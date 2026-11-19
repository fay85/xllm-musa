/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_pcp_layout.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_sp_context.h"

namespace xllm::layer::dsv4_musa_pcp {

// PCP splits one Prefill chunk across two CP groups. Attention keeps
// group-local TP/SP ranges; corresponding TP ranks exchange complete KV and
// recurrent state before the next chunk. The global MoE communicator spans both
// groups.
struct Context final {
  // Cross-group communicator: corresponding TP ranks in the two CP groups.
  ProcessGroup* cross_group = nullptr;
  int32_t cp_rank = 0;  // 0 = earlier group, 1 = later group
  int32_t cp_size = 2;
  // Split point of the CURRENT chunk in global row order.  The first chunk
  // of a prompt uses the balanced split of the full prompt; every later
  // chunk re-balances over its own rows (chunked prefill gathers rows per
  // chunk, so both groups see the same chunk row count).
  int64_t front_rows = 0;
  int64_t total_rows = 0;
  // Group-local SP context covering this group's row range.
  std::optional<dsv4_musa_sp::DeepseekV4SPContext> group_sp;
  // Non-owning communicators live for the model's entire execution lifetime.
  ProcessGroup* global_group = nullptr;
  ProcessGroup* moe_group = nullptr;
  torch::Tensor global_input_ids;
  // Same local query rows, collected in one operation over all physical ranks.
  std::optional<dsv4_musa_sp::DeepseekV4SPContext> global_sp;
  // Opt-in packed batch. All consumers must use these mappings; scalar
  // global_row_start cannot describe two disjoint request intervals.
  std::optional<SequenceLayout> batch_layout;
  torch::Tensor batch_group_indices;
  torch::Tensor batch_local_indices;
  torch::Tensor batch_restore_indices;
};

bool enabled();

// Builds a PCP context when the batch is exactly one fresh prefill long
// enough to split across two groups.  Leaves `context` empty otherwise.
void build_context(int64_t total_rows,
                   const ParallelArgs& parallel_args,
                   const torch::Tensor& global_tokens,
                   std::optional<Context>* context);

// Eager-only, validated CP2/TP4 microbatch=2. Unsupported configurations leave
// context empty; the caller must not reinterpret an admitted merged batch as
// a single-sequence request. Admission fallback belongs in the scheduler.
void build_batch_context(const SequenceLayout& layout,
                         const ParallelArgs& parallel_args,
                         const torch::Tensor& global_tokens,
                         std::optional<Context>* context);

inline bool is_batch(const Context& context) {
  return context.batch_layout.has_value();
}

// Select this rank's rows from a sequence-major tensor. Padding is appended
// only when requested and never receives a global position or a cache slot.
torch::Tensor local_rows_from_global(const torch::Tensor& global,
                                     const Context& context,
                                     bool pad_to_capacity);

// Gather the common padded rank capacity over all eight physical ranks, then
// invert the packed permutation to unpadded sequence-major order.
torch::Tensor gather_batch_global(const torch::Tensor& local_rows,
                                 const Context& context);

// Global rows owned by this rank's group.
inline int64_t group_rows(const Context& context) {
  return context.cp_rank == 0 ? context.front_rows
                              : context.total_rows - context.front_rows;
}

// Shard a full-prompt tensor down to this group's rows.
torch::Tensor shard_rows(const torch::Tensor& global, const Context& context);

// Assemble unequal compressed row ranges in global token order on both CP
// groups. Each rank carries the same global compressed slot mapping.
torch::Tensor exchange_rows_global_slots(const torch::Tensor& local_rows,
                                         const torch::Tensor& global_slots,
                                         Context& context);

// Completed compressed rows are partitioned by absolute cached positions.
// Local input is sequence-ordered inside each CP group. Result matches global
// compressed slot order [A rows, B rows], not [all front rows, all back rows].
torch::Tensor exchange_batch_compressed_rows(
    const torch::Tensor& local_rows,
    const torch::Tensor& global_slots,
    int64_t ratio,
    Context& context);

// Per-layer K/V exchange, both directions.  Each group projects only its own
// rows; after the exchange both groups hold the full chunk rows in global
// row order, so cache writes use the global slot mapping unchanged.
torch::Tensor relay_kv(torch::Tensor rows, Context& context);

// Recurrent compressor/indexer state forward per layer (front -> later
// inside a chunk).  Returns the seeding state on the later group.
torch::Tensor relay_state(torch::Tensor state, Context& context);

// Chunk-boundary backward leg (later -> front): installs the end-of-chunk
// recurrent state so the front group continues the next chunk correctly.
void relay_state_back(torch::Tensor& state_slot, Context& context);

// Restores full chunk hidden states on every participating rank.
void finish(torch::Tensor& local_hidden_states, Context& context);

}  // namespace xllm::layer::dsv4_musa_pcp
