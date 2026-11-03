/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm::layer::musa_pcp {

// Rows owned by the last rank that earlier ranks write into their KV cache
// after the layers.
struct DeferredKv final {
  torch::Tensor rows;
  torch::Tensor k_cache;
  torch::Tensor v_cache;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
};

// Recurrent-state rows that earlier ranks copy from the last rank after the
// layers.
struct DeferredState final {
  torch::Tensor cache;
  torch::Tensor indices;
};

// Rows from the preceding rank whose transfer was posted ahead of their use.
struct PendingReceive final {
  c10::intrusive_ptr<c10d::Work> work;
  torch::Tensor rows;
};

// Prefill context parallelism splits one fresh prompt into contiguous row
// ranges, one per CP rank. Inside the layers data only flows from earlier to
// later ranks, so the last rank ends with the full KV cache and the final
// recurrent state. Earlier ranks catch up after the layers only when they
// decode as well.
struct Context final {
  ProcessGroup* cp_group = nullptr;
  int32_t cp_rank = 0;
  // Rank r owns rows [row_offsets[r], row_offsets[r + 1]).
  std::vector<int32_t> row_offsets;
  int32_t max_rows = 0;
  bool replicate_final_state = true;
  torch::Tensor global_slot_mapping;
  AttentionMetadata local_metadata;
  std::vector<DeferredKv> deferred_kv;
  std::vector<DeferredState> deferred_states;
  std::deque<PendingReceive> pending_receives;
  // Buffers a root sent without waiting. A captured graph would otherwise
  // reuse their memory before the transfer stream reads them, so they live
  // until the final gather joins that stream.
  std::vector<torch::Tensor> in_flight_sends;
};

// Emplaces a context when the batch is one fresh prefill long enough to split
// across the CP group, and leaves `context` empty otherwise. `num_rows` is the
// row count of the forward, including graph padding.
void build_context(const AttentionMetadata& global_metadata,
                   const ParallelArgs& parallel_args,
                   int32_t num_rows,
                   std::optional<Context>* context);

// Whether this rank continues rows of a preceding rank and therefore takes its
// recurrent state from the preceding rank rather than from the cache.
bool receives_prefix_state(const Context& context);

// Rows of `global` owned by this rank along `dim`.
torch::Tensor shard_rows(const torch::Tensor& global,
                         const Context& context,
                         int64_t dim);

// Concatenates the rows of every rank in rank order.
torch::Tensor gather_rows(const torch::Tensor& local, const Context& context);

// Data produced on the preceding rank flows to the rank that continues its
// rows, one transfer per call in layer order on every rank. Receivers post a
// transfer before the rows are needed so it overlaps their own work, then take
// the oldest posted rows. Each call is a no-op on ranks it does not concern.
void expect_prefix_rows(torch::IntArrayRef shape,
                        const torch::TensorOptions& options,
                        Context& context);
// Posts one row of `cache`, the per-sequence state slot.
void expect_prefix_cache_row(const torch::Tensor& cache, Context& context);
torch::Tensor take_prefix_rows(Context& context);
void receive_prefix_cache_row(torch::Tensor& cache,
                              const torch::Tensor& indices,
                              Context& context);
void send_prefix_rows(const torch::Tensor& rows, Context& context);
void send_prefix_cache_row(const torch::Tensor& cache,
                           const torch::Tensor& indices,
                           Context& context);

// Posts the transfer of the preceding rank's key/value rows of one attention
// layer.
void expect_prefix_kv(int64_t num_kv_heads,
                      int64_t head_dim,
                      const torch::TensorOptions& options,
                      Context& context);

// Writes this rank's key/value rows and every earlier rank's rows into this
// rank's paged cache.
void exchange_kv(const torch::Tensor& key,
                 const torch::Tensor& value,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 KVCache& kv_cache,
                 Context& context);

// Marks final recurrent-state rows that earlier ranks copy from the last rank
// after the layers.
void defer_final_state(const torch::Tensor& cache,
                       const torch::Tensor& indices,
                       Context& context);

// Completes the deferred KV and recurrent-state copies and returns the hidden
// states of every row in rank order.
torch::Tensor finish(const torch::Tensor& local_hidden_states,
                     Context& context);

}  // namespace xllm::layer::musa_pcp
