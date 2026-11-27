/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "core/util/slice.h"

namespace xllm {

struct ModelInputParams;
struct ForwardInput;

struct DecodeBuildMeta {
  int32_t kv_max_seq_len = 0;
};

// Aggregated output vectors produced by row builders.
// Callers convert these vectors to tensors and write back to input params.
struct DecodeBuildBuffers {
  DecodeBuildMeta meta;
  std::vector<int32_t> out_token_ids;
  std::vector<int32_t> out_positions;
  std::vector<int32_t> out_kv_seq_lens;
  std::vector<int32_t> out_q_seq_lens;
  std::vector<int32_t> out_q_cu_seq_lens;
  std::vector<int32_t> out_new_cache_slots;
  std::vector<int32_t> out_block_tables;
  std::vector<int32_t> out_paged_kv_indptr;
  std::vector<int32_t> out_paged_kv_indices;
  std::vector<int32_t> out_paged_kv_last_page_len;
  std::vector<std::vector<std::vector<int32_t>>> out_multi_block_tables;
  int32_t out_block_table_rows = 0;
  int32_t out_block_table_stride = 0;
};

// CPU-side row context shared by decode row builders.
// It owns the contiguous block table tensor so row slices stay valid.
struct DecodeBatchContext {
  torch::Tensor block_tables_owner;
  std::vector<torch::Tensor> multi_block_tables_owner;
  Slice<int32_t> token_ids;
  Slice<int32_t> positions;
  int32_t num_rope_axes = 1;
  Slice<int32_t> kv_seq_lens;
  Slice<int32_t> block_tables;
  std::vector<std::vector<Slice<int32_t>>> multi_block_tables;
  int32_t num_requests = 0;
  int32_t block_table_stride = 0;
  bool model_managed_multiblock = false;
};

// Check allocated lookahead pages before issuing speculative writes. Block 0
// is valid; unallocated columns must use a negative sentinel.
bool has_decode_lookahead_capacity(const DecodeBatchContext& ctx,
                                   int32_t num_tokens,
                                   int32_t block_size,
                                   int64_t max_position_embeddings);

// Declarative spec for one emitted decode row.
// A row can selectively append token/kv/q/block fields depending on caller
// needs.
struct DecodeTokenInput {
  int32_t req_id = 0;
  int32_t token_id = 0;
  int32_t position_offset = 0;
  bool append_token = true;
  bool append_kv_len = true;
  // When set, decouples the visible KV extent from this row's position. This
  // lets block-parallel draft rows share the full block KV length.
  std::optional<int32_t> kv_len_offset;
  bool append_q_len_one = false;
  bool append_block_table = false;
};

// Builds a reusable row context from ForwardInput host-side fields.
DecodeBatchContext make_decode_batch_context(const ForwardInput& input);

// Appends one logical decode row into output buffers.
void append_decode_token(const DecodeBatchContext& ctx,
                         const DecodeTokenInput& row,
                         int32_t block_size,
                         DecodeBuildBuffers& buf);

// Resolves direct/placeholder input token to a real token and position offset.
// Placeholder tokens use negative ids (-1, -2, ...) and are read from last-step
// outputs.
std::pair<int32_t, int32_t> resolve_token_with_position_offset(
    int32_t input_token_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num);

// Computes one cache slot id from absolute position and block table mapping.
int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size);

// Circular (ring) variant of calc_slot_id for DSV4 SWA block tables.
int32_t calc_ring_slot_id(int32_t position,
                          const Slice<int32_t>& block_table_slice,
                          int32_t block_size);

// Manager 0 in a grouped-cache input is the DSV4 SWA manager.
std::vector<int32_t> build_grouped_prefill_swa_slots(const ForwardInput& input,
                                                     int32_t block_size);

// Computes sequence kv length with platform-specific seq-lens layout handling.
int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset);

// Appends one q/kv length element using current backend layout policy.
void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len);

// Appends one q length and the matching cumulative q length.
void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len);

// Sets the KPool metadata required by speculative verification.
void set_kpool_verify_metadata(ModelInputParams& input_params,
                               std::vector<int32_t> query_lens);

// Appends kv_len into output vector and updates kv_max_seq_len.
void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len);

// Updates common decode-side ModelInputParams fields from built buffers.
void update_decode_attention_metadata(ModelInputParams& input_params,
                                      DecodeBuildBuffers& buf,
                                      int32_t q_max_seq_len,
                                      std::vector<int32_t> q_seq_lens_vec,
                                      std::vector<int32_t> q_cu_seq_lens_vec,
                                      int32_t kv_max_seq_len,
                                      std::vector<int32_t> kv_seq_lens_vec,
                                      bool update_block_tables = false);

struct DecodeInputOptions {
  int32_t num_speculative_tokens = 0;
  bool per_request = false;
  bool causal = true;
};

struct DecodeExpansionOptions : DecodeInputOptions {
  int32_t block_size = 0;
  std::optional<int32_t> mask_token_id;
};

void prepare_decode_buffers(DecodeBuildBuffers& buf,
                            const DecodeBatchContext& ctx,
                            size_t num_tokens,
                            bool emit_block_tables);

DecodeBuildBuffers expand_decode_inputs(const DecodeBatchContext& ctx,
                                        const DecodeExpansionOptions& options);

void build_decode_input(ForwardInput& input,
                        DecodeBuildBuffers& buf,
                        const DecodeInputOptions& options,
                        bool materialize_tokens = true);

// Rebuilds request-scoped execution metadata after speculative validation
// expands each logical request to one or more execution tokens.
void update_execution_batch_metadata(ModelInputParams& input_params,
                                     std::vector<int32_t> num_scheduled_tokens);

// A draft executes one decode token per built row, independent of the target's
// request geometry or the attention batch-forward label.
void update_draft_execution_batch_metadata(ModelInputParams& input_params,
                                           int64_t num_rows);

// Packs a host int32 vector into a pinned CPU tensor for async H2D staging.
torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values);

// Stages token_ids/positions into both the host and device tensors of `input`
// with async H2D copies, toggling device_tensors_ready around the write.
void set_token_position_tensors(ForwardInput& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options);

}  // namespace xllm
