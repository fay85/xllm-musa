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

#include "core/kernels/musa/musa_dsv4_ops_api.h"

#include <utility>

namespace xllm::kernel::musa {

thread_local bool dsv4_legacy_graph_capture_warmup = false;
void deepseek_v4_set_graph_capture_warmup(bool enabled) {
  dsv4_legacy_graph_capture_warmup = enabled;
}
bool deepseek_v4_graph_capture_warmup() {
  return dsv4_legacy_graph_capture_warmup;
}
void deepseek_v4_mate_sparse_attention_set_metadata_key(int64_t) {}

torch::Tensor deepseek_v4_compress_decode_sequence_c4(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_steps,
    const torch::Tensor& score_steps,
    const torch::Tensor& ape,
    int64_t head_dim,
    int64_t start_position,
    const std::optional<torch::Tensor>& output);

torch::Tensor deepseek_v4_compress_decode_sequence_graph(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_steps,
    const torch::Tensor& score_steps,
    const torch::Tensor& ape,
    int64_t /*compress_ratio*/,
    int64_t head_dim,
    const torch::Tensor& /*start_position*/,
    const torch::Tensor& output) {
  // Linking a diagnostic is allowed; executing this missing ABI is not.
  // Replacing the live device position with zero silently corrupts state.
  TORCH_CHECK(false, "Legacy archive lacks device-position Graph compression; "
                     "rebuild deepseek_v4_dsa_cache_ops.cu for this candidate");
}

// Older libcuda_kernels.a archives expose state gather with positions, while
// some historical Attention objects still refer to the shorter overload.
// The current eight-argument overload is supplied by the archive itself.
torch::Tensor deepseek_v4_state_gather(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t state_rows,
    int64_t state_block_size,
    const torch::Tensor& slots,
    const torch::Tensor& output,
    const torch::Tensor& input_positions);

torch::Tensor deepseek_v4_state_gather(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t state_rows,
    int64_t state_block_size,
    const torch::Tensor& slots,
    const torch::Tensor& output);

torch::Tensor deepseek_v4_state_gather(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t state_rows,
    int64_t state_block_size,
    const torch::Tensor& slots,
    const torch::Tensor& output) {
  return deepseek_v4_state_gather(cache,
                                  block_table,
                                  sequence_index,
                                  state_rows,
                                  state_block_size,
                                  slots,
                                  output,
                                  torch::Tensor());
}

// These state carriers are new symbols in the current source.  The legacy
// q6 object does not execute the new dynamic-MATE path, so a thread-local
// no-op is sufficient for this ABI-compatible performance A/B.
// The legacy kernel archive predates the replay-time Indexer ABI.  The
// current Attention object still needs this symbol at link time even when
// the legacy q6 forward is selected (the dynamic Indexer path is disabled in
// that A/B).  Keep the compatibility implementation deliberately eager-ABI
// based; it is only a link/runtime fallback for old archives and must not be
// used as evidence for the new dynamic Graph Indexer semantics.
torch::Tensor deepseek_v4_indexer_topk(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t topk,
    int64_t query_absolute_start,
    int64_t compress_ratio,
    const std::optional<torch::Tensor>& output,
    const torch::Tensor& mate_compressed_block_table,
    const torch::Tensor& mate_swa_block_table,
    int64_t mate_kv_len,
    int64_t mate_window_size,
    int64_t mate_block_size,
    const std::optional<torch::Tensor>& mate_physical_indices,
    const std::optional<torch::Tensor>& mate_extra_physical_indices);

torch::Tensor deepseek_v4_indexer_topk_graph(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_capacity,
    int64_t topk,
    const torch::Tensor& output,
    const torch::Tensor& mate_compressed_block_table,
    const torch::Tensor& mate_swa_block_table,
    int64_t mate_window_size,
    int64_t mate_block_size,
    const torch::Tensor& mate_physical_indices,
    const torch::Tensor& mate_extra_physical_indices,
    const torch::Tensor& /*dynamic_kv_len*/) {
  TORCH_CHECK(false, "Legacy archive lacks replay-time Indexer lengths; "
                     "capacity must not substitute for dynamic_kv_len");
}

}  // namespace xllm::kernel::musa
