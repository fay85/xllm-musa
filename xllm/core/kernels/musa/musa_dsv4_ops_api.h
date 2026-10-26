/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

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
#include <tuple>

namespace xllm::kernel::musa {

bool deepseek_v4_partial_rope(const torch::Tensor& input,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin,
                              int64_t rope_start_dim,
                              int64_t rope_head_dim,
                              bool inverse,
                              torch::Tensor& output);

bool deepseek_v4_q_rmsnorm_rope_inplace(torch::Tensor& query,
                                        const torch::Tensor& cos,
                                        const torch::Tensor& sin,
                                        double eps);

bool deepseek_v4_weighted_rmsnorm_rope_inplace(torch::Tensor& input,
                                               const torch::Tensor& weight,
                                               const torch::Tensor& cos,
                                               const torch::Tensor& sin,
                                               double eps);

// Read packed, row-strided projections into distinct contiguous storage.
// Preserve the corresponding inplace kernel's reduction and BF16 rounding.
bool deepseek_v4_q_rmsnorm_rope_out(const torch::Tensor& query,
                                    const torch::Tensor& cos,
                                    const torch::Tensor& sin,
                                    double eps,
                                    const torch::Tensor& output);
bool deepseek_v4_weighted_rmsnorm_rope_out(const torch::Tensor& input,
                                           const torch::Tensor& weight,
                                           const torch::Tensor& cos,
                                           const torch::Tensor& sin,
                                           double eps,
                                           const torch::Tensor& output);

torch::Tensor deepseek_v4_dense_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_paged_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t window_size,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_paged_attention_decode_with_paged_compressed_kv(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const torch::Tensor& kv_seq_lens,
    int64_t window_size,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_sequence_index,
    int64_t compressed_kv_len,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& retained_mass_output = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_mate_sparse_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& physical_indices,
    int64_t topk_length,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& extra_physical_indices,
    int64_t extra_topk_length,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

// B1 compact verify: every query row reads its replay-time causal KV length.
// Compressed rows must already be published to the paged cache. Capacity is
// only a workspace bound, never the number of logically visible C128 rows.
torch::Tensor deepseek_v4_paged_attention_verify_compressed(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_capacity,
    int64_t compression_ratio,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const torch::Tensor& output);

torch::Tensor deepseek_v4_paged_attention_sas(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t window_size,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const torch::Tensor& output);

torch::Tensor deepseek_v4_greedy_verify_tokens(
    const torch::Tensor& draft_tokens,
    const torch::Tensor& target_tokens,
    const torch::Tensor& output);

// Sequence-major uniform queries. Each sequence has one live KV length and
// all of its queries see the same noncausal context+block window. Zero-live
// rows yield zero output. Existing scalar and causal APIs are unchanged.
torch::Tensor deepseek_v4_paged_attention_sas_batched(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    int64_t query_rows_per_sequence,
    int64_t window_size,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const torch::Tensor& output);

torch::Tensor deepseek_v4_paged_attention_verify_c128(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_capacity,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const torch::Tensor& output);

torch::Tensor deepseek_v4_mate_sparse_attention_decode_batched(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& physical_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& extra_physical_indices,
    const torch::Tensor& extra_topk_lengths,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Preserve the scalar request's KV reduction partitions while merging MATE
// execution. q/physical/length rows are flattened sequence-major fixed-width.
torch::Tensor deepseek_v4_mate_sparse_attention_decode_partitioned(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& physical_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& extra_physical_indices,
    const torch::Tensor& extra_topk_lengths,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    int64_t query_width,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Each local planner overwrites its metadata slice before every invocation;
// this in-place rebasing must not be applied twice without that rewrite.
void deepseek_v4_rebase_mate_partitions(torch::Tensor metadata,
                                       const torch::Tensor& local_splits,
                                       torch::Tensor cumulative_splits,
                                       int64_t query_width);

// Set the per-step key used to share batched MATE scheduling metadata across
// DSV4 Attention layers. The key is thread-local to the serving worker.
void deepseek_v4_mate_sparse_attention_set_metadata_key(int64_t metadata_key);
void deepseek_v4_mate_sparse_attention_set_metadata_group(int64_t metadata_group);
void deepseek_v4_mate_causal_lengths(const torch::Tensor& kv_seq_lens,
                                     int64_t compress_ratio,
                                     int64_t topk,
                                     int64_t window,
                                     torch::Tensor topk_lengths,
                                     torch::Tensor extra_lengths);

// Graph capture performs eager warmup/FFI-recording forwards before the real
// capture. DSV4 compressor/indexer state must not advance during those
// preparation forwards; the flag is thread-local to the graph worker.
void deepseek_v4_set_graph_capture_warmup(bool enabled);
bool deepseek_v4_graph_capture_warmup();

torch::Tensor deepseek_v4_mate_sparse_attention_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_mate_c4_sparse_attention_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_indices,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_mate_local_sparse_attention_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_indices,
    const torch::Tensor& attn_sink,
    int64_t q_abs_start,
    int64_t compress_ratio,
    double softmax_scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

void deepseek_v4_c128_prefill_metadata(torch::Tensor indices,
                                       torch::Tensor topk_length,
                                       int64_t token_pack,
                                       int64_t compressed_len,
                                       int64_t compiled_compressed_kv_len);

torch::Tensor deepseek_v4_gather_cache(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const std::optional<torch::Tensor>& output = std::nullopt);

void deepseek_v4_scatter_cache(torch::Tensor cache,
                               torch::Tensor slot_ids,
                               torch::Tensor values);

void deepseek_v4_pack_fp8_cache(torch::Tensor cache,
                                torch::Tensor slot_ids,
                                torch::Tensor values);

// MODEL1 cache packing uses the compatibility CUDA stream while the MATE FFI
// consumer uses the native MUSA stream.  Bridge the producer/consumer streams
// with a device event instead of a host-wide synchronize.
void deepseek_v4_wait_model1_pack_event();

void deepseek_v4_indexer_cache_quant_scatter(torch::Tensor cache,
                                             torch::Tensor scale_cache,
                                             torch::Tensor slot_ids,
                                             torch::Tensor values);

torch::Tensor deepseek_v4_indexer_cache_gather(
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output = std::nullopt);

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
    const std::optional<torch::Tensor>& output = std::nullopt,
    const torch::Tensor& mate_compressed_block_table = torch::Tensor(),
    const torch::Tensor& mate_swa_block_table = torch::Tensor(),
    int64_t mate_kv_len = 0,
    int64_t mate_window_size = 0,
    int64_t mate_block_size = 0,
    const std::optional<torch::Tensor>& mate_physical_indices = std::nullopt,
    const std::optional<torch::Tensor>& mate_extra_physical_indices =
        std::nullopt);

// Graph variant: kv_capacity is storage capacity, not the live C4 history.
// dynamic_kv_len is an int32 device vector containing each query row's
// uncompressed KV length. Query rows share one logical sequence/block table;
// scoring and completion counters are independent. The eager ABI is unchanged.
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
    const torch::Tensor& dynamic_kv_len);

// Graph variant for fixed-width multi-sequence verify.  Query rows are laid
// out as [sequence0 rows, sequence1 rows, ...].  Each sequence keeps its own
// block-table row and dynamic KV lengths; only the score/top-k launch is
// merged so this is not a semantic cross-sequence attention operation.
torch::Tensor deepseek_v4_indexer_topk_graph_batched(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    int64_t query_rows_per_sequence,
    int64_t kv_capacity,
    int64_t topk,
    const torch::Tensor& output,
    const torch::Tensor& mate_compressed_block_table,
    const torch::Tensor& mate_swa_block_table,
    int64_t mate_window_size,
    int64_t mate_block_size,
    const torch::Tensor& mate_physical_indices,
    const torch::Tensor& mate_extra_physical_indices,
    const torch::Tensor& dynamic_kv_len);

// All causal C128 rows plus each query's SWA tail, for MODEL1 batch decode.
void deepseek_v4_prepare_c128_mate_indices(
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& compressed_block_table,
    const torch::Tensor& swa_block_table,
    int64_t block_size,
    torch::Tensor logical_indices,
    torch::Tensor physical_indices,
    torch::Tensor extra_physical_indices);

// Uniform q7 requests, with a distinct block-table row for every request.
void deepseek_v4_prepare_c128_mate_indices_batched(
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& compressed_block_table,
    const torch::Tensor& swa_block_table,
    int64_t block_size,
    int64_t query_width,
    torch::Tensor logical_indices,
    torch::Tensor physical_indices,
    torch::Tensor extra_physical_indices);

torch::Tensor deepseek_v4_logical_to_physical_topk(
    const torch::Tensor& logical_indices,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t block_size,
    const std::optional<torch::Tensor>& output = std::nullopt);

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_prepare_mate_dsa_indices(
    const torch::Tensor& logical_indices,
    const torch::Tensor& compressed_block_table,
    const torch::Tensor& swa_block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t window_size,
    int64_t block_size,
    const std::optional<torch::Tensor>& physical_indices = std::nullopt,
    const std::optional<torch::Tensor>& extra_physical_indices = std::nullopt);

torch::Tensor deepseek_v4_compress_pool(
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& ape,
    int64_t compress_ratio,
    int64_t head_dim,
    int64_t compressor_coeff,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_compress_decode(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_step,
    const torch::Tensor& score_step,
    const torch::Tensor& ape,
    int64_t compress_ratio,
    int64_t head_dim,
    int64_t compressor_coeff,
    int64_t absolute_position,
    bool compute_boundary_output = true,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Advance a C4 compressor through a contiguous sequence in one launch while
// preserving the exact per-token state rotation and boundary output order of
// deepseek_v4_compress_decode.
torch::Tensor deepseek_v4_compress_decode_sequence_c4(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_steps,
    const torch::Tensor& score_steps,
    const torch::Tensor& ape,
    int64_t head_dim,
    int64_t absolute_start_position,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Graph-safe C4/C128 sequence compression. The first absolute position is
// read on device at replay time. Output has ceil(tokens / ratio) rows: valid
// boundaries are packed first and unused rows are zeroed. Cache publication
// must use the matching dynamic slot mapping (unused slots are -1).
torch::Tensor deepseek_v4_compress_decode_sequence_graph(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_steps,
    const torch::Tensor& score_steps,
    const torch::Tensor& ape,
    int64_t compress_ratio,
    int64_t head_dim,
    const torch::Tensor& start_position,
    const torch::Tensor& output);

// Request-isolated Graph recurrence on contiguous [B,state_rows,state_width]
// states. Projection rows may be strided views of a packed projection. Query
// positions stay [B*query_width], and output is [B*ceil(query_width/ratio),H].
torch::Tensor deepseek_v4_compress_decode_batch_graph(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_steps,
    const torch::Tensor& score_steps,
    const torch::Tensor& ape,
    int64_t compress_ratio,
    int64_t head_dim,
    int64_t query_width,
    const torch::Tensor& positions,
    const torch::Tensor& output);

torch::Tensor deepseek_v4_scale_to_fp32(
    const torch::Tensor& input,
    double scale,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_hc_pre_combine(
    const torch::Tensor& x_hc,
    const torch::Tensor& pre,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor deepseek_v4_hc_post(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const std::optional<torch::Tensor>& output = std::nullopt);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v4_hc_pre_metadata(const torch::Tensor& mixes,
                            const torch::Tensor& hc_scale,
                            const torch::Tensor& hc_base,
                            int64_t hc_mult,
                            int64_t sinkhorn_iters,
                            double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v4_hc_pre_decode(
    const torch::Tensor& x,
    const torch::Tensor& hc_fn,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double norm_eps,
    double hc_eps,
    const std::optional<torch::Tensor>& output = std::nullopt,
    const std::optional<torch::Tensor>& post = std::nullopt,
    const std::optional<torch::Tensor>& comb = std::nullopt,
    const std::optional<torch::Tensor>& norm_weight = std::nullopt,
    bool* output_is_normalized = nullptr);

}  // namespace xllm::kernel::musa
