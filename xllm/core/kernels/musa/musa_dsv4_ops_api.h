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

}  // namespace xllm::kernel::musa
