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

#include "core/kernels/musa/musa_dsv4_ops_api.h"

namespace xllm::kernel {

struct ScatterBySlotParams {
  torch::Tensor cache;
  torch::Tensor slot_mapping;
  torch::Tensor value;
};

struct DeepseekV4ScatterCacheParams {
  torch::Tensor cache;
  torch::Tensor slot_mapping;
  torch::Tensor value;
};

struct DeepseekV4PackFp8CacheParams {
  torch::Tensor cache;
  torch::Tensor slot_mapping;
  torch::Tensor value;
};

struct DeepseekV4IndexerCacheQuantScatterParams {
  torch::Tensor cache;
  torch::Tensor scale_cache;
  torch::Tensor slot_mapping;
  torch::Tensor value;
};

struct DeepseekV4IndexerCacheGatherParams {
  torch::Tensor cache;
  torch::Tensor scale_cache;
  torch::Tensor block_table;
  int64_t sequence_index = 0;
  int64_t kv_len = 0;
  torch::ScalarType output_dtype = torch::kBFloat16;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4IndexerTopkParams {
  torch::Tensor query;
  torch::Tensor weights;
  torch::Tensor cache;
  torch::Tensor scale_cache;
  torch::Tensor block_table;
  int64_t sequence_index = 0;
  int64_t kv_len = 0;
  int64_t topk = 0;
  int64_t query_absolute_start = 0;
  int64_t compress_ratio = 4;
  std::optional<torch::Tensor> output = std::nullopt;
  torch::Tensor mate_compressed_block_table;
  torch::Tensor mate_swa_block_table;
  int64_t mate_kv_len = 0;
  int64_t mate_window_size = 0;
  int64_t mate_block_size = 0;
  std::optional<torch::Tensor> mate_physical_indices = std::nullopt;
  std::optional<torch::Tensor> mate_extra_physical_indices = std::nullopt;
};

struct DeepseekV4LogicalToPhysicalTopkParams {
  torch::Tensor logical_indices;
  torch::Tensor block_table;
  int64_t sequence_index = 0;
  int64_t block_size = 0;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4PrepareMateDsaIndicesParams {
  torch::Tensor logical_indices;
  torch::Tensor compressed_block_table;
  torch::Tensor swa_block_table;
  int64_t sequence_index = 0;
  int64_t kv_len = 0;
  int64_t window_size = 0;
  int64_t block_size = 0;
  std::optional<torch::Tensor> physical_indices = std::nullopt;
  std::optional<torch::Tensor> extra_physical_indices = std::nullopt;
};

struct DeepseekV4CompressPoolParams {
  torch::Tensor kv;
  torch::Tensor score;
  torch::Tensor ape;
  int64_t compress_ratio = 0;
  int64_t head_dim = 0;
  int64_t compressor_coeff = 0;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4DenseAttentionDecodeParams {
  torch::Tensor q;
  torch::Tensor kv;
  torch::Tensor compressed_kv;
  torch::Tensor attn_sink;
  double softmax_scale = 1.0;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4PagedAttentionDecodeParams {
  torch::Tensor q;
  torch::Tensor cache;
  torch::Tensor block_table;
  int64_t sequence_index = 0;
  int64_t kv_len = 0;
  torch::Tensor kv_seq_lens;
  int64_t window_size = 0;
  torch::Tensor compressed_kv;
  torch::Tensor compressed_cache;
  torch::Tensor compressed_block_table;
  int64_t compressed_sequence_index = 0;
  int64_t compressed_kv_len = 0;
  torch::Tensor attn_sink;
  double softmax_scale = 1.0;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4MateSparseAttentionParams {
  torch::Tensor q;
  torch::Tensor packed_cache;
  torch::Tensor physical_indices;
  int64_t topk_length = 0;
  torch::Tensor extra_packed_cache;
  torch::Tensor extra_physical_indices;
  int64_t extra_topk_length = 0;
  torch::Tensor attn_sink;
  double softmax_scale = 1.0;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4MateSparsePrefillParams {
  torch::Tensor q;
  torch::Tensor kv;
  torch::Tensor compressed_kv;
  torch::Tensor attn_sink;
  double softmax_scale = 1.0;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4MateC4SparsePrefillParams {
  torch::Tensor q;
  torch::Tensor kv;
  torch::Tensor compressed_kv;
  torch::Tensor compressed_indices;
  torch::Tensor attn_sink;
  double softmax_scale = 1.0;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct DeepseekV4GatherCacheParams {
  torch::Tensor cache;
  torch::Tensor block_table;
  int64_t sequence_index = 0;
  int64_t kv_len = 0;
  std::optional<torch::Tensor> output = std::nullopt;
};

inline void scatter_by_slot(ScatterBySlotParams& params) {
  musa::deepseek_v4_scatter_cache(
      params.cache, params.slot_mapping, params.value);
}

inline void deepseek_v4_scatter_cache(DeepseekV4ScatterCacheParams& params) {
  musa::deepseek_v4_scatter_cache(
      params.cache, params.slot_mapping, params.value);
}

inline void deepseek_v4_pack_fp8_cache(DeepseekV4PackFp8CacheParams& params) {
  musa::deepseek_v4_pack_fp8_cache(
      params.cache, params.slot_mapping, params.value);
}

inline void deepseek_v4_indexer_cache_quant_scatter(
    DeepseekV4IndexerCacheQuantScatterParams& params) {
  musa::deepseek_v4_indexer_cache_quant_scatter(
      params.cache, params.scale_cache, params.slot_mapping, params.value);
}

inline torch::Tensor deepseek_v4_indexer_cache_gather(
    DeepseekV4IndexerCacheGatherParams& params) {
  return musa::deepseek_v4_indexer_cache_gather(params.cache,
                                                params.scale_cache,
                                                params.block_table,
                                                params.sequence_index,
                                                params.kv_len,
                                                params.output_dtype,
                                                params.output);
}

inline torch::Tensor deepseek_v4_indexer_topk(
    DeepseekV4IndexerTopkParams& params) {
  return musa::deepseek_v4_indexer_topk(params.query,
                                        params.weights,
                                        params.cache,
                                        params.scale_cache,
                                        params.block_table,
                                        params.sequence_index,
                                        params.kv_len,
                                        params.topk,
                                        params.query_absolute_start,
                                        params.compress_ratio,
                                        params.output,
                                        params.mate_compressed_block_table,
                                        params.mate_swa_block_table,
                                        params.mate_kv_len,
                                        params.mate_window_size,
                                        params.mate_block_size,
                                        params.mate_physical_indices,
                                        params.mate_extra_physical_indices);
}

inline torch::Tensor deepseek_v4_logical_to_physical_topk(
    DeepseekV4LogicalToPhysicalTopkParams& params) {
  return musa::deepseek_v4_logical_to_physical_topk(params.logical_indices,
                                                    params.block_table,
                                                    params.sequence_index,
                                                    params.block_size,
                                                    params.output);
}

inline std::tuple<torch::Tensor, torch::Tensor>
deepseek_v4_prepare_mate_dsa_indices(
    DeepseekV4PrepareMateDsaIndicesParams& params) {
  return musa::deepseek_v4_prepare_mate_dsa_indices(
      params.logical_indices,
      params.compressed_block_table,
      params.swa_block_table,
      params.sequence_index,
      params.kv_len,
      params.window_size,
      params.block_size,
      params.physical_indices,
      params.extra_physical_indices);
}

inline torch::Tensor deepseek_v4_compress_pool(
    DeepseekV4CompressPoolParams& params) {
  return musa::deepseek_v4_compress_pool(params.kv,
                                         params.score,
                                         params.ape,
                                         params.compress_ratio,
                                         params.head_dim,
                                         params.compressor_coeff,
                                         params.output);
}

inline torch::Tensor deepseek_v4_dense_attention_decode(
    DeepseekV4DenseAttentionDecodeParams& params) {
  return musa::deepseek_v4_dense_attention_decode(params.q,
                                                  params.kv,
                                                  params.compressed_kv,
                                                  params.attn_sink,
                                                  params.softmax_scale,
                                                  params.output);
}

inline torch::Tensor deepseek_v4_paged_attention_decode(
    DeepseekV4PagedAttentionDecodeParams& params) {
  return musa::deepseek_v4_paged_attention_decode_with_paged_compressed_kv(
      params.q,
      params.cache,
      params.block_table,
      params.sequence_index,
      params.kv_len,
      params.kv_seq_lens,
      params.window_size,
      params.compressed_kv,
      params.compressed_cache,
      params.compressed_block_table,
      params.compressed_sequence_index,
      params.compressed_kv_len,
      params.attn_sink,
      params.softmax_scale,
      std::nullopt,
      params.output);
}

inline torch::Tensor deepseek_v4_mate_sparse_attention_decode(
    DeepseekV4MateSparseAttentionParams& params) {
  return musa::deepseek_v4_mate_sparse_attention_decode(
      params.q,
      params.packed_cache,
      params.physical_indices,
      params.topk_length,
      params.extra_packed_cache,
      params.extra_physical_indices,
      params.extra_topk_length,
      params.attn_sink,
      params.softmax_scale,
      params.output);
}

inline torch::Tensor deepseek_v4_mate_sparse_attention_prefill(
    DeepseekV4MateSparsePrefillParams& params) {
  return musa::deepseek_v4_mate_sparse_attention_prefill(params.q,
                                                         params.kv,
                                                         params.compressed_kv,
                                                         params.attn_sink,
                                                         params.softmax_scale,
                                                         params.output);
}

inline torch::Tensor deepseek_v4_mate_c4_sparse_attention_prefill(
    DeepseekV4MateC4SparsePrefillParams& params) {
  return musa::deepseek_v4_mate_c4_sparse_attention_prefill(
      params.q,
      params.kv,
      params.compressed_kv,
      params.compressed_indices,
      params.attn_sink,
      params.softmax_scale,
      params.output);
}

inline torch::Tensor deepseek_v4_gather_cache(
    DeepseekV4GatherCacheParams& params) {
  return musa::deepseek_v4_gather_cache(params.cache,
                                        params.block_table,
                                        params.sequence_index,
                                        params.kv_len,
                                        params.output);
}

}  // namespace xllm::kernel
