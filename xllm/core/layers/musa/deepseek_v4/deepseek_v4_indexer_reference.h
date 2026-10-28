// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
#pragma once

#include <torch/torch.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace xllm::layer {

inline bool dsv4_prefill_indexer_bf16_fp32_shape(int64_t queries,
                                                int64_t heads,
                                                int64_t keys,
                                                int64_t dim) {
  return queries >= 128 && queries <= 8192 && heads == 64 && dim == 128 &&
         keys >= 1024 && keys <= 8192;
}

inline torch::Tensor dsv4_prefill_indexer_score_tile(
    const torch::Tensor& query,
    const torch::Tensor& key_transposed,
    const torch::Tensor& weights,
    bool bf16_fp32) {
  torch::Tensor score;
  if (bf16_fp32) {
    const int64_t queries = query.size(0);
    const int64_t heads = query.size(1);
    const int64_t keys = key_transposed.size(1);
    score = torch::empty({1, queries * heads, keys},
                         query.options().dtype(torch::kFloat32));
    // The separate bmm.dtype overload swaps alpha/beta in this installed
    // Torch/MUSA. Use the independently checked three-argument out API.
    torch::bmm_out(score, query.reshape({1, queries * heads, query.size(2)}),
                   key_transposed.unsqueeze(0));
    score = score.view({queries, heads, keys});
  } else {
    score = torch::matmul(query.to(torch::kFloat32), key_transposed);
  }
  score.relu_();
  score.mul_(weights.to(torch::kFloat32).unsqueeze(-1));
  return score.sum(1);
}

// Construct immutable masks per invocation; no persistent/cross-request cache.
// Scoring and TopK stay intact.
inline std::pair<torch::Tensor, torch::Tensor> dsv4_prefill_indexer_masks(
    const torch::Tensor& keys_index,
    const torch::Tensor& ranks,
    int64_t queries,
    int64_t absolute_start,
    int64_t keys,
    int64_t selected_count) {
  auto valid = torch::arange(queries, keys_index.options())
                   .add(absolute_start + 1).div(4, "floor").clamp(0, keys);
  auto score_mask = keys_index.unsqueeze(0).ge(valid.unsqueeze(1));
  auto index_mask = ranks.ge(valid.clamp_max(selected_count).unsqueeze(1));
  return {score_mask, index_mask};
}

// Same per-query score and causal TopK as the dense Indexer reference, with
// bounded scratch. No K/V truncation, head reduction, or scoring relaxation.
// Optional BF16-input GEMM keeps FP32 scores but may change dot-product rounding.
inline torch::Tensor dsv4_tiled_indexer_topk(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& weights, int64_t absolute_start, int64_t topk,
    int64_t score_bytes = 64 * 1024 * 1024,
    bool bf16_fp32 = false,
    bool prepare_masks_once = false) {
  TORCH_CHECK(query.dim() == 3 && key.dim() == 2 && weights.dim() == 2);
  TORCH_CHECK(query.size(2) == key.size(1) && query.size(0) == weights.size(0) &&
              query.size(1) == weights.size(1));
  TORCH_CHECK(key.size(0) > 0 && topk > 0 && score_bytes > 0);
  const int64_t queries = query.size(0), keys = key.size(0);
  const int64_t rows = std::max<int64_t>(1, score_bytes /
      (query.size(1) * keys * int64_t{sizeof(float)}));
  const int64_t selected_count = std::min(topk, keys);
  auto selected = torch::full({queries, topk}, -1,
      query.options().dtype(torch::kInt32));
  const bool use_bf16 = bf16_fp32 && query.device().is_privateuseone() &&
      query.scalar_type() == torch::kBFloat16 && key.scalar_type() == torch::kBFloat16 &&
      dsv4_prefill_indexer_bf16_fp32_shape(queries, query.size(1), keys, query.size(2));
  auto key_transposed = use_bf16 ? key.transpose(0, 1)
                               : key.to(torch::kFloat32).transpose(0, 1);
  auto keys_index = torch::arange(keys, query.options().dtype(torch::kInt64));
  auto ranks = torch::arange(selected_count, keys_index.options()).unsqueeze(0);
  torch::Tensor full_score_mask;
  torch::Tensor full_index_mask;
  if (prepare_masks_once) {
    auto masks = dsv4_prefill_indexer_masks(keys_index, ranks, queries,
        absolute_start, keys, selected_count);
    full_score_mask = std::move(masks.first);
    full_index_mask = std::move(masks.second);
  }
  for (int64_t start = 0; start < queries; start += rows) {
    const int64_t count = std::min(rows, queries - start);
    auto score = dsv4_prefill_indexer_score_tile(
        query.narrow(0, start, count), key_transposed,
        weights.narrow(0, start, count), use_bf16);
    torch::Tensor valid;
    torch::Tensor score_mask;
    if (prepare_masks_once) {
      score_mask = full_score_mask.narrow(/*dim=*/0, start, count);
    } else {
      valid = torch::arange(count, keys_index.options())
                  .add(absolute_start + start + 1)
                  .div(4, "floor").clamp(0, keys);
      score_mask = keys_index.unsqueeze(0).ge(valid.unsqueeze(1));
    }
    score.masked_fill_(score_mask, -std::numeric_limits<float>::infinity());
    auto indices = std::get<1>(score.topk(selected_count, -1, true, true))
                       .to(torch::kInt32);
    auto index_mask = prepare_masks_once
        ? full_index_mask.narrow(/*dim=*/0, start, count)
        : ranks.ge(valid.clamp_max(selected_count).unsqueeze(1));
    indices.masked_fill_(index_mask, -1);
    selected.narrow(0, start, count).narrow(1, 0, selected_count).copy_(indices);
  }
  return selected;
}

}  // namespace xllm::layer
