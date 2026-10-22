// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
#pragma once

#include <torch/torch.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace xllm::layer {

// Same per-query score and causal TopK as the dense Indexer reference, with
// bounded scratch. No K/V truncation, head reduction, or scoring relaxation.
inline torch::Tensor dsv4_tiled_indexer_topk(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& weights, int64_t absolute_start, int64_t topk,
    int64_t score_bytes = 64 * 1024 * 1024) {
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
  auto key_fp32_t = key.to(torch::kFloat32).transpose(0, 1);
  auto keys_index = torch::arange(keys, query.options().dtype(torch::kInt64));
  auto ranks = torch::arange(selected_count, keys_index.options()).unsqueeze(0);
  for (int64_t start = 0; start < queries; start += rows) {
    const int64_t count = std::min(rows, queries - start);
    auto score = torch::matmul(query.narrow(0, start, count).to(torch::kFloat32),
                               key_fp32_t);
    score.relu_();
    score.mul_(weights.narrow(0, start, count).to(torch::kFloat32).unsqueeze(-1));
    score = score.sum(1);
    auto valid = torch::arange(count, keys_index.options())
                     .add(absolute_start + start + 1)
                     .div(4, "floor").clamp(0, keys);
    score.masked_fill_(keys_index.unsqueeze(0).ge(valid.unsqueeze(1)),
                       -std::numeric_limits<float>::infinity());
    auto indices = std::get<1>(score.topk(selected_count, -1, true, true))
                       .to(torch::kInt32);
    indices.masked_fill_(ranks.ge(valid.clamp_max(selected_count).unsqueeze(1)), -1);
    selected.narrow(0, start, count).narrow(1, 0, selected_count).copy_(indices);
  }
  return selected;
}

}  // namespace xllm::layer
