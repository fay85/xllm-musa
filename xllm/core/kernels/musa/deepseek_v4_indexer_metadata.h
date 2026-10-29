// Copyright 2026 The xLLM Authors. Licensed under the Apache License,
// Version 2.0.
#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace xllm::kernel::musa {

// Output storage belongs to the caller and remains stable across Graph replay.
// INT8 keys retain the original FP16-scale -> BF16 -> FP32 rounding sequence.
void deepseek_v4_indexer_gather_keys_fp32(const torch::Tensor& cache,
                                          const torch::Tensor& scales,
                                          const torch::Tensor& blocks,
                                          const torch::Tensor& keys,
                                          const torch::Tensor& valid_slots);

// Same gather and BF16 rounding, without materializing a FP32 intermediate.
void deepseek_v4_indexer_gather_keys_bf16(const torch::Tensor& cache,
                                          const torch::Tensor& scales,
                                          const torch::Tensor& blocks,
                                          const torch::Tensor& keys,
                                          const torch::Tensor& valid_slots);

// In-place ReLU then FP32 head weighting. Leaves head reduction and TopK to
// the existing framework path; signed zero and NaN follow the original ReLU.
void deepseek_v4_indexer_relu_weight(const torch::Tensor& scores,
                                     const torch::Tensor& weights);

// Fuse ReLU, FP32 weighting, and the 64-head sum without modifying scores.
// Input scores: [batch, queries, 64, keys]; output: [batch * queries, keys].
torch::Tensor deepseek_v4_indexer_weighted_sum(const torch::Tensor& scores,
                                               const torch::Tensor& weights);

void deepseek_v4_indexer_mask_scores(const torch::Tensor& scores,
                                     const torch::Tensor& lengths,
                                     const torch::Tensor& valid_slots,
                                     int64_t query_rows_per_sequence);

// Does not select or sort keys. Preserve the supplied TopK order except for
// the existing all-valid short-row shortcut and its -1 padding.
void deepseek_v4_indexer_map_outputs(const torch::Tensor& selected,
                                     const torch::Tensor& lengths,
                                     const torch::Tensor& compressed_blocks,
                                     const torch::Tensor& swa_blocks,
                                     int64_t query_rows_per_sequence,
                                     int64_t capacity,
                                     int64_t window,
                                     int64_t page_size,
                                     const torch::Tensor& logical,
                                     const torch::Tensor& physical,
                                     const torch::Tensor& extra);

}  // namespace xllm::kernel::musa
