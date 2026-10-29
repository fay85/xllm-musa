/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace xllm::layer {

// Both compressor roles use the same recurrent KV/score storage contract.
// Reuse existing tensors when their shape matches to preserve Graph addresses.
inline void dsv4_ensure_compressor_state(torch::Tensor& kv,
                                         torch::Tensor& score,
                                         int64_t rows,
                                         int64_t width,
                                         const torch::TensorOptions& options) {
  auto matches = [rows, width](const torch::Tensor& value) {
    return value.defined() && value.dim() == 2 && value.size(0) == rows &&
           value.size(1) == width;
  };
  if (!matches(kv)) {
    kv = torch::zeros({rows, width}, options);
  }
  if (!matches(score)) {
    score = torch::full(
        {rows, width}, -std::numeric_limits<float>::infinity(), options);
  }
}

// C4 carries the previous complete block plus the current partial block.
// C128 carries only the partial block; it must not inherit C4 overlap.
inline std::pair<torch::Tensor, torch::Tensor> dsv4_prefill_state_tail(
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& ape,
    int64_t end,
    int64_t ratio,
    int64_t rows,
    int64_t width,
    const torch::TensorOptions& options) {
  auto kv_state = torch::zeros({rows, width}, options);
  auto score_state = torch::full(
      {rows, width}, -std::numeric_limits<float>::infinity(), options);
  const int64_t cutoff = end - end % ratio;
  const int64_t remainder = end - cutoff;
  if (ratio == 4 && cutoff >= ratio) {
    kv_state.narrow(0, 0, ratio).copy_(kv.slice(0, cutoff - ratio, cutoff));
    score_state.narrow(0, 0, ratio)
        .copy_(score.slice(0, cutoff - ratio, cutoff) + ape);
  }
  if (remainder > 0) {
    const int64_t offset = ratio == 4 ? ratio : 0;
    kv_state.narrow(0, offset, remainder).copy_(kv.slice(0, cutoff, end));
    score_state.narrow(0, offset, remainder)
        .copy_(score.slice(0, cutoff, end) + ape.narrow(0, 0, remainder));
  }
  return {kv_state.contiguous(), score_state.contiguous()};
}

}  // namespace xllm::layer
