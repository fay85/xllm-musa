/* Copyright 2026 The xLLM Authors. */

#pragma once

#include <torch/types.h>

namespace xllm::kernel::musa {

// Fused partial interleaved RoPE for [tokens, dim] and
// [tokens, heads, dim]. cos/sin use the repeated interleaved layout
// [cos0, cos0, cos1, cos1, ...] consumed by the existing Torch path.
bool deepseek_v4_partial_rope(const torch::Tensor& input,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin,
                              int64_t rope_start_dim,
                              int64_t rope_head_dim,
                              bool inverse,
                              torch::Tensor& output);

}  // namespace xllm::kernel::musa
