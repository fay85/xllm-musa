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

// In-place partial interleaved RoPE touching only the rotary slice
// [rope_start_dim, rope_start_dim + rope_head_dim). Leaves non-rotary dims
// completely untouched in memory.
bool deepseek_v4_partial_rope_inplace(torch::Tensor& input,
                                      const torch::Tensor& cos,
                                      const torch::Tensor& sin,
                                      int64_t rope_start_dim,
                                      int64_t rope_head_dim,
                                      bool inverse);

}  // namespace xllm::kernel::musa
