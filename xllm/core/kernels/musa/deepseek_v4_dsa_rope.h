/* Copyright 2026 The xLLM Authors. */

#pragma once

#include <torch/types.h>

#include <cstdint>

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

// BF16 input/output with a contiguous FP32 [cos_half, sin_half] table.
// Positions are read on device, so changing their contents remains Graph-safe.
// FP32 arithmetic and BF16 rounding are preserved; bitwise identity to the
// separate FP32-output kernel and ATen cast is not promised.
torch::Tensor deepseek_v41_rope_table(const torch::Tensor& input,
                                      const torch::Tensor& positions,
                                      const torch::Tensor& table,
                                      bool inverse);

}  // namespace xllm::kernel::musa
