/* Copyright 2026 The xLLM Authors. */

#pragma once

#include <torch/types.h>

namespace xllm::kernel::musa {

bool deepseek_v4_fp8_quantize_dequantize(torch::Tensor& tensor,
                                         int64_t quant_dim,
                                         int64_t block_size);

// Fused per-row INT8 quantize/dequantize roundtrip used by the DeepSeek-V4
// indexer query path. The result is written to output in input.dtype().
bool deepseek_v4_int8_quantize_dequantize(const torch::Tensor& input,
                                          torch::Tensor& output);

}  // namespace xllm::kernel::musa
