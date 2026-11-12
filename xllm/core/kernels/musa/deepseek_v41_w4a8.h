/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

namespace xllm::kernel::musa {

// FP8 expert-major activations, FP32 group128 scales, valid-row counts and
// the original-route -> expert-row map. Only valid prefixes are initialized.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_w4a8_dispatch(const torch::Tensor& input,
                           const torch::Tensor& expert_ids,
                           int64_t first_expert,
                           int64_t local_experts);

std::pair<torch::Tensor, torch::Tensor> deepseek_v41_w4a8_swiglu_quant(
    const torch::Tensor& gate_up,
    const torch::Tensor& counts,
    double limit);

torch::Tensor deepseek_v41_w4a8_combine(const torch::Tensor& expert_output,
                                        const torch::Tensor& expert_ids,
                                        const torch::Tensor& route_rows,
                                        const torch::Tensor& route_weights,
                                        int64_t first_expert);

torch::Tensor deepseek_v41_w4a8_masked_gemm(const torch::Tensor& input,
                                            const torch::Tensor& input_scale,
                                            const torch::Tensor& weight,
                                            const torch::Tensor& weight_scale,
                                            const torch::Tensor& counts,
                                            const std::string& library_path);

}  // namespace xllm::kernel::musa
