/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace xllm::kernel::musa {

// FP8 expert-major activations, FP32 group128 scales, valid-row counts and
// the original-route -> expert-row map. Only valid prefixes are initialized.
using DeepseekV41W4A8Dispatch =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

DeepseekV41W4A8Dispatch deepseek_v41_w4a8_dispatch(
    const torch::Tensor& input,
    const torch::Tensor& expert_ids,
    int64_t first_expert,
    int64_t local_experts,
    double static_scale = 0.0);

// Wide prefill only. Measure the exact expert occupancy before allocating
// the padded buffers. An expert above 512 rows requires the existing tiles.
std::optional<DeepseekV41W4A8Dispatch> deepseek_v41_w4a8_dispatch_compact(
    const torch::Tensor& input,
    const torch::Tensor& expert_ids,
    int64_t first_expert,
    int64_t local_experts,
    double static_scale);

// Quantize the valid expert-major BF16 prefixes already dispatched by DeepEP.
// Preserve expert/slot order and ignore uninitialized padding. No route weight
// is applied here; DeepEP combine applies it once after the down projection.
// Counts must come from the dispatcher and lie in [0, padded_capacity].
std::pair<torch::Tensor, torch::Tensor> deepseek_v41_w4a8_quant_dispatched(
    const torch::Tensor& input,
    const torch::Tensor& counts,
    double static_scale = 0.0);

std::pair<torch::Tensor, torch::Tensor> deepseek_v41_w4a8_swiglu_quant(
    const torch::Tensor& gate_up,
    const torch::Tensor& counts,
    double limit,
    double static_scale = 0.0,
    bool valid_prefix = false);

std::pair<torch::Tensor, torch::Tensor>
deepseek_v41_w4a8_swiglu_quant_prefill_parallel(const torch::Tensor& gate_up,
                                                const torch::Tensor& counts,
                                                double limit,
                                                double static_scale);

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
                                            const std::string& library_path,
                                            bool per_token_scale = false);

}  // namespace xllm::kernel::musa
