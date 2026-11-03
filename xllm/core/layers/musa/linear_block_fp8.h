/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace xllm {

struct QuantArgs;
class StateDict;

}  // namespace xllm

namespace xllm::layer::musa {

bool is_block_fp8_quant(const QuantArgs& quant_args);

void check_quantization_supported(
    const QuantArgs& quant_args,
    const std::optional<std::string>& resolved_weight_quant_method =
        std::nullopt);

void check_replicated_weight_supported(const StateDict& state_dict);

void maybe_resolve_block_fp8_unquantized(
    const StateDict& state_dict,
    const std::vector<std::string>* prefixes,
    const torch::TensorOptions& options,
    torch::Tensor& weight,
    bool& weight_is_loaded,
    bool weight_scale_inv_is_loaded,
    bool& block_fp8_resolved_unquantized);

void register_block_fp8_parameters(torch::nn::Module& module,
                                   int64_t out_features,
                                   int64_t in_features,
                                   const torch::TensorOptions& options,
                                   torch::Tensor& weight,
                                   torch::Tensor& weight_scale_inv);

// Each weight segment must be a multiple of 128 rows per rank, because a
// scale row covers 128 weight rows.
std::vector<int64_t> block_fp8_scale_shard_sizes(
    const std::vector<int64_t>& weight_shard_sizes);

// Quantizes a 16-bit [N, K] weight to float8_e4m3fn with one FP32
// dequantization scale per 128x128 block, the layout of block-FP8
// checkpoints. Returns {weight [N, K], weight_scale_inv [N / 128, K / 128]}.
std::tuple<torch::Tensor, torch::Tensor> quantize_weight_to_block_fp8(
    const torch::Tensor& weight);

torch::Tensor matmul_forward(const torch::Tensor& input,
                             const torch::Tensor& weight,
                             const std::optional<torch::Tensor>& bias,
                             torch::Tensor& output_buffer);

torch::Tensor block_fp8_or_bf16_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale_inv,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& output_buffer);

// Pre-quantized activation path (fused SwiGLU+FP8).
torch::Tensor block_fp8_quantized_forward(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale_inv,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& output_buffer);

bool supports_block_fp8_quantized_input(
    bool input_is_parallelized,
    const QuantArgs& quant_args,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale_inv);

}  // namespace xllm::layer::musa
