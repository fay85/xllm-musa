/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

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
#include <vector>

#include "core/framework/state_dict/state_dict.h"

namespace xllm::layer::musa {

torch::Tensor fp8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& input_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype);

bool load_fp8_weight_scale(const StateDict& state_dict,
                           int64_t dim,
                           int64_t rank,
                           int64_t world_size,
                           torch::Tensor& weight_scale,
                           bool& weight_scale_is_loaded,
                           const torch::Tensor& weight);

bool load_fused_fp8_weight_scale(const StateDict& state_dict,
                                 const std::vector<std::string>& prefixes,
                                 int64_t dim,
                                 int64_t rank,
                                 int64_t world_size,
                                 std::vector<torch::Tensor>& scale_list,
                                 torch::Tensor& weight_scale,
                                 bool& weight_scale_is_loaded,
                                 const torch::Tensor& weight);

}  // namespace xllm::layer::musa
