/* Copyright 2025-2026 The xLLM Authors.

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
#include <torch/types.h>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask);

void apply_sampling_penalties(torch::Tensor& logits,
                              const SamplingParameters& params);

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures);

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p);

void apply_logits_transforms(torch::Tensor& logits,
                             const torch::Tensor& temperatures,
                             const torch::Tensor& top_k,
                             const torch::Tensor& top_p);

torch::Tensor process_logits(
    torch::Tensor& logits,
    const SamplingParameters& params,
    const torch::Tensor& filter_mask = torch::Tensor());

}  // namespace xllm
