/* Copyright 2026 The xLLM Authors.

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
#include <vector>

namespace xllm {

struct SampleOutput;
class SamplingParameters;

namespace spec_verify {

void expand_verify_sampling_params(SamplingParameters& sampling_params,
                                   int32_t num_val_tokens,
                                   int32_t total_num_val_tokens,
                                   const torch::Device& device,
                                   bool repeat_unique_history = true);

}  // namespace spec_verify

// Verifies request-major target rows and samples accepted tokens from logits.
SampleOutput run_rejection_sampling(const SamplingParameters& sampling_params,
                                    const torch::Tensor& draft_token_ids,
                                    const torch::Tensor& draft_logits,
                                    const torch::Tensor& target_logits,
                                    const torch::Tensor& target_hidden_states);

}  // namespace xllm
