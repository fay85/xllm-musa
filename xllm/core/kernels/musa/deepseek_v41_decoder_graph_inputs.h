/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND. See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once
#include <torch/torch.h>

#include <array>
namespace xllm::kernel::musa {
// Same-device, bit-preserving copy of q1 decoder-tail graph inputs.
// Order: BF16 output/residual, FP32 pre/post/comb, I32 or I64 token.
void deepseek_v41_copy_decoder_graph_inputs(
    const std::array<torch::Tensor, 6>& source,
    const std::array<torch::Tensor, 6>& destination);
}  // namespace xllm::kernel::musa
