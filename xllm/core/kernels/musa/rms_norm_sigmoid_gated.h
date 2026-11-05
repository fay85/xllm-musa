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

namespace xllm::kernel::musa {

// FP32 accumulation, per-head D=128, BF16 value/gate/output. Weight [128]
// may be BF16 or FP32. This applies sigmoid to the output gate, never SiLU.
void rms_norm_sigmoid_gated(const torch::Tensor& value,
                            const torch::Tensor& gate,
                            const torch::Tensor& weight,
                            double eps,
                            torch::Tensor& output);

}  // namespace xllm::kernel::musa
