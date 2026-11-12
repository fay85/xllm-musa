/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/types.h>

namespace xllm::kernel::musa {

// Postprocess muDNN top-k without changing selected expert IDs or their order.
// scores: contiguous FP32 [rows, 288]; selected: contiguous INT64 [rows, 8].
// Caller-owned outputs: FP32 weights and INT32 IDs, both [rows, 8].
void moe_route_post(const torch::Tensor& scores,
                    const torch::Tensor& selected,
                    torch::Tensor& weights,
                    torch::Tensor& ids,
                    bool renormalize,
                    double scale);

}  // namespace xllm::kernel::musa
