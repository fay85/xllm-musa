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

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::mlu::bangc {

// Standalone MHC Pre + RMSNorm. Uses external pre_in to compress residual;
// new_pre is an output for the next boundary, never the compression weight.
// BF16: residual [T,4,5120], hc_fn [24,20480], gamma [5120].
// FP32: pre_in [T,4], hc_scale [3], hc_base [24].
// Returns norm_out BF16 [T,5120], new_pre/post FP32 [T,4], comb FP32 [T,4,4].
// Supports natural alignment, empty inputs, current streams and graph capture.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_mhc_pre(const torch::Tensor& residual,
              const torch::Tensor& pre_in,
              const torch::Tensor& hc_fn,
              const torch::Tensor& hc_scale,
              const torch::Tensor& hc_base,
              const torch::Tensor& gamma,
              int64_t sinkhorn_iter,
              double hc_eps,
              double norm_eps);

}  // namespace xllm::kernel::mlu::bangc
