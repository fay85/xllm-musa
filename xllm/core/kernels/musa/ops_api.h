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

#include <torch/torch.h>

#include <optional>

namespace xllm::kernel::musa {

// Caller-owned output storage keeps graph-captured addresses stable.
struct MatmulParams {
  torch::Tensor a;
  torch::Tensor b;
  std::optional<torch::Tensor> bias;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct Fp8BlockMatmulParams {
  torch::Tensor a;
  torch::Tensor b;
  torch::Tensor a_scale;
  torch::Tensor b_scale;
  torch::ScalarType output_dtype;
  std::optional<torch::Tensor> output;
};

torch::Tensor matmul(MatmulParams& params);

torch::Tensor fp8_block_matmul(Fp8BlockMatmulParams& params);

}  // namespace xllm::kernel::musa
