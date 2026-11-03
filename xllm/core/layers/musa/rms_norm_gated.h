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

#include <optional>
#include <tuple>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm::layer::musa {

class RmsNormGatedImpl final : public torch::nn::Module {
 public:
  RmsNormGatedImpl(int64_t dim,
                   int64_t max_rows,
                   double eps,
                   const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor& input,
                        std::optional<torch::Tensor> gate = std::nullopt);

  // Whether forward_quant_fp8() accepts `input` rows.
  bool supports_quant_fp8(const torch::Tensor& input) const;

  // Gated forward fused with group-128 FP8 quantization of the result.
  // Returns {fp8 [rows, dim], scales [rows, 1]}.
  std::tuple<torch::Tensor, torch::Tensor> forward_quant_fp8(
      const torch::Tensor& input,
      const torch::Tensor& gate);

  void load_state_dict(const StateDict& state_dict);

 private:
  DEFINE_WEIGHT(weight);
  int64_t max_rows_;
  double eps_;
  mutable torch::Tensor output_buf_;
};
TORCH_MODULE(RmsNormGated);

}  // namespace xllm::layer::musa
