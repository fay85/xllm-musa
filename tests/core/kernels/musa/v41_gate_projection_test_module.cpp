/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include <memory>
#include <utility>

#include "core/layers/musa/deepseek_v41/feed_forward.h"

PYBIND11_MODULE(v41_gate_projection_test_module, module) {
  using Gate = xllm::layer::DeepseekV41Gate;
  pybind11::class_<Gate>(module, "Gate")
      .def(pybind11::init([](torch::Tensor weight,
                             torch::Tensor bias,
                             bool cache_fp32,
                             bool use_mate) {
        return std::make_unique<Gate>(std::move(weight),
                                      std::move(bias),
                                      /*vision_bias=*/std::nullopt,
                                      /*image_token_id=*/129264,
                                      /*topk=*/6,
                                      /*normalize=*/true,
                                      /*scale=*/1.5,
                                      cache_fp32,
                                      use_mate);
      }))
      .def("forward", [](const Gate& gate, const torch::Tensor& input) {
        return gate.forward(input, /*input_ids=*/std::nullopt);
      });
  module.def(
      "projection",
      [](const torch::Tensor& input, const torch::Tensor& weight) {
        xllm::kernel::musa::TvmffiStreamGuard stream_guard(input.device());
        return xllm::kernel::musa::deepseek_v4_gate_bf16_fp32(
            input, weight, /*output=*/std::nullopt);
      });
}
