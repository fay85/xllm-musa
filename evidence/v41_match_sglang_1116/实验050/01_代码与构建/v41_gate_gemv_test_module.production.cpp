/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"

namespace {
torch::Tensor gate_gemv(const torch::Tensor &input,
                        const torch::Tensor &weight) {
  return xllm::kernel::musa::deepseek_v4_gate_gemv(input, weight,
                                                   /*output=*/std::nullopt);
}

torch::Tensor gate_reference(const torch::Tensor &input,
                             const torch::Tensor &fp32_weight) {
  return torch::linear(input.to(torch::kFloat32), fp32_weight);
}
} // namespace

PYBIND11_MODULE(v41_gate_gemv_test_module, module) {
  module.def("gemv", &gate_gemv);
  module.def("reference", &gate_reference);
  module.def("routing", &xllm::kernel::musa::deepseek_v41_gate_topk);
  using Gate = xllm::layer::DeepseekV41Gate;
  pybind11::class_<Gate, std::shared_ptr<Gate>>(module, "Gate")
      .def(pybind11::init([](torch::Tensor weight, torch::Tensor bias,
                             int64_t topk, bool normalize, double scale,
                             bool use_simt, bool cache_fp32) {
        return std::make_shared<Gate>(
            std::move(weight), std::move(bias),
            /*vision_bias=*/std::nullopt,
            /*image_token_id=*/129264, topk, normalize, scale, cache_fp32,
            /*use_bf16_fp32_projection=*/false, use_simt);
      }))
      .def("forward", [](const Gate &gate, const torch::Tensor &input) {
        return gate.forward(input, /*input_ids=*/std::nullopt);
      });
}
