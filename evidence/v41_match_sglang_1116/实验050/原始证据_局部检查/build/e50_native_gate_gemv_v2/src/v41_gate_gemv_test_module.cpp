/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include <cstdint>
#include <optional>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_ops_api.h"

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
}
