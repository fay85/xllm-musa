/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/layers/musa/deepseek_v41/deep_ep_ll_bridge.h"
#include "core/layers/musa/deepseek_v41/w4a8_experts.h"

namespace {

std::pair<torch::Tensor, torch::Tensor> routed_experts(
    xllm::layer::DeepseekV41DeepEpLlBridge& bridge,
    const xllm::layer::DeepseekV41W4A8Experts& experts,
    const torch::Tensor& input,
    const torch::Tensor& ids,
    const torch::Tensor& weights) {
  auto [received, counts] = bridge.dispatch(input, ids);
  torch::Tensor output = experts.forward_dispatched(received, counts);
  return {bridge.combine(output, ids, weights), counts};
}

}  // namespace

PYBIND11_MODULE(v41_deep_ep_w4a8_test_module, module) {
  pybind11::class_<xllm::layer::DeepseekV41W4A8Experts>(module, "Experts")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          int64_t,
                          double,
                          std::string,
                          double,
                          double,
                          bool>())
      .def("forward", &xllm::layer::DeepseekV41W4A8Experts::forward)
      .def("forward_dispatched",
           &xllm::layer::DeepseekV41W4A8Experts::forward_dispatched);
  pybind11::class_<xllm::layer::DeepseekV41DeepEpLlBridge>(module, "Bridge")
      .def(pybind11::init<pybind11::object, int64_t, int64_t>())
      .def("forward", &routed_experts);
  module.def("quant_dispatched",
             &xllm::kernel::musa::deepseek_v41_w4a8_quant_dispatched);
}
