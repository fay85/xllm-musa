/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/layers/musa/deepseek_v41/deep_ep_ll_bridge.h"

namespace {

std::pair<torch::Tensor, torch::Tensor> reference_experts(
    xllm::layer::DeepseekV41DeepEpLlBridge& bridge,
    const torch::Tensor& input,
    const torch::Tensor& expert_ids,
    const torch::Tensor& expert_weights,
    const torch::Tensor& expert_factors) {
  auto dispatched = bridge.dispatch(input, expert_ids);
  torch::Tensor expert_output = dispatched.first * expert_factors;
  return {bridge.combine(expert_output, expert_ids, expert_weights),
          dispatched.second};
}

}  // namespace

PYBIND11_MODULE(v41_deep_ep_ll_bridge_test_module, module) {
  pybind11::class_<xllm::layer::DeepseekV41DeepEpLlBridge>(module, "Bridge")
      .def(pybind11::init<pybind11::object, int64_t, int64_t>())
      .def("dispatch", &xllm::layer::DeepseekV41DeepEpLlBridge::dispatch)
      .def("combine", &xllm::layer::DeepseekV41DeepEpLlBridge::combine)
      .def("reference_experts", &reference_experts);
}
