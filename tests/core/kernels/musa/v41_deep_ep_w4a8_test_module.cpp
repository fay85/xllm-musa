/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "core/framework/parallel_state/musa/musa_native_collectives.h"
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
  pybind11::class_<xllm::MusaNativeCollectives>(module, "Collectives")
      .def(pybind11::init([](int32_t rank, uint16_t port) {
        c10d::TCPStoreOptions options;
        options.port = port;
        options.isServer = rank == 0;
        options.numWorkers = 8;
        options.waitWorkers = true;
        options.timeout = std::chrono::seconds(60);
        auto store = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", options);
        return std::make_unique<xllm::MusaNativeCollectives>(
            rank,
            /*world_size=*/8,
            /*group_name=*/"e44_mixed_graph",
            torch::Device(torch::kPrivateUse1, rank),
            store);
      }))
      .def("allgather", &xllm::MusaNativeCollectives::allgather)
      .def("broadcast", &xllm::MusaNativeCollectives::broadcast);
}
