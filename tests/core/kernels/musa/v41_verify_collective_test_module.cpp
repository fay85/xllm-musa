/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <glog/logging.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <string>

#include "core/framework/parallel_state/musa_process_group.h"
#include "core/platform/musa_graph_handle.h"

namespace {

// Uses the production communicator and its graph buffer registration. The
// test owns one group per rank and keeps it alive until every graph is gone.
class VerifyCollective final {
 public:
  VerifyCollective(int32_t rank, int32_t world_size, int32_t port)
      : device_(torch::kPrivateUse1, static_cast<c10::DeviceIndex>(rank)),
        group_(rank,
               world_size,
               world_size,
               port,
               /*trans=*/false,
               /*host=*/"127.0.0.1",
               /*group_name=*/"tp_group",
               device_) {
    CHECK(group_.custom_allreduce_enabled());
  }

  torch::Tensor mccl(torch::Tensor input) {
    torch::DeviceGuard guard(device_);
    torch::Tensor output = input.clone();
    group_.allreduce(output);
    return output;
  }

  torch::Tensor custom(torch::Tensor input) {
    torch::DeviceGuard guard(device_);
    auto output = group_.allreduce_tp_custom(input);
    CHECK(output.has_value());
    return std::move(output.value());
  }

  torch::Tensor capture(torch::Tensor input) {
    torch::DeviceGuard guard(device_);
    CHECK(!graph_);
    input_ = std::move(input);
    for (int32_t index = 0; index < 3; ++index) {
      output_ = custom(input_);
    }
    c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
    graph_ = std::make_unique<xllm::MusaGraphHandle>();
    graph_->capture_begin();
    output_ = custom(input_);
    graph_->capture_end();
    xllm::finalize_musa_custom_allreduce_graphs();
    graph_->replay();
    return output_;
  }

  torch::Tensor replay() {
    CHECK(graph_);
    torch::DeviceGuard guard(device_);
    graph_->replay();
    return output_;
  }

  void release_graph() {
    torch::DeviceGuard guard(device_);
    c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
    graph_.reset();
    input_ = torch::Tensor{};
    output_ = torch::Tensor{};
  }

 private:
  torch::Device device_;
  xllm::ProcessGroupImpl group_;
  torch::Tensor input_;
  torch::Tensor output_;
  std::unique_ptr<xllm::MusaGraphHandle> graph_;
};

}  // namespace

PYBIND11_MODULE(v41_verify_collective_test_module, module) {
  pybind11::class_<VerifyCollective>(module, "VerifyCollective")
      .def(pybind11::init<int32_t, int32_t, int32_t>())
      .def("mccl",
           &VerifyCollective::mccl,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("custom",
           &VerifyCollective::custom,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("capture",
           &VerifyCollective::capture,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("replay",
           &VerifyCollective::replay,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("release_graph",
           &VerifyCollective::release_graph,
           pybind11::call_guard<pybind11::gil_scoped_release>());
}
