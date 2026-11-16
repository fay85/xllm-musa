/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <glog/logging.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/musa_process_group.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/platform/musa_graph_handle.h"

namespace {

// Diagnostic only: compare the real vocabulary/embedding gather with an
// explicitly zero-padded custom sum. No model selects this candidate yet.
class GatherProbe final {
 public:
  GatherProbe(int32_t rank, int32_t size, int32_t port)
      : device_(torch::kPrivateUse1, static_cast<c10::DeviceIndex>(rank)),
        rank_(rank),
        size_(size),
        group_(rank,
               size,
               size,
               port,
               /*trans=*/false,
               /*host=*/"127.0.0.1",
               /*group_name=*/"tp_group",
               device_) {
    CHECK(group_.custom_allreduce_enabled());
  }

  torch::Tensor run(const torch::Tensor& input, bool use_custom) {
    torch::DeviceGuard guard(device_);
    CHECK_EQ(input.dim(), 2);
    CHECK(input.device() == device_);
    CHECK(input.is_contiguous());
    if (!use_custom) {
      return xllm::parallel_state::gather(input, &group_, /*dim=*/-1);
    }
    CHECK(input.scalar_type() == torch::kBFloat16 ||
          input.scalar_type() == torch::kFloat32);
    torch::Tensor padded =
        torch::zeros({input.size(0), input.size(1) * size_}, input.options());
    padded.narrow(1, rank_ * input.size(1), input.size(1)).copy_(input);
    auto output = group_.allreduce_tp_custom(padded);
    CHECK(output.has_value());
    return std::move(output.value());
  }

  torch::Tensor capture(torch::Tensor input, bool use_custom, int32_t calls) {
    torch::DeviceGuard guard(device_);
    CHECK(!graph_);
    CHECK_GT(calls, 0);
    CHECK_LE(calls, 40);
    input_ = std::move(input);
    outputs_.clear();
    outputs_.reserve(calls);
    for (int32_t i = 0; i < 3; ++i) {
      auto warm = run(input_, use_custom);
    }
    c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
    graph_ = std::make_unique<xllm::MusaGraphHandle>();
    graph_->capture_begin();
    for (int32_t i = 0; i < calls; ++i) {
      outputs_.emplace_back(run(input_, use_custom));
    }
    graph_->capture_end();
    xllm::finalize_musa_custom_allreduce_graphs();
    graph_->replay();
    return outputs_.back();
  }

  torch::Tensor replay() {
    torch::DeviceGuard guard(device_);
    CHECK(graph_);
    graph_->replay();
    return outputs_.back();
  }

  void release() {
    torch::DeviceGuard guard(device_);
    c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
    graph_.reset();
    outputs_.clear();
    input_ = torch::Tensor{};
  }

 private:
  torch::Device device_;
  int32_t rank_;
  int32_t size_;
  xllm::ProcessGroupImpl group_;
  torch::Tensor input_;
  std::vector<torch::Tensor> outputs_;
  std::unique_ptr<xllm::MusaGraphHandle> graph_;
};

}  // namespace

PYBIND11_MODULE(v41_gather_probe_test_module, module) {
  pybind11::class_<GatherProbe>(module, "GatherProbe")
      .def(pybind11::init<int32_t, int32_t, int32_t>())
      .def("run",
           &GatherProbe::run,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("capture",
           &GatherProbe::capture,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("replay",
           &GatherProbe::replay,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("release",
           &GatherProbe::release,
           pybind11::call_guard<pybind11::gil_scoped_release>());
}
