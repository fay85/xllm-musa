/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <torch/extension.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <cstdint>
#include <memory>

// The installed SG438 MCCL headers still use these legacy namespace names.
namespace at::musa {
using c10::musa::getCurrentMUSAStream;
using c10::musa::MUSAGuard;
using c10::musa::MUSAStream;
}  // namespace at::musa

#include "core/framework/parallel_state/musa_process_group.h"
#include "core/layers/musa/deepseek_v41/gather_bits.h"
#include "core/platform/musa_graph_handle.h"

namespace {

// Isolated hypothesis only. This is not installed in the production model.
class GatherCandidate final {
 public:
  GatherCandidate(int32_t rank, int32_t port)
      : rank_(rank),
        device_(torch::kPrivateUse1, static_cast<c10::DeviceIndex>(rank)),
        group_(rank,
               /*world_size=*/8,
               /*rank_size=*/8,
               port,
               /*trans=*/false,
               /*host=*/"127.0.0.1",
               /*group_name=*/"tp_group",
               device_) {
    CHECK(group_.custom_allreduce_enabled());
  }

  torch::Tensor forward(const torch::Tensor& input, bool use_custom) {
    torch::DeviceGuard guard(device_);
    CHECK_EQ(input.device(), device_);
    CHECK_EQ(input.dim(), 2);
    CHECK(input.is_contiguous());
    CHECK(input.scalar_type() == torch::kBFloat16 ||
          input.scalar_type() == torch::kFloat32);
    const int64_t width = input.size(1);
    if (input.size(0) == 0) {
      return torch::empty({0, width * 8}, input.options());
    }
    return xllm::layer::deepseek_v41_gather_bf16_bits(
        input, &group_, use_custom);
  }

  torch::Tensor capture(torch::Tensor input, bool use_custom, int32_t calls) {
    torch::DeviceGuard guard(device_);
    CHECK(!graph_);
    CHECK_GT(calls, 0);
    input_ = std::move(input);
    for (int32_t index = 0; index < 3; ++index) {
      output_ = forward(input_, use_custom);
    }
    c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
    graph_ = std::make_unique<xllm::MusaGraphHandle>();
    graph_->capture_begin();
    for (int32_t index = 0; index < calls; ++index) {
      output_ = forward(input_, use_custom);
    }
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
    output_ = torch::Tensor{};
    input_ = torch::Tensor{};
  }

 private:
  int32_t rank_;
  torch::Device device_;
  xllm::ProcessGroupImpl group_;
  torch::Tensor input_;
  torch::Tensor output_;
  std::unique_ptr<xllm::MusaGraphHandle> graph_;
};

}  // namespace

PYBIND11_MODULE(v41_gather_candidate_test_module, module) {
  pybind11::class_<GatherCandidate>(module, "GatherCandidate")
      .def(pybind11::init<int32_t, int32_t>())
      .def("forward",
           &GatherCandidate::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("capture",
           &GatherCandidate::capture,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("replay",
           &GatherCandidate::replay,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("release_graph",
           &GatherCandidate::release_graph,
           pybind11::call_guard<pybind11::gil_scoped_release>());
}
