/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <utility>

namespace xllm::layer {

// Adapts the frozen MUSA DeepEP Python ABI without converting GPU tensors to
// host data. Bootstrap, buffer allocation and ring reset belong outside graph
// capture. This object supports one outstanding dispatch/combine pair; callers
// must not share it between concurrent streams or model executions.
//
// This is an isolated integration primitive, not an enabled model execution
// path. The communication buffer must outlive every replay of a captured graph.
class DeepseekV41DeepEpLlBridge final {
 public:
  explicit DeepseekV41DeepEpLlBridge(pybind11::object buffer,
                                     int64_t capacity,
                                     int64_t experts)
      : buffer_(std::move(buffer)), capacity_(capacity), experts_(experts) {
    CHECK_GT(capacity_, 0);
    CHECK_GT(experts_, 0);
  }

  ~DeepseekV41DeepEpLlBridge() {
    // Release Python references while the GIL is held. The interpreter must
    // remain alive until all bridges and captured executions are destroyed.
    pybind11::gil_scoped_acquire gil;
    handle_ = pybind11::object();
    buffer_ = pybind11::object();
  }

  DeepseekV41DeepEpLlBridge(const DeepseekV41DeepEpLlBridge&) = delete;
  DeepseekV41DeepEpLlBridge& operator=(const DeepseekV41DeepEpLlBridge&) =
      delete;

  std::pair<torch::Tensor, torch::Tensor> dispatch(
      const torch::Tensor& input,
      const torch::Tensor& expert_ids) {
    CHECK(!pending_) << "The previous dispatch must be combined first";
    CHECK_EQ(input.dim(), 2);
    CHECK_EQ(input.scalar_type(), torch::kBFloat16);
    CHECK_EQ(expert_ids.scalar_type(), torch::kInt64);
    CHECK_EQ(input.size(0), expert_ids.size(0));
    CHECK_LE(input.size(0), capacity_);
    CHECK(input.device() == expert_ids.device());
    CHECK(!input.device().is_cpu());
    pybind11::gil_scoped_acquire gil;
    pybind11::tuple result = buffer_.attr("low_latency_dispatch")(
        input,
        expert_ids,
        capacity_,
        experts_,
        pybind11::arg("use_fp8") = false,
        pybind11::arg("async_finish") = true);
    CHECK_EQ(result.size(), 5);
    result[3].attr("current_stream_wait")();
    handle_ = result[2];
    pending_ = true;
    return {result[0].cast<torch::Tensor>(), result[1].cast<torch::Tensor>()};
  }

  torch::Tensor combine(const torch::Tensor& expert_output,
                        const torch::Tensor& expert_ids,
                        const torch::Tensor& expert_weights) {
    CHECK(pending_) << "Combine requires this bridge's outstanding dispatch";
    CHECK_EQ(expert_output.scalar_type(), torch::kBFloat16);
    CHECK_EQ(expert_weights.scalar_type(), torch::kFloat32);
    CHECK(expert_ids.sizes() == expert_weights.sizes());
    CHECK(expert_output.device() == expert_ids.device());
    CHECK(expert_output.device() == expert_weights.device());
    pybind11::gil_scoped_acquire gil;
    pybind11::tuple result = buffer_.attr("low_latency_combine")(
        expert_output,
        expert_ids,
        expert_weights,
        handle_,
        pybind11::arg("async_finish") = true);
    CHECK_EQ(result.size(), 3);
    result[1].attr("current_stream_wait")();
    torch::Tensor output = result[0].cast<torch::Tensor>();
    handle_ = pybind11::object();
    pending_ = false;
    return output;
  }

 private:
  pybind11::object buffer_;
  pybind11::object handle_;
  int64_t capacity_;
  int64_t experts_;
  bool pending_ = false;
};

}  // namespace xllm::layer
