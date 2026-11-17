/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <pybind11/embed.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "core/layers/musa/deepseek_v41/deep_ep_ll_bridge.h"

namespace xllm::layer {

// Owns the frozen DeepEP buffer and CPU bootstrap group for one native worker.
// Python is used only for the vendor communication ABI: model forward, expert
// math and graph execution remain native. The runtime must outlive its graphs.
// A separate TCP rendezvous is required; this does not replace xLLM's TP group.
class DeepseekV41DeepEpLlRuntime final {
 public:
  DeepseekV41DeepEpLlRuntime(int64_t rank,
                             int64_t world_size,
                             int64_t device,
                             int64_t capacity,
                             int64_t hidden,
                             int64_t experts,
                             const std::string& init_method)
      : capacity_(capacity), hidden_(hidden), experts_(experts) {
    CHECK_GT(world_size, 0);
    CHECK_GE(rank, 0);
    CHECK_LT(rank, world_size);
    CHECK_EQ(experts % world_size, 0);
    CHECK_GT(capacity_, 0);
    CHECK_GT(hidden_, 0);
    CHECK(!init_method.empty());
    initialize_interpreter();
    pybind11::gil_scoped_acquire gil;
    pybind11::module_ torch_musa = pybind11::module_::import("torch_musa");
    torch_musa.attr("set_device")(device);
    const torch::Tensor dispatch_probe =
        torch::empty({0},
                     torch::TensorOptions().device(
                         torch::Device(torch::kPrivateUse1, device)));
    CHECK(dispatch_probe.key_set().has(c10::DispatchKey::PrivateUse1))
        << "Embedded DeepEP requires libmusa_python.so to be loaded before "
           "the native Torch libraries (set the documented LD_PRELOAD order); "
           "reject CUDA-tagged MUSA tensors before communicator bootstrap";
    distributed_ = pybind11::module_::import("torch.distributed");
    CHECK(!distributed_.attr("is_initialized")().cast<bool>())
        << "Native DeepEP bootstrap requires an unused Python default group";
    distributed_.attr("init_process_group")(
        "gloo",
        pybind11::arg("init_method") = init_method,
        pybind11::arg("rank") = rank,
        pybind11::arg("world_size") = world_size,
        pybind11::arg("timeout") =
            pybind11::module_::import("datetime")
                .attr("timedelta")(pybind11::arg("seconds") = 180));
    group_ = distributed_.attr("group").attr("WORLD");
    pybind11::object buffer_type =
        pybind11::module_::import("deep_ep").attr("Buffer");
    const int64_t rdma_bytes = buffer_type
                                   .attr("get_low_latency_rdma_size_hint")(
                                       capacity_, hidden_, world_size, experts_)
                                   .cast<int64_t>();
    buffer_ =
        buffer_type(group_,
                    pybind11::arg("num_rdma_bytes") = rdma_bytes,
                    pybind11::arg("low_latency_mode") = true,
                    pybind11::arg("num_qps_per_rank") = experts_ / world_size);
    buffer_.attr("clean_low_latency_buffer")(capacity_, hidden_, experts_);
    barrier();
    bridge_ = std::make_unique<DeepseekV41DeepEpLlBridge>(
        buffer_, capacity_, experts_);
    LOG(INFO) << "Native V4.1 DeepEP LL initialized, rank=" << rank
              << ", world=" << world_size << ", capacity=" << capacity_
              << ", experts=" << experts_ << ", rdma_bytes=" << rdma_bytes;
  }

  ~DeepseekV41DeepEpLlRuntime() {
    pybind11::gil_scoped_acquire gil;
    bridge_.reset();
    buffer_ = pybind11::object();
    group_ = pybind11::object();
    distributed_ = pybind11::object();
    // The process owns interpreter/default-group shutdown. Never finalize
    // Python while xLLM may still hold tensor bindings or profiler callbacks.
  }

  DeepseekV41DeepEpLlRuntime(const DeepseekV41DeepEpLlRuntime&) = delete;
  DeepseekV41DeepEpLlRuntime& operator=(const DeepseekV41DeepEpLlRuntime&) =
      delete;

  DeepseekV41DeepEpLlBridge& bridge() { return *bridge_; }
  int64_t capacity() const { return capacity_; }

  // Only outside capture, with every EP rank participating. The local vendor
  // reset contains no barrier: both sides of it must be quiesced explicitly.
  void reset_for_capture() {
    pybind11::gil_scoped_acquire gil;
    barrier();
    buffer_.attr("reset_low_latency_flags")(
        capacity_, hidden_, experts_, pybind11::arg("start_index") = 0);
    barrier();
  }

  void barrier() {
    pybind11::gil_scoped_acquire gil;
    pybind11::module_::import("torch_musa").attr("synchronize")();
    distributed_.attr("barrier")(pybind11::arg("group") = group_);
  }

 private:
  static void initialize_interpreter() {
    static std::once_flag once;
    std::call_once(once, []() {
      if (!Py_IsInitialized()) {
        pybind11::initialize_interpreter(/*init_signal_handlers=*/false);
        PyEval_SaveThread();
      }
    });
  }

  int64_t capacity_;
  int64_t hidden_;
  int64_t experts_;
  pybind11::object distributed_;
  pybind11::object group_;
  pybind11::object buffer_;
  std::unique_ptr<DeepseekV41DeepEpLlBridge> bridge_;
};

}  // namespace xllm::layer
