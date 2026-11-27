/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/py_runtime_module.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/attention_metadata.h"
#include "core/runtime/py_attention_metadata.h"
#include "core/runtime/py_executor_impl.h"
#include "core/runtime/py_input_batch.h"
#include "models/llm/py_causal_lm.h"
#include "models/py_model_helper.h"
#include "platform/npu/device_capture_lock.h"

namespace py = pybind11;

namespace xllm {
namespace {

class PyRuntimeModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter(/*init_signal_handlers=*/false);
      thread_state_ = PyEval_SaveThread();
    }
  }

  // The registered module owns pybind11 types. Finalize the interpreter with
  // the GIL held so those types are released before the process tears down its
  // statics; otherwise pybind11 aborts the exit with "dec_ref() is being called
  // while the GIL is either not held or invalid".
  static void TearDownTestSuite() {
    if (thread_state_ == nullptr) {
      return;
    }
    PyEval_RestoreThread(thread_state_);
    thread_state_ = nullptr;
    py::finalize_interpreter();
  }

  static PyThreadState* thread_state_;
};

PyThreadState* PyRuntimeModuleTest::thread_state_ = nullptr;

TEST_F(PyRuntimeModuleTest, RegistersInRunningInterpreterAndPreservesIdentity) {
  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  py::module_ module = py::module_::import("xllm_runtime");
  py::object metadata_type = module.attr("InputBatchMetadataView");

  ensure_xllm_runtime_module();
  EXPECT_TRUE(module.is(py::module_::import("xllm_runtime")));
  EXPECT_TRUE(metadata_type.is(module.attr("InputBatchMetadataView")));
  EXPECT_TRUE(py::hasattr(module, "AttentionMetadataView"));
  EXPECT_TRUE(py::hasattr(module, "ExpandedDecodeMetadataView"));
  EXPECT_TRUE(py::hasattr(module, "LayerLoadSynchronizer"));
}

TEST_F(PyRuntimeModuleTest,
       CaptureGuardSharesDeviceLockAndReleasesOnException) {
  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  constexpr c10::DeviceIndex kDeviceIndex = 0;
  auto& capture_mutex =
      npu::DeviceCaptureLock::get_instance().get_lock(kDeviceIndex);
  py::dict globals;
  globals["guard"] = py::module_::import("xllm_runtime")
                         .attr("DeviceCaptureGuard")(kDeviceIndex);
  globals["try_prepare_lock"] = py::cpp_function([&capture_mutex]() {
    return std::async(std::launch::async,
                      [&capture_mutex]() {
                        std::unique_lock<std::mutex> lock(capture_mutex,
                                                          std::try_to_lock);
                        return lock.owns_lock();
                      })
        .get();
  });
  py::exec(R"(
assert try_prepare_lock()
with guard:
    assert not try_prepare_lock()
assert try_prepare_lock()
try:
    with guard:
        assert not try_prepare_lock()
        raise ValueError("capture failed")
except ValueError:
    pass
assert try_prepare_lock()
)",
           globals);
}

TEST_F(PyRuntimeModuleTest, CaptureGuardReleasesGilWhileWaitingForPreparation) {
  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  constexpr c10::DeviceIndex kDeviceIndex = 0;
  auto& capture_mutex =
      npu::DeviceCaptureLock::get_instance().get_lock(kDeviceIndex);
  py::object guard = py::module_::import("xllm_runtime")
                         .attr("DeviceCaptureGuard")(kDeviceIndex);
  std::promise<void> prepare_started;
  auto prepare_started_future = prepare_started.get_future();
  std::promise<void> gil_available;
  auto gil_available_future = gil_available.get_future();
  auto prepare = std::async(std::launch::async, [&]() {
    std::unique_lock<std::mutex> lock(capture_mutex);
    prepare_started.set_value();
    return gil_available_future.wait_for(std::chrono::seconds(5)) ==
           std::future_status::ready;
  });
  prepare_started_future.wait();
  auto python_thread = std::async(std::launch::async, [&gil_available]() {
    py::gil_scoped_acquire thread_gil;
    gil_available.set_value();
  });
  guard.attr("__enter__")();
  guard.attr("__exit__")(py::none(), py::none(), py::none());
  const bool released_gil = prepare.get();
  {
    py::gil_scoped_release release;
    python_thread.get();
  }
  EXPECT_TRUE(released_gil);
}

TEST_F(PyRuntimeModuleTest, CollectivesWithoutActiveModelReturnNone) {
  py::gil_scoped_acquire gil;
  py::module_::import("torch");
  ensure_xllm_runtime_module();
  py::module_ module = py::module_::import("xllm_runtime");
  torch::Tensor input = torch::arange(4, torch::kFloat32);
  // A collective without an owning PyCausalLM must report that it did not run,
  // so the Python layer can fall back to its own process group instead of
  // consuming a result that was never produced.
  EXPECT_TRUE(module.attr("tp_all_reduce")(input).is_none());
  EXPECT_TRUE(module.attr("tp_all_gather")(input, 0).is_none());
  EXPECT_TRUE(module.attr("tp_all_to_all")(input, 1).is_none());
  EXPECT_TRUE(
      module.attr("dp_all_gather")(input, std::vector<int32_t>{4}).is_none());
  EXPECT_TRUE(module.attr("moe_tp_all_reduce")(input).is_none());
  EXPECT_TRUE(module.attr("moe_ep_all_reduce")(input).is_none());
}

TEST_F(PyRuntimeModuleTest, ExposesInputBatchMetadata) {
  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  ModelInputParams params;
  params.execution_batch.num_reqs = 2;
  params.execution_batch.num_tokens = 3;
  params.execution_batch.num_scheduled_tokens = {1, 2};
  py::object view = py::cast(PyInputBatchMetadataView(params));
  EXPECT_EQ(view.attr("num_reqs").cast<int32_t>(), 2);
  EXPECT_EQ(view.attr("num_tokens").cast<int64_t>(), 3);
  EXPECT_EQ(view.attr("num_scheduled_tokens").cast<std::vector<int32_t>>(),
            (std::vector<int32_t>{1, 2}));
}

TEST_F(PyRuntimeModuleTest,
       ExposesPreparedLinearStateReadsWithoutModelDispatch) {
  py::gil_scoped_acquire gil;
  py::module_::import("torch");
  ensure_xllm_runtime_module();
  ModelInputParams params;
  params.embedding.linear_state_ids = {3};
  params.embedding.linear_state_read_ids = {2};
  params.embedding.linear_state_indices = torch::tensor({3}, torch::kInt32);
  params.embedding.linear_state_read_indices =
      torch::tensor({2}, torch::kInt32);
  auto metadata = std::make_shared<layer::AttentionMetadata>();
  for (bool is_spec_verify : {false, true}) {
    params.is_spec_verify = is_spec_verify;
    params.embedding.linear_state_read_indices =
        params.embedding.linear_state_indices;
    PyAttentionMetadataView legacy(metadata, params);
    EXPECT_TRUE(legacy.linear_state_read_indices().is_none());

    params.embedding.linear_state_read_indices =
        torch::tensor({2}, torch::kInt32);
    PyAttentionMetadataView prepared(metadata, params);
    EXPECT_TRUE(
        prepared.linear_state_read_indices().cast<torch::Tensor>().is_same(
            params.embedding.linear_state_read_indices));
    // Equal warmup values must still retain independent graph input buffers.
    params.embedding.linear_state_read_indices =
        params.embedding.linear_state_indices.clone();
    PyAttentionMetadataView warmup(metadata, params);
    EXPECT_FALSE(warmup.linear_state_read_indices().is_none());
  }
  EXPECT_TRUE(params.to(torch::kCPU)
                  .embedding.linear_state_read_indices.is_same(
                      params.embedding.linear_state_read_indices));
  params.clear_linear_attention_state();
  EXPECT_FALSE(params.embedding.linear_state_read_indices.defined());
}

TEST_F(PyRuntimeModuleTest, RegistersWeightLoaderAndReadsStateDict) {
  py::gil_scoped_acquire gil;
  py::module_::import("torch");
  ensure_xllm_weight_loader_module();
  py::module_ module = py::module_::import("xllm_weight_loader");
  py::object state_dict_type = module.attr("StateDict");
  ensure_xllm_weight_loader_module();
  EXPECT_TRUE(module.is(py::module_::import("xllm_weight_loader")));
  EXPECT_TRUE(state_dict_type.is(module.attr("StateDict")));

  torch::Tensor weight = torch::arange(4, torch::kFloat32);
  StateDict state_dict({{"weight", weight}});
  py::object view = py::cast(PyStateDict(&state_dict));
  EXPECT_TRUE(view.attr("has")("weight").cast<bool>());
  EXPECT_FALSE(view.attr("has")("missing").cast<bool>());
  EXPECT_EQ(view.attr("keys")().cast<std::vector<std::string>>(),
            (std::vector<std::string>{"weight"}));
  EXPECT_TRUE(
      view.attr("get_tensor")("weight").cast<torch::Tensor>().is_same(weight));
}

TEST_F(PyRuntimeModuleTest, PythonExecutorKeepsReplicatedDraftKvUnsharded) {
  py::gil_scoped_acquire gil;
  py::module_::import("torch");
  ensure_xllm_runtime_module();
  py::dict globals;
  globals["initialize_runtime"] = py::cpp_function(&ensure_python_interpreter);
  globals["expect_kv_split_size"] = py::cpp_function(
      [](bool is_draft_engine, const std::string& algorithm, int32_t expected) {
        torch::Device device(torch::kCPU);
        ProcessGroup tp_group(/*rank=*/0, /*world_size=*/4, device);
        ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/4, &tp_group);
        parallel_args.tp_group_ = &tp_group;
        parallel_args.moe_tp_group_ = &tp_group;
        parallel_args.kv_split_size(4);
        parallel_args.python_rendezvous_host_ = "localhost";
        parallel_args.python_rendezvous_port_ = 1;
        ModelArgs model_args;
        model_args.model_type("qwen3");
        ModelContext context(
            parallel_args,
            model_args,
            QuantArgs{},
            torch::TensorOptions().device(device).dtype(torch::kFloat32),
            /*context=*/nullptr);
        PyCausalLM model(context, /*is_vlm=*/false);
        runtime::Options options;
        options.is_draft_engine(is_draft_engine)
            .speculative_algorithm(algorithm);
        PyExecutorImpl executor(&model, model_args, device, options);
        ModelInputParams params;
        params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
        const torch::Tensor tokens = torch::zeros({1}, torch::kLong);
        params.attn_metadata->slot_mapping = tokens;
        std::vector<KVCache> caches;
        const int32_t actual = executor.run(tokens, tokens, caches, params)
                                   .hidden_states.item<int32_t>();
        EXPECT_EQ(actual, expected)
            << "algorithm=" << algorithm
            << ", role=" << (is_draft_engine ? "draft" : "target");
      });
  py::exec(R"(
import torch
from types import SimpleNamespace
from unittest.mock import patch

with patch("xllm.python.platform.current_platform.is_npu", return_value=True):
    initialize_runtime()

with (
    patch("xllm.python.distributed.init_process_group"),
    patch("xllm.python.registry.get_model_class", return_value=lambda _: torch.nn.Identity()),
    patch("xllm.python.model_executor.executor.ModelExecutor", return_value=SimpleNamespace(
        bind_kv_caches=lambda _: None,
        execute=lambda _tokens, _positions, metadata, *_args, **_kwargs: torch.tensor(metadata.kv_split_size),
    )),
):
    expect_kv_split_size(False, "DFlash2", 4)
    expect_kv_split_size(True, "DFlash2", 1)
    expect_kv_split_size(True, "DSpark", 4)
)",
           globals);
}

}  // namespace
}  // namespace xllm
