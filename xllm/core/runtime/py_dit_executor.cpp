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

#include "core/runtime/py_dit_executor.h"

#include <Python.h>
#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/runtime/py_dit_types.h"
#include "models/py_model_helper.h"

namespace py = pybind11;
namespace xllm {
namespace {
constexpr const char* kGroupKinds[] = {"tp",
                                       "sp",
                                       "cfg",
                                       "vae",
                                       "text_encoder_tp"};
void clear_python_object(py::object& object) {
  if (!object) {
    return;
  }
  if (!Py_IsInitialized()) {
    // CPython has already torn down its GIL; leaking at process exit beats
    // decref'ing during static destruction.
    (void)object.release();
    return;
  }
  py::gil_scoped_acquire gil;
  object = py::object();
}

}  // namespace

PyDiTExecutor::PyDiTExecutor(const DiTModelContext& context)
    : context_(context),
      parallel_args_(context.get_parallel_args()),
      options_(context.get_tensor_options()),
      device_(options_.device()),
      dtype_(dtype_to_string(options_)),
      model_type_(context.model_type()) {
  CHECK(context.get_dit_config().selected_policy == PolicyType::None)
      << "The Python Joyo V2 backend does not support DiT cache policies; use "
         "None.";
  ensure_python_interpreter();
  py::gil_scoped_acquire gil;
  ensure_xllm_dit_runtime_module();
  register_python_process_groups();
}

PyDiTExecutor::~PyDiTExecutor() { clear_python_object(py_executor_); }

void PyDiTExecutor::register_python_process_groups() {
  const std::string device_str = c10::str(device_);
  const int32_t global_rank = parallel_args_.rank();
  const int32_t global_world_size = parallel_args_.world_size();

  // Register all recorded layouts, including explicit singleton groups.
  // Every worker must create these groups in the same order.
  std::vector<std::pair<std::string, const std::vector<std::vector<int32_t>>*>>
      to_register;
  for (const char* kind : kGroupKinds) {
    const std::string group_name = std::string("dit_") + kind;
    auto it = parallel_args_.python_group_layouts_.find(group_name);
    if (it != parallel_args_.python_group_layouts_.end()) {
      to_register.emplace_back(group_name, &it->second);
    }
  }
  if (to_register.empty()) {
    return;
  }

  CHECK(!parallel_args_.python_rendezvous_host_.empty())
      << "python DiT backend requires the rendezvous endpoint on ParallelArgs";
  CHECK_GT(parallel_args_.python_rendezvous_port_, 0);
  py::object init_group =
      py::module_::import("xllm.python.distributed").attr("init_group");
  // DiT ranks can reach Python registration at different times during loading.
  py::object group_timeout = py::module_::import("datetime")
                                 .attr("timedelta")(py::arg("minutes") = 30);

  for (const auto& [group_name, layout] : to_register) {
    // Rank of this process inside the group that contains it.
    int32_t rank_in_group = -1;
    for (const auto& ranks : *layout) {
      auto found = std::find(ranks.begin(), ranks.end(), global_rank);
      if (found != ranks.end()) {
        rank_in_group = static_cast<int32_t>(found - ranks.begin());
        break;
      }
    }
    CHECK_GE(rank_in_group, 0)
        << "python group layout " << group_name
        << " does not contain global rank " << global_rank;
    init_group(py::str(group_name),
               parallel_args_.python_rendezvous_host_,
               parallel_args_.python_rendezvous_port_,
               py::cast(*layout),
               rank_in_group,
               device_str,
               global_rank,
               global_world_size,
               py::arg("timeout") = group_timeout);
  }
}

void PyDiTExecutor::load_model(std::unique_ptr<DiTModelLoader> loader) {
  py::gil_scoped_acquire gil;
  ensure_xllm_weight_loader_module();
  py::dict config;
  config["pipeline"] = model_type_;
  config["device"] = c10::str(device_);
  config["dtype"] = dtype_;
  config["dit_sp_group"] = "dit_sp";
  config["dit_vae_group"] = "dit_vae";
  config["dit_text_encoder_tp_group"] = "dit_text_encoder_tp";
  py::dict components;
  // StateDict wrappers borrow from these loaders only during synchronous
  // create.
  std::vector<std::unique_ptr<DiTFolderLoader>> held_loaders;
  std::shared_ptr<Tokenizer> tokenizer;
  if (loader->has_component("text_encoder")) {
    if (loader->has_component("tokenizer")) {
      auto tokenizer_loader = loader->take_component_loader("tokenizer");
      tokenizer = tokenizer_loader->tokenizer();
    }
  }
  for (const char* name :
       {"transformer", "vae", "audio_vae", "scheduler", "text_encoder"}) {
    if (!loader->has_component(name)) {
      continue;
    }
    auto component = loader->take_component_loader(name);
    py::dict spec;
    spec["config"] = py::module_::import("json").attr("loads")(
        component->component_config().dump());
    if (std::string(name) == "text_encoder") {
      if (tokenizer == nullptr) {
        tokenizer = component->tokenizer();
      }
      CHECK(tokenizer != nullptr) << "JoyOV2 text tokenizer is not loaded";
      spec["tokenizer"] = py::cast(tokenizer);
      spec["class_name"] = "JoyoTextEncoder";
    } else {
      const auto& json = component->component_config();
      CHECK(json.contains("_class_name"))
          << name << " config is missing _class_name";
      spec["class_name"] = json.at("_class_name").get<std::string>();
    }
    if (std::string(name) != "scheduler") {
      py::list weights;
      for (const auto& state : component->get_state_dicts()) {
        weights.append(
            py::cast(PyStateDict(state.get()), py::return_value_policy::move));
      }
      spec["state_dicts"] = weights;
    }
    components[name] = spec;
    held_loaders.emplace_back(std::move(component));
  }
  config["components"] = components;
  py_executor_ = py::module_::import("xllm.python.model_executor.dit_executor")
                     .attr("DiTPipelineExecutor")
                     .attr("create")(config);
}

DiTForwardOutput PyDiTExecutor::forward(const DiTForwardInput& input) {
  torch::NoGradGuard no_grad;
  COUNTER_INC(num_model_execution_total_eager);
  py::gil_scoped_acquire gil;
  CHECK(py_executor_) << "Python DiT executor is not initialized";
  // Copy the envelope so any Python reference remains valid; Tensor storage is
  // shared.
  return py_executor_
      .attr("execute")(py::cast(input, py::return_value_policy::copy))
      .cast<DiTForwardOutput>();
}

}  // namespace xllm
