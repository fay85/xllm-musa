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

#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <memory>
#include <string>

#include "core/framework/dit_model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/runtime/dit_executor.h"

namespace xllm {

// Runtime bridge only. Python owns its pipeline and every network component.
class __attribute__((visibility("hidden"))) PyDiTExecutor final
    : public DiTExecutor {
 public:
  explicit PyDiTExecutor(const DiTModelContext& context);
  ~PyDiTExecutor() override;
  void load_model(std::unique_ptr<DiTModelLoader> loader);
  DiTForwardOutput forward(const DiTForwardInput& input) override;

 private:
  void register_python_process_groups();
  DiTModelContext context_;
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  torch::Device device_;
  std::string dtype_;
  std::string model_type_;
  pybind11::object py_executor_;
};

}  // namespace xllm
