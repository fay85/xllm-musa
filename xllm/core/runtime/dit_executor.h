/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/batch/dit_batch.h"
#include "core/framework/model/dit_model.h"
#include "core/runtime/dit_forward_params.h"
#include "core/runtime/options.h"

namespace xllm {

class DiTExecutor {
 public:
  virtual ~DiTExecutor() = default;
  DiTForwardInput prepare_inputs(DiTBatch& batch);
  virtual DiTForwardOutput forward(const DiTForwardInput& input) = 0;
};

class NativeDiTExecutor final : public DiTExecutor {
 public:
  NativeDiTExecutor(DiTModel* model, const runtime::Options& options);
  DiTForwardOutput forward(const DiTForwardInput& input) override;

 private:
  // Non-owning: the worker retains the native model.
  DiTModel* model_;
};

}  // namespace xllm
