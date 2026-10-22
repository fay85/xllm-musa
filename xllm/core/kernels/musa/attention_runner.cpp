/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kernels/musa/attention_runner.h"

#include <c10/core/InferenceMode.h>
#include <glog/logging.h>

#include <limits>
#include <utility>

#include "core/kernels/musa/global_capture_instance.h"

namespace xllm::kernel::musa {

namespace {

thread_local bool piecewise_preparation_active = false;

}  // namespace

void AttentionRunner::run_capture(RunFn run_fn) {
  auto& capture = ::xllm::runtime::musa::GlobalCaptureInstance::get_instance();

  capture.temporarily_end_graph();

  run_fn_ = std::move(run_fn);
  CHECK(run_fn_ != nullptr);

  // Match the CUDA runner behavior: capture records replay state without
  // executing the runner.
  capture.temporarily_begin_graph();
}

void AttentionRunner::run_replay(const AttentionReplayParams& params) {
  CHECK(run_fn_ != nullptr);

  c10::InferenceMode guard(true);
  run_fn_(params);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
prefill_with_optional_piecewise_capture(AttentionRunner::RunFn run_fn,
                                        const torch::Tensor& output) {
  auto& capture = ::xllm::runtime::musa::GlobalCaptureInstance::get_instance();

  if (capture.is_capturing()) {
    AttentionRunner runner;
    runner.run_capture(std::move(run_fn));

    capture.register_attention_runner(std::move(runner));

    // Capture skips the runner and returns the output placeholder so the
    // following graph segment records the same output address.
    return {output, std::nullopt};
  }

  AttentionReplayParams params;
  const int64_t actual_num_tokens = output.size(0);
  CHECK_GE(actual_num_tokens, 0);
  CHECK_LE(actual_num_tokens, std::numeric_limits<uint32_t>::max());
  params.actual_num_tokens = static_cast<uint32_t>(actual_num_tokens);
  return run_fn(params);
}

PiecewisePreparationScope::PiecewisePreparationScope()
    : previous_active_(piecewise_preparation_active) {
  piecewise_preparation_active = true;
}

PiecewisePreparationScope::~PiecewisePreparationScope() {
  piecewise_preparation_active = previous_active_;
}

bool has_padded_piecewise_rows() {
  return piecewise_preparation_active ||
         ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
             .is_capturing();
}

torch::Tensor piecewise_runner_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() ||
      !::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
           .is_capturing()) {
    return tensor;
  }
  return torch::from_blob(
      tensor.data_ptr(),
      tensor.sizes(),
      tensor.strides(),
      [](void* /*unused*/) {},
      tensor.options());
}

}  // namespace xllm::kernel::musa
