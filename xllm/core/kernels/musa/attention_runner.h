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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>

namespace xllm::layer {
struct AttentionMetadata;
}

namespace xllm::kernel::musa {

struct AttentionReplayParams {
  uint32_t actual_num_tokens = 0;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
};

class AttentionRunner final {
 public:
  using RunFn =
      std::function<std::tuple<torch::Tensor, std::optional<torch::Tensor>>(
          const AttentionReplayParams&)>;

  AttentionRunner() = default;

  AttentionRunner(const AttentionRunner&) = delete;
  AttentionRunner& operator=(const AttentionRunner&) = delete;

  AttentionRunner(AttentionRunner&&) noexcept = default;
  AttentionRunner& operator=(AttentionRunner&&) noexcept = default;

  void run_capture(RunFn run_fn);

  void run_replay(const AttentionReplayParams& params);

 private:
  RunFn run_fn_;
};

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
prefill_with_optional_piecewise_capture(AttentionRunner::RunFn run_fn,
                                        const torch::Tensor& output);

// Marks the eager warmup forwards that prepare a piecewise prefill capture.
// They run on the same bucket rows as the captured graph.
class PiecewisePreparationScope final {
 public:
  PiecewisePreparationScope();
  ~PiecewisePreparationScope();

  PiecewisePreparationScope(const PiecewisePreparationScope&) = delete;
  PiecewisePreparationScope& operator=(const PiecewisePreparationScope&) =
      delete;

 private:
  bool previous_active_ = false;
};

// True while this thread captures a piecewise prefill graph or runs its
// warmups. Rows past the live tokens are then bucket padding.
bool has_padded_piecewise_rows();

// While a piecewise graph is being captured, returns a view that does not own
// the storage of `tensor`; otherwise returns `tensor`. Use it only for tensors
// produced inside the captured segments: the shared graph pool keeps their
// storage reserved for replay, and owning it would pin every captured graph's
// runner inputs and outputs.
torch::Tensor piecewise_runner_tensor(const torch::Tensor& tensor);

}  // namespace xllm::kernel::musa
