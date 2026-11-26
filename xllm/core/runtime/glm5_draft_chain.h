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

#include "core/runtime/forward_params.h"

namespace xllm {

// One synchronous worker forward may submit the remaining GLM greedy steps.
// Publish the extra outputs only after replay. An eager route leaves them
// empty.
class Glm5DraftChain final {
 public:
  Glm5DraftChain(const ForwardInput& input, int32_t steps, int32_t block_size)
      : input(input),
        steps(steps),
        block_size(block_size),
        previous_(current_) {
    current_ = this;
  }
  ~Glm5DraftChain() { current_ = previous_; }
  Glm5DraftChain(const Glm5DraftChain&) = delete;
  Glm5DraftChain& operator=(const Glm5DraftChain&) = delete;

  static Glm5DraftChain* current() { return current_; }
  const ForwardInput& input;
  const int32_t steps;
  const int32_t block_size;
  torch::Tensor proposals;

 private:
  inline static thread_local Glm5DraftChain* current_ = nullptr;
  Glm5DraftChain* previous_;
};
}  // namespace xllm
