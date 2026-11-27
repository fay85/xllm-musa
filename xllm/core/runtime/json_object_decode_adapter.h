// Copyright 2026-2026 The xLLM Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace xllm {

struct ForwardInput;
struct ForwardOutput;

namespace detail {
struct JsonDraftValidationScratch;
}  // namespace detail

class JsonObjectDecodeAdapter final {
 public:
  static void prepare_verify_mask(
      const ForwardInput& input,
      const detail::JsonDraftValidationScratch& draft_states,
      ForwardInput& verify_input,
      const torch::Device& device);
  static void prepare_block_verify_mask(const ForwardInput& input,
                                        const torch::Tensor& draft_token_ids,
                                        ForwardInput& verify_input,
                                        const torch::Device& device);

  static torch::Tensor mask_invalid_draft_tokens(
      const torch::Tensor& draft_token_ids,
      const std::vector<uint8_t>& invalid_draft);

  static std::vector<size_t> check_verified_tokens(
      const ForwardInput& input,
      const torch::Tensor& accepted_token_ids,
      ForwardOutput& output);
};

}  // namespace xllm
