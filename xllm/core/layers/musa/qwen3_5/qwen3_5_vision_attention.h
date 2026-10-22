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

#include "core/layers/common/qwen2_vision_attention.h"

namespace xllm {
namespace layer {

// Qwen3.5's MUSA vision attention has a device-specific rotary/softmax
// contract. Keep it separate from the Qwen2 attention implementation.
class Qwen3_5VisionAttentionImpl final : public Qwen2VisionAttentionImpl {
 public:
  explicit Qwen3_5VisionAttentionImpl(const ModelContext& context)
      : Qwen2VisionAttentionImpl(context) {}

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec) override;
};
TORCH_MODULE(Qwen3_5VisionAttention);

}  // namespace layer
}  // namespace xllm
