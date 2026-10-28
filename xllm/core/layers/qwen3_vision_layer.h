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

#include <torch/torch.h>

#include <functional>

#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/qwen2_vision_attention.h"
#include "core/layers/common/rms_norm.h"
#if defined(USE_MUSA)
#include "core/layers/musa/qwen3_5/qwen3_5_vision_attention.h"
#endif

namespace xllm {
namespace layer {

#if defined(USE_MUSA)
using Qwen3VisionAttention = Qwen3_5VisionAttention;
#else
using Qwen3VisionAttention = Qwen2VisionAttention;
#endif

class Qwen3_VisionLayerImpl final : public torch::nn::Module {
 public:
  explicit Qwen3_VisionLayerImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec,
                        int node_id);

 private:
  Qwen3VisionAttention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm norm1_{nullptr};
  RMSNorm norm2_{nullptr};
};
TORCH_MODULE(Qwen3_VisionLayer);

}  // namespace layer
}  // namespace xllm
