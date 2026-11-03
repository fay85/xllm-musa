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

#include "core/layers/qwen3_vision_layer.h"

namespace xllm {
namespace layer {

Qwen3_VisionLayerImpl::Qwen3_VisionLayerImpl(const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_config = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t dim = args.mm_hidden_size();

  attention_ = register_module("self_attn", Qwen3VisionAttention(context));
  norm1_ = register_module("norm1", RMSNorm(dim, args.rms_norm_eps(), options));
  norm2_ = register_module("norm2", RMSNorm(dim, args.rms_norm_eps(), options));
  norm1_->set_layernorm_mode();
  norm2_->set_layernorm_mode();
  mlp_ = register_module(
      "mlp",
      DenseMLP(dim,
               args.mm_intermediate_size(),
               /*is_gated=*/false,
               /*has_bias=*/true,
               args.mm_hidden_act(),
               /*enable_result_reduction=*/true,
               quant_config,
               parallel_args.tp_group_,
               options));
}

void Qwen3_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  mlp_->load_state_dict(
      state_dict.get_dict_with_prefix("mlp."), {"linear_fc1."}, "linear_fc2.");
  norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
  norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
}

torch::Tensor Qwen3_VisionLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec,
    int node_id) {
  auto norm_output1 = std::get<0>(norm1_(x));
  auto output = x + attention_->forward(
                          norm_output1,
                          m_cos_pos,
                          m_sin_pos,
                          cu_seq_len,
                          cu_seq_len_vec);
  auto norm_output2 = std::get<0>(norm2_(output));
  return output + mlp_(norm_output2);
}

}  // namespace layer
}  // namespace xllm

