/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class RoutedMoeGateImpl final : public torch::nn::Module {
 public:
  RoutedMoeGateImpl() = default;
  RoutedMoeGateImpl(const ModelContext& context, int32_t layer_id = 0);
  RoutedMoeGateImpl(const ModelArgs& model_args,
                    const QuantArgs& quant_args,
                    const torch::TensorOptions& options,
                    int32_t layer_id = 0);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      bool is_prefill = false,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix = "") const;

 private:
  std::tuple<torch::Tensor, torch::Tensor> select_experts(
      const torch::Tensor& router_logits,
      const std::optional<torch::Tensor>& input_ids) const;
  int64_t score_func_to_norm_type(const std::string& score_func) const;

  int64_t hidden_size_ = 0;
  int64_t n_routed_experts_ = 0;
  int64_t top_k_ = 0;
  int64_t num_expert_group_ = 1;
  int64_t topk_group_ = 1;
  int64_t n_hash_layers_ = 0;
  double route_scale_ = 1.0;
  std::string scoring_func_ = "sigmoid";
  std::string topk_method_ = "noaux_tc";
  bool norm_topk_prob_ = true;
  bool renormalize_ = true;
  bool hash_layer_ = false;
  int64_t decode_workspace_tokens_ = 16;

  DEFINE_WEIGHT(weight);
  torch::Tensor weight_bf16_;
  DEFINE_WEIGHT(bias);
  DEFINE_WEIGHT(tid2eid);
  torch::Tensor decode_logits_output_;
  torch::Tensor decode_topk_weights_output_;
  torch::Tensor decode_topk_ids_output_;
};

TORCH_MODULE(RoutedMoeGate);

}  // namespace layer
}  // namespace xllm
