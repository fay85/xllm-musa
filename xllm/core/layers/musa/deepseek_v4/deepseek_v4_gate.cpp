/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/layers/musa/deepseek_v4/deepseek_v4_gate.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>

namespace xllm {
namespace layer {
namespace {

torch::Tensor renormalize_topk_weights(const torch::Tensor& topk_weights) {
  auto denom = topk_weights.sum(-1, true);
  denom = torch::clamp_min(denom, 1e-20);
  // TorchMUSA 2.9 can silently miscompute broadcasted div.Tensor.  Keep the
  // normalization explicit as multiplication by the reciprocal.
  return topk_weights * denom.reciprocal();
}

}  // namespace

DeepseekV4GateImpl::DeepseekV4GateImpl(const ModelContext& context,
                                       int32_t layer_id)
    : DeepseekV4GateImpl(context.get_model_args(),
                         layer_id,
                         context.get_tensor_options()) {}

DeepseekV4GateImpl::DeepseekV4GateImpl(const ModelArgs& args,
                                       int32_t layer_id,
                                       const torch::TensorOptions& options) {
  hidden_size_ = args.hidden_size();
  n_routed_experts_ = args.n_routed_experts();
  topk_ = args.n_activated_experts() > 0 ? args.n_activated_experts()
                                         : args.num_experts_per_tok();
  n_hash_layers_ = args.n_hash_layers();
  route_scale_ = args.routed_scaling_factor();
  score_func_ = args.scoring_func().empty() ? std::string("softmax")
                                            : args.scoring_func();
  hash_layer_ = layer_id >= 0 && layer_id < n_hash_layers_;

  CHECK_GT(hidden_size_, 0)
      << "DeepseekV4Gate requires hidden_size > 0, got " << hidden_size_;
  CHECK_GT(n_routed_experts_, 0)
      << "DeepseekV4Gate requires n_routed_experts > 0, got "
      << n_routed_experts_;
  CHECK_GT(topk_, 0) << "DeepseekV4Gate requires topk > 0, got " << topk_;
  CHECK_LE(topk_, n_routed_experts_)
      << "DeepseekV4Gate topk must be <= n_routed_experts, got topk=" << topk_
      << ", n_routed_experts=" << n_routed_experts_;

  auto gate_options = options.dtype(torch::kFloat32);
  weight_ = register_parameter(
      "weight",
      torch::empty({n_routed_experts_, hidden_size_}, gate_options),
      /*requires_grad=*/false);

  if (hash_layer_) {
    const int64_t vocab_size = args.vocab_size();
    CHECK_GT(vocab_size, 0)
        << "DeepseekV4Gate hash layer requires vocab_size > 0, got "
        << vocab_size;
    tid2eid_ = register_parameter(
        "tid2eid",
        torch::empty({vocab_size, topk_},
                     options.dtype(torch::kInt32).device(options.device())),
        /*requires_grad=*/false);
  } else {
    bias_ = register_parameter(
        "bias",
        torch::empty({n_routed_experts_},
                     options.dtype(torch::kFloat32).device(options.device())),
        /*requires_grad=*/false);
  }
}

std::tuple<torch::Tensor, torch::Tensor> DeepseekV4GateImpl::forward(
    const torch::Tensor& hidden_states,
    const std::optional<torch::Tensor>& input_ids) {
  CHECK(hidden_states.defined())
      << "DeepseekV4Gate::forward hidden_states is undefined";
  CHECK_EQ(hidden_states.size(-1), hidden_size_)
      << "DeepseekV4Gate::forward hidden_states last dim mismatch, expected "
      << hidden_size_ << " got " << hidden_states.size(-1);

  auto gate_input = hidden_states;
  if (gate_input.scalar_type() != torch::kFloat32) {
    gate_input = gate_input.to(torch::kFloat32);
  }
  auto logits = torch::matmul(gate_input, weight_.transpose(0, 1));
  return select_experts_native(logits, input_ids);
}

std::tuple<torch::Tensor, torch::Tensor>
DeepseekV4GateImpl::select_experts_native(
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& input_ids) const {
  torch::Tensor scores;
  const int64_t norm_type = score_func_to_norm_type(score_func_);
  if (norm_type == 0) {
    scores = torch::softmax(router_logits, -1);
  } else if (norm_type == 1) {
    scores = torch::sigmoid(router_logits);
  } else {
    scores = torch::softplus(router_logits).sqrt();
  }

  auto original_scores = scores;
  if (!hash_layer_ && bias_.defined()) {
    scores = scores + bias_.to(scores.dtype());
  }

  torch::Tensor topk_idx;
  if (hash_layer_ && input_ids.has_value() && input_ids.value().defined() &&
      tid2eid_.defined()) {
    auto lookup_ids = input_ids.value().to(torch::kLong);
    topk_idx = tid2eid_.index({lookup_ids});
  } else {
    topk_idx = std::get<1>(torch::topk(scores,
                                       topk_,
                                       /*dim=*/-1,
                                       /*largest=*/true,
                                       /*sorted=*/false));
  }

  auto gather_idx = topk_idx.to(torch::kLong);
  auto topk_weights = original_scores.gather(-1, gather_idx);
  topk_weights = renormalize_topk_weights(topk_weights) * route_scale_;
  return std::make_tuple(topk_weights, gather_idx.to(torch::kInt32));
}

void DeepseekV4GateImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  auto gate_weight = state_dict.get_tensor("weight");
  CHECK(gate_weight.defined())
      << "DeepseekV4Gate missing weight in state_dict with prefix "
      << state_dict.prefix();
  {
    torch::NoGradGuard no_grad;
    weight_.copy_(gate_weight.to(weight_.device()).to(weight_.dtype()));
  }

  if (hash_layer_) {
    auto tid2eid = state_dict.get_tensor("tid2eid");
    if (!tid2eid.defined()) {
      tid2eid = state_dict.get_tensor("tid2eid.weight");
    }
    CHECK(tid2eid.defined()) << "DeepseekV4Gate hash layer missing tid2eid in "
                                "state_dict with prefix "
                             << state_dict.prefix();
    torch::NoGradGuard no_grad;
    tid2eid_.copy_(tid2eid.to(tid2eid_.device()).to(tid2eid_.dtype()));
    return;
  }

  auto bias = state_dict.get_tensor("bias");
  if (!bias.defined()) {
    bias = state_dict.get_tensor("e_score_correction_bias");
  }
  CHECK(bias.defined()) << "DeepseekV4Gate non-hash layer missing bias (or "
                        << "e_score_correction_bias) in state_dict with prefix "
                        << state_dict.prefix();
  {
    torch::NoGradGuard no_grad;
    bias_.copy_(bias.to(bias_.device()).to(bias_.dtype()));
  }
}

int64_t DeepseekV4GateImpl::score_func_to_norm_type(
    const std::string& score_func) const {
  std::string lowered = score_func;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });

  if (lowered == "softmax") {
    return 0;
  }
  if (lowered == "sigmoid") {
    return 1;
  }
  if (lowered == "sqrtsoftplus") {
    return 2;
  }

  CHECK(false) << "DeepseekV4Gate unsupported score_func: " << score_func;
  return 0;
}

}  // namespace layer
}  // namespace xllm
