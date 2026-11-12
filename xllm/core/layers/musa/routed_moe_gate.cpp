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

#include "core/layers/musa/routed_moe_gate.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor renormalize_topk_weights(const torch::Tensor& topk_weights) {
  torch::Tensor denom = topk_weights.sum(-1, true);
  denom = torch::clamp_min(denom, 1e-20);
  // TorchMUSA 2.9 can silently miscompute broadcasted div.Tensor; use
  // explicit multiplication by the reciprocal.
  return topk_weights * denom.reciprocal();
}

void validate_fused_sigmoid_gate(const torch::Tensor& logits,
                                 const torch::Tensor& bias,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& ids,
                                 bool renormalize,
                                 double route_scale) {
  static const bool enabled = []() {
    const char* value = std::getenv("XLLM_GLM5_VALIDATE_GATE");
    return value != nullptr && std::string(value) == "1";
  }();
  if (!enabled || kernel::musa::is_stream_capturing()) {
    return;
  }
  auto scores = torch::sigmoid(logits);
  auto reference_ids = std::get<1>(torch::topk(scores + bias,
                                               ids.size(1),
                                               /*dim=*/-1,
                                               /*largest=*/true,
                                               /*sorted=*/false));
  auto reference_weights = scores.gather(-1, reference_ids);
  if (renormalize) {
    reference_weights = renormalize_topk_weights(reference_weights);
  }
  reference_weights.mul_(route_scale);
  auto expected =
      torch::zeros_like(scores).scatter_(1, reference_ids, reference_weights);
  auto actual =
      torch::zeros_like(scores).scatter_(1, ids.to(torch::kInt64), weights);
  const int64_t changed_rows = std::get<0>(ids.to(torch::kInt64).sort(-1))
                                   .ne(std::get<0>(reference_ids.sort(-1)))
                                   .any(-1)
                                   .sum()
                                   .item<int64_t>();
  const double max_error = (actual - expected).abs().max().item<double>();
  LOG(INFO) << "GLM_GATE_COMPARE tokens=" << logits.size(0)
            << " changed_rows=" << changed_rows << " mixture_max=" << max_error;
  CHECK(torch::isfinite(actual).all().item<bool>());
  CHECK_EQ(changed_rows, 0);
  CHECK_LE(max_error, 1e-5)
      << "Fused sigmoid routing disagrees with the reference";
}

}  // namespace

RoutedMoeGateImpl::RoutedMoeGateImpl(const ModelContext& context,
                                     int32_t layer_id)
    : RoutedMoeGateImpl(context.get_model_args(),
                        context.get_quant_args(),
                        context.get_tensor_options(),
                        layer_id) {}

RoutedMoeGateImpl::RoutedMoeGateImpl(const ModelArgs& model_args,
                                     const QuantArgs& /*quant_args*/,
                                     const torch::TensorOptions& options,
                                     int32_t layer_id)
    : hidden_size_(model_args.hidden_size()),
      n_routed_experts_(model_args.n_routed_experts()),
      top_k_(model_args.n_activated_experts() > 0
                 ? model_args.n_activated_experts()
                 : model_args.num_experts_per_tok()),
      num_expert_group_(std::max<int64_t>(1, model_args.n_group())),
      topk_group_(std::max<int64_t>(1, model_args.topk_group())),
      n_hash_layers_(model_args.n_hash_layers()),
      route_scale_(model_args.routed_scaling_factor()),
      scoring_func_(model_args.scoring_func().empty()
                        ? std::string("softmax")
                        : model_args.scoring_func()),
      topk_method_(model_args.topk_method()),
      norm_topk_prob_(model_args.norm_topk_prob()),
      hash_layer_(layer_id >= 0 && layer_id < n_hash_layers_),
      decode_workspace_tokens_(
          std::max<int64_t>(16, model_args.num_speculative_tokens() + 1)) {
  renormalize_ = top_k_ > 1 ? norm_topk_prob_ : false;
  CHECK_GT(hidden_size_, 0)
      << "RoutedMoeGate requires hidden_size > 0, got " << hidden_size_;
  CHECK_GT(n_routed_experts_, 0)
      << "RoutedMoeGate requires n_routed_experts > 0, got "
      << n_routed_experts_;
  CHECK_GT(top_k_, 0) << "RoutedMoeGate requires top_k > 0, got " << top_k_;
  CHECK_LE(top_k_, n_routed_experts_)
      << "RoutedMoeGate top_k must be <= n_routed_experts, got top_k=" << top_k_
      << ", n_routed_experts=" << n_routed_experts_;

  const torch::TensorOptions gate_options = options.dtype(torch::kFloat32);
  weight_ = register_parameter(
      "weight",
      torch::empty({n_routed_experts_, hidden_size_}, gate_options),
      /*requires_grad=*/false);

  if (hash_layer_) {
    const int64_t vocab_size = model_args.vocab_size();
    CHECK_GT(vocab_size, 0)
        << "RoutedMoeGate hash layer requires vocab_size > 0, got "
        << vocab_size;
    tid2eid_ = register_parameter(
        "tid2eid",
        torch::empty({vocab_size, top_k_},
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

std::tuple<torch::Tensor, torch::Tensor> RoutedMoeGateImpl::forward(
    const torch::Tensor& hidden_states,
    bool /*is_prefill*/,
    const std::optional<torch::Tensor>& input_ids) {
  CHECK(hidden_states.defined())
      << "RoutedMoeGate::forward hidden_states is undefined";
  CHECK_EQ(hidden_states.size(-1), hidden_size_)
      << "RoutedMoeGate::forward hidden_states last dim mismatch, expected "
      << hidden_size_ << " got " << hidden_states.size(-1);

  const int64_t num_tokens = hidden_states.numel() / hidden_size_;
  if (score_func_to_norm_type(scoring_func_) == 2 && weight_bf16_.defined() &&
      num_tokens > 0 && num_tokens <= decode_workspace_tokens_ &&
      decode_logits_output_.defined() &&
      num_tokens <= decode_logits_output_.size(0)) {
    torch::Tensor input_2d = hidden_states.reshape({num_tokens, hidden_size_});
    torch::Tensor logits_output = decode_logits_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/num_tokens);
    torch::Tensor logits = kernel::musa::deepseek_v4_gate_gemv(
        input_2d, weight_bf16_, logits_output);
    CHECK(logits.defined()) << "RoutedMoeGate projection is unavailable";
    torch::Tensor topk_weights_output = decode_topk_weights_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/num_tokens);
    torch::Tensor topk_ids_output = decode_topk_ids_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/num_tokens);
    if (hash_layer_ && input_ids.has_value() && input_ids.value().defined() &&
        tid2eid_.defined() && input_ids.value().numel() == num_tokens) {
      return kernel::musa::deepseek_v4_gate_hash_topk(logits,
                                                      input_ids.value(),
                                                      tid2eid_,
                                                      top_k_,
                                                      route_scale_,
                                                      topk_weights_output,
                                                      topk_ids_output);
    }
    if (!hash_layer_ && bias_is_loaded_) {
      return kernel::musa::deepseek_v4_gate_topk(logits,
                                                 bias_,
                                                 top_k_,
                                                 route_scale_,
                                                 topk_weights_output,
                                                 topk_ids_output);
    }
  }

  torch::Tensor gate_input = hidden_states.reshape({num_tokens, hidden_size_});
  if (gate_input.scalar_type() != torch::kFloat32) {
    gate_input = gate_input.to(torch::kFloat32);
  }
  torch::Tensor logits = torch::matmul(gate_input, weight_.transpose(0, 1));
  return select_experts(logits, input_ids);
}

std::tuple<torch::Tensor, torch::Tensor> RoutedMoeGateImpl::select_experts(
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& input_ids) const {
  torch::Tensor scores;
  const int64_t norm_type = score_func_to_norm_type(scoring_func_);
  const char* fused_gate_env = std::getenv("XLLM_GLM5_FUSED_GATE");
  const bool use_fused_gate =
      fused_gate_env == nullptr || std::string(fused_gate_env) != "0";
  // Preserve muDNN sigmoid and top-k, including tied expert selection and
  // order. Fuse only gather, normalization, scaling and index conversion.
  // Correction bias affects selection, never the mixture weights.
  if (use_fused_gate && norm_type == 1 && !hash_layer_ && bias_is_loaded_ &&
      topk_method_ == "noaux_tc" && n_routed_experts_ == 288 &&
      num_expert_group_ == 1 && top_k_ == 8) {
    namespace km = kernel::musa;
    auto weights = torch::empty({router_logits.size(0), top_k_},
                                router_logits.options().dtype(torch::kFloat32));
    auto ids =
        torch::empty_like(weights, weights.options().dtype(torch::kInt32));
    CHECK(router_logits.is_contiguous());
    CHECK_EQ(router_logits.scalar_type(), torch::kFloat32);
    auto routing_scores = torch::sigmoid(router_logits);
    auto selected = std::get<1>(torch::topk(routing_scores + bias_,
                                            top_k_,
                                            /*dim=*/-1,
                                            /*largest=*/true,
                                            /*sorted=*/false));
    auto function = km::get_function("glm53_moe_route_post_v3", "run");
    {
      km::TvmffiStreamGuard stream_guard(router_logits.device());
      function(km::to_ffi_tensor(weights),
               km::to_ffi_tensor(ids),
               km::to_ffi_tensor(routing_scores),
               km::to_ffi_tensor(selected),
               renormalize_,
               route_scale_);
    }
    validate_fused_sigmoid_gate(
        router_logits, bias_, weights, ids, renormalize_, route_scale_);
    return {weights, ids};
  }
  if (norm_type == 0) {
    scores = torch::softmax(router_logits, -1);
  } else if (norm_type == 1) {
    scores = torch::sigmoid(router_logits);
  } else {
    scores = torch::softplus(router_logits).sqrt();
  }

  torch::Tensor original_scores = scores;
  torch::Tensor scores_for_choice = scores;
  if (!hash_layer_ && bias_is_loaded_) {
    scores_for_choice = scores_for_choice + bias_.to(scores.dtype());
  }

  const int64_t num_tokens = scores.size(0);
  if (!hash_layer_ && num_expert_group_ > 1 &&
      topk_group_ < num_expert_group_) {
    CHECK_EQ(scores.size(-1) % num_expert_group_, 0)
        << "n_routed_experts must be divisible by n_group";
    const int64_t experts_per_group = scores.size(-1) / num_expert_group_;
    torch::Tensor group_scores;
    if (topk_method_ == "noaux_tc") {
      CHECK_GE(experts_per_group, 2)
          << "noaux_tc grouped routing expects at least 2 experts per group";
      group_scores =
          std::get<0>(
              scores_for_choice
                  .view({num_tokens, num_expert_group_, experts_per_group})
                  .topk(2, -1))
              .sum(-1);
    } else {
      group_scores = std::get<0>(
          scores_for_choice
              .view({num_tokens, num_expert_group_, experts_per_group})
              .max(-1));
    }

    torch::Tensor group_idx =
        std::get<1>(group_scores.topk(topk_group_, -1, true, false));
    torch::Tensor group_mask = torch::zeros_like(group_scores);
    group_mask.scatter_(1, group_idx, 1.0f);
    torch::Tensor score_mask =
        group_mask.unsqueeze(-1)
            .expand({num_tokens, num_expert_group_, experts_per_group})
            .reshape({num_tokens, -1});
    scores_for_choice = scores_for_choice.masked_fill(score_mask == 0, 0.0f);
  }

  torch::Tensor topk_idx;
  if (hash_layer_ && input_ids.has_value() && input_ids.value().defined() &&
      tid2eid_.defined()) {
    torch::Tensor lookup_ids = input_ids.value().to(torch::kLong);
    topk_idx = tid2eid_.index({lookup_ids});
  } else {
    topk_idx = std::get<1>(torch::topk(scores_for_choice,
                                       top_k_,
                                       /*dim=*/-1,
                                       /*largest=*/true,
                                       /*sorted=*/false));
  }

  torch::Tensor gather_idx = topk_idx.to(torch::kLong);
  torch::Tensor topk_weights = original_scores.gather(-1, gather_idx);
  if (renormalize_) {
    topk_weights = renormalize_topk_weights(topk_weights);
  }
  topk_weights = topk_weights * route_scale_;
  return std::make_tuple(topk_weights.to(torch::kFloat32).contiguous(),
                         gather_idx.to(torch::kInt32).contiguous());
}

void RoutedMoeGateImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  torch::Tensor gate_weight = state_dict.get_tensor("weight");
  if (gate_weight.defined()) {
    torch::NoGradGuard no_grad;
    torch::Tensor device_weight = gate_weight.to(weight_.device()).contiguous();
    weight_.copy_(device_weight.to(weight_.dtype()));
    weight_is_loaded_ = true;
    if (gate_weight.scalar_type() == torch::kBFloat16) {
      weight_bf16_ = device_weight.to(torch::kBFloat16).contiguous();
    }
    const torch::TensorOptions decode_options =
        weight_.options().dtype(torch::kFloat32);
    decode_logits_output_ = torch::empty(
        {decode_workspace_tokens_, n_routed_experts_}, decode_options);
    decode_topk_weights_output_ =
        torch::empty({decode_workspace_tokens_, top_k_}, decode_options);
    decode_topk_ids_output_ = torch::empty({decode_workspace_tokens_, top_k_},
                                           decode_options.dtype(torch::kInt32));
  }

  if (hash_layer_) {
    torch::Tensor tid2eid = state_dict.get_tensor("tid2eid");
    if (!tid2eid.defined()) {
      tid2eid = state_dict.get_tensor("tid2eid.weight");
    }
    if (tid2eid.defined()) {
      torch::NoGradGuard no_grad;
      tid2eid_.copy_(tid2eid.to(tid2eid_.device()).to(tid2eid_.dtype()));
      tid2eid_is_loaded_ = true;
    }
    return;
  }

  torch::Tensor bias = state_dict.get_tensor("bias");
  if (!bias.defined()) {
    bias = state_dict.get_tensor("e_score_correction_bias");
  }
  if (bias.defined()) {
    torch::NoGradGuard no_grad;
    bias_.copy_(bias.to(bias_.device()).to(bias_.dtype()));
    bias_is_loaded_ = true;
  }
}

void RoutedMoeGateImpl::verify_loaded_weights(const std::string& prefix) const {
  CHECK(weight_is_loaded_) << "weight is not loaded for " << prefix + "weight";
  if (hash_layer_) {
    CHECK(tid2eid_is_loaded_)
        << "weight is not loaded for " << prefix + "tid2eid";
  } else if (topk_method_ == "noaux_tc") {
    CHECK(bias_is_loaded_) << "weight is not loaded for "
                           << prefix + "e_score_correction_bias";
  }
}

int64_t RoutedMoeGateImpl::score_func_to_norm_type(
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

  LOG(FATAL) << "RoutedMoeGate unsupported score_func: " << score_func;
  return 0;
}

}  // namespace layer
}  // namespace xllm
