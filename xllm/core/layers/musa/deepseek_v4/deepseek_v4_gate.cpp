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
#include <cstdlib>
#include <string>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_gate_policy.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_prefill_gate_policy.h"

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
  decode_workspace_tokens_ =
      std::max<int64_t>(16, args.num_speculative_tokens() + 1);

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
    const std::optional<torch::Tensor>& input_ids,
    bool batch_verify) {
  CHECK(hidden_states.defined())
      << "DeepseekV4Gate::forward hidden_states is undefined";
  CHECK_EQ(hidden_states.size(-1), hidden_size_)
      << "DeepseekV4Gate::forward hidden_states last dim mismatch, expected "
      << hidden_size_ << " got " << hidden_states.size(-1);

  const int64_t num_tokens = hidden_states.numel() / hidden_size_;
  const char* disable_gate_gemv_env =
      std::getenv("XLLM_DSV4_DISABLE_GATE_GEMV");
  const bool disable_gate_gemv = disable_gate_gemv_env != nullptr &&
                                 std::string(disable_gate_gemv_env) == "1";
  const bool batch_mate = batch_verify &&
                          num_tokens > decode_workspace_tokens_ &&
                          decode_logits_output_.defined() &&
                          num_tokens <= decode_logits_output_.size(0);
  if (!disable_gate_gemv && score_func_to_norm_type(score_func_) == 2 &&
      weight_bf16_.defined() && num_tokens > 0 &&
      (num_tokens <= decode_workspace_tokens_ || batch_mate)) {
    auto input_2d = hidden_states.reshape({num_tokens, hidden_size_});
    auto logits_output = decode_logits_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/num_tokens);
    auto logits = batch_mate ? kernel::musa::deepseek_v4_gate_bf16_fp32(
                                   input_2d, weight_bf16_, logits_output)
                             : kernel::musa::deepseek_v4_gate_gemv(
                                   input_2d, weight_bf16_, logits_output);
    CHECK(logits.defined()) << "Selected Gate projection is unavailable";
    if (batch_mate) {
      LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_GATE_MATE] rows=" << num_tokens
                           << " output_fp32=1 target_verify_only=1";
    }
    auto topk_weights_output = decode_topk_weights_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/num_tokens);
    auto topk_ids_output = decode_topk_ids_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/num_tokens);
    if (hash_layer_ && input_ids.has_value() && input_ids.value().defined() &&
        tid2eid_.defined() && input_ids.value().numel() == num_tokens) {
      return kernel::musa::deepseek_v4_gate_hash_topk(logits,
                                                      input_ids.value(),
                                                      tid2eid_,
                                                      topk_,
                                                      route_scale_,
                                                      topk_weights_output,
                                                      topk_ids_output);
    }
    if (!hash_layer_ && bias_.defined()) {
      return kernel::musa::deepseek_v4_gate_topk(logits,
                                                 bias_,
                                                 topk_,
                                                 route_scale_,
                                                 topk_weights_output,
                                                 topk_ids_output);
    }
    CHECK(!batch_mate)
        << "Batched target Gate requires valid hash IDs or correction bias";
  }

  if (!disable_gate_gemv && dsv4_prefill_gate_enabled() &&
      dsv4_prefill_gate_shape(
          num_tokens,
          hidden_size_,
          n_routed_experts_,
          score_func_to_norm_type(score_func_),
          batch_verify,
          weight_bf16_.defined() &&
              hidden_states.scalar_type() == torch::kBFloat16)) {
    // The checkpoint weight is already BF16. Only the projection backend
    // changes; retain FP32 logits and the original native expert selection.
    // Keep this separate from the small persistent Decode workspaces.
    torch::Tensor logits;
    {
      // In the preserved runtime a null/default Torch stream is bound to
      // the FFI pool stream. Order both producers and native consumers; the
      // guard must end before select_experts_native enqueues its operations.
      kernel::musa::TvmffiStreamGuard stream_guard(hidden_states.device());
      logits = kernel::musa::deepseek_v4_gate_bf16_fp32(
          hidden_states.reshape({num_tokens, hidden_size_}),
          weight_bf16_,
          /*output=*/std::nullopt);
    }
    CHECK(logits.defined()) << "Required Prefill Gate projection unavailable";
    CHECK_EQ(logits.scalar_type(), torch::kFloat32);
    LOG_FIRST_N(INFO, 1) << "[DSV4_PREFILL_GATE_BF16_FP32] rows=" << num_tokens
                         << " original_bf16_weights=1 native_selection=1";
    return select_experts_native(logits, input_ids);
  }

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
    auto device_weight = gate_weight.to(weight_.device()).contiguous();
    weight_.copy_(device_weight.to(weight_.dtype()));
    if (gate_weight.scalar_type() == torch::kBFloat16) {
      weight_bf16_ = device_weight.to(torch::kBFloat16).contiguous();
    }
  }
  const auto decode_options = weight_.options().dtype(torch::kFloat32);
  const char* batch_mate_env = std::getenv("XLLM_DSV4_BATCH_GATE_MATE_VERIFY");
  const bool batch_mate =
      batch_mate_env != nullptr && std::string(batch_mate_env) == "1";
  const int64_t capacity =
      dsv4_gate_workspace_capacity(decode_workspace_tokens_, batch_mate);
  decode_logits_output_ =
      torch::empty({capacity, n_routed_experts_}, decode_options);
  decode_topk_weights_output_ = torch::empty({capacity, topk_}, decode_options);
  decode_topk_ids_output_ =
      torch::empty({capacity, topk_}, decode_options.dtype(torch::kInt32));

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
