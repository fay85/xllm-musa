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

#include "core/layers/musa/glm5_next/glm5_next_decoder_layer.h"

#include <glog/logging.h>

#include <string>
#include <tuple>
#include <utility>

#include "core/layers/common/dsa_topk_share_plan.h"

namespace xllm::layer {
namespace {

std::string layer_type_at(const std::vector<std::string>& types,
                          int32_t layer_id,
                          const std::string& fallback) {
  CHECK_GE(layer_id, 0);
  if (layer_id < static_cast<int32_t>(types.size())) {
    return types[static_cast<size_t>(layer_id)];
  }
  return fallback;
}

}  // namespace

Glm5NextLayerRole resolve_glm5_next_layer_role(const ModelArgs& args,
                                               int32_t layer_id) {
  const std::string attention_type =
      layer_type_at(args.layer_types(), layer_id, "deepseek_sparse_attention");
  CHECK(attention_type == "linear_attention" ||
        attention_type == "deepseek_sparse_attention" ||
        attention_type == "attention" || attention_type == "full_attention")
      << "Unsupported GLM5-Next attention layer type at layer " << layer_id
      << ": " << attention_type;
  const std::string mlp_type =
      layer_type_at(args.mlp_layer_types(), layer_id, "sparse");
  CHECK(mlp_type == "dense" || mlp_type == "sparse")
      << "Unsupported GLM5-Next MLP layer type at layer " << layer_id << ": "
      << mlp_type;
  return {
      .attention = attention_type == "linear_attention"
                       ? Glm5NextAttentionRole::KDA
                       : Glm5NextAttentionRole::DSA,
      .mlp = mlp_type == "dense" ? Glm5NextMlpRole::DENSE
                                 : Glm5NextMlpRole::SPARSE,
  };
}

Glm5NextDecoderLayerImpl::Glm5NextDecoderLayerImpl(const ModelContext& context,
                                                   int32_t layer_id)
    : layer_id_(layer_id),
      role_(resolve_glm5_next_layer_role(context.get_model_args(), layer_id)) {
  const ModelArgs& args = context.get_model_args();
  const QuantArgs& quant_args = context.get_quant_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions& options = context.get_tensor_options();
  const int64_t hidden_size = args.hidden_size();
  const double norm_eps = static_cast<double>(args.rms_norm_eps());
  attn_hc_pre_ = register_module("attn_hc_pre",
                                 MHCPre(args.hc_mult(),
                                        hidden_size,
                                        args.hc_sinkhorn_iters(),
                                        static_cast<double>(args.hc_eps()),
                                        norm_eps,
                                        options));
  ffn_hc_pre_ = register_module("ffn_hc_pre",
                                MHCPre(args.hc_mult(),
                                       hidden_size,
                                       args.hc_sinkhorn_iters(),
                                       static_cast<double>(args.hc_eps()),
                                       norm_eps,
                                       options));
  hc_post_ = register_module("hc_post", MHCPost(norm_eps));
  input_norm_ = register_module(
      "input_layernorm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  post_norm_ =
      register_module("post_attention_layernorm",
                      RMSNorm(hidden_size, args.rms_norm_eps(), options));
  if (role_.attention == Glm5NextAttentionRole::KDA) {
    kda_ = register_module("self_attn", Glm5NextKDA(context));
  } else {
    enable_indexer_ = DsaTopkSharePlan(args).layer_uses_indexer(layer_id);
    sparse_attention_ = register_module(
        "self_attn",
        Glm5NextSparseAttention(args, parallel_args, options, enable_indexer_));
  }
  if (role_.mlp == Glm5NextMlpRole::DENSE) {
    dense_mlp_ =
        register_module("mlp",
                        DenseMLP(hidden_size,
                                 args.intermediate_size(),
                                 /*is_gated=*/true,
                                 /*has_bias=*/false,
                                 args.hidden_act(),
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 parallel_args.tp_group_,
                                 options,
                                 /*module_prefix=*/"",
                                 static_cast<double>(args.swiglu_limit())));
  } else {
    sparse_gate_ =
        register_module("gate", RoutedMoeGate(args, quant_args, options));
    sparse_moe_ = register_module(
        "mlp", RoutedMoe(args, quant_args, parallel_args, options, layer_id));
  }
}

void Glm5NextDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  const StateDict attention_state = state_dict.get_dict_with_prefix(
      std::vector<std::string>{"self_attn.", "attn."});
  if (kda_) {
    kda_->load_state_dict(attention_state);
  } else {
    sparse_attention_->load_state_dict(attention_state);
  }
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  const StateDict mlp_state =
      state_dict.get_dict_with_prefix(std::vector<std::string>{"mlp.", "ffn."});
  if (dense_mlp_) {
    dense_mlp_->load_state_dict(mlp_state);
  } else {
    StateDict gate_state = mlp_state.get_dict_with_prefix("gate.");
    if (gate_state.size() == 0) {
      gate_state = state_dict.get_dict_with_prefix("gate.");
    }
    if (gate_state.size() > 0) {
      sparse_gate_->load_state_dict(gate_state);
    }
    sparse_moe_->load_state_dict(mlp_state);
  }
  attn_hc_pre_->load_state_dict(
      get_mhc_state(state_dict, "attn_hc_pre.", "hc_attn_"));
  ffn_hc_pre_->load_state_dict(
      get_mhc_state(state_dict, "ffn_hc_pre.", "hc_ffn_"));
}

void Glm5NextDecoderLayerImpl::verify_loaded_weights() const {
  attn_hc_pre_->verify_loaded_weights("attn_hc_pre.");
  ffn_hc_pre_->verify_loaded_weights("ffn_hc_pre.");
  if (kda_) {
    kda_->verify_loaded_weights("self_attn.");
  } else {
    sparse_attention_->verify_loaded_weights();
  }
  if (sparse_gate_) {
    sparse_gate_->verify_loaded_weights("mlp.gate.");
  }
  if (sparse_moe_) {
    sparse_moe_->verify_loaded_weights("mlp.");
  }
}

torch::Tensor Glm5NextDecoderLayerImpl::forward(
    torch::Tensor& hidden_states,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    std::optional<PendingMHC>* pending_mhc,
    bool is_last_layer,
    kernel::musa::GlmKPoolSelection* shared_selection) {
  residual = std::nullopt;
  const bool has_pending = pending_mhc != nullptr && pending_mhc->has_value();
  torch::Tensor residual_attention;
  MHCPreOutput attention_hc;
  torch::Tensor attention_input;
  if (has_pending) {
    PendingMHC& pending = pending_mhc->value();
    std::tie(attention_input,
             residual_attention,
             attention_hc.post,
             attention_hc.comb) =
        attn_hc_pre_->post_pre_norm(pending.x,
                                    pending.residual,
                                    pending.post,
                                    pending.comb,
                                    input_norm_->weight());
    pending_mhc->reset();
  } else {
    residual_attention = hidden_states;
    attention_hc = attn_hc_pre_->forward(hidden_states);
    attention_input = std::get<0>(input_norm_->forward(attention_hc.output));
  }

  torch::Tensor attention_output;
  if (kda_) {
    attention_output =
        kda_->forward(attention_input, attn_metadata, kv_cache, input_params);
  } else {
    if (!enable_indexer_) {
      CHECK_NOTNULL(shared_selection);
    }
    auto [output, selection] = sparse_attention_->forward(
        positions,
        attention_input,
        attn_metadata,
        kv_cache,
        enable_indexer_ ? nullptr : shared_selection);
    if (enable_indexer_ && shared_selection != nullptr) {
      *shared_selection = std::move(selection);
    }
    attention_output = std::move(output);
  }

  torch::Tensor ffn_input;
  torch::Tensor residual_ffn;
  MHCPreOutput ffn_hc;
  std::tie(ffn_input, residual_ffn, ffn_hc.post, ffn_hc.comb) =
      ffn_hc_pre_->post_pre_norm(attention_output,
                                 residual_attention,
                                 attention_hc.post,
                                 attention_hc.comb,
                                 post_norm_->weight());
  torch::Tensor ffn_output;
  if (dense_mlp_) {
    ffn_output = dense_mlp_->forward(ffn_input);
  } else {
    torch::Tensor ffn_input_2d = ffn_input.reshape({-1, ffn_input.size(-1)});
    const bool is_prefill =
        attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
    auto [topk_weights, topk_ids] =
        sparse_gate_->forward(ffn_input_2d, is_prefill);
    ModelInputParams moe_input_params = input_params;
    if (moe_input_params.meta.batch_forward_type.is_empty()) {
      if (attn_metadata.is_chunked_prefill) {
        moe_input_params.meta.batch_forward_type =
            BatchForwardType::CHUNKED_PREFILL;
      } else if (attn_metadata.is_prefill) {
        moe_input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
      } else {
        moe_input_params.meta.batch_forward_type = BatchForwardType::DECODE;
      }
    }
    ffn_output = sparse_moe_->forward_with_selected_experts(
        ffn_input_2d, topk_weights, topk_ids, moe_input_params);
    ffn_output = ffn_output.reshape(ffn_input.sizes());
  }
  if (!is_last_layer && pending_mhc != nullptr) {
    pending_mhc->emplace(
        PendingMHC{ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb});
    return ffn_output;
  }
  std::tie(hidden_states, std::ignore) =
      hc_post_->forward(ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb);
  return hidden_states;
}

}  // namespace xllm::layer
