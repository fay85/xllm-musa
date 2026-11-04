/* Copyright 2026 The xLLM Authors.

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

#include "layers/mlu/deepseek_v41/deepseek_v41_decoder_layer.h"

#include <algorithm>

namespace xllm::layer {
DeepseekV41DecoderLayerImpl::DeepseekV41DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& options = context.get_tensor_options();
  const auto& quant = context.get_quant_args();
  const std::string prefix = "layers." + std::to_string(layer_id);
  attention_ = register_module("attn", DeepseekV41Attention(context, layer_id));
  const auto& engram_layers = args.engram_layer_ids();
  const auto it =
      std::find(engram_layers.begin(), engram_layers.end(), layer_id);
  if (it != engram_layers.end()) {
    engram_ = register_module(
        "engram",
        DeepseekV41Engram(context, std::distance(engram_layers.begin(), it)));
  }
  attn_norm_ = register_module("attn_norm", RMSNorm(context));
  ffn_norm_ = register_module("ffn_norm", RMSNorm(context));
  attn_hc_ = register_module("hc_attn",
                             MHCPre(args.hc_mult(),
                                    args.hidden_size(),
                                    args.hc_sinkhorn_iters(),
                                    args.hc_eps(),
                                    args.rms_norm_eps(),
                                    options));
  ffn_hc_ = register_module("hc_ffn",
                            MHCPre(args.hc_mult(),
                                   args.hidden_size(),
                                   args.hc_sinkhorn_iters(),
                                   args.hc_eps(),
                                   args.rms_norm_eps(),
                                   options));
  post_ = register_module("hc_post", MHCPost(args.rms_norm_eps()));
  QuantArgs moe_quant = quant;
  for (const auto& names : std::vector<std::pair<std::string, std::string>>{
           {"w13", "gate_proj"}, {"w13", "up_proj"}, {"w2", "down_proj"}}) {
    auto scheme = quant.module_quant_scheme(prefix + ".ffn.shared_experts." +
                                            names.first);
    if (scheme.has_value()) {
      scheme->targets = {prefix + ".ffn.shared_experts." + names.second};
      moe_quant.compressed_groups().emplace_back(std::move(*scheme));
    }
  }
  const auto& streams = context.stream_registry();
  moe_ = register_module(
      "ffn",
      SparseMoEBlock(args,
                     moe_quant,
                     context.get_parallel_args(),
                     options,
                     false,
                     streams->get(ExecutionStreamRole::COMMUNICATION),
                     streams->get(ExecutionStreamRole::AUXILIARY_COMPUTE),
                     prefix + ".ffn"));
}

void DeepseekV41DecoderLayerImpl::load_state_dict(const StateDict& state) {
  attention_->load_state_dict(state.get_dict_with_prefix("attn."));
  if (engram_) {
    engram_->load_state_dict(state.get_dict_with_prefix("engram."));
  }
  attn_norm_->load_state_dict(state.get_dict_with_prefix("attn_norm."));
  ffn_norm_->load_state_dict(state.get_dict_with_prefix("ffn_norm."));
  StateDict ffn = state.get_dict_with_prefix("ffn.");
  std::unordered_map<std::string, torch::Tensor> normalized;
  for (auto it = ffn.begin(); it != ffn.end(); ++it) {
    std::string name = it->first;
    for (const auto& alias : std::vector<std::pair<std::string, std::string>>{
             {".w1.", ".gate_proj."},
             {".w3.", ".up_proj."},
             {".w2.", ".down_proj."}}) {
      const size_t position = name.rfind(alias.first);
      if (position != std::string::npos) {
        name.replace(position, alias.first.size(), alias.second);
        break;
      }
    }
    normalized.emplace(std::move(name), it->second);
  }
  moe_->load_state_dict(
      StateDict(std::move(normalized), std::string(ffn.prefix())));
  attn_hc_->load_state_dict(get_hc_state(state, "hc_attn_pre.", "hc_attn_"));
  ffn_hc_->load_state_dict(get_hc_state(state, "hc_ffn_pre.", "hc_ffn_"));
}

std::unique_ptr<DeepseekV41EngramImpl::HashState>
DeepseekV41DecoderLayerImpl::create_hash_state(int64_t rows,
                                               int64_t query_width) const {
  return engram_ ? engram_->create_hash_state(rows, query_width) : nullptr;
}

void DeepseekV41DecoderLayerImpl::load_hash_state(const StateDict& state) {
  if (engram_) {
    engram_->load_hash_state(state);
  }
}

void DeepseekV41DecoderLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  attn_norm_->verify_loaded_weights(prefix + "attn_norm.");
  ffn_norm_->verify_loaded_weights(prefix + "ffn_norm.");
  attn_hc_->verify_loaded_weights(prefix + "hc_attn_pre.");
  ffn_hc_->verify_loaded_weights(prefix + "hc_ffn_pre.");
  attention_->verify_loaded_weights();
  if (engram_) {
    engram_->verify_loaded_weights();
  }
  moe_->verify_loaded_weights();
}

torch::Tensor DeepseekV41DecoderLayerImpl::forward(
    torch::Tensor hidden,
    torch::Tensor& pre_mix,
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& rope,
    std::vector<KVCache>& caches,
    const ModelInputParams& input,
    Csa2Context& shared,
    DeepseekV41EngramImpl::HashState* hash_state,
    const mlu_v4_cp::DeepseekV4CpContext* cp_context,
    torch::Tensor* attention_input) {
  // DSpark captures the layer input before Engram injection.
  if (attention_input != nullptr) {
    *attention_input =
        hidden.to(torch::kFloat32).mean(1).to(hidden.scalar_type());
  }
  if (engram_) {
    hidden = engram_->forward(
        hidden,
        tokens,
        input,
        hash_state,
        cp_context ? cp_context->local_row_indices : torch::Tensor());
  }
  auto attn = attn_hc_->forward(hidden, std::nullopt, pre_mix);
  torch::Tensor x = std::get<0>(attn_norm_(attn.output, std::nullopt));
  x = attention_->forward(
      x, positions, rope, caches, input, shared, cp_context);
  hidden = std::get<0>(post_->forward(x, hidden, attn.post, attn.comb));
  auto ffn = ffn_hc_->forward(hidden, std::nullopt, attn.pre);
  x = std::get<0>(ffn_norm_(ffn.output, std::nullopt));
  if (cp_context != nullptr) {
    x = moe_->forward_cp(x,
                         std::nullopt,
                         cp_context->geometry.tokens_per_rank,
                         cp_context->cp_group);
  } else {
    const auto route = moe_->prep_route(x);
    x = moe_->forward_selected(x, route.reduce_weight, route.expert_id, input);
  }
  pre_mix = ffn.pre;
  return std::get<0>(post_->forward(x, hidden, ffn.post, ffn.comb));
}

}  // namespace xllm::layer
