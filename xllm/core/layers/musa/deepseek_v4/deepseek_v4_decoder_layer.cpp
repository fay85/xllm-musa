/* Copyright 2026 The xLLM Authors.

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

#include "core/layers/musa/deepseek_v4/deepseek_v4_decoder_layer.h"

#include <glog/logging.h>

#include "core/common/flash_comm1_context.h"

namespace xllm {
namespace layer {

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  int64_t hidden_size = args.hidden_size();

  hc_mult_ = args.hc_mult();
  hc_sinkhorn_iters_ = args.hc_sinkhorn_iters();
  hc_eps_ = static_cast<double>(args.hc_eps());
  norm_eps_ = static_cast<double>(args.rms_norm_eps());

  attention_ = register_module("attn", DeepseekV4Attention(context, layer_id));
  attn_norm_ = register_module(
      "attn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  ffn_norm_ = register_module(
      "ffn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  moe_mlp_ = register_module(
      "ffn", DeepseekV4MoE(args, quant_args, parallel_args, options));
  // Register as "gate" to match Python's mlp.gate module path.
  gate_ = register_module("gate", DeepseekV4Gate(context, layer_id));

  const int64_t mix_hc = (2 + hc_mult_) * hc_mult_;
  const int64_t hc_dim = hc_mult_ * hidden_size;
  auto hc_options = options.dtype(torch::kFloat32);
  hc_attn_fn_ = register_parameter("hc_attn_fn",
                                   torch::empty({mix_hc, hc_dim}, hc_options),
                                   /*requires_grad=*/false);
  hc_ffn_fn_ = register_parameter("hc_ffn_fn",
                                  torch::empty({mix_hc, hc_dim}, hc_options),
                                  /*requires_grad=*/false);
  hc_attn_base_ = register_parameter("hc_attn_base",
                                     torch::empty({mix_hc}, hc_options),
                                     /*requires_grad=*/false);
  hc_ffn_base_ = register_parameter("hc_ffn_base",
                                    torch::empty({mix_hc}, hc_options),
                                    /*requires_grad=*/false);
  hc_attn_scale_ = register_parameter("hc_attn_scale",
                                      torch::empty({3}, hc_options),
                                      /*requires_grad=*/false);
  hc_ffn_scale_ = register_parameter("hc_ffn_scale",
                                     torch::empty({3}, hc_options),
                                     /*requires_grad=*/false);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attn_state = state_dict.get_dict_with_prefix("attn.");
  if (attn_state.size() == 0) {
    attn_state = state_dict.get_dict_with_prefix("self_attn.");
  }
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
  }

  auto attn_norm_state = state_dict.get_dict_with_prefix("attn_norm.");
  if (attn_norm_state.size() == 0) {
    attn_norm_state = state_dict.get_dict_with_prefix("input_layernorm.");
  }
  if (attn_norm_state.size() > 0) {
    attn_norm_->load_state_dict(attn_norm_state);
  }

  auto ffn_norm_state = state_dict.get_dict_with_prefix("ffn_norm.");
  if (ffn_norm_state.size() == 0) {
    ffn_norm_state =
        state_dict.get_dict_with_prefix("post_attention_layernorm.");
  }
  if (ffn_norm_state.size() > 0) {
    ffn_norm_->load_state_dict(ffn_norm_state);
  }

  auto ffn_state = state_dict.get_dict_with_prefix("ffn.");
  if (ffn_state.size() == 0) {
    ffn_state = state_dict.get_dict_with_prefix("mlp.");
  }
  if (ffn_state.size() > 0) {
    auto gate_state = ffn_state.get_dict_with_prefix("gate.");
    if (gate_state.size() == 0) {
      gate_state = state_dict.get_dict_with_prefix("gate.");
    }
    if (gate_state.size() > 0) {
      gate_->load_state_dict(gate_state);
    }
    moe_mlp_->load_state_dict(ffn_state);
  }

  LOAD_WEIGHT(hc_attn_fn);
  LOAD_WEIGHT(hc_ffn_fn);
  LOAD_WEIGHT(hc_attn_base);
  LOAD_WEIGHT(hc_ffn_base);
  LOAD_WEIGHT(hc_attn_scale);
  LOAD_WEIGHT(hc_ffn_scale);
}

void DeepseekV4DecoderLayerImpl::verify_loaded_weights() const {}

torch::Tensor DeepseekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& input_ids) {
  (void)positions;

  residual = std::nullopt;

  CHECK(attn_metadata.dsa_metadata)
      << "DeepseekV4DecoderLayer requires DSA metadata for the DSA path.";

  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();

  auto residual_attn = x;
  auto [attn_input, post_attn, comb_attn] =
      hc_pre(x, hc_attn_fn_, hc_attn_scale_, hc_attn_base_);
  attn_input = std::get<0>(attn_norm_->forward(attn_input));

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    attn_input = gather_sequence(attn_input, *fc1_ctx);
  }

  auto& dsa = *(attn_metadata.dsa_metadata);
  const auto compress_metadata = std::make_tuple(
      dsa.c1_metadata, dsa.c4_metadata, dsa.c128_metadata, dsa.qli_metadata);
  KVState kv_state{kv_cache.get_swa_cache(),
                   kv_cache.get_compress_kv_state(),
                   kv_cache.get_compress_score_state(),
                   kv_cache.get_compress_index_kv_state(),
                   kv_cache.get_compress_index_score_state()};
  auto [attn_output, attn_lse] =
      attention_->forward(dsa,
                          attn_input,
                          kv_cache,
                          kv_state,
                          attn_metadata.is_prefill,
                          attn_metadata.is_chunked_prefill,
                          compress_metadata);
  (void)attn_lse;
  attn_input = attn_output;
  x = hc_post(attn_input, residual_attn, post_attn, comb_attn);

  auto residual_ffn = x;
  auto [ffn_input, post_ffn, comb_ffn] =
      hc_pre(x, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_);
  ffn_input = std::get<0>(ffn_norm_->forward(ffn_input));

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    ffn_input = gather_sequence(ffn_input, *fc1_ctx);
  }

  auto ffn_input_2d = ffn_input.reshape({-1, ffn_input.size(-1)});
  std::optional<torch::Tensor> gate_input_ids = std::nullopt;
  if (input_ids.has_value() && input_ids.value().defined()) {
    auto flat_input_ids =
        input_ids.value().reshape({-1}).to(ffn_input.device());
    const int64_t token_count = flat_input_ids.size(0);
    const int64_t hidden_rows = ffn_input_2d.size(0);
    if (token_count == hidden_rows) {
      gate_input_ids = flat_input_ids;
    } else if (token_count > 0 && hidden_rows % token_count == 0) {
      const int64_t repeat_factor = hidden_rows / token_count;
      gate_input_ids = flat_input_ids.unsqueeze(1)
                           .repeat({1, repeat_factor})
                           .reshape({hidden_rows});
    }
  }
  if (gate_->is_hash_layer()) {
    CHECK(gate_input_ids.has_value())
        << "DeepseekV4 hash gate requires input_ids for routing";
  }
  auto [topk_weights, topk_ids] = gate_->forward(ffn_input_2d, gate_input_ids);
  ffn_input = moe_mlp_->forward_with_selected_experts(
      ffn_input, topk_weights, topk_ids, input_params);

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    ffn_input = shard_sequence(ffn_input, *fc1_ctx);
  }
  x = hc_post(ffn_input, residual_ffn, post_ffn, comb_ffn);

  return x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4DecoderLayerImpl::hc_pre(const torch::Tensor& x,
                                   const torch::Tensor& hc_fn,
                                   const torch::Tensor& hc_scale,
                                   const torch::Tensor& hc_base) {
  const int64_t dim = x.size(-1);
  torch::Tensor x_hc = x.reshape({-1, hc_mult_, dim}).contiguous();
  torch::Tensor x_flat =
      x_hc.reshape({x_hc.size(0), hc_mult_ * dim}).to(torch::kFloat32);
  torch::Tensor rsqrt = torch::rsqrt(
      x_flat.pow(2).mean(/*dim=*/-1, /*keepdim=*/true) + norm_eps_);
  torch::Tensor mixes = torch::matmul(x_flat, hc_fn.transpose(0, 1)) * rsqrt;

  const int64_t first = hc_mult_;
  const int64_t second = 2 * hc_mult_;
  torch::Tensor scale = hc_scale.to(torch::kFloat32);
  torch::Tensor base = hc_base.to(torch::kFloat32);
  torch::Tensor pre_logits = mixes.slice(/*dim=*/1, /*start=*/0, /*end=*/first);
  torch::Tensor post_logits =
      mixes.slice(/*dim=*/1, /*start=*/first, /*end=*/second);
  torch::Tensor comb_logits = mixes.slice(/*dim=*/1, /*start=*/second)
                                  .reshape({-1, hc_mult_, hc_mult_});

  torch::Tensor pre =
      torch::sigmoid(pre_logits * scale[0] +
                     base.slice(/*dim=*/0, /*start=*/0, /*end=*/first)) +
      hc_eps_;
  torch::Tensor post =
      2 *
      torch::sigmoid(post_logits * scale[1] +
                     base.slice(/*dim=*/0, /*start=*/first, /*end=*/second));
  torch::Tensor comb_base =
      base.slice(/*dim=*/0, /*start=*/second).reshape({1, hc_mult_, hc_mult_});
  torch::Tensor comb =
      torch::softmax(comb_logits * scale[2] + comb_base, /*dim=*/-1) + hc_eps_;

  comb = comb * (comb.sum(/*dim=*/-2, /*keepdim=*/true) + hc_eps_).reciprocal();
  for (int64_t i = 1; i < hc_sinkhorn_iters_; ++i) {
    comb =
        comb * (comb.sum(/*dim=*/-1, /*keepdim=*/true) + hc_eps_).reciprocal();
    comb =
        comb * (comb.sum(/*dim=*/-2, /*keepdim=*/true) + hc_eps_).reciprocal();
  }

  torch::Tensor output =
      (x_hc.to(torch::kFloat32) * pre.unsqueeze(-1)).sum(/*dim=*/1);
  std::vector<int64_t> output_shape;
  output_shape.reserve(static_cast<size_t>(x.dim() - 1));
  for (int64_t i = 0; i < x.dim() - 2; ++i) {
    output_shape.push_back(x.size(i));
  }
  output_shape.push_back(dim);

  std::vector<int64_t> post_shape = output_shape;
  post_shape.back() = hc_mult_;
  std::vector<int64_t> comb_shape = post_shape;
  comb_shape.push_back(hc_mult_);
  return std::make_tuple(output.reshape(output_shape).to(x.dtype()),
                         post.reshape(post_shape).to(torch::kFloat32),
                         comb.reshape(comb_shape).to(torch::kFloat32));
}

torch::Tensor DeepseekV4DecoderLayerImpl::hc_post(const torch::Tensor& x,
                                                  const torch::Tensor& residual,
                                                  const torch::Tensor& post,
                                                  const torch::Tensor& comb) {
  const int64_t dim = residual.size(-1);
  torch::Tensor x_flat = x.reshape({-1, dim}).to(torch::kFloat32);
  torch::Tensor residual_flat =
      residual.reshape({-1, hc_mult_, dim}).to(torch::kFloat32);
  torch::Tensor post_flat = post.reshape({-1, hc_mult_}).to(torch::kFloat32);
  torch::Tensor comb_flat =
      comb.reshape({-1, hc_mult_, hc_mult_}).to(torch::kFloat32);

  std::vector<torch::Tensor> mixed_rows;
  mixed_rows.reserve(static_cast<size_t>(hc_mult_));
  for (int64_t out_idx = 0; out_idx < hc_mult_; ++out_idx) {
    torch::Tensor row = residual_flat.select(/*dim=*/1, /*index=*/0) *
                        comb_flat.select(/*dim=*/1, /*index=*/out_idx)
                            .select(/*dim=*/1, /*index=*/0)
                            .unsqueeze(-1);
    for (int64_t in_idx = 1; in_idx < hc_mult_; ++in_idx) {
      row = row + residual_flat.select(/*dim=*/1, /*index=*/in_idx) *
                      comb_flat.select(/*dim=*/1, /*index=*/out_idx)
                          .select(/*dim=*/1, /*index=*/in_idx)
                          .unsqueeze(-1);
    }
    mixed_rows.push_back(row);
  }
  torch::Tensor mixed = torch::stack(mixed_rows, /*dim=*/1);
  torch::Tensor output = mixed + post_flat.unsqueeze(-1) * x_flat.unsqueeze(1);
  return output.reshape(residual.sizes()).to(residual.dtype());
}

}  // namespace layer
}  // namespace xllm
