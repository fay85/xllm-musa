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

#include <cstdlib>
#include <string>

#include "core/common/flash_comm1_context.h"
#include "core/kernels/musa/deepseek_v4_hc_policy.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_gate_policy.h"

namespace xllm {
namespace layer {

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  auto quant_args = context.get_quant_args();
  // DeepSeek-V4 uses `scale` aliases consumed by the private FP8 linear path;
  // Qwen block-FP8 keeps using the shared `weight_scale_inv` path unchanged.
  quant_args.weight_block_size().clear();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  int64_t hidden_size = args.hidden_size();

  hc_mult_ = args.hc_mult();
  hc_sinkhorn_iters_ = args.hc_sinkhorn_iters();
  hc_eps_ = static_cast<double>(args.hc_eps());
  norm_eps_ = static_cast<double>(args.rms_norm_eps());

  attention_ = register_module("attn", DeepseekV4Attention(context, layer_id));
  CHECK(parallel_args.single_rank_group_ != nullptr)
      << "DeepSeek-V4 prefill sequence parallel requires a single-rank "
         "process group for replicated attention";
  ParallelArgs prefill_parallel_args = parallel_args;
  prefill_parallel_args.tp_group_ = parallel_args.single_rank_group_;
  prefill_attention_ = register_module(
      "prefill_attn",
      DeepseekV4Attention(
          args, quant_args, prefill_parallel_args, options, layer_id));
  attn_norm_ = register_module(
      "attn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  ffn_norm_ = register_module(
      "ffn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  moe_mlp_ = register_module(
      "ffn", DeepseekV4MoE(args, quant_args, parallel_args, options, layer_id));
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

  constexpr int64_t kHcDecodeWorkspaceTokens = 16;
  hc_pre_output_workspace_ =
      torch::empty({kHcDecodeWorkspaceTokens, hidden_size}, options);
  hc_pre_post_workspace_ =
      torch::empty({kHcDecodeWorkspaceTokens, hc_mult_}, hc_options);
  hc_pre_comb_workspace_ =
      torch::empty({kHcDecodeWorkspaceTokens, hc_mult_, hc_mult_}, hc_options);
  hc_post_output_workspace_ =
      torch::empty({kHcDecodeWorkspaceTokens, hc_mult_, hidden_size}, options);
  rms_norm_output_workspace_ =
      torch::empty({kHcDecodeWorkspaceTokens, hidden_size}, options);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attn_state = state_dict.get_dict_with_prefix("attn.");
  if (attn_state.size() == 0) {
    attn_state = state_dict.get_dict_with_prefix("self_attn.");
  }
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
    prefill_attention_->load_state_dict(attn_state);
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
  bool attn_hc_output_is_normalized = false;
  auto [attn_input, post_attn, comb_attn] =
      hc_pre(x,
             hc_attn_fn_,
             hc_attn_scale_,
             hc_attn_base_,
             attn_norm_->weight(),
             &attn_hc_output_is_normalized);
  std::optional<torch::Tensor> rms_norm_output = std::nullopt;
  if (input_params.enable_graph && attn_input.dim() == 2 &&
      attn_input.size(0) <= rms_norm_output_workspace_.size(0)) {
    rms_norm_output = rms_norm_output_workspace_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/attn_input.size(0));
  }
  if (!attn_hc_output_is_normalized) {
    attn_input = std::get<0>(
        attn_norm_->forward(attn_input, std::nullopt, rms_norm_output));
  }

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
  DeepseekV4Attention active_attention = attention_;
  if (sequence_parallel_context_ != nullptr) {
    prefill_attention_->set_sequence_parallel_context(
        sequence_parallel_context_);
    active_attention = prefill_attention_;
  }
  auto [attn_output, attn_lse] =
      active_attention->forward(dsa,
                                attn_input,
                                kv_cache,
                                kv_state,
                                attn_metadata.is_prefill,
                                attn_metadata.is_chunked_prefill,
                                compress_metadata);
  (void)attn_lse;
  attn_input = attn_output;
  if (sequence_parallel_context_ != nullptr &&
      attn_input.size(0) == sequence_parallel_context_->local_valid_tokens) {
    attn_input =
        dsv4_musa_sp::pad_local(attn_input, *sequence_parallel_context_);
  }
  x = hc_post(attn_input, residual_attn, post_attn, comb_attn);

  auto residual_ffn = x;
  bool ffn_hc_output_is_normalized = false;
  auto [ffn_input, post_ffn, comb_ffn] = hc_pre(x,
                                                hc_ffn_fn_,
                                                hc_ffn_scale_,
                                                hc_ffn_base_,
                                                ffn_norm_->weight(),
                                                &ffn_hc_output_is_normalized);
  if (input_params.enable_graph && ffn_input.dim() == 2 &&
      ffn_input.size(0) <= rms_norm_output_workspace_.size(0)) {
    rms_norm_output = rms_norm_output_workspace_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/ffn_input.size(0));
  } else {
    rms_norm_output = std::nullopt;
  }
  if (!ffn_hc_output_is_normalized) {
    ffn_input = std::get<0>(
        ffn_norm_->forward(ffn_input, std::nullopt, rms_norm_output));
  }

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    ffn_input = gather_sequence(ffn_input, *fc1_ctx);
  }

  const char* skip_sp_moe_gather_env =
      std::getenv("XLLM_DSV4_SP_SKIP_MOE_ALLGATHER");
  const bool skip_sp_moe_gather =
      skip_sp_moe_gather_env != nullptr &&
      std::string(skip_sp_moe_gather_env) == "1" &&
      sequence_parallel_context_ != nullptr;
  if (sequence_parallel_context_ != nullptr && !skip_sp_moe_gather) {
    const dsv4_musa_sp::DeepseekV4SPContext& sp = *sequence_parallel_context_;
    CHECK_EQ(ffn_input.size(0), sp.local_tokens)
        << "DeepSeek-V4 SP MoE input has unexpected local token rows";
    ffn_input = dsv4_musa_sp::gather_global(ffn_input, sp);
  }
  auto ffn_input_2d = ffn_input.reshape({-1, ffn_input.size(-1)});
  std::optional<torch::Tensor> gate_input_ids = std::nullopt;
  if (input_ids.has_value() && input_ids.value().defined()) {
    auto flat_input_ids =
        input_ids.value().reshape({-1}).to(ffn_input.device());
    if (sequence_parallel_context_ != nullptr) {
      const auto& global_input_ids =
          sequence_parallel_context_->global_input_ids;
      CHECK(global_input_ids.defined());
      CHECK_EQ(global_input_ids.numel(),
               sequence_parallel_context_->total_tokens);
      flat_input_ids = skip_sp_moe_gather
                           ? dsv4_musa_sp::local_shard(
                                 global_input_ids, *sequence_parallel_context_)
                           : global_input_ids.reshape({-1});
    }
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
  const bool batch_gate_verify = dsv4_batch_gate_verify_allowed(
      ffn_input_2d.size(0), input_params.meta.num_sequences,
      input_params.meta.q_max_seq_len, attn_metadata.dsa_metadata->is_acl_graph,
      input_params.meta.batch_forward_type.is_decode(), input_params.is_spec_verify);
  auto [topk_weights, topk_ids] = gate_->forward(
      ffn_input_2d, gate_input_ids, batch_gate_verify);
  ffn_input = moe_mlp_->forward_with_selected_experts(
      ffn_input, topk_weights, topk_ids, input_params);

  if (sequence_parallel_context_ != nullptr && !skip_sp_moe_gather) {
    const dsv4_musa_sp::DeepseekV4SPContext& sp = *sequence_parallel_context_;
    ffn_input = dsv4_musa_sp::local_shard(ffn_input, sp);
  } else if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    ffn_input = shard_sequence(ffn_input, *fc1_ctx);
  }
  x = hc_post(ffn_input, residual_ffn, post_ffn, comb_ffn);

  return x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4DecoderLayerImpl::hc_pre(const torch::Tensor& x,
                                   const torch::Tensor& hc_fn,
                                   const torch::Tensor& hc_scale,
                                   const torch::Tensor& hc_base,
                                   const torch::Tensor& norm_weight,
                                   bool* output_is_normalized) {
  if (output_is_normalized != nullptr) {
    *output_is_normalized = false;
  }
  const int64_t dim = x.size(-1);
  torch::Tensor x_hc = x.reshape({-1, hc_mult_, dim}).contiguous();
  if (x_hc.size(0) <= hc_pre_output_workspace_.size(0)) {
    return kernel::musa::deepseek_v4_hc_pre_decode(x_hc,
                                                   hc_fn.contiguous(),
                                                   hc_scale.contiguous(),
                                                   hc_base.contiguous(),
                                                   hc_mult_,
                                                   hc_sinkhorn_iters_,
                                                   norm_eps_,
                                                   hc_eps_,
                                                   hc_pre_output_workspace_,
                                                   hc_pre_post_workspace_,
                                                   hc_pre_comb_workspace_,
                                                   norm_weight,
                                                   output_is_normalized);
  }
  if (x_hc.size(0) <= kernel::musa::deepseek_v4_hc_prefill_row_limit(dim)) {
    return kernel::musa::deepseek_v4_hc_pre_decode(x_hc,
                                                   hc_fn.contiguous(),
                                                   hc_scale.contiguous(),
                                                   hc_base.contiguous(),
                                                   hc_mult_,
                                                   hc_sinkhorn_iters_,
                                                   norm_eps_,
                                                   hc_eps_,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   norm_weight,
                                                   output_is_normalized);
  }
  torch::Tensor x_flat =
      x_hc.reshape({x_hc.size(0), hc_mult_ * dim}).to(torch::kFloat32);
  torch::Tensor rsqrt = torch::rsqrt(
      x_flat.pow(2).mean(/*dim=*/-1, /*keepdim=*/true) + norm_eps_);
  torch::Tensor mixes = torch::matmul(x_flat, hc_fn.transpose(0, 1)) * rsqrt;

  auto [pre, post, comb] =
      kernel::musa::deepseek_v4_hc_pre_metadata(mixes.contiguous(),
                                                hc_scale.contiguous(),
                                                hc_base.contiguous(),
                                                hc_mult_,
                                                hc_sinkhorn_iters_,
                                                hc_eps_);
  auto output =
      kernel::musa::deepseek_v4_hc_pre_combine(x_hc, pre.contiguous());
  return std::make_tuple(output, post, comb);
}

torch::Tensor DeepseekV4DecoderLayerImpl::hc_post(const torch::Tensor& x,
                                                  const torch::Tensor& residual,
                                                  const torch::Tensor& post,
                                                  const torch::Tensor& comb) {
  std::optional<torch::Tensor> output = std::nullopt;
  if (residual.dim() == 3 &&
      residual.size(0) <= hc_post_output_workspace_.size(0) &&
      residual.size(1) == hc_post_output_workspace_.size(1) &&
      residual.size(2) == hc_post_output_workspace_.size(2)) {
    output = hc_post_output_workspace_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/residual.size(0));
  }
  return kernel::musa::deepseek_v4_hc_post(
      x.reshape({-1, residual.size(-1)}).contiguous(),
      residual.contiguous(),
      post.contiguous(),
      comb.contiguous(),
      output);
}

}  // namespace layer
}  // namespace xllm
