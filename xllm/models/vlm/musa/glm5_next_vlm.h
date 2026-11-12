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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "models/llm/musa/glm5_next.h"
#include "models/model_registry.h"
#include "models/vlm/qwen3_vl_base.h"

namespace xllm::musa::model {

class Glm5NextVisionPatchEmbedImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionPatchEmbedImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    const ModelArgs& model_args = context.get_model_args();
    in_channels_ = model_args.mm_num_channels();
    temporal_patch_size_ = model_args.mm_temporal_patch_size();
    patch_size_ = model_args.mm_patch_size();
    embed_dim_ = model_args.mm_hidden_size();
    CHECK_GT(in_channels_, 0);
    CHECK_GT(temporal_patch_size_, 0);
    CHECK_GT(patch_size_, 0);
    CHECK_GT(embed_dim_, 0);

    proj_weight_ = register_parameter("proj_weight",
                                      torch::empty({embed_dim_,
                                                    in_channels_,
                                                    temporal_patch_size_,
                                                    patch_size_,
                                                    patch_size_},
                                                   options_),
                                      /*requires_grad=*/false);
    proj_bias_ = register_parameter("proj_bias",
                                    torch::empty({embed_dim_}, options_),
                                    /*requires_grad=*/false);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    namespace F = torch::nn::functional;
    torch::Tensor x = hidden_states.view(
        {-1, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    return F::conv3d(
               x,
               proj_weight_,
               F::Conv3dFuncOptions()
                   .bias(proj_bias_)
                   .stride({temporal_patch_size_, patch_size_, patch_size_})
                   .padding(0))
        .view({-1, embed_dim_});
  }

  void load_state_dict(const StateDict& state_dict) {
    const torch::Tensor weight = state_dict.get_tensor("proj.weight");
    if (weight.defined()) {
      CHECK_EQ(proj_weight_.sizes(), weight.sizes())
          << "proj.weight size mismatch for " << name();
      proj_weight_.copy_(weight.to(options_));
      proj_weight_loaded_ = true;
    }
    const torch::Tensor bias = state_dict.get_tensor("proj.bias");
    if (bias.defined()) {
      CHECK_EQ(proj_bias_.sizes(), bias.sizes())
          << "proj.bias size mismatch for " << name();
      proj_bias_.copy_(bias.to(options_));
      proj_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(proj_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj.bias";
  }

 private:
  int64_t in_channels_ = 0;
  int64_t temporal_patch_size_ = 0;
  int64_t patch_size_ = 0;
  int64_t embed_dim_ = 0;
  bool proj_weight_loaded_ = false;
  bool proj_bias_loaded_ = false;
  torch::TensorOptions options_;
  torch::Tensor proj_weight_;
  torch::Tensor proj_bias_;
};
TORCH_MODULE(Glm5NextVisionPatchEmbed);

class Glm5NextVisionRotaryEmbeddingImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionRotaryEmbeddingImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    const int64_t head_dim =
        model_args.mm_head_dim() > 0
            ? model_args.mm_head_dim()
            : model_args.mm_hidden_size() / model_args.mm_num_attention_heads();
    dim_ = head_dim / 2;
    theta_ = 10000.0;
    torch::TensorOptions fp32_options = options.dtype(torch::kFloat32);
    torch::Tensor inv_freq =
        1.0f / torch::pow(theta_,
                          torch::arange(0, dim_, 2, fp32_options) /
                              static_cast<double>(dim_));
    inv_freq_ = register_buffer("inv_freq", inv_freq);
  }

  void update_freqs_cache(int64_t seqlen) {
    if (seqlen <= seq_len_cached_) {
      return;
    }
    seqlen = std::max<int64_t>(seqlen * 2, 64);
    seq_len_cached_ = seqlen;
    torch::TensorOptions fp32_options = torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(inv_freq_.device());
    torch::Tensor seq = torch::arange(seqlen, fp32_options);
    freqs_cached_ = torch::outer(seq, inv_freq_);
  }

  torch::Tensor forward(int64_t seqlen) {
    update_freqs_cache(seqlen);
    return freqs_cached_.slice(/*dim=*/0, /*start=*/0, /*end=*/seqlen);
  }

 private:
  int64_t dim_ = 0;
  double theta_ = 10000.0;
  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Glm5NextVisionRotaryEmbedding);

class Glm5NextVisionAttentionImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionAttentionImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    const int64_t hidden_size = args.mm_hidden_size();
    const int64_t num_heads = args.mm_num_attention_heads();
    const int64_t head_dim =
        args.mm_head_dim() > 0 ? args.mm_head_dim() : hidden_size / num_heads;
    tp_group_ = parallel_args.tp_group_;
    const int64_t tp_size = tp_group_ != nullptr ? tp_group_->world_size() : 1;
    CHECK_EQ(num_heads % tp_size, 0)
        << "mm_num_attention_heads (" << num_heads
        << ") must be divisible by tp_size (" << tp_size << ")";

    hidden_size_per_attention_head_ = head_dim;
    num_attention_heads_per_partition_ = num_heads / tp_size;
    scale_ =
        1.0f / std::sqrt(static_cast<float>(hidden_size_per_attention_head_));

    qkv_proj_ = register_module(
        "qkv_proj",
        layer::QKVParallelLinear(hidden_size,
                                 num_attention_heads_per_partition_,
                                 num_attention_heads_per_partition_,
                                 hidden_size_per_attention_head_,
                                 /*num_kv_head_replicas=*/1,
                                 /*bias=*/true,
                                 /*gather_output=*/false,
                                 parallel_args,
                                 options,
                                 QuantArgs{}));
    q_norm_ = register_module("q_norm",
                              layer::RMSNorm(hidden_size_per_attention_head_,
                                             args.mm_layer_norm_eps(),
                                             options));
    k_norm_ = register_module("k_norm",
                              layer::RMSNorm(hidden_size_per_attention_head_,
                                             args.mm_layer_norm_eps(),
                                             options));
    proj_ = register_module(
        "proj",
        layer::RowParallelLinear(hidden_size,
                                 hidden_size,
                                 /*bias=*/true,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 QuantArgs{},
                                 tp_group_,
                                 options));
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        const torch::Tensor& precomputed_cos_sin,
                        const torch::Tensor& position_ids,
                        const std::vector<int32_t>& cu_seq_len_vec,
                        int32_t max_seqlen) {
    const int64_t total_tokens = hidden_states.size(0);
    torch::Tensor qkv = qkv_proj_->forward(hidden_states);
    std::vector<torch::Tensor> qkv_chunks = qkv.chunk(3, /*dim=*/-1);
    torch::Tensor q = qkv_chunks[0]
                          .reshape({total_tokens,
                                    num_attention_heads_per_partition_,
                                    hidden_size_per_attention_head_})
                          .contiguous();
    torch::Tensor k = qkv_chunks[1]
                          .reshape({total_tokens,
                                    num_attention_heads_per_partition_,
                                    hidden_size_per_attention_head_})
                          .contiguous();
    torch::Tensor v = qkv_chunks[2]
                          .reshape({total_tokens,
                                    num_attention_heads_per_partition_,
                                    hidden_size_per_attention_head_})
                          .contiguous();

    q = std::get<0>(q_norm_->forward(q));
    k = std::get<0>(k_norm_->forward(k));

    xllm::kernel::RotaryParams rotary_params;
    rotary_params.q = q.to(torch::kFloat32);
    rotary_params.k = k.to(torch::kFloat32);
    rotary_params.precomputed_cos_sin = precomputed_cos_sin;
    rotary_params.interleaved = false;
    rotary_params.discrete = false;
    rotary_params.position_ids = position_ids;
    rotary_params.cu_query_lens = std::nullopt;
    rotary_params.max_query_len = max_seqlen;
    xllm::kernel::apply_rotary(rotary_params);
    q = rotary_params.q.to(hidden_states.scalar_type());
    k = rotary_params.k.to(hidden_states.scalar_type());

    torch::Tensor output = torch::empty_like(q);
    const int32_t num_sequences =
        static_cast<int32_t>(cu_seq_len_vec.size()) - 1;
    for (int32_t index = 0; index < num_sequences; ++index) {
      const int32_t start = cu_seq_len_vec[index];
      const int32_t end = cu_seq_len_vec[index + 1];
      if (end <= start) {
        continue;
      }
      const torch::Tensor q_i =
          q.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0);
      const torch::Tensor k_i =
          k.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0);
      const torch::Tensor v_i =
          v.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0);
      const torch::Tensor out_i =
          torch::scaled_dot_product_attention(q_i,
                                              k_i,
                                              v_i,
                                              /*attn_mask=*/std::nullopt,
                                              /*dropout_p=*/0.0,
                                              /*is_causal=*/false,
                                              /*scale=*/scale_);
      output.slice(/*dim=*/0, start, end)
          .copy_(out_i.squeeze(/*dim=*/0).permute({1, 0, 2}));
    }

    output = output.reshape(
        {total_tokens,
         num_attention_heads_per_partition_ * hidden_size_per_attention_head_});
    return proj_->forward(output);
  }

  void load_state_dict(const StateDict& state_dict) {
    qkv_proj_->load_state_dict(state_dict.get_dict_with_prefix("qkv."));
    q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
    k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
    proj_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(qkv_proj_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "qkv.weight";
    q_norm_->verify_loaded_weights(prefix + "q_norm.");
    k_norm_->verify_loaded_weights(prefix + "k_norm.");
    CHECK(proj_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "proj.weight";
  }

 private:
  int64_t hidden_size_per_attention_head_ = 0;
  int64_t num_attention_heads_per_partition_ = 0;
  float scale_ = 1.0f;
  ProcessGroup* tp_group_ = nullptr;
  layer::QKVParallelLinear qkv_proj_{nullptr};
  layer::RMSNorm q_norm_{nullptr};
  layer::RMSNorm k_norm_{nullptr};
  layer::RowParallelLinear proj_{nullptr};
};
TORCH_MODULE(Glm5NextVisionAttention);

class Glm5NextVisionBlockImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionBlockImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    const int64_t dim = args.mm_hidden_size();
    const double swiglu_limit =
        args.swiglu_limit() > 0.0f ? args.swiglu_limit() : 10.0;

    norm1_ = register_module(
        "norm1", layer::RMSNorm(dim, args.mm_layer_norm_eps(), options));
    attn_ = register_module("attn", Glm5NextVisionAttention(context));
    norm2_ = register_module(
        "norm2", layer::RMSNorm(dim, args.mm_layer_norm_eps(), options));
    mlp_ = register_module("mlp",
                           layer::DenseMLP(dim,
                                           args.mm_intermediate_size(),
                                           /*is_gated=*/true,
                                           /*has_bias=*/true,
                                           args.mm_hidden_act(),
                                           /*enable_result_reduction=*/true,
                                           QuantArgs{},
                                           parallel_args.tp_group_,
                                           options,
                                           /*module_prefix=*/"",
                                           swiglu_limit));
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        const torch::Tensor& precomputed_cos_sin,
                        const torch::Tensor& position_ids,
                        const std::vector<int32_t>& cu_seq_len_vec,
                        int32_t max_seqlen) {
    torch::Tensor norm1_out = std::get<0>(norm1_->forward(hidden_states));
    hidden_states = hidden_states + attn_->forward(norm1_out,
                                                   precomputed_cos_sin,
                                                   position_ids,
                                                   cu_seq_len_vec,
                                                   max_seqlen);
    torch::Tensor norm2_out = std::get<0>(norm2_->forward(hidden_states));
    hidden_states = hidden_states + mlp_->forward(norm2_out);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    attn_->verify_loaded_weights(prefix + "attn.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
  }

 private:
  layer::RMSNorm norm1_{nullptr};
  Glm5NextVisionAttention attn_{nullptr};
  layer::RMSNorm norm2_{nullptr};
  layer::DenseMLP mlp_{nullptr};
};
TORCH_MODULE(Glm5NextVisionBlock);

class Glm5NextVisionPatchMergerImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionPatchMergerImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    const int64_t dim = args.mm_projection_dim();
    const int64_t context_dim = args.mm_projector_hidden_size() > 0
                                    ? args.mm_projector_hidden_size()
                                    : 10240;
    const double swiglu_limit =
        args.swiglu_limit() > 0.0f ? args.swiglu_limit() : 10.0;

    proj_ =
        register_module("proj",
                        layer::ReplicatedLinear(
                            dim, dim, /*bias=*/false, QuantArgs{}, options));
    post_projection_norm_ =
        register_module("post_projection_norm",
                        layer::RMSNorm(dim, args.mm_layer_norm_eps(), options));
    post_projection_norm_->set_layernorm_mode();
    mlp_ = register_module("mlp",
                           layer::DenseMLP(dim,
                                           context_dim,
                                           /*is_gated=*/true,
                                           /*has_bias=*/false,
                                           "silu",
                                           /*enable_result_reduction=*/true,
                                           QuantArgs{},
                                           parallel_args.tp_group_,
                                           options,
                                           /*module_prefix=*/"",
                                           swiglu_limit));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    torch::Tensor hidden_states = proj_->forward(x);
    hidden_states = std::get<0>(post_projection_norm_->forward(hidden_states));
    hidden_states = torch::gelu(hidden_states, "none");
    return mlp_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    proj_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
    post_projection_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("post_projection_norm."));
    mlp_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "proj.weight";
    post_projection_norm_->verify_loaded_weights(prefix +
                                                 "post_projection_norm.");
    mlp_->verify_loaded_weights(prefix);
  }

 private:
  layer::ReplicatedLinear proj_{nullptr};
  layer::RMSNorm post_projection_norm_{nullptr};
  layer::DenseMLP mlp_{nullptr};
};
TORCH_MODULE(Glm5NextVisionPatchMerger);

class Glm5NextVisionTransformerImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionTransformerImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    const ModelArgs& model_args = context.get_model_args();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    hidden_size_ = model_args.mm_hidden_size();
    out_hidden_size_ = model_args.mm_projection_dim();
    CHECK_GT(spatial_merge_size_, 0);
    CHECK_GT(hidden_size_, 0);
    CHECK_GT(out_hidden_size_, 0);

    patch_embed_ =
        register_module("patch_embed", Glm5NextVisionPatchEmbed(context));
    rotary_pos_emb_ = register_module("rotary_pos_emb",
                                      Glm5NextVisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    layers_.reserve(static_cast<size_t>(model_args.mm_num_hidden_layers()));
    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); ++idx) {
      Glm5NextVisionBlock block(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    post_layernorm_ = register_module(
        "post_layernorm",
        layer::RMSNorm(hidden_size_, model_args.mm_layer_norm_eps(), options_));
    downsample_weight_ = register_parameter("downsample_weight",
                                            torch::empty({out_hidden_size_,
                                                          hidden_size_,
                                                          spatial_merge_size_,
                                                          spatial_merge_size_},
                                                         options_),
                                            /*requires_grad=*/false);
    downsample_bias_ =
        register_parameter("downsample_bias",
                           torch::empty({out_hidden_size_}, options_),
                           /*requires_grad=*/false);
    merger_ = register_module("merger", Glm5NextVisionPatchMerger(context));
  }

  torch::Tensor rot_pos_emb(const torch::Tensor& grid_thw_cpu) {
    const int64_t count = grid_thw_cpu.size(0);
    std::vector<torch::Tensor> pos_ids_vec;
    pos_ids_vec.reserve(static_cast<size_t>(count));
    torch::TensorOptions long_cpu_opts =
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    int64_t max_grid_size = 0;
    for (int64_t idx = 0; idx < count; ++idx) {
      const int64_t t = grid_thw_cpu[idx][0].item<int64_t>();
      const int64_t h = grid_thw_cpu[idx][1].item<int64_t>();
      const int64_t w = grid_thw_cpu[idx][2].item<int64_t>();
      max_grid_size = std::max(max_grid_size, std::max(h, w));

      torch::Tensor hpos_ids =
          torch::arange(h, long_cpu_opts).unsqueeze(1).expand({-1, w});
      hpos_ids = hpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      torch::Tensor wpos_ids =
          torch::arange(w, long_cpu_opts).unsqueeze(0).expand({h, -1});
      wpos_ids = wpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();
      pos_ids_vec.push_back(
          torch::stack({hpos_ids, wpos_ids}, /*dim=*/-1).repeat({t, 1}));
    }
    torch::Tensor pos_ids =
        torch::cat(pos_ids_vec, /*dim=*/0).to(options_.device());
    torch::Tensor rotary_pos_emb_full = rotary_pos_emb_->forward(max_grid_size);
    return rotary_pos_emb_full.index({pos_ids}).flatten(1);
  }

  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor grid_thw) {
    torch::NoGradGuard no_grad;
    hidden_states = patch_embed_->forward(hidden_states);
    torch::Tensor grid_thw_cpu =
        grid_thw.to(torch::kCPU).to(torch::kLong).contiguous();
    torch::Tensor rotary_pos_emb = rot_pos_emb(grid_thw_cpu);
    torch::Tensor precomputed_cos_sin =
        torch::cat({rotary_pos_emb.cos(), rotary_pos_emb.sin()}, /*dim=*/-1)
            .contiguous();

    std::vector<int32_t> cu_seqlens_vec;
    const int64_t num_grids = grid_thw_cpu.size(0);
    cu_seqlens_vec.push_back(0);
    int32_t running_offset = 0;
    int32_t max_seqlen = 0;
    for (int64_t idx = 0; idx < num_grids; ++idx) {
      const int64_t t = grid_thw_cpu[idx][0].item<int64_t>();
      const int64_t h = grid_thw_cpu[idx][1].item<int64_t>();
      const int64_t w = grid_thw_cpu[idx][2].item<int64_t>();
      const int32_t frame_tokens = static_cast<int32_t>(h * w);
      max_seqlen = std::max(max_seqlen, frame_tokens);
      for (int64_t frame = 0; frame < t; ++frame) {
        running_offset += frame_tokens;
        cu_seqlens_vec.push_back(running_offset);
      }
    }

    torch::Tensor position_ids =
        torch::arange(hidden_states.size(0),
                      torch::TensorOptions()
                          .dtype(torch::kInt64)
                          .device(hidden_states.device()))
            .contiguous();

    for (size_t idx = 0; idx < layers_.size(); ++idx) {
      hidden_states = layers_[idx]->forward(hidden_states,
                                            precomputed_cos_sin,
                                            position_ids,
                                            cu_seqlens_vec,
                                            max_seqlen);
    }

    hidden_states = std::get<0>(post_layernorm_->forward(hidden_states));
    hidden_states = hidden_states.view(
        {-1, spatial_merge_size_, spatial_merge_size_, hidden_states.size(-1)});
    hidden_states = hidden_states.permute({0, 3, 1, 2}).contiguous();
    namespace F = torch::nn::functional;
    hidden_states = F::conv2d(hidden_states,
                              downsample_weight_,
                              F::Conv2dFuncOptions()
                                  .bias(downsample_bias_)
                                  .stride(spatial_merge_size_)
                                  .padding(0))
                        .view({-1, out_hidden_size_});
    hidden_states = merger_->forward(hidden_states);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (size_t idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }
    post_layernorm_->load_state_dict(
        state_dict.get_dict_with_prefix("post_layernorm."));

    const StateDict downsample_dict =
        state_dict.get_dict_with_prefix("downsample.");
    const torch::Tensor downsample_weight =
        downsample_dict.get_tensor("weight");
    if (downsample_weight.defined()) {
      CHECK_EQ(downsample_weight_.sizes(), downsample_weight.sizes())
          << "downsample.weight size mismatch for " << name();
      downsample_weight_.copy_(downsample_weight.to(options_));
      downsample_weight_loaded_ = true;
    }
    const torch::Tensor downsample_bias = downsample_dict.get_tensor("bias");
    if (downsample_bias.defined()) {
      CHECK_EQ(downsample_bias_.sizes(), downsample_bias.sizes())
          << "downsample.bias size mismatch for " << name();
      downsample_bias_.copy_(downsample_bias.to(options_));
      downsample_bias_loaded_ = true;
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    for (size_t idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->verify_loaded_weights(prefix + "blocks." +
                                          std::to_string(idx) + ".");
    }
    post_layernorm_->verify_loaded_weights(prefix + "post_layernorm.");
    CHECK(downsample_weight_loaded_)
        << "weight is not loaded for " << prefix + "downsample.weight";
    CHECK(downsample_bias_loaded_)
        << "bias is not loaded for " << prefix + "downsample.bias";
    merger_->verify_loaded_weights(prefix + "merger.");
  }

 private:
  int64_t spatial_merge_size_ = 2;
  int64_t hidden_size_ = 0;
  int64_t out_hidden_size_ = 0;
  bool downsample_weight_loaded_ = false;
  bool downsample_bias_loaded_ = false;
  torch::TensorOptions options_;
  Glm5NextVisionPatchEmbed patch_embed_{nullptr};
  Glm5NextVisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Glm5NextVisionBlock> layers_;
  layer::RMSNorm post_layernorm_{nullptr};
  torch::Tensor downsample_weight_;
  torch::Tensor downsample_bias_;
  Glm5NextVisionPatchMerger merger_{nullptr};
};
TORCH_MODULE(Glm5NextVisionTransformer);

class Glm5NextForConditionalGenerationImpl final
    : public Qwen3VLForConditionalGenerationBase<Glm5NextVisionTransformer,
                                                 Glm5NextForCausalLM> {
 public:
  explicit Glm5NextForConditionalGenerationImpl(const ModelContext& context)
      : Qwen3VLForConditionalGenerationBase<Glm5NextVisionTransformer,
                                            Glm5NextForCausalLM>(context) {}

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return language_model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    language_model_->prepare_graph_forward_metadata(state, positions, params);
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    if (!input_ids.defined() || input_ids.numel() == 0) {
      return torch::Tensor();
    }
    return Qwen3VLForConditionalGenerationBase<
        Glm5NextVisionTransformer,
        Glm5NextForCausalLM>::get_input_embeddings(input_ids, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) {
    return language_model_->logits(hidden_states, seleted_idxes, out_hidden);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }
    visual_->verify_loaded_weights("model.visual.");
    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
    }
  }
};
TORCH_MODULE(Glm5NextForConditionalGeneration);

REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(glm5_next_vlm,
                                       glm5_next,
                                       Glm5NextForConditionalGeneration);

}  // namespace xllm::musa::model
