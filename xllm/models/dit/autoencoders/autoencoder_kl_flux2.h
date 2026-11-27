/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/add_matmul.h"
#include "framework/model_context.h"
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/utils/diagonal_gaussian_distribution.h"
#include "models/dit/utils/dit_parallel_mixin.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"

// VAE model for Flux2 with VAE parallel support.
// All sub-modules are Flux2-specific (Flux2*) with VaeParallelMixin,
// so the shared modules in autoencoder_kl.h remain untouched.

namespace xllm {

class Flux2VaeResnetBlock2DImpl final : public torch::nn::Module,
                                        public dit::VaeParallelMixin {
 public:
  Flux2VaeResnetBlock2DImpl(const ModelContext& context,
                            int64_t in_channels,
                            int64_t out_channels)
      : dit::VaeParallelMixin(context),
        in_channels_(in_channels),
        out_channels_(out_channels) {
    int64_t groups = context.get_model_args().norm_num_groups();

    norm1_ = register_module(
        "norm1",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(groups, in_channels_).eps(1e-6f)));
    conv1_ = register_module(
        "conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels_,
                                                   out_channels_,
                                                   /*kernel_size=*/3)
                              .stride(1)
                              .padding(0)
                              .bias(true)));
    norm2_ = register_module(
        "norm2",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(groups, out_channels_).eps(1e-6f)));
    conv2_ = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels_,
                                                   out_channels_,
                                                   /*kernel_size=*/3)
                              .stride(1)
                              .padding(0)
                              .bias(true)));
    nonlinearity_ =
        register_module("nonlinearity", torch::nn::Functional(torch::silu));

    if (in_channels_ != out_channels_) {
      conv_shortcut_ = register_module(
          "conv_shortcut",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels_,
                                                     out_channels_,
                                                     /*kernel_size=*/1)
                                .stride(1)
                                .padding(0)
                                .bias(true)));
    } else {
      identity_ = register_module("conv_shortcut", torch::nn::Identity());
    }
  }

  torch::Tensor forward(const torch::Tensor& input_tensor) {
    auto hidden = norm1_(input_tensor);
    hidden = nonlinearity_(hidden);

    // conv1 (3×3, padding=0): manual pad handles spatial padding.
    //   parallel: exchange adds W halo, F::pad adds H only → {0,0,1,1}
    //   non-parallel: F::pad adds both H and W → {1,1,1,1}
    if (vae_parallel_enabled()) {
      hidden = vae_parallel_exchange(hidden, /*pad=*/true);
      hidden = torch::nn::functional::pad(
          hidden, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
    } else {
      hidden = torch::nn::functional::pad(
          hidden, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}));
    }
    hidden = conv1_(hidden);

    hidden = norm2_(hidden);
    hidden = nonlinearity_(hidden);

    // conv2 (3×3, padding=0): same pattern.
    if (vae_parallel_enabled()) {
      hidden = vae_parallel_exchange(hidden, /*pad=*/true);
      hidden = torch::nn::functional::pad(
          hidden, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
    } else {
      hidden = torch::nn::functional::pad(
          hidden, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}));
    }
    hidden = conv2_(hidden);

    // Shortcut: 1×1 conv (no halo needed) or identity.
    auto shortcut =
        conv_shortcut_ ? conv_shortcut_(input_tensor) : identity_(input_tensor);
    return hidden + shortcut;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "norm1.weight", norm1_->weight, is_norm1_weight_loaded_);
    weight::load_weight(
        state_dict, "norm1.bias", norm1_->bias, is_norm1_bias_loaded_);
    weight::load_weight(
        state_dict, "conv1.weight", conv1_->weight, is_conv1_weight_loaded_);
    weight::load_weight(
        state_dict, "conv1.bias", conv1_->bias, is_conv1_bias_loaded_);
    weight::load_weight(
        state_dict, "norm2.weight", norm2_->weight, is_norm2_weight_loaded_);
    weight::load_weight(
        state_dict, "norm2.bias", norm2_->bias, is_norm2_bias_loaded_);
    weight::load_weight(
        state_dict, "conv2.weight", conv2_->weight, is_conv2_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2.bias", conv2_->bias, is_conv2_bias_loaded_);
    if (conv_shortcut_) {
      weight::load_weight(state_dict,
                          "conv_shortcut.weight",
                          conv_shortcut_->weight,
                          is_shortcut_weight_loaded_);
      weight::load_weight(state_dict,
                          "conv_shortcut.bias",
                          conv_shortcut_->bias,
                          is_shortcut_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_norm1_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm1.weight";
    CHECK(is_norm1_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm1.bias";
    CHECK(is_conv1_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv1.weight";
    CHECK(is_conv1_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv1.bias";
    CHECK(is_norm2_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm2.weight";
    CHECK(is_norm2_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm2.bias";
    CHECK(is_conv2_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv2.weight";
    CHECK(is_conv2_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv2.bias";
    if (conv_shortcut_) {
      CHECK(is_shortcut_weight_loaded_)
          << "weight is not loaded for " << prefix + "conv_shortcut.weight";
      CHECK(is_shortcut_bias_loaded_)
          << "bias is not loaded for " << prefix + "conv_shortcut.bias";
    }
  }

 private:
  int64_t in_channels_;
  int64_t out_channels_;
  torch::nn::GroupNorm norm1_{nullptr};
  torch::nn::Conv2d conv1_{nullptr};
  torch::nn::GroupNorm norm2_{nullptr};
  torch::nn::Conv2d conv2_{nullptr};
  torch::nn::Functional nonlinearity_{nullptr};
  torch::nn::Conv2d conv_shortcut_{nullptr};
  torch::nn::Identity identity_{nullptr};

  bool is_norm1_weight_loaded_ = false;
  bool is_norm1_bias_loaded_ = false;
  bool is_conv1_weight_loaded_ = false;
  bool is_conv1_bias_loaded_ = false;
  bool is_norm2_weight_loaded_ = false;
  bool is_norm2_bias_loaded_ = false;
  bool is_conv2_weight_loaded_ = false;
  bool is_conv2_bias_loaded_ = false;
  bool is_shortcut_weight_loaded_ = false;
  bool is_shortcut_bias_loaded_ = false;
};
TORCH_MODULE(Flux2VaeResnetBlock2D);

class Flux2VaeDownsample2DImpl final : public torch::nn::Module,
                                       public dit::VaeParallelMixin {
 public:
  Flux2VaeDownsample2DImpl(const ModelContext& context,
                           int64_t channels,
                           int64_t padding = 1)
      : dit::VaeParallelMixin(context), padding_(padding) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels,
                                                   channels,
                                                   /*kernel_size=*/3)
                              .stride(2)
                              .padding(padding)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto hs = vae_parallel_merge(hidden_states);
    if (padding_ == 0) {
      hs = torch::nn::functional::pad(
          hs,
          torch::nn::functional::PadFuncOptions({0, 1, 0, 1})
              .mode(torch::kConstant)
              .value(0.0f));
    }
    hs = conv_(hs);
    return vae_parallel_split(hs);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "conv.weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "conv.bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv.weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "conv.bias";
  }

 private:
  int64_t padding_ = 0;
  torch::nn::Conv2d conv_{nullptr};
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(Flux2VaeDownsample2D);

class Flux2VaeUpsample2DImpl final : public torch::nn::Module,
                                     public dit::VaeParallelMixin {
 public:
  Flux2VaeUpsample2DImpl(const ModelContext& context, int64_t channels)
      : dit::VaeParallelMixin(context) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels,
                                                   channels,
                                                   /*kernel_size=*/3)
                              .padding(0)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto x = vae_parallel_merge(hidden_states);

    torch::nn::functional::InterpolateFuncOptions opts;
    opts.mode(torch::kNearest);
    opts.scale_factor(std::vector<double>{2.0, 2.0});
    x = torch::nn::functional::interpolate(x, opts);

    // conv (3×3, padding=0): manual pad handles spatial padding.
    if (vae_parallel_enabled()) {
      x = vae_parallel_exchange(x, /*pad=*/true);
      x = torch::nn::functional::pad(
          x, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
    } else {
      x = torch::nn::functional::pad(
          x, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}));
    }
    x = conv_(x);

    return vae_parallel_split(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "conv.weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "conv.bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv.weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "conv.bias";
  }

 private:
  torch::nn::Conv2d conv_{nullptr};
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(Flux2VaeUpsample2D);

class Flux2VaeAttentionImpl final : public torch::nn::Module,
                                    public dit::VaeParallelMixin {
 public:
  explicit Flux2VaeAttentionImpl(const ModelContext& context)
      : dit::VaeParallelMixin(context) {
    int64_t query_dim = context.get_model_args().block_out_channels().back();
    int64_t head_dim = query_dim;
    num_heads_ = query_dim / head_dim;
    int64_t norm_num_groups = context.get_model_args().norm_num_groups();
    auto options = context.get_tensor_options();

    if (norm_num_groups > 0) {
      group_norm_ =
          register_module("group_norm",
                          torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                   norm_num_groups, query_dim)
                                                   .eps(1e-6f)));
    }

    to_q_ = register_module(
        "to_q",
        layer::AddMatmul(query_dim, query_dim, /*with_bias=*/true, options));
    to_k_ = register_module(
        "to_k",
        layer::AddMatmul(query_dim, query_dim, /*with_bias=*/true, options));
    to_v_ = register_module(
        "to_v",
        layer::AddMatmul(query_dim, query_dim, /*with_bias=*/true, options));
    to_out_ = register_module(
        "to_out",
        layer::AddMatmul(query_dim, query_dim, /*with_bias=*/true, options));
  }

  torch::Tensor forward(const torch::Tensor& input_tensor) {
    auto residual = input_tensor;
    auto b = input_tensor.size(0);
    auto c = input_tensor.size(1);
    auto h = input_tensor.size(2);
    auto w = input_tensor.size(3);

    // Local path: GroupNorm → to_q (local W).
    auto local_normed = group_norm_->forward(input_tensor.view({b, c, h * w}))
                            .view({b, c, h, w});
    local_normed = local_normed.view({b, c, h * w}).transpose(1, 2);
    auto query = to_q_(local_normed);

    // Merged path: GroupNorm → to_k / to_v (global W).
    auto merged = vae_parallel_merge(input_tensor);
    auto w_global = merged.size(3);
    auto merged_normed = group_norm_->forward(merged.view({b, c, h * w_global}))
                             .view({b, c, h, w_global});
    merged_normed = merged_normed.view({b, c, h * w_global}).transpose(1, 2);
    auto key = to_k_(merged_normed);
    auto value = to_v_(merged_normed);

    // Reshape for multi-head attention.
    int64_t inner_dim = key.size(-1);
    int64_t head_dim = inner_dim / num_heads_;
    query = query.view({b, -1, num_heads_, head_dim}).transpose(1, 2);
    key = key.view({b, -1, num_heads_, head_dim}).transpose(1, 2);
    value = value.view({b, -1, num_heads_, head_dim}).transpose(1, 2);

    auto attn = torch::scaled_dot_product_attention(
        query, key, value, std::nullopt, 0.0, false);

    attn = attn.transpose(1, 2).reshape({b, -1, num_heads_ * head_dim});
    attn = attn.to(query.dtype());
    attn = to_out_(attn);

    // Reshape back to spatial and add residual.
    auto output = attn.transpose(-1, -2).reshape({b, c, h, w});
    return output + residual;
  }

  void load_state_dict(const StateDict& state_dict) {
    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));
    if (group_norm_) {
      weight::load_weight(state_dict,
                          "group_norm.weight",
                          group_norm_->weight,
                          is_gn_weight_loaded_);
      weight::load_weight(
          state_dict, "group_norm.bias", group_norm_->bias, is_gn_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    to_q_->verify_loaded_weights(prefix + "to_q.");
    to_k_->verify_loaded_weights(prefix + "to_k.");
    to_v_->verify_loaded_weights(prefix + "to_v.");
    to_out_->verify_loaded_weights(prefix + "to_out.0.");
    if (group_norm_) {
      CHECK(is_gn_weight_loaded_)
          << "weight is not loaded for " << prefix + "group_norm.weight";
      CHECK(is_gn_bias_loaded_)
          << "bias is not loaded for " << prefix + "group_norm.bias";
    }
  }

 private:
  torch::nn::GroupNorm group_norm_{nullptr};
  layer::AddMatmul to_q_{nullptr};
  layer::AddMatmul to_k_{nullptr};
  layer::AddMatmul to_v_{nullptr};
  layer::AddMatmul to_out_{nullptr};
  int64_t num_heads_ = 1;
  bool is_gn_weight_loaded_ = false;
  bool is_gn_bias_loaded_ = false;
};
TORCH_MODULE(Flux2VaeAttention);

class Flux2VaeUNetMidBlock2DImpl final : public torch::nn::Module {
 public:
  explicit Flux2VaeUNetMidBlock2DImpl(const ModelContext& context) {
    int64_t in_channels = context.get_model_args().block_out_channels().back();
    add_attention_ = context.get_model_args().mid_block_add_attention();

    resnets_ = register_module("resnets", torch::nn::ModuleList());
    attentions_ = register_module("attentions", torch::nn::ModuleList());
    resnet_layers_.reserve(2);
    attention_layers_.reserve(add_attention_ ? 1 : 0);

    auto block1 = Flux2VaeResnetBlock2D(context, in_channels, in_channels);
    resnets_->push_back(block1);
    resnet_layers_.push_back(block1);

    if (add_attention_) {
      auto attn = Flux2VaeAttention(context);
      attentions_->push_back(attn);
      attention_layers_.push_back(attn);
    } else {
      attentions_->push_back(torch::nn::Sequential());
    }

    auto block2 = Flux2VaeResnetBlock2D(context, in_channels, in_channels);
    resnets_->push_back(block2);
    resnet_layers_.push_back(block2);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto h = resnet_layers_[0]->forward(hidden_states);
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
      if (add_attention_) {
        h = attention_layers_[i]->forward(h);
      }
      h = resnet_layers_[i + 1]->forward(h);
    }
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "resnets." + std::to_string(i) + "."));
    }
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
      if (add_attention_) {
        attention_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
            "attentions." + std::to_string(i) + "."));
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->verify_loaded_weights(prefix + "resnets." +
                                               std::to_string(i) + ".");
    }
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
      if (add_attention_) {
        attention_layers_[i]->verify_loaded_weights(prefix + "attentions." +
                                                    std::to_string(i) + ".");
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  std::vector<Flux2VaeResnetBlock2D> resnet_layers_;
  torch::nn::ModuleList attentions_{nullptr};
  std::vector<Flux2VaeAttention> attention_layers_;
  bool add_attention_ = false;
};
TORCH_MODULE(Flux2VaeUNetMidBlock2D);

class Flux2VaeDownEncoderBlock2DImpl final : public torch::nn::Module {
 public:
  Flux2VaeDownEncoderBlock2DImpl(const ModelContext& context,
                                 int64_t in_channels,
                                 int64_t out_channels,
                                 bool add_downsample = true) {
    int64_t num_layers = context.get_model_args().layers_per_block();
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    downsamplers_ = register_module("downsamplers", torch::nn::ModuleList());
    resnet_layers_.reserve(num_layers);
    downsample_layers_.reserve(add_downsample ? 1 : 0);

    for (int64_t i = 0; i < num_layers; ++i) {
      int64_t cur_in = (i == 0) ? in_channels : out_channels;
      auto block = Flux2VaeResnetBlock2D(context, cur_in, out_channels);
      resnets_->push_back(block);
      resnet_layers_.push_back(block);
    }

    if (add_downsample) {
      auto ds = Flux2VaeDownsample2D(context, out_channels, /*padding=*/0);
      downsamplers_->push_back(ds);
      downsample_layers_.push_back(ds);
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto h = hidden_states;
    for (auto& resnet : resnet_layers_) {
      h = resnet->forward(h);
    }
    if (downsamplers_) {
      for (auto& ds : downsample_layers_) {
        h = ds->forward(h);
      }
    }
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "resnets." + std::to_string(i) + "."));
    }
    for (size_t i = 0; i < downsample_layers_.size(); ++i) {
      downsample_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "downsamplers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->verify_loaded_weights(prefix + "resnets." +
                                               std::to_string(i) + ".");
    }
    for (size_t i = 0; i < downsample_layers_.size(); ++i) {
      downsample_layers_[i]->verify_loaded_weights(prefix + "downsamplers." +
                                                   std::to_string(i) + ".");
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  std::vector<Flux2VaeResnetBlock2D> resnet_layers_;
  torch::nn::ModuleList downsamplers_{nullptr};
  std::vector<Flux2VaeDownsample2D> downsample_layers_;
};
TORCH_MODULE(Flux2VaeDownEncoderBlock2D);

class Flux2VaeUpDecoderBlock2DImpl final : public torch::nn::Module {
 public:
  Flux2VaeUpDecoderBlock2DImpl(const ModelContext& context,
                               int64_t in_channels,
                               int64_t out_channels,
                               bool add_upsample = true) {
    int64_t num_layers = context.get_model_args().layers_per_block() + 1;
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    resnet_layers_.reserve(num_layers);

    for (int64_t i = 0; i < num_layers; ++i) {
      int64_t cur_in = (i == 0) ? in_channels : out_channels;
      auto block = Flux2VaeResnetBlock2D(context, cur_in, out_channels);
      resnets_->push_back(block);
      resnet_layers_.push_back(block);
    }

    if (add_upsample) {
      add_upsample_ = true;
      upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
      upsample_layers_.reserve(1);
      auto us = Flux2VaeUpsample2D(context, out_channels);
      upsamplers_->push_back(us);
      upsample_layers_.push_back(us);
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto h = hidden_states;
    for (auto& resnet : resnet_layers_) {
      h = resnet->forward(h);
    }
    if (add_upsample_) {
      for (auto& us : upsample_layers_) {
        h = us->forward(h);
      }
    }
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "resnets." + std::to_string(i) + "."));
    }
    if (add_upsample_) {
      for (size_t i = 0; i < upsample_layers_.size(); ++i) {
        upsample_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
            "upsamplers." + std::to_string(i) + "."));
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->verify_loaded_weights(prefix + "resnets." +
                                               std::to_string(i) + ".");
    }
    if (add_upsample_) {
      for (size_t i = 0; i < upsample_layers_.size(); ++i) {
        upsample_layers_[i]->verify_loaded_weights(prefix + "upsamplers." +
                                                   std::to_string(i) + ".");
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  std::vector<Flux2VaeResnetBlock2D> resnet_layers_;
  torch::nn::ModuleList upsamplers_{nullptr};
  std::vector<Flux2VaeUpsample2D> upsample_layers_;
  bool add_upsample_ = false;
};
TORCH_MODULE(Flux2VaeUpDecoderBlock2D);

class Flux2VaeEncoderImpl final : public torch::nn::Module,
                                  public dit::VaeParallelMixin {
 public:
  explicit Flux2VaeEncoderImpl(const ModelContext& context)
      : dit::VaeParallelMixin(context) {
    auto args = context.get_model_args();
    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());

    // conv_in: 3×3, padding=0 (manual exchange+pad in forward).
    conv_in_ = register_module(
        "conv_in",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(args.in_channels(),
                                                   args.block_out_channels()[0],
                                                   /*kernel_size=*/3)
                              .stride(1)
                              .padding(0)
                              .bias(true)));

    // Down blocks.
    int32_t output_channels = args.block_out_channels()[0];
    down_layers_.reserve(args.down_block_types().size());
    for (size_t i = 0; i < args.down_block_types().size(); i++) {
      int32_t input_channels = output_channels;
      output_channels = args.block_out_channels()[i];
      bool is_final = (i == args.block_out_channels().size() - 1);
      auto down_block = Flux2VaeDownEncoderBlock2D(
          context, input_channels, output_channels, !is_final);
      down_blocks_->push_back(down_block);
      down_layers_.push_back(down_block);
    }

    // Mid block.
    mid_block_ = register_module("mid_block", Flux2VaeUNetMidBlock2D(context));

    // Output layers.
    conv_norm_out_ = register_module(
        "conv_norm_out",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(args.norm_num_groups(),
                                        args.block_out_channels().back())
                .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));

    // conv_out: 3×3, padding=0 (manual exchange+pad in forward).
    conv_out_ =
        register_module("conv_out",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                              args.block_out_channels().back(),
                                              2 * args.latent_channels(),
                                              /*kernel_size=*/3)
                                              .padding(0)
                                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& images) {
    // conv_in (3×3, padding=0): manual pad handles spatial padding.
    auto sample = images;
    if (vae_parallel_enabled()) {
      sample = vae_parallel_exchange(sample, /*pad=*/true);
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
    } else {
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}));
    }
    sample = conv_in_(sample);

    // Down blocks (sub-modules handle their own parallel ops).
    for (auto& layer : down_layers_) {
      sample = layer->forward(sample);
    }

    // Mid block.
    sample = mid_block_->forward(sample);

    // GroupNorm + SiLU (pointwise, no exchange needed).
    sample = conv_norm_out_(sample);
    sample = conv_act_(sample);

    // conv_out (3×3, padding=0): manual pad handles spatial padding.
    if (vae_parallel_enabled()) {
      sample = vae_parallel_exchange(sample, /*pad=*/true);
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
    } else {
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}));
    }
    sample = conv_out_(sample);

    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "conv_in.weight",
                        conv_in_->weight,
                        is_conv_in_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_in.bias", conv_in_->bias, is_conv_in_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv_norm_out.weight",
                        conv_norm_out_->weight,
                        is_conv_norm_out_weight_loaded_);
    weight::load_weight(state_dict,
                        "conv_norm_out.bias",
                        conv_norm_out_->bias,
                        is_conv_norm_out_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv_out.weight",
                        conv_out_->weight,
                        is_conv_out_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_out.bias", conv_out_->bias, is_conv_out_bias_loaded_);
    for (size_t i = 0; i < down_layers_.size(); ++i) {
      down_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "down_blocks." + std::to_string(i) + "."));
    }
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_conv_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_in.weight";
    CHECK(is_conv_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_in.bias";
    CHECK(is_conv_norm_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_norm_out.weight";
    CHECK(is_conv_norm_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_norm_out.bias";
    CHECK(is_conv_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_out.weight";
    CHECK(is_conv_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_out.bias";
    for (size_t i = 0; i < down_layers_.size(); ++i) {
      down_layers_[i]->verify_loaded_weights(prefix + "down_blocks." +
                                             std::to_string(i) + ".");
    }
    mid_block_->verify_loaded_weights(prefix + "mid_block.");
  }

 private:
  torch::nn::Conv2d conv_in_{nullptr};
  torch::nn::Conv2d conv_out_{nullptr};
  torch::nn::Functional conv_act_{nullptr};
  torch::nn::GroupNorm conv_norm_out_{nullptr};
  torch::nn::ModuleList down_blocks_{nullptr};
  std::vector<Flux2VaeDownEncoderBlock2D> down_layers_;
  Flux2VaeUNetMidBlock2D mid_block_{nullptr};

  bool is_conv_in_weight_loaded_ = false;
  bool is_conv_in_bias_loaded_ = false;
  bool is_conv_out_weight_loaded_ = false;
  bool is_conv_out_bias_loaded_ = false;
  bool is_conv_norm_out_weight_loaded_ = false;
  bool is_conv_norm_out_bias_loaded_ = false;
};
TORCH_MODULE(Flux2VaeEncoder);

class Flux2VaeDecoderImpl final : public torch::nn::Module,
                                  public dit::VaeParallelMixin {
 public:
  explicit Flux2VaeDecoderImpl(const ModelContext& context)
      : dit::VaeParallelMixin(context) {
    auto args = context.get_model_args();
    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());

    // conv_in: 3×3, padding=0.
    conv_in_ =
        register_module("conv_in",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                              args.latent_channels(),
                                              args.block_out_channels().back(),
                                              /*kernel_size=*/3)
                                              .stride(1)
                                              .padding(0)
                                              .bias(true)));

    // Mid block.
    mid_block_ = register_module("mid_block", Flux2VaeUNetMidBlock2D(context));

    // Up blocks.
    std::vector<int64_t> reversed_channels(args.block_out_channels().rbegin(),
                                           args.block_out_channels().rend());
    int64_t output_channel = reversed_channels[0];

    for (size_t i = 0; i < args.up_block_types().size(); ++i) {
      int64_t prev_output = output_channel;
      output_channel = reversed_channels[i];
      bool is_final = (i == args.block_out_channels().size() - 1);
      auto up_block = Flux2VaeUpDecoderBlock2D(
          context, prev_output, output_channel, !is_final);
      up_blocks_->push_back(up_block);
      up_layers_.push_back(up_block);
    }

    // Output layers.
    conv_norm_out_ =
        register_module("conv_norm_out",
                        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                 args.norm_num_groups(),
                                                 args.block_out_channels()[0])
                                                 .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));

    // conv_out: 3×3, padding=0.
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(args.block_out_channels()[0],
                                                   args.out_channels(),
                                                   /*kernel_size=*/3)
                              .padding(0)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& latents) {
    // conv_in (3×3, padding=0): manual pad handles spatial padding.
    auto sample = latents;
    if (vae_parallel_enabled()) {
      sample = vae_parallel_exchange(sample, /*pad=*/true);
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
    } else {
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}));
    }
    sample = conv_in_(sample);

    // Mid block.
    sample = mid_block_->forward(sample);

    // Up blocks.
    for (auto& layer : up_layers_) {
      sample = layer->forward(sample);
    }

    // GroupNorm + SiLU.
    sample = conv_norm_out_(sample);
    sample = conv_act_(sample);

    // conv_out (3×3, padding=0): manual pad handles spatial padding.
    if (vae_parallel_enabled()) {
      sample = vae_parallel_exchange(sample, /*pad=*/true);
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
    } else {
      sample = torch::nn::functional::pad(
          sample, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}));
    }
    sample = conv_out_(sample);

    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "conv_in.weight",
                        conv_in_->weight,
                        is_conv_in_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_in.bias", conv_in_->bias, is_conv_in_bias_loaded_);
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    for (size_t i = 0; i < up_layers_.size(); ++i) {
      up_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "up_blocks." + std::to_string(i) + "."));
    }
    weight::load_weight(state_dict,
                        "conv_norm_out.weight",
                        conv_norm_out_->weight,
                        is_conv_norm_out_weight_loaded_);
    weight::load_weight(state_dict,
                        "conv_norm_out.bias",
                        conv_norm_out_->bias,
                        is_conv_norm_out_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv_out.weight",
                        conv_out_->weight,
                        is_conv_out_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_out.bias", conv_out_->bias, is_conv_out_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_conv_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_in.weight";
    CHECK(is_conv_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_in.bias";
    mid_block_->verify_loaded_weights(prefix + "mid_block.");
    for (size_t i = 0; i < up_layers_.size(); ++i) {
      up_layers_[i]->verify_loaded_weights(prefix + "up_blocks." +
                                           std::to_string(i) + ".");
    }
    CHECK(is_conv_norm_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_norm_out.weight";
    CHECK(is_conv_norm_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_norm_out.bias";
    CHECK(is_conv_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_out.weight";
    CHECK(is_conv_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_out.bias";
  }

 private:
  torch::nn::Conv2d conv_in_{nullptr};
  Flux2VaeUNetMidBlock2D mid_block_{nullptr};
  torch::nn::ModuleList up_blocks_{nullptr};
  std::vector<Flux2VaeUpDecoderBlock2D> up_layers_;
  torch::nn::GroupNorm conv_norm_out_{nullptr};
  torch::nn::Functional conv_act_{nullptr};
  torch::nn::Conv2d conv_out_{nullptr};

  bool is_conv_in_weight_loaded_ = false;
  bool is_conv_in_bias_loaded_ = false;
  bool is_conv_norm_out_weight_loaded_ = false;
  bool is_conv_norm_out_bias_loaded_ = false;
  bool is_conv_out_weight_loaded_ = false;
  bool is_conv_out_bias_loaded_ = false;
};
TORCH_MODULE(Flux2VaeDecoder);

class AutoencoderKLFlux2Impl final : public torch::nn::Module,
                                     public dit::VaeParallelMixin {
 public:
  explicit AutoencoderKLFlux2Impl(const ModelContext& context)
      : dit::VaeParallelMixin(context), args_(context.get_model_args()) {
    encoder_ = register_module("encoder", Flux2VaeEncoder(context));
    decoder_ = register_module("decoder", Flux2VaeDecoder(context));

    if (args_.use_quant_conv()) {
      quant_conv_ = register_module(
          "quant_conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(
              2 * args_.latent_channels(), 2 * args_.latent_channels(), 1)));
    }
    if (args_.use_post_quant_conv()) {
      post_quant_conv_ = register_module(
          "post_quant_conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(
              args_.latent_channels(), args_.latent_channels(), 1)));
    }

    auto dtype = context.get_tensor_options().dtype().toScalarType();
    encoder_->to(dtype);
    decoder_->to(dtype);
    if (args_.use_quant_conv()) {
      quant_conv_->to(dtype);
    }
    if (args_.use_post_quant_conv()) {
      post_quant_conv_->to(dtype);
    }

    int64_t patch_size_prod = 1;
    for (int64_t ps : args_.ae_patch_size()) {
      patch_size_prod *= ps;
    }
    int64_t bn_num_features = patch_size_prod * args_.latent_channels();

    bn_ = register_module(
        "bn",
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(bn_num_features)
                                   .eps(args_.batch_norm_eps())
                                   .momentum(args_.batch_norm_momentum())
                                   .affine(false)
                                   .track_running_stats(true)));
    bn_->to(dtype);
  }

  torch::Tensor encode(const torch::Tensor& images, int64_t seed) {
    auto x = vae_parallel_split(images);
    auto enc = encoder_->forward(x);
    if (args_.use_quant_conv()) {
      enc = quant_conv_(enc);
    }
    enc = vae_parallel_merge(enc);
    auto posterior = DiagonalGaussianDistribution(enc);
    return posterior.sample(seed);
  }

  torch::Tensor decode(const torch::Tensor& latents) {
    auto z = vae_parallel_split(latents);
    if (args_.use_post_quant_conv()) {
      z = post_quant_conv_(z);
    }
    auto dec = decoder_->forward(z);
    dec = vae_parallel_merge(dec);
    return dec;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
      if (args_.use_quant_conv()) {
        weight::load_weight(state_dict->get_dict_with_prefix("quant_conv."),
                            "weight",
                            quant_conv_->weight,
                            is_quant_conv_weight_);
        weight::load_weight(state_dict->get_dict_with_prefix("quant_conv."),
                            "bias",
                            quant_conv_->bias,
                            is_quant_conv_bias_);
      }
      if (args_.use_post_quant_conv()) {
        weight::load_weight(
            state_dict->get_dict_with_prefix("post_quant_conv."),
            "weight",
            post_quant_conv_->weight,
            is_post_quant_conv_weight_);
        weight::load_weight(
            state_dict->get_dict_with_prefix("post_quant_conv."),
            "bias",
            post_quant_conv_->bias,
            is_post_quant_conv_bias_);
      }

      weight::load_weight(state_dict->get_dict_with_prefix("bn."),
                          "running_mean",
                          bn_->running_mean,
                          is_bn_running_mean_);
      weight::load_weight(state_dict->get_dict_with_prefix("bn."),
                          "running_var",
                          bn_->running_var,
                          is_bn_running_var_);
    }
    verify_loaded_weights("");
    LOG(INFO) << "VAE model loaded successfully.";
  }

  void verify_loaded_weights(const std::string& prefix) {
    encoder_->verify_loaded_weights(prefix + "encoder.");
    decoder_->verify_loaded_weights(prefix + "decoder.");
    if (args_.use_quant_conv()) {
      CHECK(is_quant_conv_weight_)
          << "weight is not loaded for " << prefix + "quant_conv.weight";
      CHECK(is_quant_conv_bias_)
          << "bias is not loaded for " << prefix + "quant_conv.bias";
    }
    if (args_.use_post_quant_conv()) {
      CHECK(is_post_quant_conv_weight_)
          << "weight is not loaded for " << prefix + "post_quant_conv.weight";
      CHECK(is_post_quant_conv_bias_)
          << "bias is not loaded for " << prefix + "post_quant_conv.bias";
    }
    CHECK(is_bn_running_mean_)
        << "running_mean is not loaded for " << prefix + "bn.running_mean";
    CHECK(is_bn_running_var_)
        << "running_var is not loaded for " << prefix + "bn.running_var";
  }

  torch::Tensor get_bn_running_mean() const { return bn_->running_mean; }
  torch::Tensor get_bn_running_var() const { return bn_->running_var; }
  float get_batch_norm_eps() const { return args_.batch_norm_eps(); }

 private:
  Flux2VaeEncoder encoder_{nullptr};
  Flux2VaeDecoder decoder_{nullptr};
  torch::nn::Conv2d quant_conv_{nullptr};
  torch::nn::Conv2d post_quant_conv_{nullptr};
  torch::nn::BatchNorm2d bn_{nullptr};

  bool is_quant_conv_weight_ = false;
  bool is_quant_conv_bias_ = false;
  bool is_post_quant_conv_weight_ = false;
  bool is_post_quant_conv_bias_ = false;
  bool is_bn_running_mean_ = false;
  bool is_bn_running_var_ = false;
  ModelArgs args_;
};
TORCH_MODULE(AutoencoderKLFlux2);

// register VAE model with the model registry
REGISTER_MODEL_ARGS(AutoencoderKLFlux2, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(in_channels, "in_channels", 3);
  LOAD_ARG_OR(out_channels, "out_channels", 3);
  LOAD_ARG_OR(down_block_types,
              "down_block_types",
              (std::vector<std::string>{"DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D"}));
  LOAD_ARG_OR(up_block_types,
              "up_block_types",
              (std::vector<std::string>{"UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D"}));
  LOAD_ARG_OR(block_out_channels,
              "block_out_channels",
              (std::vector<int64_t>{128, 256, 512, 512}));
  LOAD_ARG_OR(layers_per_block, "layers_per_block", 2);
  LOAD_ARG_OR(latent_channels, "latent_channels", 32);
  LOAD_ARG_OR(norm_num_groups, "norm_num_groups", 32);
  LOAD_ARG_OR(sample_size, "sample_size", 1024);
  LOAD_ARG_OR(mid_block_add_attention, "mid_block_add_attention", true);
  LOAD_ARG_OR(force_upcast, "force_upcast", true);
  LOAD_ARG_OR(use_quant_conv, "use_quant_conv", false);
  LOAD_ARG_OR(use_post_quant_conv, "use_post_quant_conv", false);
  LOAD_ARG_OR(batch_norm_eps, "batch_norm_eps", 1e-04f);
  LOAD_ARG_OR(act_fn, "act_fn", "silu");
  LOAD_ARG_OR(batch_norm_momentum, "batch_norm_momentum", 0.1f);
  LOAD_ARG_OR(ae_patch_size, "patch_size", (std::vector<int64_t>{2, 2}));
});
}  // namespace xllm
