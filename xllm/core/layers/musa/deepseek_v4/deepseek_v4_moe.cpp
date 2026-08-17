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

#include "core/layers/musa/deepseek_v4/deepseek_v4_moe.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {
namespace {

bool has_w_style_shared_expert_weights(const StateDict& state_dict) {
  static const std::vector<std::string> kSharedExpertWeightPrefixes = {
      "w1.", "w2.", "w3."};
  for (const auto& item : state_dict) {
    const std::string& name = item.first;
    for (const std::string& prefix : kSharedExpertWeightPrefixes) {
      if (name.rfind(prefix, 0) == 0) {
        return true;
      }
    }
  }
  return false;
}

torch::Tensor swiglu(const torch::Tensor& gate,
                     const torch::Tensor& up,
                     double swiglu_limit) {
  auto gated = gate;
  auto up_clamped = up;
  if (std::isfinite(swiglu_limit) && swiglu_limit > 0.0 &&
      swiglu_limit < 1000000.0) {
    gated = torch::clamp_max(gated, swiglu_limit);
    up_clamped = torch::clamp(up_clamped, -swiglu_limit, swiglu_limit);
  }
  return torch::silu(gated) * up_clamped;
}

bool all_tensors_defined(const std::vector<torch::Tensor>& tensors) {
  return std::all_of(tensors.begin(), tensors.end(), [](const auto& tensor) {
    return tensor.defined();
  });
}

bool is_packed_fp4_weight(const torch::Tensor& tensor,
                          int64_t logical_out_features,
                          int64_t logical_in_features) {
  return tensor.defined() && tensor.dim() == 2 &&
         tensor.scalar_type() == torch::kInt8 &&
         tensor.size(0) == logical_out_features &&
         tensor.size(1) * 2 == logical_in_features;
}

bool is_fp4_scale(const torch::Tensor& tensor,
                  int64_t logical_out_features,
                  int64_t logical_in_features) {
  return tensor.defined() && tensor.dim() == 2 &&
         tensor.size(0) == logical_out_features &&
         tensor.size(1) * 32 == logical_in_features;
}

bool is_fp8_weight(const torch::Tensor& tensor,
                   int64_t logical_out_features,
                   int64_t logical_in_features) {
  return tensor.defined() && tensor.dim() == 2 &&
         (tensor.scalar_type() == torch::kFloat8_e4m3fn ||
          tensor.scalar_type() == torch::kFloat8_e5m2) &&
         tensor.size(0) == logical_out_features &&
         tensor.size(1) == logical_in_features;
}

bool is_fp8_block_scale(const torch::Tensor& tensor,
                        int64_t logical_out_features,
                        int64_t logical_in_features) {
  const auto ceil_div = [](int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
  };
  return tensor.defined() && tensor.dim() == 2 &&
         (tensor.scalar_type() == torch::kFloat8_e8m0fnu ||
          tensor.scalar_type() == torch::kFloat32) &&
         tensor.size(0) == ceil_div(logical_out_features, 128) &&
         tensor.size(1) == ceil_div(logical_in_features, 128);
}

torch::Tensor dequantize_fp8_block_weight(const torch::Tensor& weight,
                                          const torch::Tensor& scale,
                                          torch::ScalarType output_dtype) {
  CHECK(is_fp8_weight(weight, weight.size(0), weight.size(1)))
      << "DeepseekV4 FP8 expert weight must be a 2D FP8 tensor.";
  CHECK(is_fp8_block_scale(scale, weight.size(0), weight.size(1)))
      << "DeepseekV4 FP8 expert scale must be a 128x128 block scale tensor.";
  const int64_t block_rows =
      (weight.size(0) + scale.size(0) - 1) / scale.size(0);
  const int64_t block_cols =
      (weight.size(1) + scale.size(1) - 1) / scale.size(1);
  auto expanded_scale =
      scale.to(torch::kFloat32)
          .unsqueeze(1)
          .unsqueeze(3)
          .repeat({1, block_rows, 1, block_cols})
          .reshape({scale.size(0) * block_rows, scale.size(1) * block_cols})
          .slice(/*dim=*/0, /*start=*/0, /*end=*/weight.size(0))
          .slice(/*dim=*/1, /*start=*/0, /*end=*/weight.size(1));
  return (weight.to(torch::kFloat32) * expanded_scale)
      .to(output_dtype)
      .contiguous();
}

torch::Tensor unpack_fp4_weight(const torch::Tensor& packed,
                                const torch::Tensor& scale,
                                torch::ScalarType dtype) {
  CHECK(is_packed_fp4_weight(packed, packed.size(0), packed.size(1) * 2))
      << "DeepseekV4 FP4 expert weight expects packed int8 [out, in/2].";
  CHECK(is_fp4_scale(scale, packed.size(0), packed.size(1) * 2))
      << "DeepseekV4 FP4 expert scale expects [out, in/32].";

  auto byte_values = packed.to(torch::kInt16);
  byte_values = torch::where(byte_values.lt(0), byte_values + 256, byte_values);
  auto byte_values_fp32 = byte_values.to(torch::kFloat32);
  auto high = torch::floor(byte_values_fp32 / 16);
  auto low = byte_values_fp32 - high * 16;

  auto table = torch::tensor(
      {0.0f,
       0.5f,
       1.0f,
       1.5f,
       2.0f,
       3.0f,
       4.0f,
       6.0f,
       0.0f,
       -0.5f,
       -1.0f,
       -1.5f,
       -2.0f,
       -3.0f,
       -4.0f,
       -6.0f},
      torch::TensorOptions().dtype(torch::kFloat32).device(packed.device()));
  auto low_values =
      table.index_select(/*dim=*/0, low.to(torch::kLong).view({-1}))
          .view(low.sizes());
  auto high_values =
      table.index_select(/*dim=*/0, high.to(torch::kLong).view({-1}))
          .view(high.sizes());
  auto values = torch::stack({low_values, high_values}, /*dim=*/-1)
                    .flatten(/*start_dim=*/1);
  auto expanded_scale =
      scale.to(torch::kFloat32).unsqueeze(-1).repeat({1, 1, 32}).flatten(1);
  return (values * expanded_scale).to(dtype).contiguous();
}

}  // namespace

DeepseekV4MoEImpl::DeepseekV4MoEImpl(const ModelArgs& model_args,
                                     const QuantArgs& quant_args,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options)
    : num_total_experts_(model_args.n_routed_experts()),
      topk_(model_args.n_activated_experts() > 0
                ? model_args.n_activated_experts()
                : model_args.num_experts_per_tok()),
      hidden_size_(model_args.hidden_size()),
      intermediate_size_(model_args.moe_intermediate_size() > 0
                             ? model_args.moe_intermediate_size()
                             : model_args.intermediate_size()),
      n_shared_experts_(model_args.n_shared_experts()),
      swiglu_limit_(static_cast<double>(model_args.swiglu_limit())),
      hidden_act_(model_args.hidden_act()),
      options_(options),
      tp_pg_(parallel_args.moe_tp_group_ != nullptr
                 ? parallel_args.moe_tp_group_
                 : parallel_args.tp_group_),
      ep_pg_(parallel_args.moe_ep_group_) {
  CHECK(tp_pg_ != nullptr) << "DeepseekV4MoE requires a valid TP group.";
  CHECK_GT(num_total_experts_, 0)
      << "DeepseekV4MoE requires n_routed_experts > 0.";
  CHECK_GT(topk_, 0) << "DeepseekV4MoE requires topk > 0.";
  CHECK_GT(hidden_size_, 0) << "DeepseekV4MoE requires hidden_size > 0.";
  CHECK_GT(intermediate_size_, 0)
      << "DeepseekV4MoE requires moe_intermediate_size/intermediate_size > 0.";

  const int64_t ep_size = std::max<int64_t>(parallel_args.ep_size(), 1);
  const int64_t ep_rank = ep_size > 1 && parallel_args.moe_ep_group_ != nullptr
                              ? parallel_args.moe_ep_group_->rank()
                              : 0;
  CHECK_EQ(num_total_experts_ % ep_size, 0)
      << "DeepseekV4MoE expects n_routed_experts divisible by ep_size.";
  num_experts_per_rank_ = num_total_experts_ / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  const int64_t world_size = std::max<int64_t>(tp_pg_->world_size(), 1);
  CHECK_EQ(intermediate_size_ % world_size, 0)
      << "DeepseekV4MoE intermediate size must be divisible by TP world size.";
  local_intermediate_size_ = intermediate_size_ / world_size;

  if (quant_args.quant_method() != "" || !quant_args.quant_descs().empty()) {
    LOG(WARNING)
        << "DeepseekV4MoE MUSA fallback supports unquantized routed experts "
           "and a functional packed-FP4 path for DeepSeek-V4-Flash. The FP4 "
           "path unpacks selected experts on demand and is not a production "
           "fused MoE kernel.";
  }

  if (n_shared_experts_ > 0) {
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size_ * n_shared_experts_,
                                 /*is_gated=*/true,
                                 /*has_bias=*/false,
                                 hidden_act_,
                                 /*enable_result_reduction=*/false,
                                 quant_args,
                                 tp_pg_,
                                 options,
                                 /*module_prefix=*/"",
                                 swiglu_limit_));
    shared_expert_gate_ = register_module(
        "shared_expert_gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, 1).bias(false)));
    shared_expert_gate_->weight.set_data(
        shared_expert_gate_->weight.to(options));
  }

  CHECK_EQ(hidden_size_ % 32, 0)
      << "DeepseekV4 FP4 expert path requires hidden_size divisible by 32.";
  CHECK_EQ(local_intermediate_size_ % 32, 0)
      << "DeepseekV4 FP4 expert path requires local intermediate size "
         "divisible by 32.";
}

void DeepseekV4MoEImpl::ensure_dense_expert_storage() {
  if (!w13_.defined()) {
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size_ * 2, hidden_size_},
            options_),
        /*requires_grad=*/false);
  }
  if (!w2_.defined()) {
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size_},
            options_),
        /*requires_grad=*/false);
  }
}

void DeepseekV4MoEImpl::ensure_fp8_expert_storage() {
  const auto weight_options = options_.dtype(torch::kFloat8_e4m3fn);
  const auto scale_options = options_.dtype(torch::kFloat32);
  if (!w13_fp8_.defined()) {
    w13_fp8_ = register_parameter(
        "w13_fp8",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size_ * 2, hidden_size_},
            weight_options),
        /*requires_grad=*/false);
  }
  if (!w13_fp8_scale_.defined()) {
    w13_fp8_scale_ = register_parameter(
        "w13_fp8_scale",
        torch::empty({num_experts_per_rank_,
                      (local_intermediate_size_ * 2 + 127) / 128,
                      (hidden_size_ + 127) / 128},
                     scale_options),
        /*requires_grad=*/false);
  }
  if (!w2_fp8_.defined()) {
    w2_fp8_ = register_parameter(
        "w2_fp8",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size_},
            weight_options),
        /*requires_grad=*/false);
  }
  if (!w2_fp8_scale_.defined()) {
    w2_fp8_scale_ = register_parameter(
        "w2_fp8_scale",
        torch::empty({num_experts_per_rank_,
                      (hidden_size_ + 127) / 128,
                      (local_intermediate_size_ + 127) / 128},
                     scale_options),
        /*requires_grad=*/false);
  }
}

void DeepseekV4MoEImpl::ensure_packed_fp4_expert_storage() {
  const auto packed_options = options_.dtype(torch::kInt8);
  const auto scale_options = options_.dtype(torch::kFloat32);
  if (!w13_packed_.defined()) {
    w13_packed_ = register_parameter("w13_packed",
                                     torch::empty({num_experts_per_rank_,
                                                   local_intermediate_size_ * 2,
                                                   hidden_size_ / 2},
                                                  packed_options),
                                     /*requires_grad=*/false);
  }
  if (!w13_scale_.defined()) {
    w13_scale_ = register_parameter("w13_scale",
                                    torch::empty({num_experts_per_rank_,
                                                  local_intermediate_size_ * 2,
                                                  hidden_size_ / 32},
                                                 scale_options),
                                    /*requires_grad=*/false);
  }
  if (!w2_packed_.defined()) {
    w2_packed_ = register_parameter(
        "w2_packed",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size_ / 2},
            packed_options),
        /*requires_grad=*/false);
  }
  if (!w2_scale_.defined()) {
    w2_scale_ = register_parameter("w2_scale",
                                   torch::empty({num_experts_per_rank_,
                                                 hidden_size_,
                                                 local_intermediate_size_ / 32},
                                                scale_options),
                                   /*requires_grad=*/false);
  }
}

bool DeepseekV4MoEImpl::try_load_packed_fp4_experts(
    const StateDict& state_dict,
    const std::vector<std::string>& w13_prefixes,
    const std::string& w2_prefix,
    int64_t rank,
    int64_t world_size) {
  CHECK_EQ(w13_prefixes.size(), 2)
      << "DeepseekV4 packed FP4 fused load expects two prefixes.";

  if (w1_packed_list_.size() < static_cast<size_t>(num_experts_per_rank_)) {
    w1_packed_list_.resize(num_experts_per_rank_);
    w3_packed_list_.resize(num_experts_per_rank_);
    w1_scale_list_.resize(num_experts_per_rank_);
    w3_scale_list_.resize(num_experts_per_rank_);
    w2_packed_list_.resize(num_experts_per_rank_);
    w2_scale_list_.resize(num_experts_per_rank_);
  }

  bool saw_packed_tensor = false;
  for (int64_t idx = 0; idx < num_experts_per_rank_; ++idx) {
    const std::string expert_prefix =
        std::to_string(start_expert_id_ + idx) + ".";
    if (!w1_packed_list_[idx].defined()) {
      auto w1 = state_dict.get_sharded_tensor(
          expert_prefix + w13_prefixes[0] + "weight", 0, rank, world_size);
      if (w1.defined()) {
        if (!is_packed_fp4_weight(w1, local_intermediate_size_, hidden_size_)) {
          return false;
        }
        auto scale = state_dict.get_sharded_tensor(
            expert_prefix + w13_prefixes[0] + "scale", 0, rank, world_size);
        CHECK(is_fp4_scale(scale, local_intermediate_size_, hidden_size_))
            << "DeepseekV4 packed FP4 w1 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w13_prefixes[0];
        w1_packed_list_[idx] = w1.contiguous();
        w1_scale_list_[idx] = scale.to(torch::kFloat32).contiguous();
        packed_fp4_experts_detected_ = true;
        saw_packed_tensor = true;
      }
    }
    if (!w3_packed_list_[idx].defined()) {
      auto w3 = state_dict.get_sharded_tensor(
          expert_prefix + w13_prefixes[1] + "weight", 0, rank, world_size);
      if (w3.defined()) {
        if (!is_packed_fp4_weight(w3, local_intermediate_size_, hidden_size_)) {
          return false;
        }
        auto scale = state_dict.get_sharded_tensor(
            expert_prefix + w13_prefixes[1] + "scale", 0, rank, world_size);
        CHECK(is_fp4_scale(scale, local_intermediate_size_, hidden_size_))
            << "DeepseekV4 packed FP4 w3 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w13_prefixes[1];
        w3_packed_list_[idx] = w3.contiguous();
        w3_scale_list_[idx] = scale.to(torch::kFloat32).contiguous();
        packed_fp4_experts_detected_ = true;
        saw_packed_tensor = true;
      }
    }
    if (!w2_packed_list_[idx].defined()) {
      auto w2 = state_dict.get_sharded_tensor(
          expert_prefix + w2_prefix + "weight", 1, rank, world_size);
      if (w2.defined()) {
        if (!is_packed_fp4_weight(w2, hidden_size_, local_intermediate_size_)) {
          return false;
        }
        auto scale = state_dict.get_sharded_tensor(
            expert_prefix + w2_prefix + "scale", 1, rank, world_size);
        CHECK(is_fp4_scale(scale, hidden_size_, local_intermediate_size_))
            << "DeepseekV4 packed FP4 w2 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w2_prefix;
        w2_packed_list_[idx] = w2.contiguous();
        w2_scale_list_[idx] = scale.to(torch::kFloat32).contiguous();
        packed_fp4_experts_detected_ = true;
        saw_packed_tensor = true;
      }
    }
  }

  if (!w13_packed_is_loaded_ && all_tensors_defined(w1_packed_list_) &&
      all_tensors_defined(w3_packed_list_) &&
      all_tensors_defined(w1_scale_list_) &&
      all_tensors_defined(w3_scale_list_)) {
    ensure_packed_fp4_expert_storage();
    std::vector<torch::Tensor> packed_vec;
    std::vector<torch::Tensor> scale_vec;
    packed_vec.reserve(num_experts_per_rank_);
    scale_vec.reserve(num_experts_per_rank_);
    for (int64_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      packed_vec.push_back(
          torch::cat({w1_packed_list_[idx], w3_packed_list_[idx]}, 0));
      scale_vec.push_back(
          torch::cat({w1_scale_list_[idx], w3_scale_list_[idx]}, 0));
    }
    w13_packed_.copy_(torch::stack(packed_vec).to(w13_packed_.device()));
    w13_scale_.copy_(torch::stack(scale_vec).to(w13_scale_.device()));
    w13_packed_is_loaded_ = true;
    w1_packed_list_.clear();
    w3_packed_list_.clear();
    w1_scale_list_.clear();
    w3_scale_list_.clear();
  }

  if (!w2_packed_is_loaded_ && all_tensors_defined(w2_packed_list_) &&
      all_tensors_defined(w2_scale_list_)) {
    ensure_packed_fp4_expert_storage();
    w2_packed_.copy_(torch::stack(w2_packed_list_).to(w2_packed_.device()));
    w2_scale_.copy_(torch::stack(w2_scale_list_).to(w2_scale_.device()));
    w2_packed_is_loaded_ = true;
    w2_packed_list_.clear();
    w2_scale_list_.clear();
  }

  return saw_packed_tensor || fp4_experts_loaded();
}

bool DeepseekV4MoEImpl::try_load_fp8_experts(
    const StateDict& state_dict,
    const std::vector<std::string>& w13_prefixes,
    const std::string& w2_prefix,
    int64_t rank,
    int64_t world_size) {
  CHECK_EQ(w13_prefixes.size(), 2)
      << "DeepseekV4 FP8 fused load expects two prefixes.";

  if (w1_fp8_list_.size() < static_cast<size_t>(num_experts_per_rank_)) {
    w1_fp8_list_.resize(num_experts_per_rank_);
    w3_fp8_list_.resize(num_experts_per_rank_);
    w1_fp8_scale_list_.resize(num_experts_per_rank_);
    w3_fp8_scale_list_.resize(num_experts_per_rank_);
    w2_fp8_list_.resize(num_experts_per_rank_);
    w2_fp8_scale_list_.resize(num_experts_per_rank_);
  }

  bool saw_fp8_tensor = false;
  for (int64_t idx = 0; idx < num_experts_per_rank_; ++idx) {
    const std::string expert_prefix =
        std::to_string(start_expert_id_ + idx) + ".";
    if (!w1_fp8_list_[idx].defined()) {
      auto w1 = state_dict.get_sharded_tensor(
          expert_prefix + w13_prefixes[0] + "weight", 0, rank, world_size);
      if (w1.defined()) {
        if (!is_fp8_weight(w1, local_intermediate_size_, hidden_size_)) {
          return false;
        }
        auto scale = state_dict.get_sharded_tensor(
            expert_prefix + w13_prefixes[0] + "scale", 0, rank, world_size);
        CHECK(is_fp8_block_scale(scale, local_intermediate_size_, hidden_size_))
            << "DeepseekV4 FP8 w1 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w13_prefixes[0];
        w1_fp8_list_[idx] = w1.contiguous();
        w1_fp8_scale_list_[idx] = scale.to(torch::kFloat32).contiguous();
        fp8_experts_detected_ = true;
        saw_fp8_tensor = true;
      }
    }
    if (!w3_fp8_list_[idx].defined()) {
      auto w3 = state_dict.get_sharded_tensor(
          expert_prefix + w13_prefixes[1] + "weight", 0, rank, world_size);
      if (w3.defined()) {
        if (!is_fp8_weight(w3, local_intermediate_size_, hidden_size_)) {
          return false;
        }
        auto scale = state_dict.get_sharded_tensor(
            expert_prefix + w13_prefixes[1] + "scale", 0, rank, world_size);
        CHECK(is_fp8_block_scale(scale, local_intermediate_size_, hidden_size_))
            << "DeepseekV4 FP8 w3 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w13_prefixes[1];
        w3_fp8_list_[idx] = w3.contiguous();
        w3_fp8_scale_list_[idx] = scale.to(torch::kFloat32).contiguous();
        fp8_experts_detected_ = true;
        saw_fp8_tensor = true;
      }
    }
    if (!w2_fp8_list_[idx].defined()) {
      auto w2 = state_dict.get_sharded_tensor(
          expert_prefix + w2_prefix + "weight", 1, rank, world_size);
      if (w2.defined()) {
        if (!is_fp8_weight(w2, hidden_size_, local_intermediate_size_)) {
          return false;
        }
        auto scale = state_dict.get_sharded_tensor(
            expert_prefix + w2_prefix + "scale", 1, rank, world_size);
        CHECK(is_fp8_block_scale(scale, hidden_size_, local_intermediate_size_))
            << "DeepseekV4 FP8 w2 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w2_prefix;
        w2_fp8_list_[idx] = w2.contiguous();
        w2_fp8_scale_list_[idx] = scale.to(torch::kFloat32).contiguous();
        fp8_experts_detected_ = true;
        saw_fp8_tensor = true;
      }
    }
  }

  if (!w13_fp8_is_loaded_ && all_tensors_defined(w1_fp8_list_) &&
      all_tensors_defined(w3_fp8_list_) &&
      all_tensors_defined(w1_fp8_scale_list_) &&
      all_tensors_defined(w3_fp8_scale_list_)) {
    ensure_fp8_expert_storage();
    std::vector<torch::Tensor> weight_vec;
    std::vector<torch::Tensor> scale_vec;
    weight_vec.reserve(num_experts_per_rank_);
    scale_vec.reserve(num_experts_per_rank_);
    for (int64_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      weight_vec.push_back(
          torch::cat({w1_fp8_list_[idx], w3_fp8_list_[idx]}, 0));
      scale_vec.push_back(
          torch::cat({w1_fp8_scale_list_[idx], w3_fp8_scale_list_[idx]}, 0));
    }
    w13_fp8_.copy_(torch::stack(weight_vec).to(w13_fp8_.device()));
    w13_fp8_scale_.copy_(torch::stack(scale_vec).to(w13_fp8_scale_.device()));
    w13_fp8_is_loaded_ = true;
    w1_fp8_list_.clear();
    w3_fp8_list_.clear();
    w1_fp8_scale_list_.clear();
    w3_fp8_scale_list_.clear();
  }

  if (!w2_fp8_is_loaded_ && all_tensors_defined(w2_fp8_list_) &&
      all_tensors_defined(w2_fp8_scale_list_)) {
    ensure_fp8_expert_storage();
    w2_fp8_.copy_(torch::stack(w2_fp8_list_).to(w2_fp8_.device()));
    w2_fp8_scale_.copy_(
        torch::stack(w2_fp8_scale_list_).to(w2_fp8_scale_.device()));
    w2_fp8_is_loaded_ = true;
    w2_fp8_list_.clear();
    w2_fp8_scale_list_.clear();
  }

  return saw_fp8_tensor || fp8_experts_loaded();
}

torch::Tensor DeepseekV4MoEImpl::apply_expert(const torch::Tensor& input,
                                              int64_t local_expert_id) const {
  torch::Tensor w1;
  torch::Tensor w3;
  torch::Tensor w2;
  if (fp4_experts_loaded()) {
    auto w13_packed = w13_packed_.select(/*dim=*/0, local_expert_id);
    auto w13_scale = w13_scale_.select(/*dim=*/0, local_expert_id);
    w1 = unpack_fp4_weight(
        w13_packed.slice(
            /*dim=*/0, /*start=*/0, /*end=*/local_intermediate_size_),
        w13_scale.slice(
            /*dim=*/0, /*start=*/0, /*end=*/local_intermediate_size_),
        input.scalar_type());
    w3 = unpack_fp4_weight(
        w13_packed.slice(/*dim=*/0,
                         /*start=*/local_intermediate_size_,
                         /*end=*/local_intermediate_size_ * 2),
        w13_scale.slice(/*dim=*/0,
                        /*start=*/local_intermediate_size_,
                        /*end=*/local_intermediate_size_ * 2),
        input.scalar_type());
    w2 = unpack_fp4_weight(w2_packed_.select(/*dim=*/0, local_expert_id),
                           w2_scale_.select(/*dim=*/0, local_expert_id),
                           input.scalar_type());
  } else if (fp8_experts_loaded()) {
    const auto w13_fp8 = w13_fp8_.select(/*dim=*/0, local_expert_id);
    const auto w13_scale = w13_fp8_scale_.select(/*dim=*/0, local_expert_id);
    const auto w2_fp8 = w2_fp8_.select(/*dim=*/0, local_expert_id);
    const auto w2_scale = w2_fp8_scale_.select(/*dim=*/0, local_expert_id);
    w1 = dequantize_fp8_block_weight(
        w13_fp8.slice(/*dim=*/0, /*start=*/0, /*end=*/local_intermediate_size_),
        w13_scale.slice(/*dim=*/0,
                        /*start=*/0,
                        /*end=*/(local_intermediate_size_ + 127) / 128),
        input.scalar_type());
    w3 = dequantize_fp8_block_weight(
        w13_fp8.slice(/*dim=*/0,
                      /*start=*/local_intermediate_size_,
                      /*end=*/local_intermediate_size_ * 2),
        w13_scale.slice(/*dim=*/0,
                        /*start=*/(local_intermediate_size_ + 127) / 128,
                        /*end=*/(local_intermediate_size_ * 2 + 127) / 128),
        input.scalar_type());
    w2 = dequantize_fp8_block_weight(w2_fp8, w2_scale, input.scalar_type());
  } else {
    auto w13 = w13_.select(/*dim=*/0, local_expert_id);
    w1 = w13.slice(/*dim=*/0, /*start=*/0, /*end=*/local_intermediate_size_);
    w3 = w13.slice(/*dim=*/0,
                   /*start=*/local_intermediate_size_,
                   /*end=*/local_intermediate_size_ * 2);
    w2 = w2_.select(/*dim=*/0, local_expert_id);
  }
  auto gate = torch::matmul(input, w1.transpose(0, 1)).to(torch::kFloat32);
  auto up = torch::matmul(input, w3.transpose(0, 1)).to(torch::kFloat32);
  auto activated = swiglu(gate, up, swiglu_limit_).to(input.scalar_type());
  return torch::matmul(activated, w2.transpose(0, 1));
}

torch::Tensor DeepseekV4MoEImpl::forward_with_selected_experts(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const ModelInputParams& input_params) {
  (void)input_params;
  CHECK((w13_is_loaded_ && w2_is_loaded_) || fp4_experts_loaded() ||
        fp8_experts_loaded())
      << "DeepseekV4MoE expert weights must be loaded before forward.";

  auto input = hidden_states;
  auto input_2d = input.reshape({-1, input.size(-1)});
  auto weights_2d = topk_weights.reshape({-1, topk_}).to(input_2d.dtype());
  auto ids_2d = topk_ids.reshape({-1, topk_}).to(torch::kLong);
  CHECK_EQ(weights_2d.size(0), input_2d.size(0))
      << "DeepseekV4MoE topk_weights token count mismatch.";
  CHECK_EQ(ids_2d.size(0), input_2d.size(0))
      << "DeepseekV4MoE topk_ids token count mismatch.";

  torch::Tensor output = torch::zeros_like(input_2d);
  auto ids_cpu = ids_2d.to(torch::kCPU).contiguous();
  std::vector<std::vector<int64_t>> token_indices(num_experts_per_rank_);
  std::vector<std::vector<int64_t>> topk_indices(num_experts_per_rank_);
  for (int64_t token = 0; token < ids_cpu.size(0); ++token) {
    for (int64_t k = 0; k < ids_cpu.size(1); ++k) {
      const int64_t global_expert = ids_cpu.index({token, k}).item<int64_t>();
      if (global_expert < start_expert_id_ ||
          global_expert >= start_expert_id_ + num_experts_per_rank_) {
        continue;
      }
      const int64_t local_expert = global_expert - start_expert_id_;
      token_indices[local_expert].push_back(token);
      topk_indices[local_expert].push_back(k);
    }
  }
  for (int64_t local_expert = 0; local_expert < num_experts_per_rank_;
       ++local_expert) {
    if (token_indices[local_expert].empty()) {
      continue;
    }
    auto index_options =
        torch::TensorOptions().dtype(torch::kLong).device(input_2d.device());
    auto token_idx = torch::tensor(token_indices[local_expert], index_options);
    auto topk_idx = torch::tensor(topk_indices[local_expert], index_options);
    auto expert_input = input_2d.index_select(/*dim=*/0, token_idx);
    auto expert_output = apply_expert(expert_input, local_expert);
    auto expert_weight = weights_2d.index({token_idx, topk_idx})
                             .unsqueeze(-1)
                             .to(output.dtype());
    output.index_add_(/*dim=*/0, token_idx, expert_output * expert_weight);
  }

  if (ep_pg_ != nullptr && ep_pg_->world_size() > 1) {
    ep_pg_->allreduce(output);
  }

  if (shared_experts_) {
    auto shared_output = shared_experts_(input);
    if (shared_expert_gate_ && shared_expert_gate_is_loaded_) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
      shared_output = shared_output * gate;
    }
    output = output + shared_output.reshape_as(output);
  }

  return output.reshape(hidden_states.sizes());
}

void DeepseekV4MoEImpl::load_experts(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = tp_pg_->rank();
  const int64_t world_size = tp_pg_->world_size();

  std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  bool saw_packed_fp4 = try_load_packed_fp4_experts(
      state_dict, prefixes, "down_proj.", rank, world_size);
  if (!saw_packed_fp4) {
    prefixes = {"w1.", "w3."};
    saw_packed_fp4 = try_load_packed_fp4_experts(
        state_dict, prefixes, "w2.", rank, world_size);
  }
  if (saw_packed_fp4 || packed_fp4_experts_detected_) {
    return;
  }

  std::vector<std::string> fp8_prefixes = {"gate_proj.", "up_proj."};
  bool saw_fp8 = try_load_fp8_experts(
      state_dict, fp8_prefixes, "down_proj.", rank, world_size);
  if (!saw_fp8) {
    fp8_prefixes = {"w1.", "w3."};
    saw_fp8 =
        try_load_fp8_experts(state_dict, fp8_prefixes, "w2.", rank, world_size);
  }
  if (saw_fp8 || fp8_experts_detected_) {
    return;
  }

  ensure_dense_expert_storage();
  prefixes = {"gate_proj.", "up_proj."};
  weight::load_moe_fused_weight(state_dict,
                                prefixes,
                                "weight",
                                rank,
                                world_size,
                                start_expert_id_,
                                num_experts_per_rank_,
                                w1_list_,
                                w3_list_,
                                w13_,
                                w1_is_loaded_,
                                w3_is_loaded_,
                                w13_is_loaded_);
  if (!w13_is_loaded_) {
    prefixes = {"w1.", "w3."};
    weight::load_moe_fused_weight(state_dict,
                                  prefixes,
                                  "weight",
                                  rank,
                                  world_size,
                                  start_expert_id_,
                                  num_experts_per_rank_,
                                  w1_list_,
                                  w3_list_,
                                  w13_,
                                  w1_is_loaded_,
                                  w3_is_loaded_,
                                  w13_is_loaded_);
  }

  weight::load_moe_weight(state_dict,
                          "down_proj.",
                          "weight",
                          /*dim=*/1,
                          rank,
                          world_size,
                          start_expert_id_,
                          num_experts_per_rank_,
                          w2_list_,
                          w2_,
                          w2_is_loaded_);
  if (!w2_is_loaded_) {
    weight::load_moe_weight(state_dict,
                            "w2.",
                            "weight",
                            /*dim=*/1,
                            rank,
                            world_size,
                            start_expert_id_,
                            num_experts_per_rank_,
                            w2_list_,
                            w2_,
                            w2_is_loaded_);
  }
}

void DeepseekV4MoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  if (n_shared_experts_ > 0) {
    StateDict shared_expert_state =
        state_dict.get_dict_with_prefix("shared_expert.");
    if (shared_expert_state.size() == 0) {
      shared_expert_state = state_dict.get_dict_with_prefix("shared_experts.");
    }
    if (shared_expert_state.size() > 0) {
      if (has_w_style_shared_expert_weights(shared_expert_state)) {
        shared_experts_->load_state_dict(
            shared_expert_state, {"w1.", "w3."}, "w2.");
      } else {
        shared_experts_->load_state_dict(shared_expert_state);
      }
    }
    torch::Tensor weight = state_dict.get_tensor("shared_expert_gate.weight");
    if (!weight.defined()) {
      weight = state_dict.get_tensor("shared_experts_gate.weight");
    }
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      CHECK_EQ(shared_expert_gate_->weight.sizes(), weight.sizes())
          << "shared_expert_gate weight size mismatch.";
      shared_expert_gate_->weight.data().copy_(
          weight.to(shared_expert_gate_->weight.device())
              .to(shared_expert_gate_->weight.dtype()));
      shared_expert_gate_is_loaded_ = true;
    }
  }

  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
