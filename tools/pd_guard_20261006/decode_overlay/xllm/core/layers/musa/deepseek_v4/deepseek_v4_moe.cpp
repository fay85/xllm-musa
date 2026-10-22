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

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_moe_policy.h"
#include "core/layers/musa/musa_fp8_linear.h"

// The shared CUDA source is remapped into kernel::musa by musamapping, while
// musa_ops_api.h retains the historical CUDA namespace declaration. Keep the
// MUSA declaration local to this implementation so the grouped FP8 path can
// link against the actual MUSA archive symbol.
namespace xllm::kernel::musa {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id,
    int64_t num_experts);
}  // namespace xllm::kernel::musa

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

// Physical B16 speculative verify is 16 * (N6+1) = 112 target rows. Keep
// those rows on the fused FP8 GEMV path; falling back to ids_2d.to(kCPU)
// invokes the scalar expert loop and is illegal during MUSA Graph capture.
constexpr int64_t kFp8GemvWorkspaceTokens = 128;
constexpr int64_t kPrefillCustomAllreduceWorkspaceTokens = 1024;

struct SharedExpertOverlapState {
  explicit SharedExpertOverlapState(int32_t device_id)
      : stream(
            c10::musa::getStreamFromPool(/*isHighPriority=*/false, device_id)) {
    CHECK_EQ(musaEventCreateWithFlags(&input_ready, musaEventDisableTiming),
             musaSuccess);
    CHECK_EQ(musaEventCreateWithFlags(&shared_ready, musaEventDisableTiming),
             musaSuccess);
  }

  c10::musa::MUSAStream stream;
  musaEvent_t input_ready = nullptr;
  musaEvent_t shared_ready = nullptr;
};

SharedExpertOverlapState* get_shared_expert_overlap_state(int32_t device_id) {
  using States =
      std::unordered_map<int32_t, std::unique_ptr<SharedExpertOverlapState>>;
  thread_local States* states = new States();
  auto& state = (*states)[device_id];
  if (!state) {
    state = std::make_unique<SharedExpertOverlapState>(device_id);
  }
  return state.get();
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

DeepseekV4SharedExpertImpl::DeepseekV4SharedExpertImpl(
    int64_t hidden_size,
    int64_t intermediate_size,
    const std::string& hidden_act,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    double swiglu_limit)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      swiglu_limit_(swiglu_limit),
      process_group_(process_group) {
  CHECK(process_group_ != nullptr)
      << "DeepseekV4SharedExpert requires a valid TP group.";
  CHECK_EQ(intermediate_size_ % process_group_->world_size(), 0)
      << "DeepseekV4SharedExpert intermediate size must be divisible by the "
         "TP world size.";
  local_intermediate_size_ = intermediate_size_ / process_group_->world_size();

  gate_up_proj_ = register_module("gate_up_proj",
                                  ColumnParallelLinear(hidden_size_,
                                                       intermediate_size_ * 2,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       quant_args,
                                                       process_group_,
                                                       options));
  act_ = register_module(
      "act", Activation(hidden_act, /*is_gated=*/true, swiglu_limit_));
  down_proj_ =
      register_module("down_proj",
                      RowParallelLinear(intermediate_size_,
                                        hidden_size_,
                                        /*bias=*/false,
                                        /*input_is_parallelized=*/true,
                                        /*enable_result_reduction=*/false,
                                        quant_args,
                                        process_group_,
                                        options));

  decode_gate_up_workspace_ = torch::empty(
      {kFp8GemvWorkspaceTokens, local_intermediate_size_ * 2}, options);
  decode_activation_workspace_ = torch::empty(
      {kFp8GemvWorkspaceTokens, local_intermediate_size_}, options);
  decode_output_workspace_ =
      torch::empty({kFp8GemvWorkspaceTokens, hidden_size_}, options);
}

bool DeepseekV4SharedExpertImpl::fp8_decode_ready() const {
  const auto gate_up_weight = gate_up_proj_->weight();
  const auto gate_up_scale = gate_up_proj_->weight_scale();
  const auto down_weight = down_proj_->weight();
  return gate_up_weight.defined() && gate_up_scale.defined() &&
         down_weight.defined() && down_weight_scale_.defined() &&
         gate_up_weight.scalar_type() == torch::kFloat8_e4m3fn &&
         gate_up_scale.scalar_type() == torch::kFloat32 &&
         down_weight.scalar_type() == torch::kFloat8_e4m3fn &&
         down_weight_scale_.scalar_type() == torch::kFloat32 &&
         gate_up_weight.dim() == 2 &&
         gate_up_weight.size(0) == local_intermediate_size_ * 2 &&
         gate_up_weight.size(1) == hidden_size_ && gate_up_scale.dim() == 2 &&
         gate_up_scale.size(0) == local_intermediate_size_ * 2 / 128 &&
         gate_up_scale.size(1) == hidden_size_ / 128 &&
         down_weight.dim() == 2 && down_weight.size(0) == hidden_size_ &&
         down_weight.size(1) == local_intermediate_size_ &&
         down_weight_scale_.dim() == 2 &&
         down_weight_scale_.size(0) == hidden_size_ / 128 &&
         down_weight_scale_.size(1) == local_intermediate_size_ / 128;
}

torch::Tensor DeepseekV4SharedExpertImpl::forward(
    const torch::Tensor& hidden_states) {
  auto input = hidden_states.reshape({-1, hidden_size_});
  const int64_t num_tokens = input.size(0);
  if (num_tokens > 0 && num_tokens <= kFp8GemvWorkspaceTokens &&
      input.is_contiguous() && fp8_decode_ready()) {
    auto gate_up = musa::fp8_linear_forward(
        input,
        gate_up_proj_->weight(),
        gate_up_proj_->weight_scale(),
        std::nullopt,
        std::nullopt,
        input.scalar_type(),
        decode_gate_up_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens));
    auto activated = kernel::musa::deepseek_v4_shared_swiglu(
        gate_up,
        swiglu_limit_,
        decode_activation_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens));
    CHECK(activated.defined())
        << "DeepSeek-V4 MUSA shared-expert decode SwiGLU contract failed.";
    auto output = musa::fp8_linear_forward(
        activated,
        down_proj_->weight(),
        down_weight_scale_,
        std::nullopt,
        std::nullopt,
        input.scalar_type(),
        decode_output_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens));
    LOG_FIRST_N(INFO, 1)
        << "DeepSeek-V4 MUSA graph-safe shared-expert decode enabled.";
    return output.reshape(hidden_states.sizes());
  }

  auto gate_up = gate_up_proj_->forward(input);
  torch::Tensor activated =
      torch::empty({num_tokens, local_intermediate_size_}, gate_up.options());
  act_->forward(gate_up, activated);
  return down_proj_->forward(activated).reshape(hidden_states.sizes());
}

void DeepseekV4SharedExpertImpl::load_down_scale(const StateDict& state_dict,
                                                 const std::string& down_name) {
  const int64_t rank = process_group_->rank();
  const int64_t world_size = process_group_->world_size();
  auto scale = state_dict.get_sharded_tensor(
      down_name + "scale", /*dim=*/1, rank, world_size);
  if (!scale.defined()) {
    scale = state_dict.get_sharded_tensor(
        down_name + "weight_scale", /*dim=*/1, rank, world_size);
  }
  if (scale.defined()) {
    down_weight_scale_ = scale.to(torch::kFloat32)
                             .to(down_proj_->weight().device())
                             .contiguous();
  }
}

void DeepseekV4SharedExpertImpl::load_state_dict(const StateDict& state_dict) {
  load_state_dict(state_dict, {"gate_proj.", "up_proj."}, "down_proj.");
}

void DeepseekV4SharedExpertImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& gate_up_names,
    const std::string& down_name) {
  CHECK_EQ(gate_up_names.size(), 2)
      << "DeepseekV4SharedExpert expects gate and up weight prefixes.";
  gate_up_proj_->load_state_dict(state_dict, gate_up_names);
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix(down_name));
  load_down_scale(state_dict, down_name);
}

DeepseekV4MoEImpl::DeepseekV4MoEImpl(const ModelArgs& model_args,
                                     const QuantArgs& quant_args,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options,
                                     int32_t layer_id)
    : num_total_experts_(model_args.n_routed_experts()),
      topk_(model_args.n_activated_experts() > 0
                ? model_args.n_activated_experts()
                : model_args.num_experts_per_tok()),
      hidden_size_(model_args.hidden_size()),
      intermediate_size_(model_args.moe_intermediate_size() > 0
                             ? model_args.moe_intermediate_size()
                             : model_args.intermediate_size()),
      n_shared_experts_(model_args.n_shared_experts()),
      layer_id_(layer_id),
      num_layers_(static_cast<int32_t>(model_args.n_layers())),
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

  // Draft and B1 retain the existing arithmetic. Only the explicit target
  // model can opt into the separately measured large-row grouped FP8 path.
  if (model_args.model_type() == "deepseek_v4") {
    const char* minimum =
        std::getenv("XLLM_DSV4_GROUPED_FP8_DECODE_MIN_TOKENS");
    if (minimum != nullptr) {
      char* end = nullptr;
      const int64_t value =
          static_cast<int64_t>(std::strtoll(minimum, &end, 10));
      CHECK(end != minimum && *end == '\0' && (value == 77 || value == 112))
          << "Grouped FP8 decode threshold must be77 or112";
      grouped_fp8_decode_min_tokens_ = value;
    }
  }

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
  if (world_size > 1) {
    tp_custom_allreduce_output_workspace_ = torch::empty(
        {kPrefillCustomAllreduceWorkspaceTokens, hidden_size_}, options_);
  }

  if (n_shared_experts_ > 0) {
    shared_experts_ = register_module(
        "shared_experts",
        DeepseekV4SharedExpert(hidden_size_,
                               intermediate_size_ * n_shared_experts_,
                               hidden_act_,
                               quant_args,
                               tp_pg_,
                               options,
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

torch::Tensor DeepseekV4MoEImpl::forward_grouped_fp8(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    bool compact_rows) const {
  if (!fp8_experts_loaded()) {
    return torch::Tensor();
  }
  // The index kernel performs an in-stream int64->int32 cast. Keeping the
  // original device tensor here avoids a torch `.to()` during Graph capture;
  // the returned sorted assignment is already int32 for quant preparation.
  auto local_ids = topk_ids.contiguous();
  if (start_expert_id_ != 0 || num_experts_per_rank_ != num_total_experts_) {
    const auto valid_ids = local_ids.ge(start_expert_id_)
                               .logical_and(local_ids.lt(
                                   start_expert_id_ + num_experts_per_rank_));
    local_ids = torch::where(valid_ids,
                             local_ids - start_expert_id_,
                             torch::full_like(local_ids, -1));
  }

  // This is the MUSA build of the shared index kernel. The source file is
  // shared with CUDA but musamapping places its symbol in kernel::musa.
  const auto route_index = kernel::musa::moe_compute_index(
      local_ids.reshape({-1}), num_experts_per_rank_);
  const auto dst_to_src = std::get<1>(route_index);
  const auto expert_counts = std::get<2>(route_index);

  constexpr int64_t kAlignmentM = 128;
  const int64_t num_assignments = input.size(0) * topk_;
  const int64_t ep_world_size =
      ep_pg_ == nullptr ? 1 : std::max<int64_t>(ep_pg_->world_size(), 1);
  CHECK(!compact_rows || (ep_world_size == 1 && start_expert_id_ == 0 &&
                         num_experts_per_rank_ == num_total_experts_))
      << "Compact grouped FP8 requires all EP1 routes on this rank";
  // With EP, each rank only consumes its local expert routes.  Reserve a
  // 2x overload budget for routing skew, then add the per-active-expert
  // alignment slack.  EP1 keeps the exact all-route bound.
  const int64_t route_capacity =
      ep_world_size > 1
          ? ((num_assignments + ep_world_size - 1) / ep_world_size) * 2
          : num_assignments;
  const int64_t max_active_experts =
      std::min<int64_t>(num_experts_per_rank_, route_capacity);
  const int64_t max_padded_assignments = compact_rows
      ? num_assignments
      : ((route_capacity + max_active_experts * (kAlignmentM - 1) +
          kAlignmentM - 1) / kAlignmentM) * kAlignmentM;

  torch::Tensor sorted_input_fp8;
  torch::Tensor sorted_input_scale;
  torch::Tensor padded_assignment;
  torch::Tensor padded_expert;
  torch::Tensor route_to_padded;
  std::tie(sorted_input_fp8,
           sorted_input_scale,
           padded_assignment,
           padded_expert,
           route_to_padded) =
      kernel::musa::deepseek_v4_fp8_moe_quant_prepare(input,
                                                      dst_to_src,
                                                      expert_counts,
                                                      topk_,
                                                      compact_rows ? 1 : kAlignmentM,
                                                      max_padded_assignments);

  auto gate_up = compact_rows
      ? kernel::musa::contiguous_moe_gemm_fp8(sorted_input_fp8,
                                              sorted_input_scale,
                                              w13_fp8_,
                                              w13_fp8_scale_,
                                              expert_counts,
                                              input.scalar_type())
      : kernel::musa::ragged_moe_gemm_fp8(sorted_input_fp8,
                                                   sorted_input_scale,
                                                   w13_fp8_,
                                                   w13_fp8_scale_,
                                                   padded_expert,
                                                   input.scalar_type(),
                                                   kAlignmentM);
  if (!gate_up.defined()) {
    return torch::Tensor();
  }

  auto [activated_fp8, activated_scale] =
      kernel::musa::deepseek_v4_fp8_moe_swiglu_quantize(
          gate_up, padded_expert, swiglu_limit_);
  if (!activated_fp8.defined() || !activated_scale.defined()) {
    return torch::Tensor();
  }
  auto down = compact_rows
      ? kernel::musa::contiguous_moe_gemm_fp8(activated_fp8,
                                              activated_scale,
                                              w2_fp8_,
                                              w2_fp8_scale_,
                                              expert_counts,
                                              input.scalar_type())
      : kernel::musa::ragged_moe_gemm_fp8(activated_fp8,
                                                activated_scale,
                                                w2_fp8_,
                                                w2_fp8_scale_,
                                                padded_expert,
                                                input.scalar_type(),
                                                kAlignmentM);
  if (!down.defined()) {
    return torch::Tensor();
  }
  return kernel::musa::deepseek_v4_fp8_moe_restore_combine(
      down, route_to_padded, topk_weights.contiguous(), /*output=*/{});
}

torch::Tensor DeepseekV4MoEImpl::forward_with_selected_experts(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const ModelInputParams& input_params,
    bool skip_tp_allreduce,
    const torch::Tensor& output_buffer) {
  CHECK((w13_is_loaded_ && w2_is_loaded_) || fp4_experts_loaded() ||
        fp8_experts_loaded())
      << "DeepseekV4MoE expert weights must be loaded before forward.";
  const bool trace_decode_moe =
      input_params.meta.batch_forward_type.is_decode() &&
      std::getenv("XLLM_DSV4_MUSA_TRACE") != nullptr;
  if (trace_decode_moe) {
    LOG_FIRST_N(INFO, 1) << "[DSV4_MOE_TRACE] begin tokens="
                         << hidden_states.size(0) << " topk=" << topk_;
  }

  auto input = hidden_states;
  auto input_2d = input.reshape({-1, input.size(-1)});
  auto weights_2d = topk_weights.reshape({-1, topk_});
  auto ids_2d = topk_ids.reshape({-1, topk_});
  const bool is_prefill = !input_params.meta.batch_forward_type.is_decode();
  const bool use_shared_expert_overlap = is_prefill && shared_experts_ &&
                                         input_2d.device().is_privateuseone() &&
                                         !kernel::musa::is_stream_capturing();
  torch::Tensor shared_output;
  SharedExpertOverlapState* shared_overlap_state = nullptr;
  if (use_shared_expert_overlap) {
    shared_overlap_state = get_shared_expert_overlap_state(
        static_cast<int32_t>(input_2d.device().index()));
    const c10::musa::MUSAStream current_stream =
        c10::musa::getCurrentMUSAStream(input_2d.device().index());
    CHECK_EQ(musaEventRecord(shared_overlap_state->input_ready,
                             current_stream.stream()),
             musaSuccess);
    CHECK_EQ(musaStreamWaitEvent(shared_overlap_state->stream.stream(),
                                 shared_overlap_state->input_ready,
                                 /*flags=*/0),
             musaSuccess);
    {
      c10::StreamGuard stream_guard(shared_overlap_state->stream.unwrap());
      shared_output = shared_experts_(input);
      if (shared_expert_gate_ && shared_expert_gate_is_loaded_) {
        auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
        shared_output.mul_(gate);
      }
    }
    CHECK_EQ(musaEventRecord(shared_overlap_state->shared_ready,
                             shared_overlap_state->stream.stream()),
             musaSuccess);
  }
  const char* disable_fp8_gemv_env =
      std::getenv("XLLM_DSV4_DISABLE_MOE_FP8_GEMV");
  const bool disable_fp8_gemv =
      disable_fp8_gemv_env != nullptr &&
      std::string(disable_fp8_gemv_env) == "1";
  const bool grouped_fp8_verify = dsv4_grouped_fp8_verify_eligible(
      grouped_fp8_decode_min_tokens_, input_2d.size(0),
      input_params.meta.q_max_seq_len, input_params.is_spec_verify,
      input_params.meta.batch_forward_type.is_decode(),
      start_expert_id_ == 0 && num_experts_per_rank_ == num_total_experts_,
      fp8_experts_loaded());
  const bool use_fp8_gemv =
      !disable_fp8_gemv && !grouped_fp8_verify &&
      input_params.meta.batch_forward_type.is_decode() &&
      input_2d.size(0) > 0 && input_2d.size(0) <= kFp8GemvWorkspaceTokens &&
      fp8_experts_loaded();
  if (!use_fp8_gemv && !grouped_fp8_verify &&
      weights_2d.scalar_type() != input_2d.scalar_type()) {
    weights_2d = weights_2d.to(input_2d.dtype());
  }
  CHECK_EQ(weights_2d.size(0), input_2d.size(0))
      << "DeepseekV4MoE topk_weights token count mismatch.";
  CHECK_EQ(ids_2d.size(0), input_2d.size(0))
      << "DeepseekV4MoE topk_ids token count mismatch.";
  torch::Tensor output;
  if (!output.defined() && use_fp8_gemv) {
    const int64_t num_tokens = input_2d.size(0);
    const int64_t num_assignments = num_tokens * topk_;
    auto local_ids = ids_2d.scalar_type() == torch::kInt32
                         ? ids_2d.contiguous()
                         : ids_2d.to(torch::kInt32).contiguous();
    if (start_expert_id_ != 0 || num_experts_per_rank_ != num_total_experts_) {
      const auto valid_ids = local_ids.ge(start_expert_id_)
                                 .logical_and(local_ids.lt(
                                     start_expert_id_ + num_experts_per_rank_));
      local_ids = torch::where(valid_ids,
                               local_ids - start_expert_id_,
                               torch::full_like(local_ids, -1));
    }
    output = kernel::musa::deepseek_v4_fp8_moe_gemv_decode(
        input_2d.contiguous(),
        local_ids,
        weights_2d.contiguous(),
        w13_fp8_,
        w13_fp8_scale_,
        w2_fp8_,
        w2_fp8_scale_,
        fp8_gemv_activation_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_assignments),
        fp8_gemv_routed_output_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_assignments),
        fp8_gemv_output_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens),
        swiglu_limit_);
  }

  if (!output.defined() && grouped_fp8_verify) {
    // Preserve FP32 router weights as in GEMV. Grouped activation FP8 rounding
    // is a deliberate experimental change and requires model accuracy gates.
    // Only the already-qualified target-q7 branch may switch row layout.
    // Prefill, B1, draft, ordinary Decode and the default path stay unchanged.
    const char* compact_env = std::getenv("XLLM_DSV4_COMPACT_GROUPED_FP8");
    const bool compact_rows = compact_env != nullptr && compact_env[0] == '1' &&
                              compact_env[1] == '\0';
    output = forward_grouped_fp8(
        input_2d.contiguous(), ids_2d, weights_2d, compact_rows);
    if (compact_rows) {
      LOG_FIRST_N(INFO, 1) << "[DSV4_COMPACT_GROUPED_FP8_VERIFY] rows="
                         << input_2d.size(0)
                         << " assignments=" << input_2d.size(0) * topk_
                         << " query_width=" << input_params.meta.q_max_seq_len;
    }
    CHECK(output.defined())
        << "Grouped FP8 verify cannot fall back to host routing in Graph";
    LOG_FIRST_N(INFO, 1) << "[DSV4_GROUPED_FP8_VERIFY] rows=" << input_2d.size(0)
                       << " minimum=" << grouped_fp8_decode_min_tokens_
                       << " query_width=" << input_params.meta.q_max_seq_len;
  }
  if (!output.defined() && !input_params.meta.batch_forward_type.is_decode() &&
      fp8_experts_loaded()) {
    output = forward_grouped_fp8(input_2d.contiguous(), ids_2d, weights_2d);
  }

  if (!output.defined()) {
    output = torch::zeros_like(input_2d);
    auto ids_cpu = ids_2d.to(torch::kLong).to(torch::kCPU).contiguous();
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
      auto token_idx =
          torch::tensor(token_indices[local_expert], index_options);
      auto topk_idx = torch::tensor(topk_indices[local_expert], index_options);
      auto expert_input = input_2d.index_select(/*dim=*/0, token_idx);
      auto expert_output = apply_expert(expert_input, local_expert);
      auto expert_weight = weights_2d.index({token_idx, topk_idx})
                               .unsqueeze(-1)
                               .to(output.dtype());
      output.index_add_(/*dim=*/0, token_idx, expert_output * expert_weight);
    }
  }

  if (ep_pg_ != nullptr && ep_pg_->world_size() > 1) {
    ep_pg_->allreduce(output);
  }

  if (shared_experts_ && !shared_output.defined()) {
    auto shared_output = shared_experts_(input);
    if (shared_expert_gate_ && shared_expert_gate_is_loaded_) {
      auto shared_output_2d =
          shared_output.reshape({-1, hidden_size_}).contiguous();
      const auto gate_weight = shared_expert_gate_->weight;
      const auto is_aligned = [](const torch::Tensor& tensor) {
        return tensor.defined() &&
               reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0;
      };
      const bool use_fused_shared_gate =
          input_2d.is_contiguous() && shared_output_2d.is_contiguous() &&
          output.is_contiguous() &&
          (input_2d.scalar_type() == torch::kBFloat16 ||
           input_2d.scalar_type() == torch::kFloat16) &&
          gate_weight.scalar_type() == input_2d.scalar_type() &&
          shared_output_2d.scalar_type() == input_2d.scalar_type() &&
          output.scalar_type() == input_2d.scalar_type() &&
          is_aligned(input_2d) && is_aligned(gate_weight) &&
          is_aligned(shared_output_2d) && is_aligned(output);
      if (use_fused_shared_gate) {
        output = kernel::musa::deepseek_v4_shared_expert_gate(
            input_2d, gate_weight, shared_output_2d, output);
      } else {
        auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
        shared_output.mul_(gate);
        output.add_(shared_output.reshape_as(output));
      }
    } else {
      output.add_(shared_output.reshape_as(output));
    }
  }

  if (shared_overlap_state != nullptr) {
    const c10::musa::MUSAStream current_stream =
        c10::musa::getCurrentMUSAStream(input_2d.device().index());
    CHECK_EQ(musaStreamWaitEvent(current_stream.stream(),
                                 shared_overlap_state->shared_ready,
                                 /*flags=*/0),
             musaSuccess);
    output.add_(shared_output.reshape_as(output));
  }

  if (output_buffer.defined()) {
    CHECK(output_buffer.dim() == 2 && output_buffer.is_contiguous());
    CHECK_GE(output_buffer.size(0), output.size(0));
    CHECK_EQ(output_buffer.size(1), output.size(1));
    auto valid_output = output_buffer.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/output.size(0));
    if (valid_output.data_ptr() != output.data_ptr()) {
      valid_output.copy_(output);
      if (output_buffer.size(0) > output.size(0)) {
        output_buffer
            .narrow(/*dim=*/0,
                    /*start=*/output.size(0),
                    /*length=*/output_buffer.size(0) - output.size(0))
            .zero_();
      }
      output = valid_output;
    }
  }

  // Routed and shared experts are sharded across the TP group. Reduce their
  // combined partial hidden-state sum once, after both branches are complete.
  if (tp_pg_ != nullptr && tp_pg_->world_size() > 1 && !skip_tp_allreduce) {
    const bool use_custom_allreduce =
        is_prefill || num_layers_ <= 4 || layer_id_ < num_layers_ - 4;
    bool used_custom_allreduce = false;
    if (use_custom_allreduce && output.dim() == 2 && output.size(0) > 0 &&
        output.size(0) <= (is_prefill ? kPrefillCustomAllreduceWorkspaceTokens
                                      : kFp8GemvWorkspaceTokens) &&
        output.size(1) == hidden_size_ &&
        tp_custom_allreduce_output_workspace_.defined()) {
      torch::Tensor output_workspace =
          tp_custom_allreduce_output_workspace_.narrow(
              /*dim=*/0, /*start=*/0, /*length=*/output.size(0));
      if (std::optional<torch::Tensor> custom_output = musa_allreduce_custom(
              tp_pg_, output, output_workspace.reshape_as(output))) {
        output = *custom_output;
        used_custom_allreduce = true;
        LOG_FIRST_N(INFO, 1)
            << "DeepSeek-V4 MUSA MoE out-of-place custom all-reduce enabled.";
      }
    }
    if (!used_custom_allreduce) {
      tp_pg_->allreduce(output);
    }
  }

  if (trace_decode_moe) {
    LOG_FIRST_N(INFO, 1) << "[DSV4_MOE_TRACE] end tokens="
                         << hidden_states.size(0);
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
  if (fp8_experts_loaded() && !fp8_gemv_activation_workspace_.defined()) {
    fp8_gemv_activation_workspace_ = torch::empty(
        {kFp8GemvWorkspaceTokens * topk_, local_intermediate_size_}, options_);
    fp8_gemv_routed_output_workspace_ =
        torch::empty({kFp8GemvWorkspaceTokens * topk_, hidden_size_}, options_);
    fp8_gemv_output_workspace_ =
        torch::empty({kFp8GemvWorkspaceTokens, hidden_size_}, options_);
  }
}

}  // namespace layer
}  // namespace xllm
