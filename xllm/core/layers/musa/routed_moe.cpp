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

#include "core/layers/musa/routed_moe.h"

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/musa/musa_fp8_linear.h"
#include "core/layers/musa/musa_mega_moe.h"

namespace xllm::kernel::musa {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id,
    int64_t num_experts);
}  // namespace xllm::kernel::musa

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kFp8GemvWorkspaceTokens = 128;
constexpr int64_t kPrefillCustomAllreduceWorkspaceTokens = 16384;

bool env_flag_enabled(const char* name, bool default_value = false) {
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  }
  return !(env[0] == '0' && env[1] == '\0');
}

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
  torch::Tensor gated = gate;
  torch::Tensor up_clamped = up;
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
         (tensor.scalar_type() == torch::kFloat8_e8m0fnu ||
          tensor.scalar_type() == torch::kFloat32) &&
         tensor.size(0) == logical_out_features &&
         tensor.size(1) * 32 == logical_in_features;
}

bool is_fp8_weight(const torch::Tensor& tensor,
                   int64_t logical_out_features,
                   int64_t logical_in_features) {
  return tensor.defined() && tensor.dim() == 2 &&
         tensor.scalar_type() == torch::kFloat8_e4m3fn &&
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

torch::Tensor get_sharded_expert_scale(const StateDict& state_dict,
                                       const std::string& weight_prefix,
                                       int64_t shard_dim,
                                       int64_t rank,
                                       int64_t world_size) {
  static const char* kScaleSuffixes[] = {
      "scale", "weight_scale_inv", "weight_scale"};
  for (const char* suffix : kScaleSuffixes) {
    torch::Tensor scale = state_dict.get_sharded_tensor(
        weight_prefix + suffix, shard_dim, rank, world_size);
    if (scale.defined()) {
      return scale;
    }
  }
  return torch::Tensor();
}

torch::Tensor dequantize_fp8_block_weight(const torch::Tensor& weight,
                                          const torch::Tensor& scale,
                                          torch::ScalarType output_dtype) {
  CHECK(is_fp8_weight(weight, weight.size(0), weight.size(1)))
      << "RoutedMoe FP8 expert weight must be a 2D FP8 tensor.";
  CHECK(is_fp8_block_scale(scale, weight.size(0), weight.size(1)))
      << "RoutedMoe FP8 expert scale must be a 128x128 block scale tensor.";
  const int64_t block_rows =
      (weight.size(0) + scale.size(0) - 1) / scale.size(0);
  const int64_t block_cols =
      (weight.size(1) + scale.size(1) - 1) / scale.size(1);
  torch::Tensor expanded_scale =
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
      << "RoutedMoe FP4 expert weight expects packed int8 [out, in/2].";
  CHECK(is_fp4_scale(scale, packed.size(0), packed.size(1) * 2))
      << "RoutedMoe FP4 expert scale expects [out, in/32].";

  torch::Tensor byte_values = packed.to(torch::kInt16);
  byte_values = torch::where(byte_values.lt(0), byte_values + 256, byte_values);
  torch::Tensor byte_values_fp32 = byte_values.to(torch::kFloat32);
  torch::Tensor high = torch::floor(byte_values_fp32 / 16);
  torch::Tensor low = byte_values_fp32 - high * 16;

  torch::Tensor table = torch::tensor(
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
  torch::Tensor low_values =
      table.index_select(/*dim=*/0, low.to(torch::kLong).view({-1}))
          .view(low.sizes());
  torch::Tensor high_values =
      table.index_select(/*dim=*/0, high.to(torch::kLong).view({-1}))
          .view(high.sizes());
  torch::Tensor values = torch::stack({low_values, high_values}, /*dim=*/-1)
                             .flatten(/*start_dim=*/1);
  torch::Tensor expanded_scale =
      scale.to(torch::kFloat32).unsqueeze(-1).repeat({1, 1, 32}).flatten(1);
  return (values * expanded_scale).to(dtype).contiguous();
}

}  // namespace

RoutedMoeTokenShard make_routed_moe_token_shard(
    const AttentionMetadata& metadata,
    const torch::Tensor& positions) {
  CHECK(metadata.glm5_prefill_cp_enabled);
  CHECK(metadata.is_prefill || metadata.is_chunked_prefill);
  const int64_t count = metadata.glm5_prefill_local_row_end -
                        metadata.glm5_prefill_local_row_begin;
  CHECK_EQ(positions.sizes(), torch::IntArrayRef({count}));
  CHECK_EQ(metadata.q_cu_seq_lens.device(), positions.device());
  torch::Tensor rows;
  if (metadata.glm5_prefill_cp_round_robin) {
    rows = metadata.glm5_prefill_local_row_indices.to(positions.device(),
                                                      torch::kInt64);
    CHECK_EQ(rows.numel(), count);
  } else {
    rows = torch::arange(metadata.glm5_prefill_local_row_begin,
                         metadata.glm5_prefill_local_row_end,
                         positions.options().dtype(torch::kInt64));
  }
  auto requests =
      torch::bucketize(rows,
                       metadata.q_cu_seq_lens.slice(0, 1).to(torch::kInt64),
                       /*out_int32=*/true,
                       /*right=*/true);
  return {metadata.glm5_prefill_cp_rank,
          metadata.glm5_prefill_cp_size,
          rows,
          requests,
          positions.to(torch::kInt64).contiguous()};
}

RoutedMoeSharedExpertImpl::RoutedMoeSharedExpertImpl(
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
      fused_shared_swiglu_(hidden_act == "silu" &&
                           options.dtype() == torch::kBFloat16 &&
                           options.device().is_privateuseone() &&
                           quant_args.quant_method() == "fp8"),
      process_group_(process_group) {
  CHECK(process_group_ != nullptr)
      << "RoutedMoeSharedExpert requires a valid TP group.";
  CHECK_EQ(intermediate_size_ % process_group_->world_size(), 0)
      << "RoutedMoeSharedExpert intermediate size must be divisible by the "
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

bool RoutedMoeSharedExpertImpl::fp8_decode_ready() const {
  const torch::Tensor gate_up_weight = gate_up_proj_->weight();
  const torch::Tensor gate_up_scale = gate_up_proj_->weight_scale();
  const torch::Tensor down_weight = down_proj_->weight();
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

torch::Tensor RoutedMoeSharedExpertImpl::forward(
    const torch::Tensor& hidden_states) {
  torch::Tensor input = hidden_states.reshape({-1, hidden_size_});
  const int64_t num_tokens = input.size(0);
  if (num_tokens == 0) {
    return torch::empty_like(hidden_states);
  }
  if (num_tokens > 0 && num_tokens <= kFp8GemvWorkspaceTokens &&
      input.is_contiguous() && fp8_decode_ready()) {
    torch::Tensor gate_up = musa::fp8_linear_forward(
        input,
        gate_up_proj_->weight(),
        gate_up_proj_->weight_scale(),
        std::nullopt,
        std::nullopt,
        input.scalar_type(),
        decode_gate_up_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens));
    torch::Tensor activated = kernel::musa::deepseek_v4_shared_swiglu(
        gate_up,
        swiglu_limit_,
        decode_activation_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens));
    CHECK(activated.defined())
        << "RoutedMoe shared-expert decode SwiGLU contract failed.";
    torch::Tensor output = musa::fp8_linear_forward(
        activated,
        down_proj_->weight(),
        down_weight_scale_,
        std::nullopt,
        std::nullopt,
        input.scalar_type(),
        decode_output_workspace_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens));
    return output.reshape(hidden_states.sizes());
  }

  torch::Tensor gate_up = gate_up_proj_->forward(input);
  torch::Tensor activated =
      torch::empty({num_tokens, local_intermediate_size_}, gate_up.options());
  if (fused_shared_swiglu_) {
    CHECK(kernel::musa::deepseek_v4_shared_swiglu(
              gate_up, swiglu_limit_, activated)
              .defined());
  } else {
    act_->forward(gate_up, activated);
  }
  return down_proj_->forward(activated).reshape(hidden_states.sizes());
}

void RoutedMoeSharedExpertImpl::load_down_scale(const StateDict& state_dict,
                                                const std::string& down_name) {
  const int64_t rank = process_group_->rank();
  const int64_t world_size = process_group_->world_size();
  torch::Tensor scale = get_sharded_expert_scale(
      state_dict, down_name, /*dim=*/1, rank, world_size);
  if (scale.defined()) {
    down_weight_scale_ = scale.to(torch::kFloat32)
                             .to(down_proj_->weight().device())
                             .contiguous();
  }
}

void RoutedMoeSharedExpertImpl::load_state_dict(const StateDict& state_dict) {
  load_state_dict(state_dict, {"gate_proj.", "up_proj."}, "down_proj.");
}

void RoutedMoeSharedExpertImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& gate_up_names,
    const std::string& down_name) {
  CHECK_EQ(gate_up_names.size(), 2)
      << "RoutedMoeSharedExpert expects gate and up weight prefixes.";
  gate_up_proj_->load_state_dict(state_dict, gate_up_names);
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix(down_name));
  load_down_scale(state_dict, down_name);
}

RoutedMoeImpl::RoutedMoeImpl(const ModelArgs& model_args,
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
  CHECK(tp_pg_ != nullptr) << "RoutedMoe requires a valid TP group.";
  CHECK_GT(num_total_experts_, 0) << "RoutedMoe requires n_routed_experts > 0.";
  CHECK_GT(topk_, 0) << "RoutedMoe requires topk > 0.";
  CHECK_GT(hidden_size_, 0) << "RoutedMoe requires hidden_size > 0.";
  CHECK_GT(intermediate_size_, 0)
      << "RoutedMoe requires moe_intermediate_size/intermediate_size > 0.";

  prefill_cp_enabled_ = model_args.model_type() == "glm5_next" &&
                        parallel_args.cp_size() > 1 &&
                        env_flag_enabled("XLLM_GLM5_PREFILL_CP");
  if (prefill_cp_enabled_) {
    CHECK(parallel_args.cp_group_ != nullptr);
    CHECK(parallel_args.tp_group_ != nullptr);
    CHECK_EQ(parallel_args.tp_group_->world_size(), 1);
    CHECK_EQ(parallel_args.dp_size(), 1);
    CHECK_EQ(tp_pg_->world_size(), 1) << "GLM CP local tokens cannot use a MoE "
                                         "TP AllReduce; moe_tp must be 1";
    CHECK(parallel_args.ep_size() == 1 || ep_pg_ != nullptr)
        << "GLM local-token MoE EP requires moe_ep_group when ep_size > 1";
    prefill_cp_rank_ = parallel_args.cp_group_->rank();
    prefill_cp_size_ = parallel_args.cp_group_->world_size();
    shared_single_rank_group_ =
        std::make_unique<ProcessGroup>(0, 1, options.device());
  }
  const int64_t ep_size = std::max<int64_t>(parallel_args.ep_size(), 1);
  if (ep_size > 1) {
    CHECK(ep_pg_ != nullptr)
        << "RoutedMoe EP requires a moe_ep_group when ep_size > 1";
    CHECK(env_flag_enabled("XLLM_GLM5_PREFILL_EP"))
        << "RoutedMoe EP1 requires XLLM_GLM5_PREFILL_EP=1";
    CHECK_EQ(ep_pg_->world_size(), ep_size)
        << "RoutedMoe EP group size must equal ep_size";
    CHECK_EQ(tp_pg_->world_size(), 1)
        << "RoutedMoe EP1 contract requires moe_tp=1";
    ep_dispatch_enabled_ = true;
  } else {
    CHECK(ep_pg_ == nullptr || ep_pg_->world_size() == 1)
        << "RoutedMoe EP0 requires a single-rank moe_ep_group";
  }
  const int64_t ep_rank = ep_size > 1 && parallel_args.moe_ep_group_ != nullptr
                              ? parallel_args.moe_ep_group_->rank()
                              : 0;
  CHECK_EQ(num_total_experts_ % ep_size, 0)
      << "RoutedMoe expects n_routed_experts divisible by ep_size.";
  num_experts_per_rank_ = num_total_experts_ / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  const int64_t world_size = std::max<int64_t>(tp_pg_->world_size(), 1);
  CHECK_EQ(intermediate_size_ % world_size, 0)
      << "RoutedMoe intermediate size must be divisible by TP world size.";
  local_intermediate_size_ = intermediate_size_ / world_size;
  if (world_size > 1) {
    tp_custom_allreduce_output_workspace_ = torch::empty(
        {kPrefillCustomAllreduceWorkspaceTokens, hidden_size_}, options_);
  }

  // The supported checkpoint has one compiled expert layout. The EP case
  // receives local CP rows. TP8 attention supplies replicated rows instead.
  if ((model_args.model_type() == "glm5_next" ||
       model_args.model_type() == "glm5_next_mtp") &&
      quant_args.quant_method() == "fp8" && num_total_experts_ == 288 &&
      topk_ == 8 && hidden_size_ == 4096 && intermediate_size_ == 2048 &&
      (ep_size == 8 || (ep_size == 1 && world_size == 8))) {
    auto* group = ep_size == 8 ? ep_pg_ : tp_pg_;
    mega_moe_ = MusaMegaMoe::acquire(group);
    num_experts_per_rank_ = num_total_experts_ / group->world_size();
    start_expert_id_ = group->rank() * num_experts_per_rank_;
    local_intermediate_size_ = intermediate_size_;
  }

  if (n_shared_experts_ > 0) {
    shared_experts_ = register_module(
        "shared_experts",
        RoutedMoeSharedExpert(hidden_size_,
                              intermediate_size_ * n_shared_experts_,
                              hidden_act_,
                              quant_args,
                              shared_single_rank_group_
                                  ? shared_single_rank_group_.get()
                                  : tp_pg_,
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
      << "RoutedMoe expert path requires hidden_size divisible by 32.";
  CHECK_EQ(local_intermediate_size_ % 32, 0)
      << "RoutedMoe expert path requires local intermediate size divisible by "
         "32.";
}

void RoutedMoeImpl::ensure_dense_expert_storage() {
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

void RoutedMoeImpl::ensure_fp8_expert_storage() {
  const torch::TensorOptions weight_options =
      options_.dtype(torch::kFloat8_e4m3fn);
  const torch::TensorOptions scale_options = options_.dtype(torch::kFloat32);
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
                      mega_moe_ ? local_intermediate_size_ / 32
                                : (local_intermediate_size_ * 2 + 127) / 128,
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

void RoutedMoeImpl::ensure_packed_fp4_expert_storage() {
  const torch::TensorOptions packed_options = options_.dtype(torch::kInt8);
  const torch::TensorOptions scale_options = options_.dtype(torch::kFloat32);
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

bool RoutedMoeImpl::try_load_packed_fp4_experts(
    const StateDict& state_dict,
    const std::vector<std::string>& w13_prefixes,
    const std::string& w2_prefix,
    int64_t rank,
    int64_t world_size) {
  return try_load_quantized_experts(state_dict,
                                    w13_prefixes,
                                    w2_prefix,
                                    rank,
                                    world_size,
                                    /*packed_fp4=*/true);
}

bool RoutedMoeImpl::try_load_fp8_experts(
    const StateDict& state_dict,
    const std::vector<std::string>& w13_prefixes,
    const std::string& w2_prefix,
    int64_t rank,
    int64_t world_size) {
  return try_load_quantized_experts(state_dict,
                                    w13_prefixes,
                                    w2_prefix,
                                    rank,
                                    world_size,
                                    /*packed_fp4=*/false);
}

bool RoutedMoeImpl::try_load_quantized_experts(
    const StateDict& state_dict,
    const std::vector<std::string>& w13_prefixes,
    const std::string& w2_prefix,
    int64_t rank,
    int64_t world_size,
    bool packed_fp4) {
  auto& w1_list = packed_fp4 ? w1_packed_list_ : w1_fp8_list_;
  auto& w3_list = packed_fp4 ? w3_packed_list_ : w3_fp8_list_;
  auto& w2_list = packed_fp4 ? w2_packed_list_ : w2_fp8_list_;
  auto& s1_list = packed_fp4 ? w1_scale_list_ : w1_fp8_scale_list_;
  auto& s3_list = packed_fp4 ? w3_scale_list_ : w3_fp8_scale_list_;
  auto& s2_list = packed_fp4 ? w2_scale_list_ : w2_fp8_scale_list_;
  torch::Tensor& w13 = packed_fp4 ? w13_packed_ : w13_fp8_;
  torch::Tensor& s13 = packed_fp4 ? w13_scale_ : w13_fp8_scale_;
  torch::Tensor& w2_storage = packed_fp4 ? w2_packed_ : w2_fp8_;
  torch::Tensor& s2 = packed_fp4 ? w2_scale_ : w2_fp8_scale_;
  bool& w13_loaded = packed_fp4 ? w13_packed_is_loaded_ : w13_fp8_is_loaded_;
  bool& w2_loaded = packed_fp4 ? w2_packed_is_loaded_ : w2_fp8_is_loaded_;
  bool& detected =
      packed_fp4 ? packed_fp4_experts_detected_ : fp8_experts_detected_;
  const auto valid_weight = packed_fp4 ? is_packed_fp4_weight : is_fp8_weight;
  const auto valid_scale = packed_fp4 ? is_fp4_scale : is_fp8_block_scale;
  auto ensure_storage = [this, packed_fp4]() {
    if (packed_fp4) {
      ensure_packed_fp4_expert_storage();
    } else {
      ensure_fp8_expert_storage();
    }
  };
  CHECK_EQ(w13_prefixes.size(), 2)
      << "RoutedMoe quantized fused load expects two prefixes.";

  if (w1_list.size() < static_cast<size_t>(num_experts_per_rank_)) {
    w1_list.resize(num_experts_per_rank_);
    w3_list.resize(num_experts_per_rank_);
    s1_list.resize(num_experts_per_rank_);
    s3_list.resize(num_experts_per_rank_);
    w2_list.resize(num_experts_per_rank_);
    s2_list.resize(num_experts_per_rank_);
  }

  bool saw_tensor = false;
  for (int64_t idx = 0; idx < num_experts_per_rank_; ++idx) {
    const std::string expert_prefix =
        std::to_string(start_expert_id_ + idx) + ".";
    if (!w1_list[idx].defined()) {
      torch::Tensor w1 = state_dict.get_sharded_tensor(
          expert_prefix + w13_prefixes[0] + "weight", 0, rank, world_size);
      if (w1.defined()) {
        if (!valid_weight(w1, local_intermediate_size_, hidden_size_)) {
          return false;
        }
        torch::Tensor scale = get_sharded_expert_scale(
            state_dict, expert_prefix + w13_prefixes[0], 0, rank, world_size);
        CHECK(valid_scale(scale, local_intermediate_size_, hidden_size_))
            << "RoutedMoe quantized w1 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w13_prefixes[0];
        w1_list[idx] = w1.contiguous();
        s1_list[idx] = scale.to(torch::kFloat32).contiguous();
        detected = true;
        saw_tensor = true;
      }
    }
    if (!w3_list[idx].defined()) {
      torch::Tensor w3 = state_dict.get_sharded_tensor(
          expert_prefix + w13_prefixes[1] + "weight", 0, rank, world_size);
      if (w3.defined()) {
        if (!valid_weight(w3, local_intermediate_size_, hidden_size_)) {
          return false;
        }
        torch::Tensor scale = get_sharded_expert_scale(
            state_dict, expert_prefix + w13_prefixes[1], 0, rank, world_size);
        CHECK(valid_scale(scale, local_intermediate_size_, hidden_size_))
            << "RoutedMoe quantized w3 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w13_prefixes[1];
        w3_list[idx] = w3.contiguous();
        s3_list[idx] = scale.to(torch::kFloat32).contiguous();
        detected = true;
        saw_tensor = true;
      }
    }
    if (!w2_list[idx].defined()) {
      torch::Tensor w2 = state_dict.get_sharded_tensor(
          expert_prefix + w2_prefix + "weight", 1, rank, world_size);
      if (w2.defined()) {
        if (!valid_weight(w2, hidden_size_, local_intermediate_size_)) {
          return false;
        }
        torch::Tensor scale = get_sharded_expert_scale(
            state_dict, expert_prefix + w2_prefix, 1, rank, world_size);
        CHECK(valid_scale(scale, hidden_size_, local_intermediate_size_))
            << "RoutedMoe quantized w2 scale size mismatch for "
            << state_dict.prefix() << expert_prefix << w2_prefix;
        w2_list[idx] = w2.contiguous();
        s2_list[idx] = scale.to(torch::kFloat32).contiguous();
        detected = true;
        saw_tensor = true;
      }
    }
  }

  if (!w13_loaded && all_tensors_defined(w1_list) &&
      all_tensors_defined(w3_list) && all_tensors_defined(s1_list) &&
      all_tensors_defined(s3_list)) {
    ensure_storage();
    std::vector<torch::Tensor> packed_vec;
    std::vector<torch::Tensor> scale_vec;
    packed_vec.reserve(num_experts_per_rank_);
    scale_vec.reserve(num_experts_per_rank_);
    for (int64_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      packed_vec.emplace_back(torch::cat({w1_list[idx], w3_list[idx]}, 0));
      scale_vec.emplace_back(torch::cat({s1_list[idx], s3_list[idx]}, 0));
    }
    if (mega_moe_) {
      CHECK(!packed_fp4);
      // Transform checkpoint tensors before H2D. Concurrent layer loading
      // must not retain both TP-style and interleaved EP GPU weights or
      // allocate a second full device tensor for the copy.
      auto transformed = MusaMegaMoe::transform_gate_up(
          torch::stack(packed_vec), torch::stack(scale_vec));
      w13.copy_(transformed.first);
      s13.copy_(transformed.second);
    } else {
      w13.copy_(torch::stack(packed_vec));
      s13.copy_(torch::stack(scale_vec));
    }
    w13_loaded = true;
    w1_list.clear();
    w3_list.clear();
    s1_list.clear();
    s3_list.clear();
  }

  if (!w2_loaded && all_tensors_defined(w2_list) &&
      all_tensors_defined(s2_list)) {
    ensure_storage();
    w2_storage.copy_(torch::stack(w2_list));
    s2.copy_(torch::stack(s2_list));
    w2_loaded = true;
    w2_list.clear();
    s2_list.clear();
  }

  return saw_tensor || (w13_loaded && w2_loaded);
}

torch::Tensor RoutedMoeImpl::apply_expert(const torch::Tensor& input,
                                          int64_t local_expert_id) const {
  torch::Tensor w1;
  torch::Tensor w3;
  torch::Tensor w2;
  if (fp4_experts_loaded()) {
    torch::Tensor w13_packed = w13_packed_.select(/*dim=*/0, local_expert_id);
    torch::Tensor w13_scale = w13_scale_.select(/*dim=*/0, local_expert_id);
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
    const torch::Tensor w13_fp8 = w13_fp8_.select(/*dim=*/0, local_expert_id);
    const torch::Tensor w13_scale =
        w13_fp8_scale_.select(/*dim=*/0, local_expert_id);
    const torch::Tensor w2_fp8 = w2_fp8_.select(/*dim=*/0, local_expert_id);
    const torch::Tensor w2_scale =
        w2_fp8_scale_.select(/*dim=*/0, local_expert_id);
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
    torch::Tensor w13 = w13_.select(/*dim=*/0, local_expert_id);
    w1 = w13.slice(/*dim=*/0, /*start=*/0, /*end=*/local_intermediate_size_);
    w3 = w13.slice(/*dim=*/0,
                   /*start=*/local_intermediate_size_,
                   /*end=*/local_intermediate_size_ * 2);
    w2 = w2_.select(/*dim=*/0, local_expert_id);
  }
  torch::Tensor gate =
      torch::matmul(input, w1.transpose(0, 1)).to(torch::kFloat32);
  torch::Tensor up =
      torch::matmul(input, w3.transpose(0, 1)).to(torch::kFloat32);
  torch::Tensor activated =
      swiglu(gate, up, swiglu_limit_).to(input.scalar_type());
  return torch::matmul(activated, w2.transpose(0, 1));
}

torch::Tensor RoutedMoeImpl::forward_grouped_fp8(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights) const {
  if (!fp8_experts_loaded()) {
    return torch::Tensor();
  }
  CHECK_EQ(topk_ids.dim(), 2);
  const int64_t topk = topk_ids.size(1);
  CHECK_EQ(topk_ids.size(0), input.size(0));
  CHECK_EQ(topk_weights.sizes(), topk_ids.sizes());
  CHECK_GT(topk, 0);
  torch::Tensor local_ids = topk_ids.contiguous();
  if (start_expert_id_ != 0 || num_experts_per_rank_ != num_total_experts_) {
    const torch::Tensor valid_ids =
        local_ids.ge(start_expert_id_)
            .logical_and(
                local_ids.lt(start_expert_id_ + num_experts_per_rank_));
    local_ids = torch::where(valid_ids,
                             local_ids - start_expert_id_,
                             torch::full_like(local_ids, -1));
  }

  const auto route_index = kernel::musa::moe_compute_index(
      local_ids.reshape({-1}), num_experts_per_rank_);
  const torch::Tensor dst_to_src = std::get<1>(route_index);
  const torch::Tensor expert_counts = std::get<2>(route_index);

  constexpr int64_t kAlignmentM = 128;
  const int64_t num_assignments = input.size(0) * topk;
  // Every assignment can belong to this rank. In particular, EP dispatch
  // has already gathered the owned rows here; dividing by EP again silently
  // truncates concentrated routes in the padded metadata builder.
  const int64_t route_capacity = num_assignments;
  const int64_t max_active_experts =
      std::min<int64_t>(num_experts_per_rank_, route_capacity);
  const int64_t max_padded_assignments =
      ((route_capacity + max_active_experts * (kAlignmentM - 1) + kAlignmentM -
        1) /
       kAlignmentM) *
      kAlignmentM;

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
                                                      topk,
                                                      kAlignmentM,
                                                      max_padded_assignments);

  torch::Tensor gate_up = kernel::musa::ragged_moe_gemm_fp8(sorted_input_fp8,
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
  torch::Tensor down = kernel::musa::ragged_moe_gemm_fp8(activated_fp8,
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

torch::Tensor RoutedMoeImpl::forward_fp8_gemv(const torch::Tensor& input_2d,
                                              const torch::Tensor& ids_2d,
                                              const torch::Tensor& weights_2d) {
  const int64_t num_tokens = input_2d.size(0);
  const int64_t num_assignments = num_tokens * topk_;
  torch::Tensor local_ids = ids_2d.scalar_type() == torch::kInt32
                                ? ids_2d.contiguous()
                                : ids_2d.to(torch::kInt32).contiguous();
  if (start_expert_id_ != 0 || num_experts_per_rank_ != num_total_experts_) {
    const torch::Tensor valid_ids =
        local_ids.ge(start_expert_id_)
            .logical_and(
                local_ids.lt(start_expert_id_ + num_experts_per_rank_));
    local_ids = torch::where(valid_ids,
                             local_ids - start_expert_id_,
                             torch::full_like(local_ids, -1));
  }
  return kernel::musa::deepseek_v4_fp8_moe_gemv_decode(
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

torch::Tensor RoutedMoeImpl::forward_reference_experts(
    const torch::Tensor& input_2d,
    const torch::Tensor& ids_2d,
    const torch::Tensor& weights_2d) {
  torch::Tensor output = torch::zeros_like(input_2d);
  torch::Tensor ids_cpu = ids_2d.to(torch::kLong).to(torch::kCPU).contiguous();
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
    torch::TensorOptions index_options =
        torch::TensorOptions().dtype(torch::kLong).device(input_2d.device());
    torch::Tensor token_idx =
        torch::tensor(token_indices[local_expert], index_options);
    torch::Tensor topk_idx =
        torch::tensor(topk_indices[local_expert], index_options);
    torch::Tensor expert_input = input_2d.index_select(/*dim=*/0, token_idx);
    torch::Tensor expert_output = apply_expert(expert_input, local_expert);
    torch::Tensor expert_weight = weights_2d.index({token_idx, topk_idx})
                                      .unsqueeze(-1)
                                      .to(output.dtype());
    output.index_add_(/*dim=*/0, token_idx, expert_output * expert_weight);
  }
  return output;
}

torch::Tensor RoutedMoeImpl::forward_dispatched_experts(
    const torch::Tensor& input,
    const torch::Tensor& expert_ids) const {
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(expert_ids.dim(), 1);
  CHECK_EQ(input.size(0), expert_ids.size(0));
  torch::Tensor output = torch::zeros_like(input);
  if (input.size(0) == 0) {
    return output;
  }

  if (fp8_experts_loaded()) {
    // Dispatch has expanded top-k into one expert assignment per row.
    // Preserve that shape: zero-weight dummy routes still consume padded
    // capacity and can displace real routes on the owner of expert zero.
    torch::Tensor grouped_ids = expert_ids.to(torch::kInt64).view({-1, 1});
    torch::Tensor grouped_weights =
        torch::ones({input.size(0), 1}, input.options().dtype(torch::kFloat32));
    torch::Tensor grouped_output =
        forward_grouped_fp8(input, grouped_ids, grouped_weights);
    if (grouped_output.defined()) {
      return grouped_output;
    }
  }

  // The route IDs stay on device.  The fixed expert loop is deliberately
  // conservative for EP1: it avoids copying token identities to host while
  // keeping the dispatch/combine contract independent of a fused MoE kernel.
  const torch::Tensor local_ids =
      expert_ids.to(torch::kInt64).contiguous() - start_expert_id_;
  for (int64_t local_expert = 0; local_expert < num_experts_per_rank_;
       ++local_expert) {
    torch::Tensor token_indices =
        torch::nonzero(local_ids.eq(local_expert)).flatten();
    if (token_indices.numel() == 0) {
      continue;
    }
    torch::Tensor expert_output = apply_expert(
        input.index_select(/*dim=*/0, token_indices), local_expert);
    output.index_copy_(/*dim=*/0, token_indices, expert_output);
  }
  return output;
}

torch::Tensor RoutedMoeImpl::forward_ep_all_to_all(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights) const {
  CHECK(ep_dispatch_enabled_);
  CHECK(ep_pg_ != nullptr);
  const int64_t world_size = ep_pg_->world_size();
  CHECK_GT(world_size, 1);
  const int64_t num_tokens = input.size(0);
  const int64_t num_assignments = num_tokens * topk_;
  const torch::Tensor input_2d = input.reshape({num_tokens, hidden_size_});

  const torch::Tensor ids =
      topk_ids.reshape({num_tokens, topk_}).to(torch::kInt64).contiguous();
  const torch::Tensor weights = topk_weights.reshape({num_tokens, topk_})
                                    .to(torch::kFloat32)
                                    .contiguous();
  CHECK(ids.ge(0).all().to(torch::kCPU).item<bool>())
      << "RoutedMoe EP received a negative expert id.";
  CHECK(ids.lt(num_total_experts_).all().to(torch::kCPU).item<bool>())
      << "RoutedMoe EP received an out-of-range expert id.";

  const torch::Tensor flat_ids = ids.reshape({-1});
  const torch::Tensor flat_weights = weights.reshape({-1});
  const torch::Tensor source_rows =
      torch::arange(num_tokens, input.options().dtype(torch::kInt64))
          .view({num_tokens, 1})
          .expand({num_tokens, topk_})
          .reshape({-1})
          .contiguous();
  const torch::Tensor topk_slots =
      torch::arange(topk_, input.options().dtype(torch::kInt64))
          .view({1, topk_})
          .expand({num_tokens, topk_})
          .reshape({-1})
          .contiguous();
  const torch::Tensor owners = flat_ids.div(num_experts_per_rank_, "floor");
  const torch::Tensor permutation = torch::argsort(owners,
                                                   /*stable=*/true,
                                                   static_cast<int64_t>(0),
                                                   /*descending=*/false);
  const torch::Tensor sorted_owners = owners.index_select(0, permutation);
  const torch::Tensor sorted_input =
      input_2d.index_select(0, source_rows.index_select(0, permutation))
          .contiguous();
  const torch::Tensor sorted_meta =
      torch::stack({flat_ids.index_select(0, permutation),
                    source_rows.index_select(0, permutation),
                    topk_slots.index_select(0, permutation)},
                   /*dim=*/1)
          .contiguous();
  const torch::Tensor sorted_weights =
      flat_weights.index_select(0, permutation).contiguous();

  std::vector<int64_t> send_counts;
  send_counts.reserve(world_size);
  for (int64_t peer = 0; peer < world_size; ++peer) {
    send_counts.emplace_back(
        sorted_owners.eq(peer).sum().to(torch::kCPU).item<int64_t>());
  }

  torch::Tensor send_counts_device =
      torch::tensor(send_counts, input.options().dtype(torch::kInt64));
  torch::Tensor recv_counts_device = torch::empty_like(send_counts_device);
  std::vector<int64_t> unit_splits(world_size, 1);
  ep_pg_->all_to_all_single(
      recv_counts_device, send_counts_device, unit_splits, unit_splits);
  std::vector<int64_t> recv_counts;
  recv_counts.reserve(world_size);
  for (int64_t peer = 0; peer < world_size; ++peer) {
    recv_counts.emplace_back(
        recv_counts_device[peer].to(torch::kCPU).item<int64_t>());
  }
  const int64_t received_assignments =
      std::accumulate(recv_counts.begin(), recv_counts.end(), int64_t{0});

  torch::Tensor received_input =
      torch::empty({received_assignments, hidden_size_}, input_2d.options());
  torch::Tensor received_meta =
      torch::empty({received_assignments, 3}, sorted_meta.options());
  torch::Tensor received_weights =
      torch::empty({received_assignments}, sorted_weights.options());
  ep_pg_->all_to_all_single(
      received_input, sorted_input, recv_counts, send_counts);
  ep_pg_->all_to_all_single(
      received_meta, sorted_meta, recv_counts, send_counts);
  ep_pg_->all_to_all_single(
      received_weights, sorted_weights, recv_counts, send_counts);

  torch::Tensor dispatched_output = forward_dispatched_experts(
      received_input,
      received_meta.select(/*dim=*/1, /*index=*/0).contiguous());
  dispatched_output.mul_(
      received_weights.to(dispatched_output.dtype()).view({-1, 1}));

  torch::Tensor combined_output = torch::empty_like(sorted_input);
  torch::Tensor combined_rows =
      torch::empty({num_assignments}, sorted_meta.options());
  ep_pg_->all_to_all_single(
      combined_output, dispatched_output, send_counts, recv_counts);
  ep_pg_->all_to_all_single(
      combined_rows,
      received_meta.select(/*dim=*/1, /*index=*/1).contiguous(),
      send_counts,
      recv_counts);

  torch::Tensor output = torch::zeros_like(input_2d);
  output.index_add_(/*dim=*/0, combined_rows.to(torch::kLong), combined_output);
  return output;
}

torch::Tensor RoutedMoeImpl::forward_generic_experts(
    const torch::Tensor& input_2d,
    const torch::Tensor& ids_2d,
    torch::Tensor weights_2d,
    const BatchForwardType& batch_type) {
  const bool use_fp8_gemv =
      !ep_dispatch_enabled_ && batch_type.is_decode() && input_2d.size(0) > 0 &&
      input_2d.size(0) <= kFp8GemvWorkspaceTokens && fp8_experts_loaded();
  if (!use_fp8_gemv && weights_2d.scalar_type() != input_2d.scalar_type()) {
    weights_2d = weights_2d.to(input_2d.dtype());
  }
  torch::Tensor output;
  if (ep_dispatch_enabled_) {
    CHECK(!kernel::musa::is_stream_capturing())
        << "Variable-split EP all-to-all requires eager execution";
    output = forward_ep_all_to_all(input_2d, ids_2d, weights_2d);
  } else if (use_fp8_gemv) {
    output = forward_fp8_gemv(input_2d, ids_2d, weights_2d);
  }

  if (!output.defined() && !batch_type.is_decode() && fp8_experts_loaded()) {
    output = forward_grouped_fp8(input_2d.contiguous(), ids_2d, weights_2d);
  }

  if (!output.defined()) {
    output = forward_reference_experts(input_2d, ids_2d, weights_2d);
  }

  return output;
}

torch::Tensor RoutedMoeImpl::forward_with_selected_experts(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const ModelInputParams& input_params,
    bool skip_tp_allreduce,
    const torch::Tensor& output_buffer,
    const RoutedMoeTokenShard* token_shard) {
  const bool is_prefill = !input_params.meta.batch_forward_type.is_decode();
  const bool use_prefill_cp = prefill_cp_enabled_ && is_prefill;

  if (token_shard != nullptr) {
    CHECK(use_prefill_cp);
    CHECK(input_params.meta.batch_forward_type.is_prefill() ||
          input_params.meta.batch_forward_type.is_chunked_prefill());
    CHECK_EQ(token_shard->cp_rank, prefill_cp_rank_);
    CHECK_EQ(token_shard->cp_size, prefill_cp_size_);
    for (const auto& rows : {token_shard->global_rows,
                             token_shard->request_rows,
                             token_shard->positions}) {
      CHECK_EQ(rows.sizes(), torch::IntArrayRef({hidden_states.size(0)}));
      CHECK_EQ(rows.device(), hidden_states.device());
    }
  }
  CHECK((w13_is_loaded_ && w2_is_loaded_) || fp4_experts_loaded() ||
        fp8_experts_loaded())
      << "RoutedMoe expert weights must be loaded before forward.";

  torch::Tensor input = hidden_states;
  torch::Tensor input_2d = input.reshape({-1, input.size(-1)});
  torch::Tensor weights_2d = topk_weights.reshape({-1, topk_});
  torch::Tensor ids_2d = topk_ids.reshape({-1, topk_});
  if (use_prefill_cp && input_2d.size(0) == 0 && !ep_dispatch_enabled_) {
    if (output_buffer.defined()) {
      output_buffer.zero_();
      return output_buffer.narrow(0, 0, 0).reshape(hidden_states.sizes());
    }
    return torch::empty_like(hidden_states);
  }
  // A CP-local shared expert needs no collective. Join it after EP combine.
  const bool use_shared_expert_overlap = is_prefill && shared_experts_ &&
                                         !use_prefill_cp &&
                                         input_2d.device().is_privateuseone() &&
                                         !kernel::musa::is_stream_capturing();
  torch::Tensor shared_output;
  SharedExpertOverlapState* shared_overlap_state = nullptr;
  if (use_prefill_cp && mega_moe_ && shared_experts_) {
    shared_output = shared_experts_(input);
    if (shared_expert_gate_ && shared_expert_gate_is_loaded_) {
      shared_output.mul_(torch::sigmoid(shared_expert_gate_->forward(input)));
    }
  } else if (use_shared_expert_overlap) {
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
        torch::Tensor gate =
            torch::sigmoid(shared_expert_gate_->forward(input));
        shared_output.mul_(gate);
      }
    }
    CHECK_EQ(musaEventRecord(shared_overlap_state->shared_ready,
                             shared_overlap_state->stream.stream()),
             musaSuccess);
  }

  CHECK_EQ(weights_2d.size(0), input_2d.size(0))
      << "RoutedMoe topk_weights token count mismatch.";
  CHECK_EQ(ids_2d.size(0), input_2d.size(0))
      << "RoutedMoe topk_ids token count mismatch.";

  torch::Tensor output;
  if (mega_moe_ && ep_dispatch_enabled_) {
    CHECK(fp8_experts_loaded());
    int64_t max_rows = input_2d.size(0);
    if (use_prefill_cp && input_params.attn_metadata &&
        input_params.attn_metadata->glm5_prefill_cp_enabled) {
      const int64_t total =
          input_params.attn_metadata->glm5_prefill_global_rows;
      max_rows = (total + prefill_cp_size_ - 1) / prefill_cp_size_;
    }
    output = mega_moe_->forward_local_shard(input_2d,
                                            ids_2d,
                                            weights_2d,
                                            w13_fp8_,
                                            w13_fp8_scale_,
                                            w2_fp8_,
                                            w2_fp8_scale_,
                                            swiglu_limit_,
                                            max_rows,
                                            is_prefill,
                                            output_buffer);
  } else if (mega_moe_) {
    CHECK(fp8_experts_loaded());
    CHECK(!skip_tp_allreduce);
    output = mega_moe_->forward(
        input_2d,
        ids_2d,
        weights_2d,
        w13_fp8_,
        w13_fp8_scale_,
        w2_fp8_,
        w2_fp8_scale_,
        swiglu_limit_,
        input_2d.size(0) <= kFp8GemvWorkspaceTokens
            ? fp8_gemv_output_workspace_.narrow(0, 0, input_2d.size(0))
            : torch::Tensor());
  } else {
    output = forward_generic_experts(
        input_2d, ids_2d, weights_2d, input_params.meta.batch_forward_type);
  }

  if (shared_experts_ && !shared_output.defined()) {
    torch::Tensor serial_shared_output = shared_experts_(input);
    if (shared_expert_gate_ && shared_expert_gate_is_loaded_) {
      torch::Tensor serial_shared_output_2d =
          serial_shared_output.reshape({-1, hidden_size_}).contiguous();
      const torch::Tensor gate_weight = shared_expert_gate_->weight;
      const auto is_aligned = [](const torch::Tensor& tensor) {
        return tensor.defined() &&
               reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0;
      };
      const bool use_fused_shared_gate =
          input_2d.is_contiguous() && serial_shared_output_2d.is_contiguous() &&
          output.is_contiguous() &&
          (input_2d.scalar_type() == torch::kBFloat16 ||
           input_2d.scalar_type() == torch::kFloat16) &&
          gate_weight.scalar_type() == input_2d.scalar_type() &&
          serial_shared_output_2d.scalar_type() == input_2d.scalar_type() &&
          output.scalar_type() == input_2d.scalar_type() &&
          is_aligned(input_2d) && is_aligned(gate_weight) &&
          is_aligned(serial_shared_output_2d) && is_aligned(output);
      if (use_fused_shared_gate) {
        output = kernel::musa::deepseek_v4_shared_expert_gate(
            input_2d, gate_weight, serial_shared_output_2d, output);
      } else {
        torch::Tensor gate =
            torch::sigmoid(shared_expert_gate_->forward(input));
        serial_shared_output.mul_(gate);
        output.add_(serial_shared_output.reshape_as(output));
      }
    } else {
      output.add_(serial_shared_output.reshape_as(output));
    }
  }

  if (shared_overlap_state != nullptr) {
    const c10::musa::MUSAStream current_stream =
        c10::musa::getCurrentMUSAStream(input_2d.device().index());
    CHECK_EQ(musaStreamWaitEvent(current_stream.stream(),
                                 shared_overlap_state->shared_ready,
                                 /*flags=*/0),
             musaSuccess);
  }
  if (shared_output.defined()) {
    output.add_(shared_output.reshape_as(output));
  }

  if (output_buffer.defined()) {
    CHECK(output_buffer.dim() == 2 && output_buffer.is_contiguous());
    CHECK_GE(output_buffer.size(0), output.size(0));
    CHECK_EQ(output_buffer.size(1), output.size(1));
    torch::Tensor valid_output = output_buffer.narrow(
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
      }
    }
    if (!used_custom_allreduce) {
      tp_pg_->allreduce(output);
    }
  }

  return output.reshape(hidden_states.sizes());
}

void RoutedMoeImpl::load_experts(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  // Expert ownership and projection tensor sharding are independent.  EP
  // selects the [start_expert_id_, num_experts_per_rank_] range in the
  // constructor; rank/world here must remain the MoE-TP group used to slice
  // each selected expert's projection.  In the EP0 contract this is rank 0 /
  // world 1, so every local expert is loaded in full without communication.
  const int64_t rank = mega_moe_ ? 0 : tp_pg_->rank();
  const int64_t world_size = mega_moe_ ? 1 : tp_pg_->world_size();
  if (ep_dispatch_enabled_) {
    CHECK_EQ(world_size, 1) << "RoutedMoe EP1 expert weights require moe_tp=1";
  }

  if (mega_moe_) {
    // A checkpoint shard can contain only experts owned by another rank.
    // That is not evidence of a dense checkpoint: do not allocate BF16
    // expert storage before this rank's FP8 tensors arrive.
    if (!try_load_fp8_experts(state_dict,
                              {"gate_proj.", "up_proj."},
                              "down_proj.",
                              rank,
                              world_size)) {
      try_load_fp8_experts(state_dict, {"w1.", "w3."}, "w2.", rank, world_size);
    }
    return;
  }

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

  // A nonempty checkpoint shard may contain only other ranks' experts.
  // Allocate dense storage only when a local weight actually arrives;
  // otherwise an FP8 model can retain unused BF16 weights for every layer.
  const auto has_local_weight = [&]() {
    for (int64_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      const std::string prefix = std::to_string(start_expert_id_ + idx) + ".";
      for (const char* name : {"gate_proj.weight",
                               "up_proj.weight",
                               "down_proj.weight",
                               "w1.weight",
                               "w3.weight",
                               "w2.weight"}) {
        if (state_dict.has(prefix + name)) {
          return true;
        }
      }
    }
    return false;
  };
  if (!has_local_weight()) {
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

void RoutedMoeImpl::load_state_dict(const StateDict& state_dict) {
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
      shared_experts_is_loaded_ = true;
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
  if (fp8_experts_loaded() && !ep_dispatch_enabled_ &&
      !fp8_gemv_output_workspace_.defined()) {
    if (!mega_moe_) {
      fp8_gemv_activation_workspace_ = torch::empty(
          {kFp8GemvWorkspaceTokens * topk_, local_intermediate_size_},
          options_);
      fp8_gemv_routed_output_workspace_ = torch::empty(
          {kFp8GemvWorkspaceTokens * topk_, hidden_size_}, options_);
    }
    fp8_gemv_output_workspace_ =
        torch::empty({kFp8GemvWorkspaceTokens, hidden_size_}, options_);
  }
}

void RoutedMoeImpl::verify_loaded_weights(const std::string& prefix) const {
  CHECK((w13_is_loaded_ && w2_is_loaded_) || fp4_experts_loaded() ||
        fp8_experts_loaded())
      << "RoutedMoe expert weights are not fully loaded for "
      << prefix + "experts";
  if (n_shared_experts_ > 0) {
    CHECK(shared_experts_is_loaded_)
        << "RoutedMoe shared_experts weights are not loaded for "
        << prefix + "shared_experts";
  }
}

}  // namespace layer
}  // namespace xllm
