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

#include "layers/musa/glm5_next/glm5_next_linear_attention.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <cmath>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>

#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "framework/state_dict/utils.h"
#include "kernels/musa/gdn_ops.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

constexpr std::array<const char*, 6> kInputProjectionNames = {
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "b_proj.weight",
    "f_a_proj.weight",
    "g_a_proj.weight"};

void load_column_linear(ColumnParallelLinear& linear,
                        const StateDict& state_dict,
                        const std::string& prefix) {
  StateDict linear_state = state_dict.get_dict_with_prefix(prefix);
  if (linear_state.size() > 0 && !linear->is_weight_loaded()) {
    linear->load_state_dict(linear_state);
  }
}

void load_conv_linear(ColumnParallelLinear& linear,
                      const StateDict& state_dict,
                      const std::string& name) {
  torch::Tensor weight = state_dict.get_tensor(name + ".weight");
  if (weight.defined() && !linear->is_weight_loaded()) {
    CHECK_EQ(weight.dim(), 3) << name << ".weight must be [channels, 1, K]";
    CHECK_EQ(weight.size(1), 1) << name << ".weight middle dimension must be 1";
    linear->load_state_dict(StateDict({{"weight", weight.squeeze(1)}}));
  }
}

torch::Tensor glm5_next_kda_apply_output_gate(
    const torch::Tensor& normalized,
    const torch::Tensor& output_gate) {
  const torch::Tensor gated = normalized.to(torch::kFloat32) *
                              torch::sigmoid(output_gate.to(torch::kFloat32));
  return gated.to(normalized.scalar_type());
}

}  // namespace

Glm5NextLinearAttentionInputProjectionImpl::
    Glm5NextLinearAttentionInputProjectionImpl(
        int64_t hidden_size,
        int64_t num_heads,
        int64_t head_dim,
        int32_t rank,
        int32_t tp_size,
        const torch::TensorOptions& options)
    : rank_(rank), tp_size_(tp_size) {
  CHECK_GT(hidden_size, 0);
  CHECK_GT(num_heads, 0);
  CHECK_GT(head_dim, 0);
  CHECK_GT(tp_size_, 0);
  CHECK_GE(rank_, 0);
  CHECK_LT(rank_, tp_size_);
  CHECK_EQ(num_heads % tp_size_, 0);
  const int64_t local_heads = num_heads / tp_size_;
  const int64_t local_size = local_heads * head_dim;
  shard_sizes_ = {
      local_size, local_size, local_size, local_heads, head_dim, head_dim};
  const int64_t output_size =
      std::accumulate(shard_sizes_.begin(), shard_sizes_.end(), int64_t{0});
  weight_ =
      register_parameter("weight",
                         torch::empty({output_size, hidden_size}, options),
                         /*requires_grad=*/false);
}

void Glm5NextLinearAttentionInputProjectionImpl::load_state_dict(
    const StateDict& state_dict) {
  int64_t offset = 0;
  for (size_t index = 0; index < shard_sizes_.size(); ++index) {
    torch::Tensor destination = weight_.narrow(0, offset, shard_sizes_[index]);
    offset += shard_sizes_[index];
    if (loaded_[index]) {
      continue;
    }
    torch::Tensor source = state_dict.get_tensor(kInputProjectionNames[index]);
    if (!source.defined()) {
      continue;
    }
    // q/k/v/b are partitioned by head; f_a/g_a retain every low-rank channel.
    const int64_t partitions = index < 4 ? tp_size_ : 1;
    CHECK_EQ(source.dim(), 2) << kInputProjectionNames[index];
    CHECK_EQ(source.size(0), shard_sizes_[index] * partitions)
        << kInputProjectionNames[index];
    CHECK_EQ(source.size(1), weight_.size(1)) << kInputProjectionNames[index];
    const int64_t source_offset = index < 4 ? rank_ * shard_sizes_[index] : 0;
    destination.copy_(source.narrow(0, source_offset, shard_sizes_[index]));
    loaded_[index] = true;
  }
}

void Glm5NextLinearAttentionInputProjectionImpl::verify_loaded_weights(
    const std::string& prefix) const {
  for (size_t index = 0; index < loaded_.size(); ++index) {
    CHECK(loaded_[index]) << "Missing required weight after all shards loaded: "
                          << prefix << kInputProjectionNames[index];
  }
}

torch::Tensor Glm5NextLinearAttentionInputProjectionImpl::forward(
    const torch::Tensor& hidden_states) {
  kernel::MatmulParams params;
  params.a = hidden_states;
  params.b = weight_;
  return kernel::matmul(params);
}

Glm5NextLinearAttentionImpl::Glm5NextLinearAttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    float gate_lower_bound)
    : gate_lower_bound_(gate_lower_bound) {
  (void)quant_args;
  CHECK_EQ(options.dtype().toScalarType(), torch::kBFloat16);
  CHECK_EQ(parallel_args.cp_size(), 1) << "KDA requires CP=1";
  CHECK(parallel_args.tp_group_ != nullptr);
  tp_size_ = parallel_args.tp_group_->world_size();
  rank_ = parallel_args.tp_group_->rank();
  num_heads_ = args.linear_num_key_heads();
  head_dim_ = args.linear_key_head_dim();
  conv_kernel_size_ = args.linear_conv_kernel_dim();

  CHECK_GT(num_heads_, 0) << "GLM5-Next KDA requires positive head count";
  CHECK_EQ(num_heads_, args.linear_num_value_heads())
      << "GLM5-Next KDA currently requires equal q/k/v head counts";
  CHECK_EQ(head_dim_, args.linear_value_head_dim())
      << "GLM5-Next KDA currently requires equal key/value head dimensions";
  CHECK_EQ(num_heads_ % tp_size_, 0)
      << "GLM5-Next KDA head count must be divisible by TP size";
  CHECK_EQ(conv_kernel_size_, 4)
      << "GLM5-Next KDA requires positive causal-conv width";
  CHECK_LT(gate_lower_bound_, 0.0f)
      << "GLM5-Next KDA gate lower bound must be negative";

  local_num_heads_ = num_heads_ / tp_size_;
  projection_size_ = num_heads_ * head_dim_;
  local_projection_size_ = local_num_heads_ * head_dim_;
  const QuantArgs no_quant_args{};

  in_proj_qkvbfg_a_ = register_module(
      "in_proj_qkvbfg_a",
      Glm5NextLinearAttentionInputProjection(args.hidden_size(),
                                             num_heads_,
                                             head_dim_,
                                             static_cast<int32_t>(rank_),
                                             static_cast<int32_t>(tp_size_),
                                             options));
  f_b_proj_ = register_module("f_b_proj",
                              ColumnParallelLinear(head_dim_,
                                                   projection_size_,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  g_b_proj_ = register_module("g_b_proj",
                              ColumnParallelLinear(head_dim_,
                                                   projection_size_,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  // The existing MUSA convolution wrapper requires BF16 weights and state.
  // Convert checkpoint FP32 convolution weights when loading.
  q_conv1d_ = register_module("q_conv1d",
                              ColumnParallelLinear(conv_kernel_size_,
                                                   projection_size_,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  k_conv1d_ = register_module("k_conv1d",
                              ColumnParallelLinear(conv_kernel_size_,
                                                   projection_size_,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  v_conv1d_ = register_module("v_conv1d",
                              ColumnParallelLinear(conv_kernel_size_,
                                                   projection_size_,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  const torch::TensorOptions fp32_options = options.dtype(torch::kFloat32);
  dt_bias_ =
      register_parameter("dt_bias",
                         torch::empty({local_projection_size_}, fp32_options),
                         /*requires_grad=*/false);
  A_log_ = register_parameter("A_log",
                              torch::empty({local_num_heads_}, fp32_options),
                              /*requires_grad=*/false);

  o_norm_ =
      register_module("o_norm",
                      musa::RmsNormGated(head_dim_,
                                         /*max_rows=*/256 * local_num_heads_,
                                         args.rms_norm_eps(),
                                         options));
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(projection_size_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              no_quant_args,
                                              parallel_args.tp_group_,
                                              options));
}

Glm5NextLinearAttentionImpl::Glm5NextLinearAttentionImpl(
    const ModelContext& context)
    : Glm5NextLinearAttentionImpl(
          context.get_model_args(),
          context.get_quant_args(),
          context.get_parallel_args(),
          context.get_tensor_options(),
          context.get_model_args().linear_lower_bound()) {}

void Glm5NextLinearAttentionImpl::load_state_dict(const StateDict& state_dict) {
  for (const auto& entry : state_dict) {
    const auto dtype = entry.second.scalar_type();
    CHECK(dtype == torch::kBFloat16 || dtype == torch::kFloat32)
        << "KDA weights must be BF16 or FP32: " << entry.first;
  }
  in_proj_qkvbfg_a_->load_state_dict(state_dict);
  load_column_linear(f_b_proj_, state_dict, "f_b_proj.");
  load_column_linear(g_b_proj_, state_dict, "g_b_proj.");

  load_conv_linear(q_conv1d_, state_dict, "q_conv1d");
  load_conv_linear(k_conv1d_, state_dict, "k_conv1d");
  load_conv_linear(v_conv1d_, state_dict, "v_conv1d");
  if (!conv_weight_.defined() && q_conv1d_->is_weight_loaded() &&
      k_conv1d_->is_weight_loaded() && v_conv1d_->is_weight_loaded()) {
    conv_weight_ =
        torch::cat(
            {q_conv1d_->weight(), k_conv1d_->weight(), v_conv1d_->weight()}, 0)
            .contiguous();
  }

  StateDict o_norm_state = state_dict.get_dict_with_prefix("o_norm.");
  if (o_norm_state.size() > 0 && !o_norm_is_loaded_) {
    o_norm_->load_state_dict(o_norm_state);
    o_norm_is_loaded_ = o_norm_state.get_tensor("weight").defined();
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));

  weight::load_sharded_weight(state_dict,
                              "dt_bias",
                              /*dim=*/0,
                              static_cast<int32_t>(rank_),
                              static_cast<int32_t>(tp_size_),
                              dt_bias_,
                              dt_bias_is_loaded_);
  torch::Tensor a_log = state_dict.get_tensor("A_log");
  if (a_log.defined()) {
    // HF KDA checkpoints exist in both [H] and [1,1,H,1] forms. Normalize the
    // latter before applying the head shard so the runtime/kernel contract is
    // always the compact FP32 [H_local] form.
    CHECK(a_log.dim() == 1 || (a_log.dim() == 4 && a_log.size(0) == 1 &&
                               a_log.size(1) == 1 && a_log.size(3) == 1))
        << "GLM5-Next KDA A_log must be [H] or [1, 1, H, 1]";
    StateDict a_log_state =
        a_log.dim() == 1
            ? state_dict
            : StateDict({{"A_log", a_log.reshape({a_log.numel()})}});
    weight::load_sharded_weight(a_log_state,
                                "A_log",
                                /*dim=*/0,
                                static_cast<int32_t>(rank_),
                                static_cast<int32_t>(tp_size_),
                                A_log_,
                                A_log_is_loaded_);
  }
}

void Glm5NextLinearAttentionImpl::verify_loaded_weights(
    const std::string& prefix) const {
  in_proj_qkvbfg_a_->verify_loaded_weights(prefix);
  CHECK(f_b_proj_ && f_b_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "f_b_proj.weight";
  CHECK(g_b_proj_ && g_b_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "g_b_proj.weight";
  CHECK(q_conv1d_ && q_conv1d_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "q_conv1d.weight";
  CHECK(k_conv1d_ && k_conv1d_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "k_conv1d.weight";
  CHECK(v_conv1d_ && v_conv1d_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "v_conv1d.weight";
  CHECK(o_norm_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "o_norm.weight";
  CHECK(o_proj_ && o_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "o_proj.weight";
  CHECK(dt_bias_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight after all shards loaded: "
                          << prefix << "A_log";
}

torch::Tensor Glm5NextLinearAttentionImpl::get_linear_state_indices(
    const ModelInputParams& input_params,
    const torch::Device& device) const {
  if (input_params.embedding.linear_state_indices.defined()) {
    const torch::Tensor& indices = input_params.embedding.linear_state_indices;
    if (indices.device() == device && indices.scalar_type() == torch::kInt &&
        indices.is_contiguous()) {
      return indices;
    }
    return indices
        .to(torch::TensorOptions().dtype(torch::kInt).device(device),
            /*non_blocking=*/true,
            /*copy=*/true)
        .contiguous();
  }
  CHECK(!input_params.embedding.linear_state_ids.empty());
  return torch::tensor(
      input_params.embedding.linear_state_ids,
      torch::TensorOptions().dtype(torch::kInt).device(device));
}

std::shared_ptr<const MusaKdaBatchMetadata> prepare_musa_kda_batch(
    const torch::Tensor& indices,
    const torch::Tensor& cu_seqlens,
    int64_t tokens,
    int64_t state_capacity) {
  CHECK_EQ(indices.dim(), 1);
  CHECK_EQ(indices.scalar_type(), torch::kInt32);
  CHECK_EQ(cu_seqlens.dim(), 1);
  CHECK_EQ(cu_seqlens.scalar_type(), torch::kInt32);
  CHECK_EQ(indices.numel() + 1, cu_seqlens.numel());
  CHECK_EQ(indices.device(), cu_seqlens.device());
  // Transfer only metadata, once, before any layer queues device work.
  const torch::Tensor ids_host = indices.cpu().contiguous();
  const torch::Tensor cu_host = cu_seqlens.cpu().contiguous();
  const int32_t* ids = ids_host.data_ptr<int32_t>();
  const int32_t* cu = cu_host.data_ptr<int32_t>();
  CHECK_EQ(cu[0], 0);
  CHECK_EQ(cu[indices.numel()], tokens);
  std::unordered_set<int32_t> unique;
  for (int64_t i = 0; i < indices.numel(); ++i) {
    CHECK_GE(ids[i], 0);
    CHECK_LT(ids[i], state_capacity);
    CHECK(unique.insert(ids[i]).second) << "KDA write slots must be unique";
    CHECK_GT(cu[i + 1], cu[i]);
  }
  auto result = std::make_shared<MusaKdaBatchMetadata>();
  result->state_indices = indices.contiguous();
  result->state_indices_long = indices.to(torch::kLong).contiguous();
  result->state_capacity = state_capacity;
  result->num_tokens = tokens;
  return result;
}

int64_t Glm5NextLinearAttentionImpl::get_checkpoint_stride(
    const KVCache& kv_cache) const {
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  CHECK_GT(conv_cache.size(0), 0) << "GLM5-Next KDA conv cache must have rows";
  CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
      << "GLM5-Next KDA SSM checkpoint layout mismatch";
  return ssm_cache.size(0) / conv_cache.size(0);
}

torch::Tensor Glm5NextLinearAttentionImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  CHECK(hidden_states.device().is_privateuseone());
  c10::DeviceGuard device_guard(hidden_states.device());
  CHECK(!kernel::musa::is_stream_capturing())
      << "KDA layer requires eager execution";
  CHECK(!attn_metadata.is_spec_verify && !input_params.is_spec_verify &&
        !attn_metadata.is_mixed)
      << "KDA does not support speculative or mixed batches";
  CHECK(!attn_metadata.linear_state_read_indices.defined() &&
        (input_params.embedding.linear_state_read_ids.empty() ||
         input_params.embedding.linear_state_read_ids ==
             input_params.embedding.linear_state_ids))
      << "KDA requires restored in-place state slots";
  CHECK(!attn_metadata.is_dummy)
      << "GLM5-Next KDA does not accept dummy batches";
  CHECK_EQ(hidden_states.dim(), 2)
      << "GLM5-Next KDA expects flattened [tokens, hidden] input";
  CHECK_EQ(hidden_states.scalar_type(), torch::kBFloat16);
  CHECK_EQ(head_dim_, 128) << "MUSA GLM5-Next KDA requires head_dim=128";
  const int64_t num_tokens = hidden_states.size(0);
  torch::Tensor projected = in_proj_qkvbfg_a_->forward(hidden_states);
  const auto projections = projected.split_with_sizes(
      {3 * local_projection_size_, local_num_heads_, head_dim_, head_dim_},
      /*dim=*/-1);
  torch::Tensor mixed_qkv = projections[0].contiguous();
  torch::Tensor beta = projections[1].contiguous();
  torch::Tensor raw_gate = f_b_proj_->forward(projections[2].contiguous())
                               .view({num_tokens, local_num_heads_, head_dim_})
                               .contiguous();
  torch::Tensor output_gate =
      g_b_proj_->forward(projections[3].contiguous())
          .view({num_tokens, local_num_heads_, head_dim_})
          .contiguous();

  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  CHECK(conv_cache.defined() && ssm_cache.defined())
      << "GLM5-Next KDA requires convolution and recurrent state caches";
  CHECK(attn_metadata.q_cu_seq_lens.defined())
      << "GLM5-Next KDA requires q_cu_seq_lens";
  const torch::Tensor cu_seqlens = attn_metadata.q_cu_seq_lens.contiguous();
  CHECK_GE(cu_seqlens.numel(), 1)
      << "GLM5-Next KDA q_cu_seq_lens cannot be empty";
  CHECK_EQ(cu_seqlens.dim(), 1);
  CHECK_EQ(cu_seqlens.device(), hidden_states.device());
  CHECK_EQ(cu_seqlens.scalar_type(), torch::kInt32);
  CHECK_EQ(conv_cache.dim(), 3);
  CHECK_EQ(conv_cache.size(1), 3 * local_projection_size_);
  CHECK_EQ(conv_cache.size(2), conv_kernel_size_ - 1);
  CHECK_EQ(ssm_cache.dim(), 4);
  CHECK_EQ(ssm_cache.size(1), local_num_heads_);
  CHECK_EQ(ssm_cache.size(2), head_dim_);
  CHECK_EQ(ssm_cache.size(3), head_dim_);
  CHECK_EQ(ssm_cache.scalar_type(), torch::kFloat32);
  CHECK_EQ(conv_cache.scalar_type(), hidden_states.scalar_type());
  CHECK_EQ(conv_cache.device(), hidden_states.device());
  CHECK_EQ(ssm_cache.device(), hidden_states.device());
  const torch::Tensor logical_state_indices =
      attn_metadata.musa_kda_batch
          ? attn_metadata.musa_kda_batch->state_indices
          : get_linear_state_indices(input_params, mixed_qkv.device());
  CHECK_EQ(logical_state_indices.dim(), 1);
  CHECK_EQ(logical_state_indices.numel(), cu_seqlens.numel() - 1)
      << "linear state count must equal attention batch";
  const auto batch = attn_metadata.musa_kda_batch
                         ? attn_metadata.musa_kda_batch
                         : prepare_musa_kda_batch(logical_state_indices,
                                                  cu_seqlens,
                                                  num_tokens,
                                                  conv_cache.size(0));
  CHECK_EQ(batch->num_tokens, num_tokens);
  CHECK_LE(batch->state_capacity, conv_cache.size(0));
  CHECK_EQ(batch->state_indices.device(), hidden_states.device());
  const int64_t checkpoint_stride = get_checkpoint_stride(kv_cache);
  CHECK_EQ(checkpoint_stride, 1)
      << "MUSA GLM5-Next KDA currently requires checkpoint_stride=1";
  const torch::Tensor& state_indices = batch->state_indices;
  if (!conv_weight_.defined()) {
    conv_weight_ =
        torch::cat(
            {q_conv1d_->weight(), k_conv1d_->weight(), v_conv1d_->weight()}, 0)
            .contiguous();
  }
  const torch::Tensor& conv_weight = conv_weight_;
  const bool is_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  if (is_prefill) {
    CHECK(attn_metadata.has_initial_states.defined())
        << "GLM5-Next KDA prefill requires has_initial_states";
    CHECK_EQ(attn_metadata.has_initial_states.scalar_type(), torch::kBool);
    CHECK_EQ(attn_metadata.has_initial_states.device(), hidden_states.device());
    CHECK_EQ(attn_metadata.has_initial_states.numel(),
             logical_state_indices.numel());
    mixed_qkv = xllm::kernel::musa::causal_conv1d_prefill(
        mixed_qkv,
        conv_weight,
        conv_cache,
        std::nullopt,
        cu_seqlens,
        state_indices,
        attn_metadata.has_initial_states,
        /*silu_activation=*/true);
  } else {
    xllm::kernel::CausalConv1dUpdateParams conv_params;
    conv_params.x = mixed_qkv;
    conv_params.conv_state = conv_cache;
    conv_params.weight = conv_weight;
    conv_params.conv_state_indices = state_indices;
    conv_params.query_start_loc = cu_seqlens;
    conv_params.max_query_len = attn_metadata.max_query_len;
    mixed_qkv = xllm::kernel::musa::causal_conv1d_update(conv_params);
  }

  auto qkv = mixed_qkv.split(
      {local_projection_size_, local_projection_size_, local_projection_size_},
      /*dim=*/-1);
  torch::Tensor q = qkv[0].view({1, num_tokens, local_num_heads_, head_dim_});
  torch::Tensor k = qkv[1].view({1, num_tokens, local_num_heads_, head_dim_});
  torch::Tensor v = qkv[2].view({1, num_tokens, local_num_heads_, head_dim_});
  torch::Tensor g = raw_gate.view({1, num_tokens, local_num_heads_, head_dim_});
  beta = beta.view({1, num_tokens, local_num_heads_});

  torch::Tensor initial_state =
      ssm_cache.index_select(0, batch->state_indices_long).contiguous();
  if (is_prefill) {
    const torch::Tensor valid =
        attn_metadata.has_initial_states.view({-1, 1, 1, 1});
    initial_state =
        torch::where(valid, initial_state, torch::zeros_like(initial_state));
  }
  torch::Tensor final_state = torch::empty_like(initial_state);
  torch::Tensor core_output = torch::empty_like(v);
  xllm::kernel::musa::flash_kda(q,
                                k,
                                v,
                                g,
                                beta,
                                1.0 / std::sqrt(128.0),
                                core_output,
                                A_log_,
                                dt_bias_.view({local_num_heads_, head_dim_}),
                                gate_lower_bound_,
                                initial_state,
                                final_state,
                                cu_seqlens);
  ssm_cache.index_copy_(
      0, batch->state_indices_long, final_state.to(ssm_cache.scalar_type()));

  core_output = core_output.squeeze(/*dim=*/0)
                    .reshape({-1, head_dim_})
                    .to(torch::kFloat32);
  output_gate = output_gate.reshape({-1, head_dim_});
  // RmsNormGated is the shared SiLU-gated primitive. GLM-5 uses a sigmoid
  // gate, so apply it after normalization to preserve vLLM's
  // FusedRMSNormGated(activation="sigmoid") semantics.
  torch::Tensor normalized = o_norm_->forward(core_output);
  normalized = glm5_next_kda_apply_output_gate(normalized, output_gate);
  return o_proj_->forward(normalized.to(hidden_states.scalar_type())
                              .view({num_tokens, local_projection_size_}));
}

}  // namespace layer
}  // namespace xllm
