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

#include "layers/musa/qwen3_next_gated_delta_net.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/true) {}

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    bool init_projections)
    : Qwen3GatedDeltaNetBaseImpl(args, quant_args, parallel_args, options) {
  if (init_projections) {
    init_next_projections(args, quant_args, parallel_args, options);
  }
}

void Qwen3NextGatedDeltaNetImpl::init_next_projections(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  // QKVZ projection used by Qwen3-Next linear attention.
  qkvz_proj_ = register_module("in_proj_qkvz",
                               ColumnParallelLinear(args.hidden_size(),
                                                    k_size_ * 2 + v_size_ * 2,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  // BA projection used to derive gating and beta terms.
  ba_proj_ = register_module("in_proj_ba",
                             ColumnParallelLinear(args.hidden_size(),
                                                  num_v_heads_ * 2,
                                                  /*bias=*/false,
                                                  /*gather_output=*/false,
                                                  quant_args,
                                                  parallel_args.tp_group_,
                                                  options));
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3NextGatedDeltaNetImpl::project_decode_inputs(
    const torch::Tensor& hidden_states) {
  auto qkvz = qkvz_proj_->forward(hidden_states);
  auto ba = ba_proj_->forward(hidden_states);
  return {qkvz.view({qkvz.size(0), -1, qkvz.size(-1)}),
          ba.view({ba.size(0), -1, ba.size(-1)})};
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3NextGatedDeltaNetImpl::project_flat_inputs(
    const torch::Tensor& hidden_states) {
  return {qkvz_proj_->forward(hidden_states), ba_proj_->forward(hidden_states)};
}

void Qwen3NextGatedDeltaNetImpl::load_state_dict(const StateDict& state_dict) {
  load_projection_state_dict(state_dict);
  load_common_state_dict(state_dict);
}

void Qwen3NextGatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  auto qkvz_state_dict = state_dict.get_dict_with_prefix("in_proj_qkvz.");
  if (qkvz_state_dict.size() > 0 && !qkvz_proj_->is_weight_loaded()) {
    qkvz_proj_->load_state_dict(qkvz_state_dict);
  }

  auto ba_state_dict = state_dict.get_dict_with_prefix("in_proj_ba.");
  if (ba_state_dict.size() > 0 && !ba_proj_->is_weight_loaded()) {
    ba_proj_->load_state_dict(ba_state_dict);
  }
}

void Qwen3NextGatedDeltaNetImpl::verify_loaded_weights(
    const std::string& prefix) const {
  verify_projection_weights(prefix);
  verify_common_loaded_weights(prefix);
}

void Qwen3NextGatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  CHECK(qkvz_proj_ && qkvz_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkvz.weight";
  CHECK(ba_proj_ && ba_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_ba.weight";
}

Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/false) {
  if (tp_size_ == 1) {
    // TP=1 fast path: use 2 merged projections (qkvz_proj_ + ba_proj_),
    // same as sglang's MergedColumnParallelLinear. The fused weight loading
    // (load_state_dict with prefixes) concatenates the 4 checkpoint weights
    // into 2 merged weights at load time. This halves the number of matmul
    // kernel launches and eliminates the merge copy_ overhead entirely.
    init_next_projections(args, quant_args, parallel_args, options);
  } else {
    // TP>1 fallback: keep 4 separate projections because in_proj_qkv needs
    // 3-shard TP partitioning (q/k/v split independently) which the fused
    // loading path does not support.
    in_proj_qkv_ = register_module("in_proj_qkv",
                                   ColumnParallelLinear(args.hidden_size(),
                                                        k_size_ * 2 + v_size_,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        parallel_args.tp_group_,
                                                        options));
    in_proj_z_ = register_module("in_proj_z",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      v_size_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
    in_proj_b_ = register_module("in_proj_b",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      num_v_heads_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
    in_proj_a_ = register_module("in_proj_a",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      num_v_heads_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
  }
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_qkvz_from_split_activations(
    const torch::Tensor& qkv,
    const torch::Tensor& z) const {
  CHECK_EQ(qkv.dim(), 3) << "Expected qkv activation to be 3D, got "
                         << qkv.sizes();
  CHECK_EQ(z.dim(), 3) << "Expected z activation to be 3D, got " << z.sizes();
  CHECK_EQ(qkv.size(0), z.size(0)) << "qkv/z batch size mismatch.";
  CHECK_EQ(qkv.size(1), z.size(1)) << "qkv/z sequence size mismatch.";
  CHECK_EQ(qkv.size(2), (2 * k_size_ + v_size_) / tp_size_)
      << "Unexpected qkv hidden size for Qwen3.5.";
  CHECK_EQ(z.size(2), v_size_ / tp_size_)
      << "Unexpected z hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = qkv.size(0);
  const int64_t seqlen = qkv.size(1);
  const int64_t qkv_cols = qkv.size(2);
  const int64_t z_cols = z.size(2);

  // Contiguous layout: [all_q | all_k | all_v | all_z]. qkv projection already
  // stores [all_q | all_k | all_v]; append z with two copy_ writes instead of
  // interleaving per head group.
  const int64_t M = bs * seqlen;
  const int64_t flat_dim = qkv_cols + z_cols;
  const bool needs_realloc =
      !qkvz_merge_buf_.defined() || qkvz_merge_buf_.size(0) < M ||
      qkvz_merge_buf_.size(1) != flat_dim ||
      qkvz_merge_buf_.scalar_type() != qkv.scalar_type() ||
      qkvz_merge_buf_.device() != qkv.device();
  if (needs_realloc) {
    const int64_t target_M = qkvz_merge_buf_.defined()
                                 ? std::max(M, qkvz_merge_buf_.size(0))
                                 : M;
    qkvz_merge_buf_ = torch::empty({target_M, flat_dim}, qkv.options());
  }
  auto buf =
      qkvz_merge_buf_.narrow(/*dim=*/0, /*start=*/0, /*length=*/M);
  buf.narrow(/*dim=*/1, /*start=*/0, /*length=*/qkv_cols)
      .copy_(qkv.reshape({M, qkv_cols}));
  buf.narrow(/*dim=*/1, qkv_cols, z_cols)
      .copy_(z.reshape({M, z_cols}));
  return buf.view({bs, seqlen, flat_dim});
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_ba_from_split_activations(
    const torch::Tensor& b,
    const torch::Tensor& a) const {
  CHECK_EQ(b.dim(), 3) << "Expected b activation to be 3D, got " << b.sizes();
  CHECK_EQ(a.dim(), 3) << "Expected a activation to be 3D, got " << a.sizes();
  CHECK_EQ(b.size(0), a.size(0)) << "b/a batch size mismatch.";
  CHECK_EQ(b.size(1), a.size(1)) << "b/a sequence size mismatch.";
  CHECK_EQ(b.size(2), num_v_heads_ / tp_size_)
      << "Unexpected b hidden size for Qwen3.5.";
  CHECK_EQ(a.size(2), num_v_heads_ / tp_size_)
      << "Unexpected a hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = b.size(0);
  const int64_t seqlen = b.size(1);
  const int64_t nv = b.size(2);

  // Contiguous layout: [all_b | all_a].
  const int64_t M = bs * seqlen;
  const int64_t flat_dim = 2 * nv;
  const bool needs_realloc = !ba_merge_buf_.defined() ||
                             ba_merge_buf_.size(0) < M ||
                             ba_merge_buf_.size(1) != flat_dim ||
                             ba_merge_buf_.scalar_type() != b.scalar_type() ||
                             ba_merge_buf_.device() != b.device();
  if (needs_realloc) {
    const int64_t target_M = ba_merge_buf_.defined()
                                 ? std::max(M, ba_merge_buf_.size(0))
                                 : M;
    ba_merge_buf_ = torch::empty({target_M, flat_dim}, b.options());
  }
  auto buf = ba_merge_buf_.narrow(/*dim=*/0, /*start=*/0, /*length=*/M);
  buf.narrow(/*dim=*/1, /*start=*/0, /*length=*/nv)
      .copy_(b.reshape({M, nv}));
  buf.narrow(/*dim=*/1, nv, nv).copy_(a.reshape({M, nv}));
  return buf.view({bs, seqlen, flat_dim});
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_decode_inputs(
    const torch::Tensor& hidden_states) {
  if (use_merged_projections()) {
    return Qwen3NextGatedDeltaNetImpl::project_decode_inputs(hidden_states);
  }
  const auto reshape_projection = [](const torch::Tensor& projection) {
    return projection.view({projection.size(0), -1, projection.size(-1)});
  };
  auto qkv = reshape_projection(in_proj_qkv_->forward(hidden_states));
  auto z_proj = reshape_projection(in_proj_z_->forward(hidden_states));
  auto b_proj = reshape_projection(in_proj_b_->forward(hidden_states));
  auto a_proj = reshape_projection(in_proj_a_->forward(hidden_states));
  return {merge_qkvz_from_split_activations(qkv, z_proj),
          merge_ba_from_split_activations(b_proj, a_proj)};
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_flat_inputs(
    const torch::Tensor& hidden_states) {
  if (use_merged_projections()) {
    return Qwen3NextGatedDeltaNetImpl::project_flat_inputs(hidden_states);
  }
  auto qkv = in_proj_qkv_->forward(hidden_states).unsqueeze(0);
  auto z_proj = in_proj_z_->forward(hidden_states).unsqueeze(0);
  auto b_proj = in_proj_b_->forward(hidden_states).unsqueeze(0);
  auto a_proj = in_proj_a_->forward(hidden_states).unsqueeze(0);
  auto qkvz = merge_qkvz_from_split_activations(qkv, z_proj);
  auto ba = merge_ba_from_split_activations(b_proj, a_proj);
  return {qkvz.view({hidden_states.size(0), qkvz.size(-1)}).contiguous(),
          ba.view({hidden_states.size(0), ba.size(-1)}).contiguous()};
}

void Qwen3_5GatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  if (use_merged_projections()) {
    // TP=1 merged path: fuse the 4 checkpoint weights into 2 merged
    // projections via load_state_dict(prefixes). This concatenates the
    // weights (and FP8 block-scale grids, if present) along dim 0.
    if (!qkvz_proj_->is_weight_loaded()) {
      qkvz_proj_->load_state_dict(
          state_dict,
          /*prefixes=*/{"in_proj_qkv.", "in_proj_z."});
    }
    if (!ba_proj_->is_weight_loaded()) {
      ba_proj_->load_state_dict(
          state_dict,
          /*prefixes=*/{"in_proj_b.", "in_proj_a."});
    }
    return;
  }

  // TP>1 fallback: load 4 separate projections.
  auto in_proj_qkv_state_dict = state_dict.get_dict_with_prefix("in_proj_qkv.");
  if (in_proj_qkv_state_dict.size() > 0 && !in_proj_qkv_->is_weight_loaded()) {
    in_proj_qkv_->load_state_dict(
        in_proj_qkv_state_dict,
        /*shard_tensor_count=*/3,
        /*shard_sizes=*/
        {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_});
  }

  auto in_proj_z_state_dict = state_dict.get_dict_with_prefix("in_proj_z.");
  if (in_proj_z_state_dict.size() > 0 && !in_proj_z_->is_weight_loaded()) {
    in_proj_z_->load_state_dict(in_proj_z_state_dict);
  }

  auto in_proj_b_state_dict = state_dict.get_dict_with_prefix("in_proj_b.");
  if (in_proj_b_state_dict.size() > 0 && !in_proj_b_->is_weight_loaded()) {
    in_proj_b_->load_state_dict(in_proj_b_state_dict);
  }

  auto in_proj_a_state_dict = state_dict.get_dict_with_prefix("in_proj_a.");
  if (in_proj_a_state_dict.size() > 0 && !in_proj_a_->is_weight_loaded()) {
    in_proj_a_->load_state_dict(in_proj_a_state_dict);
  }
}

void Qwen3_5GatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  if (use_merged_projections()) {
    CHECK(qkvz_proj_ && qkvz_proj_->is_weight_loaded())
        << "Missing required weight after all shards loaded: " << prefix
        << "in_proj_qkvz.weight (merged from in_proj_qkv + in_proj_z)";
    CHECK(ba_proj_ && ba_proj_->is_weight_loaded())
        << "Missing required weight after all shards loaded: " << prefix
        << "in_proj_ba.weight (merged from in_proj_b + in_proj_a)";
    return;
  }
  CHECK(in_proj_qkv_ && in_proj_qkv_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkv.weight";
  CHECK(in_proj_z_ && in_proj_z_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_z.weight";
  CHECK(in_proj_b_ && in_proj_b_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_b.weight";
  CHECK(in_proj_a_ && in_proj_a_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_a.weight";
}

}  // namespace layer
}  // namespace xllm
