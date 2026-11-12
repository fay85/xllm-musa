/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"
#include "core/layers/musa/deepseek_v41/rms_norm.h"

namespace xllm::layer {

// Native V4.1 query/SWA-KV preparation and grouped output projection. The
// decoder/backend owns cache writes, compressor/indexer, and TP all-reduce.
// QDQ retains original FP8 block32 semantics while GEMMs remain BF16.
class DeepseekV41AttentionProjection final {
 public:
  // Weights: wq_a, q_norm, wq_b, wkv, kv_norm, grouped_wo_a, wo_b.
  DeepseekV41AttentionProjection(std::vector<torch::Tensor> weights,
                                 int64_t heads,
                                 int64_t groups,
                                 int64_t output_rank,
                                 double epsilon,
                                 bool original_fp8_activation)
      : weights_(check_weights(std::move(weights))),
        q_a_(weights_[0], original_fp8_activation),
        q_b_(weights_[2], original_fp8_activation),
        kv_(weights_[3], original_fp8_activation),
        o_b_(weights_[6], original_fp8_activation),
        heads_(heads),
        groups_(groups),
        output_rank_(output_rank),
        epsilon_(epsilon),
        original_fp8_activation_(original_fp8_activation) {
    CHECK_GT(heads_, 0);
    CHECK_GT(groups_, 0);
    CHECK_EQ(heads_ % groups_, 0);
    CHECK_GT(output_rank_, 0);
    CHECK(std::isfinite(epsilon_) && epsilon_ > 0.0);
    CHECK_EQ(weights_[0].dim(), 2);
    CHECK_EQ(weights_[1].dim(), 1);
    CHECK_EQ(weights_[1].size(0), weights_[0].size(0));
    CHECK_EQ(weights_[2].dim(), 2);
    CHECK_EQ(weights_[2].size(1), weights_[0].size(0));
    CHECK_EQ(weights_[3].dim(), 2);
    CHECK_EQ(weights_[3].size(1), weights_[0].size(1));
    head_dim_ = weights_[3].size(0);
    CHECK_GT(head_dim_, 0);
    CHECK_EQ(weights_[2].size(0), heads_ * head_dim_);
    CHECK_EQ(weights_[4].dim(), 1);
    CHECK_EQ(weights_[4].size(0), head_dim_);
    CHECK_EQ(weights_[5].sizes(),
             torch::IntArrayRef(
                 {groups_ * output_rank_, heads_ * head_dim_ / groups_}));
    CHECK_EQ(weights_[6].sizes(),
             torch::IntArrayRef({weights_[0].size(1), groups_ * output_rank_}));
    CHECK(!original_fp8_activation_ || head_dim_ % 32 == 0);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prepare(
      const torch::Tensor& hidden,
      const torch::Tensor& positions,
      const torch::Tensor& table) const {
    check_hidden(hidden);
    check_rotary(positions, table, hidden.size(0));
    torch::DeviceGuard guard(hidden.device());
    const torch::Tensor qr = norm(q_a_.forward(hidden), weights_[1]);
    const torch::Tensor key = norm(kv_.forward(hidden), weights_[4]);
    const torch::Tensor query =
        q_b_.forward(qr).view({hidden.size(0), heads_, head_dim_});
    return {rope(query, positions, table, /*inverse=*/false),
            quantize(rope(key, positions, table, /*inverse=*/false)),
            qr};
  }

  torch::Tensor project_output(const torch::Tensor& attention,
                               const torch::Tensor& positions,
                               const torch::Tensor& table) const {
    CHECK(attention.defined());
    CHECK_EQ(attention.device(), weights_[0].device());
    CHECK_EQ(attention.scalar_type(), torch::kBFloat16);
    CHECK_EQ(attention.dim(), 3);
    CHECK_EQ(attention.size(1), heads_);
    CHECK_EQ(attention.size(2), head_dim_);
    check_rotary(positions, table, attention.size(0));
    torch::DeviceGuard guard(attention.device());
    if (attention.size(0) == 0) {
      return torch::empty({0, weights_[0].size(1)}, attention.options());
    }
    const torch::Tensor value =
        quantize(rope(attention, positions, table, /*inverse=*/true))
            .view({attention.size(0), groups_, heads_ * head_dim_ / groups_});
    const torch::Tensor grouped_weight =
        weights_[5].view({groups_, output_rank_, heads_ * head_dim_ / groups_});
    torch::Tensor low;
    if (value.size(0) == 1 && groups_ == 1 && value.size(2) == 4096 &&
        output_rank_ == 1024) {
      // Same BF16 carrier and shape as the delivered SGLang split-K GEMV.
      // Activation QDQ above remains in place; select before any execution.
      low = kernel::musa::deepseek_v4_wo_a_m1(
          value.contiguous(), grouped_weight, torch::Tensor{});
      CHECK(low.defined()) << "Validated V4.1 wo_a M1 contract was rejected";
    } else {
      low = torch::einsum("tgd,grd->tgr", {value, grouped_weight});
    }
    return o_b_.forward(
        low.reshape({attention.size(0), groups_ * output_rank_}));
  }

 private:
  static std::vector<torch::Tensor> check_weights(
      std::vector<torch::Tensor> weights) {
    CHECK_EQ(weights.size(), 7);
    CHECK(weights[0].defined());
    CHECK(weights[0].device().type() == torch::kPrivateUse1);
    for (const torch::Tensor& weight : weights) {
      CHECK(weight.defined());
      CHECK_EQ(weight.device(), weights[0].device());
      CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
      CHECK(weight.is_contiguous());
      CHECK_GT(weight.numel(), 0);
    }
    return weights;
  }

  void check_hidden(const torch::Tensor& hidden) const {
    CHECK(hidden.defined());
    CHECK_EQ(hidden.device(), weights_[0].device());
    CHECK_EQ(hidden.scalar_type(), torch::kBFloat16);
    CHECK_EQ(hidden.dim(), 2);
    CHECK_EQ(hidden.size(1), weights_[0].size(1));
  }

  void check_rotary(const torch::Tensor& positions,
                    const torch::Tensor& table,
                    int64_t rows) const {
    CHECK(positions.defined() && table.defined());
    CHECK_EQ(positions.device(), weights_[0].device());
    CHECK_EQ(table.device(), weights_[0].device());
    CHECK(positions.scalar_type() == torch::kInt64 ||
          positions.scalar_type() == torch::kInt32);
    CHECK_EQ(positions.dim(), 1);
    CHECK_EQ(positions.numel(), rows);
    CHECK_EQ(table.dim(), 2);
    CHECK_EQ(table.scalar_type(), torch::kFloat32);
    CHECK_GT(table.size(1), 0);
    CHECK_EQ(table.size(1) % 2, 0);
    CHECK_LE(table.size(1), head_dim_);
  }

  torch::Tensor norm(const torch::Tensor& input,
                     const torch::Tensor& weight) const {
    return deepseek_v41_rms_norm(input, weight, epsilon_);
  }

  torch::Tensor quantize(const torch::Tensor& input) const {
    if (!original_fp8_activation_) {
      return input;
    }
    return std::get<0>(kernel::musa::deepseek_v41_block32_qdq(
                           input.reshape({-1, input.size(-1)}).contiguous()))
        .reshape(input.sizes());
  }

  torch::Tensor rope(const torch::Tensor& input,
                     const torch::Tensor& positions,
                     const torch::Tensor& table,
                     bool inverse) const {
    if (input.size(0) == 0) {
      return input.clone();
    }
    const int64_t width = table.size(1);
    const torch::Tensor rows =
        table.index_select(0, positions.to(torch::kInt64));
    const torch::Tensor cosine =
        rows.narrow(1, 0, width / 2).repeat_interleave(2, -1).contiguous();
    const torch::Tensor sine = rows.narrow(1, width / 2, width / 2)
                                   .repeat_interleave(2, -1)
                                   .contiguous();
    const torch::Tensor value =
        input.to(torch::kFloat32)
            .reshape({input.size(0), -1, input.size(-1)})
            .contiguous();
    torch::Tensor output;
    CHECK(kernel::musa::deepseek_v4_partial_rope(
        value, cosine, sine, head_dim_ - width, width, inverse, output));
    return output.reshape(input.sizes()).to(input.scalar_type());
  }

  std::vector<torch::Tensor> weights_;
  DeepseekV41Dense q_a_;
  DeepseekV41Dense q_b_;
  DeepseekV41Dense kv_;
  DeepseekV41Dense o_b_;
  int64_t heads_;
  int64_t groups_;
  int64_t output_rank_;
  int64_t head_dim_ = 0;
  double epsilon_;
  bool original_fp8_activation_;
};

}  // namespace xllm::layer
