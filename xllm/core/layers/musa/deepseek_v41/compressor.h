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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/musa/deepseek_v41_pair_pool.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"
#include "core/layers/musa/deepseek_v41/rms_norm.h"

namespace xllm::layer {

// Original V4.1 compressor for C1/eager native execution. Ratio 2 pools in
// FP32, rounds the pooled latent to BF16, then normalizes before any RoPE.
// Partial groups use the framework's FP32 state planes, not private history.
class DeepseekV41Compressor final {
 public:
  DeepseekV41Compressor(torch::Tensor kv_weight,
                        torch::Tensor norm_weight,
                        std::optional<torch::Tensor> gate_weight,
                        std::optional<torch::Tensor> kv_state,
                        std::optional<torch::Tensor> score_state,
                        int64_t ratio,
                        int64_t swa_block_size,
                        double epsilon,
                        bool fused_pair_pool = false)
      : kv_weight_(std::move(kv_weight)),
        norm_weight_(std::move(norm_weight)),
        gate_weight_(std::move(gate_weight)),
        kv_state_(std::move(kv_state)),
        score_state_(std::move(score_state)),
        ratio_(ratio),
        swa_block_size_(swa_block_size),
        epsilon_(epsilon),
        fused_pair_pool_(fused_pair_pool) {
    CHECK(ratio_ == 1 || ratio_ == 2);
    CHECK_GT(swa_block_size_, 0);
    CHECK(std::isfinite(epsilon_) && epsilon_ > 0.0);
    CHECK(kv_weight_.defined() && norm_weight_.defined());
    CHECK(kv_weight_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(norm_weight_.device(), kv_weight_.device());
    CHECK_EQ(kv_weight_.dim(), 2);
    CHECK_GT(kv_weight_.size(0), 0);
    CHECK_GT(kv_weight_.size(1), 0);
    CHECK_EQ(norm_weight_.dim(), 1);
    CHECK_EQ(norm_weight_.size(0), kv_weight_.size(0));
    CHECK_EQ(norm_weight_.scalar_type(), torch::kBFloat16);
    CHECK_EQ(kv_weight_.scalar_type(),
             ratio_ == 1 ? torch::kBFloat16 : torch::kFloat32);
    CHECK_EQ(gate_weight_.has_value(), ratio_ == 2);
    CHECK_EQ(kv_state_.has_value(), ratio_ == 2);
    CHECK_EQ(score_state_.has_value(), ratio_ == 2);
    if (ratio_ == 2) {
      const torch::Tensor& gate = gate_weight_.value();
      CHECK(gate.defined());
      CHECK_EQ(gate.device(), kv_weight_.device());
      CHECK_EQ(gate.scalar_type(), torch::kFloat32);
      CHECK_EQ(gate.sizes(), kv_weight_.sizes());
      check_state(kv_state_.value());
      check_state(score_state_.value());
      CHECK_EQ(kv_state_->sizes(), score_state_->sizes());
    }
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& input,
      int64_t start_position,
      const torch::Tensor& swa_block_table) {
    const DeepseekV41ProfileRange range("V41/Attention/compressor");
    CHECK(input.defined() && swa_block_table.defined());
    CHECK_EQ(input.device(), kv_weight_.device());
    CHECK_EQ(input.scalar_type(), torch::kBFloat16);
    CHECK_EQ(input.dim(), 2);
    CHECK_EQ(input.size(1), kv_weight_.size(1));
    CHECK_GE(start_position, 0);
    CHECK_EQ(swa_block_table.device(), input.device());
    CHECK_EQ(swa_block_table.dim(), 2);
    CHECK_EQ(swa_block_table.size(0), 1);
    CHECK_GT(swa_block_table.size(1), 0);
    CHECK(swa_block_table.scalar_type() == torch::kInt32 ||
          swa_block_table.scalar_type() == torch::kInt64);
    torch::DeviceGuard guard(input.device());
    const auto index_options =
        torch::TensorOptions().device(input.device()).dtype(torch::kInt64);
    if (input.size(0) == 0) {
      return {torch::empty({0, kv_weight_.size(0)}, input.options()),
              torch::empty({0}, index_options)};
    }
    if (ratio_ == 1) {
      return {norm(torch::linear(input, kv_weight_)),
              torch::arange(input.size(0), index_options) + start_position};
    }
    const torch::Tensor values =
        torch::linear(input.to(torch::kFloat32), kv_weight_);
    const torch::Tensor scores =
        torch::linear(input.to(torch::kFloat32), gate_weight_.value());
    if (fused_pair_pool_) {
      auto [pooled, starts] =
          kernel::musa::deepseek_v41_pair_pool(values,
                                               scores,
                                               swa_block_table,
                                               kv_state_.value(),
                                               score_state_.value(),
                                               start_position,
                                               swa_block_size_);
      return {norm(pooled), starts};
    }
    const int64_t group_count =
        (start_position + input.size(0)) / ratio_ - start_position / ratio_;
    const torch::Tensor starts =
        (torch::arange(group_count, index_options) + start_position / ratio_) *
        ratio_;
    torch::Tensor latent =
        torch::empty({0, kv_weight_.size(0)}, input.options());
    if (group_count > 0) {
      const torch::Tensor members =
          starts.unsqueeze(-1) + torch::arange(ratio_, index_options);
      const torch::Tensor inside = members.ge(start_position).unsqueeze(-1);
      const torch::Tensor rows =
          (members - start_position).clamp(0, input.size(0) - 1).flatten();
      const torch::Tensor parked =
          state_slots(swa_block_table, members.clamp_min(0)).flatten();
      const std::vector<int64_t> shape = {
          group_count, ratio_, kv_weight_.size(0)};
      const torch::Tensor member_values =
          torch::where(inside,
                       values.index_select(0, rows).view(shape),
                       kv_state_->view({-1, kv_weight_.size(0)})
                           .index_select(0, parked)
                           .view(shape));
      const torch::Tensor member_scores =
          torch::where(inside,
                       scores.index_select(0, rows).view(shape),
                       score_state_->view({-1, kv_weight_.size(0)})
                           .index_select(0, parked)
                           .view(shape));
      const torch::Tensor pooled =
          (member_values * torch::softmax(member_scores, 1)).sum(1);
      latent = norm(pooled.to(torch::kBFloat16));
    }
    // Pool reads must precede parking: an odd chunk start may still need the
    // previous token whose ring-32 slot this chunk's tail would overwrite.
    const int64_t first =
        std::max<int64_t>(0, input.size(0) - kv_state_->size(1));
    const torch::Tensor tail =
        torch::arange(first, input.size(0), index_options);
    const torch::Tensor positions = tail + start_position;
    const torch::Tensor rows =
        tail.index({(positions + 1).remainder(ratio_).ne(0)});
    const torch::Tensor slots =
        state_slots(swa_block_table, rows + start_position);
    kv_state_->view({-1, kv_weight_.size(0)})
        .index_copy_(0, slots, values.index_select(0, rows));
    score_state_->view({-1, kv_weight_.size(0)})
        .index_copy_(0, slots, scores.index_select(0, rows));
    return {latent, starts};
  }

 private:
  void check_state(const torch::Tensor& state) const {
    CHECK(state.defined());
    CHECK_EQ(state.device(), kv_weight_.device());
    CHECK_EQ(state.scalar_type(), torch::kFloat32);
    CHECK_EQ(state.dim(), 3);
    CHECK_GT(state.size(0), 0);
    CHECK_EQ(state.size(1), 32);
    CHECK_EQ(state.size(2), kv_weight_.size(0));
    CHECK(state.is_contiguous());
  }

  torch::Tensor state_slots(const torch::Tensor& table,
                            const torch::Tensor& positions) const {
    const torch::Tensor columns =
        torch::floor_divide(positions, swa_block_size_)
            .remainder(table.size(1));
    const torch::Tensor blocks = table.select(0, 0)
                                     .index_select(0, columns.flatten())
                                     .to(torch::kInt64)
                                     .clamp_min(0)
                                     .reshape(positions.sizes());
    return blocks * kv_state_->size(1) +
           positions.remainder(kv_state_->size(1));
  }

  torch::Tensor norm(const torch::Tensor& value) const {
    return deepseek_v41_rms_norm(value.contiguous(), norm_weight_, epsilon_);
  }

  torch::Tensor kv_weight_;
  torch::Tensor norm_weight_;
  std::optional<torch::Tensor> gate_weight_;
  std::optional<torch::Tensor> kv_state_;
  std::optional<torch::Tensor> score_state_;
  int64_t ratio_;
  int64_t swa_block_size_;
  double epsilon_;
  bool fused_pair_pool_;
};

}  // namespace xllm::layer
