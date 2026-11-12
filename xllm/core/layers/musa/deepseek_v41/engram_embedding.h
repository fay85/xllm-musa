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

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/musa/deepseek_v41_engram_gather.h"

namespace xllm::layer {

// Original FP8-E4M3 table bytes and UE8M0 per-32 scales remain device resident
// and row-sharded by whole Engram heads, as in the current original model.
// Lookup decodes only requested rows, never the complete table into BF16.
class DeepseekV41EngramEmbedding final {
 public:
  DeepseekV41EngramEmbedding(torch::Tensor weight,
                             torch::Tensor scales,
                             int64_t vocab_start,
                             int64_t head_start,
                             int64_t local_heads,
                             ProcessGroup* group = nullptr,
                             std::string aot_directory = {})
      : weight_(std::move(weight)),
        scales_(std::move(scales)),
        vocab_start_(vocab_start),
        head_start_(head_start),
        local_heads_(local_heads),
        group_(group),
        aot_directory_(std::move(aot_directory)) {
    CHECK(weight_.defined() && scales_.defined());
    CHECK(weight_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(scales_.device(), weight_.device());
    CHECK_EQ(weight_.scalar_type(), torch::kUInt8);
    CHECK_EQ(scales_.scalar_type(), torch::kUInt8);
    CHECK_EQ(weight_.dim(), 2);
    CHECK_EQ(scales_.dim(), 2);
    CHECK_GT(weight_.size(0), 0);
    CHECK_GT(weight_.size(1), 0);
    CHECK_EQ(weight_.size(1) % 32, 0);
    CHECK_EQ(scales_.sizes(),
             torch::IntArrayRef({weight_.size(0), weight_.size(1) / 32}));
    CHECK_GE(vocab_start_, 0);
    CHECK_GE(head_start_, 0);
    CHECK_GT(local_heads_, 0);
  }

  torch::Tensor local(const torch::Tensor& hash_ids) const {
    CHECK(hash_ids.defined());
    CHECK_EQ(hash_ids.device(), weight_.device());
    CHECK(hash_ids.scalar_type() == torch::kInt32 ||
          hash_ids.scalar_type() == torch::kInt64);
    CHECK_EQ(hash_ids.dim(), 2);
    CHECK_GE(hash_ids.size(1), head_start_ + local_heads_);
    torch::DeviceGuard guard(weight_.device());
    const torch::Tensor columns =
        hash_ids.narrow(1, head_start_, local_heads_).to(torch::kInt64);
    if (!aot_directory_.empty()) {
      return kernel::musa::deepseek_v41_engram_gather(weight_,
                                                      scales_,
                                                      columns.contiguous(),
                                                      vocab_start_,
                                                      aot_directory_)
          .view({hash_ids.size(0), local_heads_, weight_.size(1)});
    }
    const torch::Tensor owned =
        columns.ge(vocab_start_)
            .logical_and(columns.lt(vocab_start_ + weight_.size(0)));
    const torch::Tensor ids =
        (columns - vocab_start_).clamp(0, weight_.size(0) - 1).flatten();
    const torch::Tensor payload =
        weight_.index_select(0, ids).to(torch::kInt32);
    const torch::Tensor raw_scale =
        scales_.index_select(0, ids).to(torch::kInt32);
    const torch::Tensor magnitude = torch::bitwise_and(payload, 127);
    const torch::Tensor exponent =
        torch::bitwise_and(torch::bitwise_right_shift(magnitude, 3), 15);
    const torch::Tensor mantissa =
        torch::bitwise_and(magnitude, 7).to(torch::kFloat32);
    const torch::Tensor normal =
        (1.0 + mantissa / 8.0) *
        torch::exp2(exponent.to(torch::kFloat32) - 7.0);
    const torch::Tensor subnormal = mantissa * (1.0 / 512.0);
    torch::Tensor decoded = torch::where(exponent.gt(0), normal, subnormal);
    decoded = torch::where(
        magnitude.eq(127), std::numeric_limits<float>::quiet_NaN(), decoded);
    decoded = decoded * torch::where(payload.ge(128), -1.0, 1.0);
    torch::Tensor bits = torch::where(
        raw_scale.eq(0), 0x00400000, torch::bitwise_left_shift(raw_scale, 23));
    bits = torch::where(raw_scale.eq(255), 0x7fc00000, bits);
    const torch::Tensor scale =
        bits.contiguous().view(torch::kFloat32).repeat_interleave(32, -1);
    const torch::Tensor rows =
        (decoded * scale)
            .to(torch::kBFloat16)
            .view({hash_ids.size(0), local_heads_, weight_.size(1)});
    return torch::where(owned.unsqueeze(-1), rows, 0);
  }

  torch::Tensor lookup(const torch::Tensor& hash_ids) const {
    const int64_t world_size = group_ == nullptr ? 1 : group_->world_size();
    CHECK_EQ(hash_ids.size(1), local_heads_ * world_size);
    CHECK_EQ(head_start_,
             local_heads_ * (group_ == nullptr ? 0 : group_->rank()));
    return parallel_state::gather(local(hash_ids), group_, /*dim=*/1);
  }

 private:
  torch::Tensor weight_;
  torch::Tensor scales_;
  int64_t vocab_start_;
  int64_t head_start_;
  int64_t local_heads_;
  ProcessGroup* group_;
  std::string aot_directory_;
};

}  // namespace xllm::layer
