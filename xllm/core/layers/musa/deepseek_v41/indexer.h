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
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/musa/deepseek_v41_attention_quant.h"
#include "core/kernels/musa/deepseek_v41_indexer_score.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"
#include "core/layers/musa/deepseek_v41/rms_norm.h"

namespace xllm::layer {

// Original E2M1/group32 index Q/K, replicated index heads as in the Python
// V4.1 path. A TP all-reduce here would count every head repeatedly. Unlike
// V4 QLI's INT8 path, this uses BF16 carriers after the original FP4 QDQ.
class DeepseekV41Indexer final {
 public:
  // wq_b, weights_proj; source layers additionally supply wk and k_norm.
  DeepseekV41Indexer(std::vector<torch::Tensor> weights,
                     int64_t heads,
                     int64_t head_dim,
                     double epsilon,
                     bool original_fp8_activation,
                     std::string score_aot_directory = {})
      : weights_(check_weights(std::move(weights))),
        query_(weights_[0], original_fp8_activation),
        heads_(heads),
        head_dim_(head_dim),
        epsilon_(epsilon),
        score_aot_directory_(std::move(score_aot_directory)) {
    CHECK_GT(heads_, 0);
    CHECK_GT(head_dim_, 0);
    CHECK_EQ(head_dim_ % 32, 0);
    CHECK(std::isfinite(epsilon_) && epsilon_ > 0.0);
    CHECK_EQ(weights_[0].dim(), 2);
    CHECK_EQ(weights_[0].size(0), heads_ * head_dim_);
    CHECK_EQ(weights_[1].dim(), 2);
    CHECK_EQ(weights_[1].size(0), heads_);
    if (weights_.size() == 4) {
      CHECK_EQ(weights_[2].dim(), 2);
      CHECK_EQ(weights_[2].size(0), head_dim_);
      CHECK_EQ(weights_[3].dim(), 1);
      CHECK_EQ(weights_[3].size(0), head_dim_);
    }
  }

  std::tuple<torch::Tensor, torch::Tensor> prepare_query(
      const torch::Tensor& hidden,
      const torch::Tensor& qr,
      const torch::Tensor& positions,
      const torch::Tensor& table) const {
    check_input(hidden, weights_[1].size(1));
    check_input(qr, weights_[0].size(1));
    CHECK_EQ(hidden.size(0), qr.size(0));
    torch::DeviceGuard guard(hidden.device());
    const torch::Tensor query =
        query_.forward(qr).view({qr.size(0), heads_, head_dim_});
    // Multiplication is in BF16, matching weights_proj(x) * scale, before
    // conversion to FP32 in scoring. Do not move it after the head reduction.
    const torch::Tensor head_weights =
        torch::linear(hidden, weights_[1]) *
        (std::pow(static_cast<double>(head_dim_), -0.5) *
         std::pow(static_cast<double>(heads_), -0.5));
    return {rope_quantize(query, positions, table), head_weights};
  }

  torch::Tensor prepare_key(const torch::Tensor& latent,
                            const torch::Tensor& positions,
                            const torch::Tensor& table) const {
    CHECK_EQ(weights_.size(), 4) << "Reindex layer does not own index K";
    check_input(latent, weights_[2].size(1));
    torch::DeviceGuard guard(latent.device());
    const torch::Tensor projected = torch::linear(latent, weights_[2]);
    const torch::Tensor normalized =
        deepseek_v41_rms_norm(projected, weights_[3], epsilon_);
    return rope_quantize(normalized, positions, table);
  }

  torch::Tensor score(const torch::Tensor& query,
                      const torch::Tensor& keys,
                      const torch::Tensor& head_weights,
                      const torch::Tensor& positions,
                      int64_t committed,
                      int64_t ratio) const {
    CHECK(query.defined() && keys.defined() && head_weights.defined());
    CHECK_EQ(query.device(), weights_[0].device());
    CHECK_EQ(keys.device(), query.device());
    CHECK_EQ(head_weights.device(), query.device());
    CHECK_EQ(query.scalar_type(), torch::kBFloat16);
    CHECK_EQ(keys.scalar_type(), torch::kBFloat16);
    CHECK_EQ(head_weights.scalar_type(), torch::kBFloat16);
    CHECK_EQ(query.dim(), 3);
    CHECK_EQ(query.size(1), heads_);
    CHECK_EQ(query.size(2), head_dim_);
    CHECK_EQ(keys.dim(), 2);
    CHECK_EQ(keys.size(1), head_dim_);
    CHECK_EQ(head_weights.sizes(), torch::IntArrayRef({query.size(0), heads_}));
    check_positions(positions, query.size(0));
    CHECK_GE(committed, 0);
    CHECK_LE(committed, keys.size(0));
    CHECK(ratio == 1 || ratio == 2);
    torch::DeviceGuard guard(query.device());
    if (!score_aot_directory_.empty()) {
      return kernel::musa::deepseek_v41_indexer_score(query,
                                                      keys,
                                                      head_weights,
                                                      positions,
                                                      committed,
                                                      ratio,
                                                      score_aot_directory_);
    }
    const torch::Tensor rows =
        torch::arange(keys.size(0), positions.options().dtype(torch::kInt64));
    // Unallocated tail pages can contain NaN: zero before ANY arithmetic.
    const torch::Tensor safe_keys =
        torch::where(rows.lt(committed).unsqueeze(-1), keys, 0);
    // The official inference and delivery scorer round the dot, weighted
    // product, and head reduction to BF16 separately. Continuous FP32 math
    // here changes ties near the top-k boundary even with FP4 Q/K carriers.
    const torch::Tensor dots = torch::einsum("shd,td->sht", {query, safe_keys});
    const torch::Tensor weighted =
        torch::relu(dots) * head_weights.unsqueeze(-1);
    torch::Tensor result = weighted.sum(1).to(torch::kFloat32);
    const torch::Tensor visible = torch::floor_divide(positions + 1, ratio);
    const torch::Tensor invalid =
        rows.ge(visible.unsqueeze(-1)).logical_or(rows.ge(committed));
    result.masked_fill_(invalid, -std::numeric_limits<float>::infinity());
    return result;
  }

  // Returns zero-based compressed row IDs. The paged attention component
  // applies its own window offset. Stable ties preserve chunk invariance.
  torch::Tensor select(const torch::Tensor& scores,
                       const torch::Tensor& visible,
                       int64_t topk,
                       const std::optional<torch::Tensor>& candidates) const {
    check_scores(scores, visible);
    CHECK_GT(topk, 0);
    torch::DeviceGuard guard(scores.device());
    if (!score_aot_directory_.empty() && scores.size(1) > 0 &&
        scores.size(1) <= 8192) {
      return kernel::musa::deepseek_v41_indexer_select(
          scores, visible, topk, candidates, score_aot_directory_);
    }
    torch::Tensor masked = scores;
    if (candidates.has_value()) {
      CHECK(candidates->defined());
      CHECK_EQ(candidates->device(), scores.device());
      CHECK_EQ(candidates->scalar_type(), torch::kBool);
      CHECK_EQ(candidates->sizes(), scores.sizes());
      masked = scores.masked_fill(candidates->logical_not(),
                                  -std::numeric_limits<float>::infinity());
    }
    // TorchMUSA stable argsort can order +0 ahead of an earlier -0. Numeric
    // ties must preserve the original index, matching CPU and the fused radix
    // implementation; canonicalize only signed zeros before sorting.
    masked = torch::where(masked.eq(0), 0, masked);
    const int64_t count = std::min(topk, scores.size(1));
    const torch::Tensor ordered =
        torch::argsort(masked, /*stable=*/true, /*dim=*/-1, /*descending=*/true)
            .narrow(-1, 0, count);
    const torch::Tensor indices = std::get<0>(ordered.sort(-1));
    const torch::Tensor valid =
        indices.lt(visible.unsqueeze(-1))
            .logical_and(torch::isfinite(masked.gather(-1, indices)));
    return torch::where(valid, indices, -1).to(torch::kInt32);
  }

  torch::Tensor candidates(const torch::Tensor& scores,
                           const torch::Tensor& visible,
                           int64_t topk_blocks,
                           int64_t block_size) const {
    check_scores(scores, visible);
    CHECK_GT(topk_blocks, 0);
    CHECK_GT(block_size, 0);
    torch::DeviceGuard guard(scores.device());
    const int64_t width = scores.size(1);
    if (width == 0) {
      return torch::empty_like(scores, scores.options().dtype(torch::kBool));
    }
    const int64_t padding = (block_size - width % block_size) % block_size;
    const torch::Tensor padded =
        torch::cat({scores,
                    torch::full({scores.size(0), padding},
                                -std::numeric_limits<float>::infinity(),
                                scores.options())},
                   -1);
    const torch::Tensor grouped = padded.view({scores.size(0), -1, block_size});
    torch::Tensor blocks = grouped.amax(-1);
    // This TorchMUSA reduction returns lowest-finite for all -inf input.
    // Restore unreachable-block semantics before selecting candidates.
    blocks = torch::where(
        grouped.ne(-std::numeric_limits<float>::infinity()).any(-1),
        blocks,
        -std::numeric_limits<float>::infinity());
    const torch::Tensor latest = torch::floor_divide(visible - 1, block_size);
    const torch::Tensor block_ids =
        torch::arange(blocks.size(1), visible.options().dtype(torch::kInt64));
    blocks.masked_fill_(block_ids.eq(latest.unsqueeze(-1)),
                        std::numeric_limits<float>::infinity());
    const int64_t count = std::min(topk_blocks, blocks.size(1));
    const torch::Tensor indices =
        torch::argsort(blocks, /*stable=*/true, /*dim=*/-1, /*descending=*/true)
            .narrow(-1, 0, count);
    const torch::Tensor values = blocks.gather(-1, indices);
    const torch::Tensor keep =
        torch::zeros_like(blocks, blocks.options().dtype(torch::kBool))
            .scatter_(-1,
                      indices,
                      values.gt(-std::numeric_limits<float>::infinity()));
    return keep.unsqueeze(-1)
        .expand({scores.size(0), blocks.size(1), block_size})
        .reshape({scores.size(0), -1})
        .narrow(-1, 0, width);
  }

 private:
  static std::vector<torch::Tensor> check_weights(
      std::vector<torch::Tensor> weights) {
    CHECK(weights.size() == 2 || weights.size() == 4);
    for (const torch::Tensor& weight : weights) {
      CHECK(weight.defined());
      CHECK(weight.device().type() == torch::kPrivateUse1);
      CHECK_EQ(weight.device(), weights[0].device());
      CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
    }
    return weights;
  }

  void check_input(const torch::Tensor& input, int64_t width) const {
    CHECK(input.defined());
    CHECK_EQ(input.device(), weights_[0].device());
    CHECK_EQ(input.scalar_type(), torch::kBFloat16);
    CHECK_EQ(input.dim(), 2);
    CHECK_EQ(input.size(1), width);
  }

  void check_positions(const torch::Tensor& positions, int64_t rows) const {
    CHECK(positions.defined());
    CHECK_EQ(positions.device(), weights_[0].device());
    CHECK_EQ(positions.dim(), 1);
    CHECK_EQ(positions.size(0), rows);
    CHECK(positions.scalar_type() == torch::kInt32 ||
          positions.scalar_type() == torch::kInt64);
  }

  void check_scores(const torch::Tensor& scores,
                    const torch::Tensor& visible) const {
    CHECK(scores.defined());
    CHECK_EQ(scores.device(), weights_[0].device());
    CHECK_EQ(scores.scalar_type(), torch::kFloat32);
    CHECK_EQ(scores.dim(), 2);
    check_positions(visible, scores.size(0));
  }

  torch::Tensor rope_quantize(const torch::Tensor& input,
                              const torch::Tensor& positions,
                              const torch::Tensor& table) const {
    check_positions(positions, input.size(0));
    return kernel::musa::deepseek_v41_rope_fp4_quant(input.contiguous(),
                                                     positions,
                                                     table,
                                                     /*group_size=*/32,
                                                     /*e4m3_scale=*/false);
  }

  std::vector<torch::Tensor> weights_;
  DeepseekV41Dense query_;
  int64_t heads_;
  int64_t head_dim_;
  double epsilon_;
  std::string score_aot_directory_;
};

}  // namespace xllm::layer
