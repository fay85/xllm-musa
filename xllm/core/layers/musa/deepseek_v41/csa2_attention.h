/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
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
#include <string>
#include <utility>

#include "core/kernels/musa/deepseek_v41_sparse_attention.h"

namespace xllm::layer {

// Native CSA2 sink attention. The caller owns chronological window picks,
// shared compressed-cache picks, and their fixed padding. Invalid picks are
// removed before either score or value math, including uninitialized NaNs.
// The constructor selects the FP32 oracle or the MATE BF16 sparse kernel.
// Both consume the same QDQ cache carrier; no exception-based fallback.
class DeepseekV41SparseAttention final {
 public:
  explicit DeepseekV41SparseAttention(double scale,
                                      int64_t max_gather_bytes = 32 * 1024 *
                                                                 1024,
                                      std::string library_path = {})
      : scale_(scale),
        max_gather_bytes_(max_gather_bytes),
        library_path_(std::move(library_path)) {
    CHECK(std::isfinite(scale_) && scale_ > 0.0);
    CHECK_GT(max_gather_bytes_, 0);
  }

  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& kv,
                        const torch::Tensor& indices,
                        const torch::Tensor& sink) const {
    CHECK(query.defined() && kv.defined() && indices.defined() &&
          sink.defined());
    CHECK(query.device().type() == torch::kPrivateUse1);
    CHECK_EQ(query.device(), kv.device());
    CHECK_EQ(query.device(), indices.device());
    CHECK_EQ(query.device(), sink.device());
    CHECK_EQ(query.scalar_type(), torch::kBFloat16);
    CHECK_EQ(kv.scalar_type(), torch::kBFloat16);
    CHECK(indices.scalar_type() == torch::kInt64 ||
          indices.scalar_type() == torch::kInt32);
    CHECK_EQ(sink.scalar_type(), torch::kFloat32);
    CHECK_EQ(query.dim(), 3);
    CHECK_EQ(kv.dim(), 2);
    CHECK_EQ(indices.dim(), 2);
    CHECK_EQ(sink.dim(), 1);
    CHECK_EQ(query.size(0), indices.size(0));
    CHECK_EQ(query.size(1), sink.size(0));
    CHECK_EQ(query.size(2), kv.size(1));
    CHECK_GT(query.size(1), 0);
    CHECK_GT(query.size(2), 0);
    CHECK_GT(kv.size(0), 0);
    CHECK_GT(indices.size(1), 0);
    torch::DeviceGuard guard(query.device());
    if (!library_path_.empty()) {
      return kernel::musa::deepseek_v41_mate_sparse_attention(
          query, kv, indices, sink, scale_, library_path_);
    }
    torch::Tensor output = torch::empty_like(query);
    const int64_t width = indices.size(1);
    const int64_t dim = query.size(2);
    // Same FP32 gather budget and reduction geometry as the Python baseline.
    CHECK_LE(dim, max_gather_bytes_ / 8);
    CHECK_LE(width, max_gather_bytes_ / (8 * dim));
    const int64_t bytes_per_query = 2 * width * dim * 4;
    CHECK_LE(bytes_per_query, max_gather_bytes_)
        << "CSA2 gather budget cannot fit one query";
    const int64_t tile = std::min<int64_t>(
        256, std::max<int64_t>(1, max_gather_bytes_ / bytes_per_query));
    for (int64_t first = 0; first < query.size(0); first += tile) {
      const int64_t count = std::min(tile, query.size(0) - first);
      const torch::Tensor picks =
          indices.narrow(/*dim=*/0, first, count).to(torch::kInt64);
      const torch::Tensor valid = picks.ge(0).logical_and(picks.lt(kv.size(0)));
      const torch::Tensor gathered =
          kv.index_select(/*dim=*/0, picks.clamp(0, kv.size(0) - 1).flatten())
              .view({count, width, dim})
              .to(torch::kFloat32);
      const torch::Tensor selected =
          torch::where(valid.unsqueeze(-1), gathered, 0);
      torch::Tensor scores =
          torch::einsum(
              "shd,snd->shn",
              {query.narrow(/*dim=*/0, first, count).to(torch::kFloat32),
               selected}) *
          scale_;
      scores = scores.masked_fill(valid.logical_not().unsqueeze(1),
                                  -std::numeric_limits<float>::infinity());
      const torch::Tensor row_max = scores.amax(/*dim=*/-1).clamp_min(-1e30);
      const torch::Tensor probabilities =
          torch::exp(scores - row_max.unsqueeze(-1));
      const torch::Tensor denominator =
          probabilities.sum(/*dim=*/-1) + torch::exp(sink - row_max);
      const torch::Tensor value =
          torch::einsum("shn,snd->shd", {probabilities, selected}) /
          denominator.unsqueeze(-1);
      output.narrow(/*dim=*/0, first, count).copy_(value.to(torch::kBFloat16));
    }
    return output;
  }

 private:
  double scale_;
  int64_t max_gather_bytes_;
  std::string library_path_;
};

}  // namespace xllm::layer
