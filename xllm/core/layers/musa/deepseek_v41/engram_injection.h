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

#include <optional>
#include <utility>

#include "core/kernels/musa/deepseek_v41_engram_gate.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"

namespace xllm::layer {

// Native projection and gated injection after Engram head lookup/all-gather.
// This class does not own the large table or invent a second hash algorithm.
// The decoder must supply original-order gathered BF16 embedding columns.
class DeepseekV41EngramInjection final {
 public:
  DeepseekV41EngramInjection(torch::Tensor projection,
                             torch::Tensor query_weight,
                             torch::Tensor key_weight,
                             bool original_fp8_activation)
      : projection_(check_projection(std::move(projection)),
                    original_fp8_activation),
        query_weight_(std::move(query_weight)),
        key_weight_(std::move(key_weight)) {
    CHECK(query_weight_.defined() && key_weight_.defined());
    CHECK(query_weight_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(query_weight_.device(), key_weight_.device());
    CHECK_EQ(query_weight_.sizes(), torch::IntArrayRef({4, 5120}));
    CHECK_EQ(query_weight_.sizes(), key_weight_.sizes());
    CHECK_EQ(query_weight_.scalar_type(), key_weight_.scalar_type());
    CHECK(query_weight_.scalar_type() == torch::kBFloat16 ||
          query_weight_.scalar_type() == torch::kFloat32);
    CHECK(query_weight_.is_contiguous() && key_weight_.is_contiguous());
  }

  torch::Tensor forward(const torch::Tensor& residual,
                        const torch::Tensor& gathered_embedding,
                        const std::optional<torch::Tensor>& token_mask) const {
    CHECK(residual.defined() && gathered_embedding.defined());
    CHECK_EQ(residual.device(), query_weight_.device());
    CHECK_EQ(gathered_embedding.device(), residual.device());
    CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
    CHECK_EQ(gathered_embedding.scalar_type(), torch::kBFloat16);
    CHECK_EQ(residual.dim(), 3);
    CHECK_EQ(residual.size(1), 4);
    CHECK_EQ(residual.size(2), 5120);
    CHECK_LE(residual.size(0), 512);
    CHECK_EQ(gathered_embedding.dim(), 2);
    CHECK_EQ(gathered_embedding.size(0), residual.size(0));
    const torch::Tensor kv = projection_.forward(gathered_embedding);
    return kernel::musa::deepseek_v41_engram_gate(
        residual, kv, query_weight_, key_weight_, token_mask);
  }

 private:
  static torch::Tensor check_projection(torch::Tensor weight) {
    CHECK(weight.defined());
    CHECK_EQ(weight.dim(), 2);
    CHECK_EQ(weight.size(0), 25600);
    return weight;
  }

  DeepseekV41Dense projection_;
  torch::Tensor query_weight_;
  torch::Tensor key_weight_;
};

}  // namespace xllm::layer
