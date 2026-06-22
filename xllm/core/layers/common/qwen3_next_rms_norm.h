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

#pragma once

#include <torch/torch.h>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class Qwen3NextRMSNormImpl : public torch::nn::Module {
 public:
  Qwen3NextRMSNormImpl(int64_t dim,
                       double eps,
                       const torch::TensorOptions& options);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      torch::Tensor& input,
      std::optional<torch::Tensor> residual = std::nullopt);
  torch::Tensor weight() const { return weight_; }
  bool is_weight_loaded() const { return weight_is_loaded_; }
  double eps() const { return eps_; }

  void load_state_dict(const StateDict& state_dict);

 private:
  DEFINE_WEIGHT(weight);
  int64_t norm_dim_;
  double eps_;

  // Persistent output buffer for the no-residual path. We pre-allocate this
  // on the first forward() call (which happens during warmup, OUTSIDE the
  // CUDA graph capture region) and reuse it for every subsequent call. This
  // is required for MUSA graph capture safety on torch_musa 2.7.1, where
  // calling `EmptyStridedMUSA` mid-capture raises "operation not permitted
  // when stream is capturing" -- the allocator is not capture-aware. The
  // buffer is grown on demand if a later call sees a larger shape (only
  // possible during eager prefill / warmup; once the largest decode bucket
  // has run, no further growth happens during capture).
  mutable torch::Tensor norm_out_buf_;
};
TORCH_MODULE(Qwen3NextRMSNorm);

}  // namespace layer
}  // namespace xllm
