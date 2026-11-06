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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::musa {

struct MHCPreOutput final {
  torch::Tensor layer_input;
  torch::Tensor post_mix;
  torch::Tensor comb_mix;
};

// Model-independent MHC pre. residual is [..., hc_mult, hidden_size], hc_fn is
// [hc_mult * (hc_mult + 2), hc_mult * hidden_size], and coefficients are FP32.
// hc_mult is limited to 1..12 so the one-token reduction fits device shared
// memory. GLM uses 4; DeepSeek-V4 can move to this path without a new kernel.
// The computation is performed on the input device and returns the input dtype
// for layer_input and FP32 for post/comb coefficients.
MHCPreOutput mhc_pre(const torch::Tensor& residual,
                     const torch::Tensor& hc_fn,
                     const torch::Tensor& hc_scale,
                     const torch::Tensor& hc_base,
                     double rms_eps,
                     double hc_pre_eps,
                     double hc_sinkhorn_eps,
                     double hc_post_mult_value,
                     int64_t sinkhorn_repeat);

// Model-independent MHC post. x is [..., hidden_size], residual is
// [..., hc_mult, hidden_size], post is [..., hc_mult] or [..., hc_mult, 1],
// and comb is [..., hc_mult, hc_mult].
torch::Tensor mhc_post(const torch::Tensor& x,
                       const torch::Tensor& residual,
                       const torch::Tensor& post,
                       const torch::Tensor& comb);

// Inverse RMS of the flattened residual streams. output is [..., 1].
torch::Tensor mhc_inverse_rms(const torch::Tensor& residual, double eps);

torch::Tensor mhc_head(const torch::Tensor& residual,
                       const torch::Tensor& hc_head_fn,
                       const torch::Tensor& hc_head_scale,
                       const torch::Tensor& hc_head_base,
                       double norm_eps,
                       double hc_eps);

torch::Tensor mhc_expand(const torch::Tensor& hidden_states, int64_t hc_mult);

torch::Tensor mhc_contract_mean(const torch::Tensor& residual);

}  // namespace xllm::kernel::musa
