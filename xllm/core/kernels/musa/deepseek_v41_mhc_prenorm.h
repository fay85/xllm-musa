/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <tuple>

namespace xllm::kernel::musa {

// Frozen SGLang dot160 projection and fused RMS/Sinkhorn, M=1..8.
// Returns mixes, next_pre, post, comb; it does not collapse carried_pre.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_mhc_dot_sinkhorn(const torch::Tensor& residual_flat,
                              const torch::Tensor& fn,
                              const torch::Tensor& scale,
                              const torch::Tensor& base,
                              const std::string& directory);

// MATE TF32 prenorm projection plus row square-sum. Only normalized mix
// coefficients are returned: V4.1's carried/shifted pre-mix stays in the
// native decoder and must not be replaced by V4's same-sublayer pre-mix.
torch::Tensor deepseek_v41_mhc_prenorm(const torch::Tensor& residual_flat,
                                       const torch::Tensor& fn,
                                       const std::string& library_directory,
                                       double epsilon);

// Fused GPU reduction of MATE split projection and square sums.
torch::Tensor deepseek_v41_mhc_prenorm_reduce(const torch::Tensor& partial,
                                              const torch::Tensor& square_sum,
                                              int64_t hidden_width,
                                              double epsilon);

torch::Tensor deepseek_v41_mhc_prenorm_fused(
    const torch::Tensor& residual_flat,
    const torch::Tensor& fn,
    const std::string& library_directory,
    double epsilon);

}  // namespace xllm::kernel::musa
