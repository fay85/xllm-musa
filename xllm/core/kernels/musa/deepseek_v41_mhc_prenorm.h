/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <string>

namespace xllm::kernel::musa {

// MATE TF32 prenorm projection plus row square-sum. Only normalized mix
// coefficients are returned: V4.1's carried/shifted pre-mix stays in the
// native decoder and must not be replaced by V4's same-sublayer pre-mix.
torch::Tensor deepseek_v41_mhc_prenorm(const torch::Tensor& residual_flat,
                                       const torch::Tensor& fn,
                                       const std::string& library_directory,
                                       double epsilon);

}  // namespace xllm::kernel::musa
