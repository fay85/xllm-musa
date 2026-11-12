/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <cmath>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::layer {

// Reuse xLLM's existing native MUSA weighted RMSNorm. Accumulation and weight
// multiplication remain FP32, with one final BF16 round. This is the ordinary
// norm, not Gemma's (weight+1) norm or a shifted mHC collapse.
inline torch::Tensor deepseek_v41_rms_norm(const torch::Tensor& input,
                                           const torch::Tensor& weight,
                                           double epsilon) {
  CHECK(input.defined() && weight.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.device(), weight.device());
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
  CHECK_GE(input.dim(), 2);
  CHECK_EQ(weight.dim(), 1);
  CHECK_EQ(input.size(-1), weight.numel());
  CHECK(input.is_contiguous() && weight.is_contiguous());
  CHECK(std::isfinite(epsilon) && epsilon > 0.0);
  torch::DeviceGuard guard(input.device());
  const torch::Tensor flat = input.reshape({-1, input.size(-1)});
  torch::Tensor output = torch::empty_like(flat);
  if (flat.size(0) == 0) {
    return output.reshape(input.sizes());
  }
  kernel::musa::rms_norm(output, flat, weight, epsilon);
  return output.reshape(input.sizes());
}

}  // namespace xllm::layer
