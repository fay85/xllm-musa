/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once

#include <torch/torch.h>

namespace xllm::kernel::musa {

// BF16 [rows, 128], contiguous columns and 4-element-aligned row strides.
// Computes RMSNorm(input) * sigmoid(gate) in FP32, then rounds to BF16.
torch::Tensor sigmoid_gated_rms_norm(const torch::Tensor& input,
                                     const torch::Tensor& gate,
                                     const torch::Tensor& weight,
                                     double eps);

}  // namespace xllm::kernel::musa
