/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <string>

namespace xllm::kernel::musa {

// Native launch of the offline-compiled SGLang block32 FP8 quant/GEMM port.
// Raw E4M3 weights and decoded F32 UE8M0 block scales remain unchanged.
torch::Tensor deepseek_v41_dense_block32(const torch::Tensor& input,
                                         const torch::Tensor& weight,
                                         const torch::Tensor& weight_scale,
                                         const std::string& directory);

}  // namespace xllm::kernel::musa
