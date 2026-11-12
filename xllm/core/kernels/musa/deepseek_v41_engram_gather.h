/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace xllm::kernel::musa {

torch::Tensor deepseek_v41_engram_gather(const torch::Tensor& weight,
                                         const torch::Tensor& scales,
                                         const torch::Tensor& ids,
                                         int64_t row_lo,
                                         const std::string& aot_directory);

}  // namespace xllm::kernel::musa
