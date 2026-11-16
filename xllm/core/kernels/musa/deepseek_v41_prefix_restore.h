/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once
#include <torch/torch.h>

#include <cstdint>
namespace xllm::kernel::musa {
// Restore only rows whose producing prefix exceeds accepted. Invalid slots
// are ignored, and the caller guarantees distinct valid destination rows.
void deepseek_v41_restore_suffix(const torch::Tensor& target,
                                 const torch::Tensor& slots,
                                 const torch::Tensor& saved,
                                 const torch::Tensor& prefix_lengths,
                                 int64_t accepted);
}  // namespace xllm::kernel::musa
