/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>
namespace xllm::kernel::musa {
// Produce slots, saved rows and producing-prefix lengths in one GPU launch.
// Invalid padding pages preserve the existing clamped-read/ignored-restore
// rule.
std::vector<torch::Tensor> deepseek_v41_snapshot_prefix(
    const torch::Tensor& cache,
    const torch::Tensor& table,
    int64_t begin,
    int64_t tokens,
    int64_t ratio,
    bool ring,
    bool recurrent,
    int64_t swa_block_size);

// Restore only rows whose producing prefix exceeds accepted. Invalid slots
// are ignored, and the caller guarantees distinct valid destination rows.
void deepseek_v41_restore_suffix(const torch::Tensor& target,
                                 const torch::Tensor& slots,
                                 const torch::Tensor& saved,
                                 const torch::Tensor& prefix_lengths,
                                 int64_t accepted);
}  // namespace xllm::kernel::musa
