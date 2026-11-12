/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace xllm::kernel::musa {
std::pair<torch::Tensor, torch::Tensor> deepseek_v41_paged_engram_hash(
    const torch::Tensor& ids,
    const torch::Tensor& positions,
    const torch::Tensor& table,
    torch::Tensor& history,
    const torch::Tensor& token_map,
    const torch::Tensor& multipliers,
    const torch::Tensor& primes,
    const torch::Tensor& offsets,
    int64_t pad_id,
    int64_t block_size,
    int64_t start,
    const std::optional<torch::Tensor>& lookback,
    const std::string& directory);
}  // namespace xllm::kernel::musa
