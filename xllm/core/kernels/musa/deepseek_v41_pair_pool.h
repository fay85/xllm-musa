/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::musa {

// Pool completed ratio-2 pairs before parking this chunk's unfinished tail.
// Values/scores/state stay FP32; the pooled latent rounds once to BF16.
std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_pair_pool(
    const torch::Tensor& values,
    const torch::Tensor& scores,
    const torch::Tensor& table,
    torch::Tensor& value_state,
    torch::Tensor& score_state,
    int64_t start,
    int64_t block_size);

}  // namespace xllm::kernel::musa
