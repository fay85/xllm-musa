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

// C1 decode with GPU position. The caller must select a graph whose
// closes_pair matches position % 2 == 1; no device-to-host read occurs here.
std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_pair_pool_decode(
    const torch::Tensor& values,
    const torch::Tensor& scores,
    const torch::Tensor& table,
    torch::Tensor& value_state,
    torch::Tensor& score_state,
    const torch::Tensor& position,
    bool closes_pair,
    int64_t block_size);

// Fixed 1..6-row verification. Only start parity is a host graph-bucket key;
// consecutive absolute positions stay on the device across replays.
std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_pair_pool_verify(
    const torch::Tensor& values,
    const torch::Tensor& scores,
    const torch::Tensor& table,
    torch::Tensor& value_state,
    torch::Tensor& score_state,
    const torch::Tensor& positions,
    bool starts_odd,
    int64_t block_size);

}  // namespace xllm::kernel::musa
