/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>

namespace xllm::kernel::musa {

torch::Tensor deepseek_v41_indexer_score(const torch::Tensor& query,
                                         const torch::Tensor& keys,
                                         const torch::Tensor& weights,
                                         const torch::Tensor& positions,
                                         int64_t committed,
                                         int64_t ratio,
                                         const std::string& aot_directory);

torch::Tensor deepseek_v41_indexer_select(
    const torch::Tensor& scores,
    const torch::Tensor& visible,
    int64_t topk,
    const std::optional<torch::Tensor>& candidates,
    const std::string& aot_directory);

}  // namespace xllm::kernel::musa
