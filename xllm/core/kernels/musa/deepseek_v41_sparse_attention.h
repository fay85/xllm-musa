/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <string>

namespace xllm::kernel::musa {

// H8/D512 MATE sparse MLA operating directly on the existing V4.1 QDQ BF16
// cache carrier. No additional KV quantization or change of index semantics.
torch::Tensor deepseek_v41_mate_sparse_attention(
    const torch::Tensor& query,
    const torch::Tensor& kv,
    const torch::Tensor& indices,
    const torch::Tensor& sink,
    double scale,
    const std::string& library_path);

}  // namespace xllm::kernel::musa
