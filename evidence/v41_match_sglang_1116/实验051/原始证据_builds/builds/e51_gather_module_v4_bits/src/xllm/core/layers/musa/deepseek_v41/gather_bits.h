/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/parallel_state.h"

namespace xllm::layer {

// Gather the complete BF16 embedding/vocabulary shards without floating-point
// addition of model values. Each signed 16-bit encoding is exactly
// representable as FP32, and only its owning rank contributes a nonzero
// integer. Decoding the sum therefore preserves all original bits, including -0
// and NaN payloads. This is a model-specific small-row path, not a generic
// AllGather replacement.
inline torch::Tensor deepseek_v41_gather_bf16_bits(const torch::Tensor& input,
                                                   ProcessGroup* group,
                                                   bool enabled) {
  constexpr int64_t kMaxRows = 6;
  constexpr int64_t kMaxEncodedBytes = 4 * 1024 * 1024;
  const bool supported =
      enabled && group != nullptr && group->world_size() == 8 &&
      input.device().type() == torch::kPrivateUse1 && input.dim() == 2 &&
      input.is_contiguous() && input.scalar_type() == torch::kBFloat16 &&
      input.size(0) > 0 && input.size(0) <= kMaxRows && input.size(1) > 0 &&
      input.numel() <= kMaxEncodedBytes / (8 * sizeof(float)) &&
      (input.numel() * 8 * sizeof(float)) % 16 == 0;
  if (!supported) {
    return parallel_state::gather(input, group, /*dim=*/-1);
  }
  torch::DeviceGuard guard(input.device());
  const int64_t width = input.size(1);
  const int64_t rank = group->rank();
  CHECK_GE(rank, 0);
  CHECK_LT(rank, 8);
  const torch::Tensor encoded = input.view(torch::kInt16).to(torch::kFloat32);
  torch::Tensor padded = torch::zeros({input.size(0), width * 8},
                                      input.options().dtype(torch::kFloat32));
  padded.narrow(/*dim=*/1, rank * width, width).copy_(encoded);
  auto gathered = musa_allreduce_tp_custom(group, padded);
  CHECK(gathered.has_value())
      << "Selected V4.1 BF16 bit gather requires supported custom TP reduction";
  return gathered->to(torch::kInt16).view(torch::kBFloat16);
}

}  // namespace xllm::layer
