/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
https://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

namespace xllm::layer {

inline bool dsv4_prefill_rope_assembly_eligible(const torch::Tensor& input,
                                               int64_t rope_start,
                                               int64_t rope_width) {
  return input.defined() && input.dim() == 3 && input.is_contiguous() &&
         input.scalar_type() == torch::kBFloat16 &&
         input.device().is_privateuseone() && input.size(0) >= 128 &&
         input.size(0) <= 4096 && input.size(1) == 64 &&
         input.size(2) == 128 && rope_start == 64 && rope_width == 64;
}

// Rotation arithmetic has already completed, with the original dtype/rounding.
// Copy only; no input mutation, new fusion, cross-stream work, or persistent
// allocation. Each tensor view is consumed on the caller's current stream.
inline torch::Tensor dsv4_assemble_prefill_rope(
    const torch::Tensor& input,
    const torch::Tensor& rotated_even,
    const torch::Tensor& rotated_odd) {
  CHECK(dsv4_prefill_rope_assembly_eligible(input, /*rope_start=*/64,
                                          /*rope_width=*/64));
  CHECK_EQ(rotated_even.dim(), 3);
  CHECK_EQ(rotated_even.size(0), input.size(0));
  CHECK_EQ(rotated_even.size(1), 64);
  CHECK_EQ(rotated_even.size(2), 32);
  CHECK(rotated_even.sizes() == rotated_odd.sizes());
  CHECK_EQ(rotated_even.device(), input.device());
  CHECK_EQ(rotated_odd.device(), input.device());
  CHECK_EQ(rotated_even.scalar_type(), input.scalar_type());
  CHECK_EQ(rotated_odd.scalar_type(), input.scalar_type());
  auto output = input.clone();
  output.slice(/*dim=*/-1, /*start=*/64, /*end=*/128, /*step=*/2)
      .copy_(rotated_even);
  output.slice(/*dim=*/-1, /*start=*/65, /*end=*/128, /*step=*/2)
      .copy_(rotated_odd);
  return output;
}

}  // namespace xllm::layer
