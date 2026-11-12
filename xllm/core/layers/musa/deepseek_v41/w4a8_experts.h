/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <utility>

#include "core/kernels/musa/deepseek_v41_w4a8.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"

namespace xllm::layer {

// All projection work stays in the corrected delivery MATE S4FP8 kernel.
// C++ owns dispatch/quantization/SwiGLU/combine and the EP-local weight slice.
// This object returns FP32 LOCAL routed output; its owner performs one EP
// all-reduce and adds the replicated shared expert exactly once afterwards.
class DeepseekV41W4A8Experts final {
 public:
  DeepseekV41W4A8Experts(torch::Tensor gate_up,
                         torch::Tensor gate_up_scale,
                         torch::Tensor down,
                         torch::Tensor down_scale,
                         int64_t first_expert,
                         double limit,
                         std::string library_path)
      : gate_up_(std::move(gate_up)),
        gate_up_scale_(std::move(gate_up_scale)),
        down_(std::move(down)),
        down_scale_(std::move(down_scale)),
        first_expert_(first_expert),
        limit_(limit),
        library_path_(std::move(library_path)) {
    CHECK_EQ(gate_up_.dim(), 3);
    CHECK_EQ(down_.dim(), 3);
    CHECK_EQ(gate_up_.size(0), down_.size(0));
    CHECK_EQ(gate_up_.size(1), down_.size(2) * 4);
    CHECK_EQ(gate_up_.size(2) * 2, down_.size(1));
    CHECK_GE(first_expert_, 0);
    CHECK(!library_path_.empty());
    // Detailed dtype/shape/stride validation also happens at the FFI boundary.
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& ids,
                        const torch::Tensor& route_weights) const {
    const DeepseekV41ProfileRange range("V41/MoE/W4A8_routed");
    auto [dispatched, input_scale, counts, route_rows] =
        kernel::musa::deepseek_v41_w4a8_dispatch(input.contiguous(),
                                                 ids.contiguous(),
                                                 first_expert_,
                                                 gate_up_.size(0));
    const torch::Tensor gate_up =
        kernel::musa::deepseek_v41_w4a8_masked_gemm(dispatched,
                                                    input_scale,
                                                    gate_up_,
                                                    gate_up_scale_,
                                                    counts,
                                                    library_path_);
    auto [activation, activation_scale] =
        kernel::musa::deepseek_v41_w4a8_swiglu_quant(gate_up, counts, limit_);
    const torch::Tensor down =
        kernel::musa::deepseek_v41_w4a8_masked_gemm(activation,
                                                    activation_scale,
                                                    down_,
                                                    down_scale_,
                                                    counts,
                                                    library_path_);
    return kernel::musa::deepseek_v41_w4a8_combine(down,
                                                   ids.contiguous(),
                                                   route_rows,
                                                   route_weights.contiguous(),
                                                   first_expert_);
  }

 private:
  torch::Tensor gate_up_;
  torch::Tensor gate_up_scale_;
  torch::Tensor down_;
  torch::Tensor down_scale_;
  int64_t first_expert_;
  double limit_;
  std::string library_path_;
};

}  // namespace xllm::layer
