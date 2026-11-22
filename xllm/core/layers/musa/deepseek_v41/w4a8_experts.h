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
// forward() returns FP32 local routed output for the owner's EP reduction.
// forward_dispatched() instead returns unweighted expert-major BF16 for
// DeepEP combine; the owner must not apply a second routed EP all-reduce.
class DeepseekV41W4A8Experts final {
 public:
  DeepseekV41W4A8Experts(torch::Tensor gate_up,
                         torch::Tensor gate_up_scale,
                         torch::Tensor down,
                         torch::Tensor down_scale,
                         int64_t first_expert,
                         double limit,
                         std::string library_path,
                         double gate_up_input_scale = 0.0,
                         double down_input_scale = 0.0,
                         bool valid_prefix_swiglu = false,
                         bool prefill_valid_prefix_swiglu = false,
                         bool prefill_parallel_swiglu = false)
      : gate_up_(std::move(gate_up)),
        gate_up_scale_(std::move(gate_up_scale)),
        down_(std::move(down)),
        down_scale_(std::move(down_scale)),
        first_expert_(first_expert),
        limit_(limit),
        library_path_(std::move(library_path)),
        gate_up_input_scale_(gate_up_input_scale),
        down_input_scale_(down_input_scale),
        valid_prefix_swiglu_(valid_prefix_swiglu),
        prefill_valid_prefix_swiglu_(prefill_valid_prefix_swiglu),
        prefill_parallel_swiglu_(prefill_parallel_swiglu) {
    CHECK_EQ(gate_up_input_scale_ > 0.0, down_input_scale_ > 0.0);
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
    auto dispatched =
        kernel::musa::deepseek_v41_w4a8_dispatch(input.contiguous(),
                                                 ids.contiguous(),
                                                 first_expert_,
                                                 gate_up_.size(0),
                                                 gate_up_input_scale_);
    return combine(dispatched,
                   ids,
                   route_weights,
                   (valid_prefix_swiglu_ && input.size(0) <= 8) ||
                       (prefill_valid_prefix_swiglu_ && input.size(0) > 8),
                   prefill_parallel_swiglu_ && input.size(0) > 8);
  }

  torch::Tensor forward_compact(const torch::Tensor& input,
                                const torch::Tensor& ids,
                                const torch::Tensor& route_weights) const {
    const DeepseekV41ProfileRange range("V41/MoE/W4A8_compact");
    CHECK_GT(input.size(0), 512);
    auto dispatched =
        kernel::musa::deepseek_v41_w4a8_dispatch_compact(input.contiguous(),
                                                         ids.contiguous(),
                                                         first_expert_,
                                                         gate_up_.size(0),
                                                         gate_up_input_scale_);
    if (!dispatched.has_value()) {
      LOG_FIRST_N(INFO, 1)
          << "V4.1 wide P expert occupancy exceeds 512 rows for "
          << input.size(0) << " input rows; using the original dispatch tiles";
      return {};
    }
    LOG_FIRST_N(INFO, 1) << "V4.1 compact wide P dispatch: input rows="
                         << input.size(0) << ", expert capacity="
                         << std::get<0>(*dispatched).size(1);
    const DeepseekV41ProfileRange compact("V41/MoE/W4A8_compact_active");
    return combine(dispatched.value(),
                   ids,
                   route_weights,
                   prefill_valid_prefix_swiglu_,
                   prefill_parallel_swiglu_);
  }

  // BF16 [local_experts, padded_capacity, hidden] from DeepEP dispatch.
  // Returns unweighted expert-major output for DeepEP combine; there is no
  // local re-routing, FP32 EP all-reduce, or shared-expert addition here.
  torch::Tensor forward_dispatched(const torch::Tensor& input,
                                   const torch::Tensor& counts) const {
    const DeepseekV41ProfileRange range("V41/MoE/W4A8_deepep_experts");
    CHECK_EQ(input.dim(), 3);
    CHECK_EQ(input.size(0), gate_up_.size(0));
    CHECK_EQ(input.size(2), gate_up_.size(2) * 2);
    auto [quantized, scales] = kernel::musa::deepseek_v41_w4a8_quant_dispatched(
        input, counts, gate_up_input_scale_);
    return project_dispatched(quantized, scales, counts, valid_prefix_swiglu_);
  }

 private:
  torch::Tensor combine(const kernel::musa::DeepseekV41W4A8Dispatch& routes,
                        const torch::Tensor& ids,
                        const torch::Tensor& route_weights,
                        bool valid_prefix,
                        bool parallel_prefill) const {
    const auto& [dispatched, input_scale, counts, route_rows] = routes;
    const torch::Tensor down = project_dispatched(
        dispatched, input_scale, counts, valid_prefix, parallel_prefill);
    return kernel::musa::deepseek_v41_w4a8_combine(down,
                                                   ids.contiguous(),
                                                   route_rows,
                                                   route_weights.contiguous(),
                                                   first_expert_);
  }

  torch::Tensor project_dispatched(const torch::Tensor& dispatched,
                                   const torch::Tensor& input_scale,
                                   const torch::Tensor& counts,
                                   bool valid_prefix,
                                   bool parallel_prefill = false) const {
    const torch::Tensor gate_up =
        kernel::musa::deepseek_v41_w4a8_masked_gemm(dispatched,
                                                    input_scale,
                                                    gate_up_,
                                                    gate_up_scale_,
                                                    counts,
                                                    library_path_,
                                                    gate_up_input_scale_ > 0.0);
    auto [activation, activation_scale] =
        parallel_prefill
            ? kernel::musa::deepseek_v41_w4a8_swiglu_quant_prefill_parallel(
                  gate_up, counts, limit_, down_input_scale_)
            : kernel::musa::deepseek_v41_w4a8_swiglu_quant(
                  gate_up, counts, limit_, down_input_scale_, valid_prefix);
    return kernel::musa::deepseek_v41_w4a8_masked_gemm(activation,
                                                       activation_scale,
                                                       down_,
                                                       down_scale_,
                                                       counts,
                                                       library_path_,
                                                       down_input_scale_ > 0.0);
  }

  torch::Tensor gate_up_;
  torch::Tensor gate_up_scale_;
  torch::Tensor down_;
  torch::Tensor down_scale_;
  int64_t first_expert_;
  double limit_;
  std::string library_path_;
  double gate_up_input_scale_ = 0.0;
  double down_input_scale_ = 0.0;
  bool valid_prefix_swiglu_ = false;
  // Only launch valid expert rows; keep the physical GEMM padding and counts.
  // The dispatcher enforces input <= 512 before selecting this kernel.
  bool prefill_valid_prefix_swiglu_ = false;
  bool prefill_parallel_swiglu_ = false;
};

}  // namespace xllm::layer
