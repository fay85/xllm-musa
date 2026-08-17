/* Copyright 2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 */

#include <ATen/Dispatch.h>
#include <glog/logging.h>

#include <cstdint>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kWarpsPerBlock = 8;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kThreadsPerBlock = kWarpsPerBlock * kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float stable_sigmoid(float value) {
  const bool positive = value >= 0.0f;
  const float exp_value = expf(positive ? -value : value);
  const float inverse = 1.0f / (1.0f + exp_value);
  return positive ? inverse : exp_value * inverse;
}

// This follows SGLang's MUSA fused_share_gate_sigmoid_mul layout. xLLM also
// folds the final routed + shared update into the same store.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreadsPerBlock)
    deepseek_v4_shared_expert_gate_kernel(
        scalar_t* __restrict__ routed_output,
        const scalar_t* __restrict__ hidden_states,
        const scalar_t* __restrict__ gate_weight,
        const scalar_t* __restrict__ shared_output,
        int64_t num_tokens,
        int64_t hidden_size) {
  constexpr int32_t kVectorElements = 16 / sizeof(scalar_t);
  const int32_t warp = threadIdx.x / kWarpSize;
  const int32_t lane = threadIdx.x % kWarpSize;
  const int64_t token =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (token >= num_tokens) {
    return;
  }

  const int64_t vectors_per_row = hidden_size / kVectorElements;
  float gate_sum = 0.0f;
  for (int64_t vector = lane; vector < vectors_per_row; vector += kWarpSize) {
    const int64_t offset = token * hidden_size + vector * kVectorElements;
    const int4 hidden_vector =
        *reinterpret_cast<const int4*>(hidden_states + offset);
    const int4 weight_vector =
        *reinterpret_cast<const int4*>(gate_weight + vector * kVectorElements);
    const auto* hidden_values =
        reinterpret_cast<const scalar_t*>(&hidden_vector);
    const auto* weight_values =
        reinterpret_cast<const scalar_t*>(&weight_vector);
#pragma unroll
    for (int32_t i = 0; i < kVectorElements; ++i) {
      gate_sum += static_cast<float>(hidden_values[i]) *
                  static_cast<float>(weight_values[i]);
    }
  }
  gate_sum = warp_sum(gate_sum);
  const float gate =
      stable_sigmoid(__shfl_sync(0xffffffffu, gate_sum, /*src_lane=*/0));

  for (int64_t vector = lane; vector < vectors_per_row; vector += kWarpSize) {
    const int64_t offset = token * hidden_size + vector * kVectorElements;
    const int4 shared_vector =
        *reinterpret_cast<const int4*>(shared_output + offset);
    const int4 routed_vector =
        *reinterpret_cast<const int4*>(routed_output + offset);
    int4 result_vector;
    const auto* shared_values =
        reinterpret_cast<const scalar_t*>(&shared_vector);
    const auto* routed_values =
        reinterpret_cast<const scalar_t*>(&routed_vector);
    auto* result_values = reinterpret_cast<scalar_t*>(&result_vector);
#pragma unroll
    for (int32_t i = 0; i < kVectorElements; ++i) {
      // Match the original BF16/FP16 mul followed by add rounding order.
      const scalar_t gated_shared =
          static_cast<scalar_t>(static_cast<float>(shared_values[i]) * gate);
      result_values[i] =
          static_cast<scalar_t>(static_cast<float>(routed_values[i]) +
                                static_cast<float>(gated_shared));
    }
    *reinterpret_cast<int4*>(routed_output + offset) = result_vector;
  }
}

}  // namespace

torch::Tensor deepseek_v4_shared_expert_gate(const torch::Tensor& hidden_states,
                                             const torch::Tensor& gate_weight,
                                             const torch::Tensor& shared_output,
                                             torch::Tensor routed_output) {
  CHECK(hidden_states.defined() && gate_weight.defined() &&
        shared_output.defined() && routed_output.defined());
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(gate_weight.dim(), 2);
  CHECK_EQ(shared_output.dim(), 2);
  CHECK_EQ(routed_output.dim(), 2);
  CHECK_EQ(gate_weight.size(0), 1);
  CHECK_EQ(hidden_states.sizes(), shared_output.sizes());
  CHECK_EQ(hidden_states.sizes(), routed_output.sizes());
  CHECK_EQ(hidden_states.size(1), gate_weight.size(1));
  CHECK_EQ(hidden_states.size(1) % 8, 0);
  CHECK(hidden_states.is_contiguous() && gate_weight.is_contiguous() &&
        shared_output.is_contiguous() && routed_output.is_contiguous());
  CHECK_EQ(hidden_states.device(), gate_weight.device());
  CHECK_EQ(hidden_states.device(), shared_output.device());
  CHECK_EQ(hidden_states.device(), routed_output.device());
  CHECK_EQ(hidden_states.scalar_type(), gate_weight.scalar_type());
  CHECK_EQ(hidden_states.scalar_type(), shared_output.scalar_type());
  CHECK_EQ(hidden_states.scalar_type(), routed_output.scalar_type());
  CHECK(hidden_states.scalar_type() == torch::kBFloat16 ||
        hidden_states.scalar_type() == torch::kFloat16);
  const auto is_vector_aligned = [](const torch::Tensor& tensor) {
    return reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) %
               alignof(int4) ==
           0;
  };
  CHECK(is_vector_aligned(hidden_states) && is_vector_aligned(gate_weight) &&
        is_vector_aligned(shared_output) && is_vector_aligned(routed_output));

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  if (num_tokens == 0 || hidden_size == 0) {
    return routed_output;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(hidden_states));
  const dim3 grid(static_cast<unsigned int>((num_tokens + kWarpsPerBlock - 1) /
                                            kWarpsPerBlock));
  const dim3 block(kThreadsPerBlock);
  const musaStream_t stream = current_musa_stream(hidden_states.device());
  if (hidden_states.scalar_type() == torch::kBFloat16) {
    deepseek_v4_shared_expert_gate_kernel<c10::BFloat16>
        <<<grid, block, 0, stream>>>(routed_output.data_ptr<c10::BFloat16>(),
                                     hidden_states.data_ptr<c10::BFloat16>(),
                                     gate_weight.data_ptr<c10::BFloat16>(),
                                     shared_output.data_ptr<c10::BFloat16>(),
                                     num_tokens,
                                     hidden_size);
  } else {
    deepseek_v4_shared_expert_gate_kernel<c10::Half>
        <<<grid, block, 0, stream>>>(routed_output.data_ptr<c10::Half>(),
                                     hidden_states.data_ptr<c10::Half>(),
                                     gate_weight.data_ptr<c10::Half>(),
                                     shared_output.data_ptr<c10::Half>(),
                                     num_tokens,
                                     hidden_size);
  }
  check_musa_kernel_launch("deepseek_v4_shared_expert_gate");
  return routed_output;
}

}  // namespace xllm::kernel::musa
