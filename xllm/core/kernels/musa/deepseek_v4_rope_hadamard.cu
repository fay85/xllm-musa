/* Copyright 2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 */

// MUSA-only port of SGLang's DeepSeek-V4 C4 indexer fast path.  It fuses the
// interleaved 64-wide RoPE tail with the normalized 128-wide Hadamard transform
// and updates the freshly produced query in place.

#include <c10/util/BFloat16.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeDim = 64;
constexpr int32_t kThreads = 16;
constexpr int32_t kItemsPerThread = 8;
constexpr float kHadamardScale = 0.08838834764831845f;

template <typename ScalarType, int32_t HeadsPerBlock>
__global__ void deepseek_v4_rope_hadamard_kernel(
    ScalarType* __restrict__ query,
    const ScalarType* __restrict__ cos,
    const ScalarType* __restrict__ sin,
    int64_t num_heads,
    int64_t cos_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.y);
  const int32_t head_lane = static_cast<int32_t>(threadIdx.x) / kThreads;
  const int64_t head =
      static_cast<int64_t>(blockIdx.x) * HeadsPerBlock + head_lane;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kThreads;
  const int32_t base = lane * kItemsPerThread;
  ScalarType* row = query + (token * num_heads + head) * kHeadDim;
  const ScalarType* cos_row = cos + token * cos_stride;
  const ScalarType* sin_row = sin + token * cos_stride;

  float value[kItemsPerThread];
#pragma unroll
  for (int32_t item = 0; item < kItemsPerThread; ++item) {
    value[item] = static_cast<float>(row[base + item]);
  }

  if (base >= kRopeDim) {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerThread; item += 2) {
      const int32_t rope_col = base + item - kRopeDim;
      const float even = value[item];
      const float odd = value[item + 1];
      const float c = static_cast<float>(cos_row[rope_col]);
      const float s = static_cast<float>(sin_row[rope_col]);
      value[item] = even * c - odd * s;
      value[item + 1] = even * s + odd * c;
    }
  }

  // The first three stages are local to the eight contiguous values owned by
  // a thread.  The remaining four stages exchange corresponding values among
  // the 16 lanes, matching the reference Sylvester H128 ordering.
  for (int32_t step = 1; step < kItemsPerThread; step <<= 1) {
#pragma unroll
    for (int32_t group = 0; group < kItemsPerThread; group += 2 * step) {
#pragma unroll
      for (int32_t offset = 0; offset < step; ++offset) {
        const float left = value[group + offset];
        const float right = value[group + step + offset];
        value[group + offset] = left + right;
        value[group + step + offset] = left - right;
      }
    }
  }

#pragma unroll
  for (int32_t stage = 0; stage < 4; ++stage) {
    const int32_t lane_mask = 1 << stage;
    const float sign = (lane & lane_mask) == 0 ? 1.0f : -1.0f;
#pragma unroll
    for (int32_t item = 0; item < kItemsPerThread; ++item) {
      const float other =
          __shfl_xor_sync(HeadsPerBlock == 1 ? 0x0000ffffu : 0xffffffffu,
                          value[item],
                          lane_mask,
                          kThreads);
      value[item] = sign * value[item] + other;
    }
  }

  if constexpr (std::is_same_v<ScalarType, c10::BFloat16>) {
    // Match SGLang's fast prefill kernel: four aligned 32-bit stores replace
    // eight scalar BF16 stores.  The conversion is unchanged, so this remains
    // bit-identical to the one-head implementation.
#pragma unroll
    for (int32_t item = 0; item < kItemsPerThread; item += 2) {
      const c10::BFloat16 first(value[item] * kHadamardScale);
      const c10::BFloat16 second(value[item + 1] * kHadamardScale);
      reinterpret_cast<uint32_t*>(row + base)[item / 2] =
          static_cast<uint32_t>(first.x) |
          (static_cast<uint32_t>(second.x) << 16);
    }
  } else {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerThread; ++item) {
      row[base + item] = static_cast<ScalarType>(value[item] * kHadamardScale);
    }
  }
}

template <int32_t HeadsPerBlock>
void launch_rope_hadamard(torch::Tensor& query,
                          const torch::Tensor& cos,
                          const torch::Tensor& sin) {
  const dim3 grid(static_cast<unsigned int>(query.size(1) / HeadsPerBlock),
                  static_cast<unsigned int>(query.size(0)));
  deepseek_v4_rope_hadamard_kernel<c10::BFloat16, HeadsPerBlock>
      <<<grid,
         kThreads * HeadsPerBlock,
         0,
         current_musa_stream(query.device())>>>(query.data_ptr<c10::BFloat16>(),
                                                cos.data_ptr<c10::BFloat16>(),
                                                sin.data_ptr<c10::BFloat16>(),
                                                query.size(1),
                                                cos.size(1));
}

}  // namespace

bool deepseek_v4_rope_hadamard_inplace(torch::Tensor& query,
                                       const torch::Tensor& cos,
                                       const torch::Tensor& sin) {
  if (!query.defined() || !cos.defined() || !sin.defined() ||
      query.device().is_cpu() || query.dim() != 3 || query.size(0) <= 0 ||
      query.size(1) <= 0 || query.size(2) != kHeadDim ||
      !query.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() ||
      cos.dim() != 2 || sin.dim() != 2 || cos.sizes() != sin.sizes() ||
      cos.size(0) != query.size(0) || cos.size(1) < kRopeDim ||
      query.device() != cos.device() || query.device() != sin.device() ||
      query.scalar_type() != torch::kBFloat16 ||
      cos.scalar_type() != query.scalar_type() ||
      sin.scalar_type() != query.scalar_type() ||
      query.size(0) > std::numeric_limits<unsigned int>::max() ||
      query.size(1) > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(query));
  if (query.size(0) >= 256 && query.size(1) % 8 == 0) {
    launch_rope_hadamard<8>(query, cos, sin);
  } else if (query.size(0) >= 128 && query.size(1) % 4 == 0) {
    launch_rope_hadamard<4>(query, cos, sin);
  } else {
    launch_rope_hadamard<1>(query, cos, sin);
  }
  check_musa_kernel_launch("DeepSeek-V4 RoPE Hadamard kernel");
  return true;
}

}  // namespace xllm::kernel::musa
