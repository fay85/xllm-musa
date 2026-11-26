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
    const ScalarType* __restrict__ query,
    ScalarType* __restrict__ output,
    const ScalarType* __restrict__ cos,
    const ScalarType* __restrict__ sin,
    int64_t num_heads,
    int64_t query_token_stride,
    int64_t query_head_stride,
    int64_t cos_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.y);
  const int32_t head_lane = static_cast<int32_t>(threadIdx.x) / kThreads;
  const int64_t head =
      static_cast<int64_t>(blockIdx.x) * HeadsPerBlock + head_lane;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kThreads;
  const int32_t base = lane * kItemsPerThread;
  const ScalarType* query_row =
      query + token * query_token_stride + head * query_head_stride;
  ScalarType* output_row =
      output + (token * num_heads + head) * kHeadDim;
  const ScalarType* cos_row = cos + token * cos_stride;
  const ScalarType* sin_row = sin + token * cos_stride;

  float value[kItemsPerThread];
#pragma unroll
  for (int32_t item = 0; item < kItemsPerThread; ++item) {
    value[item] = static_cast<float>(query_row[base + item]);
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
      reinterpret_cast<uint32_t*>(output_row + base)[item / 2] =
          static_cast<uint32_t>(first.x) |
          (static_cast<uint32_t>(second.x) << 16);
    }
  } else {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerThread; ++item) {
      output_row[base + item] =
          static_cast<ScalarType>(value[item] * kHadamardScale);
    }
  }
}

template <int32_t HeadsPerBlock>
void launch_rope_hadamard(const torch::Tensor& query,
                          torch::Tensor& output,
                          const torch::Tensor& cos,
                          const torch::Tensor& sin) {
  const int64_t num_heads = query.dim() == 2 ? 1 : query.size(1);
  const dim3 grid(static_cast<unsigned int>(num_heads / HeadsPerBlock),
                  static_cast<unsigned int>(query.size(0)));
  deepseek_v4_rope_hadamard_kernel<c10::BFloat16, HeadsPerBlock>
      <<<grid,
         kThreads * HeadsPerBlock,
         0,
         current_musa_stream(query.device())>>>(query.data_ptr<c10::BFloat16>(),
                                                output.data_ptr<c10::BFloat16>(),
                                                cos.data_ptr<c10::BFloat16>(),
                                                sin.data_ptr<c10::BFloat16>(),
                                                num_heads,
                                                query.stride(0),
                                                query.dim() == 2
                                                    ? query.size(1)
                                                    : query.stride(1),
                                                cos.size(1));
}

bool matches_rope_hadamard_contract(const torch::Tensor& query,
                                    const torch::Tensor& cos,
                                    const torch::Tensor& sin) {
  const bool query_shape_ok =
      query.defined() &&
      ((query.dim() == 2 && query.size(1) == kHeadDim) ||
       (query.dim() == 3 && query.size(1) > 0 &&
        query.size(2) == kHeadDim));
  const int64_t num_heads =
      query_shape_ok ? (query.dim() == 2 ? 1 : query.size(1)) : 0;
  return query_shape_ok && cos.defined() && sin.defined() &&
         !query.device().is_cpu() && query.size(0) > 0 &&
         query.stride(-1) == 1 && cos.is_contiguous() &&
         sin.is_contiguous() && cos.dim() == 2 && sin.dim() == 2 &&
         cos.sizes() == sin.sizes() && cos.size(0) == query.size(0) &&
         cos.size(1) >= kRopeDim && query.device() == cos.device() &&
         query.device() == sin.device() &&
         query.scalar_type() == torch::kBFloat16 &&
         cos.scalar_type() == query.scalar_type() &&
         sin.scalar_type() == query.scalar_type() &&
         query.size(0) <= std::numeric_limits<unsigned int>::max() &&
         num_heads <= std::numeric_limits<unsigned int>::max();
}

template <int32_t RowsPerBlock, bool ApplyRope>
__global__ void deepseek_v4_hadamard_2d_kernel(
    const c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ cos,
    const c10::BFloat16* __restrict__ sin,
    int64_t num_rows,
    int64_t cos_stride) {
  const int32_t row_lane = static_cast<int32_t>(threadIdx.x) / kThreads;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kThreads;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * RowsPerBlock + row_lane;
  const int32_t base = lane * kItemsPerThread;
  const bool valid = row < num_rows;

  float value[kItemsPerThread];
  if (valid) {
    const uint32_t* in_u32 =
        reinterpret_cast<const uint32_t*>(input + row * kHeadDim + base);
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread / 2; ++i) {
      const uint32_t packed = in_u32[i];
      c10::BFloat16 lo, hi;
      lo.x = static_cast<uint16_t>(packed & 0xffffu);
      hi.x = static_cast<uint16_t>(packed >> 16);
      value[i * 2] = static_cast<float>(lo);
      value[i * 2 + 1] = static_cast<float>(hi);
    }
    if constexpr (ApplyRope) {
      if (base >= kRopeDim) {
        const c10::BFloat16* cos_row = cos + row * cos_stride;
        const c10::BFloat16* sin_row = sin + row * cos_stride;
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
    }
  } else {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerThread; ++item) {
      value[item] = 0.0f;
    }
  }

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
          __shfl_xor_sync(0xffffffffu, value[item], lane_mask, kThreads);
      value[item] = sign * value[item] + other;
    }
  }

  if (valid) {
    uint32_t* out_u32 =
        reinterpret_cast<uint32_t*>(output + row * kHeadDim + base);
#pragma unroll
    for (int32_t item = 0; item < kItemsPerThread; item += 2) {
      const c10::BFloat16 first(value[item] * kHadamardScale);
      const c10::BFloat16 second(value[item + 1] * kHadamardScale);
      out_u32[item / 2] = static_cast<uint32_t>(first.x) |
                          (static_cast<uint32_t>(second.x) << 16);
    }
  }
}

void dispatch_rope_hadamard(const torch::Tensor& query,
                            torch::Tensor& output,
                            const torch::Tensor& cos,
                            const torch::Tensor& sin) {
  const int64_t num_heads = query.dim() == 2 ? 1 : query.size(1);
  if (query.dim() == 2 && query.is_contiguous() && query.size(0) >= 16) {
    constexpr int32_t kRowsPerBlock = 8;
    const int64_t blocks = (query.size(0) + kRowsPerBlock - 1) / kRowsPerBlock;
    deepseek_v4_hadamard_2d_kernel<kRowsPerBlock, true>
        <<<static_cast<unsigned int>(blocks),
           kThreads * kRowsPerBlock,
           0,
           current_musa_stream(query.device())>>>(
            query.data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(),
            cos.data_ptr<c10::BFloat16>(),
            sin.data_ptr<c10::BFloat16>(),
            query.size(0),
            cos.size(1));
    return;
  }
  if (query.size(0) >= 256 && num_heads % 8 == 0) {
    launch_rope_hadamard<8>(query, output, cos, sin);
  } else if (query.size(0) >= 128 && num_heads % 4 == 0) {
    launch_rope_hadamard<4>(query, output, cos, sin);
  } else {
    launch_rope_hadamard<1>(query, output, cos, sin);
  }
}

}  // namespace

bool deepseek_v4_rope_hadamard_inplace(torch::Tensor& query,
                                       const torch::Tensor& cos,
                                       const torch::Tensor& sin) {
  if (!matches_rope_hadamard_contract(query, cos, sin) ||
      !query.is_contiguous()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(query));
  dispatch_rope_hadamard(query, query, cos, sin);
  check_musa_kernel_launch("DeepSeek-V4 RoPE Hadamard kernel");
  return true;
}

bool deepseek_v4_rope_hadamard_out(const torch::Tensor& query,
                                   const torch::Tensor& cos,
                                   const torch::Tensor& sin,
                                   torch::Tensor& output) {
  if (!matches_rope_hadamard_contract(query, cos, sin) ||
      !output.defined() || output.sizes() != query.sizes() ||
      output.device() != query.device() ||
      output.scalar_type() != query.scalar_type() || !output.is_contiguous()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(query));
  dispatch_rope_hadamard(query, output, cos, sin);
  check_musa_kernel_launch("DeepSeek-V4 out-of-place RoPE Hadamard kernel");
  return true;
}

bool deepseek_v4_hadamard_out(const torch::Tensor& input,
                              torch::Tensor& output) {
  if (!input.defined() || !output.defined() || input.device().is_cpu() ||
      input.numel() == 0 || input.size(-1) != kHeadDim ||
      input.scalar_type() != torch::kBFloat16 ||
      output.scalar_type() != torch::kBFloat16 ||
      input.device() != output.device() ||
      input.sizes() != output.sizes() ||
      !input.is_contiguous() || !output.is_contiguous()) {
    return false;
  }
  const int64_t total_rows = input.numel() / kHeadDim;
  if (total_rows <= 0 ||
      total_rows > std::numeric_limits<unsigned int>::max()) {
    return false;
  }
  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  constexpr int32_t kRowsPerBlock = 8;
  const int64_t blocks = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
  deepseek_v4_hadamard_2d_kernel<kRowsPerBlock, false>
      <<<static_cast<unsigned int>(blocks),
         kThreads * kRowsPerBlock,
         0,
         current_musa_stream(input.device())>>>(
          input.data_ptr<c10::BFloat16>(),
          output.data_ptr<c10::BFloat16>(),
          nullptr,
          nullptr,
          total_rows,
          0);
  check_musa_kernel_launch("DeepSeek-V4 Hadamard H128 kernel");
  return true;
}

}  // namespace xllm::kernel::musa
