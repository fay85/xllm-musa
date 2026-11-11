/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <glog/logging.h>
#include <musa_fp8.h>

#include <algorithm>
#include <cstdint>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kQueryCastThreads = 256;
constexpr int64_t kQueryCastMaxBlocks = 4096;
constexpr int32_t kPackedQueryHeadDim = 256;
constexpr int32_t kPackedQueryElements = 4;
constexpr int32_t kPackedQueryHeadThreads =
    kPackedQueryHeadDim / kPackedQueryElements;
constexpr int32_t kPackedQueryLargeHeadsPerBlock = 4;
constexpr int64_t kPackedQuerySmallTokenLimit = 16;

__device__ __forceinline__ uint32_t
fix_fp8_exception_bytes(uint32_t result, uint64_t input_bits) {
  // Hardware SATFINITE conversion is fast but clamps overflow. Restore c10's
  // unscaled E4M3FN behavior: values above the 464 tie, Inf and NaN become a
  // signed overflow NaN. Input NaNs canonicalize to positive NaN like the
  // legacy torch_musa cast; the BF16 value 464 rounds to finite 448.
  const uint32_t value0 = static_cast<uint32_t>(input_bits) & 0xffffu;
  const uint32_t value1 = static_cast<uint32_t>(input_bits >> 16) & 0xffffu;
  const uint32_t value2 = static_cast<uint32_t>(input_bits >> 32) & 0xffffu;
  const uint32_t value3 = static_cast<uint32_t>(input_bits >> 48);
  const uint32_t mask0 = (value0 & 0x7fffu) > 0x43e8u ? 0xffu : 0u;
  const uint32_t mask1 = (value1 & 0x7fffu) > 0x43e8u ? 0xff00u : 0u;
  const uint32_t mask2 = (value2 & 0x7fffu) > 0x43e8u ? 0xff0000u : 0u;
  const uint32_t mask3 = (value3 & 0x7fffu) > 0x43e8u ? 0xff000000u : 0u;
  const uint32_t mask = mask0 | mask1 | mask2 | mask3;
  const uint32_t sign0 =
      (value0 & 0x7fffu) > 0x7f80u ? 0u : ((value0 >> 8) & 0x80u);
  const uint32_t sign1 =
      (value1 & 0x7fffu) > 0x7f80u ? 0u : ((value1 >> 8) & 0x80u);
  const uint32_t sign2 =
      (value2 & 0x7fffu) > 0x7f80u ? 0u : ((value2 >> 8) & 0x80u);
  const uint32_t sign3 =
      (value3 & 0x7fffu) > 0x7f80u ? 0u : ((value3 >> 8) & 0x80u);
  const uint32_t nan_bytes = (sign0 | 0x7fu) | ((sign1 | 0x7fu) << 8) |
                             ((sign2 | 0x7fu) << 16) | ((sign3 | 0x7fu) << 24);
  return (result & ~mask) | (nan_bytes & mask);
}

template <int32_t HeadsPerBlock>
__global__ void fp8_query_cast_packed_kernel(
    const c10::BFloat16* __restrict__ query,
    c10::Float8_e4m3fn* __restrict__ output,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t token_stride,
    int64_t head_stride) {
  const int64_t head =
      static_cast<int64_t>(blockIdx.x) * HeadsPerBlock +
      static_cast<int32_t>(threadIdx.x) / kPackedQueryHeadThreads;
  if (head >= num_heads) {
    return;
  }
  const int64_t dim =
      (static_cast<int32_t>(threadIdx.x) % kPackedQueryHeadThreads) *
      kPackedQueryElements;
  for (int64_t token = static_cast<int64_t>(blockIdx.y); token < num_tokens;
       token += static_cast<int64_t>(gridDim.y)) {
    const int64_t input_offset =
        token * token_stride + head * head_stride + dim;
    const uint64_t input_bits =
        *reinterpret_cast<const uint64_t*>(query + input_offset);
    const uint32_t lower = static_cast<uint32_t>(input_bits);
    const uint32_t upper = static_cast<uint32_t>(input_bits >> 32);
    const float4 values = make_float4(__uint_as_float(lower << 16),
                                      __uint_as_float(lower & 0xffff0000u),
                                      __uint_as_float(upper << 16),
                                      __uint_as_float(upper & 0xffff0000u));
    const uint32_t packed = static_cast<uint32_t>(
        __musa_cvt_float4_to_fp8x4(values, __MT_SATFINITE, __MT_E4M3));
    const int64_t output_offset =
        (token * num_heads + head) * kPackedQueryHeadDim + dim;
    *reinterpret_cast<uint32_t*>(output + output_offset) =
        fix_fp8_exception_bytes(packed, input_bits);
  }
}

template <int32_t HeadsPerBlock>
void launch_packed_query_cast(const torch::Tensor& query,
                              torch::Tensor& output) {
  const int64_t num_heads = query.size(1);
  const uint32_t head_blocks =
      static_cast<uint32_t>((num_heads + HeadsPerBlock - 1) / HeadsPerBlock);
  const uint32_t token_blocks =
      static_cast<uint32_t>(std::min(query.size(0), kQueryCastMaxBlocks));
  const dim3 grid(head_blocks, token_blocks);
  fp8_query_cast_packed_kernel<HeadsPerBlock>
      <<<grid,
         kPackedQueryHeadThreads * HeadsPerBlock,
         0,
         current_musa_stream(query.device())>>>(
          query.data_ptr<c10::BFloat16>(),
          output.data_ptr<c10::Float8_e4m3fn>(),
          query.size(0),
          num_heads,
          query.stride(0),
          query.stride(1));
}

__global__ void fp8_query_cast_kernel(const c10::BFloat16* __restrict__ query,
                                      c10::Float8_e4m3fn* __restrict__ output,
                                      int64_t num_rows,
                                      int64_t num_heads,
                                      int64_t head_dim,
                                      int64_t token_stride,
                                      int64_t head_stride,
                                      int64_t dim_stride) {
  // A row is one query head. Read the projection view directly and write a
  // contiguous FP8 row without an intermediate contiguous BF16 allocation.
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < num_rows;
       row += static_cast<int64_t>(gridDim.x)) {
    const int64_t token = row / num_heads;
    const int64_t head = row % num_heads;
    const int64_t input_base = token * token_stride + head * head_stride;
    const int64_t output_base = row * head_dim;
    for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < head_dim;
         dim += static_cast<int64_t>(blockDim.x)) {
      // Use the same c10 conversion as the existing unscaled FP8 cache cast.
      // In particular, do not silently introduce finite-value saturation.
      output[output_base + dim] = static_cast<c10::Float8_e4m3fn>(
          static_cast<float>(query[input_base + dim * dim_stride]));
    }
  }
}

}  // namespace

torch::Tensor query_cast_fp8_e4m3(const torch::Tensor& query) {
  CHECK(query.defined());
  CHECK(query.device().is_privateuseone())
      << "query_cast_fp8_e4m3 requires a MUSA tensor";
  CHECK_EQ(query.dim(), 3)
      << "query_cast_fp8_e4m3 expects [tokens, heads, dim]";
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "query_cast_fp8_e4m3 requires BF16 input";
  const XLLMDeviceGuard device_guard(query.device());
  torch::Tensor output =
      torch::empty(query.sizes(), query.options().dtype(torch::kFloat8_e4m3fn));
  if (query.numel() == 0) {
    return output;
  }

  // Vector loads require aligned four-BF16 groups. Projected Q with head256
  // satisfies these conditions even when tokens or heads are interleaved.
  const bool packed_supported =
      query.size(2) == kPackedQueryHeadDim && query.stride(2) == 1 &&
      query.stride(0) % kPackedQueryElements == 0 &&
      query.stride(1) % kPackedQueryElements == 0 &&
      reinterpret_cast<uintptr_t>(query.data_ptr<c10::BFloat16>()) %
              (sizeof(c10::BFloat16) * kPackedQueryElements) ==
          0;
  if (packed_supported) {
    if (query.size(0) <= kPackedQuerySmallTokenLimit) {
      launch_packed_query_cast<1>(query, output);
    } else {
      launch_packed_query_cast<kPackedQueryLargeHeadsPerBlock>(query, output);
    }
    check_musa_kernel_launch("query_cast_fp8_e4m3_packed");
    return output;
  }

  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t num_rows = query.size(0) * num_heads;
  const uint32_t num_blocks =
      static_cast<uint32_t>(std::min(num_rows, kQueryCastMaxBlocks));
  fp8_query_cast_kernel<<<num_blocks,
                          kQueryCastThreads,
                          0,
                          current_musa_stream(query.device())>>>(
      query.data_ptr<c10::BFloat16>(),
      output.data_ptr<c10::Float8_e4m3fn>(),
      num_rows,
      num_heads,
      head_dim,
      query.stride(0),
      query.stride(1),
      query.stride(2));
  check_musa_kernel_launch("query_cast_fp8_e4m3");
  return output;
}

}  // namespace xllm::kernel::musa
