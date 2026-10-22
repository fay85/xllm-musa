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

#pragma once

#include <musa_bf16.h>

#include <cstdint>

namespace xllm::kernel::musa::skinny {

constexpr int32_t kWarpThreads = 32;
constexpr int32_t kThreads = 256;
constexpr int32_t kWarps = kThreads / kWarpThreads;
constexpr int32_t kBf16PerVector = 8;
constexpr int32_t kMaxRows = 8;
constexpr int32_t kMaxVectorsPerThread = 3;

__device__ __forceinline__ void unpack_bf16x8(const uint4& raw, float* out) {
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

// One block of kThreads threads produces kCols output columns over the whole
// K, so no split-K partials or second reduction launch are needed. Every load
// of a thread is issued before the first FMA because MUSA global loads have
// long latency and a skinny GEMM has too few blocks to hide it. The sum of a
// column does not depend on kCols.
template <int32_t kRows, int32_t kCols>
__device__ __forceinline__ void block_gemm(
    const __mt_bfloat16* __restrict__ input,
    const __mt_bfloat16* __restrict__ weight,
    __mt_bfloat16* __restrict__ output,
    int32_t col_begin,
    int32_t cols,
    int32_t k,
    float (*warp_partials)[kCols * kRows]) {
  constexpr int32_t kOutputs = kCols * kRows;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid / kWarpThreads;
  const int32_t lane = tid % kWarpThreads;
  const int32_t vectors = k / kBf16PerVector;

  uint4 weight_vec[kMaxVectorsPerThread][kCols];
  uint4 input_vec[kMaxVectorsPerThread][kRows];
#pragma unroll
  for (int32_t i = 0; i < kMaxVectorsPerThread; ++i) {
    const int32_t v = tid + i * kThreads;
    const bool active = v < vectors;
    const int64_t offset = static_cast<int64_t>(v) * kBf16PerVector;
#pragma unroll
    for (int32_t c = 0; c < kCols; ++c) {
      const int32_t col = col_begin + c;
      weight_vec[i][c] =
          active && col < cols
              ? *reinterpret_cast<const uint4*>(
                    weight + static_cast<int64_t>(col) * k + offset)
              : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int32_t r = 0; r < kRows; ++r) {
      input_vec[i][r] = active
                            ? *reinterpret_cast<const uint4*>(
                                  input + static_cast<int64_t>(r) * k + offset)
                            : make_uint4(0, 0, 0, 0);
    }
  }

  float acc[kCols][kRows];
#pragma unroll
  for (int32_t c = 0; c < kCols; ++c) {
#pragma unroll
    for (int32_t r = 0; r < kRows; ++r) {
      acc[c][r] = 0.0f;
    }
  }
#pragma unroll
  for (int32_t i = 0; i < kMaxVectorsPerThread; ++i) {
#pragma unroll
    for (int32_t c = 0; c < kCols; ++c) {
      float weight_f[kBf16PerVector];
      unpack_bf16x8(weight_vec[i][c], weight_f);
#pragma unroll
      for (int32_t r = 0; r < kRows; ++r) {
        float input_f[kBf16PerVector];
        unpack_bf16x8(input_vec[i][r], input_f);
#pragma unroll
        for (int32_t e = 0; e < kBf16PerVector; ++e) {
          acc[c][r] += weight_f[e] * input_f[e];
        }
      }
    }
  }

#pragma unroll
  for (int32_t c = 0; c < kCols; ++c) {
#pragma unroll
    for (int32_t r = 0; r < kRows; ++r) {
#pragma unroll
      for (int32_t offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
        acc[c][r] +=
            __shfl_xor_sync(0xffffffff, acc[c][r], offset, kWarpThreads);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int32_t c = 0; c < kCols; ++c) {
#pragma unroll
      for (int32_t r = 0; r < kRows; ++r) {
        warp_partials[warp][c * kRows + r] = acc[c][r];
      }
    }
  }
  __syncthreads();
  if (tid < kOutputs) {
    const int32_t col = col_begin + tid / kRows;
    const int32_t row = tid % kRows;
    if (col < cols) {
      float total = 0.0f;
#pragma unroll
      for (int32_t w = 0; w < kWarps; ++w) {
        total += warp_partials[w][tid];
      }
      output[static_cast<int64_t>(row) * cols + col] =
          __float2bfloat16_rn(total);
    }
  }
}

}  // namespace xllm::kernel::musa::skinny
