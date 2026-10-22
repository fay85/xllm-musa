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

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <musa_bf16.h>

#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kWarpThreads = 32;
constexpr int32_t kWarps = 4;
constexpr int32_t kThreads = kWarps * kWarpThreads;
constexpr int32_t kStages = 2;
constexpr int32_t kKBlock = 128;
constexpr int32_t kColsPerBlock = 16;
constexpr int32_t kMaxRows = 8;
constexpr int32_t kMaxKBlocks = 160;
constexpr int32_t kAccPerLane = 4;
// K-major A and B operands; E4M3 x E4M3 products accumulated in FP32.
constexpr int32_t kMmaLayoutKMajor = 1;
constexpr int32_t kMmaTypeE4m3 = 5;
// Only LM-head-sized outputs gain inside the decode loop; for narrower
// projections muDNN is as fast or faster.
constexpr int64_t kMinCols = 65536;

struct Stage {
  uint4 b_lo[2];
  uint4 b_hi[2];
  uint4 a_lo;
  uint4 a_hi;
};

struct LaneOperands {
  const uint8_t* b_rows[2];
  const uint8_t* a_row;
  bool a_valid;
};

__device__ __forceinline__ uint32_t word_of(const uint4& v, int32_t i) {
  return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}

__device__ __forceinline__ uint4 load16(const uint8_t* ptr) {
  return __ldg(reinterpret_cast<const uint4*>(ptr));
}

__device__ __forceinline__ void mma_e4m3_m8n16k16(int32_t* acc,
                                                  int32_t a,
                                                  const int32_t* b) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  __musa_wmma_m8n16k16_mma(
      acc, &a, b, acc, 0, 0, 0, 0, 0, kMmaLayoutKMajor, kMmaTypeE4m3);
#endif
}

__device__ __forceinline__ void load_stage(const LaneOperands& lane_ops,
                                           int32_t k_block,
                                           Stage& stage) {
  const int64_t k_offset = static_cast<int64_t>(k_block) * kKBlock;
#pragma unroll
  for (int32_t h = 0; h < 2; ++h) {
    stage.b_lo[h] = load16(lane_ops.b_rows[h] + k_offset);
    stage.b_hi[h] = load16(lane_ops.b_rows[h] + k_offset + kKBlock / 2);
  }
  const uint4 zero = make_uint4(0, 0, 0, 0);
  stage.a_lo = lane_ops.a_valid ? load16(lane_ops.a_row + k_offset) : zero;
  stage.a_hi =
      lane_ops.a_valid ? load16(lane_ops.a_row + k_offset + kKBlock / 2) : zero;
}

__device__ __forceinline__ void accumulate_stage(const Stage& stage,
                                                 float low_rows_scale,
                                                 float high_rows_scale,
                                                 float* acc) {
  int32_t block_acc[kAccPerLane] = {0, 0, 0, 0};
#pragma unroll
  for (int32_t step = 0; step < 8; ++step) {
    const int32_t word = step % 4;
    const int32_t a =
        static_cast<int32_t>(word_of(step < 4 ? stage.a_lo : stage.a_hi, word));
    const int32_t b[2] = {static_cast<int32_t>(word_of(
                              step < 4 ? stage.b_lo[0] : stage.b_hi[0], word)),
                          static_cast<int32_t>(word_of(
                              step < 4 ? stage.b_lo[1] : stage.b_hi[1], word))};
    mma_e4m3_m8n16k16(block_acc, a, b);
  }
  acc[0] += __int_as_float(block_acc[0]) * low_rows_scale;
  acc[1] += __int_as_float(block_acc[1]) * low_rows_scale;
  acc[2] += __int_as_float(block_acc[2]) * high_rows_scale;
  acc[3] += __int_as_float(block_acc[3]) * high_rows_scale;
}

// A block owns 16 output columns over the whole K; its warps split the
// 128-wide K blocks and each keeps kStages of them in flight. Per K block a
// lane loads the same two 16-byte runs of one activation row and of two weight
// rows, so A and B share one k permutation inside every MMA.
__global__ void __launch_bounds__(kThreads)
    fp8_skinny_gemm_kernel(const uint8_t* __restrict__ a,
                           const float* __restrict__ a_scale,
                           const uint8_t* __restrict__ b,
                           const float* __restrict__ b_scale,
                           __mt_bfloat16* __restrict__ out,
                           int32_t rows,
                           int32_t cols,
                           int32_t k) {
  __shared__ float scales[kMaxRows][kMaxKBlocks];
  __shared__ float partials[kWarps][kAccPerLane][kWarpThreads];
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kWarpThreads;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpThreads;
  const int32_t k_blocks = k / kKBlock;
  const int32_t warp_k_blocks = k_blocks / kWarps;
  const int32_t warp_k_begin = warp * warp_k_blocks;
  const int32_t col_base = static_cast<int32_t>(blockIdx.x) * kColsPerBlock;
  const int32_t quad = lane % 4;
  const int32_t frag_row = lane / 4;
  const int32_t acc_row = lane / 8;

  const float* b_scale_row =
      b_scale + static_cast<int64_t>(col_base / kKBlock) * k_blocks;
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kMaxRows * k_blocks;
       i += kThreads) {
    const int32_t row = i / k_blocks;
    const int32_t k_block = i % k_blocks;
    scales[row][k_block] =
        row < rows ? a_scale[row * k_blocks + k_block] * b_scale_row[k_block]
                   : 0.0f;
  }

  LaneOperands lane_ops;
#pragma unroll
  for (int32_t h = 0; h < 2; ++h) {
    lane_ops.b_rows[h] =
        b + static_cast<int64_t>(col_base + frag_row + 8 * h) * k + 16 * quad;
  }
  lane_ops.a_row = a + static_cast<int64_t>(frag_row) * k + 16 * quad;
  lane_ops.a_valid = frag_row < rows;

  float acc[kAccPerLane] = {0.0f, 0.0f, 0.0f, 0.0f};
  Stage stages[kStages];
#pragma unroll
  for (int32_t i = 0; i < kStages; ++i) {
    if (i < warp_k_blocks) {
      load_stage(lane_ops, warp_k_begin + i, stages[i]);
    }
  }
  __syncthreads();
  const float* low_scales = scales[acc_row] + warp_k_begin;
  const float* high_scales = scales[acc_row + 4] + warp_k_begin;
  for (int32_t base = 0; base < warp_k_blocks; base += kStages) {
#pragma unroll
    for (int32_t i = 0; i < kStages; ++i) {
      const int32_t idx = base + i;
      if (idx < warp_k_blocks) {
        accumulate_stage(stages[i], low_scales[idx], high_scales[idx], acc);
        if (idx + kStages < warp_k_blocks) {
          load_stage(lane_ops, warp_k_begin + idx + kStages, stages[i]);
        }
      }
    }
  }

#pragma unroll
  for (int32_t i = 0; i < kAccPerLane; ++i) {
    partials[warp][i][lane] = acc[i];
  }
  __syncthreads();
  if (warp != 0) {
    return;
  }
#pragma unroll
  for (int32_t i = 0; i < kAccPerLane; ++i) {
    float total = 0.0f;
#pragma unroll
    for (int32_t w = 0; w < kWarps; ++w) {
      total += partials[w][i][lane];
    }
    const int32_t row = acc_row + (i >= 2 ? 4 : 0);
    const int32_t col = col_base + lane % 8 + (i % 2 == 1 ? 8 : 0);
    if (row < rows) {
      out[static_cast<int64_t>(row) * cols + col] = __float2bfloat16_rn(total);
    }
  }
}

bool is_16b_aligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignof(uint4) == 0;
}

bool is_mp31_device() {
  static const bool mp31 = [] {
    int32_t device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.major == 3 && prop.minor == 1;
  }();
  return mp31;
}

}  // namespace

bool fp8_skinny_gemm_supported(const torch::Tensor& a,
                               const torch::Tensor& b,
                               const torch::Tensor& a_scale,
                               const torch::Tensor& b_scale,
                               const torch::Tensor& out) {
  if (a.dim() != 2 || b.dim() != 2 || a_scale.dim() != 2 ||
      b_scale.dim() != 2 || out.dim() != 2) {
    return false;
  }
  if (a.scalar_type() != torch::kFloat8_e4m3fn ||
      b.scalar_type() != torch::kFloat8_e4m3fn ||
      a_scale.scalar_type() != torch::kFloat32 ||
      b_scale.scalar_type() != torch::kFloat32 ||
      out.scalar_type() != torch::kBFloat16) {
    return false;
  }
  const int64_t rows = a.size(0);
  const int64_t k = a.size(1);
  const int64_t cols = b.size(0);
  const int64_t k_blocks = k / kKBlock;
  return rows >= 1 && rows <= kMaxRows && b.size(1) == k && k % kKBlock == 0 &&
         k_blocks % kWarps == 0 && k_blocks <= kMaxKBlocks &&
         cols % kColsPerBlock == 0 && a_scale.size(0) == rows &&
         a_scale.size(1) == k_blocks &&
         b_scale.size(0) == (cols + kKBlock - 1) / kKBlock &&
         b_scale.size(1) == k_blocks && out.size(0) == rows &&
         out.size(1) == cols && a.is_contiguous() && b.is_contiguous() &&
         a_scale.is_contiguous() && b_scale.is_contiguous() &&
         out.is_contiguous() && is_16b_aligned(a) && is_16b_aligned(b) &&
         cols >= kMinCols && is_mp31_device();
}

void fp8_skinny_gemm(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const torch::Tensor& a_scale,
                     const torch::Tensor& b_scale,
                     torch::Tensor& out) {
  CHECK(fp8_skinny_gemm_supported(a, b, a_scale, b_scale, out))
      << "unsupported skinny FP8 GEMM: a=" << a.sizes() << ", b=" << b.sizes()
      << ", out=" << out.sizes();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int32_t cols = static_cast<int32_t>(b.size(0));
  fp8_skinny_gemm_kernel<<<cols / kColsPerBlock, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(a.data_ptr()),
      a_scale.data_ptr<float>(),
      static_cast<const uint8_t*>(b.data_ptr()),
      b_scale.data_ptr<float>(),
      reinterpret_cast<__mt_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int32_t>(a.size(0)),
      cols,
      static_cast<int32_t>(a.size(1)));
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace xllm::kernel::musa
