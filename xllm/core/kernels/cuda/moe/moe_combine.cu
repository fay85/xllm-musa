/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Fused MoE combine kernel — reorder + weighted sum in one pass.
// Replaces: torch::zeros + index_copy_ + view + multiply + sum
//
// Algorithm per token (each block handles one token):
//   1. For each of its topk experts, read gemm2 at flat_idx directly
//      (gemm2 is flat-index-ordered after scatter via index_copy_ with dst_src)
//   2. Multiply by router weight
//   3. Accumulate into output[token]
//
// Grid:  num_tokens (N) blocks
// Block: HIDDEN_DIM / HIDDEN_TILE threads

#include <c10/cuda/CUDAGuard.h>

#include "device_utils.cuh"
#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::cuda {

constexpr int32_t kCombineBlockSize = 256;

template <typename scalar_t>
__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize) moe_combine_kernel(
    const scalar_t* __restrict__ gemm2,       // [N*topk, H] flat-index-ordered
    const float* __restrict__ reduce_weight,  // [N, topk]
    scalar_t* __restrict__ output,            // [N, H]
    int64_t N,
    int32_t topk,
    int64_t H) {
  int64_t token_id = blockIdx.x;  // 0 .. N-1
  if (token_id >= N) return;

  int32_t tid = threadIdx.x;
  int32_t stride = kCombineBlockSize;

  // Accumulate over topk experts for this token
  for (int64_t h = tid; h < H; h += stride) {
    float acc = 0.0f;
    for (int32_t k = 0; k < topk; ++k) {
      int64_t flat_idx = token_id * topk + k;
      float w = reduce_weight[flat_idx];
      acc += w * static_cast<float>(gemm2[flat_idx * H + h]);
    }
    output[token_id * H + h] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize)
    moe_combine_indexed_kernel(
        const scalar_t* __restrict__ gemm2_sorted,  // [N*topk, H]
        const int32_t* __restrict__ sorted_positions,  // [N*topk]
        const float* __restrict__ reduce_weight,       // [N, topk]
        scalar_t* __restrict__ output,                 // [N, H]
        int64_t N,
    int32_t topk,
    int64_t gemm2_rows,
    int64_t H) {
  const int64_t token_id = blockIdx.x;
  if (token_id >= N) return;

  const int32_t tid = threadIdx.x;
  for (int64_t h = tid; h < H; h += kCombineBlockSize) {
    float acc = 0.0f;
    for (int32_t k = 0; k < topk; ++k) {
      const int64_t flat_idx = token_id * topk + k;
      const int32_t sorted_idx = sorted_positions[flat_idx];
      if (sorted_idx >= 0 && static_cast<int64_t>(sorted_idx) < gemm2_rows) {
        acc +=
            reduce_weight[flat_idx] *
            static_cast<float>(
                gemm2_sorted[static_cast<int64_t>(sorted_idx) * H + h]);
      }
    }
    output[token_id * H + h] = static_cast<scalar_t>(acc);
  }
}

union Bf16Pack8 {
  int4 vector;
  c10::BFloat16 values[8];
};

// BF16 vectorized specialization for MUSA prefill. Each thread owns eight
// adjacent hidden elements, so one expert position/weight load feeds eight
// accumulators and each expert row is read with one aligned 16-byte access.
__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize)
    moe_combine_indexed_bf16_vec8_kernel(
        const c10::BFloat16* __restrict__ gemm2_sorted,
        const int32_t* __restrict__ sorted_positions,
        const float* __restrict__ reduce_weight,
        c10::BFloat16* __restrict__ output,
        int64_t N,
        int32_t topk,
        int64_t gemm2_rows,
        int64_t H) {
  const int64_t token_id = blockIdx.x;
  if (token_id >= N) return;

  constexpr int32_t kValuesPerChunk = 8;
  const int64_t chunks_per_row = H / kValuesPerChunk;
  for (int64_t chunk_idx = threadIdx.x; chunk_idx < chunks_per_row;
       chunk_idx += blockDim.x) {
    float accumulators[kValuesPerChunk] = {};
    const int64_t column = chunk_idx * kValuesPerChunk;
    for (int32_t k = 0; k < topk; ++k) {
      const int64_t flat_idx = token_id * topk + k;
      const int32_t sorted_idx = sorted_positions[flat_idx];
      if (sorted_idx < 0 || static_cast<int64_t>(sorted_idx) >= gemm2_rows) {
        continue;
      }
      const float weight = reduce_weight[flat_idx];
      Bf16Pack8 input_pack;
      input_pack.vector = *reinterpret_cast<const int4*>(
          gemm2_sorted + static_cast<int64_t>(sorted_idx) * H + column);
#pragma unroll
      for (int32_t i = 0; i < kValuesPerChunk; ++i) {
        accumulators[i] += weight * static_cast<float>(input_pack.values[i]);
      }
    }

    Bf16Pack8 output_pack;
#pragma unroll
    for (int32_t i = 0; i < kValuesPerChunk; ++i) {
      output_pack.values[i] = static_cast<c10::BFloat16>(accumulators[i]);
    }
    *reinterpret_cast<int4*>(output + token_id * H + column) =
        output_pack.vector;
  }
}

// ---- Host-side orchestrator ----
torch::Tensor moe_combine_result(
    const torch::Tensor& gemm2,          // [N*topk, H] flat-index-ordered
    const torch::Tensor& reduce_weight,  // [N, topk] float or same as gemm2
    int64_t N,
    int32_t topk) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t H = gemm2.size(1);
  auto dtype = gemm2.scalar_type();

  auto output = torch::empty({N, H}, gemm2.options());
  auto rw = reduce_weight.to(gemm2.device(), torch::kFloat32).contiguous();

  if (dtype == torch::kFloat16) {
    moe_combine_kernel<c10::Half>
        <<<N, kCombineBlockSize, 0, stream>>>(gemm2.data_ptr<c10::Half>(),
                                              rw.data_ptr<float>(),
                                              output.data_ptr<c10::Half>(),
                                              N,
                                              topk,
                                              H);
  } else if (dtype == torch::kBFloat16) {
    moe_combine_kernel<c10::BFloat16>
        <<<N, kCombineBlockSize, 0, stream>>>(gemm2.data_ptr<c10::BFloat16>(),
                                              rw.data_ptr<float>(),
                                              output.data_ptr<c10::BFloat16>(),
                                              N,
                                              topk,
                                              H);
  } else {
    moe_combine_kernel<float>
        <<<N, kCombineBlockSize, 0, stream>>>(gemm2.data_ptr<float>(),
                                              rw.data_ptr<float>(),
                                              output.data_ptr<float>(),
                                              N,
                                              topk,
                                              H);
  }

  return output;
}

torch::Tensor moe_combine_result_indexed(
    const torch::Tensor& gemm2_sorted,
    const torch::Tensor& sorted_positions,
    const torch::Tensor& reduce_weight,
    int64_t N,
    int32_t topk) {
  CHECK(gemm2_sorted.dim() == 2);
  CHECK(sorted_positions.dim() == 1);
  CHECK(reduce_weight.dim() == 2);
  // Compact expert-major output has exactly N*topk rows.  The aligned MUSA
  // Ragged path keeps padding rows and supplies indices into that larger
  // buffer; the same kernel supports both layouts.
  CHECK_GE(gemm2_sorted.size(0), N * topk);
  CHECK_EQ(sorted_positions.numel(), N * topk);
  CHECK_EQ(reduce_weight.size(0), N);
  CHECK_EQ(reduce_weight.size(1), topk);
  CHECK_EQ(sorted_positions.scalar_type(), torch::kInt32);
  CHECK(gemm2_sorted.is_contiguous() && sorted_positions.is_contiguous());

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t H = gemm2_sorted.size(1);
  auto output = torch::empty({N, H}, gemm2_sorted.options());
  auto rw = reduce_weight.to(gemm2_sorted.device(), torch::kFloat32).contiguous();

  if (gemm2_sorted.scalar_type() == torch::kFloat16) {
    moe_combine_indexed_kernel<c10::Half>
        <<<N, kCombineBlockSize, 0, stream>>>(
            gemm2_sorted.data_ptr<c10::Half>(),
            sorted_positions.data_ptr<int32_t>(),
            rw.data_ptr<float>(),
            output.data_ptr<c10::Half>(),
            N,
            topk,
            gemm2_sorted.size(0),
            H);
  } else if (gemm2_sorted.scalar_type() == torch::kBFloat16) {
    if (H % 8 == 0) {
      moe_combine_indexed_bf16_vec8_kernel<<<N,
                                              kCombineBlockSize,
                                              0,
                                              stream>>>(
          gemm2_sorted.data_ptr<c10::BFloat16>(),
          sorted_positions.data_ptr<int32_t>(),
          rw.data_ptr<float>(),
          output.data_ptr<c10::BFloat16>(),
          N,
          topk,
          gemm2_sorted.size(0),
          H);
    } else {
      moe_combine_indexed_kernel<c10::BFloat16>
          <<<N, kCombineBlockSize, 0, stream>>>(
              gemm2_sorted.data_ptr<c10::BFloat16>(),
              sorted_positions.data_ptr<int32_t>(),
              rw.data_ptr<float>(),
              output.data_ptr<c10::BFloat16>(),
              N,
              topk,
              gemm2_sorted.size(0),
              H);
    }
  } else {
    moe_combine_indexed_kernel<float>
        <<<N, kCombineBlockSize, 0, stream>>>(
            gemm2_sorted.data_ptr<float>(),
            sorted_positions.data_ptr<int32_t>(),
            rw.data_ptr<float>(),
            output.data_ptr<float>(),
            N,
            topk,
            gemm2_sorted.size(0),
            H);
  }
  return output;
}

}  // namespace xllm::kernel::cuda
