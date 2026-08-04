/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

#include "core/kernels/cuda/device_utils.cuh"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kCombineBlockSize = 256;

template <typename ScalarType>
__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize)
    moe_combine_indexed_kernel(const ScalarType* __restrict__ gemm2_sorted,
                               const int32_t* __restrict__ sorted_positions,
                               const float* __restrict__ reduce_weight,
                               ScalarType* __restrict__ output,
                               int64_t num_tokens,
                               int32_t top_k,
                               int64_t gemm2_rows,
                               int64_t hidden_size) {
  const int64_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  const int32_t thread_id = threadIdx.x;
  for (int64_t hidden_idx = thread_id; hidden_idx < hidden_size;
       hidden_idx += kCombineBlockSize) {
    float accumulator = 0.0f;
    for (int32_t top_k_idx = 0; top_k_idx < top_k; ++top_k_idx) {
      const int64_t flat_idx = token_id * top_k + top_k_idx;
      const int32_t sorted_idx = sorted_positions[flat_idx];
      if (sorted_idx >= 0 && static_cast<int64_t>(sorted_idx) < gemm2_rows) {
        accumulator +=
            reduce_weight[flat_idx] *
            static_cast<float>(
                gemm2_sorted[static_cast<int64_t>(sorted_idx) * hidden_size +
                             hidden_idx]);
      }
    }
    output[token_id * hidden_size + hidden_idx] =
        static_cast<ScalarType>(accumulator);
  }
}

union Bf16Pack8 {
  int4 vector;
  c10::BFloat16 values[8];
};

// Each thread handles eight adjacent BF16 elements so one expert position and
// weight load feeds eight accumulators and each row uses aligned vector loads.
__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize)
    moe_combine_indexed_bf16_vec8_kernel(
        const c10::BFloat16* __restrict__ gemm2_sorted,
        const int32_t* __restrict__ sorted_positions,
        const float* __restrict__ reduce_weight,
        c10::BFloat16* __restrict__ output,
        int64_t num_tokens,
        int32_t top_k,
        int64_t gemm2_rows,
        int64_t hidden_size) {
  const int64_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  constexpr int32_t kValuesPerChunk = 8;
  const int64_t chunks_per_row = hidden_size / kValuesPerChunk;
  for (int64_t chunk_idx = threadIdx.x; chunk_idx < chunks_per_row;
       chunk_idx += blockDim.x) {
    float accumulators[kValuesPerChunk] = {};
    const int64_t column = chunk_idx * kValuesPerChunk;
    for (int32_t top_k_idx = 0; top_k_idx < top_k; ++top_k_idx) {
      const int64_t flat_idx = token_id * top_k + top_k_idx;
      const int32_t sorted_idx = sorted_positions[flat_idx];
      if (sorted_idx < 0 || static_cast<int64_t>(sorted_idx) >= gemm2_rows) {
        continue;
      }

      const float weight = reduce_weight[flat_idx];
      Bf16Pack8 input_pack;
      input_pack.vector = *reinterpret_cast<const int4*>(
          gemm2_sorted + static_cast<int64_t>(sorted_idx) * hidden_size +
          column);
#pragma unroll
      for (int32_t value_idx = 0; value_idx < kValuesPerChunk; ++value_idx) {
        accumulators[value_idx] +=
            weight * static_cast<float>(input_pack.values[value_idx]);
      }
    }

    Bf16Pack8 output_pack;
#pragma unroll
    for (int32_t value_idx = 0; value_idx < kValuesPerChunk; ++value_idx) {
      output_pack.values[value_idx] =
          static_cast<c10::BFloat16>(accumulators[value_idx]);
    }
    *reinterpret_cast<int4*>(output + token_id * hidden_size + column) =
        output_pack.vector;
  }
}

}  // namespace

torch::Tensor moe_combine_result_indexed(const torch::Tensor& gemm2_sorted,
                                         const torch::Tensor& sorted_positions,
                                         const torch::Tensor& reduce_weight,
                                         int64_t num_tokens,
                                         int32_t top_k) {
  CHECK_EQ(gemm2_sorted.dim(), 2);
  CHECK_EQ(sorted_positions.dim(), 1);
  CHECK_EQ(reduce_weight.dim(), 2);
  CHECK_GE(gemm2_sorted.size(0), num_tokens * top_k);
  CHECK_EQ(sorted_positions.numel(), num_tokens * top_k);
  CHECK_EQ(reduce_weight.size(0), num_tokens);
  CHECK_EQ(reduce_weight.size(1), top_k);
  CHECK_EQ(sorted_positions.scalar_type(), torch::kInt32);
  CHECK(gemm2_sorted.is_contiguous());
  CHECK(sorted_positions.is_contiguous());

  const int64_t hidden_size = gemm2_sorted.size(1);
  torch::Tensor output =
      torch::empty({num_tokens, hidden_size}, gemm2_sorted.options());
  torch::Tensor reduce_weight_fp32 =
      reduce_weight.to(gemm2_sorted.device(), torch::kFloat32).contiguous();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (gemm2_sorted.scalar_type() == torch::kFloat16) {
    moe_combine_indexed_kernel<c10::Half>
        <<<num_tokens, kCombineBlockSize, 0, stream>>>(
            gemm2_sorted.data_ptr<c10::Half>(),
            sorted_positions.data_ptr<int32_t>(),
            reduce_weight_fp32.data_ptr<float>(),
            output.data_ptr<c10::Half>(),
            num_tokens,
            top_k,
            gemm2_sorted.size(0),
            hidden_size);
  } else if (gemm2_sorted.scalar_type() == torch::kBFloat16) {
    if (hidden_size % 8 == 0) {
      moe_combine_indexed_bf16_vec8_kernel<<<num_tokens,
                                             kCombineBlockSize,
                                             0,
                                             stream>>>(
          gemm2_sorted.data_ptr<c10::BFloat16>(),
          sorted_positions.data_ptr<int32_t>(),
          reduce_weight_fp32.data_ptr<float>(),
          output.data_ptr<c10::BFloat16>(),
          num_tokens,
          top_k,
          gemm2_sorted.size(0),
          hidden_size);
    } else {
      moe_combine_indexed_kernel<c10::BFloat16>
          <<<num_tokens, kCombineBlockSize, 0, stream>>>(
              gemm2_sorted.data_ptr<c10::BFloat16>(),
              sorted_positions.data_ptr<int32_t>(),
              reduce_weight_fp32.data_ptr<float>(),
              output.data_ptr<c10::BFloat16>(),
              num_tokens,
              top_k,
              gemm2_sorted.size(0),
              hidden_size);
    }
  } else {
    CHECK_EQ(gemm2_sorted.scalar_type(), torch::kFloat32);
    moe_combine_indexed_kernel<float>
        <<<num_tokens, kCombineBlockSize, 0, stream>>>(
            gemm2_sorted.data_ptr<float>(),
            sorted_positions.data_ptr<int32_t>(),
            reduce_weight_fp32.data_ptr<float>(),
            output.data_ptr<float>(),
            num_tokens,
            top_k,
            gemm2_sorted.size(0),
            hidden_size);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace xllm::kernel::musa
