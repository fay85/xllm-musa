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
#include <glog/logging.h>
#include <musa_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kWarpThreads = 32;
constexpr int32_t kThreads = 256;
constexpr int32_t kWarps = kThreads / kWarpThreads;
constexpr int32_t kMaxTopK = 8;
constexpr int32_t kChunksPerRow = kThreads / kMaxTopK;
constexpr int32_t kBf16PerVector = 8;
constexpr int32_t kInvalidIndex = std::numeric_limits<int32_t>::max();

// Candidates of one thread, best first. Ties prefer the lower index so the
// selection is deterministic.
struct TopKList {
  float values[kMaxTopK];
  int32_t indices[kMaxTopK];
};

__device__ __forceinline__ bool ranks_before(float value,
                                             int32_t index,
                                             float other_value,
                                             int32_t other_index) {
  return value > other_value || (value == other_value && index < other_index);
}

__device__ __forceinline__ void init_list(TopKList& list) {
#pragma unroll
  for (int32_t i = 0; i < kMaxTopK; ++i) {
    list.values[i] = -INFINITY;
    list.indices[i] = kInvalidIndex;
  }
}

__device__ __forceinline__ void insert(TopKList& list,
                                       float value,
                                       int32_t index) {
  if (!ranks_before(value,
                    index,
                    list.values[kMaxTopK - 1],
                    list.indices[kMaxTopK - 1])) {
    return;
  }
  list.values[kMaxTopK - 1] = value;
  list.indices[kMaxTopK - 1] = index;
#pragma unroll
  for (int32_t i = kMaxTopK - 1; i > 0; --i) {
    if (ranks_before(list.values[i],
                     list.indices[i],
                     list.values[i - 1],
                     list.indices[i - 1])) {
      const float value_swap = list.values[i];
      list.values[i] = list.values[i - 1];
      list.values[i - 1] = value_swap;
      const int32_t index_swap = list.indices[i];
      list.indices[i] = list.indices[i - 1];
      list.indices[i - 1] = index_swap;
    }
  }
}

__device__ __forceinline__ void pop_front(TopKList& list) {
#pragma unroll
  for (int32_t i = 0; i < kMaxTopK - 1; ++i) {
    list.values[i] = list.values[i + 1];
    list.indices[i] = list.indices[i + 1];
  }
  list.values[kMaxTopK - 1] = -INFINITY;
  list.indices[kMaxTopK - 1] = kInvalidIndex;
}

__device__ __forceinline__ void reduce_best(float& value,
                                            int32_t& index,
                                            int32_t& owner) {
#pragma unroll
  for (int32_t offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    const float other_value =
        __shfl_xor_sync(0xffffffff, value, offset, kWarpThreads);
    const int32_t other_index =
        __shfl_xor_sync(0xffffffff, index, offset, kWarpThreads);
    const int32_t other_owner =
        __shfl_xor_sync(0xffffffff, owner, offset, kWarpThreads);
    if (ranks_before(other_value, other_index, value, index)) {
      value = other_value;
      index = other_index;
      owner = other_owner;
    }
  }
}

// Writes the `top_k` best candidates over all threads of the block, best
// first, by repeatedly taking the best list head of the block.
__device__ void select_block_top_k(TopKList& list,
                                   int32_t top_k,
                                   int32_t* __restrict__ out_indices) {
  __shared__ float warp_values[kWarps];
  __shared__ int32_t warp_indices[kWarps];
  __shared__ int32_t warp_owners[kWarps];
  __shared__ int32_t winner;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid % kWarpThreads;
  const int32_t warp = tid / kWarpThreads;
  for (int32_t rank = 0; rank < top_k; ++rank) {
    float value = list.values[0];
    int32_t index = list.indices[0];
    int32_t owner = tid;
    reduce_best(value, index, owner);
    if (lane == 0) {
      warp_values[warp] = value;
      warp_indices[warp] = index;
      warp_owners[warp] = owner;
    }
    __syncthreads();
    if (warp == 0) {
      value = lane < kWarps ? warp_values[lane] : -INFINITY;
      index = lane < kWarps ? warp_indices[lane] : kInvalidIndex;
      owner = lane < kWarps ? warp_owners[lane] : -1;
      reduce_best(value, index, owner);
      if (lane == 0) {
        out_indices[rank] = index;
        winner = owner;
      }
    }
    __syncthreads();
    if (tid == winner) {
      pop_front(list);
    }
  }
}

// One block selects the top-k entries of one vocab chunk of one row.
__global__ void __launch_bounds__(kThreads)
    chunk_top_k_kernel(const __mt_bfloat16* __restrict__ logits,
                       int64_t row_stride,
                       int32_t vocab,
                       int32_t top_k,
                       int32_t* __restrict__ chunk_indices) {
  const int32_t row = static_cast<int32_t>(blockIdx.y);
  const int32_t chunk = static_cast<int32_t>(blockIdx.x);
  const int32_t chunk_len = (vocab + kChunksPerRow - 1) / kChunksPerRow;
  const int32_t begin = chunk * chunk_len;
  const int32_t end = min(vocab, begin + chunk_len);
  const __mt_bfloat16* row_logits = logits + row * row_stride;
  TopKList list;
  init_list(list);
  for (int32_t i = begin + static_cast<int32_t>(threadIdx.x); i < end;
       i += kThreads) {
    insert(list, __bfloat162float(row_logits[i]), i);
  }
  select_block_top_k(
      list,
      top_k,
      chunk_indices +
          (static_cast<int64_t>(row) * kChunksPerRow + chunk) * top_k);
}

// One block merges the chunk candidates of one row into its top-k indices.
__global__ void __launch_bounds__(kThreads)
    merge_top_k_kernel(const __mt_bfloat16* __restrict__ logits,
                       int64_t row_stride,
                       int32_t top_k,
                       const int32_t* __restrict__ chunk_indices,
                       int32_t* __restrict__ top_indices) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  TopKList list;
  init_list(list);
  if (tid < kChunksPerRow * top_k) {
    const int32_t index =
        chunk_indices[static_cast<int64_t>(row) * kChunksPerRow * top_k + tid];
    if (index != kInvalidIndex) {
      list.values[0] = __bfloat162float(logits[row * row_stride + index]);
      list.indices[0] = index;
    }
  }
  select_block_top_k(
      list, top_k, top_indices + static_cast<int64_t>(row) * top_k);
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& raw, float* out) {
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

// One warp recomputes one selected logit from the unquantized weight row.
__global__ void __launch_bounds__(kThreads)
    rescore_kernel(const __mt_bfloat16* __restrict__ input,
                   const __mt_bfloat16* __restrict__ weight,
                   const int32_t* __restrict__ top_indices,
                   __mt_bfloat16* __restrict__ logits,
                   int64_t row_stride,
                   int32_t hidden,
                   int32_t top_k) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpThreads;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kWarpThreads;
  if (warp >= top_k) {
    return;
  }
  const int32_t index = top_indices[static_cast<int64_t>(row) * top_k + warp];
  if (index == kInvalidIndex) {
    return;
  }
  const __mt_bfloat16* input_row = input + static_cast<int64_t>(row) * hidden;
  const __mt_bfloat16* weight_row =
      weight + static_cast<int64_t>(index) * hidden;
  float acc = 0.0f;
  for (int32_t v = lane; v < hidden / kBf16PerVector; v += kWarpThreads) {
    float input_f[kBf16PerVector];
    float weight_f[kBf16PerVector];
    unpack_bf16x8(*reinterpret_cast<const uint4*>(input_row +
                                                  v * kBf16PerVector),
                  input_f);
    unpack_bf16x8(*reinterpret_cast<const uint4*>(weight_row +
                                                  v * kBf16PerVector),
                  weight_f);
#pragma unroll
    for (int32_t e = 0; e < kBf16PerVector; ++e) {
      acc += input_f[e] * weight_f[e];
    }
  }
#pragma unroll
  for (int32_t offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor_sync(0xffffffff, acc, offset, kWarpThreads);
  }
  if (lane == 0) {
    logits[row * row_stride + index] = __float2bfloat16_rn(acc);
  }
}

bool is_16b_aligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignof(uint4) == 0;
}

}  // namespace

void rescore_top_logits(const torch::Tensor& input,
                        const torch::Tensor& weight,
                        torch::Tensor& logits,
                        int64_t top_k) {
  CHECK(input.dim() == 2 && weight.dim() == 2 && logits.dim() == 2)
      << "rescore_top_logits expects 2-D input, weight and logits";
  CHECK(input.scalar_type() == torch::kBFloat16 &&
        weight.scalar_type() == torch::kBFloat16 &&
        logits.scalar_type() == torch::kBFloat16)
      << "rescore_top_logits expects bf16 tensors";
  CHECK(input.is_contiguous() && weight.is_contiguous() &&
        logits.stride(1) == 1)
      << "rescore_top_logits expects contiguous input/weight rows";
  CHECK(is_16b_aligned(input) && is_16b_aligned(weight));
  const int64_t rows = input.size(0);
  const int64_t hidden = input.size(1);
  const int64_t vocab = weight.size(0);
  CHECK_EQ(weight.size(1), hidden);
  CHECK_EQ(logits.size(0), rows);
  CHECK_EQ(logits.size(1), vocab);
  CHECK_EQ(hidden % kBf16PerVector, 0);
  CHECK(top_k >= 1 && top_k <= kMaxTopK) << "top_k must be in [1, 8]";
  CHECK_GE(vocab, kChunksPerRow * top_k);
  if (rows == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto index_options = input.options().dtype(torch::kInt32);
  torch::Tensor chunk_indices =
      torch::empty({rows, kChunksPerRow, top_k}, index_options);
  torch::Tensor top_indices = torch::empty({rows, top_k}, index_options);
  auto* logits_ptr =
      reinterpret_cast<__mt_bfloat16*>(logits.data_ptr<at::BFloat16>());
  const int32_t k = static_cast<int32_t>(top_k);

  chunk_top_k_kernel<<<dim3(kChunksPerRow, static_cast<uint32_t>(rows)),
                       kThreads,
                       0,
                       stream>>>(logits_ptr,
                                 logits.stride(0),
                                 static_cast<int32_t>(vocab),
                                 k,
                                 chunk_indices.data_ptr<int32_t>());
  merge_top_k_kernel<<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
      logits_ptr,
      logits.stride(0),
      k,
      chunk_indices.data_ptr<int32_t>(),
      top_indices.data_ptr<int32_t>());
  rescore_kernel<<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __mt_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      top_indices.data_ptr<int32_t>(),
      logits_ptr,
      logits.stride(0),
      static_cast<int32_t>(hidden),
      k);
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace xllm::kernel::musa
