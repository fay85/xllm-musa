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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kBf16PerVector = 8;
constexpr int64_t kTargetBlocks = 512;
constexpr int64_t kMinCols = 16384;
constexpr int64_t kNoIndex = std::numeric_limits<int64_t>::max();

// torch::argmax order: NaN ranks above every number and ties keep the lower
// index.
__device__ __forceinline__ bool ranks_before(float value,
                                             int64_t index,
                                             float other_value,
                                             int64_t other_index) {
  const bool is_nan = isnan(value);
  const bool other_is_nan = isnan(other_value);
  if (is_nan != other_is_nan) {
    return is_nan;
  }
  if (!is_nan && value != other_value) {
    return value > other_value;
  }
  return index < other_index;
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& raw, float* out) {
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

// Leaves the best (value, index) of the block in values[0] and indices[0].
__device__ void reduce_block_best(float value,
                                  int64_t index,
                                  float* values,
                                  int64_t* indices) {
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  values[tid] = value;
  indices[tid] = index;
  __syncthreads();
  for (int32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride && ranks_before(values[tid + stride],
                                     indices[tid + stride],
                                     values[tid],
                                     indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
}

// One block finds the best entry of one column chunk of one row.
__global__ void __launch_bounds__(kThreads)
    chunk_argmax_bf16_kernel(const __mt_bfloat16* __restrict__ input,
                             int64_t cols,
                             int64_t chunk_vectors,
                             float* __restrict__ chunk_values,
                             int64_t* __restrict__ chunk_indices) {
  __shared__ float values[kThreads];
  __shared__ int64_t indices[kThreads];
  const int64_t row = blockIdx.y;
  const int64_t chunk = blockIdx.x;
  const int64_t vectors = cols / kBf16PerVector;
  const int64_t begin = chunk * chunk_vectors;
  const int64_t end =
      begin + chunk_vectors < vectors ? begin + chunk_vectors : vectors;
  const uint4* row_input = reinterpret_cast<const uint4*>(input + row * cols);
  float best = -INFINITY;
  int64_t best_index = kNoIndex;
  for (int64_t vector = begin + threadIdx.x; vector < end; vector += kThreads) {
    float unpacked[kBf16PerVector];
    unpack_bf16x8(row_input[vector], unpacked);
#pragma unroll
    for (int32_t e = 0; e < kBf16PerVector; ++e) {
      const int64_t index = vector * kBf16PerVector + e;
      if (ranks_before(unpacked[e], index, best, best_index)) {
        best = unpacked[e];
        best_index = index;
      }
    }
  }
  reduce_block_best(best, best_index, values, indices);
  if (threadIdx.x == 0) {
    chunk_values[row * gridDim.x + chunk] = values[0];
    chunk_indices[row * gridDim.x + chunk] = indices[0];
  }
}

// One block merges the chunk winners of one row.
__global__ void __launch_bounds__(kThreads)
    merge_argmax_kernel(const float* __restrict__ chunk_values,
                        const int64_t* __restrict__ chunk_indices,
                        int32_t chunks,
                        const int64_t* __restrict__ column_ids,
                        int64_t* __restrict__ output) {
  __shared__ float values[kThreads];
  __shared__ int64_t indices[kThreads];
  const int64_t row = blockIdx.x;
  float value = -INFINITY;
  int64_t index = kNoIndex;
  if (static_cast<int32_t>(threadIdx.x) < chunks) {
    value = chunk_values[row * chunks + threadIdx.x];
    index = chunk_indices[row * chunks + threadIdx.x];
  }
  reduce_block_best(value, index, values, indices);
  if (threadIdx.x == 0) {
    const int64_t winner = indices[0];
    output[row] =
        (column_ids != nullptr && winner != kNoIndex) ? column_ids[winner]
                                                      : winner;
  }
}

bool is_16b_aligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignof(uint4) == 0;
}

struct CompactVocabEntry {
  const void* data_ptr = nullptr;
  int64_t cols = 0;
  torch::Tensor column_ids;
};

thread_local CompactVocabEntry s_last_compact_vocab;

}  // namespace

void register_compact_vocab_column_ids(const torch::Tensor& compact_logits,
                                       const torch::Tensor& column_ids) {
  if (!compact_logits.defined() || !column_ids.defined()) {
    s_last_compact_vocab = CompactVocabEntry{};
    return;
  }
  s_last_compact_vocab.data_ptr = compact_logits.data_ptr();
  s_last_compact_vocab.cols = compact_logits.size(-1);
  s_last_compact_vocab.column_ids = column_ids;
}

torch::Tensor lookup_compact_vocab_column_ids(const torch::Tensor& logits) {
  if (!logits.defined() || !s_last_compact_vocab.column_ids.defined()) {
    return torch::Tensor();
  }
  if (logits.data_ptr() == s_last_compact_vocab.data_ptr &&
      logits.size(-1) == s_last_compact_vocab.cols) {
    return s_last_compact_vocab.column_ids;
  }
  return torch::Tensor();
}

bool row_argmax_supported(const torch::Tensor& input) {
  return input.defined() && !input.is_cpu() && input.dim() >= 1 &&
         input.scalar_type() == torch::kBFloat16 && input.is_contiguous() &&
         input.size(-1) >= kMinCols && input.size(-1) % kBf16PerVector == 0 &&
         input.numel() > 0 && is_16b_aligned(input);
}

torch::Tensor row_argmax(const torch::Tensor& input,
                         const torch::Tensor& column_ids) {
  CHECK(row_argmax_supported(input))
      << "unsupported row argmax input: " << input.sizes() << " "
      << input.scalar_type();
  const int64_t cols = input.size(-1);
  const int64_t rows = input.numel() / cols;
  const int64_t* column_ids_ptr = nullptr;
  if (column_ids.defined()) {
    CHECK(column_ids.dim() == 1 && column_ids.size(0) == cols &&
          column_ids.scalar_type() == torch::kLong &&
          column_ids.is_contiguous() && column_ids.device() == input.device())
        << "invalid row_argmax column_ids: " << column_ids.sizes();
    column_ids_ptr = column_ids.data_ptr<int64_t>();
  }
  const std::vector<int64_t> output_sizes(input.sizes().begin(),
                                          input.sizes().end() - 1);
  torch::Tensor output =
      torch::empty(output_sizes, input.options().dtype(torch::kLong));
  const int64_t vectors = cols / kBf16PerVector;
  const int64_t wanted_chunks =
      std::clamp<int64_t>(kTargetBlocks / rows, 1, kThreads);
  const int64_t chunk_vectors = (vectors + wanted_chunks - 1) / wanted_chunks;
  const int64_t chunks = (vectors + chunk_vectors - 1) / chunk_vectors;
  torch::Tensor chunk_values =
      torch::empty({rows, chunks}, input.options().dtype(torch::kFloat));
  torch::Tensor chunk_indices =
      torch::empty({rows, chunks}, input.options().dtype(torch::kLong));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  chunk_argmax_bf16_kernel<<<dim3(static_cast<uint32_t>(chunks),
                                  static_cast<uint32_t>(rows)),
                             kThreads,
                             0,
                             stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      cols,
      chunk_vectors,
      chunk_values.data_ptr<float>(),
      chunk_indices.data_ptr<int64_t>());
  merge_argmax_kernel<<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
      chunk_values.data_ptr<float>(),
      chunk_indices.data_ptr<int64_t>(),
      static_cast<int32_t>(chunks),
      column_ids_ptr,
      output.data_ptr<int64_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  return output;
}

}  // namespace xllm::kernel::musa
