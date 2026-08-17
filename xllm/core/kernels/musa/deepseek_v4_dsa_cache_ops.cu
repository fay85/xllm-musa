/* Copyright 2026 The xLLM Authors.

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

#include <c10/cuda/CUDAStream.h>
#include <c10/util/Float8_e4m3fn.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#ifndef MUSART_INF_FP16
#define MUSART_INF_FP16 __float2half_rn(INFINITY)
#endif
#ifndef MUSART_INF_BF16
#define MUSART_INF_BF16 __ushort_as_bfloat16(0x7F80)
#endif

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kGatherCacheBlockSize = 256;
constexpr int32_t kMusaWarpSize = 32;
constexpr int32_t kHcPreCombineBlockSize = 256;
constexpr int32_t kHcPost2DBlockSize = 256;
constexpr int32_t kHcPost2DHiddenTile = 512;
constexpr int32_t kHcMetadataBlockSize = 32;
constexpr int32_t kHcMaxMult = 16;
constexpr int32_t kScatterCacheBlockSize = 256;
constexpr int32_t kModel1PackBlockSize = 256;
constexpr int32_t kIndexerCacheBlockSize = 256;
constexpr int32_t kIndexerRadixTopk = 512;
constexpr int32_t kIndexerRadixThreads = 1024;
constexpr int32_t kIndexerRadixBins = 256;
constexpr int32_t kIndexerRadixSmemBytes = 32 * 1024;
constexpr int32_t kLogicalToPhysicalBlockSize = 256;
constexpr int32_t kCompressPoolBlockSize = 256;
constexpr int32_t kCompressPoolC128BlockSize = 256;
constexpr int32_t kCompressPoolC128Warps =
    kCompressPoolC128BlockSize / kMusaWarpSize;
constexpr int32_t kCompressPoolC128DimsPerWarp = 4;
constexpr int32_t kCompressPoolC128DimsPerBlock =
    kCompressPoolC128Warps * kCompressPoolC128DimsPerWarp;
constexpr int32_t kCompressDecodeC4BlockSize = 256;
constexpr int32_t kCompressDecodeC128BlockSize = 256;
constexpr int64_t kModel1HeadDim = 512;
constexpr int64_t kModel1NopeDim = 448;
constexpr int64_t kModel1RopeDim = 64;
constexpr int64_t kModel1QuantGroup = 64;
constexpr int64_t kModel1NumScales = 7;
constexpr int64_t kModel1PayloadBytes = 576;
constexpr int64_t kModel1ScaleBytes = 8;
constexpr int64_t kModel1BytesPerToken = 584;

musaStream_t current_dsv4_cache_stream() {
  // The MUSA build is compiled through musamapping, so CUDA stream APIs map
  // to the caller's active MUSA stream.  Keep cache kernels on the same stream
  // API as the surrounding attention, RoPE, HC and MoE kernels.  Mixing the
  // native c10::musa accessor here with c10::cuda at every producer/consumer
  // boundary selected a different stream in the service runtime: the 2.5K
  // prefill regressed from about 2 s to 9.5 s and long decode became
  // nondeterministic even with the experimental decode kernel disabled.
  return c10::cuda::getCurrentCUDAStream();
}

void check_musa_runtime(musaError_t status) {
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 MUSA runtime call failed: " << musaGetErrorString(status);
}

bool dsv4_musa_cache_gather_sync_enabled() { return false; }

bool dsv4_musa_cache_scatter_sync_enabled() { return false; }

bool dsv4_musa_indexer_single_kernel_topk_enabled() { return false; }

bool dsv4_musa_indexer_radix_topk_enabled() { return true; }

bool dsv4_musa_indexer_fused_score_radix_enabled() { return true; }

bool dsv4_musa_indexer_warp_score_enabled() { return true; }

template <typename scalar_t>
__global__ void __launch_bounds__(kCompressPoolBlockSize, 1)
    dsv4_compress_pool_kernel(const scalar_t* __restrict__ kv,
                              const float* __restrict__ score,
                              const float* __restrict__ ape,
                              scalar_t* __restrict__ output,
                              int64_t compress_ratio,
                              int64_t head_dim,
                              int64_t compressor_coeff) {
  const int64_t compressed_row = blockIdx.x;
  const int64_t width = compressor_coeff * head_dim;
  const int64_t candidate_count = compressor_coeff * compress_ratio;

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float max_score = -INFINITY;
    for (int64_t candidate = 0; candidate < candidate_count; ++candidate) {
      int64_t token = 0;
      int64_t source_dim = dim;
      int64_t ape_row = candidate;
      if (compressor_coeff == 2) {
        if (candidate < compress_ratio) {
          if (compressed_row == 0) {
            continue;
          }
          token = (compressed_row - 1) * compress_ratio + candidate;
        } else {
          const int64_t local = candidate - compress_ratio;
          token = compressed_row * compress_ratio + local;
          source_dim = head_dim + dim;
          ape_row = local;
        }
      } else {
        token = compressed_row * compress_ratio + candidate;
      }
      const float candidate_score =
          score[token * width + source_dim] + ape[ape_row * width + source_dim];
      max_score = fmaxf(max_score, candidate_score);
    }

    float denominator = 0.0f;
    float weighted_sum = 0.0f;
    for (int64_t candidate = 0; candidate < candidate_count; ++candidate) {
      int64_t token = 0;
      int64_t source_dim = dim;
      int64_t ape_row = candidate;
      if (compressor_coeff == 2) {
        if (candidate < compress_ratio) {
          if (compressed_row == 0) {
            continue;
          }
          token = (compressed_row - 1) * compress_ratio + candidate;
        } else {
          const int64_t local = candidate - compress_ratio;
          token = compressed_row * compress_ratio + local;
          source_dim = head_dim + dim;
          ape_row = local;
        }
      } else {
        token = compressed_row * compress_ratio + candidate;
      }
      const float candidate_score =
          score[token * width + source_dim] + ape[ape_row * width + source_dim];
      const float weight = expf(candidate_score - max_score);
      denominator += weight;
      weighted_sum +=
          weight * musa_convert_to_float(kv[token * width + source_dim]);
    }
    output[compressed_row * head_dim + dim] =
        static_cast<scalar_t>(weighted_sum / denominator);
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kCompressPoolC128BlockSize, 1)
    dsv4_compress_pool_c128_kernel(const scalar_t* __restrict__ kv,
                                   const float* __restrict__ score,
                                   const float* __restrict__ ape,
                                   scalar_t* __restrict__ output,
                                   int64_t head_dim) {
  constexpr int64_t kCompressRatio = 128;
  const int64_t compressed_row = blockIdx.x;
  const int64_t warp = threadIdx.x / kMusaWarpSize;
  const int64_t lane = threadIdx.x % kMusaWarpSize;
  const int64_t dim_begin = blockIdx.y * kCompressPoolC128DimsPerBlock + warp;
  const int64_t dim_limit = dim_begin + kCompressPoolC128DimsPerBlock;
  const int64_t dim_end = dim_limit < head_dim ? dim_limit : head_dim;
  for (int64_t dim = dim_begin; dim < dim_end; dim += kCompressPoolC128Warps) {
    float max_score = -INFINITY;
    for (int64_t candidate = lane; candidate < kCompressRatio;
         candidate += kMusaWarpSize) {
      const int64_t token = compressed_row * kCompressRatio + candidate;
      max_score = fmaxf(
          max_score,
          score[token * head_dim + dim] + ape[candidate * head_dim + dim]);
    }
    for (int offset = kMusaWarpSize / 2; offset > 0; offset /= 2) {
      max_score =
          fmaxf(max_score, __shfl_xor_sync(0xffffffffU, max_score, offset));
    }

    float denominator = 0.0f;
    float weighted_sum = 0.0f;
    for (int64_t candidate = lane; candidate < kCompressRatio;
         candidate += kMusaWarpSize) {
      const int64_t token = compressed_row * kCompressRatio + candidate;
      const float candidate_score =
          score[token * head_dim + dim] + ape[candidate * head_dim + dim];
      const float weight = expf(candidate_score - max_score);
      denominator += weight;
      weighted_sum +=
          weight * musa_convert_to_float(kv[token * head_dim + dim]);
    }
    for (int offset = kMusaWarpSize / 2; offset > 0; offset /= 2) {
      denominator += __shfl_xor_sync(0xffffffffU, denominator, offset);
      weighted_sum += __shfl_xor_sync(0xffffffffU, weighted_sum, offset);
    }
    if (lane == 0) {
      output[compressed_row * head_dim + dim] =
          static_cast<scalar_t>(weighted_sum / denominator);
    }
  }
}

// Decode counterpart of SGLang's DeepSeek-V4 C4 compression kernel. xLLM
// keeps KV and gate projections in separate persistent state tensors, so the
// packed donor ABI is expressed directly against those tensors here. One
// thread owns one output dimension, which makes the state update, eight-way
// softmax pooling, and four-row history rotation race-free without host-side
// index_put/cat/softmax/mul/sum/copy launches.
__global__ void __launch_bounds__(kCompressDecodeC4BlockSize, 1)
    dsv4_compress_decode_c4_kernel(float* __restrict__ kv_state,
                                   float* __restrict__ score_state,
                                   const float* __restrict__ kv_step,
                                   const float* __restrict__ score_step,
                                   const float* __restrict__ ape,
                                   float* __restrict__ output,
                                   int64_t head_dim,
                                   int64_t ratio_position,
                                   bool at_boundary,
                                   bool emit_output) {
  constexpr int64_t kRatio = 4;
  constexpr int64_t kCoeff = 2;
  constexpr int64_t kStateRows = kRatio * kCoeff;
  const int64_t width = kCoeff * head_dim;
  for (int64_t dim = blockIdx.x * blockDim.x + threadIdx.x; dim < head_dim;
       dim += gridDim.x * blockDim.x) {
    const int64_t current_row = kRatio + ratio_position;
    const int64_t current_offset = current_row * width;
    const int64_t ape_offset = ratio_position * width;
    kv_state[current_offset + dim] = kv_step[dim];
    kv_state[current_offset + head_dim + dim] = kv_step[head_dim + dim];
    score_state[current_offset + dim] = score_step[dim] + ape[ape_offset + dim];
    score_state[current_offset + head_dim + dim] =
        score_step[head_dim + dim] + ape[ape_offset + head_dim + dim];

    // In state-only mode the caller still performs the exact Torch boundary
    // pool and ring rotation after this launch. Do not rotate the ring before
    // that pool, otherwise the four history rows would be overwritten before
    // they are read.
    if (!at_boundary || !emit_output) {
      continue;
    }

    if (emit_output) {
      float max_score = -INFINITY;
      for (int64_t row = 0; row < kRatio; ++row) {
        max_score = fmaxf(max_score, score_state[row * width + dim]);
      }
      for (int64_t row = kRatio; row < kStateRows; ++row) {
        max_score = fmaxf(max_score, score_state[row * width + head_dim + dim]);
      }

      float denominator = 0.0f;
      float weighted_sum = 0.0f;
      for (int64_t row = 0; row < kRatio; ++row) {
        const float weight = expf(score_state[row * width + dim] - max_score);
        denominator += weight;
        weighted_sum += weight * kv_state[row * width + dim];
      }
      for (int64_t row = kRatio; row < kStateRows; ++row) {
        const int64_t offset = row * width + head_dim + dim;
        const float weight = expf(score_state[offset] - max_score);
        denominator += weight;
        weighted_sum += weight * kv_state[offset];
      }
      output[dim] = weighted_sum / denominator;
    }

    // Preserve the just-compressed four-token group as the history candidate
    // for the next C4 boundary. Each thread owns both coefficient columns for
    // its dimension, so these copies cannot race.
    for (int64_t row = 0; row < kRatio; ++row) {
      const int64_t dst = row * width;
      const int64_t src = (kRatio + row) * width;
      kv_state[dst + dim] = kv_state[src + dim];
      kv_state[dst + head_dim + dim] = kv_state[src + head_dim + dim];
      score_state[dst + dim] = score_state[src + dim];
      score_state[dst + head_dim + dim] = score_state[src + head_dim + dim];
    }
  }
}

// C128 is rare (one pool per 128 decode tokens) but the state write happens on
// every token. Match SGLang's low-launch serial-per-dimension kernel here; the
// more expensive 128-row loop only runs at a compression boundary.
__global__ void __launch_bounds__(kCompressDecodeC128BlockSize, 1)
    dsv4_compress_decode_c128_kernel(float* __restrict__ kv_state,
                                     float* __restrict__ score_state,
                                     const float* __restrict__ kv_step,
                                     const float* __restrict__ score_step,
                                     const float* __restrict__ ape,
                                     float* __restrict__ output,
                                     int64_t head_dim,
                                     int64_t ratio_position,
                                     bool at_boundary,
                                     bool emit_output) {
  constexpr int64_t kRatio = 128;
  for (int64_t dim = blockIdx.x * blockDim.x + threadIdx.x; dim < head_dim;
       dim += gridDim.x * blockDim.x) {
    const int64_t current_offset = ratio_position * head_dim + dim;
    kv_state[current_offset] = kv_step[dim];
    score_state[current_offset] =
        score_step[dim] + ape[ratio_position * head_dim + dim];
    if (!at_boundary || !emit_output) {
      continue;
    }

    float max_score = -INFINITY;
    for (int64_t row = 0; row < kRatio; ++row) {
      max_score = fmaxf(max_score, score_state[row * head_dim + dim]);
    }

    float denominator = 0.0f;
    float weighted_sum = 0.0f;
    for (int64_t row = 0; row < kRatio; ++row) {
      const int64_t offset = row * head_dim + dim;
      const float weight = expf(score_state[offset] - max_score);
      denominator += weight;
      weighted_sum += weight * kv_state[offset];
    }
    output[dim] = weighted_sum / denominator;
  }
}

template <typename index_t, typename vec_t>
__global__ void __launch_bounds__(kScatterCacheBlockSize, 1)
    dsv4_scatter_cache_kernel(const char* __restrict__ values,
                              char* __restrict__ cache,
                              const index_t* __restrict__ slot_ids,
                              int64_t update_rows,
                              int64_t cache_rows,
                              int64_t row_bytes) {
  const int64_t row = blockIdx.x;
  if (row >= update_rows) {
    return;
  }

  const int64_t slot = static_cast<int64_t>(slot_ids[row]);
  if (slot < 0 || slot >= cache_rows) {
    return;
  }

  const int64_t vec_count = row_bytes / static_cast<int64_t>(sizeof(vec_t));
  const vec_t* src = reinterpret_cast<const vec_t*>(values + row * row_bytes);
  vec_t* dst = reinterpret_cast<vec_t*>(cache + slot * row_bytes);
  for (int64_t idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
    dst[idx] = src[idx];
  }
}

template <typename scalar_t, typename index_t>
__global__ void __launch_bounds__(kScatterCacheBlockSize, 1)
    dsv4_scatter_flashmla_cache_kernel(const scalar_t* __restrict__ values,
                                       scalar_t* __restrict__ cache,
                                       const index_t* __restrict__ slot_ids,
                                       int64_t update_rows,
                                       int64_t cache_rows,
                                       int64_t value_width,
                                       int64_t cache_width,
                                       int64_t rope_head_dim) {
  const int64_t row = blockIdx.x;
  if (row >= update_rows) {
    return;
  }

  const int64_t slot = static_cast<int64_t>(slot_ids[row]);
  if (slot < 0 || slot >= cache_rows) {
    return;
  }

  const scalar_t* src = values + row * value_width;
  scalar_t* dst = cache + slot * cache_width;
  for (int64_t col = threadIdx.x; col < cache_width; col += blockDim.x) {
    const int64_t src_col =
        col < value_width ? col
                          : value_width - rope_head_dim + (col - value_width);
    dst[col] = src[src_col];
  }
}

template <typename index_t>
void dispatch_dsv4_scatter_flashmla_cache(const torch::Tensor& value_2d,
                                          torch::Tensor& cache_2d,
                                          const torch::Tensor& slots,
                                          int64_t update_rows,
                                          int64_t cache_rows,
                                          int64_t value_width,
                                          int64_t cache_width,
                                          int64_t rope_head_dim) {
  const dim3 grid(update_rows);
  const dim3 block(kScatterCacheBlockSize);
  DISPATCH_FLOATING_TYPES(
      value_2d.scalar_type(), "dsv4_scatter_flashmla_cache", [&] {
        dsv4_scatter_flashmla_cache_kernel<scalar_t, index_t>
            <<<grid, block, 0, current_dsv4_cache_stream()>>>(
                value_2d.data_ptr<scalar_t>(),
                cache_2d.data_ptr<scalar_t>(),
                slots.data_ptr<index_t>(),
                update_rows,
                cache_rows,
                value_width,
                cache_width,
                rope_head_dim);
      });
  check_musa_kernel_launch("dsv4 kernel");
  if (dsv4_musa_cache_scatter_sync_enabled()) {
    check_musa_runtime(musaStreamSynchronize(current_dsv4_cache_stream()));
  }
}

template <typename index_t, typename vec_t>
void launch_dsv4_scatter_cache_typed(const torch::Tensor& value_2d,
                                     torch::Tensor& cache_2d,
                                     const torch::Tensor& slots,
                                     int64_t update_rows,
                                     int64_t cache_rows,
                                     int64_t row_bytes) {
  const dim3 grid(update_rows);
  const dim3 block(kScatterCacheBlockSize);
  dsv4_scatter_cache_kernel<index_t, vec_t>
      <<<grid, block, 0, current_dsv4_cache_stream()>>>(
          reinterpret_cast<const char*>(value_2d.data_ptr()),
          reinterpret_cast<char*>(cache_2d.data_ptr()),
          slots.data_ptr<index_t>(),
          update_rows,
          cache_rows,
          row_bytes);
}

template <typename index_t>
void dispatch_dsv4_scatter_cache(const torch::Tensor& value_2d,
                                 torch::Tensor& cache_2d,
                                 const torch::Tensor& slots,
                                 int64_t update_rows,
                                 int64_t cache_rows,
                                 int64_t row_bytes) {
  const uintptr_t ptr_or = reinterpret_cast<uintptr_t>(value_2d.data_ptr()) |
                           reinterpret_cast<uintptr_t>(cache_2d.data_ptr());
  if (row_bytes % 16 == 0 && (ptr_or & 0xF) == 0) {
    launch_dsv4_scatter_cache_typed<index_t, uint4>(
        value_2d, cache_2d, slots, update_rows, cache_rows, row_bytes);
  } else if (row_bytes % 4 == 0 && (ptr_or & 0x3) == 0) {
    launch_dsv4_scatter_cache_typed<index_t, uint32_t>(
        value_2d, cache_2d, slots, update_rows, cache_rows, row_bytes);
  } else if (row_bytes % 2 == 0 && (ptr_or & 0x1) == 0) {
    launch_dsv4_scatter_cache_typed<index_t, uint16_t>(
        value_2d, cache_2d, slots, update_rows, cache_rows, row_bytes);
  } else {
    launch_dsv4_scatter_cache_typed<index_t, uint8_t>(
        value_2d, cache_2d, slots, update_rows, cache_rows, row_bytes);
  }
  check_musa_kernel_launch("dsv4 kernel");
  if (dsv4_musa_cache_scatter_sync_enabled()) {
    check_musa_runtime(musaStreamSynchronize(current_dsv4_cache_stream()));
  }
}

#if !defined(USE_DCU)
template <typename scalar_t, typename index_t>
__global__ void __launch_bounds__(kModel1PackBlockSize, 1)
    dsv4_pack_model1_fp8_cache_kernel(const scalar_t* __restrict__ values,
                                      uint8_t* __restrict__ cache,
                                      const index_t* __restrict__ slot_ids,
                                      int64_t update_rows,
                                      int64_t num_blocks,
                                      int64_t block_size,
                                      int64_t page_stride_bytes) {
  const int64_t row = blockIdx.x;
  if (row >= update_rows) {
    return;
  }
  const int64_t slot = static_cast<int64_t>(slot_ids[row]);
  if (slot < 0 || slot >= num_blocks * block_size) {
    return;
  }

  const int64_t page = slot / block_size;
  const int64_t page_offset = slot % block_size;
  uint8_t* page_base = cache + page * page_stride_bytes;
  uint8_t* payload = page_base + page_offset * kModel1PayloadBytes;
  uint8_t* scales = page_base + block_size * kModel1PayloadBytes +
                    page_offset * kModel1ScaleBytes;
  const scalar_t* source = values + row * kModel1HeadDim;

  __shared__ int group_max_bits[kModel1NumScales];
  __shared__ float group_scales[kModel1NumScales];
  if (threadIdx.x < kModel1NumScales) {
    group_max_bits[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int64_t col = threadIdx.x; col < kModel1NopeDim; col += blockDim.x) {
    const int64_t group = col / kModel1QuantGroup;
    const float value = fabsf(static_cast<float>(source[col]));
    atomicMax(&group_max_bits[group], __float_as_int(value));
  }
  __syncthreads();

  if (threadIdx.x < kModel1NumScales) {
    const float max_abs = __int_as_float(group_max_bits[threadIdx.x]);
    const float raw_scale = fmaxf(max_abs / 448.0f, 1.0e-4f);
    const float scale = exp2f(ceilf(log2f(raw_scale)));
    group_scales[threadIdx.x] = scale;
    scales[threadIdx.x] =
        static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xffU);
  }
  if (threadIdx.x == kModel1NumScales) {
    scales[kModel1NumScales] = 0;
  }
  __syncthreads();

  for (int64_t col = threadIdx.x; col < kModel1NopeDim; col += blockDim.x) {
    const int64_t group = col / kModel1QuantGroup;
    float quantized = static_cast<float>(source[col]) / group_scales[group];
    quantized = fmaxf(-448.0f, fminf(quantized, 448.0f));
    payload[col] = static_cast<c10::Float8_e4m3fn>(quantized).x;
  }
  auto* rope = reinterpret_cast<scalar_t*>(payload + kModel1NopeDim);
  for (int64_t col = threadIdx.x; col < kModel1RopeDim; col += blockDim.x) {
    rope[col] = source[kModel1NopeDim + col];
  }
}

template <typename index_t>
void dispatch_dsv4_pack_model1_fp8_cache(const torch::Tensor& values,
                                         torch::Tensor& cache,
                                         const torch::Tensor& slots,
                                         int64_t update_rows,
                                         int64_t num_blocks,
                                         int64_t block_size,
                                         int64_t page_stride_bytes) {
  const dim3 grid(update_rows);
  const dim3 block(kModel1PackBlockSize);
  DISPATCH_FLOATING_TYPES(values.scalar_type(), "dsv4_pack_model1_fp8", [&] {
    dsv4_pack_model1_fp8_cache_kernel<scalar_t, index_t>
        <<<grid, block, 0, current_dsv4_cache_stream()>>>(
            values.data_ptr<scalar_t>(),
            cache.data_ptr<uint8_t>(),
            slots.data_ptr<index_t>(),
            update_rows,
            num_blocks,
            block_size,
            page_stride_bytes);
  });
  check_musa_kernel_launch("dsv4 kernel");
}
#endif

template <typename scalar_t, typename index_t>
__global__ void __launch_bounds__(kIndexerCacheBlockSize, 1)
    dsv4_indexer_cache_quant_scatter_kernel(
        const scalar_t* __restrict__ values,
        int8_t* __restrict__ cache,
        c10::Half* __restrict__ scale_cache,
        const index_t* __restrict__ slot_ids,
        int64_t update_rows,
        int64_t cache_rows,
        int64_t row_width) {
  const int64_t row = blockIdx.x;
  if (row >= update_rows) {
    return;
  }
  const int64_t slot = static_cast<int64_t>(slot_ids[row]);
  if (slot < 0 || slot >= cache_rows) {
    return;
  }

  __shared__ int max_abs_bits;
  __shared__ float row_scale;
  if (threadIdx.x == 0) {
    max_abs_bits = 0;
  }
  __syncthreads();

  const scalar_t* source = values + row * row_width;
  for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    const float value = fabsf(static_cast<float>(source[col]));
    atomicMax(&max_abs_bits, __float_as_int(value));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const float max_abs = __int_as_float(max_abs_bits);
    row_scale = fmaxf(max_abs / 127.0f, 1.0e-6f);
    scale_cache[slot] = static_cast<c10::Half>(row_scale);
  }
  __syncthreads();

  int8_t* destination = cache + slot * row_width;
  for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    float quantized = nearbyintf(static_cast<float>(source[col]) / row_scale);
    quantized = fmaxf(-127.0f, fminf(quantized, 127.0f));
    destination[col] = static_cast<int8_t>(quantized);
  }
}

template <typename scalar_t, typename block_index_t>
__global__ void __launch_bounds__(kIndexerCacheBlockSize, 1)
    dsv4_indexer_cache_gather_kernel(
        const int8_t* __restrict__ cache,
        const c10::Half* __restrict__ scale_cache,
        const block_index_t* __restrict__ block_table,
        scalar_t* __restrict__ output,
        int64_t sequence_index,
        int64_t gather_len,
        int64_t row_width,
        int64_t block_size,
        int64_t num_blocks,
        int64_t max_blocks_per_sequence) {
  const int64_t row = blockIdx.x;
  if (row >= gather_len) {
    return;
  }
  const int64_t logical_block = row / block_size;
  const int64_t block_offset = row % block_size;
  int64_t physical_block = static_cast<int64_t>(
      block_table[sequence_index * max_blocks_per_sequence + logical_block]);
  int64_t slot = physical_block * block_size + block_offset;
  if ((slot < 0 || slot >= num_blocks * block_size) && physical_block > 0) {
    slot = (physical_block - 1) * block_size + block_offset;
  }

  scalar_t* destination = output + row * row_width;
  if (slot < 0 || slot >= num_blocks * block_size) {
    for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
      destination[col] = scalar_t{};
    }
    return;
  }
  const float scale = static_cast<float>(scale_cache[slot]);
  const int8_t* source = cache + slot * row_width;
  for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    destination[col] =
        static_cast<scalar_t>(static_cast<float>(source[col]) * scale);
  }
}

template <typename scalar_t, typename block_index_t>
__global__ void __launch_bounds__(1024, 1) dsv4_indexer_paged_score_warp_kernel(
    const scalar_t* __restrict__ query,
    const float* __restrict__ weights,
    const int8_t* __restrict__ cache,
    const c10::Half* __restrict__ scale_cache,
    const block_index_t* __restrict__ block_table,
    float* __restrict__ scores,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks,
    int64_t max_blocks_per_sequence,
    int64_t query_absolute_start,
    int64_t compress_ratio) {
  const int64_t key_index = blockIdx.x;
  if (key_index >= kv_len) {
    return;
  }
  const int64_t valid_count =
      min((query_absolute_start + 1) / compress_ratio, kv_len);
  if (key_index >= valid_count) {
    if (threadIdx.x == 0) {
      scores[key_index] = -CUDART_INF_F;
    }
    return;
  }

  const int64_t logical_block = key_index / block_size;
  const int64_t block_offset = key_index % block_size;
  int64_t physical_block = static_cast<int64_t>(
      block_table[sequence_index * max_blocks_per_sequence + logical_block]);
  int64_t slot = physical_block * block_size + block_offset;
  if ((slot < 0 || slot >= num_blocks * block_size) && physical_block > 0) {
    slot = (physical_block - 1) * block_size + block_offset;
  }
  if (slot < 0 || slot >= num_blocks * block_size) {
    if (threadIdx.x == 0) {
      scores[key_index] = -CUDART_INF_F;
    }
    return;
  }

  const int64_t warp = threadIdx.x / kMusaWarpSize;
  const int64_t lane = threadIdx.x % kMusaWarpSize;
  __shared__ float head_scores[32];
  if (warp < num_heads) {
    const float key_scale = static_cast<float>(scale_cache[slot]);
    const int8_t* key = cache + slot * head_dim;
    const scalar_t* query_head = query + warp * head_dim;
    float dot = 0.0f;
    for (int64_t dim = lane; dim < head_dim; dim += kMusaWarpSize) {
      const scalar_t key_value =
          static_cast<scalar_t>(static_cast<float>(key[dim]) * key_scale);
      dot +=
          static_cast<float>(query_head[dim]) * static_cast<float>(key_value);
    }
    for (int offset = kMusaWarpSize / 2; offset > 0; offset /= 2) {
      dot += __shfl_xor_sync(0xffffffffU, dot, offset);
    }
    if (lane == 0) {
      head_scores[warp] = fmaxf(dot, 0.0f) * weights[warp];
    }
  }
  __syncthreads();

  if (warp == 0) {
    float weighted_score = lane < num_heads ? head_scores[lane] : 0.0f;
    for (int offset = kMusaWarpSize / 2; offset > 0; offset /= 2) {
      weighted_score += __shfl_xor_sync(0xffffffffU, weighted_score, offset);
    }
    if (lane == 0) {
      scores[key_index] = weighted_score;
    }
  }
}

template <typename index_t>
void dispatch_dsv4_indexer_cache_quant_scatter(const torch::Tensor& values,
                                               torch::Tensor& cache,
                                               torch::Tensor& scale_cache,
                                               const torch::Tensor& slots,
                                               int64_t update_rows,
                                               int64_t cache_rows,
                                               int64_t row_width) {
  const dim3 grid(update_rows);
  const dim3 block(kIndexerCacheBlockSize);
  DISPATCH_FLOATING_TYPES(
      values.scalar_type(), "dsv4_indexer_quant_scatter", [&] {
        dsv4_indexer_cache_quant_scatter_kernel<scalar_t, index_t>
            <<<grid, block, 0, current_dsv4_cache_stream()>>>(
                values.data_ptr<scalar_t>(),
                cache.data_ptr<int8_t>(),
                scale_cache.data_ptr<c10::Half>(),
                slots.data_ptr<index_t>(),
                update_rows,
                cache_rows,
                row_width);
      });
  check_musa_kernel_launch("dsv4 kernel");
}

template <typename scalar_t>
void dispatch_dsv4_indexer_cache_gather(const torch::Tensor& cache,
                                        const torch::Tensor& scale_cache,
                                        const torch::Tensor& block_table,
                                        torch::Tensor& output,
                                        int64_t sequence_index,
                                        int64_t gather_len,
                                        int64_t row_width,
                                        int64_t block_size,
                                        int64_t num_blocks,
                                        int64_t max_blocks_per_sequence) {
  const dim3 grid(gather_len);
  const dim3 block(std::min<int64_t>(row_width, kIndexerCacheBlockSize));
  const musaStream_t stream = current_dsv4_cache_stream();
  if (block_table.scalar_type() == torch::kInt32) {
    dsv4_indexer_cache_gather_kernel<scalar_t, int32_t>
        <<<grid, block, 0, stream>>>(cache.data_ptr<int8_t>(),
                                     scale_cache.data_ptr<c10::Half>(),
                                     block_table.data_ptr<int32_t>(),
                                     output.data_ptr<scalar_t>(),
                                     sequence_index,
                                     gather_len,
                                     row_width,
                                     block_size,
                                     num_blocks,
                                     max_blocks_per_sequence);
  } else {
    dsv4_indexer_cache_gather_kernel<scalar_t, int64_t>
        <<<grid, block, 0, stream>>>(cache.data_ptr<int8_t>(),
                                     scale_cache.data_ptr<c10::Half>(),
                                     block_table.data_ptr<int64_t>(),
                                     output.data_ptr<scalar_t>(),
                                     sequence_index,
                                     gather_len,
                                     row_width,
                                     block_size,
                                     num_blocks,
                                     max_blocks_per_sequence);
  }
  check_musa_kernel_launch("dsv4 kernel");
}

template <typename scalar_t, typename block_index_t>
__global__ void __launch_bounds__(kIndexerCacheBlockSize, 1)
    dsv4_indexer_paged_score_kernel(
        const scalar_t* __restrict__ query,
        const float* __restrict__ weights,
        const int8_t* __restrict__ cache,
        const c10::Half* __restrict__ scale_cache,
        const block_index_t* __restrict__ block_table,
        float* __restrict__ scores,
        int64_t sequence_index,
        int64_t kv_len,
        int64_t num_heads,
        int64_t head_dim,
        int64_t block_size,
        int64_t num_blocks,
        int64_t max_blocks_per_sequence,
        int64_t query_absolute_start,
        int64_t compress_ratio) {
  const int64_t key_index = blockIdx.x;
  if (key_index >= kv_len) {
    return;
  }
  const int64_t valid_count =
      min((query_absolute_start + 1) / compress_ratio, kv_len);
  if (key_index >= valid_count) {
    if (threadIdx.x == 0) {
      scores[key_index] = -CUDART_INF_F;
    }
    return;
  }

  const int64_t logical_block = key_index / block_size;
  const int64_t block_offset = key_index % block_size;
  int64_t physical_block = static_cast<int64_t>(
      block_table[sequence_index * max_blocks_per_sequence + logical_block]);
  int64_t slot = physical_block * block_size + block_offset;
  if ((slot < 0 || slot >= num_blocks * block_size) && physical_block > 0) {
    slot = (physical_block - 1) * block_size + block_offset;
  }
  if (slot < 0 || slot >= num_blocks * block_size) {
    if (threadIdx.x == 0) {
      scores[key_index] = -CUDART_INF_F;
    }
    return;
  }

  __shared__ float reduction[kIndexerCacheBlockSize];
  __shared__ float weighted_score;
  if (threadIdx.x == 0) {
    weighted_score = 0.0f;
  }
  __syncthreads();

  const float key_scale = static_cast<float>(scale_cache[slot]);
  const int8_t* key = cache + slot * head_dim;
  for (int64_t head = 0; head < num_heads; ++head) {
    float dot = 0.0f;
    const scalar_t* query_head = query + head * head_dim;
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      // Match the paged baseline: INT8 dequantization is rounded to the query
      // dtype before the FP32 dot product.
      const scalar_t key_value =
          static_cast<scalar_t>(static_cast<float>(key[dim]) * key_scale);
      dot +=
          static_cast<float>(query_head[dim]) * static_cast<float>(key_value);
    }
    reduction[threadIdx.x] = dot;
    __syncthreads();
    for (int64_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      weighted_score += fmaxf(reduction[0], 0.0f) * weights[head];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scores[key_index] = weighted_score;
  }
}

template <typename scalar_t>
void dispatch_dsv4_indexer_paged_score(const torch::Tensor& query,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& cache,
                                       const torch::Tensor& scale_cache,
                                       const torch::Tensor& block_table,
                                       torch::Tensor& scores,
                                       int64_t sequence_index,
                                       int64_t kv_len,
                                       int64_t query_absolute_start,
                                       int64_t compress_ratio) {
  const dim3 grid(kv_len);
  const bool use_warp_score = dsv4_musa_indexer_warp_score_enabled() &&
                              query.size(1) > 0 && query.size(1) <= 32;
  const dim3 block(use_warp_score ? query.size(1) * kMusaWarpSize
                                  : kIndexerCacheBlockSize);
  const musaStream_t stream = current_dsv4_cache_stream();
  if (block_table.scalar_type() == torch::kInt32) {
    if (use_warp_score) {
      dsv4_indexer_paged_score_warp_kernel<scalar_t, int32_t>
          <<<grid, block, 0, stream>>>(query.data_ptr<scalar_t>(),
                                       weights.data_ptr<float>(),
                                       cache.data_ptr<int8_t>(),
                                       scale_cache.data_ptr<c10::Half>(),
                                       block_table.data_ptr<int32_t>(),
                                       scores.data_ptr<float>(),
                                       sequence_index,
                                       kv_len,
                                       query.size(1),
                                       query.size(2),
                                       cache.size(1),
                                       cache.size(0),
                                       block_table.size(1),
                                       query_absolute_start,
                                       compress_ratio);
    } else {
      dsv4_indexer_paged_score_kernel<scalar_t, int32_t>
          <<<grid, block, 0, stream>>>(query.data_ptr<scalar_t>(),
                                       weights.data_ptr<float>(),
                                       cache.data_ptr<int8_t>(),
                                       scale_cache.data_ptr<c10::Half>(),
                                       block_table.data_ptr<int32_t>(),
                                       scores.data_ptr<float>(),
                                       sequence_index,
                                       kv_len,
                                       query.size(1),
                                       query.size(2),
                                       cache.size(1),
                                       cache.size(0),
                                       block_table.size(1),
                                       query_absolute_start,
                                       compress_ratio);
    }
  } else {
    if (use_warp_score) {
      dsv4_indexer_paged_score_warp_kernel<scalar_t, int64_t>
          <<<grid, block, 0, stream>>>(query.data_ptr<scalar_t>(),
                                       weights.data_ptr<float>(),
                                       cache.data_ptr<int8_t>(),
                                       scale_cache.data_ptr<c10::Half>(),
                                       block_table.data_ptr<int64_t>(),
                                       scores.data_ptr<float>(),
                                       sequence_index,
                                       kv_len,
                                       query.size(1),
                                       query.size(2),
                                       cache.size(1),
                                       cache.size(0),
                                       block_table.size(1),
                                       query_absolute_start,
                                       compress_ratio);
    } else {
      dsv4_indexer_paged_score_kernel<scalar_t, int64_t>
          <<<grid, block, 0, stream>>>(query.data_ptr<scalar_t>(),
                                       weights.data_ptr<float>(),
                                       cache.data_ptr<int8_t>(),
                                       scale_cache.data_ptr<c10::Half>(),
                                       block_table.data_ptr<int64_t>(),
                                       scores.data_ptr<float>(),
                                       sequence_index,
                                       kv_len,
                                       query.size(1),
                                       query.size(2),
                                       cache.size(1),
                                       cache.size(0),
                                       block_table.size(1),
                                       query_absolute_start,
                                       compress_ratio);
    }
  }
  check_musa_kernel_launch("dsv4 kernel");
}

// Adapted from SGLang's MUSA fast_topk kernel. DSA uses the selected key set,
// so the radix path deliberately does not pay for sorting the final indices.
__device__ __forceinline__ uint32_t dsv4_float_order_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) != 0 ? ~bits : (bits | 0x80000000U);
}

__device__ void dsv4_indexer_radix_topk_device(const float* __restrict__ scores,
                                               int32_t* __restrict__ indices,
                                               int32_t length) {
  constexpr int32_t kCandidateCapacity =
      kIndexerRadixSmemBytes / (2 * sizeof(int32_t));
  __shared__ int32_t histogram_buffers[2][kIndexerRadixBins + 128];
  __shared__ int32_t output_count;
  __shared__ int32_t threshold_bin;
  __shared__ int32_t candidate_counts[2];
  __shared__ int32_t last_remain;
  extern __shared__ int32_t candidate_storage[];
  int32_t* candidate_indices[2] = {candidate_storage,
                                   candidate_storage + kCandidateCapacity};

  const int32_t tid = threadIdx.x;
  int32_t remaining = kIndexerRadixTopk;
  int32_t* histogram = histogram_buffers[0];

  if (tid < kIndexerRadixBins + 1) {
    histogram[tid] = 0;
  }
  if (tid == 0) {
    output_count = 0;
    candidate_counts[0] = 0;
  }
  __syncthreads();

  for (int32_t index = tid; index < length; index += blockDim.x) {
    const int32_t bin =
        static_cast<int32_t>(dsv4_float_order_key(scores[index]) >> 24);
    atomicAdd(&histogram[bin], 1);
  }
  __syncthreads();

  auto suffix_sum = [&]() {
#pragma unroll
    for (int32_t pass = 0; pass < 8; ++pass) {
      if (tid < kIndexerRadixBins) {
        const int32_t offset = 1 << pass;
        const int32_t source = pass & 1;
        int32_t value = histogram_buffers[source][tid];
        if (tid < kIndexerRadixBins - offset) {
          value += histogram_buffers[source][tid + offset];
        }
        histogram_buffers[source ^ 1][tid] = value;
      }
      __syncthreads();
    }
  };

  suffix_sum();
  if (tid < kIndexerRadixBins && histogram[tid] > remaining &&
      histogram[tid + 1] <= remaining) {
    threshold_bin = tid;
  }
  __syncthreads();

  int32_t active_threshold = threshold_bin;
  remaining -= histogram[active_threshold + 1];
  if (remaining == 0) {
    for (int32_t index = tid; index < length; index += blockDim.x) {
      const int32_t bin =
          static_cast<int32_t>(dsv4_float_order_key(scores[index]) >> 24);
      if (bin > active_threshold) {
        indices[atomicAdd(&output_count, 1)] = index;
      }
    }
    return;
  }

  if (tid < kIndexerRadixBins + 1) {
    histogram[tid] = 0;
  }
  __syncthreads();
  for (int32_t index = tid; index < length; index += blockDim.x) {
    const uint32_t key = dsv4_float_order_key(scores[index]);
    const int32_t bin = static_cast<int32_t>(key >> 24);
    if (bin > active_threshold) {
      indices[atomicAdd(&output_count, 1)] = index;
    } else if (bin == active_threshold) {
      const int32_t position = atomicAdd(&candidate_counts[0], 1);
      if (position < kCandidateCapacity) {
        candidate_indices[0][position] = index;
        atomicAdd(&histogram[(key >> 16) & 0xffU], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int32_t round = 0; round < 3; ++round) {
    const int32_t read_buffer = round & 1;
    const int32_t write_buffer = read_buffer ^ 1;
    const int32_t input_count =
        min(candidate_counts[read_buffer], kCandidateCapacity);

    suffix_sum();
    if (tid < kIndexerRadixBins && histogram[tid] > remaining &&
        histogram[tid + 1] <= remaining) {
      threshold_bin = tid;
      candidate_counts[write_buffer] = 0;
      last_remain = remaining - histogram[tid + 1];
    }
    __syncthreads();

    active_threshold = threshold_bin;
    remaining -= histogram[active_threshold + 1];
    const int32_t shift = 16 - round * 8;
    if (remaining == 0) {
      for (int32_t i = tid; i < input_count; i += blockDim.x) {
        const int32_t index = candidate_indices[read_buffer][i];
        const int32_t bin = static_cast<int32_t>(
            (dsv4_float_order_key(scores[index]) >> shift) & 0xffU);
        if (bin > active_threshold) {
          indices[atomicAdd(&output_count, 1)] = index;
        }
      }
      return;
    }

    if (round < 2 && tid < kIndexerRadixBins + 1) {
      histogram[tid] = 0;
    }
    __syncthreads();
    for (int32_t i = tid; i < input_count; i += blockDim.x) {
      const int32_t index = candidate_indices[read_buffer][i];
      const uint32_t key = dsv4_float_order_key(scores[index]);
      const int32_t bin = static_cast<int32_t>((key >> shift) & 0xffU);
      if (bin > active_threshold) {
        indices[atomicAdd(&output_count, 1)] = index;
      } else if (bin == active_threshold) {
        if (round == 2) {
          const int32_t position = atomicAdd(&last_remain, -1);
          if (position > 0) {
            indices[kIndexerRadixTopk - position] = index;
          }
        } else {
          const int32_t position =
              atomicAdd(&candidate_counts[write_buffer], 1);
          if (position < kCandidateCapacity) {
            candidate_indices[write_buffer][position] = index;
            atomicAdd(&histogram[(key >> (shift - 8)) & 0xffU], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <typename block_index_t>
__device__ int32_t
dsv4_map_logical_to_physical(int64_t logical,
                             const block_index_t* __restrict__ block_table,
                             int64_t sequence_index,
                             int64_t max_blocks_per_sequence,
                             int64_t block_size);

template <typename scalar_t,
          typename block_index_t,
          typename mate_compressed_index_t,
          typename mate_swa_index_t>
__global__ void __launch_bounds__(kIndexerRadixThreads, 1)
    dsv4_indexer_paged_score_fused_radix_kernel(
        const scalar_t* __restrict__ query,
        const float* __restrict__ weights,
        const int8_t* __restrict__ cache,
        const c10::Half* __restrict__ scale_cache,
        const block_index_t* __restrict__ block_table,
        float* __restrict__ scores,
        int32_t* __restrict__ indices,
        int32_t* __restrict__ completion_counter,
        int64_t sequence_index,
        int64_t valid_count,
        int64_t num_heads,
        int64_t head_dim,
        int64_t block_size,
        int64_t num_blocks,
        int64_t max_blocks_per_sequence,
        const mate_compressed_index_t* __restrict__ mate_compressed_block_table,
        const mate_swa_index_t* __restrict__ mate_swa_block_table,
        int32_t* __restrict__ mate_physical_indices,
        int32_t* __restrict__ mate_extra_physical_indices,
        int64_t mate_main_count,
        int64_t mate_extra_count,
        int64_t mate_kv_len,
        int64_t mate_window_size,
        int64_t mate_block_size,
        int64_t mate_compressed_max_blocks,
        int64_t mate_swa_max_blocks,
        int64_t mate_sequence_index) {
  // Score one key with one warp per indexer head. The launch uses exactly the
  // number of warps needed by the model (subject to the radix minimum), rather
  // than a fixed 1024-thread block.
  __shared__ float head_scores[32];
  const int32_t warp = threadIdx.x / kMusaWarpSize;
  const int32_t lane = threadIdx.x % kMusaWarpSize;

  for (int64_t key_index = blockIdx.x; key_index < valid_count;
       key_index += gridDim.x) {
    const int64_t logical_block = key_index / block_size;
    const int64_t block_offset = key_index % block_size;
    int64_t physical_block = static_cast<int64_t>(
        block_table[sequence_index * max_blocks_per_sequence + logical_block]);
    int64_t slot = physical_block * block_size + block_offset;
    if ((slot < 0 || slot >= num_blocks * block_size) && physical_block > 0) {
      slot = (physical_block - 1) * block_size + block_offset;
    }
    if (slot < 0 || slot >= num_blocks * block_size) {
      if (threadIdx.x == 0) {
        scores[key_index] = -CUDART_INF_F;
      }
      __syncthreads();
      continue;
    }

    const float key_scale = static_cast<float>(scale_cache[slot]);
    const int8_t* key = cache + slot * head_dim;
    if (warp < num_heads) {
      float dot = 0.0f;
      const scalar_t* query_head = query + warp * head_dim;
      for (int64_t dim = lane; dim < head_dim; dim += kMusaWarpSize) {
        const scalar_t key_value =
            static_cast<scalar_t>(static_cast<float>(key[dim]) * key_scale);
        dot +=
            static_cast<float>(query_head[dim]) * static_cast<float>(key_value);
      }
      for (int32_t offset = kMusaWarpSize / 2; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xffffffffU, dot, offset);
      }
      if (lane == 0) {
        head_scores[warp] = fmaxf(dot, 0.0f) * weights[warp];
      }
    }
    __syncthreads();
    if (warp == 0) {
      float weighted_score = lane < num_heads ? head_scores[lane] : 0.0f;
      for (int32_t offset = kMusaWarpSize / 2; offset > 0; offset >>= 1) {
        weighted_score += __shfl_xor_sync(0xffffffffU, weighted_score, offset);
      }
      if (lane == 0) {
        scores[key_index] = weighted_score;
      }
    }
    __syncthreads();
  }

  // Publish this block's score writes before announcing completion. The last
  // block performs radix TopK in-place, avoiding a cooperative grid barrier
  // and a second kernel launch.
  __shared__ int32_t is_last_block;
  if (threadIdx.x == 0) {
    __threadfence();
    is_last_block =
        atomicAdd(completion_counter, 1) == static_cast<int32_t>(gridDim.x - 1);
  }
  __syncthreads();
  if (is_last_block != 0) {
    dsv4_indexer_radix_topk_device(
        scores, indices, static_cast<int32_t>(valid_count));
    if (mate_physical_indices != nullptr &&
        mate_extra_physical_indices != nullptr) {
      // Keep logical-to-physical mapping in the score/TopK launch for the
      // hot C4 decode path. Other variants retain the prepare fallback.
      __syncthreads();
      for (int64_t index = threadIdx.x; index < mate_main_count;
           index += blockDim.x) {
        mate_physical_indices[index] =
            dsv4_map_logical_to_physical<mate_compressed_index_t>(
                static_cast<int64_t>(indices[index]),
                mate_compressed_block_table,
                mate_sequence_index,
                mate_compressed_max_blocks,
                mate_block_size);
      }
      for (int64_t index = threadIdx.x; index < mate_extra_count;
           index += blockDim.x) {
        const int64_t row_extra_len =
            min(min(int64_t{128}, mate_window_size), mate_kv_len);
        const int64_t logical =
            index < row_extra_len ? mate_kv_len - row_extra_len + index : -1;
        mate_extra_physical_indices[index] =
            dsv4_map_logical_to_physical<mate_swa_index_t>(logical,
                                                           mate_swa_block_table,
                                                           mate_sequence_index,
                                                           mate_swa_max_blocks,
                                                           mate_block_size);
      }
    }
  }
}

template <typename scalar_t,
          typename block_index_t,
          typename mate_compressed_index_t,
          typename mate_swa_index_t>
bool dispatch_dsv4_indexer_paged_score_fused_radix(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    torch::Tensor& scores,
    torch::Tensor& indices,
    torch::Tensor& completion_counter,
    int64_t sequence_index,
    int64_t valid_count,
    const torch::Tensor& mate_compressed_block_table,
    const torch::Tensor& mate_swa_block_table,
    const torch::Tensor& mate_physical_indices,
    const torch::Tensor& mate_extra_physical_indices,
    int64_t mate_kv_len,
    int64_t mate_window_size,
    int64_t mate_block_size,
    int64_t mate_sequence_index) {
  musaDeviceProp properties{};
  const musaError_t properties_status =
      musaGetDeviceProperties(&properties, query.device().index());
  if (properties_status != musaSuccess) {
    musaGetLastError();
    return false;
  }

  // The score phase uses one warp per indexer head. DeepSeek-V4 normally has
  // 8 or 16 heads, so a fixed 1024-thread block leaves half or three quarters
  // of the block idle while it computes one key. The radix device routine only
  // needs at least one thread per 8-bit histogram bin; use the smallest valid
  // multiple of a warp and retain 1024 for larger head counts.
  const int32_t launch_threads = static_cast<int32_t>(std::min<int64_t>(
      kIndexerRadixThreads,
      std::max<int64_t>(kIndexerRadixBins, query.size(1) * kMusaWarpSize)));
  int32_t blocks_per_sm = 0;
  const musaError_t occupancy_status =
      musaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          dsv4_indexer_paged_score_fused_radix_kernel<scalar_t,
                                                      block_index_t,
                                                      mate_compressed_index_t,
                                                      mate_swa_index_t>,
          launch_threads,
          kIndexerRadixSmemBytes);
  if (occupancy_status != musaSuccess || blocks_per_sm <= 0) {
    musaGetLastError();
    return false;
  }
  const int64_t grid_blocks =
      std::max<int64_t>(1,
                        std::min<int64_t>(valid_count,
                                          static_cast<int64_t>(blocks_per_sm) *
                                              properties.multiProcessorCount));

  const scalar_t* query_ptr = query.data_ptr<scalar_t>();
  const float* weights_ptr = weights.data_ptr<float>();
  const int8_t* cache_ptr = cache.data_ptr<int8_t>();
  const c10::Half* scale_ptr = scale_cache.data_ptr<c10::Half>();
  const block_index_t* block_table_ptr = block_table.data_ptr<block_index_t>();
  float* scores_ptr = scores.data_ptr<float>();
  int32_t* indices_ptr = indices.data_ptr<int32_t>();
  int32_t* completion_counter_ptr = completion_counter.data_ptr<int32_t>();
  const bool mate_mapping_enabled =
      mate_compressed_block_table.defined() && mate_swa_block_table.defined() &&
      mate_physical_indices.defined() && mate_extra_physical_indices.defined();
  const mate_compressed_index_t* mate_compressed_block_table_ptr =
      mate_mapping_enabled
          ? mate_compressed_block_table.data_ptr<mate_compressed_index_t>()
          : nullptr;
  const mate_swa_index_t* mate_swa_block_table_ptr =
      mate_mapping_enabled ? mate_swa_block_table.data_ptr<mate_swa_index_t>()
                           : nullptr;
  int32_t* mate_physical_indices_ptr =
      mate_mapping_enabled ? mate_physical_indices.data_ptr<int32_t>()
                           : nullptr;
  int32_t* mate_extra_physical_indices_ptr =
      mate_mapping_enabled ? mate_extra_physical_indices.data_ptr<int32_t>()
                           : nullptr;
  int64_t num_heads = query.size(1);
  int64_t head_dim = query.size(2);
  int64_t block_size = cache.size(1);
  int64_t num_blocks = cache.size(0);
  int64_t max_blocks_per_sequence = block_table.size(1);
  dsv4_indexer_paged_score_fused_radix_kernel<scalar_t,
                                              block_index_t,
                                              mate_compressed_index_t,
                                              mate_swa_index_t>
      <<<dim3(static_cast<unsigned int>(grid_blocks)),
         dim3(static_cast<unsigned int>(launch_threads)),
         kIndexerRadixSmemBytes,
         current_dsv4_cache_stream()>>>(
          query_ptr,
          weights_ptr,
          cache_ptr,
          scale_ptr,
          block_table_ptr,
          scores_ptr,
          indices_ptr,
          completion_counter_ptr,
          sequence_index,
          valid_count,
          num_heads,
          head_dim,
          block_size,
          num_blocks,
          max_blocks_per_sequence,
          mate_compressed_block_table_ptr,
          mate_swa_block_table_ptr,
          mate_physical_indices_ptr,
          mate_extra_physical_indices_ptr,
          mate_mapping_enabled ? mate_physical_indices.numel() : int64_t{0},
          mate_mapping_enabled ? mate_extra_physical_indices.numel()
                               : int64_t{0},
          mate_kv_len,
          mate_window_size,
          mate_block_size,
          mate_mapping_enabled ? mate_compressed_block_table.size(1)
                               : int64_t{0},
          mate_mapping_enabled ? mate_swa_block_table.size(1) : int64_t{0},
          mate_sequence_index);
  const musaError_t launch_status = musaGetLastError();
  if (launch_status != musaSuccess) {
    musaGetLastError();
    return false;
  }
  return true;
}

template <typename scalar_t, typename block_index_t>
bool dispatch_dsv4_indexer_paged_score_fused_radix_mate_types(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    torch::Tensor& scores,
    torch::Tensor& indices,
    torch::Tensor& completion_counter,
    int64_t sequence_index,
    int64_t valid_count,
    const torch::Tensor& mate_compressed_block_table,
    const torch::Tensor& mate_swa_block_table,
    const torch::Tensor& mate_physical_indices,
    const torch::Tensor& mate_extra_physical_indices,
    int64_t mate_kv_len,
    int64_t mate_window_size,
    int64_t mate_block_size,
    int64_t mate_sequence_index) {
  if (!mate_compressed_block_table.defined()) {
    return dispatch_dsv4_indexer_paged_score_fused_radix<scalar_t,
                                                         block_index_t,
                                                         int32_t,
                                                         int32_t>(
        query,
        weights,
        cache,
        scale_cache,
        block_table,
        scores,
        indices,
        completion_counter,
        sequence_index,
        valid_count,
        mate_compressed_block_table,
        mate_swa_block_table,
        mate_physical_indices,
        mate_extra_physical_indices,
        mate_kv_len,
        mate_window_size,
        mate_block_size,
        mate_sequence_index);
  }
#define DISPATCH_DSV4_INDEXER_MATE_TYPES(COMPRESSED_T, SWA_T)         \
  return dispatch_dsv4_indexer_paged_score_fused_radix<scalar_t,      \
                                                       block_index_t, \
                                                       COMPRESSED_T,  \
                                                       SWA_T>(        \
      query,                                                          \
      weights,                                                        \
      cache,                                                          \
      scale_cache,                                                    \
      block_table,                                                    \
      scores,                                                         \
      indices,                                                        \
      completion_counter,                                             \
      sequence_index,                                                 \
      valid_count,                                                    \
      mate_compressed_block_table,                                    \
      mate_swa_block_table,                                           \
      mate_physical_indices,                                          \
      mate_extra_physical_indices,                                    \
      mate_kv_len,                                                    \
      mate_window_size,                                               \
      mate_block_size,                                                \
      mate_sequence_index)
  if (mate_compressed_block_table.scalar_type() == torch::kInt32) {
    if (mate_swa_block_table.scalar_type() == torch::kInt32) {
      DISPATCH_DSV4_INDEXER_MATE_TYPES(int32_t, int32_t);
    }
    DISPATCH_DSV4_INDEXER_MATE_TYPES(int32_t, int64_t);
  }
  if (mate_swa_block_table.scalar_type() == torch::kInt32) {
    DISPATCH_DSV4_INDEXER_MATE_TYPES(int64_t, int32_t);
  }
  DISPATCH_DSV4_INDEXER_MATE_TYPES(int64_t, int64_t);
#undef DISPATCH_DSV4_INDEXER_MATE_TYPES
}

__global__ void __launch_bounds__(kIndexerRadixThreads, 1)
    dsv4_indexer_radix_topk_kernel(const float* __restrict__ scores,
                                   int32_t* __restrict__ indices,
                                   int32_t length) {
  dsv4_indexer_radix_topk_device(scores, indices, length);
}

__global__ void __launch_bounds__(kIndexerCacheBlockSize, 1)
    dsv4_indexer_all_valid_kernel(int32_t* __restrict__ indices,
                                  int64_t selected_count,
                                  int64_t topk) {
  for (int64_t index = threadIdx.x; index < topk; index += blockDim.x) {
    indices[index] = index < selected_count ? static_cast<int32_t>(index) : -1;
  }
}

void dispatch_dsv4_indexer_radix_topk(const torch::Tensor& scores,
                                      torch::Tensor& indices,
                                      int64_t valid_count) {
  CHECK_EQ(indices.numel(), kIndexerRadixTopk);
  CHECK_GT(valid_count, kIndexerRadixTopk);
  const musaStream_t stream = current_dsv4_cache_stream();
  dsv4_indexer_radix_topk_kernel<<<1,
                                   kIndexerRadixThreads,
                                   kIndexerRadixSmemBytes,
                                   stream>>>(scores.data_ptr<float>(),
                                             indices.data_ptr<int32_t>(),
                                             static_cast<int32_t>(valid_count));
  check_musa_kernel_launch("dsv4 kernel");
}

template <typename scalar_t, typename block_index_t>
__global__ void __launch_bounds__(kIndexerCacheBlockSize, 1)
    dsv4_indexer_paged_score_topk_kernel(
        const scalar_t* __restrict__ query,
        const float* __restrict__ weights,
        const int8_t* __restrict__ cache,
        const c10::Half* __restrict__ scale_cache,
        const block_index_t* __restrict__ block_table,
        float* __restrict__ topk_values,
        int32_t* __restrict__ topk_indices,
        int32_t* __restrict__ topk_lock,
        int64_t sequence_index,
        int64_t valid_count,
        int64_t selected_count,
        int64_t num_heads,
        int64_t head_dim,
        int64_t block_size,
        int64_t num_blocks,
        int64_t max_blocks_per_sequence) {
  const int64_t key_index = blockIdx.x;
  if (key_index >= valid_count) {
    return;
  }
  const int64_t logical_block = key_index / block_size;
  const int64_t block_offset = key_index % block_size;
  int64_t physical_block = static_cast<int64_t>(
      block_table[sequence_index * max_blocks_per_sequence + logical_block]);
  int64_t slot = physical_block * block_size + block_offset;
  if ((slot < 0 || slot >= num_blocks * block_size) && physical_block > 0) {
    slot = (physical_block - 1) * block_size + block_offset;
  }
  if (slot < 0 || slot >= num_blocks * block_size) {
    return;
  }

  __shared__ float reduction[kIndexerCacheBlockSize];
  __shared__ float weighted_score;
  if (threadIdx.x == 0) {
    weighted_score = 0.0f;
  }
  __syncthreads();
  const float key_scale = static_cast<float>(scale_cache[slot]);
  const int8_t* key = cache + slot * head_dim;
  for (int64_t head = 0; head < num_heads; ++head) {
    float dot = 0.0f;
    const scalar_t* query_head = query + head * head_dim;
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const scalar_t key_value =
          static_cast<scalar_t>(static_cast<float>(key[dim]) * key_scale);
      dot +=
          static_cast<float>(query_head[dim]) * static_cast<float>(key_value);
    }
    reduction[threadIdx.x] = dot;
    __syncthreads();
    for (int64_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      weighted_score += fmaxf(reduction[0], 0.0f) * weights[head];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    while (atomicCAS(topk_lock, 0, 1) != 0) {
    }
    float candidate_value = weighted_score;
    int32_t candidate_index = static_cast<int32_t>(key_index);
    for (int64_t position = 0; position < selected_count; ++position) {
      const int32_t current_index = topk_indices[position];
      const float current_value =
          current_index >= 0 ? topk_values[position] : -CUDART_INF_F;
      const bool candidate_is_better =
          current_index < 0 || candidate_value > current_value ||
          (candidate_value == current_value && candidate_index < current_index);
      if (candidate_is_better) {
        topk_values[position] = candidate_value;
        topk_indices[position] = candidate_index;
        candidate_value = current_value;
        candidate_index = current_index;
        if (candidate_index < 0) {
          break;
        }
      }
    }
    __threadfence();
    atomicExch(topk_lock, 0);
  }
}

struct Dsv4IndexerSingleTopkWorkspace {
  int64_t topk = 0;
  int64_t score_capacity = 0;
  int64_t air_workspace_capacity = 0;
  c10::DeviceIndex device = -1;
  torch::Tensor values;
  torch::Tensor lock;
  torch::Tensor scores;
  torch::Tensor air_workspace;
};

Dsv4IndexerSingleTopkWorkspace& dsv4_indexer_single_topk_workspace(
    const torch::Tensor& query,
    int64_t topk,
    int64_t score_count = 0,
    int64_t air_workspace_bytes = 0) {
  thread_local Dsv4IndexerSingleTopkWorkspace workspace;
  const bool device_changed = workspace.device != query.device().index();
  if (workspace.topk < topk || device_changed) {
    workspace.topk = topk;
    workspace.device = query.device().index();
    workspace.values =
        torch::empty({topk}, query.options().dtype(torch::kFloat32));
    workspace.lock = torch::empty({1}, query.options().dtype(torch::kInt32));
  }
  if (workspace.score_capacity < score_count || device_changed) {
    workspace.score_capacity = score_count;
    workspace.scores =
        torch::empty({score_count}, query.options().dtype(torch::kFloat32));
  }
  if (workspace.air_workspace_capacity < air_workspace_bytes ||
      device_changed) {
    workspace.air_workspace_capacity = air_workspace_bytes;
    workspace.air_workspace = torch::empty(
        {air_workspace_bytes}, query.options().dtype(torch::kUInt8));
  }
  return workspace;
}

template <typename scalar_t>
void dispatch_dsv4_indexer_paged_score_topk(const torch::Tensor& query,
                                            const torch::Tensor& weights,
                                            const torch::Tensor& cache,
                                            const torch::Tensor& scale_cache,
                                            const torch::Tensor& block_table,
                                            torch::Tensor& topk_values,
                                            torch::Tensor& topk_indices,
                                            torch::Tensor& topk_lock,
                                            int64_t sequence_index,
                                            int64_t valid_count,
                                            int64_t selected_count) {
  const dim3 grid(valid_count);
  const dim3 block(kIndexerCacheBlockSize);
  const musaStream_t stream = current_dsv4_cache_stream();
  if (block_table.scalar_type() == torch::kInt32) {
    dsv4_indexer_paged_score_topk_kernel<scalar_t, int32_t>
        <<<grid, block, 0, stream>>>(query.data_ptr<scalar_t>(),
                                     weights.data_ptr<float>(),
                                     cache.data_ptr<int8_t>(),
                                     scale_cache.data_ptr<c10::Half>(),
                                     block_table.data_ptr<int32_t>(),
                                     topk_values.data_ptr<float>(),
                                     topk_indices.data_ptr<int32_t>(),
                                     topk_lock.data_ptr<int32_t>(),
                                     sequence_index,
                                     valid_count,
                                     selected_count,
                                     query.size(1),
                                     query.size(2),
                                     cache.size(1),
                                     cache.size(0),
                                     block_table.size(1));
  } else {
    dsv4_indexer_paged_score_topk_kernel<scalar_t, int64_t>
        <<<grid, block, 0, stream>>>(query.data_ptr<scalar_t>(),
                                     weights.data_ptr<float>(),
                                     cache.data_ptr<int8_t>(),
                                     scale_cache.data_ptr<c10::Half>(),
                                     block_table.data_ptr<int64_t>(),
                                     topk_values.data_ptr<float>(),
                                     topk_indices.data_ptr<int32_t>(),
                                     topk_lock.data_ptr<int32_t>(),
                                     sequence_index,
                                     valid_count,
                                     selected_count,
                                     query.size(1),
                                     query.size(2),
                                     cache.size(1),
                                     cache.size(0),
                                     block_table.size(1));
  }
  check_musa_kernel_launch("dsv4 kernel");
}

template <typename block_index_t>
__global__ void __launch_bounds__(kLogicalToPhysicalBlockSize, 1)
    dsv4_logical_to_physical_topk_kernel(
        const int32_t* __restrict__ logical_indices,
        const block_index_t* __restrict__ block_table,
        int32_t* __restrict__ physical_indices,
        int64_t num_indices,
        int64_t sequence_index,
        int64_t max_blocks_per_sequence,
        int64_t block_size) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < num_indices;
       index += blockDim.x * gridDim.x) {
    const int64_t logical = static_cast<int64_t>(logical_indices[index]);
    if (logical < 0) {
      physical_indices[index] = -1;
      continue;
    }
    const int64_t logical_block = logical / block_size;
    if (logical_block >= max_blocks_per_sequence) {
      physical_indices[index] = -1;
      continue;
    }
    const int64_t physical_block = static_cast<int64_t>(
        block_table[sequence_index * max_blocks_per_sequence + logical_block]);
    const int64_t physical = physical_block * block_size + logical % block_size;
    physical_indices[index] = physical_block < 0 || physical > INT32_MAX
                                  ? -1
                                  : static_cast<int32_t>(physical);
  }
}

template <typename block_index_t>
__device__ int32_t
dsv4_map_logical_to_physical(int64_t logical,
                             const block_index_t* __restrict__ block_table,
                             int64_t sequence_index,
                             int64_t max_blocks_per_sequence,
                             int64_t block_size) {
  if (logical < 0) {
    return -1;
  }
  const int64_t logical_block = logical / block_size;
  if (logical_block >= max_blocks_per_sequence) {
    return -1;
  }
  const int64_t physical_block = static_cast<int64_t>(
      block_table[sequence_index * max_blocks_per_sequence + logical_block]);
  const int64_t physical = physical_block * block_size + logical % block_size;
  return physical_block < 0 || physical > INT32_MAX
             ? -1
             : static_cast<int32_t>(physical);
}

template <typename compressed_index_t, typename swa_index_t>
__global__ void __launch_bounds__(kLogicalToPhysicalBlockSize, 1)
    dsv4_prepare_mate_dsa_indices_kernel(
        const int32_t* __restrict__ logical_indices,
        const compressed_index_t* __restrict__ compressed_block_table,
        const swa_index_t* __restrict__ swa_block_table,
        int32_t* __restrict__ physical_indices,
        int32_t* __restrict__ extra_physical_indices,
        int64_t num_main_indices,
        int64_t num_extra_indices,
        int64_t num_query_rows,
        int64_t sequence_index,
        int64_t compressed_max_blocks,
        int64_t swa_max_blocks,
        int64_t kv_len,
        int64_t window_size,
        int64_t block_size) {
  const int64_t total_indices = num_main_indices + num_extra_indices;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total_indices;
       index += blockDim.x * gridDim.x) {
    if (index < num_main_indices) {
      physical_indices[index] =
          dsv4_map_logical_to_physical<compressed_index_t>(
              static_cast<int64_t>(logical_indices[index]),
              compressed_block_table,
              sequence_index,
              compressed_max_blocks,
              block_size);
      continue;
    }
    const int64_t extra_index = index - num_main_indices;
    const int64_t query_row = extra_index / 128;
    const int64_t extra_column = extra_index % 128;
    // Compact speculative verification stores current+proposal rows in one
    // query range.  kv_len is the final visible length, while an earlier
    // query row must not read the SWA entries belonging to a later proposal.
    // Build each row's physical tail directly on device so the Mate MODEL1
    // batch ABI can be used without a host metadata copy.
    const int64_t row_kv_len =
        max(kv_len - (num_query_rows - 1 - query_row), int64_t{0});
    const int64_t row_extra_len =
        min(min(int64_t{128}, window_size), row_kv_len);
    const int64_t logical = extra_column < row_extra_len
                                ? row_kv_len - row_extra_len + extra_column
                                : -1;
    extra_physical_indices[extra_index] =
        dsv4_map_logical_to_physical<swa_index_t>(logical,
                                                  swa_block_table,
                                                  sequence_index,
                                                  swa_max_blocks,
                                                  block_size);
  }
}

template <typename scalar_t, typename index_t>
__global__ void __launch_bounds__(kGatherCacheBlockSize, 1)
    dsv4_gather_cache_kernel(const scalar_t* __restrict__ cache,
                             const index_t* __restrict__ block_table,
                             scalar_t* __restrict__ output,
                             int64_t sequence_index,
                             int64_t gather_len,
                             int64_t row_width,
                             int64_t block_size,
                             int64_t num_blocks,
                             int64_t max_blocks_per_seq) {
  const int64_t row = blockIdx.x;
  if (row >= gather_len) {
    return;
  }

  const int64_t table_idx = row / block_size;
  const int64_t block_offset = row % block_size;
  if (table_idx >= max_blocks_per_seq) {
    return;
  }

  int64_t block_id = static_cast<int64_t>(
      block_table[sequence_index * max_blocks_per_seq + table_idx]);
  int64_t flat = block_id * block_size + block_offset;
  if ((flat < 0 || flat >= num_blocks * block_size) && block_id > 0) {
    flat = (block_id - 1) * block_size + block_offset;
  }

  scalar_t* dst = output + row * row_width;
  if (flat < 0 || flat >= num_blocks * block_size) {
    for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
      dst[col] = scalar_t{};
    }
    return;
  }

  const scalar_t* src = cache + flat * row_width;
  for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    dst[col] = src[col];
  }
}

template <typename scalar_t>
void dispatch_dsv4_gather_cache(const torch::Tensor& cache_2d,
                                const torch::Tensor& block_table,
                                torch::Tensor& output,
                                int64_t sequence_index,
                                int64_t gather_len,
                                int64_t row_width,
                                int64_t block_size,
                                int64_t num_blocks,
                                int64_t max_blocks_per_seq) {
  const musaStream_t stream = current_dsv4_cache_stream();
  dim3 grid(gather_len);
  dim3 block(std::min<int64_t>(row_width, kGatherCacheBlockSize));
  if (block_table.scalar_type() == torch::kInt32) {
    dsv4_gather_cache_kernel<scalar_t, int32_t>
        <<<grid, block, 0, stream>>>(cache_2d.data_ptr<scalar_t>(),
                                     block_table.data_ptr<int32_t>(),
                                     output.data_ptr<scalar_t>(),
                                     sequence_index,
                                     gather_len,
                                     row_width,
                                     block_size,
                                     num_blocks,
                                     max_blocks_per_seq);
    check_musa_kernel_launch("dsv4 kernel");
    if (dsv4_musa_cache_gather_sync_enabled()) {
      check_musa_runtime(musaStreamSynchronize(stream));
    }
    return;
  }
  dsv4_gather_cache_kernel<scalar_t, int64_t>
      <<<grid, block, 0, stream>>>(cache_2d.data_ptr<scalar_t>(),
                                   block_table.data_ptr<int64_t>(),
                                   output.data_ptr<scalar_t>(),
                                   sequence_index,
                                   gather_len,
                                   row_width,
                                   block_size,
                                   num_blocks,
                                   max_blocks_per_seq);
  check_musa_kernel_launch("dsv4 kernel");
  if (dsv4_musa_cache_gather_sync_enabled()) {
    check_musa_runtime(musaStreamSynchronize(stream));
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kHcPreCombineBlockSize, 1)
    dsv4_hc_pre_combine_kernel(const scalar_t* __restrict__ x_hc,
                               const float* __restrict__ pre,
                               scalar_t* __restrict__ output,
                               int64_t num_tokens,
                               int64_t hc_mult,
                               int64_t hidden_size) {
  const int64_t token = blockIdx.x;
  const int64_t hidden = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= num_tokens || hidden >= hidden_size) {
    return;
  }

  float acc = 0.0f;
  const int64_t x_base = token * hc_mult * hidden_size + hidden;
  const int64_t pre_base = token * hc_mult;
  for (int64_t i = 0; i < hc_mult; ++i) {
    acc +=
        static_cast<float>(x_hc[x_base + i * hidden_size]) * pre[pre_base + i];
  }
  output[token * hidden_size + hidden] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void __launch_bounds__(kHcPreCombineBlockSize, 1)
    dsv4_hc_post_kernel(const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ residual,
                        const float* __restrict__ post,
                        const float* __restrict__ comb,
                        scalar_t* __restrict__ output,
                        int64_t hc_mult,
                        int64_t hidden_size) {
  const int64_t stream = blockIdx.x;
  const int64_t hidden = blockIdx.y * blockDim.x + threadIdx.x;
  if (hidden >= hidden_size) {
    return;
  }

  const int64_t token = stream / hc_mult;
  const int64_t output_stream = stream % hc_mult;
  const int64_t comb_base = (token * hc_mult + output_stream) * hc_mult;
  const int64_t residual_base = token * hc_mult * hidden_size + hidden;
  float value = 0.0f;
  for (int64_t input_stream = 0; input_stream < hc_mult; ++input_stream) {
    value += comb[comb_base + input_stream] *
             static_cast<float>(
                 residual[residual_base + input_stream * hidden_size]);
  }
  value += static_cast<float>(x[token * hidden_size + hidden]) *
           post[token * hc_mult + output_stream];
  output[stream * hidden_size + hidden] = static_cast<scalar_t>(value);
}

// SGLang's MUSA MHC post maps one CTA to one token/hidden tile and computes
// all four hyper-connection outputs together.  The former mapping launched a
// CTA for every output stream and reloaded x plus all four residual streams
// four times.  Sharing those values here cuts the 313x4x16384 prefill grid by
// 8x (80128 -> 10016 CTAs) and reduces the BF16 input traffic by about 4x.
template <typename scalar_t>
__global__ void __launch_bounds__(kHcPost2DBlockSize, 1)
    dsv4_hc_post_2d_h4_kernel(const scalar_t* __restrict__ x,
                              const scalar_t* __restrict__ residual,
                              const float* __restrict__ post,
                              const float* __restrict__ comb,
                              scalar_t* __restrict__ output,
                              int64_t hidden_size) {
  __shared__ float shared_post[4];
  __shared__ float shared_comb[16];

  const int64_t token = blockIdx.x;
  if (threadIdx.x < 4) {
    shared_post[threadIdx.x] = post[token * 4 + threadIdx.x];
  }
  if (threadIdx.x < 16) {
    shared_comb[threadIdx.x] = comb[token * 16 + threadIdx.x];
  }
  __syncthreads();

  const int64_t hidden_begin = blockIdx.y * kHcPost2DHiddenTile;
  const int64_t token_residual_base = token * 4 * hidden_size;
  const int64_t token_output_base = token * 4 * hidden_size;
  for (int64_t hidden_offset = threadIdx.x; hidden_offset < kHcPost2DHiddenTile;
       hidden_offset += blockDim.x) {
    const int64_t hidden = hidden_begin + hidden_offset;
    if (hidden >= hidden_size) {
      continue;
    }

    const float x_value = static_cast<float>(x[token * hidden_size + hidden]);
    float residual_values[4];
#pragma unroll
    for (int64_t input_stream = 0; input_stream < 4; ++input_stream) {
      residual_values[input_stream] = static_cast<float>(
          residual[token_residual_base + input_stream * hidden_size + hidden]);
    }

#pragma unroll
    for (int64_t output_stream = 0; output_stream < 4; ++output_stream) {
      float value = 0.0f;
#pragma unroll
      for (int64_t input_stream = 0; input_stream < 4; ++input_stream) {
        value += shared_comb[output_stream * 4 + input_stream] *
                 residual_values[input_stream];
      }
      value += x_value * shared_post[output_stream];
      output[token_output_base + output_stream * hidden_size + hidden] =
          static_cast<scalar_t>(value);
    }
  }
}

__global__ void __launch_bounds__(kHcMetadataBlockSize, 1)
    dsv4_hc_pre_metadata_kernel(const float* __restrict__ mixes,
                                const float* __restrict__ hc_scale,
                                const float* __restrict__ hc_base,
                                float* __restrict__ pre,
                                float* __restrict__ post,
                                float* __restrict__ comb,
                                int64_t hc_mult,
                                int64_t sinkhorn_iters,
                                float eps) {
  __shared__ float matrix[kHcMaxMult * kHcMaxMult];
  const int64_t token = blockIdx.x;
  const int64_t tid = threadIdx.x;
  const int64_t coeffs_per_token = (2 + hc_mult) * hc_mult;
  const float* token_mixes = mixes + token * coeffs_per_token;
  float* token_pre = pre + token * hc_mult;
  float* token_post = post + token * hc_mult;
  float* token_comb = comb + token * hc_mult * hc_mult;

  if (tid < hc_mult) {
    const float pre_value = token_mixes[tid] * hc_scale[0] + hc_base[tid];
    const float post_value =
        token_mixes[hc_mult + tid] * hc_scale[1] + hc_base[hc_mult + tid];
    token_pre[tid] = 1.0f / (1.0f + expf(-pre_value)) + eps;
    token_post[tid] = 2.0f / (1.0f + expf(-post_value));

    const int64_t row_offset = tid * hc_mult;
    const int64_t mix_offset = 2 * hc_mult + row_offset;
    float max_value = -INFINITY;
    for (int64_t col = 0; col < hc_mult; ++col) {
      max_value = fmaxf(max_value,
                        token_mixes[mix_offset + col] * hc_scale[2] +
                            hc_base[mix_offset + col]);
    }
    float row_sum = 0.0f;
    for (int64_t col = 0; col < hc_mult; ++col) {
      const float value = expf(token_mixes[mix_offset + col] * hc_scale[2] +
                               hc_base[mix_offset + col] - max_value);
      matrix[row_offset + col] = value;
      row_sum += value;
    }
    const float inv_sum = 1.0f / row_sum;
    for (int64_t col = 0; col < hc_mult; ++col) {
      matrix[row_offset + col] = matrix[row_offset + col] * inv_sum + eps;
    }
  }
  __syncthreads();

  if (tid < hc_mult) {
    float col_sum = 0.0f;
    for (int64_t row = 0; row < hc_mult; ++row) {
      col_sum += matrix[row * hc_mult + tid];
    }
    const float inv_sum = 1.0f / (col_sum + eps);
    for (int64_t row = 0; row < hc_mult; ++row) {
      matrix[row * hc_mult + tid] *= inv_sum;
    }
  }
  __syncthreads();

  for (int64_t iter = 1; iter < sinkhorn_iters; ++iter) {
    if (tid < hc_mult) {
      const int64_t row_offset = tid * hc_mult;
      float row_sum = 0.0f;
      for (int64_t col = 0; col < hc_mult; ++col) {
        row_sum += matrix[row_offset + col];
      }
      const float inv_sum = 1.0f / (row_sum + eps);
      for (int64_t col = 0; col < hc_mult; ++col) {
        matrix[row_offset + col] *= inv_sum;
      }
    }
    __syncthreads();
    if (tid < hc_mult) {
      float col_sum = 0.0f;
      for (int64_t row = 0; row < hc_mult; ++row) {
        col_sum += matrix[row * hc_mult + tid];
      }
      const float inv_sum = 1.0f / (col_sum + eps);
      for (int64_t row = 0; row < hc_mult; ++row) {
        matrix[row * hc_mult + tid] *= inv_sum;
      }
    }
    __syncthreads();
  }

  for (int64_t idx = tid; idx < hc_mult * hc_mult; idx += blockDim.x) {
    token_comb[idx] = matrix[idx];
  }
}

template <typename scalar_t>
void dispatch_dsv4_hc_pre_combine(const torch::Tensor& x_hc,
                                  const torch::Tensor& pre,
                                  torch::Tensor& output,
                                  int64_t num_tokens,
                                  int64_t hc_mult,
                                  int64_t hidden_size) {
  dim3 grid(
      num_tokens,
      (hidden_size + kHcPreCombineBlockSize - 1) / kHcPreCombineBlockSize);
  dim3 block(kHcPreCombineBlockSize);
  dsv4_hc_pre_combine_kernel<scalar_t>
      <<<grid, block, 0, current_dsv4_cache_stream()>>>(
          x_hc.data_ptr<scalar_t>(),
          pre.data_ptr<float>(),
          output.data_ptr<scalar_t>(),
          num_tokens,
          hc_mult,
          hidden_size);
}

}  // namespace

torch::Tensor deepseek_v4_gather_cache(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const std::optional<torch::Tensor>& output) {
  CHECK(cache.defined() && block_table.defined())
      << "deepseek_v4_gather_cache expects cache and block_table.";
  CHECK_GT(kv_len, 0) << "deepseek_v4_gather_cache expects positive kv_len.";
  CHECK_GE(cache.dim(), 3)
      << "DeepSeek-V4 gather cache expects [blocks, block_size, ..., dim].";
  CHECK_EQ(block_table.dim(), 2)
      << "DeepSeek-V4 gather cache expects block_table [batch, max_blocks].";
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64)
      << "DeepSeek-V4 gather cache expects int32/int64 block_table.";
  CHECK(cache.is_contiguous())
      << "DeepSeek-V4 gather cache expects contiguous cache.";
  CHECK(block_table.is_contiguous())
      << "DeepSeek-V4 gather cache expects contiguous block_table.";

  const int64_t block_size = cache.size(1);
  const int64_t num_blocks = cache.size(0);
  const int64_t max_blocks_per_seq = block_table.size(1);
  const int64_t gather_len =
      std::min<int64_t>(kv_len, max_blocks_per_seq * block_size);
  CHECK_GT(gather_len, 0);

  auto cache_2d = cache.view({num_blocks * block_size, -1});
  const int64_t row_width = cache_2d.size(1);
  torch::Tensor out = output.has_value() ? output.value()
                                         : torch::empty({gather_len, row_width},
                                                        cache.options());
  CHECK(out.defined());
  CHECK_EQ(out.dim(), 2);
  CHECK_EQ(out.size(0), gather_len);
  CHECK_EQ(out.size(1), row_width);
  CHECK_EQ(out.scalar_type(), cache.scalar_type());
  CHECK(out.is_contiguous())
      << "DeepSeek-V4 gather cache output must be contiguous.";

  DISPATCH_FLOATING_TYPES(cache.scalar_type(), "dsv4_gather_cache", [&] {
    dispatch_dsv4_gather_cache<scalar_t>(cache_2d,
                                         block_table,
                                         out,
                                         sequence_index,
                                         gather_len,
                                         row_width,
                                         block_size,
                                         num_blocks,
                                         max_blocks_per_seq);
  });
  return out;
}

void deepseek_v4_scatter_cache(torch::Tensor cache,
                               torch::Tensor slot_ids,
                               torch::Tensor values) {
  if (!cache.defined() || !slot_ids.defined() || !values.defined() ||
      cache.numel() == 0 || slot_ids.numel() == 0 || values.numel() == 0) {
    return;
  }
  CHECK_EQ(cache.scalar_type(), values.scalar_type())
      << "DeepSeek-V4 scatter cache expects cache/value dtype match.";
  CHECK_EQ(cache.device(), values.device())
      << "DeepSeek-V4 scatter cache expects cache/value device match.";
  CHECK(cache.is_contiguous())
      << "DeepSeek-V4 scatter cache expects contiguous cache.";
  CHECK_GE(values.dim(), 1)
      << "DeepSeek-V4 scatter cache expects value dim >= 1.";
  CHECK(slot_ids.scalar_type() == torch::kInt32 ||
        slot_ids.scalar_type() == torch::kInt64)
      << "DeepSeek-V4 scatter cache expects int32/int64 slot mapping.";

  auto value_2d =
      values.reshape({-1, values.size(values.dim() - 1)}).contiguous();
  const int64_t value_width = value_2d.size(1);
  const int64_t cache_width = cache.size(cache.dim() - 1);
  CHECK_GT(value_width, 0);
  CHECK_GT(cache_width, 0);
  CHECK_EQ(cache.numel() % cache_width, 0)
      << "DeepSeek-V4 scatter cache cannot flatten destination cache.";
  auto cache_2d = cache.view({-1, cache_width});
  const int64_t cache_rows = cache_2d.size(0);
  auto slots = slot_ids.reshape({-1});
  if (slots.device() != cache.device()) {
    slots = slots.to(cache.device());
  }
  slots = slots.contiguous();

  const int64_t update_rows =
      std::min<int64_t>(slots.size(0), value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }
  constexpr int64_t kFlashMlaRopeHeadDim = 64;
  if (cache_width != value_width) {
    CHECK_EQ(cache_width, value_width + kFlashMlaRopeHeadDim)
        << "DeepSeek-V4 scatter only supports equal row widths or the "
           "FlashMLA [V, Kpe] cache layout.";
    if (slots.scalar_type() == torch::kInt32) {
      dispatch_dsv4_scatter_flashmla_cache<int32_t>(value_2d,
                                                    cache_2d,
                                                    slots,
                                                    update_rows,
                                                    cache_rows,
                                                    value_width,
                                                    cache_width,
                                                    kFlashMlaRopeHeadDim);
    } else {
      dispatch_dsv4_scatter_flashmla_cache<int64_t>(value_2d,
                                                    cache_2d,
                                                    slots,
                                                    update_rows,
                                                    cache_rows,
                                                    value_width,
                                                    cache_width,
                                                    kFlashMlaRopeHeadDim);
    }
    return;
  }

  const int64_t row_bytes = value_width * value_2d.element_size();
  if (slots.scalar_type() == torch::kInt32) {
    dispatch_dsv4_scatter_cache<int32_t>(
        value_2d, cache_2d, slots, update_rows, cache_rows, row_bytes);
  } else {
    dispatch_dsv4_scatter_cache<int64_t>(
        value_2d, cache_2d, slots, update_rows, cache_rows, row_bytes);
  }
}

void deepseek_v4_pack_fp8_cache(torch::Tensor cache,
                                torch::Tensor slot_ids,
                                torch::Tensor values) {
#if defined(USE_DCU)
  CHECK(false) << "DeepSeek-V4 MODEL1 FP8 cache packing is not implemented "
                  "for DCU.";
#else
  if (!cache.defined() || !slot_ids.defined() || !values.defined() ||
      cache.numel() == 0 || slot_ids.numel() == 0 || values.numel() == 0) {
    return;
  }
  CHECK_EQ(cache.scalar_type(), torch::kUInt8)
      << "DeepSeek-V4 MODEL1 cache must be uint8.";
  CHECK_EQ(values.scalar_type(), torch::kBFloat16)
      << "DeepSeek-V4 MODEL1 pack expects BF16 latent rows.";
  CHECK_EQ(cache.device(), values.device());
  CHECK_EQ(slot_ids.device(), values.device());
  CHECK_EQ(cache.dim(), 4)
      << "DeepSeek-V4 MODEL1 cache must be [blocks, block, 1, 584].";
  CHECK_EQ(cache.size(2), 1);
  CHECK_EQ(cache.size(3), kModel1BytesPerToken);
  CHECK_EQ(cache.stride(3), 1)
      << "DeepSeek-V4 MODEL1 cache bytes must be contiguous.";
  CHECK_GE(cache.stride(0), cache.size(1) * kModel1BytesPerToken)
      << "DeepSeek-V4 MODEL1 page stride is too small.";
  CHECK(slot_ids.scalar_type() == torch::kInt32 ||
        slot_ids.scalar_type() == torch::kInt64)
      << "DeepSeek-V4 MODEL1 slot mapping must be int32/int64.";

  auto value_2d = values.reshape({-1, values.size(-1)}).contiguous();
  CHECK_EQ(value_2d.size(1), kModel1HeadDim)
      << "DeepSeek-V4 MODEL1 pack expects 512-dimensional latent rows.";
  auto slots = slot_ids.reshape({-1}).contiguous();
  const int64_t update_rows =
      std::min<int64_t>(slots.size(0), value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }
  const int64_t num_blocks = cache.size(0);
  const int64_t block_size = cache.size(1);
  const int64_t page_stride_bytes = cache.stride(0);
  if (slots.scalar_type() == torch::kInt32) {
    dispatch_dsv4_pack_model1_fp8_cache<int32_t>(value_2d,
                                                 cache,
                                                 slots,
                                                 update_rows,
                                                 num_blocks,
                                                 block_size,
                                                 page_stride_bytes);
  } else {
    dispatch_dsv4_pack_model1_fp8_cache<int64_t>(value_2d,
                                                 cache,
                                                 slots,
                                                 update_rows,
                                                 num_blocks,
                                                 block_size,
                                                 page_stride_bytes);
  }
#endif
}

void deepseek_v4_indexer_cache_quant_scatter(torch::Tensor cache,
                                             torch::Tensor scale_cache,
                                             torch::Tensor slot_ids,
                                             torch::Tensor values) {
  if (!cache.defined() || !scale_cache.defined() || !slot_ids.defined() ||
      !values.defined() || cache.numel() == 0 || scale_cache.numel() == 0 ||
      slot_ids.numel() == 0 || values.numel() == 0) {
    return;
  }
  CHECK_EQ(cache.scalar_type(), torch::kInt8);
  CHECK_EQ(scale_cache.scalar_type(), torch::kFloat16);
  CHECK_EQ(cache.device(), values.device());
  CHECK_EQ(scale_cache.device(), values.device());
  CHECK(cache.is_contiguous());
  CHECK(scale_cache.is_contiguous());
  CHECK(slot_ids.scalar_type() == torch::kInt32 ||
        slot_ids.scalar_type() == torch::kInt64);

  auto values_2d = values.reshape({-1, values.size(-1)}).contiguous();
  const int64_t row_width = values_2d.size(1);
  CHECK_EQ(cache.size(-1), row_width);
  auto cache_2d = cache.view({-1, row_width});
  auto scales_1d = scale_cache.reshape({-1});
  CHECK_EQ(scales_1d.size(0), cache_2d.size(0));
  auto slots = slot_ids.reshape({-1});
  if (slots.device() != cache.device()) {
    slots = slots.to(cache.device());
  }
  slots = slots.contiguous();
  const int64_t update_rows =
      std::min<int64_t>(values_2d.size(0), slots.size(0));
  if (update_rows <= 0) {
    return;
  }
  if (slots.scalar_type() == torch::kInt32) {
    dispatch_dsv4_indexer_cache_quant_scatter<int32_t>(values_2d,
                                                       cache_2d,
                                                       scales_1d,
                                                       slots,
                                                       update_rows,
                                                       cache_2d.size(0),
                                                       row_width);
  } else {
    dispatch_dsv4_indexer_cache_quant_scatter<int64_t>(values_2d,
                                                       cache_2d,
                                                       scales_1d,
                                                       slots,
                                                       update_rows,
                                                       cache_2d.size(0),
                                                       row_width);
  }
}

torch::Tensor deepseek_v4_indexer_cache_gather(
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output) {
  CHECK(cache.defined() && scale_cache.defined() && block_table.defined());
  CHECK_EQ(cache.scalar_type(), torch::kInt8);
  CHECK_EQ(scale_cache.scalar_type(), torch::kFloat16);
  CHECK_GE(cache.dim(), 3);
  CHECK_EQ(block_table.dim(), 2);
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK_GT(kv_len, 0);
  CHECK(cache.is_contiguous());
  CHECK(scale_cache.is_contiguous());
  CHECK(block_table.is_contiguous());
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64);

  const int64_t num_blocks = cache.size(0);
  const int64_t block_size = cache.size(1);
  const int64_t row_width = cache.size(-1);
  const int64_t gather_len =
      std::min<int64_t>(kv_len, block_table.size(1) * block_size);
  CHECK_GT(gather_len, 0);
  CHECK_EQ(scale_cache.numel(), num_blocks * block_size);
  torch::Tensor out = output.has_value()
                          ? output.value()
                          : torch::empty({gather_len, row_width},
                                         cache.options().dtype(output_dtype));
  CHECK_EQ(out.sizes(), torch::IntArrayRef({gather_len, row_width}));
  CHECK_EQ(out.scalar_type(), output_dtype);
  CHECK(out.is_contiguous());
  DISPATCH_FLOATING_TYPES(output_dtype, "dsv4_indexer_cache_gather", [&] {
    dispatch_dsv4_indexer_cache_gather<scalar_t>(cache,
                                                 scale_cache,
                                                 block_table,
                                                 out,
                                                 sequence_index,
                                                 gather_len,
                                                 row_width,
                                                 block_size,
                                                 num_blocks,
                                                 block_table.size(1));
  });
  return out;
}

torch::Tensor deepseek_v4_indexer_topk(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scale_cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t topk,
    int64_t query_absolute_start,
    int64_t compress_ratio,
    const std::optional<torch::Tensor>& output,
    const torch::Tensor& mate_compressed_block_table,
    const torch::Tensor& mate_swa_block_table,
    int64_t mate_kv_len,
    int64_t mate_window_size,
    int64_t mate_block_size,
    const std::optional<torch::Tensor>& mate_physical_indices,
    const std::optional<torch::Tensor>& mate_extra_physical_indices) {
  CHECK(query.defined() && weights.defined() && cache.defined() &&
        scale_cache.defined() && block_table.defined());
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(query.size(0), 1)
      << "Fused DeepSeek-V4 indexer currently supports decode q_tokens=1.";
  CHECK_EQ(weights.sizes(), torch::IntArrayRef({1, query.size(1)}));
  CHECK_EQ(weights.scalar_type(), torch::kFloat32);
  CHECK_EQ(cache.scalar_type(), torch::kInt8);
  CHECK_EQ(scale_cache.scalar_type(), torch::kFloat16);
  CHECK_EQ(cache.size(-1), query.size(2));
  CHECK_EQ(scale_cache.numel(), cache.size(0) * cache.size(1));
  CHECK_EQ(block_table.dim(), 2);
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64);
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK_GT(kv_len, 0);
  CHECK_GT(topk, 0);
  CHECK_GT(compress_ratio, 0);
  CHECK(query.is_contiguous() && weights.is_contiguous() &&
        cache.is_contiguous() && scale_cache.is_contiguous() &&
        block_table.is_contiguous());

  const bool mate_mapping_requested = mate_compressed_block_table.defined() ||
                                      mate_swa_block_table.defined() ||
                                      mate_physical_indices.has_value() ||
                                      mate_extra_physical_indices.has_value();
  torch::Tensor mate_main_output;
  torch::Tensor mate_extra_output;
  if (mate_mapping_requested) {
    CHECK(mate_compressed_block_table.defined() &&
          mate_swa_block_table.defined() && mate_physical_indices.has_value() &&
          mate_extra_physical_indices.has_value());
    CHECK(mate_compressed_block_table.scalar_type() == torch::kInt32 ||
          mate_compressed_block_table.scalar_type() == torch::kInt64);
    CHECK(mate_swa_block_table.scalar_type() == torch::kInt32 ||
          mate_swa_block_table.scalar_type() == torch::kInt64);
    CHECK(mate_compressed_block_table.is_contiguous());
    CHECK(mate_swa_block_table.is_contiguous());
    CHECK_EQ(mate_compressed_block_table.device(), query.device());
    CHECK_EQ(mate_swa_block_table.device(), query.device());
    CHECK_GE(mate_kv_len, 0);
    CHECK_GE(mate_window_size, 0);
    CHECK_GT(mate_block_size, 0);
    mate_main_output = mate_physical_indices.value();
    mate_extra_output = mate_extra_physical_indices.value();
    CHECK_EQ(mate_main_output.scalar_type(), torch::kInt32);
    CHECK_EQ(mate_extra_output.scalar_type(), torch::kInt32);
    CHECK_EQ(mate_main_output.sizes(), torch::IntArrayRef({1, topk}));
    CHECK_EQ(mate_extra_output.sizes(), torch::IntArrayRef({1, 128}));
    CHECK(mate_main_output.is_contiguous());
    CHECK(mate_extra_output.is_contiguous());
  }

  const int64_t valid_count = std::clamp<int64_t>(
      (query_absolute_start + 1) / compress_ratio, 0, kv_len);
  const int64_t selected_count = std::min<int64_t>(topk, valid_count);
  torch::Tensor indices =
      output.has_value()
          ? output.value()
          : torch::empty({1, topk}, cache.options().dtype(torch::kInt32));
  CHECK_EQ(indices.sizes(), torch::IntArrayRef({1, topk}));
  CHECK_EQ(indices.scalar_type(), torch::kInt32);
  CHECK(indices.is_contiguous());
  check_musa_runtime(musaMemsetAsync(indices.data_ptr<int32_t>(),
                                     0xff,
                                     indices.numel() * sizeof(int32_t),
                                     current_dsv4_cache_stream()));
  const auto prepare_mate_mapping = [&](bool mapping_already_fused) {
    if (!mate_mapping_requested || mapping_already_fused) {
      return;
    }
    deepseek_v4_prepare_mate_dsa_indices(indices,
                                         mate_compressed_block_table,
                                         mate_swa_block_table,
                                         sequence_index,
                                         mate_kv_len,
                                         mate_window_size,
                                         mate_block_size,
                                         mate_main_output,
                                         mate_extra_output);
  };
  if (selected_count == 0) {
    prepare_mate_mapping(/*mapping_already_fused=*/false);
    return indices;
  }

  if (dsv4_musa_indexer_radix_topk_enabled()) {
    // SGLang skips scoring when every causal key is selected. The sparse
    // attention result depends on the selected set, not score order.
    if (selected_count == valid_count) {
      dsv4_indexer_all_valid_kernel<<<1,
                                      kIndexerCacheBlockSize,
                                      0,
                                      current_dsv4_cache_stream()>>>(
          indices.data_ptr<int32_t>(), selected_count, topk);
      check_musa_kernel_launch("dsv4 kernel");
      prepare_mate_mapping(/*mapping_already_fused=*/false);
      return indices;
    }
    if (topk == kIndexerRadixTopk && selected_count == kIndexerRadixTopk &&
        valid_count > kIndexerRadixTopk) {
      auto& workspace = dsv4_indexer_single_topk_workspace(
          query, selected_count, valid_count);
      auto scores =
          workspace.scores.narrow(0, 0, valid_count).view({1, valid_count});
      if (dsv4_musa_indexer_fused_score_radix_enabled()) {
        check_musa_runtime(musaMemsetAsync(workspace.lock.data_ptr<int32_t>(),
                                           0,
                                           sizeof(int32_t),
                                           current_dsv4_cache_stream()));
        bool fused_launched = false;
        DISPATCH_FLOATING_TYPES(
            query.scalar_type(), "dsv4_indexer_paged_score_fused_radix", [&] {
              if (block_table.scalar_type() == torch::kInt32) {
                fused_launched =
                    dispatch_dsv4_indexer_paged_score_fused_radix_mate_types<
                        scalar_t,
                        int32_t>(query,
                                 weights,
                                 cache,
                                 scale_cache,
                                 block_table,
                                 scores,
                                 indices,
                                 workspace.lock,
                                 sequence_index,
                                 valid_count,
                                 mate_compressed_block_table,
                                 mate_swa_block_table,
                                 mate_main_output,
                                 mate_extra_output,
                                 mate_kv_len,
                                 mate_window_size,
                                 mate_block_size,
                                 sequence_index);
              } else {
                fused_launched =
                    dispatch_dsv4_indexer_paged_score_fused_radix_mate_types<
                        scalar_t,
                        int64_t>(query,
                                 weights,
                                 cache,
                                 scale_cache,
                                 block_table,
                                 scores,
                                 indices,
                                 workspace.lock,
                                 sequence_index,
                                 valid_count,
                                 mate_compressed_block_table,
                                 mate_swa_block_table,
                                 mate_main_output,
                                 mate_extra_output,
                                 mate_kv_len,
                                 mate_window_size,
                                 mate_block_size,
                                 sequence_index);
              }
            });
        if (fused_launched) {
          prepare_mate_mapping(
              /*mapping_already_fused=*/mate_mapping_requested);
          return indices;
        }
      }
      DISPATCH_FLOATING_TYPES(
          query.scalar_type(), "dsv4_indexer_paged_score_radix_topk", [&] {
            dispatch_dsv4_indexer_paged_score<scalar_t>(query,
                                                        weights,
                                                        cache,
                                                        scale_cache,
                                                        block_table,
                                                        scores,
                                                        sequence_index,
                                                        valid_count,
                                                        query_absolute_start,
                                                        compress_ratio);
          });
      dispatch_dsv4_indexer_radix_topk(scores, indices, valid_count);
      prepare_mate_mapping(/*mapping_already_fused=*/false);
      return indices;
    }
  }

  if (dsv4_musa_indexer_single_kernel_topk_enabled()) {
    auto& workspace = dsv4_indexer_single_topk_workspace(query, selected_count);
    check_musa_runtime(musaMemsetAsync(workspace.lock.data_ptr<int32_t>(),
                                       0,
                                       sizeof(int32_t),
                                       current_dsv4_cache_stream()));
    DISPATCH_FLOATING_TYPES(
        query.scalar_type(), "dsv4_indexer_paged_score_topk", [&] {
          dispatch_dsv4_indexer_paged_score_topk<scalar_t>(query,
                                                           weights,
                                                           cache,
                                                           scale_cache,
                                                           block_table,
                                                           workspace.values,
                                                           indices,
                                                           workspace.lock,
                                                           sequence_index,
                                                           valid_count,
                                                           selected_count);
        });
    prepare_mate_mapping(/*mapping_already_fused=*/false);
    return indices;
  }

  // Production MODEL1 uses the fused score/radix path above. Keep a generic
  // Torch fallback for unsupported top-k shapes instead of depending on a
  // CUDA-private AIR TopK header from the MUSA backend.
  auto& cached_workspace =
      dsv4_indexer_single_topk_workspace(query, selected_count, kv_len);
  auto scores = cached_workspace.scores.narrow(0, 0, kv_len).view({1, kv_len});
  DISPATCH_FLOATING_TYPES(query.scalar_type(), "dsv4_indexer_paged_score", [&] {
    dispatch_dsv4_indexer_paged_score<scalar_t>(query,
                                                weights,
                                                cache,
                                                scale_cache,
                                                block_table,
                                                scores,
                                                sequence_index,
                                                kv_len,
                                                query_absolute_start,
                                                compress_ratio);
  });

  auto topk_result = torch::topk(scores,
                                 selected_count,
                                 /*dim=*/-1,
                                 /*largest=*/true,
                                 /*sorted=*/true);
  indices.narrow(/*dim=*/1, /*start=*/0, selected_count)
      .copy_(std::get<1>(topk_result).to(torch::kInt32));
  prepare_mate_mapping(/*mapping_already_fused=*/false);
  return indices;
}

torch::Tensor deepseek_v4_logical_to_physical_topk(
    const torch::Tensor& logical_indices,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t block_size,
    const std::optional<torch::Tensor>& output) {
  CHECK(logical_indices.defined() && block_table.defined());
  CHECK_EQ(logical_indices.scalar_type(), torch::kInt32)
      << "DeepSeek-V4 logical TopK indices must be int32.";
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64)
      << "DeepSeek-V4 block table must be int32/int64.";
  CHECK_EQ(block_table.dim(), 2);
  CHECK_GT(block_size, 0);
  CHECK_GT(logical_indices.size(-1), 0);
  CHECK_EQ(logical_indices.numel() % logical_indices.size(-1), 0);
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK_EQ(logical_indices.device(), block_table.device());
  CHECK(logical_indices.is_contiguous());
  CHECK(block_table.is_contiguous());

  torch::Tensor physical =
      output.has_value() ? output.value() : torch::empty_like(logical_indices);
  CHECK_EQ(physical.scalar_type(), torch::kInt32);
  CHECK_EQ(physical.device(), logical_indices.device());
  CHECK_EQ(physical.sizes(), logical_indices.sizes());
  CHECK(physical.is_contiguous());
  if (logical_indices.numel() == 0) {
    return physical;
  }

  const int64_t num_indices = logical_indices.numel();
  const int64_t grid_size =
      std::min<int64_t>((num_indices + kLogicalToPhysicalBlockSize - 1) /
                            kLogicalToPhysicalBlockSize,
                        65535);
  const dim3 grid(grid_size);
  const dim3 block(kLogicalToPhysicalBlockSize);
  const musaStream_t stream = current_dsv4_cache_stream();
  if (block_table.scalar_type() == torch::kInt32) {
    dsv4_logical_to_physical_topk_kernel<int32_t>
        <<<grid, block, 0, stream>>>(logical_indices.data_ptr<int32_t>(),
                                     block_table.data_ptr<int32_t>(),
                                     physical.data_ptr<int32_t>(),
                                     num_indices,
                                     sequence_index,
                                     block_table.size(1),
                                     block_size);
  } else {
    dsv4_logical_to_physical_topk_kernel<int64_t>
        <<<grid, block, 0, stream>>>(logical_indices.data_ptr<int32_t>(),
                                     block_table.data_ptr<int64_t>(),
                                     physical.data_ptr<int32_t>(),
                                     num_indices,
                                     sequence_index,
                                     block_table.size(1),
                                     block_size);
  }
  check_musa_kernel_launch("dsv4 kernel");
  return physical;
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_prepare_mate_dsa_indices(
    const torch::Tensor& logical_indices,
    const torch::Tensor& compressed_block_table,
    const torch::Tensor& swa_block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t window_size,
    int64_t block_size,
    const std::optional<torch::Tensor>& physical_indices,
    const std::optional<torch::Tensor>& extra_physical_indices) {
  constexpr int64_t kExtraTopk = 128;
  CHECK(logical_indices.defined() && compressed_block_table.defined() &&
        swa_block_table.defined());
  CHECK_EQ(logical_indices.scalar_type(), torch::kInt32);
  CHECK(compressed_block_table.scalar_type() == torch::kInt32 ||
        compressed_block_table.scalar_type() == torch::kInt64);
  CHECK(swa_block_table.scalar_type() == torch::kInt32 ||
        swa_block_table.scalar_type() == torch::kInt64);
  CHECK_GT(logical_indices.dim(), 0);
  CHECK_EQ(compressed_block_table.dim(), 2);
  CHECK_EQ(swa_block_table.dim(), 2);
  CHECK_EQ(logical_indices.device(), compressed_block_table.device());
  CHECK_EQ(logical_indices.device(), swa_block_table.device());
  CHECK(logical_indices.is_contiguous());
  CHECK(compressed_block_table.is_contiguous());
  CHECK(swa_block_table.is_contiguous());
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, compressed_block_table.size(0));
  CHECK_LT(sequence_index, swa_block_table.size(0));
  CHECK_GE(kv_len, 0);
  CHECK_GE(window_size, 0);
  CHECK_GT(block_size, 0);

  auto main_output = physical_indices.has_value()
                         ? physical_indices.value()
                         : torch::empty_like(logical_indices);
  auto extra_sizes = logical_indices.sizes().vec();
  extra_sizes.back() = kExtraTopk;
  auto extra_output =
      extra_physical_indices.has_value()
          ? extra_physical_indices.value()
          : torch::empty(extra_sizes, logical_indices.options());
  CHECK_EQ(main_output.sizes(), logical_indices.sizes());
  CHECK_EQ(extra_output.sizes(), torch::IntArrayRef(extra_sizes));
  CHECK_EQ(main_output.scalar_type(), torch::kInt32);
  CHECK_EQ(extra_output.scalar_type(), torch::kInt32);
  CHECK_EQ(main_output.device(), logical_indices.device());
  CHECK_EQ(extra_output.device(), logical_indices.device());
  CHECK(main_output.is_contiguous());
  CHECK(extra_output.is_contiguous());

  const int64_t total_indices = main_output.numel() + extra_output.numel();
  const dim3 block(kLogicalToPhysicalBlockSize);
  const dim3 grid(
      std::min<int64_t>((total_indices + kLogicalToPhysicalBlockSize - 1) /
                            kLogicalToPhysicalBlockSize,
                        65535));
  const int64_t num_query_rows =
      logical_indices.numel() / logical_indices.size(-1);
  const musaStream_t stream = current_dsv4_cache_stream();
#define LAUNCH_DSV4_PREPARE_MATE_DSA(COMPRESSED_T, SWA_T)   \
  dsv4_prepare_mate_dsa_indices_kernel<COMPRESSED_T, SWA_T> \
      <<<grid, block, 0, stream>>>(                         \
          logical_indices.data_ptr<int32_t>(),              \
          compressed_block_table.data_ptr<COMPRESSED_T>(),  \
          swa_block_table.data_ptr<SWA_T>(),                \
          main_output.data_ptr<int32_t>(),                  \
          extra_output.data_ptr<int32_t>(),                 \
          main_output.numel(),                              \
          extra_output.numel(),                             \
          num_query_rows,                                   \
          sequence_index,                                   \
          compressed_block_table.size(1),                   \
          swa_block_table.size(1),                          \
          kv_len,                                           \
          window_size,                                      \
          block_size)
  if (compressed_block_table.scalar_type() == torch::kInt32) {
    if (swa_block_table.scalar_type() == torch::kInt32) {
      LAUNCH_DSV4_PREPARE_MATE_DSA(int32_t, int32_t);
    } else {
      LAUNCH_DSV4_PREPARE_MATE_DSA(int32_t, int64_t);
    }
  } else if (swa_block_table.scalar_type() == torch::kInt32) {
    LAUNCH_DSV4_PREPARE_MATE_DSA(int64_t, int32_t);
  } else {
    LAUNCH_DSV4_PREPARE_MATE_DSA(int64_t, int64_t);
  }
#undef LAUNCH_DSV4_PREPARE_MATE_DSA
  check_musa_kernel_launch("dsv4 kernel");
  return {main_output, extra_output};
}

torch::Tensor deepseek_v4_compress_pool(
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& ape,
    int64_t compress_ratio,
    int64_t head_dim,
    int64_t compressor_coeff,
    const std::optional<torch::Tensor>& output) {
  CHECK(kv.defined() && score.defined() && ape.defined());
  CHECK_EQ(kv.dim(), 2);
  CHECK_EQ(score.dim(), 2);
  CHECK_EQ(ape.dim(), 2);
  CHECK(kv.is_contiguous() && score.is_contiguous() && ape.is_contiguous());
  CHECK_EQ(kv.device(), score.device());
  CHECK_EQ(kv.device(), ape.device());
  CHECK_EQ(score.scalar_type(), torch::kFloat32);
  CHECK_EQ(ape.scalar_type(), torch::kFloat32);
  CHECK(compress_ratio == 4 || compress_ratio == 128);
  CHECK(compressor_coeff == 1 || compressor_coeff == 2);
  CHECK(compressor_coeff == 1 || compress_ratio == 4)
      << "Two-source compressor pooling is only defined for C4.";
  CHECK_GT(head_dim, 0);
  const int64_t width = compressor_coeff * head_dim;
  CHECK_EQ(kv.size(1), width);
  CHECK_EQ(score.sizes(), kv.sizes());
  CHECK_EQ(ape.size(0), compress_ratio);
  CHECK_EQ(ape.size(1), width);

  const int64_t compressed_rows = kv.size(0) / compress_ratio;
  auto result = output.has_value()
                    ? output.value()
                    : torch::empty({compressed_rows, head_dim}, kv.options());
  CHECK_EQ(result.scalar_type(), kv.scalar_type());
  CHECK_EQ(result.device(), kv.device());
  CHECK_EQ(result.dim(), 2);
  CHECK_EQ(result.size(0), compressed_rows);
  CHECK_EQ(result.size(1), head_dim);
  CHECK(result.is_contiguous());
  if (compressed_rows == 0) {
    return result;
  }

  const musaStream_t stream = current_dsv4_cache_stream();
  DISPATCH_FLOATING_TYPES(kv.scalar_type(), "dsv4_compress_pool", [&] {
    if (compress_ratio == 128 && compressor_coeff == 1) {
      const dim3 grid(compressed_rows,
                      (head_dim + kCompressPoolC128DimsPerBlock - 1) /
                          kCompressPoolC128DimsPerBlock);
      const dim3 block(kCompressPoolC128BlockSize);
      dsv4_compress_pool_c128_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(kv.data_ptr<scalar_t>(),
                                       score.data_ptr<float>(),
                                       ape.data_ptr<float>(),
                                       result.data_ptr<scalar_t>(),
                                       head_dim);
    } else {
      const dim3 grid(compressed_rows);
      const dim3 block(kCompressPoolBlockSize);
      dsv4_compress_pool_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(kv.data_ptr<scalar_t>(),
                                       score.data_ptr<float>(),
                                       ape.data_ptr<float>(),
                                       result.data_ptr<scalar_t>(),
                                       compress_ratio,
                                       head_dim,
                                       compressor_coeff);
    }
  });
  check_musa_kernel_launch("dsv4 kernel");
  return result;
}

torch::Tensor deepseek_v4_compress_decode(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_step,
    const torch::Tensor& score_step,
    const torch::Tensor& ape,
    int64_t compress_ratio,
    int64_t head_dim,
    int64_t compressor_coeff,
    int64_t absolute_position,
    bool compute_boundary_output,
    const std::optional<torch::Tensor>& output) {
  CHECK(kv_state.defined() && score_state.defined() && kv_step.defined() &&
        score_step.defined() && ape.defined());
  CHECK(compress_ratio == 4 || compress_ratio == 128);
  CHECK((compress_ratio == 4 && compressor_coeff == 2) ||
        (compress_ratio == 128 && compressor_coeff == 1));
  CHECK_GT(head_dim, 0);
  CHECK_GE(absolute_position, 0);
  const int64_t state_rows = compress_ratio * compressor_coeff;
  const int64_t width = compressor_coeff * head_dim;
  CHECK_EQ(kv_state.sizes(), torch::IntArrayRef({state_rows, width}));
  CHECK_EQ(score_state.sizes(), kv_state.sizes());
  CHECK_EQ(kv_step.sizes(), torch::IntArrayRef({1, width}));
  CHECK_EQ(score_step.sizes(), kv_step.sizes());
  CHECK_EQ(ape.sizes(), torch::IntArrayRef({compress_ratio, width}));
  CHECK(kv_state.is_contiguous() && score_state.is_contiguous() &&
        kv_step.is_contiguous() && score_step.is_contiguous() &&
        ape.is_contiguous());
  CHECK_EQ(kv_state.scalar_type(), torch::kFloat32);
  CHECK_EQ(score_state.scalar_type(), torch::kFloat32);
  CHECK_EQ(kv_step.scalar_type(), torch::kFloat32);
  CHECK_EQ(score_step.scalar_type(), torch::kFloat32);
  CHECK_EQ(ape.scalar_type(), torch::kFloat32);
  CHECK_EQ(kv_state.device(), score_state.device());
  CHECK_EQ(kv_state.device(), kv_step.device());
  CHECK_EQ(kv_state.device(), score_step.device());
  CHECK_EQ(kv_state.device(), ape.device());

  const bool at_boundary = (absolute_position + 1) % compress_ratio == 0;
  torch::Tensor result;
  float* output_ptr = nullptr;
  if (at_boundary && compute_boundary_output) {
    result = output.has_value()
                 ? output.value()
                 : torch::empty({1, head_dim}, kv_state.options());
    CHECK_EQ(result.sizes(), torch::IntArrayRef({1, head_dim}));
    CHECK_EQ(result.scalar_type(), torch::kFloat32);
    CHECK_EQ(result.device(), kv_state.device());
    CHECK(result.is_contiguous());
    output_ptr = result.data_ptr<float>();
  }

  const int64_t ratio_position = absolute_position % compress_ratio;
  const dim3 block(compress_ratio == 4 ? kCompressDecodeC4BlockSize
                                       : kCompressDecodeC128BlockSize);
  const dim3 grid((head_dim + block.x - 1) / block.x);
  const musaStream_t stream = current_dsv4_cache_stream();
  if (compress_ratio == 4) {
    dsv4_compress_decode_c4_kernel<<<grid, block, 0, stream>>>(
        kv_state.data_ptr<float>(),
        score_state.data_ptr<float>(),
        kv_step.data_ptr<float>(),
        score_step.data_ptr<float>(),
        ape.data_ptr<float>(),
        output_ptr,
        head_dim,
        ratio_position,
        at_boundary,
        compute_boundary_output);
  } else {
    dsv4_compress_decode_c128_kernel<<<grid, block, 0, stream>>>(
        kv_state.data_ptr<float>(),
        score_state.data_ptr<float>(),
        kv_step.data_ptr<float>(),
        score_step.data_ptr<float>(),
        ape.data_ptr<float>(),
        output_ptr,
        head_dim,
        ratio_position,
        at_boundary,
        compute_boundary_output);
  }
  check_musa_kernel_launch("dsv4 kernel");
  return result;
}

torch::Tensor deepseek_v4_hc_pre_combine(
    const torch::Tensor& x_hc,
    const torch::Tensor& pre,
    const std::optional<torch::Tensor>& output) {
  CHECK(x_hc.defined() && pre.defined())
      << "deepseek_v4_hc_pre_combine expects x_hc and pre.";
  CHECK_EQ(x_hc.dim(), 3)
      << "DeepSeek-V4 hc_pre_combine expects x_hc [tokens, hc_mult, hidden].";
  CHECK_EQ(pre.dim(), 2)
      << "DeepSeek-V4 hc_pre_combine expects pre [tokens, hc_mult].";
  CHECK(x_hc.is_contiguous())
      << "DeepSeek-V4 hc_pre_combine expects contiguous x_hc.";
  CHECK(pre.is_contiguous())
      << "DeepSeek-V4 hc_pre_combine expects contiguous pre.";
  CHECK_EQ(pre.scalar_type(), torch::kFloat32)
      << "DeepSeek-V4 hc_pre_combine expects fp32 pre.";
  CHECK_EQ(pre.device(), x_hc.device());

  const int64_t num_tokens = x_hc.size(0);
  const int64_t hc_mult = x_hc.size(1);
  const int64_t hidden_size = x_hc.size(2);
  CHECK_GT(num_tokens, 0);
  CHECK_GT(hc_mult, 0);
  CHECK_GT(hidden_size, 0);
  CHECK_EQ(pre.size(0), num_tokens);
  CHECK_EQ(pre.size(1), hc_mult);

  torch::Tensor out =
      output.has_value()
          ? output.value()
          : torch::empty({num_tokens, hidden_size}, x_hc.options());
  CHECK(out.defined());
  CHECK_EQ(out.dim(), 2);
  CHECK_EQ(out.size(0), num_tokens);
  CHECK_EQ(out.size(1), hidden_size);
  CHECK_EQ(out.scalar_type(), x_hc.scalar_type());
  CHECK_EQ(out.device(), x_hc.device());
  CHECK(out.is_contiguous())
      << "DeepSeek-V4 hc_pre_combine output must be contiguous.";

  DISPATCH_FLOATING_TYPES(x_hc.scalar_type(), "dsv4_hc_pre_combine", [&] {
    dispatch_dsv4_hc_pre_combine<scalar_t>(
        x_hc, pre, out, num_tokens, hc_mult, hidden_size);
  });
  return out;
}

torch::Tensor deepseek_v4_hc_post(const torch::Tensor& x,
                                  const torch::Tensor& residual,
                                  const torch::Tensor& post,
                                  const torch::Tensor& comb,
                                  const std::optional<torch::Tensor>& output) {
  CHECK(x.defined() && residual.defined() && post.defined() && comb.defined())
      << "deepseek_v4_hc_post expects all inputs.";
  CHECK_GE(residual.dim(), 3);
  CHECK_EQ(x.scalar_type(), residual.scalar_type());
  CHECK_EQ(post.scalar_type(), torch::kFloat32);
  CHECK_EQ(comb.scalar_type(), torch::kFloat32);
  CHECK(x.is_contiguous() && residual.is_contiguous() && post.is_contiguous() &&
        comb.is_contiguous());
  CHECK_EQ(x.device(), residual.device());
  CHECK_EQ(x.device(), post.device());
  CHECK_EQ(x.device(), comb.device());

  const int64_t hc_mult = residual.size(-2);
  const int64_t hidden_size = residual.size(-1);
  const int64_t num_tokens = residual.numel() / (hc_mult * hidden_size);
  CHECK_GT(hc_mult, 0);
  CHECK_LE(hc_mult, 16);
  CHECK_EQ(x.numel(), num_tokens * hidden_size);
  CHECK_EQ(post.numel(), num_tokens * hc_mult);
  CHECK_EQ(comb.numel(), num_tokens * hc_mult * hc_mult);

  torch::Tensor out =
      output.has_value() ? output.value() : torch::empty_like(residual);
  CHECK_EQ(out.sizes(), residual.sizes());
  CHECK_EQ(out.scalar_type(), residual.scalar_type());
  CHECK_EQ(out.device(), residual.device());
  CHECK(out.is_contiguous());
  if (num_tokens == 0) {
    return out;
  }

  const dim3 grid(
      num_tokens * hc_mult,
      (hidden_size + kHcPreCombineBlockSize - 1) / kHcPreCombineBlockSize);
  const dim3 block(kHcPreCombineBlockSize);
  DISPATCH_FLOATING_TYPES(residual.scalar_type(), "dsv4_hc_post", [&] {
    if (hc_mult == 4 && true) {
      const dim3 grid_2d(
          num_tokens,
          (hidden_size + kHcPost2DHiddenTile - 1) / kHcPost2DHiddenTile);
      dsv4_hc_post_2d_h4_kernel<scalar_t>
          <<<grid_2d, kHcPost2DBlockSize, 0, current_dsv4_cache_stream()>>>(
              x.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              post.data_ptr<float>(),
              comb.data_ptr<float>(),
              out.data_ptr<scalar_t>(),
              hidden_size);
    } else {
      dsv4_hc_post_kernel<scalar_t>
          <<<grid, block, 0, current_dsv4_cache_stream()>>>(
              x.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              post.data_ptr<float>(),
              comb.data_ptr<float>(),
              out.data_ptr<scalar_t>(),
              hc_mult,
              hidden_size);
    }
  });
  check_musa_kernel_launch("dsv4 kernel");
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v4_hc_pre_metadata(const torch::Tensor& mixes,
                            const torch::Tensor& hc_scale,
                            const torch::Tensor& hc_base,
                            int64_t hc_mult,
                            int64_t sinkhorn_iters,
                            double eps) {
  CHECK(mixes.defined() && hc_scale.defined() && hc_base.defined());
  CHECK_EQ(mixes.dim(), 2);
  CHECK_EQ(mixes.scalar_type(), torch::kFloat32);
  CHECK_EQ(hc_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(hc_base.scalar_type(), torch::kFloat32);
  CHECK(mixes.is_contiguous() && hc_scale.is_contiguous() &&
        hc_base.is_contiguous());
  CHECK_EQ(mixes.device(), hc_scale.device());
  CHECK_EQ(mixes.device(), hc_base.device());
  CHECK_GT(hc_mult, 0);
  CHECK_LE(hc_mult, kHcMaxMult);
  CHECK_GE(sinkhorn_iters, 1);
  CHECK_GE(eps, 0.0);
  CHECK_GE(hc_scale.numel(), 3);
  CHECK_EQ(mixes.size(1), (2 + hc_mult) * hc_mult);
  CHECK_EQ(hc_base.numel(), mixes.size(1));

  const int64_t num_tokens = mixes.size(0);
  auto pre = torch::empty({num_tokens, hc_mult}, mixes.options());
  auto post = torch::empty({num_tokens, hc_mult}, mixes.options());
  auto comb = torch::empty({num_tokens, hc_mult, hc_mult}, mixes.options());
  if (num_tokens == 0) {
    return std::make_tuple(pre, post, comb);
  }
  dsv4_hc_pre_metadata_kernel<<<dim3(num_tokens),
                                dim3(kHcMetadataBlockSize),
                                0,
                                current_dsv4_cache_stream()>>>(
      mixes.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_base.data_ptr<float>(),
      pre.data_ptr<float>(),
      post.data_ptr<float>(),
      comb.data_ptr<float>(),
      hc_mult,
      sinkhorn_iters,
      static_cast<float>(eps));
  check_musa_kernel_launch("dsv4 kernel");
  return std::make_tuple(pre, post, comb);
}

}  // namespace xllm::kernel::musa
