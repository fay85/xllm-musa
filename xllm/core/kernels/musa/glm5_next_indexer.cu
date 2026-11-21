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

#include <ATen/record_function.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <cub/block/block_radix_sort.cuh>
#include <limits>
#include <vector>

#include "core/kernels/musa/glm5_next_indexer.h"
#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {
constexpr int64_t kPoolPacketBytes = 132;
constexpr uint32_t kRadix = 256;
constexpr uint32_t kHistGroups = 4;
constexpr uint32_t kSglBlockSize = 512;
constexpr uint32_t kHistGroupThreads = kSglBlockSize / kHistGroups;
constexpr int64_t kMqaLogitsSkipChunkElements = 8'000'000;
constexpr int64_t kMqaLogitsFreeMemNumerator = 1;
constexpr int64_t kMqaLogitsFreeMemDenominator = 5;
constexpr int64_t kMqaLogitsTotalMemNumerator = 3;
constexpr int64_t kMqaLogitsTotalMemDenominator = 10;

// Every thread owns one channel. Each butterfly reads the previous stage
// before any lane overwrites shared memory.
__device__ float hadamard(float value, float* shared) {
  const int32_t lane = threadIdx.x;
  shared[lane] = value;
  __syncthreads();
  for (int32_t stride = 1; stride < 128; stride *= 2) {
    const float other = shared[lane ^ stride];
    const float next = (lane & stride) ? other - value : value + other;
    __syncthreads();
    shared[lane] = next;
    value = next;
    __syncthreads();
  }
  return static_cast<float>(c10::BFloat16(value * 0.08838834764831845f));
}

__device__ float quant_scale(float value, float* shared) {
  const int32_t lane = threadIdx.x;
  shared[lane] = fabsf(value);
  __syncthreads();
  for (int32_t offset = 64; offset > 0; offset /= 2) {
    if (lane < offset) {
      shared[lane] = fmaxf(shared[lane], shared[lane + offset]);
    }
    __syncthreads();
  }
  return exp2f(ceilf(log2f(fmaxf(shared[0], 1.0e-4f) / 448.0f)));
}

__global__ void update_pools(const c10::BFloat16* keys,
                             const c10::BFloat16* gates,
                             const float* ape,
                             const int64_t* positions,
                             const int32_t* cu,
                             const int32_t* state_ids,
                             const int32_t* read_ids,
                             const int32_t* table,
                             uint8_t* cache,
                             const c10::BFloat16* read_tail,
                             c10::BFloat16* tail,
                             int32_t* rows,
                             int32_t* pool_lengths,
                             int64_t states,
                             int64_t pages,
                             int64_t table_width) {
  const int32_t request = blockIdx.x;
  const int32_t lane = threadIdx.x;
  const int64_t state = state_ids[request];
  const int64_t source = read_ids[request];
  const bool valid = state > 0 && state < states;
  __shared__ float shared[128];
  if (valid && source > 0 && source < states && source != state) {
    for (int32_t field = 0; field < 8; ++field) {
      tail[state * 1024 + field * 128 + lane] =
          read_tail[source * 1024 + field * 128 + lane];
    }
  }
  __syncthreads();
  for (int32_t row = cu[request]; row < cu[request + 1]; ++row) {
    const int64_t position = positions[row];
    if (lane == 0) {
      rows[row] = valid && position >= 0 ? request : -1;
      pool_lengths[row] = valid && position >= 0 ? (position + 1) / 4 : 0;
    }
    if (!valid || position < 0) {
      continue;
    }
    if (position == 0) {
      for (int32_t field = 0; field < 8; ++field) {
        tail[state * 1024 + field * 128 + lane] = c10::BFloat16(0.0f);
      }
    }
    const int64_t ring = position % 4;
    tail[state * 1024 + ring * 128 + lane] = keys[row * 128 + lane];
    tail[state * 1024 + 512 + ring * 128 + lane] = gates[row * 128 + lane];
    __syncthreads();
    if (ring != 3) {
      continue;
    }
    const int64_t pool = position / 4;
    const int64_t column = pool / 64;
    const int64_t page =
        column < table_width ? table[request * table_width + column] : -1;
    if (page < 0 || page >= pages) {
      continue;
    }
    float logits[4];
    float maximum = -INFINITY;
    for (int32_t j = 0; j < 4; ++j) {
      logits[j] =
          static_cast<float>(tail[state * 1024 + 512 + j * 128 + lane]) +
          ape[j * 128 + lane];
      maximum = fmaxf(maximum, logits[j]);
    }
    float numerator = 0;
    float denominator = 0;
    for (int32_t j = 0; j < 4; ++j) {
      const float probability = expf(logits[j] - maximum);
      numerator +=
          probability * static_cast<float>(tail[state * 1024 + j * 128 + lane]);
      denominator += probability;
    }
    float value =
        static_cast<float>(c10::BFloat16(__fdiv_rn(numerator, denominator)));
    value = hadamard(value, shared);
    const float scale = quant_scale(value, shared);
    // MATE stores payload and scales in separate planes within each page.
    uint8_t* page_data = cache + page * 64 * 132;
    reinterpret_cast<c10::Float8_e4m3fn*>(page_data)[(pool % 64) * 128 + lane] =
        QuantConverter<c10::Float8_e4m3fn>::convert(__fdiv_rn(value, scale));
    if (lane == 0) {
      reinterpret_cast<float*>(page_data + 64 * 128)[pool % 64] = scale;
    }
    __syncthreads();
  }
}

__global__ void update_pools_parallel(const c10::BFloat16* keys,
                                      const c10::BFloat16* gates,
                                      const float* ape,
                                      const int64_t* positions,
                                      const int32_t* cu,
                                      const int32_t* state_ids,
                                      const int32_t* read_ids,
                                      const int32_t* table,
                                      uint8_t* cache,
                                      const c10::BFloat16* read_tail,
                                      int32_t* rows,
                                      int32_t* pool_lengths,
                                      int64_t states,
                                      int64_t pages,
                                      int64_t table_width,
                                      int64_t first_row = 0,
                                      int64_t key_begin = 0) {
  const int32_t row = blockIdx.x + first_row;
  const int32_t lane = threadIdx.x;
  int32_t request = 0;
  while (cu[request + 1] <= row) {
    ++request;
  }
  const int64_t state = state_ids[request];
  const int64_t source = read_ids[request];
  const int64_t position = positions[row];
  const bool valid = state > 0 && state < states && position >= 0;
  if (lane == 0) {
    rows[blockIdx.x] = valid ? request : -1;
    pool_lengths[blockIdx.x] = valid ? (position + 1) / 4 : 0;
  }
  if (!valid || position % 4 != 3) {
    return;
  }
  __shared__ float shared[128];
  float pool_keys[4];
  float pool_gates[4];
  for (int32_t j = 0; j < 4; ++j) {
    const int64_t read_state = source > 0 && source < states ? source : state;
    pool_keys[j] =
        static_cast<float>(read_tail[read_state * 1024 + j * 128 + lane]);
    pool_gates[j] =
        static_cast<float>(read_tail[read_state * 1024 + 512 + j * 128 + lane]);
    // Find the most recent write to this ring entry. Normally this reads at
    // most four positions; padding is skipped just as in the decode kernel.
    for (int32_t input_row = row;
         input_row >= cu[request] && input_row >= key_begin;
         --input_row) {
      const int64_t input_position = positions[input_row];
      if (input_position >= 0 && input_position % 4 == j) {
        pool_keys[j] =
            static_cast<float>(keys[(input_row - key_begin) * 128 + lane]);
        pool_gates[j] =
            static_cast<float>(gates[(input_row - key_begin) * 128 + lane]);
        break;
      }
      if (input_position == 0) {
        pool_keys[j] = 0;
        pool_gates[j] = 0;
        break;
      }
    }
  }
  const int64_t pool = position / 4;
  const int64_t column = pool / 64;
  const int64_t page =
      column < table_width ? table[request * table_width + column] : -1;
  if (page < 0 || page >= pages) {
    return;
  }
  float logits[4];
  float maximum = -INFINITY;
  for (int32_t j = 0; j < 4; ++j) {
    logits[j] = pool_gates[j] + ape[j * 128 + lane];
    maximum = fmaxf(maximum, logits[j]);
  }
  float numerator = 0;
  float denominator = 0;
  for (int32_t j = 0; j < 4; ++j) {
    const float probability = expf(logits[j] - maximum);
    numerator += probability * pool_keys[j];
    denominator += probability;
  }
  float value =
      static_cast<float>(c10::BFloat16(__fdiv_rn(numerator, denominator)));
  value = hadamard(value, shared);
  const float scale = quant_scale(value, shared);
  // MATE stores payload and scales in separate planes within each page.
  uint8_t* page_data = cache + page * 64 * 132;
  reinterpret_cast<c10::Float8_e4m3fn*>(page_data)[(pool % 64) * 128 + lane] =
      QuantConverter<c10::Float8_e4m3fn>::convert(__fdiv_rn(value, scale));
  if (lane == 0) {
    reinterpret_cast<float*>(page_data + 64 * 128)[pool % 64] = scale;
  }
}

// Finalize only the request's four-entry ring after all pools read its old
// tail. Keys/gates are either native row tensors or rows embedded in the
// combined CP payload.
__device__ void update_pool_tails_impl(int32_t request,
                                       int32_t lane,
                                       const c10::BFloat16* keys,
                                       const c10::BFloat16* gates,
                                       const uint8_t* gathered,
                                       const int64_t* tail_offsets,
                                       const int64_t* positions,
                                       const int32_t* cu,
                                       const int32_t* state_ids,
                                       const int32_t* read_ids,
                                       const c10::BFloat16* snapshot,
                                       c10::BFloat16* tail,
                                       int64_t states) {
  const int64_t state = state_ids[request];
  const int64_t source = read_ids[request];
  if (state <= 0 || state >= states) {
    return;
  }
  for (int32_t field = 0; field < 8; ++field) {
    if (source > 0 && source < states && source != state) {
      tail[state * 1024 + field * 128 + lane] =
          snapshot[source * 1024 + field * 128 + lane];
    }
  }
  // Keep the last write to each ring entry, including across padded rows.
  int32_t written = 0;
  for (int32_t row = cu[request + 1] - 1; row >= cu[request]; --row) {
    const int64_t position = positions[row];
    if (position < 0) {
      continue;
    }
    const c10::BFloat16* row_keys = nullptr;
    const c10::BFloat16* row_gates = nullptr;
    if (gathered != nullptr) {
      const int64_t offset = tail_offsets[row];
      row_keys = reinterpret_cast<const c10::BFloat16*>(gathered + offset);
      row_gates =
          reinterpret_cast<const c10::BFloat16*>(gathered + offset + 256);
    } else {
      row_keys = keys + row * 128;
      row_gates = gates + row * 128;
    }
    const int64_t ring = position % 4;
    if (!(written & (1 << ring))) {
      tail[state * 1024 + ring * 128 + lane] = row_keys[lane];
      tail[state * 1024 + 512 + ring * 128 + lane] = row_gates[lane];
      written |= 1 << ring;
    }
    if (position == 0) {
      for (int32_t j = 0; j < 4; ++j) {
        if (!(written & (1 << j))) {
          tail[state * 1024 + j * 128 + lane] = c10::BFloat16(0.0f);
          tail[state * 1024 + 512 + j * 128 + lane] = c10::BFloat16(0.0f);
        }
      }
      break;
    }
    if (written == 15) {
      break;
    }
  }
}

__global__ void update_pool_tails(const c10::BFloat16* keys,
                                  const c10::BFloat16* gates,
                                  const int64_t* positions,
                                  const int32_t* cu,
                                  const int32_t* state_ids,
                                  const int32_t* read_ids,
                                  const c10::BFloat16* snapshot,
                                  c10::BFloat16* tail,
                                  int64_t states) {
  update_pool_tails_impl(blockIdx.x,
                         threadIdx.x,
                         keys,
                         gates,
                         nullptr,
                         nullptr,
                         positions,
                         cu,
                         state_ids,
                         read_ids,
                         snapshot,
                         tail,
                         states);
}

// Materialize all gathered compressed pools and the recurrent tail in one
// launch. The two halves touch disjoint cache regions and run independently in
// the same block grid.
__global__ void scatter_indexer_cp(const uint8_t* gathered,
                                   const int64_t* pool_source_offsets,
                                   const int64_t* pool_cache_byte_indices,
                                   int64_t pool_count,
                                   const int64_t* tail_source_offsets,
                                   const int64_t* positions,
                                   const int32_t* cu,
                                   const int32_t* state_ids,
                                   const int32_t* read_ids,
                                   uint8_t* cache,
                                   const c10::BFloat16* snapshot,
                                   c10::BFloat16* tail,
                                   int64_t states,
                                   int32_t request_count,
                                   int64_t tail_count) {
  const int64_t block = blockIdx.x;
  if (block < pool_count) {
    for (int64_t byte = threadIdx.x; byte < kPoolPacketBytes;
         byte += blockDim.x) {
      cache[pool_cache_byte_indices[block * kPoolPacketBytes + byte]] =
          gathered[pool_source_offsets[block] + byte];
    }
  }
  if (block < request_count && tail_count > 0) {
    update_pool_tails_impl(static_cast<int32_t>(block),
                           threadIdx.x,
                           nullptr,
                           nullptr,
                           gathered,
                           tail_source_offsets,
                           positions,
                           cu,
                           state_ids,
                           read_ids,
                           snapshot,
                           tail,
                           states);
  }
}

// One warp owns one query head. Each lane loads four values, matching
// sglang's fused indexer Q kernel and keeping all Hadamard stages inside a
// warp. Four warps share a block to provide the same 128-thread launch shape.
template <int32_t kWarpsPerBlock>
__global__ void quantize_query_grouped(const c10::BFloat16* query,
                                       const float* weights,
                                       c10::Float8_e4m3fn* output,
                                       float* scaled_weights,
                                       int64_t rows) {
  constexpr int32_t kWarpThreads = 32;
  constexpr int32_t kItemsPerLane = 4;
  const int32_t warp = threadIdx.x / kWarpThreads;
  const int32_t lane = threadIdx.x % kWarpThreads;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int32_t base = lane * kItemsPerLane;
  float value[kItemsPerLane];
  if (row < rows) {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerLane; ++item) {
      value[item] = static_cast<float>(query[row * 128 + base + item]);
    }
  } else {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerLane; ++item) {
      value[item] = 0.0f;
    }
  }

  // The first two Hadamard stages stay within the four values held by a lane.
  // The remaining five stages exchange values through lane shuffles.
#pragma unroll
  for (int32_t step = 1; step < kItemsPerLane; step <<= 1) {
#pragma unroll
    for (int32_t group = 0; group < kItemsPerLane; group += 2 * step) {
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
  for (int32_t lane_mask = 1; lane_mask < kWarpThreads; lane_mask <<= 1) {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerLane; ++item) {
      const float other =
          __shfl_xor_sync(0xffffffffu, value[item], lane_mask, kWarpThreads);
      value[item] =
          (lane & lane_mask) ? other - value[item] : value[item] + other;
    }
  }

#pragma unroll
  for (int32_t item = 0; item < kItemsPerLane; ++item) {
    value[item] =
        static_cast<float>(c10::BFloat16(value[item] * 0.08838834764831845f));
  }
  float max_value = 0.0f;
#pragma unroll
  for (int32_t item = 0; item < kItemsPerLane; ++item) {
    max_value = fmaxf(max_value, fabsf(value[item]));
  }
#pragma unroll
  for (int32_t offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    max_value =
        fmaxf(max_value,
              __shfl_xor_sync(0xffffffffu, max_value, offset, kWarpThreads));
  }
  const float scale = exp2f(ceilf(log2f(fmaxf(max_value, 1.0e-4f) / 448.0f)));
  if (row < rows) {
#pragma unroll
    for (int32_t item = 0; item < kItemsPerLane; ++item) {
      output[row * 128 + base + item] =
          QuantConverter<c10::Float8_e4m3fn>::convert(
              __fdiv_rn(value[item], scale));
    }
    if (lane == 0) {
      scaled_weights[row] = weights[row] * scale * (1.0f / 64.0f);
    }
  }
}

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint32_t warp_inclusive_sum_musa(uint32_t lane_id,
                                                            uint32_t value) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(0xFFFFFFFF, value, offset);
    if (lane_id >= offset) value += other;
  }
  return value;
}

template <bool kRegisterThresholdScan,
          uint32_t kCandidateSMEM = 12 * 1024 * sizeof(uint32_t),
          bool kExactOverflow = false>
__device__ __forceinline__ void radix_topk_musa_split_hist(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    const uint32_t length,
    const uint32_t wanted) {
  constexpr uint32_t SMEM_INPUT_SIZE = kCandidateSMEM / (2 * sizeof(int32_t));
  alignas(128) __shared__ uint32_t _s_histogram_buf[2][kRadix + 32];
  alignas(128) __shared__ uint32_t s_group_hist[kHistGroups][kRadix + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_threshold_above_next;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ uint32_t s_sub_bin24_min;
  alignas(128) __shared__ uint32_t s_sub_bin24_max;
  alignas(128) __shared__ uint32_t s_threshold_key_min;
  alignas(128) __shared__ uint32_t s_threshold_key_max;
  alignas(128)
      __shared__ uint32_t s_thread_key_min[kExactOverflow ? kSglBlockSize : 1];
  alignas(128)
      __shared__ uint32_t s_thread_key_max[kExactOverflow ? kSglBlockSize : 1];
  alignas(128) __shared__ int32_t s_last_remain;
  alignas(128) __shared__ uint32_t s_warp_sum[kRadix / 32];

  extern __shared__ uint32_t s_input_idx[];
  const auto candidate_at = [&](uint32_t buffer, uint32_t index) -> uint32_t& {
    return s_input_idx[buffer * SMEM_INPUT_SIZE + index];
  };

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = wanted;
  auto& s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
    if (tx < kRadix) {
      const uint32_t lane_id = tx & 31u;
      const uint32_t warp_id = tx >> 5;
      const uint32_t value = s_histogram[tx];
      const uint32_t warp_inc = warp_inclusive_sum_musa(lane_id, value);
      if (lane_id == 31u) s_warp_sum[warp_id] = warp_inc;
    }
    __syncthreads_lm();

    if (tx < kRadix) {
      const uint32_t lane_id = tx & 31u;
      const uint32_t warp_id = tx >> 5;
      uint32_t total = 0;
      uint32_t inter_prefix = 0;
#pragma unroll
      for (uint32_t i = 0; i < kRadix / 32; ++i) {
        const uint32_t part = s_warp_sum[i];
        total += part;
        if (i < warp_id) inter_prefix += part;
      }
      const uint32_t value = s_histogram[tx];
      const uint32_t warp_inc = warp_inclusive_sum_musa(lane_id, value);
      s_histogram[tx] = total - (inter_prefix + warp_inc) + value;
    } else if (tx == kRadix) {
      s_histogram[kRadix] = 0;
    }
    __syncthreads_lm();
  };

  const auto find_threshold_from_hist = [&](uint32_t target,
                                            int32_t reset_slot,
                                            bool reset_counter,
                                            bool set_last_remain) {
    if constexpr (kRegisterThresholdScan) {
      if (tx < kRadix) {
        const uint32_t lane_id = tx & 31u;
        const uint32_t warp_id = tx >> 5;
        const uint32_t count = s_histogram[tx];
        const uint32_t warp_inc = warp_inclusive_sum_musa(lane_id, count);
        if (lane_id == 31u) s_warp_sum[warp_id] = warp_inc;
      }
      __syncthreads_lm();

      if (tx < kRadix) {
        const uint32_t lane_id = tx & 31u;
        const uint32_t warp_id = tx >> 5;
        uint32_t total = 0;
        uint32_t inter_prefix = 0;
#pragma unroll
        for (uint32_t i = 0; i < kRadix / 32; ++i) {
          const uint32_t part = s_warp_sum[i];
          total += part;
          if (i < warp_id) inter_prefix += part;
        }
        const uint32_t count = s_histogram[tx];
        const uint32_t prefix =
            inter_prefix + warp_inclusive_sum_musa(lane_id, count);
        const uint32_t above = total - prefix + count;
        const uint32_t above_next = above - count;
        if (above > target && above_next <= target) {
          s_threshold_bin_id = tx;
          s_threshold_above_next = above_next;
          if (reset_slot >= 0) s_num_input[reset_slot] = 0;
          if (reset_counter) s_counter = 0;
          if (set_last_remain) s_last_remain = target - above_next;
        }
      }
      __syncthreads_lm();
    } else {
      run_cumsum();
      if (tx < kRadix && s_histogram[tx] > target &&
          s_histogram[tx + 1] <= target) {
        s_threshold_bin_id = tx;
        s_threshold_above_next = s_histogram[tx + 1];
        if (reset_slot >= 0) s_num_input[reset_slot] = 0;
        if (reset_counter) s_counter = 0;
        if (set_last_remain) s_last_remain = target - s_histogram[tx + 1];
      }
      __syncthreads_lm();
    }
  };

  // Stage 1: split coarse histogram to reduce shared-atomic contention on MUSA.
  for (uint32_t i = tx; i < kHistGroups * (kRadix + 32); i += kSglBlockSize) {
    reinterpret_cast<uint32_t*>(s_group_hist)[i] = 0;
  }
  __syncthreads_lm();

  const uint32_t hist_group = tx / kHistGroupThreads;
  uint32_t local_key_min = 0xFFFFFFFFu;
  uint32_t local_key_max = 0;
  for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
    const uint32_t bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_group_hist[hist_group][bin], 1);
  }
  __syncthreads_lm();

  if (tx < kRadix + 1) {
    uint32_t sum = 0;
#pragma unroll
    for (uint32_t g = 0; g < kHistGroups; ++g) sum += s_group_hist[g][tx];
    s_histogram[tx] = sum;
  }
  __syncthreads_lm();

  find_threshold_from_hist(remain_topk, 0, true, false);

  const auto threshold_bin_0 = s_threshold_bin_id;
  remain_topk -= s_threshold_above_next;
  if (remain_topk == 0) {
    for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
      const uint32_t bin = convert_to_uint8(input[idx]);
      if (bin > threshold_bin_0) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    __syncthreads_lm();
    return;
  }

  if constexpr (!kExactOverflow) {
    // The common K-pool path has a very small coarse-bin overflow. Keep the
    // SGLang radix histogram and emit that boundary bin directly. This avoids
    // rescanning the full score row for lower radix digits while preserving the
    // requested pool count.
    if (tx == 0) s_counter = 0;
    __syncthreads_lm();
    for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
      const uint32_t bin = convert_to_uint8(input[idx]);
      if (bin >= threshold_bin_0) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        if (pos < wanted) output[pos] = idx;
      }
    }
    __syncthreads_lm();
    return;
  }

  __syncthreads_lm();
  if (tx < kRadix + 1) s_histogram[tx] = 0;
  if (tx == 0) {
    s_num_input[1] = 0;
    s_sub_bin24_min = 0xFFFFFFFFu;
    s_sub_bin24_max = 0;
    s_threshold_key_min = 0xFFFFFFFFu;
    s_threshold_key_max = 0;
  }
  __syncthreads_lm();

  for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
    const float raw_input = input[idx];
    const uint32_t bin = convert_to_uint8(raw_input);
    if (bin > threshold_bin_0) {
      const auto pos = ::atomicAdd(&s_counter, 1);
      output[pos] = idx;
    } else if (bin == threshold_bin_0) {
      const auto pos = ::atomicAdd(&s_num_input[1], 1);
      const auto key = convert_to_uint32(raw_input);
      if constexpr (kExactOverflow) {
        local_key_min = key < local_key_min ? key : local_key_min;
        local_key_max = key > local_key_max ? key : local_key_max;
      }
      if (pos < SMEM_INPUT_SIZE) {
        candidate_at(1, pos) = idx;
        const auto sub_bin24 = (key >> 24) & 0xFF;
        const auto sub_bin16 = (key >> 16) & 0xFF;
        ::atomicMin(&s_sub_bin24_min, sub_bin24);
        ::atomicMax(&s_sub_bin24_max, sub_bin24);
        ::atomicAdd(&s_histogram[sub_bin16], 1);
      }
    }
  }
  __syncthreads_lm();
  if constexpr (kExactOverflow) {
    if (s_num_input[1] <= remain_topk) {
      if (tx == 0) s_counter = 0;
      __syncthreads_lm();
      for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
        const uint32_t bin = convert_to_uint8(input[idx]);
        if (bin >= threshold_bin_0) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          if (pos < wanted) output[pos] = idx;
        }
      }
      __syncthreads_lm();
      return;
    }
  }
  if constexpr (kExactOverflow) {
    s_thread_key_min[tx] = local_key_min;
    s_thread_key_max[tx] = local_key_max;
    __syncthreads_lm();
#pragma unroll
    for (uint32_t offset = kSglBlockSize / 2; offset > 0; offset >>= 1) {
      if (tx < offset) {
        const uint32_t other_min = s_thread_key_min[tx + offset];
        const uint32_t other_max = s_thread_key_max[tx + offset];
        s_thread_key_min[tx] =
            other_min < s_thread_key_min[tx] ? other_min : s_thread_key_min[tx];
        s_thread_key_max[tx] =
            other_max > s_thread_key_max[tx] ? other_max : s_thread_key_max[tx];
      }
      __syncthreads_lm();
    }
    if (tx == 0) {
      s_threshold_key_min = s_thread_key_min[0];
      s_threshold_key_max = s_thread_key_max[0];
    }
    __syncthreads_lm();

    if (s_num_input[1] > SMEM_INPUT_SIZE &&
        s_threshold_key_min != s_threshold_key_max) {
      uint32_t prefix = 0;
      uint32_t prefix_mask = 0;
      uint32_t target = wanted;
#pragma unroll 4
      for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t i = tx; i < kHistGroups * (kRadix + 32);
             i += kSglBlockSize) {
          reinterpret_cast<uint32_t*>(s_group_hist)[i] = 0;
        }
        __syncthreads_lm();
        for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
          const uint32_t key = convert_to_uint32(input[idx]);
          if ((key & prefix_mask) == prefix) {
            const uint32_t bin = (key >> shift) & 0xFF;
            ::atomicAdd(&s_group_hist[hist_group][bin], 1);
          }
        }
        __syncthreads_lm();
        if (tx < kRadix + 1) {
          uint32_t sum = 0;
#pragma unroll
          for (uint32_t group = 0; group < kHistGroups; ++group) {
            sum += s_group_hist[group][tx];
          }
          s_histogram[tx] = sum;
        }
        __syncthreads_lm();
        find_threshold_from_hist(target, -1, false, false);
        const uint32_t threshold = s_threshold_bin_id;
        target -= s_threshold_above_next;
        prefix |= threshold << shift;
        prefix_mask |= 0xFFu << shift;
        if (target == 0) {
          if (shift > 0) prefix |= (1u << shift) - 1u;
          break;
        }
      }

      if (tx == 0) s_counter = 0;
      __syncthreads_lm();
      for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
        if (convert_to_uint32(input[idx]) > prefix) {
          const uint32_t pos = ::atomicAdd(&s_counter, 1);
          if (pos < wanted) output[pos] = idx;
        }
      }
      __syncthreads_lm();
      for (uint32_t idx = tx; idx < length; idx += kSglBlockSize) {
        if (convert_to_uint32(input[idx]) == prefix) {
          const uint32_t pos = ::atomicAdd(&s_counter, 1);
          if (pos < wanted) output[pos] = idx;
        }
      }
      __syncthreads_lm();
      return;
    }
  }

  // Long decode top candidates commonly share the top FP32 byte after the
  // fp16 coarse threshold; skip that no-op radix round when it is uniform.
  int start_round = 1;
  if (s_sub_bin24_min != s_sub_bin24_max) {
    if (tx < kRadix + 1) s_histogram[tx] = 0;
    if (tx == 0) s_num_input[0] = 0;
    __syncthreads_lm();
    const auto raw_num_input = s_num_input[1];
    const auto num_input =
        raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;
    for (uint32_t i = tx; i < num_input; i += kSglBlockSize) {
      const auto idx = candidate_at(1, i);
      const auto key = convert_to_uint32(input[idx]);
      const auto pos = ::atomicAdd(&s_num_input[0], 1);
      if (pos < SMEM_INPUT_SIZE) {
        candidate_at(0, pos) = idx;
        const auto sub_bin24 = (key >> 24) & 0xFF;
        ::atomicAdd(&s_histogram[sub_bin24], 1);
      }
    }
    start_round = 0;
    __syncthreads_lm();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    if (round < start_round) continue;
    const auto r_idx = round % 2;
    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input =
        raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    find_threshold_from_hist(remain_topk, r_idx ^ 1, false, true);

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_threshold_above_next;

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += kSglBlockSize) {
        const auto idx = candidate_at(r_idx, i);
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      }
      __syncthreads_lm();
      break;
    }

    __syncthreads_lm();
    if (tx < kRadix + 1) s_histogram[tx] = 0;
    __syncthreads_lm();
    for (uint32_t i = tx; i < num_input; i += kSglBlockSize) {
      const auto idx = candidate_at(r_idx, i);
      const auto raw_input = input[idx];
      const auto offset = 24 - round * 8;
      const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) output[wanted - pos] = idx;
        } else {
          const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
          if (pos < SMEM_INPUT_SIZE) {
            candidate_at(r_idx ^ 1, pos) = idx;
            const auto sub_bin =
                (convert_to_uint32(raw_input) >> (offset - 8)) & 0xFF;
            ::atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads_lm();
  }
}

// The SGLang MUSA transform selects complete rate-4 pools with an exact
// four-pass radix selector. Keeping this in the same launch as expansion avoids
// materializing pool ids or sorting a second tensor.
__global__ void select_pool_tokens_sgl(const float* scores,
                                       const int32_t* pool_lengths,
                                       const int64_t* positions,
                                       const int32_t* rows,
                                       int32_t* tokens,
                                       int32_t* lengths,
                                       int64_t score_stride,
                                       int64_t pools,
                                       int64_t first_request,
                                       int64_t request_end,
                                       int64_t width,
                                       int64_t token_topk) {
  constexpr int32_t kThreads = 512;
  __shared__ int32_t selected[512];
  const int64_t row = blockIdx.x;
  const int32_t lane = threadIdx.x;
  const int64_t request = max(rows[row], 0);
  if (request < first_request || request >= request_end) {
    return;
  }
  const int64_t visible = rows[row] >= 0 ? positions[row] + 1 : 0;
  const int64_t tail = visible % 4;
  const int64_t end = visible <= token_topk ? visible : token_topk + tail;
  const uint32_t pool_length = static_cast<uint32_t>(max(pool_lengths[row], 0));
  const uint32_t wanted = static_cast<uint32_t>(
      (token_topk / 4 < pool_length) ? token_topk / 4 : pool_length);
  if (lane == 0) {
    lengths[row] = static_cast<int32_t>(end);
  }
  if (visible > token_topk && wanted > 0) {
    if (pool_length == wanted) {
      if (lane < static_cast<int32_t>(pool_length)) {
        selected[lane] = lane;
      }
    } else {
      radix_topk_musa_split_hist<true, 12 * 1024 * sizeof(uint32_t), true>(
          scores + row * score_stride + (request - first_request) * pools,
          selected,
          pool_length,
          wanted);
      // Radix selection appends pool IDs with atomics. Use logical pool order
      // so sparse attention reduces the same values in eager and graph replay.
      using BlockSort = cub::BlockRadixSort<uint32_t, kThreads, 1>;
      static_assert(sizeof(BlockSort::TempStorage) <=
                    12 * 1024 * sizeof(uint32_t));
      extern __shared__ uint32_t sort_scratch[];
      __syncthreads_lm();
      uint32_t pool_id[1] = {static_cast<uint32_t>(lane) < wanted
                                 ? static_cast<uint32_t>(selected[lane])
                                 : UINT32_MAX};
      auto& sort_storage =
          *reinterpret_cast<BlockSort::TempStorage*>(sort_scratch);
      BlockSort(sort_storage).Sort(pool_id, 0, 32 - __clz(pool_length));
      selected[lane] = static_cast<int32_t>(pool_id[0]);
    }
  }
  __syncthreads_lm();
  for (int64_t column = lane; column < width; column += kThreads) {
    int32_t token = -1;
    if (visible <= token_topk) {
      if (column < end) {
        token = static_cast<int32_t>(column);
      }
    } else if (column < token_topk) {
      const int64_t pool_slot = column / 4;
      if (pool_slot < wanted) {
        token = selected[pool_slot] * 4 + static_cast<int32_t>(column % 4);
      }
    } else if (column < end) {
      token = static_cast<int32_t>((visible / 4) * 4 + column - token_topk);
    }
    tokens[row * width + column] = token;
  }
}

// A physical page stores 64 FP8 keys followed by 64 FP32 scales.
__global__ void gather_pool_prefix(const uint8_t* cache,
                                   const int32_t* table,
                                   uint8_t* keys,
                                   float* scales,
                                   int64_t pools,
                                   int64_t table_stride,
                                   int64_t physical_pages) {
  const int64_t page = blockIdx.x;
  const int64_t request = blockIdx.y;
  const int64_t count = min(int64_t{64}, pools - page * 64);
  const int32_t physical = table[request * table_stride + page];
  const int64_t offset = request * pools + page * 64;
  // Shorter requests and graph padding can have unmapped trailing pages.
  if (physical < 0 || physical >= physical_pages) {
    return;
  }
  const uint8_t* source = cache + static_cast<int64_t>(physical) * 64 * 132;
  for (int64_t i = threadIdx.x; i < count * 128; i += blockDim.x) {
    keys[offset * 128 + i] = source[i];
  }
  if (threadIdx.x < count) {
    scales[offset + threadIdx.x] =
        reinterpret_cast<const float*>(source + 64 * 128)[threadIdx.x];
  }
}

void check_tensor(const torch::Tensor& value,
                  const torch::Tensor& anchor,
                  torch::ScalarType dtype) {
  CHECK(value.defined());
  CHECK_EQ(value.device(), anchor.device());
  CHECK_EQ(value.scalar_type(), dtype);
  CHECK(value.is_contiguous());
}

// Keep the foreign allocator's tensor alive until the Torch consumer finishes.
// The existing FFI stream guard orders its producer before Torch reads it.
torch::Tensor own_logits(ffi::Tensor tensor,
                         const torch::TensorOptions& options) {
  std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<int64_t> strides;
  for (int32_t dim = 0; dim < tensor.ndim(); ++dim) {
    strides.push_back(tensor.stride(dim));
  }
  void* data = tensor.data_ptr();
  return torch::from_blob(
      data,
      shape,
      strides,
      [owner = std::move(tensor)](void*) {},
      options.dtype(torch::kFloat32));
}
}  // namespace

Glm5NextIndexerSelection glm5_next_indexer(const torch::Tensor& query,
                                           const torch::Tensor& raw_key,
                                           const torch::Tensor& gate,
                                           const torch::Tensor& weights,
                                           const torch::Tensor& ape,
                                           const torch::Tensor& positions,
                                           const torch::Tensor& cu_seqlens,
                                           const torch::Tensor& state_ids,
                                           const torch::Tensor& read_state_ids,
                                           const torch::Tensor& block_table,
                                           torch::Tensor& cache,
                                           torch::Tensor& tail,
                                           int64_t max_context,
                                           int64_t token_topk) {
  RECORD_USER_SCOPE("kernel: glm5_next_indexer");
  CHECK(query.device().is_privateuseone());
  XLLMDeviceGuard device_guard(query.device());
  check_tensor(query, query, torch::kBFloat16);
  check_tensor(raw_key, query, torch::kBFloat16);
  check_tensor(gate, query, torch::kBFloat16);
  check_tensor(weights, query, torch::kFloat32);
  check_tensor(ape, query, torch::kFloat32);
  check_tensor(positions, query, torch::kInt64);
  check_tensor(cu_seqlens, query, torch::kInt32);
  check_tensor(state_ids, query, torch::kInt32);
  check_tensor(read_state_ids, query, torch::kInt32);
  check_tensor(block_table, query, torch::kInt32);
  check_tensor(cache, query, torch::kUInt8);
  check_tensor(tail, query, torch::kBFloat16);
  CHECK_EQ(query.dim(), 3);
  const int64_t tokens = query.size(0);
  CHECK_EQ(query.sizes(), torch::IntArrayRef({tokens, 32, 128}));
  CHECK_EQ(raw_key.sizes(), torch::IntArrayRef({tokens, 128}));
  CHECK_EQ(gate.sizes(), raw_key.sizes());
  CHECK_EQ(weights.sizes(), torch::IntArrayRef({tokens, 32}));
  CHECK_EQ(ape.sizes(), torch::IntArrayRef({4, 128}));
  CHECK_EQ(positions.sizes(), torch::IntArrayRef({tokens}));
  CHECK_EQ(block_table.dim(), 2);
  const int64_t batch = block_table.size(0);
  CHECK_EQ(cu_seqlens.sizes(), torch::IntArrayRef({batch + 1}));
  CHECK_EQ(state_ids.sizes(), torch::IntArrayRef({batch}));
  CHECK_EQ(read_state_ids.sizes(), state_ids.sizes());
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.sizes().slice(1), torch::IntArrayRef({64, 1, 132}));
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.sizes().slice(1), torch::IntArrayRef({2, 4, 128}));
  CHECK_GT(token_topk, 0);
  CHECK_EQ(token_topk % 4, 0);
  CHECK_LE(token_topk, 2048);
  CHECK_GT(max_context, 0);
  CHECK_LE(max_context, block_table.size(1) * 256);
  CHECK_LE(max_context, std::numeric_limits<int32_t>::max());
  auto ints = query.options().dtype(torch::kInt32);
  if (tokens == 0) {
    return glm5_next_indexer_select(query,
                                    weights,
                                    positions,
                                    torch::empty({0}, ints),
                                    block_table,
                                    cache,
                                    max_context,
                                    token_topk);
  }
  auto query_rows = torch::empty({tokens}, ints);
  musaDeviceProp properties{};
  CHECK_EQ(musaGetDeviceProperties(&properties, query.device().index()),
           musaSuccess);
  CHECK_EQ(properties.major * 10 + properties.minor, 31);
  if (max_context > token_topk) {
    get_function("glm_kpool_nonpaged_v1", "fp8_mqa_logits");
  }
  const auto stream = current_musa_stream(query.device());
  auto pool_lengths = torch::empty({tokens}, ints);
  auto snapshot =
      read_state_ids.data_ptr() != state_ids.data_ptr() ? tail.clone() : tail;
  if (tokens > batch) {
    update_pools_parallel<<<tokens, 128, 0, stream>>>(
        raw_key.data_ptr<c10::BFloat16>(),
        gate.data_ptr<c10::BFloat16>(),
        ape.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        state_ids.data_ptr<int32_t>(),
        read_state_ids.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        cache.data_ptr<uint8_t>(),
        snapshot.data_ptr<c10::BFloat16>(),
        query_rows.data_ptr<int32_t>(),
        pool_lengths.data_ptr<int32_t>(),
        tail.size(0),
        cache.size(0),
        block_table.size(1));
    update_pool_tails<<<batch, 128, 0, stream>>>(
        raw_key.data_ptr<c10::BFloat16>(),
        gate.data_ptr<c10::BFloat16>(),
        positions.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        state_ids.data_ptr<int32_t>(),
        read_state_ids.data_ptr<int32_t>(),
        snapshot.data_ptr<c10::BFloat16>(),
        tail.data_ptr<c10::BFloat16>(),
        tail.size(0));
  } else {
    update_pools<<<batch, 128, 0, stream>>>(raw_key.data_ptr<c10::BFloat16>(),
                                            gate.data_ptr<c10::BFloat16>(),
                                            ape.data_ptr<float>(),
                                            positions.data_ptr<int64_t>(),
                                            cu_seqlens.data_ptr<int32_t>(),
                                            state_ids.data_ptr<int32_t>(),
                                            read_state_ids.data_ptr<int32_t>(),
                                            block_table.data_ptr<int32_t>(),
                                            cache.data_ptr<uint8_t>(),
                                            snapshot.data_ptr<c10::BFloat16>(),
                                            tail.data_ptr<c10::BFloat16>(),
                                            query_rows.data_ptr<int32_t>(),
                                            pool_lengths.data_ptr<int32_t>(),
                                            tail.size(0),
                                            cache.size(0),
                                            block_table.size(1));
  }
  check_musa_kernel_launch("GLM IndexPool update");
  return glm5_next_indexer_select(query,
                                  weights,
                                  positions,
                                  query_rows,
                                  block_table,
                                  cache,
                                  max_context,
                                  token_topk);
}

Glm5NextIndexerSelection glm5_next_indexer_select(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& positions,
    const torch::Tensor& rows,
    const torch::Tensor& block_table,
    const torch::Tensor& cache,
    int64_t max_context,
    int64_t token_topk) {
  RECORD_USER_SCOPE("kernel: glm5_next_indexer_select");
  XLLMDeviceGuard device_guard(query.device());
  check_tensor(query, query, torch::kBFloat16);
  check_tensor(weights, query, torch::kFloat32);
  check_tensor(positions, query, torch::kInt64);
  check_tensor(rows, query, torch::kInt32);
  check_tensor(block_table, query, torch::kInt32);
  check_tensor(cache, query, torch::kUInt8);
  const int64_t tokens = query.size(0);
  CHECK_EQ(query.sizes(), torch::IntArrayRef({tokens, 32, 128}));
  CHECK_EQ(weights.sizes(), torch::IntArrayRef({tokens, 32}));
  CHECK_EQ(positions.sizes(), torch::IntArrayRef({tokens}));
  CHECK_EQ(rows.sizes(), positions.sizes());
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.sizes().slice(1), torch::IntArrayRef({64, 1, 132}));
  CHECK_GT(max_context, 0);
  CHECK_LE(max_context, std::numeric_limits<int32_t>::max());
  CHECK_LE(max_context, block_table.size(1) * 256);
  CHECK_GT(token_topk, 0);
  CHECK_EQ(token_topk % 4, 0);
  CHECK_LE(token_topk, 2048);
  const int64_t width = ((token_topk + 3 + 127) / 128) * 128;
  auto ints = query.options().dtype(torch::kInt32);
  Glm5NextIndexerSelection result{
      torch::empty({tokens, width}, ints), torch::empty({tokens}, ints), rows};
  if (tokens == 0) {
    return result;
  }
  const torch::Tensor pool_lengths =
      torch::where(rows.ge(0), (positions + 1).div(4, "floor"), 0)
          .to(torch::kInt32);
  musaDeviceProp properties{};
  CHECK_EQ(musaGetDeviceProperties(&properties, query.device().index()),
           musaSuccess);
  CHECK_EQ(properties.major * 10 + properties.minor, 31);
  const auto stream = current_musa_stream(query.device());
  const int64_t pools = max_context / 4;
  const int64_t batch = block_table.size(0);
  CHECK_GT(batch, 0);
  const auto select_rows = [&](const float* scores,
                               int64_t stride,
                               int64_t first,
                               int64_t size,
                               int64_t first_request,
                               int64_t request_end) {
    select_pool_tokens_sgl<<<size, 512, 12 * 1024 * sizeof(uint32_t), stream>>>(
        scores,
        pool_lengths.data_ptr<int32_t>() + first,
        positions.data_ptr<int64_t>() + first,
        rows.data_ptr<int32_t>() + first,
        result.tokens.data_ptr<int32_t>() + first * width,
        result.lengths.data_ptr<int32_t>() + first,
        stride,
        pools,
        first_request,
        request_end,
        width,
        token_topk);
    check_musa_kernel_launch("GLM IndexPool fused candidate selection");
  };
  if (max_context <= token_topk) {
    select_rows(nullptr, 0, 0, tokens, 0, batch);
    return result;
  }

  auto quant = torch::empty({tokens, 32, 128},
                            query.options().dtype(torch::kFloat8_e4m3fn));
  // MATE checks the row stride even for single-token decode.
  auto scaled_weights = torch::empty({tokens, 32}, weights.options());
  const int64_t query_rows = tokens * 32;
  quantize_query_grouped<4><<<(query_rows + 3) / 4, 128, 0, stream>>>(
      query.data_ptr<c10::BFloat16>(),
      weights.data_ptr<float>(),
      quant.data_ptr<c10::Float8_e4m3fn>(),
      scaled_weights.data_ptr<float>(),
      query_rows);
  check_musa_kernel_launch("GLM IndexPool query quantization");

  // Pack consecutive requests so every Q has a contiguous logical K range.
  // Bound the replicated prefix storage for batches with widely varying
  // lengths.
  constexpr int64_t kGatherBudget = 64 * 1024 * 1024;
  const int64_t group_size =
      std::clamp<int64_t>(kGatherBudget / (pools * kPoolPacketBytes),
                          1,
                          std::min<int64_t>(batch, 65535));
  const int64_t max_keys = group_size * pools;
  int64_t chunk = tokens;
  if (tokens * max_keys >= kMqaLogitsSkipChunkElements) {
    int64_t budget = kMqaLogitsSkipChunkElements * sizeof(float);
    // Allocation recording and graph capture must agree on Q chunk shapes.
    if (get_ffi_alloc_mode() == FfiAllocMode::kPassthrough &&
        !is_stream_capturing()) {
      size_t free_memory = 0;
      size_t total_memory = 0;
      CHECK_EQ(musaMemGetInfo(&free_memory, &total_memory), musaSuccess);
      budget = std::min(free_memory * kMqaLogitsFreeMemNumerator /
                            kMqaLogitsFreeMemDenominator,
                        total_memory * kMqaLogitsTotalMemNumerator /
                            kMqaLogitsTotalMemDenominator);
    }
    chunk = std::clamp<int64_t>(
        budget / ((max_keys + 128) * sizeof(float)), 1, tokens);
  }
  auto score_fn = get_function("glm_kpool_nonpaged_v1", "fp8_mqa_logits");
  for (int64_t request = 0; request < batch; request += group_size) {
    const int64_t count = std::min(group_size, batch - request);
    auto keys = torch::empty({count * pools, 128}, quant.options());
    auto scales = torch::empty({count * pools}, weights.options());
    gather_pool_prefix<<<dim3((pools + 63) / 64, count), 256, 0, stream>>>(
        cache.data_ptr<uint8_t>(),
        block_table.data_ptr<int32_t>() + request * block_table.size(1),
        static_cast<uint8_t*>(keys.data_ptr()),
        scales.data_ptr<float>(),
        pools,
        block_table.size(1),
        cache.size(0));
    check_musa_kernel_launch("GLM IndexPool prefix gather");
    auto starts = torch::zeros_like(pool_lengths);
    auto ends = pool_lengths;
    if (batch > 1) {
      starts = (rows - request).clamp(0, count - 1) * pools;
      ends = starts + torch::where(rows.ge(request) & rows.lt(request + count),
                                   pool_lengths,
                                   0);
    }
    for (int64_t first = 0; first < tokens; first += chunk) {
      const int64_t size = std::min(chunk, tokens - first);
      ffi::Tensor logits;
      {
        TvmffiStreamGuard guard(query.device());
        logits = score_fn(to_ffi_tensor(quant.narrow(0, first, size)),
                          to_ffi_tensor(keys),
                          to_ffi_tensor(scaled_weights.narrow(0, first, size)),
                          to_ffi_tensor(starts.narrow(0, first, size)),
                          to_ffi_tensor(ends.narrow(0, first, size)),
                          to_ffi_tensor(scales),
                          false,
                          int64_t{0})
                     .cast<ffi::Tensor>();
      }
      auto scores = own_logits(std::move(logits), query.options());
      select_rows(scores.data_ptr<float>(),
                  scores.stride(0),
                  first,
                  size,
                  request,
                  request + count);
    }
  }
  return result;
}
// CP owns a contiguous range of pool-ending rows. Only the preceding three
// projected key/gate rows are needed from other ranks. The recurrent tail is
// read here and finalized after the compressed pool exchange.
void glm5_next_indexer_update_cp(const torch::Tensor& keys,
                                 const torch::Tensor& gates,
                                 const torch::Tensor& ape,
                                 const torch::Tensor& positions,
                                 const torch::Tensor& cu,
                                 const torch::Tensor& ids,
                                 const torch::Tensor& read_ids,
                                 const torch::Tensor& table,
                                 torch::Tensor& cache,
                                 const torch::Tensor& tail,
                                 int64_t first_row,
                                 int64_t count,
                                 int64_t key_begin) {
  RECORD_USER_SCOPE("kernel: glm5_next_indexer_update_cp");
  XLLMDeviceGuard device_guard(keys.device());
  CHECK_GE(count, 0);
  CHECK_GE(first_row, key_begin);
  CHECK_GE(key_begin, 0);
  CHECK_LE(first_row + count, positions.numel());
  CHECK_EQ(keys.sizes(),
           torch::IntArrayRef({first_row + count - key_begin, 128}));
  CHECK_EQ(gates.sizes(), keys.sizes());
  CHECK_EQ(ape.sizes(), torch::IntArrayRef({4, 128}));
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(cu.sizes(), torch::IntArrayRef({table.size(0) + 1}));
  CHECK_EQ(ids.sizes(), torch::IntArrayRef({table.size(0)}));
  CHECK_EQ(read_ids.sizes(), ids.sizes());
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.sizes().slice(1), torch::IntArrayRef({64, 1, 132}));
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.sizes().slice(1), torch::IntArrayRef({2, 4, 128}));
  for (const auto& value : {keys, gates, tail}) {
    check_tensor(value, keys, torch::kBFloat16);
  }
  check_tensor(ape, keys, torch::kFloat32);
  check_tensor(positions, keys, torch::kInt64);
  for (const auto& value : {cu, ids, read_ids, table}) {
    check_tensor(value, keys, torch::kInt32);
  }
  check_tensor(cache, keys, torch::kUInt8);
  if (count == 0) {
    return;
  }
  auto rows = torch::empty({count}, keys.options().dtype(torch::kInt32));
  auto lengths = torch::empty_like(rows);
  update_pools_parallel<<<count, 128, 0, current_musa_stream(keys.device())>>>(
      keys.data_ptr<c10::BFloat16>(),
      gates.data_ptr<c10::BFloat16>(),
      ape.data_ptr<float>(),
      positions.data_ptr<int64_t>(),
      cu.data_ptr<int32_t>(),
      ids.data_ptr<int32_t>(),
      read_ids.data_ptr<int32_t>(),
      table.data_ptr<int32_t>(),
      cache.data_ptr<uint8_t>(),
      tail.data_ptr<c10::BFloat16>(),
      rows.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(),
      tail.size(0),
      cache.size(0),
      table.size(1),
      first_row,
      key_begin);
  check_musa_kernel_launch("GLM CP IndexPool update");
}

void glm5_next_indexer_update_tail(const torch::Tensor& keys,
                                   const torch::Tensor& gates,
                                   const torch::Tensor& positions,
                                   const torch::Tensor& cu,
                                   const torch::Tensor& ids,
                                   const torch::Tensor& read_ids,
                                   torch::Tensor& tail) {
  RECORD_USER_SCOPE("kernel: glm5_next_indexer_update_tail");
  XLLMDeviceGuard device_guard(keys.device());
  for (const auto& value : {keys, gates, tail}) {
    check_tensor(value, keys, torch::kBFloat16);
  }
  for (const auto& value : {cu, ids, read_ids}) {
    check_tensor(value, keys, torch::kInt32);
  }
  check_tensor(positions, keys, torch::kInt64);
  CHECK_EQ(keys.sizes(), torch::IntArrayRef({positions.numel(), 128}));
  CHECK_EQ(gates.sizes(), keys.sizes());
  CHECK_GT(ids.numel(), 0);
  CHECK_EQ(cu.numel(), ids.numel() + 1);
  CHECK_EQ(read_ids.sizes(), ids.sizes());
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.sizes().slice(1), torch::IntArrayRef({2, 4, 128}));
  auto snapshot = tail.clone();
  update_pool_tails<<<ids.numel(),
                      128,
                      0,
                      current_musa_stream(keys.device())>>>(
      keys.data_ptr<c10::BFloat16>(),
      gates.data_ptr<c10::BFloat16>(),
      positions.data_ptr<int64_t>(),
      cu.data_ptr<int32_t>(),
      ids.data_ptr<int32_t>(),
      read_ids.data_ptr<int32_t>(),
      snapshot.data_ptr<c10::BFloat16>(),
      tail.data_ptr<c10::BFloat16>(),
      tail.size(0));
  check_musa_kernel_launch("GLM CP IndexPool tail update");
}

void glm5_next_indexer_scatter_cp(const torch::Tensor& gathered,
                                  const torch::Tensor& pool_source_offsets,
                                  const torch::Tensor& pool_cache_byte_indices,
                                  const torch::Tensor& tail_source_offsets,
                                  const torch::Tensor& positions,
                                  const torch::Tensor& cu,
                                  const torch::Tensor& ids,
                                  const torch::Tensor& read_ids,
                                  torch::Tensor& cache,
                                  torch::Tensor& tail) {
  RECORD_USER_SCOPE("kernel: glm5_next_indexer_scatter_cp");
  XLLMDeviceGuard device_guard(gathered.device());
  check_tensor(gathered, gathered, torch::kUInt8);
  for (const auto& value : {pool_source_offsets,
                            pool_cache_byte_indices,
                            tail_source_offsets,
                            positions}) {
    check_tensor(value, gathered, torch::kInt64);
  }
  for (const auto& value : {cu, ids, read_ids}) {
    check_tensor(value, gathered, torch::kInt32);
  }
  check_tensor(cache, gathered, torch::kUInt8);
  check_tensor(tail, gathered, torch::kBFloat16);
  CHECK_EQ(gathered.dim(), 1);
  CHECK_EQ(pool_cache_byte_indices.numel(),
           pool_source_offsets.numel() * kPoolPacketBytes);
  CHECK_EQ(tail_source_offsets.numel(), positions.numel());
  CHECK_GT(ids.numel(), 0);
  CHECK_EQ(cu.numel(), ids.numel() + 1);
  CHECK_EQ(read_ids.sizes(), ids.sizes());
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.sizes().slice(1), torch::IntArrayRef({64, 1, 132}));
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.sizes().slice(1), torch::IntArrayRef({2, 4, 128}));
  const int64_t pool_count = pool_source_offsets.numel();
  const int64_t blocks = std::max(pool_count, ids.numel());
  CHECK_LE(blocks, std::numeric_limits<int32_t>::max());
  auto snapshot = tail.clone();
  scatter_indexer_cp<<<static_cast<int32_t>(blocks),
                       128,
                       0,
                       current_musa_stream(gathered.device())>>>(
      gathered.data_ptr<uint8_t>(),
      pool_source_offsets.data_ptr<int64_t>(),
      pool_cache_byte_indices.data_ptr<int64_t>(),
      pool_count,
      tail_source_offsets.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      cu.data_ptr<int32_t>(),
      ids.data_ptr<int32_t>(),
      read_ids.data_ptr<int32_t>(),
      cache.data_ptr<uint8_t>(),
      snapshot.data_ptr<c10::BFloat16>(),
      tail.data_ptr<c10::BFloat16>(),
      tail.size(0),
      static_cast<int32_t>(ids.numel()),
      tail_source_offsets.numel());
  check_musa_kernel_launch("GLM CP Indexed cache and tail scatter");
}
}  // namespace xllm::kernel::musa
