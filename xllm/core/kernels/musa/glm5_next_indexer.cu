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
#include <limits>
#include <vector>

#include "core/kernels/musa/glm5_next_indexer.h"
#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {
constexpr int64_t kPoolPacketBytes = 132;

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

__device__ __forceinline__ uint32_t topk_rank_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint32_t topk_rank_key8(float value) {
  return topk_rank_key(value) >> 24;
}

// Radix-select the top pool ids for one query.  The ordering matches stable
// descending top-k: a larger score wins, and the smaller pool id wins a tie.
// The selected ids are unordered; token expansion below does not depend on
// their order.
__device__ int32_t radix_select_pools(const float* scores,
                                      int32_t pool_length,
                                      int32_t wanted,
                                      int32_t* selected) {
  constexpr int32_t kThreads = 256;
  constexpr int32_t kRadix = 256;
  constexpr int32_t kScratch = 4096;
  __shared__ int32_t histogram[2][kRadix + 32];
  __shared__ int32_t threshold_bin;
  __shared__ int32_t scratch_count;
  __shared__ int32_t selected_count;
  __shared__ int32_t scratch[kScratch];
  const int32_t lane = threadIdx.x;
  if (lane == 0) {
    selected_count = 0;
    scratch_count = 0;
    threshold_bin = 0;
  }
  if (lane < kRadix + 1) {
    histogram[0][lane] = 0;
  }
  __syncthreads();
  for (int32_t pool = lane; pool < pool_length; pool += kThreads) {
    atomicAdd(&histogram[0][topk_rank_key8(scores[pool])], 1);
  }
  __syncthreads();
  // Eight steps end in histogram[0].  Each value is the count of this bin and
  // every higher bin, which is the descending rank boundary.
  for (int32_t stride = 1; stride < kRadix; stride *= 2) {
    const int32_t source = (stride & 1) == 1 ? 1 : 0;
    if (lane < kRadix) {
      int32_t value = histogram[source][lane];
      if (lane + stride < kRadix) {
        value += histogram[source][lane + stride];
      }
      histogram[source ^ 1][lane] = value;
    }
    __syncthreads();
  }
  if (lane < kRadix && histogram[0][lane] >= wanted &&
      (lane + 1 == kRadix || histogram[0][lane + 1] < wanted)) {
    threshold_bin = lane;
  }
  __syncthreads();
  const int32_t bin = threshold_bin;
  const int32_t strict =
      lane + 1 < kRadix ? histogram[0][lane == bin ? bin + 1 : lane] : 0;
  int32_t kept_above = 0;
  if (lane == 0) {
    kept_above = bin + 1 < kRadix ? histogram[0][bin + 1] : 0;
  }
  __shared__ int32_t above_count;
  if (lane == 0) {
    above_count = bin + 1 < kRadix ? histogram[0][bin + 1] : 0;
  }
  __syncthreads();
  for (int32_t pool = lane; pool < pool_length; pool += kThreads) {
    const uint32_t key = topk_rank_key8(scores[pool]);
    if (static_cast<int32_t>(key) > bin) {
      const int32_t pos = atomicAdd(&selected_count, 1);
      if (pos < wanted) {
        selected[pos] = pool;
      }
    } else if (static_cast<int32_t>(key) == bin) {
      const int32_t pos = atomicAdd(&scratch_count, 1);
      if (pos < kScratch) {
        scratch[pos] = pool;
      }
    }
  }
  __syncthreads();
  const int32_t need = wanted - above_count;
  const int32_t candidates = scratch_count;
  // Fill the remaining slots from the threshold bin in increasing pool-id
  // order. Sorting that bin dominated a 20k prefill; sparse attention consumes
  // the selected set, and the smaller id remains the stable tie break.
  if (need > 0 && candidates <= kScratch) {
    const int32_t limit = min(candidates, kScratch);
    const int32_t emit = min(need, limit);
    for (int32_t item = lane; item < emit; item += kThreads) {
      selected[above_count + item] = scratch[item];
    }
    if (lane == 0) {
      selected_count = above_count + emit;
    }
    __syncthreads();
  } else if (need > 0) {
    for (int32_t pool = lane; pool < pool_length && need > 0;
         pool += kThreads) {
      if (static_cast<int32_t>(topk_rank_key8(scores[pool])) == bin) {
        const int32_t slot = atomicAdd(&selected_count, 1);
        if (slot < wanted) {
          selected[slot] = pool;
        }
      }
    }
    __syncthreads();
  }
  return selected_count;
}

// One block owns one query row.  It selects token_topk/4 pools, expands each
// pool to four logical token ids, and appends the unpadded causal tail.
__global__ void select_pool_tokens(const float* scores,
                                   const int32_t* pool_lengths,
                                   const int64_t* positions,
                                   const int32_t* rows,
                                   int32_t* tokens,
                                   int32_t* lengths,
                                   int64_t score_width,
                                   int64_t selected_width,
                                   int64_t width,
                                   int64_t token_topk) {
  constexpr int32_t kThreads = 256;
  constexpr int32_t kMaxSelected = 512;
  __shared__ int32_t selected[kMaxSelected];
  __shared__ int32_t selected_count;
  const int64_t row = blockIdx.x;
  const int32_t lane = threadIdx.x;
  const int64_t visible = rows[row] >= 0 ? positions[row] + 1 : 0;
  const int64_t tail = visible % 4;
  const int64_t end = visible <= token_topk ? visible : token_topk + tail;
  const int32_t pool_length = max(pool_lengths[row], 0);
  const int32_t wanted = static_cast<int32_t>(
      min(selected_width, static_cast<int64_t>(pool_length)));
  if (lane == 0) {
    lengths[row] = static_cast<int32_t>(end);
    selected_count = 0;
  }
  __syncthreads();
  if (visible > token_topk && wanted > 0) {
    const int32_t found = radix_select_pools(
        scores + row * score_width, pool_length, wanted, selected);
    if (lane == 0) {
      selected_count = min(found, wanted);
    }
  }
  __syncthreads();
  for (int64_t col = lane; col < width; col += kThreads) {
    int32_t token = -1;
    if (col < end && visible <= token_topk) {
      token = static_cast<int32_t>(col);
    } else if (visible > token_topk && col < token_topk) {
      const int64_t pool_slot = col / 4;
      if (pool_slot < selected_count) {
        const int32_t pool = selected[pool_slot];
        if (pool >= 0 && pool < pool_length) {
          token = pool * 4 + static_cast<int32_t>(col % 4);
        }
      }
    } else if (col >= token_topk && col < end) {
      token = static_cast<int32_t>((visible / 4) * 4 + col - token_topk);
    }
    tokens[row * width + col] = token;
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
  // Resolve both scoring functions before any persistent state is mutated.
  auto metadata_fn =
      get_function("glm_kpool_metadata_v1", "get_paged_mqa_logits_metadata");
  auto score_fn = get_function("glm_kpool_score_v1", "fp8_paged_mqa_logits");
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
  auto metadata_fn =
      get_function("glm_kpool_metadata_v1", "get_paged_mqa_logits_metadata");
  auto score_fn = get_function("glm_kpool_score_v1", "fp8_paged_mqa_logits");
  const auto stream = current_musa_stream(query.device());
  const int64_t pools = (max_context + 3) / 4;
  const int64_t count = std::min<int64_t>(token_topk / 4, pools);
  torch::Tensor scores;
  if (max_context > token_topk) {
    auto quant = torch::empty({tokens, 1, 32, 128},
                              query.options().dtype(torch::kFloat8_e4m3fn));
    auto scaled_weights = torch::empty_like(weights);
    const int64_t query_rows = tokens * 32;
    quantize_query_grouped<4>
        <<<static_cast<int32_t>((query_rows + 3) / 4), 128, 0, stream>>>(
            query.data_ptr<c10::BFloat16>(),
            weights.data_ptr<float>(),
            quant.data_ptr<c10::Float8_e4m3fn>(),
            scaled_weights.data_ptr<float>(),
            query_rows);
    check_musa_kernel_launch("GLM IndexPool query quantization");
    // At most 16 MiB of FP32 logits per chunk, and <=128 query rows to match
    // the exported scheduler. Main history is never expanded per head.
    const int64_t chunk = std::max<int64_t>(
        1, std::min<int64_t>(128, (16 * 1024 * 1024) / (pools * 4)));
    auto pool_ids = torch::arange(pools, ints).unsqueeze(0);
    for (int64_t first = 0; first < tokens; first += chunk) {
      const int64_t size = std::min<int64_t>(chunk, tokens - first);
      auto lens = pool_lengths.narrow(0, first, size);
      auto table_rows =
          result.rows.narrow(0, first, size).clamp_min(0).to(torch::kInt64);
      auto table = block_table.index_select(0, table_rows).contiguous();
      auto scheduling =
          torch::empty({properties.multiProcessorCount + 1, 2}, ints);
      ffi::Tensor ffi_logits;
      {
        TvmffiStreamGuard guard(query.device());
        metadata_fn(
            to_ffi_tensor(lens), int64_t{64}, to_ffi_tensor(scheduling));
        ffi_logits =
            score_fn(to_ffi_tensor(quant.narrow(0, first, size)),
                     to_ffi_tensor(cache),
                     to_ffi_tensor(scaled_weights.narrow(0, first, size)),
                     to_ffi_tensor(lens),
                     to_ffi_tensor(table),
                     to_ffi_tensor(scheduling),
                     pools,
                     true)
                .cast<ffi::Tensor>();
      }
      auto chunk_scores = own_logits(std::move(ffi_logits), query.options())
                              .narrow(1, 0, pools);
      chunk_scores.masked_fill_(pool_ids.ge(lens.unsqueeze(1)), -INFINITY);
      if (!scores.defined()) {
        scores = torch::empty({tokens, pools},
                              query.options().dtype(torch::kFloat32));
      }
      scores.narrow(0, first, size).copy_(chunk_scores);
    }
  }
  if (!scores.defined()) {
    scores = torch::empty({tokens, std::max<int64_t>(pools, 1)},
                          query.options().dtype(torch::kFloat32));
  }
  CHECK_LE(count, 512);
  select_pool_tokens<<<tokens, 256, 0, stream>>>(
      scores.data_ptr<float>(),
      pool_lengths.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(),
      result.rows.data_ptr<int32_t>(),
      result.tokens.data_ptr<int32_t>(),
      result.lengths.data_ptr<int32_t>(),
      scores.size(1),
      count,
      width,
      token_topk);
  check_musa_kernel_launch("GLM IndexPool fused candidate selection");
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
