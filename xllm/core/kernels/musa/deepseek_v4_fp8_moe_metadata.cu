/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// MUSA-only device metadata for Mate ragged FP8 MoE. This replaces the Torch
// composite chain (cumsum/where/index_copy/nonzero) used by the first bridge.
// Keeping route ids and expert padding in int32 avoids PrivateUse1 integer
// promotion and removes host synchronization from the prefill data path.

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kMetadataBlock = 256;
constexpr int32_t kQuantBlock = 128;
constexpr int32_t kWarpSize = 32;
constexpr float kFp8Max = 448.0f;

constexpr bool metadata_single_cta_enabled() { return false; }

constexpr bool metadata_expert_parallel_enabled() { return true; }

bool metadata_parallel_prefix_enabled() {
  const char* value = std::getenv("XLLM_DSV4_MOE_PARALLEL_PREFIX");
  return value != nullptr && std::string(value) == "1";
}

constexpr int32_t quant_groups_per_cta() { return 1; }

constexpr bool warp_quant_scatter_enabled() { return true; }

constexpr int32_t warp_quant_scatter_warps_per_cta() { return 16; }

constexpr bool warp_swiglu_quant_enabled() { return false; }

bool long_warp_swiglu_quant_enabled(const torch::Tensor& gate_up) {
  // Restrict this opt-in experiment to the measured long TP8 BF16 shape.
  // The original dispatch remains unchanged for short rows and other widths.
  const char* value = std::getenv("XLLM_DSV4_LONG_MOE_WARP_SWIGLU");
  return value != nullptr && std::string(value) == "1" &&
         gate_up.scalar_type() == torch::kBFloat16 &&
         gate_up.size(1) == 512 && gate_up.size(0) >= 4096 * 6 &&
         gate_up.size(0) <= 16384 * 6;
}

constexpr bool restore_combine_2d_enabled() { return true; }

constexpr int64_t restore_combine_2d_min_tokens() { return 2; }

constexpr int32_t restore_combine_2d_block_h() { return 512; }

constexpr int32_t activation_quant_groups_per_cta() { return 1; }

constexpr int32_t restore_combine_threads() { return kMetadataBlock; }

void check_last_error(const char* stage) {
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 FP8 MoE " << stage
      << " kernel failed: " << musaGetErrorString(status);
}

__global__ void build_padded_metadata_kernel(
    const int32_t* __restrict__ sorted_assignment,
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ padded_assignment,
    int32_t* __restrict__ m_indices,
    int32_t* __restrict__ route_to_padded,
    int64_t padded_m,
    int32_t num_experts,
    int32_t alignment_m) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= padded_m) {
    return;
  }

  int32_t route = -1;
  int32_t expert = -1;
  int32_t source_offset = 0;
  int32_t padded_offset = 0;
  for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    const int32_t count = expert_counts[expert_id];
    const int32_t padded_count = (count + alignment_m - 1) & ~(alignment_m - 1);
    const int64_t row_in_expert = row - padded_offset;
    if (row_in_expert >= 0 && row_in_expert < count) {
      route = sorted_assignment[source_offset + row_in_expert];
      expert = expert_id;
      break;
    }
    if (row_in_expert >= count && row_in_expert < padded_count) {
      break;
    }
    source_offset += count;
    padded_offset += padded_count;
  }
  padded_assignment[row] = route;
  m_indices[row] = expert;
  if (route_to_padded != nullptr && route >= 0) {
    route_to_padded[route] = static_cast<int32_t>(row);
  }
}

// The row-parallel kernel above makes every padded row rescan all expert
// counts.  For the DeepSeek shape that is roughly 36K independent 256-entry
// scans per layer.  One CTA can instead build both prefix arrays once and let
// one thread per expert fill its own contiguous interval.  Route order,
// padding values and route_to_padded are bit-identical to the original path.
__global__ void build_padded_metadata_single_cta_kernel(
    const int32_t* __restrict__ sorted_assignment,
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ padded_assignment,
    int32_t* __restrict__ m_indices,
    int32_t* __restrict__ route_to_padded,
    int64_t padded_m,
    int32_t num_experts,
    int32_t alignment_m) {
  __shared__ int32_t source_offsets[kMetadataBlock];
  __shared__ int32_t padded_offsets[kMetadataBlock];
  __shared__ int32_t total_padded;

  if (threadIdx.x == 0) {
    int32_t source_offset = 0;
    int32_t padded_offset = 0;
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      source_offsets[expert] = source_offset;
      padded_offsets[expert] = padded_offset;
      const int32_t count = expert_counts[expert];
      const int32_t padded_count =
          (count + alignment_m - 1) & ~(alignment_m - 1);
      source_offset += count;
      padded_offset += padded_count;
    }
    total_padded = padded_offset;
  }
  __syncthreads();

  const int32_t expert = threadIdx.x;
  if (expert < num_experts) {
    const int32_t count = expert_counts[expert];
    const int32_t padded_count = (count + alignment_m - 1) & ~(alignment_m - 1);
    const int32_t source_offset = source_offsets[expert];
    const int32_t padded_offset = padded_offsets[expert];
    for (int32_t row_in_expert = 0; row_in_expert < padded_count;
         ++row_in_expert) {
      const int64_t row = static_cast<int64_t>(padded_offset) + row_in_expert;
      if (row >= padded_m) {
        break;
      }
      int32_t route = -1;
      int32_t row_expert = -1;
      if (row_in_expert < count) {
        route = sorted_assignment[source_offset + row_in_expert];
        row_expert = expert;
      }
      padded_assignment[row] = route;
      m_indices[row] = row_expert;
      if (route_to_padded != nullptr && route >= 0) {
        route_to_padded[route] = static_cast<int32_t>(row);
      }
    }
  }

  for (int64_t row = static_cast<int64_t>(total_padded) + threadIdx.x;
       row < padded_m;
       row += blockDim.x) {
    padded_assignment[row] = -1;
    m_indices[row] = -1;
  }
}

// Use one CTA per expert.  Only thread zero scans the preceding expert counts,
// then all threads cooperatively fill that expert's interval.  Compared with
// one scan per padded row this reduces count loads by roughly two orders of
// magnitude without putting the whole layer behind one long-running CTA.
__global__ void build_padded_metadata_expert_parallel_kernel(
    const int32_t* __restrict__ sorted_assignment,
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ padded_assignment,
    int32_t* __restrict__ m_indices,
    int32_t* __restrict__ route_to_padded,
    int64_t padded_m,
    int32_t num_experts,
    int32_t alignment_m) {
  const int32_t expert = blockIdx.x;
  if (expert >= num_experts) {
    return;
  }

  __shared__ int32_t source_offset;
  __shared__ int32_t padded_offset;
  __shared__ int32_t count;
  __shared__ int32_t padded_count;
  if (threadIdx.x == 0) {
    int32_t source = 0;
    int32_t padded = 0;
    for (int32_t preceding = 0; preceding < expert; ++preceding) {
      const int32_t preceding_count = expert_counts[preceding];
      source += preceding_count;
      padded += (preceding_count + alignment_m - 1) & ~(alignment_m - 1);
    }
    const int32_t current_count = expert_counts[expert];
    source_offset = source;
    padded_offset = padded;
    count = current_count;
    padded_count = (current_count + alignment_m - 1) & ~(alignment_m - 1);
  }
  __syncthreads();

  for (int32_t row_in_expert = threadIdx.x; row_in_expert < padded_count;
       row_in_expert += blockDim.x) {
    const int64_t row = static_cast<int64_t>(padded_offset) + row_in_expert;
    if (row >= padded_m) {
      break;
    }
    int32_t route = -1;
    int32_t row_expert = -1;
    if (row_in_expert < count) {
      route = sorted_assignment[source_offset + row_in_expert];
      row_expert = expert;
    }
    padded_assignment[row] = route;
    m_indices[row] = row_expert;
    if (route_to_padded != nullptr && route >= 0) {
      route_to_padded[route] = static_cast<int32_t>(row);
    }
  }

  if (expert == num_experts - 1) {
    const int64_t total_padded =
        static_cast<int64_t>(padded_offset) + padded_count;
    for (int64_t row = total_padded + threadIdx.x; row < padded_m;
         row += blockDim.x) {
      padded_assignment[row] = -1;
      m_indices[row] = -1;
    }
  }
}

// The DSV4/EP1 family has256 experts. Each CTA computes the exact integer
// prefix in parallel, and every CTA participates in clearing the disjoint
// tail. No global barrier or floating-point operation is introduced.
__global__ void build_padded_metadata_parallel_prefix_kernel(
    const int32_t* __restrict__ sorted_assignment,
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ padded_assignment,
    int32_t* __restrict__ m_indices,
    int32_t* __restrict__ route_to_padded,
    int64_t padded_m,
    int32_t alignment_m) {
  constexpr int32_t kExperts = 256;
  constexpr int32_t kWarps = kExperts / kWarpSize;
  const int32_t expert = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid % kWarpSize;
  const int32_t warp = tid / kWarpSize;
  const int32_t count_i = expert_counts[tid];
  const int32_t padded_i = (count_i + alignment_m - 1) & ~(alignment_m - 1);
  int32_t source = tid < expert ? count_i : 0;
  int32_t prefix = tid < expert ? padded_i : 0;
  int32_t total = padded_i;
#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    source += __shfl_down_sync(0xffffffffU, source, offset);
    prefix += __shfl_down_sync(0xffffffffU, prefix, offset);
    total += __shfl_down_sync(0xffffffffU, total, offset);
  }
  __shared__ int32_t partial_source[kWarps];
  __shared__ int32_t partial_prefix[kWarps];
  __shared__ int32_t partial_total[kWarps];
  __shared__ int32_t source_offset;
  __shared__ int32_t padded_offset;
  __shared__ int32_t total_padded;
  __shared__ int32_t count;
  __shared__ int32_t padded_count;
  if (lane == 0) {
    partial_source[warp] = source;
    partial_prefix[warp] = prefix;
    partial_total[warp] = total;
  }
  __syncthreads();
  if (tid == 0) {
    int32_t source_sum = 0;
    int32_t prefix_sum = 0;
    int32_t total_sum = 0;
#pragma unroll
    for (int32_t index = 0; index < kWarps; ++index) {
      source_sum += partial_source[index];
      prefix_sum += partial_prefix[index];
      total_sum += partial_total[index];
    }
    source_offset = source_sum;
    padded_offset = prefix_sum;
    total_padded = total_sum;
    count = expert_counts[expert];
    padded_count = (count + alignment_m - 1) & ~(alignment_m - 1);
  }
  __syncthreads();
  for (int32_t index = tid; index < padded_count; index += kExperts) {
    const int64_t row = static_cast<int64_t>(padded_offset) + index;
    if (row >= padded_m) {
      break;
    }
    const int32_t route = index < count ? sorted_assignment[source_offset + index] : -1;
    padded_assignment[row] = route;
    m_indices[row] = index < count ? expert : -1;
    if (route_to_padded != nullptr && route >= 0) {
      route_to_padded[route] = static_cast<int32_t>(row);
    }
  }
  // All writes here are >=total_padded, and all expert intervals end before
  // that boundary. Therefore tail clearing cannot race an expert's publish.
  for (int64_t row = static_cast<int64_t>(total_padded) +
                     static_cast<int64_t>(expert) * kExperts + tid;
       row < padded_m;
       row += static_cast<int64_t>(kExperts) * kExperts) {
    padded_assignment[row] = -1;
    m_indices[row] = -1;
  }
}

void launch_build_padded_metadata(const int32_t* sorted_assignment,
                                  const int32_t* expert_counts,
                                  int32_t* padded_assignment,
                                  int32_t* m_indices,
                                  int32_t* route_to_padded,
                                  int64_t padded_m,
                                  int32_t num_experts,
                                  int32_t alignment_m,
                                  musaStream_t stream) {
  if (metadata_parallel_prefix_enabled() && num_experts == 256) {
    build_padded_metadata_parallel_prefix_kernel<<<256, 256, 0, stream>>>(
        sorted_assignment, expert_counts, padded_assignment, m_indices,
        route_to_padded, padded_m, alignment_m);
    LOG_FIRST_N(INFO, 1) << "[DSV4_MOE_PARALLEL_PREFIX] experts=256 padded_m="
                        << padded_m << " alignment=" << alignment_m
                        << " integer_only=1 distributed_tail=1";
    return;
  }
  if (metadata_expert_parallel_enabled() && num_experts > 0) {
    build_padded_metadata_expert_parallel_kernel<<<num_experts,
                                                   6 * kWarpSize,
                                                   0,
                                                   stream>>>(sorted_assignment,
                                                             expert_counts,
                                                             padded_assignment,
                                                             m_indices,
                                                             route_to_padded,
                                                             padded_m,
                                                             num_experts,
                                                             alignment_m);
    return;
  }
  if (metadata_single_cta_enabled() && num_experts <= kMetadataBlock) {
    build_padded_metadata_single_cta_kernel<<<1, kMetadataBlock, 0, stream>>>(
        sorted_assignment,
        expert_counts,
        padded_assignment,
        m_indices,
        route_to_padded,
        padded_m,
        num_experts,
        alignment_m);
    return;
  }
  const int64_t metadata_grid =
      (padded_m + kMetadataBlock - 1) / kMetadataBlock;
  build_padded_metadata_kernel<<<metadata_grid, kMetadataBlock, 0, stream>>>(
      sorted_assignment,
      expert_counts,
      padded_assignment,
      m_indices,
      route_to_padded,
      padded_m,
      num_experts,
      alignment_m);
}

template <typename scalar_t>
__global__ void gather_padded_input_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ padded_assignment,
    scalar_t* __restrict__ sorted_input,
    int64_t padded_m,
    int64_t hidden_size,
    int32_t topk) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = padded_m * hidden_size;
  if (linear >= total) {
    return;
  }
  const int64_t row = linear / hidden_size;
  const int64_t column = linear - row * hidden_size;
  const int32_t route = padded_assignment[row];
  scalar_t value = static_cast<scalar_t>(0.0f);
  if (route >= 0) {
    const int64_t token = route / topk;
    value = input[token * hidden_size + column];
  }
  sorted_input[linear] = value;
}

template <typename scalar_t>
__global__ void token_blockwise_fp8_scatter_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ route_to_padded,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t blocks_per_row,
    int32_t topk) {
  const int64_t token = blockIdx.x;
  const int64_t block = blockIdx.y;
  if (token >= num_tokens || block >= blocks_per_row) {
    return;
  }

  const int32_t tid = threadIdx.x;
  const int64_t column = block * kQuantBlock + tid;
  const float value =
      column < hidden_size
          ? static_cast<float>(input[token * hidden_size + column])
          : 0.0f;
  float max_abs = fabsf(value);

#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }

  __shared__ float warp_max[kQuantBlock / kWarpSize];
  __shared__ float block_scale;
  const int32_t lane = tid % kWarpSize;
  const int32_t warp = tid / kWarpSize;
  if (lane == 0) {
    warp_max[warp] = max_abs;
  }
  __syncthreads();

  if (warp == 0) {
    max_abs = lane < kQuantBlock / kWarpSize ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
      const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
      block_scale = exp2f(ceilf(log2f(raw_scale)));
    }
  }
  __syncthreads();

  const auto quantized = scaled_fp8_conversion<true, c10::Float8_e4m3fn>(
      value, 1.0f / block_scale);
  const int64_t route_base = token * topk;
  for (int32_t route_slot = 0; route_slot < topk; ++route_slot) {
    const int32_t dst = route_to_padded[route_base + route_slot];
    if (dst < 0) {
      continue;
    }
    if (tid == 0) {
      scales[static_cast<int64_t>(dst) * blocks_per_row + block] = block_scale;
    }
    if (column < hidden_size) {
      output[static_cast<int64_t>(dst) * hidden_size + column] = quantized;
    }
  }
}

// Preserve the exact 128-value quantization groups and reduction order while
// amortizing CTA scheduling over several independent groups.  The original
// prefill shape launches 2500 * 32 tiny CTAs per MoE layer; grouping four of
// them cuts that to 20,000 without changing any scale or FP8 value.
template <typename scalar_t, int32_t kGroupsPerCta>
__global__ void token_blockwise_fp8_scatter_grouped_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ route_to_padded,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t blocks_per_row,
    int32_t topk) {
  const int64_t token = blockIdx.x;
  const int32_t group = threadIdx.x / kQuantBlock;
  const int32_t tid = threadIdx.x - group * kQuantBlock;
  const int64_t quant_block =
      static_cast<int64_t>(blockIdx.y) * kGroupsPerCta + group;
  const bool active = token < num_tokens && quant_block < blocks_per_row;
  const int64_t column = quant_block * kQuantBlock + tid;
  const float value =
      active && column < hidden_size
          ? static_cast<float>(input[token * hidden_size + column])
          : 0.0f;
  float max_abs = fabsf(value);

#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }

  constexpr int32_t kWarpsPerGroup = kQuantBlock / kWarpSize;
  __shared__ float warp_max[kGroupsPerCta][kWarpsPerGroup];
  __shared__ float block_scale[kGroupsPerCta];
  const int32_t lane = tid % kWarpSize;
  const int32_t warp_in_group = tid / kWarpSize;
  if (lane == 0) {
    warp_max[group][warp_in_group] = max_abs;
  }
  __syncthreads();

  if (warp_in_group == 0) {
    max_abs = lane < kWarpsPerGroup ? warp_max[group][lane] : 0.0f;
#pragma unroll
    for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
      const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
      block_scale[group] = exp2f(ceilf(log2f(raw_scale)));
    }
  }
  __syncthreads();

  if (!active) {
    return;
  }
  const auto quantized = scaled_fp8_conversion<true, c10::Float8_e4m3fn>(
      value, 1.0f / block_scale[group]);
  const int64_t route_base = token * topk;
  for (int32_t route_slot = 0; route_slot < topk; ++route_slot) {
    const int32_t dst = route_to_padded[route_base + route_slot];
    if (dst < 0) {
      continue;
    }
    if (tid == 0) {
      scales[static_cast<int64_t>(dst) * blocks_per_row + quant_block] =
          block_scale[group];
    }
    if (column < hidden_size) {
      output[static_cast<int64_t>(dst) * hidden_size + column] = quantized;
    }
  }
}

// SGLang's production prefill scatter assigns one 128-value quantization
// group to a warp: every lane loads/converts four adjacent values, the warp
// computes one scale, and the four FP8 bytes are emitted with one aligned
// 32-bit store for each selected route.  The established xLLM kernel assigns
// a full 128-thread CTA to the same group, which creates 80K tiny CTAs for a
// 2500x4096 input.  Keep xLLM's exact power-of-two scale rule and route map,
// but use the SGLang launch geometry behind an opt-in gate.
template <typename scalar_t, int32_t kWarpsPerCta>
__global__ void token_blockwise_fp8_scatter_warp_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ route_to_padded,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t blocks_per_row,
    int32_t topk) {
  constexpr int32_t kValuesPerLane = kQuantBlock / kWarpSize;
  static_assert(kValuesPerLane == 4);
  static_assert(sizeof(c10::Float8_e4m3fn) == 1);

  const int32_t warp = threadIdx.x / kWarpSize;
  const int32_t lane = threadIdx.x % kWarpSize;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * kWarpsPerCta + warp;
  const int64_t total_groups = num_tokens * blocks_per_row;
  if (group >= total_groups) {
    return;
  }
  const int64_t token = group / blocks_per_row;
  const int64_t quant_block = group - token * blocks_per_row;
  const int64_t column_base = quant_block * kQuantBlock + lane * kValuesPerLane;

  float values[kValuesPerLane];
  float max_abs = 0.0f;
#pragma unroll
  for (int32_t value_id = 0; value_id < kValuesPerLane; ++value_id) {
    const int64_t column = column_base + value_id;
    values[value_id] =
        column < hidden_size
            ? static_cast<float>(input[token * hidden_size + column])
            : 0.0f;
    max_abs = fmaxf(max_abs, fabsf(values[value_id]));
  }
#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }
  max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
  const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
  const float scale = exp2f(ceilf(log2f(raw_scale)));
  const float inverse_scale = 1.0f / scale;

  uint32_t packed = 0;
#pragma unroll
  for (int32_t value_id = 0; value_id < kValuesPerLane; ++value_id) {
    const auto quantized = scaled_fp8_conversion<true, c10::Float8_e4m3fn>(
        values[value_id], inverse_scale);
    packed |= static_cast<uint32_t>(quantized.x) << (value_id * 8);
  }

  const int64_t route_base = token * topk;
#pragma unroll
  for (int32_t route_slot = 0; route_slot < topk; ++route_slot) {
    const int32_t dst = route_to_padded[route_base + route_slot];
    if (dst < 0) {
      continue;
    }
    if (lane == 0) {
      scales[static_cast<int64_t>(dst) * blocks_per_row + quant_block] = scale;
    }
    if (column_base + kValuesPerLane <= hidden_size) {
      auto* output_bytes = reinterpret_cast<uint8_t*>(output) +
                           static_cast<int64_t>(dst) * hidden_size +
                           column_base;
      *reinterpret_cast<uint32_t*>(output_bytes) = packed;
    }
  }
}

__global__ void prequantized_fp8_scatter_kernel(
    const c10::Float8_e4m3fn* __restrict__ input,
    const float* __restrict__ input_scales,
    const int32_t* __restrict__ route_to_padded,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ output_scales,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t blocks_per_row,
    int32_t topk) {
  const int64_t token = blockIdx.x;
  const int64_t block = blockIdx.y;
  if (token >= num_tokens || block >= blocks_per_row) {
    return;
  }
  const int32_t tid = threadIdx.x;
  const int64_t column = block * kQuantBlock + tid;
  const c10::Float8_e4m3fn value = input[token * hidden_size + column];
  const float scale = input_scales[token * blocks_per_row + block];
  const int64_t route_base = token * topk;
  for (int32_t route_slot = 0; route_slot < topk; ++route_slot) {
    const int32_t dst = route_to_padded[route_base + route_slot];
    if (dst < 0) {
      continue;
    }
    output[static_cast<int64_t>(dst) * hidden_size + column] = value;
    if (tid == 0) {
      output_scales[static_cast<int64_t>(dst) * blocks_per_row + block] = scale;
    }
  }
}

template <typename scalar_t>
__global__ void masked_decode_quant_scatter_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ local_ids,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int32_t* __restrict__ masked_tokens,
    int64_t hidden_size,
    int64_t blocks_per_row,
    int32_t topk,
    int32_t num_experts) {
  const int64_t block = blockIdx.x;
  if (block >= blocks_per_row) {
    return;
  }

  const int32_t tid = threadIdx.x;
  const int64_t column = block * kQuantBlock + tid;
  const float value =
      column < hidden_size ? static_cast<float>(input[column]) : 0.0f;
  float max_abs = fabsf(value);

#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }

  __shared__ float warp_max[kQuantBlock / kWarpSize];
  __shared__ float block_scale;
  const int32_t lane = tid % kWarpSize;
  const int32_t warp = tid / kWarpSize;
  if (lane == 0) {
    warp_max[warp] = max_abs;
  }
  __syncthreads();

  if (warp == 0) {
    max_abs = lane < kQuantBlock / kWarpSize ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
      const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
      block_scale = exp2f(ceilf(log2f(raw_scale)));
    }
  }
  __syncthreads();

  const auto quantized = scaled_fp8_conversion<true, c10::Float8_e4m3fn>(
      value, 1.0f / block_scale);
  for (int32_t route_slot = 0; route_slot < topk; ++route_slot) {
    const int32_t expert = local_ids[route_slot];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
    if (tid == 0) {
      scales[static_cast<int64_t>(expert) * blocks_per_row + block] =
          block_scale;
      if (block == 0) {
        atomicExch(masked_tokens + expert, 1);
      }
    }
    if (column < hidden_size) {
      output[static_cast<int64_t>(expert) * hidden_size + column] = quantized;
    }
  }
}

struct QuantPrepareWorkspaceKey {
  int32_t device = -1;
  int32_t dtype = 0;
  int64_t hidden_size = 0;

  bool operator==(const QuantPrepareWorkspaceKey& other) const {
    return device == other.device && dtype == other.dtype &&
           hidden_size == other.hidden_size;
  }
};

struct QuantPrepareWorkspaceKeyHash {
  size_t operator()(const QuantPrepareWorkspaceKey& key) const {
    size_t seed = static_cast<size_t>(key.device + 1);
    seed ^= static_cast<size_t>(key.dtype) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
    seed ^= static_cast<size_t>(key.hidden_size) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct QuantPrepareWorkspace {
  int64_t capacity_rows = 0;
  int64_t capacity_routes = 0;
  int64_t capacity_experts = 0;
  torch::Tensor input_fp8;
  torch::Tensor input_scale;
  torch::Tensor padded_assignment;
  torch::Tensor m_indices;
  torch::Tensor route_to_padded;
  torch::Tensor masked_tokens;
};

struct SwigluQuantWorkspaceKey {
  int32_t device = -1;
  int32_t dtype = 0;
  int64_t hidden_size = 0;

  bool operator==(const SwigluQuantWorkspaceKey& other) const {
    return device == other.device && dtype == other.dtype &&
           hidden_size == other.hidden_size;
  }
};

struct SwigluQuantWorkspaceKeyHash {
  size_t operator()(const SwigluQuantWorkspaceKey& key) const {
    size_t seed = static_cast<size_t>(key.device + 1);
    seed ^= static_cast<size_t>(key.dtype) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
    seed ^= static_cast<size_t>(key.hidden_size) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct SwigluQuantWorkspace {
  int64_t capacity_rows = 0;
  torch::Tensor output;
  torch::Tensor scale;
};

QuantPrepareWorkspace& get_quant_prepare_workspace(const torch::Tensor& input,
                                                   int64_t rows,
                                                   int64_t routes,
                                                   int64_t experts = 0) {
  static thread_local std::unordered_map<QuantPrepareWorkspaceKey,
                                         QuantPrepareWorkspace,
                                         QuantPrepareWorkspaceKeyHash>
      workspaces;
  const QuantPrepareWorkspaceKey key{input.device().index(),
                                     static_cast<int32_t>(input.scalar_type()),
                                     input.size(1)};
  auto& workspace = workspaces[key];
  if (workspace.capacity_rows < rows || !workspace.input_fp8.defined()) {
    const int64_t capacity = std::max<int64_t>(
        rows, workspace.capacity_rows > 0 ? workspace.capacity_rows * 2 : 1);
    const int64_t scale_cols = (input.size(1) + kQuantBlock - 1) / kQuantBlock;
    workspace.capacity_rows = capacity;
    workspace.input_fp8 =
        torch::empty({capacity, input.size(1)},
                     input.options().dtype(torch::kFloat8_e4m3fn));
    workspace.input_scale = torch::empty(
        {capacity, scale_cols}, input.options().dtype(torch::kFloat32));
    workspace.padded_assignment =
        torch::empty({capacity}, input.options().dtype(torch::kInt32));
    workspace.m_indices =
        torch::empty({capacity}, input.options().dtype(torch::kInt32));
  }
  if (workspace.capacity_routes < routes ||
      !workspace.route_to_padded.defined()) {
    const int64_t capacity = std::max<int64_t>(
        routes,
        workspace.capacity_routes > 0 ? workspace.capacity_routes * 2 : 1);
    workspace.capacity_routes = capacity;
    workspace.route_to_padded =
        torch::empty({capacity}, input.options().dtype(torch::kInt32));
  }
  if (workspace.capacity_experts < experts ||
      (experts > 0 && !workspace.masked_tokens.defined())) {
    const int64_t capacity = std::max<int64_t>(
        experts,
        workspace.capacity_experts > 0 ? workspace.capacity_experts * 2 : 1);
    workspace.capacity_experts = capacity;
    workspace.masked_tokens =
        torch::empty({capacity}, input.options().dtype(torch::kInt32));
  }
  return workspace;
}

SwigluQuantWorkspace& get_swiglu_quant_workspace(const torch::Tensor& gate_up,
                                                 int64_t rows,
                                                 int64_t hidden_size,
                                                 int64_t scale_cols) {
  static thread_local std::unordered_map<SwigluQuantWorkspaceKey,
                                         SwigluQuantWorkspace,
                                         SwigluQuantWorkspaceKeyHash>
      workspaces;
  const SwigluQuantWorkspaceKey key{gate_up.device().index(),
                                    static_cast<int32_t>(gate_up.scalar_type()),
                                    hidden_size};
  auto& workspace = workspaces[key];
  if (workspace.capacity_rows < rows || !workspace.output.defined() ||
      !workspace.scale.defined()) {
    const int64_t capacity = std::max<int64_t>(
        rows, workspace.capacity_rows > 0 ? workspace.capacity_rows * 2 : 1);
    workspace.capacity_rows = capacity;
    workspace.output =
        torch::empty({capacity, hidden_size},
                     gate_up.options().dtype(torch::kFloat8_e4m3fn));
    workspace.scale = torch::empty({capacity, scale_cols},
                                   gate_up.options().dtype(torch::kFloat32));
  }
  return workspace;
}

template <typename scalar_t>
__global__ void swiglu_blockwise_fp8_quant_kernel(
    const scalar_t* __restrict__ gate_up,
    const int32_t* __restrict__ ragged_tokens_info,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t hidden_size,
    int64_t blocks_per_row,
    float swiglu_limit) {
  const int64_t row = blockIdx.x;
  const int64_t block = blockIdx.y;
  if (row >= rows || block >= blocks_per_row || ragged_tokens_info[row] < 0) {
    return;
  }

  const int32_t tid = threadIdx.x;
  const int64_t column = block * kQuantBlock + tid;
  float value = 0.0f;
  if (column < hidden_size) {
    const int64_t input_offset = row * hidden_size * 2 + column;
    float gate = static_cast<float>(gate_up[input_offset]);
    float up = static_cast<float>(gate_up[input_offset + hidden_size]);
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    // Match the existing split path: active() stores BF16/FP16 before the
    // blockwise quantizer reads the value back as FP32.
    const scalar_t rounded =
        static_cast<scalar_t>(gate / (1.0f + expf(-gate)) * up);
    value = static_cast<float>(rounded);
  }
  float max_abs = fabsf(value);

#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }

  __shared__ float warp_max[kQuantBlock / kWarpSize];
  __shared__ float block_scale;
  const int32_t lane = tid % kWarpSize;
  const int32_t warp = tid / kWarpSize;
  if (lane == 0) {
    warp_max[warp] = max_abs;
  }
  __syncthreads();

  if (warp == 0) {
    max_abs = lane < kQuantBlock / kWarpSize ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
      const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
      block_scale = exp2f(ceilf(log2f(raw_scale)));
      scales[row * blocks_per_row + block] = block_scale;
    }
  }
  __syncthreads();

  if (column < hidden_size) {
    output[row * hidden_size + column] =
        scaled_fp8_conversion<true, c10::Float8_e4m3fn>(value,
                                                        1.0f / block_scale);
  }
}

template <typename scalar_t, int32_t kGroupsPerCta>
__global__ void swiglu_blockwise_fp8_quant_grouped_kernel(
    const scalar_t* __restrict__ gate_up,
    const int32_t* __restrict__ ragged_tokens_info,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t hidden_size,
    int64_t blocks_per_row,
    float swiglu_limit) {
  const int64_t row = blockIdx.x;
  const int32_t group = threadIdx.x / kQuantBlock;
  const int32_t tid = threadIdx.x - group * kQuantBlock;
  const int64_t quant_block =
      static_cast<int64_t>(blockIdx.y) * kGroupsPerCta + group;
  const bool active = row < rows && quant_block < blocks_per_row &&
                      ragged_tokens_info[row] >= 0;
  const int64_t column = quant_block * kQuantBlock + tid;
  float value = 0.0f;
  if (active && column < hidden_size) {
    const int64_t input_offset = row * hidden_size * 2 + column;
    float gate = static_cast<float>(gate_up[input_offset]);
    float up = static_cast<float>(gate_up[input_offset + hidden_size]);
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const scalar_t rounded =
        static_cast<scalar_t>(gate / (1.0f + expf(-gate)) * up);
    value = static_cast<float>(rounded);
  }
  float max_abs = fabsf(value);

#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }

  constexpr int32_t kWarpsPerGroup = kQuantBlock / kWarpSize;
  __shared__ float warp_max[kGroupsPerCta][kWarpsPerGroup];
  __shared__ float block_scale[kGroupsPerCta];
  const int32_t lane = tid % kWarpSize;
  const int32_t warp_in_group = tid / kWarpSize;
  if (lane == 0) {
    warp_max[group][warp_in_group] = max_abs;
  }
  __syncthreads();
  if (warp_in_group == 0) {
    max_abs = lane < kWarpsPerGroup ? warp_max[group][lane] : 0.0f;
#pragma unroll
    for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
      const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
      block_scale[group] = exp2f(ceilf(log2f(raw_scale)));
      if (active) {
        scales[row * blocks_per_row + quant_block] = block_scale[group];
      }
    }
  }
  __syncthreads();
  if (active && column < hidden_size) {
    output[row * hidden_size + column] =
        scaled_fp8_conversion<true, c10::Float8_e4m3fn>(
            value, 1.0f / block_scale[group]);
  }
}

// Mirror the production SGLang scatter geometry for the activation side of
// the MoE: one warp owns one 128-value quant group and each lane processes four
// adjacent values.  This removes tens of thousands of tiny four-warp CTAs at
// the 36K-padded-row prefill shape.  The BF16/FP16 rounding point and xLLM's
// power-of-two scale rule remain identical to the established kernel.
template <typename scalar_t, int32_t kWarpsPerCta>
__global__ void swiglu_blockwise_fp8_quant_warp_kernel(
    const scalar_t* __restrict__ gate_up,
    const int32_t* __restrict__ ragged_tokens_info,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t hidden_size,
    int64_t blocks_per_row,
    float swiglu_limit) {
  constexpr int32_t kValuesPerLane = kQuantBlock / kWarpSize;
  static_assert(kValuesPerLane == 4);
  static_assert(sizeof(c10::Float8_e4m3fn) == 1);

  const int32_t warp = threadIdx.x / kWarpSize;
  const int32_t lane = threadIdx.x % kWarpSize;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * kWarpsPerCta + warp;
  const int64_t total_groups = rows * blocks_per_row;
  if (group >= total_groups) {
    return;
  }
  const int64_t row = group / blocks_per_row;
  const int64_t quant_block = group - row * blocks_per_row;
  if (ragged_tokens_info[row] < 0) {
    return;
  }
  const int64_t column_base = quant_block * kQuantBlock + lane * kValuesPerLane;

  float values[kValuesPerLane];
  float max_abs = 0.0f;
#pragma unroll
  for (int32_t value_id = 0; value_id < kValuesPerLane; ++value_id) {
    const int64_t column = column_base + value_id;
    float value = 0.0f;
    if (column < hidden_size) {
      const int64_t input_offset = row * hidden_size * 2 + column;
      float gate = static_cast<float>(gate_up[input_offset]);
      float up = static_cast<float>(gate_up[input_offset + hidden_size]);
      if (swiglu_limit > 0.0f) {
        gate = fminf(gate, swiglu_limit);
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
      }
      const scalar_t rounded =
          static_cast<scalar_t>(gate / (1.0f + expf(-gate)) * up);
      value = static_cast<float>(rounded);
    }
    values[value_id] = value;
    max_abs = fmaxf(max_abs, fabsf(value));
  }
#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }
  max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
  const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
  const float scale = exp2f(ceilf(log2f(raw_scale)));
  const float inverse_scale = 1.0f / scale;
  if (lane == 0) {
    scales[row * blocks_per_row + quant_block] = scale;
  }

  uint32_t packed = 0;
#pragma unroll
  for (int32_t value_id = 0; value_id < kValuesPerLane; ++value_id) {
    const auto quantized = scaled_fp8_conversion<true, c10::Float8_e4m3fn>(
        values[value_id], inverse_scale);
    packed |= static_cast<uint32_t>(quantized.x) << (value_id * 8);
  }
  if (column_base + kValuesPerLane <= hidden_size) {
    auto* output_bytes =
        reinterpret_cast<uint8_t*>(output) + row * hidden_size + column_base;
    *reinterpret_cast<uint32_t*>(output_bytes) = packed;
  }
}

template <typename scalar_t>
__global__ void restore_padded_output_kernel(
    const scalar_t* __restrict__ sorted_output,
    const int32_t* __restrict__ padded_assignment,
    scalar_t* __restrict__ output,
    int64_t padded_m,
    int64_t hidden_size,
    int64_t num_assignments) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = padded_m * hidden_size;
  if (linear >= total) {
    return;
  }
  const int64_t row = linear / hidden_size;
  const int64_t column = linear - row * hidden_size;
  const int32_t route = padded_assignment[row];
  if (route >= 0 && route < num_assignments) {
    output[static_cast<int64_t>(route) * hidden_size + column] =
        sorted_output[linear];
  }
}

template <typename scalar_t, typename weight_t>
__global__ void restore_combine_kernel(
    const scalar_t* __restrict__ sorted_output,
    const int32_t* __restrict__ route_to_padded,
    const weight_t* __restrict__ topk_weights,
    scalar_t* __restrict__ output,
    int64_t num_tokens,
    int64_t hidden_size,
    int32_t topk) {
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  for (int64_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int32_t slot = 0; slot < topk; ++slot) {
      const int64_t route = token * topk + slot;
      const int32_t row = route_to_padded[route];
      if (row >= 0) {
        value += static_cast<float>(topk_weights[route]) *
                 static_cast<float>(
                     sorted_output[static_cast<int64_t>(row) * hidden_size +
                                   column]);
      }
    }
    output[token * hidden_size + column] = static_cast<scalar_t>(value);
  }
}

// SGLang's cached post-combine divides the hidden dimension into 1024-value
// tiles and caches the six route rows/weights once per CTA.  The old xLLM
// kernel launches one CTA per token and makes each thread walk 16 columns,
// leaving little parallelism at BS1 prefill.  Keep the same accumulation order
// but expose four independent hidden tiles for the DeepSeek H=4096 shape.
template <typename scalar_t,
          typename weight_t,
          int32_t kBlockH,
          int32_t kMaxTopK>
__global__ void restore_combine_2d_kernel(
    const scalar_t* __restrict__ sorted_output,
    const int32_t* __restrict__ route_to_padded,
    const weight_t* __restrict__ topk_weights,
    scalar_t* __restrict__ output,
    int64_t num_tokens,
    int64_t hidden_size,
    int32_t topk) {
  constexpr int32_t kValuesPerThread = kBlockH / kMetadataBlock;
  static_assert(kValuesPerThread == 2 || kValuesPerThread == 4 ||
                kValuesPerThread == 8);
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }

  __shared__ int32_t route_rows[kMaxTopK];
  __shared__ float route_weights[kMaxTopK];
  if (threadIdx.x < topk) {
    const int64_t route = token * topk + threadIdx.x;
    route_rows[threadIdx.x] = route_to_padded[route];
    route_weights[threadIdx.x] = static_cast<float>(topk_weights[route]);
  }
  __syncthreads();

  const int64_t column_base = static_cast<int64_t>(blockIdx.y) * kBlockH +
                              threadIdx.x * kValuesPerThread;
  float values[kValuesPerThread] = {};
#pragma unroll
  for (int32_t slot = 0; slot < topk; ++slot) {
    const int32_t row = route_rows[slot];
    if (row < 0) {
      continue;
    }
    const float weight = route_weights[slot];
#pragma unroll
    for (int32_t value_id = 0; value_id < kValuesPerThread; ++value_id) {
      const int64_t column = column_base + value_id;
      if (column < hidden_size) {
        values[value_id] +=
            weight * static_cast<float>(
                         sorted_output[static_cast<int64_t>(row) * hidden_size +
                                       column]);
      }
    }
  }
#pragma unroll
  for (int32_t value_id = 0; value_id < kValuesPerThread; ++value_id) {
    const int64_t column = column_base + value_id;
    if (column < hidden_size) {
      output[token * hidden_size + column] =
          static_cast<scalar_t>(values[value_id]);
    }
  }
}

template <typename Scalar, typename Weight>
__global__ void restore_combine_vector8_kernel(
    const Scalar* __restrict__ sorted_output,
    const int32_t* __restrict__ route_to_padded,
    const Weight* __restrict__ topk_weights,
    Scalar* __restrict__ output) {
  static_assert(sizeof(Scalar) == 2);
  constexpr int32_t kHidden = 4096;
  constexpr int32_t kTopk = 6;
  constexpr int32_t kValues = 8;
  const int64_t token = blockIdx.x;
  const int32_t column = blockIdx.y * kMetadataBlock * kValues +
                         threadIdx.x * kValues;
  __shared__ int32_t route_rows[kTopk];
  __shared__ float route_weights[kTopk];
  if (threadIdx.x < kTopk) {
    const int64_t route = token * kTopk + threadIdx.x;
    route_rows[threadIdx.x] = route_to_padded[route];
    route_weights[threadIdx.x] = static_cast<float>(topk_weights[route]);
  }
  __syncthreads();
  float accumulators[kValues] = {};
#pragma unroll
  for (int32_t slot = 0; slot < kTopk; ++slot) {
    const int32_t row = route_rows[slot];
    if (row < 0) {
      continue;
    }
    alignas(16) Scalar values[kValues];
    *reinterpret_cast<uint4*>(values) = *reinterpret_cast<const uint4*>(
        sorted_output + static_cast<int64_t>(row) * kHidden + column);
    const float weight = route_weights[slot];
#pragma unroll
    for (int32_t value = 0; value < kValues; ++value) {
      accumulators[value] += weight * static_cast<float>(values[value]);
    }
  }
  alignas(16) Scalar result[kValues];
#pragma unroll
  for (int32_t value = 0; value < kValues; ++value) {
    result[value] = static_cast<Scalar>(accumulators[value]);
  }
  *reinterpret_cast<uint4*>(output + token * kHidden + column) =
      *reinterpret_cast<const uint4*>(result);
}

template <typename scalar_t, typename weight_t>
void launch_restore_combine(const torch::Tensor& sorted_output,
                            const torch::Tensor& route_to_padded,
                            const torch::Tensor& topk_weights,
                            torch::Tensor& output,
                            musaStream_t stream) {
  if constexpr (sizeof(scalar_t) == 2) {
    const char* enabled = std::getenv("XLLM_DSV4_LONG_MOE_VECTOR_RESTORE");
    if (enabled != nullptr && std::string(enabled) == "1" &&
        topk_weights.size(0) >= 4096 && topk_weights.size(0) <= 16384 &&
        topk_weights.size(1) == 6 && sorted_output.size(1) == 4096 &&
        sorted_output.scalar_type() == torch::kBFloat16 &&
        topk_weights.scalar_type() == torch::kBFloat16 &&
        reinterpret_cast<uintptr_t>(sorted_output.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0) {
      restore_combine_vector8_kernel<scalar_t, weight_t>
          <<<dim3(topk_weights.size(0), 2), kMetadataBlock, 0, stream>>>(
              sorted_output.data_ptr<scalar_t>(),
              route_to_padded.data_ptr<int32_t>(),
              topk_weights.data_ptr<weight_t>(), output.data_ptr<scalar_t>());
      LOG_FIRST_N(INFO, 1) << "[DSV4_LONG_MOE_VECTOR_RESTORE] rows=" << topk_weights.size(0)
                          << " hidden=4096 topk=6 bf16_router=1";
      return;
    }
  }
  constexpr int32_t kMaxTopK = 16;
  if (restore_combine_2d_enabled() &&
      topk_weights.size(0) >= restore_combine_2d_min_tokens() &&
      topk_weights.size(1) <= kMaxTopK) {
    const int32_t block_h = restore_combine_2d_block_h();
    auto launch_2d = [&](auto block_h_tag) {
      constexpr int32_t kBlockH = decltype(block_h_tag)::value;
      const dim3 grid(topk_weights.size(0),
                      (sorted_output.size(1) + kBlockH - 1) / kBlockH);
      restore_combine_2d_kernel<scalar_t, weight_t, kBlockH, kMaxTopK>
          <<<grid, kMetadataBlock, 0, stream>>>(
              sorted_output.data_ptr<scalar_t>(),
              route_to_padded.data_ptr<int32_t>(),
              topk_weights.data_ptr<weight_t>(),
              output.data_ptr<scalar_t>(),
              topk_weights.size(0),
              sorted_output.size(1),
              static_cast<int32_t>(topk_weights.size(1)));
    };
    if (block_h == 512) {
      launch_2d(std::integral_constant<int32_t, 512>{});
    } else if (block_h == 2048) {
      launch_2d(std::integral_constant<int32_t, 2048>{});
    } else {
      launch_2d(std::integral_constant<int32_t, 1024>{});
    }
  } else {
    restore_combine_kernel<scalar_t, weight_t>
        <<<topk_weights.size(0), restore_combine_threads(), 0, stream>>>(
            sorted_output.data_ptr<scalar_t>(),
            route_to_padded.data_ptr<int32_t>(),
            topk_weights.data_ptr<weight_t>(),
            output.data_ptr<scalar_t>(),
            topk_weights.size(0),
            sorted_output.size(1),
            static_cast<int32_t>(topk_weights.size(1)));
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v4_fp8_moe_prepare(const torch::Tensor& input,
                            const torch::Tensor& sorted_assignment,
                            const torch::Tensor& expert_counts,
                            int64_t topk,
                            int64_t alignment_m,
                            int64_t max_padded_assignments) {
  CHECK(input.defined() && input.dim() == 2 && input.is_contiguous());
  CHECK(sorted_assignment.defined() && sorted_assignment.dim() == 1 &&
        sorted_assignment.is_contiguous());
  CHECK(expert_counts.defined() && expert_counts.dim() == 1 &&
        expert_counts.is_contiguous());
  CHECK(sorted_assignment.scalar_type() == torch::kInt32);
  CHECK(expert_counts.scalar_type() == torch::kInt32);
  CHECK(input.device() == sorted_assignment.device());
  CHECK(input.device() == expert_counts.device());
  CHECK(topk > 0);
  // Alignment1 keeps sorted valid routes contiguous for count-based GEMM.
  // Invalid routes still have inverse=-1 and never enter restore/combine.
  CHECK(alignment_m == 1 || alignment_m == 128 || alignment_m == 256);
  CHECK(max_padded_assignments > 0);
  if (alignment_m == 1) {
    CHECK_GE(max_padded_assignments, input.size(0) * topk);
  }

  const auto int_options = input.options().dtype(torch::kInt32);
  auto padded_assignment = torch::empty({max_padded_assignments}, int_options);
  auto m_indices = torch::empty({max_padded_assignments}, int_options);
  auto sorted_input =
      torch::empty({max_padded_assignments, input.size(1)}, input.options());

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const musaStream_t stream = current_musa_stream(input.device());
  launch_build_padded_metadata(sorted_assignment.data_ptr<int32_t>(),
                               expert_counts.data_ptr<int32_t>(),
                               padded_assignment.data_ptr<int32_t>(),
                               m_indices.data_ptr<int32_t>(),
                               nullptr,
                               max_padded_assignments,
                               static_cast<int32_t>(expert_counts.numel()),
                               static_cast<int32_t>(alignment_m),
                               stream);
  check_last_error("metadata");

  const int64_t values = max_padded_assignments * input.size(1);
  const int64_t gather_grid = (values + kMetadataBlock - 1) / kMetadataBlock;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      input.scalar_type(),
      "deepseek_v4_fp8_moe_gather",
      [&] {
        gather_padded_input_kernel<scalar_t>
            <<<gather_grid, kMetadataBlock, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                padded_assignment.data_ptr<int32_t>(),
                sorted_input.data_ptr<scalar_t>(),
                max_padded_assignments,
                input.size(1),
                static_cast<int32_t>(topk));
      });
  check_last_error("gather");
  return {sorted_input, padded_assignment, m_indices};
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
deepseek_v4_fp8_moe_quant_prepare(const torch::Tensor& input,
                                  const torch::Tensor& sorted_assignment,
                                  const torch::Tensor& expert_counts,
                                  int64_t topk,
                                  int64_t alignment_m,
                                  int64_t max_padded_assignments) {
  CHECK(input.defined() && input.dim() == 2 && input.is_contiguous());
  CHECK(sorted_assignment.defined() && sorted_assignment.dim() == 1 &&
        sorted_assignment.is_contiguous());
  CHECK(expert_counts.defined() && expert_counts.dim() == 1 &&
        expert_counts.is_contiguous());
  CHECK(sorted_assignment.scalar_type() == torch::kInt32);
  CHECK(expert_counts.scalar_type() == torch::kInt32);
  CHECK(input.device() == sorted_assignment.device());
  CHECK(input.device() == expert_counts.device());
  CHECK(input.size(1) > 0 && input.size(1) % kQuantBlock == 0);
  CHECK(topk > 0);
  CHECK(alignment_m == 1 || alignment_m == 128 || alignment_m == 256);
  CHECK(max_padded_assignments > 0);

  const int64_t num_assignments = input.size(0) * topk;
  if (alignment_m == 1) {
    CHECK_GE(max_padded_assignments, num_assignments);
  }
  auto& workspace = get_quant_prepare_workspace(
      input, max_padded_assignments, num_assignments);
  auto input_fp8 = workspace.input_fp8.narrow(0, 0, max_padded_assignments);
  auto input_scale = workspace.input_scale.narrow(0, 0, max_padded_assignments);
  auto padded_assignment =
      workspace.padded_assignment.narrow(0, 0, max_padded_assignments);
  auto m_indices = workspace.m_indices.narrow(0, 0, max_padded_assignments);
  auto route_to_padded =
      workspace.route_to_padded.narrow(0, 0, num_assignments);

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const musaStream_t stream = current_musa_stream(input.device());
  const musaError_t route_init_status =
      musaMemsetAsync(route_to_padded.data_ptr<int32_t>(),
                      0xff,
                      route_to_padded.numel() * sizeof(int32_t),
                      stream);
  CHECK(route_init_status == musaSuccess)
      << "DeepSeek-V4 FP8 MoE route metadata initialization failed: "
      << musaGetErrorString(route_init_status);
  launch_build_padded_metadata(sorted_assignment.data_ptr<int32_t>(),
                               expert_counts.data_ptr<int32_t>(),
                               padded_assignment.data_ptr<int32_t>(),
                               m_indices.data_ptr<int32_t>(),
                               route_to_padded.data_ptr<int32_t>(),
                               max_padded_assignments,
                               static_cast<int32_t>(expert_counts.numel()),
                               static_cast<int32_t>(alignment_m),
                               stream);
  check_last_error("quant metadata");

  const int64_t scale_cols = input.size(1) / kQuantBlock;
  const musaError_t scale_init_status =
      musaMemsetAsync(input_scale.data_ptr<float>(),
                      0,
                      input_scale.numel() * sizeof(float),
                      stream);
  CHECK(scale_init_status == musaSuccess)
      << "DeepSeek-V4 FP8 MoE scale initialization failed: "
      << musaGetErrorString(scale_init_status);
  const int32_t quant_groups = quant_groups_per_cta();
  const dim3 grid(input.size(0),
                  (scale_cols + quant_groups - 1) / quant_groups);
  const dim3 block(kQuantBlock * quant_groups);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      input.scalar_type(),
      "deepseek_v4_fp8_moe_quant_prepare",
      [&] {
        if (warp_quant_scatter_enabled()) {
          const int64_t total_groups = input.size(0) * scale_cols;
          const int32_t warps_per_cta = warp_quant_scatter_warps_per_cta();
          auto launch_warp_scatter = [&](auto warps_per_cta_tag) {
            constexpr int32_t kWarpsPerCta = decltype(warps_per_cta_tag)::value;
            const int64_t warp_grid =
                (total_groups + kWarpsPerCta - 1) / kWarpsPerCta;
            token_blockwise_fp8_scatter_warp_kernel<scalar_t, kWarpsPerCta>
                <<<warp_grid, kWarpsPerCta * kWarpSize, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    route_to_padded.data_ptr<int32_t>(),
                    input_fp8.data_ptr<c10::Float8_e4m3fn>(),
                    input_scale.data_ptr<float>(),
                    input.size(0),
                    input.size(1),
                    scale_cols,
                    static_cast<int32_t>(topk));
          };
          if (warps_per_cta == 4) {
            launch_warp_scatter(std::integral_constant<int32_t, 4>{});
          } else if (warps_per_cta == 16) {
            launch_warp_scatter(std::integral_constant<int32_t, 16>{});
          } else {
            launch_warp_scatter(std::integral_constant<int32_t, 8>{});
          }
        } else if (quant_groups == 4) {
          token_blockwise_fp8_scatter_grouped_kernel<scalar_t, 4>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(),
                  route_to_padded.data_ptr<int32_t>(),
                  input_fp8.data_ptr<c10::Float8_e4m3fn>(),
                  input_scale.data_ptr<float>(),
                  input.size(0),
                  input.size(1),
                  scale_cols,
                  static_cast<int32_t>(topk));
        } else if (quant_groups == 2) {
          token_blockwise_fp8_scatter_grouped_kernel<scalar_t, 2>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(),
                  route_to_padded.data_ptr<int32_t>(),
                  input_fp8.data_ptr<c10::Float8_e4m3fn>(),
                  input_scale.data_ptr<float>(),
                  input.size(0),
                  input.size(1),
                  scale_cols,
                  static_cast<int32_t>(topk));
        } else {
          token_blockwise_fp8_scatter_kernel<scalar_t>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(),
                  route_to_padded.data_ptr<int32_t>(),
                  input_fp8.data_ptr<c10::Float8_e4m3fn>(),
                  input_scale.data_ptr<float>(),
                  input.size(0),
                  input.size(1),
                  scale_cols,
                  static_cast<int32_t>(topk));
        }
      });
  check_last_error("quant gather");
  return {
      input_fp8, input_scale, padded_assignment, m_indices, route_to_padded};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v4_fp8_moe_masked_decode_prepare(const torch::Tensor& input,
                                          const torch::Tensor& local_ids,
                                          int64_t num_experts) {
  CHECK(input.defined() && input.dim() == 2 && input.is_contiguous());
  CHECK(input.scalar_type() == torch::kBFloat16 ||
        input.scalar_type() == torch::kFloat16);
  CHECK(input.size(0) == 1 && input.size(1) > 0 &&
        input.size(1) % kQuantBlock == 0);
  CHECK(local_ids.defined() && local_ids.dim() == 2 &&
        local_ids.is_contiguous());
  CHECK(local_ids.scalar_type() == torch::kInt32);
  CHECK(local_ids.size(0) == 1 && local_ids.size(1) > 0);
  CHECK(local_ids.device() == input.device());
  CHECK(num_experts > 0);

  const int64_t scale_cols = input.size(1) / kQuantBlock;
  auto& workspace = get_quant_prepare_workspace(
      input, num_experts, /*routes=*/0, num_experts);
  auto input_fp8 = workspace.input_fp8.narrow(0, 0, num_experts)
                       .view({num_experts, 1, input.size(1)});
  auto input_scale = workspace.input_scale.narrow(0, 0, num_experts)
                         .view({num_experts, 1, scale_cols});
  auto masked_tokens = workspace.masked_tokens.narrow(0, 0, num_experts);

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const musaStream_t stream = current_musa_stream(input.device());
  const musaError_t count_init_status =
      musaMemsetAsync(masked_tokens.data_ptr<int32_t>(),
                      0,
                      masked_tokens.numel() * sizeof(int32_t),
                      stream);
  CHECK(count_init_status == musaSuccess)
      << "DeepSeek-V4 masked decode count initialization failed: "
      << musaGetErrorString(count_init_status);

  const dim3 grid(scale_cols);
  const dim3 block(kQuantBlock);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      input.scalar_type(),
      "deepseek_v4_fp8_moe_masked_decode_prepare",
      [&] {
        masked_decode_quant_scatter_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                local_ids.data_ptr<int32_t>(),
                input_fp8.data_ptr<c10::Float8_e4m3fn>(),
                input_scale.data_ptr<float>(),
                masked_tokens.data_ptr<int32_t>(),
                input.size(1),
                scale_cols,
                static_cast<int32_t>(local_ids.size(1)),
                static_cast<int32_t>(num_experts));
      });
  check_last_error("masked decode quant scatter");
  return {input_fp8, input_scale, masked_tokens};
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
deepseek_v4_fp8_moe_scatter_prepare(const torch::Tensor& input_fp8,
                                    const torch::Tensor& input_scale,
                                    const torch::Tensor& sorted_assignment,
                                    const torch::Tensor& expert_counts,
                                    int64_t topk,
                                    int64_t alignment_m,
                                    int64_t max_padded_assignments) {
  CHECK(input_fp8.defined() && input_fp8.dim() == 2 &&
        input_fp8.is_contiguous());
  CHECK(input_fp8.scalar_type() == torch::kFloat8_e4m3fn);
  CHECK(input_scale.defined() && input_scale.dim() == 2 &&
        input_scale.is_contiguous());
  CHECK(input_scale.scalar_type() == torch::kFloat32);
  CHECK(input_scale.size(0) == input_fp8.size(0));
  CHECK(input_fp8.size(1) > 0 && input_fp8.size(1) % kQuantBlock == 0);
  CHECK(input_scale.size(1) == input_fp8.size(1) / kQuantBlock);
  CHECK(sorted_assignment.defined() && sorted_assignment.dim() == 1 &&
        sorted_assignment.is_contiguous());
  CHECK(expert_counts.defined() && expert_counts.dim() == 1 &&
        expert_counts.is_contiguous());
  CHECK(sorted_assignment.scalar_type() == torch::kInt32);
  CHECK(expert_counts.scalar_type() == torch::kInt32);
  CHECK(input_fp8.device() == input_scale.device());
  CHECK(input_fp8.device() == sorted_assignment.device());
  CHECK(input_fp8.device() == expert_counts.device());
  CHECK(topk > 0);
  CHECK(alignment_m == 128 || alignment_m == 256);
  CHECK(max_padded_assignments > 0);

  const int64_t num_assignments = input_fp8.size(0) * topk;
  auto& workspace = get_quant_prepare_workspace(
      input_fp8, max_padded_assignments, num_assignments);
  auto output_fp8 = workspace.input_fp8.narrow(0, 0, max_padded_assignments);
  auto output_scale =
      workspace.input_scale.narrow(0, 0, max_padded_assignments);
  auto padded_assignment =
      workspace.padded_assignment.narrow(0, 0, max_padded_assignments);
  auto m_indices = workspace.m_indices.narrow(0, 0, max_padded_assignments);
  auto route_to_padded =
      workspace.route_to_padded.narrow(0, 0, num_assignments);

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input_fp8));
  const musaStream_t stream = current_musa_stream(input_fp8.device());
  const musaError_t route_init_status =
      musaMemsetAsync(route_to_padded.data_ptr<int32_t>(),
                      0xff,
                      route_to_padded.numel() * sizeof(int32_t),
                      stream);
  CHECK(route_init_status == musaSuccess)
      << "DeepSeek-V4 prequantized route initialization failed: "
      << musaGetErrorString(route_init_status);
  launch_build_padded_metadata(sorted_assignment.data_ptr<int32_t>(),
                               expert_counts.data_ptr<int32_t>(),
                               padded_assignment.data_ptr<int32_t>(),
                               m_indices.data_ptr<int32_t>(),
                               route_to_padded.data_ptr<int32_t>(),
                               max_padded_assignments,
                               static_cast<int32_t>(expert_counts.numel()),
                               static_cast<int32_t>(alignment_m),
                               stream);
  check_last_error("prequantized metadata");

  const musaError_t scale_init_status =
      musaMemsetAsync(output_scale.data_ptr<float>(),
                      0,
                      output_scale.numel() * sizeof(float),
                      stream);
  CHECK(scale_init_status == musaSuccess)
      << "DeepSeek-V4 prequantized scale initialization failed: "
      << musaGetErrorString(scale_init_status);
  const int64_t scale_cols = input_scale.size(1);
  const dim3 grid(input_fp8.size(0), scale_cols);
  const dim3 block(kQuantBlock);
  prequantized_fp8_scatter_kernel<<<grid, block, 0, stream>>>(
      input_fp8.data_ptr<c10::Float8_e4m3fn>(),
      input_scale.data_ptr<float>(),
      route_to_padded.data_ptr<int32_t>(),
      output_fp8.data_ptr<c10::Float8_e4m3fn>(),
      output_scale.data_ptr<float>(),
      input_fp8.size(0),
      input_fp8.size(1),
      scale_cols,
      static_cast<int32_t>(topk));
  check_last_error("prequantized scatter");
  return {
      output_fp8, output_scale, padded_assignment, m_indices, route_to_padded};
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_fp8_moe_swiglu_quantize(
    const torch::Tensor& gate_up,
    const torch::Tensor& ragged_tokens_info,
    double swiglu_limit) {
  CHECK(gate_up.defined() && gate_up.dim() == 2 && gate_up.is_contiguous());
  CHECK(gate_up.scalar_type() == torch::kBFloat16 ||
        gate_up.scalar_type() == torch::kFloat16);
  CHECK(gate_up.size(1) > 0 && gate_up.size(1) % (2 * kQuantBlock) == 0);
  CHECK(ragged_tokens_info.defined() && ragged_tokens_info.dim() == 1 &&
        ragged_tokens_info.is_contiguous());
  CHECK(ragged_tokens_info.scalar_type() == torch::kInt32);
  CHECK(gate_up.size(0) == ragged_tokens_info.size(0));
  CHECK(gate_up.device() == ragged_tokens_info.device());

  const int64_t rows = gate_up.size(0);
  const int64_t hidden_size = gate_up.size(1) / 2;
  const int64_t scale_cols = hidden_size / kQuantBlock;
  auto& workspace =
      get_swiglu_quant_workspace(gate_up, rows, hidden_size, scale_cols);
  auto output = workspace.output.narrow(0, 0, rows);
  auto scale = workspace.scale.narrow(0, 0, rows);
  if (rows == 0) {
    return {output, scale};
  }

  float limit = static_cast<float>(swiglu_limit);
  if (!std::isfinite(limit) || limit >= 1000000.0f) {
    limit = 0.0f;
  }
  const XLLMOptionalDeviceGuard device_guard(at::device_of(gate_up));
  const musaStream_t stream = current_musa_stream(gate_up.device());
  const int32_t quant_groups = activation_quant_groups_per_cta();
  const dim3 grid(rows, (scale_cols + quant_groups - 1) / quant_groups);
  const dim3 block(kQuantBlock * quant_groups);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      gate_up.scalar_type(),
      "deepseek_v4_fp8_moe_swiglu_quantize",
      [&] {
        if (warp_swiglu_quant_enabled() ||
            long_warp_swiglu_quant_enabled(gate_up)) {
          constexpr int32_t kWarpsPerCta = 8;
          const int64_t total_groups = rows * scale_cols;
          const int64_t warp_grid =
              (total_groups + kWarpsPerCta - 1) / kWarpsPerCta;
          swiglu_blockwise_fp8_quant_warp_kernel<scalar_t, kWarpsPerCta>
              <<<warp_grid, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  gate_up.data_ptr<scalar_t>(),
                  ragged_tokens_info.data_ptr<int32_t>(),
                  output.data_ptr<c10::Float8_e4m3fn>(),
                  scale.data_ptr<float>(),
                  rows,
                  hidden_size,
                  scale_cols,
                  limit);
        } else if (quant_groups == 4) {
          swiglu_blockwise_fp8_quant_grouped_kernel<scalar_t, 4>
              <<<grid, block, 0, stream>>>(
                  gate_up.data_ptr<scalar_t>(),
                  ragged_tokens_info.data_ptr<int32_t>(),
                  output.data_ptr<c10::Float8_e4m3fn>(),
                  scale.data_ptr<float>(),
                  rows,
                  hidden_size,
                  scale_cols,
                  limit);
        } else if (quant_groups == 2) {
          swiglu_blockwise_fp8_quant_grouped_kernel<scalar_t, 2>
              <<<grid, block, 0, stream>>>(
                  gate_up.data_ptr<scalar_t>(),
                  ragged_tokens_info.data_ptr<int32_t>(),
                  output.data_ptr<c10::Float8_e4m3fn>(),
                  scale.data_ptr<float>(),
                  rows,
                  hidden_size,
                  scale_cols,
                  limit);
        } else {
          swiglu_blockwise_fp8_quant_kernel<scalar_t>
              <<<grid, block, 0, stream>>>(
                  gate_up.data_ptr<scalar_t>(),
                  ragged_tokens_info.data_ptr<int32_t>(),
                  output.data_ptr<c10::Float8_e4m3fn>(),
                  scale.data_ptr<float>(),
                  rows,
                  hidden_size,
                  scale_cols,
                  limit);
        }
      });
  check_last_error("SwiGLU quant");
  return {output, scale};
}

torch::Tensor deepseek_v4_fp8_moe_restore(
    const torch::Tensor& sorted_output,
    const torch::Tensor& padded_assignment,
    int64_t num_assignments) {
  CHECK(sorted_output.defined() && sorted_output.dim() == 2 &&
        sorted_output.is_contiguous());
  CHECK(padded_assignment.defined() && padded_assignment.dim() == 1 &&
        padded_assignment.is_contiguous());
  CHECK(padded_assignment.scalar_type() == torch::kInt32);
  CHECK(sorted_output.size(0) == padded_assignment.size(0));
  CHECK(sorted_output.device() == padded_assignment.device());
  CHECK(num_assignments > 0);

  auto output = torch::zeros({num_assignments, sorted_output.size(1)},
                             sorted_output.options());
  const int64_t values = sorted_output.numel();
  const int64_t grid = (values + kMetadataBlock - 1) / kMetadataBlock;
  const XLLMOptionalDeviceGuard device_guard(at::device_of(sorted_output));
  const musaStream_t stream = current_musa_stream(sorted_output.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      sorted_output.scalar_type(),
      "deepseek_v4_fp8_moe_restore",
      [&] {
        restore_padded_output_kernel<scalar_t>
            <<<grid, kMetadataBlock, 0, stream>>>(
                sorted_output.data_ptr<scalar_t>(),
                padded_assignment.data_ptr<int32_t>(),
                output.data_ptr<scalar_t>(),
                sorted_output.size(0),
                sorted_output.size(1),
                num_assignments);
      });
  check_last_error("restore");
  return output;
}

torch::Tensor deepseek_v4_fp8_moe_restore_combine(
    const torch::Tensor& sorted_output,
    const torch::Tensor& route_to_padded,
    const torch::Tensor& topk_weights,
    const torch::Tensor& output_buffer) {
  CHECK(sorted_output.defined() && sorted_output.dim() == 2 &&
        sorted_output.is_contiguous());
  CHECK(route_to_padded.defined() && route_to_padded.dim() == 1 &&
        route_to_padded.is_contiguous());
  CHECK(route_to_padded.scalar_type() == torch::kInt32);
  CHECK(topk_weights.defined() && topk_weights.dim() == 2 &&
        topk_weights.is_contiguous());
  CHECK(topk_weights.scalar_type() == torch::kFloat32 ||
        topk_weights.scalar_type() == torch::kFloat16 ||
        topk_weights.scalar_type() == torch::kBFloat16);
  CHECK(sorted_output.device() == route_to_padded.device());
  CHECK(sorted_output.device() == topk_weights.device());
  CHECK(route_to_padded.numel() == topk_weights.numel());

  torch::Tensor full_output = output_buffer;
  if (!full_output.defined()) {
    full_output = torch::empty({topk_weights.size(0), sorted_output.size(1)},
                               sorted_output.options());
  } else {
    CHECK(full_output.dim() == 2 && full_output.is_contiguous());
    CHECK(full_output.size(0) >= topk_weights.size(0));
    CHECK(full_output.size(1) == sorted_output.size(1));
    CHECK(full_output.scalar_type() == sorted_output.scalar_type());
    CHECK(full_output.device() == sorted_output.device());
  }
  auto output = full_output.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/topk_weights.size(0));
  const XLLMOptionalDeviceGuard device_guard(at::device_of(sorted_output));
  const musaStream_t stream = current_musa_stream(sorted_output.device());
  if (full_output.size(0) > output.size(0)) {
    const int64_t tail_values =
        (full_output.size(0) - output.size(0)) * full_output.size(1);
    auto* tail = static_cast<char*>(full_output.data_ptr()) +
                 output.numel() * output.element_size();
    const musaError_t tail_status = musaMemsetAsync(
        tail, 0, tail_values * full_output.element_size(), stream);
    CHECK(tail_status == musaSuccess)
        << "DeepSeek-V4 FP8 MoE reduce-scatter tail zero failed: "
        << musaGetErrorString(tail_status);
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      sorted_output.scalar_type(),
      "deepseek_v4_fp8_moe_restore_combine_output",
      [&] {
        if (topk_weights.scalar_type() == torch::kFloat32) {
          launch_restore_combine<scalar_t, float>(
              sorted_output, route_to_padded, topk_weights, output, stream);
        } else if (topk_weights.scalar_type() == torch::kFloat16) {
          launch_restore_combine<scalar_t, c10::Half>(
              sorted_output, route_to_padded, topk_weights, output, stream);
        } else {
          launch_restore_combine<scalar_t, c10::BFloat16>(
              sorted_output, route_to_padded, topk_weights, output, stream);
        }
      });
  check_last_error("restore combine");
  return output;
}

}  // namespace xllm::kernel::musa
