// Copyright 2026 The xLLM Authors. Licensed under the Apache License,
// Version 2.0.
#include <glog/logging.h>
#include <musa_fp8.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include "core/kernels/musa/deepseek_v4_indexer_metadata.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kThreads = 256;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kTopk = 512;
constexpr int64_t kExtra = 128;

bool tiled_metadata_enabled() {
  const char* value = std::getenv("XLLM_DSV4_INDEXER_METADATA_TILED");
  return value != nullptr && std::string(value) == "1";
}

int32_t launch_blocks(int64_t elements) {
  return static_cast<int32_t>(
      std::min<int64_t>((elements + kThreads - 1) / kThreads, 4096));
}

void check_tensor(const torch::Tensor& tensor,
                  const torch::Tensor& anchor,
                  torch::ScalarType dtype) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), anchor.device());
  CHECK(tensor.is_contiguous());
  CHECK_EQ(tensor.scalar_type(), dtype);
}

void check_table(const torch::Tensor& table,
                 const torch::Tensor& anchor,
                 int64_t batch) {
  CHECK(table.defined());
  CHECK_EQ(table.device(), anchor.device());
  CHECK(table.is_contiguous());
  CHECK_EQ(table.dim(), 2);
  CHECK_GE(table.size(0), batch);
  CHECK_GT(table.size(1), 0);
  CHECK(table.scalar_type() == torch::kInt32 ||
        table.scalar_type() == torch::kInt64);
}

__global__ void indexer_relu_weight_kernel(float* __restrict__ scores,
                                           const float* __restrict__ weights,
                                           int32_t elements,
                                           int32_t capacity) {
  for (int32_t offset =
           static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
       offset < elements;
       offset += static_cast<int32_t>(gridDim.x * blockDim.x)) {
    const float value = scores[offset];
    // Do not use fmaxf: the framework preserves NaN and negative zero here.
    const float rectified = value < 0.0f ? 0.0f : value;
    scores[offset] = rectified * weights[offset / capacity];
  }
}

__global__ void indexer_weighted_head_sum_kernel(
    const float4* __restrict__ scores,
    const float* __restrict__ weights,
    float4* __restrict__ output,
    int32_t total_tiles,
    int32_t key_tiles,
    int32_t capacity_vectors) {
  constexpr int32_t kHeadLanes = 8;
  constexpr int32_t kKeyVectors = 32;
  constexpr int32_t kHeads = 64;
  __shared__ float4 partial[kHeadLanes][kKeyVectors];
  const int32_t key_lane = static_cast<int32_t>(threadIdx.x);
  const int32_t head_lane = static_cast<int32_t>(threadIdx.y);
  for (int32_t tile = static_cast<int32_t>(blockIdx.x); tile < total_tiles;
       tile += static_cast<int32_t>(gridDim.x)) {
    const int32_t row = tile / key_tiles;
    const int32_t column = (tile % key_tiles) * kKeyVectors + key_lane;
    float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
    if (column < capacity_vectors) {
#pragma unroll
      for (int32_t group = 0; group < kHeads / kHeadLanes; ++group) {
        const int32_t head = group * kHeadLanes + head_lane;
        const float4 value =
            scores[(row * kHeads + head) * capacity_vectors + column];
        const float weight = weights[row * kHeads + head];
        sum.x = __fadd_rn(sum.x,
                          __fmul_rn(value.x < 0.0f ? 0.0f : value.x, weight));
        sum.y = __fadd_rn(sum.y,
                          __fmul_rn(value.y < 0.0f ? 0.0f : value.y, weight));
        sum.z = __fadd_rn(sum.z,
                          __fmul_rn(value.z < 0.0f ? 0.0f : value.z, weight));
        sum.w = __fadd_rn(sum.w,
                          __fmul_rn(value.w < 0.0f ? 0.0f : value.w, weight));
      }
    }
    partial[head_lane][key_lane] = sum;
    __syncthreads();
    if (head_lane == 0 && column < capacity_vectors) {
      float4 total{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int32_t lane = 0; lane < kHeadLanes; ++lane) {
        const float4 value = partial[lane][key_lane];
        total.x = __fadd_rn(total.x, value.x);
        total.y = __fadd_rn(total.y, value.y);
        total.z = __fadd_rn(total.z, value.z);
        total.w = __fadd_rn(total.w, value.w);
      }
      output[row * capacity_vectors + column] = total;
    }
    // A block can handle another tile; do not overwrite partial sums while
    // lane zero still consumes the previous tile.
    __syncthreads();
  }
}

template <int32_t Capacity>
__global__ void indexer_relu_weight_vector_kernel(
    float4* __restrict__ scores,
    const float* __restrict__ weights,
    int32_t vectors,
    int32_t capacity_vectors) {
  for (int32_t offset =
           static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
       offset < vectors;
       offset += static_cast<int32_t>(gridDim.x * blockDim.x)) {
    const int32_t head =
        offset / (Capacity > 0 ? Capacity / 4 : capacity_vectors);
    const float weight = weights[head];
    float4 value = scores[offset];
    value.x = (value.x < 0.0f ? 0.0f : value.x) * weight;
    value.y = (value.y < 0.0f ? 0.0f : value.y) * weight;
    value.z = (value.z < 0.0f ? 0.0f : value.z) * weight;
    value.w = (value.w < 0.0f ? 0.0f : value.w) * weight;
    scores[offset] = value;
  }
}

template <int32_t Capacity>
void launch_indexer_relu_weight_vector(const torch::Tensor& scores,
                                       const torch::Tensor& weights) {
  indexer_relu_weight_vector_kernel<Capacity>
      <<<launch_blocks(scores.numel() / 4),
         kThreads,
         0,
         current_musa_stream(scores.device())>>>(
          reinterpret_cast<float4*>(scores.data_ptr<float>()),
          weights.data_ptr<float>(),
          static_cast<int32_t>(scores.numel() / 4),
          static_cast<int32_t>(scores.size(3) / 4));
}

template <typename BlockIndex, typename Output>
__global__ void indexer_gather_keys_kernel(
    const int8_t* __restrict__ cache,
    const c10::Half* __restrict__ scales,
    const BlockIndex* __restrict__ blocks,
    Output* __restrict__ keys,
    bool* __restrict__ valid_slots,
    int64_t elements,
    int64_t capacity,
    int64_t table_columns,
    int64_t page,
    int64_t physical_pages) {
  const int64_t slots = physical_pages * page;
  for (int64_t offset =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < elements;
       offset += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = offset / kHeadDim;
    const int64_t channel = offset % kHeadDim;
    const int64_t sequence = row / capacity;
    const int64_t position = row % capacity;
    const int64_t block = static_cast<int64_t>(
        blocks[sequence * table_columns + position / page]);
    // Preserve the helper's terminal-page compatibility. Invalid values
    // use its safe-slot clamp, even though their scores will later be masked.
    const int64_t adjusted_block =
        block >= physical_pages && block > 0 ? block - 1 : block;
    const bool valid = adjusted_block >= 0 && adjusted_block < physical_pages;
    const int64_t safe_slot = adjusted_block < 0 ? 0
                              : adjusted_block >= physical_pages
                                  ? slots - 1
                                  : adjusted_block * page + position % page;
    const float scaled =
        static_cast<float>(cache[safe_slot * kHeadDim + channel]) *
        static_cast<float>(scales[safe_slot]);
    keys[offset] = static_cast<Output>(static_cast<c10::BFloat16>(scaled));
    if (channel == 0) {
      valid_slots[row] = valid;
    }
  }
}

__global__ void indexer_mask_scores_kernel(float* __restrict__ scores,
                                           const int32_t* __restrict__ lengths,
                                           const bool* __restrict__ valid_slots,
                                           int64_t elements,
                                           int64_t capacity,
                                           int64_t query_rows_per_sequence) {
  for (int64_t offset =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < elements;
       offset += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = offset / capacity;
    const int64_t position = offset % capacity;
    const int64_t live = lengths[row] > 0 ? lengths[row] : 0;
    if (position >= live / 4 ||
        !valid_slots[(row / query_rows_per_sequence) * capacity + position]) {
      scores[offset] = -INFINITY;
    }
  }
}

template <typename BlockIndex, typename Output>
__global__ void indexer_gather_keys_tiled_kernel(
    const int8_t* __restrict__ cache,
    const c10::Half* __restrict__ scales,
    const BlockIndex* __restrict__ blocks,
    Output* __restrict__ keys,
    bool* __restrict__ valid_slots,
    int32_t capacity,
    int64_t table_columns,
    int64_t physical_pages) {
  // One grid-y plane owns one sequence. Page128 and dim128 are guaranteed by
  // dispatch, so no per-element dynamic 64-bit quotient/remainder is needed.
  const int64_t sequence = static_cast<int64_t>(blockIdx.y);
  const int64_t slots = physical_pages * 128;
  const int64_t output_base = sequence * capacity * 128;
  for (int32_t offset =
           static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
       offset < capacity * 128;
       offset += static_cast<int32_t>(gridDim.x * blockDim.x)) {
    const int32_t position = offset >> 7;
    const int32_t channel = offset & 127;
    const int64_t block = static_cast<int64_t>(
        blocks[sequence * table_columns + (position >> 7)]);
    const int64_t adjusted =
        block >= physical_pages && block > 0 ? block - 1 : block;
    const bool valid = adjusted >= 0 && adjusted < physical_pages;
    const int64_t slot = adjusted < 0 ? 0
                         : adjusted >= physical_pages
                             ? slots - 1
                             : adjusted * 128 + (position & 127);
    const float scaled = static_cast<float>(cache[slot * 128 + channel]) *
                         static_cast<float>(scales[slot]);
    keys[output_base + offset] =
        static_cast<Output>(static_cast<c10::BFloat16>(scaled));
    if (channel == 0) {
      valid_slots[sequence * capacity + position] = valid;
    }
  }
}

__global__ void indexer_mask_scores_tiled_kernel(
    float* __restrict__ scores,
    const int32_t* __restrict__ lengths,
    const bool* __restrict__ valid_slots,
    int32_t capacity,
    int32_t query_rows_per_sequence) {
  const int32_t row = static_cast<int32_t>(blockIdx.y);
  const int32_t live = lengths[row] > 0 ? lengths[row] : 0;
  const int32_t owner = row / query_rows_per_sequence;
  for (int32_t position =
           static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
       position < capacity;
       position += static_cast<int32_t>(gridDim.x * blockDim.x)) {
    if (position >= live / 4 ||
        !valid_slots[static_cast<int64_t>(owner) * capacity + position]) {
      scores[static_cast<int64_t>(row) * capacity + position] = -INFINITY;
    }
  }
}

template <typename BlockIndex>
__device__ int32_t map_physical_index(int64_t logical,
                                      const BlockIndex* table,
                                      int64_t owner,
                                      int64_t columns,
                                      int64_t page) {
  if (logical < 0 || logical / page >= columns) {
    return -1;
  }
  const int64_t block =
      static_cast<int64_t>(table[owner * columns + logical / page]);
  const int64_t offset = logical % page;
  constexpr int64_t kMaxPhysical = 2147483647;
  if (block < 0 || block > (kMaxPhysical - offset) / page) {
    return -1;
  }
  return static_cast<int32_t>(block * page + offset);
}

template <typename BlockIndex>
__device__ int32_t map_physical_index_page128(int64_t logical,
                                              const BlockIndex* table,
                                              int32_t owner,
                                              int64_t columns) {
  if (logical < 0 || (logical >> 7) >= columns) {
    return -1;
  }
  const int64_t block = static_cast<int64_t>(
      table[static_cast<int64_t>(owner) * columns + (logical >> 7)]);
  // All offsets0..127 fit int32 exactly when the block fits24 positive bits.
  if (block < 0 || block > 16777215) {
    return -1;
  }
  return static_cast<int32_t>(block * 128 + (logical & 127));
}

template <typename CompressedIndex, typename SwaIndex>
__global__ void indexer_map_outputs_tiled_kernel(
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ lengths,
    const CompressedIndex* __restrict__ compressed,
    const SwaIndex* __restrict__ swa,
    int32_t* __restrict__ logical,
    int32_t* __restrict__ physical,
    int32_t* __restrict__ extra,
    int32_t capacity,
    int32_t query_rows_per_sequence,
    int64_t compressed_columns,
    int64_t swa_columns,
    int32_t window) {
  const int32_t row = static_cast<int32_t>(blockIdx.y);
  const int32_t index =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= 640) {
    return;
  }
  const int32_t owner = row / query_rows_per_sequence;
  const int32_t live = lengths[row] > 0 ? lengths[row] : 0;
  const int32_t valid = live / 4 < capacity ? live / 4 : capacity;
  if (index < 512) {
    const int64_t key = index >= valid ? -1
                        : valid <= 512
                            ? index
                            : selected[static_cast<int64_t>(row) * 512 + index];
    logical[static_cast<int64_t>(row) * 512 + index] =
        static_cast<int32_t>(key);
    physical[static_cast<int64_t>(row) * 512 + index] =
        map_physical_index_page128(key, compressed, owner, compressed_columns);
    return;
  }
  const int32_t extra_index = index - 512;
  const int32_t count = live < window ? live : window;
  const int64_t key = extra_index < count
                          ? static_cast<int64_t>(live) - count + extra_index
                          : -1;
  extra[static_cast<int64_t>(row) * 128 + extra_index] =
      map_physical_index_page128(key, swa, owner, swa_columns);
}

template <typename CompressedIndex, typename SwaIndex>
__global__ void indexer_map_outputs_kernel(
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ lengths,
    const CompressedIndex* __restrict__ compressed,
    const SwaIndex* __restrict__ swa,
    int32_t* __restrict__ logical,
    int32_t* __restrict__ physical,
    int32_t* __restrict__ extra,
    int64_t elements,
    int64_t capacity,
    int64_t query_rows_per_sequence,
    int64_t compressed_columns,
    int64_t swa_columns,
    int64_t window,
    int64_t page) {
  for (int64_t offset =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < elements;
       offset += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = offset / (kTopk + kExtra);
    const int64_t index = offset % (kTopk + kExtra);
    const int64_t owner = row / query_rows_per_sequence;
    const int64_t live = lengths[row] > 0 ? lengths[row] : 0;
    const int64_t valid = live / 4 < capacity ? live / 4 : capacity;
    if (index < kTopk) {
      const int64_t key = index >= valid   ? -1
                          : valid <= kTopk ? index
                                           : selected[row * kTopk + index];
      logical[row * kTopk + index] = static_cast<int32_t>(key);
      physical[row * kTopk + index] =
          map_physical_index(key, compressed, owner, compressed_columns, page);
      continue;
    }
    const int64_t extra_index = index - kTopk;
    const int64_t count = live < window ? live : window;
    const int64_t key = extra_index < count ? live - count + extra_index : -1;
    extra[row * kExtra + extra_index] =
        map_physical_index(key, swa, owner, swa_columns, page);
  }
}

}  // namespace

namespace {

void gather_keys(const torch::Tensor& cache,
                 const torch::Tensor& scales,
                 const torch::Tensor& blocks,
                 const torch::Tensor& keys,
                 const torch::Tensor& valid_slots,
                 torch::ScalarType output_dtype) {
  CHECK(cache.defined());
  CHECK(cache.device().is_privateuseone());
  CHECK(cache.dim() == 3 || (cache.dim() == 4 && cache.size(2) == 1));
  CHECK_GT(cache.size(0), 0);
  CHECK_GT(cache.size(1), 0);
  CHECK_EQ(cache.size(-1), kHeadDim);
  check_tensor(cache, cache, torch::kInt8);
  check_tensor(scales, cache, torch::kFloat16);
  CHECK_EQ(scales.numel(), cache.size(0) * cache.size(1));
  check_tensor(keys, cache, output_dtype);
  CHECK_EQ(keys.dim(), 3);
  CHECK_GT(keys.size(0), 0);
  CHECK_GT(keys.size(1), 0);
  CHECK_EQ(keys.size(2), kHeadDim);
  check_tensor(valid_slots, cache, torch::kBool);
  CHECK_EQ(valid_slots.sizes(), keys.sizes().slice(0, 2));
  check_table(blocks, cache, keys.size(0));
  CHECK_GE(blocks.size(1) * cache.size(1), keys.size(1));
  const XLLMOptionalDeviceGuard guard(cache.device());
  const bool tiled = tiled_metadata_enabled() && cache.size(1) == 128 &&
                     keys.size(0) <= 16 && keys.size(1) <= 65536;
  auto launch = [&](const auto* table, auto* destination) {
    if (tiled) {
      indexer_gather_keys_tiled_kernel<<<
          dim3(static_cast<uint32_t>(launch_blocks(keys.size(1) * 128)),
               static_cast<uint32_t>(keys.size(0))),
          kThreads,
          0,
          current_musa_stream(cache.device())>>>(
          cache.data_ptr<int8_t>(),
          scales.data_ptr<c10::Half>(),
          table,
          destination,
          valid_slots.data_ptr<bool>(),
          static_cast<int32_t>(keys.size(1)),
          blocks.size(1),
          cache.size(0));
      return;
    }
    indexer_gather_keys_kernel<<<launch_blocks(keys.numel()),
                                 kThreads,
                                 0,
                                 current_musa_stream(cache.device())>>>(
        cache.data_ptr<int8_t>(),
        scales.data_ptr<c10::Half>(),
        table,
        destination,
        valid_slots.data_ptr<bool>(),
        keys.numel(),
        keys.size(1),
        blocks.size(1),
        cache.size(1),
        cache.size(0));
  };
  auto dispatch_table = [&](auto* destination) {
    if (blocks.scalar_type() == torch::kInt32) {
      launch(blocks.data_ptr<int32_t>(), destination);
    } else {
      launch(blocks.data_ptr<int64_t>(), destination);
    }
  };
  if (output_dtype == torch::kBFloat16) {
    dispatch_table(keys.data_ptr<c10::BFloat16>());
  } else {
    CHECK_EQ(output_dtype, torch::kFloat32);
    dispatch_table(keys.data_ptr<float>());
  }
  check_musa_kernel_launch("dsv4 Indexer fused paged gather");
  if (tiled && keys.size(0) == 8) {
    LOG_FIRST_N(INFO, 1) << "[DSV4_TILED_INDEXER_GATHER_B8] capacity="
                         << keys.size(1)
                         << " page=128 float_rounding_unchanged=1";
  }
}

}  // namespace

void deepseek_v4_indexer_gather_keys_fp32(const torch::Tensor& cache,
                                          const torch::Tensor& scales,
                                          const torch::Tensor& blocks,
                                          const torch::Tensor& keys,
                                          const torch::Tensor& valid_slots) {
  gather_keys(cache, scales, blocks, keys, valid_slots, torch::kFloat32);
}

void deepseek_v4_indexer_gather_keys_bf16(const torch::Tensor& cache,
                                          const torch::Tensor& scales,
                                          const torch::Tensor& blocks,
                                          const torch::Tensor& keys,
                                          const torch::Tensor& valid_slots) {
  gather_keys(cache, scales, blocks, keys, valid_slots, torch::kBFloat16);
}

torch::Tensor deepseek_v4_indexer_weighted_sum(const torch::Tensor& scores,
                                               const torch::Tensor& weights) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  check_tensor(scores, scores, torch::kFloat32);
  check_tensor(weights, scores, torch::kFloat32);
  CHECK_EQ(scores.dim(), 4);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(scores.size(2), 64);
  CHECK_GT(scores.numel(), 0);
  CHECK_EQ(scores.size(3) % 4, 0);
  CHECK_LE(scores.numel(), std::numeric_limits<int32_t>::max());
  CHECK_EQ(weights.size(0), scores.size(0) * scores.size(1));
  CHECK_EQ(weights.size(1), scores.size(2));
  CHECK_EQ(
      reinterpret_cast<uintptr_t>(scores.data_ptr<float>()) % alignof(float4),
      0);
  const XLLMOptionalDeviceGuard guard(scores.device());
  torch::Tensor output = torch::empty(
      {scores.size(0) * scores.size(1), scores.size(3)}, scores.options());
  const int64_t key_tiles = (scores.size(3) / 4 + 31) / 32;
  const int64_t total_tiles = output.size(0) * key_tiles;
  const int32_t blocks =
      static_cast<int32_t>(std::min<int64_t>(total_tiles, 4096));
  indexer_weighted_head_sum_kernel<<<blocks,
                                     dim3(32, 8),
                                     0,
                                     current_musa_stream(scores.device())>>>(
      reinterpret_cast<const float4*>(scores.data_ptr<float>()),
      weights.data_ptr<float>(),
      reinterpret_cast<float4*>(output.data_ptr<float>()),
      static_cast<int32_t>(total_tiles),
      static_cast<int32_t>(key_tiles),
      static_cast<int32_t>(scores.size(3) / 4));
  check_musa_kernel_launch("dsv4 Indexer fused weighted head sum");
  return output;
}

void deepseek_v4_indexer_relu_weight(const torch::Tensor& scores,
                                     const torch::Tensor& weights) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  check_tensor(scores, scores, torch::kFloat32);
  check_tensor(weights, scores, torch::kFloat32);
  CHECK_EQ(scores.dim(), 4);
  CHECK_EQ(weights.dim(), 2);
  CHECK_GT(scores.numel(), 0);
  CHECK_LE(scores.numel(),
           std::numeric_limits<int32_t>::max() -
               static_cast<int64_t>(4096) * kThreads);
  CHECK_EQ(weights.size(0), scores.size(0) * scores.size(1));
  CHECK_EQ(weights.size(1), scores.size(2));
  const XLLMOptionalDeviceGuard guard(scores.device());
  if (scores.size(3) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(scores.data_ptr<float>()) % alignof(float4) ==
          0) {
    if (scores.size(3) == 4096) {
      launch_indexer_relu_weight_vector<4096>(scores, weights);
    } else if (scores.size(3) == 8192) {
      launch_indexer_relu_weight_vector<8192>(scores, weights);
    } else {
      launch_indexer_relu_weight_vector<0>(scores, weights);
    }
    check_musa_kernel_launch("dsv4 Indexer vector ReLU head weight");
    return;
  }
  indexer_relu_weight_kernel<<<launch_blocks(scores.numel()),
                               kThreads,
                               0,
                               current_musa_stream(scores.device())>>>(
      scores.data_ptr<float>(),
      weights.data_ptr<float>(),
      static_cast<int32_t>(scores.numel()),
      static_cast<int32_t>(scores.size(3)));
  check_musa_kernel_launch("dsv4 Indexer ReLU head weight");
}

void deepseek_v4_indexer_mask_scores(const torch::Tensor& scores,
                                     const torch::Tensor& lengths,
                                     const torch::Tensor& valid_slots,
                                     int64_t query_rows_per_sequence) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  check_tensor(scores, scores, torch::kFloat32);
  CHECK_EQ(scores.dim(), 2);
  CHECK_GT(scores.size(0), 0);
  CHECK_GT(scores.size(1), 0);
  CHECK_GT(query_rows_per_sequence, 0);
  CHECK_EQ(scores.size(0) % query_rows_per_sequence, 0);
  check_tensor(lengths, scores, torch::kInt32);
  CHECK_EQ(lengths.numel(), scores.size(0));
  check_tensor(valid_slots, scores, torch::kBool);
  CHECK_EQ(valid_slots.dim(), 2);
  CHECK_EQ(valid_slots.size(0), scores.size(0) / query_rows_per_sequence);
  CHECK_EQ(valid_slots.size(1), scores.size(1));
  const XLLMOptionalDeviceGuard guard(scores.device());
  if (tiled_metadata_enabled() && scores.size(0) <= 112 &&
      scores.size(1) <= 65536) {
    indexer_mask_scores_tiled_kernel<<<
        dim3(static_cast<uint32_t>(launch_blocks(scores.size(1))),
             static_cast<uint32_t>(scores.size(0))),
        kThreads,
        0,
        current_musa_stream(scores.device())>>>(
        scores.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        valid_slots.data_ptr<bool>(),
        static_cast<int32_t>(scores.size(1)),
        static_cast<int32_t>(query_rows_per_sequence));
    check_musa_kernel_launch("dsv4 Indexer tiled score mask");
    return;
  }
  indexer_mask_scores_kernel<<<launch_blocks(scores.numel()),
                               kThreads,
                               0,
                               current_musa_stream(scores.device())>>>(
      scores.data_ptr<float>(),
      lengths.data_ptr<int32_t>(),
      valid_slots.data_ptr<bool>(),
      scores.numel(),
      scores.size(1),
      query_rows_per_sequence);
  check_musa_kernel_launch("dsv4 Indexer fused score mask");
}

namespace {

constexpr int32_t kPrefillTopK512 = 512;
constexpr int32_t kPrefillRadixBins = 256;
constexpr int32_t kPrefillHistGroups = 4;
constexpr int32_t kPrefillMaxRegChunks = 8;  // 512 * 8 = 4096 keys in regs

__device__ __forceinline__ uint32_t prefill_float_to_ordered_u32(float v) {
  if (isnan(v)) {
    return 0u;
  }
  if (v == 0.0f) {
    return 0x80000000u;
  }
  const uint32_t bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__global__ void __launch_bounds__(kPrefillTopK512)
indexer_prefill_causal_topk512_kernel(
    const float* __restrict__ scores,
    int64_t row_stride,
    int64_t absolute_start,
    int32_t keys,
    int32_t* __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  int32_t* out_row = output + static_cast<int64_t>(row) * kPrefillTopK512;

  const int64_t num = absolute_start + static_cast<int64_t>(row) + 1;
  const int32_t valid =
      num <= 0 ? 0 : static_cast<int32_t>(num / 4 < keys ? num / 4 : keys);

  if (valid <= kPrefillTopK512) {
    out_row[tid] = tid < valid ? tid : -1;
    return;
  }

  const float* row_scores = scores + static_cast<int64_t>(row) * row_stride;

  __shared__ uint32_t s_group_hist[kPrefillHistGroups][kPrefillRadixBins];
  __shared__ uint32_t s_hist[kPrefillRadixBins];
  __shared__ uint32_t s_prefix;
  __shared__ uint32_t s_needed;
  __shared__ uint32_t s_exact_done;
  __shared__ uint32_t s_gt_pos;
  __shared__ uint32_t s_eq_pos;

  if (tid == 0) {
    s_prefix = 0u;
    s_needed = static_cast<uint32_t>(kPrefillTopK512);
    s_exact_done = 0u;
    s_gt_pos = 0u;
    s_eq_pos = 0u;
  }

  const bool use_reg_cache = (valid <= kPrefillTopK512 * kPrefillMaxRegChunks);
  uint32_t reg_keys[kPrefillMaxRegChunks];
  if (use_reg_cache) {
#pragma unroll
    for (int32_t c = 0; c < kPrefillMaxRegChunks; ++c) {
      const int32_t col = tid + c * kPrefillTopK512;
      if (col < valid) {
        reg_keys[c] = prefill_float_to_ordered_u32(row_scores[col]);
      }
    }
  }

  const int32_t hist_group = tid & (kPrefillHistGroups - 1);
  uint32_t mask = 0u;

#pragma unroll
  for (int32_t pass = 0; pass < 4; ++pass) {
    const int32_t shift = 24 - pass * 8;
    s_group_hist[tid >> 8][tid & 255] = 0u;
    s_group_hist[(tid >> 8) + 2][tid & 255] = 0u;
    __syncthreads();

    const uint32_t prefix = s_prefix;
    if (use_reg_cache) {
#pragma unroll
      for (int32_t c = 0; c < kPrefillMaxRegChunks; ++c) {
        const int32_t col = tid + c * kPrefillTopK512;
        if (col < valid) {
          const uint32_t k = reg_keys[c];
          if ((k & mask) == prefix) {
            atomicAdd(&s_group_hist[hist_group][(k >> shift) & 0xffu], 1u);
          }
        }
      }
    } else {
      for (int32_t col = tid; col < valid; col += kPrefillTopK512) {
        const uint32_t k = prefill_float_to_ordered_u32(row_scores[col]);
        if ((k & mask) == prefix) {
          atomicAdd(&s_group_hist[hist_group][(k >> shift) & 0xffu], 1u);
        }
      }
    }
    __syncthreads();

    if (tid < kPrefillRadixBins) {
      s_hist[tid] = s_group_hist[0][tid] + s_group_hist[1][tid] +
                    s_group_hist[2][tid] + s_group_hist[3][tid];
    }
    __syncthreads();

    if (tid == 0) {
      uint32_t acc = 0u;
      const uint32_t needed = s_needed;
      uint32_t chosen_bin = 0u;
      for (int32_t b = kPrefillRadixBins - 1; b >= 0; --b) {
        const uint32_t cnt = s_hist[b];
        if (acc + cnt >= needed) {
          chosen_bin = static_cast<uint32_t>(b);
          break;
        }
        acc += cnt;
      }
      s_prefix = prefix | (chosen_bin << shift);
      const uint32_t remaining = needed - acc;
      if (s_hist[chosen_bin] == remaining) {
        s_exact_done = 1u;
        s_needed = 0u;
      } else {
        s_needed = remaining;
      }
    }
    __syncthreads();

    if (s_exact_done != 0u) {
      break;
    }
    mask |= (0xffu << shift);
  }

  if (tid == 0) {
    s_gt_pos = 0u;
    s_eq_pos = static_cast<uint32_t>(kPrefillTopK512) - s_needed;
  }
  __syncthreads();

  const uint32_t threshold = s_prefix;
  const bool exact_ge = (s_exact_done != 0u);

  if (use_reg_cache) {
#pragma unroll
    for (int32_t c = 0; c < kPrefillMaxRegChunks; ++c) {
      const int32_t col = tid + c * kPrefillTopK512;
      if (col < valid) {
        const uint32_t k = reg_keys[c];
        if (exact_ge) {
          if (k >= threshold) {
            const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
            if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
              out_row[pos] = col;
            }
          }
        } else {
          if (k > threshold) {
            const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
            if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
              out_row[pos] = col;
            }
          } else if (k == threshold) {
            const uint32_t pos = atomicAdd(&s_eq_pos, 1u);
            if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
              out_row[pos] = col;
            }
          }
        }
      }
    }
  } else {
    for (int32_t col = tid; col < valid; col += kPrefillTopK512) {
      const uint32_t k = prefill_float_to_ordered_u32(row_scores[col]);
      if (exact_ge) {
        if (k >= threshold) {
          const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
          if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
            out_row[pos] = col;
          }
        }
      } else {
        if (k > threshold) {
          const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
          if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
            out_row[pos] = col;
          }
        } else if (k == threshold) {
          const uint32_t pos = atomicAdd(&s_eq_pos, 1u);
          if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
            out_row[pos] = col;
          }
        }
      }
    }
  }
}

constexpr int32_t kMqaSubwarpThreads = 16;
constexpr int32_t kMqaItemsPerThread = 8;
constexpr int32_t kMqaRopeDim = 64;
constexpr float kMqaHadamardScale = 0.08838834764831845f;
constexpr float kMqaFp8Max = 448.0f;
constexpr float kMqaScaleMin = 1.0e-6f;
constexpr char kMateMqaLogitsOutUri[] =
    "mate_fp8_mqa_logits_h64_d128_q2_kv128_full_out";

template <bool ApplyRope, typename WeightT, int32_t HeadsPerBlock>
__global__ void indexer_q_rope_hadamard_fp8_quant_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ cos,
    const c10::BFloat16* __restrict__ sin,
    const WeightT* __restrict__ raw_weights,
    float score_scale,
    int64_t q_token_stride,
    int64_t q_head_stride,
    int64_t cos_stride,
    int64_t w_row_stride,
    int64_t absolute_start,
    int64_t kv_tokens,
    c10::Float8_e4m3fn* __restrict__ q_fp8,
    float* __restrict__ w_scaled,
    int32_t* __restrict__ ks,
    int32_t* __restrict__ ke) {
  const int64_t token = static_cast<int64_t>(blockIdx.y);
  const int32_t head_lane = static_cast<int32_t>(threadIdx.x) / kMqaSubwarpThreads;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kMqaSubwarpThreads;
  const int64_t head = static_cast<int64_t>(blockIdx.x) * HeadsPerBlock + head_lane;
  const int32_t base = lane * kMqaItemsPerThread;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ks[token] = 0;
    const int64_t num = absolute_start + token + 1;
    const int32_t valid =
        num <= 0 ? 0
                 : static_cast<int32_t>(num / 4 < kv_tokens ? num / 4 : kv_tokens);
    ke[token] = valid;
  }

  const c10::BFloat16* q_row =
      query + token * q_token_stride + head * q_head_stride + base;
  float value[kMqaItemsPerThread];
#pragma unroll
  for (int32_t item = 0; item < kMqaItemsPerThread; ++item) {
    value[item] = static_cast<float>(q_row[item]);
  }

  if constexpr (ApplyRope) {
    if (base >= kMqaRopeDim) {
      const c10::BFloat16* cos_row = cos + token * cos_stride;
      const c10::BFloat16* sin_row = sin + token * cos_stride;
#pragma unroll
      for (int32_t item = 0; item < kMqaItemsPerThread; item += 2) {
        const int32_t rope_col = base + item - kMqaRopeDim;
        const float even = value[item];
        const float odd = value[item + 1];
        const float c = static_cast<float>(cos_row[rope_col]);
        const float s = static_cast<float>(sin_row[rope_col]);
        value[item] = even * c - odd * s;
        value[item + 1] = even * s + odd * c;
      }
    }
  }

  // Sylvester H128: 3 local stages + 4 subwarp shuffle stages
  for (int32_t step = 1; step < kMqaItemsPerThread; step <<= 1) {
#pragma unroll
    for (int32_t group = 0; group < kMqaItemsPerThread; group += 2 * step) {
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
  for (int32_t stage = 0; stage < 4; ++stage) {
    const int32_t lane_mask = 1 << stage;
    const float sign = (lane & lane_mask) == 0 ? 1.0f : -1.0f;
#pragma unroll
    for (int32_t item = 0; item < kMqaItemsPerThread; ++item) {
      const float other =
          __shfl_xor_sync(0xffffffffu, value[item], lane_mask, kMqaSubwarpThreads);
      value[item] = sign * value[item] + other;
    }
  }

  float max_abs = 0.0f;
#pragma unroll
  for (int32_t item = 0; item < kMqaItemsPerThread; ++item) {
    value[item] *= kMqaHadamardScale;
    max_abs = fmaxf(max_abs, fabsf(value[item]));
  }
#pragma unroll
  for (int32_t stage = 0; stage < 4; ++stage) {
    const float other =
        __shfl_xor_sync(0xffffffffu, max_abs, 1 << stage, kMqaSubwarpThreads);
    max_abs = fmaxf(max_abs, other);
  }

  const float q_scale = fmaxf(max_abs / kMqaFp8Max, kMqaScaleMin);
  const float inv_scale = 1.0f / q_scale;
  const float4 s0 = make_float4(value[0] * inv_scale,
                                value[1] * inv_scale,
                                value[2] * inv_scale,
                                value[3] * inv_scale);
  const float4 s1 = make_float4(value[4] * inv_scale,
                                value[5] * inv_scale,
                                value[6] * inv_scale,
                                value[7] * inv_scale);
  const uint32_t p0 = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(s0, __MT_SATFINITE, __MT_E4M3));
  const uint32_t p1 = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(s1, __MT_SATFINITE, __MT_E4M3));
  uint32_t* out_u32 = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(q_fp8) + (token * 64 + head) * kHeadDim + base);
  out_u32[0] = p0;
  out_u32[1] = p1;

  if (lane == 0) {
    w_scaled[token * 64 + head] =
        static_cast<float>(raw_weights[token * w_row_stride + head]) *
        score_scale * q_scale;
  }
}

template <int32_t RowsPerBlock>
__global__ void indexer_k_fp8_quant_kernel(
    const c10::BFloat16* __restrict__ key,
    c10::Float8_e4m3fn* __restrict__ k_fp8,
    float* __restrict__ k_scale,
    int64_t kv_tokens) {
  const int32_t row_lane = static_cast<int32_t>(threadIdx.x) / kMqaSubwarpThreads;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kMqaSubwarpThreads;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * RowsPerBlock + row_lane;
  const int32_t base = lane * kMqaItemsPerThread;
  const bool valid = row < kv_tokens;

  float value[kMqaItemsPerThread];
  float max_abs = 0.0f;
  if (valid) {
    const uint32_t* in_u32 =
        reinterpret_cast<const uint32_t*>(key + row * kHeadDim + base);
#pragma unroll
    for (int32_t i = 0; i < kMqaItemsPerThread / 2; ++i) {
      const uint32_t packed = in_u32[i];
      c10::BFloat16 lo, hi;
      lo.x = static_cast<uint16_t>(packed & 0xffffu);
      hi.x = static_cast<uint16_t>(packed >> 16);
      const float v0 = static_cast<float>(lo);
      const float v1 = static_cast<float>(hi);
      value[i * 2] = v0;
      value[i * 2 + 1] = v1;
      max_abs = fmaxf(max_abs, fmaxf(fabsf(v0), fabsf(v1)));
    }
  } else {
#pragma unroll
    for (int32_t i = 0; i < kMqaItemsPerThread; ++i) {
      value[i] = 0.0f;
    }
  }

#pragma unroll
  for (int32_t stage = 0; stage < 4; ++stage) {
    const float other =
        __shfl_xor_sync(0xffffffffu, max_abs, 1 << stage, kMqaSubwarpThreads);
    max_abs = fmaxf(max_abs, other);
  }

  if (valid) {
    const float scale = fmaxf(max_abs / kMqaFp8Max, kMqaScaleMin);
    const float inv_scale = 1.0f / scale;
    const float4 s0 = make_float4(value[0] * inv_scale,
                                  value[1] * inv_scale,
                                  value[2] * inv_scale,
                                  value[3] * inv_scale);
    const float4 s1 = make_float4(value[4] * inv_scale,
                                  value[5] * inv_scale,
                                  value[6] * inv_scale,
                                  value[7] * inv_scale);
    const uint32_t p0 = static_cast<uint32_t>(
        __musa_cvt_float4_to_fp8x4(s0, __MT_SATFINITE, __MT_E4M3));
    const uint32_t p1 = static_cast<uint32_t>(
        __musa_cvt_float4_to_fp8x4(s1, __MT_SATFINITE, __MT_E4M3));
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uint8_t*>(k_fp8) + row * kHeadDim + base);
    out_u32[0] = p0;
    out_u32[1] = p1;
    if (lane == 0) {
      k_scale[row] = scale;
    }
  }
}

struct MqaLogitsWorkspace {
  int64_t cap_q = 0;
  int64_t cap_kv = 0;
  int64_t cap_logits_rows = 0;
  int64_t cap_logits_cols = 0;
  torch::Tensor q_fp8;
  torch::Tensor w_scaled;
  torch::Tensor ks;
  torch::Tensor ke;
  torch::Tensor k_fp8;
  torch::Tensor k_scale;
  torch::Tensor logits_storage;
};

MqaLogitsWorkspace& get_mqa_logits_workspace(const torch::Device& device,
                                             int64_t q_tokens,
                                             int64_t kv_tokens) {
  static thread_local std::unordered_map<int32_t, MqaLogitsWorkspace> map;
  auto& ws = map[static_cast<int32_t>(device.index())];
  const auto opts = torch::TensorOptions().device(device);
  if (ws.cap_q < q_tokens || !ws.q_fp8.defined()) {
    const int64_t cap = std::max<int64_t>(q_tokens, ws.cap_q > 0 ? ws.cap_q * 2 : 1);
    ws.cap_q = cap;
    ws.q_fp8 = torch::empty({cap, 64, kHeadDim}, opts.dtype(torch::kFloat8_e4m3fn));
    ws.w_scaled = torch::empty({cap, 64}, opts.dtype(torch::kFloat32));
    ws.ks = torch::empty({cap}, opts.dtype(torch::kInt32));
    ws.ke = torch::empty({cap}, opts.dtype(torch::kInt32));
  }
  if (ws.cap_kv < kv_tokens || !ws.k_fp8.defined()) {
    const int64_t cap = std::max<int64_t>(kv_tokens, ws.cap_kv > 0 ? ws.cap_kv * 2 : 1);
    ws.cap_kv = cap;
    ws.k_fp8 = torch::empty({cap, kHeadDim}, opts.dtype(torch::kFloat8_e4m3fn));
    ws.k_scale = torch::empty({cap}, opts.dtype(torch::kFloat32));
  }
  const int64_t req_rows = ((q_tokens + 1) / 2) * 2;
  const int64_t req_cols = ((kv_tokens + 128 + 7) / 8) * 8;
  if (ws.cap_logits_rows < req_rows || ws.cap_logits_cols < req_cols ||
      !ws.logits_storage.defined()) {
    ws.cap_logits_rows = std::max<int64_t>(req_rows, ws.cap_logits_rows);
    ws.cap_logits_cols = std::max<int64_t>(req_cols, ws.cap_logits_cols);
    ws.logits_storage = torch::empty(
        {ws.cap_logits_rows, ws.cap_logits_cols}, opts.dtype(torch::kFloat32));
  }
  return ws;
}

ffi::Optional<ffi::Function> get_mate_mqa_logits_out_fn() {
  static std::once_flag once;
  static ffi::Optional<ffi::Function> cached_fn;
  std::call_once(once, []() {
    const char* root = std::getenv("FLASHINFER_OPS_PATH");
    if (root == nullptr || root[0] == 0) {
      return;
    }
    std::ifstream so_file(path_to_uri_so_lib(kMateMqaLogitsOutUri));
    if (!so_file.good()) {
      return;
    }
    try {
      cached_fn = get_module(kMateMqaLogitsOutUri)->GetFunction("fp8_mqa_logits_out");
    } catch (const std::exception& e) {
      LOG(WARNING) << "[DSV4_PREFILL_FP8_MQA_LOGITS] failed loading "
                   << kMateMqaLogitsOutUri << ": " << e.what();
    }
  });
  return cached_fn;
}

}  // namespace

void deepseek_v4_indexer_prefill_causal_topk512(
    const torch::Tensor& scores,
    int64_t absolute_start,
    torch::Tensor output) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(scores.dim(), 2);
  CHECK_GT(scores.size(0), 0);
  CHECK_GT(scores.size(1), 0);
  CHECK_EQ(scores.stride(1), 1);
  CHECK_GE(scores.stride(0), scores.size(1));
  check_tensor(output, scores, torch::kInt32);
  CHECK_EQ(output.dim(), 2);
  CHECK_EQ(output.size(0), scores.size(0));
  CHECK_EQ(output.size(1), kPrefillTopK512);

  const XLLMOptionalDeviceGuard guard(scores.device());
  indexer_prefill_causal_topk512_kernel<<<
      static_cast<uint32_t>(scores.size(0)),
      kPrefillTopK512,
      0,
      current_musa_stream(scores.device())>>>(
      scores.data_ptr<float>(),
      scores.stride(0),
      absolute_start,
      static_cast<int32_t>(scores.size(1)),
      output.data_ptr<int32_t>());
  check_musa_kernel_launch("dsv4 Indexer prefill causal topk512");
}

torch::Tensor deepseek_v4_indexer_prefill_fp8_mqa_topk512(
    const torch::Tensor& query_raw,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& key,
    const torch::Tensor& raw_head_weights,
    float score_scale,
    int64_t absolute_start) {
  if (!query_raw.defined() || !key.defined() || !raw_head_weights.defined() ||
      !query_raw.device().is_privateuseone() ||
      query_raw.device() != key.device() ||
      query_raw.device() != raw_head_weights.device() ||
      query_raw.dim() != 3 || key.dim() != 2 || raw_head_weights.dim() != 2 ||
      query_raw.size(1) != 64 || query_raw.size(2) != kHeadDim ||
      key.size(1) != kHeadDim || raw_head_weights.size(0) != query_raw.size(0) ||
      raw_head_weights.size(1) != 64 || query_raw.size(0) <= 0 ||
      key.size(0) <= 0 || absolute_start < 0 ||
      query_raw.scalar_type() != torch::kBFloat16 ||
      key.scalar_type() != torch::kBFloat16 ||
      (raw_head_weights.scalar_type() != torch::kBFloat16 &&
       raw_head_weights.scalar_type() != torch::kFloat32) ||
      query_raw.stride(-1) != 1 || raw_head_weights.stride(-1) != 1) {
    return torch::Tensor();
  }
  const bool apply_rope =
      cos.defined() && sin.defined() && cos.dim() == 2 && sin.dim() == 2 &&
      cos.size(0) >= query_raw.size(0) && sin.size(0) >= query_raw.size(0) &&
      cos.size(1) >= kMqaRopeDim && sin.size(1) >= kMqaRopeDim &&
      cos.scalar_type() == torch::kBFloat16 &&
      sin.scalar_type() == torch::kBFloat16 && cos.device() == query_raw.device() &&
      sin.device() == query_raw.device() && cos.is_contiguous() &&
      sin.is_contiguous();
  if ((cos.defined() || sin.defined()) && !apply_rope) {
    return torch::Tensor();
  }

  const int64_t q_tokens = query_raw.size(0);
  const int64_t kv_tokens = key.size(0);
  auto selected = torch::empty(
      {q_tokens, kPrefillTopK512}, query_raw.options().dtype(torch::kInt32));

  // Fast path: if every query row has valid <= 512, topk512 returns [0..valid-1, -1...]
  // directly without scoring at all.
  const int64_t max_valid =
      std::min<int64_t>(kv_tokens, (absolute_start + q_tokens) / 4);
  if (max_valid <= kPrefillTopK512) {
    const XLLMOptionalDeviceGuard guard(query_raw.device());
    indexer_prefill_causal_topk512_kernel<<<
        static_cast<uint32_t>(q_tokens),
        kPrefillTopK512,
        0,
        current_musa_stream(query_raw.device())>>>(
        nullptr,
        0,
        absolute_start,
        static_cast<int32_t>(kv_tokens),
        selected.data_ptr<int32_t>());
    check_musa_kernel_launch("dsv4 Indexer prefill causal topk512 short-circuit");
    return selected;
  }

  auto mqa_fn = get_mate_mqa_logits_out_fn();
  if (!mqa_fn.defined()) {
    return torch::Tensor();
  }

  const torch::Tensor key_contig = key.is_contiguous() ? key : key.contiguous();
  auto& ws = get_mqa_logits_workspace(query_raw.device(), q_tokens, kv_tokens);
  auto q_fp8 = ws.q_fp8.narrow(/*dim=*/0, 0, q_tokens);
  auto w_scaled = ws.w_scaled.narrow(/*dim=*/0, 0, q_tokens);
  auto ks = ws.ks.narrow(/*dim=*/0, 0, q_tokens);
  auto ke = ws.ke.narrow(/*dim=*/0, 0, q_tokens);
  auto k_fp8 = ws.k_fp8.narrow(/*dim=*/0, 0, kv_tokens);
  auto k_scale = ws.k_scale.narrow(/*dim=*/0, 0, kv_tokens);
  const int64_t aligned_q = ((q_tokens + 1) / 2) * 2;
  auto logits_storage = ws.logits_storage.narrow(/*dim=*/0, 0, aligned_q);

  {
    const XLLMOptionalDeviceGuard guard(query_raw.device());
    const musaStream_t stream = current_musa_stream(query_raw.device());
    constexpr int32_t kHeadsPerBlock = 8;
    const dim3 q_grid(static_cast<unsigned int>(64 / kHeadsPerBlock),
                      static_cast<unsigned int>(q_tokens));
    const c10::BFloat16* cos_ptr =
        apply_rope ? cos.data_ptr<c10::BFloat16>() : nullptr;
    const c10::BFloat16* sin_ptr =
        apply_rope ? sin.data_ptr<c10::BFloat16>() : nullptr;
    const int64_t cos_stride = apply_rope ? cos.size(1) : 0;

    auto launch_q = [&](auto apply_rope_tag, auto* w_ptr) {
      using WeightT = std::remove_const_t<std::remove_pointer_t<decltype(w_ptr)>>;
      indexer_q_rope_hadamard_fp8_quant_kernel<decltype(apply_rope_tag)::value,
                                               WeightT,
                                               kHeadsPerBlock>
          <<<q_grid, kMqaSubwarpThreads * kHeadsPerBlock, 0, stream>>>(
              query_raw.data_ptr<c10::BFloat16>(),
              cos_ptr,
              sin_ptr,
              w_ptr,
              score_scale,
              query_raw.stride(0),
              query_raw.stride(1),
              cos_stride,
              raw_head_weights.stride(0),
              absolute_start,
              kv_tokens,
              q_fp8.data_ptr<c10::Float8_e4m3fn>(),
              w_scaled.data_ptr<float>(),
              ks.data_ptr<int32_t>(),
              ke.data_ptr<int32_t>());
    };

    if (raw_head_weights.scalar_type() == torch::kBFloat16) {
      if (apply_rope) {
        launch_q(std::true_type{}, raw_head_weights.data_ptr<c10::BFloat16>());
      } else {
        launch_q(std::false_type{}, raw_head_weights.data_ptr<c10::BFloat16>());
      }
    } else {
      if (apply_rope) {
        launch_q(std::true_type{}, raw_head_weights.data_ptr<float>());
      } else {
        launch_q(std::false_type{}, raw_head_weights.data_ptr<float>());
      }
    }
    check_musa_kernel_launch("dsv4 Indexer Q rope+hadamard+fp8 quant");

    constexpr int32_t kRowsPerBlock = 8;
    const int64_t k_blocks = (kv_tokens + kRowsPerBlock - 1) / kRowsPerBlock;
    indexer_k_fp8_quant_kernel<kRowsPerBlock>
        <<<static_cast<unsigned int>(k_blocks),
           kMqaSubwarpThreads * kRowsPerBlock,
           0,
           stream>>>(key_contig.data_ptr<c10::BFloat16>(),
                     k_fp8.data_ptr<c10::Float8_e4m3fn>(),
                     k_scale.data_ptr<float>(),
                     kv_tokens);
    check_musa_kernel_launch("dsv4 Indexer K fp8 quant");
  }

  try {
    TvmffiStreamGuard stream_guard(query_raw.device());
    mqa_fn.value()(to_ffi_tensor(q_fp8),
                   to_ffi_tensor(k_fp8),
                   to_ffi_tensor(w_scaled),
                   to_ffi_tensor(ks),
                   to_ffi_tensor(ke),
                   to_ffi_tensor(k_scale),
                   to_ffi_tensor(logits_storage));
  } catch (const std::exception& e) {
    LOG_FIRST_N(WARNING, 2)
        << "[DSV4_PREFILL_FP8_MQA_LOGITS] launch failed, falling back: "
        << e.what();
    return torch::Tensor();
  }

  {
    const XLLMOptionalDeviceGuard guard(query_raw.device());
    indexer_prefill_causal_topk512_kernel<<<
        static_cast<uint32_t>(q_tokens),
        kPrefillTopK512,
        0,
        current_musa_stream(query_raw.device())>>>(
        logits_storage.data_ptr<float>(),
        logits_storage.stride(0),
        absolute_start,
        static_cast<int32_t>(kv_tokens),
        selected.data_ptr<int32_t>());
    check_musa_kernel_launch("dsv4 Indexer prefill fp8 mqa topk512");
  }

  LOG_FIRST_N(INFO, 2) << "[DSV4_PREFILL_FP8_MQA_LOGITS] queries=" << q_tokens
                       << " keys=" << kv_tokens
                       << " absolute_start=" << absolute_start
                       << " logits_stride=" << logits_storage.stride(0);
  return selected;
}

namespace {

template <typename WeightT, int32_t HeadsPerBlock>
__global__ void indexer_decode_q_fp8_quant_kernel(
    const c10::BFloat16* __restrict__ query,
    const WeightT* __restrict__ weights,
    const int32_t* __restrict__ lengths,
    int64_t capacity,
    c10::Float8_e4m3fn* __restrict__ q_fp8,
    float* __restrict__ w_scaled,
    int32_t* __restrict__ ks,
    int32_t* __restrict__ ke) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int32_t head_lane =
      static_cast<int32_t>(threadIdx.x) / kMqaSubwarpThreads;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kMqaSubwarpThreads;
  const int64_t head =
      static_cast<int64_t>(blockIdx.x) * HeadsPerBlock + head_lane;
  const int32_t base = lane * kMqaItemsPerThread;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ks[row] = 0;
    const int32_t live = lengths[row] > 0 ? lengths[row] : 0;
    const int32_t valid =
        live / 4 < capacity ? live / 4 : static_cast<int32_t>(capacity);
    ke[row] = valid;
  }

  const c10::BFloat16* q_row = query + (row * 64 + head) * kHeadDim + base;
  float value[kMqaItemsPerThread];
  float max_abs = 0.0f;
#pragma unroll
  for (int32_t item = 0; item < kMqaItemsPerThread; ++item) {
    const float v = static_cast<float>(q_row[item]);
    value[item] = v;
    max_abs = fmaxf(max_abs, fabsf(v));
  }
#pragma unroll
  for (int32_t stage = 0; stage < 4; ++stage) {
    const float other =
        __shfl_xor_sync(0xffffffffu, max_abs, 1 << stage, kMqaSubwarpThreads);
    max_abs = fmaxf(max_abs, other);
  }

  const float q_scale = fmaxf(max_abs / kMqaFp8Max, kMqaScaleMin);
  const float inv_scale = 1.0f / q_scale;
  const float4 s0 = make_float4(value[0] * inv_scale,
                                value[1] * inv_scale,
                                value[2] * inv_scale,
                                value[3] * inv_scale);
  const float4 s1 = make_float4(value[4] * inv_scale,
                                value[5] * inv_scale,
                                value[6] * inv_scale,
                                value[7] * inv_scale);
  const uint32_t p0 = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(s0, __MT_SATFINITE, __MT_E4M3));
  const uint32_t p1 = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(s1, __MT_SATFINITE, __MT_E4M3));
  uint32_t* out_u32 = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(q_fp8) + (row * 64 + head) * kHeadDim + base);
  out_u32[0] = p0;
  out_u32[1] = p1;

  if (lane == 0) {
    w_scaled[row * 64 + head] = static_cast<float>(weights[row * 64 + head]) * q_scale;
  }
}

template <typename BlockIndex>
__global__ void indexer_gather_keys_fp8_kernel(
    const int8_t* __restrict__ cache,
    const c10::Half* __restrict__ scales,
    const BlockIndex* __restrict__ blocks,
    c10::Float8_e4m3fn* __restrict__ k_fp8,
    float* __restrict__ k_scale,
    bool* __restrict__ valid_slots,
    int32_t capacity,
    int64_t table_columns,
    int64_t page,
    int64_t physical_pages) {
  const int64_t seq = static_cast<int64_t>(blockIdx.y);
  const int64_t total_slots = physical_pages * page;
  const int32_t total_vec4 = capacity * 32;  // 128 / 4 = 32 vec4s per key row
  uint32_t* out_u32_base = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(k_fp8) + seq * capacity * kHeadDim);
  for (int32_t offset =
           static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
       offset < total_vec4;
       offset += static_cast<int32_t>(gridDim.x * blockDim.x)) {
    const int32_t position = offset >> 5;
    const int32_t vec_idx = offset & 31;
    const int64_t table_col = position / page;
    const int64_t in_page = position - table_col * page;
    const int64_t block =
        static_cast<int64_t>(blocks[seq * table_columns + table_col]);
    const int64_t adjusted =
        block >= physical_pages && block > 0 ? block - 1 : block;
    const bool valid = adjusted >= 0 && adjusted < physical_pages;
    const int64_t slot = adjusted < 0 ? 0
                         : adjusted >= physical_pages
                             ? total_slots - 1
                             : adjusted * page + in_page;
    const uint32_t raw4 = *reinterpret_cast<const uint32_t*>(
        cache + slot * kHeadDim + vec_idx * 4);
    const int8_t c0 = static_cast<int8_t>(raw4 & 0xffu);
    const int8_t c1 = static_cast<int8_t>((raw4 >> 8) & 0xffu);
    const int8_t c2 = static_cast<int8_t>((raw4 >> 16) & 0xffu);
    const int8_t c3 = static_cast<int8_t>((raw4 >> 24) & 0xffu);
    const float4 vals = make_float4(static_cast<float>(c0),
                                    static_cast<float>(c1),
                                    static_cast<float>(c2),
                                    static_cast<float>(c3));
    out_u32_base[offset] = static_cast<uint32_t>(
        __musa_cvt_float4_to_fp8x4(vals, __MT_SATFINITE, __MT_E4M3));
    if (vec_idx == 0) {
      k_scale[seq * capacity + position] =
          valid ? static_cast<float>(scales[slot]) : 0.0f;
      valid_slots[seq * capacity + position] = valid;
    }
  }
}

__global__ void indexer_extract_mask_mqa_scores_kernel(
    const float* __restrict__ logits_storage,
    const int32_t* __restrict__ lengths,
    const bool* __restrict__ valid_slots,
    float* __restrict__ scores_out,
    int32_t capacity,
    int32_t query_rows_per_sequence,
    int32_t aligned_q,
    int64_t logits_stride) {
  const int32_t row = static_cast<int32_t>(blockIdx.y);
  const int32_t seq = row / query_rows_per_sequence;
  const int32_t q_in_seq = row - seq * query_rows_per_sequence;
  const int32_t live = lengths[row] > 0 ? lengths[row] : 0;
  const int32_t valid_len = live / 4;
  const float* src_row =
      logits_storage +
      (static_cast<int64_t>(seq) * aligned_q + q_in_seq) * logits_stride;
  float* dst_row = scores_out + static_cast<int64_t>(row) * capacity;
  const bool* seq_valid = valid_slots + static_cast<int64_t>(seq) * capacity;
  for (int32_t pos =
           static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
       pos < capacity;
       pos += static_cast<int32_t>(gridDim.x * blockDim.x)) {
    dst_row[pos] = (pos < valid_len && seq_valid[pos]) ? src_row[pos] : -INFINITY;
  }
}


__global__ void __launch_bounds__(kPrefillTopK512)
indexer_decode_topk512_kernel(
    const float* __restrict__ scores,
    const int32_t* __restrict__ lengths,
    int32_t capacity,
    int64_t* __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  int64_t* out_row = output + static_cast<int64_t>(row) * kPrefillTopK512;

  const int32_t live = lengths[row] > 0 ? lengths[row] : 0;
  const int32_t valid = live / 4 < capacity ? (live / 4) : capacity;

  if (valid <= kPrefillTopK512) {
    out_row[tid] = tid < valid ? static_cast<int64_t>(tid) : int64_t{0};
    return;
  }

  const float* row_scores = scores + static_cast<int64_t>(row) * capacity;

  __shared__ uint32_t s_group_hist[kPrefillHistGroups][kPrefillRadixBins];
  __shared__ uint32_t s_hist[kPrefillRadixBins];
  __shared__ uint32_t s_prefix;
  __shared__ uint32_t s_needed;
  __shared__ uint32_t s_exact_done;
  __shared__ uint32_t s_gt_pos;
  __shared__ uint32_t s_eq_pos;

  if (tid == 0) {
    s_prefix = 0u;
    s_needed = static_cast<uint32_t>(kPrefillTopK512);
    s_exact_done = 0u;
    s_gt_pos = 0u;
    s_eq_pos = 0u;
  }

  const bool use_reg_cache = (valid <= kPrefillTopK512 * kPrefillMaxRegChunks);
  uint32_t reg_keys[kPrefillMaxRegChunks];
  if (use_reg_cache) {
#pragma unroll
    for (int32_t c = 0; c < kPrefillMaxRegChunks; ++c) {
      const int32_t col = tid + c * kPrefillTopK512;
      if (col < valid) {
        reg_keys[c] = prefill_float_to_ordered_u32(row_scores[col]);
      }
    }
  }

  const int32_t hist_group = tid & (kPrefillHistGroups - 1);
  uint32_t mask = 0u;

#pragma unroll
  for (int32_t pass = 0; pass < 4; ++pass) {
    const int32_t shift = 24 - pass * 8;
    s_group_hist[tid >> 8][tid & 255] = 0u;
    s_group_hist[(tid >> 8) + 2][tid & 255] = 0u;
    __syncthreads();

    const uint32_t prefix = s_prefix;
    if (use_reg_cache) {
#pragma unroll
      for (int32_t c = 0; c < kPrefillMaxRegChunks; ++c) {
        const int32_t col = tid + c * kPrefillTopK512;
        if (col < valid) {
          const uint32_t k = reg_keys[c];
          if ((k & mask) == prefix) {
            atomicAdd(&s_group_hist[hist_group][(k >> shift) & 0xffu], 1u);
          }
        }
      }
    } else {
      for (int32_t col = tid; col < valid; col += kPrefillTopK512) {
        const uint32_t k = prefill_float_to_ordered_u32(row_scores[col]);
        if ((k & mask) == prefix) {
          atomicAdd(&s_group_hist[hist_group][(k >> shift) & 0xffu], 1u);
        }
      }
    }
    __syncthreads();

    if (tid < kPrefillRadixBins) {
      s_hist[tid] = s_group_hist[0][tid] + s_group_hist[1][tid] +
                    s_group_hist[2][tid] + s_group_hist[3][tid];
    }
    __syncthreads();

    if (tid == 0) {
      uint32_t acc = 0u;
      const uint32_t needed = s_needed;
      uint32_t chosen_bin = 0u;
      for (int32_t b = kPrefillRadixBins - 1; b >= 0; --b) {
        const uint32_t cnt = s_hist[b];
        if (acc + cnt >= needed) {
          chosen_bin = static_cast<uint32_t>(b);
          break;
        }
        acc += cnt;
      }
      s_prefix = prefix | (chosen_bin << shift);
      const uint32_t remaining = needed - acc;
      if (s_hist[chosen_bin] == remaining) {
        s_exact_done = 1u;
        s_needed = 0u;
      } else {
        s_needed = remaining;
      }
    }
    __syncthreads();

    if (s_exact_done != 0u) {
      break;
    }
    mask |= (0xffu << shift);
  }

  if (tid == 0) {
    s_gt_pos = 0u;
    s_eq_pos = static_cast<uint32_t>(kPrefillTopK512) - s_needed;
  }
  __syncthreads();

  const uint32_t threshold = s_prefix;
  const bool exact_ge = (s_exact_done != 0u);

  if (use_reg_cache) {
#pragma unroll
    for (int32_t c = 0; c < kPrefillMaxRegChunks; ++c) {
      const int32_t col = tid + c * kPrefillTopK512;
      if (col < valid) {
        const uint32_t k = reg_keys[c];
        if (exact_ge) {
          if (k >= threshold) {
            const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
            if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
              out_row[pos] = static_cast<int64_t>(col);
            }
          }
        } else {
          if (k > threshold) {
            const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
            if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
              out_row[pos] = static_cast<int64_t>(col);
            }
          } else if (k == threshold) {
            const uint32_t pos = atomicAdd(&s_eq_pos, 1u);
            if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
              out_row[pos] = static_cast<int64_t>(col);
            }
          }
        }
      }
    }
  } else {
    for (int32_t col = tid; col < valid; col += kPrefillTopK512) {
      const uint32_t k = prefill_float_to_ordered_u32(row_scores[col]);
      if (exact_ge) {
        if (k >= threshold) {
          const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
          if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
            out_row[pos] = static_cast<int64_t>(col);
          }
        }
      } else {
        if (k > threshold) {
          const uint32_t pos = atomicAdd(&s_gt_pos, 1u);
          if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
            out_row[pos] = static_cast<int64_t>(col);
          }
        } else if (k == threshold) {
          const uint32_t pos = atomicAdd(&s_eq_pos, 1u);
          if (pos < static_cast<uint32_t>(kPrefillTopK512)) {
            out_row[pos] = static_cast<int64_t>(col);
          }
        }
      }
    }
  }
}

}  // namespace


torch::Tensor deepseek_v4_indexer_decode_topk512(
    const torch::Tensor& scores,
    const torch::Tensor& lengths) {
  CHECK(scores.defined() && lengths.defined());
  CHECK(scores.device().is_privateuseone());
  CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(scores.dim(), 2);
  CHECK(scores.is_contiguous());
  CHECK_EQ(lengths.scalar_type(), torch::kInt32);
  CHECK(lengths.is_contiguous());
  CHECK_EQ(lengths.numel(), scores.size(0));
  const int64_t rows = scores.size(0);
  const int64_t capacity = scores.size(1);
  CHECK_GE(capacity, kPrefillTopK512);
  auto output = torch::empty(
      {rows, kPrefillTopK512}, scores.options().dtype(torch::kInt64));
  const XLLMOptionalDeviceGuard guard(scores.device());
  indexer_decode_topk512_kernel<<<static_cast<uint32_t>(rows),
                                  kPrefillTopK512,
                                  0,
                                  current_musa_stream(scores.device())>>>(
      scores.data_ptr<float>(),
      lengths.data_ptr<int32_t>(),
      static_cast<int32_t>(capacity),
      output.data_ptr<int64_t>());
  check_musa_kernel_launch("dsv4 decode indexer radix topk512");
  return output;
}

torch::Tensor deepseek_v4_indexer_decode_fp8_mqa_scores(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scales,
    const torch::Tensor& blocks,
    const torch::Tensor& lengths,
    int64_t query_rows_per_sequence,
    int64_t capacity) {
  if (!query.defined() || !weights.defined() || !cache.defined() ||
      !scales.defined() || !blocks.defined() || !lengths.defined() ||
      !query.device().is_privateuseone() || query.dim() != 3 ||
      query.size(1) != 64 || query.size(2) != kHeadDim ||
      query.scalar_type() != torch::kBFloat16 ||
      (weights.scalar_type() != torch::kFloat32 &&
       weights.scalar_type() != torch::kBFloat16) ||
      cache.scalar_type() != torch::kInt8 || !cache.is_contiguous() ||
      scales.scalar_type() != torch::kFloat16 || !scales.is_contiguous() ||
      lengths.scalar_type() != torch::kInt32 || !lengths.is_contiguous() ||
      !blocks.is_contiguous() ||
      (blocks.scalar_type() != torch::kInt32 &&
       blocks.scalar_type() != torch::kInt64) ||
      query_rows_per_sequence <= 0 ||
      query.size(0) % query_rows_per_sequence != 0 || capacity < kTopk) {
    return torch::Tensor();
  }
  auto mqa_fn = get_mate_mqa_logits_out_fn();
  if (!mqa_fn.defined()) {
    return torch::Tensor();
  }

  const int64_t total_q = query.size(0);
  const int64_t batch = total_q / query_rows_per_sequence;
  const int64_t physical_pages = cache.size(0);
  const int64_t page = cache.size(1);
  const int64_t table_columns = blocks.size(1);
  const int64_t aligned_q = ((query_rows_per_sequence + 1) / 2) * 2;
  const int64_t logits_stride = ((capacity + 128 + 7) / 8) * 8;

  const auto q_contig = query.is_contiguous() ? query : query.contiguous();
  const auto w_contig = weights.is_contiguous() ? weights : weights.contiguous();
  const auto opts = query.options();

  // Graph-local allocations prevent VMM scratch aliasing across captured graphs.
  auto q_fp8 = torch::empty({total_q, 64, kHeadDim}, opts.dtype(torch::kFloat8_e4m3fn));
  auto w_scaled = torch::empty({total_q, 64}, opts.dtype(torch::kFloat32));
  auto ks = torch::empty({total_q}, opts.dtype(torch::kInt32));
  auto ke = torch::empty({total_q}, opts.dtype(torch::kInt32));
  auto k_fp8 = torch::empty({batch, capacity, kHeadDim}, opts.dtype(torch::kFloat8_e4m3fn));
  auto k_scale = torch::empty({batch, capacity}, opts.dtype(torch::kFloat32));
  auto valid_slots = torch::empty({batch, capacity}, opts.dtype(torch::kBool));
  auto logits_storage =
      torch::empty({batch, aligned_q, logits_stride}, opts.dtype(torch::kFloat32));
  auto scores_out = torch::empty({total_q, capacity}, opts.dtype(torch::kFloat32));

  {
    const XLLMOptionalDeviceGuard guard(query.device());
    const musaStream_t stream = current_musa_stream(query.device());
    constexpr int32_t kHeadsPerBlock = 8;
    const dim3 q_grid(static_cast<unsigned int>(64 / kHeadsPerBlock),
                      static_cast<unsigned int>(total_q));
    if (w_contig.scalar_type() == torch::kFloat32) {
      indexer_decode_q_fp8_quant_kernel<float, kHeadsPerBlock>
          <<<q_grid, kMqaSubwarpThreads * kHeadsPerBlock, 0, stream>>>(
              q_contig.data_ptr<c10::BFloat16>(),
              w_contig.data_ptr<float>(),
              lengths.data_ptr<int32_t>(),
              capacity,
              q_fp8.data_ptr<c10::Float8_e4m3fn>(),
              w_scaled.data_ptr<float>(),
              ks.data_ptr<int32_t>(),
              ke.data_ptr<int32_t>());
    } else {
      indexer_decode_q_fp8_quant_kernel<c10::BFloat16, kHeadsPerBlock>
          <<<q_grid, kMqaSubwarpThreads * kHeadsPerBlock, 0, stream>>>(
              q_contig.data_ptr<c10::BFloat16>(),
              w_contig.data_ptr<c10::BFloat16>(),
              lengths.data_ptr<int32_t>(),
              capacity,
              q_fp8.data_ptr<c10::Float8_e4m3fn>(),
              w_scaled.data_ptr<float>(),
              ks.data_ptr<int32_t>(),
              ke.data_ptr<int32_t>());
    }
    check_musa_kernel_launch("dsv4 decode indexer Q fp8 quant");

    const int32_t k_blocks_x =
        static_cast<int32_t>(std::min<int64_t>((capacity * 32 + kThreads - 1) / kThreads, 512));
    const dim3 k_grid(static_cast<unsigned int>(k_blocks_x),
                      static_cast<unsigned int>(batch));
    if (blocks.scalar_type() == torch::kInt32) {
      indexer_gather_keys_fp8_kernel<int32_t><<<k_grid, kThreads, 0, stream>>>(
          cache.data_ptr<int8_t>(),
          scales.data_ptr<c10::Half>(),
          blocks.data_ptr<int32_t>(),
          k_fp8.data_ptr<c10::Float8_e4m3fn>(),
          k_scale.data_ptr<float>(),
          valid_slots.data_ptr<bool>(),
          static_cast<int32_t>(capacity),
          table_columns,
          page,
          physical_pages);
    } else {
      indexer_gather_keys_fp8_kernel<int64_t><<<k_grid, kThreads, 0, stream>>>(
          cache.data_ptr<int8_t>(),
          scales.data_ptr<c10::Half>(),
          blocks.data_ptr<int64_t>(),
          k_fp8.data_ptr<c10::Float8_e4m3fn>(),
          k_scale.data_ptr<float>(),
          valid_slots.data_ptr<bool>(),
          static_cast<int32_t>(capacity),
          table_columns,
          page,
          physical_pages);
    }
    check_musa_kernel_launch("dsv4 decode indexer gather keys fp8");
  }

  try {
    TvmffiStreamGuard stream_guard(query.device());
    for (int64_t b = 0; b < batch; ++b) {
      auto q_b = q_fp8.narrow(/*dim=*/0, b * query_rows_per_sequence, query_rows_per_sequence);
      auto k_b = k_fp8.select(/*dim=*/0, b);
      auto w_b = w_scaled.narrow(/*dim=*/0, b * query_rows_per_sequence, query_rows_per_sequence);
      auto ks_b = ks.narrow(/*dim=*/0, b * query_rows_per_sequence, query_rows_per_sequence);
      auto ke_b = ke.narrow(/*dim=*/0, b * query_rows_per_sequence, query_rows_per_sequence);
      auto sfk_b = k_scale.select(/*dim=*/0, b);
      auto out_b = logits_storage.select(/*dim=*/0, b);
      mqa_fn.value()(to_ffi_tensor(q_b),
                     to_ffi_tensor(k_b),
                     to_ffi_tensor(w_b),
                     to_ffi_tensor(ks_b),
                     to_ffi_tensor(ke_b),
                     to_ffi_tensor(sfk_b),
                     to_ffi_tensor(out_b));
    }
  } catch (const std::exception& e) {
    LOG_FIRST_N(WARNING, 2)
        << "[DSV4_DECODE_FP8_MQA_LOGITS] launch failed, falling back: "
        << e.what();
    return torch::Tensor();
  }

  {
    const XLLMOptionalDeviceGuard guard(query.device());
    const musaStream_t stream = current_musa_stream(query.device());
    const int32_t m_blocks_x =
        static_cast<int32_t>(std::min<int64_t>((capacity + kThreads - 1) / kThreads, 64));
    const dim3 m_grid(static_cast<unsigned int>(m_blocks_x),
                      static_cast<unsigned int>(total_q));
    indexer_extract_mask_mqa_scores_kernel<<<m_grid, kThreads, 0, stream>>>(
        logits_storage.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        valid_slots.data_ptr<bool>(),
        scores_out.data_ptr<float>(),
        static_cast<int32_t>(capacity),
        static_cast<int32_t>(query_rows_per_sequence),
        static_cast<int32_t>(aligned_q),
        logits_stride);
    check_musa_kernel_launch("dsv4 decode indexer extract+mask mqa scores");
  }
  return scores_out;
}

void deepseek_v4_indexer_map_outputs(const torch::Tensor& selected,
                                     const torch::Tensor& lengths,
                                     const torch::Tensor& compressed_blocks,
                                     const torch::Tensor& swa_blocks,
                                     int64_t query_rows_per_sequence,
                                     int64_t capacity,
                                     int64_t window,
                                     int64_t page_size,
                                     const torch::Tensor& logical,
                                     const torch::Tensor& physical,
                                     const torch::Tensor& extra) {
  CHECK(selected.defined());
  CHECK(selected.device().is_privateuseone());
  check_tensor(selected, selected, torch::kInt64);
  CHECK_EQ(selected.dim(), 2);
  CHECK_GT(selected.size(0), 0);
  CHECK_EQ(selected.size(1), kTopk);
  CHECK_GT(query_rows_per_sequence, 0);
  CHECK_EQ(selected.size(0) % query_rows_per_sequence, 0);
  CHECK_GE(capacity, kTopk);
  CHECK_LE(capacity, std::numeric_limits<int32_t>::max());
  CHECK_GE(window, 0);
  CHECK_GT(page_size, 0);
  const int64_t batch = selected.size(0) / query_rows_per_sequence;
  check_table(compressed_blocks, selected, batch);
  check_table(swa_blocks, selected, batch);
  check_tensor(lengths, selected, torch::kInt32);
  CHECK_EQ(lengths.numel(), selected.size(0));
  check_tensor(logical, selected, torch::kInt32);
  check_tensor(physical, selected, torch::kInt32);
  check_tensor(extra, selected, torch::kInt32);
  CHECK_EQ(logical.sizes(), selected.sizes());
  CHECK_EQ(physical.sizes(), selected.sizes());
  CHECK_EQ(extra.sizes(), torch::IntArrayRef({selected.size(0), kExtra}));
  const XLLMOptionalDeviceGuard guard(selected.device());
  const int64_t elements = selected.size(0) * (kTopk + kExtra);
  const bool tiled = tiled_metadata_enabled() && page_size == 128 &&
                     selected.size(0) <= 112 && capacity <= 65536;
  auto launch = [&](const auto* compressed, const auto* swa) {
    if (tiled) {
      indexer_map_outputs_tiled_kernel<<<
          dim3(/*x=*/3, static_cast<uint32_t>(selected.size(0))),
          kThreads,
          0,
          current_musa_stream(selected.device())>>>(
          selected.data_ptr<int64_t>(),
          lengths.data_ptr<int32_t>(),
          compressed,
          swa,
          logical.data_ptr<int32_t>(),
          physical.data_ptr<int32_t>(),
          extra.data_ptr<int32_t>(),
          static_cast<int32_t>(capacity),
          static_cast<int32_t>(query_rows_per_sequence),
          compressed_blocks.size(1),
          swa_blocks.size(1),
          static_cast<int32_t>(std::min<int64_t>(window, kExtra)));
      return;
    }
    indexer_map_outputs_kernel<<<launch_blocks(elements),
                                 kThreads,
                                 0,
                                 current_musa_stream(selected.device())>>>(
        selected.data_ptr<int64_t>(),
        lengths.data_ptr<int32_t>(),
        compressed,
        swa,
        logical.data_ptr<int32_t>(),
        physical.data_ptr<int32_t>(),
        extra.data_ptr<int32_t>(),
        elements,
        capacity,
        query_rows_per_sequence,
        compressed_blocks.size(1),
        swa_blocks.size(1),
        std::min<int64_t>(window, kExtra),
        page_size);
  };
  if (compressed_blocks.scalar_type() == torch::kInt32) {
    if (swa_blocks.scalar_type() == torch::kInt32) {
      launch(compressed_blocks.data_ptr<int32_t>(),
             swa_blocks.data_ptr<int32_t>());
    } else {
      launch(compressed_blocks.data_ptr<int32_t>(),
             swa_blocks.data_ptr<int64_t>());
    }
  } else if (swa_blocks.scalar_type() == torch::kInt32) {
    launch(compressed_blocks.data_ptr<int64_t>(),
           swa_blocks.data_ptr<int32_t>());
  } else {
    launch(compressed_blocks.data_ptr<int64_t>(),
           swa_blocks.data_ptr<int64_t>());
  }
  check_musa_kernel_launch("dsv4 Indexer fused output mapping");
}

}  // namespace xllm::kernel::musa
