// Copyright 2026 The xLLM Authors. Licensed under the Apache License,
// Version 2.0.
#include <glog/logging.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "core/kernels/musa/deepseek_v4_indexer_metadata.h"
#include "core/kernels/musa/musa_kernel_utils.h"

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
