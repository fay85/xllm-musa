/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <c10/macros/Macros.h>
#include <glog/logging.h>
#include <torch/utils.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "core/kernels/musa/deepseek_v41_swa_decode_read.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

__global__ void swa_draft_read_kernel(const c10::BFloat16* cache,
                                      const c10::BFloat16* current,
                                      const int64_t* table,
                                      const int64_t* previous_context,
                                      c10::BFloat16* rows,
                                      int64_t* picks,
                                      int64_t physical_blocks,
                                      int64_t block_size,
                                      int64_t table_columns,
                                      int64_t width,
                                      int64_t window,
                                      int64_t queries) {
  const int64_t context = previous_context[0];
  CUDA_KERNEL_ASSERT(context >= 0);
  const int64_t row = blockIdx.x;
  const int64_t position = context - window + row;
  int64_t slot = 0;
  if (row < window && position >= 0) {
    const int64_t block = table[(position / block_size) % table_columns];
    const int64_t read_block = block < 0 ? 0 : block;
    CUDA_KERNEL_ASSERT(read_block < physical_blocks);
    slot = read_block * block_size + position % block_size;
  }
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    c10::BFloat16 value = static_cast<c10::BFloat16>(0.0f);
    if (row >= window) {
      value = current[(row - window) * width + column];
    } else if (position >= 0) {
      value = cache[slot * width + column];
    }
    rows[row * width + column] = value;
  }
  if (threadIdx.x < queries) {
    // Every draft query sees every proposed KV row, including future rows.
    picks[threadIdx.x * (window + queries) + row] = position >= 0 ? row : -1;
  }
}

template <bool PackSelected>
__device__ int64_t verify_pick(const int64_t* extra,
                               int64_t position,
                               int64_t window,
                               int64_t compressed_start,
                               int64_t compressed_rows,
                               int64_t extra_columns,
                               int64_t query,
                               int64_t column) {
  if (column < window) {
    return position + query - window + 1 + column >= 0 ? query + column + 1
                                                       : -1;
  }
  const int64_t offset = query * extra_columns + column - window;
  const int64_t selected = extra[offset];
  return selected >= 0 && selected < compressed_rows
             ? compressed_start + (PackSelected ? offset : selected)
             : -1;
}

__device__ int64_t verify_write_slot(const int64_t* table,
                                     int64_t position,
                                     int64_t physical_blocks,
                                     int64_t block_size,
                                     int64_t table_columns) {
  const int64_t block = table[(position / block_size) % table_columns];
  CUDA_KERNEL_ASSERT(block < physical_blocks);
  // The existing scatter filters the negative slots.
  return block * block_size + position % block_size;
}

template <bool PackSelected, typename Pick>
__global__ void swa_verify_read_kernel(const c10::BFloat16* cache,
                                       const c10::BFloat16* current,
                                       const int64_t* table,
                                       const int64_t* positions,
                                       const c10::BFloat16* compressed,
                                       const int64_t* extra,
                                       c10::BFloat16* rows,
                                       Pick* picks,
                                       int64_t* slot,
                                       int64_t physical_blocks,
                                       int64_t block_size,
                                       int64_t table_columns,
                                       int64_t width,
                                       int64_t window,
                                       int64_t compressed_rows,
                                       int64_t extra_columns,
                                       int64_t queries) {
  const int64_t position = positions[0];
  CUDA_KERNEL_ASSERT(position >= 0);
  const int64_t row = blockIdx.x;
  const int64_t history_position = position - window + row;
  int64_t physical = 0;
  if (row < window && history_position >= 0) {
    const int64_t block =
        table[(history_position / block_size) % table_columns];
    const int64_t read_block = block < 0 ? 0 : block;
    CUDA_KERNEL_ASSERT(read_block < physical_blocks);
    physical = read_block * block_size + history_position % block_size;
  }
  const int64_t compressed_start = window + queries;
  const c10::BFloat16* source = nullptr;
  if (row < window) {
    if (history_position >= 0) {
      source = cache + physical * width;
    }
  } else if (row < compressed_start) {
    source = current + (row - window) * width;
  } else {
    const int64_t offset = row - compressed_start;
    const int64_t selected = PackSelected ? extra[offset] : offset;
    if (selected >= 0 && selected < compressed_rows) {
      source = compressed + selected * width;
    }
  }
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    rows[row * width + column] =
        source == nullptr ? static_cast<c10::BFloat16>(0.0f) : source[column];
  }
  const int64_t stride = window + extra_columns;
  if constexpr (std::is_same_v<Pick, int32_t>) {
    // Assign one query's contiguous picks to each of the first query CTAs.
    if (row >= queries) {
      return;
    }
    for (int64_t column = threadIdx.x; column < stride; column += blockDim.x) {
      picks[row * stride + column] =
          static_cast<int32_t>(verify_pick<PackSelected>(extra,
                                                         position,
                                                         window,
                                                         compressed_start,
                                                         compressed_rows,
                                                         extra_columns,
                                                         row,
                                                         column));
    }
    if (threadIdx.x == 0) {
      const int64_t current = positions[row];
      CUDA_KERNEL_ASSERT(current == position + row);
      slot[row] = verify_write_slot(
          table, current, physical_blocks, block_size, table_columns);
    }
  } else {
    if (row != 0) {
      return;
    }
    for (int64_t index = threadIdx.x; index < queries * stride;
         index += blockDim.x) {
      const int64_t query = index / stride;
      const int64_t column = index % stride;
      CUDA_KERNEL_ASSERT(positions[query] == position + query);
      picks[index] = verify_pick<PackSelected>(extra,
                                               position,
                                               window,
                                               compressed_start,
                                               compressed_rows,
                                               extra_columns,
                                               query,
                                               column);
    }
    if (threadIdx.x < queries) {
      slot[threadIdx.x] = verify_write_slot(table,
                                            positions[threadIdx.x],
                                            physical_blocks,
                                            block_size,
                                            table_columns);
    }
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_swa_draft_read(
    const torch::Tensor& cache,
    const torch::Tensor& current_kv,
    const torch::Tensor& table,
    const torch::Tensor& previous_context,
    int64_t window) {
  CHECK(cache.defined() && current_kv.defined() && table.defined() &&
        previous_context.defined());
  CHECK(cache.device().type() == torch::kPrivateUse1);
  CHECK_EQ(current_kv.device(), cache.device());
  CHECK_EQ(table.device(), cache.device());
  CHECK_EQ(previous_context.device(), cache.device());
  CHECK_EQ(cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(current_kv.scalar_type(), torch::kBFloat16);
  CHECK_EQ(previous_context.scalar_type(), torch::kInt64);
  CHECK_EQ(cache.dim(), 3);
  CHECK_GT(cache.size(0), 0);
  CHECK_GT(cache.size(1), 0);
  CHECK_GT(cache.size(2), 0);
  CHECK_EQ(current_kv.dim(), 2);
  CHECK_GT(current_kv.size(0), 0);
  CHECK_LE(current_kv.size(0), 8);
  CHECK_EQ(current_kv.size(1), cache.size(2));
  CHECK(cache.is_contiguous() && current_kv.is_contiguous());
  CHECK_EQ(previous_context.numel(), 1);
  CHECK(previous_context.is_contiguous());
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(table.size(0), 1);
  CHECK_GT(table.size(1), 0);
  CHECK(table.scalar_type() == torch::kInt32 ||
        table.scalar_type() == torch::kInt64);
  CHECK_GT(window, 0);
  CHECK_GE(table.size(1) * cache.size(1), window);
  torch::DeviceGuard guard(cache.device());
  const torch::Tensor pages = table.to(torch::kInt64).contiguous();
  const int64_t queries = current_kv.size(0);
  const int64_t count = window + queries;
  torch::Tensor rows = torch::empty({count, cache.size(2)}, cache.options());
  torch::Tensor picks = torch::empty({queries, count}, pages.options());
  swa_draft_read_kernel<<<static_cast<uint32_t>(count),
                          128,
                          0,
                          current_musa_stream(cache.device())>>>(
      cache.data_ptr<c10::BFloat16>(),
      current_kv.data_ptr<c10::BFloat16>(),
      pages.data_ptr<int64_t>(),
      previous_context.data_ptr<int64_t>(),
      rows.data_ptr<c10::BFloat16>(),
      picks.data_ptr<int64_t>(),
      cache.size(0),
      cache.size(1),
      table.size(1),
      cache.size(2),
      window,
      queries);
  check_musa_kernel_launch("V4.1 fused noncausal draft SWA read");
  return {rows, picks};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_swa_verify_read(const torch::Tensor& cache,
                             const torch::Tensor& current_kv,
                             const torch::Tensor& table,
                             const torch::Tensor& positions,
                             int64_t window,
                             const std::optional<torch::Tensor>& compressed,
                             const std::optional<torch::Tensor>& extra) {
  CHECK(cache.defined() && current_kv.defined() && table.defined() &&
        positions.defined());
  CHECK(cache.device().type() == torch::kPrivateUse1);
  CHECK_EQ(current_kv.device(), cache.device());
  CHECK_EQ(table.device(), cache.device());
  CHECK_EQ(positions.device(), cache.device());
  CHECK_EQ(cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(current_kv.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cache.dim(), 3);
  CHECK_EQ(current_kv.dim(), 2);
  CHECK_GT(current_kv.size(0), 0);
  CHECK_LE(current_kv.size(0), 6);
  CHECK_LE(current_kv.size(0), cache.size(1));
  CHECK_EQ(current_kv.size(1), cache.size(2));
  CHECK(cache.is_contiguous() && current_kv.is_contiguous());
  CHECK_GT(cache.size(0), 0);
  CHECK_GT(cache.size(1), 0);
  CHECK_GT(cache.size(2), 0);
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(table.size(0), 1);
  CHECK_GT(table.size(1), 0);
  CHECK(table.scalar_type() == torch::kInt64 ||
        table.scalar_type() == torch::kInt32);
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(positions.numel(), current_kv.size(0));
  CHECK(positions.scalar_type() == torch::kInt64 ||
        positions.scalar_type() == torch::kInt32);
  CHECK_GT(window, 0);
  CHECK_GE(table.size(1) * cache.size(1), window);
  CHECK_EQ(compressed.has_value(), extra.has_value());
  torch::DeviceGuard guard(cache.device());
  const torch::Tensor pages = table.to(torch::kInt64).contiguous();
  const torch::Tensor pos = positions.to(torch::kInt64).contiguous();
  int64_t compressed_rows = 0;
  int64_t extra_columns = 0;
  torch::Tensor extra_long;
  if (compressed.has_value()) {
    CHECK(compressed->defined() && extra->defined());
    CHECK_EQ(compressed->device(), cache.device());
    CHECK_EQ(extra->device(), cache.device());
    CHECK_EQ(compressed->scalar_type(), torch::kBFloat16);
    CHECK_EQ(compressed->dim(), 2);
    CHECK_EQ(compressed->size(1), cache.size(2));
    CHECK(compressed->is_contiguous());
    CHECK(extra->scalar_type() == torch::kInt64 ||
          extra->scalar_type() == torch::kInt32);
    CHECK_EQ(extra->dim(), 2);
    CHECK_EQ(extra->size(0), current_kv.size(0));
    extra_long = extra->to(torch::kInt64).contiguous();
    compressed_rows = compressed->size(0);
    extra_columns = extra->size(1);
  }
  const int64_t queries = current_kv.size(0);
  const char* packing = std::getenv("XLLM_DSV41_VERIFY_PACKED_COMPRESSED_READ");
  // Keep the sparse pick order unchanged. Each query gets a private copy of
  // its selected compressed rows, so no device-wide deduplication is needed.
  const int64_t selected_rows = queries * extra_columns;
  const bool pack_selected =
      packing != nullptr && packing[0] == '1' && packing[1] == '\0' &&
      queries > 1 && selected_rows > 0 && selected_rows < compressed_rows;
  const int64_t rows_count =
      window + queries + (pack_selected ? selected_rows : compressed_rows);
  if (pack_selected) {
    LOG_FIRST_N(INFO, 1)
        << "Native V4.1 verify packs selected compressed KV rows: source="
        << compressed_rows << ", packed=" << selected_rows
        << ", queries=" << queries;
  }
  const char* normalized = std::getenv("XLLM_DSV41_VERIFY_NORMALIZED_PICKS");
  const int64_t stride = window + extra_columns;
  const bool prepare_picks = pack_selected && normalized != nullptr &&
                             normalized[0] == '1' && normalized[1] == '\0' &&
                             stride % 64 == 0 &&
                             rows_count <= std::numeric_limits<int32_t>::max();
  if (prepare_picks) {
    LOG_FIRST_N(INFO, 1)
        << "Native V4.1 verify writes normalized int32 sparse picks: columns="
        << stride << ", queries=" << queries;
  }
  torch::Tensor rows =
      torch::empty({rows_count, cache.size(2)}, cache.options());
  torch::Tensor picks = torch::empty(
      {queries, stride},
      pages.options().dtype(prepare_picks ? torch::kInt32 : torch::kInt64));
  torch::Tensor slot = torch::empty({queries}, pages.options());
  const auto launch = [&](auto mode, auto pick_type) {
    using Pick = decltype(pick_type);
    swa_verify_read_kernel<decltype(mode)::value, Pick>
        <<<static_cast<uint32_t>(rows_count),
           128,
           0,
           current_musa_stream(cache.device())>>>(
            cache.data_ptr<c10::BFloat16>(),
            current_kv.data_ptr<c10::BFloat16>(),
            pages.data_ptr<int64_t>(),
            pos.data_ptr<int64_t>(),
            compressed.has_value() ? compressed->data_ptr<c10::BFloat16>()
                                   : nullptr,
            extra.has_value() ? extra_long.data_ptr<int64_t>() : nullptr,
            rows.data_ptr<c10::BFloat16>(),
            picks.data_ptr<Pick>(),
            slot.data_ptr<int64_t>(),
            cache.size(0),
            cache.size(1),
            table.size(1),
            cache.size(2),
            window,
            compressed_rows,
            extra_columns,
            queries);
  };
  if (prepare_picks) {
    launch(std::true_type{}, int32_t{});
  } else if (pack_selected) {
    launch(std::true_type{}, int64_t{});
  } else {
    launch(std::false_type{}, int64_t{});
  }
  check_musa_kernel_launch("V4.1 causal verify fused SWA read and picks");
  return {rows, picks, slot};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_swa_decode_read(const torch::Tensor& cache,
                             const torch::Tensor& current_kv,
                             const torch::Tensor& table,
                             const torch::Tensor& positions,
                             int64_t window,
                             const std::optional<torch::Tensor>& compressed,
                             const std::optional<torch::Tensor>& extra) {
  CHECK_EQ(current_kv.size(0), 1);
  return deepseek_v41_swa_verify_read(
      cache, current_kv, table, positions, window, compressed, extra);
}

}  // namespace xllm::kernel::musa
