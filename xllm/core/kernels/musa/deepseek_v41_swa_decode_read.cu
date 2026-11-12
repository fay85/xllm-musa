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

#include "core/kernels/musa/deepseek_v41_swa_decode_read.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

__global__ void swa_decode_read_kernel(const c10::BFloat16* cache,
                                       const c10::BFloat16* current,
                                       const int64_t* table,
                                       const int64_t* positions,
                                       const c10::BFloat16* compressed,
                                       const int64_t* extra,
                                       c10::BFloat16* rows,
                                       int64_t* picks,
                                       int64_t* slot,
                                       int64_t physical_blocks,
                                       int64_t block_size,
                                       int64_t table_columns,
                                       int64_t width,
                                       int64_t window,
                                       int64_t compressed_rows,
                                       int64_t extra_columns) {
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
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    c10::BFloat16 value = static_cast<c10::BFloat16>(0.0f);
    if (row < window) {
      if (history_position >= 0) {
        value = cache[physical * width + column];
      }
    } else if (row == window) {
      value = current[column];
    } else {
      value = compressed[(row - window - 1) * width + column];
    }
    rows[row * width + column] = value;
  }
  if (row != 0) {
    return;
  }
  for (int64_t column = threadIdx.x; column < window; column += blockDim.x) {
    picks[column] = position - window + 1 + column >= 0 ? column + 1 : -1;
  }
  for (int64_t column = threadIdx.x; column < extra_columns;
       column += blockDim.x) {
    const int64_t index = extra[column];
    picks[window + column] =
        index >= 0 && index < compressed_rows ? index + window + 1 : -1;
  }
  if (threadIdx.x == 0) {
    const int64_t block = table[(position / block_size) % table_columns];
    CUDA_KERNEL_ASSERT(block < physical_blocks);
    // Preserve invalid-write sign; the existing scatter filters invalid slots.
    slot[0] = block * block_size + position % block_size;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_swa_decode_read(const torch::Tensor& cache,
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
  CHECK_EQ(current_kv.size(0), 1);
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
  CHECK_EQ(positions.numel(), 1);
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
    CHECK_EQ(extra->size(0), 1);
    extra_long = extra->to(torch::kInt64).contiguous();
    compressed_rows = compressed->size(0);
    extra_columns = extra->size(1);
  }
  const int64_t rows_count = window + 1 + compressed_rows;
  torch::Tensor rows =
      torch::empty({rows_count, cache.size(2)}, cache.options());
  torch::Tensor picks =
      torch::empty({1, window + extra_columns}, pages.options());
  torch::Tensor slot = torch::empty({1}, pages.options());
  swa_decode_read_kernel<<<static_cast<uint32_t>(rows_count),
                           128,
                           0,
                           current_musa_stream(cache.device())>>>(
      cache.data_ptr<c10::BFloat16>(),
      current_kv.data_ptr<c10::BFloat16>(),
      pages.data_ptr<int64_t>(),
      pos.data_ptr<int64_t>(),
      compressed.has_value() ? compressed->data_ptr<c10::BFloat16>() : nullptr,
      extra.has_value() ? extra_long.data_ptr<int64_t>() : nullptr,
      rows.data_ptr<c10::BFloat16>(),
      picks.data_ptr<int64_t>(),
      slot.data_ptr<int64_t>(),
      cache.size(0),
      cache.size(1),
      table.size(1),
      cache.size(2),
      window,
      compressed_rows,
      extra_columns);
  check_musa_kernel_launch("V4.1 q1 fused SWA read and picks");
  return {rows, picks, slot};
}

}  // namespace xllm::kernel::musa
