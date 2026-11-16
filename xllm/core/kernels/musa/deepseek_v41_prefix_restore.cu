/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <glog/logging.h>
#include <torch/utils.h>

#include <cstdint>
#include <utility>

#include "core/kernels/musa/deepseek_v41_prefix_restore.h"
#include "core/kernels/musa/musa_kernel_utils.h"
namespace xllm::kernel::musa {
namespace {
template <typename TableType>
__global__ void snapshot_prefix_kernel(const uint8_t* target,
                                       const TableType* table,
                                       int64_t* slots,
                                       uint8_t* saved,
                                       int64_t* prefixes,
                                       int64_t begin,
                                       int64_t first,
                                       int64_t ratio,
                                       bool ring,
                                       bool recurrent,
                                       int64_t block_size,
                                       int64_t swa_block_size,
                                       int64_t table_columns,
                                       int64_t target_rows,
                                       int64_t row_bytes) {
  const int64_t row = blockIdx.x;
  const int64_t logical = first + row * (recurrent ? 2 : 1);
  int64_t column = logical / (recurrent ? swa_block_size : block_size);
  if (ring || recurrent) {
    column %= table_columns;
  }
  const int64_t block = static_cast<int64_t>(table[column]);
  const int64_t slot = block * block_size + logical % block_size;
  const int64_t prefix =
      recurrent ? logical - begin + 1 : (logical + 1) * ratio - begin;
  if (threadIdx.x == 0) {
    slots[row] = slot;
    prefixes[row] = prefix;
  }
  const int64_t read_slot =
      slot < 0 ? 0 : (slot >= target_rows ? target_rows - 1 : slot);
  for (int64_t offset = threadIdx.x; offset < row_bytes; offset += blockDim.x) {
    saved[row * row_bytes + offset] = target[read_slot * row_bytes + offset];
  }
}

__global__ void restore_suffix_kernel(uint8_t* target,
                                      const int64_t* slots,
                                      const uint8_t* saved,
                                      const int64_t* prefix_lengths,
                                      int64_t accepted,
                                      int64_t target_rows,
                                      int64_t row_bytes) {
  const int64_t row = blockIdx.x;
  const int64_t slot = slots[row];
  if (prefix_lengths[row] <= accepted || slot < 0 || slot >= target_rows) {
    return;
  }
  for (int64_t offset = threadIdx.x; offset < row_bytes; offset += blockDim.x) {
    target[slot * row_bytes + offset] = saved[row * row_bytes + offset];
  }
}
}  // namespace

std::vector<torch::Tensor> deepseek_v41_snapshot_prefix(
    const torch::Tensor& cache,
    const torch::Tensor& table,
    int64_t begin,
    int64_t tokens,
    int64_t ratio,
    bool ring,
    bool recurrent,
    int64_t swa_block_size) {
  CHECK(cache.defined() && table.defined());
  CHECK(cache.device().type() == torch::kPrivateUse1);
  CHECK_EQ(cache.device(), table.device());
  CHECK(cache.is_contiguous() && table.is_contiguous());
  CHECK_EQ(cache.dim(), 3);
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(table.size(0), 1);
  CHECK_GT(table.size(1), 0);
  CHECK(table.scalar_type() == torch::kInt32 ||
        table.scalar_type() == torch::kInt64);
  CHECK_GT(cache.size(0), 0);
  CHECK_GT(cache.size(1), 0);
  CHECK_GT(cache.size(2), 0);
  CHECK_GE(begin, 0);
  CHECK_GT(tokens, 0);
  CHECK_LE(tokens, 6);
  CHECK(ratio == 1 || ratio == 2);
  if (recurrent) {
    CHECK_EQ(cache.size(1), 32);
    CHECK_GT(swa_block_size, 0);
    CHECK_EQ(swa_block_size % 32, 0);
  } else {
    CHECK_GE(cache.size(1), tokens);
  }
  const int64_t first = recurrent ? begin + begin % 2 : begin / ratio;
  const int64_t count = recurrent ? (begin + tokens - first + 1) / 2
                                  : (begin + tokens) / ratio - first;
  if (!ring && !recurrent) {
    CHECK_LE(first + count, table.size(1) * cache.size(1));
  }
  torch::DeviceGuard guard(cache.device());
  auto slots = torch::empty({count}, table.options().dtype(torch::kInt64));
  auto prefixes = torch::empty_like(slots);
  auto values = torch::empty({count, cache.size(2)}, cache.options());
  if (count > 0) {
    const auto launch = [&](auto* page_ids) {
      snapshot_prefix_kernel<<<static_cast<uint32_t>(count),
                               256,
                               0,
                               current_musa_stream(cache.device())>>>(
          static_cast<const uint8_t*>(cache.data_ptr()),
          page_ids,
          slots.data_ptr<int64_t>(),
          static_cast<uint8_t*>(values.data_ptr()),
          prefixes.data_ptr<int64_t>(),
          begin,
          first,
          ratio,
          ring,
          recurrent,
          cache.size(1),
          swa_block_size,
          table.size(1),
          cache.size(0) * cache.size(1),
          cache.size(2) * cache.element_size());
    };
    if (table.scalar_type() == torch::kInt32) {
      launch(table.data_ptr<int32_t>());
    } else {
      launch(table.data_ptr<int64_t>());
    }
    CHECK_EQ(musaGetLastError(), musaSuccess);
  }
  return {std::move(slots), std::move(values), std::move(prefixes)};
}

void deepseek_v41_restore_suffix(const torch::Tensor& target,
                                 const torch::Tensor& slots,
                                 const torch::Tensor& saved,
                                 const torch::Tensor& prefix_lengths,
                                 int64_t accepted) {
  CHECK(target.defined() && slots.defined() && saved.defined() &&
        prefix_lengths.defined());
  CHECK(target.device().type() == torch::kPrivateUse1);
  CHECK_EQ(target.dim(), 2);
  CHECK_EQ(saved.dim(), 2);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(prefix_lengths.sizes(), slots.sizes());
  CHECK_EQ(saved.size(0), slots.numel());
  CHECK_EQ(saved.size(1), target.size(1));
  CHECK_EQ(saved.scalar_type(), target.scalar_type());
  CHECK_EQ(slots.scalar_type(), torch::kInt64);
  CHECK_EQ(prefix_lengths.scalar_type(), torch::kInt64);
  CHECK_GE(accepted, 0);
  CHECK_LE(slots.numel(), 6);
  for (const torch::Tensor& tensor : {target, slots, saved, prefix_lengths}) {
    CHECK_EQ(tensor.device(), target.device());
    CHECK(tensor.is_contiguous());
  }
  if (slots.numel() == 0) {
    return;
  }
  torch::DeviceGuard guard(target.device());
  restore_suffix_kernel<<<static_cast<uint32_t>(slots.numel()),
                          256,
                          0,
                          current_musa_stream(target.device())>>>(
      static_cast<uint8_t*>(target.data_ptr()),
      slots.data_ptr<int64_t>(),
      static_cast<const uint8_t*>(saved.data_ptr()),
      prefix_lengths.data_ptr<int64_t>(),
      accepted,
      target.size(0),
      target.size(1) * target.element_size());
  CHECK_EQ(musaGetLastError(), musaSuccess);
}
}  // namespace xllm::kernel::musa
