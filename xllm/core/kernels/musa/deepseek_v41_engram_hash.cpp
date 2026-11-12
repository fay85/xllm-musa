/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include "core/kernels/musa/deepseek_v41_engram_hash.h"

#include <glog/logging.h>
#include <torch/utils.h>

#include <array>
#include <limits>

#include "core/kernels/musa/musa_aot_kernel.h"

namespace xllm::kernel::musa {
std::pair<torch::Tensor, torch::Tensor> deepseek_v41_paged_engram_hash(
    const torch::Tensor& ids,
    const torch::Tensor& positions,
    const torch::Tensor& table,
    torch::Tensor& history,
    const torch::Tensor& token_map,
    const torch::Tensor& multipliers,
    const torch::Tensor& primes,
    const torch::Tensor& offsets,
    int64_t pad_id,
    int64_t block_size,
    int64_t start,
    const std::optional<torch::Tensor>& lookback,
    const std::string& directory) {
  CHECK(ids.defined() && positions.defined() && table.defined() &&
        history.defined());
  CHECK(ids.device().type() == torch::kPrivateUse1);
  for (const torch::Tensor& tensor :
       {positions, table, history, token_map, multipliers, primes, offsets}) {
    CHECK_EQ(tensor.device(), ids.device());
  }
  CHECK_EQ(ids.dim(), 1);
  CHECK_EQ(positions.sizes(), ids.sizes());
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(table.size(0), 1);
  CHECK_GT(table.size(1), 0);
  CHECK_EQ(history.dim(), 1);
  CHECK_EQ(history.scalar_type(), torch::kInt32);
  CHECK(history.is_contiguous());
  CHECK_GT(history.numel(), 0);
  CHECK_LE(history.numel(), std::numeric_limits<int32_t>::max());
  CHECK_EQ(token_map.dim(), 1);
  CHECK_GT(token_map.numel(), 0);
  CHECK_EQ(multipliers.sizes(), torch::IntArrayRef({2, 4}));
  CHECK_EQ(primes.sizes(), torch::IntArrayRef({2, 3, 8}));
  CHECK_EQ(offsets.sizes(), torch::IntArrayRef({2, 24}));
  CHECK_GT(block_size, 0);
  CHECK_GE(start, 0);
  CHECK_LE(ids.numel(), 512);
  CHECK(!directory.empty());
  torch::DeviceGuard guard(ids.device());
  const torch::Tensor tokens = ids.to(torch::kInt64).contiguous();
  const torch::Tensor pos = positions.to(torch::kInt64).contiguous();
  const torch::Tensor pages = table.to(torch::kInt64).contiguous();
  for (const torch::Tensor& tensor :
       {token_map, multipliers, primes, offsets}) {
    CHECK_EQ(tensor.scalar_type(), torch::kInt64);
    CHECK(tensor.is_contiguous());
  }
  torch::Tensor previous = tokens;
  int64_t lookback_size = 0;
  if (lookback.has_value()) {
    CHECK(lookback->defined());
    CHECK_EQ(lookback->device(), ids.device());
    CHECK_EQ(lookback->dim(), 2);
    CHECK_EQ(lookback->size(0), 1);
    CHECK_GT(lookback->size(1), 0);
    previous = lookback->to(torch::kInt64).contiguous();
    lookback_size = previous.numel();
  }
  torch::Tensor output =
      torch::empty({ids.numel(), 2, 24}, ids.options().dtype(torch::kInt32));
  // The AOT byte-mask writer publishes aligned packed u32 words. Padding
  // protects the final partial word; the returned mask retains exact M rows.
  torch::Tensor active_storage = torch::empty(
      {(ids.numel() + 3) / 4 * 4}, ids.options().dtype(torch::kBool));
  torch::Tensor active = active_storage.narrow(0, 0, ids.numel());
  if (ids.numel() == 0) {
    return {output, active};
  }
  torch::Tensor slots =
      torch::empty({ids.numel()}, ids.options().dtype(torch::kInt32));
  torch::Tensor mapped = torch::empty({ids.numel()}, history.options());
  int32_t count = static_cast<int32_t>(ids.numel());
  int64_t map_size = token_map.numel();
  int64_t history_size = history.numel();
  int64_t table_size = table.size(1);
  void* tokens_ptr = tokens.data_ptr();
  void* pos_ptr = pos.data_ptr();
  void* table_ptr = pages.data_ptr();
  void* history_ptr = history.data_ptr();
  void* map_ptr = token_map.data_ptr();
  void* mult_ptr = multipliers.data_ptr();
  void* prime_ptr = primes.data_ptr();
  void* offset_ptr = offsets.data_ptr();
  void* previous_ptr = previous.data_ptr();
  void* output_ptr = output.data_ptr();
  void* active_ptr = active.data_ptr();
  void* slots_ptr = slots.data_ptr();
  void* mapped_ptr = mapped.data_ptr();
  std::array<void*, 21> args = {
      &tokens_ptr,   &pos_ptr,    &table_ptr,  &history_ptr,  &map_ptr,
      &mult_ptr,     &prime_ptr,  &offset_ptr, &previous_ptr, &output_ptr,
      &active_ptr,   &slots_ptr,  &mapped_ptr, &count,        &map_size,
      &history_size, &table_size, &block_size, &start,        &lookback_size,
      &pad_id};
  launch_musa_aot_kernel(ids.device(),
                         directory,
                         "hash",
                         {static_cast<uint32_t>(count), 1, 1},
                         args.data());
  // Kernel completion on this same stream orders all old-history reads
  // before any ring slot is overwritten, even when pages repeat.
  std::array<void*, 5> commit_args = {
      &slots_ptr, &mapped_ptr, &history_ptr, &count, &history_size};
  launch_musa_aot_kernel(ids.device(),
                         directory,
                         count <= 32 ? "commit32" : "commit512",
                         {static_cast<uint32_t>(count), 1, 1},
                         commit_args.data());
  return {output, active};
}
}  // namespace xllm::kernel::musa
