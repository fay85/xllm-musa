/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "core/kernels/musa/deepseek_v41_prefix_restore.h"
namespace xllm::layer {
struct DeepseekV41PrefixSavedRows {
  torch::Tensor target;
  torch::Tensor slots;
  torch::Tensor values;
  torch::Tensor prefix_lengths;
};
// A C1 transaction for <=6 consecutive verify rows. Each saved cache row is
// tagged with the smallest accepted input prefix that makes its write valid.
// Causal prefix outputs are retained; only rejected writes are rolled back.
// It does not repair non-causal verify math or commit scheduler/draft state.
class DeepseekV41PrefixCheckpoint final {
 public:
  DeepseekV41PrefixCheckpoint(int64_t begin,
                              int64_t tokens,
                              int64_t max_bytes,
                              bool fused_snapshot = false)
      : begin_(begin),
        tokens_(tokens),
        max_bytes_(max_bytes),
        fused_snapshot_(fused_snapshot) {
    CHECK_GE(begin_, 0);
    CHECK_GT(tokens_, 0);
    CHECK_LE(tokens_, 6);
    CHECK_GT(max_bytes_, 0);
    saved_.reserve(256);
  }
  void add_paged(const torch::Tensor& cache,
                 const torch::Tensor& table,
                 int64_t ratio,
                 bool ring) {
    check_inputs(cache, table);
    CHECK(ratio == 1 || ratio == 2);
    // No two speculative rows can address the same offset within a page.
    CHECK_GE(cache.size(1), tokens_);
    const int64_t first = begin_ / ratio;
    const int64_t count = (begin_ + tokens_) / ratio - first;
    if (count == 0) {
      return;
    }
    if (!ring) {
      CHECK_LE(first + count, table.size(1) * cache.size(1));
    }
    if (fused_snapshot_) {
      save_fused(cache, table, ratio, ring, false, cache.size(1), count);
      return;
    }
    torch::DeviceGuard guard(cache.device());
    const torch::Tensor logical =
        torch::arange(count, table.options().dtype(torch::kInt64)) + first;
    torch::Tensor columns = torch::floor_divide(logical, cache.size(1));
    if (ring) {
      columns = columns.remainder(table.size(1));
    }
    const torch::Tensor blocks =
        table.select(0, 0).index_select(0, columns).to(torch::kInt64);
    save(cache.view({-1, cache.size(2)}),
         blocks * cache.size(1) + logical.remainder(cache.size(1)),
         (logical + 1) * ratio - begin_);
  }
  void add_recurrent(const torch::Tensor& state,
                     const torch::Tensor& swa_table,
                     int64_t swa_block_size) {
    check_inputs(state, swa_table);
    CHECK_EQ(state.size(1), 32);
    CHECK_GT(swa_block_size, 0);
    CHECK_EQ(swa_block_size % 32, 0);
    const int64_t first = begin_ + begin_ % 2;
    const int64_t count = (begin_ + tokens_ - first + 1) / 2;
    if (count == 0) {
      return;
    }
    if (fused_snapshot_) {
      save_fused(state, swa_table, 1, true, true, swa_block_size, count);
      return;
    }
    torch::DeviceGuard guard(state.device());
    const torch::Tensor positions =
        torch::arange(count, swa_table.options().dtype(torch::kInt64)) * 2 +
        first;
    const torch::Tensor columns = torch::floor_divide(positions, swa_block_size)
                                      .remainder(swa_table.size(1));
    const torch::Tensor blocks =
        swa_table.select(0, 0).index_select(0, columns).to(torch::kInt64);
    save(state.view({-1, state.size(2)}),
         blocks * 32 + positions.remainder(32),
         positions - begin_ + 1);
  }
  void restore_suffix(int64_t accepted) {
    CHECK(!restored_);
    CHECK_GE(accepted, 0);
    CHECK_LE(accepted, tokens_);
    if (accepted != tokens_) {
      for (const auto& row : saved_) {
        kernel::musa::deepseek_v41_restore_suffix(
            row.target, row.slots, row.values, row.prefix_lengths, accepted);
      }
    }
    restored_ = true;
  }
  int64_t saved_bytes() const { return saved_bytes_; }

 private:
  void check_inputs(const torch::Tensor& tensor,
                    const torch::Tensor& table) const {
    CHECK(!restored_);
    CHECK(tensor.defined() && table.defined());
    CHECK(tensor.device().type() == torch::kPrivateUse1);
    CHECK_EQ(tensor.device(), table.device());
    CHECK(tensor.is_contiguous() && table.is_contiguous());
    CHECK_EQ(tensor.dim(), 3);
    CHECK_GT(tensor.size(0), 0);
    CHECK_GT(tensor.size(1), 0);
    CHECK_GT(tensor.size(2), 0);
    CHECK_EQ(table.dim(), 2);
    CHECK_EQ(table.size(0), 1);
    CHECK_GT(table.size(1), 0);
    CHECK(table.scalar_type() == torch::kInt32 ||
          table.scalar_type() == torch::kInt64);
  }
  void save_fused(const torch::Tensor& cache,
                  const torch::Tensor& table,
                  int64_t ratio,
                  bool ring,
                  bool recurrent,
                  int64_t swa_block_size,
                  int64_t count) {
    const int64_t bytes = count * cache.size(2) * cache.element_size();
    CHECK_LE(bytes, max_bytes_ - saved_bytes_);
    auto rows = kernel::musa::deepseek_v41_snapshot_prefix(
        cache, table, begin_, tokens_, ratio, ring, recurrent, swa_block_size);
    CHECK_EQ(rows[0].numel(), count);
    saved_bytes_ += bytes;
    saved_.emplace_back(
        DeepseekV41PrefixSavedRows{cache.view({-1, cache.size(2)}),
                                   std::move(rows[0]),
                                   std::move(rows[1]),
                                   std::move(rows[2])});
  }
  void save(torch::Tensor target,
            torch::Tensor slots,
            torch::Tensor prefix_lengths) {
    const int64_t bytes =
        slots.numel() * target.size(1) * target.element_size();
    CHECK_LE(bytes, max_bytes_ - saved_bytes_);
    saved_bytes_ += bytes;
    // Invalid padding pages are never written by restore_suffix; clamp only
    // this read to keep the fixed row count without a D2H nonzero boundary.
    torch::Tensor values =
        target.index_select(0, slots.clamp(0, target.size(0) - 1));
    saved_.emplace_back(DeepseekV41PrefixSavedRows{std::move(target),
                                                   std::move(slots),
                                                   std::move(values),
                                                   std::move(prefix_lengths)});
  }
  int64_t begin_;
  int64_t tokens_;
  int64_t max_bytes_;
  int64_t saved_bytes_ = 0;
  bool restored_ = false;
  bool fused_snapshot_ = false;
  std::vector<DeepseekV41PrefixSavedRows> saved_;
};
}  // namespace xllm::layer
