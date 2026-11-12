/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/
#pragma once
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace xllm::layer {

struct DeepseekV41SavedRows {
  torch::Tensor target;
  torch::Tensor indices;
  torch::Tensor values;
};

// A bounded, request-local transaction over existing GPU cache tensors.
// It does not restore scheduler lengths, draft state, or runtime metadata.
// Caller serializes all writes and snapshots BEFORE a speculative forward.
class DeepseekV41StateCheckpoint final {
 public:
  explicit DeepseekV41StateCheckpoint(int64_t max_bytes)
      : max_bytes_(max_bytes) {
    CHECK_GT(max_bytes_, 0);
  }

  void add_full(const torch::Tensor& tensor) {
    check_tensor(tensor);
    reserve_bytes(tensor.nbytes());
    torch::DeviceGuard guard(tensor.device());
    saved_.emplace_back(DeepseekV41SavedRows{tensor, {}, tensor.clone()});
  }

  void add_paged(const torch::Tensor& cache,
                 const torch::Tensor& table,
                 int64_t first,
                 int64_t count,
                 bool ring) {
    check_tensor(cache);
    CHECK_EQ(cache.dim(), 3);
    CHECK(table.defined() && table.is_contiguous());
    CHECK_EQ(table.device(), cache.device());
    CHECK(table.scalar_type() == torch::kInt32 ||
          table.scalar_type() == torch::kInt64);
    CHECK_EQ(table.dim(), 2);
    CHECK_EQ(table.size(0), 1);
    CHECK_GT(table.size(1), 0);
    CHECK_GE(first, 0);
    CHECK_GE(count, 0);
    CHECK_GT(cache.size(0), 0);
    CHECK_GT(cache.size(1), 0);
    CHECK_GT(cache.size(2), 0);
    CHECK_LE(count, max_bytes_ / (cache.size(2) * cache.element_size()));
    if (!ring) {
      CHECK_LE(first, table.size(1) * cache.size(1));
      CHECK_LE(count, table.size(1) * cache.size(1) - first);
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
    const torch::Tensor slots =
        blocks * cache.size(1) + logical.remainder(cache.size(1));
    const torch::Tensor valid =
        slots.ge(0).logical_and(slots.lt(cache.size(0) * cache.size(1)));
    const torch::Tensor sorted = std::get<0>(torch::sort(slots.index({valid})));
    const torch::Tensor indices =
        std::get<0>(torch::unique_consecutive(sorted, false, false));
    const torch::Tensor target = cache.view({-1, cache.size(2)});
    reserve_bytes(indices.numel() * cache.size(2) * cache.element_size());
    saved_.emplace_back(
        DeepseekV41SavedRows{target, indices, target.index_select(0, indices)});
  }

  // Recurrent rows follow the SWA physical page id, but their row count is
  // not the SWA token block size. Save whole selected state pages explicitly.
  void add_blocks(const torch::Tensor& tensor, const torch::Tensor& block_ids) {
    check_tensor(tensor);
    CHECK_GE(tensor.dim(), 1);
    CHECK_GT(tensor.size(0), 0);
    CHECK(block_ids.defined() && block_ids.is_contiguous());
    CHECK_EQ(block_ids.device(), tensor.device());
    CHECK_EQ(block_ids.dim(), 1);
    CHECK(block_ids.scalar_type() == torch::kInt32 ||
          block_ids.scalar_type() == torch::kInt64);
    torch::DeviceGuard guard(tensor.device());
    const torch::Tensor ids = block_ids.to(torch::kInt64);
    const torch::Tensor valid = ids.ge(0).logical_and(ids.lt(tensor.size(0)));
    const torch::Tensor ordered = std::get<0>(torch::sort(ids.index({valid})));
    const torch::Tensor indices =
        std::get<0>(torch::unique_consecutive(ordered, false, false));
    reserve_bytes(indices.numel() * (tensor.nbytes() / tensor.size(0)));
    saved_.emplace_back(
        DeepseekV41SavedRows{tensor, indices, tensor.index_select(0, indices)});
  }

  void restore() {
    CHECK(!restored_) << "A cache transaction can be rolled back only once";
    for (const auto& entry : saved_) {
      torch::DeviceGuard guard(entry.target.device());
      if (entry.indices.defined()) {
        entry.target.index_copy_(0, entry.indices, entry.values);
      } else {
        CHECK_EQ(entry.target.sizes(), entry.values.sizes());
        entry.target.copy_(entry.values);
      }
    }
    restored_ = true;
  }

  int64_t saved_bytes() const { return saved_bytes_; }

 private:
  void check_tensor(const torch::Tensor& tensor) const {
    CHECK(!restored_);
    CHECK(tensor.defined() && tensor.is_contiguous());
    CHECK(tensor.device().type() == torch::kPrivateUse1);
  }

  void reserve_bytes(int64_t bytes) {
    CHECK_GE(bytes, 0);
    CHECK_LE(bytes, max_bytes_ - saved_bytes_)
        << "Cache transaction exceeds its explicit GPU memory budget";
    saved_bytes_ += bytes;
  }

  std::vector<DeepseekV41SavedRows> saved_;
  int64_t max_bytes_;
  int64_t saved_bytes_ = 0;
  bool restored_ = false;
};
}  // namespace xllm::layer
