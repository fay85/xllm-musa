/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "core/layers/musa/deepseek_v41/csa2_attention.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"

namespace xllm::layer {

// Native chronological SWA gather, sparse sink attention, and deferred ring
// commit. Uses the actual scheduler's paged tensor views and preserves their
// physical layout for host swap/P-D transfer. C1/eager; no model Python call.
// The caller supplies committed compressed rows and source-layer top-k picks.
class DeepseekV41PagedAttention final {
 public:
  DeepseekV41PagedAttention(torch::Tensor cache,
                            int64_t window,
                            double scale,
                            std::string sparse_library_path = {})
      : cache_(std::move(cache)),
        window_(window),
        attention_(scale, 32 * 1024 * 1024, std::move(sparse_library_path)) {
    CHECK(cache_.defined());
    CHECK(cache_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(cache_.scalar_type(), torch::kBFloat16);
    CHECK_EQ(cache_.dim(), 3);
    CHECK(cache_.is_contiguous());
    CHECK_GT(cache_.size(0), 0);
    CHECK_GT(cache_.size(1), 0);
    CHECK_GT(cache_.size(2), 0);
    CHECK_GT(window_, 0);
  }

  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& current_kv,
                        const torch::Tensor& block_table,
                        int64_t context_length,
                        const torch::Tensor& sink,
                        const std::optional<torch::Tensor>& compressed_kv,
                        const std::optional<torch::Tensor>& compressed_picks) {
    const DeepseekV41ProfileRange range("V41/Attention/SWA_sparse_and_commit");
    CHECK(query.defined() && current_kv.defined() && block_table.defined());
    CHECK_EQ(query.device(), cache_.device());
    CHECK_EQ(current_kv.device(), cache_.device());
    CHECK_EQ(block_table.device(), cache_.device());
    CHECK_EQ(query.scalar_type(), torch::kBFloat16);
    CHECK_EQ(current_kv.scalar_type(), torch::kBFloat16);
    CHECK_EQ(query.dim(), 3);
    CHECK_EQ(query.size(2), cache_.size(2));
    CHECK_EQ(current_kv.sizes(),
             torch::IntArrayRef({query.size(0), cache_.size(2)}));
    CHECK_EQ(block_table.dim(), 2);
    CHECK_EQ(block_table.size(0), 1) << "Native paged attention currently C1";
    CHECK_GT(block_table.size(1), 0);
    CHECK(block_table.scalar_type() == torch::kInt32 ||
          block_table.scalar_type() == torch::kInt64);
    CHECK_GE(block_table.size(1) * cache_.size(1), window_);
    CHECK_GE(context_length, query.size(0));
    CHECK_EQ(compressed_kv.has_value(), compressed_picks.has_value());
    torch::DeviceGuard guard(cache_.device());
    if (query.size(0) == 0) {
      return torch::empty_like(query);
    }
    const int64_t previous = context_length - query.size(0);
    const auto indices_options =
        torch::TensorOptions().device(cache_.device()).dtype(torch::kInt64);
    const torch::Tensor ring_positions =
        torch::arange(window_, indices_options) + previous - window_;
    const torch::Tensor flat_cache = cache_.view({-1, cache_.size(2)});
    const torch::Tensor gathered =
        flat_cache.index_select(0,
                                physical_slots(block_table,
                                               ring_positions.clamp_min(0),
                                               /*padding_read=*/true));
    const torch::Tensor ring =
        torch::where(ring_positions.ge(0).unsqueeze(-1), gathered, 0);
    torch::Tensor rows = torch::cat({ring, current_kv}, 0);
    const torch::Tensor positions =
        torch::arange(query.size(0), indices_options) + previous;
    const torch::Tensor desired = positions.unsqueeze(-1) - window_ + 1 +
                                  torch::arange(window_, indices_options);
    torch::Tensor picks =
        torch::where(desired.ge(0), desired - previous + window_, -1);
    if (compressed_kv.has_value()) {
      const torch::Tensor& compressed = compressed_kv.value();
      const torch::Tensor& extra = compressed_picks.value();
      CHECK(compressed.defined() && extra.defined());
      CHECK_EQ(compressed.device(), cache_.device());
      CHECK_EQ(extra.device(), cache_.device());
      CHECK_EQ(compressed.scalar_type(), torch::kBFloat16);
      CHECK(extra.scalar_type() == torch::kInt32 ||
            extra.scalar_type() == torch::kInt64);
      CHECK_EQ(compressed.dim(), 2);
      CHECK_EQ(compressed.size(1), cache_.size(2));
      CHECK_EQ(extra.dim(), 2);
      CHECK_EQ(extra.size(0), query.size(0));
      const torch::Tensor valid =
          extra.ge(0).logical_and(extra.lt(compressed.size(0)));
      const torch::Tensor mapped =
          torch::where(valid, extra.to(torch::kInt64) + rows.size(0), -1);
      picks = torch::cat({picks, mapped}, -1);
      rows = torch::cat({rows, compressed}, 0);
    }
    // Always read before committing this chunk, including M=1. The oldest
    // ring candidate overwritten by a decode token is outside its window.
    const torch::Tensor result = attention_.forward(query, rows, picks, sink);
    commit(physical_slots(block_table, positions, /*padding_read=*/false),
           current_kv);
    return result;
  }

 private:
  torch::Tensor physical_slots(const torch::Tensor& table,
                               const torch::Tensor& positions,
                               bool padding_read) const {
    const int64_t block_size = cache_.size(1);
    const torch::Tensor columns =
        torch::floor_divide(positions, block_size).remainder(table.size(1));
    const torch::Tensor blocks =
        table.select(0, 0).index_select(0, columns).to(torch::kInt64);
    // Read-side ring candidates may refer to unused scheduler padding. Write
    // slots retain the sign and are filtered rather than corrupting block 0.
    return (padding_read ? blocks.clamp_min(0) : blocks) * block_size +
           positions.remainder(block_size);
  }

  void commit(const torch::Tensor& slots, const torch::Tensor& values) {
    const DeepseekV41ProfileRange range("V41/Attention/SWA_commit");
    const torch::Tensor order =
        torch::argsort(slots, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    const torch::Tensor sorted = slots.index_select(0, order);
    torch::Tensor keep =
        sorted.ge(0).logical_and(sorted.lt(cache_.size(0) * cache_.size(1)));
    if (sorted.numel() > 1) {
      keep.narrow(0, 0, sorted.numel() - 1)
          .logical_and_(sorted.narrow(0, 0, sorted.numel() - 1)
                            .ne(sorted.narrow(0, 1, sorted.numel() - 1)));
    }
    const torch::Tensor latest = order.index({keep});
    cache_.view({-1, cache_.size(2)})
        .index_copy_(
            0, slots.index_select(0, latest), values.index_select(0, latest));
  }

  torch::Tensor cache_;
  int64_t window_;
  DeepseekV41SparseAttention attention_;
};

}  // namespace xllm::layer
