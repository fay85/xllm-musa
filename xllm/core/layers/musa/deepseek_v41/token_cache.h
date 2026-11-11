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
#include <utility>

namespace xllm::layer {

// The original-QDQ BF16 token cache carrier, with scheduler-owned physical
// storage and page tables. Neither the layer nor this view owns a private
// copy of history; host swap and Mooncake transfer see this same tensor.
class DeepseekV41TokenCache final {
 public:
  explicit DeepseekV41TokenCache(torch::Tensor cache)
      : cache_(std::move(cache)) {
    CHECK(cache_.defined());
    CHECK(cache_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(cache_.scalar_type(), torch::kBFloat16);
    CHECK_EQ(cache_.dim(), 3);
    CHECK(cache_.is_contiguous());
    CHECK_GT(cache_.size(0), 0);
    CHECK_GT(cache_.size(1), 0);
    CHECK_GT(cache_.size(2), 0);
  }

  void commit(const torch::Tensor& values,
              const torch::Tensor& rows,
              const torch::Tensor& table) {
    check_table(table);
    CHECK(values.defined() && rows.defined());
    CHECK_EQ(values.device(), cache_.device());
    CHECK_EQ(rows.device(), cache_.device());
    CHECK_EQ(values.scalar_type(), cache_.scalar_type());
    CHECK_EQ(values.dim(), 2);
    CHECK_EQ(values.size(1), cache_.size(2));
    CHECK_EQ(rows.dim(), 1);
    CHECK_EQ(rows.size(0), values.size(0));
    CHECK(rows.scalar_type() == torch::kInt64 ||
          rows.scalar_type() == torch::kInt32);
    torch::DeviceGuard guard(cache_.device());
    if (values.size(0) == 0) {
      return;
    }
    // Compressor commits are chronological and unique. Invalid real write
    // metadata is an error; never clamp it into physical block zero.
    cache_.view({-1, cache_.size(2)})
        .index_copy_(0, slots(rows.to(torch::kInt64), table, false), values);
  }

  torch::Tensor gather(const torch::Tensor& table,
                       int64_t committed,
                       int64_t capacity) const {
    check_table(table);
    CHECK_GE(committed, 0);
    CHECK_GE(capacity, committed);
    CHECK_LE(capacity, table.size(1) * cache_.size(1));
    torch::DeviceGuard guard(cache_.device());
    const torch::Tensor rows =
        torch::arange(capacity, table.options().dtype(torch::kInt64));
    const torch::Tensor values = cache_.view({-1, cache_.size(2)})
                                     .index_select(0, slots(rows, table, true));
    return torch::where(rows.lt(committed).unsqueeze(-1), values, 0);
  }

 private:
  void check_table(const torch::Tensor& table) const {
    CHECK(table.defined());
    CHECK_EQ(table.device(), cache_.device());
    CHECK_EQ(table.dim(), 2);
    CHECK_EQ(table.size(0), 1) << "Native token cache currently C1/eager";
    CHECK_GT(table.size(1), 0);
    CHECK(table.scalar_type() == torch::kInt32 ||
          table.scalar_type() == torch::kInt64);
  }

  torch::Tensor slots(const torch::Tensor& rows,
                      const torch::Tensor& table,
                      bool padding_read) const {
    const torch::Tensor blocks =
        table.select(0, 0)
            .index_select(0, torch::floor_divide(rows, cache_.size(1)))
            .to(torch::kInt64);
    return (padding_read ? blocks.clamp_min(0) : blocks) * cache_.size(1) +
           rows.remainder(cache_.size(1));
  }

  torch::Tensor cache_;
};

}  // namespace xllm::layer
