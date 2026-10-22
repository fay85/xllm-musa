/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm::layer {

struct Dsv4PendingCacheRows {
  torch::Tensor values;
  int64_t row_begin = 0;
  int64_t row_end = 0;
  int64_t tensor_end = 0;
};

inline Dsv4PendingCacheRows dsv4_pending_cache_rows(
    const std::vector<torch::Tensor>& parts,
    int64_t first_tensor,
    int64_t first_row) {
  const int64_t tensor_count = static_cast<int64_t>(parts.size());
  CHECK_GE(first_tensor, 0);
  CHECK_LE(first_tensor, tensor_count);
  CHECK_GE(first_row, 0);
  if (first_tensor == tensor_count) {
    return {torch::Tensor(), first_row, first_row, tensor_count};
  }
  // Earlier parts have already been published. Concatenating them again only
  // to slice them away creates quadratic copying across a request loop.
  auto pending = tensor_count - first_tensor == 1
                     ? parts[first_tensor].contiguous()
                     : torch::cat(torch::TensorList(parts).slice(
                                      static_cast<size_t>(first_tensor)),
                                  /*dim=*/0);
  CHECK(pending.defined());
  CHECK_GE(pending.dim(), 1);
  return {pending, first_row, first_row + pending.size(0), tensor_count};
}

}  // namespace xllm::layer
