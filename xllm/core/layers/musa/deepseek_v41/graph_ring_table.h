/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

namespace xllm::layer {

// Expand a ring's logical page mapping into a fixed Graph table. Callers must
// bound every accessed logical page to target.size(1). Within that domain,
// target[b] == source[b % source.size(1)], including a one-page physical ring.
// Padding with -1 would change valid ring accesses, and retaining the source
// width in the Graph key creates a capture for every growing context page.
inline void deepseek_v41_copy_graph_ring_table(torch::Tensor& target,
                                               const torch::Tensor& source) {
  CHECK_EQ(source.dim(), 2);
  CHECK_EQ(target.dim(), 2);
  CHECK_EQ(source.size(0), 1);
  CHECK_EQ(target.size(0), 1);
  CHECK_GT(source.size(1), 0);
  CHECK_GT(target.size(1), 0);
  CHECK_EQ(source.device(), target.device());
  CHECK_EQ(source.scalar_type(), target.scalar_type());
  if (source.size(1) == target.size(1)) {
    target.copy_(source);
    return;
  }
  const auto columns =
      torch::arange(target.size(1), source.options().dtype(torch::kInt64))
          .remainder(source.size(1));
  target.copy_(source.index_select(1, columns));
}

}  // namespace xllm::layer
