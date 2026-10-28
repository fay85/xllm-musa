// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

namespace xllm::layer {

inline torch::Tensor dsv4_paged_compressed_table_view(
    const torch::Tensor& registered_table,
    const torch::Tensor& historical_table,
    bool dynamic_causal,
    bool direct_multi_sequence,
    int64_t sequence_index) {
  // Graph C128 reserves one output row even before the first128-token
  // boundary. That capacity can exceed the real compressed history and leave
  // historical_table unset. Dynamic kernels consume published cache directly
  // and derive validity from each query's device KV length, not that tail.
  const auto& table = dynamic_causal ? registered_table : historical_table;
  if (dynamic_causal) {
    CHECK(table.defined() && table.dim() == 2)
        << "Dynamic DSV4 attention requires the registered2D cache table";
  }
  if (!table.defined() || !direct_multi_sequence) {
    return table;
  }
  CHECK_EQ(table.dim(), 2);
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, table.size(0));
  return table.narrow(/*dim=*/0, sequence_index, /*length=*/1);
}

}  // namespace xllm::layer
