/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::layer {

// Preserve the request-local GEMM shapes and output-buffer dispatch of the
// independent compressor path. A single wider GEMM can choose a different
// reduction kernel (including bypassing the <=8-row Indexer GEMM) and change
// BF16 rounding. Packing these results still enables batched compression.
inline torch::Tensor dsv4_pack_compressor_projections(
    const torch::Tensor& input,
    const torch::Tensor& packed_weight,
    int64_t query_width,
    int64_t main_width,
    int64_t index_width,
    const std::optional<torch::Tensor>& output = std::nullopt) {
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(packed_weight.dim(), 2);
  CHECK_GT(query_width, 0);
  CHECK_GT(main_width, 0);
  CHECK_GT(index_width, 0);
  CHECK_EQ(input.size(0) % query_width, 0);
  CHECK_EQ(input.size(1), packed_weight.size(1));
  CHECK_EQ(packed_weight.size(0), 2 * main_width + 2 * index_width);
  const int64_t sequences = input.size(0) / query_width;
  CHECK_GT(sequences, 0);
  const std::array<int64_t, 4> widths{
      main_width, main_width, index_width, index_width};
  std::vector<torch::Tensor> packed_sequences;
  packed_sequences.reserve(sequences);
  for (int64_t sequence = 0; sequence < sequences; ++sequence) {
    auto local_input = input.narrow(0, sequence * query_width, query_width);
    std::vector<torch::Tensor> projections;
    projections.reserve(widths.size());
    int64_t offset = 0;
    for (const int64_t width : widths) {
      auto buffer = torch::empty({query_width, width}, input.options());
      projections.emplace_back(
          kernel::musa::matmul(local_input,
                               packed_weight.narrow(0, offset, width),
                               /*bias=*/std::nullopt,
                               buffer));
      offset += width;
    }
    packed_sequences.emplace_back(torch::cat(projections, /*dim=*/1));
  }
  torch::Tensor packed = sequences == 1
                             ? packed_sequences.front()
                             : torch::cat(packed_sequences, /*dim=*/0);
  if (!output.has_value()) {
    return packed;
  }
  CHECK(output->defined());
  CHECK(output->sizes() == packed.sizes());
  CHECK(output->device() == packed.device());
  CHECK(output->scalar_type() == packed.scalar_type());
  output->copy_(packed);
  return output.value();
}

// C4 projections are packed as [main KV, main score, indexer KV, indexer
// score]. Dense short-context attention consumes only the main pair. Select its
// view without changing the producer's row stride or allocating a Graph buffer.
inline torch::Tensor dsv4_compressor_projection_view(
    const torch::Tensor& projections,
    int64_t main_width,
    int64_t index_width,
    bool uses_indexer) {
  CHECK_EQ(projections.dim(), 2);
  CHECK_GT(main_width, 0);
  CHECK_GE(index_width, 0);
  const int64_t main_columns = 2 * main_width;
  const int64_t joint_columns = main_columns + 2 * index_width;
  const int64_t actual_columns = projections.size(1);
  CHECK(actual_columns == main_columns ||
        (index_width > 0 && actual_columns == joint_columns))
      << "Unexpected packed compressor projection width: " << actual_columns;
  if (uses_indexer) {
    CHECK_GT(index_width, 0);
    CHECK_EQ(actual_columns, joint_columns)
        << "Missing Indexer projection columns";
    return projections;
  }
  if (actual_columns == main_columns) {
    return projections;
  }
  return projections.narrow(/*dim=*/1, /*start=*/0, /*length=*/main_columns);
}

}  // namespace xllm::layer
