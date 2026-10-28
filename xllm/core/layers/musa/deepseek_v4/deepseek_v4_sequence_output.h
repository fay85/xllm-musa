// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
#pragma once
#include <torch/torch.h>

#include <vector>

namespace xllm::layer {

// The caller must finish all writes before use, and keep the backing storage
// alive until its consumer completes. Disjoint scratch allocations or reordered
// request rows retain the original cat semantics instead of manufacturing a view.
inline torch::Tensor dsv4_adjacent_row_view_or_cat(
    const std::vector<torch::Tensor>& rows) {
  TORCH_CHECK(!rows.empty());
  const auto& first = rows.front();
  TORCH_CHECK(first.dim() > 0);
  int64_t total_rows = 0;
  int64_t next_offset = first.storage_offset();
  bool adjacent = first.is_contiguous();
  for (const auto& row : rows) {
    adjacent = adjacent && row.dim() == first.dim() && row.is_contiguous() &&
        row.scalar_type() == first.scalar_type() && row.device() == first.device() &&
        row.is_alias_of(first) && row.strides() == first.strides() &&
        row.storage_offset() == next_offset;
    if (!adjacent) {
      return torch::cat(rows, 0);
    }
    for (int64_t dim = 1; dim < row.dim(); ++dim) {
      if (row.size(dim) != first.size(dim)) {
        return torch::cat(rows, 0);
      }
    }
    total_rows += row.size(0);
    next_offset += row.numel();
  }
  auto shape = first.sizes().vec();
  shape[0] = total_rows;
  return first.as_strided(shape, first.strides(), first.storage_offset());
}

inline torch::Tensor dsv4_retain_sequence_output(
    const torch::Tensor& output, int64_t sequence_count) {
  TORCH_CHECK(sequence_count > 0);
  // Both paged compact verify and MATE may return shared per-layer scratch.
  // Each retained row must outlive the next request's kernel invocation.
  // B1 keeps its original alias and incurs no allocation/copy.
  return sequence_count > 1 ? output.clone() : output;
}

inline torch::Tensor dsv4_sequence_output_buffer(
    const torch::Tensor& query, const torch::Tensor& workspace,
    int64_t sequence_count) {
  TORCH_CHECK(sequence_count > 0);
  if (sequence_count == 1) {
    return workspace.narrow(0, 0, query.size(0));
  }
  // The caller retains all sequence outputs until concatenation. Returning
  // the same scratch prefix for every request would make every retained
  // output contain the last request's attention result.
  return torch::empty(query.sizes(), query.options());
}

}  // namespace xllm::layer
