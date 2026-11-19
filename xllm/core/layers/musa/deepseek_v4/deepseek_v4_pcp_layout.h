/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace xllm::layer::dsv4_musa_pcp {

struct SequenceLayoutSpan {
  int32_t sequence_id = 0;
  int64_t packed_start = 0;  // Sequence-major unpadded input row.
  int64_t length = 0;        // Query count, never a cumulative length.
  int64_t cached = 0;
  int64_t front_rows = 0;
  std::array<int64_t, 2> group_offsets{0, 0};
};

struct SequenceRow {
  int32_t sequence_id = 0;
  int64_t query_position = 0;
  int64_t absolute_position = 0;
  int64_t sequence_major_row = 0;
};

struct RankSequenceSpan {
  int32_t sequence_id = 0;
  int64_t local_begin = 0;           // Row inside this rank's padded buffer.
  int64_t length = 0;               // Valid rows only; excludes all padding.
  int64_t sequence_major_begin = 0;
  int64_t absolute_begin = 0;        // RoPE/causal position within this request.
};

struct CompressedLayoutSpan {
  int32_t sequence_id = 0;
  int64_t first_block = 0;
  int64_t split_block = 0;
  int64_t end_block = 0;
  int64_t output_offset = 0;  // Offset among completed rows of this batch.
  std::array<int64_t, 2> group_offsets{0, 0};
};

// Pure host geometry. No state or tensor ownership is implied by this layout.
// In particular, partial compressor blocks still need per-request relay state.
class SequenceLayout final {
 public:
  // MUSA's batch builder transports q_seq_lens and kv_seq_lens as cumulative
  // arrays with a leading zero, not independent per-request lengths.
  static std::optional<SequenceLayout> build_from_cumulative(
      const std::vector<int64_t>& query_offsets,
      const std::vector<int64_t>& kv_offsets,
      bool enable_microbatch = false) {
    if (query_offsets.size() < 2 || query_offsets.size() > 3 ||
        query_offsets.size() != kv_offsets.size() ||
        query_offsets.front() != 0 || kv_offsets.front() != 0) {
      return std::nullopt;
    }
    std::vector<int64_t> queries;
    std::vector<int64_t> cached;
    queries.reserve(query_offsets.size() - 1);
    cached.reserve(query_offsets.size() - 1);
    for (size_t index = 1; index < query_offsets.size(); ++index) {
      if (query_offsets[index] <= query_offsets[index - 1] ||
          kv_offsets[index] <= kv_offsets[index - 1]) {
        return std::nullopt;
      }
      const int64_t query = query_offsets[index] - query_offsets[index - 1];
      const int64_t kv = kv_offsets[index] - kv_offsets[index - 1];
      if (kv < query) {
        return std::nullopt;
      }
      queries.emplace_back(query);
      cached.emplace_back(kv - query);
    }
    return build(queries, cached, enable_microbatch);
  }

  // Rejecting admission is normal: callers retain their single-sequence queue.
  // This first contract is deliberately restricted to equal-shape microbatch=2.
  static std::optional<SequenceLayout> build(
      const std::vector<int64_t>& query_lengths,
      const std::vector<int64_t>& cached_lengths,
      bool enable_microbatch = false,
      int32_t cp_size = 2,
      int32_t tp_size = 4) {
    if (cp_size != 2 || tp_size != 4 || query_lengths.empty() ||
        query_lengths.size() > 2 ||
        query_lengths.size() != cached_lengths.size() ||
        (query_lengths.size() == 2 && !enable_microbatch)) {
      return std::nullopt;
    }
    constexpr int64_t kMinGroupRows = 1024;
    constexpr int64_t kSplitAlignment = 128;
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    SequenceLayout layout;
    layout.spans_.reserve(query_lengths.size());
    for (size_t index = 0; index < query_lengths.size(); ++index) {
      const int64_t length = query_lengths[index];
      const int64_t cached = cached_lengths[index];
      if (length < 2 * kMinGroupRows || cached < 0 ||
          length != query_lengths.front() ||
          cached != cached_lengths.front() || cached > kMax - length ||
          layout.total_rows_ > kMax - length) {
        return std::nullopt;
      }
      const int64_t front = (length / 2 / kSplitAlignment) * kSplitAlignment;
      const int64_t back = length - front;
      if (front < kMinGroupRows || back < kMinGroupRows) {
        return std::nullopt;
      }
      layout.spans_.emplace_back(SequenceLayoutSpan{
          static_cast<int32_t>(index), layout.total_rows_, length, cached,
          front, layout.group_rows_});
      layout.total_rows_ += length;
      layout.group_rows_[0] += front;
      layout.group_rows_[1] += back;
    }
    const int64_t max_group =
        std::max(layout.group_rows_[0], layout.group_rows_[1]);
    // All eight ranks use one common capacity; padding is never assigned a
    // position or slot. Division first avoids overflow in ceil(n / tp).
    layout.rank_capacity_ = max_group / tp_size + (max_group % tp_size != 0);
    if (layout.rank_capacity_ > kMax / (cp_size * tp_size)) {
      return std::nullopt;
    }
    layout.padded_rows_ = layout.rank_capacity_ * cp_size * tp_size;
    return layout;
  }

  const std::vector<SequenceLayoutSpan>& spans() const { return spans_; }
  int64_t total_rows() const { return total_rows_; }
  int64_t padded_rows() const { return padded_rows_; }
  int64_t rank_capacity() const { return rank_capacity_; }

  std::optional<int64_t> group_rows(int32_t cp_rank) const {
    if (cp_rank < 0 || cp_rank >= 2) {
      return std::nullopt;
    }
    return group_rows_[static_cast<size_t>(cp_rank)];
  }

  std::optional<SequenceRow> sequence_row(int64_t row) const {
    if (row < 0 || row >= total_rows_) {
      return std::nullopt;
    }
    for (const auto& span : spans_) {
      if (row < span.packed_start || row - span.packed_start >= span.length) {
        continue;
      }
      const int64_t position = row - span.packed_start;
      return SequenceRow{span.sequence_id, position, span.cached + position, row};
    }
    return std::nullopt;
  }

  std::optional<SequenceRow> rank_row(int32_t cp_rank,
                                    int32_t tp_rank,
                                    int64_t local_row) const {
    if (cp_rank < 0 || cp_rank >= 2 || tp_rank < 0 || tp_rank >= 4 ||
        local_row < 0 || local_row >= rank_capacity_) {
      return std::nullopt;
    }
    const size_t cp = static_cast<size_t>(cp_rank);
    const int64_t group_row = tp_rank * rank_capacity_ + local_row;
    if (group_row >= group_rows_[cp]) {
      return std::nullopt;
    }
    for (const auto& span : spans_) {
      const int64_t count = cp_rank == 0 ? span.front_rows
                                         : span.length - span.front_rows;
      if (group_row < span.group_offsets[cp] ||
          group_row - span.group_offsets[cp] >= count) {
        continue;
      }
      const int64_t position = group_row - span.group_offsets[cp] +
                               (cp_rank == 0 ? 0 : span.front_rows);
      return SequenceRow{span.sequence_id, position, span.cached + position,
                         span.packed_start + position};
    }
    return std::nullopt;
  }

  // A rank may straddle the boundary between two requests. Do not infer one
  // absolute position from its first row and apply that position to all rows.
  // These ordered intervals can slice Attention Q and gather per-request RoPE.
  std::optional<std::vector<RankSequenceSpan>> rank_spans(
      int32_t cp_rank, int32_t tp_rank) const {
    if (cp_rank < 0 || cp_rank >= 2 || tp_rank < 0 || tp_rank >= 4) {
      return std::nullopt;
    }
    const size_t cp = static_cast<size_t>(cp_rank);
    const int64_t rank_begin = tp_rank * rank_capacity_;
    const int64_t rank_end =
        std::min(rank_begin + rank_capacity_, group_rows_[cp]);
    std::vector<RankSequenceSpan> result;
    result.reserve(spans_.size());
    for (const auto& span : spans_) {
      const int64_t group_count = cp_rank == 0 ? span.front_rows
                                               : span.length - span.front_rows;
      const int64_t begin = std::max(rank_begin, span.group_offsets[cp]);
      const int64_t end =
          std::min(rank_end, span.group_offsets[cp] + group_count);
      if (begin >= end) {
        continue;
      }
      const int64_t query_begin = begin - span.group_offsets[cp] +
                                  (cp_rank == 0 ? 0 : span.front_rows);
      result.emplace_back(RankSequenceSpan{
          span.sequence_id, begin - rank_begin, end - begin,
          span.packed_start + query_begin, span.cached + query_begin});
    }
    return result;
  }

  // Index into flattened [cp, tp, rank_capacity] for a sequence-major row.
  std::optional<int64_t> rank_major_index(int64_t sequence_major_row) const {
    const auto row = sequence_row(sequence_major_row);
    if (!row.has_value()) {
      return std::nullopt;
    }
    const auto& span = spans_[static_cast<size_t>(row->sequence_id)];
    const int32_t cp_rank = row->query_position < span.front_rows ? 0 : 1;
    const int64_t group_row = span.group_offsets[static_cast<size_t>(cp_rank)] +
                              row->query_position -
                              (cp_rank == 0 ? 0 : span.front_rows);
    return cp_rank * 4 * rank_capacity_ + group_row;
  }

  // Gather indices: destination rank-major row -> source sequence-major row.
  // -1 denotes padding. An explicit cap prevents accidental huge allocations.
  std::optional<std::vector<int64_t>> pack_indices(int64_t max_rows) const {
    if (max_rows < padded_rows_ ||
        static_cast<uint64_t>(padded_rows_) >
            std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
      return std::nullopt;
    }
    std::vector<int64_t> indices;
    indices.reserve(static_cast<size_t>(padded_rows_));
    for (int32_t cp_rank = 0; cp_rank < 2; ++cp_rank) {
      for (int32_t tp_rank = 0; tp_rank < 4; ++tp_rank) {
        for (int64_t local_row = 0; local_row < rank_capacity_; ++local_row) {
          const auto row = rank_row(cp_rank, tp_rank, local_row);
          indices.emplace_back(row.has_value() ? row->sequence_major_row : -1);
        }
      }
    }
    return indices;
  }

  // Gather indices: destination sequence-major row -> source rank-major row.
  std::optional<std::vector<int64_t>> restore_indices(int64_t max_rows) const {
    if (max_rows < total_rows_ ||
        static_cast<uint64_t>(total_rows_) >
            std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
      return std::nullopt;
    }
    std::vector<int64_t> indices;
    indices.reserve(static_cast<size_t>(total_rows_));
    for (int64_t row = 0; row < total_rows_; ++row) {
      indices.emplace_back(*rank_major_index(row));
    }
    return indices;
  }

  // Half-open ranges of COMPLETED compressed blocks; never front_rows/ratio.
  // C128 continuation may enter and leave with partially filled state blocks.
  std::optional<std::vector<CompressedLayoutSpan>> compressed_spans(
      int64_t ratio) const {
    if (ratio != 1 && ratio != 4 && ratio != 128) {
      return std::nullopt;
    }
    std::vector<CompressedLayoutSpan> result;
    result.reserve(spans_.size());
    int64_t offset = 0;
    std::array<int64_t, 2> group_offsets{0, 0};
    for (const auto& span : spans_) {
      const int64_t first = span.cached / ratio;
      const int64_t split = (span.cached + span.front_rows) / ratio;
      const int64_t end = (span.cached + span.length) / ratio;
      result.emplace_back(CompressedLayoutSpan{
          span.sequence_id, first, split, end, offset, group_offsets});
      group_offsets[0] += split - first;
      group_offsets[1] += end - split;
      offset += end - first;
    }
    return result;
  }

 private:
  SequenceLayout() = default;
  std::vector<SequenceLayoutSpan> spans_;
  std::array<int64_t, 2> group_rows_{0, 0};
  int64_t total_rows_ = 0;
  int64_t padded_rows_ = 0;
  int64_t rank_capacity_ = 0;
};

}  // namespace xllm::layer::dsv4_musa_pcp
