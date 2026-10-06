// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace xllm::layer {

inline constexpr char kDsv4GraphSequenceView[] =
    "__musa_dsv4_graph_sequence_view";

inline constexpr char kDsv4BatchedSasKvLengths[] =
    "__musa_dsv4_batched_sas_kv_lengths";

inline bool dsv4_single_indexer_gemm_shape(int64_t rows,
                                           int64_t heads,
                                           int64_t head_dim,
                                           int64_t capacity,
                                           int64_t topk) {
  return rows == 7 && heads == 64 && head_dim == 128 &&
         topk == 512 && capacity >= topk;
}

inline bool dsv4_batched_indexer_bf16_fp32_shape(int64_t batch,
                                                int64_t query_width,
                                                int64_t heads,
                                                int64_t head_dim) {
  return batch > 1 && batch <= 16 && query_width == 7 && heads == 64 &&
         head_dim == 128;
}

inline bool dsv4_single_indexer_bf16_fp32_shape(int64_t batch,
                                                int64_t query_width,
                                                int64_t heads,
                                                int64_t head_dim,
                                                int64_t capacity) {
  return batch == 1 && query_width == 7 && heads == 64 && head_dim == 128 &&
         capacity >= 2560 && capacity <= 8192 && capacity % 512 == 0;
}

inline bool dsv4_projection_norm_copy_shape(int64_t rows,
                                             bool prefill,
                                             bool chunked_prefill) {
  return rows > 1 && rows <= 112 && !prefill && !chunked_prefill;
}

inline bool dsv4_indexer_relu_weight_shape(int64_t batch,
                                           int64_t query_width,
                                           int64_t heads,
                                           int64_t head_dim,
                                           int64_t capacity) {
  // The fused pointwise kernel improves measured B16 long verify, but smaller
  // batches regress. Keep them and unmeasured capacities on native operators.
  return batch == 16 &&
         dsv4_batched_indexer_bf16_fp32_shape(
             batch, query_width, heads, head_dim) &&
         (capacity == 4096 || capacity == 8192);
}

inline bool dsv4_batched_sas_query_layout(
    const std::vector<int32_t>& query_lengths, bool is_graph) {
  if (is_graph || query_lengths.size() < 2 || query_lengths.size() > 16) {
    return false;
  }
  const int32_t width = query_lengths.front();
  return width >= 5 && width <= 7 &&
      std::all_of(query_lengths.begin(), query_lengths.end(),
                  [width](int32_t value) { return value == width; });
}

inline bool dsv4_graph_sequence_views_enabled() {
  const char* value = std::getenv("XLLM_DSV4_GRAPH_SEQUENCE_VIEWS");
  return value != nullptr && std::string(value) == "1";
}

inline bool dsv4_fixed_graph_sequences(int64_t token_capacity,
                                       const std::vector<int32_t>& q_lens) {
  return dsv4_graph_sequence_views_enabled() && q_lens.size() > 1 &&
         q_lens.front() > 1 &&
         token_capacity ==
             q_lens.front() * static_cast<int64_t>(q_lens.size()) &&
         std::all_of(q_lens.begin(), q_lens.end(), [&](int32_t q) {
           return q == q_lens.front();
         });
}

inline int64_t dsv4_graph_compressed_stride(int64_t width, int64_t ratio) {
  return (width + ratio - 1) / ratio;
}

inline int64_t dsv4_graph_indexer_bucket(int64_t rows, bool middle_buckets) {
  const int64_t fine_capacity =
      ((std::max<int64_t>(rows, 1) + 511) / 512) * 512;
  int64_t capacity = 512;
  while (capacity < fine_capacity) {
    capacity *= 2;
  }
  // Add one fixed intermediate bucket per long power-of-two interval. The
  // bucket, not capture-time live length, is part of every structural key.
  const int64_t middle = capacity / 2 + capacity / 8;
  if (middle_buckets && capacity >= 4096 && fine_capacity <= middle) {
    return middle;
  }
  return capacity;
}

// Allocation capacity, never causal key length. Use this same policy for the
// Graph key, persistent block-table capacity and the attention launch.
inline int64_t dsv4_graph_indexer_capacity(int64_t rows, bool multi_sequence) {
  if (!multi_sequence) {
    return ((std::max<int64_t>(rows, 1) + 511) / 512) * 512;
  }
  const char* value = std::getenv("XLLM_DSV4_MID_INDEXER_BUCKETS");
  const bool middle = value != nullptr && value[0] == '1' && value[1] == '\0';
  return dsv4_graph_indexer_bucket(rows, middle);
}

// Reuse is deliberately limited to the long sparse path with one common C4
// capacity. Short/full-attention and long C128 fallbacks keep exact keys.
inline int64_t dsv4_reusable_graph_capacity(
    const std::vector<int32_t>& cumulative_kv,
    int64_t query_width) {
  if (query_width <= 1 || cumulative_kv.size() < 3 ||
      cumulative_kv.front() != 0) {
    return 0;
  }
  int64_t capacity = 0;
  for (size_t i = 1; i < cumulative_kv.size(); ++i) {
    const int64_t length =
        static_cast<int64_t>(cumulative_kv[i]) - cumulative_kv[i - 1];
    if (length < 4096 || length > 65536 || length < query_width) {
      return 0;
    }
    const int64_t row_capacity = dsv4_graph_indexer_capacity(length / 4, true);
    if (capacity != 0 && capacity != row_capacity) {
      return 0;
    }
    capacity = row_capacity;
  }
  return capacity;
}

// The structural graph key includes the C4 allocation bucket. All replay KV
// lengths in that bucket satisfy floor(KV / 4) <= capacity. Include the three
// uncompressed tail tokens and round UP to pages, never to the current length.
// A zero return leaves B1 and incompatible graph families unchanged.
inline int64_t dsv4_graph_block_table_columns(
    const std::vector<int32_t>& cumulative_kv,
    int64_t query_width,
    int64_t block_size) {
  const int64_t capacity =
      dsv4_reusable_graph_capacity(cumulative_kv, query_width);
  if (capacity == 0) {
    return 0;
  }
  CHECK_GT(block_size, 0);
  const int64_t maximum_kv = capacity * 4 + 3;
  return (maximum_kv + block_size - 1) / block_size;
}

template <typename HasFullCapture>
inline uint32_t dsv4_select_existing_graph_sequences(
    uint32_t active,
    uint32_t maximum,
    HasFullCapture has_full_capture) {
  CHECK(active <= maximum);
  if (active <= 1) {
    return active;
  }
  for (uint32_t sequences = active; sequences <= maximum; ++sequences) {
    if (has_full_capture(sequences)) {
      return sequences;
    }
  }
  // Never create a larger graph using dummy rows. Capture the actual batch.
  return active;
}

inline std::vector<int32_t> dsv4_graph_query_lengths(int32_t active,
                                                     int32_t physical,
                                                     int32_t width,
                                                     bool fixed_storage) {
  CHECK(active >= 0 && physical >= active && width > 0);
  std::vector<int32_t> lengths(static_cast<size_t>(physical), 0);
  std::fill_n(lengths.begin(), fixed_storage ? physical : active, width);
  return lengths;
}

// Query rows describe storage, while KV rows describe validity. In a reused
// graph inactive trailing requests still own q-width rows, but zero KV length.
inline bool dsv4_fixed_graph_query_layout(
    const std::vector<int32_t>& cumulative_q,
    int64_t physical,
    int64_t width) {
  if (physical <= 1 || width <= 1 ||
      cumulative_q.size() != static_cast<size_t>(physical + 1) ||
      cumulative_q.front() != 0) {
    return false;
  }
  for (int64_t row = 0; row < physical; ++row) {
    if (static_cast<int64_t>(cumulative_q[row + 1]) - cumulative_q[row] !=
        width) {
      return false;
    }
  }
  return true;
}

inline std::vector<int32_t> dsv4_compact_graph_kv_lengths(
    const std::vector<int32_t>& source,
    int32_t active,
    int32_t physical,
    int32_t width,
    int64_t actual_tokens,
    int32_t fallback_max) {
  CHECK(active >= 0 && physical >= active && width > 0);
  CHECK(actual_tokens >= static_cast<int64_t>(active) * width);
  const bool cumulative = !source.empty() && source.front() == 0;
  const bool per_sequence =
      cumulative && source.size() == static_cast<size_t>(active + 1);
  const bool per_token =
      cumulative && source.size() == static_cast<size_t>(actual_tokens + 1);
  const bool raw_rows = source.size() == static_cast<size_t>(actual_tokens);
  CHECK(active == 0 || physical == active || per_sequence || per_token ||
        raw_rows)
      << "Padded DSV4 Graph requires a recognized live KV length layout";
  std::vector<int32_t> result;
  result.reserve(static_cast<size_t>(physical + 1));
  result.emplace_back(0);
  for (int32_t row = 0; row < physical; ++row) {
    int32_t length = 0;
    if (row < active) {
      const size_t last = static_cast<size_t>((row + 1) * width);
      length = per_sequence ? source[row + 1] - source[row]
               : per_token  ? source[last] - source[last - 1]
               : raw_rows   ? source[last - 1]
                            : fallback_max;
      length = std::max<int32_t>(length, 1);
    }
    result.emplace_back(result.back() + length);
  }
  return result;
}

}  // namespace xllm::layer
