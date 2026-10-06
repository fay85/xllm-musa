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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace xllm::layer {

inline int64_t dsv4_full_kv_threshold() {
  const char* value = std::getenv("XLLM_DSV4_FULL_KV_THRESHOLD");
  if (value != nullptr && *value != '\0') {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed >= 0) {
      return static_cast<int64_t>(parsed);
    }
  }
  return 2048;
}

// The executor and Attention must agree on this host-side branch. A replay
// cannot add the Indexer/compressed-history nodes missing from a short capture.
inline bool dsv4_uses_full_kv(int64_t kv_len,
                              int64_t threshold,
                              int64_t window_size) {
  return threshold > 0 && kv_len <= threshold &&
         (window_size <= 0 || kv_len <= window_size);
}

// Direct q7 short-context Graph kernels receive live per-query KV lengths.
// Keep the full/sparse bit and allocation buckets in the caller's key, but
// omit exact length only for the explicitly validated <=128-token family.
// Legacy scalar compressed fallback and extreme C128 lengths remain exact.
inline int64_t dsv4_graph_baked_kv_length(int64_t kv_len,
                                        bool full_kv,
                                        bool dynamic_short) {
  const bool live_short = dynamic_short && kv_len >= 0 && kv_len <= 128;
  return ((full_kv && !live_short) || kv_len / 128 > 512) ? kv_len : 0;
}

// Direct multi-sequence Attention selects full-KV versus sparse attention
// from the batch maximum, not each row's length. This opt-in family is bounded
// to the first C4 allocation bucket. Validity still comes from live device
// positions; callers must keep the batch-wide full/sparse bit in the key.
// Return zero for B1, dummy rows, other query widths or unvalidated lengths.
inline int64_t dsv4_graph_short_batch_max_kv(
    const std::vector<int32_t>& cumulative_kv,
    int64_t query_width) {
  if (query_width != 7 || cumulative_kv.size() < 3 ||
      cumulative_kv.front() != 0) {
    return 0;
  }
  int64_t maximum = 0;
  for (std::size_t row = 1; row < cumulative_kv.size(); ++row) {
    const int64_t length = static_cast<int64_t>(cumulative_kv[row]) -
                           cumulative_kv[row - 1];
    if (length < query_width || length > 2048) {
      return 0;
    }
    maximum = length > maximum ? length : maximum;
  }
  return maximum;
}

inline int64_t dsv4_compression_boundaries(int64_t start,
                                           int64_t tokens,
                                           int64_t ratio) {
  const int64_t first = (ratio - ((start + 1) % ratio)) % ratio;
  return first < tokens ? 1 + (tokens - 1 - first) / ratio : 0;
}

// Callers validate field capacities. C4 and C128 output tensor shapes both
// enter the key, independently of their replay-time absolute positions.
inline uint32_t dsv4_compact_graph_key(uint32_t token_bucket,
                                       uint32_t indexer_capacity,
                                       uint32_t c4_outputs,
                                       uint32_t c128_outputs,
                                       bool full_kv) {
  return (uint32_t{1} << 31) | (token_bucket << 16) |
         (indexer_capacity << 6) | (static_cast<uint32_t>(full_kv) << 5) |
         (c128_outputs << 2) | c4_outputs;
}

}  // namespace xllm::layer
