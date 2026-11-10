/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/decode_graph_bucket.h"

#include <glog/logging.h>

#include <algorithm>

#include "core/framework/config/execution_config.h"
#include "core/runtime/options.h"

namespace xllm::runtime {
namespace {

constexpr int64_t kGraphTokenStep = 16;

}  // namespace

std::vector<int32_t> normalize_decode_graph_capture_sizes(
    const std::vector<int32_t>& raw_sizes,
    int64_t query_width,
    int64_t max_tokens,
    int32_t max_sequences) {
  if (raw_sizes.empty() || query_width <= 0 || max_tokens <= 0 ||
      max_sequences <= 0) {
    return {};
  }
  std::vector<int32_t> normalized_sizes = raw_sizes;
  std::sort(normalized_sizes.begin(), normalized_sizes.end());
  CHECK_GT(normalized_sizes.front(), 0)
      << "graph_capture_sizes must contain positive token rows";
  // Check the width before multiplication / rounding to avoid overflow even
  // when a caller provides an invalid width larger than the token budget.
  const int64_t token_limit =
      std::min<int64_t>(normalized_sizes.back(), max_tokens);
  if (query_width > token_limit) {
    return {};
  }
  const int64_t sequence_limit =
      std::min<int64_t>(max_sequences, token_limit / query_width);
  const int64_t max_capture_tokens = sequence_limit * query_width;
  std::vector<int32_t> sizes;
  sizes.reserve(raw_sizes.size());
  for (int32_t size : normalized_sizes) {
    const int64_t rounded_size =
        ((static_cast<int64_t>(size) + query_width - 1) / query_width) *
        query_width;
    if (rounded_size > max_capture_tokens) {
      continue;
    }
    if (sizes.empty() || sizes.back() != rounded_size) {
      sizes.emplace_back(static_cast<int32_t>(rounded_size));
    }
  }
  return sizes;
}

std::vector<int32_t> get_decode_graph_capture_sizes(const Options& options,
                                                    int64_t query_width) {
  const auto& config = ExecutionConfig::get_instance();
  if (config.graph_capture_sizes().empty()) {
    return {};
  }
  auto sizes = normalize_decode_graph_capture_sizes(
      config.graph_capture_sizes(),
      query_width,
      options.max_tokens_per_batch(),
      std::min(
          options.max_seqs_per_batch(),
          std::max<int32_t>(1, config.acl_graph_decode_batch_size_limit())));
  CHECK(!sizes.empty())
      << "graph_capture_sizes has no shape within decode query width and "
         "capacity";
  return sizes;
}

int64_t get_decode_graph_token_bucket(
    int64_t num_tokens,
    bool enable_no_padding,
    const std::vector<int32_t>& capture_sizes) {
  if (capture_sizes.empty()) {
    return get_decode_graph_token_bucket(num_tokens, enable_no_padding);
  }
  const auto bucket =
      std::lower_bound(capture_sizes.begin(), capture_sizes.end(), num_tokens);
  // Startup rejects this combination; keep exact-only semantics for direct
  // callers instead of silently padding a no-padding request.
  if (bucket == capture_sizes.end() ||
      (enable_no_padding && *bucket != num_tokens)) {
    return 0;
  }
  return *bucket;
}

int64_t get_decode_graph_token_bucket(int64_t num_tokens,
                                      bool enable_no_padding) {
  if (enable_no_padding) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  }
  if (num_tokens <= 2) {
    return 2;
  }
  if (num_tokens <= 4) {
    return 4;
  }
  if (num_tokens <= 8) {
    return 8;
  }

  return ((num_tokens + kGraphTokenStep - 1) / kGraphTokenStep) *
         kGraphTokenStep;
}

std::vector<int32_t> get_decode_graph_dp_token_counts(
    const std::vector<int32_t>& token_counts,
    int32_t graph_token_count) {
  if (token_counts.empty()) {
    return {};
  }
  std::vector<int32_t> padded_counts = token_counts;
  for (int32_t& token_count : padded_counts) {
    token_count = token_count > 0 ? graph_token_count : 1;
  }
  return padded_counts;
}

int64_t get_decode_graph_dp_layout_token_count(int32_t dp_size,
                                               int32_t graph_token_count) {
  return static_cast<int64_t>(dp_size) * graph_token_count;
}

}  // namespace xllm::runtime
