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

#pragma once

#include <cstdint>
#include <vector>

namespace xllm::runtime {

struct Options;

// Decode graph warmup fields supplied by the owning Engine. For MTP,
// num_decoding_tokens is the number of token rows produced per sequence in a
// decode step; it is normally num_speculative_tokens + 1.
struct DecodeGraphWarmupConfig {
  int64_t num_decoding_tokens = 1;
  int32_t num_speculative_tokens = 0;
  bool enable_graph_mode_decode_no_padding = false;
  int32_t max_graph_batch_size = 0;
  // Effective, sorted token-row shapes supplied by the NPU engine.
  std::vector<int32_t> graph_capture_sizes;
};

// vLLM-Ascend's uniform-decode policy: round each raw size up to query_width,
// deduplicate, and exclude shapes outside the raw maximum or token/sequence
// capacities. Positive raw sizes are sorted and deduplicated here as well as
// in ExecutionConfig parsing, since programmatic setters can bypass parsing.
// Empty input means the default policy; empty output from a
// nonempty input is an invalid configuration, not a request for defaults.
std::vector<int32_t> normalize_decode_graph_capture_sizes(
    const std::vector<int32_t>& raw_sizes,
    int64_t query_width,
    int64_t max_tokens,
    int32_t max_sequences);

// Get effective shapes using the same query width and local capacities for
// engine warmup, executors and worker DP padding. Every shape is a multiple
// of query_width; consumers select from this list without normalizing again.
// Empty config stays empty.
std::vector<int32_t> get_decode_graph_capture_sizes(const Options& options,
                                                    int64_t query_width);

// Returns the padded token-row bucket shared by decode graph executors. When
// no-padding mode is enabled each exact token count is its own graph shape.
// With a nonempty effective list, return the smallest compatible bucket or
// zero if no configured shape covers the request.
int64_t get_decode_graph_token_bucket(int64_t num_tokens,
                                      bool enable_no_padding);

int64_t get_decode_graph_token_bucket(
    int64_t num_tokens,
    bool enable_no_padding,
    const std::vector<int32_t>& capture_sizes);

// Returns the per-DP token rows used to build graph-mode DP metadata. Active
// shards use the graph execution width, while empty shards retain their
// single fake row; raw token counts remain separate for lm-head indices.
std::vector<int32_t> get_decode_graph_dp_token_counts(
    const std::vector<int32_t>& token_counts,
    int32_t graph_token_count);

// Returns the dense token layout consumed after DP padding.
int64_t get_decode_graph_dp_layout_token_count(int32_t dp_size,
                                               int32_t graph_token_count);

}  // namespace xllm::runtime
