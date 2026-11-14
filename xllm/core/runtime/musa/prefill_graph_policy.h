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

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace xllm::runtime::musa {

// Minimum physical free memory floor (1.5 GiB) required before starting a new
// MUSA prefill graph capture, even if musa_prefill_warmup_reserve_mb is set
// lower.
inline constexpr std::size_t kMinPrefillCaptureFreeBytes =
    1536ULL * 1024ULL * 1024ULL;

// Padding is for a single sequence only. Multi-sequence GDN capture must keep
// the exact host layout in both its key and metadata.
inline int64_t prefill_graph_token_bucket(uint32_t num_tokens, int32_t stride) {
  if (stride > 256 && num_tokens <= 256) {
    return 256;
  }
  return ((static_cast<int64_t>(num_tokens) + stride - 1) / stride) * stride;
}

// Never evict a live graph here: its logical VA is owned by the shared VMM
// allocator until executor destruction. Reject new keys before preparation.
inline bool can_admit_prefill_graph(std::size_t cached_entries,
                                    int32_t max_entries,
                                    bool cache_hit) {
  return cache_hit || cached_entries < static_cast<std::size_t>(max_entries);
}

inline std::size_t prefill_capture_required_free_bytes(int32_t reserve_mb) {
  const std::size_t configured_bytes =
      static_cast<std::size_t>(std::max<int32_t>(reserve_mb, 0)) * 1024ULL *
      1024ULL;
  return std::max<std::size_t>(kMinPrefillCaptureFreeBytes, configured_bytes);
}

// Existing cached keys always replay (zero new memory). Unseen keys require
// the executor not to have latched an OOM, entry count < max_entries, and
// sufficient free physical memory to survive warmup + VMM/graph instantiation.
inline bool can_admit_prefill_graph_with_memory(std::size_t cached_entries,
                                                int32_t max_entries,
                                                bool cache_hit,
                                                bool oom_latched,
                                                std::size_t physical_free_bytes,
                                                int32_t reserve_mb) {
  if (cache_hit) {
    return true;
  }
  if (oom_latched ||
      !can_admit_prefill_graph(cached_entries, max_entries, cache_hit)) {
    return false;
  }
  return physical_free_bytes >= prefill_capture_required_free_bytes(reserve_mb);
}

inline int64_t prefill_warmup_reserve_bytes(int32_t reserve_mb,
                                            bool enable_graph,
                                            bool enable_prefill_graph) {
  return enable_graph && enable_prefill_graph
             ? static_cast<int64_t>(reserve_mb) * 1024 * 1024
             : 0;
}

}  // namespace xllm::runtime::musa
