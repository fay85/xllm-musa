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

#include <cstddef>
#include <cstdint>

namespace xllm::runtime::musa {

// Padding is for a single sequence only. Multi-sequence GDN capture must keep
// the exact host layout in both its key and metadata.
inline int64_t prefill_graph_token_bucket(uint32_t num_tokens, int32_t stride) {
  return ((static_cast<int64_t>(num_tokens) + stride - 1) / stride) * stride;
}

// Never evict a live graph here: its logical VA is owned by the shared VMM
// allocator until executor destruction. Reject new keys before preparation.
inline bool can_admit_prefill_graph(std::size_t cached_entries,
                                    int32_t max_entries,
                                    bool cache_hit) {
  return cache_hit || cached_entries < static_cast<std::size_t>(max_entries);
}

inline int64_t prefill_warmup_reserve_bytes(int32_t reserve_mb,
                                            bool enable_graph,
                                            bool enable_prefill_graph) {
  return enable_graph && enable_prefill_graph
             ? static_cast<int64_t>(reserve_mb) * 1024 * 1024
             : 0;
}

}  // namespace xllm::runtime::musa
