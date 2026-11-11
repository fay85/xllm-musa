/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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
#include <string_view>

namespace xllm {

// Portable FNV-1a, shared by the constraint-state fingerprints so the value
// stays stable across compilers / standard-library versions (and across
// processes once persisted for PD spec transfer). std::hash<std::string> is
// not portable and must not be used for this.
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

inline uint64_t fnv1a_byte(uint64_t hash, uint8_t value) {
  return (hash ^ value) * kFnvPrime;
}

inline uint64_t fnv1a(std::string_view data) {
  uint64_t hash = kFnvOffsetBasis;
  for (unsigned char c : data) {
    hash = fnv1a_byte(hash, c);
  }
  return hash;
}

}  // namespace xllm
