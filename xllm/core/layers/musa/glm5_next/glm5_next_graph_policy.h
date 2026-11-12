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

#include <glog/logging.h>

#include <algorithm>
#include <bit>
#include <cstdint>

namespace xllm::layer {

inline uint32_t glm5_next_graph_context_capacity(int64_t length) {
  CHECK_GT(length, 0);
  CHECK_LE(length, int64_t{1} << 30);
  return std::bit_ceil(std::max(uint32_t{4096}, static_cast<uint32_t>(length)));
}

inline uint32_t glm5_next_graph_key(uint32_t tokens, int64_t length) {
  CHECK_GT(tokens, 0);
  CHECK_LT(tokens, uint32_t{1} << 16);
  const uint32_t capacity = glm5_next_graph_context_capacity(length);
  return (std::bit_width(capacity) << 16) | tokens;
}

}  // namespace xllm::layer
