// Copyright 2026 The xLLM Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace xllm::layer {

// Only for the C1 static decoder. Zero means the selected bucket set does not
// support this length; callers must select eager before mutating cache state.
// Length already includes every verification row, not just committed tokens.
constexpr int64_t deepseek_v41_decode_capacity(int64_t length,
                                               bool middle,
                                               bool long_context = false) {
  if (length <= 0) {
    return 0;
  }
  if (length <= 512) {
    return 512;
  }
  if (length <= 4096) {
    return middle && length <= 3072 ? 3072 : 4096;
  }
  // 20000 input + 1024 output + up to six verification rows; aligned for
  // both compressed-cache ratios and 128-row physical pages.
  return long_context && length <= 21504 ? 21504 : 0;
}

}  // namespace xllm::layer
