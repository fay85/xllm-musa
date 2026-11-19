// Copyright 2026 The xLLM Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace xllm::layer {

// Only for the C1 static decoder. The caller first checks 0 < length <= 4096.
// Length already includes every verification row, not just committed tokens.
constexpr int64_t deepseek_v41_decode_capacity(int64_t length, bool middle) {
  if (length <= 512) {
    return 512;
  }
  return middle && length <= 3072 ? 3072 : 4096;
}

}  // namespace xllm::layer
