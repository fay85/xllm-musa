/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace xllm::layer {
inline bool dsv4_pd_reduce_scatter_enabled() {
  const char *value = std::getenv("XLLM_DSV4_PREFILL_NATIVE_REDUCE_SCATTER");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool dsv4_pd_reduce_scatter_eligible(int64_t rows, int64_t local_rows,
                                            int32_t world, bool prefill,
                                            bool bf16, bool skip_gather) {
  return prefill && bf16 && !skip_gather && world == 8 && rows >= 4096 &&
         rows <= 16384 && rows % world == 0 && local_rows == rows / world;
}
} // namespace xllm::layer
