// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstdint>

namespace xllm::layer {

// Opt-in N6/q7 target-only EP1 experiment. The caller leaves minimum_rows=0
// for draft models. Never let B1 or unsupported shapes switch arithmetic.
inline bool dsv4_grouped_fp8_verify_eligible(int64_t minimum_rows,
                                             int64_t rows,
                                             int64_t query_width,
                                             bool is_verify,
                                             bool is_decode,
                                             bool all_experts_local,
                                             bool fp8_ready) {
  return (minimum_rows == 77 || minimum_rows == 112) && rows >= minimum_rows &&
         rows <= 128 && query_width == 7 && rows % query_width == 0 &&
         is_verify && is_decode && all_experts_local && fp8_ready;
}

}  // namespace xllm::layer
