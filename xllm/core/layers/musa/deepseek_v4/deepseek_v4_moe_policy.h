// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstdint>

namespace xllm::layer {

inline bool dsv4_compact_prefill_fp8_eligible(int64_t rows,
                                              bool is_prefill_like,
                                              bool all_experts_local,
                                              bool fp8_ready) {
  // These are post-SP-gather rows, not the local Attention shard. Preserve
  // small Prefill/hot and draft blocks. The extended 13616/16384-row full
  // pipeline and dynamic replay checks retain the BF16 router contract.
  return rows >= 4096 && rows <= 16384 && is_prefill_like &&
         all_experts_local && fp8_ready;
}

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
