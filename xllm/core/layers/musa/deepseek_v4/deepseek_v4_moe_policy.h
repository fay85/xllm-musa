// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstdint>

#include "core/layers/musa/deepseek_v4/deepseek_v4_execution_limits.h"

namespace xllm::layer {

inline bool dsv4_compact_prefill_fp8_eligible(int64_t rows,
                                              bool is_prefill_like,
                                              bool all_experts_local,
                                              bool fp8_ready) {
  // These are post-SP-gather rows, not the local Attention shard. Preserve
  // small Prefill/hot and draft blocks. The extended 13616/16384-row full
  // pipeline and dynamic replay checks retain the BF16 router contract.
  return rows >= kDsv4PrefillReduceMinRows &&
         rows <= kDsv4PrefillReduceMaxRows && is_prefill_like &&
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
  return dsv4_grouped_verify_minimum_supported(minimum_rows) &&
         rows >= minimum_rows && rows <= kDsv4VerifyWorkspaceRows &&
         query_width == kDsv4VerifyQueryWidth && rows % query_width == 0 &&
         is_verify && is_decode && all_experts_local && fp8_ready;
}

}  // namespace xllm::layer
