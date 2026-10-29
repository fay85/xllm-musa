/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "core/layers/musa/deepseek_v4/deepseek_v4_execution_limits.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_runtime_options.h"

namespace xllm::layer {
inline bool dsv4_pd_reduce_scatter_enabled() {
  return dsv4_runtime_option<DeepseekV4Env::PREFILL_NATIVE_REDUCE_SCATTER>()
      .is_one();
}

inline bool dsv4_pd_reduce_scatter_eligible(int64_t rows,
                                            int64_t local_rows,
                                            int32_t world,
                                            bool prefill,
                                            bool bf16,
                                            bool skip_gather) {
  return prefill && bf16 && !skip_gather &&
         world == kDsv4PrefillReduceWorldSize &&
         rows >= kDsv4PrefillReduceMinRows &&
         rows <= kDsv4PrefillReduceMaxRows && rows % world == 0 &&
         local_rows == rows / world;
}
}  // namespace xllm::layer
