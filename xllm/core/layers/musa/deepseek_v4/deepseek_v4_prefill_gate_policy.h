/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace xllm::layer {

inline bool dsv4_prefill_gate_enabled() {
  const char* value = std::getenv("XLLM_DSV4_PREFILL_GATE_BF16_FP32");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool dsv4_prefill_gate_shape(int64_t rows,
                                    int64_t hidden,
                                    int64_t experts,
                                    int64_t score_kind,
                                    bool batch_verify,
                                    bool original_bf16) {
  return rows >= 4096 && rows <= 16384 && hidden == 4096 && experts == 256 &&
         score_kind == 2 && !batch_verify && original_bf16;
}

}  // namespace xllm::layer
