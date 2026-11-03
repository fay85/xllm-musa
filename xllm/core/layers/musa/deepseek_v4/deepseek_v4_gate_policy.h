// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <cstdint>

namespace xllm::layer {

inline int64_t dsv4_gate_workspace_capacity(int64_t original, bool batch_mate) {
  return batch_mate ? std::max<int64_t>(original, 128) : original;
}

// Preserve the original <=16-row GEMV path and all non-target phases. The
// direct Graph family carries one q7 segment per physical sequence row.
inline bool dsv4_batch_gate_verify_allowed(int64_t tokens,
                                          int64_t sequences,
                                          int64_t query_width,
                                          bool graph,
                                          bool decode,
                                          bool target_verify) {
  return graph && decode && target_verify && query_width == 7 &&
         sequences > 2 && sequences <= 16 && tokens == sequences * query_width;
}

}  // namespace xllm::layer
