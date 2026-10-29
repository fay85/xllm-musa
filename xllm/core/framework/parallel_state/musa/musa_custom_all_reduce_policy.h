// Copyright 2026 The xLLM Authors. Licensed under the Apache License,
// Version 2.0.
#pragma once

#include <cstdint>
#include <string_view>

namespace xllm {

inline bool musa_graph_tp_bf16_two_shot_shape(std::string_view group,
                                              int32_t world_size,
                                              bool is_bf16,
                                              int64_t rows,
                                              int64_t width) {
  // Only the measured large, uniform q7 target-verify family. B1, draft q6,
  // other dtypes/groups, and arbitrary projection widths retain their policy.
  return group == "tp_group" && world_size == 8 && is_bf16 && width == 4096 &&
         rows >= 56 && rows <= 112 && rows % 7 == 0;
}

// The caller serializes all calls sharing this TP handle. The library selects
// the concrete kernel at capture time; restoring host algorithm1 does not alter
// a recorded Graph. Do not leave eager/B1 on algorithm2 after capture or
// failure.
template <typename Setter, typename Operation>
inline void musa_with_tp_two_shot(bool enabled,
                                  Setter set_algorithm,
                                  Operation operation) {
  if (!enabled) {
    operation();
    return;
  }
  set_algorithm(int64_t{2});
  try {
    operation();
  } catch (...) {
    set_algorithm(int64_t{1});
    throw;
  }
  set_algorithm(int64_t{1});
}

}  // namespace xllm
