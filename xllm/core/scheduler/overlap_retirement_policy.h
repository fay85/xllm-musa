/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/
#pragma once

#include <cstdint>

namespace xllm {

struct OverlapRetirementState {
  bool enabled = false;
  bool pd_decode = false;
  bool deepseek_v4 = false;
  bool dspark = false;
  int32_t dp_size = 1;
  bool in_flight = false;
  bool retiring_in_flight_request = false;
};

inline bool should_track_retiring_overlap(const OverlapRetirementState& state) {
  return state.enabled && state.pd_decode && state.deepseek_v4 && state.dspark &&
         state.dp_size == 1 && state.in_flight;
}

inline bool should_drain_retiring_overlap(const OverlapRetirementState& state) {
  return should_track_retiring_overlap(state) &&
         state.retiring_in_flight_request;
}

}  // namespace xllm
