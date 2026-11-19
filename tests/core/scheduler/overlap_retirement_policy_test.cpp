/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

#include <gtest/gtest.h>

#include "core/scheduler/overlap_retirement_policy.h"

namespace xllm {
namespace {

TEST(OverlapRetirementPolicyTest, TracksOnlyValidatedDsparkPdState) {
  OverlapRetirementState state;
  state.enabled = true;
  state.pd_decode = true;
  state.deepseek_v4 = true;
  state.dspark = true;
  state.in_flight = true;
  EXPECT_TRUE(should_track_retiring_overlap(state));
  state.retiring_in_flight_request = true;
  EXPECT_TRUE(should_drain_retiring_overlap(state));
}

TEST(OverlapRetirementPolicyTest, RejectsOtherModesAndDp) {
  OverlapRetirementState state;
  state.enabled = true;
  state.pd_decode = true;
  state.deepseek_v4 = true;
  state.dspark = true;
  state.in_flight = true;
  state.dp_size = 2;
  EXPECT_FALSE(should_track_retiring_overlap(state));
  state.dp_size = 1;
  state.pd_decode = false;
  EXPECT_FALSE(should_track_retiring_overlap(state));
}

}  // namespace
}  // namespace xllm
