/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <gtest/gtest.h>

#include "core/layers/musa/deepseek_v4/deepseek_v4_execution_limits.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_moe_policy.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_prefill_gate_policy.h"

namespace xllm::layer {
namespace {

TEST(DeepseekV4JointBuildPolicyTest, RetainsQualifiedCachedPrefillTail) {
  for (const int64_t rows : {2048, 4095, 4096, 16384}) {
    EXPECT_TRUE(dsv4_compact_prefill_fp8_eligible(rows, true, true, true));
    EXPECT_TRUE(dsv4_prefill_gate_shape(rows, 4096, 256, 2, false, true));
  }
  for (const int64_t rows : {0, 7, 2047, 16385}) {
    EXPECT_FALSE(dsv4_compact_prefill_fp8_eligible(rows, true, true, true));
    EXPECT_FALSE(dsv4_prefill_gate_shape(rows, 4096, 256, 2, false, true));
  }
  EXPECT_FALSE(dsv4_compact_prefill_fp8_eligible(2048, false, true, true));
  EXPECT_FALSE(dsv4_compact_prefill_fp8_eligible(2048, true, false, true));
  EXPECT_FALSE(dsv4_compact_prefill_fp8_eligible(2048, true, true, false));
  EXPECT_EQ(kDsv4PrefillReduceMinRows, 4096);
}

TEST(DeepseekV4JointBuildPolicyTest,
     RetainsGroupedVerifyThresholdWithoutWideningWidth) {
  for (const int64_t minimum : {63, 77, 112}) {
    EXPECT_TRUE(dsv4_grouped_fp8_verify_eligible(
        minimum, minimum, 7, true, true, true, true));
    EXPECT_FALSE(dsv4_grouped_fp8_verify_eligible(
        minimum, minimum - 7, 7, true, true, true, true));
    EXPECT_FALSE(dsv4_grouped_fp8_verify_eligible(
        minimum, minimum, 6, true, true, true, true));
  }
  EXPECT_FALSE(dsv4_grouped_verify_minimum_supported(62));
  EXPECT_FALSE(dsv4_grouped_verify_minimum_supported(64));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(63, 133, 7, true, true, true, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(63, 64, 7, true, true, true, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(63, 63, 7, false, true, true, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(63, 63, 7, true, false, true, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(63, 63, 7, true, true, false, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(63, 63, 7, true, true, true, false));
}

}  // namespace
}  // namespace xllm::layer
