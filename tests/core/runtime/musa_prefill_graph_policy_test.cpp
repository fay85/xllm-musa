/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>

#include "core/runtime/musa/prefill_graph_policy.h"

namespace xllm::runtime::musa {
namespace {

TEST(MusaPrefillGraphPolicyTest, ShortTailsShareOneBucket) {
  for (uint32_t rows :
       {1U, 16U, 54U, 64U, 140U, 144U, 232U, 240U, 255U, 256U}) {
    EXPECT_EQ(prefill_graph_token_bucket(rows, 256), 256);
  }
  EXPECT_EQ(prefill_graph_token_bucket(257, 256), 512);
}

TEST(MusaPrefillGraphPolicyTest, PaddingCoversRowsAndHasBoundedCardinality) {
  std::set<int64_t> buckets;
  for (uint32_t rows = 1; rows <= 8192; ++rows) {
    const int64_t bucket = prefill_graph_token_bucket(rows, 256);
    EXPECT_GE(bucket, rows);
    EXPECT_LT(bucket - rows, 256);
    EXPECT_LE(bucket, 8192);
    buckets.insert(bucket);
  }
  EXPECT_EQ(buckets.size(), 32);
  EXPECT_EQ(prefill_graph_token_bucket(8193, 256), 8448);
}

TEST(MusaPrefillGraphPolicyTest, RoundingDoesNotOverflowUint32) {
  EXPECT_EQ(
      prefill_graph_token_bucket(std::numeric_limits<uint32_t>::max(), 256),
      int64_t{1} << 32);
}

TEST(MusaPrefillGraphPolicyTest, CacheLimitKeepsHitsAndRejectsNewKeys) {
  EXPECT_TRUE(can_admit_prefill_graph(31, 32, false));
  EXPECT_FALSE(can_admit_prefill_graph(32, 32, false));
  EXPECT_TRUE(can_admit_prefill_graph(32, 32, true));
  EXPECT_TRUE(can_admit_prefill_graph(33, 32, true));
  EXPECT_FALSE(can_admit_prefill_graph(33, 32, false));
}

TEST(MusaPrefillGraphPolicyTest, WarmupReserveOnlyAppliesToPrefillGraphs) {
  constexpr int64_t kReserve = int64_t{2048} * 1024 * 1024;
  EXPECT_EQ(prefill_warmup_reserve_bytes(2048, true, true), kReserve);
  EXPECT_EQ(prefill_warmup_reserve_bytes(2048, true, false), 0);
  EXPECT_EQ(prefill_warmup_reserve_bytes(2048, false, true), 0);
  EXPECT_EQ(prefill_warmup_reserve_bytes(0, true, true), 0);
  // Shared-device target/draft capacity takes min(), so one reserve is charged.
  constexpr int64_t kAvailable = int64_t{48} * 1024 * 1024 * 1024;
  const int64_t target =
      kAvailable - prefill_warmup_reserve_bytes(2048, true, true);
  const int64_t draft =
      kAvailable - prefill_warmup_reserve_bytes(2048, true, true);
  EXPECT_EQ(std::min(target, draft), kAvailable - kReserve);
}

}  // namespace
}  // namespace xllm::runtime::musa
