/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/kv_cache_transfer/staged_read_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace xllm {
namespace {

TEST(StagedReadPlanTest, JoinsTp2RowsWithoutCopyingPadding) {
  StagedReadRun run{/*remote_buffer_id=*/3,
                    /*remote_offset=*/0,
                    /*length=*/16,
                    /*payload_bytes=*/16,
                    /*begin=*/0,
                    /*end=*/1};
  for (size_t row = 1; row < 8; ++row) {
    ASSERT_TRUE(try_extend_staged_read_run(run, 3, row * 32, 16, row + 1, 256));
  }
  EXPECT_EQ(run.length, 240);
  EXPECT_EQ(run.payload_bytes, 128);
  EXPECT_EQ(run.end, 8);
  std::vector<uint8_t> remote(256);
  for (size_t i = 0; i < remote.size(); ++i) {
    remote[i] = static_cast<uint8_t>(i);
  }
  const std::vector<uint8_t> staged(remote.begin(),
                                    remote.begin() + run.length);
  std::vector<uint8_t> destination(128, 0);
  for (size_t row = run.begin; row < run.end; ++row) {
    std::copy_n(staged.begin() + row * 32, 16, destination.begin() + row * 16);
  }
  for (size_t row = 0; row < 8; ++row) {
    for (size_t byte = 0; byte < 16; ++byte) {
      EXPECT_EQ(destination[row * 16 + byte], remote[row * 32 + byte]);
    }
  }
}

TEST(StagedReadPlanTest, BoundsStagingAndPreservesExistingRunOnRejection) {
  StagedReadRun run{1, 100, 16, 16, 0, 1};
  ASSERT_TRUE(try_extend_staged_read_run(run, 1, 132, 16, 2, 48));
  EXPECT_FALSE(try_extend_staged_read_run(run, 1, 164, 16, 3, 48));
  EXPECT_EQ(run.remote_offset, 100);
  EXPECT_EQ(run.length, 48);
  EXPECT_EQ(run.payload_bytes, 32);
  EXPECT_EQ(run.end, 2);
}

TEST(StagedReadPlanTest, RejectsOtherBuffersLargeGapsAndOverlap) {
  StagedReadRun run{1, 128, 16, 16, 0, 1};
  EXPECT_FALSE(try_extend_staged_read_run(run, 2, 144, 16, 2, 1024));
  EXPECT_FALSE(try_extend_staged_read_run(run, 1, 176, 16, 2, 1024));
  EXPECT_FALSE(try_extend_staged_read_run(run, 1, 136, 16, 2, 1024));
  EXPECT_EQ(run.length, 16);
  EXPECT_EQ(run.end, 1);
}

TEST(StagedReadPlanTest, RejectsAddressOverflow) {
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  StagedReadRun run{0, kMax - 16, 8, 8, 0, 1};
  EXPECT_FALSE(try_extend_staged_read_run(run, 0, kMax - 4, 8, 2, 1024));
  StagedReadRun invalid{0, kMax - 4, 8, 8, 0, 1};
  EXPECT_FALSE(try_extend_staged_read_run(invalid, 0, 128, 8, 2, 1024));
}

TEST(StagedReadPlanTest, RandomizedScatterMatchesDirectCopies) {
  std::mt19937 rng(20261028);
  for (size_t trial = 0; trial < 1000; ++trial) {
    std::vector<uint8_t> remote(8192);
    for (size_t i = 0; i < remote.size(); ++i) {
      remote[i] = static_cast<uint8_t>(rng());
    }
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> lengths;
    uint64_t next = 0;
    for (size_t i = 0; i < 32; ++i) {
      offsets.emplace_back(next);
      lengths.emplace_back(1 + rng() % 32);
      next += lengths.back() + rng() % 64;
    }
    std::vector<StagedReadRun> runs;
    runs.reserve(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
      const bool extended =
          !runs.empty() &&
          try_extend_staged_read_run(
              runs.back(), 0, offsets[i], lengths[i], i + 1, 256);
      if (!extended) {
        runs.emplace_back(
            StagedReadRun{0, offsets[i], lengths[i], lengths[i], i, i + 1});
      }
    }
    std::vector<uint8_t> actual;
    std::vector<uint8_t> expected;
    actual.reserve(next);
    expected.reserve(next);
    for (const StagedReadRun& run : runs) {
      ASSERT_LE(run.length, 256);
      ASSERT_LE(run.length, run.payload_bytes * 2);
      const std::vector<uint8_t> staging(
          remote.begin() + run.remote_offset,
          remote.begin() + run.remote_offset + run.length);
      for (size_t i = run.begin; i < run.end; ++i) {
        const uint64_t offset = offsets[i] - run.remote_offset;
        actual.insert(actual.end(),
                      staging.begin() + offset,
                      staging.begin() + offset + lengths[i]);
        expected.insert(expected.end(),
                        remote.begin() + offsets[i],
                        remote.begin() + offsets[i] + lengths[i]);
      }
    }
    EXPECT_EQ(actual, expected);
  }
}

}  // namespace
}  // namespace xllm
