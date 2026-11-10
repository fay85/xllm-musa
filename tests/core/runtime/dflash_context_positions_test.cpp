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

#include "runtime/dflash/dflash_context_positions.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {
namespace {

std::vector<int32_t> make_range(int32_t begin, int32_t end) {
  std::vector<int32_t> values;
  values.reserve(static_cast<size_t>(end - begin));
  for (int32_t value = begin; value < end; ++value) {
    values.emplace_back(value);
  }
  return values;
}

class DFlashContextPositionsDeathTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST(DFlashContextPositionsTest, TextPositionsPassThroughWithoutRebuild) {
  const torch::Tensor positions = torch::tensor({4, 5, 6}, torch::kInt);

  EXPECT_FALSE(dflash_detail::is_mrope_context_positions(positions, 3));
}

TEST(DFlashContextPositionsTest, MropePositionsRequireThreeAlignedAxes) {
  const torch::Tensor positions = torch::zeros({3, 4}, torch::kInt);

  EXPECT_TRUE(dflash_detail::is_mrope_context_positions(positions, 4));
}

TEST_F(DFlashContextPositionsDeathTest, RejectsNonMropeAxisCount) {
  const torch::Tensor positions = torch::zeros({2, 4}, torch::kInt);

  EXPECT_DEATH(dflash_detail::is_mrope_context_positions(positions, 4),
               "three axes");
}

TEST_F(DFlashContextPositionsDeathTest, RejectsUnsupportedPositionRank) {
  const torch::Tensor positions = torch::zeros({1, 3, 4}, torch::kInt);

  EXPECT_DEATH(dflash_detail::is_mrope_context_positions(positions, 4),
               "DFlash context positions");
}

TEST_F(DFlashContextPositionsDeathTest, RejectsMropeTokenCountMismatch) {
  const torch::Tensor positions = torch::zeros({3, 5}, torch::kInt);

  EXPECT_DEATH(dflash_detail::is_mrope_context_positions(positions, 4),
               "align with context tokens");
}

TEST_F(DFlashContextPositionsDeathTest, RejectsTextTokenCountMismatch) {
  const torch::Tensor positions = torch::zeros({5}, torch::kInt);

  EXPECT_DEATH(dflash_detail::is_mrope_context_positions(positions, 4),
               "hidden/position count mismatch");
}

TEST(DFlashContextPositionsTest, RebuildsFullPrefillPositions) {
  const std::vector<int32_t> positions =
      dflash_detail::build_token_aligned_context_positions({100}, {100}, 100);

  EXPECT_EQ(positions, make_range(0, 100));
}

TEST(DFlashContextPositionsTest, RebuildsChunkedPrefillPositions) {
  const std::vector<int32_t> positions =
      dflash_detail::build_token_aligned_context_positions({100}, {50}, 50);

  EXPECT_EQ(positions, make_range(50, 100));
}

TEST(DFlashContextPositionsTest, RebuildsMultipleSequencesWithoutCrossing) {
  const std::vector<int32_t> positions =
      dflash_detail::build_token_aligned_context_positions(
          {5, 12, 20}, {5, 4, 2}, 11);
  const std::vector<int32_t> expected = {0, 1, 2, 3, 4, 8, 9, 10, 11, 18, 19};

  EXPECT_EQ(positions, expected);
}

TEST(DFlashContextPositionsTest, RebuildsEmptyContext) {
  const std::vector<int32_t> positions =
      dflash_detail::build_token_aligned_context_positions({}, {}, 0);

  EXPECT_TRUE(positions.empty());
}

TEST_F(DFlashContextPositionsDeathTest, RejectsMismatchedSequenceMetadata) {
  EXPECT_DEATH(
      dflash_detail::build_token_aligned_context_positions({10}, {10, 20}, 10),
      "sequence count mismatch");
}

TEST_F(DFlashContextPositionsDeathTest, RejectsQueryLongerThanKv) {
  EXPECT_DEATH(
      dflash_detail::build_token_aligned_context_positions({10}, {11}, 11),
      "kv_len must cover q_len");
}

TEST_F(DFlashContextPositionsDeathTest, RejectsTotalTokenCountMismatch) {
  EXPECT_DEATH(dflash_detail::build_token_aligned_context_positions(
                   {10, 20}, {4, 5}, 10),
               "must match the context token count");
}

}  // namespace
}  // namespace xllm
