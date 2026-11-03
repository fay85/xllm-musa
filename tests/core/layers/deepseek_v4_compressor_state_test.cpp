/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include "core/layers/musa/deepseek_v4/deepseek_v4_compressor_state.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace xllm::layer {
namespace {

TEST(DeepseekV4CompressorStateTest, TailMatchesTokenReference) {
  for (const int64_t ratio : {4, 128}) {
    const int64_t rows = ratio == 4 ? 8 : 128;
    constexpr int64_t kWidth = 6;
    auto kv = torch::arange(400 * kWidth, torch::kFloat32).view({400, kWidth});
    auto score = kv * 0.125;
    auto ape =
        torch::arange(ratio * kWidth, torch::kFloat32).view({ratio, kWidth}) *
        0.5;
    for (const int64_t end : {0, 1, 3, 4, 5, 7, 8, 127, 128, 129, 256, 259}) {
      auto actual = dsv4_prefill_state_tail(
          kv, score, ape, end, ratio, rows, kWidth, kv.options());
      auto expected_kv = torch::zeros({rows, kWidth});
      auto expected_score =
          torch::full({rows, kWidth}, -std::numeric_limits<float>::infinity());
      const int64_t cutoff = end - end % ratio;
      if (ratio == 4 && cutoff >= ratio) {
        for (int64_t row = 0; row < ratio; ++row) {
          expected_kv[row].copy_(kv[cutoff - ratio + row]);
          expected_score[row].copy_(score[cutoff - ratio + row] + ape[row]);
        }
      }
      for (int64_t row = 0; row < end - cutoff; ++row) {
        const int64_t destination = (ratio == 4 ? ratio : 0) + row;
        expected_kv[destination].copy_(kv[cutoff + row]);
        expected_score[destination].copy_(score[cutoff + row] + ape[row]);
      }
      EXPECT_TRUE(torch::equal(actual.first, expected_kv))
          << "ratio=" << ratio << " end=" << end;
      EXPECT_TRUE(torch::equal(actual.second, expected_score))
          << "ratio=" << ratio << " end=" << end;
    }
  }
}

TEST(DeepseekV4CompressorStateTest, ReusesMatchingStorageAndValues) {
  auto kv = torch::ones({8, 6});
  auto score = torch::full({8, 6}, 2.0);
  const void* kv_address = kv.data_ptr();
  const void* score_address = score.data_ptr();
  dsv4_ensure_compressor_state(kv, score, 8, 6, kv.options());
  EXPECT_EQ(kv.data_ptr(), kv_address);
  EXPECT_EQ(score.data_ptr(), score_address);
  EXPECT_TRUE(torch::equal(kv, torch::ones({8, 6})));
  EXPECT_TRUE(torch::equal(score, torch::full({8, 6}, 2.0)));
}

TEST(DeepseekV4CompressorStateTest, InitializesMissingState) {
  torch::Tensor kv;
  torch::Tensor score;
  dsv4_ensure_compressor_state(kv, score, 8, 6, torch::TensorOptions());
  EXPECT_TRUE(torch::equal(kv, torch::zeros({8, 6})));
  EXPECT_TRUE(torch::equal(
      score, torch::full({8, 6}, -std::numeric_limits<float>::infinity())));
}

TEST(DeepseekV4CompressorStateTest, ReplacesOnlyMismatchingState) {
  auto kv = torch::ones({8, 6});
  auto score = torch::ones({4, 6});
  const void* kv_address = kv.data_ptr();
  dsv4_ensure_compressor_state(kv, score, 8, 6, kv.options());
  EXPECT_EQ(kv.data_ptr(), kv_address);
  EXPECT_EQ(score.size(0), 8);
  EXPECT_TRUE(torch::isneginf(score).all().item<bool>());
}

TEST(DeepseekV4CompressorStateTest, SnapshotDoesNotAliasSource) {
  auto kv = torch::ones({8, 6});
  auto score = torch::ones_like(kv);
  auto ape = torch::zeros({4, 6});
  auto snapshot =
      dsv4_prefill_state_tail(kv, score, ape, 4, 4, 8, 6, kv.options());
  kv.zero_();
  score.zero_();
  EXPECT_TRUE(
      torch::equal(snapshot.first.narrow(0, 0, 4), torch::ones({4, 6})));
  EXPECT_TRUE(
      torch::equal(snapshot.second.narrow(0, 0, 4), torch::ones({4, 6})));
}

}  // namespace
}  // namespace xllm::layer
