/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include "core/kernels/musa/deepseek_v4_indexer_metadata.h"

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <optional>
#include <vector>

namespace xllm::kernel::musa {
namespace {

TEST(DeepseekV4IndexerMetadataDeathTest, RejectsCpuScoresBeforeLaunch) {
  EXPECT_DEATH(
      {
        torch::Tensor scores = torch::zeros({1, 1, 64, 4});
        torch::Tensor weights = torch::ones({1, 64});
        deepseek_v4_indexer_weighted_sum(scores, weights);
      },
      "privateuseone");
}

class DeepseekV4IndexerMetadataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
      GTEST_SKIP() << "MUSA device unavailable";
    }
    at::detail::getMUSAHooks().initMUSA();
    no_grad_.emplace();
    stream_.emplace(
        c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    guard_.emplace(stream_->unwrap());
    torch::manual_seed(/*seed=*/20261026);
  }

  void TearDown() override {
    if (stream_.has_value()) {
      stream_->synchronize();
    }
  }

  torch::Tensor reference(const torch::Tensor& scores,
                          const torch::Tensor& weights) const {
    return (torch::relu(scores) *
            weights.view({scores.size(0), scores.size(1), 64, 1}))
        .sum(2)
        .reshape({-1, scores.size(3)});
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::optional<torch::NoGradGuard> no_grad_;
  std::optional<c10::musa::MUSAStream> stream_;
  std::optional<c10::StreamGuard> guard_;
};

TEST_F(DeepseekV4IndexerMetadataTest, MatchesFp32ReferenceWithSignedWeights) {
  const std::array<std::array<int64_t, 4>, 5> shapes{{{1, 26, 64, 10000},
                                                      {1, 65, 64, 10000},
                                                      {2, 3, 64, 128},
                                                      {1, 1, 64, 16384},
                                                      {1, 3, 64, 10004}}};
  float maximum_error = 0.0f;
  for (const auto& shape : shapes) {
    torch::Tensor scores_cpu = torch::randn(shape, torch::kFloat32);
    torch::Tensor weights_cpu = torch::randn({shape[0] * shape[1], 64});
    torch::Tensor scores = scores_cpu.to(device_);
    torch::Tensor weights = weights_cpu.to(device_);
    torch::Tensor expected = reference(scores, weights);
    torch::Tensor actual = deepseek_v4_indexer_weighted_sum(scores, weights);
    stream_->synchronize();
    ASSERT_EQ(actual.sizes(), expected.sizes());
    ASSERT_EQ(actual.scalar_type(), torch::kFloat32);
    torch::Tensor expected_cpu = expected.to(torch::kCPU);
    torch::Tensor actual_cpu = actual.to(torch::kCPU);
    const float error = (actual_cpu - expected_cpu).abs().max().item<float>();
    maximum_error = std::max(maximum_error, error);
    EXPECT_TRUE(torch::allclose(actual_cpu, expected_cpu, 1e-5, 3e-5));
    EXPECT_TRUE(torch::equal(scores.to(torch::kCPU), scores_cpu));
  }
  RecordProperty("maximum_absolute_error", std::to_string(maximum_error));
}

TEST_F(DeepseekV4IndexerMetadataTest, OrdersWritesOnNonDefaultStream) {
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor scores = torch::empty({1, 3, 64, 10000}, options);
  torch::Tensor weights = torch::full({3, 64}, 0.25, options);
  scores.fill_(1.0f);
  torch::Tensor first = deepseek_v4_indexer_weighted_sum(scores, weights);
  scores.fill_(-1.0f);
  torch::Tensor second = deepseek_v4_indexer_weighted_sum(scores, weights);
  stream_->synchronize();
  EXPECT_NE(first.data_ptr(), second.data_ptr());
  EXPECT_TRUE(
      torch::equal(first.to(torch::kCPU), torch::full({3, 10000}, 16.0f)));
  EXPECT_TRUE(torch::equal(second.to(torch::kCPU), torch::zeros({3, 10000})));
}

TEST_F(DeepseekV4IndexerMetadataTest, PreservesReluSpecialValues) {
  torch::Tensor values = torch::zeros({1, 2, 64, 4});
  values.index_put_({0, 0, 0, 0}, std::numeric_limits<float>::quiet_NaN());
  values.index_put_({0, 0, 0, 1}, std::numeric_limits<float>::infinity());
  values.index_put_({0, 0, 0, 2}, -std::numeric_limits<float>::infinity());
  values.index_put_({0, 0, 0, 3}, -0.0f);
  values.index({0, 1}).copy_(values.index({0, 0}));
  torch::Tensor weights_cpu = torch::ones({2, 64});
  weights_cpu.index_put_({1, 0}, -2.0f);
  torch::Tensor scores = values.to(device_);
  torch::Tensor weights = weights_cpu.to(device_);
  // Use an independent CPU oracle for non-finite values: backend reductions
  // must not define which columns receive NaN or infinity.
  torch::Tensor expected = reference(values, weights_cpu);
  torch::Tensor actual_device =
      deepseek_v4_indexer_weighted_sum(scores, weights);
  stream_->synchronize();
  torch::Tensor actual = actual_device.to(torch::kCPU);
  EXPECT_TRUE(torch::equal(torch::isnan(actual), torch::isnan(expected)))
      << "actual=" << actual << " expected=" << expected;
  EXPECT_TRUE(torch::equal(torch::isinf(actual), torch::isinf(expected)))
      << "actual=" << actual << " expected=" << expected;
  const torch::Tensor finite = torch::isfinite(expected);
  EXPECT_TRUE(torch::equal(actual.masked_select(finite),
                           expected.masked_select(finite)));
  EXPECT_EQ(actual.index({0, 1}).item<float>(),
            std::numeric_limits<float>::infinity());
  EXPECT_EQ(actual.index({1, 1}).item<float>(),
            -std::numeric_limits<float>::infinity());
}

TEST_F(DeepseekV4IndexerMetadataTest, PreservesCausalTopkWithSeparatedScores) {
  constexpr int64_t kQueries = 26;
  constexpr int64_t kKeys = 10000;
  torch::Tensor scores_cpu = torch::arange(kKeys, torch::kFloat32)
                                 .mul(0.001f)
                                 .view({1, 1, 1, kKeys})
                                 .expand({1, kQueries, 64, kKeys})
                                 .contiguous();
  torch::Tensor scores = scores_cpu.to(device_);
  torch::Tensor weights = torch::full({kQueries, 64}, 0.125f).to(device_);
  torch::Tensor expected_device = reference(scores, weights);
  torch::Tensor actual_device =
      deepseek_v4_indexer_weighted_sum(scores, weights);
  stream_->synchronize();
  torch::Tensor expected = expected_device.to(torch::kCPU);
  torch::Tensor actual = actual_device.to(torch::kCPU);
  torch::Tensor valid =
      (torch::arange(kQueries, torch::kInt64) + 32001).div(4, "floor");
  torch::Tensor mask =
      torch::arange(kKeys, torch::kInt64).unsqueeze(0).ge(valid.unsqueeze(1));
  expected.masked_fill_(mask, -std::numeric_limits<float>::infinity());
  actual.masked_fill_(mask, -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(torch::equal(std::get<1>(expected.topk(512, 1)),
                           std::get<1>(actual.topk(512, 1))));
}

TEST_F(DeepseekV4IndexerMetadataTest, ReportsLongTileReductionTiming) {
  torch::Tensor scores = torch::randn({1, 26, 64, 10000}).to(device_);
  torch::Tensor weights = torch::randn({26, 64}).to(device_);
  std::vector<double> old_times;
  std::vector<double> fused_times;
  old_times.reserve(5);
  fused_times.reserve(5);
  for (int32_t iteration = 0; iteration < 7; ++iteration) {
    torch::Tensor scratch = scores.clone();
    stream_->synchronize();
    auto begin = std::chrono::steady_clock::now();
    deepseek_v4_indexer_relu_weight(scratch, weights);
    torch::Tensor old_output = scratch.sum(2);
    stream_->synchronize();
    const double old_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - begin)
                              .count();
    begin = std::chrono::steady_clock::now();
    torch::Tensor output = deepseek_v4_indexer_weighted_sum(scores, weights);
    stream_->synchronize();
    const double fused_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - begin)
                                .count();
    ASSERT_TRUE(old_output.defined() && output.defined());
    if (iteration >= 2) {
      old_times.emplace_back(old_ms);
      fused_times.emplace_back(fused_ms);
    }
  }
  std::sort(old_times.begin(), old_times.end());
  std::sort(fused_times.begin(), fused_times.end());
  RecordProperty("existing_median_ms", std::to_string(old_times[2]));
  RecordProperty("fused_median_ms", std::to_string(fused_times[2]));
}

}  // namespace
}  // namespace xllm::kernel::musa
