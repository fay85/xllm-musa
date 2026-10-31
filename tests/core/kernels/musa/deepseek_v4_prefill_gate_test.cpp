/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_prefill_gate_policy.h"

namespace xllm::kernel::musa {
namespace {

TEST(DeepseekV4PrefillGatePolicyTest, IncludesCachedTailAndExcludesDecode) {
  for (const int64_t rows : {2048, 4095, 4096, 7232, 8000, 8192, 16384}) {
    EXPECT_TRUE(
        layer::dsv4_prefill_gate_shape(rows, 4096, 256, 2, false, true));
  }
  for (const int64_t rows : {1, 7, 112, 1000, 2047, 16385}) {
    EXPECT_FALSE(
        layer::dsv4_prefill_gate_shape(rows, 4096, 256, 2, false, true));
  }
  EXPECT_FALSE(layer::dsv4_prefill_gate_shape(8000, 2048, 256, 2, false, true));
  EXPECT_FALSE(layer::dsv4_prefill_gate_shape(8000, 4096, 128, 2, false, true));
  EXPECT_FALSE(layer::dsv4_prefill_gate_shape(8000, 4096, 256, 0, false, true));
  EXPECT_FALSE(layer::dsv4_prefill_gate_shape(8000, 4096, 256, 2, true, true));
  EXPECT_FALSE(
      layer::dsv4_prefill_gate_shape(8000, 4096, 256, 2, false, false));
}

class DeepseekV4PrefillGateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (musaGetDeviceCount(&count) != musaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device unavailable";
    }
    at::detail::getMUSAHooks().initMUSA();
    no_grad_.emplace();
    stream_.emplace(c10::musa::getStreamFromPool(false, 0));
    guard_.emplace(stream_->unwrap());
    torch::manual_seed(20261026);
  }

  void TearDown() override {
    if (stream_.has_value()) {
      stream_->synchronize();
    }
  }

  torch::Tensor project(const torch::Tensor& input,
                        const torch::Tensor& weight) {
    TvmffiStreamGuard ffi_guard(input.device());
    return deepseek_v4_gate_bf16_fp32(input, weight);
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::optional<torch::NoGradGuard> no_grad_;
  std::optional<c10::musa::MUSAStream> stream_;
  std::optional<c10::StreamGuard> guard_;
};

TEST_F(DeepseekV4PrefillGateTest,
       MatchesFp32ProjectionAndRoutingWithinRoundingBound) {
  for (const int64_t rows : {2048, 4096, 7232, 8000, 8192}) {
    torch::Tensor input =
        (torch::randn({rows, 4096}) * 0.3).to(torch::kBFloat16).to(device_);
    torch::Tensor weight =
        (torch::randn({256, 4096}) * 0.02).to(torch::kBFloat16).to(device_);
    torch::Tensor expected = torch::matmul(
        input.to(torch::kFloat32), weight.to(torch::kFloat32).transpose(0, 1));
    torch::Tensor actual = project(input, weight);
    ASSERT_TRUE(actual.defined());
    stream_->synchronize();
    ASSERT_EQ(actual.sizes(), expected.sizes());
    ASSERT_EQ(actual.scalar_type(), torch::kFloat32);
    torch::Tensor reference_cpu = expected.to(torch::kCPU);
    torch::Tensor actual_cpu = actual.to(torch::kCPU);
    const float max_error =
        (reference_cpu - actual_cpu).abs().max().item<float>();
    EXPECT_LT(max_error, 3e-5f);
    torch::Tensor bias = torch::randn({256}) * 0.01;
    torch::Tensor reference_scores =
        torch::softplus(reference_cpu).sqrt() + bias;
    torch::Tensor actual_scores = torch::softplus(actual_cpu).sqrt() + bias;
    torch::Tensor reference_ids = std::get<1>(reference_scores.topk(8, -1));
    torch::Tensor actual_ids = std::get<1>(actual_scores.topk(8, -1));
    const int64_t changed_rows = (std::get<0>(reference_ids.sort(-1)) !=
                                  std::get<0>(actual_ids.sort(-1)))
                                     .any(-1)
                                     .sum()
                                     .item<int64_t>();
    const float score_error =
        (reference_scores - actual_scores).abs().max().item<float>();
    torch::Tensor reference_cutoff =
        std::get<0>(reference_scores.topk(8, -1)).select(1, 7);
    torch::Tensor selected_min =
        std::get<0>(reference_scores.gather(1, actual_ids).min(1));
    const float max_regret =
        (reference_cutoff - selected_min).max().item<float>();
    EXPECT_LE(max_regret, 2 * score_error + 1e-7f);
    RecordProperty("rows" + std::to_string(rows) + "_max_logit_error",
                   std::to_string(max_error));
    RecordProperty("rows" + std::to_string(rows) + "_changed_topk_rows",
                   std::to_string(changed_rows));
    RecordProperty("rows" + std::to_string(rows) + "_max_selection_regret",
                   std::to_string(max_regret));
  }
}

TEST_F(DeepseekV4PrefillGateTest, OrdersFreshInputsAndPreservesSeparatedTopk) {
  torch::Tensor input = torch::ones({8000, 4096}, torch::kBFloat16).to(device_);
  torch::Tensor weight = torch::arange(256, torch::kFloat32)
                             .mul(1.0f / 256)
                             .view({256, 1})
                             .expand({256, 4096})
                             .contiguous()
                             .to(torch::kBFloat16)
                             .to(device_);
  torch::Tensor first = project(input, weight);
  input.fill_(-1);
  torch::Tensor second = project(input, weight);
  stream_->synchronize();
  EXPECT_NE(first.data_ptr(), second.data_ptr());
  torch::Tensor first_cpu = first.to(torch::kCPU);
  torch::Tensor second_cpu = second.to(torch::kCPU);
  EXPECT_TRUE(torch::equal(first_cpu, -second_cpu));
  torch::Tensor selected = std::get<1>(first_cpu.topk(8, -1));
  torch::Tensor expected_ids =
      torch::arange(255, 247, -1, torch::kLong).unsqueeze(0).expand({8000, 8});
  EXPECT_TRUE(torch::equal(selected, expected_ids));
}

TEST_F(DeepseekV4PrefillGateTest, ReportsCachedTailProjectionTiming) {
  torch::Tensor input =
      torch::randn({8000, 4096}).to(torch::kBFloat16).to(device_);
  torch::Tensor weight =
      torch::randn({256, 4096}).to(torch::kBFloat16).to(device_);
  torch::Tensor weight_fp32 = weight.to(torch::kFloat32);
  std::vector<double> original_times;
  std::vector<double> candidate_times;
  original_times.reserve(7);
  candidate_times.reserve(7);
  for (int32_t iteration = 0; iteration < 10; ++iteration) {
    stream_->synchronize();
    auto start = std::chrono::steady_clock::now();
    torch::Tensor original =
        torch::matmul(input.to(torch::kFloat32), weight_fp32.transpose(0, 1));
    stream_->synchronize();
    const double original_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
    start = std::chrono::steady_clock::now();
    torch::Tensor candidate = project(input, weight);
    stream_->synchronize();
    const double candidate_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
    ASSERT_TRUE(original.defined() && candidate.defined());
    if (iteration >= 3) {
      original_times.emplace_back(original_ms);
      candidate_times.emplace_back(candidate_ms);
    }
  }
  std::sort(original_times.begin(), original_times.end());
  std::sort(candidate_times.begin(), candidate_times.end());
  RecordProperty("original_median_ms", std::to_string(original_times[3]));
  RecordProperty("candidate_median_ms", std::to_string(candidate_times[3]));
}

}  // namespace
}  // namespace xllm::kernel::musa
