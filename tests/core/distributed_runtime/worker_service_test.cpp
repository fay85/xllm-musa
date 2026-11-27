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

#include "core/distributed_runtime/worker_service.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/speculative_token_stats.h"
#include "core/runtime/options.h"

namespace xllm {

class WorkerServiceTestPeer final {
 public:
  static std::vector<SpeculativeTokenStats> record_speculative_metrics(
      WorkerService& service,
      const ForwardOutput& output,
      const torch::Tensor& tokens) {
    return service.record_speculative_metrics_from_output(output, tokens);
  }
};

namespace {

TEST(SpeculativeTokenStatsTest, CountsTokensPerSequence) {
  // Row 0 accepts two draft tokens; row 1 accepts all five.
  const torch::Tensor tokens = torch::tensor(
      {{10, 11, 12, -1, -1, -1}, {20, 21, 22, 23, 24, 25}}, torch::kInt32);
  const auto stats = calculate_speculative_output_stats(tokens);

  EXPECT_EQ(stats.committed_tokens, 9);
  EXPECT_EQ(stats.accepted_per_position, (std::vector<int64_t>{2, 2, 1, 1, 1}));
  ASSERT_EQ(stats.sequence_stats.size(), 2);
  EXPECT_EQ(stats.sequence_stats[0].accepted_tokens, 2);
  EXPECT_EQ(stats.sequence_stats[0].proposed_tokens, 5);
  EXPECT_EQ(stats.sequence_stats[1].accepted_tokens, 5);
  EXPECT_EQ(stats.sequence_stats[1].proposed_tokens, 5);
}

TEST(SpeculativeTokenStatsTest, BlockGreedyExcludesTargetTokens) {
  const auto masked_tokens = torch::tensor(
      {{9, -1, -1, -1}, {1, 9, -1, -1}, {1, 2, 9, -1}, {1, 2, 3, 8}},
      /*options=*/torch::kInt64);
  const auto stats =
      calculate_speculative_output_stats(masked_tokens).sequence_stats;
  ASSERT_EQ(stats.size(), 4);
  for (size_t row = 0; row < stats.size(); ++row) {
    EXPECT_EQ(stats[row].accepted_tokens, static_cast<int64_t>(row));
    EXPECT_EQ(stats[row].proposed_tokens, 3);
  }
}

TEST(SpeculativeTokenStatsTest, BlockRandomExcludesTargetTokens) {
  const auto masked_tokens = torch::tensor(
      {{1, -1, -1, -1}, {0, 1, -1, -1}, {0, 0, 1, -1}, {0, 0, 0, 1}},
      /*options=*/torch::kInt64);
  const auto stats =
      calculate_speculative_output_stats(masked_tokens).sequence_stats;
  ASSERT_EQ(stats.size(), 4);
  for (size_t row = 0; row < stats.size(); ++row) {
    EXPECT_EQ(stats[row].accepted_tokens, static_cast<int64_t>(row));
    EXPECT_EQ(stats[row].proposed_tokens, 3);
  }
}

TEST(SpeculativeTokenStatsTest, BlockHandlesSingleAndZeroDrafts) {
  const auto tokens =
      torch::tensor({{9, -1}, {1, 8}}, /*options=*/torch::kInt64);
  const auto stats = calculate_speculative_output_stats(tokens).sequence_stats;
  ASSERT_EQ(stats.size(), 2);
  EXPECT_EQ(stats[0].accepted_tokens, 0);
  EXPECT_EQ(stats[0].proposed_tokens, 1);
  EXPECT_EQ(stats[1].accepted_tokens, 1);
  EXPECT_EQ(stats[1].proposed_tokens, 1);

  // An empty draft block emits only the target bonus.
  const auto bonus_tokens =
      torch::tensor({{8}, {8}}, /*options=*/torch::kInt64);
  const auto zero_stats =
      calculate_speculative_output_stats(bonus_tokens).sequence_stats;
  ASSERT_EQ(zero_stats.size(), 2);
  for (const SpeculativeTokenStats& row_stats : zero_stats) {
    EXPECT_EQ(row_stats.accepted_tokens, 0);
    EXPECT_EQ(row_stats.proposed_tokens, 0);
  }
}

TEST(WorkerServiceMetricsTest, FixedWidthMtpUpdatesMetricsOnce) {
  runtime::Options options;
  options.enable_speculative_decode(true)
      .num_speculative_tokens(3)
      .speculative_algorithm("mtp");
  WorkerService service(options, torch::Device("npu:0"));
  // The three rows accept two, one, and zero draft tokens.
  const torch::Tensor tokens = torch::tensor(
      {{10, 11, 12, -1}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kInt32);
  const std::vector<SpeculativeTokenStats> output_stats = {
      {2, 3}, {1, 3}, {0, 3}};

  const double drafts_before = COUNTER_VALUE(speculative_num_drafts_total);
  const double committed_before =
      COUNTER_VALUE(speculative_num_committed_tokens_total);
  std::vector<double> positions_before;
  positions_before.reserve(3);
  for (int32_t position = 0; position < 3; ++position) {
    positions_before.emplace_back(
        MULTI_COUNTER_speculative_num_accepted_tokens_per_pos
            .get_stats({std::to_string(position)})
            ->get_value());
  }

  const double proposed_before =
      COUNTER_VALUE(speculative_num_draft_tokens_total);
  const double accepted_before =
      COUNTER_VALUE(speculative_num_accepted_tokens_total);
  const double constrained_proposals_before =
      COUNTER_VALUE(speculative_num_draft_tokens_constrained_total);
  const double constrained_accepted_before =
      COUNTER_VALUE(speculative_num_accepted_tokens_constrained_total);
  ForwardOutput output;
  output.is_speculative_verify = true;
  output.speculative_constrained_mask = {0, 1, 1};
  const auto result = WorkerServiceTestPeer::record_speculative_metrics(
      service, output, tokens);
  EXPECT_DOUBLE_EQ(
      COUNTER_VALUE(speculative_num_draft_tokens_constrained_total),
      constrained_proposals_before + 6);
  EXPECT_DOUBLE_EQ(
      COUNTER_VALUE(speculative_num_accepted_tokens_constrained_total),
      constrained_accepted_before + 1);

  ASSERT_EQ(result.size(), output_stats.size());
  for (size_t row = 0; row < result.size(); ++row) {
    EXPECT_EQ(result[row].accepted_tokens, output_stats[row].accepted_tokens);
    EXPECT_EQ(result[row].proposed_tokens, output_stats[row].proposed_tokens);
  }
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_drafts_total),
                   drafts_before + 3);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_committed_tokens_total),
                   committed_before + 6);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_draft_tokens_total),
                   proposed_before + 9);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_accepted_tokens_total),
                   accepted_before + 3);
  const std::vector<double> expected_positions = {2, 1, 0};
  for (int32_t position = 0; position < 3; ++position) {
    EXPECT_DOUBLE_EQ(MULTI_COUNTER_speculative_num_accepted_tokens_per_pos
                         .get_stats({std::to_string(position)})
                         ->get_value(),
                     positions_before[position] + expected_positions[position]);
  }
  const double mean_tokens = GAUGE_VALUE(speculative_mean_acceptance_length);
  EXPECT_DOUBLE_EQ(mean_tokens, (committed_before + 6) / (drafts_before + 3));
}

TEST(WorkerServiceMetricsTest, SkipsPrefillAndGraphWarmup) {
  runtime::Options options;
  options.enable_speculative_decode(true)
      .num_speculative_tokens(3)
      .speculative_algorithm("mtp");
  WorkerService service(options, torch::Device("npu:0"));
  const torch::Tensor tokens = torch::tensor({{10, 11, -1, -1}}, torch::kInt32);
  const double drafts_before = COUNTER_VALUE(speculative_num_drafts_total);
  const double proposed_before =
      COUNTER_VALUE(speculative_num_draft_tokens_total);
  const double accepted_before =
      COUNTER_VALUE(speculative_num_accepted_tokens_total);
  const double committed_before =
      COUNTER_VALUE(speculative_num_committed_tokens_total);

  ForwardOutput output;
  EXPECT_TRUE(
      WorkerServiceTestPeer::record_speculative_metrics(service, output, tokens)
          .empty());
  output.is_speculative_verify = true;
  output.is_graph_warmup = true;
  EXPECT_TRUE(
      WorkerServiceTestPeer::record_speculative_metrics(service, output, tokens)
          .empty());
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_drafts_total), drafts_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_draft_tokens_total),
                   proposed_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_accepted_tokens_total),
                   accepted_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_committed_tokens_total),
                   committed_before);
}

TEST(WorkerServiceMetricsTest, BlockDiffusionRecordsVerifiedOutput) {
  const torch::Tensor tokens = torch::tensor({{10, 11, -1, -1}}, torch::kInt32);
  ForwardOutput output;
  output.is_speculative_verify = true;
  const double drafts_before = COUNTER_VALUE(speculative_num_drafts_total);
  const double proposed_before =
      COUNTER_VALUE(speculative_num_draft_tokens_total);
  const double accepted_before =
      COUNTER_VALUE(speculative_num_accepted_tokens_total);
  const double committed_before =
      COUNTER_VALUE(speculative_num_committed_tokens_total);

  for (const std::string& algorithm :
       {std::string("dflash"), std::string("dspark")}) {
    runtime::Options options;
    options.enable_speculative_decode(true)
        .num_speculative_tokens(3)
        .speculative_algorithm(algorithm);
    WorkerService service(options, torch::Device("npu:0"));
    const auto result = WorkerServiceTestPeer::record_speculative_metrics(
        service, output, tokens);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].accepted_tokens, 1);
    EXPECT_EQ(result[0].proposed_tokens, 3);
  }
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_drafts_total),
                   drafts_before + 2);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_draft_tokens_total),
                   proposed_before + 6);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_accepted_tokens_total),
                   accepted_before + 2);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_committed_tokens_total),
                   committed_before + 4);
}

}  // namespace
}  // namespace xllm
