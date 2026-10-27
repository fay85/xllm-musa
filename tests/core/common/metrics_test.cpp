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

#include "common/metrics.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace {

class MetricsDumper final : public bvar::Dumper {
 public:
  bool dump(const std::string& name,
            const butil::StringPiece& description) override {
    values_[name] = description.as_string();
    return true;
  }

  const std::unordered_map<std::string, std::string>& values() const {
    return values_;
  }

 private:
  std::unordered_map<std::string, std::string> values_;
};

TEST(MultimodalMetricsTest, ExportsFixedDimensionNames) {
  MULTI_COUNTER_ADD(mm_input_stage_failures_total, "transfer", 1);
  MULTI_COUNTER_ADD(mm_input_stage_failures_total, "processor", 0);
  MULTI_COUNTER_ADD_3_LABELS(
      mm_input_items_total, "image", "raw", "success", 1);
  MULTI_COUNTER_ADD_3_LABELS(
      mm_input_items_total, "unknown", "unknown", "failed", 1);
  MULTI_COUNTER_ADD_2_LABELS(mm_cache_lookups_total, "processor", "miss", 1);
  MULTI_COUNTER_ADD_2_LABELS(mm_cache_lookups_total, "encoder", "hit", 1);
  MetricsDumper dumper;
  const bvar::DumpOptions options;
  MULTI_COUNTER_mm_input_stage_failures_total.dump(&dumper, &options);
  MULTI_COUNTER_mm_input_items_total.dump(&dumper, &options);
  MULTI_COUNTER_mm_cache_lookups_total.dump(&dumper, &options);

  const auto& values = dumper.values();
  EXPECT_EQ(values.count("mm_input_stage_failures_total_stage_transfer"), 1u);
  EXPECT_EQ(values.count("mm_input_stage_failures_total_stage_processor"), 1u);
  EXPECT_EQ(values.count("mm_input_items_total_modality_image_representation_"
                         "raw_result_success"),
            1u);
  EXPECT_EQ(values.count("mm_input_items_total_modality_unknown_representation_"
                         "unknown_result_failed"),
            1u);
  EXPECT_EQ(values.count("mm_cache_lookups_total_cache_processor_outcome_miss"),
            1u);
  EXPECT_EQ(values.count("mm_cache_lookups_total_cache_encoder_outcome_hit"),
            1u);
}

TEST(MultimodalMetricsTest, ExposesExistingStageCountDenominators) {
  EXPECT_EQ(bvar::Variable::describe_exposed(
                "mm_transfer_latency_milliseconds_count"),
            std::to_string(HISTOGRAM_mm_transfer_latency_milliseconds.count()));
  EXPECT_EQ(
      bvar::Variable::describe_exposed(
          "mm_processor_latency_milliseconds_count"),
      std::to_string(HISTOGRAM_mm_processor_latency_milliseconds.count()));
}

TEST(MultimodalMetricsTest, AccumulatesConcurrentUpdates) {
  constexpr size_t kThreadCount = 4;
  constexpr size_t kUpdatesPerThread = 100;
  bvar::Adder<double>* lookups =
      MULTI_COUNTER_mm_cache_lookups_total.get_stats({"encoder", "hit"});
  bvar::Adder<double>* items =
      MULTI_COUNTER_mm_input_items_total.get_stats({"image", "raw", "success"});
  ASSERT_NE(lookups, nullptr);
  ASSERT_NE(items, nullptr);
  const double failures_before =
      MULTI_COUNTER_VALUE(mm_input_stage_failures_total, "transfer");
  const double lookups_before = lookups->get_value();
  const double items_before = items->get_value();
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t index = 0; index < kThreadCount; ++index) {
    threads.emplace_back([] {
      for (size_t update = 0; update < kUpdatesPerThread; ++update) {
        MULTI_COUNTER_ADD(mm_input_stage_failures_total, "transfer", 2);
        MULTI_COUNTER_ADD_2_LABELS(mm_cache_lookups_total, "encoder", "hit", 1);
        MULTI_COUNTER_ADD_3_LABELS(
            mm_input_items_total, "image", "raw", "success", 1);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  const double updates = static_cast<double>(kThreadCount * kUpdatesPerThread);
  EXPECT_EQ(MULTI_COUNTER_VALUE(mm_input_stage_failures_total, "transfer"),
            failures_before + 2 * updates);
  EXPECT_EQ(lookups->get_value(), lookups_before + updates);
  EXPECT_EQ(items->get_value(), items_before + updates);
}

TEST(DiTMetricsTest, ExportsRequestCountsAndFixedInputStages) {
  EXPECT_EQ(
      bvar::Variable::describe_exposed(
          "dit_input_parse_latency_milliseconds_count"),
      std::to_string(HISTOGRAM_dit_input_parse_latency_milliseconds.count()));
  EXPECT_EQ(
      bvar::Variable::describe_exposed(
          "dit_output_build_latency_milliseconds_count"),
      std::to_string(HISTOGRAM_dit_output_build_latency_milliseconds.count()));

  for (const char* stage : {"media_decode", "tensor_deserialize"}) {
    ASSERT_NE(
        MULTI_HISTOGRAM_dit_input_stage_host_latency_microseconds.get_stats(
            {stage}),
        nullptr);
    ASSERT_NE(MULTI_COUNTER_dit_input_stage_failures_total.get_stats({stage}),
              nullptr);
  }
  MetricsDumper dumper;
  const bvar::DumpOptions options;
  MULTI_HISTOGRAM_dit_input_stage_host_latency_microseconds.dump(&dumper,
                                                                 &options);
  MULTI_COUNTER_dit_input_stage_failures_total.dump(&dumper, &options);
  for (const char* stage : {"media_decode", "tensor_deserialize"}) {
    const std::string count_name =
        std::string("dit_input_stage_host_latency_microseconds_count_stage_") +
        stage;
    const std::string failure_name =
        std::string("dit_input_stage_failures_total_stage_") + stage;
    ASSERT_EQ(dumper.values().count(count_name), 1u);
    EXPECT_EQ(
        dumper.values().at(count_name),
        std::to_string(MULTI_HISTOGRAM_dit_input_stage_host_latency_microseconds
                           .get_stats({stage})
                           ->count()));
    EXPECT_EQ(dumper.values().count(failure_name), 1u);
  }
}

TEST(PrefixCacheHitMetricsTest, UpdatesCumulativeTokenHitRate) {
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);

  record_prefix_cache_hit_metrics(/*prompt_tokens=*/100, /*hit_tokens=*/40);
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/50, /*hit_tokens=*/30);

  const double expected_prompt = prompt_before + 150.0;
  const double expected_hit = hit_before + 70.0;
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   expected_prompt);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), expected_hit);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   expected_hit * 100.0 / expected_prompt);
}

TEST(PrefixCacheHitMetricsTest, ClampsHitsAndIgnoresZeroPrompt) {
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);

  record_prefix_cache_hit_metrics(/*prompt_tokens=*/10, /*hit_tokens=*/20);
  const double rate_after_clamp = GAUGE_VALUE(prefix_cache_token_hit_rate_perc);
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/0, /*hit_tokens=*/20);

  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   prompt_before + 10.0);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total),
                   hit_before + 10.0);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   rate_after_clamp);
}

TEST(PrefixCacheHitMetricsTest, KeepsCountersAndGaugeConsistentConcurrently) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kUpdatesPerThread = 100;
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([]() {
      for (size_t update = 0; update < kUpdatesPerThread; ++update) {
        record_prefix_cache_hit_metrics(/*prompt_tokens=*/4, /*hit_tokens=*/1);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  const double expected_prompt =
      prompt_before + static_cast<double>(kThreadCount * kUpdatesPerThread * 4);
  const double expected_hit =
      hit_before + static_cast<double>(kThreadCount * kUpdatesPerThread);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   expected_prompt);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), expected_hit);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   expected_hit * 100.0 / expected_prompt);
}

}  // namespace
}  // namespace xllm
