/* Copyright 2026 The xLLM Authors.

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

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
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
  EXPECT_EQ(values.count("mm_input_stage_failures_total{stage=\"transfer\"}"),
            1u);
  EXPECT_EQ(values.count("mm_input_stage_failures_total{stage=\"processor\"}"),
            1u);
  EXPECT_EQ(values.count("mm_input_items_total{modality=\"image\","
                         "representation=\"raw\",result=\"success\"}"),
            1u);
  EXPECT_EQ(values.count("mm_input_items_total{modality=\"unknown\","
                         "representation=\"unknown\",result=\"failed\"}"),
            1u);
  EXPECT_EQ(values.count(
                "mm_cache_lookups_total{cache=\"processor\",outcome=\"miss\"}"),
            1u);
  EXPECT_EQ(
      values.count("mm_cache_lookups_total{cache=\"encoder\",outcome=\"hit\"}"),
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
  const double failures_before = MULTI_COUNTER_mm_input_stage_failures_total
                                     .get_stats(/*labels=*/{"transfer"})
                                     ->get_value();
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
  EXPECT_EQ(MULTI_COUNTER_mm_input_stage_failures_total
                .get_stats(/*labels=*/{"transfer"})
                ->get_value(),
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
        std::string(
            "dit_input_stage_host_latency_microseconds_count{stage=\"") +
        stage + "\"}";
    const std::string failure_name =
        std::string("dit_input_stage_failures_total{stage=\"") + stage + "\"}";
    ASSERT_EQ(dumper.values().count(count_name), 1u);
    EXPECT_EQ(
        dumper.values().at(count_name),
        std::to_string(MULTI_HISTOGRAM_dit_input_stage_host_latency_microseconds
                           .get_stats({stage})
                           ->count()));
    EXPECT_EQ(dumper.values().count(failure_name), 1u);
  }
}

TEST(StorePrefetchMetricsTest, ExportsExactBytesIncludingSmallTiBIncrements) {
  const std::string total_name = "store_prefetch_worker_read_bytes_total";
  bvar::Adder<uint64_t>* memory_counter =
      MULTI_COUNTER_store_prefetch_worker_read_tier_bytes_total.get_stats(
          {"memory"});
  bvar::Adder<uint64_t>* disk_counter =
      MULTI_COUNTER_store_prefetch_worker_read_tier_bytes_total.get_stats(
          {"disk"});
  ASSERT_NE(memory_counter, nullptr);
  ASSERT_NE(disk_counter, nullptr);
  uint64_t total = COUNTER_VALUE(store_prefetch_worker_read_bytes_total);
  uint64_t memory = memory_counter->get_value();
  uint64_t disk = disk_counter->get_value();
  for (const uint64_t bytes :
       {uint64_t{44040192}, uint64_t{1} << 40, uint64_t{1024}}) {
    SCOPED_TRACE(bytes);
    COUNTER_ADD(store_prefetch_worker_read_bytes_total, bytes);
    *memory_counter << bytes - 7;
    *disk_counter << uint64_t{7};
    total += bytes;
    memory += bytes - 7;
    disk += 7;
    MetricsDumper dumper;
    const bvar::DumpOptions options;
    ASSERT_GE(bvar::Variable::dump_exposed(&dumper, &options), 0);
    MULTI_COUNTER_store_prefetch_worker_read_tier_bytes_total.dump(&dumper,
                                                                   &options);
    EXPECT_EQ(dumper.values().at(total_name), std::to_string(total));
    EXPECT_EQ(
        dumper.values().at(
            "store_prefetch_worker_read_tier_bytes_total{tier=\"memory\"}"),
        std::to_string(memory));
    EXPECT_EQ(dumper.values().at(
                  "store_prefetch_worker_read_tier_bytes_total{tier=\"disk\"}"),
              std::to_string(disk));
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

TEST(PrefixCacheHitMetricsTest,
     PairedWindowExpiresAndRecoversWithoutFalseZeros) {
  PrefixCacheHitWindow::Clock::time_point now{};
  PrefixCacheHitWindow window([&now] { return now; });
  bvar::PassiveStatus<double> exported(
      "prefix_cache_idle_window_test",
      [](void* value) {
        return static_cast<PrefixCacheHitWindow*>(value)->hit_rate();
      },
      &window);
  window.record(/*prompt_tokens=*/10, /*hit_tokens=*/20);
  EXPECT_FALSE(window.has_samples());
  EXPECT_TRUE(std::isnan(std::stod(
      bvar::Variable::describe_exposed("prefix_cache_idle_window_test"))));
  window.record(/*prompt_tokens=*/100, /*hit_tokens=*/100);
  EXPECT_TRUE(window.has_samples());
  now += std::chrono::seconds(30);
  window.record(/*prompt_tokens=*/900, /*hit_tokens=*/0);
  window.record(/*prompt_tokens=*/0, /*hit_tokens=*/1000);
  // Averaging request percentages would incorrectly report 50%.
  EXPECT_DOUBLE_EQ(window.hit_rate(), 10.0);
  now += std::chrono::seconds(29);
  EXPECT_DOUBLE_EQ(window.hit_rate(), 10.0);
  now += std::chrono::seconds(1);
  EXPECT_TRUE(window.has_samples());
  EXPECT_DOUBLE_EQ(window.hit_rate(), 0.0);
  now += std::chrono::seconds(30);
  EXPECT_FALSE(window.has_samples());
  EXPECT_TRUE(std::isnan(window.hit_rate()));
  window.record(/*prompt_tokens=*/200, /*hit_tokens=*/50);
  EXPECT_TRUE(window.has_samples());
  EXPECT_DOUBLE_EQ(window.hit_rate(), 25.0);
}

TEST(PrefixCacheHitMetricsTest, RejectsInvalidPairsWithoutClampingThem) {
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/100, /*hit_tokens=*/40);
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);
  const double rate_before = GAUGE_VALUE(prefix_cache_token_hit_rate_perc);
  const double window_before =
      GAUGE_VALUE(prefix_cache_token_hit_rate_window_perc);
  const double invalid_before =
      COUNTER_VALUE(prefix_cache_invalid_hit_samples_total);

  record_prefix_cache_hit_metrics(/*prompt_tokens=*/10, /*hit_tokens=*/20);
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/0, /*hit_tokens=*/20);

  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   prompt_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), hit_before);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc), rate_before);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_window_perc),
                   window_before);
  EXPECT_EQ(COUNTER_VALUE(prefix_cache_invalid_hit_samples_total),
            invalid_before + 2);
}

TEST(PrefixCacheHitMetricsTest, KeepsCountersAndGaugeConsistentConcurrently) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kUpdatesPerThread = 100;
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);
  PrefixCacheHitWindow window;
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&window]() {
      for (size_t update = 0; update < kUpdatesPerThread; ++update) {
        record_prefix_cache_hit_metrics(/*prompt_tokens=*/4, /*hit_tokens=*/1);
        window.record(/*prompt_tokens=*/4, /*hit_tokens=*/1);
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
  EXPECT_DOUBLE_EQ(window.hit_rate(), 25.0);
  EXPECT_EQ(bvar::Variable::describe_exposed(
                "prefix_cache_token_hit_rate_window_valid"),
            "1");
  EXPECT_FALSE(bvar::Variable::describe_exposed(
                   "prefix_cache_token_hit_rate_window_perc")
                   .empty());
}

}  // namespace
}  // namespace xllm
