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

#include "core/scheduler/dit_scheduler.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/request/dit_request.h"
#include "core/framework/request/dit_request_state.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

constexpr auto kWaitTimeout = std::chrono::seconds(10);

DiTScheduler::Options make_options(int64_t server_idx) {
  DiTScheduler::Options options;
  options.server_idx(server_idx).max_request_per_batch(2);
  return options;
}

class TestDiTScheduler final : public DiTDynamicBatchScheduler {
 public:
  explicit TestDiTScheduler(int64_t server_idx)
      : DiTDynamicBatchScheduler(make_options(server_idx)) {}

  void report_queues() { report_request_queue_metrics(); }

  std::vector<DiTBatch> prepare() { return prepare_batch(); }

  void clear_running_requests() { running_requests_.clear(); }
};

std::shared_ptr<DiTRequest> make_request(const std::string& request_id,
                                         int32_t width = 512) {
  DiTGenerationParams generation_params;
  generation_params.width = width;
  DiTRequestState state(
      DiTInputParams{},
      std::move(generation_params),
      [](const DiTRequestOutput&) { return true; },
      DiTOutputsFunc{},
      DiTRequestKind::kText);
  return std::make_shared<DiTRequest>(
      request_id, "x-request", "time", std::move(state));
}

TEST(DiTSchedulerTest, WaitingTracksQueuesSelectionCancellationAndDestruction) {
  TestDiTScheduler other(436);
  auto other_request = make_request("other");
  ASSERT_TRUE(other.add_request(other_request));
  other.report_queues();
  auto* other_waiting =
      MULTI_GAUGE_scheduler_waiting_requests.get_stats({"436"});
  auto* ingress =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"ingress", "433"});
  auto* deferred =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"dit_deferred", "433"});
  auto* waiting = MULTI_GAUGE_scheduler_waiting_requests.get_stats({"433"});
  ASSERT_NE(other_waiting, nullptr);
  ASSERT_NE(ingress, nullptr);
  ASSERT_NE(deferred, nullptr);
  ASSERT_NE(waiting, nullptr);
  {
    TestDiTScheduler scheduler(433);
    auto first = make_request("first");
    auto second = make_request("second", 256);
    auto third = make_request("third", 128);
    ASSERT_TRUE(scheduler.add_request(first));
    ASSERT_TRUE(scheduler.add_request(second));
    ASSERT_TRUE(scheduler.add_request(third));
    EXPECT_EQ(GAUGE_VALUE(num_waiting_requests), 3);
    scheduler.report_queues();
    EXPECT_DOUBLE_EQ(ingress->get_value(), 3.0);
    EXPECT_DOUBLE_EQ(waiting->get_value(), 3.0);
    scheduler.prepare();
    EXPECT_DOUBLE_EQ(ingress->get_value(), 1.0);
    EXPECT_DOUBLE_EQ(deferred->get_value(), 1.0);
    EXPECT_DOUBLE_EQ(waiting->get_value(), 2.0);
    EXPECT_EQ(GAUGE_VALUE(num_waiting_requests), 2);
    EXPECT_EQ(GAUGE_VALUE(num_running_requests), 1);
    scheduler.clear_running_requests();
    scheduler.prepare();
    ASSERT_EQ(scheduler.get_running_requests().front(), second);
    EXPECT_DOUBLE_EQ(ingress->get_value(), 0.0);
    EXPECT_DOUBLE_EQ(deferred->get_value(), 1.0);
    third->set_cancel();
    scheduler.clear_running_requests();
    EXPECT_TRUE(scheduler.prepare().front().empty());
    EXPECT_DOUBLE_EQ(deferred->get_value(), 0.0);
    EXPECT_DOUBLE_EQ(waiting->get_value(), 0.0);
    EXPECT_EQ(GAUGE_VALUE(num_waiting_requests), 0);
    EXPECT_EQ(GAUGE_VALUE(num_running_requests), 0);
    auto remaining = make_request(/*request_id=*/"remaining");
    ASSERT_TRUE(scheduler.add_request(remaining));
    scheduler.report_queues();
    EXPECT_DOUBLE_EQ(waiting->get_value(), 1.0);
  }
  EXPECT_DOUBLE_EQ(ingress->get_value(), 0.0);
  EXPECT_DOUBLE_EQ(waiting->get_value(), 0.0);
  EXPECT_DOUBLE_EQ(other_waiting->get_value(), 1.0);
}

std::shared_ptr<DiTRequest> make_request(DiTOutputFunc callback) {
  DiTInputParams input;
  input.prompt = "prompt";
  DiTGenerationParams generation;
  generation.height = 2;
  generation.width = 2;
  DiTRequestState state(std::move(input),
                        std::move(generation),
                        std::move(callback),
                        nullptr,
                        DiTRequestKind::kImage);
  return std::make_shared<DiTRequest>(
      "request", "x-request", "time", std::move(state));
}

TEST(DiTResponseMetricsTest, OutputBuildEndsBeforeCallback) {
  DiTAsyncResponseProcessor processor(/*disable_log_stats=*/true);
  std::promise<void> callback_started;
  auto started = callback_started.get_future();
  std::promise<void> callback_release;
  auto release = callback_release.get_future().share();
  const int64_t output_before =
      HISTOGRAM_dit_output_build_latency_milliseconds.count();
  const double response_before =
      COUNTER_VALUE(responsing_latency_seconds_non_stream);
  auto request = make_request([&](const DiTRequestOutput& output) {
    EXPECT_TRUE(output.status.has_value() && output.status->ok());
    EXPECT_EQ(output.outputs.size(), 1u);
    callback_started.set_value();
    release.wait();
    return true;
  });
  request->handle_forward_output(torch::zeros({1, 3, 2, 2}));
  processor.process_completed_request(request);
  ScopeGuard unblock([&] {
    callback_release.set_value();
    processor.wait_completion();
  });
  ASSERT_EQ(started.wait_for(kWaitTimeout), std::future_status::ready);
  EXPECT_EQ(HISTOGRAM_dit_output_build_latency_milliseconds.count(),
            output_before + 1);
  EXPECT_EQ(COUNTER_VALUE(responsing_latency_seconds_non_stream),
            response_before);
  callback_release.set_value();
  unblock.dismiss();
  processor.wait_completion();
  EXPECT_GT(COUNTER_VALUE(responsing_latency_seconds_non_stream),
            response_before);
}

TEST(DiTResponseMetricsTest, CancelledAndFailedResponsesDoNotBuildOutput) {
  DiTAsyncResponseProcessor processor(/*disable_log_stats=*/true);
  std::atomic<int32_t> callbacks{0};
  auto request = make_request([&](const DiTRequestOutput& output) {
    EXPECT_TRUE(output.status.has_value() &&
                output.status->code() == StatusCode::CANCELLED);
    ++callbacks;
    return true;
  });
  request->set_cancel();
  const int64_t output_before =
      HISTOGRAM_dit_output_build_latency_milliseconds.count();
  const int64_t e2e_before = HISTOGRAM_end_2_end_latency_milliseconds.count();
  const double response_before =
      COUNTER_VALUE(responsing_latency_seconds_non_stream);
  processor.process_completed_request(request);
  processor.process_failed_request(request,
                                   {StatusCode::CANCELLED, "cancelled"});
  processor.wait_completion();
  EXPECT_EQ(callbacks.load(), 1);
  EXPECT_EQ(HISTOGRAM_dit_output_build_latency_milliseconds.count(),
            output_before);
  EXPECT_EQ(HISTOGRAM_end_2_end_latency_milliseconds.count(), e2e_before);
  EXPECT_EQ(COUNTER_VALUE(responsing_latency_seconds_non_stream),
            response_before);
}

}  // namespace
}  // namespace xllm
