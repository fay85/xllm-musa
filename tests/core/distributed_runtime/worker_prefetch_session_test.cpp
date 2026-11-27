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

#include "core/distributed_runtime/worker_prefetch_session.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "common/metrics.h"

namespace xllm {
namespace {

StoragePrefetchRequest make_request() {
  StoragePrefetchRequest request;
  request.request_id = "prefetch-session-test";
  request.units.reserve(5);
  for (int32_t unit = 0; unit < 5; ++unit) {
    PrefetchUnit prefetch_unit;
    prefetch_unit.gated_blocks.emplace_back(/*src_block_id=*/-1,
                                            /*dst_block_id=*/unit);
    request.units.emplace_back(std::move(prefetch_unit));
  }
  return request;
}

class SessionHarness final {
 public:
  using Get = decltype(PrefetchSessionCallbacks::get);

  explicit SessionHarness(Get get = nullptr,
                          bool stats_enabled = true,
                          std::function<void()> before_close = nullptr,
                          std::function<void()> before_write = nullptr,
                          bool capture_report = true) {
    if (stats_enabled) {
      stats_pool_ = std::make_unique<ThreadPool>(/*num_threads=*/1);
    }
    copy_pool_ = std::make_unique<ThreadPool>(/*num_threads=*/1);
    PrefetchSessionCallbacks callbacks;
    callbacks.get = [this, get = std::move(get)](
                        Slice<BlockTransferInfo>& blocks,
                        StoreGetStats* stats) {
      ++get_calls;
      if (get != nullptr) {
        return get(blocks, stats);
      }
      EXPECT_EQ(stats != nullptr, stats_pool_ != nullptr);
      if (stats != nullptr) {
        stats->read_bytes = blocks.size() * 100;
        stats->memory_bytes = stats->read_bytes;
        stats->memory_objects = blocks.size();
      }
      return std::vector<uint8_t>(blocks.size(), /*value=*/1);
    };
    callbacks.probe = [this](Slice<BlockTransferInfo>& blocks) {
      ++probe_calls;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        probe_ids.reserve(blocks.size());
        for (const BlockTransferInfo& block : blocks) {
          probe_ids.emplace_back(block.dst_block_id);
        }
      }
      return std::vector<uint8_t>(invalid_probe ? 0 : blocks.size(),
                                  /*value=*/1);
    };
    callbacks.write = [this, before_write = std::move(before_write)](
                          brpc::StreamId /*id*/, const butil::IOBuf& result) {
      if (before_write != nullptr) {
        before_write();
      }
      std::vector<uint8_t> bytes(result.length());
      result.copy_to(bytes.data(), bytes.size());
      {
        std::lock_guard<std::mutex> lock(mutex_);
        writes.emplace_back(std::move(bytes));
      }
      changed_.notify_all();
      return write_error ? -1 : 0;
    };
    callbacks.close =
        [this, before_close = std::move(before_close)](brpc::StreamId id) {
          if (before_close != nullptr) {
            before_close();
          }
          ++close_calls;
          session->on_closed(id);
        };
    if (capture_report) {
      callbacks.report = [this](const PrefetchSessionReport& report) {
        EXPECT_GT(close_calls.load(), 0);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          reports.emplace_back(report);
        }
        changed_.notify_all();
      };
    }
    session = std::make_shared<WorkerPrefetchSession>(copy_pool_.get(),
                                                      stats_pool_.get(),
                                                      make_request(),
                                                      /*batch_size=*/4,
                                                      /*worker_rank=*/0,
                                                      std::move(callbacks));
  }

  ~SessionHarness() { drain(); }

  void start() { session->start(/*stream_id=*/1); }

  void remote_close() {
    ++close_calls;
    session->on_closed(/*id=*/1);
  }

  int32_t control(PrefetchControl control) {
    const uint8_t byte = static_cast<uint8_t>(control);
    butil::IOBuf message;
    message.append(&byte, sizeof(byte));
    butil::IOBuf* messages[] = {&message};
    return session->on_received_messages(/*id=*/1, messages, /*size=*/1);
  }

  bool wait_writes(size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [this, count]() {
      return writes.size() >= count;
    });
  }

  bool wait_reports(size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [this, count]() {
      return reports.size() >= count;
    });
  }

  void drain() {
    session->shutdown();
    copy_pool_.reset();
    stats_pool_.reset();
  }

  std::shared_ptr<WorkerPrefetchSession> session;
  std::atomic<int32_t> get_calls = 0;
  std::atomic<int32_t> probe_calls = 0;
  std::atomic<int32_t> close_calls = 0;
  bool invalid_probe = false;
  bool write_error = false;
  // Inspect these after wait_reports() or drain() synchronizes with the tasks.
  std::vector<std::vector<uint8_t>> writes;
  std::vector<int32_t> probe_ids;
  std::vector<PrefetchSessionReport> reports;

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::unique_ptr<ThreadPool> stats_pool_;
  std::unique_ptr<ThreadPool> copy_pool_;
};

TEST(WorkerPrefetchSessionTest, ReportsShortFinalBatchAndEmptyTail) {
  const double completed_before =
      MULTI_COUNTER_store_prefetch_worker_sessions_total
          .get_stats(/*labels=*/{"completed"})
          ->get_value();
  const uint64_t bytes_before =
      COUNTER_VALUE(store_prefetch_worker_read_bytes_total);
  bvar::Adder<uint64_t>* memory_bytes =
      MULTI_COUNTER_store_prefetch_worker_read_tier_bytes_total.get_stats(
          {"memory"});
  ASSERT_NE(memory_bytes, nullptr);
  const uint64_t memory_before = memory_bytes->get_value();
  std::vector<std::pair<bvar::Adder<double>*, double>> expected = {
      {&COUNTER_store_prefetch_worker_requested_units_total, 5},
      {&COUNTER_store_prefetch_worker_executed_units_total, 5},
      {&COUNTER_store_prefetch_worker_hit_units_total, 5},
      {&COUNTER_store_prefetch_worker_requested_blocks_total, 5},
      {&COUNTER_store_prefetch_worker_hit_blocks_total, 5},
      {&COUNTER_store_prefetch_worker_batches_total, 2}};
  for (auto& [counter, value] : expected) {
    value += counter->get_value();
  }
  std::vector<std::pair<bvar::LatencyRecorder*, int64_t>> latencies;
  latencies.reserve(5);
  for (const char* stage :
       {"queue_wait", "get", "tier_query", "control_wait", "total"}) {
    bvar::LatencyRecorder* latency =
        MULTI_HISTOGRAM_store_prefetch_worker_latency_microseconds.get_stats(
            {stage});
    ASSERT_NE(latency, nullptr);
    latencies.emplace_back(latency, latency->count());
  }
  SessionHarness harness;
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  EXPECT_EQ(harness.control(PrefetchControl::CONTINUE), 0);
  ASSERT_TRUE(harness.wait_writes(/*count=*/2));
  EXPECT_EQ(harness.control(PrefetchControl::STOP), 0);
  ASSERT_TRUE(harness.wait_reports(/*count=*/1));
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  const PrefetchSessionReport& report = harness.reports.front();
  EXPECT_FALSE(report.failed);
  EXPECT_EQ(report.stats.executed_units, 5U);
  EXPECT_EQ(report.stats.hit_units, 5U);
  EXPECT_EQ(report.stats.get.read_bytes, 500U);
  EXPECT_EQ(report.stats.batches, 2U);
  EXPECT_EQ(report.probed_units, 0U);
  EXPECT_EQ(report.probed_present_units, 0U);
  EXPECT_EQ(harness.probe_calls, 0);
  EXPECT_EQ(harness.writes.back(),
            std::vector<uint8_t>({1, 0, 0, 0, 1, 0, 0, 0}));
  EXPECT_EQ(MULTI_COUNTER_store_prefetch_worker_sessions_total
                .get_stats(/*labels=*/{"completed"})
                ->get_value(),
            completed_before + 1);
  EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_read_bytes_total),
            bytes_before + 500);
  EXPECT_EQ(memory_bytes->get_value(), memory_before + 500);
  for (const auto& [counter, value] : expected) {
    EXPECT_EQ(counter->get_value(), value);
  }
  for (const auto& [latency, samples_before] : latencies) {
    EXPECT_EQ(latency->count(), samples_before + 1);
  }
}

TEST(WorkerPrefetchSessionTest, ProbesReadHitsAfterGatedMiss) {
  const uint64_t bytes_before =
      COUNTER_VALUE(store_prefetch_worker_read_bytes_total);
  const double hit_before =
      COUNTER_VALUE(store_prefetch_worker_hit_units_total);
  SessionHarness harness(
      [](Slice<BlockTransferInfo>& /*blocks*/, StoreGetStats* stats) {
        stats->read_bytes = 300;
        return std::vector<uint8_t>({1, 1, 0, 1});
      });
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  EXPECT_EQ(harness.control(PrefetchControl::STOP), 0);
  ASSERT_TRUE(harness.wait_reports(/*count=*/1));
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_EQ(harness.reports.front().stats.probe_begin_unit, 2U);
  EXPECT_EQ(harness.reports.front().probed_present_units, 3U);
  EXPECT_EQ(harness.probe_ids, std::vector<int32_t>({2, 3, 4}));
  EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_read_bytes_total),
            bytes_before + 300);
  EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_hit_units_total),
            hit_before + 3);
  EXPECT_EQ(harness.get_calls, 1);
  EXPECT_EQ(harness.probe_calls, 1);
}

TEST(WorkerPrefetchSessionTest, InvalidProbeRetainsReadStatistics) {
  const double invalid_before =
      MULTI_COUNTER_store_prefetch_worker_probe_sessions_total
          .get_stats(/*labels=*/{"invalid"})
          ->get_value();
  const double present_before =
      COUNTER_VALUE(store_prefetch_worker_probed_present_units_total);
  SessionHarness harness;
  harness.invalid_probe = true;
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  EXPECT_EQ(harness.control(PrefetchControl::STOP), 0);
  ASSERT_TRUE(harness.wait_reports(/*count=*/1));
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_FALSE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_FALSE(harness.reports.front().probed_present_units.has_value());
  EXPECT_EQ(MULTI_COUNTER_store_prefetch_worker_probe_sessions_total
                .get_stats(/*labels=*/{"invalid"})
                ->get_value(),
            invalid_before + 1);
  EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_probed_present_units_total),
            present_before);
}

TEST(WorkerPrefetchSessionTest, FailureWhileWaitingReportsOnceWithoutProbe) {
  const double failed_before =
      MULTI_COUNTER_store_prefetch_worker_sessions_total
          .get_stats(/*labels=*/{"failed"})
          ->get_value();
  const uint64_t bytes_before =
      COUNTER_VALUE(store_prefetch_worker_read_bytes_total);
  SessionHarness harness;
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  harness.session->on_idle_timeout(/*id=*/1);
  harness.session->on_failed(/*id=*/1, /*error_code=*/-1, "stream failure");
  harness.session->on_closed(/*id=*/1);
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_FALSE(harness.reports.front().probed_present_units.has_value());
  EXPECT_EQ(harness.probe_calls, 0);
  EXPECT_EQ(MULTI_COUNTER_store_prefetch_worker_sessions_total
                .get_stats(/*labels=*/{"failed"})
                ->get_value(),
            failed_before + 1);
  EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_read_bytes_total),
            bytes_before + 400);
}

TEST(WorkerPrefetchSessionTest, ShutdownDuringGetDrainsStatistics) {
  std::promise<void> entered;
  std::promise<void> release;
  const std::shared_future<void> released = release.get_future().share();
  SessionHarness harness([&entered, released](Slice<BlockTransferInfo>& blocks,
                                              StoreGetStats* stats) {
    entered.set_value();
    released.wait();
    stats->read_bytes = 400;
    return std::vector<uint8_t>(blocks.size(), /*value=*/1);
  });
  harness.start();
  const std::future_status status =
      entered.get_future().wait_for(std::chrono::seconds(5));
  // Always release the Get before assertions so teardown cannot hang.
  harness.session->shutdown();
  release.set_value();
  EXPECT_EQ(status, std::future_status::ready);
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_EQ(harness.reports.front().stats.executed_units, 4U);
  EXPECT_EQ(harness.probe_calls, 0);
  EXPECT_TRUE(harness.writes.empty());
  EXPECT_EQ(harness.control(PrefetchControl::CONTINUE), 0);
  harness.session->on_closed(/*id=*/1);
  EXPECT_EQ(harness.get_calls, 1);
  EXPECT_EQ(harness.reports.size(), 1U);
}

TEST(WorkerPrefetchSessionTest, ShutdownWaitsForConcurrentStreamClosure) {
  std::promise<void> closing;
  std::promise<void> release;
  const std::shared_future<void> released = release.get_future().share();
  SessionHarness harness(
      /*get=*/nullptr, /*stats_enabled=*/true, [&closing, released]() {
        closing.set_value();
        released.wait();
      });
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  auto failure = std::async(std::launch::async, [&harness]() {
    harness.session->on_idle_timeout(/*id=*/1);
  });
  const std::future_status started =
      closing.get_future().wait_for(std::chrono::seconds(5));
  auto teardown =
      std::async(std::launch::async, [&harness]() { harness.drain(); });
  const std::future_status pending =
      teardown.wait_for(std::chrono::milliseconds(20));
  release.set_value();
  failure.get();
  teardown.get();
  EXPECT_EQ(started, std::future_status::ready);
  EXPECT_EQ(pending, std::future_status::timeout);
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
}

TEST(WorkerPrefetchSessionTest, RemoteCloseDuringGetRetainsStatistics) {
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    const double failed_before =
        MULTI_COUNTER_store_prefetch_worker_sessions_total
            .get_stats(/*labels=*/{"failed"})
            ->get_value();
    const double executed_before =
        COUNTER_VALUE(store_prefetch_worker_executed_units_total);
    std::promise<void> entered;
    std::promise<void> release;
    const std::shared_future<void> released = release.get_future().share();
    SessionHarness harness(
        [&entered, released](Slice<BlockTransferInfo>& blocks,
                             StoreGetStats* stats) {
          entered.set_value();
          released.wait();
          if (stats != nullptr) {
            stats->read_bytes = 400;
          }
          return std::vector<uint8_t>(blocks.size(), /*value=*/1);
        },
        /*stats_enabled=*/enabled);
    harness.start();
    const std::future_status started =
        entered.get_future().wait_for(std::chrono::seconds(5));
    harness.remote_close();
    release.set_value();
    harness.drain();
    EXPECT_EQ(started, std::future_status::ready);
    if (enabled) {
      ASSERT_EQ(harness.reports.size(), 1U);
      EXPECT_TRUE(harness.reports.front().failed);
      EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
      EXPECT_EQ(harness.reports.front().stats.executed_units, 4U);
    } else {
      EXPECT_TRUE(harness.reports.empty());
    }
    harness.session->on_closed(/*id=*/1);
    EXPECT_EQ(MULTI_COUNTER_store_prefetch_worker_sessions_total
                  .get_stats(/*labels=*/{"failed"})
                  ->get_value(),
              failed_before + 1);
    EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_executed_units_total),
              executed_before + 4);
    EXPECT_EQ(harness.get_calls, 1);
    EXPECT_EQ(harness.close_calls, 1);
    EXPECT_EQ(harness.probe_calls, 0);
    EXPECT_TRUE(harness.writes.empty());
  }
}

TEST(WorkerPrefetchSessionTest, RemoteCloseBeforeWriteReportsFailureOnce) {
  std::promise<void> entered;
  std::promise<void> release;
  const std::shared_future<void> released = release.get_future().share();
  SessionHarness harness(
      /*get=*/nullptr,
      /*stats_enabled=*/true,
      /*before_close=*/nullptr,
      [&entered, released]() {
        entered.set_value();
        released.wait();
      });
  harness.write_error = true;
  harness.start();
  const std::future_status started =
      entered.get_future().wait_for(std::chrono::seconds(5));
  harness.remote_close();
  release.set_value();
  harness.drain();
  EXPECT_EQ(started, std::future_status::ready);
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_EQ(harness.reports.front().stats.executed_units, 4U);
  EXPECT_EQ(harness.get_calls, 1);
  EXPECT_EQ(harness.close_calls, 1);
  EXPECT_EQ(harness.probe_calls, 0);
  EXPECT_EQ(harness.writes.size(), 1U);
}

TEST(WorkerPrefetchSessionTest, InvalidGetResultReportsFailure) {
  SessionHarness harness(
      [](Slice<BlockTransferInfo>& /*blocks*/, StoreGetStats* stats) {
        stats->read_bytes = 200;
        return std::vector<uint8_t>({1, 1});
      });
  harness.start();
  ASSERT_TRUE(harness.wait_reports(/*count=*/1));
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 200U);
  EXPECT_TRUE(harness.writes.empty());
  EXPECT_EQ(harness.probe_calls, 0);
}

TEST(WorkerPrefetchSessionTest, WriteFailureReportsWithoutProbe) {
  SessionHarness harness;
  harness.write_error = true;
  harness.start();
  ASSERT_TRUE(harness.wait_reports(/*count=*/1));
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_EQ(harness.probe_calls, 0);
}

TEST(WorkerPrefetchSessionTest, ProtocolErrorReportsAccumulatedStatistics) {
  SessionHarness harness;
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  butil::IOBuf message;
  butil::IOBuf* messages[] = {&message};
  EXPECT_EQ(
      harness.session->on_received_messages(/*id=*/1, messages, /*size=*/1),
      -1);
  ASSERT_TRUE(harness.wait_reports(/*count=*/1));
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_EQ(harness.probe_calls, 0);
}

TEST(WorkerPrefetchSessionTest, FailedCallbackReportsAccumulatedStatistics) {
  SessionHarness harness;
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  harness.session->on_failed(/*id=*/1, /*error_code=*/-1, "stream failure");
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_EQ(harness.probe_calls, 0);
}

TEST(WorkerPrefetchSessionTest, UnexpectedClosureReportsWithoutProbe) {
  SessionHarness harness;
  harness.start();
  ASSERT_TRUE(harness.wait_writes(/*count=*/1));
  harness.remote_close();
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_TRUE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_EQ(harness.probe_calls, 0);
}

TEST(WorkerPrefetchSessionTest, StopDuringGetCompletesAfterWritingResult) {
  std::promise<void> entered;
  std::promise<void> release;
  const std::shared_future<void> released = release.get_future().share();
  SessionHarness harness([&entered, released](Slice<BlockTransferInfo>& blocks,
                                              StoreGetStats* stats) {
    entered.set_value();
    released.wait();
    stats->read_bytes = 400;
    return std::vector<uint8_t>(blocks.size(), /*value=*/1);
  });
  harness.start();
  const std::future_status started =
      entered.get_future().wait_for(std::chrono::seconds(5));
  const int32_t control_result = harness.control(PrefetchControl::STOP);
  release.set_value();
  EXPECT_EQ(started, std::future_status::ready);
  EXPECT_EQ(control_result, 0);
  ASSERT_TRUE(harness.wait_reports(/*count=*/1));
  harness.drain();
  ASSERT_EQ(harness.reports.size(), 1U);
  EXPECT_FALSE(harness.reports.front().failed);
  EXPECT_EQ(harness.reports.front().stats.get.read_bytes, 400U);
  EXPECT_EQ(harness.writes.size(), 1U);
  EXPECT_EQ(harness.probe_ids, std::vector<int32_t>({4}));
}

TEST(WorkerPrefetchSessionTest, ExportsBasicMetricsWithoutDiagnostics) {
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    const uint64_t bytes_before =
        COUNTER_VALUE(store_prefetch_worker_read_bytes_total);
    bvar::LatencyRecorder* latency =
        MULTI_HISTOGRAM_store_prefetch_worker_latency_microseconds.get_stats(
            {"total"});
    ASSERT_NE(latency, nullptr);
    const int64_t samples_before = latency->count();
    const int64_t blocks_samples_before =
        HISTOGRAM_store_prefetch_worker_hit_blocks_per_session.count();
    const int64_t rate_samples_before =
        HISTOGRAM_store_prefetch_worker_block_hit_rate_basis_points.count();
    const double blocks_before =
        COUNTER_VALUE(store_prefetch_worker_hit_blocks_total);
    SessionHarness harness(
        [enabled](Slice<BlockTransferInfo>& blocks, StoreGetStats* stats) {
          EXPECT_EQ(stats != nullptr, enabled);
          if (stats != nullptr) {
            stats->read_bytes = blocks.size() * 100;
          }
          return std::vector<uint8_t>(blocks.size(), enabled ? 1 : 0);
        },
        /*stats_enabled=*/enabled,
        /*before_close=*/nullptr,
        /*before_write=*/nullptr,
        /*capture_report=*/!enabled);
    harness.start();
    const bool wrote = harness.wait_writes(/*count=*/1);
    EXPECT_EQ(harness.control(PrefetchControl::STOP), 0);
    harness.drain();
    EXPECT_TRUE(wrote);
    EXPECT_TRUE(harness.reports.empty());
    EXPECT_EQ(harness.get_calls, 1);
    EXPECT_EQ(harness.probe_calls, enabled ? 1 : 0);
    EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_read_bytes_total),
              bytes_before + (enabled ? 400 : 0));
    EXPECT_EQ(latency->count(), samples_before + 1);
    EXPECT_EQ(COUNTER_VALUE(store_prefetch_worker_hit_blocks_total),
              blocks_before + (enabled ? 4 : 0));
    EXPECT_EQ(HISTOGRAM_store_prefetch_worker_hit_blocks_per_session.count(),
              blocks_samples_before + 1);
    EXPECT_EQ(
        HISTOGRAM_store_prefetch_worker_block_hit_rate_basis_points.count(),
        rate_samples_before + 1);
  }
}

}  // namespace
}  // namespace xllm
