/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <utility>

#include "core/framework/config/distributed_config.h"
#include "core/runtime/xservice_client.h"

namespace xllm {

class XServiceClientLifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    old_interval_ = DistributedConfig::get_instance().heart_beat_interval();
    DistributedConfig::get_instance().heart_beat_interval(60.0);
    client_ = std::unique_ptr<XServiceClient>(new XServiceClient());
  }

  void TearDown() override {
    client_.reset();
    DistributedConfig::get_instance().heart_beat_interval(old_interval_);
  }

  void start_periodic_tasks() {
    client_->heartbeat_thread_ = std::make_unique<std::thread>(
        &XServiceClient::heartbeat, client_.get());
    client_->reconcile_thread_ = std::make_unique<std::thread>(
        &XServiceClient::reconcile_registration_loop, client_.get());
  }

  bool tasks_joined() const {
    return (!client_->heartbeat_thread_ ||
            !client_->heartbeat_thread_->joinable()) &&
           (!client_->reconcile_thread_ ||
            !client_->reconcile_thread_->joinable());
  }

  void start_inflight_task(std::promise<void>& entered,
                           std::shared_future<void> release,
                           std::atomic_bool& finished) {
    client_->heartbeat_thread_ = std::make_unique<std::thread>(
        [&entered, release = std::move(release), &finished]() {
          entered.set_value();
          release.wait();
          finished.store(true);
        });
  }

  std::unique_ptr<XServiceClient> client_;
  double old_interval_ = 0.5;
};

TEST_F(XServiceClientLifecycleTest, StopIsIdempotentWithoutInitialization) {
  client_->stop_background_tasks();
  client_->stop_background_tasks();
  EXPECT_TRUE(tasks_joined());
}

TEST_F(XServiceClientLifecycleTest, StopWakesLongIntervalPeriodicTasks) {
  start_periodic_tasks();
  const std::chrono::steady_clock::time_point started =
      std::chrono::steady_clock::now();
  client_->stop_background_tasks();
  EXPECT_LT(std::chrono::steady_clock::now() - started,
            std::chrono::seconds(2));
  EXPECT_TRUE(tasks_joined());
}

TEST_F(XServiceClientLifecycleTest, ConcurrentStopsJoinEachTaskOnlyOnce) {
  start_periodic_tasks();
  std::thread first([this]() { client_->stop_background_tasks(); });
  std::thread second([this]() { client_->stop_background_tasks(); });
  first.join();
  second.join();
  EXPECT_TRUE(tasks_joined());
}

TEST_F(XServiceClientLifecycleTest, StopWaitsForInflightBorrowBeforeReturning) {
  std::promise<void> entered;
  std::promise<void> release;
  std::atomic_bool finished{false};
  start_inflight_task(entered, release.get_future().share(), finished);
  entered.get_future().wait();
  std::future<void> stopped = std::async(
      std::launch::async, [this]() { client_->stop_background_tasks(); });
  EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(30)),
            std::future_status::timeout);
  release.set_value();
  stopped.get();
  EXPECT_TRUE(finished.load());
  EXPECT_TRUE(tasks_joined());
}

}  // namespace xllm
