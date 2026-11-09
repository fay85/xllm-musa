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

#include <folly/futures/Future.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/distributed_runtime/pd_pull_queue_policy.h"
#include "core/util/threadpool.h"

namespace {

using namespace std::chrono_literals;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class PullRelease final {
 public:
  PullRelease() : future_(promise_.get_future().share()) {}
  ~PullRelease() { release(); }

  PullRelease(const PullRelease&) = delete;
  PullRelease& operator=(const PullRelease&) = delete;

  std::shared_future<void> future() const { return future_; }

  void release() {
    if (!released_) {
      released_ = true;
      promise_.set_value();
    }
  }

 private:
  std::promise<void> promise_;
  std::shared_future<void> future_;
  bool released_ = false;
};

void check_pool_progress(size_t request_threads,
                         size_t copy_threads,
                         bool expect_copy,
                         const std::string& role) {
  xllm::ThreadPool request_pool(request_threads, false, role + ".request");
  xllm::ThreadPool copy_pool(copy_threads, false, role + ".copy");
  // Release is destroyed before the pools even if a check throws. The queued
  // lambdas retain all other shared state, so a failing assertion cannot leave
  // a dangling promise or hang ThreadPool's draining destructor.
  PullRelease release;
  const std::shared_future<void> release_future = release.future();
  auto started = std::make_shared<xllm::BlockingCounter>(
      static_cast<int32_t>(request_threads));
  auto completed = std::make_shared<std::atomic<int32_t>>(0);
  std::vector<folly::SemiFuture<bool>> pull_results;
  pull_results.reserve(request_threads);

  xllm::ThreadPool& selected =
      xllm::select_pd_pull_queue(request_pool, copy_pool);
  require((&selected == &copy_pool) == expect_copy,
          role + ": selector returned wrong actual ThreadPool");
  for (size_t index = 0; index < request_threads; ++index) {
    folly::Promise<bool> result;
    pull_results.emplace_back(result.getSemiFuture());
    selected.schedule([started,
                       completed,
                       release_future,
                       result = std::move(result)]() mutable {
      started->decrement_count();
      release_future.wait();
      completed->fetch_add(1, std::memory_order_relaxed);
      result.setValue(true);
    });
  }
  require(started->wait_for(2000ms), role + ": PULL tasks did not all start");

  auto control = std::make_shared<std::promise<void>>();
  std::future<void> control_result = control->get_future();
  request_pool.schedule([control]() { control->set_value(); });
  const std::future_status observed =
      control_result.wait_for(expect_copy ? 2000ms : 75ms);
  require((observed == std::future_status::ready) == expect_copy,
          role + ": unexpected control progress while PULL is blocked");
  require(completed->load(std::memory_order_relaxed) == 0,
          role + ": blocked PULL completed without release");

  release.release();
  require(control_result.wait_for(2000ms) == std::future_status::ready,
          role + ": control did not finish after PULL release");
  control_result.get();
  for (folly::SemiFuture<bool>& result : pull_results) {
    require(std::move(result).get(), role + ": PULL result was not propagated");
  }
  require(completed->load(std::memory_order_relaxed) ==
              static_cast<int32_t>(request_threads),
          role + ": PULL tasks were lost or replayed");
}

void check_future_results(bool expect_copy) {
  xllm::ThreadPool request_pool(1, false, "future.request");
  xllm::ThreadPool copy_pool(4, false, "future.copy");
  xllm::ThreadPool& selected =
      xllm::select_pd_pull_queue(request_pool, copy_pool);
  require((&selected == &copy_pool) == expect_copy,
          "future selector returned wrong pool");

  folly::Promise<bool> rejected;
  auto rejected_future = rejected.getSemiFuture();
  selected.schedule(
      [rejected = std::move(rejected)]() mutable { rejected.setValue(false); });
  require(!std::move(rejected_future).get(),
          "false PULL status was converted to success");

  folly::Promise<bool> exceptional;
  auto exceptional_future = exceptional.getSemiFuture();
  selected.schedule([exceptional = std::move(exceptional)]() mutable {
    exceptional.setException(std::runtime_error("expected_future_exception"));
  });
  bool exception_observed = false;
  try {
    static_cast<void>(std::move(exceptional_future).get());
  } catch (const std::runtime_error& exception) {
    exception_observed =
        std::string(exception.what()) == "expected_future_exception";
  }
  require(exception_observed, "Folly future exception was not preserved");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected --policy, --request, or --copy");
    const std::string operation = argv[1];
    if (operation == "--policy") {
      std::cout << (xllm::pd_pull_copy_queue_enabled() ? "copy" : "request")
                << '\n';
      return 0;
    }
    require(operation == "--request" || operation == "--copy",
            "unknown operation");
    const bool expect_copy = operation == "--copy";
    require(xllm::pd_pull_copy_queue_enabled() == expect_copy,
            "policy did not match externally requested expectation");
    check_pool_progress(/*request_threads=*/1,
                        /*copy_threads=*/4,
                        expect_copy,
                        "RemoteWorker");
    check_pool_progress(/*request_threads=*/4,
                        /*copy_threads=*/5,
                        expect_copy,
                        "WorkerService");
    check_future_results(expect_copy);
    std::cout << "{\"passed\":true,\"route\":\""
              << (expect_copy ? "copy" : "request")
              << "\",\"real_threadpool\":true,\"rpc_or_gpu_test\":false}\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
