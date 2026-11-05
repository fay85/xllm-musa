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

#include "core/util/folly_bthread_compat.h"

#include <gtest/gtest.h>
#include <pthread.h>
#include <sys/mman.h>
#include <ucontext.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

namespace {

constexpr size_t kStackBytes = 8 * 1024 * 1024;

struct StackFixture {
  void* mapping = nullptr;
  std::atomic<uint32_t> futex{0};
  std::atomic<uint32_t> advised_calls{0};
  std::atomic<bool> entered{false};
  uint32_t outside_pthread_stack_calls = 0;
  size_t maximum_advised_bytes = 0;
  bool wait_returned = false;
  int32_t context_error = 0;
};

thread_local StackFixture* current_fixture = nullptr;

class IdleTimeoutGuard final {
 public:
  IdleTimeoutGuard()
      : previous_(folly::detail::MemoryIdler::defaultIdleTimeout.load()) {
    folly::detail::MemoryIdler::defaultIdleTimeout.store(
        std::chrono::steady_clock::duration::zero());
  }
  ~IdleTimeoutGuard() {
    folly::detail::MemoryIdler::defaultIdleTimeout.store(previous_);
  }

 private:
  std::chrono::steady_clock::duration previous_;
};

void wait_on_alternate_stack() {
  current_fixture->entered.store(true);
  folly::detail::MemoryIdler::futexWait(current_fixture->futex,
                                        /*expected=*/0);
  current_fixture->wait_returned = true;
}

void* run_alternate_stack(void* argument) {
  current_fixture = static_cast<StackFixture*>(argument);
  ucontext_t original{};
  ucontext_t alternate{};
  if (getcontext(&alternate) != 0) {
    current_fixture->context_error = 1;
    return nullptr;
  }
  alternate.uc_stack.ss_sp =
      static_cast<char*>(current_fixture->mapping) + 2 * kStackBytes;
  alternate.uc_stack.ss_size = kStackBytes;
  alternate.uc_link = &original;
  makecontext(&alternate, wait_on_alternate_stack, /*argc=*/0);
  if (swapcontext(&original, &alternate) != 0) {
    current_fixture->context_error = 2;
  }
  return nullptr;
}

void exercise_alternate_stack(bool compatible) {
  IdleTimeoutGuard guard;
  if (compatible) {
    xllm::initialize_folly_bthread_compatibility();
  }

  StackFixture fixture;
  fixture.mapping = mmap(nullptr,
                         3 * kStackBytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         /*fd=*/-1,
                         /*offset=*/0);
  ASSERT_NE(fixture.mapping, MAP_FAILED);
  pthread_attr_t attributes;
  ASSERT_EQ(pthread_attr_init(&attributes), 0);
  ASSERT_EQ(pthread_attr_setguardsize(&attributes, /*guardsize=*/0), 0);
  ASSERT_EQ(pthread_attr_setstack(&attributes, fixture.mapping, kStackBytes),
            0);
  pthread_t worker;
  ASSERT_EQ(pthread_create(&worker, &attributes, run_alternate_stack, &fixture),
            0);
  ASSERT_EQ(pthread_attr_destroy(&attributes), 0);
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!fixture.entered.load() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  if (compatible) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  } else {
    while (fixture.advised_calls.load() == 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  }
  fixture.futex.store(1);
  folly::detail::futexWake(&fixture.futex, /*count=*/1);
  ASSERT_EQ(pthread_join(worker, nullptr), 0);
  EXPECT_EQ(fixture.context_error, 0);
  EXPECT_TRUE(fixture.wait_returned);
  if (compatible) {
    EXPECT_EQ(fixture.advised_calls.load(), 0);
    EXPECT_EQ(fixture.outside_pthread_stack_calls, 0);
  } else {
    EXPECT_GT(fixture.advised_calls.load(), 0);
    EXPECT_GT(fixture.outside_pthread_stack_calls, 0);
    EXPECT_GT(fixture.maximum_advised_bytes, kStackBytes);
  }
  ::testing::Test::RecordProperty("intercepted_madvise_calls",
                                  std::to_string(fixture.advised_calls.load()));
  ::testing::Test::RecordProperty(
      "outside_pthread_stack_calls",
      std::to_string(fixture.outside_pthread_stack_calls));
  ::testing::Test::RecordProperty(
      "maximum_advised_bytes", std::to_string(fixture.maximum_advised_bytes));
  EXPECT_EQ(munmap(fixture.mapping, 3 * kStackBytes), 0);
}

}  // namespace

// Linker interception is restricted to this CPU test executable. Never invoke
// the real syscall: the negative control intentionally computes an unsafe
// range.
extern "C" int __wrap_madvise(void* address, size_t length, int advice) {
  if (current_fixture == nullptr || advice != MADV_DONTNEED) {
    return 0;
  }
  ++current_fixture->advised_calls;
  current_fixture->maximum_advised_bytes =
      std::max(current_fixture->maximum_advised_bytes, length);
  const uintptr_t begin = std::bit_cast<uintptr_t>(address);
  const uintptr_t stack_begin =
      std::bit_cast<uintptr_t>(current_fixture->mapping);
  const uintptr_t stack_end = stack_begin + kStackBytes;
  if (begin < stack_begin || begin > stack_end || length > stack_end - begin) {
    ++current_fixture->outside_pthread_stack_calls;
  }
  return 0;
}

TEST(FollyBthreadCompatibilityTest, ReproducesUnsafeRangeWithoutDiscardingIt) {
  exercise_alternate_stack(/*compatible=*/false);
}

TEST(FollyBthreadCompatibilityTest,
     DisablesReclamationButPreservesWaitAndWake) {
  exercise_alternate_stack(/*compatible=*/true);
}

TEST(FollyBthreadCompatibilityTest, InitializationIsIdempotent) {
  IdleTimeoutGuard guard;
  xllm::initialize_folly_bthread_compatibility();
  xllm::initialize_folly_bthread_compatibility();
  EXPECT_LT(folly::detail::MemoryIdler::defaultIdleTimeout.load(),
            std::chrono::steady_clock::duration::zero());
}
