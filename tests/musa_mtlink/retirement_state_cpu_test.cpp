/* Copyright 2026 The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <glog/logging.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "core/runtime/pd_retirement_trace.h"

namespace {

void test_out_of_order_and_early_submit() {
  int32_t identity = 0;
  auto& registry = xllm::detail::pd_retire_trace_registry();
  registry.add(&identity, /*device=*/0);
  auto state = registry.find(&identity);
  const uint64_t first = xllm::detail::pd_retire_trace_admit(state);
  const uint64_t second = xllm::detail::pd_retire_trace_admit(state);
  xllm::detail::pd_retire_trace_mark(state, second, /*submitted=*/true);
  xllm::detail::pd_retire_trace_mark(state, second, /*submitted=*/false);
  CHECK_EQ(state->max_submitted, second);
  CHECK_EQ(state->submitted_frontier, 0);
  CHECK_EQ(state->enqueue_ack_frontier, 0);
  xllm::detail::pd_retire_trace_mark(state, first, /*submitted=*/false);
  CHECK_EQ(state->enqueue_ack_frontier, second);
  CHECK_EQ(state->submitted_frontier, 0);
  xllm::detail::pd_retire_trace_mark(state, first, /*submitted=*/true);
  CHECK_EQ(state->submitted_frontier, second);
  CHECK(state->pending.empty());
  CHECK(state->detail_valid);
  registry.remove(&identity);
}

void test_concurrent_hook_order() {
  int32_t identity = 0;
  auto& registry = xllm::detail::pd_retire_trace_registry();
  registry.add(&identity, /*device=*/0);
  auto state = registry.find(&identity);
  std::vector<std::thread> producers;
  producers.reserve(4);
  for (int32_t index = 0; index < 4; ++index) {
    producers.emplace_back([state, index] {
      for (int32_t step = 0; step < 8; ++step) {
        const uint64_t epoch = xllm::detail::pd_retire_trace_admit(state);
        if (index % 2 == 0) {
          xllm::detail::pd_retire_trace_mark(state, epoch, /*submitted=*/true);
          std::this_thread::yield();
          xllm::detail::pd_retire_trace_mark(state, epoch, /*submitted=*/false);
        } else {
          xllm::detail::pd_retire_trace_mark(state, epoch, /*submitted=*/false);
          std::this_thread::yield();
          xllm::detail::pd_retire_trace_mark(state, epoch, /*submitted=*/true);
        }
      }
    });
  }
  for (std::thread& producer : producers) {
    producer.join();
  }
  CHECK_EQ(state->admitted, 32);
  CHECK_EQ(state->enqueue_ack_count, 32);
  CHECK_EQ(state->submitted_count, 32);
  CHECK_EQ(state->enqueue_ack_frontier, 32);
  CHECK_EQ(state->submitted_frontier, 32);
  CHECK(state->pending.empty());
  CHECK(state->detail_valid);
  registry.remove(&identity);
}

void test_pointer_reuse_and_retention_limit() {
  int32_t identity = 0;
  auto& registry = xllm::detail::pd_retire_trace_registry();
  registry.add(&identity, /*device=*/0);
  auto old = registry.find(&identity);
  const uint64_t epoch = xllm::detail::pd_retire_trace_admit(old);
  registry.remove(&identity);
  registry.add(&identity, /*device=*/0);
  auto fresh = registry.find(&identity);
  CHECK_GT(fresh->incarnation, old->incarnation);
  xllm::detail::pd_retire_trace_mark(old, epoch, /*submitted=*/true);
  xllm::detail::pd_retire_trace_mark(old, epoch, /*submitted=*/false);
  CHECK(old->closed);
  CHECK_EQ(old->late_hooks, 2);
  CHECK_EQ(fresh->admitted, 0);
  for (size_t count = 0; count <= xllm::detail::kPdTracePendingLimit; ++count) {
    xllm::detail::pd_retire_trace_admit(fresh);
  }
  CHECK(!fresh->detail_valid);
  CHECK(fresh->pending.empty());
  CHECK_EQ(fresh->submitted_frontier, 0);
  registry.remove(&identity);
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK_EQ(argc, 2);
  if (std::strcmp(argv[1], "on") == 0) {
    CHECK_EQ(setenv("XLLM_MUSA_PD_RETIRE_TRACE", "1", /*replace=*/1), 0);
    CHECK(xllm::detail::pd_retire_trace_enabled());
    test_out_of_order_and_early_submit();
    test_concurrent_hook_order();
    test_pointer_reuse_and_retention_limit();
    std::cout << "{\"cpu_state_tests\":\"passed\",\"gpu_tests\":\"not_run\"}\n";
    return 0;
  }
  if (std::strcmp(argv[1], "off") == 0) {
    CHECK_EQ(unsetenv("XLLM_MUSA_PD_RETIRE_TRACE"), 0);
    CHECK(!xllm::detail::pd_retire_trace_enabled());
    std::cout << "{\"default_off\":true}\n";
    return 0;
  }
  CHECK_EQ(setenv("XLLM_MUSA_PD_RETIRE_TRACE", argv[1], /*replace=*/1), 0);
  const bool enabled = xllm::detail::pd_retire_trace_enabled();
  std::cout << "{\"enabled\":" << enabled << "}\n";
  return 0;
}
