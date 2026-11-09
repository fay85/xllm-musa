/* Copyright 2026 The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <glog/logging.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "core/runtime/pd_retirement_trace.h"

namespace {

namespace trace = xllm::detail;

void test_direct_self_and_closed_receiver() {
  int32_t worker = 0;
  auto& registry = trace::pd_retire_trace_registry();
  registry.add(&worker, /*device=*/0);
  auto state = registry.find(&worker);
  const uint64_t epoch = trace::pd_retire_trace_admit(state);
  trace::pd_retire_trace_mark(state, epoch, /*submitted=*/false);
  trace::pd_retire_trace_mark(state, epoch, /*submitted=*/true);
  CHECK(trace::pd_retire_trace_bind_execution_parent(&worker, &worker));
  const trace::PdTraceExecutionSnapshot snapshot =
      trace::pd_retire_trace_execution_snapshot(state);
  CHECK(snapshot.scope_available);
  CHECK_EQ(std::strcmp(snapshot.association, "self"), 0);
  CHECK_EQ(snapshot.receiver_incarnation, state->incarnation);
  CHECK_EQ(snapshot.execution_owner_incarnation, state->incarnation);
  CHECK_EQ(snapshot.admitted, 1);
  CHECK_EQ(snapshot.submitted_frontier, 1);
  registry.remove(&worker);
  const trace::PdTraceExecutionSnapshot closed =
      trace::pd_retire_trace_execution_snapshot(state);
  CHECK(!closed.scope_available);
  CHECK(!closed.detail_valid);
  CHECK(closed.receiver_closed);
  CHECK_EQ(std::strcmp(closed.association, "receiver_closed"), 0);
}

void test_parent_execution_and_sticky_incarnation() {
  int32_t parent = 0;
  int32_t receiver = 0;
  int32_t fresh_receiver = 0;
  auto& registry = trace::pd_retire_trace_registry();
  registry.add(&parent, /*device=*/0);
  registry.add(&receiver, /*device=*/0);
  auto owner = registry.find(&parent);
  auto child = registry.find(&receiver);
  const uint64_t old_owner_incarnation = owner->incarnation;
  const uint64_t first = trace::pd_retire_trace_admit(owner);
  const uint64_t second = trace::pd_retire_trace_admit(owner);
  trace::pd_retire_trace_mark(owner, first, /*submitted=*/false);
  trace::pd_retire_trace_mark(owner, first, /*submitted=*/true);
  trace::pd_retire_trace_mark(owner, second, /*submitted=*/false);
  CHECK(trace::pd_retire_trace_bind_execution_parent(&parent, &receiver));
  trace::PdTraceExecutionSnapshot snapshot =
      trace::pd_retire_trace_execution_snapshot(child);
  CHECK(snapshot.scope_available);
  CHECK_EQ(std::strcmp(snapshot.association, "explicit_parent_binding"), 0);
  CHECK_EQ(snapshot.receiver_incarnation, child->incarnation);
  CHECK_EQ(snapshot.execution_owner_incarnation, old_owner_incarnation);
  CHECK_EQ(snapshot.receiver_admitted, 0);
  CHECK_EQ(snapshot.admitted, 2);
  CHECK_EQ(snapshot.enqueue_ack_frontier, 2);
  CHECK_EQ(snapshot.submitted_frontier, 1);
  CHECK_EQ(snapshot.pending, 1);
  const uint64_t pull =
      trace::pd_retire_trace_pull(child, /*pull_id=*/0, /*entering=*/true);
  CHECK_EQ(pull, 1);
  CHECK_EQ(trace::pd_retire_trace_pull(child, pull, /*entering=*/false), pull);

  // V12 snapshots deliberately retain the execution owner while readers query
  // its GPU ledger. Release this reader before checking the weak parent link.
  snapshot.execution_owner_state.reset();

  std::weak_ptr<trace::PdRetireTraceState> weak_owner = owner;
  registry.remove(&parent);
  CHECK(!trace::pd_retire_trace_execution_snapshot(child).scope_available);
  owner.reset();
  CHECK(weak_owner.expired());
  const trace::PdTraceExecutionSnapshot expired =
      trace::pd_retire_trace_execution_snapshot(child);
  CHECK(!expired.scope_available);
  CHECK(!expired.detail_valid);
  CHECK_EQ(std::strcmp(expired.association, "expired_parent_binding"), 0);
  CHECK_EQ(expired.execution_owner_incarnation, old_owner_incarnation);

  registry.add(&parent, /*device=*/0);
  auto replacement = registry.find(&parent);
  CHECK_GT(replacement->incarnation, old_owner_incarnation);
  CHECK(!trace::pd_retire_trace_bind_execution_parent(&parent, &receiver));
  const trace::PdTraceExecutionSnapshot rejected =
      trace::pd_retire_trace_execution_snapshot(child);
  CHECK(!rejected.scope_available);
  CHECK(!rejected.detail_valid);
  CHECK_EQ(std::strcmp(rejected.association, "parent_incarnation_changed"), 0);
  CHECK_EQ(rejected.execution_owner_incarnation, old_owner_incarnation);

  registry.add(&fresh_receiver, /*device=*/0);
  CHECK(trace::pd_retire_trace_bind_execution_parent(&parent, &fresh_receiver));
  const trace::PdTraceExecutionSnapshot fresh =
      trace::pd_retire_trace_execution_snapshot(registry.find(&fresh_receiver));
  CHECK(fresh.scope_available);
  CHECK_EQ(fresh.execution_owner_incarnation, replacement->incarnation);
  registry.remove(&fresh_receiver);
  registry.remove(&receiver);
  registry.remove(&parent);
}

void test_missing_and_cross_device_parent() {
  int32_t missing = 0;
  int32_t parent = 0;
  int32_t receiver = 0;
  auto& registry = trace::pd_retire_trace_registry();
  CHECK(!trace::pd_retire_trace_execution_snapshot(nullptr).scope_available);
  registry.add(&receiver, /*device=*/0);
  CHECK(!trace::pd_retire_trace_bind_execution_parent(&missing, &receiver));
  const trace::PdTraceExecutionSnapshot missing_scope =
      trace::pd_retire_trace_execution_snapshot(registry.find(&receiver));
  CHECK(!missing_scope.scope_available);
  CHECK(!missing_scope.detail_valid);
  CHECK_EQ(std::strcmp(missing_scope.association, "parent_unregistered"), 0);
  registry.add(&parent, /*device=*/1);
  CHECK(!trace::pd_retire_trace_bind_execution_parent(&parent, &receiver));
  const trace::PdTraceExecutionSnapshot wrong_device =
      trace::pd_retire_trace_execution_snapshot(registry.find(&receiver));
  CHECK(!wrong_device.scope_available);
  CHECK(!wrong_device.detail_valid);
  CHECK_EQ(std::strcmp(wrong_device.association, "device_mismatch_binding"), 0);
  registry.remove(&receiver);
  registry.remove(&parent);
}

void test_nested_and_late_nested_scope() {
  int32_t outer = 0;
  int32_t middle = 0;
  int32_t receiver = 0;
  auto& registry = trace::pd_retire_trace_registry();
  registry.add(&outer, /*device=*/0);
  registry.add(&middle, /*device=*/0);
  registry.add(&receiver, /*device=*/0);
  CHECK(trace::pd_retire_trace_bind_execution_parent(&outer, &middle));
  CHECK(!trace::pd_retire_trace_bind_execution_parent(&middle, &receiver));
  CHECK(!trace::pd_retire_trace_execution_snapshot(registry.find(&receiver))
             .scope_available);
  registry.remove(&receiver);
  registry.remove(&middle);
  registry.remove(&outer);

  registry.add(&outer, /*device=*/0);
  registry.add(&middle, /*device=*/0);
  registry.add(&receiver, /*device=*/0);
  CHECK(trace::pd_retire_trace_bind_execution_parent(&middle, &receiver));
  CHECK(trace::pd_retire_trace_execution_snapshot(registry.find(&receiver))
            .scope_available);
  CHECK(trace::pd_retire_trace_bind_execution_parent(&outer, &middle));
  const trace::PdTraceExecutionSnapshot late =
      trace::pd_retire_trace_execution_snapshot(registry.find(&receiver));
  CHECK(!late.scope_available);
  CHECK(!late.detail_valid);
  CHECK_EQ(std::strcmp(late.association, "nested_parent_binding_unsupported"),
           0);
  registry.remove(&receiver);
  registry.remove(&middle);
  registry.remove(&outer);
}

void test_concurrent_binding_snapshot_and_progress() {
  int32_t parent = 0;
  int32_t receiver = 0;
  auto& registry = trace::pd_retire_trace_registry();
  registry.add(&parent, /*device=*/0);
  registry.add(&receiver, /*device=*/0);
  auto owner = registry.find(&parent);
  auto child = registry.find(&receiver);
  CHECK(trace::pd_retire_trace_bind_execution_parent(&parent, &receiver));
  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int32_t producer = 0; producer < 2; ++producer) {
    threads.emplace_back([owner] {
      for (int32_t index = 0; index < 16; ++index) {
        const uint64_t epoch = trace::pd_retire_trace_admit(owner);
        trace::pd_retire_trace_mark(owner, epoch, /*submitted=*/true);
        std::this_thread::yield();
        trace::pd_retire_trace_mark(owner, epoch, /*submitted=*/false);
      }
    });
  }
  for (int32_t reader = 0; reader < 2; ++reader) {
    threads.emplace_back([child, owner, &parent, &receiver] {
      for (int32_t index = 0; index < 64; ++index) {
        CHECK(trace::pd_retire_trace_bind_execution_parent(&parent, &receiver));
        const trace::PdTraceExecutionSnapshot value =
            trace::pd_retire_trace_execution_snapshot(child);
        CHECK(value.scope_available);
        CHECK_EQ(value.execution_owner_incarnation, owner->incarnation);
        CHECK_LE(value.submitted_frontier, value.admitted);
        CHECK_LE(value.enqueue_ack_frontier, value.admitted);
        CHECK_LE(value.max_submitted, value.admitted);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  const trace::PdTraceExecutionSnapshot final =
      trace::pd_retire_trace_execution_snapshot(child);
  CHECK(final.scope_available);
  CHECK_EQ(final.receiver_admitted, 0);
  CHECK_EQ(final.admitted, 32);
  CHECK_EQ(final.submitted_frontier, 32);
  CHECK_EQ(final.enqueue_ack_frontier, 32);
  CHECK_EQ(final.pending, 0);
  registry.remove(&receiver);
  registry.remove(&parent);
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK_EQ(argc, 2);
  if (std::strcmp(argv[1], "on") == 0) {
    CHECK_EQ(setenv("XLLM_MUSA_PD_RETIRE_TRACE", "1", /*replace=*/1), 0);
    CHECK(trace::pd_retire_trace_enabled());
    test_direct_self_and_closed_receiver();
    test_parent_execution_and_sticky_incarnation();
    test_missing_and_cross_device_parent();
    test_nested_and_late_nested_scope();
    test_concurrent_binding_snapshot_and_progress();
    std::cout
        << "{\"binding_cpu_tests\":\"passed\",\"gpu_tests\":\"not_run\"}\n";
    return 0;
  }
  CHECK_EQ(unsetenv("XLLM_MUSA_PD_RETIRE_TRACE"), 0);
  CHECK(!trace::pd_retire_trace_enabled());
  CHECK(!trace::pd_retire_trace_bind_execution_parent(nullptr, nullptr));
  CHECK(!trace::pd_retire_trace_execution_snapshot(nullptr).scope_available);
  std::cout << "{\"default_off\":true}\n";
  return 0;
}
