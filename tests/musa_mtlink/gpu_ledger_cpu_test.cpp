/* Copyright 2026 The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "core/runtime/pd_gpu_ticket_ledger.h"
#include "core/runtime/pd_retirement_trace.h"

namespace {

namespace trace = xllm::detail;

class CpuMockTicket final : public trace::PdGpuTicket {
 public:
  CpuMockTicket(uint64_t epoch,
                trace::PdGpuCompletion state,
                uint32_t mask = 3,
                uint32_t count = 2)
      : epoch_(epoch), state_(state), mask_(mask), count_(count) {}

  trace::PdGpuTicketObservation query() const override {
    trace::PdGpuTicketObservation value;
    value.state = state_.load();
    value.reason = "cpu_mock_not_gpu_evidence";
    value.epoch = epoch_;
    value.recorded_mask = mask_;
    value.event_count = count_;
    return value;
  }

  void set(trace::PdGpuCompletion state) { state_.store(state); }

 private:
  uint64_t epoch_;
  std::atomic<trace::PdGpuCompletion> state_;
  uint32_t mask_;
  uint32_t count_;
};

class CpuBorrowedObservationTicket final : public trace::PdGpuTicket {
 public:
  trace::PdGpuTicketObservation query() const override {
    trace::PdGpuTicketObservation value;
    value.state = unsupported_.load() ? trace::PdGpuCompletion::UNSUPPORTED
                                      : trace::PdGpuCompletion::READY;
    value.reason = "cpu_provenance_mock_not_gpu_evidence";
    value.epoch = 2;
    value.recorded_mask = 3;
    value.event_count = 2;
    value.context_id = 111;
    value.query_context_borrowed = !unsupported_.load();
    value.original_query_context_id = unsupported_.load() ? 222 : 0;
    value.query_context_id = unsupported_.load() ? 222 : 111;
    return value;
  }
  void make_unsupported() { unsupported_.store(true); }

 private:
  std::atomic<bool> unsupported_{false};
};

class ClosingTicket final : public trace::PdGpuTicket {
 public:
  explicit ClosingTicket(trace::PdGpuTicketLedger& ledger) : ledger_(ledger) {}
  trace::PdGpuTicketObservation query() const override {
    ledger_.close();
    return CpuMockTicket(/*epoch=*/1, trace::PdGpuCompletion::READY).query();
  }

 private:
  trace::PdGpuTicketLedger& ledger_;
};

class ClosingOwnerTicket final : public trace::PdGpuTicket {
 public:
  explicit ClosingOwnerTicket(const void* owner) : owner_(owner) {}
  trace::PdGpuTicketObservation query() const override {
    trace::pd_retire_trace_registry().remove(owner_);
    return CpuMockTicket(/*epoch=*/1, trace::PdGpuCompletion::READY).query();
  }

 private:
  const void* owner_;
};

void test_missing_and_pending_frontier() {
  trace::PdGpuTicketLedger ledger;
  CHECK(ledger.snapshot().latest_observation.state ==
        trace::PdGpuCompletion::NO_TICKET);
  auto second =
      std::make_shared<CpuMockTicket>(2, trace::PdGpuCompletion::READY);
  ledger.publish(2, second);
  CHECK_EQ(ledger.snapshot().completed_frontier, 0);
  auto first =
      std::make_shared<CpuMockTicket>(1, trace::PdGpuCompletion::PENDING);
  ledger.publish(1, first);
  const trace::PdGpuLedgerSnapshot pending = ledger.snapshot();
  CHECK(pending.frontier_valid);
  CHECK_EQ(pending.completed_frontier, 0);
  CHECK(pending.latest_observation.state == trace::PdGpuCompletion::READY);
  first->set(trace::PdGpuCompletion::READY);
  const trace::PdGpuLedgerSnapshot ready = ledger.snapshot();
  CHECK_EQ(ready.completed_frontier, 2);
  CHECK_EQ(ready.retained_tickets, 0);
  const trace::PdGpuLedgerSnapshot cached = ledger.snapshot();
  CHECK(cached.latest_observation.state == trace::PdGpuCompletion::READY);
  CHECK(!cached.observation_from_current_query);
}

void test_nonready_and_invalid_certificates() {
  for (trace::PdGpuCompletion state : {trace::PdGpuCompletion::UNSUPPORTED,
                                       trace::PdGpuCompletion::ERROR,
                                       trace::PdGpuCompletion::PENDING}) {
    trace::PdGpuTicketLedger ledger;
    ledger.publish(1, std::make_shared<CpuMockTicket>(1, state));
    ledger.publish(
        2, std::make_shared<CpuMockTicket>(2, trace::PdGpuCompletion::READY));
    CHECK_EQ(ledger.snapshot().completed_frontier, 0);
  }
  for (uint32_t mask : {0U, 1U, 2U}) {
    trace::PdGpuTicketLedger ledger;
    ledger.publish(1,
                   std::make_shared<CpuMockTicket>(
                       1, trace::PdGpuCompletion::READY, mask));
    const trace::PdGpuLedgerSnapshot invalid = ledger.snapshot();
    CHECK_EQ(invalid.completed_frontier, 0);
    CHECK(invalid.latest_observation.state == trace::PdGpuCompletion::ERROR);
  }
  {
    trace::PdGpuTicketLedger ledger;
    ledger.publish(1,
                   std::make_shared<CpuMockTicket>(
                       1, trace::PdGpuCompletion::READY, 3, 0));
    CHECK(ledger.snapshot().latest_observation.state ==
          trace::PdGpuCompletion::ERROR);
    CHECK_EQ(ledger.snapshot().completed_frontier, 0);
  }
  {
    trace::PdGpuTicketLedger ledger;
    ledger.publish(
        1, std::make_shared<CpuMockTicket>(2, trace::PdGpuCompletion::READY));
    CHECK(ledger.snapshot().latest_observation.state ==
          trace::PdGpuCompletion::ERROR);
    CHECK_EQ(ledger.snapshot().completed_frontier, 0);
  }
  {
    trace::PdGpuTicketLedger ledger;
    ledger.publish(1, nullptr);
    CHECK(ledger.snapshot().latest_observation.state ==
          trace::PdGpuCompletion::NO_TICKET);
    CHECK_EQ(ledger.snapshot().completed_frontier, 0);
  }
}

void test_bounds_duplicate_and_close() {
  {
    trace::PdGpuTicketLedger ledger;
    auto first =
        std::make_shared<CpuMockTicket>(1, trace::PdGpuCompletion::PENDING);
    ledger.publish(1, first);
    ledger.publish(1, first);
    const trace::PdGpuLedgerSnapshot invalid = ledger.snapshot();
    CHECK(!invalid.frontier_valid);
    CHECK_EQ(invalid.completed_frontier, 0);
    CHECK_EQ(invalid.retained_tickets, 0);
  }
  {
    trace::PdGpuTicketLedger ledger;
    for (uint64_t epoch = 1; epoch <= trace::kPdGpuTicketLimit + 1; ++epoch) {
      ledger.publish(epoch,
                     std::make_shared<CpuMockTicket>(
                         epoch, trace::PdGpuCompletion::PENDING));
    }
    const trace::PdGpuLedgerSnapshot invalid = ledger.snapshot();
    CHECK(!invalid.frontier_valid);
    CHECK_EQ(invalid.completed_frontier, 0);
    CHECK_EQ(invalid.retained_tickets, 0);
  }
  {
    trace::PdGpuTicketLedger ledger;
    ledger.publish(1, std::make_shared<ClosingTicket>(ledger));
    const trace::PdGpuLedgerSnapshot closed = ledger.snapshot();
    CHECK(closed.closed);
    CHECK(!closed.frontier_valid);
    CHECK_EQ(closed.completed_frontier, 0);
    CHECK(closed.latest_observation.state ==
          trace::PdGpuCompletion::UNSUPPORTED);
  }
  {
    trace::PdGpuTicketLedger ledger;
    auto ticket =
        std::make_shared<CpuMockTicket>(1, trace::PdGpuCompletion::PENDING);
    std::weak_ptr<trace::PdGpuTicket> weak = ticket;
    ledger.publish(1, ticket);
    ticket.reset();
    CHECK(!weak.expired());
    ledger.close();
    CHECK(weak.expired());
    CHECK(!ledger.snapshot().frontier_valid);
  }
}

void test_historical_ready_is_not_current_query() {
  trace::PdGpuTicketLedger ledger;
  auto later =
      std::make_shared<CpuMockTicket>(2, trace::PdGpuCompletion::READY);
  ledger.publish(2, later);
  CHECK(ledger.snapshot().latest_observation.state ==
        trace::PdGpuCompletion::READY);
  later->set(trace::PdGpuCompletion::UNSUPPORTED);
  const trace::PdGpuLedgerSnapshot unavailable = ledger.snapshot();
  CHECK_EQ(unavailable.completed_frontier, 0);
  CHECK(unavailable.latest_observation.state == trace::PdGpuCompletion::READY);
  CHECK(!unavailable.observation_from_current_query);
}

void test_borrowed_provenance_is_not_fresh_when_cached() {
  trace::PdGpuTicketLedger ledger;
  auto second = std::make_shared<CpuBorrowedObservationTicket>();
  ledger.publish(2, second);
  const auto fresh = ledger.snapshot();
  CHECK(fresh.observation_from_current_query);
  CHECK(fresh.latest_observation.query_context_borrowed);
  CHECK_EQ(fresh.latest_observation.original_query_context_id, 0);
  CHECK_EQ(fresh.latest_observation.query_context_id, 111);
  CHECK_EQ(fresh.completed_frontier, 0);
  second->make_unsupported();
  const auto historical = ledger.snapshot();
  CHECK(!historical.observation_from_current_query);
  CHECK(historical.latest_observation.state == trace::PdGpuCompletion::READY);
  CHECK(historical.latest_observation.query_context_borrowed);
  CHECK_EQ(historical.latest_observation.original_query_context_id, 0);
  CHECK_EQ(historical.latest_observation.query_context_id, 111);
  CHECK_EQ(historical.completed_frontier, 0);
  ledger.publish(
      1, std::make_shared<CpuMockTicket>(1, trace::PdGpuCompletion::READY));
  CHECK_EQ(ledger.snapshot().completed_frontier, 1);
}

void test_actual_owner_binding_lifetime_wrappers() {
  int32_t parent = 0;
  int32_t receiver = 0;
  auto& registry = trace::pd_retire_trace_registry();
  registry.add(&parent, /*device=*/0);
  registry.add(&receiver, /*device=*/0);
  auto owner = registry.find(&parent);
  auto child = registry.find(&receiver);
  const uint64_t epoch = trace::pd_retire_trace_admit(owner);
  trace::pd_retire_trace_mark(owner, epoch, /*submitted=*/false);
  trace::pd_retire_trace_mark(owner, epoch, /*submitted=*/true);
  trace::pd_retire_trace_publish_gpu(
      owner,
      epoch,
      std::make_shared<CpuMockTicket>(epoch, trace::PdGpuCompletion::READY));
  CHECK(trace::pd_retire_trace_bind_execution_parent(&parent, &receiver));
  const trace::PdTraceExecutionSnapshot execution =
      trace::pd_retire_trace_execution_snapshot(child);
  CHECK(execution.scope_available);
  CHECK_EQ(execution.receiver_admitted, 0);
  CHECK_EQ(trace::pd_retire_trace_gpu_snapshot(execution.execution_owner_state)
               .completed_frontier,
           1);
  CHECK(trace::pd_retire_trace_gpu_snapshot(child).latest_observation.state ==
        trace::PdGpuCompletion::NO_TICKET);
  registry.remove(&parent);
  CHECK(!trace::pd_retire_trace_gpu_snapshot(owner).frontier_valid);
  registry.add(&parent, /*device=*/0);
  CHECK(!trace::pd_retire_trace_bind_execution_parent(&parent, &receiver));
  CHECK(!trace::pd_retire_trace_execution_snapshot(child).scope_available);
  registry.remove(&receiver);
  registry.remove(&parent);

  registry.add(&parent, /*device=*/0);
  owner = registry.find(&parent);
  const uint64_t closing_epoch = trace::pd_retire_trace_admit(owner);
  trace::pd_retire_trace_mark(owner, closing_epoch, /*submitted=*/false);
  trace::pd_retire_trace_mark(owner, closing_epoch, /*submitted=*/true);
  trace::pd_retire_trace_publish_gpu(
      owner, closing_epoch, std::make_shared<ClosingOwnerTicket>(&parent));
  const trace::PdGpuLedgerSnapshot closed_during_query =
      trace::pd_retire_trace_gpu_snapshot(owner);
  CHECK(closed_during_query.closed);
  CHECK(!closed_during_query.frontier_valid);
  CHECK(closed_during_query.latest_observation.state ==
        trace::PdGpuCompletion::UNSUPPORTED);
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK_EQ(argc, 1);
  CHECK_EQ(setenv("XLLM_MUSA_PD_RETIRE_TRACE", "1", /*replace=*/1), 0);
  test_missing_and_pending_frontier();
  test_nonready_and_invalid_certificates();
  test_bounds_duplicate_and_close();
  test_historical_ready_is_not_current_query();
  test_actual_owner_binding_lifetime_wrappers();
  test_borrowed_provenance_is_not_fresh_when_cached();
  std::cout << "{\"actual_cpu_ledger_tests\":\"passed\",\"mock_queries_are_not_"
               "gpu_proof\":true}\n";
  return 0;
}
