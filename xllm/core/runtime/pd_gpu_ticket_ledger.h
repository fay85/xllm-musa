/* Copyright 2026 The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "core/runtime/pd_gpu_ticket.h"

namespace xllm::detail {

inline constexpr size_t kPdGpuTicketLimit = 32;

struct PdGpuLedgerSnapshot {
  uint64_t latest_epoch = 0;
  uint64_t completed_frontier = 0;
  bool frontier_valid = true;
  bool closed = false;
  size_t retained_tickets = 0;
  bool observation_from_current_query = false;
  PdGpuTicketObservation latest_observation;
};

class PdGpuTicketLedger final {
 public:
  void publish(uint64_t epoch, std::shared_ptr<PdGpuTicket> ticket) {
    // Reap only by real nonblocking queries, outside the ledger mutex.
    snapshot();
    std::map<uint64_t, std::shared_ptr<PdGpuTicket>> released;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;
      }
      if (epoch > latest_epoch_) {
        latest_epoch_ = epoch;
        latest_observation_ = PdGpuTicketObservation();
        latest_observation_.epoch = epoch;
      }
      if (!frontier_valid_) {
        return;
      }
      if (epoch == 0 || epoch <= completed_frontier_ ||
          tickets_.find(epoch) != tickets_.end() ||
          tickets_.size() >= kPdGpuTicketLimit) {
        frontier_valid_ = false;
        latest_observation_.state = PdGpuCompletion::UNSUPPORTED;
        latest_observation_.reason = "ledger_detail_invalid";
        released.swap(tickets_);
        return;
      }
      tickets_.emplace(epoch, std::move(ticket));
    }
  }

  PdGpuLedgerSnapshot snapshot() {
    std::vector<std::pair<uint64_t, std::shared_ptr<PdGpuTicket>>> captured;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      captured.reserve(tickets_.size());
      for (const auto& item : tickets_) {
        captured.emplace_back(item);
      }
    }
    std::map<uint64_t, PdGpuTicketObservation> queried;
    for (const auto& item : captured) {
      PdGpuTicketObservation result;
      result.epoch = item.first;
      if (item.second != nullptr) {
        result = item.second->query();
        if (result.epoch != item.first ||
            (result.state == PdGpuCompletion::READY &&
             (result.recorded_mask != 3 || result.event_count == 0))) {
          result.state = PdGpuCompletion::ERROR;
          result.reason = "invalid_ticket_certificate";
        }
      }
      queried.emplace(item.first, result);
    }
    PdGpuLedgerSnapshot result;
    bool used_current_query = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!closed_ && frontier_valid_) {
        auto latest = queried.find(latest_epoch_);
        if (latest != queried.end()) {
          // Readiness is monotonic for immutable, never-re-recorded tickets.
          if (latest_observation_.state != PdGpuCompletion::READY ||
              latest->second.state == PdGpuCompletion::READY) {
            latest_observation_ = latest->second;
            used_current_query = true;
          }
        }
        for (;;) {
          const uint64_t next = completed_frontier_ + 1;
          auto current = tickets_.find(next);
          auto observation = queried.find(next);
          if (current == tickets_.end() || observation == queried.end() ||
              observation->second.state != PdGpuCompletion::READY) {
            break;
          }
          // A concurrent close/publish cannot replace a matching epoch.
          ++completed_frontier_;
          tickets_.erase(current);
        }
      }
      result.latest_epoch = latest_epoch_;
      result.completed_frontier = completed_frontier_;
      result.frontier_valid = frontier_valid_;
      result.closed = closed_;
      result.retained_tickets = tickets_.size();
      result.latest_observation = latest_observation_;
      result.observation_from_current_query = used_current_query;
      if (closed_) {
        result.frontier_valid = false;
        result.observation_from_current_query = false;
        result.latest_observation.state = PdGpuCompletion::UNSUPPORTED;
        result.latest_observation.reason = "ledger_closed";
      }
    }
    // Captured shared tickets are destroyed after the mutex is released.
    return result;
  }

  void close() {
    std::map<uint64_t, std::shared_ptr<PdGpuTicket>> released;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
      frontier_valid_ = false;
      latest_observation_.state = PdGpuCompletion::UNSUPPORTED;
      latest_observation_.reason = "ledger_closed";
      released.swap(tickets_);
    }
    // Native event destruction, if any, occurs outside the mutex.
  }

 private:
  std::mutex mutex_;
  std::map<uint64_t, std::shared_ptr<PdGpuTicket>> tickets_;
  uint64_t latest_epoch_ = 0;
  uint64_t completed_frontier_ = 0;
  bool frontier_valid_ = true;
  bool closed_ = false;
  PdGpuTicketObservation latest_observation_;
};

}  // namespace xllm::detail
