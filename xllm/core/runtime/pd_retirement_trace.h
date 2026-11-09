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

#pragma once

#if defined(USE_MUSA)

#include <folly/Indestructible.h>
#include <glog/logging.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "core/runtime/pd_gpu_ticket_ledger.h"

namespace xllm::detail {

inline constexpr size_t kPdTraceWorkerLimit = 64;
inline constexpr size_t kPdTracePendingLimit = 128;
inline constexpr uint64_t kPdTraceEpochLogLimit = 512;
inline constexpr uint64_t kPdTracePullLogLimit = 256;

struct PdTraceEpoch {
  bool enqueue_ack = false;
  bool submitted = false;
};

struct PdRetireTraceState {
  std::mutex mutex;
  uint64_t incarnation = 0;
  int32_t device = -1;
  uint64_t admitted = 0;
  uint64_t enqueue_ack_count = 0;
  uint64_t submitted_count = 0;
  uint64_t max_enqueue_ack = 0;
  uint64_t max_submitted = 0;
  uint64_t enqueue_ack_frontier = 0;
  uint64_t submitted_frontier = 0;
  uint64_t pulls = 0;
  uint64_t late_hooks = 0;
  bool detail_valid = true;
  bool closed = false;
  std::map<uint64_t, PdTraceEpoch> pending;
  std::shared_ptr<PdGpuTicketLedger> gpu_ledger =
      std::make_shared<PdGpuTicketLedger>();
  std::weak_ptr<PdRetireTraceState> execution_parent;
  uint64_t bound_parent_incarnation = 0;
  bool parent_binding_attempted = false;
  const char* binding_failure = "none";
};

inline bool pd_retire_trace_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("XLLM_MUSA_PD_RETIRE_TRACE");
    if (value == nullptr || std::strcmp(value, "0") == 0) {
      return false;
    }
    CHECK_EQ(std::strcmp(value, "1"), 0)
        << "XLLM_MUSA_PD_RETIRE_TRACE must be exactly 0 or 1";
    return true;
  }();
  return enabled;
}

class PdRetireTraceRegistry final {
 public:
  void add(const void* worker, int32_t device) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK(entries_.find(worker) == entries_.end());
    if (entries_.size() >= kPdTraceWorkerLimit) {
      LOG(ERROR) << "[PD_RETIRE_TRACE] event=registry_limit trace_available=0";
      return;
    }
    CHECK_LT(next_incarnation_, std::numeric_limits<uint64_t>::max());
    auto state = std::make_shared<PdRetireTraceState>();
    state->incarnation = ++next_incarnation_;
    state->device = device;
    entries_.emplace(worker, state);
    LOG(INFO) << "[PD_RETIRE_TRACE] event=worker_open incarnation="
              << state->incarnation << " device=" << device
              << " gpu_completion=unknown event_query_supported=0"
              << " owner_generation_proof=0";
  }

  std::shared_ptr<PdRetireTraceState> find(const void* worker) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(worker);
    return it == entries_.end() ? nullptr : it->second;
  }

  void remove(const void* worker) {
    std::shared_ptr<PdRetireTraceState> state;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = entries_.find(worker);
      if (it == entries_.end()) {
        return;
      }
      state = it->second;
      entries_.erase(it);
    }
    std::shared_ptr<PdGpuTicketLedger> ledger;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->closed = true;
      LOG(INFO) << "[PD_RETIRE_TRACE] event=worker_close incarnation="
                << state->incarnation << " admitted=" << state->admitted
                << " enqueue_ack_count=" << state->enqueue_ack_count
                << " submitted_count=" << state->submitted_count
                << " pending=" << state->pending.size()
                << " detail_valid=" << state->detail_valid;
      ledger = state->gpu_ledger;
    }
    ledger->close();
  }

 private:
  std::mutex mutex_;
  uint64_t next_incarnation_ = 0;
  std::unordered_map<const void*, std::shared_ptr<PdRetireTraceState>> entries_;
};

inline PdRetireTraceRegistry& pd_retire_trace_registry() {
  // Fixed process-lifetime bookkeeping stays available during late destruction.
  // Worker entries are bounded and erased; queued hooks own only shared state.
  static folly::Indestructible<PdRetireTraceRegistry> registry;
  return *registry;
}

inline std::shared_ptr<PdRetireTraceState> pd_retire_trace_find(
    const void* worker) {
  return pd_retire_trace_enabled() ? pd_retire_trace_registry().find(worker)
                                   : nullptr;
}

inline void pd_retire_trace_advance(PdRetireTraceState& state) {
  if (!state.detail_valid) {
    return;
  }
  for (;;) {
    auto it = state.pending.find(state.enqueue_ack_frontier + 1);
    if (it == state.pending.end() || !it->second.enqueue_ack) {
      break;
    }
    ++state.enqueue_ack_frontier;
  }
  for (;;) {
    auto it = state.pending.find(state.submitted_frontier + 1);
    if (it == state.pending.end() || !it->second.submitted) {
      break;
    }
    ++state.submitted_frontier;
  }
  const uint64_t retired_cpu_metadata =
      state.enqueue_ack_frontier < state.submitted_frontier
          ? state.enqueue_ack_frontier
          : state.submitted_frontier;
  state.pending.erase(state.pending.begin(),
                      state.pending.upper_bound(retired_cpu_metadata));
}

inline uint64_t pd_retire_trace_admit(
    const std::shared_ptr<PdRetireTraceState>& state) {
  if (state == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  CHECK_LT(state->admitted, std::numeric_limits<uint64_t>::max());
  const uint64_t epoch = ++state->admitted;
  if (state->closed) {
    ++state->late_hooks;
  }
  if (state->detail_valid && state->pending.size() >= kPdTracePendingLimit) {
    state->detail_valid = false;
    state->pending.clear();
    LOG(ERROR) << "[PD_RETIRE_TRACE] event=pending_limit incarnation="
               << state->incarnation << " epoch=" << epoch
               << " detail_valid=0 gpu_completion=unknown";
  }
  if (state->detail_valid) {
    state->pending.emplace(epoch, PdTraceEpoch());
  }
  if (epoch <= kPdTraceEpochLogLimit) {
    LOG(INFO) << "[PD_RETIRE_TRACE] event=admitted incarnation="
              << state->incarnation << " epoch=" << epoch
              << " closed=" << state->closed;
  }
  return epoch;
}

inline void pd_retire_trace_mark(
    const std::shared_ptr<PdRetireTraceState>& state,
    uint64_t epoch,
    bool submitted,
    bool output_present = false) {
  if (state == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  if (state->closed) {
    ++state->late_hooks;
  }
  if (submitted) {
    ++state->submitted_count;
    state->max_submitted =
        epoch > state->max_submitted ? epoch : state->max_submitted;
  } else {
    ++state->enqueue_ack_count;
    state->max_enqueue_ack =
        epoch > state->max_enqueue_ack ? epoch : state->max_enqueue_ack;
  }
  if (state->detail_valid) {
    auto it = state->pending.find(epoch);
    if (it == state->pending.end() ||
        (submitted ? it->second.submitted : it->second.enqueue_ack)) {
      state->detail_valid = false;
      state->pending.clear();
    } else {
      if (submitted) {
        it->second.submitted = true;
      } else {
        it->second.enqueue_ack = true;
      }
      pd_retire_trace_advance(*state);
    }
  }
  if (epoch <= kPdTraceEpochLogLimit) {
    LOG(INFO) << "[PD_RETIRE_TRACE] event="
              << (submitted ? "post_step_submit" : "enqueue_ack")
              << " incarnation=" << state->incarnation << " epoch=" << epoch
              << " enqueue_ack_frontier=" << state->enqueue_ack_frontier
              << " submitted_frontier=" << state->submitted_frontier
              << " max_submitted=" << state->max_submitted
              << " output_present=" << output_present
              << " detail_valid=" << state->detail_valid
              << " closed=" << state->closed
              << " gpu_completion=unknown event_query_supported=0";
  }
}

struct PdTraceExecutionSnapshot {
  std::shared_ptr<PdRetireTraceState> execution_owner_state;
  uint64_t receiver_incarnation = 0;
  uint64_t execution_owner_incarnation = 0;
  const char* association = "receiver_unavailable";
  bool scope_available = false;
  bool receiver_closed = false;
  uint64_t receiver_admitted = 0;
  uint64_t admitted = 0;
  uint64_t max_enqueue_ack = 0;
  uint64_t enqueue_ack_frontier = 0;
  uint64_t max_submitted = 0;
  uint64_t submitted_frontier = 0;
  size_t pending = 0;
  bool detail_valid = false;
};

inline bool pd_retire_trace_bind_execution_parent(const void* parent_worker,
                                                  const void* receiver_worker) {
  if (!pd_retire_trace_enabled()) {
    return false;
  }
  auto parent = pd_retire_trace_registry().find(parent_worker);
  auto receiver = pd_retire_trace_registry().find(receiver_worker);
  if (receiver == nullptr) {
    LOG(ERROR)
        << "[PD_RETIRE_TRACE] event=execution_binding incarnation=0"
        << " association=receiver_unavailable execution_scope_available=0";
    return false;
  }
  if (parent == receiver) {
    return true;
  }
  if (parent == nullptr) {
    std::lock_guard<std::mutex> lock(receiver->mutex);
    receiver->parent_binding_attempted = true;
    receiver->binding_failure = "parent_unregistered";
    LOG(ERROR)
        << "[PD_RETIRE_TRACE] event=execution_binding incarnation="
        << receiver->incarnation
        << " association=parent_unregistered execution_scope_available=0";
    return false;
  }
  std::scoped_lock lock(receiver->mutex, parent->mutex);
  const char* failure = nullptr;
  if (receiver->closed || parent->closed) {
    failure = "closed_worker_binding";
  } else if (receiver->device != parent->device) {
    failure = "device_mismatch_binding";
  } else if (receiver->bound_parent_incarnation != 0 &&
             receiver->bound_parent_incarnation != parent->incarnation) {
    failure = "parent_incarnation_changed";
  } else if (parent->parent_binding_attempted) {
    failure = "nested_parent_binding_unsupported";
  }
  receiver->parent_binding_attempted = true;
  if (failure != nullptr) {
    receiver->binding_failure = failure;
    LOG(ERROR) << "[PD_RETIRE_TRACE] event=execution_binding incarnation="
               << receiver->incarnation
               << " receiver_incarnation=" << receiver->incarnation
               << " execution_owner_incarnation=" << parent->incarnation
               << " association=" << failure << " execution_scope_available=0";
    return false;
  }
  const bool first = receiver->bound_parent_incarnation == 0;
  receiver->execution_parent = parent;
  receiver->bound_parent_incarnation = parent->incarnation;
  receiver->binding_failure = "none";
  if (first) {
    LOG(INFO)
        << "[PD_RETIRE_TRACE] event=execution_binding incarnation="
        << receiver->incarnation
        << " receiver_incarnation=" << receiver->incarnation
        << " execution_owner_incarnation=" << parent->incarnation
        << " association=explicit_parent_binding execution_scope_available=1"
        << " gpu_completion=unknown owner_generation_proof=0";
  }
  return true;
}

inline void pd_retire_trace_copy_execution(const PdRetireTraceState& owner,
                                           PdTraceExecutionSnapshot& snapshot) {
  snapshot.execution_owner_incarnation = owner.incarnation;
  snapshot.scope_available = true;
  snapshot.admitted = owner.admitted;
  snapshot.max_enqueue_ack = owner.max_enqueue_ack;
  snapshot.enqueue_ack_frontier = owner.enqueue_ack_frontier;
  snapshot.max_submitted = owner.max_submitted;
  snapshot.submitted_frontier = owner.submitted_frontier;
  snapshot.pending = owner.pending.size();
  snapshot.detail_valid = owner.detail_valid;
}

inline PdTraceExecutionSnapshot pd_retire_trace_execution_snapshot(
    const std::shared_ptr<PdRetireTraceState>& receiver) {
  PdTraceExecutionSnapshot snapshot;
  if (receiver == nullptr) {
    return snapshot;
  }
  std::shared_ptr<PdRetireTraceState> parent;
  {
    std::lock_guard<std::mutex> lock(receiver->mutex);
    snapshot.receiver_incarnation = receiver->incarnation;
    snapshot.receiver_closed = receiver->closed;
    snapshot.receiver_admitted = receiver->admitted;
    if (receiver->closed) {
      snapshot.association = "receiver_closed";
      return snapshot;
    }
    if (!receiver->parent_binding_attempted) {
      snapshot.association = "self";
      snapshot.execution_owner_state = receiver;
      pd_retire_trace_copy_execution(*receiver, snapshot);
      return snapshot;
    }
    snapshot.execution_owner_incarnation = receiver->bound_parent_incarnation;
    if (std::strcmp(receiver->binding_failure, "none") != 0) {
      snapshot.association = receiver->binding_failure;
      return snapshot;
    }
    parent = receiver->execution_parent.lock();
    if (parent == nullptr) {
      snapshot.association = "expired_parent_binding";
      return snapshot;
    }
  }
  std::scoped_lock lock(receiver->mutex, parent->mutex);
  snapshot.receiver_closed = receiver->closed;
  snapshot.receiver_admitted = receiver->admitted;
  if (receiver->closed || parent->closed) {
    snapshot.association = "closed_worker_binding";
  } else if (std::strcmp(receiver->binding_failure, "none") != 0) {
    snapshot.association = receiver->binding_failure;
  } else if (receiver->execution_parent.lock() != parent ||
             receiver->bound_parent_incarnation != parent->incarnation) {
    snapshot.association = "parent_binding_changed";
  } else if (receiver->device != parent->device) {
    snapshot.association = "device_mismatch_binding";
  } else if (parent->parent_binding_attempted) {
    snapshot.association = "nested_parent_binding_unsupported";
  } else {
    snapshot.association = "explicit_parent_binding";
    snapshot.execution_owner_state = parent;
    pd_retire_trace_copy_execution(*parent, snapshot);
  }
  return snapshot;
}

inline void pd_retire_trace_publish_gpu(
    const std::shared_ptr<PdRetireTraceState>& state,
    uint64_t epoch,
    std::shared_ptr<PdGpuTicket> ticket) {
  if (state == nullptr) {
    return;
  }
  std::shared_ptr<PdGpuTicketLedger> ledger;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closed) {
      return;
    }
    ledger = state->gpu_ledger;
  }
  ledger->publish(epoch, std::move(ticket));
}

inline PdGpuLedgerSnapshot pd_retire_trace_gpu_snapshot(
    const std::shared_ptr<PdRetireTraceState>& state) {
  PdGpuLedgerSnapshot result;
  if (state == nullptr) {
    result.frontier_valid = false;
    return result;
  }
  std::shared_ptr<PdGpuTicketLedger> ledger;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closed || !state->detail_valid) {
      result.frontier_valid = false;
      result.closed = state->closed;
      result.latest_observation.state = PdGpuCompletion::UNSUPPORTED;
      result.latest_observation.reason = "execution_owner_unavailable";
      return result;
    }
    ledger = state->gpu_ledger;
  }
  result = ledger->snapshot();
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closed || !state->detail_valid) {
      result.frontier_valid = false;
      result.closed = state->closed;
      result.latest_observation.state = PdGpuCompletion::UNSUPPORTED;
      result.latest_observation.reason = "execution_owner_closed_during_query";
      result.observation_from_current_query = false;
    }
  }
  return result;
}

inline uint64_t pd_retire_trace_pull(
    const std::shared_ptr<PdRetireTraceState>& state,
    uint64_t pull_id,
    bool entering) {
  if (state == nullptr) {
    return 0;
  }
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (entering) {
      CHECK_LT(state->pulls, std::numeric_limits<uint64_t>::max());
      pull_id = ++state->pulls;
    }
  }
  if (pull_id <= kPdTracePullLogLimit) {
    const PdTraceExecutionSnapshot snapshot =
        pd_retire_trace_execution_snapshot(state);
    const PdGpuLedgerSnapshot gpu =
        pd_retire_trace_gpu_snapshot(snapshot.execution_owner_state);
    LOG(INFO) << "[PD_RETIRE_TRACE] event="
              << (entering ? "pull_entry" : "pull_dispatch_return")
              << " incarnation=" << snapshot.receiver_incarnation
              << " receiver_incarnation=" << snapshot.receiver_incarnation
              << " execution_owner_incarnation="
              << snapshot.execution_owner_incarnation
              << " association=" << snapshot.association
              << " execution_scope_available=" << snapshot.scope_available
              << " receiver_closed=" << snapshot.receiver_closed
              << " receiver_admitted=" << snapshot.receiver_admitted
              << " pull_id=" << pull_id << " admitted=" << snapshot.admitted
              << " max_enqueue_ack=" << snapshot.max_enqueue_ack
              << " enqueue_ack_frontier=" << snapshot.enqueue_ack_frontier
              << " max_submitted=" << snapshot.max_submitted
              << " submitted_frontier=" << snapshot.submitted_frontier
              << " pending=" << snapshot.pending
              << " detail_valid=" << snapshot.detail_valid << " gpu_completion="
              << pd_gpu_completion_name(gpu.latest_observation.state)
              << " gpu_observation_reason=" << gpu.latest_observation.reason
              << " gpu_observation_current_query="
              << gpu.observation_from_current_query
              << " gpu_latest_epoch=" << gpu.latest_epoch
              << " gpu_completed_frontier=" << gpu.completed_frontier
              << " gpu_frontier_valid=" << gpu.frontier_valid
              << " gpu_recorded_mask=" << gpu.latest_observation.recorded_mask
              << " gpu_event_count=" << gpu.latest_observation.event_count
              << " gpu_compute_result=" << gpu.latest_observation.compute_result
              << " gpu_prepare_result=" << gpu.latest_observation.prepare_result
              << " gpu_context_id=" << gpu.latest_observation.context_id
              << " gpu_query_context_id="
              << gpu.latest_observation.query_context_id
              << " gpu_original_query_context_id="
              << gpu.latest_observation.original_query_context_id
              << " gpu_query_context_borrowed="
              << gpu.latest_observation.query_context_borrowed
              << " gpu_compute_stream_id="
              << gpu.latest_observation.compute_stream_id
              << " gpu_prepare_stream_id="
              << gpu.latest_observation.prepare_stream_id
              << " gpu_scope=recorded_compute_prepare_only"
              << " owner_generation_proof=0 owner_closed=0 "
                 "physical_id_intersection=unsupported";
  }
  return pull_id;
}

}  // namespace xllm::detail

#endif  // USE_MUSA
