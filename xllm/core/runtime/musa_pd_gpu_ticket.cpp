/* Copyright 2026 The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "core/runtime/musa_pd_gpu_ticket.h"

#include <glog/logging.h>

// Bind plain Driver symbols even if SDK per-thread-default-stream macros are
// enabled by another include. NULL is the legacy stream contract here.
extern "C" MUresult MUSAAPI
    pd_ticket_event_record(MUevent, MUstream) __asm__("muEventRecord");
extern "C" MUresult MUSAAPI
pd_ticket_stream_context(MUstream, MUcontext*) __asm__("muStreamGetCtx");

namespace xllm::detail {
namespace {

bool is_legacy(MUstream stream) {
  return stream == nullptr || stream == MU_STREAM_LEGACY;
}

MUstream canonical_stream(MUstream stream) {
  return is_legacy(stream) ? nullptr : stream;
}

void require_success(MUresult result, const char* operation) {
  CHECK_EQ(result, MUSA_SUCCESS)
      << "MUSA PD GPU ticket " << operation
      << " failed, driver_result=" << static_cast<int32_t>(result);
}

class ScopedEmptyQueryContext final {
 public:
  explicit ScopedEmptyQueryContext(MUcontext recorded, bool borrowed)
      : context_(recorded), borrowed_(borrowed) {
    if (borrowed_) {
      require_success(muCtxPushCurrent(context_),
                      "muCtxPushCurrent(empty query)");
    }
  }
  ~ScopedEmptyQueryContext() {
    if (borrowed_) {
      MUcontext popped = nullptr;
      require_success(muCtxPopCurrent(&popped), "muCtxPopCurrent(empty query)");
      CHECK(popped == context_) << "MUSA PD query popped wrong context";
      MUcontext restored = nullptr;
      require_success(muCtxGetCurrent(&restored),
                      "muCtxGetCurrent(restored query)");
      CHECK(restored == nullptr)
          << "MUSA PD query did not restore empty context";
    }
  }
  ScopedEmptyQueryContext(const ScopedEmptyQueryContext&) = delete;
  ScopedEmptyQueryContext& operator=(const ScopedEmptyQueryContext&) = delete;

 private:
  MUcontext context_;
  bool borrowed_;
};

bool validate_stream(MUstream stream,
                     MUcontext current,
                     PdGpuTicketObservation& observation,
                     bool compute) {
  if (is_legacy(stream)) {
    return true;
  }
  MUcontext context = nullptr;
  const MUresult result = pd_ticket_stream_context(stream, &context);
  if (result == MUSA_ERROR_INVALID_HANDLE ||
      result == MUSA_ERROR_INVALID_CONTEXT ||
      result == MUSA_ERROR_INVALID_VALUE) {
    observation.reason = "stream_context_query_rejected";
    if (compute) {
      observation.compute_result = static_cast<int32_t>(result);
    } else {
      observation.prepare_result = static_cast<int32_t>(result);
    }
    return false;
  }
  require_success(result, "muStreamGetCtx");
  if (context == nullptr || context != current) {
    observation.reason = "record_stream_context_mismatch";
    return false;
  }
  return true;
}

}  // namespace

MusaPdGpuTicket::MusaPdGpuTicket(uint64_t epoch,
                                 MUstream compute,
                                 MUstream prepare) {
  observation_.state = PdGpuCompletion::UNSUPPORTED;
  observation_.reason = "not_recorded";
  observation_.epoch = epoch;
  observation_.compute_stream_id = reinterpret_cast<uintptr_t>(compute);
  observation_.prepare_stream_id = reinterpret_cast<uintptr_t>(prepare);
}

std::shared_ptr<PdGpuTicket> MusaPdGpuTicket::record(uint64_t epoch,
                                                     MUstream compute,
                                                     MUstream prepare,
                                                     bool eligible) {
  auto ticket = std::shared_ptr<MusaPdGpuTicket>(
      new MusaPdGpuTicket(epoch, compute, prepare));
  if (!eligible) {
    ticket->observation_.reason = "execution_scope_ineligible";
    return ticket;
  }
  if (compute == MU_STREAM_PER_THREAD || prepare == MU_STREAM_PER_THREAD) {
    ticket->observation_.reason = "per_thread_stream_unsupported";
    return ticket;
  }
  MUcontext current = nullptr;
  const MUresult context_result = muCtxGetCurrent(&current);
  if (context_result == MUSA_ERROR_NOT_INITIALIZED) {
    ticket->observation_.reason = "driver_not_initialized";
    return ticket;
  }
  require_success(context_result, "muCtxGetCurrent(record)");
  ticket->observation_.context_id = reinterpret_cast<uintptr_t>(current);
  if (current == nullptr) {
    ticket->observation_.reason = "record_context_unavailable";
    return ticket;
  }
  // Validate BOTH roles before creating or recording any event.
  if (!validate_stream(compute, current, ticket->observation_, true) ||
      !validate_stream(prepare, current, ticket->observation_, false)) {
    return ticket;
  }
  ticket->context_ = current;
  const MUstream compute_stream = canonical_stream(compute);
  const MUstream prepare_stream = canonical_stream(prepare);
  require_success(muEventCreate(&ticket->events_[0], MU_EVENT_DISABLE_TIMING),
                  "muEventCreate(compute)");
  ticket->event_count_ = 1;
  ticket->role_masks_[0] = compute_stream == prepare_stream ? 3U : 1U;
  require_success(pd_ticket_event_record(ticket->events_[0], compute_stream),
                  "muEventRecord(compute)");
  if (compute_stream != prepare_stream) {
    require_success(muEventCreate(&ticket->events_[1], MU_EVENT_DISABLE_TIMING),
                    "muEventCreate(prepare)");
    ticket->event_count_ = 2;
    ticket->role_masks_[1] = 2;
    require_success(pd_ticket_event_record(ticket->events_[1], prepare_stream),
                    "muEventRecord(prepare)");
  }
  ticket->observation_.recorded_mask = 3;
  ticket->observation_.event_count = ticket->event_count_;
  ticket->observation_.state = PdGpuCompletion::PENDING;
  ticket->observation_.reason = "recorded_not_queried";
  return ticket;
}

PdGpuTicketObservation MusaPdGpuTicket::query() const {
  PdGpuTicketObservation observation = observation_;
  if (observation.state == PdGpuCompletion::UNSUPPORTED) {
    return observation;
  }
  MUcontext current = nullptr;
  require_success(muCtxGetCurrent(&current), "muCtxGetCurrent(query)");
  observation.original_query_context_id = reinterpret_cast<uintptr_t>(current);
  observation.query_context_id = reinterpret_cast<uintptr_t>(current);
  if (current != nullptr && current != context_) {
    observation.state = PdGpuCompletion::UNSUPPORTED;
    observation.reason = "query_context_mismatch";
    return observation;
  }
  // Auxiliary CPU readers may have no Driver context after Runtime set-device.
  // Borrow only this ticket's existing recording context, never create one or
  // override a nonempty caller context. The scope restores NULL before return.
  observation.query_context_borrowed = current == nullptr;
  ScopedEmptyQueryContext query_scope(context_,
                                      observation.query_context_borrowed);
  observation.query_context_id = reinterpret_cast<uintptr_t>(context_);
  bool ready = true;
  for (uint32_t index = 0; index < event_count_; ++index) {
    const MUresult result = muEventQuery(events_[index]);
    if (result != MUSA_SUCCESS && result != MUSA_ERROR_NOT_READY) {
      require_success(result, "muEventQuery");
    }
    ready = ready && result == MUSA_SUCCESS;
    if ((role_masks_[index] & 1U) != 0) {
      observation.compute_result = static_cast<int32_t>(result);
    }
    if ((role_masks_[index] & 2U) != 0) {
      observation.prepare_result = static_cast<int32_t>(result);
    }
  }
  observation.state = ready ? PdGpuCompletion::READY : PdGpuCompletion::PENDING;
  observation.reason =
      ready ? "all_recorded_events_ready" : "recorded_events_pending";
  return observation;
}

MusaPdGpuTicket::~MusaPdGpuTicket() {
  if (event_count_ == 0) {
    return;
  }
  MUcontext current = nullptr;
  require_success(muCtxGetCurrent(&current), "muCtxGetCurrent(cleanup)");
  const bool pushed = current != context_;
  if (pushed) {
    require_success(muCtxPushCurrent(context_), "muCtxPushCurrent(cleanup)");
  }
  // Driver contract permits pending-event destruction without a completion
  // wait.
  for (uint32_t index = 0; index < event_count_; ++index) {
    require_success(muEventDestroy(events_[index]), "muEventDestroy");
  }
  if (pushed) {
    MUcontext popped = nullptr;
    require_success(muCtxPopCurrent(&popped), "muCtxPopCurrent(cleanup)");
    CHECK(popped == context_) << "MUSA PD ticket cleanup popped wrong context";
  }
}

}  // namespace xllm::detail
