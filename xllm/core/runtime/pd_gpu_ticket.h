/* Copyright 2026 The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace xllm::detail {

enum class PdGpuCompletion : uint8_t {
  NO_TICKET,
  UNSUPPORTED,
  PENDING,
  READY,
  ERROR
};

struct PdGpuTicketObservation {
  PdGpuCompletion state = PdGpuCompletion::NO_TICKET;
  const char* reason = "no_ticket";
  uint64_t epoch = 0;
  uint32_t recorded_mask = 0;
  uint32_t event_count = 0;
  int32_t compute_result = -1;
  int32_t prepare_result = -1;
  uintptr_t context_id = 0;
  uintptr_t compute_stream_id = 0;
  uintptr_t prepare_stream_id = 0;
  uintptr_t query_context_id = 0;
  bool owner_closed = false;
  bool query_context_borrowed = false;
  uintptr_t original_query_context_id = 0;
};

class PdGpuTicket {
 public:
  virtual ~PdGpuTicket() = default;
  virtual PdGpuTicketObservation query() const = 0;
};

inline const char* pd_gpu_completion_name(PdGpuCompletion state) {
  switch (state) {
    case PdGpuCompletion::NO_TICKET:
      return "no_ticket";
    case PdGpuCompletion::UNSUPPORTED:
      return "unsupported";
    case PdGpuCompletion::PENDING:
      return "pending";
    case PdGpuCompletion::READY:
      return "ready";
    case PdGpuCompletion::ERROR:
      return "error";
  }
  return "error";
}

}  // namespace xllm::detail
