/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
==============================================================================*/
#pragma once

#include <cerrno>
#include <cstdint>

namespace xllm::detail {

struct SessionOpenRpcResult {
  bool transport_failed = false;
  int32_t error_code = 0;
  bool peer_ok = false;
};

inline constexpr int64_t kSessionOpenBudgetMs = 5000;
inline constexpr int32_t kSessionOpenMaxAttempts = 9;
inline constexpr int32_t kSessionOpenRetryDelayMs = 500;

inline int32_t session_open_retry_delay_ms(const SessionOpenRpcResult& result,
                                         int32_t attempt,
                                         int64_t remaining_ms) {
  // EHOSTDOWN is BRPC's cached, not-connected socket result. Do not retry
  // timeouts/resets: the peer may already have incremented its reference count.
  if (!result.transport_failed || result.peer_ok || result.error_code != EHOSTDOWN ||
      attempt < 0 || attempt >= kSessionOpenMaxAttempts - 1 ||
      remaining_ms <= kSessionOpenRetryDelayMs) {
    return 0;
  }
  return kSessionOpenRetryDelayMs;
}

// The RPC callback must disable BRPC's immediate internal retries and honor
// the supplied timeout. This is only the RPC phase, before local openSegment
// or handle insertion, and never replays already-successful peer endpoints.
template <typename Call, typename Wait, typename Now>
SessionOpenRpcResult run_session_open_rpc(Call&& call,
                                         Wait&& wait,
                                         Now&& now) {
  const int64_t start_ms = now();
  for (int32_t attempt = 0; attempt < kSessionOpenMaxAttempts; ++attempt) {
    const int64_t remaining_ms = kSessionOpenBudgetMs - (now() - start_ms);
    if (remaining_ms <= 0) {
      return {true, ETIMEDOUT, false};
    }
    const SessionOpenRpcResult result = call(remaining_ms);
    if (!result.transport_failed && result.peer_ok) {
      return result;
    }
    const int32_t delay_ms = session_open_retry_delay_ms(
        result, attempt, kSessionOpenBudgetMs - (now() - start_ms));
    if (delay_ms == 0) {
      return result;
    }
    wait(delay_ms);
  }
  return {true, ETIMEDOUT, false};
}

}  // namespace xllm::detail
