/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>

namespace xllm {

struct SingleRequestOverlapState {
  bool enabled = false;
  bool in_flight = false;
  int32_t dp_size = 1;
  int64_t last_requests = 0;
  int64_t last_sequences = 0;
  bool same_running_request = false;
  bool queued_work = false;
  bool frontend_queue_empty = true;
  int64_t pending_admissions = 0;
  int64_t pending_prefetch = 0;
};

inline bool should_drain_single_request_overlap(
    const SingleRequestOverlapState& state) {
  // pending_admissions is a frontend counter,not an active-request count.
  // The one live request must be identified from the in-flight batch and all
  // other work queues must be empty before making this latency-only choice.
  return state.enabled && state.in_flight && state.dp_size == 1 &&
         state.last_requests == 1 && state.last_sequences == 1 &&
         state.same_running_request && !state.queued_work &&
         state.frontend_queue_empty && state.pending_admissions == 0 &&
         state.pending_prefetch == 0;
}

}  // namespace xllm
