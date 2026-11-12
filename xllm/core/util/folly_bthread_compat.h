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

#include <folly/detail/MemoryIdler.h>

#include <atomic>
#include <chrono>

namespace xllm {

// Folly's pthread-stack reclamation is unsafe while a brpc bthread is using
// a different stack: release builds may MADV_DONTNEED the memory between the
// pthread stack and the bthread stack. A negative timeout is Folly's explicit
// no-idling mode. Set it before starting servers, including embedded users.
// This trades idle host-stack/cache reclamation for correctness; ordinary
// allocation/free and inference semantics are unchanged.
inline void initialize_folly_bthread_compatibility() {
  folly::detail::MemoryIdler::defaultIdleTimeout.store(
      std::chrono::steady_clock::duration(-1), std::memory_order_release);
}

}  // namespace xllm
