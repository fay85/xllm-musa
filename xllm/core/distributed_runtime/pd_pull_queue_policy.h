/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <cstdlib>
#include <cstring>

namespace xllm {

class ThreadPool;

// This experiment changes only the host queue used to wait for a PULL.
// It does not establish GPU retirement or change any transfer fence.
inline bool pd_pull_copy_queue_enabled() {
#if defined(USE_MUSA)
  const char* value = std::getenv("XLLM_MUSA_PD_COPY_QUEUE");
  if (value == nullptr) {
    return false;
  }
  CHECK(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0)
      << "XLLM_MUSA_PD_COPY_QUEUE must be exactly 0 or 1";
  return std::strcmp(value, "1") == 0;
#else
  return false;
#endif
}

// Both pools outlive their scheduled tasks. Keep result collection on the
// request pool while an enabled PULL waits independently on the copy pool.
inline ThreadPool& select_pd_pull_queue(ThreadPool& request_pool,
                                        ThreadPool& copy_pool) {
  return pd_pull_copy_queue_enabled() ? copy_pool : request_pool;
}

}  // namespace xllm
