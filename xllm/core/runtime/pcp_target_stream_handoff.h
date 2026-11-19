/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once
#include <glog/logging.h>

#include "core/platform/stream.h"

namespace xllm::speculative_worker {

// The caller retains the returned event through the original completion wait.
// This only enqueues a dependency; it does not introduce a host/device wait.
inline StreamEventPtr record_target_stream_handoff(const Stream& producer,
                                                  const Stream& consumer) {
  auto ready = producer.record_event();
  CHECK(ready != nullptr) << "Target stream handoff requires a device event";
  CHECK(consumer.wait_event(ready)) << "Failed to enqueue target stream handoff";
  return ready;
}

}  // namespace xllm::speculative_worker
