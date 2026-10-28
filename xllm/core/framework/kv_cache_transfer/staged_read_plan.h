/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <limits>

namespace xllm {

struct StagedReadRun {
  uint64_t remote_buffer_id = 0;
  uint64_t remote_offset = 0;
  uint64_t length = 0;
  uint64_t payload_bytes = 0;
  size_t begin = 0;
  size_t end = 0;
};

// Regions are ordered by (remote buffer, offset). Reading the unused half of a
// TP2 source row replaces many tiny RDMA reads with one registered-buffer span.
// Scatter still copies only the original regions. Bound both staging space and
// read amplification; never join overlapping regions or distinct buffers.
inline bool try_extend_staged_read_run(StagedReadRun& run,
                                       uint64_t remote_buffer_id,
                                       uint64_t remote_offset,
                                       uint64_t length,
                                       size_t end,
                                       uint64_t staging_bytes) {
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  if (run.remote_buffer_id != remote_buffer_id || length == 0 ||
      run.remote_offset > kMax - run.length || remote_offset > kMax - length ||
      run.payload_bytes > kMax - length) {
    return false;
  }
  const uint64_t run_end = run.remote_offset + run.length;
  if (remote_offset < run_end || remote_offset - run_end > length) {
    return false;
  }
  const uint64_t span = remote_offset + length - run.remote_offset;
  const uint64_t payload = run.payload_bytes + length;
  if (span > staging_bytes || span / 2 + span % 2 > payload) {
    return false;
  }
  run.length = span;
  run.payload_bytes = payload;
  run.end = end;
  return true;
}

}  // namespace xllm
