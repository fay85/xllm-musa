/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/platform/stream.h"

#if defined(USE_MUSA)
#include <musa_runtime_api.h>
#include <torch/utils.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>
#endif

namespace xllm {

// Sample runtime phase boundaries on its actual compute stream. This is a
// separate diagnostic, not kernel busy time. Never synchronize to read events
// and never insert markers into a captured model graph.
class MusaSpecStepTiming final {
 public:
  MusaSpecStepTiming(const Stream& stream,
                     int64_t proposal_rows,
                     int64_t batch_size,
                     bool graph_warmup,
                     const std::vector<std::string>& request_ids) {
#if defined(USE_MUSA)
    State& state = local_state();
    if (state.path.empty() || graph_warmup || current_ != nullptr) {
      return;
    }
    torch::DeviceGuard guard(stream.get_stream()->device());
    flush(state);
    const bool armed =
        !state.arm_file.empty() && ::access(state.arm_file.c_str(), F_OK) == 0;
    if (!armed || state.samples >= state.limit || !state.pending.empty() ||
        state.steps++ % state.every != 0) {
      return;
    }
    musaStreamCaptureStatus capture = musaStreamCaptureStatusNone;
    if (musaStreamIsCapturing(stream.get_stream()->stream(), &capture) !=
            musaSuccess ||
        capture != musaStreamCaptureStatusNone) {
      return;
    }
    stream_ = stream.get_stream()->stream();
    current_ = this;
    active_ = true;
    ++state.samples;
    state.metadata = {{"kind", "sampled_spec_runtime_boundaries"},
                      {"pid", ::getpid()},
                      {"sample", state.samples},
                      {"step", state.steps},
                      {"proposal_rows", proposal_rows},
                      {"batch_size", batch_size},
                      {"request_ids", request_ids},
                      {"stream", reinterpret_cast<uintptr_t>(stream_)}};
    state.marks.clear();
    mark_active("step_begin");
#endif
  }

  ~MusaSpecStepTiming() {
#if defined(USE_MUSA)
    if (!active_) {
      return;
    }
    mark_active("step_end");
    State& state = local_state();
    state.pending = std::move(state.marks);
    current_ = nullptr;
    flush(state);
#endif
  }

  MusaSpecStepTiming(const MusaSpecStepTiming&) = delete;
  MusaSpecStepTiming& operator=(const MusaSpecStepTiming&) = delete;

  static void mark_active(const char* name) {
#if defined(USE_MUSA)
    if (current_ == nullptr) {
      return;
    }
    State& state = local_state();
    Mark mark;
    mark.name = name;
    mark.host_ns = now();
    mark.status = static_cast<int32_t>(musaEventCreate(&mark.event));
    if (mark.status == 0) {
      mark.status =
          static_cast<int32_t>(musaEventRecord(mark.event, current_->stream_));
    }
    state.marks.emplace_back(std::move(mark));
#endif
  }

 private:
#if defined(USE_MUSA)
  struct Mark {
    std::string name;
    int64_t host_ns = 0;
    musaEvent_t event = nullptr;
    int32_t status = 0;
  };
  struct State {
    std::string path;
    std::string arm_file;
    uint64_t steps = 0, samples = 0, every = 16, limit = 64;
    nlohmann::json metadata;
    std::vector<Mark> marks, pending;
  };

  static uint64_t positive(const char* key, uint64_t fallback) {
    const char* value = std::getenv(key);
    if (value == nullptr) {
      return fallback;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 && parsed <= 4096
               ? static_cast<uint64_t>(parsed)
               : fallback;
  }

  static State make_state() {
    State state;
    const char* prefix = std::getenv("XLLM_MUSA_SPEC_TIMING_PREFIX");
    const char* arm = std::getenv("XLLM_MUSA_SPEC_TIMING_ARM_FILE");
    if (prefix != nullptr && *prefix && arm != nullptr && *arm) {
      state.path = std::string(prefix) + "." + std::to_string(::getpid()) +
                   "." + std::to_string(reinterpret_cast<uintptr_t>(&tag_)) +
                   ".jsonl";
      state.arm_file = arm;
      state.every = positive("XLLM_MUSA_SPEC_TIMING_EVERY", 16);
      state.limit = positive("XLLM_MUSA_SPEC_TIMING_LIMIT", 64);
      state.marks.reserve(16);
    }
    return state;
  }

  static State& local_state() {
    thread_local State state = make_state();
    return state;
  }

  static void flush(State& state) {
    if (state.pending.empty()) {
      return;
    }
    const Mark& last = state.pending.back();
    if (last.status == 0 && musaEventQuery(last.event) == musaErrorNotReady) {
      return;
    }
    nlohmann::json rows = nlohmann::json::array();
    for (size_t index = 0; index < state.pending.size(); ++index) {
      const Mark& mark = state.pending[index];
      nlohmann::json row = {{"name", mark.name},
                            {"host_ns", mark.host_ns},
                            {"record_status", mark.status}};
      if (index > 0) {
        const Mark& previous = state.pending[index - 1];
        float elapsed_ms = 0.0f;
        const int32_t status =
            previous.status != 0 ? previous.status
            : mark.status != 0   ? mark.status
                                 : static_cast<int32_t>(musaEventElapsedTime(
                                     &elapsed_ms, previous.event, mark.event));
        row["elapsed_status"] = status;
        row["gpu_stream_span_ms"] =
            status == 0 ? nlohmann::json(elapsed_ms) : nlohmann::json(nullptr);
        row["host_span_ms"] = (mark.host_ns - previous.host_ns) / 1e6;
      }
      rows.emplace_back(std::move(row));
    }
    state.metadata["boundaries"] = std::move(rows);
    std::ofstream out(state.path, std::ios::app);
    out << state.metadata.dump() << '\n';
    for (const Mark& mark : state.pending) {
      if (mark.event != nullptr) {
        musaEventDestroy(mark.event);
      }
    }
    state.pending.clear();
  }

  static int64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  inline static thread_local char tag_;
  inline static thread_local MusaSpecStepTiming* current_ = nullptr;
  musaStream_t stream_ = nullptr;
  bool active_ = false;
#endif
};
}  // namespace xllm
