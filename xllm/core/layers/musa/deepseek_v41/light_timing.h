/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(USE_MUSA)
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#endif

namespace xllm::layer {

// Diagnostic host wall-clock scopes without backtrace, Kineto or
// implicit device synchronization. Optional device events measure stream
// spans, not kernel busy time. Enabled only in a separate calibration run.
class DeepseekV41LightTiming final {
 public:
  DeepseekV41LightTiming(std::string_view name,
                         int64_t layer,
                         int64_t tokens,
                         int64_t context,
                         int64_t payload_bytes = -1,
                         int64_t group_size = -1) {
    State& s = state();
    if (s.path.empty()) {
      return;
    }
    root_ = name == "V41/step";
    if (root_) {
      // Every thread owns its sampling and buffer; do not cross request steps.
      flush(s);  // query only: never synchronize the device to collect data.
      s.selected = (s.steps++ % s.every == 0) && s.samples < s.limit &&
                   s.pending.empty();
      if (s.selected) {
        ++s.samples;
        s.rows.clear();
      }
    }
    if (!s.selected || name.substr(0, 4) != "V41/") {
      return;
    }
    active_ = true;
    begin_ = now();
    index_ = s.rows.size();
    depth_ = s.depth++;
    s.rows.push_back({std::string(name),
                      layer,
                      tokens,
                      context,
                      begin_,
                      0,
                      depth_,
                      payload_bytes,
                      group_size});
#if defined(USE_MUSA)
    if (s.device_events) {
      Row& row = s.rows.back();
      row.event_status = static_cast<int32_t>(musaEventCreate(&row.gpu_begin));
      if (!row.event_status) {
        row.event_status = static_cast<int32_t>(musaEventCreate(&row.gpu_end));
      }
      if (!row.event_status) {
        row.event_status = static_cast<int32_t>(
            musaEventRecord(row.gpu_begin, c10::musa::getCurrentMUSAStream()));
      }
    }
#endif
  }

  ~DeepseekV41LightTiming() noexcept {
    if (!active_) {
      return;
    }
    State& s = state();
    Row& row = s.rows[index_];
#if defined(USE_MUSA)
    if (s.device_events && !row.event_status) {
      row.event_status = static_cast<int32_t>(
          musaEventRecord(row.gpu_end, c10::musa::getCurrentMUSAStream()));
    }
#endif
    row.end = now();
    --s.depth;
    if (!root_) {
      return;
    }
    s.pending = std::move(s.rows);
    s.selected = false;
    flush(s);
  }

  DeepseekV41LightTiming(const DeepseekV41LightTiming&) = delete;
  DeepseekV41LightTiming& operator=(const DeepseekV41LightTiming&) = delete;

 private:
  struct Row {
    std::string name;
    int64_t layer, tokens, context, begin, end;
    uint64_t depth;
    int64_t payload_bytes, group_size;
#if defined(USE_MUSA)
    musaEvent_t gpu_begin = nullptr, gpu_end = nullptr;
    int32_t event_status = 0;
#endif
  };
  static uint64_t positive(const char* key, uint64_t fallback) {
    const char* p = std::getenv(key);
    if (!p) {
      return fallback;
    }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(p, &end, 10);
    return end != p && *end == '\0' && value > 0 && value <= 4096
               ? static_cast<uint64_t>(value)
               : fallback;
  }
  struct State {
    std::string path;
    uint64_t every = positive("XLLM_V41_LIGHT_EVERY", 32);
    uint64_t limit = positive("XLLM_V41_LIGHT_LIMIT", 16);
    uint64_t steps = 0, samples = 0, depth = 0;
    bool selected = false;
    std::vector<Row> rows, pending;
    bool device_events = std::getenv("XLLM_V41_LIGHT_DEVICE_EVENTS") != nullptr;
  };
  static State make_state() {
    State s;
    const char* prefix = std::getenv("XLLM_V41_LIGHT_PREFIX");
    if (prefix && *prefix) {
      s.path = std::string(prefix) + "." + std::to_string(::getpid()) + "." +
               std::to_string(reinterpret_cast<uintptr_t>(&state_thread_id_)) +
               ".jsonl";
      s.rows.reserve(2048);
    }
    return s;
  }

  static void flush(State& s) noexcept {
    if (s.pending.empty()) {
      return;
    }
#if defined(USE_MUSA)
    // The root end is ordered after child scope ends on the model stream.
    // Keep at most one pending sample; skip new samples until it is ready.
    const Row& root = s.pending.front();
    if (s.device_events && !root.event_status) {
      const musaError_t ready = musaEventQuery(root.gpu_end);
      if (ready == musaErrorNotReady) {
        return;
      }
    }
#endif
    std::ofstream out(s.path, std::ios::app);
    for (Row& r : s.pending) {
      out << "{\"kind\":\"host_wall_scope\",\"pid\":" << ::getpid()
          << ",\"thread\":\"" << static_cast<const void*>(&s) << "\""
          << ",\"sample\":" << s.samples << ",\"step\":" << s.steps
          << ",\"name\":\"" << r.name << "\",\"layer\":" << r.layer
          << ",\"q\":" << r.tokens << ",\"kv\":" << r.context
          << ",\"begin_ns\":" << r.begin << ",\"end_ns\":" << r.end
          << ",\"depth\":" << r.depth
          << ",\"payload_bytes\":" << r.payload_bytes
          << ",\"group_size\":" << r.group_size;
#if defined(USE_MUSA)
      if (s.device_events) {
        float ms = 0.0f;
        int32_t status = r.event_status;
        if (!status) {
          status = static_cast<int32_t>(
              musaEventElapsedTime(&ms, r.gpu_begin, r.gpu_end));
        }
        out << ",\"device_event_status\":" << status
            << ",\"gpu_stream_span_ms\":";
        if (!status) {
          out << ms;
        } else {
          out << "null";
        }
      }
      if (r.gpu_begin) {
        musaEventDestroy(r.gpu_begin);
      }
      if (r.gpu_end) {
        musaEventDestroy(r.gpu_end);
      }
#endif
      out << "}\n";
    }
    s.pending.clear();
  }
  inline static thread_local char state_thread_id_;
  static State& state() {
    thread_local State s = make_state();
    return s;
  }
  static int64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  bool active_ = false, root_ = false;
  int64_t begin_ = 0;
  size_t index_ = 0;
  uint64_t depth_ = 0;
};
}  // namespace xllm::layer
