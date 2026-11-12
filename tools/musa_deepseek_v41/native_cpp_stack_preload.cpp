/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
// Optional profiling-only sidecar. Captures actual native PCs, independently
// of PyTorch's Python/JIT with_stack support. It does not synthesize frames.
#include <ATen/record_function.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/syscall.h>
#include <torch/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string quoted(const std::string& text) {
  std::string result = "\"";
  for (unsigned char ch : text) {
    switch (ch) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (ch < 32) {
          char escaped[7];
          std::snprintf(escaped, sizeof(escaped), "\\u%04x", ch);
          result += escaped;
        } else {
          result += static_cast<char>(ch);
        }
    }
  }
  return result + "\"";
}

class CaptureContext final : public torch::ObserverContext {
 public:
  int64_t begin = now_ns();
  int64_t os_thread = static_cast<int64_t>(::syscall(SYS_gettid));
  uint64_t record_id = 0;
  uint64_t torch_thread = 0;
  std::string name;
  std::vector<void*> frames;
};

class CaptureState final {
 public:
  std::atomic<bool> active{false};
  std::mutex mutex;
  std::ofstream output;
  std::unordered_map<std::string, uint64_t> stacks;
  torch::CallbackHandle callback = torch::INVALID_CALLBACK_HANDLE;
};

CaptureState& state() {
  // Keep callback resources valid through libtorch's process teardown.
  static CaptureState* capture = new CaptureState();
  return *capture;
}

std::unique_ptr<torch::ObserverContext> start_record(
    const torch::RecordFunction& record) {
  if (!state().active.load(std::memory_order_acquire)) {
    return nullptr;
  }
  auto context = std::make_unique<CaptureContext>();
  context->record_id = record.handle();
  context->torch_thread = record.threadId();
  context->name = record.name();
  void* frames[64];
  const int32_t count = ::backtrace(frames, 64);
  context->frames.assign(frames, frames + count);
  return context;
}

void finish_record(const torch::RecordFunction& /*record*/,
                   torch::ObserverContext* observer) {
  const int64_t end = now_ns();
  if (observer == nullptr) {
    return;
  }
  const auto& context = *static_cast<CaptureContext*>(observer);
  CaptureState& capture = state();
  std::lock_guard<std::mutex> lock(capture.mutex);
  if (!capture.active.load(std::memory_order_relaxed)) {
    return;
  }
  const std::string identity(
      reinterpret_cast<const char*>(context.frames.data()),
      context.frames.size() * sizeof(void*));
  auto [entry, inserted] =
      capture.stacks.emplace(identity, capture.stacks.size());
  const uint64_t stack_id = entry->second;
  if (inserted) {
    capture.output << "{\"kind\":\"stack\",\"stack_id\":" << stack_id
                   << ",\"frames\":[";
    bool first = true;
    for (void* frame : context.frames) {
      Dl_info info{};
      const bool found = ::dladdr(frame, &info) != 0;
      const uintptr_t address = reinterpret_cast<uintptr_t>(frame);
      const uintptr_t base =
          found ? reinterpret_cast<uintptr_t>(info.dli_fbase) : 0;
      if (!first) {
        capture.output << ',';
      }
      first = false;
      capture.output << "{\"pc\":" << address << ",\"module_base\":" << base
                     << ",\"module_offset\":" << address - base
                     << ",\"module\":"
                     << quoted(found && info.dli_fname ? info.dli_fname : "")
                     << ",\"symbol\":"
                     << quoted(found && info.dli_sname ? info.dli_sname : "")
                     << '}';
    }
    capture.output << "]}\n";
  }
  capture.output << "{\"kind\":\"record\",\"record_id\":" << context.record_id
                 << ",\"name\":" << quoted(context.name)
                 << ",\"torch_thread\":" << context.torch_thread
                 << ",\"os_thread\":" << context.os_thread
                 << ",\"begin_steady_ns\":" << context.begin
                 << ",\"end_steady_ns\":" << end << ",\"stack_id\":" << stack_id
                 << "}\n";
}

}  // namespace

extern "C" int32_t xllm_cpp_stack_start(const char* directory) noexcept {
  try {
    CaptureState& capture = state();
    std::lock_guard<std::mutex> lock(capture.mutex);
    if (directory == nullptr || capture.active.load()) {
      return -1;
    }
    std::filesystem::create_directories(directory);
    const std::string path = std::string(directory) + "/native_cpp_stacks_" +
                             std::to_string(::getpid()) + ".jsonl";
    capture.output.open(path, std::ios::out | std::ios::trunc);
    if (!capture.output.good()) {
      return -1;
    }
    capture.stacks.clear();
    const int64_t epoch_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    capture.output << "{\"kind\":\"clock_anchor\",\"pid\":" << ::getpid()
                   << ",\"epoch_ns\":" << epoch_ns
                   << ",\"steady_ns\":" << now_ns() << "}\n";
    capture.active.store(true, std::memory_order_release);
    capture.callback = torch::addGlobalCallback(
        torch::RecordFunctionCallback(start_record, finish_record)
            .needsIds(true));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "native C++ stack start failed: %s\n", error.what());
    return -1;
  }
}

extern "C" int32_t xllm_cpp_stack_stop() noexcept {
  try {
    CaptureState& capture = state();
    capture.active.store(false, std::memory_order_release);
    if (capture.callback != torch::INVALID_CALLBACK_HANDLE) {
      torch::removeCallback(capture.callback);
      capture.callback = torch::INVALID_CALLBACK_HANDLE;
    }
    std::lock_guard<std::mutex> lock(capture.mutex);
    capture.output.flush();
    const bool okay = capture.output.good();
    capture.output.close();
    return okay ? 0 : -1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "native C++ stack stop failed: %s\n", error.what());
    return -1;
  }
}
