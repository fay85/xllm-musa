/* Copyright 2025-2026 The xLLM Authors.

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

#include "platform/torch_profiler.h"

#include <dlfcn.h>
#include <glog/logging.h>
#include <torch/csrc/autograd/profiler.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <set>

namespace xllm {
namespace {

namespace tp = torch::profiler::impl;
namespace ap = torch::autograd::profiler;

using NativeStackStart = int32_t (*)(const char*);
using NativeStackStop = int32_t (*)();

NativeStackStop native_stack_stop() {
  return reinterpret_cast<NativeStackStop>(
      ::dlsym(RTLD_DEFAULT, "xllm_cpp_stack_stop"));
}

tp::ActivityType gpu_activity_type() {
#if defined(USE_MUSA)
  return tp::ActivityType::PrivateUse1;
#else
  return tp::ActivityType::CUDA;
#endif
}

// Build the destination path for a worker's Chrome trace. Mirrors the
// ".pt.trace.json" suffix produced by torch.profiler's tensorboard handler so
// the file is recognized by the usual viewers (Perfetto, chrome://tracing,
// TensorBoard).
std::string make_trace_path(const std::string& profile_dir, int32_t rank) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto ts =
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const std::string file_name = "xllm_rank" + std::to_string(rank) + "_" +
                                std::to_string(::getpid()) + "_" +
                                std::to_string(ts) + ".pt.trace.json";
  if (profile_dir.empty()) {
    return file_name;
  }
  return (std::filesystem::path(profile_dir) / file_name).string();
}

}  // namespace

TorchProfiler& TorchProfiler::get_instance() {
  static TorchProfiler instance;
  return instance;
}

TorchProfiler::~TorchProfiler() {
  if (running_) {
    // Best-effort: drop the trace rather than write it during teardown.
    try {
      if (native_stack_running_) {
        const NativeStackStop stop = native_stack_stop();
        if (stop != nullptr) {
          stop();
        }
        native_stack_running_ = false;
      }
      ap::disableProfiler();
    } catch (const std::exception& e) {
      LOG(WARNING) << "disableProfiler during shutdown failed: " << e.what();
    }
    running_ = false;
  }
}

bool TorchProfiler::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    LOG(WARNING)
        << "Torch profiler is already running, ignoring start request.";
    return true;
  }

  const char* stack_directory = std::getenv("XLLM_CPP_STACK_DIR");
  NativeStackStart stack_start = nullptr;
  if (stack_directory != nullptr && stack_directory[0] != '\0') {
    stack_start = reinterpret_cast<NativeStackStart>(
        ::dlsym(RTLD_DEFAULT, "xllm_cpp_stack_start"));
    if (stack_start == nullptr || native_stack_stop() == nullptr) {
      LOG(ERROR)
          << "XLLM_CPP_STACK_DIR requires the native-stack preload helper";
      return false;
    }
  }
  try {
    const std::set<tp::ActivityType> activities = {tp::ActivityType::CPU,
                                                   gpu_activity_type()};
    const tp::ProfilerConfig config(tp::ProfilerState::KINETO,
                                    /*report_input_shapes=*/false,
                                    /*profile_memory=*/false,
                                    /*with_stack=*/true,
                                    /*with_flops=*/false,
                                    /*with_modules=*/false);
    ap::prepareProfiler(config, activities);
    ap::enableProfiler(config, activities);
    if (stack_start != nullptr) {
      if (stack_start(stack_directory) != 0) {
        ap::disableProfiler();
        LOG(ERROR) << "Native-stack collector did not start; profile rejected";
        return false;
      }
      native_stack_running_ = true;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to start torch profiler: " << e.what();
    return false;
  }

  running_ = true;
  LOG(INFO) << "Torch profiler started (in-process Kineto). The trace will be "
               "written on /stop_profile; no external profiler is required.";
  return true;
}

bool TorchProfiler::stop(const std::string& profile_dir, int32_t rank) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    LOG(WARNING) << "Torch profiler is not running, ignoring stop request.";
    return true;
  }
  running_ = false;

  const std::string path = make_trace_path(profile_dir, rank);
  bool stack_saved = true;
  try {
    if (!profile_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(profile_dir, ec);
      if (ec) {
        LOG(WARNING) << "Failed to create profile dir '" << profile_dir
                     << "': " << ec.message();
      }
    }
    if (native_stack_running_) {
      const NativeStackStop stop = native_stack_stop();
      stack_saved = stop != nullptr && stop() == 0;
      native_stack_running_ = false;
      if (!stack_saved) {
        LOG(ERROR) << "Native-stack sidecar did not finish; profile incomplete";
      }
    }
    auto result = ap::disableProfiler();
    if (result == nullptr) {
      LOG(ERROR) << "disableProfiler returned no result; no trace written.";
      return false;
    }
    result->save(path);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to stop torch profiler / save trace: " << e.what();
    return false;
  }

  LOG(INFO) << "Torch profiler stopped. Trace written to: "
            << std::filesystem::absolute(path).string();
  return stack_saved;
}

bool TorchProfiler::is_running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}

}  // namespace xllm
