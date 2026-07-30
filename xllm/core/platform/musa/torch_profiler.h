/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <mutex>
#include <string>

namespace xllm {

// MUSA TorchProfiler: in-process Kineto timeline (CPU + PrivateUse1 / MUPTI).
// Also writes a top-kernel summary (.summary.txt) and gzips the Chrome trace.
// IMPORTANT: start()/stop() must run on the compute thread that executes the
// model forward pass (Kineto CPU capture is thread-local).
class TorchProfiler {
 public:
  static TorchProfiler& get_instance();

  bool start();
  bool stop(const std::string& profile_dir, int32_t rank);
  bool is_running() const;

 private:
  TorchProfiler() = default;
  ~TorchProfiler();
  TorchProfiler(const TorchProfiler&) = delete;
  TorchProfiler& operator=(const TorchProfiler&) = delete;

  mutable std::mutex mutex_;
  bool running_ = false;
};

}  // namespace xllm