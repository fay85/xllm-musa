/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>

namespace xllm {

// In-process Kineto profiling for xLLM decode or prefill steps.
//
// Option B (recommended): XLLM_ENABLE_TORCH_KINETO_PROFILE=1
//   Uses libtorch Kineto profiler (aten ops + GPU kernels in one Chrome trace).
//
// Option A: XLLM_ENABLE_KINETO_TRACE=1
//   Direct libkineto API (GPU timeline; useful when torch profiler is off).
//
// When both are set, torch capture runs first, then a standalone libkineto
// capture on the next selected step window (two JSON files).
//
// Common env:
//   XLLM_KINETO_WARMUP_DECODE_STEPS=1
//   XLLM_KINETO_TRACE_DECODE_STEPS=128
//   XLLM_KINETO_PROFILE_PREFILL=1
//   XLLM_KINETO_WARMUP_PREFILL_STEPS=1
//   XLLM_KINETO_TRACE_PREFILL_STEPS=1
//   XLLM_TORCH_KINETO_TRACE_PATH=logs/xllm_torch_kineto_trace.json
//   XLLM_KINETO_TRACE_PATH=logs/xllm_libkineto_trace.json
//   XLLM_KINETO_SUMMARY_PATH=logs/xllm_kineto_summary.txt
class XllmKinetoProfiler {
 public:
  static bool is_torch_kineto_enabled();
  static bool is_libkineto_trace_enabled();
  static bool is_enabled();

  static void on_profile_step_begin();
  static void on_profile_step_end();

  class StepScope {
   public:
    StepScope(bool is_decode_step, bool is_prefill_step);
    ~StepScope();

    StepScope(const StepScope&) = delete;
    StepScope& operator=(const StepScope&) = delete;

   private:
    bool is_profile_step_;
  };

  class UserScope {
   public:
    explicit UserScope(const char* name);
    ~UserScope();

    UserScope(const UserScope&) = delete;
    UserScope& operator=(const UserScope&) = delete;

   private:
    const char* name_;
#if defined(USE_CUDA) || defined(USE_MUSA)
    struct TorchGuard;
    TorchGuard* torch_guard_;
#endif
  };
};

#define XLLM_KINETO_USER_SCOPE(name) \
  xllm::XllmKinetoProfiler::UserScope _xllm_kineto_user_scope(name)

}  // namespace xllm
