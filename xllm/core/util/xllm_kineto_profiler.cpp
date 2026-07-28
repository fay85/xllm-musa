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

#include "core/util/xllm_kineto_profiler.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "core/util/env_var.h"

#if defined(USE_CUDA) || defined(USE_MUSA)
#include <ATen/record_function.h>
#if defined(USE_MUSA)
#include <c10/musa/MUSAFunctions.h>
#else
#include <c10/cuda/CUDAFunctions.h>
#endif
#include <kineto/libkineto.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#endif

namespace xllm {
namespace {

#if defined(USE_CUDA) || defined(USE_MUSA)

enum class CapturePhase {
  kIdle = 0,
  kWarmup,
  kTorchCapture,
  kLibkinetoCapture,
  kDone,
};

std::mutex g_mu;
int64_t g_profile_steps = 0;
CapturePhase g_phase = CapturePhase::kIdle;
bool g_libkineto_initialized = false;
bool g_libkineto_prepared = false;
bool g_torch_prepared = false;
bool g_torch_capture_active = false;
bool g_libkineto_capture_active = false;
bool g_child_profiler_joined = false;
int64_t g_capture_window_start_step = 0;

bool profile_prefill_steps() {
  return util::get_bool_env("XLLM_KINETO_PROFILE_PREFILL", false);
}

int64_t warmup_steps() {
  if (profile_prefill_steps()) {
    return util::get_int_env("XLLM_KINETO_WARMUP_PREFILL_STEPS", 1);
  }
  return util::get_int_env("XLLM_KINETO_WARMUP_DECODE_STEPS", 1);
}

int64_t trace_steps() {
  if (profile_prefill_steps()) {
    return util::get_int_env("XLLM_KINETO_TRACE_PREFILL_STEPS", 1);
  }
  return util::get_int_env("XLLM_KINETO_TRACE_DECODE_STEPS", 128);
}

const char* profile_step_label() {
  return profile_prefill_steps() ? "prefill" : "decode";
}

std::string torch_trace_path() {
  const auto path =
      util::get_optional_string_env("XLLM_TORCH_KINETO_TRACE_PATH");
  if (path.has_value() && !path->empty()) {
    return *path;
  }
  return "logs/xllm_torch_kineto_trace.json";
}

std::string libkineto_trace_path() {
  const auto path = util::get_optional_string_env("XLLM_KINETO_TRACE_PATH");
  if (path.has_value() && !path->empty()) {
    return *path;
  }
  return "logs/xllm_libkineto_trace.json";
}

std::string summary_path() {
  const auto path = util::get_optional_string_env("XLLM_KINETO_SUMMARY_PATH");
  if (path.has_value() && !path->empty()) {
    return *path;
  }
  return "logs/xllm_kineto_summary.txt";
}

void ensure_parent_dir(const std::string& path) {
  const std::filesystem::path file_path(path);
  if (file_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(file_path.parent_path(), ec);
  }
}

void sync_gpu() {
#if defined(USE_MUSA)
  c10::musa::device_synchronize();
#else
  c10::cuda::device_synchronize();
#endif
}

torch::profiler::impl::ActivityType gpu_activity_type() {
#if defined(USE_MUSA)
  return torch::profiler::impl::ActivityType::PrivateUse1;
#else
  return torch::profiler::impl::ActivityType::CUDA;
#endif
}

std::set<torch::profiler::impl::ActivityType> default_torch_activities() {
  return {torch::profiler::impl::ActivityType::CPU, gpu_activity_type()};
}

std::set<libkineto::ActivityType> default_libkineto_types() {
  return {libkineto::ActivityType::CONCURRENT_KERNEL,
          libkineto::ActivityType::GPU_MEMCPY,
          libkineto::ActivityType::GPU_MEMSET,
          libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
          libkineto::ActivityType::PRIVATEUSE1_DRIVER,
          libkineto::ActivityType::CPU_OP,
          libkineto::ActivityType::USER_ANNOTATION,
          libkineto::ActivityType::GPU_USER_ANNOTATION,
          libkineto::ActivityType::EXTERNAL_CORRELATION};
}

torch::profiler::impl::ProfilerConfig make_torch_config() {
  return torch::profiler::impl::ProfilerConfig(
      torch::profiler::impl::ProfilerState::KINETO,
      /*report_input_shapes=*/true,
      /*profile_memory=*/false,
      /*with_stack=*/true,
      /*with_flops=*/false,
      /*with_modules=*/false);
}

void ensure_libkineto_initialized() {
  if (g_libkineto_initialized) {
    return;
  }
  libkineto_init(/*cpuOnly=*/false, /*logOnError=*/true);
  libkineto::api().initProfilerIfRegistered();
  g_libkineto_initialized = true;
  LOG(INFO) << "XllmKinetoProfiler: libkineto initialized";
}

void prepare_libkineto_once() {
  if (g_libkineto_prepared) {
    return;
  }
  ensure_libkineto_initialized();
  auto& profiler = libkineto::api().activityProfiler();
  profiler.prepareTrace(default_libkineto_types());
  g_libkineto_prepared = true;
  LOG(INFO) << "XllmKinetoProfiler: libkineto prepareTrace done";
}

void prepare_torch_once() {
  if (g_torch_prepared) {
    return;
  }
  const auto activities = default_torch_activities();
  torch::autograd::profiler::prepareProfiler(make_torch_config(), activities);
  g_torch_prepared = true;
  LOG(INFO) << "XllmKinetoProfiler: torch prepareProfiler done";
}

void join_child_threads_to_profiler() {
  if (g_child_profiler_joined) {
    return;
  }
  if (torch::autograd::profiler::isProfilerEnabledInMainThread()) {
    torch::autograd::profiler::enableProfilerInChildThread();
    g_child_profiler_joined = true;
  }
}

void write_torch_summary(
    const torch::autograd::profiler::ProfilerResult& result,
    const std::string& path) {
  struct OpStat {
    std::string name;
    int64_t total_us = 0;
    int64_t count = 0;
  };

  auto accumulate =
      [](std::vector<OpStat>& stats, const std::string& name, int64_t us) {
        if (us <= 0 || name.empty()) {
          return;
        }
        auto it = std::find_if(stats.begin(),
                               stats.end(),
                               [&](const OpStat& s) { return s.name == name; });
        if (it == stats.end()) {
          stats.push_back({name, us, 1});
        } else {
          it->total_us += us;
          it->count += 1;
        }
      };

  std::vector<OpStat> gpu_stats;
  std::vector<OpStat> cpu_stats;
  for (const auto& event : result.events()) {
    const int64_t duration_us = static_cast<int64_t>(event.durationNs() / 1000);
    if (event.deviceType() == c10::DeviceType::PrivateUse1 ||
        event.deviceType() == c10::DeviceType::CUDA) {
      const int64_t us = event.privateuse1ElapsedUs() > 0
                             ? event.privateuse1ElapsedUs()
                             : event.cudaElapsedUs();
      accumulate(gpu_stats, event.name(), us > 0 ? us : duration_us);
      continue;
    }
    accumulate(cpu_stats, event.name(), duration_us);
  }

  auto sort_stats = [](std::vector<OpStat>& stats) {
    std::sort(stats.begin(), stats.end(), [](const OpStat& a, const OpStat& b) {
      return a.total_us > b.total_us;
    });
  };
  sort_stats(gpu_stats);
  sort_stats(cpu_stats);

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    LOG(WARNING) << "XllmKinetoProfiler: failed to write summary " << path;
    return;
  }
  out << "# xLLM torch.kineto profile summary\n";
  out << "# trace_start_ns=" << result.trace_start_ns() << "\n";
  out << "# events=" << result.events().size() << "\n\n";

  out << "===== top GPU kernels / device ops =====\n";
  const int32_t top_k = 100;
  int32_t rank = 0;
  for (const auto& stat : gpu_stats) {
    if (rank >= top_k) {
      break;
    }
    out << rank + 1 << "\t" << stat.total_us << " us total\t" << stat.count
        << " calls\t" << stat.name << "\n";
    rank += 1;
  }

  out << "\n===== top CPU aten / user scopes =====\n";
  rank = 0;
  for (const auto& stat : cpu_stats) {
    if (rank >= top_k) {
      break;
    }
    out << rank + 1 << "\t" << stat.total_us << " us total\t" << stat.count
        << " calls\t" << stat.name << "\n";
    rank += 1;
  }
  LOG(INFO) << "XllmKinetoProfiler: wrote profile summary to " << path;
}

void stop_torch_capture_and_save() {
  if (!g_torch_capture_active) {
    return;
  }
  sync_gpu();
  auto result = torch::autograd::profiler::disableProfiler();
  g_torch_capture_active = false;
  g_child_profiler_joined = false;
  if (result == nullptr) {
    LOG(ERROR) << "XllmKinetoProfiler: disableProfiler returned null";
    return;
  }
  const std::string trace_path = torch_trace_path();
  ensure_parent_dir(trace_path);
  result->save(trace_path);
  const std::string summary = summary_path();
  ensure_parent_dir(summary);
  write_torch_summary(*result, summary);
  LOG(INFO) << "XllmKinetoProfiler: saved torch kineto trace to " << trace_path
            << " events=" << result->events().size();
}

void stop_libkineto_capture_and_save() {
  if (!g_libkineto_capture_active) {
    return;
  }
  sync_gpu();
  auto& profiler = libkineto::api().activityProfiler();
  auto trace = profiler.stopTrace();
  g_libkineto_capture_active = false;
  if (trace == nullptr) {
    LOG(ERROR) << "XllmKinetoProfiler: libkineto stopTrace returned null";
    return;
  }
  const std::string trace_path = libkineto_trace_path();
  ensure_parent_dir(trace_path);
  trace->save(trace_path);
  LOG(INFO) << "XllmKinetoProfiler: saved libkineto trace to " << trace_path
            << " activities=" << trace->activities()->size();
}

void start_torch_capture() {
  if (g_torch_capture_active) {
    return;
  }
  const auto activities = default_torch_activities();
  torch::autograd::profiler::enableProfiler(make_torch_config(), activities);
  g_torch_capture_active = true;
  LOG(INFO) << "XllmKinetoProfiler: torch capture started at "
            << profile_step_label() << " step " << g_profile_steps;
}

void start_libkineto_capture() {
  if (g_libkineto_capture_active) {
    return;
  }
  prepare_libkineto_once();
  auto& profiler = libkineto::api().activityProfiler();
  profiler.startTrace();
  g_libkineto_capture_active = true;
  g_capture_window_start_step = g_profile_steps;
  LOG(INFO) << "XllmKinetoProfiler: libkineto capture started at "
            << profile_step_label() << " step " << g_profile_steps;
}

void finish_all_profiling() {
  g_phase = CapturePhase::kDone;
  LOG(INFO) << "XllmKinetoProfiler: profiling complete after "
            << g_profile_steps << " " << profile_step_label() << " steps";
}

void on_warmup_complete() {
  // Torch and direct libkineto share one ActivityProfiler; never prepare both
  // at once. Torch prepareProfiler already initializes libkineto internally.
  if (XllmKinetoProfiler::is_torch_kineto_enabled()) {
    prepare_torch_once();
  } else if (XllmKinetoProfiler::is_libkineto_trace_enabled()) {
    prepare_libkineto_once();
  }
  g_phase = CapturePhase::kIdle;
}

void maybe_advance_capture() {
  const int64_t warmup = warmup_steps();
  const int64_t window = trace_steps();
  const bool want_torch = XllmKinetoProfiler::is_torch_kineto_enabled();
  const bool want_libkineto = XllmKinetoProfiler::is_libkineto_trace_enabled();

  if (g_phase == CapturePhase::kDone) {
    return;
  }

  if (g_profile_steps <= warmup) {
    if (g_profile_steps == warmup) {
      on_warmup_complete();
      if (want_torch) {
        g_phase = CapturePhase::kTorchCapture;
        start_torch_capture();
      } else if (want_libkineto) {
        g_phase = CapturePhase::kLibkinetoCapture;
        start_libkineto_capture();
      } else {
        finish_all_profiling();
      }
    }
    return;
  }

  const int64_t steps_after_warmup = g_profile_steps - warmup;

  if (g_phase == CapturePhase::kTorchCapture) {
    if (steps_after_warmup >= window) {
      stop_torch_capture_and_save();
      if (want_libkineto && want_torch) {
        g_libkineto_prepared = false;
        g_phase = CapturePhase::kLibkinetoCapture;
        prepare_libkineto_once();
        start_libkineto_capture();
      } else {
        finish_all_profiling();
      }
    }
    return;
  }

  if (g_phase == CapturePhase::kLibkinetoCapture) {
    const int64_t steps_in_window =
        g_profile_steps - g_capture_window_start_step;
    if (steps_in_window >= window) {
      stop_libkineto_capture_and_save();
      finish_all_profiling();
    }
  }
}

#endif  // defined(USE_CUDA) || defined(USE_MUSA)

}  // namespace

bool XllmKinetoProfiler::is_torch_kineto_enabled() {
#if defined(USE_CUDA) || defined(USE_MUSA)
  return util::get_bool_env("XLLM_ENABLE_TORCH_KINETO_PROFILE", false);
#else
  return false;
#endif
}

bool XllmKinetoProfiler::is_libkineto_trace_enabled() {
#if defined(USE_CUDA) || defined(USE_MUSA)
  return util::get_bool_env("XLLM_ENABLE_KINETO_TRACE", false);
#else
  return false;
#endif
}

bool XllmKinetoProfiler::is_enabled() {
  return is_torch_kineto_enabled() || is_libkineto_trace_enabled();
}

void XllmKinetoProfiler::on_profile_step_begin() {
#if defined(USE_CUDA) || defined(USE_MUSA)
  // Model execution runs on the worker thread that also calls enableProfiler;
  // do not call enableProfilerInChildThread on the same thread.
#endif
}

void XllmKinetoProfiler::on_profile_step_end() {
#if defined(USE_CUDA) || defined(USE_MUSA)
  if (!is_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_phase == CapturePhase::kDone) {
    return;
  }
  g_profile_steps += 1;
  maybe_advance_capture();
#endif
}

XllmKinetoProfiler::StepScope::StepScope(bool is_decode_step,
                                         bool is_prefill_step)
    : is_profile_step_(false) {
#if defined(USE_CUDA) || defined(USE_MUSA)
  is_profile_step_ =
      is_enabled() &&
      (profile_prefill_steps() ? is_prefill_step : is_decode_step);
#else
  (void)is_decode_step;
  (void)is_prefill_step;
#endif
  if (!is_profile_step_) {
    return;
  }
  on_profile_step_begin();
}

XllmKinetoProfiler::StepScope::~StepScope() {
  if (!is_profile_step_) {
    return;
  }
  on_profile_step_end();
}

#if defined(USE_CUDA) || defined(USE_MUSA)
struct XllmKinetoProfiler::UserScope::TorchGuard {
  at::RecordFunction guard;

  explicit TorchGuard(const char* name) : guard(at::RecordScope::USER_SCOPE) {
    if (guard.isActive()) {
      ::at::detail::record_function_with_scope(
          guard, name, c10::ArrayRef<const c10::IValue>{});
    }
  }
};
#endif

XllmKinetoProfiler::UserScope::UserScope(const char* name)
    : name_(name)
#if defined(USE_CUDA) || defined(USE_MUSA)
      ,
      torch_guard_(nullptr)
#endif
{
#if defined(USE_CUDA) || defined(USE_MUSA)
  if (!is_enabled()) {
    return;
  }
  if (g_torch_capture_active ||
      torch::autograd::profiler::isProfilerEnabledInMainThread()) {
    torch_guard_ = new TorchGuard(name);
  }
#endif
}

XllmKinetoProfiler::UserScope::~UserScope() {
#if defined(USE_CUDA) || defined(USE_MUSA)
  delete torch_guard_;
#endif
}

}  // namespace xllm
