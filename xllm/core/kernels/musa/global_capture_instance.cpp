/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kernels/musa/global_capture_instance.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/kernels/musa/attention_runner.h"
#include "core/kernels/musa/piecewise_graphs.h"

namespace xllm::runtime::musa {

std::mutex GlobalCaptureInstance::capture_mutex_;

GlobalCaptureInstance& GlobalCaptureInstance::get_instance() {
  thread_local GlobalCaptureInstance instance;
  return instance;
}

GlobalCaptureInstance::GlobalCaptureInstance() = default;
GlobalCaptureInstance::~GlobalCaptureInstance() = default;

void GlobalCaptureInstance::begin_capture(const MusaMempoolId& pool) {
  CHECK(!is_capturing_) << "Already capturing, call end_capture() first";

  capture_lock_ = std::make_unique<std::lock_guard<std::mutex>>(capture_mutex_);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "MusaGlobalCaptureInstance::begin_capture()";

  is_capturing_ = true;
  graph_pool_ = pool;

  current_piecewise_graph_ = std::make_unique<PiecewiseGraphs>();

  current_graph_ = std::make_unique<MusaGraphSegment>();
  current_graph_->capture_begin(pool, musaStreamCaptureModeThreadLocal);
}

std::unique_ptr<PiecewiseGraphs> GlobalCaptureInstance::end_capture() {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  CHECK(current_graph_)
      << "Current graph is null, cannot end without active graph";

  current_graph_->capture_end();
  current_piecewise_graph_->add_graph(std::move(current_graph_));

  is_capturing_ = false;

  VLOG(kGraphExecutorLogVerboseLevel)
      << "MusaGlobalCaptureInstance::end_capture(), total graphs: "
      << current_piecewise_graph_->size()
      << ", total runners: " << current_piecewise_graph_->num_runners();

  std::unique_ptr<PiecewiseGraphs> result = std::move(current_piecewise_graph_);

  capture_lock_.reset();

  return result;
}

void GlobalCaptureInstance::temporarily_end_graph() {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  temporarily_end_graph_locked();
}

void GlobalCaptureInstance::temporarily_begin_graph() {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  temporarily_begin_graph_locked();
}

void GlobalCaptureInstance::temporarily_end_graph_locked() {
  CHECK(current_graph_) << "Current graph is null, cannot end";
  CHECK(current_piecewise_graph_) << "Current piecewise graph is null";

  current_graph_->capture_end();
  current_piecewise_graph_->add_graph(std::move(current_graph_));

  VLOG(kGraphExecutorLogVerboseLevel)
      << "MusaGlobalCaptureInstance::temporarily_end_graph(), total graphs: "
      << current_piecewise_graph_->size();
}

void GlobalCaptureInstance::temporarily_begin_graph_locked() {
  CHECK(!current_graph_) << "Current graph already exists";

  current_graph_ = std::make_unique<MusaGraphSegment>();
  current_graph_->capture_begin(graph_pool_, musaStreamCaptureModeThreadLocal);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "MusaGlobalCaptureInstance::temporarily_begin_graph()";
}

void GlobalCaptureInstance::register_attention_runner(
    ::xllm::kernel::musa::AttentionRunner&& runner) {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  CHECK(current_piecewise_graph_) << "Current piecewise graph is null";

  current_piecewise_graph_->add_attention_runner(std::move(runner));

  VLOG(kGraphExecutorLogVerboseLevel)
      << "MusaGlobalCaptureInstance::register_attention_runner(), total "
         "runners: "
      << current_piecewise_graph_->num_runners();
}

}  // namespace xllm::runtime::musa
