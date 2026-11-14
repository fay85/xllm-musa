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

#pragma once

#include <torch_musa/csrc/core/MUSACachingAllocator.h>
#include <torch_musa/csrc/core/MUSAStream.h>

// The installed native MUSAGraph header expects this stream alias.
namespace at::musa {
using c10::musa::MUSAStream;
}  // namespace at::musa

#include <torch_musa/csrc/aten/musa/MUSAGraph.h>

#include "core/platform/graph_pool_ref.h"

namespace xllm {

// All instances are owned through this exact type, never through a native
// MUSAGraph pointer (its destructor is not virtual).
class MusaGraphHandle final : public at::musa::MUSAGraph {
 public:
  MusaGraphHandle() = default;
  ~MusaGraphHandle() { abort_capture(); }

  MusaGraphHandle(const MusaGraphHandle&) = delete;
  MusaGraphHandle& operator=(const MusaGraphHandle&) = delete;

  void capture_begin(c10::musa::MempoolId_t pool = {0, 0},
                     musaStreamCaptureMode mode = musaStreamCaptureModeGlobal,
                     bool owns_user_pool_ref = false) {
    try {
      at::musa::MUSAGraph::capture_begin(pool, mode);
    } catch (...) {
      clear_generator_states_noexcept();
      (void)musaGetLastError();
      throw;
    }
    capture_started_ = true;
    if (owns_user_pool_ref) {
      pool_ref_.mark_captured();
    }
  }

  void capture_end() {
    at::musa::MUSAGraph::capture_end();
    capture_started_ = false;
  }

  void abort_capture() noexcept {
    try {
      if (capture_started_) {
        musaStreamCaptureStatus status = musaStreamCaptureStatusNone;
        if (musaStreamIsCapturing(capture_stream_.stream(), &status) ==
                musaSuccess &&
            status != musaStreamCaptureStatusNone) {
          musaGraph_t aborted_graph = nullptr;
          (void)musaStreamEndCapture(capture_stream_.stream(), &aborted_graph);
          if (aborted_graph != nullptr) {
            (void)musaGraphDestroy(aborted_graph);
          }
          (void)musaGetLastError();
          c10::musa::MUSACachingAllocator::endAllocateToPool(capture_dev_,
                                                             mempool_id_);
          for (auto& kv : captured_generator_states_) {
            if (kv.first) {
              try {
                kv.first->capture_epilogue();
              } catch (...) {
              }
            }
          }
        } else if (has_graph_ && !has_graph_exec_) {
          (void)musaGetLastError();
          for (auto& kv : captured_generator_states_) {
            if (kv.first) {
              try {
                kv.first->capture_epilogue();
              } catch (...) {
              }
            }
          }
        }
        clear_generator_states_noexcept();
        capture_started_ = false;
      }
      reset();
    } catch (...) {
      clear_generator_states_noexcept();
      capture_started_ = false;
      (void)musaGetLastError();
    }
  }

  void reset() {
    pool_ref_.reset([this]() { at::musa::MUSAGraph::reset(); },
                    [this]() {
                      return c10::musa::MUSACachingAllocator::getPoolUseCount(
                          capture_dev_, mempool_id_);
                    },
                    [this]() {
                      c10::musa::MUSACachingAllocator::releasePool(capture_dev_,
                                                                   mempool_id_);
                    });
  }

 private:
  void clear_generator_states_noexcept() noexcept {
    for (auto& kv : captured_generator_states_) {
      if (kv.first) {
        try {
          kv.first->unregister_graph(this);
        } catch (...) {
        }
      }
    }
    captured_generator_states_.clear();
  }

  GraphPoolRef pool_ref_;
  bool capture_started_ = false;
};

}  // namespace xllm
