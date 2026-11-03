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
  ~MusaGraphHandle() { reset(); }

  MusaGraphHandle(const MusaGraphHandle&) = delete;
  MusaGraphHandle& operator=(const MusaGraphHandle&) = delete;

  void capture_begin(c10::musa::MempoolId_t pool = {0, 0},
                     musaStreamCaptureMode mode = musaStreamCaptureModeGlobal,
                     bool owns_user_pool_ref = false) {
    at::musa::MUSAGraph::capture_begin(pool, mode);
    if (owns_user_pool_ref) {
      pool_ref_.mark_captured();
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
  GraphPoolRef pool_ref_;
};

}  // namespace xllm
