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

#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSACachingAllocator.h>
#include <torch_musa/csrc/core/MUSAException.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_set>

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
    CHECK(!owns_user_pool_ref || supports_user_pool_refs())
        << "This TorchMUSA allocator cannot verify user MemPool references; "
           "disable graph VMM pools before capture";
    at::musa::MUSAGraph::capture_begin(pool, mode);
    if (owns_user_pool_ref) {
      pool_ref_.mark_captured();
    }
  }

  void capture_end() {
    at::musa::MUSAGraph::capture_end();
    auto& registry = captured_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    capture_thread_ = std::this_thread::get_id();
    registry.graphs.insert(this);
  }

  // Called only on the owning compute thread at a profiler boundary. Native
  // graphs are not registered in torch_musa's Python WeakSet, so the Python
  // profiler's automatic re-instantiation cannot discover them. Refreshing
  // the exec preserves the graph's nodes, addresses, pools and cache contents.
  // Other worker threads' graphs are deliberately outside this method's scope.
  static int64_t reinstantiate_current_thread_graphs() {
    auto& registry = captured_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    int64_t count = 0;
    for (MusaGraphHandle* graph : registry.graphs) {
      if (graph->capture_thread_ != std::this_thread::get_id()) {
        continue;
      }
      torch::DeviceGuard guard(
          torch::Device(torch::kPrivateUse1, graph->capture_dev_));
      C10_MUSA_CHECK(musaDeviceSynchronize());
      graph->reinstantiate_graph();
      ++count;
    }
    return count;
  }

  void reset() {
    auto& registry = captured_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.graphs.erase(this);
    reset_with_allocator(c10::musa::MUSACachingAllocator::get());
  }

  static constexpr bool supports_user_pool_refs() {
    using Allocator =
        std::remove_pointer_t<decltype(c10::musa::MUSACachingAllocator::get())>;
    return allocator_has_pool_use_count<Allocator>();
  }

 private:
  template <typename Allocator>
  static constexpr bool allocator_has_pool_use_count() {
    return requires(Allocator& allocator,
                    c10::DeviceIndex device,
                    c10::musa::MempoolId_t pool) {
      allocator.getPoolUseCount(device, pool);
    };
  }

  template <typename Allocator>
  void reset_with_allocator(Allocator* allocator) {
    if constexpr (allocator_has_pool_use_count<Allocator>()) {
      pool_ref_.reset([this]() { at::musa::MUSAGraph::reset(); },
                      [this, allocator]() {
                        return allocator->getPoolUseCount(capture_dev_,
                                                          mempool_id_);
                      },
                      [this, allocator]() {
                        allocator->releasePool(capture_dev_, mempool_id_);
                      });
    } else {
      // Older TorchMUSA supports ordinary native graph-owned pools, but not
      // the reference-count check needed by our custom VMM allocator.
      // capture_begin rejects user-owned pools before starting capture.
      at::musa::MUSAGraph::reset();
    }
  }

  struct CapturedRegistry {
    std::mutex mutex;
    // Non-owning references: capture_end inserts; reset/destruction erase under
    // the same lock held during refresh, so no reset graph can be visited.
    std::unordered_set<MusaGraphHandle*> graphs;
  };

  static CapturedRegistry& captured_registry() {
    static CapturedRegistry registry;
    return registry;
  }

  std::thread::id capture_thread_;
  GraphPoolRef pool_ref_;
};

}  // namespace xllm
