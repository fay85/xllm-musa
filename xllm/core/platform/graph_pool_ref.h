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

#include <glog/logging.h>

#include <cstdint>
#include <utility>

namespace xllm {

// A user MemPool owns one reference independently of its captured graphs.
// Supported SDKs either release a graph's reference in reset(), or leave it
// outstanding. Reconcile exactly that one reference, never drain other owners.
// The pool must remain alive and its graph resets must be serialized.
class GraphPoolRef final {
 public:
  GraphPoolRef() = default;
  GraphPoolRef(const GraphPoolRef&) = delete;
  GraphPoolRef& operator=(const GraphPoolRef&) = delete;

  void mark_captured() {
    CHECK(!active_) << "A captured graph must be reset before recapture";
    active_ = true;
  }

  template <typename NativeReset, typename UseCount, typename Release>
  void reset(NativeReset&& native_reset,
             UseCount&& use_count,
             Release&& release) {
    if (!active_) {
      std::forward<NativeReset>(native_reset)();
      return;
    }

    const int64_t before = std::forward<UseCount>(use_count)();
    CHECK_GT(before, 1) << "Captured graph lost its user MemPool owner";
    std::forward<NativeReset>(native_reset)();
    const int64_t after = use_count();
    if (after == before) {
      // Compatibility with TorchMUSA builds whose native reset omits
      // releasePool(). Fixed SDKs take the other, already-balanced branch.
      std::forward<Release>(release)();
    } else {
      CHECK_EQ(after, before - 1)
          << "Native graph reset changed unrelated MemPool references";
    }
    CHECK_EQ(use_count(), before - 1)
        << "Graph reset did not release exactly one MemPool reference";
    active_ = false;
  }

 private:
  bool active_ = false;
};

}  // namespace xllm
