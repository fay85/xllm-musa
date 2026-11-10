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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "framework/request/request.h"

namespace xllm {

class DPRequestLoad final {
 public:
  explicit DPRequestLoad(size_t dp_size)
      : waiting_(dp_size, 0), running_(dp_size, 0) {}

  const std::vector<size_t>& waiting() const { return waiting_; }
  const std::vector<size_t>& running() const { return running_; }

  void track_waiting(const std::shared_ptr<Request>& request) {
    if (!request || request->finished() || request->cancelled() ||
        request->sequences().empty()) {
      return;
    }
    const auto& sequence = request->sequences().front();
    const int32_t rank = sequence->dp_rank();
    if (rank < 0 || static_cast<size_t>(rank) >= waiting_.size()) {
      return;
    }
    if (entries_.emplace(request.get(), Entry{rank, false}).second) {
      ++waiting_[rank];
    }
  }

  void mark_running(const std::shared_ptr<Request>& request) {
    track_waiting(request);
    transition(request, true);
  }

  void mark_waiting(const std::shared_ptr<Request>& request) {
    transition(request, false);
  }

  void remove(const std::shared_ptr<Request>& request) {
    const auto iter = entries_.find(request.get());
    if (iter != entries_.end()) {
      --(iter->second.running ? running_[iter->second.rank]
                              : waiting_[iter->second.rank]);
      entries_.erase(iter);
    }
  }

 private:
  void transition(const std::shared_ptr<Request>& request, bool running) {
    const auto iter = entries_.find(request.get());
    if (iter == entries_.end() || iter->second.running == running) {
      return;
    }
    const int32_t rank = iter->second.rank;
    --(running ? waiting_[rank] : running_[rank]);
    ++(running ? running_[rank] : waiting_[rank]);
    iter->second.running = running;
  }

  struct Entry {
    int32_t rank;
    bool running;
  };
  std::unordered_map<const Request*, Entry> entries_;
  std::vector<size_t> waiting_;
  std::vector<size_t> running_;
};

}  // namespace xllm
