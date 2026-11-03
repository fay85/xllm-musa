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

#include <torch_musa/csrc/core/MUSAGuard.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "core/platform/shared_vmm_allocator.h"
#include "core/platform/vmm_torch_allocator.h"

namespace xllm {

// Returned logical pools must be destroyed before this physical pool owner.
// Graph cache hits do not create a pool; every new capture calls create_pool().
class MusaVmmCapturePool final {
 public:
  explicit MusaVmmCapturePool(int32_t device_index, size_t reserve_size = 0)
      : device_index_(device_index), torch_allocator_(&allocator_) {
    c10::musa::MUSAGuard guard(device_index_);
    allocator_.init(device_index_, reserve_size);
  }

  MusaVmmCapturePool(const MusaVmmCapturePool&) = delete;
  MusaVmmCapturePool& operator=(const MusaVmmCapturePool&) = delete;

  std::unique_ptr<c10::musa::MemPool> create_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    c10::musa::MUSAGuard guard(device_index_);
    if (captures_created_ > 0) {
      allocator_.switch_to_new_virtual_space();
    }
    auto pool = std::make_unique<c10::musa::MemPool>(&torch_allocator_,
                                                     /*is_user_created=*/true);
    ++captures_created_;
    return pool;
  }

  size_t mapped_size() const { return allocator_.mapped_size(); }
  size_t current_offset() const { return allocator_.current_offset(); }
  size_t high_water_mark() const { return allocator_.high_water_mark(); }

 private:
  const int32_t device_index_;
  SharedVMMAllocator allocator_;
  VMMTorchAllocator torch_allocator_;
  size_t captures_created_ = 0;
  std::mutex mutex_;
};

}  // namespace xllm
