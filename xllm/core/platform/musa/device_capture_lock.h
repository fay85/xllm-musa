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

#include <c10/core/Device.h>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "common/macros.h"

namespace xllm::musa {

// Device-level read-write lock manager for protecting MUSA Graph capture
// operations. Shared locks allow concurrent prepare and replay operations,
// while capture takes exclusive access to the device.
class DeviceCaptureLock final {
 public:
  static DeviceCaptureLock& get_instance() {
    static DeviceCaptureLock instance;
    return instance;
  }

  std::shared_mutex& get_write_lock(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = locks_.find(device_index);
    if (it == locks_.end()) {
      locks_[device_index] = std::make_unique<std::shared_mutex>();
      return *locks_[device_index];
    }
    return *it->second;
  }

  std::shared_mutex& get_read_lock(c10::DeviceIndex device_index) {
    return get_write_lock(device_index);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(DeviceCaptureLock);
  DeviceCaptureLock() = default;
  ~DeviceCaptureLock() = default;

  std::unordered_map<c10::DeviceIndex, std::unique_ptr<std::shared_mutex>>
      locks_;
  std::mutex map_mutex_;
};

}  // namespace xllm::musa
