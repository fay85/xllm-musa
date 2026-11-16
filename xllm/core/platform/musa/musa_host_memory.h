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

#include <cstddef>
#include <memory>

namespace xllm::musa {

// Owns a page-aligned MUSA portable pinned-host allocation. The implementation
// records the allocating MUSA context (if any) across moves and restores the
// releasing thread's current context after freeing the allocation.
class MUSAHostMemoryRegion final {
 public:
  MUSAHostMemoryRegion();
  explicit MUSAHostMemoryRegion(size_t bytes);
  MUSAHostMemoryRegion(const MUSAHostMemoryRegion&) = delete;
  MUSAHostMemoryRegion& operator=(const MUSAHostMemoryRegion&) = delete;
  MUSAHostMemoryRegion(MUSAHostMemoryRegion&& other) noexcept;
  MUSAHostMemoryRegion& operator=(MUSAHostMemoryRegion&& other) noexcept;
  ~MUSAHostMemoryRegion();

  void* data() const;
  size_t size() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm::musa
