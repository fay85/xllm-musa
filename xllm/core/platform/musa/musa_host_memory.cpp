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

#include "platform/musa/musa_host_memory.h"

#include <glog/logging.h>
#include <musa.h>
#include <musa_runtime.h>
#include <unistd.h>

#include <cstdint>
#include <limits>
#include <string>

namespace xllm::musa {
namespace {

std::string mu_error_text(MUresult result) {
  const char* text = nullptr;
  const MUresult text_result = muGetErrorString(result, &text);
  if (text_result != MUSA_SUCCESS || text == nullptr) {
    return "unknown MUSA driver error";
  }
  return text;
}

size_t page_aligned_bytes(size_t bytes) {
  const int64_t system_page_size = sysconf(_SC_PAGESIZE);
  CHECK_GT(system_page_size, 0) << "Failed to query system page size.";
  const size_t page_size = static_cast<size_t>(system_page_size);
  CHECK_LE(bytes, std::numeric_limits<size_t>::max() - (page_size - 1))
      << "MUSA host region byte count overflows page rounding: bytes=" << bytes
      << ", page_size=" << page_size;
  return ((bytes + page_size - 1) / page_size) * page_size;
}

}  // namespace

class MUSAHostMemoryRegion::Impl final {
 public:
  explicit Impl(size_t bytes) : size_(page_aligned_bytes(bytes)) {
    const MUresult ctx_result = muCtxGetCurrent(&owner_context_);
    if (ctx_result != MUSA_SUCCESS) {
      owner_context_ = nullptr;
    }

    const musaError_t alloc_result =
        musaHostAlloc(&alloc_ptr_, size_, musaHostAllocPortable);
    CHECK_EQ(alloc_result, musaSuccess)
        << "musaHostAlloc failed: result=" << static_cast<int32_t>(alloc_result)
        << ", error=" << musaGetErrorString(alloc_result)
        << ", bytes=" << size_;

    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (reinterpret_cast<uintptr_t>(alloc_ptr_) % page_size == 0) {
      data_ = alloc_ptr_;
    } else {
      // MUSA's internal pinned slab allocator may sub-allocate small buffers at
      // sub-page alignment when an earlier allocation was not page-sized.
      // Re-allocate with one extra page of padding and align data_ up while
      // retaining alloc_ptr_ for musaFreeHost.
      const musaError_t free_result = musaFreeHost(alloc_ptr_);
      if (free_result != musaSuccess) {
        LOG(ERROR) << "musaFreeHost for unaligned allocation failed: result="
                   << static_cast<int32_t>(free_result)
                   << ", error=" << musaGetErrorString(free_result);
      }
      alloc_ptr_ = nullptr;
      CHECK_LE(size_, std::numeric_limits<size_t>::max() - page_size)
          << "MUSA host region padded byte count overflows: size_=" << size_;
      const size_t padded_size = size_ + page_size;
      const musaError_t padded_result =
          musaHostAlloc(&alloc_ptr_, padded_size, musaHostAllocPortable);
      CHECK_EQ(padded_result, musaSuccess)
          << "musaHostAlloc (padded) failed: result="
          << static_cast<int32_t>(padded_result)
          << ", error=" << musaGetErrorString(padded_result)
          << ", bytes=" << padded_size;
      const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(alloc_ptr_);
      const uintptr_t aligned_addr =
          (raw_addr + page_size - 1) & ~(page_size - 1);
      data_ = reinterpret_cast<void*>(aligned_addr);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  ~Impl() { release(); }

  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void release() {
    if (alloc_ptr_ == nullptr || size_ == 0) {
      alloc_ptr_ = nullptr;
      data_ = nullptr;
      size_ = 0;
      owner_context_ = nullptr;
      return;
    }

    MUcontext previous_context = nullptr;
    const MUresult get_result = muCtxGetCurrent(&previous_context);
    bool switched_context = false;
    if (owner_context_ != nullptr && get_result == MUSA_SUCCESS &&
        previous_context != owner_context_) {
      const MUresult set_result = muCtxSetCurrent(owner_context_);
      if (set_result != MUSA_SUCCESS) {
        LOG(ERROR) << "muCtxSetCurrent before musaFreeHost failed: result="
                   << static_cast<int32_t>(set_result)
                   << ", error=" << mu_error_text(set_result);
      } else {
        switched_context = true;
      }
    }

    const musaError_t free_result = musaFreeHost(alloc_ptr_);
    if (free_result != musaSuccess) {
      LOG(ERROR) << "musaFreeHost failed: result="
                 << static_cast<int32_t>(free_result)
                 << ", error=" << musaGetErrorString(free_result)
                 << ", bytes=" << size_ << ", ptr=" << alloc_ptr_;
    }

    if (switched_context) {
      const MUresult restore_result = muCtxSetCurrent(previous_context);
      if (restore_result != MUSA_SUCCESS) {
        LOG(ERROR) << "muCtxSetCurrent restore failed: result="
                   << static_cast<int32_t>(restore_result)
                   << ", error=" << mu_error_text(restore_result);
      }
    }

    alloc_ptr_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    owner_context_ = nullptr;
  }

  void* alloc_ptr_ = nullptr;
  void* data_ = nullptr;
  size_t size_ = 0;
  MUcontext owner_context_ = nullptr;
};

MUSAHostMemoryRegion::MUSAHostMemoryRegion() = default;

MUSAHostMemoryRegion::MUSAHostMemoryRegion(size_t bytes) {
  if (bytes > 0) {
    impl_ = std::make_unique<Impl>(bytes);
  }
}

MUSAHostMemoryRegion::MUSAHostMemoryRegion(
    MUSAHostMemoryRegion&& other) noexcept = default;

MUSAHostMemoryRegion& MUSAHostMemoryRegion::operator=(
    MUSAHostMemoryRegion&& other) noexcept = default;

MUSAHostMemoryRegion::~MUSAHostMemoryRegion() = default;

void* MUSAHostMemoryRegion::data() const {
  return impl_ == nullptr ? nullptr : impl_->data();
}

size_t MUSAHostMemoryRegion::size() const {
  return impl_ == nullptr ? 0 : impl_->size();
}

}  // namespace xllm::musa
