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

#include "shared_vmm_allocator.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "common/global_flags.h"
#include "core/util/env_var.h"
#include "device.h"

namespace xllm {

namespace {

template <typename PtrT>
inline PtrT add_offset(PtrT base, size_t offset_bytes) {
  if constexpr (std::is_pointer_v<PtrT>) {
    auto* base_bytes = reinterpret_cast<std::uint8_t*>(base);
    return reinterpret_cast<PtrT>(base_bytes + offset_bytes);
  } else {
    return static_cast<PtrT>(static_cast<std::uintptr_t>(base) + offset_bytes);
  }
}

template <typename PtrT>
inline void* to_void_ptr(PtrT ptr) {
  if constexpr (std::is_pointer_v<PtrT>) {
    return ptr;
  } else {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr));
  }
}

inline size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

SharedVMMAllocator::~SharedVMMAllocator() {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Unmap and release all virtual address spaces
  for (auto& space : virtual_spaces_) {
    if (space.mapped_size > 0) {
      vmm::unmap(space.base_ptr, space.mapped_size);
    }
    if (space.reserved_size > 0) {
      vmm::release_vir_ptr(space.base_ptr, space.reserved_size);
    }
  }
  virtual_spaces_.clear();

  // Release all physical memory handles
  for (auto& handle : handles_) {
    vmm::release_phy_mem_handle(handle);
  }
  handles_.clear();

  initialized_ = false;
  VLOG(20) << "SharedVMMAllocator destroyed. High water mark: "
           << high_water_mark_ << " bytes, total virtual spaces created: "
           << virtual_spaces_.size();
}

void SharedVMMAllocator::init(int32_t device_id, size_t reserve_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    LOG(WARNING) << "SharedVMMAllocator already initialized";
    return;
  }

  device_id_ = device_id;

#if defined(USE_DCU)
  CHECK_EQ(hipSetDevice(device_id_), hipSuccess)
      << "hipSetDevice failed for device " << device_id_;
  CHECK_EQ(hipFree(nullptr), hipSuccess)
      << "hipFree(nullptr) failed when initializing HIP context";
#endif

  granularity_ = vmm::get_recommended_granularity(device_id_);
  CHECK_GT(granularity_, 0u) << "Invalid VMM granularity size";

  if (reserve_size == 0) {
    Device device(device_id);
    int64_t total_mem = device.total_memory();

#if defined(USE_DCU)
    constexpr int64_t kDefaultDcuVmmReserveMb = 64;
    int64_t reserve_mb =
        util::get_int_env("XLLM_DCU_VMM_RESERVE_MB", kDefaultDcuVmmReserveMb);
    if (reserve_mb <= 0) {
      reserve_mb = kDefaultDcuVmmReserveMb;
    }

    reserve_size = static_cast<size_t>(reserve_mb) * 1024ULL * 1024ULL;
    reserve_size =
        std::min<size_t>(static_cast<size_t>(total_mem), reserve_size);
#else
    reserve_size = static_cast<size_t>(total_mem + total_mem / 8);
#endif
  }

  reserved_size_ = align_up(reserve_size, granularity_);

  LOG(INFO) << "SharedVMMAllocator init: device=" << device_id_
            << ", reserve_size=" << reserve_size << " bytes ("
            << reserve_size / (1024 * 1024) << " MB)"
            << ", granularity=" << granularity_ << " bytes ("
            << granularity_ / (1024 * 1024) << " MB)"
            << ", aligned reserved_size=" << reserved_size_ << " bytes ("
            << reserved_size_ / (1024 * 1024) << " MB)";

  VirtualSpace first_space;
  vmm::create_vir_ptr(first_space.base_ptr, reserved_size_);
  first_space.reserved_size = reserved_size_;
  first_space.mapped_size = 0;
  virtual_spaces_.push_back(first_space);
  current_space_index_ = 0;

  initialized_ = true;
  VLOG(20) << "SharedVMMAllocator initialized on device " << device_id_
           << ". Reserved virtual address space: " << reserved_size_
           << " bytes (" << reserved_size_ / (1024 * 1024) << " MB)";
}

void SharedVMMAllocator::reset_allocation_pointer() {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t prev_offset = current_offset_;
  current_offset_ = 0;
  VLOG(20) << "SharedVMMAllocator::reset_allocation_pointer() called. "
           << "Previous offset: " << prev_offset << " bytes, "
           << "High water mark: " << high_water_mark_ << " bytes, "
           << "Mapped size: " << mapped_size_ << " bytes";
}

void SharedVMMAllocator::switch_to_new_virtual_space() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_) {
    LOG(FATAL) << "SharedVMMAllocator not initialized";
    return;
  }

  size_t prev_offset = current_offset_;
  size_t prev_space_index = current_space_index_;

  // Create a new virtual address space
  VirtualSpace new_space;
  if (!vmm::try_create_vir_ptr(new_space.base_ptr, reserved_size_)) {
    LOG(ERROR) << "SharedVMMAllocator: failed to reserve virtual address space "
               << reserved_size_ << " bytes on device " << device_id_;
    throw std::runtime_error(
        "SharedVMMAllocator: failed to reserve virtual address space");
  }
  new_space.reserved_size = reserved_size_;
  new_space.mapped_size = 0;

  // Map all existing physical memory to the new virtual space
  size_t granularity = get_granularity();
  for (size_t i = 0; i < handles_.size(); ++i) {
    VirPtr map_addr = add_offset(new_space.base_ptr, i * granularity);
    if (!vmm::try_map(map_addr, handles_[i], granularity, device_id_)) {
      LOG(ERROR) << "SharedVMMAllocator: failed to map physical handle " << i
                 << " of " << handles_.size()
                 << " into new virtual space on device " << device_id_;
      if (new_space.mapped_size > 0) {
        vmm::unmap(new_space.base_ptr, new_space.mapped_size);
      }
      vmm::release_vir_ptr(new_space.base_ptr, new_space.reserved_size);
      throw std::runtime_error(
          "SharedVMMAllocator: failed to map physical memory into new virtual "
          "space");
    }
    new_space.mapped_size += granularity;
  }

  // Add the new space and switch to it
  virtual_spaces_.push_back(new_space);
  current_space_index_ = virtual_spaces_.size() - 1;
  current_offset_ = 0;

  VLOG(20) << "SharedVMMAllocator::switch_to_new_virtual_space() called. "
           << "Previous space: " << prev_space_index
           << ", previous offset: " << prev_offset << " bytes, "
           << "New space: " << current_space_index_ << ", base_ptr: 0x"
           << std::hex << new_space.base_ptr << std::dec
           << ", physical memory mapped: " << new_space.mapped_size
           << " bytes, "
           << "Total virtual spaces: " << virtual_spaces_.size();
}

void SharedVMMAllocator::map_physical_to_virtual(VirtualSpace& space,
                                                 size_t start_handle_idx) {
  // This function is called with mutex_ held
  size_t granularity = get_granularity();

  for (size_t i = start_handle_idx; i < handles_.size(); ++i) {
    VirPtr map_addr = add_offset(space.base_ptr, i * granularity);
    vmm::map(map_addr, handles_[i], granularity, device_id_);
  }

  space.mapped_size = handles_.size() * granularity;
}

void* SharedVMMAllocator::allocate(size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_ || virtual_spaces_.empty()) {
    LOG(FATAL) << "SharedVMMAllocator not initialized";
    return nullptr;
  }

  if (size == 0) {
    return nullptr;
  }

  // Align size to granularity
  size_t granularity = get_granularity();
  size_t aligned_size = (size + granularity - 1) / granularity * granularity;

  // Check if we need to extend physical memory mapping
  if (current_offset_ + aligned_size > mapped_size_) {
    extend_mapping(current_offset_ + aligned_size);
  }

  // Get current virtual space
  VirtualSpace& current_space = virtual_spaces_[current_space_index_];

  // Allocate from current offset in current virtual space
  VirPtr vir_ptr = add_offset(current_space.base_ptr, current_offset_);
  void* ptr = to_void_ptr(vir_ptr);
  current_offset_ += aligned_size;
  high_water_mark_ = std::max(high_water_mark_, current_offset_);

  VLOG(20) << "SharedVMMAllocator: Allocated " << size
           << " bytes (aligned: " << aligned_size
           << "), offset: " << (current_offset_ - aligned_size)
           << ", ptr: " << ptr << ", space: " << current_space_index_;

  return ptr;
}

void SharedVMMAllocator::deallocate(void* ptr) {
  // No-op: VMM allocator does not free individual allocations
  // Memory will be reused after switch_to_new_virtual_space()
  VLOG(20) << "SharedVMMAllocator: deallocate called (no-op), ptr: " << ptr;
}

void SharedVMMAllocator::shrink_mapping_locked(size_t target_mapped_size) {
  const size_t granularity = get_granularity();
  if (granularity == 0) {
    return;
  }
  while (mapped_size_ > target_mapped_size && !handles_.empty()) {
    const size_t unmap_offset = mapped_size_ - granularity;
    for (auto& space : virtual_spaces_) {
      if (space.mapped_size > unmap_offset) {
        VirPtr map_addr = add_offset(space.base_ptr, unmap_offset);
        vmm::unmap(map_addr, granularity);
        space.mapped_size = unmap_offset;
      }
    }
    PhyMemHandle handle = handles_.back();
    handles_.pop_back();
    vmm::release_phy_mem_handle(handle);
    mapped_size_ = unmap_offset;
  }
}

void SharedVMMAllocator::rollback_capture(size_t target_mapped_size,
                                          size_t target_num_spaces) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return;
  }
  shrink_mapping_locked(target_mapped_size);
  const size_t min_spaces = std::max<size_t>(1, target_num_spaces);
  while (virtual_spaces_.size() > min_spaces) {
    VirtualSpace space = virtual_spaces_.back();
    virtual_spaces_.pop_back();
    if (space.mapped_size > 0) {
      vmm::unmap(space.base_ptr, space.mapped_size);
    }
    if (space.reserved_size > 0) {
      vmm::release_vir_ptr(space.base_ptr, space.reserved_size);
    }
  }
  current_space_index_ =
      virtual_spaces_.empty() ? 0 : virtual_spaces_.size() - 1;
  current_offset_ = 0;
  high_water_mark_ = std::min(high_water_mark_, mapped_size_);
}

void SharedVMMAllocator::extend_mapping(size_t new_size) {
  // This function is called with mutex_ held

  if (new_size > reserved_size_) {
    LOG(ERROR) << "SharedVMMAllocator: Requested size " << new_size
               << " exceeds reserved virtual address space " << reserved_size_;
    throw std::runtime_error(
        "SharedVMMAllocator: requested size exceeds reserved virtual address "
        "space");
  }

  size_t granularity = get_granularity();
  size_t prev_mapped_size = mapped_size_;
  size_t prev_handle_count = handles_.size();

  // Create new physical memory and map to all virtual spaces
  while (mapped_size_ < new_size) {
    // Create physical memory handle
    PhyMemHandle handle;
    if (!vmm::try_create_phy_mem_handle(handle, device_id_)) {
      LOG(ERROR) << "SharedVMMAllocator: OOM creating physical memory handle "
                 << "at mapped_size=" << mapped_size_
                 << ", requested=" << new_size
                 << ", rolling back to prev_mapped_size=" << prev_mapped_size;
      shrink_mapping_locked(prev_mapped_size);
      throw std::runtime_error(
          "SharedVMMAllocator: out of physical memory while extending VMM "
          "mapping");
    }

    // Map to all existing virtual address spaces
    size_t mapped_spaces = 0;
    bool map_ok = true;
    for (auto& space : virtual_spaces_) {
      VirPtr map_addr = add_offset(space.base_ptr, mapped_size_);
      if (!vmm::try_map(map_addr, handle, granularity, device_id_)) {
        map_ok = false;
        break;
      }
      space.mapped_size = mapped_size_ + granularity;
      ++mapped_spaces;
    }
    if (!map_ok) {
      for (size_t s = 0; s < mapped_spaces; ++s) {
        VirPtr map_addr = add_offset(virtual_spaces_[s].base_ptr, mapped_size_);
        vmm::unmap(map_addr, granularity);
        virtual_spaces_[s].mapped_size = mapped_size_;
      }
      vmm::release_phy_mem_handle(handle);
      shrink_mapping_locked(prev_mapped_size);
      throw std::runtime_error(
          "SharedVMMAllocator: failed to map physical handle across virtual "
          "spaces");
    }

    handles_.push_back(handle);
    mapped_size_ += granularity;
  }

  VLOG(20) << "SharedVMMAllocator::extend_mapping() extended from "
           << prev_mapped_size << " to " << mapped_size_ << " bytes ("
           << mapped_size_ / (1024 * 1024) << " MB), "
           << "requested: " << new_size << " bytes, "
           << "new handles: " << (handles_.size() - prev_handle_count)
           << ", total virtual spaces: " << virtual_spaces_.size();
}

size_t SharedVMMAllocator::get_granularity() const { return granularity_; }

}  // namespace xllm
