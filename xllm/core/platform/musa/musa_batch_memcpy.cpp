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

#include "platform/musa/musa_batch_memcpy.h"

#include "platform/platform.h"

#include <glog/logging.h>
#include <musa_runtime.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <string>

namespace xllm::musa {
namespace {

const char* direction_name(bool h2d) { return h2d ? "H2D" : "D2H"; }

void drain_stream_after_failure(musaStream_t stream,
                                const char* operation,
                                size_t descriptor_count) {
  const musaError_t drain_result = musaStreamSynchronize(stream);
  LOG(ERROR) << "MUSA batch memcpy " << operation
             << " failure recovery stream sync: descriptor_count="
             << descriptor_count
             << ", result=" << static_cast<int32_t>(drain_result)
             << ", error=" << musaGetErrorString(drain_result);
  if (drain_result != musaSuccess) {
    LOG(FATAL) << "Failed to drain MUSA batch memcpy stream after submission "
                  "failure: operation="
               << operation << ", descriptor_count=" << descriptor_count
               << ", result=" << static_cast<int32_t>(drain_result)
               << ", error=" << musaGetErrorString(drain_result);
  }
}

}  // namespace

void MUSABatchMemcpy::init(int32_t device_id) {
  if (initialized_) {
    CHECK_EQ(device_id_, device_id)
        << "MUSABatchMemcpy cannot be initialized for another device.";
    return;
  }
  CHECK_GE(device_id, 0) << "MUSABatchMemcpy device id must be non-negative.";
  device_id_ = device_id;
  initialized_ = true;
}

bool MUSABatchMemcpy::submit_h2d(const std::vector<torch::Tensor>& src_tensors,
                                 const std::vector<torch::Tensor>& dst_tensors,
                                 Stream* stream) {
  return copy(src_tensors,
              dst_tensors,
              stream,
              Direction::H2D,
              CompletionMode::SUBMIT_ONLY);
}

bool MUSABatchMemcpy::submit_d2h(const std::vector<torch::Tensor>& src_tensors,
                                 const std::vector<torch::Tensor>& dst_tensors,
                                 Stream* stream) {
  return copy(src_tensors,
              dst_tensors,
              stream,
              Direction::D2H,
              CompletionMode::SUBMIT_ONLY);
}

bool MUSABatchMemcpy::copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                               const std::vector<torch::Tensor>& dst_tensors,
                               Stream* stream) {
  return copy(src_tensors,
              dst_tensors,
              stream,
              Direction::D2H,
              CompletionMode::SYNCHRONIZE);
}

bool MUSABatchMemcpy::submit_h2d_raw(const std::vector<void*>& src_addrs,
                                     const std::vector<void*>& dst_addrs,
                                     const std::vector<size_t>& byte_counts,
                                     Stream* stream) {
  return copy_raw(src_addrs,
                  dst_addrs,
                  byte_counts,
                  stream,
                  Direction::H2D,
                  CompletionMode::SUBMIT_ONLY);
}

bool MUSABatchMemcpy::submit_d2h_raw(const std::vector<void*>& src_addrs,
                                     const std::vector<void*>& dst_addrs,
                                     const std::vector<size_t>& byte_counts,
                                     Stream* stream) {
  return copy_raw(src_addrs,
                  dst_addrs,
                  byte_counts,
                  stream,
                  Direction::D2H,
                  CompletionMode::SUBMIT_ONLY);
}

bool MUSABatchMemcpy::copy_d2h_raw(const std::vector<void*>& src_addrs,
                                   const std::vector<void*>& dst_addrs,
                                   const std::vector<size_t>& byte_counts,
                                   Stream* stream) {
  return copy_raw(src_addrs,
                  dst_addrs,
                  byte_counts,
                  stream,
                  Direction::D2H,
                  CompletionMode::SYNCHRONIZE);
}

bool MUSABatchMemcpy::valid_inputs(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<torch::Tensor>& dst_tensors,
    const Stream* stream,
    Direction direction) const {
  const bool h2d = direction == Direction::H2D;
  const char* operation = direction_name(h2d);
  if (!initialized_) {
    LOG(ERROR) << "MUSABatchMemcpy " << operation << " is not initialized.";
    return false;
  }
  if (stream == nullptr) {
    LOG(ERROR) << "MUSABatchMemcpy " << operation << " stream must not be null.";
    return false;
  }
  if (stream->get_stream()->device_index() != device_id_) {
    LOG(ERROR) << "MUSABatchMemcpy " << operation
               << " stream device mismatch: expected=" << device_id_
               << ", actual=" << stream->get_stream()->device_index();
    return false;
  }
  if (src_tensors.size() != dst_tensors.size()) {
    LOG(ERROR) << "MUSABatchMemcpy " << operation
               << " tensor count mismatch: src=" << src_tensors.size()
               << ", dst=" << dst_tensors.size();
    return false;
  }
  for (size_t index = 0; index < src_tensors.size(); ++index) {
    const torch::Tensor& src = src_tensors[index];
    const torch::Tensor& dst = dst_tensors[index];
    if (!src.defined() || !dst.defined()) {
      LOG(ERROR) << "MUSABatchMemcpy " << operation
                 << " has undefined tensor at index=" << index;
      return false;
    }
    if (!src.is_contiguous() || !dst.is_contiguous()) {
      LOG(ERROR) << "MUSABatchMemcpy " << operation
                 << " requires contiguous tensors at index=" << index;
      return false;
    }
    if (src.nbytes() != dst.nbytes()) {
      LOG(ERROR) << "MUSABatchMemcpy " << operation
                 << " byte count mismatch at index=" << index
                 << ", src=" << src.nbytes() << ", dst=" << dst.nbytes();
      return false;
    }
    const torch::Tensor& host = h2d ? src : dst;
    const torch::Tensor& musa = h2d ? dst : src;
    if (!host.device().is_cpu()) {
      LOG(ERROR) << "MUSABatchMemcpy " << operation
                 << " host tensor is not CPU at index=" << index;
      return false;
    }
    if (musa.device().type() != Platform::type_torch() ||
        !musa.device().has_index() || musa.device().index() != device_id_) {
      LOG(ERROR) << "MUSABatchMemcpy " << operation
                 << " device tensor mismatch at index=" << index
                 << ", expected_device=" << device_id_
                 << ", actual_device=" << musa.device();
      return false;
    }
  }
  return true;
}

bool MUSABatchMemcpy::copy(const std::vector<torch::Tensor>& src_tensors,
                           const std::vector<torch::Tensor>& dst_tensors,
                           Stream* stream,
                           Direction direction,
                           CompletionMode completion_mode) {
  musaStream_t musa_stream = nullptr;
  bool stream_has_submitted_work = false;
  const bool h2d = direction == Direction::H2D;
  try {
    if (!valid_inputs(src_tensors, dst_tensors, stream, direction)) {
      return false;
    }
    if (src_tensors.empty()) {
      return true;
    }

    std::vector<void*> src_addrs;
    std::vector<void*> dst_addrs;
    std::vector<size_t> byte_counts;
    src_addrs.reserve(src_tensors.size());
    dst_addrs.reserve(src_tensors.size());
    byte_counts.reserve(src_tensors.size());
    for (size_t index = 0; index < src_tensors.size(); ++index) {
      const size_t nbytes = static_cast<size_t>(src_tensors[index].nbytes());
      if (nbytes == 0) {
        continue;
      }
      src_addrs.emplace_back(src_tensors[index].data_ptr());
      dst_addrs.emplace_back(dst_tensors[index].data_ptr());
      byte_counts.emplace_back(nbytes);
    }
    if (byte_counts.empty()) {
      return true;
    }

    const size_t count = byte_counts.size();
    musaMemcpyAttributes attr{};
    attr.srcAccessOrder = musaMemcpySrcAccessOrderStream;
    attr.srcLocHint.type =
        h2d ? musaMemLocationTypeHost : musaMemLocationTypeDevice;
    attr.srcLocHint.id = h2d ? 0 : device_id_;
    attr.dstLocHint.type =
        h2d ? musaMemLocationTypeDevice : musaMemLocationTypeHost;
    attr.dstLocHint.id = h2d ? device_id_ : 0;
    attr.flags = musaMemcpyFlagDefault;
    size_t attr_index = 0;

    const c10::StreamGuard guard = stream->set_stream_guard();
    musa_stream = stream->get_stream()->stream();
    for (size_t offset = 0; offset < count; offset += kMaxBatchCopyCount) {
      const size_t chunk = std::min(kMaxBatchCopyCount, count - offset);
      size_t fail_index = std::numeric_limits<size_t>::max();
      const musaError_t submit_result =
          musaMemcpyBatchAsync(dst_addrs.data() + offset,
                               src_addrs.data() + offset,
                               byte_counts.data() + offset,
                               chunk,
                               &attr,
                               &attr_index,
                               /*numAttrs=*/1,
                               &fail_index,
                               musa_stream);
      if (submit_result != musaSuccess ||
          fail_index != std::numeric_limits<size_t>::max()) {
        LOG(ERROR) << "MUSA batch memcpy " << direction_name(h2d)
                   << " submission failed: chunk_offset=" << offset
                   << ", chunk_count=" << chunk
                   << ", fail_index="
                   << (fail_index == std::numeric_limits<size_t>::max()
                           ? fail_index
                           : offset + fail_index)
                   << ", result=" << static_cast<int32_t>(submit_result)
                   << ", error=" << musaGetErrorString(submit_result);
        drain_stream_after_failure(musa_stream, direction_name(h2d), count);
        return false;
      }
      stream_has_submitted_work = true;
    }

    if (completion_mode == CompletionMode::SUBMIT_ONLY) {
      return true;
    }

    const musaError_t sync_result = musaStreamSynchronize(musa_stream);
    if (sync_result != musaSuccess) {
      LOG(ERROR) << "MUSA batch memcpy " << direction_name(h2d)
                 << " stream sync failed: count=" << count
                 << ", result=" << static_cast<int32_t>(sync_result)
                 << ", error=" << musaGetErrorString(sync_result);
    }
    return sync_result == musaSuccess;
  } catch (const std::exception& error) {
    LOG(ERROR) << "MUSA batch memcpy " << direction_name(h2d)
               << " raised an exception: " << error.what();
  } catch (...) {
    LOG(ERROR) << "MUSA batch memcpy " << direction_name(h2d)
               << " raised an unknown exception.";
  }
  if (stream_has_submitted_work) {
    drain_stream_after_failure(
        musa_stream, direction_name(h2d), src_tensors.size());
  }
  return false;
}

bool MUSABatchMemcpy::copy_raw(const std::vector<void*>& src_addrs,
                               const std::vector<void*>& dst_addrs,
                               const std::vector<size_t>& byte_counts,
                               Stream* stream,
                               Direction direction,
                               CompletionMode completion_mode) {
  musaStream_t musa_stream = nullptr;
  bool stream_has_submitted_work = false;
  const bool h2d = direction == Direction::H2D;
  try {
    if (!initialized_) {
      LOG(ERROR) << "MUSABatchMemcpy " << direction_name(h2d)
                 << " is not initialized.";
      return false;
    }
    if (stream == nullptr || stream->get_stream()->device_index() != device_id_) {
      LOG(ERROR) << "MUSABatchMemcpy " << direction_name(h2d)
                 << " invalid stream.";
      return false;
    }
    if (src_addrs.size() != dst_addrs.size() ||
        src_addrs.size() != byte_counts.size()) {
      LOG(ERROR) << "MUSABatchMemcpy " << direction_name(h2d)
                 << " raw vector size mismatch.";
      return false;
    }
    if (byte_counts.empty()) {
      return true;
    }

    const size_t count = byte_counts.size();
    musaMemcpyAttributes attr{};
    attr.srcAccessOrder = musaMemcpySrcAccessOrderStream;
    attr.srcLocHint.type =
        h2d ? musaMemLocationTypeHost : musaMemLocationTypeDevice;
    attr.srcLocHint.id = h2d ? 0 : device_id_;
    attr.dstLocHint.type =
        h2d ? musaMemLocationTypeDevice : musaMemLocationTypeHost;
    attr.dstLocHint.id = h2d ? device_id_ : 0;
    attr.flags = musaMemcpyFlagDefault;
    size_t attr_index = 0;

    const c10::StreamGuard guard = stream->set_stream_guard();
    musa_stream = stream->get_stream()->stream();
    for (size_t offset = 0; offset < count; offset += kMaxBatchCopyCount) {
      const size_t chunk = std::min(kMaxBatchCopyCount, count - offset);
      size_t fail_index = std::numeric_limits<size_t>::max();
      const musaError_t submit_result = musaMemcpyBatchAsync(
          const_cast<void**>(dst_addrs.data() + offset),
          const_cast<void**>(src_addrs.data() + offset),
          const_cast<size_t*>(byte_counts.data() + offset),
          chunk,
          &attr,
          &attr_index,
          /*numAttrs=*/1,
          &fail_index,
          musa_stream);
      if (submit_result != musaSuccess ||
          fail_index != std::numeric_limits<size_t>::max()) {
        LOG(ERROR) << "MUSA batch memcpy raw " << direction_name(h2d)
                   << " submission failed: chunk_offset=" << offset
                   << ", chunk_count=" << chunk
                   << ", result=" << static_cast<int32_t>(submit_result)
                   << ", error=" << musaGetErrorString(submit_result);
        drain_stream_after_failure(musa_stream, direction_name(h2d), count);
        return false;
      }
      stream_has_submitted_work = true;
    }

    if (completion_mode == CompletionMode::SUBMIT_ONLY) {
      return true;
    }

    const musaError_t sync_result = musaStreamSynchronize(musa_stream);
    if (sync_result != musaSuccess) {
      LOG(ERROR) << "MUSA batch memcpy raw " << direction_name(h2d)
                 << " stream sync failed: count=" << count
                 << ", result=" << static_cast<int32_t>(sync_result)
                 << ", error=" << musaGetErrorString(sync_result);
    }
    return sync_result == musaSuccess;
  } catch (const std::exception& error) {
    LOG(ERROR) << "MUSA batch memcpy raw " << direction_name(h2d)
               << " raised an exception: " << error.what();
  } catch (...) {
    LOG(ERROR) << "MUSA batch memcpy raw " << direction_name(h2d)
               << " raised an unknown exception.";
  }
  if (stream_has_submitted_work) {
    drain_stream_after_failure(
        musa_stream, direction_name(h2d), byte_counts.size());
  }
  return false;
}

}  // namespace xllm::musa
