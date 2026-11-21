// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <cstdint>
#include <limits>

#include "op_utils.hpp"

namespace {

constexpr int32_t kWarpSize = 32;
// The MP31 paged score kernel assigns four 64-entry KV blocks to one segment.
constexpr int32_t kBlockKV = 64;
constexpr int32_t kSegmentSize = 4 * kBlockKV;

__device__ int64_t segment_count(const int32_t* lengths,
                                 int64_t row,
                                 int32_t next_n) {
  const int64_t length = lengths[(row + 1) * next_n - 1];
  return (length + kSegmentSize - 1) / kSegmentSize;
}

// One warp finds one MP boundary. Each lane owns a contiguous interval of
// rows. Only the lane containing the boundary needs to scan its interval again.
__global__ void build_metadata(const int32_t* lengths,
                               int32_t rows,
                               int32_t next_n,
                               int32_t num_mps,
                               int32_t* schedule) {
  const int32_t lane = threadIdx.x;
  const int32_t mp = blockIdx.x;
  const int64_t rows_per_lane =
      (static_cast<int64_t>(rows) + kWarpSize - 1) / kWarpSize;
  const int64_t begin = lane * rows_per_lane;
  const int64_t end = min(begin + rows_per_lane, static_cast<int64_t>(rows));
  int64_t count = 0;
  for (int64_t row = begin; row < end; ++row) {
    count += segment_count(lengths, row, next_n);
  }

  int64_t prefix = count;
  for (int32_t delta = 1; delta < kWarpSize; delta *= 2) {
    const int64_t other = __shfl_up_sync(0xffffffff, prefix, delta);
    if (lane >= delta) {
      prefix += other;
    }
  }
  const int64_t total = __shfl_sync(0xffffffff, prefix, kWarpSize - 1);
  const int64_t target =
      (total / num_mps) * mp + min(static_cast<int64_t>(mp), total % num_mps);
  if (target == total) {
    if (lane == 0) {
      schedule[2 * mp] = rows;
      schedule[2 * mp + 1] = 0;
    }
    return;
  }

  int64_t offset = target - (prefix - count);
  if (offset >= 0 && offset < count) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t segments = segment_count(lengths, row, next_n);
      if (offset < segments) {
        schedule[2 * mp] = static_cast<int32_t>(row);
        schedule[2 * mp + 1] = static_cast<int32_t>(offset);
        return;
      }
      offset -= segments;
    }
  }
}

void get_paged_mqa_logits_metadata(ffi::TensorView lengths,
                                   int64_t block_kv,
                                   ffi::TensorView schedule) {
  CHECK_INPUT_AND_TYPE(lengths, dl_int32);
  CHECK_INPUT_AND_TYPE(schedule, dl_int32);
  CHECK_DEVICE(schedule, lengths);
  CHECK_DIM(2, schedule);
  TVM_FFI_ICHECK(lengths.dim() == 1 || lengths.dim() == 2);
  const int64_t next_n = lengths.dim() == 1 ? 1 : lengths.size(1);
  TVM_FFI_ICHECK(next_n == 1 || next_n == 2 || next_n == 4);
  TVM_FFI_ICHECK_EQ(block_kv, kBlockKV);
  TVM_FFI_ICHECK_LE(lengths.size(0), std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_EQ(schedule.size(1), 2);
  ffi::MUSADeviceGuard guard(lengths.device().device_id);
  musaDeviceProp properties{};
  MATE_MUSA_RUNTIME_CHECK(
      musaGetDeviceProperties(&properties, lengths.device().device_id));
  const int64_t num_mps = schedule.size(0) - 1;
  TVM_FFI_ICHECK(num_mps > 0 && num_mps <= properties.multiProcessorCount);
  build_metadata<<<num_mps + 1, kWarpSize, 0, get_stream(lengths.device())>>>(
      static_cast<const int32_t*>(lengths.data_ptr()),
      static_cast<int32_t>(lengths.size(0)),
      static_cast<int32_t>(next_n),
      static_cast<int32_t>(num_mps),
      static_cast<int32_t*>(schedule.data_ptr()));
  MATE_CHECK_MUSA_KERNEL_LAUNCH();
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_paged_mqa_logits_metadata,
                              get_paged_mqa_logits_metadata);
