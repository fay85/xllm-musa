/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <glog/logging.h>
#include <torch/utils.h>

#include <cstdint>

#include "core/kernels/musa/deepseek_v41_prefix_restore.h"
#include "core/kernels/musa/musa_kernel_utils.h"
namespace xllm::kernel::musa {
namespace {
__global__ void restore_suffix_kernel(uint8_t* target,
                                      const int64_t* slots,
                                      const uint8_t* saved,
                                      const int64_t* prefix_lengths,
                                      int64_t accepted,
                                      int64_t target_rows,
                                      int64_t row_bytes) {
  const int64_t row = blockIdx.x;
  const int64_t slot = slots[row];
  if (prefix_lengths[row] <= accepted || slot < 0 || slot >= target_rows) {
    return;
  }
  for (int64_t offset = threadIdx.x; offset < row_bytes; offset += blockDim.x) {
    target[slot * row_bytes + offset] = saved[row * row_bytes + offset];
  }
}
}  // namespace
void deepseek_v41_restore_suffix(const torch::Tensor& target,
                                 const torch::Tensor& slots,
                                 const torch::Tensor& saved,
                                 const torch::Tensor& prefix_lengths,
                                 int64_t accepted) {
  CHECK(target.defined() && slots.defined() && saved.defined() &&
        prefix_lengths.defined());
  CHECK(target.device().type() == torch::kPrivateUse1);
  CHECK_EQ(target.dim(), 2);
  CHECK_EQ(saved.dim(), 2);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(prefix_lengths.sizes(), slots.sizes());
  CHECK_EQ(saved.size(0), slots.numel());
  CHECK_EQ(saved.size(1), target.size(1));
  CHECK_EQ(saved.scalar_type(), target.scalar_type());
  CHECK_EQ(slots.scalar_type(), torch::kInt64);
  CHECK_EQ(prefix_lengths.scalar_type(), torch::kInt64);
  CHECK_GE(accepted, 0);
  CHECK_LE(slots.numel(), 6);
  for (const torch::Tensor& tensor : {target, slots, saved, prefix_lengths}) {
    CHECK_EQ(tensor.device(), target.device());
    CHECK(tensor.is_contiguous());
  }
  if (slots.numel() == 0) {
    return;
  }
  torch::DeviceGuard guard(target.device());
  restore_suffix_kernel<<<static_cast<uint32_t>(slots.numel()),
                          256,
                          0,
                          current_musa_stream(target.device())>>>(
      static_cast<uint8_t*>(target.data_ptr()),
      slots.data_ptr<int64_t>(),
      static_cast<const uint8_t*>(saved.data_ptr()),
      prefix_lengths.data_ptr<int64_t>(),
      accepted,
      target.size(0),
      target.size(1) * target.element_size());
  CHECK_EQ(musaGetLastError(), musaSuccess);
}
}  // namespace xllm::kernel::musa
