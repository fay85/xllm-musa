/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/

#include "core/kernels/musa/deepseek_v41_engram_gather.h"

#include <glog/logging.h>
#include <torch/utils.h>

#include <array>
#include <limits>

#include "core/kernels/musa/musa_aot_kernel.h"

namespace xllm::kernel::musa {

torch::Tensor deepseek_v41_engram_gather(const torch::Tensor& weight,
                                         const torch::Tensor& scales,
                                         const torch::Tensor& ids,
                                         int64_t row_lo,
                                         const std::string& aot_directory) {
  CHECK(weight.defined() && scales.defined() && ids.defined());
  CHECK(weight.device().type() == torch::kPrivateUse1);
  CHECK_EQ(weight.device(), scales.device());
  CHECK_EQ(weight.device(), ids.device());
  CHECK_EQ(weight.scalar_type(), torch::kUInt8);
  CHECK_EQ(scales.scalar_type(), torch::kUInt8);
  CHECK_EQ(ids.scalar_type(), torch::kInt64);
  CHECK_EQ(weight.dim(), 2);
  CHECK_EQ(weight.size(1), 256);
  CHECK_EQ(scales.sizes(), torch::IntArrayRef({weight.size(0), 8}));
  CHECK(weight.is_contiguous() && scales.is_contiguous() &&
        ids.is_contiguous());
  CHECK_GE(row_lo, 0);
  CHECK_LE(weight.size(0), std::numeric_limits<int64_t>::max() - row_lo);
  CHECK_LE(ids.numel(), std::numeric_limits<int32_t>::max());
  CHECK(!aot_directory.empty());
  torch::DeviceGuard guard(weight.device());
  torch::Tensor output = torch::empty({ids.numel(), 256},
                                      weight.options().dtype(torch::kBFloat16));
  if (ids.numel() == 0) {
    return output;
  }
  int64_t row_hi = row_lo + weight.size(0);
  void* weight_pointer = weight.data_ptr();
  void* scale_pointer = scales.data_ptr();
  void* ids_pointer = ids.data_ptr();
  void* output_pointer = output.data_ptr();
  std::array<void*, 6> arguments = {&weight_pointer,
                                    &scale_pointer,
                                    &ids_pointer,
                                    &output_pointer,
                                    &row_lo,
                                    &row_hi};
  launch_musa_aot_kernel(weight.device(),
                         aot_directory,
                         "engram_gather",
                         {static_cast<uint32_t>(ids.numel()), 1, 1},
                         arguments.data());
  return output;
}

}  // namespace xllm::kernel::musa
