/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/

#include "core/kernels/musa/deepseek_v41_dense_block32.h"

#include <glog/logging.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "core/kernels/musa/musa_aot_kernel.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace xllm::kernel::musa {

torch::Tensor deepseek_v41_dense_block32(const torch::Tensor& input,
                                         const torch::Tensor& weight,
                                         const torch::Tensor& weight_scale,
                                         const std::string& directory) {
  CHECK(input.defined() && weight.defined() && weight_scale.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.device(), weight.device());
  CHECK_EQ(input.device(), weight_scale.device());
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weight.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weight.dim(), 2);
  CHECK_EQ(weight_scale.dim(), 2);
  CHECK(input.is_contiguous() && weight.is_contiguous() &&
        weight_scale.is_contiguous());
  CHECK_EQ(input.size(1), weight.size(1));
  CHECK_GT(input.size(0), 0);
  CHECK_LE(input.size(0), 512);
  CHECK_EQ(weight.size(0) % 32, 0);
  CHECK_EQ(weight.size(1) % 128, 0);
  CHECK_EQ(weight_scale.size(0), weight.size(0) / 32);
  CHECK_EQ(weight_scale.size(1), weight.size(1) / 32);
  CHECK_LE(input.size(0) * weight.size(0), std::numeric_limits<int32_t>::max());
  CHECK(!directory.empty());
  c10::musa::MUSAGuard guard(input.device().index());
  const int32_t splits = weight.size(1) % 512 == 0 ? 16 : 4;
  int32_t rows = static_cast<int32_t>(input.size(0));
  int32_t columns = static_cast<int32_t>(weight.size(0));
  int32_t hidden = static_cast<int32_t>(weight.size(1));
  int32_t elements = rows * columns;
  const torch::Tensor partials = torch::empty(
      {splits, rows, columns}, input.options().dtype(torch::kFloat32));
  torch::Tensor output = torch::empty({rows, columns}, input.options());
  void* input_pointer = input.data_ptr();
  void* weight_pointer = weight.data_ptr();
  void* scale_pointer = weight_scale.data_ptr();
  void* partial_pointer = partials.data_ptr();
  void* output_pointer = output.data_ptr();
  std::array<void*, 7> arguments = {&input_pointer,
                                    &weight_pointer,
                                    &scale_pointer,
                                    &partial_pointer,
                                    &rows,
                                    &columns,
                                    &hidden};
  launch_musa_aot_kernel(input.device(),
                         directory,
                         "split" + std::to_string(splits),
                         {static_cast<uint32_t>((columns + 31) / 32),
                          static_cast<uint32_t>(splits),
                          static_cast<uint32_t>((rows + 31) / 32)},
                         arguments.data());
  std::array<void*, 3> reduction = {
      &partial_pointer, &output_pointer, &elements};
  launch_musa_aot_kernel(input.device(),
                         directory,
                         "reduce" + std::to_string(splits),
                         {static_cast<uint32_t>((elements + 255) / 256), 1, 1},
                         reduction.data());
  return output;
}

}  // namespace xllm::kernel::musa
