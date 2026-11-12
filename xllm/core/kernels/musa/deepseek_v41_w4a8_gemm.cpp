/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/

#include <glog/logging.h>
#include <musa_runtime_api.h>
#include <torch/utils.h>

#include "core/kernels/musa/deepseek_v41_w4a8.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

torch::Tensor deepseek_v41_w4a8_masked_gemm(const torch::Tensor& input,
                                            const torch::Tensor& input_scale,
                                            const torch::Tensor& weight,
                                            const torch::Tensor& weight_scale,
                                            const torch::Tensor& counts,
                                            const std::string& library_path) {
  CHECK(input.defined() && input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.dim(), 3);
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight.scalar_type(), torch::kInt8);
  CHECK_EQ(weight_scale.scalar_type(), torch::kBFloat16);
  CHECK_EQ(counts.scalar_type(), torch::kInt32);
  CHECK_EQ(input_scale.dim(), 3);
  CHECK_EQ(weight.dim(), 3);
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(counts.dim(), 1);
  CHECK_GT(input.size(2), 0);
  CHECK_EQ(input.size(2) % 128, 0);
  CHECK_EQ(input.size(1) % 128, 0);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1), input.size(1));
  CHECK_EQ(input_scale.size(2), input.size(2) / 128);
  CHECK_EQ(weight.size(0), input.size(0));
  CHECK_EQ(weight.size(2) * 2, input.size(2));
  CHECK_EQ(weight_scale.size(0), weight.size(0));
  CHECK_EQ(weight_scale.size(1), weight.size(1));
  CHECK_EQ(weight_scale.size(2), input.size(2) / 128);
  CHECK_EQ(counts.size(0), input.size(0));
  for (const torch::Tensor& tensor :
       {input, input_scale, weight, weight_scale, counts}) {
    CHECK_EQ(tensor.device(), input.device());
    CHECK(tensor.is_contiguous());
  }
  CHECK(!library_path.empty());
  torch::DeviceGuard guard(input.device());
  // Both full-model shapes have an even number of K128 scale blocks. Reject
  // an unsupported shape before launch instead of silently changing strides.
  CHECK_EQ(weight_scale.size(2) % 2, 0);
  musaDeviceProp properties;
  CHECK(musaGetDeviceProperties(&properties, input.get_device()) ==
        musaSuccess);
  torch::Tensor output =
      torch::empty({input.size(0), input.size(1), weight.size(1)},
                   input.options().dtype(torch::kBFloat16));
  const ffi::Function run = get_function_from_file(
      library_path,
      "mp31_masked_moe_gemm_mixed_dtype_bf16_32x256x256_s3_Apergroup");
  TvmffiStreamGuard stream_guard(input.device());
  run(to_ffi_tensor(input),
      to_ffi_tensor(input_scale),
      to_ffi_tensor(weight),
      to_ffi_tensor(weight_scale),
      to_ffi_tensor(output),
      to_ffi_tensor(counts),
      input.size(1),
      static_cast<int64_t>(properties.multiProcessorCount));
  return output;
}

}  // namespace xllm::kernel::musa
