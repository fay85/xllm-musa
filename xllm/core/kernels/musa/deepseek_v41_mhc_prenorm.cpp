/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/

#include "core/kernels/musa/deepseek_v41_mhc_prenorm.h"

#include <glog/logging.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>
#include <filesystem>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

torch::Tensor mhc_prenorm(const torch::Tensor& residual_flat,
                          const torch::Tensor& fn,
                          const std::string& library_directory,
                          double epsilon,
                          bool fused_reduce) {
  CHECK(residual_flat.defined() && fn.defined());
  CHECK(residual_flat.device().type() == torch::kPrivateUse1);
  CHECK_EQ(fn.device(), residual_flat.device());
  CHECK_EQ(residual_flat.scalar_type(), torch::kBFloat16);
  CHECK_EQ(fn.scalar_type(), torch::kFloat32);
  CHECK_EQ(residual_flat.dim(), 2);
  CHECK_EQ(fn.dim(), 2);
  CHECK(residual_flat.is_contiguous() && fn.is_contiguous());
  CHECK_EQ(fn.size(0), 24);
  CHECK_EQ(fn.size(1), residual_flat.size(1));
  CHECK_GT(residual_flat.size(0), 0);
  CHECK_LE(residual_flat.size(0), 512);
  CHECK_GE(residual_flat.size(1), 1024);
  CHECK_EQ(residual_flat.size(1) % 1024, 0);
  CHECK(std::isfinite(epsilon) && epsilon > 0.0);
  CHECK(!library_directory.empty());
  torch::DeviceGuard guard(residual_flat.device());
  const int64_t rows = residual_flat.size(0);
  const int64_t splits = rows <= 64 ? 32 : 16;
  const std::filesystem::path library =
      std::filesystem::path(library_directory) /
      (rows <= 32 ? "libdeepseek_v41_mhc_m32.so"
                  : "libdeepseek_v41_mhc_m64.so");
  const ffi::Function run =
      get_function_from_file(library.string(), "tf32_hc_prenorm_gemm");
  const torch::Tensor partial =
      torch::empty({splits, rows, fn.size(0)},
                   residual_flat.options().dtype(torch::kFloat32));
  const torch::Tensor square_sum = torch::empty(
      {splits, rows}, residual_flat.options().dtype(torch::kFloat32));
  {
    TvmffiStreamGuard stream_guard(residual_flat.device());
    run(to_ffi_tensor(residual_flat),
        to_ffi_tensor(fn),
        to_ffi_tensor(partial),
        to_ffi_tensor(square_sum),
        splits);
  }
  if (fused_reduce) {
    return deepseek_v41_mhc_prenorm_reduce(
        partial, square_sum, residual_flat.size(1), epsilon);
  }
  const torch::Tensor projection = partial.sum(/*dim=*/0);
  const torch::Tensor mean_square =
      square_sum.sum(/*dim=*/0) / residual_flat.size(1);
  return projection * torch::rsqrt(mean_square.unsqueeze(-1) + epsilon);
}

}  // namespace

torch::Tensor deepseek_v41_mhc_prenorm(const torch::Tensor& residual_flat,
                                       const torch::Tensor& fn,
                                       const std::string& library_directory,
                                       double epsilon) {
  return mhc_prenorm(residual_flat,
                     fn,
                     library_directory,
                     epsilon,
                     /*fused_reduce=*/false);
}

torch::Tensor deepseek_v41_mhc_prenorm_fused(
    const torch::Tensor& residual_flat,
    const torch::Tensor& fn,
    const std::string& library_directory,
    double epsilon) {
  return mhc_prenorm(residual_flat,
                     fn,
                     library_directory,
                     epsilon,
                     /*fused_reduce=*/true);
}

}  // namespace xllm::kernel::musa
