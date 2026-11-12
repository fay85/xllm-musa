/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/extra/stl.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

// Preserve original K32 quantization using BF16 mantissas and FP32 partials.
// The tested FP8 BMM backend altered group dot products enough to fail the
// real-expert oracle; it is intentionally not used by this explicit mode.
// This is a BF16 tensor-core path, not native FP4 or FP8 tensor-core GEMM.
torch::Tensor deepseek_v41_partition_linear(const torch::Tensor& input,
                                            const torch::Tensor& packed,
                                            const torch::Tensor& scales,
                                            const std::string& bmm_so_path,
                                            int64_t max_partial_bytes) {
  CHECK(input.defined() && packed.defined() && scales.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(packed.scalar_type(), torch::kUInt8);
  CHECK_EQ(scales.scalar_type(), torch::kUInt8);
  CHECK_EQ(input.device(), packed.device());
  CHECK_EQ(input.device(), scales.device());
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(packed.dim(), 2);
  CHECK_EQ(scales.dim(), 2);
  CHECK(input.is_contiguous() && packed.is_contiguous() &&
        scales.is_contiguous());
  const int64_t rows = input.size(0);
  const int64_t columns = packed.size(0);
  const int64_t inner = input.size(1);
  CHECK_GE(rows, 0);
  CHECK_GT(columns, 0);
  CHECK_GT(inner, 0);
  CHECK_EQ(inner % 32, 0);
  CHECK_EQ(packed.size(1) * 2, inner);
  CHECK_EQ(scales.size(0), columns);
  CHECK_EQ(scales.size(1) * 32, inner);
  CHECK(!bmm_so_path.empty());
  CHECK_GT(max_partial_bytes, 0);
  torch::DeviceGuard guard(input.device());
  if (rows == 0) {
    return torch::empty({rows, columns}, input.options());
  }
  const int64_t groups = inner / 32;
  const int64_t chunk_rows =
      std::min<int64_t>(256, max_partial_bytes / 4 / groups / columns);
  CHECK_GT(chunk_rows, 0)
      << "K32 partition byte budget must fit at least one output row";
  // Resolve dependency before launching any model kernels. No exception
  // fallback.
  const ffi::Function run = get_function_from_file(bmm_so_path, "bmm");
  if (rows > chunk_rows) {
    torch::Tensor output = torch::empty({rows, columns}, input.options());
    for (int64_t start = 0; start < rows; start += chunk_rows) {
      const int64_t count = std::min(chunk_rows, rows - start);
      const torch::Tensor chunk =
          deepseek_v41_partition_linear(input.narrow(/*dim=*/0, start, count),
                                        packed,
                                        scales,
                                        bmm_so_path,
                                        max_partial_bytes);
      output.narrow(/*dim=*/0, start, count).copy_(chunk);
    }
    return output;
  }
  auto [quantized, input_scale] = deepseek_v41_block32_batch_quant(input);
  auto [weight, weight_scale] =
      deepseek_v41_block32_batch_unpack(packed, scales);
  const torch::Tensor input_mantissa = quantized.to(torch::kBFloat16);
  const torch::Tensor weight_mantissa = weight.to(torch::kBFloat16);
  const torch::Tensor partial = torch::empty(
      {groups, rows, columns}, input.options().dtype(torch::kFloat32));
  {
    TvmffiStreamGuard stream_guard(input.device());
    run(to_ffi_tensor(input_mantissa),
        to_ffi_tensor(weight_mantissa),
        to_ffi_tensor(partial),
        to_ffi_optional_tensor(std::nullopt),
        to_ffi_optional_tensor(std::nullopt),
        to_ffi_optional_tensor(std::nullopt),
        ffi::Tuple<int64_t, int64_t>(int64_t{1}, int64_t{-1}),
        ffi::Tuple<int64_t, int64_t>(int64_t{1}, int64_t{-1}),
        /*trans_a=*/false,
        /*trans_b=*/true,
        /*fixed_scale_layout=*/false);
  }
  return deepseek_v41_block32_batch_reduce_scaled(
      partial, input_scale, weight_scale);
}

}  // namespace xllm::kernel::musa
