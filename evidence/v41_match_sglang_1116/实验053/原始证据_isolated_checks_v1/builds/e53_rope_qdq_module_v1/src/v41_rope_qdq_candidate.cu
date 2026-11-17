/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <glog/logging.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

template <typename Position>
__global__ void __launch_bounds__(256)
    rope_qdq_kernel(const torch::BFloat16* __restrict__ input,
                    const Position* __restrict__ positions,
                    const float* __restrict__ table,
                    torch::BFloat16* __restrict__ output,
                    int64_t heads,
                    int64_t rotary,
                    int64_t table_rows,
                    bool inverse) {
  constexpr int32_t kHidden = 512;
  __shared__ torch::BFloat16 intermediate[kHidden];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t position = static_cast<int64_t>(positions[row / heads]);
  CUDA_KERNEL_ASSERT(position >= 0 && position < table_rows);
  const int64_t start = kHidden - rotary;
  const float* values = table + position * rotary;
  const torch::BFloat16* input_row = input + row * kHidden;
  for (int64_t column = threadIdx.x; column < kHidden; column += 256) {
    if (column < start) {
      intermediate[column] = input_row[column];
      continue;
    }
    const int64_t local = column - start;
    if ((local & 1) != 0) {
      continue;
    }
    const int64_t pair = local / 2;
    const float x = static_cast<float>(input_row[column]);
    const float y = static_cast<float>(input_row[column + 1]);
    const float c = values[pair];
    const float s =
        inverse ? -values[rotary / 2 + pair] : values[rotary / 2 + pair];
    // The BF16 intermediate is required by the original quantization order.
    intermediate[column] = static_cast<torch::BFloat16>(x * c - y * s);
    intermediate[column + 1] = static_cast<torch::BFloat16>(y * c + x * s);
  }
  __syncthreads();
  for (int32_t column = threadIdx.x; column < kHidden; column += 256) {
    const float value = static_cast<float>(intermediate[column]);
    float maximum = fabsf(value);
#pragma unroll
    for (int32_t delta = 16; delta > 0; delta /= 2) {
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    }
    const float raw_scale = fmaxf(maximum, 1.0e-4f) * (1.0f / 448.0f);
    const uint32_t bits = __float_as_uint(raw_scale);
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 255u) - 127 +
                             ((bits & 0x7fffffu) != 0);
    const float scale = ldexpf(1.0f, exponent);
    const torch::Float8_e4m3fn quantized =
        scaled_fp8_conversion<false, torch::Float8_e4m3fn>(value, scale);
    output[row * kHidden + column] =
        static_cast<torch::BFloat16>(static_cast<float>(quantized) * scale);
  }
}

template <typename Position>
void launch_rope_qdq(const torch::Tensor& input,
                     const torch::Tensor& positions,
                     const torch::Tensor& table,
                     bool inverse,
                     torch::Tensor& output) {
  const int64_t heads = input.dim() == 3 ? input.size(1) : 1;
  rope_qdq_kernel<Position><<<static_cast<uint32_t>(input.size(0) * heads),
                              256,
                              0,
                              current_musa_stream(input.device())>>>(
      input.data_ptr<torch::BFloat16>(),
      positions.data_ptr<Position>(),
      table.data_ptr<float>(),
      output.data_ptr<torch::BFloat16>(),
      heads,
      table.size(1),
      table.size(0),
      inverse);
}

}  // namespace

torch::Tensor test_rope_qdq(const torch::Tensor& input,
                            const torch::Tensor& positions,
                            const torch::Tensor& table,
                            bool inverse) {
  CHECK(input.defined() && positions.defined() && table.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK(input.dim() == 2 || input.dim() == 3);
  CHECK_EQ(input.size(-1), 512);
  CHECK(input.is_contiguous() && positions.is_contiguous() &&
        table.is_contiguous());
  CHECK_EQ(input.device(), positions.device());
  CHECK_EQ(input.device(), table.device());
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(positions.numel(), input.size(0));
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  CHECK_EQ(table.scalar_type(), torch::kFloat32);
  CHECK_EQ(table.dim(), 2);
  CHECK_GT(table.size(0), 0);
  CHECK_GT(table.size(1), 0);
  CHECK_EQ(table.size(1) % 32, 0);
  CHECK_LE(table.size(1), input.size(-1));
  CHECK_LE(input.numel() / 512, std::numeric_limits<uint32_t>::max());
  torch::DeviceGuard guard(input.device());
  torch::Tensor output = torch::empty_like(input);
  if (input.numel() == 0) {
    return output;
  }
  if (positions.scalar_type() == torch::kInt32) {
    launch_rope_qdq<int32_t>(input, positions, table, inverse, output);
  } else {
    launch_rope_qdq<int64_t>(input, positions, table, inverse, output);
  }
  check_musa_kernel_launch(
      "V4.1 direct RoPE and original block32 QDQ candidate");
  return output;
}

}  // namespace xllm::kernel::musa
