/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <glog/logging.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

template <typename Position, int32_t BlockSize>
__global__ void __launch_bounds__(BlockSize)
    rope_table_kernel(const torch::BFloat16* __restrict__ input,
                      const Position* __restrict__ positions,
                      const float* __restrict__ table,
                      torch::BFloat16* __restrict__ output,
                      int64_t heads,
                      int64_t hidden,
                      int64_t rotary,
                      int64_t table_rows,
                      bool inverse,
                      int32_t arithmetic) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t token = row / heads;
  const int64_t position = static_cast<int64_t>(positions[token]);
  CUDA_KERNEL_ASSERT(position >= 0 && position < table_rows);
  const int64_t start = hidden - rotary;
  const float* values = table + position * rotary;
  const torch::BFloat16* input_row = input + row * hidden;
  torch::BFloat16* output_row = output + row * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += BlockSize) {
    if (column < start) {
      output_row[column] = input_row[column];
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
    // Preserve the existing FP32 pair arithmetic and final BF16 conversion.
    float even;
    float odd;
    if (arithmetic == 0) {
      even = x * c - y * s;
      odd = y * c + x * s;
    } else if (arithmetic == 1) {
      even = __fsub_rn(__fmul_rn(x, c), __fmul_rn(y, s));
      odd = __fadd_rn(__fmul_rn(y, c), __fmul_rn(x, s));
    } else {
      const bool even_x = arithmetic == 2 || arithmetic == 4;
      const bool odd_y = arithmetic == 2 || arithmetic == 5;
      even = even_x ? __fmaf_rn(x, c, -__fmul_rn(y, s))
                    : __fmaf_rn(-y, s, __fmul_rn(x, c));
      odd = odd_y ? __fmaf_rn(y, c, __fmul_rn(x, s))
                  : __fmaf_rn(x, s, __fmul_rn(y, c));
    }
    output_row[column] = static_cast<torch::BFloat16>(even);
    output_row[column + 1] = static_cast<torch::BFloat16>(odd);
  }
}

template <typename Position, int32_t BlockSize>
void launch_rope_table(const torch::Tensor& input,
                       const torch::Tensor& positions,
                       const torch::Tensor& table,
                       bool inverse,
                       int32_t arithmetic,
                       torch::Tensor& output) {
  const int64_t heads = input.dim() == 3 ? input.size(1) : 1;
  rope_table_kernel<Position, BlockSize>
      <<<static_cast<unsigned int>(input.size(0) * heads),
         BlockSize,
         0,
         current_musa_stream(input.device())>>>(
          input.data_ptr<torch::BFloat16>(),
          positions.data_ptr<Position>(),
          table.data_ptr<float>(),
          output.data_ptr<torch::BFloat16>(),
          heads,
          input.size(-1),
          table.size(1),
          table.size(0),
          inverse,
          arithmetic);
}

}  // namespace

torch::Tensor deepseek_v41_rope_table_candidate(const torch::Tensor& input,
                                                const torch::Tensor& positions,
                                                const torch::Tensor& table,
                                                bool inverse,
                                                int32_t block_size,
                                                int32_t arithmetic) {
  CHECK(input.defined() && positions.defined() && table.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK(input.dim() == 2 || input.dim() == 3);
  CHECK(input.is_contiguous() && positions.is_contiguous() &&
        table.is_contiguous());
  CHECK_EQ(positions.device(), input.device());
  CHECK_EQ(table.device(), input.device());
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(positions.numel(), input.size(0));
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  CHECK_EQ(table.scalar_type(), torch::kFloat32);
  CHECK_EQ(table.dim(), 2);
  CHECK_GT(table.size(0), 0);
  CHECK_GT(table.size(1), 0);
  CHECK_EQ(table.size(1) % 2, 0);
  CHECK_LE(table.size(1), input.size(-1));
  CHECK(block_size == 128 || block_size == 256);
  CHECK_GE(arithmetic, 0);
  CHECK_LE(arithmetic, 5);
  torch::DeviceGuard guard(input.device());
  torch::Tensor output = torch::empty_like(input);
  if (input.numel() == 0) {
    return output;
  }
  const int64_t heads = input.dim() == 3 ? input.size(1) : 1;
  CHECK_GT(heads, 0);
  CHECK_LE(input.size(0) * heads, std::numeric_limits<unsigned int>::max());
  if (positions.scalar_type() == torch::kInt64) {
    if (block_size == 128) {
      launch_rope_table<int64_t, 128>(
          input, positions, table, inverse, arithmetic, output);
    } else {
      launch_rope_table<int64_t, 256>(
          input, positions, table, inverse, arithmetic, output);
    }
  } else {
    if (block_size == 128) {
      launch_rope_table<int32_t, 128>(
          input, positions, table, inverse, arithmetic, output);
    } else {
      launch_rope_table<int32_t, 256>(
          input, positions, table, inverse, arithmetic, output);
    }
  }
  check_musa_kernel_launch("deepseek_v41_rope_table_candidate");
  return output;
}

}  // namespace xllm::kernel::musa
