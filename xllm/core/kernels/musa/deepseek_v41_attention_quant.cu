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
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

#include "core/kernels/musa/deepseek_v41_attention_quant.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

__device__ __forceinline__ uint8_t fp4_round_even(float value, float scale) {
  const float magnitude = fabsf(value);
  // Compare in input units. A reciprocal-based division can perturb exact
  // midpoints for E4M3 scales (for example -75/60 became slightly <-1.25).
  // Each threshold*scale is exact in FP32 for a finite E4M3/UE8M0 scale.
  // E2M1 codes are monotone in magnitude. At a midpoint, choose the even
  // code (the significand's retained low bit), including signed zero.
  uint8_t code = 0;
  code += magnitude > 0.25f * scale;
  code += magnitude >= 0.75f * scale;
  code += magnitude > 1.25f * scale;
  code += magnitude >= 1.75f * scale;
  code += magnitude > 2.5f * scale;
  code += magnitude >= 3.5f * scale;
  code += magnitude > 5.0f * scale;
  return code | (signbit(value) ? 8u : 0u);
}

__device__ __forceinline__ float fp4_decode(uint8_t code) {
  const int32_t magnitude = code & 7;
  const float value = magnitude < 2
                          ? static_cast<float>(magnitude) * 0.5f
                          : ldexpf(static_cast<float>(2 + (magnitude & 1)),
                                   (magnitude >> 1) - 2);
  return (code & 8) ? -value : value;
}

template <bool E4m3Scale, bool WithRope = false>
__global__ void attention_fp4_quant_kernel(const c10::BFloat16* input,
                                           c10::BFloat16* output,
                                           uint8_t* packed,
                                           uint8_t* scale_codes,
                                           int64_t group_count,
                                           int32_t group_size,
                                           const int64_t* positions = nullptr,
                                           const float* table = nullptr,
                                           int64_t heads = 0,
                                           int64_t head_dim = 0,
                                           int64_t rope_dim = 0) {
  const int32_t lane = threadIdx.x;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.y;
  if (group >= group_count) {
    return;
  }
  float value = lane < group_size
                    ? static_cast<float>(input[group * group_size + lane])
                    : 0.0f;
  if constexpr (WithRope) {
    const int64_t element = group * group_size + lane;
    const int64_t column = element % head_dim;
    if (lane < group_size && column >= head_dim - rope_dim) {
      const int64_t token = element / (heads * head_dim);
      const int64_t pair = (column - head_dim + rope_dim) / 2;
      const int64_t even = element - (column & 1);
      const float real = static_cast<float>(input[even]);
      const float imaginary = static_cast<float>(input[even + 1]);
      const int64_t frequency = positions[token] * rope_dim + pair;
      const float cosine = table[frequency];
      const float sine = table[frequency + rope_dim / 2];
      // Preserve the existing V4 partial-RoPE arithmetic and its BF16
      // boundary before the V4.1 delivery's group16/group32 quantization.
      const float rotated = (column & 1) ? imaginary * cosine + real * sine
                                         : real * cosine - imaginary * sine;
      value = static_cast<float>(static_cast<c10::BFloat16>(rotated));
    }
  }
  float maximum = fabsf(value);
#pragma unroll
  for (int32_t delta = 16; delta > 0; delta /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
  }
  float scale;
  uint8_t stored_scale;
  if constexpr (E4m3Scale) {
    const float raw =
        fminf(fmaxf(maximum, 6.0f * 0x1p-9f) * (1.0f / 6.0f), 448.0f);
    // Match the reference E4M3 cast, including its subnormal scale range.
    const c10::Float8_e4m3fn rounded(raw);
    scale = static_cast<float>(rounded);
    stored_scale = rounded.x;
  } else {
    const float raw = fmaxf(maximum, 6.0f * 0x1p-126f) * (1.0f / 6.0f);
    const uint32_t bits = __float_as_uint(raw);
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 255u) - 127 +
                             ((bits & 0x7fffffu) != 0);
    scale = ldexpf(1.0f, exponent);
    stored_scale = static_cast<uint8_t>(exponent + 127);
  }
  const uint8_t code = fp4_round_even(value, scale);
  const uint8_t next_code = static_cast<uint8_t>(
      __shfl_down_sync(0xffffffffu, static_cast<int32_t>(code), 1));
  if (lane < group_size) {
    output[group * group_size + lane] =
        static_cast<c10::BFloat16>(fp4_decode(code) * scale);
    if (packed != nullptr && (lane & 1) == 0) {
      packed[(group * group_size + lane) / 2] = code | (next_code << 4);
    }
  }
  if (scale_codes != nullptr && lane == 0) {
    scale_codes[group] = stored_scale;
  }
}

}  // namespace

// Produce QDQ BF16 values plus exact original packed bytes and scale bytes.
// This primitive does not choose or mutate any attention cache format.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_attention_fp4_quant(const torch::Tensor& input,
                                 int64_t group_size,
                                 bool e4m3_scale) {
  CHECK(input.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(input.dim(), 2);
  CHECK(input.is_contiguous());
  CHECK(group_size == 16 || group_size == 32);
  CHECK_GT(input.size(1), 0);
  CHECK_EQ(input.size(1) % group_size, 0);
  torch::DeviceGuard guard(input.device());
  torch::Tensor output = torch::empty_like(input);
  torch::Tensor packed = torch::empty({input.size(0), input.size(1) / 2},
                                      input.options().dtype(torch::kUInt8));
  torch::Tensor scale_codes =
      torch::empty({input.size(0), input.size(1) / group_size},
                   input.options().dtype(torch::kUInt8));
  const int64_t group_count = input.numel() / group_size;
  if (group_count == 0) {
    return {output, packed, scale_codes};
  }
  const dim3 grid(static_cast<uint32_t>((group_count + 3) / 4));
  const dim3 block(32, 4);
  if (e4m3_scale) {
    attention_fp4_quant_kernel<true>
        <<<grid, block, 0, current_musa_stream(input.device())>>>(
            input.data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(),
            packed.data_ptr<uint8_t>(),
            scale_codes.data_ptr<uint8_t>(),
            group_count,
            static_cast<int32_t>(group_size));
  } else {
    attention_fp4_quant_kernel<false>
        <<<grid, block, 0, current_musa_stream(input.device())>>>(
            input.data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(),
            packed.data_ptr<uint8_t>(),
            scale_codes.data_ptr<uint8_t>(),
            group_count,
            static_cast<int32_t>(group_size));
  }
  check_musa_kernel_launch("V4.1 attention original FP4 quantization");
  return {output, packed, scale_codes};
}

torch::Tensor deepseek_v41_rope_fp4_quant(const torch::Tensor& input,
                                          const torch::Tensor& positions,
                                          const torch::Tensor& table,
                                          int64_t group_size,
                                          bool e4m3_scale) {
  CHECK(input.defined() && positions.defined() && table.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.device(), positions.device());
  CHECK_EQ(input.device(), table.device());
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK(input.dim() == 2 || input.dim() == 3);
  CHECK(input.is_contiguous() && table.is_contiguous());
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(positions.size(0), input.size(0));
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(table.scalar_type(), torch::kFloat32);
  CHECK_GT(table.size(0), 0);
  CHECK_GT(table.size(1), 0);
  CHECK_EQ(table.size(1) % 2, 0);
  CHECK_LE(table.size(1), input.size(-1));
  CHECK(group_size == 16 || group_size == 32);
  CHECK_EQ(input.size(-1) % group_size, 0);
  torch::DeviceGuard guard(input.device());
  torch::Tensor output = torch::empty_like(input);
  const int64_t group_count = input.numel() / group_size;
  if (group_count == 0) {
    return output;
  }
  const torch::Tensor token_positions =
      positions.to(torch::kInt64).contiguous();
  const int64_t heads = input.dim() == 3 ? input.size(1) : 1;
  const dim3 grid(static_cast<uint32_t>((group_count + 3) / 4));
  const dim3 block(32, 4);
  if (e4m3_scale) {
    attention_fp4_quant_kernel<true, true>
        <<<grid, block, 0, current_musa_stream(input.device())>>>(
            input.data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(),
            nullptr,
            nullptr,
            group_count,
            static_cast<int32_t>(group_size),
            token_positions.data_ptr<int64_t>(),
            table.data_ptr<float>(),
            heads,
            input.size(-1),
            table.size(1));
  } else {
    attention_fp4_quant_kernel<false, true>
        <<<grid, block, 0, current_musa_stream(input.device())>>>(
            input.data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(),
            nullptr,
            nullptr,
            group_count,
            static_cast<int32_t>(group_size),
            token_positions.data_ptr<int64_t>(),
            table.data_ptr<float>(),
            heads,
            input.size(-1),
            table.size(1));
  }
  check_musa_kernel_launch("V4.1 fused RoPE and original FP4 QDQ");
  return output;
}

}  // namespace xllm::kernel::musa
