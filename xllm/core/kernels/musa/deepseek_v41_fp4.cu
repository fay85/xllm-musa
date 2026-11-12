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

#include "core/kernels/musa/deepseek_v41_fp4_accum.cuh"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

// Official act_quant: max(abs(x), 1e-4) / 448, then ceil to a power of two.
// V4's group128 quantizer clamps AFTER division, so it cannot be reused as is.
__global__ void block32_qdq_kernel(const c10::BFloat16* input,
                                   c10::BFloat16* output,
                                   float* scales,
                                   int64_t groups) {
  const int32_t lane = threadIdx.x;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.y;
  if (group >= groups) {
    return;
  }
  const int64_t index = group * 32 + lane;
  const float value = static_cast<float>(input[index]);
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
  const c10::Float8_e4m3fn quantized =
      scaled_fp8_conversion<false, c10::Float8_e4m3fn>(value, scale);
  output[index] =
      static_cast<c10::BFloat16>(static_cast<float>(quantized) * scale);
  if (lane == 0) {
    scales[group] = scale;
  }
}

__device__ __forceinline__ float fp4_value(uint8_t nibble) {
  const int32_t magnitude = nibble & 7;
  const float positive = magnitude < 2
                             ? static_cast<float>(magnitude) * 0.5f
                             : ldexpf(static_cast<float>(2 + (magnitude & 1)),
                                      (magnitude >> 1) - 2);
  return (nibble & 8) ? -positive : positive;
}

// Produce the exact original group32 quantization directly in BMM batch
// layout [K/32, M, 32], with scales [K/32, M, 1].
__global__ void block32_batch_quant_kernel(const c10::BFloat16* input,
                                           c10::Float8_e4m3fn* output,
                                           float* scales,
                                           int64_t rows,
                                           int64_t groups_per_row) {
  const int32_t lane = threadIdx.x;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.y;
  if (group >= rows * groups_per_row) {
    return;
  }
  const float value = static_cast<float>(input[group * 32 + lane]);
  float maximum = fabsf(value);
#pragma unroll
  for (int32_t delta = 16; delta > 0; delta /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
  }
  const float raw = fmaxf(maximum, 1.0e-4f) * (1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(raw);
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 255u) - 127 +
                           ((bits & 0x7fffffu) != 0);
  const float scale = ldexpf(1.0f, exponent);
  const int64_t destination =
      (group % groups_per_row) * rows + group / groups_per_row;
  output[destination * 32 + lane] =
      scaled_fp8_conversion<false, c10::Float8_e4m3fn>(value, scale);
  if (lane == 0) {
    scales[destination] = scale;
  }
}

// E2M1 mantissas are exactly representable in E4M3. Preserve the original
// scale for every group; this is expansion, not re-quantization of weights.
__global__ void block32_batch_unpack_kernel(const uint8_t* packed,
                                            const uint8_t* input_scales,
                                            c10::Float8_e4m3fn* output,
                                            float* scales,
                                            int64_t rows,
                                            int64_t groups_per_row) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * groups_per_row * 32) {
    return;
  }
  const int64_t lane = index % 32;
  const int64_t row = (index / 32) % rows;
  const int64_t group = index / (rows * 32);
  const int64_t k = group * 32 + lane;
  const uint8_t byte = packed[row * groups_per_row * 16 + k / 2];
  const uint8_t nibble = (lane & 1) ? byte >> 4 : byte & 15;
  output[index] =
      QuantConverter<c10::Float8_e4m3fn>::convert(fp4_value(nibble));
  if (lane == 0) {
    const uint8_t raw = input_scales[row * groups_per_row + group];
    scales[group * rows + row] = raw == 255 ? NAN : ldexpf(1.0f, raw - 127);
  }
}

// The K-partition BMM returns FP32 per-group partials. Compensate the final
// accumulation before the only BF16 output cast, including cancellation.
template <bool ApplyScales>
__global__ void block32_batch_reduce_kernel(const float* partial,
                                            const float* input_scales,
                                            const float* weight_scales,
                                            c10::BFloat16* output,
                                            int64_t groups,
                                            int64_t elements,
                                            int64_t rows,
                                            int64_t columns) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  float sum = 0.0f;
  float correction = 0.0f;
  for (int64_t group = 0; group < groups; ++group) {
    float value = partial[group * elements + index];
    if constexpr (ApplyScales) {
      value *= input_scales[group * rows + index / columns] *
               weight_scales[group * columns + index % columns];
    }
    const float adjusted = value - correction;
    const float next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  output[index] = static_cast<c10::BFloat16>(sum - correction);
}

// Small-M memory path: consume ORIGINAL packed E2M1 and group32 UE8M0 scales.
// This is a BF16 x dequantized-FP4 vector kernel, not tensor-core FP4 GEMM.
__global__ void packed_fp4_gemv_kernel(const c10::BFloat16* input,
                                       const uint8_t* packed,
                                       const uint8_t* scales,
                                       c10::BFloat16* output,
                                       int64_t rows,
                                       int64_t columns,
                                       int64_t inner) {
  const int32_t lane = threadIdx.x;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.y;
  const int64_t row = blockIdx.y;
  if (column >= columns || row >= rows) {
    return;
  }
  V41CompensatedAccumulator accumulator;
  for (int64_t k = lane; k < inner; k += 32) {
    const uint8_t byte = packed[column * (inner / 2) + k / 2];
    const uint8_t nibble = (k & 1) ? byte >> 4 : byte & 15;
    const uint8_t raw_scale = scales[column * (inner / 32) + k / 32];
    // UE8M0 byte zero encodes 2^-127, not zero. FF encodes NaN.
    const float scale = raw_scale == 255 ? NAN : ldexpf(1.0f, raw_scale - 127);
    const c10::BFloat16 weight =
        static_cast<c10::BFloat16>(fp4_value(nibble) * scale);
    accumulator.add(static_cast<float>(input[row * inner + k]) *
                    static_cast<float>(weight));
  }
  const float sum = accumulator.warp_sum();
  if (lane == 0) {
    output[row * columns + column] = static_cast<c10::BFloat16>(sum);
  }
}

void check_input(const torch::Tensor& input) {
  CHECK(input.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(input.dim(), 2);
  CHECK_GT(input.size(1), 0);
  CHECK_EQ(input.size(1) % 32, 0);
  CHECK(input.is_contiguous());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_block32_qdq(
    const torch::Tensor& input) {
  check_input(input);
  torch::DeviceGuard guard(input.device());
  torch::Tensor output = torch::empty_like(input);
  torch::Tensor scales = torch::empty({input.size(0), input.size(1) / 32},
                                      input.options().dtype(torch::kFloat32));
  const int64_t groups = input.numel() / 32;
  if (groups == 0) {
    return {output, scales};
  }
  block32_qdq_kernel<<<static_cast<uint32_t>((groups + 3) / 4),
                       dim3(32, 4),
                       0,
                       current_musa_stream(input.device())>>>(
      input.data_ptr<c10::BFloat16>(),
      output.data_ptr<c10::BFloat16>(),
      scales.data_ptr<float>(),
      groups);
  check_musa_kernel_launch("V4.1 block32 activation QDQ");
  return {output, scales};
}

torch::Tensor deepseek_v41_packed_fp4_gemv(const torch::Tensor& input,
                                           const torch::Tensor& packed,
                                           const torch::Tensor& scales) {
  check_input(input);
  CHECK_EQ(packed.device(), input.device());
  CHECK_EQ(scales.device(), input.device());
  CHECK_EQ(packed.scalar_type(), torch::kUInt8);
  CHECK_EQ(scales.scalar_type(), torch::kUInt8);
  CHECK_EQ(packed.dim(), 2);
  CHECK_EQ(scales.dim(), 2);
  CHECK(packed.is_contiguous() && scales.is_contiguous());
  CHECK_EQ(packed.size(1) * 2, input.size(1));
  CHECK_EQ(scales.size(0), packed.size(0));
  CHECK_EQ(scales.size(1) * 32, input.size(1));
  CHECK_GT(packed.size(0), 0);
  CHECK_LE(input.size(0), 8) << "Packed GEMV is an explicit small-M path";
  torch::DeviceGuard guard(input.device());
  torch::Tensor output =
      torch::empty({input.size(0), packed.size(0)}, input.options());
  if (input.size(0) == 0) {
    return output;
  }
  packed_fp4_gemv_kernel<<<dim3(static_cast<uint32_t>((packed.size(0) + 3) / 4),
                                static_cast<uint32_t>(input.size(0))),
                           dim3(32, 4),
                           0,
                           current_musa_stream(input.device())>>>(
      input.data_ptr<c10::BFloat16>(),
      packed.data_ptr<uint8_t>(),
      scales.data_ptr<uint8_t>(),
      output.data_ptr<c10::BFloat16>(),
      input.size(0),
      packed.size(0),
      input.size(1));
  check_musa_kernel_launch("V4.1 packed FP4 GEMV");
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_block32_batch_quant(
    const torch::Tensor& input) {
  check_input(input);
  torch::DeviceGuard guard(input.device());
  const int64_t rows = input.size(0);
  const int64_t groups = input.size(1) / 32;
  torch::Tensor quantized = torch::empty(
      {groups, rows, 32}, input.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor scales =
      torch::empty({groups, rows, 1}, input.options().dtype(torch::kFloat32));
  if (rows == 0) {
    return {quantized, scales};
  }
  block32_batch_quant_kernel<<<static_cast<uint32_t>((rows * groups + 3) / 4),
                               dim3(32, 4),
                               0,
                               current_musa_stream(input.device())>>>(
      input.data_ptr<c10::BFloat16>(),
      quantized.data_ptr<c10::Float8_e4m3fn>(),
      scales.data_ptr<float>(),
      rows,
      groups);
  check_musa_kernel_launch("V4.1 group32 batch quantization");
  return {quantized, scales};
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_block32_batch_unpack(
    const torch::Tensor& packed,
    const torch::Tensor& input_scales) {
  CHECK(packed.defined() && input_scales.defined());
  CHECK(packed.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input_scales.device(), packed.device());
  CHECK_EQ(packed.scalar_type(), torch::kUInt8);
  CHECK_EQ(input_scales.scalar_type(), torch::kUInt8);
  CHECK_EQ(packed.dim(), 2);
  CHECK_EQ(input_scales.dim(), 2);
  CHECK(packed.is_contiguous() && input_scales.is_contiguous());
  CHECK_GT(packed.size(0), 0);
  CHECK_GT(packed.size(1), 0);
  CHECK_EQ(packed.size(1) % 16, 0);
  CHECK_EQ(input_scales.size(0), packed.size(0));
  CHECK_EQ(input_scales.size(1) * 16, packed.size(1));
  torch::DeviceGuard guard(packed.device());
  const int64_t rows = packed.size(0);
  const int64_t groups = input_scales.size(1);
  torch::Tensor values = torch::empty(
      {groups, rows, 32}, packed.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor scales =
      torch::empty({groups, rows, 1}, packed.options().dtype(torch::kFloat32));
  block32_batch_unpack_kernel<<<static_cast<uint32_t>((values.numel() + 255) /
                                                      256),
                                256,
                                0,
                                current_musa_stream(packed.device())>>>(
      packed.data_ptr<uint8_t>(),
      input_scales.data_ptr<uint8_t>(),
      values.data_ptr<c10::Float8_e4m3fn>(),
      scales.data_ptr<float>(),
      rows,
      groups);
  check_musa_kernel_launch("V4.1 group32 packed weight expansion");
  return {values, scales};
}

torch::Tensor deepseek_v41_block32_batch_reduce(const torch::Tensor& partial) {
  CHECK(partial.defined());
  CHECK(partial.device().type() == torch::kPrivateUse1);
  CHECK_EQ(partial.scalar_type(), torch::kFloat32);
  CHECK_EQ(partial.dim(), 3);
  CHECK_GT(partial.size(0), 0);
  CHECK(partial.is_contiguous());
  torch::DeviceGuard guard(partial.device());
  torch::Tensor output =
      torch::empty({partial.size(1), partial.size(2)},
                   partial.options().dtype(torch::kBFloat16));
  if (output.numel() == 0) {
    return output;
  }
  block32_batch_reduce_kernel<false>
      <<<static_cast<uint32_t>((output.numel() + 255) / 256),
         256,
         0,
         current_musa_stream(partial.device())>>>(
          partial.data_ptr<float>(),
          nullptr,
          nullptr,
          output.data_ptr<c10::BFloat16>(),
          partial.size(0),
          output.numel(),
          partial.size(1),
          partial.size(2));
  check_musa_kernel_launch("V4.1 group32 partition reduction");
  return output;
}

torch::Tensor deepseek_v41_block32_batch_reduce_scaled(
    const torch::Tensor& partial,
    const torch::Tensor& input_scales,
    const torch::Tensor& weight_scales) {
  CHECK(partial.defined() && input_scales.defined() && weight_scales.defined());
  CHECK(partial.device().type() == torch::kPrivateUse1);
  CHECK_EQ(partial.scalar_type(), torch::kFloat32);
  CHECK_EQ(input_scales.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scales.scalar_type(), torch::kFloat32);
  CHECK_EQ(input_scales.device(), partial.device());
  CHECK_EQ(weight_scales.device(), partial.device());
  CHECK_EQ(partial.dim(), 3);
  CHECK_EQ(input_scales.dim(), 3);
  CHECK_EQ(weight_scales.dim(), 3);
  CHECK_GT(partial.size(0), 0);
  CHECK_EQ(input_scales.size(0), partial.size(0));
  CHECK_EQ(weight_scales.size(0), partial.size(0));
  CHECK_EQ(input_scales.size(1), partial.size(1));
  CHECK_EQ(weight_scales.size(1), partial.size(2));
  CHECK_EQ(input_scales.size(2), 1);
  CHECK_EQ(weight_scales.size(2), 1);
  CHECK(partial.is_contiguous() && input_scales.is_contiguous() &&
        weight_scales.is_contiguous());
  torch::DeviceGuard guard(partial.device());
  torch::Tensor output =
      torch::empty({partial.size(1), partial.size(2)},
                   partial.options().dtype(torch::kBFloat16));
  if (output.numel() == 0) {
    return output;
  }
  block32_batch_reduce_kernel<true>
      <<<static_cast<uint32_t>((output.numel() + 255) / 256),
         256,
         0,
         current_musa_stream(partial.device())>>>(
          partial.data_ptr<float>(),
          input_scales.data_ptr<float>(),
          weight_scales.data_ptr<float>(),
          output.data_ptr<c10::BFloat16>(),
          partial.size(0),
          output.numel(),
          partial.size(1),
          partial.size(2));
  check_musa_kernel_launch("V4.1 group32 scaled partition reduction");
  return output;
}

}  // namespace xllm::kernel::musa
