/* Copyright 2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 */

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "core/kernels/musa/deepseek_v4_fp8_roundtrip.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int kRoundtripBlockSize = 256;

template <typename scalar_t>
__global__ void __launch_bounds__(kRoundtripBlockSize)
    deepseek_v4_fp8_roundtrip_kernel(scalar_t* __restrict__ values,
                                     int64_t rows,
                                     int64_t row_stride,
                                     int64_t quant_dim,
                                     int64_t block_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const int64_t num_groups = (quant_dim + block_size - 1) / block_size;
  __shared__ int max_abs_bits[32];
  __shared__ float scales[32];
  if (threadIdx.x < num_groups) {
    max_abs_bits[threadIdx.x] = 0;
  }
  __syncthreads();

  scalar_t* row_values = values + row * row_stride;
  for (int64_t col = threadIdx.x; col < quant_dim; col += blockDim.x) {
    const float value = static_cast<float>(row_values[col]);
    atomicMax(&max_abs_bits[col / block_size], __float_as_int(fabsf(value)));
  }
  __syncthreads();

  if (threadIdx.x < num_groups) {
    const float max_abs = __int_as_float(max_abs_bits[threadIdx.x]);
    const float raw_scale = fmaxf(max_abs / 448.0f, 1.0e-4f);
    scales[threadIdx.x] = exp2f(ceilf(log2f(raw_scale)));
  }
  __syncthreads();

  for (int64_t col = threadIdx.x; col < quant_dim; col += blockDim.x) {
    const int64_t group = col / block_size;
    float quantized = static_cast<float>(row_values[col]) / scales[group];
    quantized = fmaxf(-448.0f, fminf(quantized, 448.0f));
    const c10::Float8_e4m3fn fp8_value(quantized);
    row_values[col] =
        static_cast<scalar_t>(static_cast<float>(fp8_value) * scales[group]);
  }
}

template <typename scalar_t>
void launch_deepseek_v4_fp8_roundtrip(torch::Tensor& tensor,
                                      int64_t rows,
                                      int64_t row_stride,
                                      int64_t quant_dim,
                                      int64_t block_size) {
  deepseek_v4_fp8_roundtrip_kernel<scalar_t>
      <<<static_cast<unsigned int>(rows),
         kRoundtripBlockSize,
         0,
         current_musa_stream(tensor.device())>>>(
          tensor.data_ptr<scalar_t>(), rows, row_stride, quant_dim, block_size);
}

template <typename scalar_t>
__global__ void __launch_bounds__(kRoundtripBlockSize)
    deepseek_v4_int8_roundtrip_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      int64_t rows,
                                      int64_t row_stride) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float max_abs[kRoundtripBlockSize];
  const scalar_t* row_input = input + row * row_stride;
  scalar_t* row_output = output + row * row_stride;

  float local_max = 0.0f;
  for (int64_t col = threadIdx.x; col < row_stride; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(static_cast<float>(row_input[col])));
  }
  max_abs[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      max_abs[threadIdx.x] =
          fmaxf(max_abs[threadIdx.x], max_abs[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  // Match the native path: an all-zero row uses safe_max=1, while non-zero
  // rows retain their exact amax/127 scale.
  const float safe_max = max_abs[0] > 0.0f ? max_abs[0] : 1.0f;
  const float scale = safe_max * (1.0f / 127.0f);
  for (int64_t col = threadIdx.x; col < row_stride; col += blockDim.x) {
    const float value = static_cast<float>(row_input[col]);
    float quantized = nearbyintf(value / scale);
    quantized = fmaxf(-128.0f, fminf(quantized, 127.0f));
    row_output[col] = static_cast<scalar_t>(quantized * scale);
  }
}

template <typename scalar_t>
void launch_deepseek_v4_int8_roundtrip(const torch::Tensor& input,
                                       torch::Tensor& output,
                                       int64_t rows,
                                       int64_t row_stride) {
  deepseek_v4_int8_roundtrip_kernel<scalar_t>
      <<<static_cast<unsigned int>(rows),
         kRoundtripBlockSize,
         0,
         current_musa_stream(input.device())>>>(input.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(),
                                                rows,
                                                row_stride);
}

}  // namespace

bool deepseek_v4_fp8_quantize_dequantize(torch::Tensor& tensor,
                                         int64_t quant_dim,
                                         int64_t block_size) {
  if (!tensor.defined() || tensor.device().is_cpu() ||
      !tensor.is_contiguous() || tensor.dim() < 1 || quant_dim <= 0 ||
      block_size <= 0 || quant_dim % block_size != 0 ||
      tensor.size(-1) < quant_dim) {
    return false;
  }

  const int64_t last_dim = tensor.size(-1);
  const int64_t rows = tensor.numel() / last_dim;
  if (rows <= 0 || rows > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(tensor));
  switch (tensor.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_deepseek_v4_fp8_roundtrip<c10::BFloat16>(
          tensor, rows, last_dim, quant_dim, block_size);
      break;
    case at::ScalarType::Half:
      launch_deepseek_v4_fp8_roundtrip<c10::Half>(
          tensor, rows, last_dim, quant_dim, block_size);
      break;
    case at::ScalarType::Float:
      launch_deepseek_v4_fp8_roundtrip<float>(
          tensor, rows, last_dim, quant_dim, block_size);
      break;
    default:
      return false;
  }
  check_musa_kernel_launch("deepseek_v4_fp8_quantize_dequantize");
  return true;
}

bool deepseek_v4_int8_quantize_dequantize(const torch::Tensor& input,
                                          torch::Tensor& output) {
  if (!input.defined() || input.device().is_cpu() || !input.is_contiguous() ||
      input.dim() < 1 || input.numel() == 0) {
    return false;
  }

  const int64_t row_stride = input.size(-1);
  const int64_t rows = input.numel() / row_stride;
  if (row_stride <= 0 || rows <= 0 ||
      rows > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      output = torch::empty_like(input);
      launch_deepseek_v4_int8_roundtrip<c10::BFloat16>(
          input, output, rows, row_stride);
      break;
    case at::ScalarType::Half:
      output = torch::empty_like(input);
      launch_deepseek_v4_int8_roundtrip<c10::Half>(
          input, output, rows, row_stride);
      break;
    case at::ScalarType::Float:
      output = torch::empty_like(input);
      launch_deepseek_v4_int8_roundtrip<float>(input, output, rows, row_stride);
      break;
    default:
      return false;
  }
  check_musa_kernel_launch("deepseek_v4_int8_quantize_dequantize");
  return true;
}

}  // namespace xllm::kernel::musa
