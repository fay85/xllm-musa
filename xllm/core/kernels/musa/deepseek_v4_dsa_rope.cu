/* Copyright 2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 */

#include <ATen/Dispatch.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#include "core/kernels/musa/deepseek_v4_dsa_rope.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int kRopeBlockSize = 256;

template <typename scalar_t>
__global__ void __launch_bounds__(kRopeBlockSize)
    deepseek_v4_partial_rope_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ cos,
                                    const scalar_t* __restrict__ sin,
                                    scalar_t* __restrict__ output,
                                    int64_t rows,
                                    int64_t heads,
                                    int64_t hidden_dim,
                                    int64_t rope_start_dim,
                                    int64_t rope_head_dim,
                                    int64_t cos_stride,
                                    bool inverse) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const int64_t token = row / heads;
  const scalar_t* input_row = input + row * hidden_dim;
  scalar_t* output_row = output + row * hidden_dim;
  const scalar_t* cos_row = cos + token * cos_stride;
  const scalar_t* sin_row = sin + token * cos_stride;

  // One thread owns each non-rotary element and each rotary pair. This
  // avoids races while writing the even/odd pair and keeps the whole output
  // in one contiguous allocation.
  for (int64_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    if (col < rope_start_dim || col >= rope_start_dim + rope_head_dim) {
      output_row[col] = input_row[col];
      continue;
    }
    const int64_t local = col - rope_start_dim;
    if ((local & 1) != 0) {
      continue;
    }
    const int64_t odd_col = col + 1;
    const int64_t pair_offset = local;
    const float x = static_cast<float>(input_row[col]);
    const float y = static_cast<float>(input_row[odd_col]);
    const float c = static_cast<float>(cos_row[pair_offset]);
    const float s = inverse ? -static_cast<float>(sin_row[pair_offset])
                            : static_cast<float>(sin_row[pair_offset]);
    output_row[col] = static_cast<scalar_t>(x * c - y * s);
    output_row[odd_col] = static_cast<scalar_t>(y * c + x * s);
  }
}

template <typename scalar_t>
void launch_deepseek_v4_partial_rope(const torch::Tensor& input,
                                     const torch::Tensor& cos,
                                     const torch::Tensor& sin,
                                     int64_t rows,
                                     int64_t heads,
                                     int64_t hidden_dim,
                                     int64_t rope_start_dim,
                                     int64_t rope_head_dim,
                                     int64_t cos_stride,
                                     bool inverse,
                                     torch::Tensor& output) {
  deepseek_v4_partial_rope_kernel<scalar_t>
      <<<static_cast<unsigned int>(rows),
         kRopeBlockSize,
         0,
         current_musa_stream(input.device())>>>(input.data_ptr<scalar_t>(),
                                                cos.data_ptr<scalar_t>(),
                                                sin.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(),
                                                rows,
                                                heads,
                                                hidden_dim,
                                                rope_start_dim,
                                                rope_head_dim,
                                                cos_stride,
                                                inverse);
}

}  // namespace

bool deepseek_v4_partial_rope(const torch::Tensor& input,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin,
                              int64_t rope_start_dim,
                              int64_t rope_head_dim,
                              bool inverse,
                              torch::Tensor& output) {
  if (!input.defined() || !cos.defined() || !sin.defined() ||
      input.device().is_cpu() || input.dim() < 2 || input.dim() > 3 ||
      !input.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() ||
      input.scalar_type() != cos.scalar_type() ||
      input.scalar_type() != sin.scalar_type() || cos.dim() != 2 ||
      sin.dim() != 2 || cos.sizes() != sin.sizes() || rope_start_dim < 0 ||
      rope_head_dim <= 0 || (rope_head_dim & 1) != 0 ||
      input.size(-1) < rope_start_dim + rope_head_dim ||
      cos.size(0) != input.size(0) || sin.size(0) != input.size(0) ||
      cos.size(1) < rope_head_dim || sin.size(1) < rope_head_dim ||
      input.device() != cos.device() || input.device() != sin.device()) {
    return false;
  }

  const int64_t tokens = input.size(0);
  const int64_t hidden_dim = input.size(-1);
  const int64_t heads = input.dim() == 3 ? input.size(1) : 1;
  const int64_t rows = tokens * heads;
  if (rows <= 0 || rows > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  // Whole-graph decode cannot allocate while the stream is capturing.  Let
  // callers provide stable storage, while preserving the original allocating
  // behavior for all existing call sites.
  if (!output.defined() || output.sizes() != input.sizes() ||
      output.scalar_type() != input.scalar_type() ||
      output.device() != input.device() || !output.is_contiguous()) {
    output = torch::empty(input.sizes(), input.options());
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
                                  at::ScalarType::BFloat16,
                                  input.scalar_type(),
                                  "deepseek_v4_partial_rope",
                                  [&] {
                                    launch_deepseek_v4_partial_rope<scalar_t>(
                                        input,
                                        cos,
                                        sin,
                                        rows,
                                        heads,
                                        hidden_dim,
                                        rope_start_dim,
                                        rope_head_dim,
                                        cos.size(1),
                                        inverse,
                                        output);
                                  });
  check_musa_kernel_launch("deepseek_v4_partial_rope");
  return true;
}

}  // namespace xllm::kernel::musa
