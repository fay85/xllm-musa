/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/

#include <glog/logging.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

#include "core/kernels/musa/deepseek_v41_engram_gate.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

// Port of the delivered SGLang V4.1 exact gate's 8x32 reduction geometry.
// Source SHA d3e944fc8edd3d967a9f6a42998455174b280b774e1af725775256de167f9b83.
// Preserve official (h * (qw * kw)) * key and FP32 product boundaries.
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset /= 2) {
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

template <typename Scalar, typename Weight>
__global__ void engram_gate_kernel(const Scalar* input,
                                   const Scalar* kv,
                                   const Weight* qw,
                                   const Weight* kw,
                                   const bool* mask,
                                   Scalar* output) {
  constexpr int64_t kDim = 5120;
  constexpr int64_t kCopies = 4;
  const int64_t row = blockIdx.x;
  const int64_t token = row / kCopies;
  const int64_t copy = row % kCopies;
  const int32_t thread = threadIdx.x;
  const int32_t lane = thread % 32;
  const int32_t warp = thread / 32;
  __shared__ float partial[3][8];
  __shared__ float coefficient;
  float sx = 0.0f;
  float sk = 0.0f;
  float sd = 0.0f;
#pragma unroll
  for (int32_t j = 0; j < 20; ++j) {
    const int64_t col = thread * 4 + (j / 4) * 1024 + j % 4;
    const float x = static_cast<float>(input[row * kDim + col]);
    const float key = static_cast<float>(
        kv[token * (kCopies + 1) * kDim + copy * kDim + col]);
    const float weight = __fmul_rn(static_cast<float>(qw[copy * kDim + col]),
                                   static_cast<float>(kw[copy * kDim + col]));
    sx = __fadd_rn(sx, __fmul_rn(x, x));
    sk = __fadd_rn(sk, __fmul_rn(key, key));
    sd = __fadd_rn(sd, __fmul_rn(__fmul_rn(x, weight), key));
  }
  sx = warp_sum(sx);
  sk = warp_sum(sk);
  sd = warp_sum(sd);
  if (lane == 0) {
    partial[0][warp] = sx;
    partial[1][warp] = sk;
    partial[2][warp] = sd;
  }
  __syncthreads();
  if (warp == 0) {
    sx = lane < 8 ? partial[0][lane] : 0.0f;
    sk = lane < 8 ? partial[1][lane] : 0.0f;
    sd = lane < 8 ? partial[2][lane] : 0.0f;
    sx = warp_sum(sx);
    sk = warp_sum(sk);
    sd = warp_sum(sd);
    if (lane == 0) {
      const float rx = rsqrtf(__fadd_rn(__fmul_rn(sx, 1.0f / 5120.0f), 1e-20f));
      const float rk = rsqrtf(__fadd_rn(__fmul_rn(sk, 1.0f / 5120.0f), 1e-20f));
      const float rstd = __fmul_rn(rx, rk);
      const float dot = __fmul_rn(__fmul_rn(sd, rstd), 0.013975424859373685f);
      const float signed_root = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
      const float gate = 1.0f / __fadd_rn(1.0f, __expf(-signed_root));
      coefficient = mask == nullptr || mask[token] ? gate : 0.0f;
    }
  }
  __syncthreads();
  for (int64_t col = thread; col < kDim; col += 256) {
    const float x = static_cast<float>(input[row * kDim + col]);
    const float value = static_cast<float>(
        kv[token * (kCopies + 1) * kDim + kCopies * kDim + col]);
    output[row * kDim + col] =
        static_cast<Scalar>(__fadd_rn(x, __fmul_rn(coefficient, value)));
  }
}

template <typename Scalar, typename Weight>
void launch_gate(const torch::Tensor& input,
                 const torch::Tensor& kv,
                 const torch::Tensor& qw,
                 const torch::Tensor& kw,
                 const std::optional<torch::Tensor>& mask,
                 torch::Tensor& output) {
  engram_gate_kernel<Scalar, Weight><<<static_cast<uint32_t>(input.size(0) * 4),
                                       256,
                                       0,
                                       current_musa_stream(input.device())>>>(
      input.data_ptr<Scalar>(),
      kv.data_ptr<Scalar>(),
      qw.data_ptr<Weight>(),
      kw.data_ptr<Weight>(),
      mask.has_value() ? mask->data_ptr<bool>() : nullptr,
      output.data_ptr<Scalar>());
}

}  // namespace

torch::Tensor deepseek_v41_engram_gate(
    const torch::Tensor& input,
    const torch::Tensor& kv,
    const torch::Tensor& qw,
    const torch::Tensor& kw,
    const std::optional<torch::Tensor>& mask) {
  CHECK(input.defined() && kv.defined() && qw.defined() && kw.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.dim(), 3);
  CHECK_EQ(input.size(1), 4);
  CHECK_EQ(input.size(2), 5120);
  CHECK_LE(input.size(0), 512);
  CHECK_EQ(kv.sizes(), torch::IntArrayRef({input.size(0), 25600}));
  CHECK_EQ(qw.sizes(), torch::IntArrayRef({4, 5120}));
  CHECK_EQ(kw.sizes(), qw.sizes());
  CHECK_EQ(kv.scalar_type(), input.scalar_type());
  CHECK_EQ(qw.scalar_type(), kw.scalar_type());
  CHECK(input.scalar_type() == torch::kBFloat16 ||
        input.scalar_type() == torch::kFloat32);
  CHECK(qw.scalar_type() == torch::kBFloat16 ||
        qw.scalar_type() == torch::kFloat32);
  CHECK_EQ(input.device(), kv.device());
  CHECK_EQ(input.device(), qw.device());
  CHECK_EQ(input.device(), kw.device());
  CHECK(input.is_contiguous() && kv.is_contiguous() && qw.is_contiguous() &&
        kw.is_contiguous());
  if (mask.has_value()) {
    CHECK_EQ(mask->device(), input.device());
    CHECK_EQ(mask->scalar_type(), torch::kBool);
    CHECK_EQ(mask->dim(), 1);
    CHECK_EQ(mask->size(0), input.size(0));
    CHECK(mask->is_contiguous());
  }
  torch::DeviceGuard guard(input.device());
  torch::Tensor output = torch::empty_like(input);
  if (input.size(0) == 0) {
    return output;
  }
  if (input.scalar_type() == torch::kBFloat16 &&
      qw.scalar_type() == torch::kBFloat16) {
    launch_gate<torch::BFloat16, torch::BFloat16>(
        input, kv, qw, kw, mask, output);
  } else if (input.scalar_type() == torch::kBFloat16) {
    launch_gate<torch::BFloat16, float>(input, kv, qw, kw, mask, output);
  } else if (qw.scalar_type() == torch::kBFloat16) {
    launch_gate<float, torch::BFloat16>(input, kv, qw, kw, mask, output);
  } else {
    launch_gate<float, float>(input, kv, qw, kw, mask, output);
  }
  check_musa_kernel_launch("deepseek_v41_engram_gate");
  return output;
}
}  // namespace xllm::kernel::musa
