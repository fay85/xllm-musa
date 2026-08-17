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

#include <ATen/Dispatch.h>
#include <glog/logging.h>

#include <cstdlib>
#include <mutex>
#include <string>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kMaxThreads = 256;
constexpr int32_t kWarpSize = 32;

int32_t gate_gemv_threads() { return kMaxThreads; }

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMaxThreads)
    deepseek_v4_gate_gemv_kernel(const scalar_t* __restrict__ input,
                                 const c10::BFloat16* __restrict__ weight,
                                 float* __restrict__ output,
                                 int64_t hidden_size,
                                 int64_t num_experts) {
  const int64_t expert = blockIdx.x;
  const int64_t token = blockIdx.y;
  const scalar_t* input_row = input + token * hidden_size;
  const c10::BFloat16* weight_row = weight + expert * hidden_size;

  float acc = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    acc += musa_convert_to_float(input_row[col]) *
           musa_convert_to_float(weight_row[col]);
  }

  acc = warp_sum(acc);
  __shared__ float warp_partials[kMaxThreads / kWarpSize];
  const int32_t lane = threadIdx.x % kWarpSize;
  const int32_t warp = threadIdx.x / kWarpSize;
  const int32_t num_warps = blockDim.x / kWarpSize;
  if (lane == 0) {
    warp_partials[warp] = acc;
  }
  __syncthreads();

  if (warp == 0) {
    acc = lane < num_warps ? warp_partials[lane] : 0.0f;
    acc = warp_sum(acc);
  }
  if (threadIdx.x == 0) {
    output[token * num_experts + expert] = acc;
  }
}

}  // namespace

torch::Tensor deepseek_v4_gate_gemv(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& output) {
  CHECK(input.defined() && weight.defined())
      << "deepseek_v4_gate_gemv expects input and weight.";
  CHECK_EQ(input.dim(), 2) << "deepseek_v4_gate_gemv input must be [T, H].";
  CHECK_EQ(weight.dim(), 2) << "deepseek_v4_gate_gemv weight must be [E, H].";
  CHECK(input.scalar_type() == torch::kBFloat16 ||
        input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kFloat32)
      << "deepseek_v4_gate_gemv input must be bf16, fp16, or fp32.";
  CHECK_EQ(weight.scalar_type(), torch::kBFloat16)
      << "deepseek_v4_gate_gemv weight must preserve checkpoint BF16.";
  CHECK_EQ(input.size(1), weight.size(1));
  CHECK_EQ(input.device(), weight.device());

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  const int64_t num_experts = weight.size(0);
  torch::Tensor result;
  if (output.has_value() && output.value().defined()) {
    result = output.value();
    CHECK_EQ(result.sizes(), torch::IntArrayRef({num_tokens, num_experts}));
    CHECK_EQ(result.scalar_type(), torch::kFloat32);
    CHECK_EQ(result.device(), input.device());
  } else {
    result = torch::empty({num_tokens, num_experts},
                          input.options().dtype(torch::kFloat32));
  }
  if (num_tokens == 0 || hidden_size == 0 || num_experts == 0) {
    return result;
  }

  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();
  auto result_contig = result.contiguous();
  const XLLMOptionalDeviceGuard device_guard(at::device_of(input_contig));
  const musaStream_t stream = current_musa_stream(input_contig.device());
  const dim3 grid(static_cast<unsigned int>(num_experts),
                  static_cast<unsigned int>(num_tokens));
  const dim3 block(gate_gemv_threads());
  if (input_contig.scalar_type() == torch::kBFloat16) {
    deepseek_v4_gate_gemv_kernel<c10::BFloat16>
        <<<grid, block, 0, stream>>>(input_contig.data_ptr<c10::BFloat16>(),
                                     weight_contig.data_ptr<c10::BFloat16>(),
                                     result_contig.data_ptr<float>(),
                                     hidden_size,
                                     num_experts);
  } else if (input_contig.scalar_type() == torch::kFloat16) {
    deepseek_v4_gate_gemv_kernel<c10::Half>
        <<<grid, block, 0, stream>>>(input_contig.data_ptr<c10::Half>(),
                                     weight_contig.data_ptr<c10::BFloat16>(),
                                     result_contig.data_ptr<float>(),
                                     hidden_size,
                                     num_experts);
  } else {
    deepseek_v4_gate_gemv_kernel<float>
        <<<grid, block, 0, stream>>>(input_contig.data_ptr<float>(),
                                     weight_contig.data_ptr<c10::BFloat16>(),
                                     result_contig.data_ptr<float>(),
                                     hidden_size,
                                     num_experts);
  }
  check_musa_kernel_launch("deepseek_v4_gate_gemv");
  if (!result.is_contiguous()) {
    result.copy_(result_contig);
  }
  return result;
}

}  // namespace xllm::kernel::musa
