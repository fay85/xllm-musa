/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#include "core/kernels/musa/kda_ops.h"
#include "core/kernels/musa/kda_validation.h"

namespace xllm::kernel::musa {
namespace {

// Each thread owns one value row of the V-first recurrent matrix. This
// correctness-first kernel preserves FP32 state without a BF16 state roundtrip.
__global__ void kda_decode_kernel(const torch::BFloat16* query,
                                  const torch::BFloat16* key,
                                  const torch::BFloat16* value,
                                  const torch::BFloat16* raw_forget,
                                  const torch::BFloat16* raw_beta,
                                  const float* a_log,
                                  const float* dt_bias,
                                  const float* initial_state,
                                  torch::BFloat16* output,
                                  float* final_state,
                                  int32_t heads,
                                  float scale,
                                  float gate_lower_bound) {
  __shared__ float q[128];
  __shared__ float k[128];
  __shared__ float decay[128];
  __shared__ float inv_q;
  __shared__ float inv_k;
  const int32_t channel = static_cast<int32_t>(threadIdx.x);
  const int64_t row = blockIdx.x;
  const int64_t head = row % heads;
  const int64_t position = row * 128 + channel;
  q[channel] = static_cast<float>(query[position]);
  k[channel] = static_cast<float>(key[position]);
  const float gate =
      expf(a_log[head]) * (static_cast<float>(raw_forget[position]) +
                           dt_bias[head * 128 + channel]);
  decay[channel] = expf(gate_lower_bound / (1.0f + expf(-gate)));
  __syncthreads();
  if (channel == 0) {
    float sum_q = 1e-6f;
    float sum_k = 1e-6f;
    for (int32_t d = 0; d < 128; ++d) {
      sum_q += q[d] * q[d];
      sum_k += k[d] * k[d];
    }
    inv_q = rsqrtf(sum_q) * scale;
    inv_k = rsqrtf(sum_k);
  }
  __syncthreads();
  q[channel] *= inv_q;
  k[channel] *= inv_k;
  __syncthreads();
  const int64_t state_row = (row * 128 + channel) * 128;
  float prediction = 0.0f;
  for (int32_t d = 0; d < 128; ++d) {
    prediction += initial_state[state_row + d] * decay[d] * k[d];
  }
  const float beta = 1.0f / (1.0f + expf(-static_cast<float>(raw_beta[row])));
  const float delta = (static_cast<float>(value[position]) - prediction) * beta;
  float result = 0.0f;
  for (int32_t d = 0; d < 128; ++d) {
    const float updated =
        initial_state[state_row + d] * decay[d] + delta * k[d];
    final_state[state_row + d] = updated;
    result += updated * q[d];
  }
  output[position] = static_cast<torch::BFloat16>(result);
}

}  // namespace

void kda_decode(const KdaInputs& inputs,
                torch::Tensor& output,
                torch::Tensor& final_state) {
  detail::validate_kda(inputs, output, final_state);
  CHECK_EQ(inputs.query.size(1), 1) << "KDA decode requires one token";
  const int64_t rows = inputs.query.size(0) * inputs.query.size(2);
  CHECK_LE(rows, std::numeric_limits<int32_t>::max());
  const c10::cuda::CUDAGuard guard(inputs.query.device());
  kda_decode_kernel<<<static_cast<uint32_t>(rows),
                      128,
                      0,
                      at::cuda::getCurrentCUDAStream()>>>(
      inputs.query.data_ptr<torch::BFloat16>(),
      inputs.key.data_ptr<torch::BFloat16>(),
      inputs.value.data_ptr<torch::BFloat16>(),
      inputs.raw_forget.data_ptr<torch::BFloat16>(),
      inputs.raw_beta.data_ptr<torch::BFloat16>(),
      inputs.a_log.data_ptr<float>(),
      inputs.dt_bias.data_ptr<float>(),
      inputs.initial_state.data_ptr<float>(),
      output.data_ptr<torch::BFloat16>(),
      final_state.data_ptr<float>(),
      static_cast<int32_t>(inputs.query.size(2)),
      inputs.options.scale,
      inputs.options.gate_lower_bound);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::musa
