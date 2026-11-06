/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cmath>
#include <vector>

#include "core/kernels/musa/mhc_ops.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kThreads = 256;
constexpr int64_t kMaxHcMult = 12;

__device__ float stable_sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

// Learned HC head: RMS-normalize the flattened residual, apply the FP32 head,
// then reduce weighted streams. GLM final contraction uses mhc_contract_mean
// instead of this kernel.
template <typename scalar_t>
__global__ void mhc_head_kernel(const scalar_t* __restrict__ residual,
                                const float* __restrict__ hc_fn,
                                const float* __restrict__ hc_scale,
                                const float* __restrict__ hc_base,
                                scalar_t* __restrict__ output,
                                int64_t hc_mult,
                                int64_t hidden_size,
                                float norm_eps,
                                float hc_eps) {
  extern __shared__ float shared[];
  const int32_t thread_id = threadIdx.x;
  const int64_t token = blockIdx.x;
  const int64_t flat_size = hc_mult * hidden_size;
  const int64_t input_offset = token * flat_size;
  float* partials = shared;
  float* weights = partials + (hc_mult + 1) * blockDim.x;

  for (int64_t slot = thread_id; slot < (hc_mult + 1) * blockDim.x;
       slot += blockDim.x) {
    partials[slot] = 0.0f;
  }
  __syncthreads();
  for (int64_t index = thread_id; index < flat_size; index += blockDim.x) {
    const float value = musa_convert_to_float(residual[input_offset + index]);
    partials[hc_mult * blockDim.x + thread_id] += value * value;
    for (int64_t row = 0; row < hc_mult; ++row) {
      partials[row * blockDim.x + thread_id] +=
          value * hc_fn[row * flat_size + index];
    }
  }
  __syncthreads();

  for (int64_t row = 0; row <= hc_mult; ++row) {
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride) {
        partials[row * blockDim.x + thread_id] +=
            partials[row * blockDim.x + thread_id + stride];
      }
      __syncthreads();
    }
  }

  if (thread_id == 0) {
    const float inverse_rms =
        rsqrtf(partials[hc_mult * blockDim.x] / static_cast<float>(flat_size) +
               norm_eps);
    const float scale = hc_scale[0];
    for (int64_t row = 0; row < hc_mult; ++row) {
      weights[row] =
          stable_sigmoid(partials[row * blockDim.x] * inverse_rms * scale +
                         hc_base[row]) +
          hc_eps;
    }
  }
  __syncthreads();

  for (int64_t hidden = thread_id; hidden < hidden_size; hidden += blockDim.x) {
    float value = 0.0f;
    for (int64_t row = 0; row < hc_mult; ++row) {
      value += musa_convert_to_float(
                   residual[input_offset + row * hidden_size + hidden]) *
               weights[row];
    }
    output[token * hidden_size + hidden] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void mhc_contract_mean_kernel(const scalar_t* __restrict__ residual,
                                         scalar_t* __restrict__ output,
                                         int64_t hc_mult,
                                         int64_t hidden_size) {
  const int64_t token = blockIdx.x;
  for (int64_t hidden = threadIdx.x + blockIdx.y * blockDim.x;
       hidden < hidden_size;
       hidden += blockDim.x * gridDim.y) {
    float value = 0.0f;
    const scalar_t* residual_token =
        residual + token * hc_mult * hidden_size + hidden;
    for (int64_t row = 0; row < hc_mult; ++row) {
      value += musa_convert_to_float(residual_token[row * hidden_size]);
    }
    output[token * hidden_size + hidden] =
        static_cast<scalar_t>(value / static_cast<float>(hc_mult));
  }
}

}  // namespace

torch::Tensor mhc_head(const torch::Tensor& residual,
                       const torch::Tensor& hc_head_fn,
                       const torch::Tensor& hc_head_scale,
                       const torch::Tensor& hc_head_base,
                       double norm_eps,
                       double hc_eps) {
  CHECK(residual.defined() && residual.dim() >= 3);
  CHECK_EQ(residual.device().type(), torch::kPrivateUse1);
  CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
  CHECK(residual.is_contiguous());
  const int64_t hc_mult = residual.size(-2);
  const int64_t hidden_size = residual.size(-1);
  CHECK_GT(hc_mult, 0);
  CHECK_LE(hc_mult, kMaxHcMult);
  CHECK_GT(hidden_size, 0);
  CHECK_EQ(hidden_size % 8, 0);
  for (const auto& parameter : {hc_head_fn, hc_head_scale, hc_head_base}) {
    CHECK(parameter.defined() && parameter.device() == residual.device());
    CHECK(parameter.is_contiguous());
    CHECK_EQ(parameter.scalar_type(), torch::kFloat32);
  }
  CHECK(std::isfinite(norm_eps) && norm_eps > 0);
  CHECK(std::isfinite(hc_eps) && hc_eps >= 0);
  CHECK_EQ(hc_head_fn.sizes(),
           torch::IntArrayRef({hc_mult, hc_mult * hidden_size}));
  CHECK_EQ(hc_head_scale.numel(), 1);
  CHECK_EQ(hc_head_base.numel(), hc_mult);

  torch::Tensor residual_flat =
      residual.reshape({-1, hc_mult, hidden_size}).contiguous();
  const int64_t tokens = residual_flat.size(0);
  torch::Tensor output =
      torch::empty({tokens, hidden_size}, residual.options());
  if (tokens > 0) {
    const size_t shared_bytes =
        (static_cast<size_t>(hc_mult) + 1) * kThreads * sizeof(float) +
        static_cast<size_t>(hc_mult) * sizeof(float);
    mhc_head_kernel<c10::BFloat16><<<static_cast<uint32_t>(tokens),
                                     kThreads,
                                     shared_bytes,
                                     current_musa_stream(residual.device())>>>(
        residual_flat.data_ptr<c10::BFloat16>(),
        hc_head_fn.data_ptr<float>(),
        hc_head_scale.data_ptr<float>(),
        hc_head_base.data_ptr<float>(),
        output.data_ptr<c10::BFloat16>(),
        hc_mult,
        hidden_size,
        static_cast<float>(norm_eps),
        static_cast<float>(hc_eps));
    check_musa_kernel_launch("mhc_head");
  }
  std::vector<int64_t> output_shape(residual.sizes().begin(),
                                    residual.sizes().end() - 2);
  output_shape.emplace_back(hidden_size);
  return output.reshape(output_shape);
}

torch::Tensor mhc_contract_mean(const torch::Tensor& residual) {
  CHECK(residual.defined() && residual.dim() >= 3);
  CHECK_EQ(residual.device().type(), torch::kPrivateUse1);
  CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
  const int64_t hc_mult = residual.size(-2);
  const int64_t hidden_size = residual.size(-1);
  CHECK_GT(hc_mult, 0);
  CHECK_LE(hc_mult, kMaxHcMult);
  CHECK_GT(hidden_size, 0);
  torch::Tensor residual_flat =
      residual.reshape({-1, hc_mult, hidden_size}).contiguous();
  const int64_t tokens = residual_flat.size(0);
  torch::Tensor output =
      torch::empty({tokens, hidden_size}, residual.options());
  if (tokens > 0) {
    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<uint32_t>(tokens),
        static_cast<uint32_t>((hidden_size + kThreads - 1) / kThreads));
    mhc_contract_mean_kernel<c10::BFloat16>
        <<<grid, block, 0, current_musa_stream(residual.device())>>>(
            residual_flat.data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(),
            hc_mult,
            hidden_size);
    check_musa_kernel_launch("mhc_contract_mean");
  }
  std::vector<int64_t> output_shape(residual.sizes().begin(),
                                    residual.sizes().end() - 2);
  output_shape.emplace_back(hidden_size);
  return output.reshape(output_shape);
}

}  // namespace xllm::kernel::musa
