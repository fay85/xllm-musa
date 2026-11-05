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

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "core/kernels/musa/musa_tensor_validation.h"
#include "core/kernels/musa/rms_norm_sigmoid_gated.h"

namespace xllm::kernel::musa {
namespace {

template <typename Weight>
__global__ void rms_norm_sigmoid_gated_kernel(const torch::BFloat16* value,
                                              const torch::BFloat16* gate,
                                              const Weight* weight,
                                              torch::BFloat16* output,
                                              float eps) {
  __shared__ float squares[128];
  __shared__ float inverse_rms;
  const int32_t d = static_cast<int32_t>(threadIdx.x);
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * 128 + d;
  const float x = static_cast<float>(value[offset]);
  squares[d] = x * x;
  __syncthreads();
  if (d == 0) {
    float sum = 0.0f;
    for (int32_t i = 0; i < 128; ++i) {
      sum += squares[i];
    }
    inverse_rms = rsqrtf(sum / 128.0f + eps);
  }
  __syncthreads();
  const float activated_gate =
      1.0f / (1.0f + expf(-static_cast<float>(gate[offset])));
  output[offset] = static_cast<torch::BFloat16>(
      x * inverse_rms * static_cast<float>(weight[d]) * activated_gate);
}

}  // namespace

void rms_norm_sigmoid_gated(const torch::Tensor& value,
                            const torch::Tensor& gate,
                            const torch::Tensor& weight,
                            double eps,
                            torch::Tensor& output) {
  CHECK(value.defined());
  CHECK(value.device().is_privateuseone());
  CHECK_GT(value.dim(), 0);
  CHECK_EQ(value.size(-1), 128);
  CHECK_GT(value.numel(), 0);
  CHECK(std::isfinite(eps) && eps > 0);
  for (const torch::Tensor* tensor :
       std::initializer_list<const torch::Tensor*>{&value, &gate, &output}) {
    detail::validate_musa_tensor(*tensor, value.device(), torch::kBFloat16);
    CHECK(tensor->sizes() == value.sizes());
  }
  CHECK(weight.defined());
  CHECK(weight.scalar_type() == torch::kBFloat16 ||
        weight.scalar_type() == torch::kFloat32);
  detail::validate_musa_tensor(weight, value.device(), weight.scalar_type());
  CHECK(weight.sizes() == torch::IntArrayRef({128}));
  CHECK(!output.is_alias_of(value));
  CHECK(!output.is_alias_of(gate));
  CHECK(!output.is_alias_of(weight));
  const int64_t rows = value.numel() / 128;
  CHECK_LE(rows, std::numeric_limits<int32_t>::max());
  const c10::cuda::CUDAGuard guard(value.device());
  const auto launch = [&](const auto* weights) {
    rms_norm_sigmoid_gated_kernel<<<static_cast<uint32_t>(rows),
                                    128,
                                    0,
                                    at::cuda::getCurrentCUDAStream()>>>(
        value.data_ptr<torch::BFloat16>(),
        gate.data_ptr<torch::BFloat16>(),
        weights,
        output.data_ptr<torch::BFloat16>(),
        static_cast<float>(eps));
  };
  if (weight.scalar_type() == torch::kFloat32) {
    launch(weight.data_ptr<float>());
  } else {
    launch(weight.data_ptr<torch::BFloat16>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::musa
