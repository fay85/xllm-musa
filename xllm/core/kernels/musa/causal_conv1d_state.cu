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

#include "core/kernels/musa/causal_conv1d_state.h"
#include "core/kernels/musa/musa_tensor_validation.h"

namespace xllm::kernel::musa {
namespace {

__global__ void causal_conv1d_state_kernel(const torch::BFloat16* input,
                                           const torch::BFloat16* weight,
                                           const torch::BFloat16* history,
                                           torch::BFloat16* output,
                                           torch::BFloat16* final_history,
                                           int64_t batch,
                                           int64_t tokens,
                                           int64_t channels) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= batch * channels) {
    return;
  }
  const int64_t b = row / channels;
  const int64_t c = row % channels;
  float h0 = static_cast<float>(history[(b * 3) * channels + c]);
  float h1 = static_cast<float>(history[(b * 3 + 1) * channels + c]);
  float h2 = static_cast<float>(history[(b * 3 + 2) * channels + c]);
  const float w0 = static_cast<float>(weight[c * 4]);
  const float w1 = static_cast<float>(weight[c * 4 + 1]);
  const float w2 = static_cast<float>(weight[c * 4 + 2]);
  const float w3 = static_cast<float>(weight[c * 4 + 3]);
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t offset = (b * tokens + t) * channels + c;
    const float current = static_cast<float>(input[offset]);
    const float sum = h0 * w0 + h1 * w1 + h2 * w2 + current * w3;
    output[offset] = static_cast<torch::BFloat16>(sum / (1.0f + expf(-sum)));
    h0 = h1;
    h1 = h2;
    h2 = current;
  }
  final_history[(b * 3) * channels + c] = static_cast<torch::BFloat16>(h0);
  final_history[(b * 3 + 1) * channels + c] = static_cast<torch::BFloat16>(h1);
  final_history[(b * 3 + 2) * channels + c] = static_cast<torch::BFloat16>(h2);
}

}  // namespace

void causal_conv1d_with_state(const torch::Tensor& input,
                              const torch::Tensor& weight,
                              const torch::Tensor& initial_history,
                              torch::Tensor& output,
                              torch::Tensor& final_history) {
  CHECK(input.defined());
  CHECK(input.device().is_privateuseone());
  CHECK_EQ(input.dim(), 3);
  CHECK_GT(input.size(0), 0);
  CHECK_GT(input.size(1), 0);
  CHECK_GT(input.size(2), 0);
  for (const torch::Tensor* tensor :
       std::initializer_list<const torch::Tensor*>{
           &input, &weight, &initial_history, &output, &final_history}) {
    detail::validate_musa_tensor(*tensor, input.device(), torch::kBFloat16);
  }
  CHECK(output.sizes() == input.sizes());
  CHECK(weight.sizes() == torch::IntArrayRef({input.size(2), 4}));
  CHECK(initial_history.sizes() ==
        torch::IntArrayRef({input.size(0), 3, input.size(2)}));
  CHECK(final_history.sizes() == initial_history.sizes());
  for (const torch::Tensor* tensor : {&input, &weight, &initial_history}) {
    CHECK(!output.is_alias_of(*tensor));
    CHECK(!final_history.is_alias_of(*tensor));
  }
  CHECK(!output.is_alias_of(final_history));
  const int64_t blocks = (input.size(0) * input.size(2) + 255) / 256;
  CHECK_LE(blocks, std::numeric_limits<int32_t>::max());
  const c10::cuda::CUDAGuard guard(input.device());
  causal_conv1d_state_kernel<<<static_cast<uint32_t>(blocks),
                               256,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<torch::BFloat16>(),
      weight.data_ptr<torch::BFloat16>(),
      initial_history.data_ptr<torch::BFloat16>(),
      output.data_ptr<torch::BFloat16>(),
      final_history.data_ptr<torch::BFloat16>(),
      input.size(0),
      input.size(1),
      input.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace xllm::kernel::musa
