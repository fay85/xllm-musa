/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND. See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <glog/logging.h>
#include <torch/utils.h>

#include <cstdint>

#include "core/kernels/musa/deepseek_v41_decoder_graph_inputs.h"
#include "core/kernels/musa/musa_kernel_utils.h"
namespace xllm::kernel::musa {
namespace {
template <typename Token>
__global__ void copy_decoder_inputs(const c10::BFloat16* attention,
                                    const c10::BFloat16* residual,
                                    const float* pre,
                                    const float* post,
                                    const float* comb,
                                    const Token* token,
                                    c10::BFloat16* out_attention,
                                    c10::BFloat16* out_residual,
                                    float* out_pre,
                                    float* out_post,
                                    float* out_comb,
                                    Token* out_token,
                                    int64_t hidden) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < hidden) {
    out_attention[index] = attention[index];
  }
  if (index < 4 * hidden) {
    out_residual[index] = residual[index];
  }
  if (index < 4) {
    out_pre[index] = pre[index];
    out_post[index] = post[index];
  }
  if (index < 16) {
    out_comb[index] = comb[index];
  }
  if (index == 0) {
    out_token[0] = token[0];
  }
}

template <typename Token>
void launch_copy(const std::array<torch::Tensor, 6>& src,
                 const std::array<torch::Tensor, 6>& dst) {
  const int64_t hidden = src[0].numel();
  copy_decoder_inputs<Token><<<static_cast<uint32_t>((4 * hidden + 255) / 256),
                               256,
                               0,
                               current_musa_stream(src[0].device())>>>(
      src[0].data_ptr<c10::BFloat16>(),
      src[1].data_ptr<c10::BFloat16>(),
      src[2].data_ptr<float>(),
      src[3].data_ptr<float>(),
      src[4].data_ptr<float>(),
      src[5].data_ptr<Token>(),
      dst[0].data_ptr<c10::BFloat16>(),
      dst[1].data_ptr<c10::BFloat16>(),
      dst[2].data_ptr<float>(),
      dst[3].data_ptr<float>(),
      dst[4].data_ptr<float>(),
      dst[5].data_ptr<Token>(),
      hidden);
}
}  // namespace
void deepseek_v41_copy_decoder_graph_inputs(
    const std::array<torch::Tensor, 6>& source,
    const std::array<torch::Tensor, 6>& destination) {
  CHECK(source[0].defined() &&
        source[0].device().type() == torch::kPrivateUse1);
  for (size_t index = 0; index < source.size(); ++index) {
    CHECK(source[index].defined() && destination[index].defined());
    CHECK_EQ(source[index].device(), source[0].device());
    CHECK_EQ(destination[index].device(), source[0].device());
    CHECK_EQ(source[index].sizes(), destination[index].sizes());
    CHECK_EQ(source[index].scalar_type(), destination[index].scalar_type());
    CHECK(source[index].is_contiguous() && destination[index].is_contiguous());
  }
  CHECK_EQ(source[0].scalar_type(), torch::kBFloat16);
  CHECK_EQ(source[1].scalar_type(), torch::kBFloat16);
  CHECK_EQ(source[0].dim(), 2);
  CHECK_EQ(source[0].size(0), 1);
  CHECK_EQ(source[1].dim(), 3);
  CHECK_EQ(source[1].size(0), 1);
  CHECK_EQ(source[1].size(1), 4);
  CHECK_EQ(source[1].size(2), source[0].size(1));
  CHECK_GT(source[0].numel(), 0);
  for (size_t index = 2; index < 5; ++index) {
    CHECK_EQ(source[index].scalar_type(), torch::kFloat32);
    CHECK_EQ(source[index].numel(), index == 4 ? 16 : 4);
  }
  CHECK_EQ(source[5].numel(), 1);
  CHECK(source[5].scalar_type() == torch::kInt32 ||
        source[5].scalar_type() == torch::kInt64);
  torch::DeviceGuard guard(source[0].device());
  if (source[5].scalar_type() == torch::kInt32) {
    launch_copy<int32_t>(source, destination);
  } else {
    launch_copy<int64_t>(source, destination);
  }
  check_musa_kernel_launch("V4.1 decoder Graph input fused copy");
}
}  // namespace xllm::kernel::musa
