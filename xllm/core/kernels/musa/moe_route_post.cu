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
#include <musa_runtime.h>

#include "core/kernels/musa/moe_route_post.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

// Keep muDNN's sigmoid and top-k, including its tie handling and output order.
// Only fuse the operations after expert selection. One warp handles each row.
__global__ void route_post_kernel(float* weights,
                                  int32_t* ids,
                                  const float* scores,
                                  const int64_t* selected,
                                  int64_t rows,
                                  bool renormalize,
                                  float scale) {
  const int64_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int32_t lane = threadIdx.x % 32;
  if (row >= rows) {
    return;
  }
  const int64_t expert = lane < 8 ? selected[row * 8 + lane] : 0;
  const float weight = lane < 8 ? scores[row * 288 + expert] : 0.0f;
  float sum = weight;
  if (renormalize) {
    for (int32_t offset = 4; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);
  }
  if (lane < 8) {
    const float normalized =
        renormalize ? weight * __frcp_rn(fmaxf(sum, 1.0e-20f)) : weight;
    weights[row * 8 + lane] = normalized * scale;
    ids[row * 8 + lane] = static_cast<int32_t>(expert);
  }
}

}  // namespace

void moe_route_post(const torch::Tensor& scores,
                    const torch::Tensor& selected,
                    torch::Tensor& weights,
                    torch::Tensor& ids,
                    bool renormalize,
                    double scale) {
  CHECK(scores.defined() && selected.defined() && weights.defined() &&
        ids.defined());
  CHECK(scores.device().is_privateuseone());
  CHECK(scores.is_contiguous() && selected.is_contiguous() &&
        weights.is_contiguous() && ids.is_contiguous());
  CHECK_EQ(scores.dim(), 2);
  CHECK_EQ(selected.dim(), 2);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(ids.dim(), 2);
  CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(selected.scalar_type(), torch::kInt64);
  CHECK_EQ(weights.scalar_type(), torch::kFloat32);
  CHECK_EQ(ids.scalar_type(), torch::kInt32);
  CHECK_EQ(selected.device(), scores.device());
  CHECK_EQ(weights.device(), scores.device());
  CHECK_EQ(ids.device(), scores.device());
  const int64_t rows = scores.size(0);
  CHECK_EQ(scores.size(1), 288);
  CHECK_EQ(selected.sizes(), torch::IntArrayRef({rows, 8}));
  CHECK_EQ(weights.sizes(), selected.sizes());
  CHECK_EQ(ids.sizes(), selected.sizes());
  if (rows == 0) {
    return;
  }
  const XLLMDeviceGuard device_guard(scores.device());
  const musaStream_t stream = current_musa_stream(scores.device());
  route_post_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
      weights.data_ptr<float>(),
      ids.data_ptr<int32_t>(),
      scores.data_ptr<float>(),
      selected.data_ptr<int64_t>(),
      rows,
      renormalize,
      static_cast<float>(scale));
  check_musa_kernel_launch("moe_route_post");
}

}  // namespace xllm::kernel::musa
