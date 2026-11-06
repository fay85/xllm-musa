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

// output[t, j, h] = post[t, j] * x[t, h] + sum_i comb[t, i, j] * residual[t, i,
// h]
template <typename scalar_t>
__global__ void mhc_post_kernel(const scalar_t* __restrict__ x,
                                const scalar_t* __restrict__ residual,
                                const float* __restrict__ post,
                                const float* __restrict__ comb,
                                scalar_t* __restrict__ output,
                                int64_t hc_mult,
                                int64_t hidden_size) {
  const int64_t token = blockIdx.x;
  const int64_t stream = blockIdx.y;
  const int64_t hidden = blockIdx.z * blockDim.x + threadIdx.x;
  if (hidden >= hidden_size || stream >= hc_mult) {
    return;
  }
  const float activation =
      musa_convert_to_float(x[token * hidden_size + hidden]);
  float value = post[token * hc_mult + stream] * activation;
  const scalar_t* residual_token =
      residual + (token * hc_mult) * hidden_size + hidden;
  const float* comb_token = comb + token * hc_mult * hc_mult;
  for (int64_t source = 0; source < hc_mult; ++source) {
    value += comb_token[source * hc_mult + stream] *
             musa_convert_to_float(residual_token[source * hidden_size]);
  }
  output[(token * hc_mult + stream) * hidden_size + hidden] =
      static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void mhc_inverse_rms_kernel(const scalar_t* __restrict__ residual,
                                       float* __restrict__ inverse_rms,
                                       int64_t width,
                                       float eps) {
  float partial = 0.0f;
  const scalar_t* row = residual + static_cast<int64_t>(blockIdx.x) * width;
  for (int64_t index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = musa_convert_to_float(row[index]);
    partial += value * value;
  }
  constexpr int32_t kWarp = 32;
  __shared__ float warp_sums[8];
  for (int32_t offset = kWarp / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffff, partial, offset);
  }
  if ((threadIdx.x & (kWarp - 1)) == 0) {
    warp_sums[threadIdx.x / kWarp] = partial;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int32_t warps = blockDim.x / kWarp;
    for (int32_t warp = 0; warp < warps; ++warp) {
      total += warp_sums[warp];
    }
    inverse_rms[blockIdx.x] = rsqrtf(total / static_cast<float>(width) + eps);
  }
}

}  // namespace

torch::Tensor mhc_post(const torch::Tensor& x,
                       const torch::Tensor& residual,
                       const torch::Tensor& post,
                       const torch::Tensor& comb) {
  CHECK(x.defined() && residual.defined() && post.defined() && comb.defined());
  CHECK_GE(residual.dim(), 3);
  CHECK_EQ(x.scalar_type(), residual.scalar_type());
  CHECK_EQ(x.scalar_type(), torch::kBFloat16);
  CHECK_EQ(x.device().type(), torch::kPrivateUse1);
  CHECK_EQ(post.scalar_type(), torch::kFloat32);
  CHECK_EQ(comb.scalar_type(), torch::kFloat32);
  CHECK(x.device() == residual.device());
  CHECK(x.device() == post.device());
  CHECK(x.device() == comb.device());

  const int64_t hc_mult = residual.size(-2);
  const int64_t hidden_size = residual.size(-1);
  CHECK_GT(hc_mult, 0);
  CHECK_LE(hc_mult, kMaxHcMult);
  CHECK_GT(hidden_size, 0);
  CHECK_EQ(hidden_size % 8, 0);
  const int64_t tokens = residual.numel() / (hc_mult * hidden_size);
  CHECK_EQ(x.numel(), tokens * hidden_size);
  CHECK_EQ(post.numel(), tokens * hc_mult);
  CHECK_EQ(comb.numel(), tokens * hc_mult * hc_mult);

  torch::Tensor x_flat = x.reshape({tokens, hidden_size}).contiguous();
  torch::Tensor residual_flat =
      residual.reshape({tokens, hc_mult, hidden_size}).contiguous();
  torch::Tensor post_flat = post.reshape({tokens, hc_mult}).contiguous();
  torch::Tensor comb_flat =
      comb.reshape({tokens, hc_mult, hc_mult}).contiguous();
  torch::Tensor output =
      torch::empty({tokens, hc_mult, hidden_size}, residual_flat.options());
  if (tokens > 0) {
    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<uint32_t>(tokens),
        static_cast<uint32_t>(hc_mult),
        static_cast<uint32_t>((hidden_size + kThreads - 1) / kThreads));
    mhc_post_kernel<c10::BFloat16>
        <<<grid, block, 0, current_musa_stream(x.device())>>>(
            x_flat.data_ptr<c10::BFloat16>(),
            residual_flat.data_ptr<c10::BFloat16>(),
            post_flat.data_ptr<float>(),
            comb_flat.data_ptr<float>(),
            output.data_ptr<c10::BFloat16>(),
            hc_mult,
            hidden_size);
    check_musa_kernel_launch("mhc_post");
  }
  return output.reshape(residual.sizes());
}

torch::Tensor mhc_inverse_rms(const torch::Tensor& residual, double eps) {
  CHECK(residual.defined() && residual.dim() >= 2);
  CHECK_EQ(residual.device().type(), torch::kPrivateUse1);
  CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
  CHECK(std::isfinite(eps) && eps > 0);
  CHECK_GE(residual.dim(), 3);
  torch::Tensor flat = residual.flatten(-2).contiguous();
  const int64_t rows = flat.size(0);
  const int64_t width = flat.size(1);
  CHECK_GT(width, 0);
  torch::Tensor inverse =
      torch::empty({rows, 1}, flat.options().dtype(torch::kFloat32));
  if (rows > 0) {
    mhc_inverse_rms_kernel<c10::BFloat16>
        <<<static_cast<uint32_t>(rows),
           kThreads,
           0,
           current_musa_stream(residual.device())>>>(
            flat.data_ptr<c10::BFloat16>(),
            inverse.data_ptr<float>(),
            width,
            static_cast<float>(eps));
    check_musa_kernel_launch("mhc_inverse_rms");
  }
  std::vector<int64_t> shape(residual.sizes().begin(),
                             residual.sizes().end() - 2);
  shape.emplace_back(1);
  return inverse.reshape(shape);
}

}  // namespace xllm::kernel::musa
