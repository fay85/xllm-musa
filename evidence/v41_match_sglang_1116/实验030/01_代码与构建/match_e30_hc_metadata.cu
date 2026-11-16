/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <glog/logging.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

__global__ void hc_metadata_warp_kernel(const float* mixes,
                                        const float* scale,
                                        const float* base,
                                        float* pre,
                                        float* post,
                                        float* comb,
                                        int64_t iterations,
                                        float eps) {
  const int32_t lane = threadIdx.x;
  const int64_t token = blockIdx.x;
  const float* input = mixes + token * 24;
  float row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (lane < 4) {
    const float pre_value = input[lane] * scale[0] + base[lane];
    const float post_value = input[4 + lane] * scale[1] + base[4 + lane];
    pre[token * 4 + lane] = 1.0f / (1.0f + expf(-pre_value)) + eps;
    post[token * 4 + lane] = 2.0f / (1.0f + expf(-post_value));
    const int32_t offset = 8 + lane * 4;
    float maximum = -INFINITY;
#pragma unroll
    for (int32_t col = 0; col < 4; ++col) {
      maximum =
          fmaxf(maximum, input[offset + col] * scale[2] + base[offset + col]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int32_t col = 0; col < 4; ++col) {
      row[col] =
          expf(input[offset + col] * scale[2] + base[offset + col] - maximum);
      sum += row[col];
    }
    const float reciprocal = 1.0f / sum;
#pragma unroll
    for (int32_t col = 0; col < 4; ++col) {
      row[col] = row[col] * reciprocal + eps;
    }
  }
  // All 32 lanes execute each shuffle; only lanes 0..3 own matrix rows.
  // Preserve the original row-0, row-1, row-2, row-3 addition order.
#pragma unroll
  for (int32_t col = 0; col < 4; ++col) {
    float sum = 0.0f;
#pragma unroll
    for (int32_t source = 0; source < 4; ++source) {
      sum += __shfl_sync(0xffffffffu, row[col], source, 32);
    }
    row[col] *= 1.0f / (sum + eps);
  }
  for (int64_t iteration = 1; iteration < iterations; ++iteration) {
    float sum = 0.0f;
#pragma unroll
    for (int32_t col = 0; col < 4; ++col) {
      sum += row[col];
    }
    const float reciprocal = 1.0f / (sum + eps);
#pragma unroll
    for (int32_t col = 0; col < 4; ++col) {
      row[col] *= reciprocal;
    }
#pragma unroll
    for (int32_t col = 0; col < 4; ++col) {
      float column_sum = 0.0f;
#pragma unroll
      for (int32_t source = 0; source < 4; ++source) {
        column_sum += __shfl_sync(0xffffffffu, row[col], source, 32);
      }
      row[col] *= 1.0f / (column_sum + eps);
    }
  }
  if (lane < 4) {
#pragma unroll
    for (int32_t col = 0; col < 4; ++col) {
      comb[token * 16 + lane * 4 + col] = row[col];
    }
  }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_metadata_warp_probe(
    const torch::Tensor& mixes,
    const torch::Tensor& scale,
    const torch::Tensor& base,
    int64_t iterations,
    double eps) {
  CHECK_EQ(mixes.dim(), 2);
  CHECK_EQ(mixes.size(1), 24);
  CHECK_EQ(scale.numel(), 3);
  CHECK_EQ(base.numel(), 24);
  CHECK_GE(iterations, 1);
  CHECK_GE(eps, 0.0);
  CHECK(mixes.device().type() == torch::kPrivateUse1);
  for (const torch::Tensor& tensor : {mixes, scale, base}) {
    CHECK_EQ(tensor.scalar_type(), torch::kFloat32);
    CHECK_EQ(tensor.device(), mixes.device());
    CHECK(tensor.is_contiguous());
  }
  torch::DeviceGuard guard(mixes.device());
  const int64_t tokens = mixes.size(0);
  torch::Tensor pre = torch::empty({tokens, 4}, mixes.options());
  torch::Tensor post = torch::empty({tokens, 4}, mixes.options());
  torch::Tensor comb = torch::empty({tokens, 4, 4}, mixes.options());
  if (tokens > 0) {
    const musaStream_t stream = is_stream_capturing()
                                    ? c10::musa::getCurrentMUSAStream().stream()
                                    : current_musa_stream(mixes.device());
    hc_metadata_warp_kernel<<<dim3(static_cast<uint32_t>(tokens)),
                              dim3(32),
                              0,
                              stream>>>(mixes.data_ptr<float>(),
                                        scale.data_ptr<float>(),
                                        base.data_ptr<float>(),
                                        pre.data_ptr<float>(),
                                        post.data_ptr<float>(),
                                        comb.data_ptr<float>(),
                                        iterations,
                                        static_cast<float>(eps));
    check_musa_kernel_launch("HC metadata warp probe");
  }
  return {pre, post, comb};
}
}  // namespace xllm::kernel::musa
