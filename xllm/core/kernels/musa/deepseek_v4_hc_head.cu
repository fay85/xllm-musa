/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// MUSA-only fused DeepSeek-V4 HC head. The algorithm follows SGLang's MUSA
// split-K implementation: stage 0 computes four projection partials and the
// residual square sum in one pass; stage 1 reduces them, applies RMS/sigmoid,
// and writes the weighted sum of the four residual streams.

#include <torch/torch.h>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kHcMult = 4;
constexpr int64_t kHiddenSize = 4096;
constexpr int64_t kFlatHiddenSize = kHcMult * kHiddenSize;
constexpr int64_t kSplitK = 32;
constexpr int64_t kThreads = 128;
constexpr int64_t kPartialWidth = kHcMult + 1;

void check_last_error(const char* stage) {
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 HC head " << stage
      << " kernel failed: " << musaGetErrorString(status);
}

template <typename scalar_t>
__global__ void hc_head_splitk_stage0_kernel(
    const scalar_t* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ partial) {
  const int64_t token = blockIdx.x;
  const int64_t split = blockIdx.y;
  const int64_t split_size = kFlatHiddenSize / kSplitK;
  const int64_t begin = split * split_size;
  const int64_t end = begin + split_size;

  float accum[kPartialWidth] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const int64_t token_offset = token * kFlatHiddenSize;
  for (int64_t k = begin + threadIdx.x; k < end; k += blockDim.x) {
    const float value = static_cast<float>(residual[token_offset + k]);
#pragma unroll
    for (int64_t branch = 0; branch < kHcMult; ++branch) {
      accum[branch] += value * weight[branch * kFlatHiddenSize + k];
    }
    accum[kHcMult] += value * value;
  }

  __shared__ float reduction[kPartialWidth][kThreads];
#pragma unroll
  for (int64_t value = 0; value < kPartialWidth; ++value) {
    reduction[value][threadIdx.x] = accum[value];
  }
  __syncthreads();

  for (int64_t stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int64_t value = 0; value < kPartialWidth; ++value) {
        reduction[value][threadIdx.x] += reduction[value][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int64_t partial_offset = (token * kSplitK + split) * kPartialWidth;
#pragma unroll
    for (int64_t value = 0; value < kPartialWidth; ++value) {
      partial[partial_offset + value] = reduction[value][0];
    }
  }
}

template <typename scalar_t>
__global__ void hc_head_splitk_stage1_kernel(
    const scalar_t* __restrict__ residual,
    const float* __restrict__ partial,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    scalar_t* __restrict__ output,
    float norm_eps,
    float hc_eps) {
  const int64_t token = blockIdx.x;
  __shared__ float coefficients[kHcMult];

  if (threadIdx.x < kPartialWidth) {
    float total = 0.0f;
    for (int64_t split = 0; split < kSplitK; ++split) {
      total += partial[(token * kSplitK + split) * kPartialWidth + threadIdx.x];
    }
    if (threadIdx.x < kHcMult) {
      float square_sum = 0.0f;
      for (int64_t split = 0; split < kSplitK; ++split) {
        square_sum +=
            partial[(token * kSplitK + split) * kPartialWidth + kHcMult];
      }
      const float inv_rms =
          rsqrtf(square_sum / static_cast<float>(kFlatHiddenSize) + norm_eps);
      const float gate = total * inv_rms * scale[0] + base[threadIdx.x];
      coefficients[threadIdx.x] = 1.0f / (1.0f + expf(-gate)) + hc_eps;
    }
  }
  __syncthreads();

  const int64_t residual_offset = token * kFlatHiddenSize;
  const int64_t output_offset = token * kHiddenSize;
  for (int64_t hidden = threadIdx.x; hidden < kHiddenSize;
       hidden += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int64_t branch = 0; branch < kHcMult; ++branch) {
      value += coefficients[branch] *
               static_cast<float>(
                   residual[residual_offset + branch * kHiddenSize + hidden]);
    }
    output[output_offset + hidden] = static_cast<scalar_t>(value);
  }
}

}  // namespace

torch::Tensor deepseek_v4_hc_head(
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& base,
    double norm_eps,
    double hc_eps,
    const std::optional<torch::Tensor>& output,
    const std::optional<torch::Tensor>& workspace) {
  CHECK(residual.defined() && residual.dim() == 3 && residual.is_contiguous());
  CHECK(residual.size(1) == kHcMult && residual.size(2) == kHiddenSize);
  CHECK(weight.defined() && weight.scalar_type() == torch::kFloat32 &&
        weight.is_contiguous() && weight.dim() == 2 &&
        weight.size(0) == kHcMult && weight.size(1) == kFlatHiddenSize);
  CHECK(scale.defined() && scale.scalar_type() == torch::kFloat32 &&
        scale.is_contiguous() && scale.numel() == 1);
  CHECK(base.defined() && base.scalar_type() == torch::kFloat32 &&
        base.is_contiguous() && base.numel() == kHcMult);
  CHECK(residual.device() == weight.device() &&
        residual.device() == scale.device() &&
        residual.device() == base.device());
  CHECK(residual.size(0) > 0 && residual.size(0) <= 64);

  torch::Tensor result =
      output.has_value()
          ? output.value()
          : torch::empty({residual.size(0), kHiddenSize}, residual.options());
  CHECK(result.device() == residual.device() &&
        result.scalar_type() == residual.scalar_type() &&
        result.is_contiguous() && result.size(0) == residual.size(0) &&
        result.size(1) == kHiddenSize);
  torch::Tensor partial =
      workspace.has_value()
          ? workspace.value()
          : torch::empty({residual.size(0), kSplitK, kPartialWidth},
                         residual.options().dtype(torch::kFloat32));
  CHECK(partial.device() == residual.device() &&
        partial.scalar_type() == torch::kFloat32 && partial.is_contiguous() &&
        partial.size(0) == residual.size(0) && partial.size(1) == kSplitK &&
        partial.size(2) == kPartialWidth);

  const XLLMOptionalDeviceGuard device_guard(at::device_of(residual));
  const musaStream_t stream = current_musa_stream(residual.device());
  const dim3 stage0_grid(static_cast<unsigned int>(residual.size(0)),
                         static_cast<unsigned int>(kSplitK));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      residual.scalar_type(),
      "deepseek_v4_hc_head",
      [&] {
        hc_head_splitk_stage0_kernel<scalar_t>
            <<<stage0_grid, kThreads, 0, stream>>>(
                residual.data_ptr<scalar_t>(),
                weight.data_ptr<float>(),
                partial.data_ptr<float>());
        check_last_error("stage0");
        hc_head_splitk_stage1_kernel<scalar_t>
            <<<static_cast<unsigned int>(residual.size(0)),
               kThreads,
               0,
               stream>>>(residual.data_ptr<scalar_t>(),
                         partial.data_ptr<float>(),
                         scale.data_ptr<float>(),
                         base.data_ptr<float>(),
                         result.data_ptr<scalar_t>(),
                         static_cast<float>(norm_eps),
                         static_cast<float>(hc_eps));
      });
  check_last_error("stage1");
  return result;
}

}  // namespace xllm::kernel::musa
