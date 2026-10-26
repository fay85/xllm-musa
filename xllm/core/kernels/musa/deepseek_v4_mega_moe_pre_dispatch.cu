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

// Adapted from SGLang's DeepSeek-V4 mega_moe_pre_dispatch data flow. The
// donor writes group-32 UE8M0 scales for FP4 MegaMoE; xLLM's current Mate
// FP8-FP8 ABI consumes group-128 FP32 power-of-two scales, so the quantization
// and output layout are implemented locally instead of reusing the donor ABI.

#include <torch/torch.h>

#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kQuantBlock = 128;
constexpr int64_t kThreads = 128;
constexpr int64_t kWarpSize = 32;
constexpr float kFp8Max = 448.0f;

template <typename index_t>
__global__ void mega_moe_pre_dispatch_kernel(
    const at::BFloat16* __restrict__ input,
    const index_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ output_scale,
    int64_t* __restrict__ output_topk_ids,
    float* __restrict__ output_topk_weights,
    int64_t hidden_size,
    int64_t scale_columns,
    int64_t topk) {
  const int64_t token = blockIdx.x;
  const int64_t quant_block = blockIdx.y;
  const int64_t thread = threadIdx.x;
  const int64_t column = quant_block * kQuantBlock + thread;
  const int64_t offset = token * hidden_size + column;
  const float value = static_cast<float>(input[offset]);
  float max_abs = fabsf(value);

#pragma unroll
  for (int64_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }

  __shared__ float warp_max[kThreads / kWarpSize];
  __shared__ float block_scale;
  const int64_t lane = thread % kWarpSize;
  const int64_t warp = thread / kWarpSize;
  if (lane == 0) {
    warp_max[warp] = max_abs;
  }
  __syncthreads();

  if (warp == 0) {
    max_abs = lane < kThreads / kWarpSize ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int64_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
      const float raw_scale = fmaxf(max_abs / kFp8Max, 1.0e-4f);
      block_scale = exp2f(ceilf(log2f(raw_scale)));
      output_scale[token * scale_columns + quant_block] = block_scale;
    }
  }
  __syncthreads();

  output[offset] = scaled_fp8_conversion<true, c10::Float8_e4m3fn>(
      value, 1.0f / block_scale);

  if (quant_block == 0 && thread < topk) {
    const int64_t route_offset = token * topk + thread;
    output_topk_ids[route_offset] =
        static_cast<int64_t>(topk_ids[route_offset]);
    output_topk_weights[route_offset] = topk_weights[route_offset];
  }
}

void check_last_error() {
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 MegaMoE pre-dispatch kernel failed: "
      << musaGetErrorString(status);
}

}  // namespace

bool deepseek_v4_mega_moe_pre_dispatch(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& output,
    const torch::Tensor& output_scale,
    const torch::Tensor& output_topk_ids,
    const torch::Tensor& output_topk_weights) {
  const int64_t num_tokens = input.dim() == 2 ? input.size(0) : 0;
  const int64_t hidden_size = input.dim() == 2 ? input.size(1) : 0;
  const int64_t topk = topk_ids.dim() == 2 ? topk_ids.size(1) : 0;
  const bool valid =
      input.defined() && topk_ids.defined() && topk_weights.defined() &&
      output.defined() && output_scale.defined() && output_topk_ids.defined() &&
      output_topk_weights.defined() && input.dim() == 2 &&
      topk_ids.dim() == 2 && topk_weights.dim() == 2 && output.dim() == 2 &&
      output_scale.dim() == 2 && output_topk_ids.dim() == 2 &&
      output_topk_weights.dim() == 2 && num_tokens > 0 && hidden_size > 0 &&
      hidden_size % kQuantBlock == 0 && topk > 0 &&
      input.scalar_type() == torch::kBFloat16 &&
      (topk_ids.scalar_type() == torch::kInt32 ||
       topk_ids.scalar_type() == torch::kInt64) &&
      topk_weights.scalar_type() == torch::kFloat32 &&
      output.scalar_type() == torch::kFloat8_e4m3fn &&
      output_scale.scalar_type() == torch::kFloat32 &&
      output_topk_ids.scalar_type() == torch::kInt64 &&
      output_topk_weights.scalar_type() == torch::kFloat32 &&
      input.is_contiguous() && topk_ids.is_contiguous() &&
      topk_weights.is_contiguous() && output.is_contiguous() &&
      output_scale.is_contiguous() && output_topk_ids.is_contiguous() &&
      output_topk_weights.is_contiguous() && topk_ids.size(0) == num_tokens &&
      topk_weights.sizes() == topk_ids.sizes() &&
      output.size(0) >= num_tokens && output.size(1) == hidden_size &&
      output_scale.size(0) >= num_tokens &&
      output_scale.size(1) == hidden_size / kQuantBlock &&
      output_topk_ids.size(0) >= num_tokens &&
      output_topk_ids.size(1) == topk &&
      output_topk_weights.sizes() == output_topk_ids.sizes() &&
      input.device() == topk_ids.device() &&
      input.device() == topk_weights.device() &&
      input.device() == output.device() &&
      input.device() == output_scale.device() &&
      input.device() == output_topk_ids.device() &&
      input.device() == output_topk_weights.device();
  if (!valid) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const musaStream_t stream = current_musa_stream(input.device());
  const dim3 grid(static_cast<uint32_t>(num_tokens),
                  static_cast<uint32_t>(hidden_size / kQuantBlock));
  const dim3 block(static_cast<uint32_t>(kThreads));
  if (topk_ids.scalar_type() == torch::kInt32) {
    mega_moe_pre_dispatch_kernel<int32_t>
        <<<grid, block, 0, stream>>>(input.data_ptr<at::BFloat16>(),
                                     topk_ids.data_ptr<int32_t>(),
                                     topk_weights.data_ptr<float>(),
                                     output.data_ptr<c10::Float8_e4m3fn>(),
                                     output_scale.data_ptr<float>(),
                                     output_topk_ids.data_ptr<int64_t>(),
                                     output_topk_weights.data_ptr<float>(),
                                     hidden_size,
                                     hidden_size / kQuantBlock,
                                     topk);
  } else {
    mega_moe_pre_dispatch_kernel<int64_t>
        <<<grid, block, 0, stream>>>(input.data_ptr<at::BFloat16>(),
                                     topk_ids.data_ptr<int64_t>(),
                                     topk_weights.data_ptr<float>(),
                                     output.data_ptr<c10::Float8_e4m3fn>(),
                                     output_scale.data_ptr<float>(),
                                     output_topk_ids.data_ptr<int64_t>(),
                                     output_topk_weights.data_ptr<float>(),
                                     hidden_size,
                                     hidden_size / kQuantBlock,
                                     topk);
  }
  check_last_error();
  return true;
}

}  // namespace xllm::kernel::musa
