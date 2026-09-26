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

// Low-M FP8 expert GEMV adapted from SGLang's MUSA moe_gemv_swiglu block
// organization. W13 consumes BF16/FP16 activations directly and fuses SwiGLU;
// W2 combines all routes and their router weights in one launch. This avoids
// padding six single-token routes to TensorCore block-M tiles.

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

using Fp8LoadVector = uint32_t __attribute__((vector_size(16)));
using InputLoadVector = float __attribute__((vector_size(32)));
using Half4 = _Float16 __attribute__((ext_vector_type(4)));

template <typename scalar_t>
__device__ __forceinline__ float bf16_fp8_dot16(
    const scalar_t* __restrict__ input,
    const c10::Float8_e4m3fn* __restrict__ weight,
    float scale) {
  scalar_t input_values[16];
  uint8_t weight_values[16];
  *reinterpret_cast<InputLoadVector*>(input_values) =
      *reinterpret_cast<const InputLoadVector*>(input);
  *reinterpret_cast<Fp8LoadVector*>(weight_values) =
      *reinterpret_cast<const Fp8LoadVector*>(weight);
  float value = 0.0f;
#pragma unroll
  for (int32_t chunk = 0; chunk < 4; ++chunk) {
    Half4 weight_half4;
    asm("CVT.RTE.DENORM.BST4 %0.f16, %1.e4m3;"
        : "=R"(weight_half4)
        : "R"(reinterpret_cast<float*>(weight_values)[chunk]));
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      value += musa_convert_to_float(input_values[chunk * 4 + item]) *
               static_cast<float>(weight_half4[item]);
    }
  }
  return scale * value;
}

template <typename scalar_t, int32_t kBlockN, int32_t kBlockK>
__global__ void fp8_moe_w13_gemv_swiglu_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ local_ids,
    const c10::Float8_e4m3fn* __restrict__ weight,
    const float* __restrict__ weight_scale,
    scalar_t* __restrict__ activated,
    int64_t num_tokens,
    int64_t topk,
    int64_t num_experts,
    int64_t hidden,
    int64_t intermediate,
    int64_t scale_k_cols,
    float swiglu_limit) {
  __shared__ float gate_partials[kBlockN * kBlockK];
  __shared__ float up_partials[kBlockN * kBlockK];
  const int32_t thread_n = threadIdx.x / kBlockK;
  const int32_t thread_k = threadIdx.x % kBlockK;
  const int64_t token = blockIdx.z;
  const int64_t route = blockIdx.y;
  const int64_t assignment = token * topk + route;
  const int32_t expert = local_ids[assignment];
  const int64_t output_n =
      static_cast<int64_t>(blockIdx.x) * kBlockN + thread_n;
  const bool valid = token < num_tokens && route < topk && expert >= 0 &&
                     expert < num_experts && output_n < intermediate;
  float gate = 0.0f;
  float up = 0.0f;
  if (valid) {
    const scalar_t* input_row = input + token * hidden;
    const int64_t expert_weight_offset =
        static_cast<int64_t>(expert) * 2 * intermediate * hidden;
    const int64_t expert_scale_offset =
        static_cast<int64_t>(expert) * (2 * intermediate / 128) * scale_k_cols;
    const int64_t gate_row = output_n;
    const int64_t up_row = intermediate + output_n;
    for (int64_t column = static_cast<int64_t>(thread_k) * 16; column < hidden;
         column += static_cast<int64_t>(kBlockK) * 16) {
      const int64_t scale_k = column / 128;
      const float gate_scale =
          weight_scale[expert_scale_offset + (gate_row / 128) * scale_k_cols +
                       scale_k];
      const float up_scale =
          weight_scale[expert_scale_offset + (up_row / 128) * scale_k_cols +
                       scale_k];
      gate += bf16_fp8_dot16(
          input_row + column,
          weight + expert_weight_offset + gate_row * hidden + column,
          gate_scale);
      up += bf16_fp8_dot16(
          input_row + column,
          weight + expert_weight_offset + up_row * hidden + column,
          up_scale);
    }
  }
  gate_partials[threadIdx.x] = gate;
  up_partials[threadIdx.x] = up;
  __syncthreads();
  if (thread_k == 0 && valid) {
    gate = 0.0f;
    up = 0.0f;
#pragma unroll
    for (int32_t k = 0; k < kBlockK; ++k) {
      gate += gate_partials[thread_n * kBlockK + k];
      up += up_partials[thread_n * kBlockK + k];
    }
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    activated[assignment * intermediate + output_n] =
        static_cast<scalar_t>((gate / (1.0f + expf(-gate))) * up);
  }
}

template <typename scalar_t,
          typename weight_t,
          int32_t kBlockN,
          int32_t kBlockK>
__global__ void fp8_moe_w2_gemv_kernel(
    const scalar_t* __restrict__ activated,
    const weight_t* __restrict__ topk_weights,
    const int32_t* __restrict__ local_ids,
    const c10::Float8_e4m3fn* __restrict__ weight,
    const float* __restrict__ weight_scale,
    scalar_t* __restrict__ routed_output,
    int64_t num_tokens,
    int64_t topk,
    int64_t num_experts,
    int64_t hidden,
    int64_t intermediate,
    int64_t scale_k_cols) {
  __shared__ float partials[kBlockN * kBlockK];
  const int32_t thread_n = threadIdx.x / kBlockK;
  const int32_t thread_k = threadIdx.x % kBlockK;
  const int64_t token = blockIdx.z;
  const int64_t route = blockIdx.y;
  const int64_t assignment = token * topk + route;
  const int32_t expert = local_ids[assignment];
  const int64_t output_n =
      static_cast<int64_t>(blockIdx.x) * kBlockN + thread_n;
  float value = 0.0f;
  const bool valid = token < num_tokens && route < topk && expert >= 0 &&
                     expert < num_experts && output_n < hidden;
  if (valid) {
    const scalar_t* activated_row = activated + assignment * intermediate;
    const int64_t expert_weight_offset =
        static_cast<int64_t>(expert) * hidden * intermediate;
    const int64_t expert_scale_offset =
        static_cast<int64_t>(expert) * (hidden / 128) * scale_k_cols;
    for (int64_t column = static_cast<int64_t>(thread_k) * 16;
         column < intermediate;
         column += static_cast<int64_t>(kBlockK) * 16) {
      const float scale =
          weight_scale[expert_scale_offset + (output_n / 128) * scale_k_cols +
                       column / 128];
      value += bf16_fp8_dot16(
          activated_row + column,
          weight + expert_weight_offset + output_n * intermediate + column,
          scale);
    }
  }
  partials[threadIdx.x] = value;
  __syncthreads();
  if (thread_k == 0 && token < num_tokens && route < topk &&
      output_n < hidden) {
    value = 0.0f;
#pragma unroll
    for (int32_t k = 0; k < kBlockK; ++k) {
      value += partials[thread_n * kBlockK + k];
    }
    if (valid) {
      value *= musa_convert_to_float(topk_weights[assignment]);
    }
    routed_output[assignment * hidden + output_n] =
        static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void fp8_moe_route_combine_kernel(
    const scalar_t* __restrict__ routed_output,
    scalar_t* __restrict__ output,
    int64_t num_tokens,
    int64_t topk,
    int64_t hidden) {
  const int64_t token = blockIdx.y;
  const int64_t output_n =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token < num_tokens && output_n < hidden) {
    float value = 0.0f;
#pragma unroll
    for (int64_t route = 0; route < topk; ++route) {
      value += musa_convert_to_float(
          routed_output[(token * topk + route) * hidden + output_n]);
    }
    output[token * hidden + output_n] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, typename weight_t, int32_t kW13BlockN, int32_t kW2BlockN>
void launch_fp8_moe_gemv(const torch::Tensor& input,
                         const torch::Tensor& local_ids,
                         const torch::Tensor& topk_weights,
                         const torch::Tensor& w13,
                         const torch::Tensor& w13_scale,
                         const torch::Tensor& w2,
                         const torch::Tensor& w2_scale,
                         float swiglu_limit,
                         const torch::Tensor& activated,
                         const torch::Tensor& routed_output,
                         const torch::Tensor& output,
                         musaStream_t stream) {
  constexpr int32_t kW13BlockK = 32;
  constexpr int32_t kW2BlockK = 4;
  const int64_t tokens = input.size(0);
  const int64_t topk = local_ids.size(1);
  const int64_t experts = w13.size(0);
  const int64_t hidden = input.size(1);
  const int64_t intermediate = w2.size(2);
  const dim3 w13_grid(
      static_cast<unsigned int>((intermediate + kW13BlockN - 1) / kW13BlockN),
      static_cast<unsigned int>(topk),
      static_cast<unsigned int>(tokens));
  fp8_moe_w13_gemv_swiglu_kernel<scalar_t, kW13BlockN, kW13BlockK>
      <<<w13_grid, kW13BlockN * kW13BlockK, 0, stream>>>(
          input.data_ptr<scalar_t>(),
          local_ids.data_ptr<int32_t>(),
          w13.data_ptr<c10::Float8_e4m3fn>(),
          w13_scale.data_ptr<float>(),
          activated.data_ptr<scalar_t>(),
          tokens,
          topk,
          experts,
          hidden,
          intermediate,
          w13_scale.size(2),
          swiglu_limit);
  const dim3 w2_grid(
      static_cast<unsigned int>((hidden + kW2BlockN - 1) / kW2BlockN),
      static_cast<unsigned int>(topk),
      static_cast<unsigned int>(tokens));
  fp8_moe_w2_gemv_kernel<scalar_t, weight_t, kW2BlockN, kW2BlockK>
      <<<w2_grid, kW2BlockN * kW2BlockK, 0, stream>>>(
          activated.data_ptr<scalar_t>(),
          topk_weights.data_ptr<weight_t>(),
          local_ids.data_ptr<int32_t>(),
          w2.data_ptr<c10::Float8_e4m3fn>(),
          w2_scale.data_ptr<float>(),
          routed_output.data_ptr<scalar_t>(),
          tokens,
          topk,
          experts,
          hidden,
          intermediate,
          w2_scale.size(2));
  constexpr int32_t kCombineThreads = 256;
  const dim3 combine_grid(static_cast<unsigned int>(
                              (hidden + kCombineThreads - 1) / kCombineThreads),
                          static_cast<unsigned int>(tokens));
  fp8_moe_route_combine_kernel<scalar_t>
      <<<combine_grid, kCombineThreads, 0, stream>>>(
          routed_output.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          tokens,
          topk,
          hidden);
}

template <typename Scalar, typename Weight>
void dispatch_fp8_moe_tiles(
    const torch::Tensor& input, const torch::Tensor& local_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& w13,
    const torch::Tensor& w13_scale, const torch::Tensor& w2,
    const torch::Tensor& w2_scale, float limit, const torch::Tensor& activated,
    const torch::Tensor& routed_output, const torch::Tensor& output,
    musaStream_t stream) {
  const char* w13_env = std::getenv("XLLM_DSV4_MOE_W13_TILE_N");
  const char* w2_env = std::getenv("XLLM_DSV4_MOE_W2_TILE_N");
  const std::string w13_tile = w13_env == nullptr ? "4" : w13_env;
  const std::string w2_tile = w2_env == nullptr ? "32" : w2_env;
  CHECK(w13_tile == "4" || w13_tile == "8" || w13_tile == "16");
  CHECK(w2_tile == "32" || w2_tile == "64" || w2_tile == "128");
  auto launch = [&](auto w13_tag, auto w2_tag) {
    launch_fp8_moe_gemv<Scalar, Weight, decltype(w13_tag)::value,
                       decltype(w2_tag)::value>(
        input, local_ids, topk_weights, w13, w13_scale, w2, w2_scale, limit,
        activated, routed_output, output, stream);
  };
  auto dispatch_w2 = [&](auto w13_tag) {
    if (w2_tile == "64") {
      launch(w13_tag, std::integral_constant<int32_t, 64>{});
    } else if (w2_tile == "128") {
      launch(w13_tag, std::integral_constant<int32_t, 128>{});
    } else {
      launch(w13_tag, std::integral_constant<int32_t, 32>{});
    }
  };
  if (w13_tile == "8") {
    dispatch_w2(std::integral_constant<int32_t, 8>{});
  } else if (w13_tile == "16") {
    dispatch_w2(std::integral_constant<int32_t, 16>{});
  } else {
    dispatch_w2(std::integral_constant<int32_t, 4>{});
  }
}

}  // namespace

torch::Tensor deepseek_v4_fp8_moe_gemv_decode(
    const torch::Tensor& input,
    const torch::Tensor& local_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& activated_workspace,
    const torch::Tensor& routed_output_workspace,
    const torch::Tensor& output_workspace,
    double swiglu_limit) {
  CHECK(input.defined() && input.dim() == 2 && input.is_contiguous());
  CHECK(input.size(0) > 0 && input.size(1) > 0 && input.size(1) % 128 == 0);
  CHECK(input.scalar_type() == torch::kBFloat16 ||
        input.scalar_type() == torch::kFloat16);
  CHECK(local_ids.defined() && local_ids.dim() == 2 &&
        local_ids.is_contiguous() && local_ids.scalar_type() == torch::kInt32);
  CHECK(local_ids.size(0) == input.size(0) && local_ids.size(1) > 0 &&
        local_ids.size(1) <= 8);
  CHECK(topk_weights.defined() && topk_weights.dim() == 2 &&
        topk_weights.is_contiguous() &&
        topk_weights.sizes() == local_ids.sizes());
  CHECK(topk_weights.scalar_type() == torch::kFloat32 ||
        topk_weights.scalar_type() == torch::kBFloat16 ||
        topk_weights.scalar_type() == torch::kFloat16);
  CHECK(w13.defined() && w13.dim() == 3 && w13.is_contiguous() &&
        w13.scalar_type() == torch::kFloat8_e4m3fn);
  CHECK(w2.defined() && w2.dim() == 3 && w2.is_contiguous() &&
        w2.scalar_type() == torch::kFloat8_e4m3fn);
  CHECK(w13_scale.defined() && w13_scale.dim() == 3 &&
        w13_scale.is_contiguous() &&
        w13_scale.scalar_type() == torch::kFloat32);
  CHECK(w2_scale.defined() && w2_scale.dim() == 3 && w2_scale.is_contiguous() &&
        w2_scale.scalar_type() == torch::kFloat32);
  CHECK(w13.size(0) == w2.size(0) && w13.size(2) == input.size(1));
  CHECK(w13.size(1) == 2 * w2.size(2) && w2.size(1) == input.size(1));
  CHECK(w13.size(1) % 128 == 0 && w2.size(2) % 128 == 0);
  CHECK(w13_scale.size(0) == w13.size(0) &&
        w13_scale.size(1) == w13.size(1) / 128 &&
        w13_scale.size(2) == w13.size(2) / 128);
  CHECK(w2_scale.size(0) == w2.size(0) &&
        w2_scale.size(1) == w2.size(1) / 128 &&
        w2_scale.size(2) == w2.size(2) / 128);
  CHECK(input.device() == local_ids.device() &&
        input.device() == topk_weights.device() &&
        input.device() == w13.device() &&
        input.device() == w13_scale.device() && input.device() == w2.device() &&
        input.device() == w2_scale.device());
  const std::vector<int64_t> activated_shape{input.size(0) * local_ids.size(1),
                                             w2.size(2)};
  const std::vector<int64_t> routed_output_shape{
      input.size(0) * local_ids.size(1), input.size(1)};
  CHECK(activated_workspace.defined() &&
        activated_workspace.sizes() == activated_shape &&
        activated_workspace.scalar_type() == input.scalar_type() &&
        activated_workspace.device() == input.device() &&
        activated_workspace.is_contiguous());
  CHECK(routed_output_workspace.defined() &&
        routed_output_workspace.sizes() == routed_output_shape &&
        routed_output_workspace.scalar_type() == input.scalar_type() &&
        routed_output_workspace.device() == input.device() &&
        routed_output_workspace.is_contiguous());
  CHECK(output_workspace.defined() &&
        output_workspace.sizes() == input.sizes() &&
        output_workspace.scalar_type() == input.scalar_type() &&
        output_workspace.device() == input.device() &&
        output_workspace.is_contiguous());
  float limit = static_cast<float>(swiglu_limit);
  if (!std::isfinite(limit) || limit >= 1000000.0f) {
    limit = 0.0f;
  }
  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const musaStream_t stream = current_musa_stream(input.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      input.scalar_type(),
      "deepseek_v4_fp8_moe_gemv_decode_input",
      [&] {
        if (topk_weights.scalar_type() == torch::kFloat32) {
          dispatch_fp8_moe_tiles<scalar_t, float>(input,
                                               local_ids,
                                               topk_weights,
                                               w13,
                                               w13_scale,
                                               w2,
                                               w2_scale,
                                               limit,
                                               activated_workspace,
                                               routed_output_workspace,
                                               output_workspace,
                                               stream);
        } else if (topk_weights.scalar_type() == torch::kFloat16) {
          dispatch_fp8_moe_tiles<scalar_t, c10::Half>(input,
                                                   local_ids,
                                                   topk_weights,
                                                   w13,
                                                   w13_scale,
                                                   w2,
                                                   w2_scale,
                                                   limit,
                                                   activated_workspace,
                                                   routed_output_workspace,
                                                   output_workspace,
                                                   stream);
        } else {
          dispatch_fp8_moe_tiles<scalar_t, c10::BFloat16>(input,
                                                       local_ids,
                                                       topk_weights,
                                                       w13,
                                                       w13_scale,
                                                       w2,
                                                       w2_scale,
                                                       limit,
                                                       activated_workspace,
                                                       routed_output_workspace,
                                                       output_workspace,
                                                       stream);
        }
      });
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess) << "DeepSeek-V4 FP8 MoE GEMV launch failed: "
                               << musaGetErrorString(status);
  return output_workspace;
}

}  // namespace xllm::kernel::musa
