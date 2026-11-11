/* Copyright 2026 The xLLM Authors.

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
#include <torch/utils.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "core/kernels/musa/deepseek_v41_fp4_accum.cuh"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

__device__ __forceinline__ float packed_weight(const uint8_t* weight,
                                               const uint8_t* scale,
                                               int64_t row,
                                               int64_t k,
                                               int64_t inner) {
  const uint8_t byte = weight[row * (inner / 2) + k / 2];
  const uint8_t nibble = (k & 1) ? byte >> 4 : byte & 15;
  const int32_t magnitude = nibble & 7;
  float value = magnitude < 2 ? static_cast<float>(magnitude) * 0.5f
                              : ldexpf(static_cast<float>(2 + (magnitude & 1)),
                                       (magnitude >> 1) - 2);
  value = (nibble & 8) ? -value : value;
  const uint8_t raw_scale = scale[row * (inner / 32) + k / 32];
  const float factor = raw_scale == 255 ? NAN : ldexpf(1.0f, raw_scale - 127);
  // Match the established weight dequantization and BF16 GEMM inputs.
  return static_cast<float>(static_cast<c10::BFloat16>(value * factor));
}

// Donor V4 decode dataflow: selected expert IDs index local packed weights
// directly. No host histogram, unique_consecutive, or per-expert launch loop.
__global__ void decode_gate_up_kernel(const c10::BFloat16* input,
                                      const int64_t* expert_ids,
                                      const float* route_weights,
                                      const uint8_t* gate,
                                      const uint8_t* gate_scale,
                                      const uint8_t* up,
                                      const uint8_t* up_scale,
                                      c10::BFloat16* hidden,
                                      int64_t inner,
                                      int64_t intermediate,
                                      int64_t topk,
                                      int64_t first_expert,
                                      int64_t local_experts,
                                      float limit) {
  const int32_t lane = threadIdx.x;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.y;
  const int64_t route = blockIdx.y;
  if (column >= intermediate) {
    return;
  }
  const int64_t global_expert = expert_ids[route];
  if (global_expert < first_expert ||
      global_expert >= first_expert + local_experts) {
    if (lane == 0) {
      hidden[route * intermediate + column] = static_cast<c10::BFloat16>(0.0f);
    }
    return;
  }
  const int64_t local = global_expert - first_expert;
  const int64_t weight_row = local * intermediate + column;
  const int64_t token = route / topk;
  V41CompensatedAccumulator gate_accumulator;
  V41CompensatedAccumulator up_accumulator;
  for (int64_t k = lane; k < inner; k += 32) {
    const float value = static_cast<float>(input[token * inner + k]);
    gate_accumulator.add(value *
                         packed_weight(gate, gate_scale, weight_row, k, inner));
    up_accumulator.add(value *
                       packed_weight(up, up_scale, weight_row, k, inner));
  }
  const float gate_sum = gate_accumulator.warp_sum();
  const float up_sum = up_accumulator.warp_sum();
  if (lane == 0) {
    // Each linear first produces BF16, exactly as the non-fused model path.
    const float rounded_gate =
        static_cast<float>(static_cast<c10::BFloat16>(gate_sum));
    const float rounded_up =
        static_cast<float>(static_cast<c10::BFloat16>(up_sum));
    const float g = rounded_gate > limit ? limit : rounded_gate;
    const float u = rounded_up > limit
                        ? limit
                        : (rounded_up < -limit ? -limit : rounded_up);
    const float activation = (g / (1.0f + expf(-g))) * u;
    hidden[route * intermediate + column] =
        static_cast<c10::BFloat16>(activation * route_weights[route]);
  }
}

__global__ void decode_down_kernel(const c10::BFloat16* hidden,
                                   const int64_t* expert_ids,
                                   const uint8_t* down,
                                   const uint8_t* down_scale,
                                   c10::BFloat16* output,
                                   int64_t hidden_size,
                                   int64_t intermediate,
                                   int64_t first_expert,
                                   int64_t local_experts) {
  const int32_t lane = threadIdx.x;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.y;
  const int64_t route = blockIdx.y;
  if (column >= hidden_size) {
    return;
  }
  const int64_t global_expert = expert_ids[route];
  if (global_expert < first_expert ||
      global_expert >= first_expert + local_experts) {
    if (lane == 0) {
      output[route * hidden_size + column] = static_cast<c10::BFloat16>(0.0f);
    }
    return;
  }
  const int64_t local = global_expert - first_expert;
  const int64_t weight_row = local * hidden_size + column;
  V41CompensatedAccumulator accumulator;
  for (int64_t k = lane; k < intermediate; k += 32) {
    accumulator.add(
        static_cast<float>(hidden[route * intermediate + k]) *
        packed_weight(down, down_scale, weight_row, k, intermediate));
  }
  const float sum = accumulator.warp_sum();
  if (lane == 0) {
    output[route * hidden_size + column] = static_cast<c10::BFloat16>(sum);
  }
}

__global__ void decode_combine_kernel(const c10::BFloat16* expert_output,
                                      const int64_t* expert_ids,
                                      float* output,
                                      int64_t hidden_size,
                                      int64_t topk) {
  const int64_t token = blockIdx.y;
  __shared__ int32_t order[32];
  if (threadIdx.x == 0) {
    // Stable ascending expert order matches the prior FP32 index_add loop.
    for (int32_t i = 0; i < topk; ++i) {
      order[i] = i;
    }
    for (int32_t i = 1; i < topk; ++i) {
      const int32_t route = order[i];
      int32_t j = i;
      while (j > 0 && expert_ids[token * topk + order[j - 1]] >
                          expert_ids[token * topk + route]) {
        order[j] = order[j - 1];
        --j;
      }
      order[j] = route;
    }
  }
  __syncthreads();
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= hidden_size) {
    return;
  }
  float sum = 0.0f;
  for (int32_t i = 0; i < topk; ++i) {
    sum += static_cast<float>(
        expert_output[(token * topk + order[i]) * hidden_size + column]);
  }
  output[token * hidden_size + column] = sum;
}

void check_packed(const torch::Tensor& weight,
                  const torch::Tensor& scale,
                  const torch::Tensor& input,
                  int64_t experts,
                  int64_t out_dim,
                  int64_t in_dim) {
  CHECK(weight.defined() && scale.defined());
  CHECK_EQ(weight.device(), input.device());
  CHECK_EQ(scale.device(), input.device());
  CHECK_EQ(weight.scalar_type(), torch::kUInt8);
  CHECK_EQ(scale.scalar_type(), torch::kUInt8);
  CHECK_EQ(weight.dim(), 3);
  CHECK_EQ(scale.dim(), 3);
  CHECK_EQ(weight.size(0), experts);
  CHECK_EQ(weight.size(1), out_dim);
  CHECK_EQ(weight.size(2) * 2, in_dim);
  CHECK_EQ(scale.size(0), experts);
  CHECK_EQ(scale.size(1), out_dim);
  CHECK_EQ(scale.size(2) * 32, in_dim);
  CHECK(weight.is_contiguous() && scale.is_contiguous());
}

}  // namespace

torch::Tensor deepseek_v41_decode_moe(const torch::Tensor& input,
                                      const torch::Tensor& expert_ids,
                                      const torch::Tensor& route_weights,
                                      const torch::Tensor& gate,
                                      const torch::Tensor& gate_scale,
                                      const torch::Tensor& up,
                                      const torch::Tensor& up_scale,
                                      const torch::Tensor& down,
                                      const torch::Tensor& down_scale,
                                      int64_t first_expert,
                                      double limit,
                                      bool block32_quant) {
  CHECK(input.defined());
  CHECK(input.device().type() == torch::kPrivateUse1);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(input.dim(), 2);
  CHECK_LE(input.size(0), 8);
  CHECK_GT(input.size(1), 0);
  CHECK_EQ(input.size(1) % 32, 0);
  CHECK(input.is_contiguous());
  CHECK_EQ(expert_ids.device(), input.device());
  CHECK_EQ(route_weights.device(), input.device());
  CHECK_EQ(expert_ids.scalar_type(), torch::kInt64);
  CHECK_EQ(route_weights.scalar_type(), torch::kFloat32);
  CHECK_EQ(expert_ids.dim(), 2);
  CHECK_EQ(route_weights.sizes(), expert_ids.sizes());
  CHECK_EQ(expert_ids.size(0), input.size(0));
  CHECK_GT(expert_ids.size(1), 0);
  CHECK_LE(expert_ids.size(1), 32);
  CHECK(expert_ids.is_contiguous() && route_weights.is_contiguous());
  CHECK_EQ(gate.dim(), 3);
  const int64_t local_experts = gate.size(0);
  const int64_t intermediate = gate.size(1);
  const int64_t inner = input.size(1);
  const int64_t topk = expert_ids.size(1);
  CHECK_GT(local_experts, 0);
  CHECK_GT(intermediate, 0);
  CHECK_EQ(intermediate % 32, 0);
  CHECK_GE(first_expert, 0);
  CHECK_LE(first_expert, INT64_MAX - local_experts);
  CHECK(std::isfinite(limit) && limit > 0.0 && limit < 1000000.0);
  check_packed(gate, gate_scale, input, local_experts, intermediate, inner);
  check_packed(up, up_scale, input, local_experts, intermediate, inner);
  check_packed(down, down_scale, input, local_experts, inner, intermediate);
  torch::DeviceGuard guard(input.device());
  torch::Tensor output =
      torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));
  if (input.size(0) == 0) {
    return output;
  }
  const int64_t routes = input.size(0) * topk;
  const torch::Tensor quantized =
      block32_quant ? std::get<0>(deepseek_v41_block32_qdq(input)) : input;
  const torch::Tensor hidden =
      torch::empty({routes, intermediate}, input.options());
  const torch::Tensor per_route =
      torch::empty({routes, inner}, input.options());
  const musaStream_t stream = current_musa_stream(input.device());
  decode_gate_up_kernel<<<dim3(static_cast<uint32_t>((intermediate + 3) / 4),
                               static_cast<uint32_t>(routes)),
                          dim3(32, 4),
                          0,
                          stream>>>(quantized.data_ptr<c10::BFloat16>(),
                                    expert_ids.data_ptr<int64_t>(),
                                    route_weights.data_ptr<float>(),
                                    gate.data_ptr<uint8_t>(),
                                    gate_scale.data_ptr<uint8_t>(),
                                    up.data_ptr<uint8_t>(),
                                    up_scale.data_ptr<uint8_t>(),
                                    hidden.data_ptr<c10::BFloat16>(),
                                    inner,
                                    intermediate,
                                    topk,
                                    first_expert,
                                    local_experts,
                                    static_cast<float>(limit));
  check_musa_kernel_launch("V4.1 decode gate/up and weighted SwiGLU");
  const torch::Tensor quantized_hidden =
      block32_quant ? std::get<0>(deepseek_v41_block32_qdq(hidden)) : hidden;
  decode_down_kernel<<<dim3(static_cast<uint32_t>((inner + 3) / 4),
                            static_cast<uint32_t>(routes)),
                       dim3(32, 4),
                       0,
                       stream>>>(quantized_hidden.data_ptr<c10::BFloat16>(),
                                 expert_ids.data_ptr<int64_t>(),
                                 down.data_ptr<uint8_t>(),
                                 down_scale.data_ptr<uint8_t>(),
                                 per_route.data_ptr<c10::BFloat16>(),
                                 inner,
                                 intermediate,
                                 first_expert,
                                 local_experts);
  check_musa_kernel_launch("V4.1 decode selected expert down");
  decode_combine_kernel<<<dim3(static_cast<uint32_t>((inner + 255) / 256),
                               static_cast<uint32_t>(input.size(0))),
                          256,
                          0,
                          stream>>>(per_route.data_ptr<c10::BFloat16>(),
                                    expert_ids.data_ptr<int64_t>(),
                                    output.data_ptr<float>(),
                                    inner,
                                    topk);
  check_musa_kernel_launch("V4.1 decode ascending-expert combine");
  return output;
}

}  // namespace xllm::kernel::musa
