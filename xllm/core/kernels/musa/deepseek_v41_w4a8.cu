/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/

#include <glog/logging.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

#include "core/kernels/musa/deepseek_v41_w4a8.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kGroupSize = 128;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int32_t delta = 16; delta > 0; delta /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, delta));
  }
  return value;
}

// The caller has replicated token activations across EP ranks, so this
// permutation is local. It does not pretend to implement DeepEP all-to-all.
// Row order may differ between launches; route_rows preserves the inverse.
__global__ void dispatch_quant_kernel(const c10::BFloat16* input,
                                      const int64_t* expert_ids,
                                      c10::Float8_e4m3fn* output,
                                      float* scales,
                                      int32_t* counts,
                                      int32_t* route_rows,
                                      int32_t hidden,
                                      int32_t topk,
                                      int32_t first_expert,
                                      int32_t experts,
                                      int32_t capacity) {
  const int32_t route = blockIdx.x;
  const int64_t expert = expert_ids[route] - first_expert;
  if (expert < 0 || expert >= experts) {
    if (threadIdx.x == 0) {
      route_rows[route] = -1;
    }
    return;
  }
  __shared__ int32_t row;
  if (threadIdx.x == 0) {
    row = atomicAdd(counts + expert, 1);
    route_rows[route] = row;
  }
  __syncthreads();
  const int32_t token = route / topk;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t groups = hidden / kGroupSize;
  const int64_t target_row = expert * capacity + row;
  for (int32_t group = warp; group < groups; group += blockDim.x / 32) {
    float values[4];
    float maximum = 0.0f;
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t column = group * kGroupSize + lane + item * 32;
      values[item] = static_cast<float>(input[token * hidden + column]);
      maximum = fmaxf(maximum, fabsf(values[item]));
    }
    const float scale = fmaxf(warp_max(maximum), 1.0e-4f) / kFp8Max;
    if (lane == 0) {
      scales[target_row * groups + group] = scale;
    }
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t column = group * kGroupSize + lane + item * 32;
      const float value = fminf(kFp8Max, fmaxf(-kFp8Max, values[item] / scale));
      output[target_row * hidden + column] = c10::Float8_e4m3fn(value);
    }
  }
}

// The delivery kernel keeps SwiGLU in FP32 until FP8 quantization. Its
// group128 scale floor is 1e-10, unlike the input quantizer's 1e-4 floor.
__global__ void swiglu_quant_kernel(const c10::BFloat16* gate_up,
                                    const int32_t* counts,
                                    c10::Float8_e4m3fn* output,
                                    float* scales,
                                    int32_t intermediate,
                                    int32_t capacity,
                                    float limit) {
  const int32_t expert = blockIdx.z;
  const int32_t row = blockIdx.y * 4 + threadIdx.y;
  if (row >= counts[expert]) {
    return;
  }
  const int32_t group = blockIdx.x;
  const int32_t lane = threadIdx.x;
  const int64_t offset = static_cast<int64_t>(expert) * capacity + row;
  float values[4];
  float maximum = 0.0f;
#pragma unroll
  for (int32_t item = 0; item < 4; ++item) {
    const int32_t column = group * kGroupSize + lane + item * 32;
    const float gate = fminf(
        static_cast<float>(gate_up[offset * 2 * intermediate + column]), limit);
    const float up = fminf(
        limit,
        fmaxf(-limit,
              static_cast<float>(
                  gate_up[offset * 2 * intermediate + intermediate + column])));
    values[item] = (gate / (1.0f + __expf(-gate))) * up;
    maximum = fmaxf(maximum, fabsf(values[item]));
  }
  const float scale = fmaxf(warp_max(maximum), 1.0e-10f) / kFp8Max;
  if (lane == 0) {
    scales[offset * (intermediate / kGroupSize) + group] = scale;
  }
  const float inverse_scale = 1.0f / scale;
#pragma unroll
  for (int32_t item = 0; item < 4; ++item) {
    const int32_t column = group * kGroupSize + lane + item * 32;
    const float value =
        fminf(kFp8Max, fmaxf(-kFp8Max, values[item] * inverse_scale));
    output[offset * intermediate + column] = c10::Float8_e4m3fn(value);
  }
}

__global__ void combine_kernel(const c10::BFloat16* expert_output,
                               const int64_t* expert_ids,
                               const int32_t* route_rows,
                               const float* route_weights,
                               float* output,
                               int32_t hidden,
                               int32_t topk,
                               int32_t first_expert,
                               int32_t capacity) {
  const int32_t token = blockIdx.y;
  const int32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= hidden) {
    return;
  }
  float value = 0.0f;
  for (int32_t selected = 0; selected < topk; ++selected) {
    const int32_t route = token * topk + selected;
    const int32_t row = route_rows[route];
    if (row < 0) {
      continue;
    }
    const int64_t expert = expert_ids[route] - first_expert;
    const int64_t offset = (expert * capacity + row) * hidden + column;
    // Weights multiply the completed down projection, as in the delivery
    // W4A8/DeepEP combine path, not before activation quantization.
    value += static_cast<float>(expert_output[offset]) * route_weights[route];
  }
  output[token * hidden + column] = value;
}

void check_gpu_contiguous(const torch::Tensor& tensor,
                          const torch::Device& device,
                          torch::ScalarType dtype) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.is_contiguous());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_w4a8_dispatch(const torch::Tensor& input,
                           const torch::Tensor& expert_ids,
                           int64_t first_expert,
                           int64_t local_experts) {
  CHECK(input.defined() && input.device().type() == torch::kPrivateUse1);
  check_gpu_contiguous(input, input.device(), torch::kBFloat16);
  check_gpu_contiguous(expert_ids, input.device(), torch::kInt64);
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(expert_ids.dim(), 2);
  CHECK_EQ(input.size(0), expert_ids.size(0));
  CHECK_GT(input.size(0), 0);
  CHECK_LE(input.size(0), 512);
  CHECK_GT(input.size(1), 0);
  CHECK_EQ(input.size(1) % kGroupSize, 0);
  CHECK_GT(expert_ids.size(1), 0);
  CHECK_LE(expert_ids.size(1), 32);
  CHECK_GE(first_expert, 0);
  CHECK_GT(local_experts, 0);
  CHECK_LE(first_expert + local_experts, 1024);
  torch::DeviceGuard guard(input.device());
  const int64_t capacity = (input.size(0) + 127) / 128 * 128;
  torch::Tensor output =
      torch::empty({local_experts, capacity, input.size(1)},
                   input.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor scales =
      torch::empty({local_experts, capacity, input.size(1) / kGroupSize},
                   input.options().dtype(torch::kFloat32));
  torch::Tensor counts =
      torch::zeros({local_experts}, input.options().dtype(torch::kInt32));
  torch::Tensor rows =
      torch::empty(expert_ids.sizes(), input.options().dtype(torch::kInt32));
  dispatch_quant_kernel<<<expert_ids.numel(),
                          256,
                          0,
                          current_musa_stream(input.device())>>>(
      input.data_ptr<c10::BFloat16>(),
      expert_ids.data_ptr<int64_t>(),
      output.data_ptr<c10::Float8_e4m3fn>(),
      scales.data_ptr<float>(),
      counts.data_ptr<int32_t>(),
      rows.data_ptr<int32_t>(),
      static_cast<int32_t>(input.size(1)),
      static_cast<int32_t>(expert_ids.size(1)),
      static_cast<int32_t>(first_expert),
      static_cast<int32_t>(local_experts),
      static_cast<int32_t>(capacity));
  check_musa_kernel_launch("V4.1 W4A8 group128 dispatch");
  return {output, scales, counts, rows};
}

std::pair<torch::Tensor, torch::Tensor> deepseek_v41_w4a8_swiglu_quant(
    const torch::Tensor& gate_up,
    const torch::Tensor& counts,
    double limit) {
  CHECK(gate_up.defined() && gate_up.device().type() == torch::kPrivateUse1);
  check_gpu_contiguous(gate_up, gate_up.device(), torch::kBFloat16);
  check_gpu_contiguous(counts, gate_up.device(), torch::kInt32);
  CHECK_EQ(gate_up.dim(), 3);
  CHECK_EQ(counts.dim(), 1);
  CHECK_EQ(counts.size(0), gate_up.size(0));
  CHECK_GT(gate_up.size(1), 0);
  CHECK_GT(gate_up.size(2), 0);
  CHECK_EQ(gate_up.size(2) % (2 * kGroupSize), 0);
  CHECK(std::isfinite(limit) && limit > 0.0);
  torch::DeviceGuard guard(gate_up.device());
  const int64_t intermediate = gate_up.size(2) / 2;
  torch::Tensor output =
      torch::empty({gate_up.size(0), gate_up.size(1), intermediate},
                   gate_up.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor scales = torch::empty(
      {gate_up.size(0), gate_up.size(1), intermediate / kGroupSize},
      gate_up.options().dtype(torch::kFloat32));
  const dim3 grid(static_cast<uint32_t>(intermediate / kGroupSize),
                  static_cast<uint32_t>((gate_up.size(1) + 3) / 4),
                  static_cast<uint32_t>(gate_up.size(0)));
  swiglu_quant_kernel<<<grid,
                        dim3(32, 4),
                        0,
                        current_musa_stream(gate_up.device())>>>(
      gate_up.data_ptr<c10::BFloat16>(),
      counts.data_ptr<int32_t>(),
      output.data_ptr<c10::Float8_e4m3fn>(),
      scales.data_ptr<float>(),
      static_cast<int32_t>(intermediate),
      static_cast<int32_t>(gate_up.size(1)),
      static_cast<float>(limit));
  check_musa_kernel_launch("V4.1 W4A8 SwiGLU group128 quantization");
  return {output, scales};
}

torch::Tensor deepseek_v41_w4a8_combine(const torch::Tensor& expert_output,
                                        const torch::Tensor& expert_ids,
                                        const torch::Tensor& route_rows,
                                        const torch::Tensor& route_weights,
                                        int64_t first_expert) {
  CHECK(expert_output.defined() &&
        expert_output.device().type() == torch::kPrivateUse1);
  check_gpu_contiguous(expert_output, expert_output.device(), torch::kBFloat16);
  check_gpu_contiguous(expert_ids, expert_output.device(), torch::kInt64);
  check_gpu_contiguous(route_rows, expert_output.device(), torch::kInt32);
  check_gpu_contiguous(route_weights, expert_output.device(), torch::kFloat32);
  CHECK_EQ(expert_output.dim(), 3);
  CHECK_EQ(expert_ids.dim(), 2);
  CHECK_EQ(expert_ids.sizes(), route_rows.sizes());
  CHECK_EQ(expert_ids.sizes(), route_weights.sizes());
  CHECK_GT(expert_ids.size(0), 0);
  CHECK_GT(expert_output.size(2), 0);
  torch::DeviceGuard guard(expert_output.device());
  torch::Tensor output =
      torch::empty({expert_ids.size(0), expert_output.size(2)},
                   expert_output.options().dtype(torch::kFloat32));
  const dim3 grid(static_cast<uint32_t>((expert_output.size(2) + 255) / 256),
                  static_cast<uint32_t>(expert_ids.size(0)));
  combine_kernel<<<grid, 256, 0, current_musa_stream(expert_output.device())>>>(
      expert_output.data_ptr<c10::BFloat16>(),
      expert_ids.data_ptr<int64_t>(),
      route_rows.data_ptr<int32_t>(),
      route_weights.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int32_t>(expert_output.size(2)),
      static_cast<int32_t>(expert_ids.size(1)),
      static_cast<int32_t>(first_expert),
      static_cast<int32_t>(expert_output.size(1)));
  check_musa_kernel_launch("V4.1 W4A8 weighted combine");
  return output;
}

}  // namespace xllm::kernel::musa
