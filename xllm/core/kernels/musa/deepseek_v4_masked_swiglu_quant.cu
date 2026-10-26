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

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kQuantBlock = 128;
constexpr int32_t kWarpSize = 32;
constexpr float kFp8Max = 448.0f;

class MaskedSwigluWorkspace final {
 public:
  int64_t capacity_tokens = 0;
  torch::Tensor output;
  torch::Tensor scale;
};

struct MaskedSwigluWorkspaceKey {
  int32_t device = -1;
  int32_t dtype = -1;
  int64_t num_experts = 0;
  int64_t hidden_size = 0;

  bool operator==(const MaskedSwigluWorkspaceKey& other) const {
    return device == other.device && dtype == other.dtype &&
           num_experts == other.num_experts && hidden_size == other.hidden_size;
  }
};

struct MaskedSwigluWorkspaceKeyHash {
  size_t operator()(const MaskedSwigluWorkspaceKey& key) const {
    size_t value = std::hash<int32_t>{}(key.device);
    value ^= std::hash<int32_t>{}(key.dtype) + 0x9e3779b9 + (value << 6) +
             (value >> 2);
    value ^= std::hash<int64_t>{}(key.num_experts) + 0x9e3779b9 + (value << 6) +
             (value >> 2);
    value ^= std::hash<int64_t>{}(key.hidden_size) + 0x9e3779b9 + (value << 6) +
             (value >> 2);
    return value;
  }
};

MaskedSwigluWorkspace& get_workspace(const torch::Tensor& gate_up,
                                     int64_t max_tokens,
                                     int64_t hidden_size) {
  static thread_local std::unordered_map<MaskedSwigluWorkspaceKey,
                                         MaskedSwigluWorkspace,
                                         MaskedSwigluWorkspaceKeyHash>
      workspaces;
  const MaskedSwigluWorkspaceKey key{
      gate_up.device().index(),
      static_cast<int32_t>(gate_up.scalar_type()),
      gate_up.size(0),
      hidden_size};
  MaskedSwigluWorkspace& workspace = workspaces[key];
  if (workspace.capacity_tokens < max_tokens || !workspace.output.defined() ||
      !workspace.scale.defined()) {
    const int64_t capacity = std::max<int64_t>(
        max_tokens,
        workspace.capacity_tokens > 0 ? workspace.capacity_tokens * 2 : 1);
    workspace.capacity_tokens = capacity;
    workspace.output =
        torch::empty({gate_up.size(0), capacity, hidden_size},
                     gate_up.options().dtype(torch::kFloat8_e4m3fn));
    workspace.scale =
        torch::empty({gate_up.size(0), capacity, hidden_size / kQuantBlock},
                     gate_up.options().dtype(torch::kFloat32));
  }
  return workspace;
}

template <typename ScalarType>
__global__ void masked_swiglu_fp8_quant_kernel(
    const ScalarType* __restrict__ gate_up,
    const int32_t* __restrict__ masked_tokens,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scale,
    int64_t max_tokens,
    int64_t hidden_size,
    int64_t output_token_stride,
    float swiglu_limit) {
  const int64_t block_column = blockIdx.x;
  const int64_t token = blockIdx.y;
  const int64_t expert = blockIdx.z;
  if (token >= masked_tokens[expert]) {
    return;
  }

  const int32_t thread = threadIdx.x;
  const int64_t column = block_column * kQuantBlock + thread;
  const int64_t input_row = expert * max_tokens + token;
  float value = 0.0f;
  if (column < hidden_size) {
    const int64_t input_offset = input_row * hidden_size * 2 + column;
    float gate = static_cast<float>(gate_up[input_offset]);
    float up = static_cast<float>(gate_up[input_offset + hidden_size]);
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const ScalarType rounded =
        static_cast<ScalarType>(gate / (1.0f + expf(-gate)) * up);
    value = static_cast<float>(rounded);
  }

  float max_abs = fabsf(value);
#pragma unroll
  for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
  }

  __shared__ float warp_max[kQuantBlock / kWarpSize];
  __shared__ float block_scale;
  const int32_t lane = thread % kWarpSize;
  const int32_t warp = thread / kWarpSize;
  if (lane == 0) {
    warp_max[warp] = max_abs;
  }
  __syncthreads();

  if (warp == 0) {
    max_abs = lane < kQuantBlock / kWarpSize ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int32_t delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
      block_scale = fmaxf(max_abs, 1.0e-10f) / kFp8Max;
      scale[(expert * output_token_stride + token) *
                (hidden_size / kQuantBlock) +
            block_column] = block_scale;
    }
  }
  __syncthreads();

  if (column < hidden_size) {
    output[(expert * output_token_stride + token) * hidden_size + column] =
        scaled_fp8_conversion<true, c10::Float8_e4m3fn>(value,
                                                        1.0f / block_scale);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_masked_swiglu_quantize(
    const torch::Tensor& gate_up,
    const torch::Tensor& masked_tokens,
    double swiglu_limit) {
  CHECK(gate_up.defined() && gate_up.dim() == 3 && gate_up.is_contiguous());
  CHECK(gate_up.scalar_type() == torch::kBFloat16 ||
        gate_up.scalar_type() == torch::kFloat16);
  CHECK(gate_up.size(0) > 0 && gate_up.size(1) > 0 && gate_up.size(2) > 0 &&
        gate_up.size(2) % (2 * kQuantBlock) == 0);
  CHECK(masked_tokens.defined() && masked_tokens.dim() == 1 &&
        masked_tokens.is_contiguous());
  CHECK(masked_tokens.scalar_type() == torch::kInt32);
  CHECK(masked_tokens.size(0) == gate_up.size(0));
  CHECK(masked_tokens.device() == gate_up.device());

  const int64_t max_tokens = gate_up.size(1);
  const int64_t hidden_size = gate_up.size(2) / 2;
  MaskedSwigluWorkspace& workspace =
      get_workspace(gate_up, max_tokens, hidden_size);
  torch::Tensor output = workspace.output.narrow(
      /*dim=*/1, /*start=*/0, /*length=*/max_tokens);
  torch::Tensor scale = workspace.scale.narrow(
      /*dim=*/1, /*start=*/0, /*length=*/max_tokens);

  float limit = static_cast<float>(swiglu_limit);
  if (!std::isfinite(limit) || limit >= 1000000.0f) {
    limit = 0.0f;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(gate_up));
  const musaStream_t stream = current_musa_stream(gate_up.device());
  const dim3 grid(hidden_size / kQuantBlock, max_tokens, gate_up.size(0));
  const dim3 block(kQuantBlock);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      gate_up.scalar_type(),
      "deepseek_v4_masked_swiglu_quantize",
      [&] {
        masked_swiglu_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            gate_up.data_ptr<scalar_t>(),
            masked_tokens.data_ptr<int32_t>(),
            workspace.output.data_ptr<c10::Float8_e4m3fn>(),
            workspace.scale.data_ptr<float>(),
            max_tokens,
            hidden_size,
            workspace.capacity_tokens,
            limit);
      });
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 masked SwiGLU quant kernel failed: "
      << musaGetErrorString(status);
  return {output, scale};
}

}  // namespace xllm::kernel::musa
