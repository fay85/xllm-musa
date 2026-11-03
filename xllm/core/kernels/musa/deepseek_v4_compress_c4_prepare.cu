/* Copyright 2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MUSA-only C4 decode-state update.  At a four-token boundary it also writes
// the two disjoint candidate slices into contiguous [8, D] tensors and rotates
// history in the same launch.  The caller deliberately retains Torch's exact
// softmax/multiply/sum sequence over those tensors; this removes cat/copy
// launches without changing recurrent pooling arithmetic.

#include <torch/torch.h>

#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kRatio = 4;
constexpr int64_t kCoeff = 2;
constexpr int64_t kStateRows = kRatio * kCoeff;
constexpr int32_t kThreads = 256;

__global__ void deepseek_v4_compress_c4_prepare_kernel(
    float* __restrict__ kv_state,
    float* __restrict__ score_state,
    const float* __restrict__ kv_step,
    const float* __restrict__ score_step,
    const float* __restrict__ ape,
    float* __restrict__ pooled_kv,
    float* __restrict__ pooled_score,
    int64_t head_dim,
    int64_t ratio_position,
    bool at_boundary) {
  const int64_t width = kCoeff * head_dim;
  for (int64_t dim =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       dim < head_dim;
       dim += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t current_row = kRatio + ratio_position;
    const int64_t current_offset = current_row * width;
    const int64_t ape_offset = ratio_position * width;
    kv_state[current_offset + dim] = kv_step[dim];
    kv_state[current_offset + head_dim + dim] = kv_step[head_dim + dim];
    score_state[current_offset + dim] = score_step[dim] + ape[ape_offset + dim];
    score_state[current_offset + head_dim + dim] =
        score_step[head_dim + dim] + ape[ape_offset + head_dim + dim];

    if (!at_boundary) {
      continue;
    }

#pragma unroll
    for (int64_t row = 0; row < kRatio; ++row) {
      const int64_t history_offset = row * width;
      const int64_t current_group_offset = (kRatio + row) * width;
      pooled_kv[row * head_dim + dim] = kv_state[history_offset + dim];
      pooled_score[row * head_dim + dim] = score_state[history_offset + dim];
      pooled_kv[(kRatio + row) * head_dim + dim] =
          kv_state[current_group_offset + head_dim + dim];
      pooled_score[(kRatio + row) * head_dim + dim] =
          score_state[current_group_offset + head_dim + dim];

      // The pooled values above are read before the next history state is
      // installed. Each thread exclusively owns both coefficient columns for
      // this dimension, so no block-wide barrier is required.
      kv_state[history_offset + dim] = kv_state[current_group_offset + dim];
      kv_state[history_offset + head_dim + dim] =
          kv_state[current_group_offset + head_dim + dim];
      score_state[history_offset + dim] =
          score_state[current_group_offset + dim];
      score_state[history_offset + head_dim + dim] =
          score_state[current_group_offset + head_dim + dim];
    }
  }
}

__global__ void deepseek_v4_compress_c4_weighted_sum_kernel(
    const float* __restrict__ pooled_kv,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int64_t head_dim) {
  for (int64_t dim =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       dim < head_dim;
       dim += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float sum = 0.0f;
#pragma unroll
    for (int64_t row = 0; row < kStateRows; ++row) {
      const int64_t offset = row * head_dim + dim;
      sum += pooled_kv[offset] * weights[offset];
    }
    output[dim] = sum;
  }
}

bool valid_float_tensor(const torch::Tensor& tensor,
                        const torch::Device& device) {
  return tensor.defined() && !tensor.device().is_cpu() &&
         tensor.device() == device && tensor.is_contiguous() &&
         tensor.scalar_type() == torch::kFloat32;
}

}  // namespace

bool deepseek_v4_compress_c4_update_and_prepare(torch::Tensor& kv_state,
                                                torch::Tensor& score_state,
                                                const torch::Tensor& kv_step,
                                                const torch::Tensor& score_step,
                                                const torch::Tensor& ape,
                                                int64_t absolute_position,
                                                torch::Tensor& pooled_kv,
                                                torch::Tensor& pooled_score) {
  if (!kv_state.defined() || kv_state.device().is_cpu() ||
      kv_state.dim() != 2 || kv_state.size(0) != kStateRows ||
      kv_state.size(1) <= 0 || kv_state.size(1) % kCoeff != 0 ||
      absolute_position < 0) {
    return false;
  }
  const auto device = kv_state.device();
  const int64_t width = kv_state.size(1);
  const int64_t head_dim = width / kCoeff;
  if (!valid_float_tensor(kv_state, device) ||
      !valid_float_tensor(score_state, device) ||
      !valid_float_tensor(kv_step, device) ||
      !valid_float_tensor(score_step, device) ||
      !valid_float_tensor(ape, device) ||
      score_state.sizes() != kv_state.sizes() || kv_step.dim() != 2 ||
      kv_step.size(0) != 1 || kv_step.size(1) != width ||
      score_step.sizes() != kv_step.sizes() || ape.dim() != 2 ||
      ape.size(0) != kRatio || ape.size(1) != width ||
      head_dim > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const bool at_boundary = (absolute_position + 1) % kRatio == 0;
  if (at_boundary) {
    if (!valid_float_tensor(pooled_kv, device) ||
        !valid_float_tensor(pooled_score, device) || pooled_kv.dim() != 2 ||
        pooled_kv.size(0) != kStateRows || pooled_kv.size(1) != head_dim ||
        pooled_score.sizes() != pooled_kv.sizes()) {
      return false;
    }
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(kv_state));
  const dim3 grid(
      static_cast<unsigned int>((head_dim + kThreads - 1) / kThreads));
  deepseek_v4_compress_c4_prepare_kernel<<<grid,
                                           kThreads,
                                           0,
                                           current_musa_stream(device)>>>(
      kv_state.data_ptr<float>(),
      score_state.data_ptr<float>(),
      kv_step.data_ptr<float>(),
      score_step.data_ptr<float>(),
      ape.data_ptr<float>(),
      at_boundary ? pooled_kv.data_ptr<float>() : nullptr,
      at_boundary ? pooled_score.data_ptr<float>() : nullptr,
      head_dim,
      absolute_position % kRatio,
      at_boundary);
  check_musa_kernel_launch("DeepSeek-V4 C4 compressor kernel");
  return true;
}

bool deepseek_v4_compress_c4_weighted_sum(const torch::Tensor& pooled_kv,
                                          const torch::Tensor& weights,
                                          torch::Tensor& output) {
  if (!pooled_kv.defined() || pooled_kv.device().is_cpu() ||
      pooled_kv.dim() != 2 || pooled_kv.size(0) != kStateRows ||
      pooled_kv.size(1) <= 0) {
    return false;
  }
  const auto device = pooled_kv.device();
  const int64_t head_dim = pooled_kv.size(1);
  if (!valid_float_tensor(pooled_kv, device) ||
      !valid_float_tensor(weights, device) ||
      weights.sizes() != pooled_kv.sizes() ||
      !valid_float_tensor(output, device) || output.dim() != 2 ||
      output.size(0) != 1 || output.size(1) != head_dim ||
      head_dim > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(pooled_kv));
  const dim3 grid(
      static_cast<unsigned int>((head_dim + kThreads - 1) / kThreads));
  deepseek_v4_compress_c4_weighted_sum_kernel<<<grid,
                                                kThreads,
                                                0,
                                                current_musa_stream(device)>>>(
      pooled_kv.data_ptr<float>(),
      weights.data_ptr<float>(),
      output.data_ptr<float>(),
      head_dim);
  check_musa_kernel_launch("DeepSeek-V4 C4 compressor kernel");
  return true;
}

}  // namespace xllm::kernel::musa
