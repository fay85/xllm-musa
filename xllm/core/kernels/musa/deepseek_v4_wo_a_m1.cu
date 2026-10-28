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

// MUSA-only DeepSeek-V4 wo_a decode GEMV.  This is the C++/MUSA equivalent of
// SGLang's TileLang m1_splitk implementation: two output rows share a block,
// while 128 lanes split the 4096-element reduction.  Keeping this in the MUSA
// directory avoids changing the CUDA implementation used by other vendors.

#include <torch/torch.h>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kInputSize = 4096;
constexpr int64_t kOutputSize = 1024;
constexpr int64_t kRowsPerBlock = 2;
constexpr int64_t kReduceThreads = 128;
constexpr int64_t kValuesPerLane = 8;
constexpr int64_t kChunks = kInputSize / (kReduceThreads * kValuesPerLane);

constexpr int64_t kSharedIntermediate = 256;
constexpr int64_t kSharedGateUpRows = 2 * kSharedIntermediate;
constexpr int64_t kSharedDownRowsPerBlock = 8;
constexpr int64_t kSharedDownReduceThreads = 32;

template <typename scalar_t, bool kHasHeadScales>
__global__ void deepseek_v4_wo_a_m1_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float* __restrict__ head_scales,
    scalar_t* __restrict__ output) {
  const int64_t output_row =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.x;
  const int64_t reduce_lane = threadIdx.y;
  float sum = 0.0f;

  __shared__ float staged_head_scales[8];
  if constexpr (kHasHeadScales) {
    // Both output rows consume the same eight per-head scales.  Stage them
    // once per block instead of issuing 2 * 128 * 4 redundant global loads
    // from the reduction loop.
    if (threadIdx.x == 0 && reduce_lane < 8) {
      staged_head_scales[reduce_lane] = head_scales[reduce_lane];
    }
    __syncthreads();
  }

#pragma unroll
  for (int64_t chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t begin =
        chunk * kReduceThreads * kValuesPerLane + reduce_lane * kValuesPerLane;
    // The production SGLang TileLang kernel uses one 16-byte load for the
    // eight BF16 values in this inner loop.  Keep the same aligned load shape
    // here; the contract guarantees that begin and every weight row are
    // 16-byte aligned.
    const uint4 input_vec = *reinterpret_cast<const uint4*>(input + begin);
    const uint4 weight_vec = *reinterpret_cast<const uint4*>(
        weight + output_row * kInputSize + begin);
    const auto* input_values =
        reinterpret_cast<const at::BFloat16*>(&input_vec);
    const auto* weight_values =
        reinterpret_cast<const at::BFloat16*>(&weight_vec);
    float chunk_sum = 0.0f;
#pragma unroll
    for (int64_t item = 0; item < kValuesPerLane; ++item) {
      chunk_sum += static_cast<float>(input_values[item]) *
                   static_cast<float>(weight_values[item]);
    }
    if constexpr (kHasHeadScales) {
      sum += chunk_sum * staged_head_scales[begin / 512];
    } else {
      sum += chunk_sum;
    }
  }

  __shared__ float partial[kRowsPerBlock][kReduceThreads];
  partial[threadIdx.x][reduce_lane] = sum;
  __syncthreads();

#pragma unroll
  for (int64_t stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
    if (reduce_lane < stride) {
      partial[threadIdx.x][reduce_lane] +=
          partial[threadIdx.x][reduce_lane + stride];
    }
    __syncthreads();
  }

  if (reduce_lane == 0) {
    output[output_row] = static_cast<scalar_t>(partial[threadIdx.x][0]);
  }
}

__global__ void deepseek_v4_shared_gate_up_swiglu_m1_kernel(
    const at::BFloat16* __restrict__ input,
    const at::BFloat16* __restrict__ weight,
    at::BFloat16* __restrict__ output,
    float swiglu_limit) {
  const int64_t intermediate_row = blockIdx.x;
  const int64_t branch = threadIdx.x;
  const int64_t reduce_lane = threadIdx.y;
  const int64_t weight_row = intermediate_row + branch * kSharedIntermediate;
  float sum = 0.0f;

#pragma unroll
  for (int64_t chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t begin =
        chunk * kReduceThreads * kValuesPerLane + reduce_lane * kValuesPerLane;
    const uint4 input_vec = *reinterpret_cast<const uint4*>(input + begin);
    const uint4 weight_vec = *reinterpret_cast<const uint4*>(
        weight + weight_row * kInputSize + begin);
    const auto* input_values =
        reinterpret_cast<const at::BFloat16*>(&input_vec);
    const auto* weight_values =
        reinterpret_cast<const at::BFloat16*>(&weight_vec);
#pragma unroll
    for (int64_t item = 0; item < kValuesPerLane; ++item) {
      sum += static_cast<float>(input_values[item]) *
             static_cast<float>(weight_values[item]);
    }
  }

  __shared__ float partial[2][kReduceThreads];
  partial[branch][reduce_lane] = sum;
  __syncthreads();
#pragma unroll
  for (int64_t stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
    if (reduce_lane < stride) {
      partial[branch][reduce_lane] += partial[branch][reduce_lane + stride];
    }
    __syncthreads();
  }
  if (branch == 0 && reduce_lane == 0) {
    float gate = partial[0][0];
    float up = partial[1][0];
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const float activated = gate / (1.0f + expf(-gate));
    output[intermediate_row] = static_cast<at::BFloat16>(activated * up);
  }
}

__global__ void deepseek_v4_shared_down_m1_kernel(
    const at::BFloat16* __restrict__ input,
    const at::BFloat16* __restrict__ weight,
    at::BFloat16* __restrict__ output) {
  const int64_t output_row =
      static_cast<int64_t>(blockIdx.x) * kSharedDownRowsPerBlock + threadIdx.x;
  const int64_t reduce_lane = threadIdx.y;
  const int64_t begin = reduce_lane * kValuesPerLane;
  const uint4 input_vec = *reinterpret_cast<const uint4*>(input + begin);
  const uint4 weight_vec = *reinterpret_cast<const uint4*>(
      weight + output_row * kSharedIntermediate + begin);
  const auto* input_values = reinterpret_cast<const at::BFloat16*>(&input_vec);
  const auto* weight_values =
      reinterpret_cast<const at::BFloat16*>(&weight_vec);
  float sum = 0.0f;
#pragma unroll
  for (int64_t item = 0; item < kValuesPerLane; ++item) {
    sum += static_cast<float>(input_values[item]) *
           static_cast<float>(weight_values[item]);
  }

  __shared__ float partial[kSharedDownRowsPerBlock][kSharedDownReduceThreads];
  partial[threadIdx.x][reduce_lane] = sum;
  __syncthreads();
#pragma unroll
  for (int64_t stride = kSharedDownReduceThreads / 2; stride > 0;
       stride >>= 1) {
    if (reduce_lane < stride) {
      partial[threadIdx.x][reduce_lane] +=
          partial[threadIdx.x][reduce_lane + stride];
    }
    __syncthreads();
  }
  if (reduce_lane == 0) {
    output[output_row] = static_cast<at::BFloat16>(partial[threadIdx.x][0]);
  }
}

__global__ void deepseek_v4_shared_swiglu_kernel(
    const at::BFloat16* __restrict__ input,
    at::BFloat16* __restrict__ output,
    int64_t intermediate_size,
    int64_t total_elements,
    float swiglu_limit) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= total_elements) {
    return;
  }

  const int64_t row = output_index / intermediate_size;
  const int64_t column = output_index - row * intermediate_size;
  const int64_t input_row_offset = row * 2 * intermediate_size;
  float gate = static_cast<float>(input[input_row_offset + column]);
  float up =
      static_cast<float>(input[input_row_offset + intermediate_size + column]);
  if (swiglu_limit > 0.0f) {
    gate = fminf(gate, swiglu_limit);
    up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
  }
  output[output_index] =
      static_cast<at::BFloat16>(gate / (1.0f + expf(-gate)) * up);
}

bool matches_wo_a_m1_contract(const torch::Tensor& input,
                              const torch::Tensor& weight,
                              const torch::Tensor& head_scales) {
  const bool scales_valid =
      !head_scales.defined() ||
      (!head_scales.device().is_cpu() &&
       head_scales.device() == input.device() &&
       head_scales.scalar_type() == torch::kFloat32 && head_scales.dim() == 1 &&
       head_scales.size(0) == 8 && head_scales.is_contiguous());
  return scales_valid && input.defined() && weight.defined() &&
         !input.device().is_cpu() && input.device() == weight.device() &&
         input.scalar_type() == torch::kBFloat16 &&
         weight.scalar_type() == torch::kBFloat16 && input.dim() == 3 &&
         weight.dim() == 3 && input.size(0) == 1 && input.size(1) == 1 &&
         input.size(2) == kInputSize && weight.size(0) == 1 &&
         weight.size(1) == kOutputSize && weight.size(2) == kInputSize &&
         input.is_contiguous() && weight.is_contiguous();
}

}  // namespace

torch::Tensor deepseek_v4_wo_a_m1(const torch::Tensor& input,
                                  const torch::Tensor& weight,
                                  const torch::Tensor& head_scales,
                                  const std::optional<torch::Tensor>& output) {
  if (!matches_wo_a_m1_contract(input, weight, head_scales)) {
    return torch::Tensor();
  }

  torch::Tensor result;
  if (output.has_value() && output->defined()) {
    if (output->device() != input.device() ||
        output->scalar_type() != torch::kBFloat16 || output->dim() != 3 ||
        output->size(0) != 1 || output->size(1) != 1 ||
        output->size(2) != kOutputSize || !output->is_contiguous()) {
      return torch::Tensor();
    }
    result = *output;
  } else {
    result = torch::empty({1, 1, kOutputSize}, input.options());
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const dim3 grid(kOutputSize / kRowsPerBlock);
  const dim3 block(kRowsPerBlock, kReduceThreads);
  if (head_scales.defined()) {
    deepseek_v4_wo_a_m1_kernel<at::BFloat16, true>
        <<<grid, block, 0, current_musa_stream(input.device())>>>(
            input.data_ptr<at::BFloat16>(),
            weight.data_ptr<at::BFloat16>(),
            head_scales.data_ptr<float>(),
            result.data_ptr<at::BFloat16>());
  } else {
    deepseek_v4_wo_a_m1_kernel<at::BFloat16, false>
        <<<grid, block, 0, current_musa_stream(input.device())>>>(
            input.data_ptr<at::BFloat16>(),
            weight.data_ptr<at::BFloat16>(),
            nullptr,
            result.data_ptr<at::BFloat16>());
  }
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 wo_a M=1 kernel failed: " << musaGetErrorString(status);
  return result;
}

torch::Tensor deepseek_v4_shared_expert_m1(
    const torch::Tensor& input,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_weight,
    double swiglu_limit,
    const std::optional<torch::Tensor>& intermediate,
    const std::optional<torch::Tensor>& output) {
  const bool contract =
      input.defined() && gate_up_weight.defined() && down_weight.defined() &&
      !input.device().is_cpu() && input.device() == gate_up_weight.device() &&
      input.device() == down_weight.device() &&
      input.scalar_type() == torch::kBFloat16 &&
      gate_up_weight.scalar_type() == torch::kBFloat16 &&
      down_weight.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
      input.size(0) == 1 && input.size(1) == kInputSize &&
      gate_up_weight.dim() == 2 &&
      gate_up_weight.size(0) == kSharedGateUpRows &&
      gate_up_weight.size(1) == kInputSize && down_weight.dim() == 2 &&
      down_weight.size(0) == kInputSize &&
      down_weight.size(1) == kSharedIntermediate && input.is_contiguous() &&
      gate_up_weight.is_contiguous() && down_weight.is_contiguous();
  if (!contract || !intermediate.has_value() || !intermediate->defined()) {
    return torch::Tensor();
  }
  auto hidden = *intermediate;
  if (hidden.device() != input.device() ||
      hidden.scalar_type() != torch::kBFloat16 || hidden.numel() != 256 ||
      !hidden.is_contiguous()) {
    return torch::Tensor();
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const auto stream = current_musa_stream(input.device());
  deepseek_v4_shared_gate_up_swiglu_m1_kernel<<<kSharedIntermediate,
                                                dim3(2, kReduceThreads),
                                                0,
                                                stream>>>(
      input.data_ptr<at::BFloat16>(),
      gate_up_weight.data_ptr<at::BFloat16>(),
      hidden.data_ptr<at::BFloat16>(),
      static_cast<float>(swiglu_limit));
  if (!output.has_value() || !output->defined()) {
    const musaError_t status = musaGetLastError();
    CHECK(status == musaSuccess)
        << "DeepSeek-V4 shared gate/up M=1 kernel failed: "
        << musaGetErrorString(status);
    return hidden;
  }
  auto result = *output;
  if (result.device() != input.device() ||
      result.scalar_type() != torch::kBFloat16 ||
      result.numel() != kInputSize || !result.is_contiguous()) {
    return torch::Tensor();
  }
  deepseek_v4_shared_down_m1_kernel<<<kInputSize / kSharedDownRowsPerBlock,
                                      dim3(kSharedDownRowsPerBlock,
                                           kSharedDownReduceThreads),
                                      0,
                                      stream>>>(
      hidden.data_ptr<at::BFloat16>(),
      down_weight.data_ptr<at::BFloat16>(),
      result.data_ptr<at::BFloat16>());
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 shared expert M=1 kernels failed: "
      << musaGetErrorString(status);
  return result;
}

torch::Tensor deepseek_v4_shared_swiglu(const torch::Tensor& input,
                                        double swiglu_limit,
                                        const torch::Tensor& output) {
  const bool contract =
      input.defined() && output.defined() && !input.device().is_cpu() &&
      input.device() == output.device() &&
      input.scalar_type() == torch::kBFloat16 &&
      output.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
      input.size(0) > 0 && input.size(1) > 0 && input.size(1) % 2 == 0 &&
      output.dim() == 2 && output.size(0) == input.size(0) &&
      output.size(1) * 2 == input.size(1) && input.is_contiguous() &&
      output.is_contiguous();
  if (!contract) {
    return torch::Tensor();
  }

  const int64_t total_elements = output.numel();
  constexpr int64_t kThreads = 256;
  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  deepseek_v4_shared_swiglu_kernel<<<(total_elements + kThreads - 1) / kThreads,
                                     kThreads,
                                     0,
                                     current_musa_stream(input.device())>>>(
      input.data_ptr<at::BFloat16>(),
      output.data_ptr<at::BFloat16>(),
      output.size(1),
      total_elements,
      static_cast<float>(swiglu_limit));
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess) << "DeepSeek-V4 shared SwiGLU kernel failed: "
                               << musaGetErrorString(status);
  return output;
}

}  // namespace xllm::kernel::musa
