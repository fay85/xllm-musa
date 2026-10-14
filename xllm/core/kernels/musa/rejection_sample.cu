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

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kBlockSize = 256;

constexpr int32_t kGreedyBlockSize = 32;

__device__ __forceinline__ float positive_probability(float value) {
  return (value > 0.0f && isfinite(value)) ? value : 0.0f;
}

__global__ void rejection_sample_target_only_kernel(
    const int64_t* __restrict__ draft_token_ids,
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    const float* __restrict__ uniform_rand,
    const float* __restrict__ recovery_exponential,
    const int64_t* __restrict__ bonus_token_ids,
    int32_t n_speculative_tokens,
    int32_t vocab_size,
    int64_t* __restrict__ output) {
  const int32_t batch_index = static_cast<int32_t>(blockIdx.x);
  const int32_t thread_index = static_cast<int32_t>(threadIdx.x);
  const int64_t draft_offset =
      static_cast<int64_t>(batch_index) * n_speculative_tokens;
  const int64_t target_offset =
      static_cast<int64_t>(batch_index) * n_speculative_tokens * vocab_size;
  const int64_t output_offset =
      static_cast<int64_t>(batch_index) * (n_speculative_tokens + 1);

  __shared__ bool rejected_seen;
  __shared__ bool recover_current;
  __shared__ float shared_scores[kBlockSize];
  __shared__ int64_t shared_tokens[kBlockSize];
  if (thread_index == 0) {
    rejected_seen = false;
  }
  __syncthreads();

  for (int32_t spec_index = 0; spec_index < n_speculative_tokens;
       ++spec_index) {
    if (thread_index == 0) {
      recover_current = false;
      const int64_t draft_token =
          draft_token_ids[draft_offset + spec_index];
      if (rejected_seen) {
        output[output_offset + spec_index] = -1;
      } else {
        const float draft_probability = positive_probability(
            draft_probs[draft_offset + spec_index]);
        const int64_t target_row =
            target_offset + static_cast<int64_t>(spec_index) * vocab_size;
        const float target_probability =
            (draft_token >= 0 && draft_token < vocab_size)
                ? positive_probability(target_probs[target_row + draft_token])
                : 0.0f;
        const float acceptance_probability =
            draft_probability > 0.0f
                ? target_probability / draft_probability
                : (target_probability > 0.0f ? 1.0f : 0.0f);
        if (uniform_rand[draft_offset + spec_index] <
            acceptance_probability) {
          output[output_offset + spec_index] = draft_token;
        } else {
          recover_current = true;
        }
      }
    }
    __syncthreads();

    if (recover_current) {
      const int64_t draft_token =
          draft_token_ids[draft_offset + spec_index];
      const int64_t target_row =
          target_offset + static_cast<int64_t>(spec_index) * vocab_size;
      float best_score = -1.0f;
      int64_t best_token = 0;
      for (int32_t token = thread_index; token < vocab_size;
           token += static_cast<int32_t>(blockDim.x)) {
        if (static_cast<int64_t>(token) == draft_token) {
          continue;
        }
        const float probability =
            positive_probability(target_probs[target_row + token]);
        const float exponential = fmaxf(
            recovery_exponential[target_row + token], FLT_MIN);
        const float score = probability / exponential;
        if (score > best_score ||
            (score == best_score &&
             static_cast<int64_t>(token) < best_token)) {
          best_score = score;
          best_token = token;
        }
      }
      shared_scores[thread_index] = best_score;
      shared_tokens[thread_index] = best_token;
      __syncthreads();

      for (int32_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
          const float other_score = shared_scores[thread_index + stride];
          const int64_t other_token = shared_tokens[thread_index + stride];
          if (other_score > shared_scores[thread_index] ||
              (other_score == shared_scores[thread_index] &&
               other_token < shared_tokens[thread_index])) {
            shared_scores[thread_index] = other_score;
            shared_tokens[thread_index] = other_token;
          }
        }
        __syncthreads();
      }

      if (thread_index == 0) {
        output[output_offset + spec_index] = shared_tokens[0];
        rejected_seen = true;
      }
    }
    __syncthreads();
  }

  if (thread_index == 0) {
    output[output_offset + n_speculative_tokens] =
        rejected_seen ? -1 : bonus_token_ids[batch_index];
  }
}

__global__ void greedy_rejection_sample_kernel(
    const int64_t* __restrict__ draft_token_ids,
    const int64_t* __restrict__ target_token_ids,
    const int64_t* __restrict__ bonus_token_ids,
    int64_t batch_size,
    int32_t num_speculative_tokens,
    int64_t draft_stride_0,
    int64_t draft_stride_1,
    int64_t target_stride_0,
    int64_t target_stride_1,
    int64_t bonus_stride_0,
    int64_t* __restrict__ accepted_token_ids,
    int64_t* __restrict__ masked_token_ids) {
  const int64_t batch_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const int64_t output_offset =
      batch_index * static_cast<int64_t>(num_speculative_tokens + 1);
  bool rejected = false;
  for (int32_t index = 0; index < num_speculative_tokens; ++index) {
    const int64_t target_token =
        target_token_ids[batch_index * target_stride_0 +
                         static_cast<int64_t>(index) * target_stride_1];
    accepted_token_ids[output_offset + index] = target_token;
    masked_token_ids[output_offset + index] = rejected ? -1 : target_token;
    rejected =
        rejected ||
        target_token !=
            draft_token_ids[batch_index * draft_stride_0 +
                            static_cast<int64_t>(index) * draft_stride_1];
  }

  const int64_t bonus_token = bonus_token_ids[batch_index * bonus_stride_0];
  accepted_token_ids[output_offset + num_speculative_tokens] = bonus_token;
  masked_token_ids[output_offset + num_speculative_tokens] =
      rejected ? -1 : bonus_token;
}
}  // namespace

torch::Tensor rejection_sample_target_only(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& recovery_exponential,
    const torch::Tensor& bonus_token_ids) {
  CHECK_EQ(draft_token_ids.dim(), 2);
  CHECK_GT(draft_token_ids.size(1), 0);
  CHECK_EQ(draft_probs.sizes(), draft_token_ids.sizes());
  CHECK_EQ(target_probs.dim(), 3);
  CHECK_EQ(target_probs.size(0), draft_token_ids.size(0));
  CHECK_EQ(target_probs.size(1), draft_token_ids.size(1));
  CHECK_EQ(uniform_rand.sizes(), draft_token_ids.sizes());
  CHECK_EQ(recovery_exponential.sizes(), target_probs.sizes());
  CHECK_EQ(bonus_token_ids.numel(), draft_token_ids.size(0));
  CHECK_GT(target_probs.size(2), 0);

  const int64_t batch_size = draft_token_ids.size(0);
  const int64_t n_speculative_tokens = draft_token_ids.size(1);
  auto ids = draft_token_ids.to(torch::kInt64).contiguous();
  auto draft_probability = draft_probs.to(torch::kFloat32).contiguous();
  auto target_probability = target_probs.to(torch::kFloat32).contiguous();
  auto acceptance_rand = uniform_rand.to(torch::kFloat32).contiguous();
  auto resample_exponential =
      recovery_exponential.to(torch::kFloat32).contiguous();
  auto bonus =
      bonus_token_ids.to(torch::kInt64).reshape({batch_size}).contiguous();
  auto output = torch::empty(
      {batch_size, n_speculative_tokens + 1},
      torch::TensorOptions().dtype(torch::kInt64).device(ids.device()));
  if (batch_size == 0) {
    return output;
  }

  const at::cuda::OptionalCUDAGuard device_guard(ids.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  rejection_sample_target_only_kernel<<<batch_size, kBlockSize, 0, stream>>>(
      ids.data_ptr<int64_t>(),
      draft_probability.data_ptr<float>(),
      target_probability.data_ptr<float>(),
      acceptance_rand.data_ptr<float>(),
      resample_exponential.data_ptr<float>(),
      bonus.data_ptr<int64_t>(),
      static_cast<int32_t>(n_speculative_tokens),
      static_cast<int32_t>(target_probability.size(2)),
      output.data_ptr<int64_t>());
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> greedy_rejection_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& target_token_ids,
    const torch::Tensor& bonus_token_ids) {
  CHECK_EQ(draft_token_ids.dim(), 2);
  CHECK_GT(draft_token_ids.size(1), 0);
  CHECK_EQ(target_token_ids.sizes(), draft_token_ids.sizes());
  CHECK_EQ(bonus_token_ids.dim(), 2);
  CHECK_EQ(bonus_token_ids.size(0), draft_token_ids.size(0));
  CHECK_EQ(bonus_token_ids.size(1), 1);
  CHECK(draft_token_ids.device() == target_token_ids.device());
  CHECK(draft_token_ids.device() == bonus_token_ids.device());
  CHECK_EQ(draft_token_ids.scalar_type(), torch::kLong);
  CHECK_EQ(target_token_ids.scalar_type(), torch::kLong);
  CHECK_EQ(bonus_token_ids.scalar_type(), torch::kLong);

  const int64_t batch_size = draft_token_ids.size(0);
  const int64_t num_speculative_tokens = draft_token_ids.size(1);
  auto options = torch::TensorOptions()
                     .dtype(torch::kLong)
                     .device(draft_token_ids.device());
  auto accepted_token_ids =
      torch::empty({batch_size, num_speculative_tokens + 1}, options);
  auto masked_token_ids =
      torch::empty({batch_size, num_speculative_tokens + 1}, options);
  if (batch_size == 0) {
    return {accepted_token_ids, masked_token_ids};
  }

  const at::cuda::OptionalCUDAGuard device_guard(draft_token_ids.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const int64_t grid_size =
      (batch_size + kGreedyBlockSize - 1) / kGreedyBlockSize;
  greedy_rejection_sample_kernel<<<grid_size, kGreedyBlockSize, 0, stream>>>(
      draft_token_ids.data_ptr<int64_t>(),
      target_token_ids.data_ptr<int64_t>(),
      bonus_token_ids.data_ptr<int64_t>(),
      batch_size,
      static_cast<int32_t>(num_speculative_tokens),
      draft_token_ids.stride(0),
      draft_token_ids.stride(1),
      target_token_ids.stride(0),
      target_token_ids.stride(1),
      bonus_token_ids.stride(0),
      accepted_token_ids.data_ptr<int64_t>(),
      masked_token_ids.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {accepted_token_ids, masked_token_ids};
}

}  // namespace xllm::kernel::musa
