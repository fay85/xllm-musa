/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cassert>
#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kMaxExperts = 1024;

__global__ void moe_align_kernel(const int32_t* ids,
                                 int32_t* sorted_ids,
                                 int32_t* expert_ids,
                                 int32_t* padded_count,
                                 int32_t assignments,
                                 int32_t num_experts,
                                 int32_t block_size,
                                 int32_t capacity) {
  __shared__ int32_t offsets[kMaxExperts + 1];
  for (int32_t expert = threadIdx.x; expert < num_experts; expert += kThreads) {
    int32_t count = 0;
    for (int32_t row = 0; row < assignments; ++row) {
      count += ids[row] == expert;
    }
    offsets[expert + 1] = (count + block_size - 1) / block_size * block_size;
  }
  for (int32_t row = threadIdx.x; row < capacity; row += kThreads) {
    sorted_ids[row] = assignments;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    offsets[0] = 0;
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      offsets[expert + 1] += offsets[expert];
    }
    *padded_count = offsets[num_experts];
  }
  __syncthreads();
  for (int32_t expert = threadIdx.x; expert < num_experts; expert += kThreads) {
    for (int32_t block = offsets[expert] / block_size;
         block < offsets[expert + 1] / block_size;
         ++block) {
      expert_ids[block] = expert;
    }
  }
  for (int32_t row = threadIdx.x; row < assignments; row += kThreads) {
    const int32_t expert = ids[row];
    if (expert < 0 || expert >= num_experts) {
      __assert_fail(
          "expert >= 0 && expert < num_experts", __FILE__, __LINE__, __func__);
      return;
    }
    int32_t position = offsets[expert];
    for (int32_t previous = 0; previous < row; ++previous) {
      position += ids[previous] == expert;
    }
    sorted_ids[position] = row;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_align_block_size(
    const torch::Tensor& topk_ids,
    int64_t num_experts,
    int64_t block_size) {
  CHECK_EQ(topk_ids.dim(), 2);
  CHECK_EQ(topk_ids.scalar_type(), torch::kInt32);
  CHECK(topk_ids.is_contiguous());
  CHECK(topk_ids.device().is_privateuseone());
  CHECK_GT(topk_ids.size(0), 0);
  CHECK_LE(topk_ids.size(0), 8);
  CHECK_EQ(topk_ids.size(1), 8);
  CHECK_GT(num_experts, 0);
  CHECK_LE(num_experts, kMaxExperts);
  CHECK(block_size == 16 || block_size == 32);
  const int64_t assignments = topk_ids.numel();
  const int64_t capacity =
      assignments < num_experts + 1
          ? assignments * block_size
          : assignments + (num_experts + 1) * (block_size - 1);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
  auto sorted_ids = torch::empty({capacity}, topk_ids.options());
  auto expert_ids = torch::empty({(capacity + block_size - 1) / block_size},
                                 topk_ids.options());
  auto padded_count = torch::empty({1}, topk_ids.options());
  const musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  moe_align_kernel<<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(),
      sorted_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      padded_count.data_ptr<int32_t>(),
      static_cast<int32_t>(assignments),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(block_size),
      static_cast<int32_t>(capacity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sorted_ids, expert_ids, padded_count};
}

}  // namespace xllm::kernel::musa
