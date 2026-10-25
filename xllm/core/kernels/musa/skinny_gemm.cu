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
#include <musa_bf16.h>

#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/skinny_gemm.cuh"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kColsPerBlock = 2;

template <int32_t kRows>
__global__ void __launch_bounds__(skinny::kThreads)
    skinny_gemm_kernel(const __mt_bfloat16* __restrict__ input,
                       const __mt_bfloat16* __restrict__ weight,
                       __mt_bfloat16* __restrict__ output,
                       int32_t cols,
                       int32_t k) {
  __shared__ float warp_partials[skinny::kWarps][kColsPerBlock * kRows];
  skinny::block_gemm<kRows, kColsPerBlock>(
      input,
      weight,
      output,
      static_cast<int32_t>(blockIdx.x) * kColsPerBlock,
      cols,
      k,
      warp_partials);
}

template <int32_t kRows>
void launch_skinny_gemm(const torch::Tensor& input,
                        const torch::Tensor& weight,
                        torch::Tensor& output,
                        cudaStream_t stream) {
  const int32_t k = static_cast<int32_t>(input.size(1));
  const int32_t cols = static_cast<int32_t>(weight.size(0));
  const int32_t blocks = (cols + kColsPerBlock - 1) / kColsPerBlock;
  skinny_gemm_kernel<kRows><<<blocks, skinny::kThreads, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __mt_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
      cols,
      k);
}

bool is_16b_aligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignof(uint4) == 0;
}

}  // namespace

bool skinny_bf16_gemm_supported(const torch::Tensor& input,
                                const torch::Tensor& weight,
                                const torch::Tensor& output) {
  if (input.dim() != 2 || weight.dim() != 2 || output.dim() != 2) {
    return false;
  }
  if (input.scalar_type() != torch::kBFloat16 ||
      weight.scalar_type() != torch::kBFloat16 ||
      output.scalar_type() != torch::kBFloat16) {
    return false;
  }
  const int64_t rows = input.size(0);
  const int64_t k = input.size(1);
  const int64_t cols = weight.size(0);
  return rows >= 1 && rows <= skinny::kMaxRows && cols >= 1 && cols <= 256 &&
         weight.size(1) == k && k % skinny::kBf16PerVector == 0 &&
         k / skinny::kBf16PerVector <=
             skinny::kThreads * skinny::kMaxVectorsPerThread &&
         output.size(0) == rows && output.size(1) == cols &&
         input.is_contiguous() && weight.is_contiguous() &&
         output.is_contiguous() && is_16b_aligned(input) &&
         is_16b_aligned(weight);
}

void skinny_bf16_gemm(const torch::Tensor& input,
                      const torch::Tensor& weight,
                      torch::Tensor& output) {
  CHECK(skinny_bf16_gemm_supported(input, weight, output))
      << "unsupported skinny BF16 GEMM: input=" << input.sizes()
      << ", weight=" << weight.sizes() << ", output=" << output.sizes();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (input.size(0)) {
    case 1:
      launch_skinny_gemm<1>(input, weight, output, stream);
      break;
    case 2:
      launch_skinny_gemm<2>(input, weight, output, stream);
      break;
    case 3:
      launch_skinny_gemm<3>(input, weight, output, stream);
      break;
    case 4:
      launch_skinny_gemm<4>(input, weight, output, stream);
      break;
    case 5:
      launch_skinny_gemm<5>(input, weight, output, stream);
      break;
    case 6:
      launch_skinny_gemm<6>(input, weight, output, stream);
      break;
    case 7:
      launch_skinny_gemm<7>(input, weight, output, stream);
      break;
    default:
      launch_skinny_gemm<skinny::kMaxRows>(input, weight, output, stream);
      break;
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace xllm::kernel::musa
