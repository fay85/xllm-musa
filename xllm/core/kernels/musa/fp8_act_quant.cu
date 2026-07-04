/* Copyright 2025-2026 The xLLM Authors.

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

// clang-format off
#include <musa_bf16.h>
#include <musa_fp8.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
// clang-format on

#include <cstdint>
#include <tuple>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::cuda {

namespace {

// DeepSeek block-FP8 activation quantization, specialized for the Qwen3.5
// serving path: bf16 input, e4m3 output, group size 128 along K, row-major
// scale grid [M, K/128]. Each 128-element group is handled by one 32-lane
// subwarp (4 elems/lane, vectorized 8-byte loads + a single float4->fp8x4
// hardware conversion), 8 groups per 256-thread block.
constexpr int kGroupSize = 128;
constexpr int kThreadsPerGroup = 32;
constexpr int kElemsPerThread = kGroupSize / kThreadsPerGroup;  // 4
constexpr int kSubwarpsPerBlock = 8;
constexpr int kBlockThreads = kSubwarpsPerBlock * kThreadsPerGroup;  // 256
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kEps = 1e-10f;

__global__ void per_token_group_quant_fp8_bf16_g128_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8x4_storage_t* __restrict__ output_q_vec4,  // uint32 = 4 fp8
    float* __restrict__ output_s,
    int64_t num_groups) {
  const int subwarp = threadIdx.x / kThreadsPerGroup;
  const int lane = threadIdx.x % kThreadsPerGroup;
  const int64_t group =
      static_cast<int64_t>(blockIdx.x) * kSubwarpsPerBlock + subwarp;
  if (group >= num_groups) {
    return;
  }

  // Vectorized 8-byte (4 x bf16) load for this lane's slice of the group.
  const int64_t in_base =
      group * kGroupSize + static_cast<int64_t>(lane) * kElemsPerThread;
  const uint64_t raw = *reinterpret_cast<const uint64_t*>(input + in_base);
  const __mt_bfloat16* raw_bf16 = reinterpret_cast<const __mt_bfloat16*>(&raw);

  float vals[kElemsPerThread];
  float local_absmax = 0.0f;
#pragma unroll
  for (int j = 0; j < kElemsPerThread; ++j) {
    vals[j] = __bfloat162float(raw_bf16[j]);
    local_absmax = fmaxf(local_absmax, fabsf(vals[j]));
  }

  // Butterfly max-reduce across the 32-lane subwarp; every lane ends with the
  // group absmax (all lanes need the scale to quantize their own elements).
#pragma unroll
  for (int offset = kThreadsPerGroup / 2; offset > 0; offset >>= 1) {
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(0xffffffffu, local_absmax, offset, kThreadsPerGroup));
  }

  const float scale_inv = fmaxf(local_absmax / kFp8E4M3Max, kEps);
  if (lane == 0) {
    output_s[group] = scale_inv;
  }

  const float scale = 1.0f / scale_inv;
  const float4 scaled = make_float4(
      vals[0] * scale, vals[1] * scale, vals[2] * scale, vals[3] * scale);
  const __mt_fp8x4_storage_t packed =
      __musa_cvt_float4_to_fp8x4(scaled, __MT_SATFINITE, __MT_E4M3);
  output_q_vec4[group * (kGroupSize / kElemsPerThread) + lane] = packed;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> per_token_group_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size) {
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "per_token_group_quant_fp8 (MUSA g128) supports bf16 input only");
  TORCH_CHECK(group_size == kGroupSize,
              "per_token_group_quant_fp8 (MUSA g128) requires group_size=128");
  TORCH_CHECK(input.stride(-1) == 1,
              "per_token_group_quant_fp8: input last dim must be contiguous");

  const int64_t k = input.size(-1);
  TORCH_CHECK(k % kGroupSize == 0,
              "per_token_group_quant_fp8: K must be divisible by 128");
  const int64_t k_groups = k / kGroupSize;
  const int64_t m = input.numel() / k;
  const int64_t num_groups = m * k_groups;

  auto out_q =
      torch::empty(input.sizes(), input.options().dtype(torch::kFloat8_e4m3fn));
  auto scale_sizes = input.sizes().vec();
  scale_sizes.back() = k_groups;
  auto out_scale =
      torch::empty(scale_sizes, input.options().dtype(torch::kFloat32));

  if (num_groups == 0) {
    return std::make_tuple(out_q, out_scale);
  }

  const int64_t grid =
      (num_groups + kSubwarpsPerBlock - 1) / kSubwarpsPerBlock;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  per_token_group_quant_fp8_bf16_g128_kernel<<<static_cast<unsigned int>(grid),
                                               kBlockThreads,
                                               0,
                                               stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__mt_fp8x4_storage_t*>(
          out_q.data_ptr<c10::Float8_e4m3fn>()),
      out_scale.data_ptr<float>(),
      num_groups);
  C10_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(out_q, out_scale);
}

}  // namespace xllm::kernel::cuda
