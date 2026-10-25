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

// MUSA-native port of SGLang's TileLang
// dsv4_fused_q_rmsnorm_rope_h512_r64 kernel. One warp owns one query head,
// reducing the 512-wide variance and rotating the final interleaved 64 values
// without an intermediate normalized-query allocation.

#include <c10/util/BFloat16.h>
#include <torch/torch.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <limits>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kHiddenDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kNopeDim = kHiddenDim - kRopeDim;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = 4;
constexpr int32_t kThreads = kWarpSize * kWarpsPerBlock;
constexpr int32_t kKvCompatThreads = 1024;

using CubAddOp = cub::Sum;

__global__ void deepseek_v4_q_rmsnorm_rope_by_token_kernel(
    c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ cos,
    const c10::BFloat16* __restrict__ sin,
    int64_t heads,
    int64_t cos_stride,
    float eps) {
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
  const int64_t head = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (head >= heads) {
    return;
  }
  const int64_t token = static_cast<int64_t>(blockIdx.y);
  c10::BFloat16* row_ptr = query + (token * heads + head) * kHiddenDim;

  float sumsq = 0.0f;
#pragma unroll
  for (int32_t tile = 0; tile < kHiddenDim / kWarpSize; ++tile) {
    const float value = static_cast<float>(row_ptr[tile * kWarpSize + lane]);
    sumsq += value * value;
  }
#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset, kWarpSize);
  }
  const float rrms =
      __shfl_sync(0xffffffffu,
                  rsqrtf(sumsq / static_cast<float>(kHiddenDim) + eps),
                  0,
                  kWarpSize);

  // Match SGLang's numerical order: store the normalized non-RoPE prefix as
  // BF16, and round each normalized RoPE value to BF16 before the FP32 rotate.
#pragma unroll
  for (int32_t tile = 0; tile < kNopeDim / kWarpSize; ++tile) {
    const int32_t col = tile * kWarpSize + lane;
    const float value = static_cast<float>(row_ptr[col]);
    row_ptr[col] = static_cast<c10::BFloat16>(value * rrms);
  }

  const int32_t even_col = static_cast<int32_t>(kNopeDim) + lane * 2;
  const int32_t odd_col = even_col + 1;
  const float even = static_cast<float>(
      static_cast<c10::BFloat16>(static_cast<float>(row_ptr[even_col]) * rrms));
  const float odd = static_cast<float>(
      static_cast<c10::BFloat16>(static_cast<float>(row_ptr[odd_col]) * rrms));
  const c10::BFloat16* cos_row = cos + token * cos_stride;
  const c10::BFloat16* sin_row = sin + token * cos_stride;
  const float c = static_cast<float>(cos_row[lane * 2]);
  const float s = static_cast<float>(sin_row[lane * 2]);
  row_ptr[even_col] = static_cast<c10::BFloat16>(even * c - odd * s);
  row_ptr[odd_col] = static_cast<c10::BFloat16>(odd * c + even * s);
}

template <int32_t HiddenDim>
__global__ void deepseek_v4_weighted_rmsnorm_rope_kernel(
    c10::BFloat16* __restrict__ input,
    const c10::BFloat16* __restrict__ weight,
    const c10::BFloat16* __restrict__ cos,
    const c10::BFloat16* __restrict__ sin,
    int64_t cos_stride,
    float eps) {
  constexpr int32_t NopeDim = HiddenDim - kRopeDim;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  c10::BFloat16* row_ptr = input + token * HiddenDim;

  // Reproduce xLLM's generic rms_norm reduction exactly. KV is recurrent
  // state, so the faster warp-per-row summation can accumulate a one-token
  // divergence after a long decode even though short tests pass.
  float variance = 0.0f;
  if (tid < HiddenDim) {
    const float value = static_cast<float>(row_ptr[tid]);
    variance = value * value;
  }
  using BlockReduce = cub::BlockReduce<float, kKvCompatThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  __shared__ float s_rrms;
  variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  if (tid == 0) {
    s_rrms = rsqrtf(variance / static_cast<float>(HiddenDim) + eps);
  }
  __syncthreads();

  if (tid < NopeDim) {
    const float value = static_cast<float>(row_ptr[tid]);
    row_ptr[tid] = static_cast<c10::BFloat16>(value * s_rrms) * weight[tid];
  }

  if (tid < kRopeDim / 2) {
    const int32_t even_col = NopeDim + tid * 2;
    const int32_t odd_col = even_col + 1;
    const c10::BFloat16 even_weighted =
        static_cast<c10::BFloat16>(static_cast<float>(row_ptr[even_col]) *
                                   s_rrms) *
        weight[even_col];
    const c10::BFloat16 odd_weighted =
        static_cast<c10::BFloat16>(static_cast<float>(row_ptr[odd_col]) *
                                   s_rrms) *
        weight[odd_col];
    const float even = static_cast<float>(even_weighted);
    const float odd = static_cast<float>(odd_weighted);
    const c10::BFloat16* cos_row = cos + token * cos_stride;
    const c10::BFloat16* sin_row = sin + token * cos_stride;
    const float c = static_cast<float>(cos_row[tid * 2]);
    const float s = static_cast<float>(sin_row[tid * 2]);
    row_ptr[even_col] = static_cast<c10::BFloat16>(even * c - odd * s);
    row_ptr[odd_col] = static_cast<c10::BFloat16>(odd * c + even * s);
  }
}

// Keep the established inplace kernels unchanged. These out-of-place variants
// fuse only packed-projection materialization, not a different reduction tree.
__global__ void deepseek_v4_q_rmsnorm_rope_out_kernel(
    const c10::BFloat16* __restrict__ query,
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ cos,
    const c10::BFloat16* __restrict__ sin,
    int64_t heads,
    int64_t row_stride,
    int64_t cos_stride,
    float eps) {
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
  const int64_t head = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (head >= heads) {
    return;
  }
  const int64_t token = static_cast<int64_t>(blockIdx.y);
  const c10::BFloat16* row_ptr = query + token * row_stride + head * kHiddenDim;
  c10::BFloat16* out = output + (token * heads + head) * kHiddenDim;
  float sumsq = 0.0f;
#pragma unroll
  for (int32_t tile = 0; tile < kHiddenDim / kWarpSize; ++tile) {
    const float value = static_cast<float>(row_ptr[tile * kWarpSize + lane]);
    sumsq += value * value;
  }
#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset, kWarpSize);
  }
  const float rrms = __shfl_sync(0xffffffffu,
      rsqrtf(sumsq / static_cast<float>(kHiddenDim) + eps), 0, kWarpSize);
#pragma unroll
  for (int32_t tile = 0; tile < kNopeDim / kWarpSize; ++tile) {
    const int32_t col = tile * kWarpSize + lane;
    const float value = static_cast<float>(row_ptr[col]);
    out[col] = static_cast<c10::BFloat16>(value * rrms);
  }
  const int32_t even_col = static_cast<int32_t>(kNopeDim) + lane * 2;
  const int32_t odd_col = even_col + 1;
  const float even = static_cast<float>(
      static_cast<c10::BFloat16>(static_cast<float>(row_ptr[even_col]) * rrms));
  const float odd = static_cast<float>(
      static_cast<c10::BFloat16>(static_cast<float>(row_ptr[odd_col]) * rrms));
  const float c = static_cast<float>(cos[token * cos_stride + lane * 2]);
  const float s = static_cast<float>(sin[token * cos_stride + lane * 2]);
  out[even_col] = static_cast<c10::BFloat16>(even * c - odd * s);
  out[odd_col] = static_cast<c10::BFloat16>(odd * c + even * s);
}

__global__ void deepseek_v4_weighted_rmsnorm_rope_out_kernel(
    const c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ weight,
    const c10::BFloat16* __restrict__ cos,
    const c10::BFloat16* __restrict__ sin,
    int64_t row_stride,
    int64_t cos_stride,
    float eps) {
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const c10::BFloat16* row_ptr = input + token * row_stride;
  c10::BFloat16* out = output + token * kHiddenDim;
  float variance = 0.0f;
  if (tid < kHiddenDim) {
    const float value = static_cast<float>(row_ptr[tid]);
    variance = value * value;
  }
  using BlockReduce = cub::BlockReduce<float, kKvCompatThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  __shared__ float s_rrms;
  variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  if (tid == 0) {
    s_rrms = rsqrtf(variance / static_cast<float>(kHiddenDim) + eps);
  }
  __syncthreads();
  if (tid < kNopeDim) {
    const float value = static_cast<float>(row_ptr[tid]);
    out[tid] = static_cast<c10::BFloat16>(value * s_rrms) * weight[tid];
  }
  if (tid < kRopeDim / 2) {
    const int32_t even_col = static_cast<int32_t>(kNopeDim) + tid * 2;
    const int32_t odd_col = even_col + 1;
    const c10::BFloat16 even_weighted =
        static_cast<c10::BFloat16>(static_cast<float>(row_ptr[even_col]) * s_rrms) * weight[even_col];
    const c10::BFloat16 odd_weighted =
        static_cast<c10::BFloat16>(static_cast<float>(row_ptr[odd_col]) * s_rrms) * weight[odd_col];
    const float even = static_cast<float>(even_weighted);
    const float odd = static_cast<float>(odd_weighted);
    const float c = static_cast<float>(cos[token * cos_stride + tid * 2]);
    const float s = static_cast<float>(sin[token * cos_stride + tid * 2]);
    out[even_col] = static_cast<c10::BFloat16>(even * c - odd * s);
    out[odd_col] = static_cast<c10::BFloat16>(odd * c + even * s);
  }
}

bool valid_rope_out(const torch::Tensor& input,
                    const torch::Tensor& cos,
                    const torch::Tensor& sin,
                    const torch::Tensor& output,
                    int64_t heads) {
  return input.defined() && output.defined() && cos.defined() && sin.defined() &&
         input.device().is_privateuseone() && (input.dim() == 2 || input.dim() == 3) &&
         input.size(0) > 0 && heads > 0 && input.size(-1) == kHiddenDim &&
         input.stride(-1) == 1 && input.stride(0) >= heads * kHiddenDim &&
         (input.dim() != 3 || (input.size(1) == heads && input.stride(1) == kHiddenDim)) &&
         output.sizes() == input.sizes() && output.is_contiguous() &&
         output.device() == input.device() && output.scalar_type() == torch::kBFloat16 &&
         input.scalar_type() == torch::kBFloat16 && !input.is_alias_of(output) &&
         cos.dim() == 2 && sin.dim() == 2 && cos.sizes() == sin.sizes() &&
         cos.size(0) == input.size(0) && cos.size(1) >= kRopeDim &&
         cos.is_contiguous() && sin.is_contiguous() &&
         cos.device() == input.device() && sin.device() == input.device() &&
         cos.scalar_type() == torch::kBFloat16 && sin.scalar_type() == torch::kBFloat16 &&
         !cos.is_alias_of(output) && !sin.is_alias_of(output) &&
         input.size(0) <= std::numeric_limits<uint32_t>::max() &&
         (heads + kWarpsPerBlock - 1) / kWarpsPerBlock <= std::numeric_limits<uint32_t>::max();
}

}  // namespace

bool deepseek_v4_q_rmsnorm_rope_out(const torch::Tensor& query,
                                    const torch::Tensor& cos,
                                    const torch::Tensor& sin,
                                    double eps,
                                    const torch::Tensor& output) {
  if (!query.defined() || query.dim() != 3 ||
      !valid_rope_out(query, cos, sin, output, query.size(1))) {
    return false;
  }
  const XLLMOptionalDeviceGuard guard(query.device());
  const dim3 grid(static_cast<uint32_t>((query.size(1) + kWarpsPerBlock - 1) / kWarpsPerBlock),
                  static_cast<uint32_t>(query.size(0)));
  deepseek_v4_q_rmsnorm_rope_out_kernel<<<grid, kThreads, 0, current_musa_stream(query.device())>>>(
      query.data_ptr<c10::BFloat16>(), output.data_ptr<c10::BFloat16>(),
      cos.data_ptr<c10::BFloat16>(), sin.data_ptr<c10::BFloat16>(),
      query.size(1), query.stride(0), cos.size(1), static_cast<float>(eps));
  check_musa_kernel_launch("DSV4 packed Q copy norm RoPE");
  return true;
}

bool deepseek_v4_weighted_rmsnorm_rope_out(const torch::Tensor& input,
                                           const torch::Tensor& weight,
                                           const torch::Tensor& cos,
                                           const torch::Tensor& sin,
                                           double eps,
                                           const torch::Tensor& output) {
  if (!valid_rope_out(input, cos, sin, output, 1) || !weight.defined() ||
      weight.device() != input.device() || weight.scalar_type() != torch::kBFloat16 ||
      weight.dim() != 1 || weight.size(0) != kHiddenDim || !weight.is_contiguous() ||
      weight.is_alias_of(output)) {
    return false;
  }
  const XLLMOptionalDeviceGuard guard(input.device());
  deepseek_v4_weighted_rmsnorm_rope_out_kernel<<<static_cast<uint32_t>(input.size(0)),
      kKvCompatThreads, 0, current_musa_stream(input.device())>>>(
      input.data_ptr<c10::BFloat16>(), output.data_ptr<c10::BFloat16>(),
      weight.data_ptr<c10::BFloat16>(), cos.data_ptr<c10::BFloat16>(),
      sin.data_ptr<c10::BFloat16>(), input.stride(0), cos.size(1), static_cast<float>(eps));
  check_musa_kernel_launch("DSV4 packed KV copy norm RoPE");
  return true;
}

bool deepseek_v4_q_rmsnorm_rope_inplace(torch::Tensor& query,
                                        const torch::Tensor& cos,
                                        const torch::Tensor& sin,
                                        double eps) {
  if (!query.defined() || !cos.defined() || !sin.defined() ||
      query.device().is_cpu() || query.dim() != 3 || query.size(0) <= 0 ||
      query.size(1) <= 0 || query.size(2) != kHiddenDim ||
      !query.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() ||
      cos.dim() != 2 || sin.dim() != 2 || cos.sizes() != sin.sizes() ||
      cos.size(0) != query.size(0) || cos.size(1) < kRopeDim ||
      query.device() != cos.device() || query.device() != sin.device() ||
      query.scalar_type() != torch::kBFloat16 ||
      cos.scalar_type() != query.scalar_type() ||
      sin.scalar_type() != query.scalar_type() ||
      query.size(0) > std::numeric_limits<unsigned int>::max() ||
      (query.size(1) + kWarpsPerBlock - 1) / kWarpsPerBlock >
          std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(query));
  const dim3 grid(static_cast<unsigned int>(
                      (query.size(1) + kWarpsPerBlock - 1) / kWarpsPerBlock),
                  static_cast<unsigned int>(query.size(0)));
  deepseek_v4_q_rmsnorm_rope_by_token_kernel<<<grid,
                                               kThreads,
                                               0,
                                               current_musa_stream(
                                                   query.device())>>>(
      query.data_ptr<c10::BFloat16>(),
      cos.data_ptr<c10::BFloat16>(),
      sin.data_ptr<c10::BFloat16>(),
      query.size(1),
      cos.size(1),
      static_cast<float>(eps));
  check_musa_kernel_launch("DeepSeek-V4 fused RMSNorm RoPE kernel");
  return true;
}

bool deepseek_v4_weighted_rmsnorm_rope_inplace(torch::Tensor& input,
                                               const torch::Tensor& weight,
                                               const torch::Tensor& cos,
                                               const torch::Tensor& sin,
                                               double eps) {
  const int64_t hidden_dim = input.defined() && input.dim() >= 2
                                 ? input.size(input.dim() - 1)
                                 : int64_t{0};
  const bool shape_ok =
      input.defined() && hidden_dim == kHiddenDim &&
      ((input.dim() == 2 && input.size(1) == hidden_dim) ||
       (input.dim() == 3 && input.size(1) == 1 && input.size(2) == hidden_dim));
  if (!shape_ok || !weight.defined() || !cos.defined() || !sin.defined() ||
      input.device().is_cpu() || input.size(0) <= 0 || !input.is_contiguous() ||
      !weight.is_contiguous() || weight.dim() != 1 ||
      weight.size(0) != hidden_dim || !cos.is_contiguous() ||
      !sin.is_contiguous() || cos.dim() != 2 || sin.dim() != 2 ||
      cos.sizes() != sin.sizes() || cos.size(0) != input.size(0) ||
      cos.size(1) < kRopeDim || input.device() != weight.device() ||
      input.device() != cos.device() || input.device() != sin.device() ||
      input.scalar_type() != torch::kBFloat16 ||
      weight.scalar_type() != input.scalar_type() ||
      cos.scalar_type() != input.scalar_type() ||
      sin.scalar_type() != input.scalar_type() ||
      input.size(0) > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  const XLLMOptionalDeviceGuard device_guard(at::device_of(input));
  const unsigned int blocks = static_cast<unsigned int>(input.size(0));
  const musaStream_t stream = current_musa_stream(input.device());
  deepseek_v4_weighted_rmsnorm_rope_kernel<kHiddenDim>
      <<<blocks, kKvCompatThreads, 0, stream>>>(
          input.data_ptr<c10::BFloat16>(),
          weight.data_ptr<c10::BFloat16>(),
          cos.data_ptr<c10::BFloat16>(),
          sin.data_ptr<c10::BFloat16>(),
          cos.size(1),
          static_cast<float>(eps));
  check_musa_kernel_launch("DeepSeek-V4 fused RMSNorm RoPE kernel");
  return true;
}

}  // namespace xllm::kernel::musa
