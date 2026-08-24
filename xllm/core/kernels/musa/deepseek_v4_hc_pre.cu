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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kThreads = 256;
constexpr int64_t kHcMult = 4;
constexpr int64_t kMixHc = 24;
constexpr int64_t kMateHcPrenormSplits = 32;
constexpr int32_t kHcFinishHiddenTile = 512;
constexpr char kMateHcPrenormUri[] = "mhc_prenorm_ops";
constexpr char kMateHcVerifyPrenormUri[] =
    "mate_jit_tf32_hc_prenorm_64x32x32_3stages";
constexpr char kMateHcBigFuseUri[] = "mhc_pre_big_fuse_ops";

template <typename scalar_t>
__global__ void dsv4_hc_pre_finish_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gemm_partials,
    const float* __restrict__ sqr_sum_partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    scalar_t* __restrict__ output,
    float* __restrict__ post,
    float* __restrict__ comb,
    int64_t dim,
    int64_t sinkhorn_iters,
    float norm_eps,
    float hc_eps);

template <typename scalar_t>
__global__ void dsv4_hc_pre_finish_split_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gemm_partials,
    const float* __restrict__ sqr_sum_partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ output_sqr_partials,
    int32_t* __restrict__ finish_counter,
    const scalar_t* __restrict__ norm_weight,
    scalar_t* __restrict__ output,
    float* __restrict__ post,
    float* __restrict__ comb,
    int64_t dim,
    int64_t sinkhorn_iters,
    float norm_eps,
    float hc_eps);

void check_last_error();

struct MateHcWorkspace {
  torch::Tensor gemm_partials;
  torch::Tensor sqr_sum_partials;
  torch::Tensor finish_sqr_partials;
  torch::Tensor finish_counter;
};

MateHcWorkspace& mate_hc_workspace(const torch::Tensor& x, int64_t tokens) {
  static thread_local std::unordered_map<int64_t, MateHcWorkspace> workspaces;
  const int64_t key = (static_cast<int64_t>(x.device().index()) << 32) | tokens;
  auto& workspace = workspaces[key];
  if (!workspace.gemm_partials.defined() ||
      workspace.gemm_partials.size(1) != tokens) {
    const auto options = x.options().dtype(torch::kFloat32);
    workspace.gemm_partials =
        torch::empty({kMateHcPrenormSplits, tokens, kMixHc}, options);
    workspace.sqr_sum_partials =
        torch::empty({kMateHcPrenormSplits, tokens}, options);
  }
  const int64_t finish_tiles =
      (x.size(2) + kHcFinishHiddenTile - 1) / kHcFinishHiddenTile;
  if (!workspace.finish_sqr_partials.defined() ||
      workspace.finish_sqr_partials.size(0) != tokens ||
      workspace.finish_sqr_partials.size(1) != finish_tiles) {
    const auto options = x.options().dtype(torch::kFloat32);
    workspace.finish_sqr_partials =
        torch::empty({tokens, finish_tiles}, options);
    workspace.finish_counter =
        torch::zeros({tokens}, x.options().dtype(torch::kInt32));
  }
  return workspace;
}

ffi::Optional<ffi::Function> get_mate_hc_function(const char* uri,
                                                  const char* name) {
  const char* root = std::getenv("FLASHINFER_OPS_PATH");
  if (root == nullptr || root[0] == '\0') {
    return ffi::Optional<ffi::Function>();
  }
  std::ifstream so_file(path_to_uri_so_lib(uri));
  if (!so_file.good()) {
    return ffi::Optional<ffi::Function>();
  }
  try {
    return get_module(uri)->GetFunction(name);
  } catch (const std::exception&) {
    return ffi::Optional<ffi::Function>();
  }
}

bool try_mate_hc_pre_decode(const torch::Tensor& x,
                            const torch::Tensor& hc_fn,
                            const torch::Tensor& hc_scale,
                            const torch::Tensor& hc_base,
                            const torch::Tensor& output,
                            const torch::Tensor& post,
                            const torch::Tensor& comb,
                            int64_t sinkhorn_iters,
                            double norm_eps,
                            double hc_eps,
                            const std::optional<torch::Tensor>& norm_weight,
                            bool* output_is_normalized) {
  // The Mate prefill pair is model-level validated through the largest local
  // TP8/SP8 shard produced by the fixed 2.5K gate (313 rows).  Keep larger or
  // otherwise unsupported shapes on the local correctness-first kernel.
  constexpr int64_t kMatePrefillValidatedMaxTokens = 320;
  const int64_t tokens = x.size(0);
  if (x.scalar_type() != torch::kBFloat16 || tokens < 1 ||
      tokens > kMatePrefillValidatedMaxTokens || sinkhorn_iters != 20) {
    return false;
  }
  const bool is_prefill = tokens > 1;
  const char* prenorm_uri =
      is_prefill ? kMateHcVerifyPrenormUri : kMateHcPrenormUri;
  auto prenorm = get_mate_hc_function(prenorm_uri, "tf32_hc_prenorm_gemm");
  if (!prenorm.defined()) {
    return false;
  }
  auto big_fuse = is_prefill
                      ? get_mate_hc_function(kMateHcBigFuseUri,
                                             "mhc_pre_big_fuse_kernel")
                      : ffi::Optional<ffi::Function>();
  if (is_prefill && !big_fuse.defined()) {
    return false;
  }
  try {
    auto& workspace = mate_hc_workspace(x, tokens);
    TvmffiStreamGuard stream_guard(x.device());
    prenorm.value()(to_ffi_tensor(x.reshape({tokens, kHcMult * x.size(2)})),
                    to_ffi_tensor(hc_fn),
                    to_ffi_tensor(workspace.gemm_partials),
                    to_ffi_tensor(workspace.sqr_sum_partials),
                    kMateHcPrenormSplits);
    if (is_prefill) {
      big_fuse.value()(to_ffi_tensor(workspace.gemm_partials),
                       to_ffi_tensor(workspace.sqr_sum_partials),
                       to_ffi_tensor(hc_scale),
                       to_ffi_tensor(hc_base),
                       to_ffi_tensor(x),
                       to_ffi_tensor(post),
                       to_ffi_tensor(comb),
                       to_ffi_tensor(output));
      if (output_is_normalized != nullptr) {
        *output_is_normalized = false;
      }
      LOG_FIRST_N(INFO, 1)
          << "DeepSeek-V4 MUSA Mate prefill HC prenorm and big-fuse enabled; "
          << "validated tokens <= " << kMatePrefillValidatedMaxTokens << ".";
      return true;
    }
    const XLLMOptionalDeviceGuard device_guard(at::device_of(x));
    const musaStream_t stream = current_musa_stream(x.device());
    const bool fuse_norm = norm_weight.has_value();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        x.scalar_type(),
        "deepseek_v4_mate_hc_pre_finish",
        [&] {
          if (fuse_norm) {
            const int64_t finish_tiles =
                (x.size(2) + kHcFinishHiddenTile - 1) / kHcFinishHiddenTile;
            dsv4_hc_pre_finish_split_kernel<scalar_t>
                <<<dim3(1, static_cast<unsigned int>(finish_tiles)),
                   kThreads,
                   0,
                   stream>>>(x.data_ptr<scalar_t>(),
                             workspace.gemm_partials.data_ptr<float>(),
                             workspace.sqr_sum_partials.data_ptr<float>(),
                             hc_scale.data_ptr<float>(),
                             hc_base.data_ptr<float>(),
                             workspace.finish_sqr_partials.data_ptr<float>(),
                             workspace.finish_counter.data_ptr<int32_t>(),
                             norm_weight->data_ptr<scalar_t>(),
                             output.data_ptr<scalar_t>(),
                             post.data_ptr<float>(),
                             comb.data_ptr<float>(),
                             x.size(2),
                             sinkhorn_iters,
                             static_cast<float>(norm_eps),
                             static_cast<float>(hc_eps));
          } else {
            dsv4_hc_pre_finish_kernel<scalar_t><<<1, kThreads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                workspace.gemm_partials.data_ptr<float>(),
                workspace.sqr_sum_partials.data_ptr<float>(),
                hc_scale.data_ptr<float>(),
                hc_base.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                post.data_ptr<float>(),
                comb.data_ptr<float>(),
                x.size(2),
                sinkhorn_iters,
                static_cast<float>(norm_eps),
                static_cast<float>(hc_eps));
          }
        });
    check_last_error();
    if (output_is_normalized != nullptr) {
      *output_is_normalized = fuse_norm;
    }
    LOG_FIRST_N(INFO, 1)
        << "DeepSeek-V4 MUSA Mate HC prenorm and split finish enabled; "
        << "fused RMSNorm=" << fuse_norm << ".";
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

__device__ __forceinline__ float stable_sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  const float exp_value = expf(value);
  return exp_value / (1.0f + exp_value);
}

template <typename scalar_t>
__global__ void dsv4_hc_pre_finish_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gemm_partials,
    const float* __restrict__ sqr_sum_partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    scalar_t* __restrict__ output,
    float* __restrict__ post,
    float* __restrict__ comb,
    int64_t dim,
    int64_t sinkhorn_iters,
    float norm_eps,
    float hc_eps) {
  __shared__ float
      shared[kMixHc + kHcMult + kHcMult + kHcMult * kHcMult + kThreads + 1];
  const int32_t tid = threadIdx.x;
  const int64_t token = blockIdx.x;
  const int64_t x_size = kHcMult * dim;
  const int64_t x_offset = token * x_size;
  float* mixes = shared;
  float* pre = mixes + kMixHc;
  float* post_shared = pre + kHcMult;
  float* comb_shared = post_shared + kHcMult;
  float* input_inv_rms = comb_shared + kHcMult * kHcMult + kThreads;

  if (tid < kMixHc) {
    float value = 0.0f;
#pragma unroll
    for (int64_t split = 0; split < kMateHcPrenormSplits; ++split) {
      value += gemm_partials[(split * gridDim.x + token) * kMixHc + tid];
    }
    mixes[tid] = value;
  }
  if (tid == 0) {
    float sqr_sum = 0.0f;
#pragma unroll
    for (int64_t split = 0; split < kMateHcPrenormSplits; ++split) {
      sqr_sum += sqr_sum_partials[split * gridDim.x + token];
    }
    input_inv_rms[0] = rsqrtf(sqr_sum / static_cast<float>(x_size) + norm_eps);
  }
  __syncthreads();

  if (tid < kMixHc) {
    mixes[tid] *= input_inv_rms[0];
  }
  __syncthreads();

  if (tid == 0) {
    for (int64_t row = 0; row < kHcMult; ++row) {
      pre[row] =
          stable_sigmoid(mixes[row] * hc_scale[0] + hc_base[row]) + hc_eps;
      post_shared[row] =
          2.0f * stable_sigmoid(mixes[kHcMult + row] * hc_scale[1] +
                                hc_base[kHcMult + row]);
    }

    constexpr int64_t kCombOffset = 2 * kHcMult;
    for (int64_t row = 0; row < kHcMult; ++row) {
      float max_value = -INFINITY;
      for (int64_t col = 0; col < kHcMult; ++col) {
        max_value =
            fmaxf(max_value,
                  mixes[kCombOffset + row * kHcMult + col] * hc_scale[2] +
                      hc_base[kCombOffset + row * kHcMult + col]);
      }
      float denominator = 0.0f;
      for (int64_t col = 0; col < kHcMult; ++col) {
        const float logit =
            mixes[kCombOffset + row * kHcMult + col] * hc_scale[2] +
            hc_base[kCombOffset + row * kHcMult + col];
        const float value = expf(logit - max_value);
        comb_shared[row * kHcMult + col] = value;
        denominator += value;
      }
      for (int64_t col = 0; col < kHcMult; ++col) {
        comb_shared[row * kHcMult + col] =
            comb_shared[row * kHcMult + col] / denominator + hc_eps;
      }
    }
    for (int64_t col = 0; col < kHcMult; ++col) {
      float denominator = hc_eps;
      for (int64_t row = 0; row < kHcMult; ++row) {
        denominator += comb_shared[row * kHcMult + col];
      }
      for (int64_t row = 0; row < kHcMult; ++row) {
        comb_shared[row * kHcMult + col] /= denominator;
      }
    }
    for (int64_t iter = 1; iter < sinkhorn_iters; ++iter) {
      for (int64_t row = 0; row < kHcMult; ++row) {
        float denominator = hc_eps;
        for (int64_t col = 0; col < kHcMult; ++col) {
          denominator += comb_shared[row * kHcMult + col];
        }
        for (int64_t col = 0; col < kHcMult; ++col) {
          comb_shared[row * kHcMult + col] /= denominator;
        }
      }
      for (int64_t col = 0; col < kHcMult; ++col) {
        float denominator = hc_eps;
        for (int64_t row = 0; row < kHcMult; ++row) {
          denominator += comb_shared[row * kHcMult + col];
        }
        for (int64_t row = 0; row < kHcMult; ++row) {
          comb_shared[row * kHcMult + col] /= denominator;
        }
      }
    }
  }
  __syncthreads();

  for (int64_t hidden = tid; hidden < dim; hidden += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int64_t row = 0; row < kHcMult; ++row) {
      value +=
          musa_convert_to_float(x[x_offset + row * dim + hidden]) * pre[row];
    }
    output[token * dim + hidden] = static_cast<scalar_t>(value);
  }
  if (tid < kHcMult) {
    post[token * kHcMult + tid] = post_shared[tid];
  }
  if (tid < kHcMult * kHcMult) {
    comb[token * kHcMult * kHcMult + tid] = comb_shared[tid];
  }
}

// Decode only: split the hidden combine across independent 512-wide tiles.
// The last arriving tile performs the in-place RMSNorm, preserving the BF16
// rounding boundary while avoiding a separate normalization kernel.
template <typename scalar_t>
__global__ void dsv4_hc_pre_finish_split_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gemm_partials,
    const float* __restrict__ sqr_sum_partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ output_sqr_partials,
    int32_t* __restrict__ finish_counter,
    const scalar_t* __restrict__ norm_weight,
    scalar_t* __restrict__ output,
    float* __restrict__ post,
    float* __restrict__ comb,
    int64_t dim,
    int64_t sinkhorn_iters,
    float norm_eps,
    float hc_eps) {
  __shared__ float
      shared[kMixHc + kHcMult + kHcMult + kHcMult * kHcMult + kThreads + 1];
  __shared__ int32_t is_last_tile;
  __shared__ float output_inv_rms;
  const int32_t tid = threadIdx.x;
  const int64_t token = blockIdx.x;
  const int32_t hidden_tile = blockIdx.y;
  const int32_t num_hidden_tiles = gridDim.y;
  const int64_t x_size = kHcMult * dim;
  const int64_t x_offset = token * x_size;
  float* mixes = shared;
  float* pre = mixes + kMixHc;
  float* post_shared = pre + kHcMult;
  float* comb_shared = post_shared + kHcMult;
  float* reduction = comb_shared + kHcMult * kHcMult;
  float* input_inv_rms = reduction + kThreads;

  const int32_t required_mixes = hidden_tile == 0 ? kMixHc : kHcMult;
  if (tid < required_mixes) {
    float value = 0.0f;
#pragma unroll
    for (int64_t split = 0; split < kMateHcPrenormSplits; ++split) {
      value += gemm_partials[(split * gridDim.x + token) * kMixHc + tid];
    }
    mixes[tid] = value;
  }
  if (tid == 0) {
    float sqr_sum = 0.0f;
#pragma unroll
    for (int64_t split = 0; split < kMateHcPrenormSplits; ++split) {
      sqr_sum += sqr_sum_partials[split * gridDim.x + token];
    }
    input_inv_rms[0] = rsqrtf(sqr_sum / static_cast<float>(x_size) + norm_eps);
  }
  __syncthreads();

  if (tid < required_mixes) {
    mixes[tid] *= input_inv_rms[0];
  }
  __syncthreads();

  if (tid < kHcMult) {
    pre[tid] = stable_sigmoid(mixes[tid] * hc_scale[0] + hc_base[tid]) + hc_eps;
    if (hidden_tile == 0) {
      post_shared[tid] =
          2.0f * stable_sigmoid(mixes[kHcMult + tid] * hc_scale[1] +
                                hc_base[kHcMult + tid]);
    }
  }
  __syncthreads();

  if (hidden_tile == 0 && tid < 32) {
    constexpr unsigned int kFullWarpMask = 0xffffffffu;
    const int32_t matrix_lane = tid & 15;
    constexpr int64_t kCombOffset = 2 * kHcMult;
    float comb_value = mixes[kCombOffset + matrix_lane] * hc_scale[2] +
                       hc_base[kCombOffset + matrix_lane];
    float row_max = fmaxf(
        comb_value, __shfl_xor_sync(kFullWarpMask, comb_value, 1, kHcMult));
    row_max =
        fmaxf(row_max, __shfl_xor_sync(kFullWarpMask, row_max, 2, kHcMult));
    comb_value = expf(comb_value - row_max);
    float row_sum =
        comb_value + __shfl_xor_sync(kFullWarpMask, comb_value, 1, kHcMult);
    row_sum += __shfl_xor_sync(kFullWarpMask, row_sum, 2, kHcMult);
    comb_value = comb_value / row_sum + hc_eps;

    float col_sum =
        comb_value + __shfl_xor_sync(kFullWarpMask, comb_value, 4, 16);
    col_sum += __shfl_xor_sync(kFullWarpMask, col_sum, 8, 16);
    comb_value /= col_sum + hc_eps;
    for (int64_t iter = 1; iter < sinkhorn_iters; ++iter) {
      row_sum =
          comb_value + __shfl_xor_sync(kFullWarpMask, comb_value, 1, kHcMult);
      row_sum += __shfl_xor_sync(kFullWarpMask, row_sum, 2, kHcMult);
      comb_value /= row_sum + hc_eps;
      col_sum = comb_value + __shfl_xor_sync(kFullWarpMask, comb_value, 4, 16);
      col_sum += __shfl_xor_sync(kFullWarpMask, col_sum, 8, 16);
      comb_value /= col_sum + hc_eps;
    }
    if (tid < kHcMult * kHcMult) {
      comb_shared[tid] = comb_value;
    }
  }

  const int64_t hidden_begin =
      static_cast<int64_t>(hidden_tile) * kHcFinishHiddenTile;
  const int64_t hidden_limit = hidden_begin + kHcFinishHiddenTile;
  const int64_t hidden_end = hidden_limit < dim ? hidden_limit : dim;
  float output_norm_sum = 0.0f;
  for (int64_t hidden = hidden_begin + tid; hidden < hidden_end;
       hidden += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int64_t row = 0; row < kHcMult; ++row) {
      value +=
          musa_convert_to_float(x[x_offset + row * dim + hidden]) * pre[row];
    }
    const scalar_t quantized = static_cast<scalar_t>(value);
    output[token * dim + hidden] = quantized;
    const float quantized_float = musa_convert_to_float(quantized);
    output_norm_sum += quantized_float * quantized_float;
  }
  reduction[tid] = output_norm_sum;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output_sqr_partials[token * num_hidden_tiles + hidden_tile] = reduction[0];
    if (hidden_tile == 0) {
      for (int64_t row = 0; row < kHcMult; ++row) {
        post[token * kHcMult + row] = post_shared[row];
      }
      for (int64_t index = 0; index < kHcMult * kHcMult; ++index) {
        comb[token * kHcMult * kHcMult + index] = comb_shared[index];
      }
    }
    __threadfence();
    const unsigned int ticket =
        atomicInc(reinterpret_cast<unsigned int*>(finish_counter + token),
                  static_cast<unsigned int>(num_hidden_tiles - 1));
    is_last_tile = ticket == static_cast<unsigned int>(num_hidden_tiles - 1);
  }
  __syncthreads();
  if (!is_last_tile) {
    return;
  }
  if (tid == 0) {
    float sum = 0.0f;
    for (int32_t tile = 0; tile < num_hidden_tiles; ++tile) {
      sum += output_sqr_partials[token * num_hidden_tiles + tile];
    }
    output_inv_rms = rsqrtf(sum / static_cast<float>(dim) + norm_eps);
  }
  __syncthreads();
  for (int64_t hidden = tid; hidden < dim; hidden += blockDim.x) {
    const int64_t offset = token * dim + hidden;
    const scalar_t normalized = static_cast<scalar_t>(
        musa_convert_to_float(output[offset]) * output_inv_rms);
    output[offset] =
        static_cast<scalar_t>(musa_convert_to_float(normalized) *
                              musa_convert_to_float(norm_weight[hidden]));
  }
}

template <typename scalar_t>
__global__ void dsv4_hc_pre_decode_kernel(const scalar_t* __restrict__ x,
                                          const float* __restrict__ hc_fn,
                                          const float* __restrict__ hc_scale,
                                          const float* __restrict__ hc_base,
                                          scalar_t* __restrict__ output,
                                          float* __restrict__ post,
                                          float* __restrict__ comb,
                                          int64_t dim,
                                          int64_t sinkhorn_iters,
                                          float norm_eps,
                                          float hc_eps) {
  extern __shared__ float shared[];
  const int32_t tid = threadIdx.x;
  const int64_t token = blockIdx.x;
  const int64_t x_size = kHcMult * dim;
  const int64_t x_offset = token * x_size;
  float* reduction = shared;
  float* mixes = reduction + (kMixHc + 1) * blockDim.x;
  float* pre = mixes + kMixHc;
  float* post_shared = pre + kHcMult;
  float* comb_shared = post_shared + kHcMult;

  float norm_sum = 0.0f;
  float dot[kMixHc] = {};
  for (int64_t index = tid; index < x_size; index += blockDim.x) {
    const float value = musa_convert_to_float(x[x_offset + index]);
    norm_sum += value * value;
#pragma unroll
    for (int64_t row = 0; row < kMixHc; ++row) {
      dot[row] += value * hc_fn[row * x_size + index];
    }
  }
#pragma unroll
  for (int64_t row = 0; row < kMixHc; ++row) {
    reduction[row * blockDim.x + tid] = dot[row];
  }
  reduction[kMixHc * blockDim.x + tid] = norm_sum;
  __syncthreads();

  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int64_t row = 0; row <= kMixHc; ++row) {
        reduction[row * blockDim.x + tid] +=
            reduction[row * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float inv_rms = rsqrtf(
        reduction[kMixHc * blockDim.x] / static_cast<float>(x_size) + norm_eps);
#pragma unroll
    for (int64_t row = 0; row < kMixHc; ++row) {
      mixes[row] = reduction[row * blockDim.x] * inv_rms;
    }
#pragma unroll
    for (int64_t row = 0; row < kHcMult; ++row) {
      pre[row] =
          stable_sigmoid(mixes[row] * hc_scale[0] + hc_base[row]) + hc_eps;
      post_shared[row] =
          2.0f * stable_sigmoid(mixes[kHcMult + row] * hc_scale[1] +
                                hc_base[kHcMult + row]);
    }

    constexpr int64_t kCombOffset = 2 * kHcMult;
    for (int64_t row = 0; row < kHcMult; ++row) {
      float max_value = -INFINITY;
      for (int64_t col = 0; col < kHcMult; ++col) {
        max_value =
            fmaxf(max_value,
                  mixes[kCombOffset + row * kHcMult + col] * hc_scale[2] +
                      hc_base[kCombOffset + row * kHcMult + col]);
      }
      float denominator = 0.0f;
      for (int64_t col = 0; col < kHcMult; ++col) {
        const float logit =
            mixes[kCombOffset + row * kHcMult + col] * hc_scale[2] +
            hc_base[kCombOffset + row * kHcMult + col];
        const float value = expf(logit - max_value);
        comb_shared[row * kHcMult + col] = value;
        denominator += value;
      }
      for (int64_t col = 0; col < kHcMult; ++col) {
        comb_shared[row * kHcMult + col] =
            comb_shared[row * kHcMult + col] / denominator + hc_eps;
      }
    }

    for (int64_t col = 0; col < kHcMult; ++col) {
      float denominator = hc_eps;
      for (int64_t row = 0; row < kHcMult; ++row) {
        denominator += comb_shared[row * kHcMult + col];
      }
      for (int64_t row = 0; row < kHcMult; ++row) {
        comb_shared[row * kHcMult + col] /= denominator;
      }
    }
    for (int64_t iter = 1; iter < sinkhorn_iters; ++iter) {
      for (int64_t row = 0; row < kHcMult; ++row) {
        float denominator = hc_eps;
        for (int64_t col = 0; col < kHcMult; ++col) {
          denominator += comb_shared[row * kHcMult + col];
        }
        for (int64_t col = 0; col < kHcMult; ++col) {
          comb_shared[row * kHcMult + col] /= denominator;
        }
      }
      for (int64_t col = 0; col < kHcMult; ++col) {
        float denominator = hc_eps;
        for (int64_t row = 0; row < kHcMult; ++row) {
          denominator += comb_shared[row * kHcMult + col];
        }
        for (int64_t row = 0; row < kHcMult; ++row) {
          comb_shared[row * kHcMult + col] /= denominator;
        }
      }
    }
  }
  __syncthreads();

  for (int64_t hidden = tid; hidden < dim; hidden += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int64_t row = 0; row < kHcMult; ++row) {
      value +=
          musa_convert_to_float(x[x_offset + row * dim + hidden]) * pre[row];
    }
    output[token * dim + hidden] = static_cast<scalar_t>(value);
  }
  if (tid < kHcMult) {
    post[token * kHcMult + tid] = post_shared[tid];
  }
  if (tid < kHcMult * kHcMult) {
    comb[token * kHcMult * kHcMult + tid] = comb_shared[tid];
  }
}

void check_last_error() {
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 HC pre kernel failed: " << musaGetErrorString(status);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v4_hc_pre_decode(const torch::Tensor& x,
                          const torch::Tensor& hc_fn,
                          const torch::Tensor& hc_scale,
                          const torch::Tensor& hc_base,
                          int64_t hc_mult,
                          int64_t sinkhorn_iters,
                          double norm_eps,
                          double hc_eps,
                          const std::optional<torch::Tensor>& output,
                          const std::optional<torch::Tensor>& post,
                          const std::optional<torch::Tensor>& comb,
                          const std::optional<torch::Tensor>& norm_weight,
                          bool* output_is_normalized) {
  if (output_is_normalized != nullptr) {
    *output_is_normalized = false;
  }
  CHECK(x.defined() && hc_fn.defined() && hc_scale.defined() &&
        hc_base.defined());
  CHECK(x.is_contiguous() && hc_fn.is_contiguous() &&
        hc_scale.is_contiguous() && hc_base.is_contiguous());
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(hc_mult, kHcMult);
  CHECK_EQ(x.size(1), kHcMult);
  CHECK_EQ(hc_fn.scalar_type(), torch::kFloat32);
  CHECK_EQ(hc_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(hc_base.scalar_type(), torch::kFloat32);
  CHECK_EQ(hc_fn.device(), x.device());
  CHECK_EQ(hc_scale.device(), x.device());
  CHECK_EQ(hc_base.device(), x.device());
  CHECK_GE(sinkhorn_iters, 1);

  const int64_t tokens = x.size(0);
  const int64_t dim = x.size(2);
  CHECK_GT(tokens, 0);
  CHECK_EQ(hc_fn.sizes(), torch::IntArrayRef({kMixHc, kHcMult * dim}));
  CHECK_EQ(hc_scale.numel(), 3);
  CHECK_EQ(hc_base.numel(), kMixHc);
  if (norm_weight.has_value()) {
    CHECK(norm_weight->defined() && norm_weight->is_contiguous());
    CHECK_EQ(norm_weight->scalar_type(), x.scalar_type());
    CHECK_EQ(norm_weight->device(), x.device());
    CHECK_EQ(norm_weight->numel(), dim);
  }

  auto result = output.has_value() ? output.value().narrow(0, 0, tokens)
                                   : torch::empty({tokens, dim}, x.options());
  auto post_result = post.has_value()
                         ? post.value().narrow(0, 0, tokens)
                         : torch::empty({tokens, kHcMult}, hc_scale.options());
  auto comb_result = comb.has_value() ? comb.value().narrow(0, 0, tokens)
                                      : torch::empty({tokens, kHcMult, kHcMult},
                                                     hc_scale.options());
  CHECK(result.is_contiguous() && post_result.is_contiguous() &&
        comb_result.is_contiguous());
  CHECK_EQ(result.scalar_type(), x.scalar_type());
  CHECK_EQ(post_result.scalar_type(), torch::kFloat32);
  CHECK_EQ(comb_result.scalar_type(), torch::kFloat32);

  // Match worker2's decode fast path: Mate split-K prenorm plus its fused HC
  // finish writes directly into the persistent decode workspaces. If the
  // cached modules are unavailable, retain the local MUSA implementation.
  if (try_mate_hc_pre_decode(x,
                             hc_fn,
                             hc_scale,
                             hc_base,
                             result,
                             post_result,
                             comb_result,
                             sinkhorn_iters,
                             norm_eps,
                             hc_eps,
                             norm_weight,
                             output_is_normalized)) {
    return {result, post_result, comb_result};
  }

  const size_t shared_bytes =
      (static_cast<size_t>(kMixHc) + 1) * kThreads * sizeof(float) +
      static_cast<size_t>(kMixHc + 2 * kHcMult + kHcMult * kHcMult) *
          sizeof(float);
  const XLLMOptionalDeviceGuard device_guard(at::device_of(x));
  const musaStream_t stream = current_musa_stream(x.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf,
                                  torch::kBFloat16,
                                  x.scalar_type(),
                                  "deepseek_v4_hc_pre_decode",
                                  [&] {
                                    dsv4_hc_pre_decode_kernel<scalar_t>
                                        <<<static_cast<unsigned int>(tokens),
                                           kThreads,
                                           shared_bytes,
                                           stream>>>(
                                            x.data_ptr<scalar_t>(),
                                            hc_fn.data_ptr<float>(),
                                            hc_scale.data_ptr<float>(),
                                            hc_base.data_ptr<float>(),
                                            result.data_ptr<scalar_t>(),
                                            post_result.data_ptr<float>(),
                                            comb_result.data_ptr<float>(),
                                            dim,
                                            sinkhorn_iters,
                                            static_cast<float>(norm_eps),
                                            static_cast<float>(hc_eps));
                                  });
  check_last_error();
  return {result, post_result, comb_result};
}

}  // namespace xllm::kernel::musa
