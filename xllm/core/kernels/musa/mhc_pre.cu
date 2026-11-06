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

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "core/kernels/musa/mhc_ops.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kThreads = 256;
constexpr int64_t kMaxHcMult = 12;
constexpr int64_t kPrenormSplits = 32;

__device__ float stable_sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

// One token per block. Shared memory holds the mix reduction, then the
// pre/post/comb coefficients. hc_mult is a runtime argument so GLM and a
// future DeepSeek-V4 migration share this kernel.
template <typename scalar_t>
__global__ void mhc_pre_kernel(const scalar_t* __restrict__ residual,
                               const float* __restrict__ hc_fn,
                               const float* __restrict__ hc_scale,
                               const float* __restrict__ hc_base,
                               scalar_t* __restrict__ layer_input,
                               float* __restrict__ post,
                               float* __restrict__ comb,
                               int64_t hc_mult,
                               int64_t hidden_size,
                               int64_t mix_size,
                               int64_t sinkhorn_repeat,
                               float rms_eps,
                               float hc_pre_eps,
                               float hc_sinkhorn_eps,
                               float hc_post_mult_value) {
  extern __shared__ float shared[];
  const int32_t thread_id = threadIdx.x;
  const int64_t token = blockIdx.x;
  const int64_t flat_size = hc_mult * hidden_size;
  const int64_t input_offset = token * flat_size;
  float* partials = shared;
  float* mixes = partials + (mix_size + 1) * blockDim.x;
  float* pre = mixes + mix_size;
  float* post_shared = pre + hc_mult;
  float* comb_shared = post_shared + hc_mult;

  for (int64_t slot = thread_id; slot < (mix_size + 1) * blockDim.x;
       slot += blockDim.x) {
    partials[slot] = 0.0f;
  }
  __syncthreads();
  for (int64_t index = thread_id; index < flat_size; index += blockDim.x) {
    const float value = musa_convert_to_float(residual[input_offset + index]);
    partials[mix_size * blockDim.x + thread_id] += value * value;
    for (int64_t row = 0; row < mix_size; ++row) {
      partials[row * blockDim.x + thread_id] +=
          value * hc_fn[row * flat_size + index];
    }
  }
  __syncthreads();

  for (int64_t row = 0; row <= mix_size; ++row) {
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride) {
        partials[row * blockDim.x + thread_id] +=
            partials[row * blockDim.x + thread_id + stride];
      }
      __syncthreads();
    }
  }

  if (thread_id == 0) {
    const float inverse_rms =
        rsqrtf(partials[mix_size * blockDim.x] / static_cast<float>(flat_size) +
               rms_eps);
    for (int64_t row = 0; row < mix_size; ++row) {
      mixes[row] = partials[row * blockDim.x] * inverse_rms;
    }
    for (int64_t row = 0; row < hc_mult; ++row) {
      pre[row] =
          stable_sigmoid(mixes[row] * hc_scale[0] + hc_base[row]) + hc_pre_eps;
      post_shared[row] = hc_post_mult_value *
                         stable_sigmoid(mixes[hc_mult + row] * hc_scale[1] +
                                        hc_base[hc_mult + row]);
    }
    const int64_t comb_offset = 2 * hc_mult;
    for (int64_t row = 0; row < hc_mult; ++row) {
      float maximum = -INFINITY;
      for (int64_t column = 0; column < hc_mult; ++column) {
        const int64_t mix_index = comb_offset + row * hc_mult + column;
        maximum =
            fmaxf(maximum, mixes[mix_index] * hc_scale[2] + hc_base[mix_index]);
      }
      float row_sum = 0.0f;
      for (int64_t column = 0; column < hc_mult; ++column) {
        const int64_t mix_index = comb_offset + row * hc_mult + column;
        const float value =
            expf(mixes[mix_index] * hc_scale[2] + hc_base[mix_index] - maximum);
        comb_shared[row * hc_mult + column] = value;
        row_sum += value;
      }
      for (int64_t column = 0; column < hc_mult; ++column) {
        comb_shared[row * hc_mult + column] =
            comb_shared[row * hc_mult + column] / row_sum + hc_sinkhorn_eps;
      }
    }
    for (int64_t iteration = 0; iteration < sinkhorn_repeat; ++iteration) {
      if (iteration > 0) {
        for (int64_t row = 0; row < hc_mult; ++row) {
          float row_sum = hc_sinkhorn_eps;
          for (int64_t column = 0; column < hc_mult; ++column) {
            row_sum += comb_shared[row * hc_mult + column];
          }
          for (int64_t column = 0; column < hc_mult; ++column) {
            comb_shared[row * hc_mult + column] /= row_sum;
          }
        }
      }
      for (int64_t column = 0; column < hc_mult; ++column) {
        float column_sum = hc_sinkhorn_eps;
        for (int64_t row = 0; row < hc_mult; ++row) {
          column_sum += comb_shared[row * hc_mult + column];
        }
        for (int64_t row = 0; row < hc_mult; ++row) {
          comb_shared[row * hc_mult + column] /= column_sum;
        }
      }
    }
  }
  __syncthreads();

  for (int64_t hidden = thread_id; hidden < hidden_size; hidden += blockDim.x) {
    float value = 0.0f;
    for (int64_t row = 0; row < hc_mult; ++row) {
      value += musa_convert_to_float(
                   residual[input_offset + row * hidden_size + hidden]) *
               pre[row];
    }
    layer_input[token * hidden_size + hidden] = static_cast<scalar_t>(value);
  }
  if (thread_id < hc_mult) {
    post[token * hc_mult + thread_id] = post_shared[thread_id];
  }
  if (thread_id < hc_mult * hc_mult) {
    comb[token * hc_mult * hc_mult + thread_id] = comb_shared[thread_id];
  }
}

// Applies RMS, sigmoid, and Sinkhorn to MATE prenorm partials. Partial mixes
// are [splits, tokens, mix] and partial squares are [splits, tokens].
template <typename scalar_t>
__global__ void mhc_pre_finish_kernel(const scalar_t* __restrict__ residual,
                                      const float* __restrict__ partial_mixes,
                                      const float* __restrict__ partial_squares,
                                      const float* __restrict__ hc_scale,
                                      const float* __restrict__ hc_base,
                                      scalar_t* __restrict__ layer_input,
                                      float* __restrict__ post,
                                      float* __restrict__ comb,
                                      int64_t splits,
                                      int64_t hc_mult,
                                      int64_t hidden_size,
                                      int64_t mix_size,
                                      int64_t sinkhorn_repeat,
                                      float rms_eps,
                                      float hc_pre_eps,
                                      float hc_sinkhorn_eps,
                                      float hc_post_mult_value) {
  extern __shared__ float shared[];
  const int32_t thread_id = threadIdx.x;
  const int64_t token = blockIdx.x;
  float* mixes = shared;
  float* pre = mixes + mix_size;
  float* comb_shared = pre + hc_mult;

  if (thread_id == 0) {
    float square_sum = 0.0f;
    for (int64_t split = 0; split < splits; ++split) {
      square_sum += partial_squares[split * gridDim.x + token];
    }
    const float inverse_rms = rsqrtf(
        square_sum / static_cast<float>(hc_mult * hidden_size) + rms_eps);
    for (int64_t row = 0; row < mix_size; ++row) {
      float mix = 0.0f;
      for (int64_t split = 0; split < splits; ++split) {
        mix += partial_mixes[(split * gridDim.x + token) * mix_size + row];
      }
      mixes[row] = mix * inverse_rms;
    }
    for (int64_t row = 0; row < hc_mult; ++row) {
      pre[row] =
          stable_sigmoid(mixes[row] * hc_scale[0] + hc_base[row]) + hc_pre_eps;
      post[token * hc_mult + row] =
          hc_post_mult_value *
          stable_sigmoid(mixes[hc_mult + row] * hc_scale[1] +
                         hc_base[hc_mult + row]);
    }
    const int64_t comb_offset = 2 * hc_mult;
    for (int64_t row = 0; row < hc_mult; ++row) {
      float maximum = -INFINITY;
      for (int64_t column = 0; column < hc_mult; ++column) {
        maximum =
            fmaxf(maximum,
                  mixes[comb_offset + row * hc_mult + column] * hc_scale[2] +
                      hc_base[comb_offset + row * hc_mult + column]);
      }
      float row_sum = 0.0f;
      for (int64_t column = 0; column < hc_mult; ++column) {
        const float logit =
            mixes[comb_offset + row * hc_mult + column] * hc_scale[2] +
            hc_base[comb_offset + row * hc_mult + column];
        const float value = expf(logit - maximum);
        comb_shared[row * hc_mult + column] = value;
        row_sum += value;
      }
      for (int64_t column = 0; column < hc_mult; ++column) {
        comb_shared[row * hc_mult + column] =
            comb_shared[row * hc_mult + column] / row_sum + hc_sinkhorn_eps;
      }
    }
    for (int64_t column = 0; column < hc_mult; ++column) {
      float column_sum = hc_sinkhorn_eps;
      for (int64_t row = 0; row < hc_mult; ++row) {
        column_sum += comb_shared[row * hc_mult + column];
      }
      for (int64_t row = 0; row < hc_mult; ++row) {
        comb_shared[row * hc_mult + column] /= column_sum;
      }
    }
    for (int64_t iteration = 1; iteration < sinkhorn_repeat; ++iteration) {
      for (int64_t row = 0; row < hc_mult; ++row) {
        float row_sum = hc_sinkhorn_eps;
        for (int64_t column = 0; column < hc_mult; ++column) {
          row_sum += comb_shared[row * hc_mult + column];
        }
        for (int64_t column = 0; column < hc_mult; ++column) {
          comb_shared[row * hc_mult + column] /= row_sum;
        }
      }
      for (int64_t column = 0; column < hc_mult; ++column) {
        float column_sum = hc_sinkhorn_eps;
        for (int64_t row = 0; row < hc_mult; ++row) {
          column_sum += comb_shared[row * hc_mult + column];
        }
        for (int64_t row = 0; row < hc_mult; ++row) {
          comb_shared[row * hc_mult + column] /= column_sum;
        }
      }
    }
    for (int64_t index = 0; index < hc_mult * hc_mult; ++index) {
      comb[token * hc_mult * hc_mult + index] = comb_shared[index];
    }
  }
  __syncthreads();

  const int64_t input_offset = token * hc_mult * hidden_size;
  for (int64_t hidden = thread_id; hidden < hidden_size; hidden += blockDim.x) {
    float value = 0.0f;
    for (int64_t row = 0; row < hc_mult; ++row) {
      value += musa_convert_to_float(
                   residual[input_offset + row * hidden_size + hidden]) *
               pre[row];
    }
    layer_input[token * hidden_size + hidden] = static_cast<scalar_t>(value);
  }
}

bool try_mate_prenorm(const torch::Tensor& residual,
                      const torch::Tensor& hc_fn,
                      int64_t tokens,
                      int64_t hc_mult,
                      int64_t hidden_size,
                      int64_t mix_size,
                      torch::Tensor* partial_mixes,
                      torch::Tensor* partial_squares) {
  const std::string prenorm_uri =
      tokens <= 64 ? "mate_jit_tf32_hc_prenorm_32x32x32_3stages"
                   : "mate_jit_tf32_hc_prenorm_64x32x32_3stages";
  if (!std::ifstream(path_to_uri_so_lib(prenorm_uri)).good()) {
    return false;
  }
  *partial_mixes = torch::empty({kPrenormSplits, tokens, mix_size},
                                residual.options().dtype(torch::kFloat32));
  *partial_squares = torch::empty({kPrenormSplits, tokens},
                                  residual.options().dtype(torch::kFloat32));
  {
    TvmffiStreamGuard stream_guard(residual.device());
    ffi::Function prenorm = get_function(prenorm_uri, "tf32_hc_prenorm_gemm");
    prenorm(to_ffi_tensor(residual.reshape({tokens, hc_mult * hidden_size})),
            to_ffi_tensor(hc_fn),
            to_ffi_tensor(*partial_mixes),
            to_ffi_tensor(*partial_squares),
            static_cast<int64_t>(kPrenormSplits));
  }
  return partial_mixes->defined() && partial_squares->defined();
}

}  // namespace

MHCPreOutput mhc_pre(const torch::Tensor& residual,
                     const torch::Tensor& hc_fn,
                     const torch::Tensor& hc_scale,
                     const torch::Tensor& hc_base,
                     double rms_eps,
                     double hc_pre_eps,
                     double hc_sinkhorn_eps,
                     double hc_post_mult_value,
                     int64_t sinkhorn_repeat) {
  CHECK(residual.defined() && residual.dim() >= 3);
  CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
  CHECK_EQ(residual.device().type(), torch::kPrivateUse1);
  CHECK(residual.is_contiguous() && hc_fn.is_contiguous());
  CHECK(hc_scale.is_contiguous() && hc_base.is_contiguous());
  CHECK_EQ(hc_fn.scalar_type(), torch::kFloat32);
  CHECK_EQ(hc_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(hc_base.scalar_type(), torch::kFloat32);
  CHECK(residual.device() == hc_fn.device());
  CHECK(residual.device() == hc_scale.device());
  CHECK(residual.device() == hc_base.device());
  const int64_t hc_mult = residual.size(-2);
  const int64_t hidden_size = residual.size(-1);
  const int64_t mix_size = hc_mult * (hc_mult + 2);
  CHECK_GT(hc_mult, 0);
  CHECK_LE(hc_mult, kMaxHcMult);
  CHECK_GT(hidden_size, 0);
  CHECK_EQ(hidden_size % 8, 0);
  CHECK_EQ(hc_fn.sizes(),
           torch::IntArrayRef({mix_size, hc_mult * hidden_size}));
  CHECK_EQ(hc_scale.numel(), 3);
  CHECK_EQ(hc_base.numel(), mix_size);
  CHECK_GE(sinkhorn_repeat, 1);
  CHECK(std::isfinite(rms_eps) && rms_eps > 0);
  CHECK(std::isfinite(hc_pre_eps) && hc_pre_eps >= 0);
  CHECK(std::isfinite(hc_sinkhorn_eps) && hc_sinkhorn_eps > 0);
  CHECK(std::isfinite(hc_post_mult_value));

  const std::vector<int64_t> outer_shape(residual.sizes().begin(),
                                         residual.sizes().end() - 2);
  torch::Tensor residual_flat =
      residual.reshape({-1, hc_mult, hidden_size}).contiguous();
  const int64_t tokens = residual_flat.size(0);
  torch::Tensor layer_input =
      torch::empty({tokens, hidden_size}, residual.options());
  torch::Tensor post = torch::empty({tokens, hc_mult},
                                    residual.options().dtype(torch::kFloat32));
  torch::Tensor comb = torch::empty({tokens, hc_mult, hc_mult},
                                    residual.options().dtype(torch::kFloat32));
  if (tokens > 0) {
    torch::Tensor partial_mixes;
    torch::Tensor partial_squares;
    const bool used_mate = try_mate_prenorm(residual_flat,
                                            hc_fn,
                                            tokens,
                                            hc_mult,
                                            hidden_size,
                                            mix_size,
                                            &partial_mixes,
                                            &partial_squares);
    const musaStream_t stream = current_musa_stream(residual.device());
    if (used_mate) {
      const size_t shared_bytes =
          static_cast<size_t>(mix_size + hc_mult + hc_mult * hc_mult) *
          sizeof(float);
      mhc_pre_finish_kernel<c10::BFloat16>
          <<<static_cast<uint32_t>(tokens), kThreads, shared_bytes, stream>>>(
              residual_flat.data_ptr<c10::BFloat16>(),
              partial_mixes.data_ptr<float>(),
              partial_squares.data_ptr<float>(),
              hc_scale.data_ptr<float>(),
              hc_base.data_ptr<float>(),
              layer_input.data_ptr<c10::BFloat16>(),
              post.data_ptr<float>(),
              comb.data_ptr<float>(),
              kPrenormSplits,
              hc_mult,
              hidden_size,
              mix_size,
              sinkhorn_repeat,
              static_cast<float>(rms_eps),
              static_cast<float>(hc_pre_eps),
              static_cast<float>(hc_sinkhorn_eps),
              static_cast<float>(hc_post_mult_value));
      check_musa_kernel_launch("mhc_pre_finish");
    } else {
      const size_t shared_bytes =
          (static_cast<size_t>(mix_size) + 1) * kThreads * sizeof(float) +
          static_cast<size_t>(mix_size + 2 * hc_mult + hc_mult * hc_mult) *
              sizeof(float);
      mhc_pre_kernel<c10::BFloat16>
          <<<static_cast<uint32_t>(tokens), kThreads, shared_bytes, stream>>>(
              residual_flat.data_ptr<c10::BFloat16>(),
              hc_fn.data_ptr<float>(),
              hc_scale.data_ptr<float>(),
              hc_base.data_ptr<float>(),
              layer_input.data_ptr<c10::BFloat16>(),
              post.data_ptr<float>(),
              comb.data_ptr<float>(),
              hc_mult,
              hidden_size,
              mix_size,
              sinkhorn_repeat,
              static_cast<float>(rms_eps),
              static_cast<float>(hc_pre_eps),
              static_cast<float>(hc_sinkhorn_eps),
              static_cast<float>(hc_post_mult_value));
      check_musa_kernel_launch("mhc_pre");
    }
  }

  std::vector<int64_t> post_shape = outer_shape;
  post_shape.reserve(outer_shape.size() + 2);
  post_shape.emplace_back(hc_mult);
  post_shape.emplace_back(1);
  std::vector<int64_t> comb_shape = outer_shape;
  comb_shape.reserve(outer_shape.size() + 2);
  comb_shape.emplace_back(hc_mult);
  comb_shape.emplace_back(hc_mult);
  std::vector<int64_t> input_shape = outer_shape;
  input_shape.reserve(outer_shape.size() + 1);
  input_shape.emplace_back(hidden_size);
  return {layer_input.reshape(input_shape),
          post.reshape(post_shape),
          comb.reshape(comb_shape)};
}

}  // namespace xllm::kernel::musa
