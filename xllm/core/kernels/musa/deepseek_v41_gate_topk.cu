/* Copyright 2026 The xLLM Authors.

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
#include <torch/utils.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

__device__ __forceinline__ float v41_score(float x) {
  // Evaluate sqrt(softplus(x)) as a fused function. Below -20, the
  // omitted correction is < 1e-9 relative; exp(x/2) avoids rounding the
  // intermediate exp(x) to a subnormal before taking its square root.
  if (x < -20.0f) {
    return expf(x * 0.5f);
  }
  return sqrtf(x > 20.0f ? x : log1pf(expf(x)));
}

__device__ __forceinline__ bool prefer_candidate(float score,
                                                 int32_t id,
                                                 float best,
                                                 int32_t best_id) {
  if (id == INT32_MAX) {
    return false;
  }
  if (best_id == INT32_MAX) {
    return true;
  }
  if (isnan(score) != isnan(best)) {
    return isnan(score);
  }
  return score > best ||
         ((score == best || (isnan(score) && isnan(best))) && id < best_id);
}

// Reuse the donor V4 warp-selection organization, generalized to 384
// experts and per-token modality bias. Recover weights from ORIGINAL logits
// rather than subtracting correction bias (which loses small scores).
template <int32_t Slots>
__global__ void v41_gate_kernel(const float* logits,
                                const float* bias,
                                float* weights,
                                int64_t* indices,
                                int64_t tokens,
                                int32_t experts,
                                int32_t topk,
                                bool per_token_bias,
                                bool normalize,
                                float scale) {
  const int32_t lane = threadIdx.x;
  const int64_t token = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.y;
  if (token >= tokens) {
    return;
  }
  const float* row_bias = bias + (per_token_bias ? token * experts : 0);
  float scores[Slots];
  bool eligible[Slots];
#pragma unroll
  for (int32_t i = 0; i < Slots; ++i) {
    const int32_t id = i * 32 + lane;
    eligible[i] = id < experts;
    scores[i] = id < experts
                    ? v41_score(logits[token * experts + id]) + row_bias[id]
                    : -INFINITY;
  }
  float total = 0.0f;
  for (int32_t selected = 0; selected < topk; ++selected) {
    int32_t best_id = INT32_MAX;
    float best = -INFINITY;
#pragma unroll
    for (int32_t i = 0; i < Slots; ++i) {
      const int32_t id = eligible[i] ? i * 32 + lane : INT32_MAX;
      if (prefer_candidate(scores[i], id, best, best_id)) {
        best = scores[i];
        best_id = id;
      }
    }
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2) {
      const float other = __shfl_xor_sync(0xffffffffu, best, offset, 32);
      const int32_t other_id =
          __shfl_xor_sync(0xffffffffu, best_id, offset, 32);
      if (prefer_candidate(other, other_id, best, best_id)) {
        best = other;
        best_id = other_id;
      }
    }
    if (lane == 0) {
      const float original = v41_score(logits[token * experts + best_id]);
      indices[token * topk + selected] = best_id;
      weights[token * topk + selected] = original;
      total += original;
    }
    if (lane == best_id % 32) {
      eligible[best_id / 32] = false;
    }
  }
  if (lane == 0) {
    for (int32_t selected = 0; selected < topk; ++selected) {
      const int64_t at = token * topk + selected;
      float weight = weights[at];
      if (normalize && topk > 1) {
        weight /= total + 1.0e-20f;
      }
      weights[at] = weight * scale;
    }
  }
}

template <int32_t Slots>
void launch_gate(const torch::Tensor& logits,
                 const torch::Tensor& bias,
                 torch::Tensor& weights,
                 torch::Tensor& indices,
                 int64_t topk,
                 bool normalize,
                 double scale) {
  const dim3 grid(static_cast<uint32_t>((logits.size(0) + 3) / 4));
  const dim3 block(32, 4);
  v41_gate_kernel<Slots>
      <<<grid, block, 0, current_musa_stream(logits.device())>>>(
          logits.data_ptr<float>(),
          bias.data_ptr<float>(),
          weights.data_ptr<float>(),
          indices.data_ptr<int64_t>(),
          logits.size(0),
          static_cast<int32_t>(logits.size(1)),
          static_cast<int32_t>(topk),
          bias.dim() == 2,
          normalize,
          static_cast<float>(scale));
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_gate_topk(
    const torch::Tensor& logits,
    const torch::Tensor& bias,
    int64_t topk,
    bool normalize,
    double scale) {
  CHECK(logits.defined() && bias.defined());
  CHECK(logits.device().type() == torch::kPrivateUse1);
  CHECK_EQ(logits.device(), bias.device());
  CHECK_EQ(logits.scalar_type(), torch::kFloat32);
  CHECK_EQ(bias.scalar_type(), torch::kFloat32);
  CHECK_EQ(logits.dim(), 2);
  CHECK(bias.dim() == 1 || bias.dim() == 2);
  CHECK_EQ(bias.size(-1), logits.size(1));
  if (bias.dim() == 2) {
    CHECK_EQ(bias.size(0), logits.size(0));
  }
  CHECK(logits.is_contiguous() && bias.is_contiguous());
  CHECK_GT(logits.size(1), 0);
  CHECK_LE(logits.size(1), 1024);
  CHECK_GT(topk, 0);
  CHECK_LE(topk, logits.size(1));
  CHECK(std::isfinite(scale));
  torch::DeviceGuard guard(logits.device());
  torch::Tensor weights =
      torch::empty({logits.size(0), topk}, logits.options());
  torch::Tensor ids = torch::empty({logits.size(0), topk},
                                   logits.options().dtype(torch::kInt64));
  if (logits.size(0) == 0) {
    return {weights, ids};
  }
  if (logits.size(1) <= 128) {
    launch_gate<4>(logits, bias, weights, ids, topk, normalize, scale);
  } else if (logits.size(1) <= 256) {
    launch_gate<8>(logits, bias, weights, ids, topk, normalize, scale);
  } else if (logits.size(1) <= 384) {
    launch_gate<12>(logits, bias, weights, ids, topk, normalize, scale);
  } else if (logits.size(1) <= 512) {
    launch_gate<16>(logits, bias, weights, ids, topk, normalize, scale);
  } else {
    launch_gate<32>(logits, bias, weights, ids, topk, normalize, scale);
  }
  check_musa_kernel_launch("deepseek_v41_gate_topk");
  return {weights, ids};
}
}  // namespace xllm::kernel::musa
