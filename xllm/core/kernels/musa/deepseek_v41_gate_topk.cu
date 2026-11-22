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
#include "core/util/env_var.h"

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

// Small verify batches benefit from parallelizing the expert dimension, as in
// SGLang's MUSA fused gate. Keep v41_score, tie breaking, and the serial sum of
// selected unbiased scores identical to the existing warp-per-token path.
// One CTA owns each token; every expert has one thread (unused lanes invalid).
template <int32_t Threads>
__global__ void v41_gate_block_kernel(const float* logits,
                                      const float* bias,
                                      float* weights,
                                      int64_t* indices,
                                      int32_t experts,
                                      int32_t topk,
                                      bool per_token_bias,
                                      bool normalize,
                                      float scale) {
  constexpr int32_t kWarps = Threads / 32;
  __shared__ float warp_scores[kWarps];
  __shared__ int32_t warp_ids[kWarps];
  __shared__ int32_t selected_id;
  __shared__ float original_scores[Threads];
  const int32_t tid = threadIdx.x;
  const int32_t lane = tid % 32;
  const int32_t warp = tid / 32;
  const int64_t token = blockIdx.x;
  const float* row_logits = logits + token * experts;
  const float* row_bias = bias + (per_token_bias ? token * experts : 0);
  bool eligible = tid < experts;
  const float original = eligible ? v41_score(row_logits[tid]) : 0.0f;
  original_scores[tid] = original;
  const float score = eligible ? original + row_bias[tid] : -INFINITY;
  float total = 0.0f;
  for (int32_t selected = 0; selected < topk; ++selected) {
    float best = score;
    int32_t best_id = eligible ? tid : INT32_MAX;
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
      warp_scores[warp] = best;
      warp_ids[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0) {
      best = lane < kWarps ? warp_scores[lane] : -INFINITY;
      best_id = lane < kWarps ? warp_ids[lane] : INT32_MAX;
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
        selected_id = best_id;
        // Reuse the unbiased score, never subtract correction bias. The first
        // CTA barrier also publishes every thread's original_scores entry.
        const float original = original_scores[best_id];
        indices[token * topk + selected] = best_id;
        weights[token * topk + selected] = original;
        total += original;
      }
    }
    __syncthreads();
    eligible = eligible && tid != selected_id;
    // The next iteration's first CTA barrier ensures all selected_id reads
    // finish before warp 0 can publish a different selected expert.
  }
  if (tid == 0) {
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

// Canonicalize numeric ties before packing the score and its stable ID.
__device__ __forceinline__ uint64_t route_key(float score, int32_t expert) {
  constexpr uint32_t kSign = 0x80000000u;
  constexpr uint32_t kMaxExpert = 511u;
  uint32_t encoded;
  if (isnan(score)) {
    encoded = UINT32_MAX;
  } else {
    const uint32_t bits = score == 0.0f ? 0u : __float_as_uint(score);
    encoded = (bits & kSign) ? ~bits : (bits ^ kSign);
  }
  return (static_cast<uint64_t>(encoded) << 9) | (kMaxExpert - expert);
}

__device__ __forceinline__ uint64_t route_shuffle_xor(uint64_t value,
                                                      int32_t offset) {
  const uint32_t lo =
      __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(value), offset, 32);
  const uint32_t hi = __shfl_xor_sync(
      0xffffffffu, static_cast<uint32_t>(value >> 32), offset, 32);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

// First keep six sorted candidates per warp. One warp then merges those
// sorted lists. A single CTA barrier publishes the candidates and their
// unbiased scores.
__global__ void v41_gate_merge_six_kernel(const float* logits,
                                          const float* bias,
                                          float* weights,
                                          int64_t* indices,
                                          bool per_token_bias,
                                          bool normalize,
                                          float scale) {
  constexpr int32_t kExperts = 384;
  constexpr int32_t kWarps = kExperts / 32;
  constexpr int32_t kRoutes = 6;
  const int32_t expert = threadIdx.x;
  const int32_t lane = expert % 32;
  const int32_t warp = expert / 32;
  const int64_t token = blockIdx.x;
  const float original = v41_score(logits[token * kExperts + expert]);
  const float* row_bias = bias + (per_token_bias ? token * kExperts : 0);
  uint64_t key = route_key(original + row_bias[expert], expert);
  __shared__ uint64_t warp_keys[kRoutes][kWarps];
  __shared__ float original_scores[kExperts];
  original_scores[expert] = original;
#pragma unroll
  for (int32_t width = 2; width <= 32; width *= 2) {
#pragma unroll
    for (int32_t offset = width / 2; offset != 0; offset /= 2) {
      const uint64_t peer = route_shuffle_xor(key, offset);
      const bool keep_max = ((lane & width) == 0) == ((lane & offset) == 0);
      key = keep_max == (key > peer) ? key : peer;
    }
  }
  if (lane < kRoutes) {
    warp_keys[lane][warp] = key;
  }
  __syncthreads();
  if (warp != 0) {
    return;
  }
  int32_t used = 0;
  float total = 0.0f;
  float selected_scores[kRoutes];
#pragma unroll
  for (int32_t rank = 0; rank < kRoutes; ++rank) {
    uint64_t best = lane < kWarps ? warp_keys[used][lane] : 0;
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2) {
      const uint64_t peer = route_shuffle_xor(best, offset);
      best = best > peer ? best : peer;
    }
    const int32_t id = 511 - static_cast<int32_t>(best & 511u);
    if (lane == 0) {
      indices[token * kRoutes + rank] = id;
      selected_scores[rank] = original_scores[id];
      total += selected_scores[rank];
    }
    used += static_cast<int32_t>(id / 32 == lane);
  }
  if (lane == 0) {
#pragma unroll
    for (int32_t rank = 0; rank < kRoutes; ++rank) {
      float weight = selected_scores[rank];
      if (normalize) {
        weight /= total + 1.0e-20f;
      }
      weights[token * kRoutes + rank] = weight * scale;
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
  const bool block_gate =
      util::get_bool_env("XLLM_DSV41_PARALLEL_GATE_TOPK", false);
  const bool merge_routes =
      block_gate && logits.size(0) > 1 && logits.size(0) <= 6 &&
      logits.size(1) == 384 && topk == 6 &&
      util::get_bool_env("XLLM_DSV41_VERIFY_MERGE_GATE_ROUTES", false);
  if (merge_routes) {
    v41_gate_merge_six_kernel<<<dim3(static_cast<uint32_t>(logits.size(0))),
                                dim3(384),
                                0,
                                current_musa_stream(logits.device())>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        weights.data_ptr<float>(),
        ids.data_ptr<int64_t>(),
        bias.dim() == 2,
        normalize,
        static_cast<float>(scale));
    LOG_FIRST_N(INFO, 1) << "Native V4.1 merges six verify gate routes: rows="
                         << logits.size(0);
  } else if (block_gate && logits.size(0) <= 6 && logits.size(1) <= 512) {
    v41_gate_block_kernel<512><<<dim3(static_cast<uint32_t>(logits.size(0))),
                                 dim3(512),
                                 0,
                                 current_musa_stream(logits.device())>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        weights.data_ptr<float>(),
        ids.data_ptr<int64_t>(),
        static_cast<int32_t>(logits.size(1)),
        static_cast<int32_t>(topk),
        bias.dim() == 2,
        normalize,
        static_cast<float>(scale));
  } else if (logits.size(1) <= 128) {
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
