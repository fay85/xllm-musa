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

#include <cmath>
#include <cstdint>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kGateTopkThreads = 256;
constexpr int32_t kGateTopkWarps = 4;
constexpr int32_t kGateTopkWarpSize = 32;
constexpr int32_t kDeepseekV4Experts = 256;
constexpr int32_t kScoresPerLane = 8;
constexpr int32_t kMaxTopk = 32;

__device__ __forceinline__ float stable_softplus(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return expf(value);
  }
  return log1pf(expf(value));
}

__global__ void __launch_bounds__(kGateTopkThreads)
    deepseek_v4_gate_topk_kernel(const float* __restrict__ router_logits,
                                 const float* __restrict__ correction_bias,
                                 float* __restrict__ topk_weights,
                                 int32_t* __restrict__ topk_ids,
                                 int64_t num_experts,
                                 int64_t topk,
                                 float route_scale) {
  const int64_t token = blockIdx.x;
  const int32_t expert = threadIdx.x;
  extern __shared__ float workspace[];
  float* original_scores = workspace;
  float* biased_scores = workspace + num_experts;

  if (expert < num_experts) {
    const float score = sqrtf(fmaxf(
        stable_softplus(router_logits[token * num_experts + expert]), 0.0f));
    original_scores[expert] = score;
    biased_scores[expert] = score + correction_bias[expert];
  }
  __syncthreads();

  if (expert == 0) {
    float selected_sum = 0.0f;
    for (int64_t selected = 0; selected < topk; ++selected) {
      int32_t selected_expert = 0;
      float selected_score = biased_scores[0];
      for (int32_t candidate = 1; candidate < num_experts; ++candidate) {
        const float candidate_score = biased_scores[candidate];
        if (candidate_score > selected_score) {
          selected_score = candidate_score;
          selected_expert = candidate;
        }
      }
      const int64_t output_offset = token * topk + selected;
      const float original_score = original_scores[selected_expert];
      topk_ids[output_offset] = selected_expert;
      topk_weights[output_offset] = original_score;
      selected_sum += original_score;
      biased_scores[selected_expert] = -INFINITY;
    }
    const float inverse_sum = 1.0f / fmaxf(selected_sum, 1.0e-20f);
    for (int64_t selected = 0; selected < topk; ++selected) {
      topk_weights[token * topk + selected] *= inverse_sum * route_scale;
    }
  }
}

// DeepSeek V4 has exactly 256 routed experts. This is the donor's register-
// compact layout: a warp owns one token, each lane owns two float4 loads, and
// the original score is recovered only for the selected experts. Keeping a
// single eight-value register array avoids the spill-prone duplicate score
// arrays used by the earlier candidate.
__global__ void __launch_bounds__(kGateTopkWarps * kGateTopkWarpSize)
    deepseek_v4_gate_topk_warp_kernel(
        const float* __restrict__ router_logits,
        const float* __restrict__ correction_bias,
        float* __restrict__ topk_weights,
        int32_t* __restrict__ topk_ids,
        int64_t num_tokens,
        int64_t topk,
        float route_scale) {
  const int32_t lane = threadIdx.x;
  const int64_t token =
      static_cast<int64_t>(blockIdx.x) * kGateTopkWarps + threadIdx.y;
  if (token >= num_tokens) {
    return;
  }

  const int32_t first_expert = lane * 4;
  const float4 logits_low = *reinterpret_cast<const float4*>(
      router_logits + token * kDeepseekV4Experts + first_expert);
  const float4 logits_high = *reinterpret_cast<const float4*>(
      router_logits + token * kDeepseekV4Experts + 128 + first_expert);
  float scores[kScoresPerLane] = {logits_low.x,
                                  logits_low.y,
                                  logits_low.z,
                                  logits_low.w,
                                  logits_high.x,
                                  logits_high.y,
                                  logits_high.z,
                                  logits_high.w};
#pragma unroll
  for (int32_t i = 0; i < kScoresPerLane; ++i) {
    const int32_t expert =
        first_expert + (i >= 4 ? 128 : 0) + (i & 3);
    scores[i] = sqrtf(fmaxf(stable_softplus(scores[i]), 0.0f)) +
                correction_bias[expert];
  }

  float selected_sum = 0.0f;
  for (int32_t selected = 0; selected < topk; ++selected) {
    float selected_score = scores[0];
    int32_t selected_expert = first_expert;
#pragma unroll
    for (int32_t i = 1; i < kScoresPerLane; ++i) {
      const int32_t expert =
          first_expert + (i >= 4 ? 128 : 0) + (i & 3);
      if (scores[i] > selected_score) {
        selected_score = scores[i];
        selected_expert = expert;
      }
    }

#pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2) {
      const float other_score =
          __shfl_xor_sync(0xffffffffu, selected_score, offset, 32);
      const int32_t other_expert =
          __shfl_xor_sync(0xffffffffu, selected_expert, offset, 32);
      if (other_score > selected_score ||
          (other_score == selected_score && other_expert < selected_expert)) {
        selected_score = other_score;
        selected_expert = other_expert;
      }
    }

    if (lane == 0) {
      const float original_score =
          selected_score - correction_bias[selected_expert];
      const int64_t output_offset = token * topk + selected;
      topk_ids[output_offset] = selected_expert;
      topk_weights[output_offset] = original_score;
      selected_sum += original_score;
    }

    const int32_t owner_lane = (selected_expert & 127) / 4;
    if (lane == owner_lane) {
      const int32_t local_index =
          (selected_expert >= 128 ? 4 : 0) + (selected_expert & 3);
      scores[local_index] = -INFINITY;
    }
  }

  if (lane == 0) {
    const float inverse_sum = 1.0f / fmaxf(selected_sum, 1.0e-20f);
    for (int32_t selected = 0; selected < topk; ++selected) {
      topk_weights[token * topk + selected] *= inverse_sum * route_scale;
    }
  }
}

template <typename InputIdType, typename ExpertIdType>
__global__ void __launch_bounds__(kGateTopkThreads)
    deepseek_v4_gate_hash_topk_kernel(const float* __restrict__ router_logits,
                                      const InputIdType* __restrict__ input_ids,
                                      const ExpertIdType* __restrict__ tid2eid,
                                      float* __restrict__ topk_weights,
                                      int32_t* __restrict__ topk_ids,
                                      int64_t num_experts,
                                      int64_t topk,
                                      float route_scale) {
  const int64_t token = blockIdx.x;
  const int32_t selected = threadIdx.x;
  extern __shared__ float selected_scores[];

  if (selected < topk) {
    const int64_t input_id = static_cast<int64_t>(input_ids[token]);
    const int64_t expert =
        static_cast<int64_t>(tid2eid[input_id * topk + selected]);
    float score = 0.0f;
    if (expert >= 0 && expert < num_experts) {
      const float logit = router_logits[token * num_experts + expert];
      score = sqrtf(fmaxf(stable_softplus(logit), 0.0f));
    }
    selected_scores[selected] = score;
    topk_ids[token * topk + selected] = static_cast<int32_t>(expert);
  }
  __syncthreads();

  if (selected == 0) {
    float selected_sum = 0.0f;
    for (int64_t i = 0; i < topk; ++i) {
      selected_sum += selected_scores[i];
    }
    const float inverse_sum = 1.0f / fmaxf(selected_sum, 1.0e-20f);
    for (int64_t i = 0; i < topk; ++i) {
      topk_weights[token * topk + i] =
          selected_scores[i] * inverse_sum * route_scale;
    }
  }
}

template <typename InputIdType, typename ExpertIdType>
void launch_deepseek_v4_gate_hash_topk(const torch::Tensor& router_logits,
                                       const torch::Tensor& input_ids,
                                       const torch::Tensor& tid2eid,
                                       double route_scale,
                                       const torch::Tensor& topk_weights_output,
                                       const torch::Tensor& topk_ids_output,
                                       musaStream_t stream) {
  const int64_t topk = tid2eid.size(1);
  const dim3 grid(static_cast<unsigned int>(router_logits.size(0)));
  const dim3 block(kGateTopkThreads);
  const size_t shared_memory = static_cast<size_t>(topk) * sizeof(float);
  deepseek_v4_gate_hash_topk_kernel<InputIdType, ExpertIdType>
      <<<grid, block, shared_memory, stream>>>(
          router_logits.data_ptr<float>(),
          input_ids.data_ptr<InputIdType>(),
          tid2eid.data_ptr<ExpertIdType>(),
          topk_weights_output.data_ptr<float>(),
          topk_ids_output.data_ptr<int32_t>(),
          router_logits.size(1),
          topk,
          static_cast<float>(route_scale));
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_gate_topk(
    const torch::Tensor& router_logits,
    const torch::Tensor& correction_bias,
    int64_t topk,
    double route_scale,
    const torch::Tensor& topk_weights_output,
    const torch::Tensor& topk_ids_output) {
  CHECK(router_logits.defined() && correction_bias.defined());
  CHECK_EQ(router_logits.dim(), 2);
  CHECK_EQ(router_logits.scalar_type(), torch::kFloat32);
  CHECK(router_logits.is_contiguous());
  CHECK_EQ(correction_bias.dim(), 1);
  CHECK_EQ(correction_bias.scalar_type(), torch::kFloat32);
  CHECK(correction_bias.is_contiguous());
  CHECK_EQ(correction_bias.size(0), router_logits.size(1));
  CHECK_EQ(correction_bias.device(), router_logits.device());
  CHECK_GT(topk, 0);
  CHECK_LE(topk, kMaxTopk);
  CHECK_LE(router_logits.size(1), kGateTopkThreads);
  CHECK_EQ(topk_weights_output.sizes(),
           torch::IntArrayRef({router_logits.size(0), topk}));
  CHECK_EQ(topk_weights_output.scalar_type(), torch::kFloat32);
  CHECK(topk_weights_output.is_contiguous());
  CHECK_EQ(topk_ids_output.sizes(),
           torch::IntArrayRef({router_logits.size(0), topk}));
  CHECK_EQ(topk_ids_output.scalar_type(), torch::kInt32);
  CHECK(topk_ids_output.is_contiguous());
  CHECK_EQ(topk_weights_output.device(), router_logits.device());
  CHECK_EQ(topk_ids_output.device(), router_logits.device());

  const XLLMOptionalDeviceGuard device_guard(at::device_of(router_logits));
  const musaStream_t stream = current_musa_stream(router_logits.device());
  const bool warp_inputs_aligned =
      reinterpret_cast<uintptr_t>(router_logits.data_ptr<float>()) %
              alignof(float4) ==
          0 &&
      reinterpret_cast<uintptr_t>(correction_bias.data_ptr<float>()) %
              alignof(float4) ==
          0;
  if (router_logits.size(1) == kDeepseekV4Experts && warp_inputs_aligned) {
    const dim3 grid(static_cast<unsigned int>(
        (router_logits.size(0) + kGateTopkWarps - 1) / kGateTopkWarps));
    const dim3 block(kGateTopkWarpSize, kGateTopkWarps);
    deepseek_v4_gate_topk_warp_kernel<<<grid, block, 0, stream>>>(
        router_logits.data_ptr<float>(),
        correction_bias.data_ptr<float>(),
        topk_weights_output.data_ptr<float>(),
        topk_ids_output.data_ptr<int32_t>(),
        router_logits.size(0),
        topk,
        static_cast<float>(route_scale));
    check_musa_kernel_launch("deepseek_v4_gate_topk_warp");
    return std::make_tuple(topk_weights_output, topk_ids_output);
  }

  const dim3 grid(static_cast<unsigned int>(router_logits.size(0)));
  const dim3 block(kGateTopkThreads);
  const size_t shared_memory = static_cast<size_t>(
      2 * router_logits.size(1) * static_cast<int64_t>(sizeof(float)));
  deepseek_v4_gate_topk_kernel<<<grid, block, shared_memory, stream>>>(
      router_logits.data_ptr<float>(),
      correction_bias.data_ptr<float>(),
      topk_weights_output.data_ptr<float>(),
      topk_ids_output.data_ptr<int32_t>(),
      router_logits.size(1),
      topk,
      static_cast<float>(route_scale));
  check_musa_kernel_launch("deepseek_v4_gate_topk");
  return std::make_tuple(topk_weights_output, topk_ids_output);
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_gate_hash_topk(
    const torch::Tensor& router_logits,
    const torch::Tensor& input_ids,
    const torch::Tensor& tid2eid,
    int64_t topk,
    double route_scale,
    const torch::Tensor& topk_weights_output,
    const torch::Tensor& topk_ids_output) {
  CHECK(router_logits.defined() && input_ids.defined() && tid2eid.defined());
  CHECK_EQ(router_logits.dim(), 2);
  CHECK_EQ(router_logits.scalar_type(), torch::kFloat32);
  CHECK(router_logits.is_contiguous());
  CHECK(input_ids.scalar_type() == torch::kInt32 ||
        input_ids.scalar_type() == torch::kInt64);
  CHECK(input_ids.is_contiguous());
  CHECK_EQ(input_ids.numel(), router_logits.size(0));
  CHECK_EQ(tid2eid.dim(), 2);
  CHECK(tid2eid.scalar_type() == torch::kInt32 ||
        tid2eid.scalar_type() == torch::kInt64);
  CHECK(tid2eid.is_contiguous());
  CHECK_EQ(tid2eid.size(1), topk);
  CHECK_EQ(tid2eid.device(), router_logits.device());
  CHECK_EQ(input_ids.device(), router_logits.device());
  CHECK_GT(topk, 0);
  CHECK_LE(topk, kMaxTopk);
  CHECK_EQ(topk_weights_output.sizes(),
           torch::IntArrayRef({router_logits.size(0), topk}));
  CHECK_EQ(topk_weights_output.scalar_type(), torch::kFloat32);
  CHECK(topk_weights_output.is_contiguous());
  CHECK_EQ(topk_ids_output.sizes(),
           torch::IntArrayRef({router_logits.size(0), topk}));
  CHECK_EQ(topk_ids_output.scalar_type(), torch::kInt32);
  CHECK(topk_ids_output.is_contiguous());
  CHECK_EQ(topk_weights_output.device(), router_logits.device());
  CHECK_EQ(topk_ids_output.device(), router_logits.device());

  const XLLMOptionalDeviceGuard device_guard(at::device_of(router_logits));
  const musaStream_t stream = current_musa_stream(router_logits.device());
  if (input_ids.scalar_type() == torch::kInt32 &&
      tid2eid.scalar_type() == torch::kInt32) {
    launch_deepseek_v4_gate_hash_topk<int32_t, int32_t>(router_logits,
                                                        input_ids,
                                                        tid2eid,
                                                        route_scale,
                                                        topk_weights_output,
                                                        topk_ids_output,
                                                        stream);
  } else if (input_ids.scalar_type() == torch::kInt32) {
    launch_deepseek_v4_gate_hash_topk<int32_t, int64_t>(router_logits,
                                                        input_ids,
                                                        tid2eid,
                                                        route_scale,
                                                        topk_weights_output,
                                                        topk_ids_output,
                                                        stream);
  } else if (tid2eid.scalar_type() == torch::kInt32) {
    launch_deepseek_v4_gate_hash_topk<int64_t, int32_t>(router_logits,
                                                        input_ids,
                                                        tid2eid,
                                                        route_scale,
                                                        topk_weights_output,
                                                        topk_ids_output,
                                                        stream);
  } else {
    launch_deepseek_v4_gate_hash_topk<int64_t, int64_t>(router_logits,
                                                        input_ids,
                                                        tid2eid,
                                                        route_scale,
                                                        topk_weights_output,
                                                        topk_ids_output,
                                                        stream);
  }
  check_musa_kernel_launch("deepseek_v4_gate_hash_topk");
  return std::make_tuple(topk_weights_output, topk_ids_output);
}

}  // namespace xllm::kernel::musa
