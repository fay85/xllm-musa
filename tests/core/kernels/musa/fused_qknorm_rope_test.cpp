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

#include <Python.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

bool is_musa_available() {
  int device_count = 0;
  const musaError_t status = musaGetDeviceCount(&device_count);
  return status == musaSuccess && device_count > 0;
}

bool initialize_musa_backend() {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(/*initsigs=*/0);
  }
  const PyGILState_STATE gil_state = PyGILState_Ensure();
  const int32_t result = PyRun_SimpleString("import torch_musa");
  if (result != 0 && PyErr_Occurred()) {
    PyErr_Print();
  }
  PyGILState_Release(gil_state);
  return result == 0;
}

torch::Tensor apply_neox_rope(const torch::Tensor& input,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin,
                              int64_t rotary_dim) {
  const torch::Tensor rotary = input.slice(/*dim=*/-1, /*start=*/0, rotary_dim);
  const torch::Tensor passthrough = input.slice(
      /*dim=*/-1, /*start=*/rotary_dim, /*end=*/input.size(-1));
  const std::vector<torch::Tensor> halves =
      rotary.chunk(/*chunks=*/2, /*dim=*/-1);
  const torch::Tensor rotated = torch::cat({-halves[1], halves[0]}, /*dim=*/-1);
  const torch::Tensor cos_full =
      torch::cat({cos, cos}, /*dim=*/-1).unsqueeze(1);
  const torch::Tensor sin_full =
      torch::cat({sin, sin}, /*dim=*/-1).unsqueeze(1);
  const torch::Tensor rope_output = rotary * cos_full + rotated * sin_full;
  return torch::cat({rope_output, passthrough}, /*dim=*/-1);
}

void apply_qg_reference(torch::Tensor& qkv,
                        int64_t num_heads_q,
                        int64_t num_heads_k,
                        int64_t head_dim,
                        double eps,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin_cache,
                        const torch::Tensor& position_ids,
                        bool qg_interleaved) {
  const int64_t num_tokens = qkv.size(0);
  const int64_t qg_size = 2 * num_heads_q * head_dim;
  const int64_t k_size = num_heads_k * head_dim;
  const int64_t rotary_dim = cos_sin_cache.size(1);
  const int64_t half_rotary_dim = rotary_dim / 2;

  torch::Tensor query =
      qg_interleaved ? qkv.slice(/*dim=*/-1, /*start=*/0, /*end=*/qg_size)
                           .view({num_tokens, num_heads_q, 2, head_dim})
                           .select(/*dim=*/2, /*index=*/0)
                     : qkv.slice(/*dim=*/-1,
                                 /*start=*/0,
                                 /*end=*/num_heads_q * head_dim)
                           .view({num_tokens, num_heads_q, head_dim});
  torch::Tensor key =
      qkv.slice(/*dim=*/-1, /*start=*/qg_size, /*end=*/qg_size + k_size)
          .view({num_tokens, num_heads_k, head_dim});

  const torch::Tensor query_float = query.to(torch::kFloat32);
  const torch::Tensor key_float = key.to(torch::kFloat32);
  const torch::Tensor query_scale =
      (q_weight.to(torch::kFloat32) + 1.0).view({1, 1, head_dim});
  const torch::Tensor key_scale =
      (k_weight.to(torch::kFloat32) + 1.0).view({1, 1, head_dim});
  torch::Tensor normalized_query =
      query_float *
      torch::rsqrt(
          (query_float * query_float).mean(/*dim=*/-1, /*keepdim=*/true) +
          eps) *
      query_scale;
  torch::Tensor normalized_key =
      key_float *
      torch::rsqrt((key_float * key_float).mean(/*dim=*/-1, /*keepdim=*/true) +
                   eps) *
      key_scale;

  const torch::Tensor selected =
      cos_sin_cache.index_select(/*dim=*/0, position_ids.to(torch::kLong))
          .to(torch::kFloat32);
  const torch::Tensor cos = selected.slice(
      /*dim=*/-1, /*start=*/0, /*end=*/half_rotary_dim);
  const torch::Tensor sin = selected.slice(
      /*dim=*/-1, /*start=*/half_rotary_dim, /*end=*/rotary_dim);
  normalized_query = apply_neox_rope(normalized_query, cos, sin, rotary_dim);
  normalized_key = apply_neox_rope(normalized_key, cos, sin, rotary_dim);

  query.copy_(normalized_query.to(query.scalar_type()));
  key.copy_(normalized_key.to(key.scalar_type()));
}

TEST(FusedQknormRopeMusaTest, MatchesInterleavedQGateReference) {
  if (!is_musa_available()) {
    GTEST_SKIP() << "MUSA device is unavailable";
  }
  ASSERT_TRUE(initialize_musa_backend());
  ASSERT_EQ(musaSetDevice(/*device=*/0), musaSuccess);

  constexpr int64_t kNumTokens = 7;
  constexpr int64_t kNumHeadsQ = 4;
  constexpr int64_t kNumHeadsK = 2;
  constexpr int64_t kNumHeadsV = 2;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kRotaryDim = 64;
  constexpr int64_t kMaxPosition = 128;
  constexpr double kEps = 1e-6;
  constexpr int64_t kSeed = 20260828;
  constexpr double kTolerance = 3e-2;

  torch::manual_seed(kSeed);
  const torch::TensorOptions cpu_bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  const torch::TensorOptions cpu_int =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  const int64_t total_heads = 2 * kNumHeadsQ + kNumHeadsK + kNumHeadsV;
  const torch::Tensor source_cpu =
      torch::randn({kNumTokens, total_heads * kHeadDim}, cpu_bf16) * 0.1;
  const torch::Tensor q_weight_cpu = torch::randn({kHeadDim}, cpu_bf16) * 0.05;
  const torch::Tensor k_weight_cpu = torch::randn({kHeadDim}, cpu_bf16) * 0.05;
  const torch::Tensor cos_sin_cpu =
      torch::randn({kMaxPosition, kRotaryDim}, cpu_bf16);
  const torch::Tensor position_ids_cpu = torch::randint(
      /*low=*/0, /*high=*/kMaxPosition, {kNumTokens}, cpu_int);

  torch::Tensor reference = source_cpu.clone();
  apply_qg_reference(reference,
                     kNumHeadsQ,
                     kNumHeadsK,
                     kHeadDim,
                     kEps,
                     q_weight_cpu,
                     k_weight_cpu,
                     cos_sin_cpu,
                     position_ids_cpu,
                     /*qg_interleaved=*/true);

  const torch::Device musa_device(torch::kPrivateUse1, /*index=*/0);
  torch::Tensor actual = source_cpu.to(musa_device);
  fused_qk_norm_rope(actual,
                     kNumHeadsQ,
                     kNumHeadsK,
                     kNumHeadsV,
                     kHeadDim,
                     kEps,
                     q_weight_cpu.to(musa_device),
                     k_weight_cpu.to(musa_device),
                     cos_sin_cpu.to(musa_device),
                     /*interleaved=*/false,
                     position_ids_cpu.to(musa_device),
                     /*k_head_offset=*/2 * kNumHeadsQ,
                     /*q_head_stride=*/2);
  actual = actual.to(torch::kCPU);

  EXPECT_TRUE(torch::allclose(actual, reference, kTolerance, kTolerance));
  const torch::Tensor actual_qg =
      actual.slice(/*dim=*/-1, /*start=*/0, /*end=*/2 * kNumHeadsQ * kHeadDim)
          .view({kNumTokens, kNumHeadsQ, 2, kHeadDim});
  const torch::Tensor source_qg =
      source_cpu
          .slice(/*dim=*/-1, /*start=*/0, /*end=*/2 * kNumHeadsQ * kHeadDim)
          .view({kNumTokens, kNumHeadsQ, 2, kHeadDim});
  EXPECT_TRUE(torch::equal(actual_qg.select(/*dim=*/2, /*index=*/1),
                           source_qg.select(/*dim=*/2, /*index=*/1)));
  const int64_t value_start = (2 * kNumHeadsQ + kNumHeadsK) * kHeadDim;
  EXPECT_TRUE(torch::equal(actual.slice(/*dim=*/-1, value_start),
                           source_cpu.slice(/*dim=*/-1, value_start)));
}

TEST(FusedQknormRopeMusaTest, KeepsGroupedQGateLayoutCompatible) {
  if (!is_musa_available()) {
    GTEST_SKIP() << "MUSA device is unavailable";
  }
  ASSERT_TRUE(initialize_musa_backend());
  ASSERT_EQ(musaSetDevice(/*device=*/0), musaSuccess);

  constexpr int64_t kNumTokens = 5;
  constexpr int64_t kNumHeadsQ = 4;
  constexpr int64_t kNumHeadsK = 2;
  constexpr int64_t kNumHeadsV = 2;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kRotaryDim = 64;
  constexpr int64_t kMaxPosition = 128;
  constexpr double kEps = 1e-6;
  constexpr int64_t kSeed = 20260829;
  constexpr double kTolerance = 3e-2;

  torch::manual_seed(kSeed);
  const torch::TensorOptions cpu_bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  const torch::TensorOptions cpu_int =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  const int64_t total_heads = 2 * kNumHeadsQ + kNumHeadsK + kNumHeadsV;
  const torch::Tensor source_cpu =
      torch::randn({kNumTokens, total_heads * kHeadDim}, cpu_bf16) * 0.1;
  const torch::Tensor q_weight_cpu = torch::randn({kHeadDim}, cpu_bf16) * 0.05;
  const torch::Tensor k_weight_cpu = torch::randn({kHeadDim}, cpu_bf16) * 0.05;
  const torch::Tensor cos_sin_cpu =
      torch::randn({kMaxPosition, kRotaryDim}, cpu_bf16);
  const torch::Tensor position_ids_cpu = torch::randint(
      /*low=*/0, /*high=*/kMaxPosition, {kNumTokens}, cpu_int);

  torch::Tensor reference = source_cpu.clone();
  apply_qg_reference(reference,
                     kNumHeadsQ,
                     kNumHeadsK,
                     kHeadDim,
                     kEps,
                     q_weight_cpu,
                     k_weight_cpu,
                     cos_sin_cpu,
                     position_ids_cpu,
                     /*qg_interleaved=*/false);

  const torch::Device musa_device(torch::kPrivateUse1, /*index=*/0);
  torch::Tensor actual = source_cpu.to(musa_device);
  fused_qk_norm_rope(actual,
                     kNumHeadsQ,
                     kNumHeadsK,
                     kNumHeadsV,
                     kHeadDim,
                     kEps,
                     q_weight_cpu.to(musa_device),
                     k_weight_cpu.to(musa_device),
                     cos_sin_cpu.to(musa_device),
                     /*interleaved=*/false,
                     position_ids_cpu.to(musa_device),
                     /*k_head_offset=*/2 * kNumHeadsQ,
                     /*q_head_stride=*/1);
  actual = actual.to(torch::kCPU);

  EXPECT_TRUE(torch::allclose(actual, reference, kTolerance, kTolerance));
  const int64_t gate_start = kNumHeadsQ * kHeadDim;
  const int64_t gate_end = 2 * kNumHeadsQ * kHeadDim;
  EXPECT_TRUE(torch::equal(actual.slice(/*dim=*/-1, gate_start, gate_end),
                           source_cpu.slice(/*dim=*/-1, gate_start, gate_end)));
  const int64_t value_start = (2 * kNumHeadsQ + kNumHeadsK) * kHeadDim;
  EXPECT_TRUE(torch::equal(actual.slice(/*dim=*/-1, value_start),
                           source_cpu.slice(/*dim=*/-1, value_start)));
}

}  // namespace
}  // namespace xllm::kernel::musa
