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

#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>

#include "core/kernels/cuda/mate_gdn_ops.h"
#include "core/kernels/param.h"

namespace xllm::kernel::cuda {
namespace test {

namespace {

// Torch reference for the fused GDN decode (T=1). Mirrors the kernel math so
// any divergence is a kernel bug. The state tensor is stored in production
// layout [pool, Hv, V, K] (K innermost); we expose a [K, V] view via
// transpose(-2, -1) for compact math, which shares memory with the storage so
// in-place updates remain consistent. Returns decode output [B, Hv, V].
torch::Tensor fused_gdn_decode_reference(const torch::Tensor& mixed_qkv,
                                         torch::Tensor& state,
                                         const torch::Tensor& A_log,
                                         const torch::Tensor& a,
                                         const torch::Tensor& dt_bias,
                                         const torch::Tensor& b,
                                         const torch::Tensor& state_indices,
                                         int64_t num_k_heads,
                                         int64_t num_v_heads,
                                         int64_t head_k_dim,
                                         int64_t head_v_dim,
                                         double scale,
                                         double softplus_beta,
                                         double softplus_threshold) {
  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;
  const int64_t group = num_v_heads / num_k_heads;
  auto idx_long = state_indices.to(torch::kLong).cpu();
  auto output = torch::zeros({batch_size, num_v_heads, head_v_dim},
                             mixed_qkv.options());
  // state_view shares memory with state; updates through h.mul_ / h.add_
  // therefore write back into the production [pool, Hv, V, K] storage at the
  // right positions without requiring an explicit copy.
  auto state_view = state.transpose(-2, -1);  // [pool, Hv, K, V]

  for (int64_t bi = 0; bi < batch_size; ++bi) {
    const int64_t slot = idx_long[bi].item<int64_t>();
    if (slot < 0) {
      continue;
    }
    auto row = mixed_qkv.select(0, bi).to(torch::kFloat32);
    auto q_all = row.slice(0, 0, qk_cols).view({num_k_heads, head_k_dim});
    auto k_all = row.slice(0, qk_cols, 2 * qk_cols)
                     .view({num_k_heads, head_k_dim});
    auto v_all = row.slice(0, 2 * qk_cols, 2 * qk_cols + v_cols)
                     .view({num_v_heads, head_v_dim});

    for (int64_t hv = 0; hv < num_v_heads; ++hv) {
      const int64_t hk = hv / group;
      auto q_h = q_all.select(0, hk);
      auto k_h = k_all.select(0, hk);
      auto v_h = v_all.select(0, hv);
      const float q_norm = std::sqrt(q_h.pow(2).sum().item<float>() + 1e-6f);
      const float k_norm = std::sqrt(k_h.pow(2).sum().item<float>() + 1e-6f);
      auto q_hat = q_h / q_norm * static_cast<float>(scale);
      auto k_hat = k_h / k_norm;

      const float a_val = a.index({bi, hv}).to(torch::kFloat32).item<float>();
      const float b_val = b.index({bi, hv}).to(torch::kFloat32).item<float>();
      const float A_log_val =
          A_log.index({hv}).to(torch::kFloat32).item<float>();
      const float dt_bias_val =
          dt_bias.index({hv}).to(torch::kFloat32).item<float>();
      const float pre = a_val + dt_bias_val;
      const float bx = static_cast<float>(softplus_beta) * pre;
      const float sp =
          (bx > static_cast<float>(softplus_threshold))
              ? pre
              : (std::log1p(std::exp(bx)) /
                 static_cast<float>(softplus_beta));
      const float g = -std::exp(A_log_val) * sp;
      const float g_exp = std::exp(g);
      const float beta_gate = 1.f / (1.f + std::exp(-b_val));

      // h is a [K, V] view into the production [pool, Hv, V, K] storage; its
      // in-place mutations are written back to the underlying tensor.
      auto h = state_view.select(0, slot).select(0, hv);
      h.mul_(g_exp);
      auto kv = (h * k_hat.unsqueeze(-1)).sum(0);
      auto delta = (v_h - kv) * beta_gate;
      h.add_(k_hat.unsqueeze(-1) * delta.unsqueeze(0));
      auto out_v = (h * q_hat.unsqueeze(-1)).sum(0);
      output.select(0, bi)
          .select(0, hv)
          .copy_(out_v.to(mixed_qkv.scalar_type()));
    }
  }
  return output;
}

class FusedGdnDecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping fused GDN decode test.";
    }
    torch::manual_seed(2026);
    device_ = torch::Device(torch::kCUDA, 0);
  }
  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(FusedGdnDecodeTest, MatchesReferenceGQA) {
  const int64_t batch_size = 3;
  const int64_t num_k_heads = 4;
  const int64_t num_v_heads = 8;
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 128;
  const int64_t pool = 16;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));

  auto bf16_opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto fp32_opts =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto i32_opts = torch::TensorOptions().device(device_).dtype(torch::kInt32);

  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;
  auto mixed_qkv =
      torch::randn({batch_size, 2 * qk_cols + v_cols}, bf16_opts) * 0.15;
  auto A_log = torch::randn({num_v_heads}, fp32_opts) * 0.5 - 1.0;
  auto dt_bias = torch::randn({num_v_heads}, fp32_opts) * 0.1;
  auto a_tensor = torch::randn({batch_size, num_v_heads}, bf16_opts) * 0.1;
  auto b_tensor = torch::randn({batch_size, num_v_heads}, bf16_opts) * 0.1;
  // State is fp32 to mirror the production Qwen3.5 SSM cache (mamba_ssm_dtype).
  // Storage layout is [pool, Hv, V, K] -- K is innermost, matching how the
  // layer writes back chunk_gated_delta_rule output via transpose(-1, -2).
  auto state =
      torch::randn({pool, num_v_heads, head_v_dim, head_k_dim}, fp32_opts) *
      0.05;
  auto state_indices = torch::tensor({0, 5, 12}, i32_opts);

  auto state_ref = state.clone();
  auto state_kernel = state.clone();

  auto output_ref = fused_gdn_decode_reference(mixed_qkv,
                                               state_ref,
                                               A_log,
                                               a_tensor,
                                               dt_bias,
                                               b_tensor,
                                               state_indices,
                                               num_k_heads,
                                               num_v_heads,
                                               head_k_dim,
                                               head_v_dim,
                                               scale,
                                               1.0,
                                               20.0);

  xllm::kernel::MateGatedDeltaRuleDecodeParams params;
  params.mixed_qkv = mixed_qkv;
  params.state = state_kernel;
  params.A_log = A_log;
  params.a = a_tensor;
  params.dt_bias = dt_bias;
  params.b = b_tensor;
  params.state_indices = state_indices;
  params.num_k_heads = num_k_heads;
  params.num_v_heads = num_v_heads;
  params.head_k_dim = head_k_dim;
  params.head_v_dim = head_v_dim;
  params.scale = scale;
  params.use_qk_l2norm = true;

  auto output_kernel = fused_gated_delta_rule_decode(params);

  EXPECT_TRUE(torch::allclose(output_kernel, output_ref, 1e-2, 1e-2))
      << "decode output diverges from reference";
  EXPECT_TRUE(torch::allclose(state_kernel, state_ref, 1e-2, 1e-2))
      << "in-place state cache diverges from reference";
}

TEST_F(FusedGdnDecodeTest, PaddedSlotSkipsCacheUpdate) {
  const int64_t batch_size = 2;
  const int64_t num_k_heads = 2;
  const int64_t num_v_heads = 4;
  const int64_t head_k_dim = 64;
  const int64_t head_v_dim = 64;
  const int64_t pool = 8;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));

  auto bf16_opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto fp32_opts =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto i32_opts = torch::TensorOptions().device(device_).dtype(torch::kInt32);

  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;
  auto mixed_qkv =
      torch::randn({batch_size, 2 * qk_cols + v_cols}, bf16_opts) * 0.15;
  auto A_log = torch::randn({num_v_heads}, fp32_opts) * 0.5 - 1.0;
  auto dt_bias = torch::randn({num_v_heads}, fp32_opts) * 0.1;
  auto a_tensor = torch::randn({batch_size, num_v_heads}, bf16_opts) * 0.1;
  auto b_tensor = torch::randn({batch_size, num_v_heads}, bf16_opts) * 0.1;
  auto state =
      torch::randn({pool, num_v_heads, head_v_dim, head_k_dim}, fp32_opts) *
      0.05;
  // Batch 0 valid (slot 3), batch 1 padded (-1).
  auto state_indices = torch::tensor({3, -1}, i32_opts);

  auto state_before = state.clone();
  auto state_kernel = state.clone();

  xllm::kernel::MateGatedDeltaRuleDecodeParams params;
  params.mixed_qkv = mixed_qkv;
  params.state = state_kernel;
  params.A_log = A_log;
  params.a = a_tensor;
  params.dt_bias = dt_bias;
  params.b = b_tensor;
  params.state_indices = state_indices;
  params.num_k_heads = num_k_heads;
  params.num_v_heads = num_v_heads;
  params.head_k_dim = head_k_dim;
  params.head_v_dim = head_v_dim;
  params.scale = scale;
  params.use_qk_l2norm = true;

  (void)fused_gated_delta_rule_decode(params);

  const float slot_3_diff =
      (state_kernel.select(0, 3) - state_before.select(0, 3))
          .abs()
          .max()
          .item<float>();
  EXPECT_GT(slot_3_diff, 0.f) << "valid slot was not touched";

  for (int64_t s = 0; s < pool; ++s) {
    if (s == 3) {
      continue;
    }
    EXPECT_TRUE(torch::equal(state_kernel.select(0, s),
                             state_before.select(0, s)))
        << "padded slot path leaked writes into pool index " << s;
  }
}

}  // namespace

}  // namespace test
}  // namespace xllm::kernel::cuda
