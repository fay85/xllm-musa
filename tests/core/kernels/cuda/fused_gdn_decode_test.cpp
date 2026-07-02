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
#include <torch/torch.h>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/param.h"

namespace xllm::kernel::cuda {
namespace test {

class FusedGdnDecodeKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    torch::manual_seed(2026);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  torch::Device device_{torch::kCPU};
};

TEST_F(FusedGdnDecodeKernelTest, PadSlotZerosOutput) {
  const int64_t batch_size = 2;
  const int64_t num_k_heads = 1;
  const int64_t num_v_heads = 1;
  const int64_t head_k_dim = 4;
  const int64_t head_v_dim = 4;
  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;
  const int64_t mixed_dim = 2 * qk_cols + v_cols;

  const auto bf16 = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  torch::Tensor mixed_qkv = torch::randn({batch_size, mixed_dim}, bf16);
  torch::Tensor state = torch::zeros(
      {2, num_v_heads, head_v_dim, head_k_dim},
      torch::TensorOptions().device(device_).dtype(torch::kFloat32));
  torch::Tensor A_log = torch::zeros({num_v_heads}, bf16);
  torch::Tensor a = torch::zeros({batch_size, num_v_heads}, bf16);
  torch::Tensor dt_bias = torch::zeros({num_v_heads}, bf16);
  torch::Tensor b = torch::zeros({batch_size, num_v_heads}, bf16);
  torch::Tensor state_indices =
      torch::tensor({0, -1}, torch::TensorOptions().device(device_).dtype(torch::kInt32));
  torch::Tensor output =
      torch::full({batch_size, num_v_heads, head_v_dim}, 42.0f, bf16);

  MateGatedDeltaRuleDecodeParams params;
  params.mixed_qkv = mixed_qkv;
  params.state = state;
  params.A_log = A_log;
  params.a = a;
  params.dt_bias = dt_bias;
  params.b = b;
  params.state_indices = state_indices;
  params.num_k_heads = num_k_heads;
  params.num_v_heads = num_v_heads;
  params.head_k_dim = head_k_dim;
  params.head_v_dim = head_v_dim;
  params.scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));
  params.use_qk_l2norm = true;
  params.decode_output = output;

  fused_gated_delta_rule_decode(params);

  torch::Tensor row0 = output.index({0}).cpu();
  torch::Tensor row1 = output.index({1}).cpu();
  EXPECT_FALSE(torch::allclose(row0, torch::full_like(row0, 42.0f), /*rtol=*/0,
                               /*atol=*/0))
      << "active slot should overwrite prefilled output";
  EXPECT_TRUE(torch::allclose(row1, torch::zeros_like(row1), /*rtol=*/0,
                              /*atol=*/0))
      << "pad slot (state_index < 0) must zero output";
}

TEST_F(FusedGdnDecodeKernelTest, ProducesFiniteOutputForActiveSlots) {
  const int64_t batch_size = 1;
  const int64_t num_k_heads = 1;
  const int64_t num_v_heads = 1;
  const int64_t head_k_dim = 8;
  const int64_t head_v_dim = 8;
  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;
  const int64_t mixed_dim = 2 * qk_cols + v_cols;

  const auto bf16 = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  torch::Tensor mixed_qkv = torch::randn({batch_size, mixed_dim}, bf16) * 0.05f;
  torch::Tensor state = torch::zeros(
      {1, num_v_heads, head_v_dim, head_k_dim},
      torch::TensorOptions().device(device_).dtype(torch::kFloat32));
  torch::Tensor A_log = torch::full({num_v_heads}, -2.0f, bf16);
  torch::Tensor a = torch::randn({batch_size, num_v_heads}, bf16) * 0.01f;
  torch::Tensor dt_bias = torch::zeros({num_v_heads}, bf16);
  torch::Tensor b = torch::randn({batch_size, num_v_heads}, bf16) * 0.01f;
  torch::Tensor state_indices =
      torch::zeros({batch_size}, torch::TensorOptions().device(device_).dtype(torch::kInt32));

  MateGatedDeltaRuleDecodeParams params;
  params.mixed_qkv = mixed_qkv;
  params.state = state;
  params.A_log = A_log;
  params.a = a;
  params.dt_bias = dt_bias;
  params.b = b;
  params.state_indices = state_indices;
  params.num_k_heads = num_k_heads;
  params.num_v_heads = num_v_heads;
  params.head_k_dim = head_k_dim;
  params.head_v_dim = head_v_dim;
  params.scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));
  params.use_qk_l2norm = true;

  torch::Tensor output = fused_gated_delta_rule_decode(params);
  EXPECT_TRUE(torch::isfinite(output.cpu()).all().item<bool>());
}

}  // namespace test
}  // namespace xllm::kernel::cuda
