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

namespace xllm::kernel::cuda {
namespace test {

namespace {

torch::Tensor reference_gemma_rms_norm(const torch::Tensor& input,
                                       const torch::Tensor& weight,
                                       double eps) {
  auto x = input.to(torch::kFloat32);
  auto w = weight.to(torch::kFloat32);
  auto mean_sq = x.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
  auto normed = x * torch::rsqrt(mean_sq + eps);
  return (normed * (1.0f + w)).to(input.scalar_type());
}

}  // namespace

class GemmaNormKernelTest : public ::testing::Test {
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

TEST_F(GemmaNormKernelTest, GemmaRmsNormMatchesReference) {
  for (const auto dtype : {torch::kBFloat16, torch::kFloat16, torch::kFloat32}) {
    const auto opts = torch::TensorOptions().device(device_).dtype(dtype);
    const int64_t rows = 4;
    const int64_t hidden = 128;
    torch::Tensor input = torch::randn({rows, hidden}, opts) * 0.1;
    torch::Tensor weight = torch::randn({hidden}, opts) * 0.01;
    torch::Tensor output = torch::empty_like(input);
    const double eps = 1e-6;

    gemma_rms_norm(output, input, weight, eps);

    torch::Tensor expected = reference_gemma_rms_norm(input, weight, eps);
    EXPECT_TRUE(torch::allclose(output.cpu(), expected.cpu(), /*rtol=*/1e-2,
                                /*atol=*/1e-2))
        << "dtype mismatch for scalar type " << static_cast<int>(dtype);
  }
}

}  // namespace test
}  // namespace xllm::kernel::cuda
