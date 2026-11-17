/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <optional>

#include "core/kernels/musa/mhc_ops.h"
#include "core/kernels/musa/sigmoid_gated_rms_norm.h"

namespace xllm::kernel::musa {
namespace {

class MhcNormFusionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (musaGetDeviceCount(&count) != musaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device unavailable";
    }
    at::detail::getMUSAHooks().initMUSA();
    no_grad_.emplace();
    stream_.emplace(c10::musa::getStreamFromPool(false, 0));
    guard_.emplace(stream_->unwrap());
    torch::manual_seed(20261116);
  }

  void TearDown() override {
    if (stream_.has_value()) {
      stream_->synchronize();
    }
  }

  void check_post(int64_t rows,
                  int64_t streams,
                  int64_t hidden,
                  bool unaligned = false,
                  bool strided = false) {
    torch::Tensor x = torch::randn({rows, hidden}).to(torch::kBFloat16);
    torch::Tensor residual =
        torch::randn({rows, streams, hidden}).to(torch::kBFloat16);
    torch::Tensor post = torch::randn({rows, streams});
    // Asymmetric coefficients catch a transposed comb matrix.
    torch::Tensor comb = torch::randn({rows, streams, streams});
    torch::Tensor expected =
        post.unsqueeze(-1) * x.to(torch::kFloat32).unsqueeze(1);
    for (int64_t source = 0; source < streams; ++source) {
      expected += comb.select(1, source).unsqueeze(-1) *
                  residual.select(1, source).to(torch::kFloat32).unsqueeze(1);
    }
    expected = expected.to(torch::kBFloat16).to(torch::kFloat32);
    torch::Tensor x_device = x.to(device_);
    torch::Tensor residual_device = residual.to(device_);
    if (unaligned) {
      torch::Tensor backing = torch::empty({x.numel() + 1}, x_device.options());
      x_device = backing.narrow(0, 1, x.numel()).view(x.sizes());
      x_device.copy_(x);
    }
    if (strided) {
      residual_device =
          residual.transpose(1, 2).contiguous().to(device_).transpose(1, 2);
    }
    const torch::Tensor actual =
        mhc_post(x_device, residual_device, post.to(device_), comb.to(device_));
    stream_->synchronize();
    ASSERT_EQ(actual.sizes(), residual.sizes());
    EXPECT_TRUE(torch::allclose(
        actual.to(torch::kFloat32).cpu(), expected, 0.008, 0.001));
  }

  void check_norm(int64_t rows, double eps, bool strided, bool small) {
    torch::Tensor x =
        torch::randn({rows, 128}).mul(small ? 1e-4 : 1.0).to(torch::kBFloat16);
    torch::Tensor gate = torch::randn({rows, 128}).mul(10).to(torch::kBFloat16);
    torch::Tensor weight = torch::randn({128}).to(torch::kBFloat16);
    const torch::Tensor xf = x.to(torch::kFloat32);
    const torch::Tensor expected =
        (xf * torch::rsqrt(xf.square().mean(-1, true) + eps) *
         weight.to(torch::kFloat32) * torch::sigmoid(gate.to(torch::kFloat32)))
            .to(torch::kBFloat16)
            .to(torch::kFloat32);
    torch::Tensor xd = x.to(device_);
    torch::Tensor gd = gate.to(device_);
    if (strided) {
      xd = torch::empty({rows, 140}, xd.options()).narrow(1, 0, 128);
      gd = torch::empty({rows, 144}, gd.options()).narrow(1, 0, 128);
      xd.copy_(x);
      gd.copy_(gate);
    }
    const torch::Tensor actual =
        sigmoid_gated_rms_norm(xd, gd, weight.to(device_), eps);
    stream_->synchronize();
    ASSERT_EQ(actual.sizes(), x.sizes());
    ASSERT_EQ(actual.scalar_type(), torch::kBFloat16);
    EXPECT_TRUE(torch::allclose(
        actual.to(torch::kFloat32).cpu(), expected, 0.008, 1e-5));
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::optional<torch::NoGradGuard> no_grad_;
  std::optional<c10::musa::MUSAStream> stream_;
  std::optional<c10::StreamGuard> guard_;
};

TEST_F(MhcNormFusionTest, PostTilesAndGenericInputsMatchReference) {
  check_post(/*rows=*/1, /*streams=*/4, /*hidden=*/8);
  check_post(/*rows=*/9, /*streams=*/4, /*hidden=*/520);
  check_post(/*rows=*/2500, /*streams=*/4, /*hidden=*/4096);
  check_post(/*rows=*/3, /*streams=*/4, /*hidden=*/520, /*unaligned=*/true);
  check_post(/*rows=*/3,
             /*streams=*/4,
             /*hidden=*/520,
             /*unaligned=*/false,
             /*strided=*/true);
  check_post(/*rows=*/3, /*streams=*/2, /*hidden=*/128);
  check_post(/*rows=*/0, /*streams=*/4, /*hidden=*/128);
}

TEST_F(MhcNormFusionTest, SigmoidNormMatchesReferenceAcrossRowsAndEpsilon) {
  for (const int64_t rows : {0, 1, 7, 9, 160000}) {
    SCOPED_TRACE(::testing::Message() << "rows=" << rows);
    check_norm(rows, /*eps=*/1e-6, /*strided=*/false, /*small=*/false);
  }
  check_norm(/*rows=*/9, /*eps=*/1e-5, /*strided=*/true, /*small=*/true);
  check_norm(/*rows=*/9, /*eps=*/1e-6, /*strided=*/true, /*small=*/true);
}

TEST_F(MhcNormFusionTest, SigmoidGateHasCorrectSaturationAndZero) {
  const torch::Tensor x =
      torch::ones({3, 128}, torch::TensorOptions().dtype(torch::kBFloat16));
  const torch::Tensor weight = torch::ones({128}, x.options());
  const torch::Tensor gate = torch::tensor({-1000.0f, 0.0f, 1000.0f})
                                 .unsqueeze(1)
                                 .expand({3, 128})
                                 .contiguous()
                                 .to(torch::kBFloat16);
  const torch::Tensor actual = sigmoid_gated_rms_norm(
      x.to(device_), gate.to(device_), weight.to(device_), 1e-6);
  stream_->synchronize();
  const torch::Tensor expected =
      torch::tensor({0.0f, 0.5f, 1.0f}).unsqueeze(1).expand({3, 128});
  EXPECT_TRUE(torch::equal(actual.to(torch::kFloat32).cpu(), expected));
}

}  // namespace
}  // namespace xllm::kernel::musa
