/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <cmath>
#include <limits>
#include <optional>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

class FusedSwigluQuantTest : public ::testing::Test {
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

  void check_reference(const torch::Tensor& input, double limit) {
    const int64_t width = input.size(1) / 2;
    // Reference is the existing unfused Activation followed by dense quant.
    torch::Tensor gate = input.slice(1, 0, width).to(torch::kFloat32);
    torch::Tensor up = input.slice(1, width).to(torch::kFloat32);
    if (std::isfinite(limit) && limit > 0.0 && limit < 1000000.0) {
      gate = gate.clamp_max(limit);
      up = up.clamp(-limit, limit);
    }
    torch::Tensor activated = (torch::silu(gate) * up).to(torch::kBFloat16);
    auto [expected_q, expected_s] =
        per_token_group_quant_fp8(activated, /*group_size=*/128);
    auto [actual_q, actual_s] =
        fused_swiglu_quant_fp8(input, /*group_size=*/128, limit);
    stream_->synchronize();
    ASSERT_EQ(actual_q.sizes(), expected_q.sizes());
    ASSERT_EQ(actual_s.sizes(), expected_s.sizes());
    ASSERT_EQ(actual_q.scalar_type(), torch::kFloat8_e4m3fn);
    ASSERT_EQ(actual_s.scalar_type(), torch::kFloat32);
    ASSERT_TRUE(torch::allclose(actual_s.cpu(), expected_s.cpu(), 1e-5, 1e-12));
    const torch::Tensor actual = actual_q.to(torch::kFloat32).cpu();
    const torch::Tensor expected = expected_q.to(torch::kFloat32).cpu();
    // FP32 SiLU implementations can straddle a BF16 rounding boundary.
    // Permit at most one FP8 step for a tiny fraction of values.
    EXPECT_TRUE(torch::allclose(actual, expected, 0.125, 0.015625));
    EXPECT_LE((actual != expected).to(torch::kFloat32).mean().item<float>(),
              0.002f);
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::optional<torch::NoGradGuard> no_grad_;
  std::optional<c10::musa::MUSAStream> stream_;
  std::optional<c10::StreamGuard> guard_;
};

TEST_F(FusedSwigluQuantTest, ClampBoundariesAndScaleFloor) {
  // Both signs, adjacent BF16 values around +/-7, and negative gates far
  // below -limit distinguish upper-only gate clamp from symmetric clamp.
  torch::Tensor values = torch::tensor({-100.0f,
                                        -8.0f,
                                        -7.03125f,
                                        -7.0f,
                                        -6.96875f,
                                        -0.001f,
                                        0.0f,
                                        0.001f,
                                        6.96875f,
                                        7.0f,
                                        7.03125f,
                                        8.0f,
                                        100.0f,
                                        0.5f,
                                        -0.5f,
                                        1.0f});
  torch::Tensor gate = values.repeat({8}).reshape({1, 128});
  torch::Tensor up = gate.flip({1});
  torch::Tensor input =
      torch::cat({gate, up}, 1).to(torch::kBFloat16).to(device_);
  check_reference(input, /*limit=*/7.0);
  check_reference(input, /*limit=*/10.0);
  // Independently check the dense scale floor, including tiny nonzero groups.
  input = torch::full({2, 256}, 1e-8f).to(torch::kBFloat16).to(device_);
  input[0].zero_();
  auto [q, scales] = fused_swiglu_quant_fp8(input, /*group_size=*/128, 7.0);
  stream_->synchronize();
  EXPECT_TRUE(torch::equal(scales.cpu(), torch::full({2, 1}, 1e-10f)));
  EXPECT_EQ(q.to(torch::kFloat32).cpu().abs().max().item<float>(), 0.0f);
}

TEST_F(FusedSwigluQuantTest, DecodeTailTilesAndPrefillMatchUnfused) {
  for (const int64_t rows : {1, 7, 2500}) {
    for (const int64_t width : {128, 4736, 12288}) {
      SCOPED_TRACE(::testing::Message()
                   << "rows=" << rows << " width=" << width);
      torch::Tensor input = (torch::randn({rows, 2 * width}) * 9)
                                .to(torch::kBFloat16)
                                .to(device_);
      check_reference(input, /*limit=*/7.0);
      check_reference(input, /*limit=*/10.0);
    }
  }
}

TEST_F(FusedSwigluQuantTest, DisabledLimitAndEmptyInput) {
  const torch::Tensor input =
      torch::randn({3, 256}).to(torch::kBFloat16).to(device_);
  for (const double limit : {0.0,
                             -1.0,
                             1000000.0,
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    check_reference(input, limit);
  }
  const torch::Tensor empty = torch::empty({0, 256}, input.options());
  auto [q, scales] = fused_swiglu_quant_fp8(empty, /*group_size=*/128, 7.0);
  EXPECT_EQ(q.sizes(), torch::IntArrayRef({0, 128}));
  EXPECT_EQ(scales.sizes(), torch::IntArrayRef({0, 1}));
}

}  // namespace
}  // namespace xllm::kernel::musa
