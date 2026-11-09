/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "kernels/mlu/bangc/include/fused_mhc_decode.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

constexpr int64_t kHc = 4;
constexpr int64_t kDim = 5120;
constexpr int64_t kMix = 24;
constexpr int64_t kSinkhornIter = 20;
constexpr double kHcEps = 1e-6;
constexpr double kNormEps = 1e-20;

struct MhcCase {
  torch::Tensor x;
  torch::Tensor residual_in;
  torch::Tensor post_in;
  torch::Tensor comb_in;
  torch::Tensor pre_in;
  torch::Tensor hc_fn;
  torch::Tensor hc_scale;
  torch::Tensor hc_base;
  torch::Tensor gamma;
};

struct MhcReference {
  torch::Tensor y;
  torch::Tensor norm;
  torch::Tensor pre;
  torch::Tensor post;
  torch::Tensor comb;
};

MhcCase make_case(int64_t tokens, const torch::Device& device) {
  torch::manual_seed(tokens);
  const torch::TensorOptions bf16 = torch::dtype(torch::kBFloat16);
  MhcCase c;
  c.x = torch::randn({tokens, kDim}).to(bf16).to(device);
  c.residual_in = torch::randn({tokens, kHc, kDim}).to(bf16).to(device);
  c.post_in = torch::randn({tokens, kHc}).to(device);
  c.comb_in = torch::randn({tokens, kHc, kHc}).to(device);
  c.pre_in = (torch::rand({tokens, kHc}) + 1e-6).to(device);
  c.hc_fn = (torch::randn({kMix, kHc * kDim}) * 0.02).to(bf16).to(device);
  c.hc_scale = (torch::rand({3}) * 0.5 + 0.1).to(device);
  c.hc_base = torch::randn({kMix}).to(device);
  c.gamma = torch::randn({kDim}).to(bf16).to(device);
  return c;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reference_sinkhorn(
    const torch::Tensor& mixes,
    const torch::Tensor& scale,
    const torch::Tensor& base) {
  using torch::indexing::Slice;
  const int64_t tokens = mixes.size(0);
  torch::Tensor pre =
      torch::sigmoid(mixes.index({Slice(), Slice(0, 4)}) * scale[0] +
                     base.index({Slice(0, 4)})) +
      kHcEps;
  torch::Tensor post =
      2.0 * torch::sigmoid(mixes.index({Slice(), Slice(4, 8)}) * scale[1] +
                           base.index({Slice(4, 8)}));
  torch::Tensor comb =
      mixes.index({Slice(), Slice(8)}).view({tokens, kHc, kHc}) * scale[2] +
      base.index({Slice(8)}).view({kHc, kHc});
  comb = torch::softmax(comb, /*dim=*/-1) + kHcEps;
  comb = comb / (comb.sum(/*dim=*/-2, /*keepdim=*/true) + kHcEps);
  for (int64_t i = 0; i < kSinkhornIter - 1; ++i) {
    comb = comb / (comb.sum(/*dim=*/-1, /*keepdim=*/true) + kHcEps);
    comb = comb / (comb.sum(/*dim=*/-2, /*keepdim=*/true) + kHcEps);
  }
  return {pre, post, comb};
}

// fp32 reference of the delayed-pre boundary. Everything downstream of y uses
// the kernel's bf16 residual_out, as the kernel does.
MhcReference reference_mhc(
    const MhcCase& c,
    const torch::Tensor& residual_out = torch::Tensor()) {
  const torch::Tensor x = c.x.to(torch::kFloat32);
  const torch::Tensor residual = c.residual_in.to(torch::kFloat32);
  MhcReference r;
  r.y = c.post_in.unsqueeze(-1) * x.unsqueeze(-2) +
        torch::sum(c.comb_in.unsqueeze(-1) * residual.unsqueeze(-2),
                   /*dim=*/1);
  const torch::Tensor y_rounded =
      residual_out.defined() ? residual_out.to(torch::kFloat32)
                             : r.y.to(torch::kBFloat16).to(torch::kFloat32);
  const torch::Tensor rms = torch::rsqrt(
      r.y.flatten(1).square().mean(/*dim=*/-1, /*keepdim=*/true) + kNormEps);
  const torch::Tensor mixes = torch::nn::functional::linear(
      y_rounded.flatten(1), c.hc_fn.to(torch::kFloat32));
  std::tie(r.pre, r.post, r.comb) =
      reference_sinkhorn(mixes * rms, c.hc_scale, c.hc_base);
  torch::Tensor z = torch::sum(c.pre_in.unsqueeze(-1) * y_rounded, /*dim=*/-2);
  r.norm =
      z *
      torch::rsqrt(z.square().mean(/*dim=*/-1, /*keepdim=*/true) + kNormEps) *
      c.gamma.to(torch::kFloat32);
  return r;
}

// Contiguous copy of `t` whose data starts one element past an allocation,
// so it is not 64-byte aligned.
torch::Tensor misaligned_copy(const torch::Tensor& t) {
  torch::Tensor storage = torch::empty({t.numel() + 1}, t.options());
  torch::Tensor view =
      storage.narrow(/*dim=*/0, /*start=*/1, /*length=*/t.numel())
          .view(t.sizes());
  view.copy_(t);
  return view;
}

double relative_l2(const torch::Tensor& actual, const torch::Tensor& expected) {
  const torch::Tensor a = actual.to(torch::kFloat64);
  const torch::Tensor e = expected.to(torch::kFloat64);
  const double den = e.square().sum().item<double>();
  return std::sqrt((a - e).square().sum().item<double>() /
                   std::max(den, 1e-300));
}

TEST(FusedMhcDelayedPreTest, MatchesReferenceIncludingTails) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  // Decode token counts (d-split kernel, every group size and tail), the
  // large-T handoff to the prefill kernel (640, 4097) and the d-split rounds
  // between them (300).
  for (const int64_t tokens :
       {1, 3, 8, 9, 17, 33, 63, 65, 89, 128, 129, 257, 300, 640, 4097}) {
    SCOPED_TRACE(tokens);
    const MhcCase c = make_case(tokens, device);
    const auto [norm_out, residual_out, next_pre, post_out, comb_out] =
        kernel::mlu::fused_mhc_delayed_pre(c.x,
                                           c.residual_in,
                                           c.hc_fn,
                                           c.gamma,
                                           c.post_in,
                                           c.comb_in,
                                           c.pre_in,
                                           c.hc_scale,
                                           c.hc_base,
                                           kSinkhornIter,
                                           kHcEps,
                                           kNormEps);
    const MhcReference r = reference_mhc(c, residual_out);
    for (const torch::Tensor& out :
         {norm_out, residual_out, next_pre, post_out, comb_out}) {
      EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
    }
    const double residual_mismatch = residual_out.ne(r.y.to(torch::kBFloat16))
                                         .to(torch::kFloat64)
                                         .mean()
                                         .item<double>();
    EXPECT_LT(residual_mismatch, 1e-3);
    EXPECT_LT(relative_l2(norm_out, r.norm), 3e-3);
    EXPECT_LT(relative_l2(next_pre, r.pre), 1e-5);
    EXPECT_LT(relative_l2(post_out, r.post), 1e-5);
    EXPECT_LT(relative_l2(comb_out, r.comb), 1e-5);
  }
}

// Independent reference: never feed a kernel output into the oracle.
TEST(FusedMhcDelayedPreTest,
     MatchesIndependentReferenceAtSchedulingBoundaries) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  const std::vector<int64_t> tokens_to_test = {
      0,   2,   7,   8,   9,   15,  16,  31,  32,  64,   87,   88,  89,  90,
      255, 256, 312, 313, 320, 321, 360, 361, 364, 365,  368,  369, 384, 385,
      416, 417, 448, 449, 464, 465, 511, 512, 513, 2047, 2048, 2049};
  for (const int64_t tokens : tokens_to_test) {
    SCOPED_TRACE(tokens);
    const MhcCase c = make_case(tokens, device);
    const auto [norm, residual, pre, post, comb] =
        kernel::mlu::fused_mhc_delayed_pre(c.x,
                                           c.residual_in,
                                           c.hc_fn,
                                           c.gamma,
                                           c.post_in,
                                           c.comb_in,
                                           c.pre_in,
                                           c.hc_scale,
                                           c.hc_base,
                                           kSinkhornIter,
                                           kHcEps,
                                           kNormEps);
    EXPECT_EQ(norm.sizes(), c.x.sizes());
    EXPECT_EQ(residual.sizes(), c.residual_in.sizes());
    EXPECT_EQ(pre.sizes(), c.pre_in.sizes());
    EXPECT_EQ(post.sizes(), c.post_in.sizes());
    EXPECT_EQ(comb.sizes(), c.comb_in.sizes());
    if (tokens == 0) {
      continue;
    }
    const MhcReference r = reference_mhc(c);
    EXPECT_LT(relative_l2(residual, r.y), 3e-3);
    EXPECT_LT(relative_l2(norm, r.norm), 3e-3);
    EXPECT_LT(relative_l2(pre, r.pre), 3e-3);
    EXPECT_LT(relative_l2(post, r.post), 3e-3);
    EXPECT_LT(relative_l2(comb, r.comb), 3e-3);
  }
}

TEST(FusedMhcDelayedPreTest, ZeroInputsProduceZeroResidualAndNorm) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  for (const int64_t tokens : {1, 89, 321, 640}) {
    SCOPED_TRACE(tokens);
    MhcCase c = make_case(tokens, device);
    c.x.zero_();
    c.residual_in.zero_();
    const auto [norm, residual, pre, post, comb] =
        kernel::mlu::fused_mhc_delayed_pre(c.x,
                                           c.residual_in,
                                           c.hc_fn,
                                           c.gamma,
                                           c.post_in,
                                           c.comb_in,
                                           c.pre_in,
                                           c.hc_scale,
                                           c.hc_base,
                                           kSinkhornIter,
                                           kHcEps,
                                           kNormEps);
    EXPECT_EQ(norm.count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(residual.count_nonzero().item<int64_t>(), 0);
    EXPECT_TRUE(torch::isfinite(pre).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(post).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(comb).all().item<bool>());
  }
}

TEST(FusedMhcDelayedPreTest, AcceptsMisalignedTensors) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  // Large enough that aligned tensors would go to the prefill kernel.
  MhcCase c = make_case(/*tokens=*/700, device);
  c.x = misaligned_copy(c.x);
  c.residual_in = misaligned_copy(c.residual_in);
  c.post_in = misaligned_copy(c.post_in);
  ASSERT_NE(reinterpret_cast<uintptr_t>(c.x.data_ptr()) % 64, 0);
  const auto [norm_out, residual_out, next_pre, post_out, comb_out] =
      kernel::mlu::fused_mhc_delayed_pre(c.x,
                                         c.residual_in,
                                         c.hc_fn,
                                         c.gamma,
                                         c.post_in,
                                         c.comb_in,
                                         c.pre_in,
                                         c.hc_scale,
                                         c.hc_base,
                                         kSinkhornIter,
                                         kHcEps,
                                         kNormEps);
  const MhcReference r = reference_mhc(c, residual_out);
  const double residual_mismatch = residual_out.ne(r.y.to(torch::kBFloat16))
                                       .to(torch::kFloat64)
                                       .mean()
                                       .item<double>();
  EXPECT_LT(residual_mismatch, 1e-3);
  EXPECT_LT(relative_l2(norm_out, r.norm), 3e-3);
  EXPECT_LT(relative_l2(next_pre, r.pre), 1e-5);
  EXPECT_LT(relative_l2(post_out, r.post), 1e-5);
  EXPECT_LT(relative_l2(comb_out, r.comb), 1e-5);
}

// Cross-queue ordering bugs show up as run-to-run differences. Every group
// size, the multi-tile pipelines (>= 3 tiles per round, P = 8 and P = 4) and
// the prefill handoff must give bitwise identical results on repeated
// launches.
TEST(FusedMhcDelayedPreTest, RepeatedLaunchesAreBitwiseIdentical) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  for (const int64_t tokens : {4, 40, 88, 128, 255, 640}) {
    SCOPED_TRACE(tokens);
    const MhcCase c = make_case(tokens, device);
    auto run = [&]() {
      return kernel::mlu::fused_mhc_delayed_pre(c.x,
                                                c.residual_in,
                                                c.hc_fn,
                                                c.gamma,
                                                c.post_in,
                                                c.comb_in,
                                                c.pre_in,
                                                c.hc_scale,
                                                c.hc_base,
                                                kSinkhornIter,
                                                kHcEps,
                                                kNormEps);
    };
    const auto first = run();
    for (int32_t repeat = 0; repeat < 16; ++repeat) {
      const auto again = run();
      ASSERT_TRUE(torch::equal(std::get<0>(again), std::get<0>(first)));
      ASSERT_TRUE(torch::equal(std::get<1>(again), std::get<1>(first)));
      ASSERT_TRUE(torch::equal(std::get<2>(again), std::get<2>(first)));
      ASSERT_TRUE(torch::equal(std::get<3>(again), std::get<3>(first)));
      ASSERT_TRUE(torch::equal(std::get<4>(again), std::get<4>(first)));
    }
  }
}

TEST(FusedMhcDelayedPreTest, CombinesWithPreInNotTheNewPre) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  const MhcCase c = make_case(/*tokens=*/33, device);
  const auto [norm_out, residual_out, next_pre, post_out, comb_out] =
      kernel::mlu::fused_mhc_delayed_pre(c.x,
                                         c.residual_in,
                                         c.hc_fn,
                                         c.gamma,
                                         c.post_in,
                                         c.comb_in,
                                         c.pre_in,
                                         c.hc_scale,
                                         c.hc_base,
                                         kSinkhornIter,
                                         kHcEps,
                                         kNormEps);
  // Normalizing with the newly computed pre must give a different result.
  MhcCase with_new_pre = c;
  with_new_pre.pre_in = next_pre;
  const MhcReference r_new = reference_mhc(with_new_pre, residual_out);
  EXPECT_GT(relative_l2(norm_out, r_new.norm), 1e-2);
}

TEST(FusedMhcDelayedPreTest, AcceptsZeroTokens) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  const MhcCase c = make_case(/*tokens=*/0, device);
  const auto [norm_out, residual_out, next_pre, post_out, comb_out] =
      kernel::mlu::fused_mhc_delayed_pre(c.x,
                                         c.residual_in,
                                         c.hc_fn,
                                         c.gamma,
                                         c.post_in,
                                         c.comb_in,
                                         c.pre_in,
                                         c.hc_scale,
                                         c.hc_base,
                                         kSinkhornIter,
                                         kHcEps,
                                         kNormEps);
  EXPECT_EQ(norm_out.size(0), 0);
  EXPECT_EQ(residual_out.size(0), 0);
  EXPECT_EQ(next_pre.size(0), 0);
}

}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TEST(FusedMhcDelayedPreTest, GraphReplayReadsUpdatedInputsOnNonDefaultStream) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  for (const int64_t tokens : {1, 321, 640}) {
    SCOPED_TRACE(tokens);
    MhcCase c = make_case(tokens, device);
    auto call = [&]() {
      return kernel::mlu::fused_mhc_delayed_pre(c.x,
                                                c.residual_in,
                                                c.hc_fn,
                                                c.gamma,
                                                c.post_in,
                                                c.comb_in,
                                                c.pre_in,
                                                c.hc_scale,
                                                c.hc_base,
                                                kSinkhornIter,
                                                kHcEps,
                                                kNormEps);
    };
    auto actual = call();
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      actual = call();
      graph.capture_end();
    }
    c.x.mul_(0.5);
    c.residual_in.mul_(0.75);
    c.pre_in.fill_(0.25);
    c.post_in.mul_(0.8);
    c.comb_in.mul_(0.9);
    c.hc_fn.mul_(0.9);
    c.hc_scale.mul_(0.8);
    c.hc_base.add_(0.1);
    c.gamma.mul_(0.5);
    torch_mlu::synchronize();
    graph.replay();
    torch_mlu::synchronize();
    const auto expected = call();
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(std::get<0>(actual), std::get<0>(expected)));
    EXPECT_TRUE(torch::equal(std::get<1>(actual), std::get<1>(expected)));
    EXPECT_TRUE(torch::equal(std::get<2>(actual), std::get<2>(expected)));
    EXPECT_TRUE(torch::equal(std::get<3>(actual), std::get<3>(expected)));
    EXPECT_TRUE(torch::equal(std::get<4>(actual), std::get<4>(expected)));
  }
}
}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TEST(FusedMhcDelayedPreDeathTest, RejectsInvalidInputContracts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const int32_t kind : {0, 1, 2, 3, 4, 5, 6}) {
    EXPECT_DEATH(
        {
          const torch::Device device(torch::kPrivateUse1, 0);
          torch::DeviceGuard guard(device);
          MhcCase c = make_case(1, device);
          int64_t iterations = kSinkhornIter;
          double hc_eps = kHcEps;
          double norm_eps = kNormEps;
          if (kind == 0) {
            c.x = c.x.to(torch::kFloat32);
          } else if (kind == 1) {
            c.x = c.x.slice(1, 0, kDim - 1).contiguous();
          } else if (kind == 2) {
            c.x = torch::empty({1, kDim * 2}, c.x.options())
                      .slice(1, 0, kDim * 2, 2);
          } else if (kind == 3) {
            iterations = 0;
          } else if (kind == 4) {
            hc_eps = -1e-6;
          } else if (kind == 5) {
            norm_eps = 0.0;
          } else {
            norm_eps = std::nan("");
          }
          kernel::mlu::fused_mhc_delayed_pre(c.x,
                                             c.residual_in,
                                             c.hc_fn,
                                             c.gamma,
                                             c.post_in,
                                             c.comb_in,
                                             c.pre_in,
                                             c.hc_scale,
                                             c.hc_base,
                                             iterations,
                                             hc_eps,
                                             norm_eps);
        },
        "Check failed");
  }
}
}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TEST(FusedMhcDelayedPreTest, PreservesInputsAndHandlesEveryMisalignedTensor) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  for (const int32_t index : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) {
    SCOPED_TRACE(index);
    MhcCase c = make_case(640, device);
    std::vector<torch::Tensor*> inputs = {&c.x,
                                          &c.residual_in,
                                          &c.post_in,
                                          &c.comb_in,
                                          &c.pre_in,
                                          &c.hc_fn,
                                          &c.hc_scale,
                                          &c.hc_base,
                                          &c.gamma};
    std::vector<torch::Tensor> outputs = {torch::empty_like(c.x),
                                          torch::empty_like(c.residual_in),
                                          torch::empty_like(c.pre_in),
                                          torch::empty_like(c.post_in),
                                          torch::empty_like(c.comb_in)};
    if (index < 9) {
      *inputs[index] = misaligned_copy(*inputs[index]);
    } else {
      outputs[index - 9] = misaligned_copy(outputs[index - 9]);
    }
    std::vector<torch::Tensor> saved;
    saved.reserve(inputs.size());
    for (const auto* input : inputs) {
      saved.emplace_back(input->clone());
    }
    kernel::mlu::bangc::fused_mhc_decode(c.x,
                                         c.residual_in,
                                         c.post_in,
                                         c.comb_in,
                                         c.pre_in,
                                         c.hc_fn,
                                         c.hc_scale,
                                         c.hc_base,
                                         c.gamma,
                                         kSinkhornIter,
                                         kHcEps,
                                         kNormEps,
                                         outputs[0],
                                         outputs[1],
                                         outputs[2],
                                         outputs[3],
                                         outputs[4]);
    const MhcReference expected = reference_mhc(c);
    EXPECT_LT(relative_l2(outputs[0], expected.norm), 3e-3);
    EXPECT_LT(relative_l2(outputs[1], expected.y), 3e-3);
    EXPECT_LT(relative_l2(outputs[2], expected.pre), 3e-3);
    EXPECT_LT(relative_l2(outputs[3], expected.post), 3e-3);
    EXPECT_LT(relative_l2(outputs[4], expected.comb), 3e-3);
    for (size_t i = 0; i < inputs.size(); ++i) {
      EXPECT_TRUE(torch::equal(*inputs[i], saved[i]));
    }
  }
}
}  // namespace
}  // namespace xllm
