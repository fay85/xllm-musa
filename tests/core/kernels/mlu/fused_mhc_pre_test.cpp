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

#include "kernels/mlu/bangc/include/fused_mhc_pre.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <vector>

namespace xllm {
namespace {
constexpr int64_t kDim = 5120;
constexpr double kEps = 1e-6;
using Inputs = std::array<torch::Tensor, 6>;
using Outputs =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

Inputs make_inputs(int64_t tokens) {
  torch::manual_seed(42 + tokens);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  return {torch::randn({tokens, 4, kDim}).to(bf16).to(device),
          torch::randn({tokens, 4}).to(device),
          (torch::randn({24, 4 * kDim}) * 0.02).to(bf16).to(device),
          torch::tensor({0.1f, 0.2f, 0.15f}).to(device),
          torch::randn({24}).to(device),
          torch::randn({kDim}).to(bf16).to(device)};
}
Outputs run(const Inputs& in,
            int64_t iterations,
            double hc_eps,
            double norm_eps) {
  return kernel::mlu::bangc::fused_mhc_pre(
      in[0], in[1], in[2], in[3], in[4], in[5], iterations, hc_eps, norm_eps);
}
Outputs run(const Inputs& in, int64_t iterations = 20, double eps = kEps) {
  return run(in, iterations, eps, eps);
}

// Independent CPU formula of the reference model: FP32 projection, and a BF16
// boundary only after the external weighted reduction before RMSNorm. No
// candidate outputs feed it.
Outputs reference(const Inputs& in,
                  int64_t iterations,
                  double hc_eps,
                  double norm_eps) {
  using torch::indexing::Slice;
  const auto residual = in[0].cpu().to(torch::kFloat32);
  const auto pre_in = in[1].cpu();
  const auto fn = in[2].cpu().to(torch::kFloat32);
  const auto scale = in[3].cpu();
  const auto base = in[4].cpu();
  const auto gamma = in[5].cpu().to(torch::kFloat32);
  const int64_t tokens = residual.size(0);
  const auto rms =
      torch::rsqrt(residual.flatten(1).square().mean(-1, true) + norm_eps);
  const auto mixes = torch::mm(residual.flatten(1), fn.transpose(0, 1)) * rms;
  const auto pre =
      torch::sigmoid(mixes.index({Slice(), Slice(0, 4)}) * scale[0] +
                     base.index({Slice(0, 4)})) +
      hc_eps;
  const auto post =
      2.0 * torch::sigmoid(mixes.index({Slice(), Slice(4, 8)}) * scale[1] +
                           base.index({Slice(4, 8)}));
  auto comb =
      torch::softmax(
          mixes.index({Slice(), Slice(8)}).view({tokens, 4, 4}) * scale[2] +
              base.index({Slice(8)}).view({4, 4}),
          -1) +
      hc_eps;
  comb = comb / (comb.sum(-2, true) + hc_eps);
  for (int64_t i = 1; i < iterations; ++i) {
    comb = comb / (comb.sum(-1, true) + hc_eps);
    comb = comb / (comb.sum(-2, true) + hc_eps);
  }
  const auto z = (residual * pre_in.unsqueeze(-1))
                     .sum(1)
                     .to(torch::kBFloat16)
                     .to(torch::kFloat32);
  const auto norm =
      (z * torch::rsqrt(z.square().mean(-1, true) + norm_eps) * gamma)
          .to(torch::kBFloat16);
  return {norm, pre, post, comb};
}
Outputs reference(const Inputs& in,
                  int64_t iterations = 20,
                  double eps = kEps) {
  return reference(in, iterations, eps, eps);
}
void expect_close(const Outputs& actual, const Outputs& expected) {
  const std::array<torch::Tensor, 4> a = {std::get<0>(actual),
                                          std::get<1>(actual),
                                          std::get<2>(actual),
                                          std::get<3>(actual)};
  const std::array<torch::Tensor, 4> e = {std::get<0>(expected),
                                          std::get<1>(expected),
                                          std::get<2>(expected),
                                          std::get<3>(expected)};
  for (size_t i = 0; i < a.size(); ++i) {
    SCOPED_TRACE(i);
    const auto left = a[i].cpu().to(torch::kFloat64);
    const auto right = e[i].cpu().to(torch::kFloat64);
    ASSERT_EQ(left.sizes(), right.sizes());
    EXPECT_TRUE(torch::isfinite(left).all().item<bool>());
    const double relative_l2 =
        ((left - right).square().sum() / right.square().sum().clamp_min(1e-30))
            .sqrt()
            .item<double>();
    EXPECT_LT(relative_l2, 4e-3);
    EXPECT_TRUE(torch::allclose(left, right, 2e-2, 2e-2));
  }
}

TEST(FusedMhcPreTest, MatchesIndependentReferenceAcrossTailsAndRounds) {
  torch::InferenceMode inference;
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  for (const int64_t tokens :
       {1,  3,  7,   8,   9,   17,  31,  32,  33,  63,  64,   65,   87,
        88, 89, 127, 128, 129, 255, 256, 257, 321, 640, 1025, 4096, 4097}) {
    SCOPED_TRACE(tokens);
    const auto in = make_inputs(tokens);
    expect_close(run(in), reference(in));
  }
}
// {tokens, seed, token}: inputs from torch::manual_seed(seed) whose FP32
// projection of mix row 5 for `token` lies next to a BF16 rounding midpoint,
// where any BF16 rounding of the projection moves post by about 1e-3.
void expect_post_matches_fp64_projection(
    std::initializer_list<std::array<int64_t, 3>> samples) {
  torch::InferenceMode inference;
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  for (const auto& sample : samples) {
    const int64_t tokens = sample[0];
    const int64_t seed = sample[1];
    const int64_t token = sample[2];
    SCOPED_TRACE(tokens);
    torch::manual_seed(seed);
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
    const Inputs in = {
        torch::randn({tokens, 4, kDim}).to(bf16).to(device),
        torch::randn({tokens, 4}).to(device),
        (torch::randn({24, 4 * kDim}) * 0.02).to(bf16).to(device),
        torch::tensor({0.1f, 0.2f, 0.15f}).to(device),
        torch::randn({24}).to(device),
        torch::randn({kDim}).to(bf16).to(device)};
    // The projection feeds rms/scale/base unrounded, as in the reference
    // model.
    const auto residual = in[0][token].cpu().to(torch::kFloat64).flatten();
    const auto weight = in[2][5].cpu().to(torch::kFloat64);
    const double projection = torch::dot(residual, weight).item<double>();
    const double rms =
        1.0 / std::sqrt(residual.square().mean().item<double>() + 1e-20);
    const double scale = in[3].cpu()[1].item<double>();
    const double bias = in[4].cpu()[5].item<double>();
    const double expected =
        2.0 / (1.0 + std::exp(-(projection * rms * scale + bias)));
    const auto actual = run(in, 20, 1e-6, 1e-20);
    EXPECT_NEAR(
        std::get<2>(actual).cpu()[token][1].item<double>(), expected, 2e-6);
  }
}
TEST(FusedMhcPreTest, ProjectionNearBf16MidpointStaysFp32) {
  // 4 runs P = 16, 24 / 88 P = 8, 128 P = 4 and 512 the prefill kernel
  // (aligned, T >= 257).
  expect_post_matches_fp64_projection({std::array<int64_t, 3>{4, 427, 2},
                                       std::array<int64_t, 3>{24, 42, 11},
                                       std::array<int64_t, 3>{88, 28, 64},
                                       std::array<int64_t, 3>{128, 96, 57},
                                       std::array<int64_t, 3>{512, 0, 18}});
}

// Cross-queue ordering bugs show up as run-to-run differences. Every group
// size and the multi-tile pipelines (>= 3 tiles per round) must give bitwise
// identical results on repeated launches.
TEST(FusedMhcPreTest, RepeatedLaunchesAreBitwiseIdentical) {
  torch::InferenceMode inference;
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  for (const int64_t tokens : {4, 40, 88, 128, 255, 512}) {
    SCOPED_TRACE(tokens);
    const auto in = make_inputs(tokens);
    const auto first = run(in);
    for (int32_t repeat = 0; repeat < 16; ++repeat) {
      const auto again = run(in);
      ASSERT_TRUE(torch::equal(std::get<0>(again), std::get<0>(first)));
      ASSERT_TRUE(torch::equal(std::get<1>(again), std::get<1>(first)));
      ASSERT_TRUE(torch::equal(std::get<2>(again), std::get<2>(first)));
      ASSERT_TRUE(torch::equal(std::get<3>(again), std::get<3>(first)));
    }
  }
}

TEST(FusedMhcPreTest, ExternalMixOnlyControlsNormAndInputsArePreserved) {
  torch::InferenceMode inference;
  const auto in = make_inputs(33);
  auto changed = in;
  changed[1] = torch::zeros_like(in[1]);
  const auto before = run(in);
  const auto zero = run(changed);
  EXPECT_EQ(std::get<0>(zero).abs().max().item<float>(), 0);
  EXPECT_TRUE(torch::equal(std::get<1>(before), std::get<1>(zero)));
  EXPECT_TRUE(torch::equal(std::get<2>(before), std::get<2>(zero)));
  EXPECT_TRUE(torch::equal(std::get<3>(before), std::get<3>(zero)));
  Inputs saved;
  for (size_t i = 0; i < in.size(); ++i) {
    saved[i] = in[i].clone();
  }
  changed[1] = torch::zeros_like(in[1]);
  changed[1].select(1, 0).fill_(1);
  expect_close(run(changed), reference(changed));
  run(in);
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_TRUE(torch::equal(in[i], saved[i]));
  }
}
TEST(FusedMhcPreTest, SupportsNaturalAlignmentAndEpsilonScales) {
  torch::InferenceMode inference;
  for (const int64_t tokens : {9, 257, 640}) {
    SCOPED_TRACE(tokens);
    for (const double input_scale : {0.0, 1e-4, 1.0, 16.0}) {
      SCOPED_TRACE(input_scale);
      auto in = make_inputs(tokens);
      in[0].mul_(input_scale);
      auto offset = in;
      // Shift all six tensors by one element. Large token counts must then
      // retain the general d-split path, while in uses aligned prefill.
      for (auto& tensor : offset) {
        auto storage = torch::empty({tensor.numel() + 1}, tensor.options());
        auto view = storage.narrow(0, 1, tensor.numel()).view(tensor.sizes());
        view.copy_(tensor);
        tensor = view;
      }
      for (const int64_t iterations : {1, 4, 20}) {
        SCOPED_TRACE(iterations);
        const auto expected = reference(in, iterations);
        expect_close(run(in, iterations), expected);
        expect_close(run(offset, iterations), expected);
      }
    }
  }
}
TEST(FusedMhcPreTest, KeepsHcAndNormEpsilonIndependentForNearZeroResiduals) {
  torch::InferenceMode inference;
  constexpr double kNormEps = 1e-20;
  for (const int64_t tokens : {1, 9, 257, 640}) {
    SCOPED_TRACE(tokens);
    for (const double input_scale : {1e-12, 1e-8}) {
      SCOPED_TRACE(input_scale);
      auto in = make_inputs(tokens);
      in[0].mul_(input_scale);
      expect_close(run(in, /*iterations=*/20, kEps, kNormEps),
                   reference(in, /*iterations=*/20, kEps, kNormEps));
    }
  }
}
TEST(FusedMhcPreTest, MatchesDecodeAndFlattenedVerifyLocalTokenCounts) {
  torch::InferenceMode inference;
  // Decode batches 8/16/32 and q=3/5 flattened verification rows.
  for (const int64_t tokens : {8, 16, 32, 24, 40, 48, 80, 96, 160}) {
    SCOPED_TRACE(tokens);
    const auto in = make_inputs(tokens);
    expect_close(run(in), reference(in));
  }
}
TEST(FusedMhcPreTest, RestoresUnevenAndEmptyLocalRowShards) {
  torch::InferenceMode inference;
  constexpr int64_t kTokens = 321;
  const auto in = make_inputs(kTokens);
  // Model independent token-row ownership, including an empty rank. The
  // larger local shard exercises prefill while smaller shards use d-split.
  const std::array<int64_t, 5> shard_sizes = {0, 1, 8, 17, 295};
  const auto host_rows =
      torch::arange(kTokens, torch::kInt64).mul(37).remainder(kTokens);
  const auto rows = host_rows.to(in[0].device());
  const auto inverse_rows = torch::argsort(host_rows).to(in[0].device());
  auto shuffled = in;
  shuffled[0] = in[0].index_select(0, rows);
  shuffled[1] = in[1].index_select(0, rows);
  std::array<std::vector<torch::Tensor>, 4> pieces;
  for (auto& tensors : pieces) {
    tensors.reserve(shard_sizes.size());
  }
  int64_t begin = 0;
  for (const int64_t local_tokens : shard_sizes) {
    SCOPED_TRACE(local_tokens);
    auto local = shuffled;
    local[0] = shuffled[0].narrow(0, begin, local_tokens).clone();
    local[1] = shuffled[1].narrow(0, begin, local_tokens).clone();
    const auto [norm, pre, post, comb] = run(local);
    pieces[0].emplace_back(norm);
    pieces[1].emplace_back(pre);
    pieces[2].emplace_back(post);
    pieces[3].emplace_back(comb);
    begin += local_tokens;
  }
  ASSERT_EQ(begin, kTokens);
  const Outputs restored = {
      torch::cat(pieces[0], /*dim=*/0).index_select(0, inverse_rows),
      torch::cat(pieces[1], /*dim=*/0).index_select(0, inverse_rows),
      torch::cat(pieces[2], /*dim=*/0).index_select(0, inverse_rows),
      torch::cat(pieces[3], /*dim=*/0).index_select(0, inverse_rows)};
  expect_close(restored, reference(in));
  expect_close(restored, run(in));
}
TEST(FusedMhcPreTest, AcceptsEmptyTokens) {
  const auto in = make_inputs(0);
  const auto [norm, pre, post, comb] = run(in);
  EXPECT_EQ(norm.sizes(), (torch::IntArrayRef{0, kDim}));
  EXPECT_EQ(pre.sizes(), (torch::IntArrayRef{0, 4}));
  EXPECT_EQ(post.sizes(), (torch::IntArrayRef{0, 4}));
  EXPECT_EQ(comb.sizes(), (torch::IntArrayRef{0, 4, 4}));
}
TEST(FusedMhcPreTest, GraphReplayReadsAllUpdatedInputsOnNonDefaultStream) {
  torch::InferenceMode inference;
  for (const int64_t tokens : {1, 33, 89, 321, 640}) {
    SCOPED_TRACE(tokens);
    auto in = make_inputs(tokens);
    auto actual = run(in);
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      actual = run(in);
      graph.capture_end();
    }
    for (int32_t step = 0; step < 3; ++step) {
      in[0].mul_(0.75);
      in[1].add_(0.1);
      in[2].mul_(0.9);
      in[3].mul_(0.8);
      in[4].add_(0.2);
      in[5].mul_(0.5);
      torch_mlu::synchronize();
      graph.replay();
      torch_mlu::synchronize();
      expect_close(actual, reference(in));
      const auto eager = run(in);
      EXPECT_TRUE(torch::equal(std::get<0>(actual), std::get<0>(eager)));
      EXPECT_TRUE(torch::equal(std::get<1>(actual), std::get<1>(eager)));
      EXPECT_TRUE(torch::equal(std::get<2>(actual), std::get<2>(eager)));
      EXPECT_TRUE(torch::equal(std::get<3>(actual), std::get<3>(eager)));
    }
  }
}
TEST(FusedMhcPreDeathTest, RejectsInvalidContracts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const int32_t kind : {0, 1, 2, 3, 4, 5}) {
    SCOPED_TRACE(kind);
    EXPECT_DEATH(
        {
          auto in = make_inputs(1);
          int64_t iterations = 20;
          if (kind == 0) {
            in[0] = in[0].to(torch::kFloat32);
          }
          if (kind == 1) {
            in[1] = in[1].to(torch::kBFloat16);
          }
          if (kind == 2) {
            in[0] = in[0].slice(2, 0, kDim - 1).contiguous();
          }
          if (kind == 3) {
            in[2] = in[2].transpose(0, 1);
          }
          if (kind == 4) {
            in[5] = in[5].cpu();
          }
          if (kind == 5) {
            iterations = 0;
          }
          run(in, iterations);
        },
        "Check failed");
  }
}
}  // namespace
}  // namespace xllm
