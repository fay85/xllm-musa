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

#include "core/kernels/musa/mhc_ops.h"

#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <cmath>
#include <vector>

#include "core/layers/musa/hyper_connection.h"

namespace xllm::kernel::musa {
namespace {

struct MHCReference final {
  torch::Tensor layer_input;
  torch::Tensor post;
  torch::Tensor comb;
};

MHCReference reference_pre(const torch::Tensor& residual,
                           const torch::Tensor& hc_fn,
                           const torch::Tensor& hc_scale,
                           const torch::Tensor& hc_base,
                           double rms_eps,
                           double hc_eps,
                           int64_t sinkhorn_repeat) {
  const int64_t hc_mult = residual.size(-2);
  const int64_t hidden_size = residual.size(-1);
  torch::Tensor residual_flat =
      residual.reshape({-1, hc_mult, hidden_size}).to(torch::kFloat32);
  torch::Tensor x = residual_flat.reshape({-1, hc_mult * hidden_size});
  torch::Tensor mixes =
      torch::matmul(x, hc_fn.to(torch::kFloat32).transpose(0, 1)) *
      torch::rsqrt(x.square().sum(-1, true) /
                       static_cast<double>(hc_mult * hidden_size) +
                   rms_eps);
  torch::Tensor pre = torch::sigmoid(mixes.slice(-1, 0, hc_mult) * hc_scale[0] +
                                     hc_base.slice(0, 0, hc_mult)) +
                      hc_eps;
  torch::Tensor post =
      torch::sigmoid(mixes.slice(-1, hc_mult, 2 * hc_mult) * hc_scale[1] +
                     hc_base.slice(0, hc_mult, 2 * hc_mult)) *
      2.0;
  torch::Tensor comb =
      mixes.slice(-1, 2 * hc_mult).reshape({-1, hc_mult, hc_mult}) *
          hc_scale[2] +
      hc_base.slice(0, 2 * hc_mult).reshape({hc_mult, hc_mult});
  comb = torch::softmax(comb, -1) + hc_eps;
  comb = comb / (comb.sum(-2, true) + hc_eps);
  for (int64_t iteration = 1; iteration < sinkhorn_repeat; ++iteration) {
    comb = comb / (comb.sum(-1, true) + hc_eps);
    comb = comb / (comb.sum(-2, true) + hc_eps);
  }
  std::vector<int64_t> outer(residual.sizes().begin(),
                             residual.sizes().end() - 2);
  std::vector<int64_t> layer_shape = outer;
  layer_shape.emplace_back(hidden_size);
  std::vector<int64_t> post_shape = outer;
  post_shape.emplace_back(hc_mult);
  post_shape.emplace_back(1);
  std::vector<int64_t> comb_shape = outer;
  comb_shape.emplace_back(hc_mult);
  comb_shape.emplace_back(hc_mult);
  torch::Tensor layer_input =
      (pre.unsqueeze(-1) * residual_flat).sum(1).to(torch::kBFloat16);
  return {layer_input.reshape(layer_shape),
          post.reshape(post_shape),
          comb.reshape(comb_shape)};
}

void expect_close(const torch::Tensor& actual,
                  const torch::Tensor& expected,
                  double atol,
                  double rtol) {
  torch::Tensor actual_float = actual.to(torch::kFloat32).cpu();
  torch::Tensor expected_float = expected.to(torch::kFloat32).cpu();
  const double max_error =
      (actual_float - expected_float).abs().max().item<double>();
  const double limit = atol + rtol * expected_float.abs().max().item<double>();
  EXPECT_LE(max_error, limit) << "max error " << max_error;
}

class MHCOperationsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int32_t device_count = 0;
    if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
      GTEST_SKIP() << "MUSA device unavailable";
    }
    at::detail::getMUSAHooks().initMUSA();
    device_ = torch::Device(torch::kPrivateUse1, 0);
  }

  torch::Tensor seeded(const std::string& name,
                       torch::IntArrayRef sizes,
                       torch::ScalarType dtype) const {
    torch::manual_seed(
        static_cast<uint64_t>(std::hash<std::string>{}(name) % 1000003));
    return torch::randn(sizes,
                        torch::TensorOptions().dtype(dtype).device(device_));
  }

  torch::Device device_{torch::kCPU};
};

TEST_F(MHCOperationsTest, PrePostMatchReferenceAcrossTokenCounts) {
  const std::vector<int64_t> token_counts = {1, 2, 16, 64, 129};
  constexpr int64_t kHcMult = 4;
  constexpr int64_t kHidden = 4096;
  constexpr int64_t kRepeat = 20;
  for (int64_t tokens : token_counts) {
    torch::Tensor residual = seeded("residual-" + std::to_string(tokens),
                                    {tokens, kHcMult, kHidden},
                                    torch::kBFloat16);
    torch::Tensor fn_bf16 = seeded("fn-" + std::to_string(tokens),
                                   {24, kHcMult * kHidden},
                                   torch::kBFloat16) *
                            0.01;
    torch::Tensor fn = fn_bf16.to(torch::kFloat32);
    torch::Tensor scale = seeded("scale", {3}, torch::kFloat32).mul(0.1);
    torch::Tensor base = seeded("base", {24}, torch::kFloat32).mul(0.1);

    MHCPreOutput actual =
        mhc_pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, kRepeat);
    MHCReference expected =
        reference_pre(residual, fn, scale, base, 1e-6, 1e-6, kRepeat);
    expect_close(actual.layer_input, expected.layer_input, 2e-2, 2e-2);
    expect_close(actual.post_mix, expected.post, 2e-3, 2e-3);
    expect_close(actual.comb_mix, expected.comb, 2e-3, 2e-3);

    torch::Tensor x = seeded(
        "x-" + std::to_string(tokens), {tokens, kHidden}, torch::kBFloat16);
    torch::Tensor output =
        mhc_post(x, residual, actual.post_mix, actual.comb_mix);
    torch::Tensor expected_output =
        (torch::einsum("tij,tih->tjh",
                       actual.comb_mix.to(torch::kFloat32),
                       residual.to(torch::kFloat32)) +
         actual.post_mix.to(torch::kFloat32) *
             x.unsqueeze(-2).to(torch::kFloat32))
            .to(torch::kBFloat16);
    expect_close(output, expected_output, 2e-2, 2e-2);
    EXPECT_TRUE(torch::allclose(mhc_contract_mean(output).to(torch::kFloat32),
                                output.to(torch::kFloat32).mean(-2),
                                1e-3,
                                1e-3));
  }
}

TEST_F(MHCOperationsTest, LayerLoadsBFloat16FnAsFloat32) {
  constexpr int64_t kHcMult = 4;
  constexpr int64_t kHidden = 1024;
  layer::MHCPre pre(
      kHcMult,
      kHidden,
      /*sinkhorn_iters=*/4,
      /*hc_eps=*/1e-6,
      /*norm_eps=*/1e-6,
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
  torch::Tensor fn =
      seeded("layer-fn", {24, kHcMult * kHidden}, torch::kBFloat16);
  torch::Tensor base = seeded("layer-base", {24}, torch::kFloat32);
  torch::Tensor scale = seeded("layer-scale", {3}, torch::kFloat32);
  pre->load_state_dict(
      StateDict({{"hc_fn", fn}, {"hc_base", base}, {"hc_scale", scale}}));
  torch::Tensor loaded = pre->named_parameters()["hc_fn"];
  EXPECT_EQ(loaded.scalar_type(), torch::kFloat32);
  expect_close(loaded, fn.to(torch::kFloat32), 0.0, 0.0);
}

TEST_F(MHCOperationsTest, PendingChainComposesPostAndNextPre) {
  constexpr int64_t kTokens = 3;
  constexpr int64_t kHcMult = 4;
  constexpr int64_t kHidden = 1024;
  torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  layer::MHCPre first(kHcMult, kHidden, 4, 1e-6, 1e-6, options);
  layer::MHCPre second(kHcMult, kHidden, 4, 1e-6, 1e-6, options);
  auto weights = [&](const std::string& name) {
    return StateDict(
        {{"hc_fn",
          seeded(name + "-fn", {24, kHcMult * kHidden}, torch::kBFloat16) *
              0.01},
         {"hc_base", seeded(name + "-base", {24}, torch::kFloat32) * 0.1},
         {"hc_scale", seeded(name + "-scale", {3}, torch::kFloat32) * 0.1}});
  };
  first->load_state_dict(weights("first"));
  second->load_state_dict(weights("second"));
  torch::Tensor residual =
      seeded("chain-residual", {kTokens, kHcMult, kHidden}, torch::kBFloat16);
  torch::Tensor x = seeded("chain-x", {kTokens, kHidden}, torch::kBFloat16);
  torch::Tensor norm =
      seeded("chain-norm", {kHidden}, torch::kBFloat16).abs() + 0.5;
  layer::MHCPreOutput first_output = first->forward(residual);
  torch::Tensor normalized;
  torch::Tensor next_residual;
  torch::Tensor next_post;
  torch::Tensor next_comb;
  std::tie(normalized, next_residual, next_post, next_comb) =
      second->post_pre_norm(
          x, residual, first_output.post, first_output.comb, norm);
  torch::Tensor expected_residual =
      mhc_post(x, residual, first_output.post, first_output.comb);
  layer::MHCPreOutput expected_pre = second->forward(expected_residual);
  torch::Tensor expected_norm =
      (expected_pre.output.to(torch::kFloat32) *
       torch::rsqrt(
           expected_pre.output.to(torch::kFloat32).square().mean(-1, true) +
           1e-6) *
       norm.to(torch::kFloat32))
          .to(torch::kBFloat16);
  expect_close(next_residual, expected_residual, 2e-2, 2e-2);
  expect_close(normalized, expected_norm, 2e-2, 2e-2);
  expect_close(next_post, expected_pre.post, 2e-3, 2e-3);
  expect_close(next_comb, expected_pre.comb, 2e-3, 2e-3);
  EXPECT_EQ(mhc_expand(x, kHcMult).sizes(), residual.sizes());
}

TEST_F(MHCOperationsTest, GenericMultiplicityMatchesReference) {
  constexpr int64_t kTokens = 5;
  constexpr int64_t kHcMult = 2;
  constexpr int64_t kHidden = 128;
  constexpr int64_t kRepeat = 3;
  const int64_t mix_size = kHcMult * (kHcMult + 2);
  torch::Tensor residual = seeded(
      "generic-residual", {2, kTokens, kHcMult, kHidden}, torch::kBFloat16);
  torch::Tensor fn =
      seeded("generic-fn", {mix_size, kHcMult * kHidden}, torch::kBFloat16)
          .to(torch::kFloat32) *
      0.02f;
  torch::Tensor scale = seeded("generic-scale", {3}, torch::kFloat32) * 0.1;
  torch::Tensor base =
      seeded("generic-base", {mix_size}, torch::kFloat32) * 0.1;
  MHCPreOutput actual =
      mhc_pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, kRepeat);
  MHCReference expected =
      reference_pre(residual, fn, scale, base, 1e-6, 1e-6, kRepeat);
  expect_close(actual.layer_input, expected.layer_input, 2e-2, 2e-2);
  expect_close(actual.post_mix, expected.post, 2e-3, 2e-3);
  expect_close(actual.comb_mix, expected.comb, 2e-3, 2e-3);

  torch::Tensor x =
      seeded("generic-x", {2, kTokens, kHidden}, torch::kBFloat16);
  torch::Tensor output =
      mhc_post(x, residual, actual.post_mix, actual.comb_mix);
  torch::Tensor expected_output =
      (torch::einsum("...ij,...ih->...jh",
                     actual.comb_mix.to(torch::kFloat32),
                     residual.to(torch::kFloat32)) +
       actual.post_mix.to(torch::kFloat32) *
           x.unsqueeze(-2).to(torch::kFloat32))
          .to(torch::kBFloat16);
  expect_close(output, expected_output, 2e-2, 2e-2);
  expect_close(mhc_contract_mean(output),
               output.to(torch::kFloat32).mean(-2).to(torch::kBFloat16),
               2e-2,
               2e-2);

  torch::Tensor head_fn =
      seeded("generic-head-fn", {kHcMult, kHcMult * kHidden}, torch::kFloat32) *
      0.02f;
  torch::Tensor head_scale = seeded("generic-head-scale", {1}, torch::kFloat32);
  torch::Tensor head_base =
      seeded("generic-head-base", {kHcMult}, torch::kFloat32) * 0.1;
  torch::Tensor head =
      mhc_head(residual, head_fn, head_scale, head_base, 1e-6, 1e-6);
  torch::Tensor flat =
      residual.reshape({-1, kHcMult * kHidden}).to(torch::kFloat32);
  torch::Tensor weights =
      torch::sigmoid(torch::matmul(flat, head_fn.t()) *
                         torch::rsqrt(flat.square().mean(-1, true) + 1e-6) *
                         head_scale +
                     head_base) +
      1e-6;
  torch::Tensor expected_head =
      (weights.unsqueeze(-1) *
       residual.reshape({-1, kHcMult, kHidden}).to(torch::kFloat32))
          .sum(1)
          .to(torch::kBFloat16)
          .reshape({2, kTokens, kHidden});
  expect_close(head, expected_head, 2e-2, 2e-2);
}

}  // namespace
}  // namespace xllm::kernel::musa
