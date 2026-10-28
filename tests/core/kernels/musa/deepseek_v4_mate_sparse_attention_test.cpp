/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace xllm::util {
std::string get_string_env(const std::string& name) {
  const char* value = std::getenv(name.c_str());
  return value == nullptr ? std::string() : std::string(value);
}
int ceil_pow2(int value) {
  if (value <= 1) return 1;
  int result = 1;
  while (result < value) result <<= 1;
  return result;
}
}  // namespace xllm::util

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_graph_policy.h"

namespace xllm::kernel::musa {
namespace {

TEST(DeepseekV4SamplingTest, GreedyPrefixMaskMatchesSequentialOracle) {
  int32_t devices = 0;
  ASSERT_EQ(musaGetDeviceCount(&devices), musaSuccess);
  ASSERT_GT(devices, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  for (const int64_t batch : {1, 4}) {
    for (const int64_t width : {1, 5, 6, 7, 9, 31}) {
      auto target = torch::zeros({batch, width + 1},
          torch::TensorOptions().dtype(torch::kInt64).device(device));
      auto draft = torch::zeros({batch, width}, target.options());
      auto output = torch::empty_like(target);
      musaGraph_t graph = nullptr;
      musaGraphExec_t executable = nullptr;
      stream.synchronize();
      ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal), musaSuccess);
      deepseek_v4_greedy_verify_tokens(draft, target, output);
      ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
      ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
      for (int64_t frame = 0; frame <= width; ++frame) {
        auto th = (torch::arange(batch * (width + 1), torch::kInt64) + frame * 100 + 5000)
                      .view({batch, width + 1}).flip({0}).contiguous();
        auto dh = th.narrow(1, 0, width).contiguous().clone();
        auto expected = th.clone();
        auto* dp = dh.data_ptr<int64_t>();
        auto* ep = expected.data_ptr<int64_t>();
        for (int64_t row = 0; row < batch; ++row) {
          const int64_t first_rejected = frame == width ? width : (frame + row) % width;
          if (first_rejected < width) {
            dp[row * width + first_rejected] += 17;
          }
          for (int64_t index = first_rejected + 1; index <= width; ++index) {
            ep[row * (width + 1) + index] = -1;
          }
        }
        draft.copy_(dh.to(device));
        target.copy_(th.to(device));
        output.fill_(-99);
        ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
        stream.synchronize();
        EXPECT_TRUE(torch::equal(output.cpu(), expected)) << "batch=" << batch
            << " width=" << width << " frame=" << frame;
      }
      ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
      ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
    }
  }
}

TEST(DeepseekV4ProjectionTest, PackedC4ViewsBoundedBf16Roundoff) {
  int32_t devices = 0;
  ASSERT_EQ(musaGetDeviceCount(&devices), musaSuccess);
  ASSERT_GT(devices, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int64_t kHidden = 4096;
  auto expect_close = [](const torch::Tensor& actual, const torch::Tensor& reference,
                         const char* label) {
    auto a = actual.cpu().to(torch::kFloat32);
    auto r = reference.cpu().to(torch::kFloat32);
    const auto difference = (a - r).abs();
    LOG(INFO) << "[JOINT_PROJECTION_ROUNDOFF] " << label
              << " max_abs=" << difference.max().item<float>()
              << " mean_abs=" << difference.mean().item<float>()
              << " differing=" << difference.ne(0).sum().item<int64_t>();
    // GEMM shape may change FP32 accumulation order before the BF16 store.
    // Require BF16-scale relative error, with a small cancellation floor;
    // the original strict-equality failure remains an explicit experiment.
    EXPECT_TRUE(torch::allclose(a, r, /*rtol=*/0.008, /*atol=*/2.0e-5));
  };
  // C128 joint projection was rejected after a numeric state regression;
  // production keeps its original two GEMMs, so this tests the adopted C4 path.
  for (const int64_t ratio : {4}) {
    const int64_t coefficient = ratio == 4 ? 2 : 1;
    const int64_t main_width = coefficient * 512;
    const int64_t index_width = ratio == 4 ? 256 : 0;
    const int64_t packed_width = 2 * (main_width + index_width);
    auto weight = torch::arange(packed_width * kHidden, fp).mul_(0.019f).sin_()
                      .mul_(0.007f).view({packed_width, kHidden}).to(torch::kBFloat16);
    const int64_t workspace_tokens = 16;
    const int64_t projection_rows =
        (workspace_tokens * packed_width + main_width - 1) / main_width;
    auto storage = torch::empty({projection_rows, main_width}, weight.options());
    EXPECT_GE(storage.numel(), workspace_tokens * packed_width);
    auto reference_weight = weight.cpu().to(torch::kFloat64);
    for (const int64_t tokens : {2, 6, 7, 8}) {
      auto input_source = torch::arange(tokens * kHidden, fp).mul_(0.023f).sin_()
                              .view({tokens, kHidden}).to(torch::kBFloat16);
      auto input = input_source.clone();
      auto packed = storage.view({-1}).narrow(0, 0, tokens * packed_width)
                        .view({tokens, packed_width});
      auto state = torch::zeros({coefficient * ratio, main_width}, fp);
      auto scores = torch::zeros_like(state);
      auto ape = torch::zeros({ratio, main_width}, fp);
      auto position = torch::tensor({125}, fp.dtype(torch::kInt64));
      auto compressed = torch::empty({ratio == 4 ? 2 : 1, 512}, weight.options());
      auto execute = [&] {
        torch::mm_out(packed, input, weight.transpose(0, 1));
        return deepseek_v4_compress_decode_sequence_graph(
            state, scores, packed.narrow(1, 0, main_width),
            packed.narrow(1, main_width, main_width), ape, ratio, 512, position, compressed);
      };
      execute();
      stream.synchronize();
      musaGraph_t graph = nullptr;
      musaGraphExec_t executable = nullptr;
      ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal), musaSuccess);
      execute();
      ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
      ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
      for (int64_t frame = 0; frame < 3; ++frame) {
        SCOPED_TRACE(::testing::Message() << "ratio=" << ratio << " tokens=" << tokens
                     << " frame=" << frame);
        input.copy_(input_source.roll({frame}, {0}));
        position.fill_(125 + frame);
        state.zero_();
        scores.zero_();
        auto kv = torch::matmul(input, weight.narrow(0, 0, main_width).transpose(0, 1));
        auto gate = torch::matmul(input, weight.narrow(0, main_width, main_width).transpose(0, 1));
        auto reference_state = state.clone();
        auto reference_scores = scores.clone();
        auto reference_output = torch::empty_like(compressed);
        deepseek_v4_compress_decode_sequence_graph(reference_state, reference_scores, kv, gate,
                                                  ape, ratio, 512, position, reference_output);
        ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
        stream.synchronize();
        auto fp64_reference = torch::matmul(input.cpu().to(torch::kFloat64),
                                            reference_weight.transpose(0, 1)).to(torch::kBFloat16);
        expect_close(packed, fp64_reference, "packed_vs_independent_fp64");
        expect_close(packed.narrow(1, 0, main_width), kv, "main_kv_vs_separate");
        expect_close(packed.narrow(1, main_width, main_width), gate, "main_score_vs_separate");
        expect_close(state, reference_state, "state_from_strided_projection");
        expect_close(scores, reference_scores, "score_state_from_strided_projection");
        expect_close(compressed, reference_output, "compressed_from_strided_projection");
        EXPECT_TRUE(torch::equal(packed.narrow(1, 0, main_width).cpu(), kv.cpu()));
        EXPECT_TRUE(torch::equal(packed.narrow(1, main_width, main_width).cpu(), gate.cpu()));
        EXPECT_TRUE(torch::equal(state.cpu(), reference_state.cpu()));
        EXPECT_TRUE(torch::equal(scores.cpu(), reference_scores.cpu()));
        EXPECT_TRUE(torch::equal(compressed.cpu(), reference_output.cpu()));
        if (index_width > 0) {
          for (const int64_t offset : {2 * main_width, 2 * main_width + index_width}) {
            auto reference = torch::matmul(input, weight.narrow(0, offset, index_width).transpose(0, 1));
            expect_close(packed.narrow(1, offset, index_width), reference, "indexer_vs_separate");
          }
        }
      }
      ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
      ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
    }
  }
}

TEST(DeepseekV4MoEGemvTest, TileShapesPreserveOutputsAndGraphReplay) {
  int32_t devices = 0;
  ASSERT_EQ(musaGetDeviceCount(&devices), musaSuccess);
  ASSERT_GT(devices, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int64_t kTokens = 6;
  constexpr int64_t kExperts = 64;
  constexpr int64_t kHidden = 4096;
  constexpr int64_t kIntermediate = 256;
  constexpr int64_t kTopk = 6;
  auto w13 = torch::arange(kExperts * 2 * kIntermediate * kHidden, fp)
                 .mul_(0.037f).sin_().mul_(4.0f)
                 .reshape({kExperts, 2 * kIntermediate, kHidden})
                 .to(torch::kFloat8_e4m3fn);
  auto w2 = torch::arange(kExperts * kHidden * kIntermediate, fp)
                .mul_(0.029f).cos_().mul_(4.0f)
                .reshape({kExperts, kHidden, kIntermediate})
                .to(torch::kFloat8_e4m3fn);
  auto s13 = torch::full({kExperts, 2 * kIntermediate / 128, kHidden / 128}, 0.03125f, fp);
  auto s2 = torch::full({kExperts, kHidden / 128, kIntermediate / 128}, 0.015625f, fp);
  auto ids_base = torch::arange(kTokens * kTopk, fp.dtype(torch::kInt32))
                      .mul_(13).remainder_(kExperts).view({kTokens, kTopk});
  auto ids = ids_base.clone();
  auto weights = torch::full({kTokens, kTopk}, 1.0f / kTopk, fp);
  for (const torch::ScalarType dtype : {torch::kBFloat16, torch::kFloat16}) {
    auto input_base = torch::arange(kTokens * kHidden, fp).mul_(0.023f).sin_()
                          .mul_(0.25f).view({kTokens, kHidden}).to(dtype);
    auto input = input_base.clone();
    auto activated = torch::empty({kTokens * kTopk, kIntermediate}, fp.dtype(dtype));
    auto routed = torch::empty({kTokens * kTopk, kHidden}, fp.dtype(dtype));
    auto output = torch::empty_like(input);
    auto execute = [&] {
      return deepseek_v4_fp8_moe_gemv_decode(input, ids, weights, w13, s13, w2, s2,
                                            activated, routed, output, 7.0);
    };
    for (const std::string w13_tile : {"4", "8", "16"}) {
      for (const std::string w2_tile : {"32", "64", "128"}) {
        SCOPED_TRACE(w13_tile + "/" + w2_tile);
        setenv("XLLM_DSV4_MOE_W13_TILE_N", w13_tile.c_str(), 1);
        setenv("XLLM_DSV4_MOE_W2_TILE_N", w2_tile.c_str(), 1);
        execute();
        stream.synchronize();
        musaGraph_t graph = nullptr;
        musaGraphExec_t executable = nullptr;
        ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal), musaSuccess);
        execute();
        ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
        ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
        for (int32_t frame = 0; frame < 3; ++frame) {
          input.copy_(input_base.roll({frame}, {0}));
          ids.copy_((ids_base + frame * 3).remainder(kExperts));
          ids[0][0] = -1;
          ids[1][1] = kExperts;
          activated.fill_(-19.0f);
          setenv("XLLM_DSV4_MOE_W13_TILE_N", "4", 1);
          setenv("XLLM_DSV4_MOE_W2_TILE_N", "32", 1);
          execute();
          stream.synchronize();
          auto reference_a = activated.cpu().clone();
          auto reference_r = routed.cpu().clone();
          auto reference_o = output.cpu().clone();
          activated.fill_(-19.0f);
          output.fill_(-31.0f);
          ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
          stream.synchronize();
          EXPECT_TRUE(torch::equal(activated.cpu(), reference_a));
          EXPECT_TRUE(torch::equal(routed.cpu(), reference_r));
          EXPECT_TRUE(torch::equal(output.cpu(), reference_o));
          EXPECT_TRUE(torch::isfinite(output).all().cpu().item<bool>());
        }
        if (dtype == torch::kBFloat16) {
          ids.copy_(ids_base);
          musaEvent_t begin = nullptr;
          musaEvent_t end = nullptr;
          ASSERT_EQ(musaEventCreate(&begin), musaSuccess);
          ASSERT_EQ(musaEventCreate(&end), musaSuccess);
          ASSERT_EQ(musaEventRecord(begin, stream.stream()), musaSuccess);
          for (int32_t repeat = 0; repeat < 100; ++repeat) {
            ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
          }
          ASSERT_EQ(musaEventRecord(end, stream.stream()), musaSuccess);
          ASSERT_EQ(musaEventSynchronize(end), musaSuccess);
          float elapsed_ms = 0.0f;
          ASSERT_EQ(musaEventElapsedTime(&elapsed_ms, begin, end), musaSuccess);
          LOG(INFO) << "[MOE_TILE_BENCH] w13_n=" << w13_tile << " w2_n=" << w2_tile
                    << " tokens=6 hidden=4096 intermediate=256 mean_us=" << elapsed_ms * 10.0f;
          ASSERT_EQ(musaEventDestroy(begin), musaSuccess);
          ASSERT_EQ(musaEventDestroy(end), musaSuccess);
        }
        ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
        ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
      }
    }
  }
  unsetenv("XLLM_DSV4_MOE_W13_TILE_N");
  unsetenv("XLLM_DSV4_MOE_W2_TILE_N");
}

TEST(DeepseekV4HCPostTest, NonsymmetricInputOutputMixing) {
  int device_count = 0;
  ASSERT_EQ(musaGetDeviceCount(&device_count), musaSuccess);
  ASSERT_GT(device_count, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  c10::musa::MUSAGuard guard(device);
  for (const int64_t hc : {int64_t{3}, int64_t{4}}) {
    for (const auto dtype : {torch::kFloat32, torch::kBFloat16}) {
      constexpr int64_t kTokens = 6;
      constexpr int64_t kHidden = 4096;
      auto residual_cpu = torch::arange(kTokens * hc * kHidden, torch::kFloat32)
                              .remainder(97).sub(48).div(128)
                              .reshape({kTokens, hc, kHidden}).to(dtype);
      auto x_cpu = torch::full({kTokens, kHidden}, 0.25f, torch::kFloat32).to(dtype);
      auto post_cpu = torch::full({kTokens, hc}, 0.125f, torch::kFloat32);
      // Non-symmetric, doubly-stochastic mixer. Identity/symmetric matrices
      // and identical residual lanes cannot distinguish M from M^T.
      auto identity = torch::eye(hc, torch::kFloat32);
      auto comb_cpu = (identity.roll({1}, {1}) * 0.75f + identity * 0.25f)
                          .unsqueeze(0).repeat({kTokens, 1, 1});
      auto expected = (torch::matmul(comb_cpu.transpose(1, 2), residual_cpu.to(torch::kFloat32)) +
                       post_cpu.unsqueeze(-1) * x_cpu.to(torch::kFloat32).unsqueeze(1)).to(dtype);
      auto actual = deepseek_v4_hc_post(x_cpu.to(device), residual_cpu.to(device),
                                       post_cpu.to(device), comb_cpu.to(device), std::nullopt);
      const float error = (actual.cpu().to(torch::kFloat32) - expected.to(torch::kFloat32))
                              .abs().max().item<float>();
      LOG(INFO) << "[HC_POST_ORIENTATION] hc=" << hc << " dtype=" << dtype << " max_abs=" << error;
      EXPECT_EQ(error, 0.0f);
    }
  }
}

TEST(DeepseekV4HCPreTest, BatchedFinishAndNormMatchesMateGraph) {
  const char* small_tile_env = std::getenv("XLLM_DSV4_HC_PRE_SMALL_TILE");
  const bool compare_small_tile = small_tile_env != nullptr && std::string(small_tile_env) == "1";
  int32_t devices = 0;
  ASSERT_EQ(musaGetDeviceCount(&devices), musaSuccess);
  ASSERT_GT(devices, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  TvmffiStreamOverrideGuard ffi_guard(device, reinterpret_cast<void*>(stream.stream()));
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int64_t kDim = 4096;
  auto fn = torch::arange(24 * 4 * kDim, fp).mul_(0.011f).sin_()
                .mul_(0.003f).view({24, 4 * kDim});
  auto scale = torch::tensor({0.7f, 0.8f, 0.9f}, fp);
  auto base = torch::arange(24, fp).mul_(0.2f).sin_();
  auto gamma = (torch::arange(kDim, fp).mul_(0.03f).sin_().mul_(0.1f) + 1.0f)
                   .to(torch::kBFloat16);
  for (const int64_t tokens : {2, 5, 6, 7, 8}) {
    auto source = torch::arange(tokens * 4 * kDim, fp).mul_(0.017f).sin_()
                      .view({tokens, 4, kDim}).to(torch::kBFloat16);
    auto x = source.clone();
    auto out = torch::empty({tokens, kDim}, x.options());
    auto post = torch::empty({tokens, 4}, fp);
    auto comb = torch::empty({tokens, 4, 4}, fp);
    bool normalized = false;
    auto execute = [&] {
      return deepseek_v4_hc_pre_decode(x, fn, scale, base, 4, 20, 1.0e-6, 1.0e-6,
                                       out, post, comb, gamma, &normalized);
    };
    auto norm_reference = [&](const torch::Tensor& raw) {
      auto variance = raw.to(torch::kFloat32).square().mean(-1, true);
      return ((raw.to(torch::kFloat32) * torch::rsqrt(variance + 1.0e-6))
                  .to(torch::kBFloat16) * gamma).contiguous();
    };
    setenv("XLLM_DSV4_HC_PRE_BATCHED_FINISH", "0", 1);
    setenv("XLLM_DSV4_HC_PRE_SMALL_TILE", "0", 1);
    execute();
    ASSERT_FALSE(normalized);
    stream.synchronize();
    setenv("XLLM_DSV4_HC_PRE_BATCHED_FINISH", "1", 1);
    setenv("XLLM_DSV4_HC_PRE_SMALL_TILE", compare_small_tile ? "1" : "0", 1);
    execute();
    ASSERT_TRUE(normalized);
    stream.synchronize();
    begin_ffi_alloc_record();
    execute();
    auto allocations = end_ffi_alloc_record();
    stream.synchronize();
    musaGraph_t graph = nullptr;
    musaGraphExec_t executable = nullptr;
    ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal), musaSuccess);
    begin_ffi_alloc_replay(&allocations);
    execute();
    ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
    end_ffi_alloc_replay();
    ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
    for (int32_t frame = 0; frame < 4; ++frame) {
      SCOPED_TRACE(::testing::Message() << "tokens=" << tokens << " frame=" << frame);
      x.copy_(source.roll({frame}, {0}) * (0.5f + 0.3f * frame));
      setenv("XLLM_DSV4_HC_PRE_BATCHED_FINISH", "0", 1);
      setenv("XLLM_DSV4_HC_PRE_SMALL_TILE", "0", 1);
      execute();
      stream.synchronize();
      auto expected = norm_reference(out).clone();
      auto expected_post = post.clone();
      auto expected_comb = comb.clone();
      out.fill_(-99);
      ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
      stream.synchronize();
      const float out_error = (out.to(torch::kFloat32) - expected.to(torch::kFloat32))
                                  .abs().max().cpu().item<float>();
      const float post_error = (post - expected_post).abs().max().cpu().item<float>();
      const float comb_error = (comb - expected_comb).abs().max().cpu().item<float>();
      LOG(INFO) << "[HC_BATCHED_FINISH_UNIT] tokens=" << tokens << " frame=" << frame
                << " output_max=" << out_error << " post_max=" << post_error
                << " comb_max=" << comb_error;
      EXPECT_LT(out_error, 0.032f);
      EXPECT_LT(post_error, 2.0e-5f);
      EXPECT_LT(comb_error, 2.0e-5f);
      EXPECT_TRUE(torch::isfinite(out).all().cpu().item<bool>());
    }
    musaEvent_t begin = nullptr;
    musaEvent_t end = nullptr;
    ASSERT_EQ(musaEventCreate(&begin), musaSuccess);
    ASSERT_EQ(musaEventCreate(&end), musaSuccess);
    ASSERT_EQ(musaEventRecord(begin, stream.stream()), musaSuccess);
    for (int32_t repeat = 0; repeat < 200; ++repeat) {
      ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
    }
    ASSERT_EQ(musaEventRecord(end, stream.stream()), musaSuccess);
    ASSERT_EQ(musaEventSynchronize(end), musaSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(musaEventElapsedTime(&elapsed_ms, begin, end), musaSuccess);
    LOG(INFO) << "[HC_BATCHED_FINISH_BENCH] tokens=" << tokens
              << " fused_pair_mean_us=" << elapsed_ms * 5.0f;
    ASSERT_EQ(musaEventDestroy(begin), musaSuccess);
    ASSERT_EQ(musaEventDestroy(end), musaSuccess);
    ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
    ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
  }
  unsetenv("XLLM_DSV4_HC_PRE_BATCHED_FINISH");
  if (compare_small_tile) {
    setenv("XLLM_DSV4_HC_PRE_SMALL_TILE", "1", 1);
  } else {
    unsetenv("XLLM_DSV4_HC_PRE_SMALL_TILE");
  }
}

TEST(DeepseekV4HCPostTest, GraphReplayWithAliasedResidual) {
  int device_count = 0;
  ASSERT_EQ(musaGetDeviceCount(&device_count), musaSuccess);
  ASSERT_GT(device_count, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  c10::musa::MUSAGuard guard(device);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard stream_guard(stream.unwrap());
  constexpr int64_t kTokens = 6;
  constexpr int64_t kHidden = 4096;
  auto cpu = torch::arange(kTokens * 4 * kHidden, torch::kFloat32)
                 .remainder(97).sub(48).div(128).reshape({kTokens, 4, kHidden}).to(torch::kBFloat16);
  auto residual = cpu.to(device);
  auto x = torch::full({kTokens, kHidden}, 0.25f, residual.options());
  auto post = torch::full({kTokens, 4}, 0.125f, residual.options().dtype(torch::kFloat32));
  auto eye = torch::eye(4, torch::kFloat32);
  auto comb_cpu = (eye.roll({1}, {1}) * 0.75f + eye * 0.25f).unsqueeze(0).repeat({kTokens, 1, 1});
  auto comb = comb_cpu.to(device);
  deepseek_v4_hc_post(x, residual, post, comb, residual);
  ASSERT_EQ(musaStreamSynchronize(stream.stream()), musaSuccess);
  musaGraph_t graph;
  musaGraphExec_t executable;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeGlobal), musaSuccess);
  deepseek_v4_hc_post(x, residual, post, comb, residual);
  ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), musaSuccess);
  for (int64_t repeat = 0; repeat < 3; ++repeat) {
    auto mixer = repeat == 1 ? comb_cpu.transpose(1, 2).contiguous() : comb_cpu;
    comb.copy_(mixer);
    residual.copy_(cpu);
    ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
    ASSERT_EQ(musaStreamSynchronize(stream.stream()), musaSuccess);
    auto expected = (torch::matmul(mixer.transpose(1, 2), cpu.to(torch::kFloat32)) + 0.03125f)
                        .to(torch::kBFloat16);
    EXPECT_TRUE(torch::equal(residual.cpu(), expected));
  }
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

constexpr int64_t kHeads = 8;
constexpr int64_t kHeadDim = 512;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kPackedBytes = 584;
constexpr int64_t kPayloadBytes = 576;
constexpr int64_t kNopeDim = 448;
constexpr int64_t kQuantGroup = 64;
constexpr double kSoftmaxScale = 0.044194173824159216;

TEST(DeepseekV4GraphPolicyTest, SeparatesCapturedAttentionBranches) {
  auto key = [](int64_t start, int64_t threshold, int64_t window) {
    constexpr int64_t kTokens = 6;
    const int64_t kv_len = start + kTokens;
    return layer::dsv4_compact_graph_key(
        /*token_bucket=*/6,
        static_cast<uint32_t>((std::max<int64_t>(kv_len / 4, 1) + 511) / 512),
        static_cast<uint32_t>(layer::dsv4_compression_boundaries(start, kTokens, 4)),
        static_cast<uint32_t>(layer::dsv4_compression_boundaries(start, kTokens, 128)),
        layer::dsv4_uses_full_kv(kv_len, threshold, window));
  };
  // Same C4 shape and indexer capacity, but a different host Attention route.
  EXPECT_NE(key(117, 2048, 128), key(129, 2048, 128));
  // These both use compressed history and have two C4 outputs; only the
  // C128 emission count changes. Reusing the graph would omit its cache write.
  EXPECT_NE(key(123, 2048, 128), key(131, 2048, 128));
  EXPECT_EQ(key(131, 2048, 128), key(135, 2048, 128));
  EXPECT_NE(key(121, 2048, 128), key(121, 0, 128));
  EXPECT_TRUE(layer::dsv4_uses_full_kv(128, 2048, 128));
  EXPECT_FALSE(layer::dsv4_uses_full_kv(129, 2048, 128));
  EXPECT_FALSE(layer::dsv4_uses_full_kv(64, 0, 128));
  EXPECT_TRUE(layer::dsv4_uses_full_kv(512, 2048, 0));
  // Exhaustively verify q6 shapes around page/compression boundaries.
  for (int64_t start = 0; start < 8192; ++start) {
    for (const int64_t ratio : {int64_t{4}, int64_t{128}}) {
      const int64_t expected = (start + 6) / ratio - start / ratio;
      EXPECT_EQ(layer::dsv4_compression_boundaries(start, 6, ratio), expected);
    }
  }
}

torch::Tensor decode_packed_rows(const torch::Tensor& cache,
                                 const torch::Tensor& physical) {
  TORCH_CHECK(cache.defined());
  TORCH_CHECK(physical.defined());
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cache.dim() == 4);
  TORCH_CHECK(cache.size(2) == 1);
  TORCH_CHECK(cache.size(3) == kPackedBytes);

  auto flat = physical.to(torch::kCPU).reshape({-1}).to(torch::kLong).contiguous();
  auto valid = flat.ge(0).logical_and(
      flat.lt(cache.size(0) * cache.size(1)));
  flat = torch::masked_select(flat, valid).contiguous();
  if (flat.numel() == 0) {
    return torch::empty({0, kHeadDim},
                        torch::TensorOptions().dtype(torch::kFloat32));
  }
  const int64_t page_stride = cache.stride(0);
  auto pages = torch::floor_divide(flat, kBlockSize);
  auto offsets = torch::remainder(flat, kBlockSize);
  auto storage = cache.as_strided({cache.size(0) * page_stride}, {1}).to(torch::kCPU);
  auto payload_base = pages * page_stride + offsets * kPayloadBytes;
  auto payload_offsets = torch::arange(
      kPayloadBytes,
      torch::TensorOptions().dtype(torch::kLong));
  auto payload = storage
                     .index_select(
                         0,
                         (payload_base.unsqueeze(1) +
                          payload_offsets.unsqueeze(0))
                             .reshape({-1}))
                     .view({flat.numel(), kPayloadBytes});

  auto scale_base = pages * page_stride + kBlockSize * kPayloadBytes +
                    offsets * 8;
  auto scale_offsets = torch::arange(
      7, torch::TensorOptions().dtype(torch::kLong));
  auto scale_exponents = storage
                             .index_select(
                                 0,
                                 (scale_base.unsqueeze(1) +
                                  scale_offsets.unsqueeze(0))
                                     .reshape({-1}))
                             .view({flat.numel(), 7})
                             .to(torch::kFloat32) -
                         127.0;
  auto scales = torch::pow(torch::full_like(scale_exponents, 2.0),
                           scale_exponents)
                    .repeat_interleave(kQuantGroup, /*dim=*/1);
  auto nope = payload.slice(1, 0, kNopeDim)
                  .view(torch::kFloat8_e4m3fn)
                  .to(torch::kFloat32) *
              scales;
  auto rope = payload.slice(1, kNopeDim, kPayloadBytes)
                  .view(torch::kBFloat16)
                  .to(torch::kFloat32);
  return torch::cat({nope, rope}, /*dim=*/1).contiguous();
}

torch::Tensor dense_reference(const torch::Tensor& q,
                              const torch::Tensor& cache,
                              const torch::Tensor& main_physical,
                              int64_t topk_length,
                              const torch::Tensor& extra_physical,
                              int64_t extra_topk_length,
                              const torch::Tensor& sink) {
  TORCH_CHECK(q.dim() == 3);
  const int64_t batch = q.size(0);
  std::vector<torch::Tensor> rows;
  rows.reserve(static_cast<size_t>(batch));
  for (int64_t row = 0; row < batch; ++row) {
    auto main_row = main_physical.select(0, row).slice(0, 0, topk_length);
    auto extra_row =
        extra_physical.select(0, row).slice(0, 0, extra_topk_length);
    auto main_kv = decode_packed_rows(cache, main_row);
    auto extra_kv = decode_packed_rows(cache, extra_row);
    auto selected = torch::cat({main_kv, extra_kv}, /*dim=*/0);
    auto q_row = q.select(0, row).to(torch::kCPU).to(torch::kFloat32);
    auto scores = torch::matmul(q_row, selected.transpose(0, 1)) *
                  kSoftmaxScale;
    auto max_score = std::get<0>(scores.max(/*dim=*/-1, /*keepdim=*/true));
    auto exp_scores = torch::exp(scores - max_score);
    auto denom = exp_scores.sum(/*dim=*/-1, /*keepdim=*/true);
    if (sink.defined()) {
      denom = denom + torch::exp(sink.to(torch::kCPU).to(torch::kFloat32).view({kHeads, 1}) -
                                 max_score);
    }
    rows.push_back(torch::matmul(exp_scores / denom, selected)
                       .to(torch::kBFloat16));
  }
  return torch::stack(rows, /*dim=*/0).contiguous();
}

struct MateCaseResult {
  torch::Tensor actual;
  torch::Tensor expected;
};

MateCaseResult run_mate_case(const torch::Tensor& q,
                             const torch::Tensor& values,
                             const torch::Tensor& main_physical,
                             int64_t topk_length,
                             const torch::Tensor& extra_physical,
                             int64_t extra_topk_length) {
  const torch::Device device = q.device();
  const int64_t cache_rows = values.size(0);
  const int64_t blocks = (cache_rows + kBlockSize - 1) / kBlockSize;
  const int64_t page_stride =
      ((kBlockSize * kPackedBytes + 576 - 1) / 576) * 576;
  auto storage = torch::zeros({blocks * page_stride},
                              torch::TensorOptions()
                                  .dtype(torch::kUInt8)
                                  .device(device));
  auto cache = storage.as_strided({blocks, kBlockSize, 1, kPackedBytes},
                                  {page_stride, kPackedBytes, kPackedBytes, 1});
  auto slots = torch::arange(cache_rows,
                             torch::TensorOptions()
                                 .dtype(torch::kInt32)
                                 .device(device));
  deepseek_v4_pack_fp8_cache(cache, slots, values);
  auto sink = torch::full({kHeads},
                          -std::numeric_limits<float>::infinity(),
                          torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(device));
  auto actual = deepseek_v4_mate_sparse_attention_decode(
      q,
      cache,
      main_physical,
      topk_length,
      cache,
      extra_physical,
      extra_topk_length,
      sink,
      kSoftmaxScale,
      std::nullopt);
  if (actual.defined()) {
    actual = actual.contiguous();
  }
  auto expected = dense_reference(q,
                                 cache,
                                 main_physical,
                                 topk_length,
                                 extra_physical,
                                 extra_topk_length,
                                 sink);
  return {actual, expected};
}

torch::Tensor deterministic_values(int64_t rows, const torch::Device& device) {
  auto cpu = torch::arange(rows * kHeadDim,
                           torch::TensorOptions().dtype(torch::kFloat32))
                 .reshape({rows, kHeadDim});
  // Keep the values bounded so the FP8 payload and the BF16 rope payload both
  // exercise nontrivial scales without overflowing the MODEL1 format.
  return ((cpu * 0.0017).sin() * 1.7).to(device, torch::kBFloat16);
}

void expect_close(const MateCaseResult& result, const char* name) {
  ASSERT_TRUE(result.actual.defined())
      << name << ": MATE artifact/kernel returned undefined output";
  ASSERT_EQ(result.actual.sizes(), result.expected.sizes()) << name;
  auto diff = (result.actual.to(torch::kCPU).to(torch::kFloat32) -
               result.expected.to(torch::kFloat32))
                  .abs();
  const float max_abs = diff.max().item<float>();
  const float mean_abs = diff.mean().item<float>();
  LOG(INFO) << "[MATE_UNIT] case=" << name << " max_abs=" << max_abs
            << " mean_abs=" << mean_abs;
  EXPECT_LT(max_abs, 0.08f) << name;
  EXPECT_LT(mean_abs, 0.01f) << name;
}

TEST(DeepseekV4MateSparseAttentionTest, Q1AndQ2MatchDenseReference) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  if (!std::ifstream(path_to_uri_so_lib(
          "dsa_model1_h8_topk512_extra128_bs128"))
           .good()) {
    GTEST_SKIP() << "MATE MODEL1 h8 decode artifact unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  c10::musa::MUSAStream stream =
      c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());

  const int64_t rows = 512;
  auto values = deterministic_values(rows, device);
  auto q1 = deterministic_values(kHeads, device).view({1, kHeads, kHeadDim});
  auto main1 = torch::arange(
                   8, torch::TensorOptions().dtype(torch::kInt32).device(device))
                   .view({1, 8});
  auto extra1 = torch::arange(
                     128,
                     torch::TensorOptions().dtype(torch::kInt32).device(device))
                     .view({1, 128});
  auto q1_result = run_mate_case(
      q1, values, main1, /*topk_length=*/8, extra1, /*extra_topk_length=*/128);
  stream.synchronize();
  expect_close(q1_result, "q1_scalar");

  auto q2 = deterministic_values(2 * kHeads, device).view({2, kHeads, kHeadDim});
  auto main2 = torch::stack({
      torch::arange(8, torch::TensorOptions().dtype(torch::kInt32).device(device)),
      torch::arange(8, 16, torch::TensorOptions().dtype(torch::kInt32).device(device))});
  auto extra2 = torch::stack({
      torch::arange(128, torch::TensorOptions().dtype(torch::kInt32).device(device)),
      torch::arange(128, 256,
                    torch::TensorOptions().dtype(torch::kInt32).device(device))});
  auto q2_result = run_mate_case(
      q2, values, main2, /*topk_length=*/8, extra2, /*extra_topk_length=*/128);
  stream.synchronize();
  expect_close(q2_result, "q2_batched_rows");
}

TEST(DeepseekV4MateSparseAttentionTest,
     GraphC4StateMatchesScalarAcrossAbsoluteBoundaries) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  c10::musa::MUSAStream stream =
      c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());

  constexpr int64_t kHeadDim = 16;
  constexpr int64_t kWidth = 2 * kHeadDim;
  constexpr int64_t kStateRows = 8;
  constexpr int64_t kTokens = 6;
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto state = torch::arange(kStateRows * kWidth, options)
                   .reshape({kStateRows, kWidth})
                   .mul_(0.013f);
  auto score = state.mul(0.17f);
  auto kv_steps = torch::arange(kTokens * kWidth, options)
                      .reshape({kTokens, kWidth})
                      .mul_(0.007f);
  auto score_steps = kv_steps.mul(0.11f);
  auto ape = torch::arange(4 * kWidth, options).reshape({4, kWidth}).mul_(0.03f);

  for (const int64_t start_position : {int64_t{0}, int64_t{1}, int64_t{2},
                                       int64_t{3}, int64_t{127}}) {
    auto scalar_state = state.clone();
    auto scalar_score = score.clone();
    std::vector<torch::Tensor> scalar_outputs;
    for (int64_t token = 0; token < kTokens; ++token) {
      auto output = deepseek_v4_compress_decode(
          scalar_state,
          scalar_score,
          kv_steps.narrow(/*dim=*/0, token, 1),
          score_steps.narrow(/*dim=*/0, token, 1),
          ape,
          /*compress_ratio=*/4,
          kHeadDim,
          /*compressor_coeff=*/2,
          start_position + token,
          /*compute_boundary_output=*/true);
      if (output.defined()) {
        scalar_outputs.push_back(output);
      }
    }
    auto scalar_output = scalar_outputs.empty()
                             ? torch::empty({0, kHeadDim}, options)
                             : torch::cat(scalar_outputs, /*dim=*/0);

    auto graph_state = state.clone();
    auto graph_score = score.clone();
    const int64_t first_boundary =
        (4 - ((start_position + 1) % 4)) % 4;
    const int64_t boundary_count =
        first_boundary < kTokens
            ? 1 + (kTokens - 1 - first_boundary) / 4
            : 0;
    auto graph_output = torch::empty({boundary_count, kHeadDim}, options);
    auto position = torch::tensor({start_position},
                                  torch::TensorOptions()
                                      .dtype(torch::kInt64)
                                      .device(device));
    deepseek_v4_compress_decode_sequence_graph(
        graph_state,
        graph_score,
        kv_steps,
        score_steps,
        ape,
        /*compress_ratio=*/4,
        kHeadDim,
        position,
        graph_output);
    stream.synchronize();
    ASSERT_EQ(graph_output.size(0), scalar_output.size(0))
        << "start_position=" << start_position;
    auto state_diff = (graph_state - scalar_state)
                          .abs()
                          .max()
                          .to(torch::kCPU)
                          .item<float>();
    auto score_diff = (graph_score - scalar_score)
                          .abs()
                          .max()
                          .to(torch::kCPU)
                          .item<float>();
    auto output_diff =
        scalar_output.numel() == 0
            ? 0.0f
            : (graph_output - scalar_output)
                  .abs()
                  .max()
                  .to(torch::kCPU)
                  .item<float>();
    LOG(INFO) << "[DSV4_GRAPH_STATE_UNIT] start=" << start_position
              << " state_max_abs=" << state_diff
              << " score_max_abs=" << score_diff
              << " output_max_abs=" << output_diff;
    EXPECT_LT(state_diff, 1.0e-5f);
    EXPECT_LT(score_diff, 1.0e-5f);
    EXPECT_LT(output_diff, 1.0e-5f);
  }
}

TEST(DeepseekV4MateSparseAttentionTest,
     CapturedC128AttentionUsesPerRowCausalLengths) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const c10::musa::MUSAStream stream =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());
  constexpr int64_t kQueries = 6;
  constexpr int64_t kCapacity = 64;
  constexpr int64_t kRows = 4 * 128;
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  for (const torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
    auto values = torch::arange(kRows * kHeadDim, fp32)
                      .mul_(0.017f).sin_().reshape({4, 128, 1, kHeadDim}).to(dtype);
    auto compressed = values.add(0.5f);
    auto q = torch::arange(kQueries * kHeads * kHeadDim, fp32)
                 .mul_(0.013f).sin_().mul_(0.1f)
                 .reshape({kQueries, kHeads, kHeadDim}).to(dtype);
    auto sink = torch::zeros({kHeads}, fp32);
    auto lengths = torch::arange(121, 121 + kQueries, fp32.dtype(torch::kInt32));
    auto output = torch::empty_like(q);
    for (const torch::ScalarType index_dtype : {torch::kInt32, torch::kInt64}) {
      auto table = torch::arange(kCapacity, fp32.dtype(index_dtype))
                       .remainder(4).view({1, kCapacity});
      auto cmp_table = torch::tensor({2, 1, 0, 3}, fp32.dtype(index_dtype))
                           .view({1, 4});
      auto execute = [&] {
        deepseek_v4_paged_attention_verify_c128(
            q, values, table, lengths, compressed, cmp_table, kCapacity,
            sink, kSoftmaxScale, output);
      };
      execute();
      stream.synchronize();
      musaGraph_t graph = nullptr;
      musaGraphExec_t executable = nullptr;
      ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                      musaStreamCaptureModeThreadLocal), musaSuccess);
      execute();
      ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
      size_t nodes = 0;
      ASSERT_EQ(musaGraphGetNodes(graph, nullptr, &nodes), musaSuccess);
      ASSERT_GT(nodes, 0);
      ASSERT_EQ(musaGraphInstantiate(&executable, graph, /*flags=*/0), musaSuccess);
      for (const int64_t start : {int64_t{120}, int64_t{125}, int64_t{439},
                                 int64_t{1023}, int64_t{4095}, int64_t{0}}) {
        SCOPED_TRACE(::testing::Message() << "start=" << start
                     << " dtype=" << dtype << " table_dtype=" << index_dtype);
        lengths.copy_(torch::arange(start + 1, start + 1 + kQueries,
                                   lengths.options()));
        const int64_t offset = (start / 128) % 4;
        table.copy_(torch::arange(kCapacity, table.options())
                        .add_(offset).remainder_(4).view_as(table));
        cmp_table.fill_((offset + 2) % 4);
        stream.synchronize();
        ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
        stream.synchronize();
        for (int64_t row = 0; row < kQueries; ++row) {
          const int64_t kv_len = start + row + 1;
          std::vector<int64_t> physical;
          physical.reserve(128);
          for (int64_t pos = std::max<int64_t>(0, kv_len - 128);
               pos < kv_len; ++pos) {
            physical.emplace_back(((pos / 128 + offset) % 4) * 128 + pos % 128);
          }
          auto indices = torch::tensor(physical, fp32.dtype(torch::kInt64));
          auto selected = values.view({kRows, kHeadDim}).index_select(0, indices);
          const int64_t cmp_len = kv_len / 128;
          if (cmp_len > 0) {
            auto selected_cmp = compressed.view({kRows, kHeadDim}).narrow(
                /*dim=*/0, ((offset + 2) % 4) * 128, cmp_len);
            selected = torch::cat({selected, selected_cmp}, /*dim=*/0);
          }
          auto scores = torch::matmul(q[row].to(torch::kFloat32),
                                     selected.to(torch::kFloat32).transpose(0, 1))
                            .mul_(kSoftmaxScale);
          auto probabilities = torch::softmax(
              torch::cat({scores, sink.view({kHeads, 1})}, /*dim=*/1), /*dim=*/1);
          auto reference = torch::matmul(
              probabilities.narrow(/*dim=*/1, /*start=*/0, selected.size(0)),
              selected.to(torch::kFloat32));
          const float difference = (output[row].to(torch::kFloat32) - reference)
                                       .abs().max().to(torch::kCPU).item<float>();
          EXPECT_LT(difference, dtype == torch::kFloat32 ? 2.0e-5f : 2.0e-3f)
              << "row=" << row;
        }
      }
      ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
      ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
      if (dtype == torch::kBFloat16 && index_dtype == torch::kInt32) {
        lengths.copy_(torch::arange(2500, 2500 + kQueries, lengths.options()));
        stream.synchronize();
        ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                        musaStreamCaptureModeThreadLocal), musaSuccess);
        for (int32_t call = 0; call < 32; ++call) {
          execute();
        }
        ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
        ASSERT_EQ(musaGraphInstantiate(&executable, graph, /*flags=*/0), musaSuccess);
        ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
        musaEvent_t begin = nullptr;
        musaEvent_t end = nullptr;
        ASSERT_EQ(musaEventCreate(&begin), musaSuccess);
        ASSERT_EQ(musaEventCreate(&end), musaSuccess);
        ASSERT_EQ(musaEventRecord(begin, stream.stream()), musaSuccess);
        for (int32_t replay = 0; replay < 10; ++replay) {
          ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
        }
        ASSERT_EQ(musaEventRecord(end, stream.stream()), musaSuccess);
        ASSERT_EQ(musaEventSynchronize(end), musaSuccess);
        float elapsed_ms = 0.0f;
        ASSERT_EQ(musaEventElapsedTime(&elapsed_ms, begin, end), musaSuccess);
        LOG(INFO) << "[C128_VERIFY_BENCH] q=6 heads=8 dim=512 capacity=64 calls=320 mean_us="
                  << elapsed_ms * 1000.0f / 320;
        ASSERT_EQ(musaEventDestroy(begin), musaSuccess);
        ASSERT_EQ(musaEventDestroy(end), musaSuccess);
        ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
        ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
      }
    }
  }
}

TEST(DeepseekV4MateSparseAttentionTest,
     CapturedShortC4MasksStaleRowsAfterRequestSwitch) {
  int32_t device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int64_t kQueries = 6;
  for (const auto dtype : {torch::kFloat32, torch::kBFloat16}) {
    auto values = torch::arange(2 * 128 * kHeadDim, fp).mul_(0.017f).sin_()
                      .reshape({2, 128, 1, kHeadDim}).to(dtype);
    auto compressed_base = values.add(0.25f);
    auto compressed = compressed_base.clone();
    auto q_base = torch::arange(kQueries * kHeads * kHeadDim, fp)
                      .mul_(0.013f).sin_().mul_(0.1f)
                      .reshape({kQueries, kHeads, kHeadDim}).to(dtype);
    auto q = q_base.clone();
    auto sink = torch::zeros({kHeads}, fp);
    auto lengths = torch::arange(92, 92 + kQueries, fp.dtype(torch::kInt32));
    auto output = torch::empty_like(q);
    for (const auto index_dtype : {torch::kInt32, torch::kInt64}) {
      auto table = torch::tensor({0, 1}, fp.dtype(index_dtype)).view({1, 2});
      auto compressed_table = torch::ones({1, 1}, fp.dtype(index_dtype));
      auto attend = [&] {
        return deepseek_v4_paged_attention_verify_compressed(
            q, values, table, lengths, compressed, compressed_table,
            /*compressed_capacity=*/32, /*compression_ratio=*/4,
            sink, kSoftmaxScale, output);
      };
      ASSERT_TRUE(attend().defined());
      stream.synchronize();
      musaGraph_t graph = nullptr;
      musaGraphExec_t executable = nullptr;
      ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                       musaStreamCaptureModeThreadLocal), musaSuccess);
      attend();
      ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
      ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
      int64_t iteration = 0;
      for (const int64_t first_kv : {112, 68, 63, 66, 109, 123, 0, 92}) {
        SCOPED_TRACE(::testing::Message() << "first_kv=" << first_kv
                     << " dtype=" << dtype << " table_dtype=" << index_dtype);
        const int64_t page = iteration % 2;
        table.copy_(torch::tensor({page, 1 - page}, table.options()).view_as(table));
        compressed_table.fill_(1 - page);
        lengths.copy_(torch::arange(first_kv, first_kv + kQueries, lengths.options()));
        q.copy_(q_base.roll({iteration % kQueries}, {0}));
        compressed.copy_(compressed_base);
        const int64_t max_valid = (first_kv + kQueries - 1) / 4;
        // Poison the capacity beyond this request's longest causal prefix.
        compressed.select(0, 1 - page).narrow(0, max_valid, 128 - max_valid)
            .fill_(101.0f);
        ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
        stream.synchronize();
        for (int64_t row = 0; row < kQueries; ++row) {
          const int64_t kv_length = first_kv + row;
          const int64_t swa_length = std::min<int64_t>(kv_length, 128);
          auto selected = values.select(0, page)
                              .narrow(0, kv_length - swa_length, swa_length)
                              .reshape({swa_length, kHeadDim});
          const int64_t compressed_length = kv_length / 4;
          auto selected_compressed = compressed.select(0, 1 - page)
                                         .narrow(0, 0, compressed_length)
                                         .reshape({compressed_length, kHeadDim});
          auto valid = torch::cat({selected, selected_compressed}, 0)
                           .to(torch::kFloat32);
          auto scores = torch::matmul(q[row].to(torch::kFloat32),
                                     valid.transpose(0, 1)).mul_(kSoftmaxScale);
          auto probabilities = torch::softmax(
              torch::cat({scores, sink.view({kHeads, 1})}, 1), 1);
          auto reference = torch::matmul(probabilities.narrow(1, 0, valid.size(0)), valid);
          const float error = (output[row].to(torch::kFloat32) - reference)
                                  .abs().max().cpu().item<float>();
          EXPECT_LT(error, dtype == torch::kFloat32 ? 2.0e-5f : 2.0e-3f);
        }
        ++iteration;
      }
      ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
      ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
    }
  }
}

TEST(DeepseekV4MateSparseAttentionTest, C128Model1CausalGraphMatchesDense) {
  int devices = 0;
  ASSERT_EQ(musaGetDeviceCount(&devices), musaSuccess);
  ASSERT_GT(devices, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  TvmffiStreamOverrideGuard ffi_guard(device, reinterpret_cast<void*>(stream.stream()));
  constexpr int64_t kRows = 6;
  constexpr int64_t kPageStride = ((kBlockSize * kPackedBytes + 575) / 576) * 576;
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto storage = torch::zeros({4 * kPageStride}, i32.dtype(torch::kUInt8));
  auto cache = storage.as_strided({4, kBlockSize, 1, kPackedBytes},
                                  {kPageStride, kPackedBytes, kPackedBytes, 1});
  auto values = deterministic_values(512, device);
  auto slots = torch::arange(512, i32);
  deepseek_v4_pack_fp8_cache(cache, slots, values);
  stream.synchronize();
  auto all_values = decode_packed_rows(cache, slots);
  auto q = deterministic_values(kRows * kHeads, device)
               .reshape({kRows, kHeads, kHeadDim}).mul_(0.1f);
  auto sink = torch::arange(kHeads, i32.dtype(torch::kFloat32)).mul_(0.05f);
  auto lengths = torch::arange(123, 123 + kRows, i32);
  auto cmp = torch::arange(4, i32).reshape({1, 4});
  auto swa = torch::arange(512, i32).remainder_(4).reshape({1, 512});
  auto logical = torch::empty({kRows, 512}, i32);
  auto main = torch::empty_like(logical);
  auto extra = torch::empty({kRows, 128}, i32);
  auto main_len = torch::empty({kRows}, i32);
  auto extra_len = torch::empty_like(main_len);
  auto output = torch::empty_like(q);
  auto output_again = torch::empty_like(q);
  auto peer_output = torch::empty_like(q);
  auto peer_main = torch::arange(512, i32).unsqueeze(0).repeat({kRows, 1});
  auto peer_extra = torch::full({kRows, 128}, -1, i32);
  auto peer_lengths = torch::empty_like(main_len);
  auto zero_lengths = torch::zeros_like(main_len);
  auto execute = [&] {
    deepseek_v4_prepare_c128_mate_indices(lengths, cmp, swa, 128, logical, main, extra);
    deepseek_v4_mate_causal_lengths(lengths, 128, 512, 128, main_len, extra_len);
    deepseek_v4_mate_sparse_attention_set_metadata_group(128);
    deepseek_v4_mate_sparse_attention_set_metadata_key(42);
    auto first = deepseek_v4_mate_sparse_attention_decode_batched(
        q, cache, main, main_len, cache, extra, extra_len, sink, kSoftmaxScale, output);
    // Interleave another ratio's distinct visible-key set, then reuse C128.
    // New graph capture must still record the first writer for BOTH groups.
    deepseek_v4_mate_causal_lengths(lengths, 4, 512, 0, peer_lengths, zero_lengths);
    deepseek_v4_mate_sparse_attention_set_metadata_group(4);
    deepseek_v4_mate_sparse_attention_set_metadata_key(42);
    deepseek_v4_mate_sparse_attention_decode_batched(q, cache, peer_main, peer_lengths,
        cache, peer_extra, zero_lengths, sink, kSoftmaxScale, peer_output);
    deepseek_v4_mate_sparse_attention_set_metadata_group(128);
    deepseek_v4_mate_sparse_attention_set_metadata_key(42);
    deepseek_v4_mate_sparse_attention_decode_batched(q, cache, main, main_len,
        cache, extra, extra_len, sink, kSoftmaxScale, output_again);
    return first;
  };
  ASSERT_TRUE(execute().defined());
  stream.synchronize();
  begin_ffi_alloc_record();
  ASSERT_TRUE(execute().defined());
  auto allocations = end_ffi_alloc_record();
  stream.synchronize();
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal), musaSuccess);
  begin_ffi_alloc_replay(&allocations);
  auto captured = execute();
  const musaError_t capture_status = musaStreamEndCapture(stream.stream(), &graph);
  end_ffi_alloc_replay();
  ASSERT_EQ(capture_status, musaSuccess);
  ASSERT_TRUE(captured.defined());
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
  // Create a SECOND capture at the same host key, after warmup/first capture
  // populated the metadata cache. Omitting its first writer would freeze it.
  musaGraph_t second_graph = nullptr;
  musaGraphExec_t second_exec = nullptr;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal), musaSuccess);
  begin_ffi_alloc_replay(&allocations);
  execute();
  const musaError_t second_status = musaStreamEndCapture(stream.stream(), &second_graph);
  end_ffi_alloc_replay();
  ASSERT_EQ(second_status, musaSuccess);
  ASSERT_EQ(musaGraphInstantiate(&second_exec, second_graph, 0), musaSuccess);
  int64_t phase = 0;
  for (const int64_t start : {int64_t{0}, int64_t{1}, int64_t{123}, int64_t{127},
                            int64_t{439}, int64_t{1023}, int64_t{2497},
                            int64_t{4093}, int64_t{32763}, int64_t{112}}) {
    SCOPED_TRACE(::testing::Message() << "start=" << start);
    lengths.copy_(torch::arange(start, start + kRows, i32));
    cmp.copy_(torch::arange(4, i32).add_(phase).remainder_(4).reshape({1, 4}));
    swa.copy_(torch::arange(512, i32).add_(phase + 1).remainder_(4).reshape({1, 512}));
    q.copy_(q.flip({0}).contiguous());
    stream.synchronize();
    ASSERT_EQ(musaGraphLaunch(phase % 2 == 0 ? executable : second_exec, stream.stream()), musaSuccess);
    stream.synchronize();
    EXPECT_TRUE(torch::equal(output.cpu(), output_again.cpu()));
    auto cmp_cpu = cmp.cpu().to(torch::kLong).view({-1});
    auto swa_cpu = swa.cpu().to(torch::kLong).view({-1});
    auto main_cpu = main.cpu();
    auto extra_cpu = extra.cpu();
    std::vector<torch::Tensor> references;
    references.reserve(kRows);
    for (int64_t row = 0; row < kRows; ++row) {
      const int64_t length = start + row;
      const int64_t compressed = length / 128;
      auto ids = torch::arange(compressed, torch::kLong);
      auto mapped = cmp_cpu.index_select(0, ids.floor_divide(128)).mul(128).add(ids.remainder(128));
      EXPECT_TRUE(torch::equal(main_cpu[row].narrow(0, 0, compressed).to(torch::kLong), mapped));
      EXPECT_TRUE(main_cpu[row].narrow(0, compressed, 512 - compressed).eq(-1).all().item<bool>());
      auto tail = torch::arange(std::max<int64_t>(length - 128, 0), length, torch::kLong);
      auto mapped_tail = swa_cpu.index_select(0, tail.floor_divide(128)).mul(128).add(tail.remainder(128));
      EXPECT_TRUE(torch::equal(extra_cpu[row].narrow(0, 0, tail.numel()).to(torch::kLong), mapped_tail));
      auto selected = all_values.index_select(0, torch::cat({mapped, mapped_tail}));
      auto scores = torch::matmul(q[row].cpu().to(torch::kFloat32), selected.transpose(0, 1))
                        .mul(kSoftmaxScale);
      auto probability = torch::softmax(torch::cat({scores, sink.cpu().view({kHeads, 1})}, 1), 1);
      references.emplace_back(torch::matmul(probability.narrow(1, 0, selected.size(0)), selected));
    }
    auto reference = torch::stack(references);
    auto diff = (output.cpu().to(torch::kFloat32) - reference).abs();
    LOG(INFO) << "[C128_MODEL1_UNIT] start=" << start << " max_abs=" << diff.max().item<float>()
              << " mean_abs=" << diff.mean().item<float>();
    EXPECT_LT(diff.max().item<float>(), 0.02f);
    EXPECT_LT(diff.mean().item<float>(), 0.003f);
    for (int64_t row = 0; row < kRows; ++row) {
      const int64_t valid = std::min<int64_t>((start + row) / 4, 512);
      auto selected = all_values.narrow(0, 0, valid);
      auto scores = torch::matmul(q[row].cpu().to(torch::kFloat32), selected.transpose(0, 1))
                        .mul(kSoftmaxScale);
      auto probability = torch::softmax(torch::cat({scores, sink.cpu().view({kHeads, 1})}, 1), 1);
      auto peer_reference = torch::matmul(probability.narrow(1, 0, valid), selected);
      EXPECT_LT((peer_output[row].cpu().to(torch::kFloat32) - peer_reference)
                    .abs().max().item<float>(), 0.02f);
    }
    ++phase;
  }
  lengths.copy_(torch::arange(2500, 2500 + kRows, i32));
  stream.synchronize();
  musaEvent_t begin = nullptr;
  musaEvent_t end = nullptr;
  ASSERT_EQ(musaEventCreate(&begin), musaSuccess);
  ASSERT_EQ(musaEventCreate(&end), musaSuccess);
  ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
  ASSERT_EQ(musaEventRecord(begin, stream.stream()), musaSuccess);
  for (int32_t repeat = 0; repeat < 100; ++repeat) {
    ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
  }
  ASSERT_EQ(musaEventRecord(end, stream.stream()), musaSuccess);
  ASSERT_EQ(musaEventSynchronize(end), musaSuccess);
  float ms = 0.0f;
  ASSERT_EQ(musaEventElapsedTime(&ms, begin, end), musaSuccess);
  LOG(INFO) << "[GROUPED_MODEL1_BENCH] C128+C4+C128 q=6 kv=2500 mean_us=" << ms * 10.0f;
  ASSERT_EQ(musaEventDestroy(begin), musaSuccess);
  ASSERT_EQ(musaEventDestroy(end), musaSuccess);
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
  ASSERT_EQ(musaGraphExecDestroy(second_exec), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(second_graph), musaSuccess);
  deepseek_v4_mate_sparse_attention_set_metadata_group(0);
  deepseek_v4_mate_sparse_attention_set_metadata_key(std::numeric_limits<int64_t>::min());
}

TEST(DeepseekV4MateSparseAttentionTest, PagedSasMatchesNoncausalDenseWindow) {
  int32_t devices = 0;
  ASSERT_EQ(musaGetDeviceCount(&devices), musaSuccess);
  ASSERT_GT(devices, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int64_t kQueries = 5;
  constexpr int64_t kWidth = 576;
  for (const torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
    auto source = torch::arange(4 * 128 * kWidth, fp).mul_(0.017f).sin_()
                      .reshape({4, 128, 1, kWidth}).to(dtype);
    auto cache = source.clone();
    auto query_source = torch::arange(kQueries * kHeads * kHeadDim, fp)
                            .mul_(0.011f).sin_().mul_(0.1f)
                            .view({kQueries, kHeads, kHeadDim}).to(dtype);
    auto q = query_source.clone();
    auto output = torch::empty_like(q);
    auto sink = torch::zeros({kHeads}, fp);
    for (const torch::ScalarType index_dtype : {torch::kInt32, torch::kInt64}) {
      auto table = torch::zeros({2, 257}, fp.dtype(index_dtype));
      for (const int64_t kv_len : {1, 31, 127, 128, 129, 133, 439, 2505, 32768}) {
        SCOPED_TRACE(::testing::Message() << "kv=" << kv_len << " dtype=" << dtype
                     << " index_dtype=" << index_dtype);
        auto execute = [&] {
          return deepseek_v4_paged_attention_sas(
              q, cache, table, /*sequence_index=*/1, kv_len, /*window_size=*/133,
              sink, kSoftmaxScale, output);
        };
        musaGraph_t graph = nullptr;
        musaGraphExec_t executable = nullptr;
        stream.synchronize();
        ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal), musaSuccess);
        execute();
        ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
        ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
        for (int64_t frame = 0; frame < 3; ++frame) {
          table.select(0, 0).fill_(3);
          table.select(0, 1).copy_(torch::arange(257, table.options()).add_(frame).remainder_(4));
          cache.copy_(source);
          q.copy_(query_source.roll({frame}, {0}));
          const int64_t visible = std::min<int64_t>(kv_len, 133);
          std::vector<int64_t> physical;
          physical.reserve(visible);
          for (int64_t logical = kv_len - visible; logical < kv_len; ++logical) {
            physical.emplace_back(((logical / 128 + frame) % 4) * 128 + logical % 128);
          }
          auto indices = torch::tensor(physical, fp.dtype(torch::kInt64));
          if (frame == 1) {
            // With zero Q, excluding later diffusion rows changes the mean.
            // This protects SAS's all-query noncausal semantics, especially row0.
            q.zero_();
            for (int64_t i = std::max<int64_t>(visible - kQueries, 0); i < visible; ++i) {
              cache.view({512, kWidth}).select(0, physical[i]).fill_(12.0f);
            }
          }
          auto selected = cache.view({512, kWidth}).index_select(0, indices)
                              .narrow(1, 0, kHeadDim).to(torch::kFloat32);
          auto qf = q.to(torch::kFloat32).permute({1, 0, 2}).contiguous();
          auto scores = torch::matmul(qf, selected.transpose(0, 1)).mul_(kSoftmaxScale);
          auto mass = torch::exp(scores - std::get<0>(scores.max(-1, true)));
          auto denominator = mass.sum(-1, true) +
              torch::exp(sink.view({kHeads, 1, 1}) - std::get<0>(scores.max(-1, true)));
          auto expected = torch::matmul(mass / denominator, selected)
                              .permute({1, 0, 2}).contiguous();
          ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
          stream.synchronize();
          const float error = (output.to(torch::kFloat32) - expected)
                                  .abs().max().cpu().item<float>();
          EXPECT_LT(error, dtype == torch::kFloat32 ? 2.0e-5f : 0.004f);
        }
        if (kv_len == 2505 && dtype == torch::kBFloat16 && index_dtype == torch::kInt32) {
          musaEvent_t begin = nullptr;
          musaEvent_t end = nullptr;
          ASSERT_EQ(musaEventCreate(&begin), musaSuccess);
          ASSERT_EQ(musaEventCreate(&end), musaSuccess);
          ASSERT_EQ(musaEventRecord(begin, stream.stream()), musaSuccess);
          for (int32_t repeat = 0; repeat < 200; ++repeat) {
            ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
          }
          ASSERT_EQ(musaEventRecord(end, stream.stream()), musaSuccess);
          ASSERT_EQ(musaEventSynchronize(end), musaSuccess);
          float elapsed_ms = 0.0f;
          ASSERT_EQ(musaEventElapsedTime(&elapsed_ms, begin, end), musaSuccess);
          LOG(INFO) << "[PAGED_SAS_BENCH] q=5 kv=2505 window=133 mean_us=" << elapsed_ms * 5.0f;
          ASSERT_EQ(musaEventDestroy(begin), musaSuccess);
          ASSERT_EQ(musaEventDestroy(end), musaSuccess);
        }
        ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
        ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
      }
    }
  }
}

// Negative control: a correct compression kernel does not make speculative
// state transactional. The integration must restore the accepted prefix, not
// reuse the recurrent state produced after all six proposed tokens.
TEST(DeepseekV4MateSparseAttentionTest,
     RejectedProposalStateNeedsAcceptedPrefixRestore) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int64_t kDim = 16;
  constexpr int64_t kTokens = 6;
  for (const int64_t ratio : {int64_t{4}, int64_t{128}}) {
    const int64_t coeff = ratio == 4 ? 2 : 1;
    const int64_t width = coeff * kDim;
    const int64_t rows = ratio * coeff;
    const int64_t start = ratio - 3;
    auto initial = torch::arange(rows * width, options)
                       .reshape({rows, width}).mul_(0.027f).sin_();
    auto initial_score = initial.mul(0.2f);
    auto first = torch::arange(kTokens * width, options)
                     .reshape({kTokens, width}).mul_(0.013f).cos_();
    auto second = first.mul(-0.7f);
    auto ape = torch::zeros({ratio, width}, options);
    auto run = [&](torch::Tensor state, torch::Tensor score,
                   const torch::Tensor& steps, int64_t position) {
      auto device_position = torch::tensor({position}, options.dtype(torch::kInt64));
      auto output = torch::empty({2, kDim}, options);
      deepseek_v4_compress_decode_sequence_graph(
          state, score, steps, steps.mul(0.1f), ape, ratio, kDim,
          device_position, output);
      return output;
    };
    auto stale_state = initial.clone();
    auto stale_score = initial_score.clone();
    run(stale_state, stale_score, first, start);
    // Only the first proposal was accepted. The next verify overwrites the
    // rejected suffix, but cannot undo the earlier recurrent pooling/shift.
    auto stale_output = run(stale_state, stale_score, second, start + 1);
    auto correct_state = initial.clone();
    auto correct_score = initial_score.clone();
    run(correct_state, correct_score, first.narrow(0, 0, 1), start);
    auto correct_output = run(correct_state, correct_score, second, start + 1);
    stream.synchronize();
    const float difference = (stale_output - correct_output)
                                 .abs().max().to(torch::kCPU).item<float>();
    LOG(INFO) << "[REJECTED_PREFIX_NEGATIVE_CONTROL] ratio=" << ratio
              << " accepted=1 start=" << start
              << " stale_output_max_abs=" << difference;
    EXPECT_GT(difference, 1.0e-4f);
  }
}

TEST(DeepseekV4MateSparseAttentionTest, ScalarPagedOddHistoryMatchesDense) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto values = torch::arange(4 * 128 * kHeadDim, fp)
                    .mul_(0.013f).sin_().reshape({4, 128, 1, kHeadDim})
                    .to(torch::kBFloat16);
  auto compressed = values.add(0.25f);
  auto q = torch::arange(kHeads * kHeadDim, fp).mul_(0.037f).sin_().mul_(0.1f)
               .view({1, kHeads, kHeadDim}).to(torch::kBFloat16);
  auto output = torch::empty_like(q);
  auto sink = torch::zeros({kHeads}, fp);
  auto kv_len = torch::tensor({2500}, fp.dtype(torch::kInt32));
  auto logical = torch::arange(2500 - 128, 2500, fp.dtype(torch::kInt64));
  auto physical = logical.remainder(4 * 128);
  auto selected_swa = values.view({4 * 128, kHeadDim}).index_select(0, physical);
  for (const torch::ScalarType index_dtype : {torch::kInt32, torch::kInt64}) {
    auto table = torch::arange(64, fp.dtype(index_dtype)).remainder(4).view({1, 64});
    auto cmp_table = torch::tensor({2}, fp.dtype(index_dtype)).view({1, 1});
    for (const int64_t compressed_len : {int64_t{1}, int64_t{3}, int64_t{19}}) {
      auto actual = deepseek_v4_paged_attention_decode_with_paged_compressed_kv(
          q, values, table, /*sequence_index=*/0, /*kv_len=*/2500, kv_len,
          /*window_size=*/128, torch::Tensor(), compressed, cmp_table,
          /*compressed_sequence_index=*/0, compressed_len, sink, kSoftmaxScale,
          std::nullopt, output);
      auto selected_cmp = compressed.view({4 * 128, kHeadDim}).narrow(
          /*dim=*/0, /*start=*/2 * 128, compressed_len);
      auto selected = torch::cat({selected_swa, selected_cmp}, /*dim=*/0)
                          .to(torch::kFloat32);
      auto scores = torch::matmul(q[0].to(torch::kFloat32), selected.transpose(0, 1))
                        .mul_(kSoftmaxScale);
      auto probs = torch::softmax(torch::cat({scores, sink.view({kHeads, 1})}, 1), 1);
      auto reference = torch::matmul(probs.narrow(1, 0, selected.size(0)), selected);
      stream.synchronize();
      EXPECT_LT((actual[0].to(torch::kFloat32) - reference).abs().max()
                    .to(torch::kCPU).item<float>(), 2.0e-3f);
    }
  }
}

// Diagnostic negative control for the legacy scalar fallback. Its explicit
// compressed length is a host scalar, whereas only the SWA length changes on
// replay. A caller must not reuse that capture for a shorter compressed prefix.
TEST(DeepseekV4MateSparseAttentionTest,
     CapturedScalarC4CapacityIsNotDynamicValidity) {
  int32_t device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto cache = torch::ones({1, 128, 1, kHeadDim}, fp);
  auto compressed = torch::ones_like(cache);
  auto table = torch::zeros({1, 1}, fp.dtype(torch::kInt32));
  auto lengths = torch::full({1}, 112, fp.dtype(torch::kInt32));
  auto q = torch::zeros({1, kHeads, kHeadDim}, fp);
  auto sink = torch::full({kHeads}, -INFINITY, fp);
  auto output = torch::empty_like(q);
  auto attend = [&] {
    return deepseek_v4_paged_attention_decode_with_paged_compressed_kv(
        q, cache, table, 0, 112, lengths, 128, torch::Tensor(),
        compressed, table, 0, 28, sink, kSoftmaxScale, std::nullopt, output);
  };
  ASSERT_TRUE(attend().defined());
  stream.synchronize();
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                   musaStreamCaptureModeThreadLocal), musaSuccess);
  attend();
  ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
  ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
  stream.synchronize();
  EXPECT_LT((output - 1.0f).abs().max().cpu().item<float>(), 1.0e-5f);

  lengths.fill_(67);
  // Only floor(67/4)=16 compressed rows belong to the current request.
  // The following twelve rows stand for another request's leftover cache.
  compressed.view({128, kHeadDim}).narrow(0, 16, 12).fill_(101.0f);
  ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
  stream.synchronize();
  const float mismatch = (output - 1.0f).abs().max().cpu().item<float>();
  LOG(INFO) << "[C4_STATIC_LENGTH_NEGATIVE_CONTROL] capture_kv=112 replay_kv=67"
            << " captured_compressed=28 valid_compressed=16 max_abs=" << mismatch;
  // Passing this negative control demonstrates the limitation, not a fix.
  EXPECT_GT(mismatch, 5.0f);
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

TEST(DeepseekV4MateSparseAttentionTest,
     CapturedCompressionReplaysChangedPositionsAndClearsPadding) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const c10::musa::MUSAStream stream =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());
  constexpr int64_t kTokens = 6;
  constexpr int64_t kStateHeadDim = 128;
  const auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  for (const int64_t ratio : {int64_t{4}, int64_t{128}}) {
    const int64_t coeff = ratio == 4 ? 2 : 1;
    const int64_t width = coeff * kStateHeadDim;
    const int64_t capacity = (kTokens + ratio - 1) / ratio;
    for (const torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
      auto initial_kv = torch::arange(ratio * coeff * width, fp32_options)
                            .reshape({ratio * coeff, width}).mul_(0.001f).sin_();
      auto initial_score = initial_kv.mul(0.17f);
      auto kv_steps = torch::arange(kTokens * width, fp32_options)
                          .reshape({kTokens, width}).mul_(0.017f).sin_().to(dtype);
      auto score_steps = kv_steps.mul(0.25f);
      auto ape = torch::arange(ratio * width, fp32_options)
                     .reshape({ratio, width}).mul_(0.003f).sin_();
      auto replay_kv = initial_kv.clone();
      auto replay_score = initial_score.clone();
      auto output = torch::empty({capacity, kStateHeadDim}, kv_steps.options());
      for (const torch::ScalarType position_dtype : {torch::kInt32, torch::kInt64}) {
        auto position = torch::zeros({1}, fp32_options.dtype(position_dtype));
        deepseek_v4_compress_decode_sequence_graph(
            replay_kv, replay_score, kv_steps, score_steps, ape, ratio,
            kStateHeadDim, position, output);
        ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
        musaGraph_t graph = nullptr;
        musaGraphExec_t executable = nullptr;
        ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                         musaStreamCaptureModeThreadLocal),
                  musaSuccess);
        deepseek_v4_compress_decode_sequence_graph(
            replay_kv, replay_score, kv_steps, score_steps, ape, ratio,
            kStateHeadDim, position, output);
        ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
        size_t node_count = 0;
        ASSERT_EQ(musaGraphGetNodes(graph, nullptr, &node_count), musaSuccess);
        ASSERT_GT(node_count, 0);
        ASSERT_EQ(musaGraphInstantiate(&executable, graph, /*flags=*/0), musaSuccess);
        // One graph, same pointers; updates include both compression and
        // page boundaries and the historic counting 439/440 transition.
        for (const int64_t start : {int64_t{0}, int64_t{1}, int64_t{2},
                                    int64_t{3}, int64_t{125}, int64_t{127},
                                    int64_t{128}, int64_t{439}, int64_t{440},
                                    int64_t{1023}, int64_t{4095}}) {
          SCOPED_TRACE(::testing::Message() << "ratio=" << ratio
                       << " dtype=" << dtype << " start=" << start);
          replay_kv.copy_(initial_kv);
          replay_score.copy_(initial_score);
          output.fill_(97.0f);
          position.fill_(start);
          stream.synchronize();
          ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
          stream.synchronize();
          auto expected_kv = initial_kv.clone();
          auto expected_score = initial_score.clone();
          auto expected_output = torch::zeros_like(output);
          int64_t output_row = 0;
          for (int64_t token = 0; token < kTokens; ++token) {
            auto step_output = torch::empty({1, kStateHeadDim}, kv_steps.options());
            auto result = deepseek_v4_compress_decode(
                expected_kv, expected_score, kv_steps.narrow(0, token, 1),
                score_steps.narrow(0, token, 1), ape, ratio, kStateHeadDim,
                coeff, start + token, /*compute_boundary_output=*/true,
                step_output);
            if (result.defined()) {
              expected_output.narrow(0, output_row++, 1).copy_(result);
            }
          }
          ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
          EXPECT_TRUE(torch::allclose(replay_kv.to(torch::kCPU),
                                      expected_kv.to(torch::kCPU), 1e-6, 1e-6));
          EXPECT_TRUE(torch::allclose(replay_score.to(torch::kCPU),
                                      expected_score.to(torch::kCPU), 1e-6, 1e-6));
          EXPECT_TRUE(torch::allclose(output.to(torch::kCPU),
                                      expected_output.to(torch::kCPU), 1e-6, 1e-6));
        }
        ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
        ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
      }
    }
  }
}

TEST(DeepseekV4MateSparseAttentionTest,
     CapturedIndexerUpdatesLengthsAndBlockTables) {
  int devices = 0;
  if (musaGetDeviceCount(&devices) != musaSuccess || devices == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto i32 = f32.dtype(torch::kInt32);
  constexpr int64_t capacity = 1024;
  auto query = torch::arange(64 * 128, f32).mul_(0.137f).sin_()
                   .reshape({1, 64, 128}).to(torch::kBFloat16);
  auto weights = torch::arange(64, f32).mul_(0.037f).cos_().abs_().add_(0.1f)
                     .reshape({1, 64});
  auto cache = torch::arange(capacity * 128, f32).mul_(0.043f).sin_()
                   .mul_(113).round_().to(torch::kInt8).reshape({8, 128, 128});
  auto scales = torch::full({8, 128}, 0.015625, f32.dtype(torch::kFloat16));
  auto compressed_table = torch::arange(8, i32).reshape({1, 8});
  auto swa_table = torch::arange(32, i32).reshape({1, 32});
  auto live_kv = torch::full({1}, 101, i32);
  auto logical = torch::empty({1, 512}, i32);
  auto physical = torch::empty_like(logical);
  auto extra = torch::empty({1, 128}, i32);
  auto run = [&] {
    return deepseek_v4_indexer_topk_graph(
        query, weights, cache, scales, compressed_table, 0, capacity, 512,
        logical, compressed_table, swa_table, 128, 128, physical, extra, live_kv);
  };
  run();
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal),
            musaSuccess);
  run();
  ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
  size_t nodes = 0;
  ASSERT_EQ(musaGraphGetNodes(graph, nullptr, &nodes), musaSuccess);
  EXPECT_GT(nodes, 0);
  // Same graph and buffer addresses; expand, cross TopK=512, contract, and
  // recycle request/page identities. No reference answer or model is involved.
  int replay = 0;
  for (const int64_t length : {101, 118, 439, 440, 2047, 2048, 2052,
                               2500, 4096, 112, 0}) {
    SCOPED_TRACE(::testing::Message() << "kv=" << length);
    auto cmp_host = torch::remainder(torch::arange(8, torch::kInt32) + replay, 8)
                        .reshape({1, 8});
    auto swa_host = torch::remainder(torch::arange(32, torch::kInt32) + replay, 32)
                        .reshape({1, 32});
    compressed_table.copy_(cmp_host);
    swa_table.copy_(swa_host);
    live_kv.fill_(length);
    logical.fill_(123456);
    physical.fill_(123456);
    extra.fill_(123456);
    stream.synchronize();
    ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
    stream.synchronize();
    auto got = logical.cpu();
    auto got_physical = physical.cpu();
    auto got_extra = extra.cpu();
    const int32_t* ids = got.data_ptr<int32_t>();
    const int32_t* mapped = got_physical.data_ptr<int32_t>();
    const int32_t* cmp = cmp_host.data_ptr<int32_t>();
    int64_t valid = 0;
    for (int64_t j = 0; j < 512; ++j) {
      if (ids[j] >= 0) {
        ++valid;
        EXPECT_LT(ids[j], length / 4);
        EXPECT_EQ(mapped[j], cmp[ids[j] / 128] * 128 + ids[j] % 128);
      } else {
        EXPECT_EQ(mapped[j], -1);
      }
    }
    EXPECT_EQ(valid, std::min<int64_t>(length / 4, 512));
    LOG(INFO) << "[DSV4_GRAPH_INDEXER_UNIT] replay=" << replay
              << " kv=" << length << " valid=" << valid;
    ++replay;
  }
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

TEST(DeepseekV4MateSparseAttentionTest, BatchedIndexerCausalRowsAndGraphReplay) {
  int devices = 0;
  ASSERT_EQ(musaGetDeviceCount(&devices), musaSuccess);
  ASSERT_GT(devices, 0);
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto i32 = f32.dtype(torch::kInt32);
  constexpr int64_t kCapacity = 1024;
  const std::vector<int32_t> lengths = {0, 127, 439, 2047, 2048,
                                       2052, 2500, 4096, 128};
  for (const int64_t rows : {int64_t{2}, int64_t{6}}) {
    for (const torch::ScalarType table_type : {torch::kInt32, torch::kInt64}) {
      auto query = torch::arange(rows * 64 * 128, f32).mul_(0.137f).sin_()
                       .reshape({rows, 64, 128}).to(torch::kBFloat16);
      auto weights = torch::arange(rows * 64, f32).mul_(0.037f).cos_()
                         .abs_().add_(0.1f).reshape({rows, 64});
      auto cache = torch::arange(kCapacity * 128, f32).mul_(0.043f).sin_()
                       .mul_(113).round_().to(torch::kInt8)
                       .reshape({8, 128, 128});
      auto scales = torch::full({8, 128}, 0.015625, f32.dtype(torch::kFloat16));
      auto cmp = torch::arange(8, f32.dtype(table_type)).reshape({1, 8});
      auto swa = torch::arange(32, f32.dtype(table_type)).reshape({1, 32});
      auto live = torch::full({rows}, 101, i32);
      auto logical = torch::empty({rows, 512}, i32);
      auto physical = torch::empty_like(logical);
      auto extra = torch::empty({rows, 128}, i32);
      auto scalar_logical = torch::empty_like(logical);
      auto scalar_physical = torch::empty_like(physical);
      auto scalar_extra = torch::empty_like(extra);
      auto batched = [&] {
        deepseek_v4_indexer_topk_graph(query, weights, cache, scales, cmp,
            0, kCapacity, 512, logical, cmp, swa, 128, 128,
            physical, extra, live);
      };
      auto scalar = [&] {
        for (int64_t row = 0; row < rows; ++row) {
          deepseek_v4_indexer_topk_graph(
              query.narrow(0, row, 1), weights.narrow(0, row, 1), cache,
              scales, cmp, 0, kCapacity, 512, scalar_logical.narrow(0, row, 1),
              cmp, swa, 128, 128, scalar_physical.narrow(0, row, 1),
              scalar_extra.narrow(0, row, 1), live.narrow(0, row, 1));
        }
      };
      batched();
      scalar();
      stream.synchronize();
      musaGraph_t graph = nullptr;
      musaGraphExec_t executable = nullptr;
      ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal),
                musaSuccess);
      batched();
      ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
      ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
      const auto cache_cpu = cache.cpu().reshape({kCapacity, 128}).to(torch::kFloat32);
      for (int64_t phase = 0; phase < 6; ++phase) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " phase=" << phase
                                       << " table=" << table_type);
        std::vector<int32_t> host_lengths;
        host_lengths.reserve(rows);
        for (int64_t row = 0; row < rows; ++row) {
          host_lengths.emplace_back(lengths[(phase + row) % lengths.size()]);
        }
        live.copy_(torch::tensor(host_lengths, torch::kInt32));
        auto cmp_cpu = torch::arange(8, torch::kInt64).add_(phase).remainder_(8);
        auto swa_cpu = torch::arange(32, torch::kInt64).add_(phase).remainder_(32);
        cmp.copy_(cmp_cpu.view({1, 8}));
        swa.copy_(swa_cpu.view({1, 32}));
        // Change row identities without changing captured buffer addresses.
        query.copy_(query.flip({0}).contiguous());
        weights.copy_(weights.flip({0}).contiguous());
        logical.fill_(123456);
        physical.fill_(123456);
        extra.fill_(123456);
        stream.synchronize();
        ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
        stream.synchronize();
        scalar();
        stream.synchronize();
        const auto got = logical.cpu();
        const auto mapped = physical.cpu();
        const auto got_extra = extra.cpu();
        const auto q_cpu = query.cpu().to(torch::kFloat32);
        const auto w_cpu = weights.cpu();
        const auto scalar_ids = scalar_logical.cpu();
        for (int64_t row = 0; row < rows; ++row) {
          const int64_t valid = host_lengths[row] / 4;
          const auto selected = got[row].masked_select(got[row].ge(0)).to(torch::kLong);
          EXPECT_EQ(selected.numel(), std::min<int64_t>(valid, 512));
          if (selected.numel() > 0) {
            EXPECT_LT(selected.max().item<int64_t>(), valid);
            auto sorted = std::get<0>(selected.sort());
            EXPECT_TRUE(sorted.numel() < 2 ||
                        sorted.slice(0, 1).ne(sorted.slice(0, 0, -1)).all().item<bool>());
            auto expected_map = cmp_cpu.index_select(0, selected.floor_divide(128))
                                    .mul(128).add(selected.remainder(128));
            EXPECT_TRUE(torch::equal(mapped[row].narrow(0, 0, selected.numel()).to(torch::kLong),
                                     expected_map));
          }
          if (valid <= 512) {
            EXPECT_TRUE(torch::equal(got[row], scalar_ids[row]));
          } else {
            // Independent FP32 score oracle. Radix tie ordering is not part
            // of the API; verify selected scores, not arbitrary tie order.
            auto ids = torch::arange(valid, torch::kLong);
            auto physical_ids = cmp_cpu.index_select(0, ids.floor_divide(128))
                                    .mul(128).add(ids.remainder(128));
            auto keys = cache_cpu.index_select(0, physical_ids).mul(0.015625);
            auto scores = (torch::matmul(q_cpu[row], keys.transpose(0, 1)).relu() *
                           w_cpu[row].unsqueeze(-1)).sum(0);
            const float cutoff = std::get<0>(scores.topk(512)).min().item<float>();
            EXPECT_GE(scores.index_select(0, selected).min().item<float>(), cutoff - 0.02f);
          }
          const int64_t visible = std::min<int64_t>(host_lengths[row], 128);
          for (int64_t column = 0; column < 128; ++column) {
            const int64_t pos = host_lengths[row] - visible + column;
            const int64_t expected = column < visible
                ? swa_cpu[pos / 128].item<int64_t>() * 128 + pos % 128 : -1;
            EXPECT_EQ(got_extra[row][column].item<int32_t>(), expected);
          }
        }
      }
      if (rows == 6 && table_type == torch::kInt32) {
        live.copy_(torch::arange(2500, 2500 + rows, i32));
        musaGraph_t scalar_graph = nullptr;
        musaGraphExec_t scalar_exec = nullptr;
        stream.synchronize();
        ASSERT_EQ(musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal),
                  musaSuccess);
        scalar();
        ASSERT_EQ(musaStreamEndCapture(stream.stream(), &scalar_graph), musaSuccess);
        ASSERT_EQ(musaGraphInstantiate(&scalar_exec, scalar_graph, 0), musaSuccess);
        for (int32_t mode = 0; mode < 2; ++mode) {
          musaGraphExec_t current = mode == 0 ? scalar_exec : executable;
          musaEvent_t begin = nullptr;
          musaEvent_t end = nullptr;
          ASSERT_EQ(musaEventCreate(&begin), musaSuccess);
          ASSERT_EQ(musaEventCreate(&end), musaSuccess);
          for (int32_t warm = 0; warm < 5; ++warm) {
            ASSERT_EQ(musaGraphLaunch(current, stream.stream()), musaSuccess);
          }
          ASSERT_EQ(musaEventRecord(begin, stream.stream()), musaSuccess);
          for (int32_t replay = 0; replay < 50; ++replay) {
            ASSERT_EQ(musaGraphLaunch(current, stream.stream()), musaSuccess);
          }
          ASSERT_EQ(musaEventRecord(end, stream.stream()), musaSuccess);
          ASSERT_EQ(musaEventSynchronize(end), musaSuccess);
          float ms = 0.0f;
          ASSERT_EQ(musaEventElapsedTime(&ms, begin, end), musaSuccess);
          LOG(INFO) << "[BATCHED_INDEXER_BENCH] mode=" << (mode == 0 ? "scalar6" : "batch6")
                    << " kv=2500 capacity=1024 mean_us=" << ms * 1000.0f / 50;
          ASSERT_EQ(musaEventDestroy(begin), musaSuccess);
          ASSERT_EQ(musaEventDestroy(end), musaSuccess);
        }
        ASSERT_EQ(musaGraphExecDestroy(scalar_exec), musaSuccess);
        ASSERT_EQ(musaGraphDestroy(scalar_graph), musaSuccess);
      }
      ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
      ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
    }
  }
}

TEST(DeepseekV4MateSparseAttentionTest, CapturedPackIgnoresStaleEagerEvent) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const c10::musa::MUSAStream stream =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  auto values = deterministic_values(/*rows=*/2, device);
  auto slots = torch::tensor({1, -1}, torch::kInt32).to(device);
  auto cache = torch::zeros({1, kBlockSize, 1, kPackedBytes},
                            torch::TensorOptions().dtype(torch::kUInt8).device(device));
  auto expected = torch::zeros_like(cache);
  // The eager completion event exists before graph capture on another stream.
  deepseek_v4_pack_fp8_cache(cache, slots, values);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  c10::StreamGuard stream_guard(stream.unwrap());
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                   musaStreamCaptureModeThreadLocal), musaSuccess);
  deepseek_v4_wait_model1_pack_event();
  deepseek_v4_pack_fp8_cache(cache, slots, values);
  deepseek_v4_wait_model1_pack_event();
  ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
  size_t node_count = 0;
  ASSERT_EQ(musaGraphGetNodes(graph, nullptr, &node_count), musaSuccess);
  ASSERT_GT(node_count, 0);
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, /*flags=*/0), musaSuccess);
  for (const int32_t slot : {2, 127, 0}) {
    values.mul_(0.5f);
    slots.narrow(0, 0, 1).fill_(slot);
    cache.zero_();
    expected.zero_();
    stream.synchronize();
    ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
    stream.synchronize();
    deepseek_v4_pack_fp8_cache(expected, slots, values);
    ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
    EXPECT_TRUE(torch::equal(cache.to(torch::kCPU), expected.to(torch::kCPU)));
  }
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

TEST(DeepseekV4MateSparseAttentionTest, RollingSwaBoundaryMappingMatchesDense) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  if (!std::ifstream(path_to_uri_so_lib(
          "dsa_model1_h8_topk512_extra128_bs128"))
           .good()) {
    GTEST_SKIP() << "MATE MODEL1 h8 decode artifact unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  c10::musa::MUSAStream stream =
      c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());

  auto values = deterministic_values(4 * kBlockSize, device);
  auto q2 = deterministic_values(2 * kHeads, device).view({2, kHeads, kHeadDim});
  auto logical_main = torch::tensor(
      {{0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14, 15}},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto compressed_table = torch::tensor(
      {{1, 2}},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto rolling_swa_table = torch::tensor(
      {{-1, 2, 3}},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto mapped = deepseek_v4_prepare_mate_dsa_indices(logical_main,
                                                     compressed_table,
                                                     rolling_swa_table,
                                                     /*sequence_index=*/0,
                                                     /*kv_len=*/258,
                                                     /*window_size=*/128,
                                                     /*block_size=*/128);
  auto result = run_mate_case(q2,
                              values,
                              std::get<0>(mapped),
                              /*topk_length=*/8,
                              std::get<1>(mapped),
                              /*extra_topk_length=*/128);
  stream.synchronize();
  expect_close(result, "q2_swa_boundary_minus1_2_3");
}

TEST(DeepseekV4MateSparseAttentionTest, CapturedQ6PackedSwaPublication) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  if (!std::ifstream(path_to_uri_so_lib(
          "dsa_model1_h8_topk512_extra128_bs128")).good()) {
    GTEST_SKIP() << "MATE MODEL1 h8 decode artifact unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  TvmffiStreamOverrideGuard ffi_guard(
      device, reinterpret_cast<void*>(stream.stream()));
  constexpr int64_t q_tokens = 6;
  constexpr int64_t page_stride =
      ((kBlockSize * kPackedBytes + 575) / 576) * 576;
  const auto bytes = torch::TensorOptions().dtype(torch::kUInt8).device(device);
  auto storage = torch::zeros({2 * page_stride}, bytes);
  auto cache = storage.as_strided({2, kBlockSize, 1, kPackedBytes},
                                  {page_stride, kPackedBytes, kPackedBytes, 1});
  auto reference_storage = torch::zeros_like(storage);
  auto reference_cache = reference_storage.as_strided(
      cache.sizes(), cache.strides());
  auto values = torch::full({q_tokens, kHeadDim}, -1.0,
      bytes.dtype(torch::kBFloat16));
  auto slots = torch::arange(128, 128 + q_tokens,
                            bytes.dtype(torch::kInt32));
  auto q = torch::zeros({q_tokens, kHeads, kHeadDim}, values.options());
  auto sink = torch::full({kHeads}, -INFINITY, bytes.dtype(torch::kFloat32));
  auto main = torch::full({q_tokens, 512}, -1, bytes.dtype(torch::kInt32));
  auto extra = torch::full({q_tokens, 128}, -1, bytes.dtype(torch::kInt32));
  auto main_lengths = torch::zeros({q_tokens}, bytes.dtype(torch::kInt32));
  auto extra_lengths = torch::arange(1, q_tokens + 1,
                                    bytes.dtype(torch::kInt32));
  auto output = torch::empty_like(q);
  auto set_indices = [&](int32_t first_slot) {
    auto host = torch::full({q_tokens, 128}, -1, torch::kInt32);
    auto* ptr = host.data_ptr<int32_t>();
    for (int64_t row = 0; row < q_tokens; ++row) {
      for (int64_t col = 0; col <= row; ++col) {
        ptr[row * 128 + col] = first_slot + col;
      }
    }
    extra.copy_(host.to(device));
    slots.copy_(torch::arange(first_slot, first_slot + q_tokens,
                              torch::kInt32).to(device));
  };
  auto attend = [&] {
    return deepseek_v4_mate_sparse_attention_decode_batched(
        q, cache, main, main_lengths, cache, extra, extra_lengths,
        sink, kSoftmaxScale, output);
  };
  set_indices(128);
  deepseek_v4_pack_fp8_cache(cache, slots, values);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  ASSERT_TRUE(attend().defined());
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  begin_ffi_alloc_record();
  ASSERT_TRUE(attend().defined());
  auto allocations = end_ffi_alloc_record();
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                  musaStreamCaptureModeThreadLocal), musaSuccess);
  begin_ffi_alloc_replay(&allocations);
  deepseek_v4_pack_fp8_cache(cache, slots, values);
  auto captured = attend();
  const auto capture_status = musaStreamEndCapture(stream.stream(), &graph);
  end_ffi_alloc_replay();
  ASSERT_EQ(capture_status, musaSuccess);
  ASSERT_TRUE(captured.defined());
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
  // Different physical rows, including the 127/128 page boundary; reuse the
  // SAME values/slot/index buffers with changing data at every replay.
  for (const int32_t first_slot : {128, 125, 0}) {
    set_indices(first_slot);
    values.fill_(-1.0);
    deepseek_v4_pack_fp8_cache(cache, slots, values);
    ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
    ASSERT_TRUE(attend().defined());
    ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
    auto stale = output.to(torch::kCPU).clone();
    values.fill_(0.75);
    deepseek_v4_pack_fp8_cache(reference_cache, slots, values);
    ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
    auto expected = dense_reference(q, reference_cache, main, 0, extra,
                                     q_tokens, sink);
    // A test comparing against the same stale cache would incorrectly pass.
    EXPECT_GT((stale.to(torch::kFloat32) - expected.to(torch::kFloat32))
                  .abs().max().item<float>(), 1.0f);
    ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
    stream.synchronize();
    expect_close({output, expected}, "graph_q6_swa_publication");
  }
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

TEST(DeepseekV4MateSparseAttentionTest, GroupedStateRestoreExactReplay) {
  int32_t count = 0;
  if (musaGetDeviceCount(&count) != musaSuccess || count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  // Same Graph addresses; change row identity, logical holes, physical page
  // order and previous-page position at each replay. Roles do NOT share tables.
  for (const int64_t role_count : {2, 4}) {
    for (const int64_t capacity : {1, 257, 8193}) {
      for (const auto position_dtype : {torch::kInt32, torch::kInt64}) {
        std::vector<torch::Tensor> caches, tables, slots, outputs;
        std::vector<torch::Tensor> scalar_slots, scalar_outputs;
        std::vector<int64_t> rows;
        auto position = torch::zeros({1}, fp.dtype(position_dtype));
        for (int64_t role = 0; role < role_count; ++role) {
          const int64_t width = role < 2 ? 1024 : 256;
          const int64_t state_rows = role_count == 2 ? 128 : 8;
          rows.push_back(state_rows);
          caches.push_back((torch::arange(5 * 128 * width, torch::kFloat32) + role)
                               .reshape({5, 128, width}).to(device));
          tables.push_back(torch::full({2, capacity}, -1,
              fp.dtype(role % 2 == 0 ? torch::kInt32 : torch::kInt64)));
          slots.push_back(torch::empty({state_rows}, fp.dtype(torch::kInt64)));
          outputs.push_back(torch::empty({state_rows, width}, fp));
          scalar_slots.push_back(torch::empty_like(slots.back()));
          scalar_outputs.push_back(torch::empty_like(outputs.back()));
        }
        musaGraph_t graph = nullptr;
        musaGraphExec_t executable = nullptr;
        ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
        ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                         musaStreamCaptureModeThreadLocal), musaSuccess);
        deepseek_v4_state_gather_many(caches, tables, rows, slots, outputs, 1, position);
        for (int64_t role = 0; role < role_count; ++role) {
          deepseek_v4_state_gather(caches[role], tables[role], 1, rows[role], 128,
                                   scalar_slots[role], scalar_outputs[role], position);
        }
        ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
        ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
        for (int64_t frame = 0; frame < 8; ++frame) {
          const int64_t input_position = frame % 4 == 0 ? 0 :
              (frame % 4 == 1 ? 128 : (frame % 4 == 2 ? 129 : 440));
          position.fill_(input_position);
          for (int64_t role = 0; role < role_count; ++role) {
            auto host = torch::full({2, capacity}, -1, torch::kInt64);
            host[0].fill_(4);  // Wrong request identity would read page 4.
            const bool empty = (frame + role) % 4 == 0;
            const int64_t previous = (frame + role + 1) % 5;
            const int64_t tail = (frame + 2 * role) % 5;
            if (!empty) {
              host[1][0] = previous;
              host[1][capacity - 1] = tail;
            }
            tables[role].copy_(host.to(device, tables[role].scalar_type()));
            outputs[role].fill_(-19);
            scalar_outputs[role].fill_(-19);
            slots[role].fill_(-99);
            scalar_slots[role].fill_(-99);
          }
          ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
          stream.synchronize();
          for (int64_t role = 0; role < role_count; ++role) {
            SCOPED_TRACE("role=" + std::to_string(role) + " frame=" + std::to_string(frame));
            EXPECT_TRUE(torch::equal(slots[role].cpu(), scalar_slots[role].cpu()));
            EXPECT_TRUE(torch::equal(outputs[role].cpu(), scalar_outputs[role].cpu()));
            // Independent CPU oracle prevents two identically wrong kernels
            // from hiding a row/page mapping regression.
            const bool empty = (frame + role) % 4 == 0;
            const int64_t tail = (frame + 2 * role) % 5;
            const int64_t source = capacity > 1 && input_position > 0 &&
                (input_position - 1) / 128 == 0 ? (frame + role + 1) % 5 : tail;
            auto expected_slots = torch::arange(rows[role], torch::kInt64) +
                                  (empty ? -128 : tail * 128);
            auto expected = empty
                ? torch::full({rows[role], outputs[role].size(1)}, -19, torch::kFloat32)
                : caches[role][source].narrow(0, 0, rows[role]).cpu();
            EXPECT_TRUE(torch::equal(slots[role].cpu(), expected_slots));
            EXPECT_TRUE(torch::equal(outputs[role].cpu(), expected));
          }
        }
        ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
        ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
      }
    }
  }
}

TEST(DeepseekV4MateSparseAttentionTest, StateLookupCapacityAndReplay) {
  int count = 0;
  if (musaGetDeviceCount(&count) != musaSuccess || count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard guard(stream.unwrap());
  const auto fp = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto cache = torch::arange(5 * 128 * 1024, torch::kFloat32)
                   .reshape({5, 128, 1024}).to(device);
  auto slots = torch::empty({128}, fp.dtype(torch::kInt64));
  auto separate_slots = torch::empty_like(slots);
  auto output = torch::empty({128, 1024}, fp);
  auto position = torch::zeros({1}, fp.dtype(torch::kInt64));
  for (const auto dtype : {torch::kInt32, torch::kInt64}) {
    for (const int64_t capacity : {1, 3, 257, 8193}) {
      auto table = torch::full({2, capacity}, -1, fp.dtype(dtype));
      for (const int64_t state_rows : {8, 128}) {
        auto gather = [&] {
          return deepseek_v4_state_gather(cache, table, 1, state_rows, 128,
                                          slots, output, position);
        };
        musaGraph_t graph = nullptr;
        musaGraphExec_t executable = nullptr;
        ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
        ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                         musaStreamCaptureModeThreadLocal), musaSuccess);
        gather();
        deepseek_v4_state_slots(table, 1, state_rows, 128, separate_slots);
        ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
        ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
        for (const bool empty : {false, true, false}) {
          auto host = torch::full({2, capacity}, -1, torch::kInt64);
          // Row 0 is a distractor; row 1 has holes and a non-monotonic page
          // mapping. The last logical column is NOT the largest physical ID.
          host.select(0, 0).fill_(4);
          if (!empty) {
            host[1][0] = 4;
            host[1][capacity - 1] = 2;
          }
          table.copy_(host.to(device, dtype));
          position.fill_(1);  // Restore previous page 4, publish current page 2.
          output.fill_(-19);
          ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
          ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
          stream.synchronize();
          auto expected_slots = torch::arange(state_rows, torch::kInt64) +
                                (empty ? -128 : 256);
          EXPECT_TRUE(torch::equal(slots.narrow(0, 0, state_rows).cpu(),
                                   expected_slots));
          EXPECT_TRUE(torch::equal(separate_slots.narrow(0, 0, state_rows).cpu(),
                                   expected_slots));
          auto expected = empty
              ? torch::full({state_rows, 1024}, -19, torch::kFloat32)
              : cache.select(0, capacity == 1 ? 2 : 4)
                    .narrow(0, 0, state_rows).cpu();
          EXPECT_TRUE(torch::equal(output.narrow(0, 0, state_rows).cpu(), expected));
        }
        ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
        ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
      }
    }
  }
  // Same graph workload against old/new archives. Event timing excludes host
  // dispatch and does not assert hardware-specific performance thresholds.
  auto table = torch::full({1, 8193}, -1, fp.dtype(torch::kInt32));
  table[0][0] = 1;
  position.zero_();
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(musaStreamBeginCapture(stream.stream(),
                                   musaStreamCaptureModeThreadLocal), musaSuccess);
  for (int i = 0; i < 64; ++i) {
    deepseek_v4_state_gather(cache, table, 0, 8, 128, slots, output, position);
  }
  ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, 0), musaSuccess);
  ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
  stream.synchronize();
  musaEvent_t start, end;
  ASSERT_EQ(musaEventCreate(&start), musaSuccess);
  ASSERT_EQ(musaEventCreate(&end), musaSuccess);
  ASSERT_EQ(musaEventRecord(start, stream.stream()), musaSuccess);
  for (int repeat = 0; repeat < 10; ++repeat) {
    ASSERT_EQ(musaGraphLaunch(executable, stream.stream()), musaSuccess);
  }
  ASSERT_EQ(musaEventRecord(end, stream.stream()), musaSuccess);
  ASSERT_EQ(musaEventSynchronize(end), musaSuccess);
  float elapsed_ms = 0;
  ASSERT_EQ(musaEventElapsedTime(&elapsed_ms, start, end), musaSuccess);
  LOG(INFO) << "[STATE_LOOKUP_BENCH] capacity=8193 rows=8 calls=640 "
            << "mean_us=" << elapsed_ms * 1000.0 / 640;
  ASSERT_EQ(musaEventDestroy(start), musaSuccess);
  ASSERT_EQ(musaEventDestroy(end), musaSuccess);
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

}  // namespace
}  // namespace xllm::kernel::musa
