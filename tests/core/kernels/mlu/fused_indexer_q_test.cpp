/* Copyright 2026 The xLLM Authors.

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

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
struct Inputs {
  torch::Tensor qr, wq, positions, rope;
};

Inputs make_inputs(int64_t tokens, int64_t table_rows = 256) {
  torch::manual_seed(20261108);
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor angles =
      torch::arange(table_rows, torch::kFloat32).unsqueeze(1) *
      (torch::arange(32, torch::kFloat32) + 1)
          .repeat_interleave(2)
          .unsqueeze(0) *
      0.003;
  return {(torch::randn({tokens, 1280})).to(bf16).to(device),
          (torch::randn({4096, 1280}) / std::sqrt(1280.0)).to(bf16).to(device),
          (torch::arange(tokens, torch::kInt32) * 17)
              .remainder(table_rows)
              .to(device),
          torch::cat({angles.cos(), angles.sin()}, /*dim=*/1).to(device)};
}

torch::Tensor run(const Inputs& in) {
  return fused_indexer_q(in.qr, in.wq, in.positions, in.rope);
}

// The reference expresses the two rounding boundaries independently of the
// production kernel. The decomposed MLU baseline only supplies error budgets.
torch::Tensor reference(const Inputs& in, bool cpu) {
  const torch::Tensor qr = cpu ? in.qr.cpu().to(torch::kFloat64) : in.qr;
  const torch::Tensor wq = cpu ? in.wq.cpu().to(torch::kFloat64) : in.wq;
  const torch::Tensor rope = cpu ? in.rope.cpu().to(torch::kFloat64) : in.rope;
  const torch::Tensor positions =
      (cpu ? in.positions.cpu() : in.positions).to(torch::kInt64);
  torch::Tensor projected =
      torch::matmul(qr, wq.t()).to(torch::kBFloat16).view({-1, 32, 128});
  torch::Tensor output = projected.clone();
  const torch::Tensor values =
      projected.to(cpu ? torch::kFloat64 : torch::kFloat32);
  const torch::Tensor table = rope.index_select(0, positions).unsqueeze(1);
  const torch::Tensor even = values.slice(-1, 64, 128, 2);
  const torch::Tensor odd = values.slice(-1, 65, 128, 2);
  output.slice(-1, 64, 128, 2)
      .copy_(even * table.slice(-1, 0, 64, 2) -
             odd * table.slice(-1, 64, 128, 2));
  output.slice(-1, 65, 128, 2)
      .copy_(odd * table.slice(-1, 1, 64, 2) +
             even * table.slice(-1, 65, 128, 2));
  return output;
}

bool exact(const torch::Tensor& left, const torch::Tensor& right) {
  return torch::all((left == right) |
                    (torch::isnan(left) & torch::isnan(right)))
      .item<bool>();
}

void check_accuracy(const Inputs& in, const torch::Tensor& actual) {
  const torch::Tensor golden = reference(in, /*cpu=*/true).to(torch::kFloat64);
  const torch::Tensor base =
      reference(in, /*cpu=*/false).cpu().to(torch::kFloat64);
  const torch::Tensor result = actual.cpu().to(torch::kFloat64);
  ASSERT_TRUE(torch::equal(torch::isnan(golden), torch::isnan(result)));
  ASSERT_TRUE(torch::equal(torch::isposinf(golden), torch::isposinf(result)));
  ASSERT_TRUE(torch::equal(torch::isneginf(golden), torch::isneginf(result)));
  ASSERT_TRUE(torch::equal(torch::isnan(golden), torch::isnan(base)));
  ASSERT_TRUE(torch::equal(torch::isposinf(golden), torch::isposinf(base)));
  ASSERT_TRUE(torch::equal(torch::isneginf(golden), torch::isneginf(base)));
  const torch::Tensor finite = torch::isfinite(golden);
  const torch::Tensor g = golden.masked_select(finite);
  const torch::Tensor b = base.masked_select(finite);
  const torch::Tensor c = result.masked_select(finite);
  const torch::Tensor safe = torch::where(g == 0, 1e-9, g).abs();
  const double denominator1 = safe.sum().item<double>();
  const double denominator2 = safe.square().sum().item<double>();
  const double base1 = (b - g).abs().sum().item<double>() / denominator1;
  const double diff1 = (c - g).abs().sum().item<double>() / denominator1;
  const double base2 =
      std::sqrt((b - g).square().sum().item<double>() / denominator2);
  const double diff2 =
      std::sqrt((c - g).square().sum().item<double>() / denominator2);
  EXPECT_LE(diff1, 10 * std::max(base1, 1e-6));
  EXPECT_LE(diff2, 10 * std::max(base2, 1e-6));
  const int64_t greater = (c > g).sum().item<int64_t>();
  const int64_t less = (c < g).sum().item<int64_t>();
  const int64_t base_greater = (b > g).sum().item<int64_t>();
  const int64_t base_less = (b < g).sum().item<int64_t>();
  EXPECT_TRUE(
      greater + less < 100 || (greater > 0 && less > 0) ||
      (base_greater + base_less > 0 && (base_greater == 0 || base_less == 0)));
}

TEST(FusedIndexerQTest, MatchesReferenceAcrossDispatch) {
  for (const int64_t tokens :
       {1,   6,   12,  23,  24,  25,  27,  31,  32,   34,   40,  41,
        47,  48,  49,  65,  95,  96,  97,  127, 128,  129,  143, 144,
        145, 191, 192, 193, 257, 575, 576, 577, 1024, 2048, 4096}) {
    SCOPED_TRACE(tokens);
    const Inputs in = make_inputs(tokens);
    check_accuracy(in, run(in));
  }
}

TEST(FusedIndexerQTest, SupportsShortWidthsAndBatch96) {
  for (const int64_t batch : {1, 9, 17, 32, 65, 96}) {
    for (const int64_t width : {1, 2, 3, 6}) {
      SCOPED_TRACE(batch);
      SCOPED_TRACE(width);
      const Inputs in = make_inputs(batch * width);
      check_accuracy(in, run(in));
    }
  }
}

TEST(FusedIndexerQTest, PreservesInputsAndOutputStorage) {
  const Inputs in = make_inputs(/*tokens=*/34);
  const std::vector<torch::Tensor> saved{
      in.qr.clone(), in.wq.clone(), in.positions.clone(), in.rope.clone()};
  const torch::Tensor out = run(in);
  const torch::Tensor repeat = run(in);
  EXPECT_EQ(out.sizes(), (torch::IntArrayRef{34, 32, 128}));
  EXPECT_EQ(out.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(out.device(), in.qr.device());
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_NE(out.data_ptr(), repeat.data_ptr());
  EXPECT_TRUE(exact(out, repeat));
  const std::vector<torch::Tensor> inputs{in.qr, in.wq, in.positions, in.rope};
  for (int64_t i = 0; i < inputs.size(); ++i) {
    EXPECT_NE(out.data_ptr(), inputs[i].data_ptr());
    EXPECT_TRUE(torch::equal(inputs[i], saved[i]));
  }
}

TEST(FusedIndexerQTest, SupportsShortTableAndMaximumPosition) {
  Inputs in = make_inputs(/*tokens=*/3, /*table_rows=*/1);
  check_accuracy(in, run(in));
  in.rope = torch::zeros({1048576, 128}, in.rope.options());
  in.rope.slice(/*dim=*/1, /*start=*/0, /*end=*/64).fill_(1);
  in.positions.copy_(
      torch::tensor({1048575, 0, 1048575}, in.positions.options()));
  check_accuracy(in, run(in));
}

TEST(FusedIndexerQTest, PreservesRoundingAndNonfiniteValues) {
  Inputs in = make_inputs(/*tokens=*/6);
  in.qr.zero_();
  EXPECT_TRUE(
      torch::equal(run(in), torch::zeros({6, 32, 128}, in.qr.options())));
  in.qr.select(/*dim=*/1, /*index=*/0).fill_(1);
  in.qr.select(/*dim=*/1, /*index=*/1).fill_(std::ldexp(1.0, -8));
  in.wq.zero_();
  in.wq.slice(/*dim=*/1, /*start=*/0, /*end=*/2).fill_(1);
  const torch::Tensor out = run(in);
  EXPECT_TRUE(torch::equal(out.slice(-1, 0, 64),
                           torch::ones({6, 32, 64}, in.qr.options())));
  check_accuracy(in, out);
  in.wq.select(/*dim=*/0, /*index=*/0)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  in.wq.select(/*dim=*/0, /*index=*/1)
      .fill_(std::numeric_limits<float>::infinity());
  in.wq.select(/*dim=*/0, /*index=*/2)
      .fill_(-std::numeric_limits<float>::infinity());
  check_accuracy(in, run(in));
}

TEST(FusedIndexerQTest, UsesCurrentStreamAndReplaysUpdatedInputs) {
  for (const int64_t tokens : {24, 32, 34, 96, 97, 576, 577}) {
    SCOPED_TRACE(tokens);
    Inputs in = make_inputs(tokens);
    const std::vector<torch::Tensor> saved{
        in.qr.clone(), in.wq.clone(), in.positions.clone(), in.rope.clone()};
    torch_mlu::synchronize();
    torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(
        /*isHighPriority=*/false, in.qr.device().index()));
    for (int32_t i = 0; i < 10; ++i) {
      run(in);
    }
    torch_mlu::MLUGraph graph;
    graph.capture_begin();
    const torch::Tensor output = run(in);
    graph.capture_end();
    graph.replay();
    const torch::Tensor initial = output.clone();
    EXPECT_TRUE(exact(output, run(in)));
    for (int32_t update = 0; update < 7; ++update) {
      in.qr.copy_(saved[0]);
      in.wq.copy_(saved[1]);
      in.positions.copy_(saved[2]);
      in.rope.copy_(saved[3]);
      switch (update) {
        case 0:
          in.qr.mul_(0.5);
          break;
        case 1:
          in.wq.mul_(0.75);
          break;
        case 2:
          in.positions.fill_(1);
          break;
        case 3:
          in.rope.mul_(0.5);
          break;
        case 4:
          in.qr.slice(0, tokens / 2).zero_();
          break;
        case 5:
          in.qr.slice(0, 1).zero_();
          break;
        case 6:
          break;
      }
      graph.replay();
      EXPECT_TRUE(exact(output, run(in)));
      if (update < 6) {
        EXPECT_FALSE(exact(output, initial));
      }
    }
    graph.replay();
    EXPECT_TRUE(exact(initial, output));
    for (int32_t i = 0; i < 200; ++i) {
      graph.replay();
    }
    EXPECT_TRUE(exact(initial, output));
  }
}

// Opt-in latency evidence, excluded from default CTest correctness checks.
TEST(FusedIndexerQTest, DISABLED_ReportsPairedLatency) {
  for (const int64_t tokens :
       {6, 12, 24, 27, 34, 48, 96, 97, 576, 1024, 2048, 4096}) {
    const Inputs in = make_inputs(tokens);
    std::vector<float> fused, base;
    fused.reserve(3);
    base.reserve(3);
    for (int32_t path = 0; path < 2; ++path) {
      auto call = [&]() { return path == 0 ? run(in) : reference(in, false); };
      for (int32_t i = 0; i < 10; ++i) {
        call();
      }
      for (int32_t group = 0; group < 3; ++group) {
        cnrtNotifier_t start, end;
        ASSERT_EQ(cnrtNotifierCreate(&start), cnrtSuccess);
        ASSERT_EQ(cnrtNotifierCreate(&end), cnrtSuccess);
        ASSERT_EQ(cnrtPlaceNotifier(start, torch_mlu::getCurMLUStream()),
                  cnrtSuccess);
        for (int32_t i = 0; i < 100; ++i) {
          call();
        }
        ASSERT_EQ(cnrtPlaceNotifier(end, torch_mlu::getCurMLUStream()),
                  cnrtSuccess);
        torch_mlu::synchronize();
        float duration = 0;
        ASSERT_EQ(cnrtNotifierDuration(start, end, &duration), cnrtSuccess);
        (path == 0 ? fused : base).emplace_back(duration / 100);
        ASSERT_EQ(cnrtNotifierDestroy(start), cnrtSuccess);
        ASSERT_EQ(cnrtNotifierDestroy(end), cnrtSuccess);
      }
    }
    std::sort(fused.begin(), fused.end());
    std::sort(base.begin(), base.end());
    LOG(INFO) << "T=" << tokens << " fused_us=" << fused[1]
              << " decomposed_us=" << base[1];
  }
}

TEST(FusedIndexerQTest, RejectsInvalidContracts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const Inputs in = make_inputs(/*tokens=*/6);
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(in.qr.cpu(), in.wq, in.positions, in.rope),
      "privateuseone");
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(in.qr.to(torch::kFloat32), in.wq, in.positions, in.rope),
      "scalar_type");
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(in.qr, in.wq, in.positions.to(torch::kInt64), in.rope),
      "scalar_type");
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(in.qr, in.wq, in.positions, in.rope.to(torch::kBFloat16)),
      "scalar_type");
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(
          in.qr.slice(0, 0, 0), in.wq, in.positions.slice(0, 0, 0), in.rope),
      "qr.size");
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(in.qr, in.wq.slice(0, 0, 4095), in.positions, in.rope),
      "sizes");
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(in.qr, in.wq, in.positions.slice(0, 0, 5), in.rope),
      "sizes");
  const torch::Tensor strided =
      torch::empty({6, 2560}, in.qr.options()).slice(1, 0, 2560, 2);
  EXPECT_DEATH_IF_SUPPORTED(
      fused_indexer_q(strided, in.wq, in.positions, in.rope), "contiguous");
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
