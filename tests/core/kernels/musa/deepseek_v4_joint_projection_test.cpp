/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/aten/musa/MUSAGraph.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <optional>
#include <vector>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_compressor_projection.h"

namespace xllm::kernel::musa {
namespace {

TEST(DeepseekV4JointProjectionTest, KeepsMainValuesAndPhysicalRowStride) {
  torch::Tensor packed = torch::arange(35 * 2560).reshape({35, 2560});
  torch::Tensor main = layer::dsv4_compressor_projection_view(
      packed, 1024, 256, /*uses_indexer=*/false);
  EXPECT_EQ(main.size(1), 2048);
  EXPECT_EQ(main.stride(0), 2560);
  EXPECT_EQ(main.data_ptr(), packed.data_ptr());
  EXPECT_TRUE(torch::equal(main.narrow(0, 7, 7),
                           packed.narrow(0, 7, 7).narrow(1, 0, 2048)));
  torch::Tensor full = layer::dsv4_compressor_projection_view(
      packed, 1024, 256, /*uses_indexer=*/true);
  EXPECT_TRUE(torch::equal(full, packed));
  EXPECT_EQ(full.data_ptr(), packed.data_ptr());
}

TEST(DeepseekV4JointProjectionTest, AcceptsMainOnlyC4AndC128Layouts) {
  for (const int64_t main_width : {512, 1024}) {
    torch::Tensor packed = torch::randn({14, 2 * main_width});
    torch::Tensor selected =
        layer::dsv4_compressor_projection_view(packed,
                                               main_width,
                                               main_width == 512 ? 0 : 256,
                                               /*uses_indexer=*/false);
    EXPECT_TRUE(torch::equal(selected, packed));
    EXPECT_EQ(selected.data_ptr(), packed.data_ptr());
  }
}

TEST(DeepseekV4JointProjectionDeathTest, RejectsUnknownPackingWidth) {
  EXPECT_DEATH(layer::dsv4_compressor_projection_view(
                   torch::zeros({14, 2304}), 1024, 256, false),
               "Unexpected packed compressor projection width");
}

TEST(DeepseekV4JointProjectionDeathTest, RejectsMissingRequiredIndexer) {
  EXPECT_DEATH(layer::dsv4_compressor_projection_view(
                   torch::zeros({14, 2048}), 1024, 256, true),
               "Missing Indexer projection columns");
}

class DeepseekV4JointProjectionGraphTest : public ::testing::Test {
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
    torch::manual_seed(20261026);
  }

  void TearDown() override { EXPECT_EQ(musaDeviceSynchronize(), musaSuccess); }

  void check_compression(int64_t sequences, bool indexer) {
    constexpr int64_t kQueryWidth = 7;
    const int64_t head_dim = indexer ? 128 : 512;
    const int64_t width = 2 * head_dim;
    const int64_t offset = indexer ? 2048 : 0;
    const auto fp32 =
        torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    const auto bf16 = fp32.dtype(torch::kBFloat16);
    torch::Tensor packed = torch::zeros({sequences * kQueryWidth, 2560}, bf16);
    torch::Tensor actual_kv = torch::zeros({sequences, 8, width}, fp32);
    torch::Tensor actual_score = torch::zeros_like(actual_kv);
    torch::Tensor expected_kv = torch::zeros_like(actual_kv);
    torch::Tensor expected_score = torch::zeros_like(actual_kv);
    torch::Tensor ape = torch::randn({4, width}, fp32) * 0.1;
    torch::Tensor positions =
        torch::zeros({sequences * kQueryWidth}, fp32.dtype(torch::kInt64));
    torch::Tensor actual = torch::zeros({sequences * 2, head_dim}, bf16);
    torch::Tensor expected = torch::zeros_like(actual);
    auto launch = [&]() {
      torch::Tensor view =
          layer::dsv4_compressor_projection_view(packed, 1024, 256, indexer);
      return deepseek_v4_compress_decode_batch_graph(
          actual_kv,
          actual_score,
          view.narrow(1, offset, width),
          view.narrow(1, offset + width, width),
          ape,
          4,
          head_dim,
          kQueryWidth,
          positions,
          actual);
    };
    EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    launch();
    EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    at::musa::MUSAGraph graph;
    graph.capture_begin();
    launch();
    graph.capture_end();
    for (int32_t iteration = 0; iteration < 4; ++iteration) {
      packed.copy_(torch::randn_like(packed));
      torch::Tensor state = torch::randn_like(actual_kv) * 0.1;
      torch::Tensor scores = torch::randn_like(actual_score) * 0.1;
      actual_kv.copy_(state);
      expected_kv.copy_(state);
      actual_score.copy_(scores);
      expected_score.copy_(scores);
      torch::Tensor bases =
          (torch::arange(sequences, torch::kInt64) + 1) * 128 + 1 + iteration;
      torch::Tensor new_positions =
          (bases.unsqueeze(1) +
           torch::arange(kQueryWidth, torch::kInt64).unsqueeze(0))
              .flatten();
      positions.copy_(new_positions.to(device_));
      torch::Tensor kv_copy = packed.narrow(1, offset, width).contiguous();
      torch::Tensor score_copy =
          packed.narrow(1, offset + width, width).contiguous();
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
      for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        deepseek_v4_compress_decode_sequence_graph(
            expected_kv.select(0, sequence),
            expected_score.select(0, sequence),
            kv_copy.narrow(0, sequence * kQueryWidth, kQueryWidth),
            score_copy.narrow(0, sequence * kQueryWidth, kQueryWidth),
            ape,
            4,
            head_dim,
            positions.narrow(0, sequence * kQueryWidth, 1),
            expected.narrow(0, sequence * 2, 2));
      }
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
      graph.replay();
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
      EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()))
          << sequences << ":" << iteration;
      EXPECT_TRUE(torch::equal(actual_kv.cpu(), expected_kv.cpu()));
      EXPECT_TRUE(torch::equal(actual_score.cpu(), expected_score.cpu()));
    }
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::optional<torch::NoGradGuard> no_grad_;
  std::optional<c10::musa::MUSAStream> stream_;
  std::optional<c10::StreamGuard> guard_;
};

TEST_F(DeepseekV4JointProjectionGraphTest,
       ShortMainCompressionMatchesSeparateColumns) {
  for (const int64_t sequences : {2, 3, 5}) {
    check_compression(sequences, /*indexer=*/false);
  }
}

TEST_F(DeepseekV4JointProjectionGraphTest,
       IndexerCompressionKeepsJointOffsets) {
  for (const int64_t sequences : {2, 3, 5}) {
    check_compression(sequences, /*indexer=*/true);
  }
}

TEST_F(DeepseekV4JointProjectionGraphTest,
       JointGemmMatchesSeparateProjections) {
  for (const int64_t rows : {14, 35}) {
    torch::Tensor input =
        (torch::randn({rows, 4096}) * 0.3).to(torch::kBFloat16).to(device_);
    std::vector<torch::Tensor> weights;
    std::vector<torch::Tensor> separate;
    weights.reserve(4);
    separate.reserve(4);
    for (const int64_t width : {1024, 1024, 256, 256}) {
      weights.emplace_back((torch::randn({width, 4096}) * 0.02)
                               .to(torch::kBFloat16)
                               .to(device_));
    }
    torch::Tensor packed_weight = torch::cat(weights, 0);
    EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    torch::Tensor joint = ::xllm::kernel::musa::matmul(
        input, packed_weight, std::nullopt, std::nullopt);
    EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    for (const torch::Tensor& weight : weights) {
      separate.emplace_back(::xllm::kernel::musa::matmul(
          input, weight, std::nullopt, std::nullopt));
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    }
    torch::Tensor expected = torch::cat(separate, 1).cpu().to(torch::kFloat32);
    torch::Tensor actual = joint.cpu().to(torch::kFloat32);
    const float max_error = (expected - actual).abs().max().item<float>();
    const int64_t changed = (expected != actual).sum().item<int64_t>();
    EXPECT_TRUE(torch::allclose(actual, expected, 8e-3, 2e-5));
    RecordProperty("rows" + std::to_string(rows) + "_max_bf16_error",
                   std::to_string(max_error));
    RecordProperty("rows" + std::to_string(rows) + "_changed_elements",
                   std::to_string(changed));
  }
}

TEST_F(DeepseekV4JointProjectionGraphTest,
       PackedGraphPreservesRequestLocalGemmValues) {
  for (const int64_t sequences : {1, 2, 5}) {
    const int64_t rows = sequences * 7;
    torch::Tensor input =
        (torch::randn({rows, 4096}) * 0.3).to(torch::kBFloat16).to(device_);
    std::vector<torch::Tensor> weights;
    weights.reserve(4);
    for (const int64_t width : {1024, 1024, 256, 256}) {
      weights.emplace_back((torch::randn({width, 4096}) * 0.02)
                               .to(torch::kBFloat16)
                               .to(device_));
    }
    torch::Tensor packed_weight = torch::cat(weights, 0);
    torch::Tensor expected = torch::empty({rows, 2560}, input.options());
    EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    layer::dsv4_pack_compressor_projections(input, packed_weight, 7, 1024, 256);
    EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    at::musa::MUSAGraph graph;
    graph.capture_begin();
    torch::Tensor actual = layer::dsv4_pack_compressor_projections(
        input, packed_weight, 7, 1024, 256);
    graph.capture_end();
    for (int32_t iteration = 0; iteration < 3; ++iteration) {
      input.copy_(torch::randn_like(input));
      packed_weight.copy_(torch::randn_like(packed_weight));
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
      for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        int64_t offset = 0;
        for (const int64_t width : {1024, 1024, 256, 256}) {
          torch::Tensor output = torch::empty({7, width}, input.options());
          ::xllm::kernel::musa::matmul(input.narrow(0, sequence * 7, 7),
                                       packed_weight.narrow(0, offset, width),
                                       std::nullopt,
                                       output);
          EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
          expected.narrow(0, sequence * 7, 7)
              .narrow(1, offset, width)
              .copy_(output);
          offset += width;
        }
      }
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
      graph.replay();
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
      EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()))
          << "sequences=" << sequences << " iteration=" << iteration;
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::musa
