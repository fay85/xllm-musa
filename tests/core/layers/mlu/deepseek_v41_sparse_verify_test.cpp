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

#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_indexer.h"
#include "layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {

TEST(DeepseekV41GraphTest, SparseVerificationIgnoresCausallyMaskedSelections) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.compress_ratios({2}).index_topk(2).candidate_source_layer_id(-1);
  DeepseekV41Indexer indexer(ModelContext(parallel, args, QuantArgs(), options),
                             /*layer_id=*/0);
  Csa2Context shared;
  shared.topk.resize(1);
  shared.candidates.resize(1);

  // Four verification rows at positions 0..3 see 0, 1, 1, 2 compressed
  // tokens. Use the real selector so masked entries arise from causality.
  auto query = torch::ones({4, 1, 1}, options.dtype(torch::kFloat32));
  auto weights = torch::ones({4, 1}, query.options());
  auto keys = torch::tensor({1.0f, 2.0f}, query.options()).view({2, 1});
  indexer->select_indices(query, weights, keys, /*begin=*/0, /*seq=*/0, shared);
  auto expected_indices =
      torch::tensor({-1, -1, 0, -1, 0, -1, 1, 0}, torch::kInt64).view({4, 2});
  ASSERT_TRUE(torch::equal(shared.topk[0].cpu(), expected_indices));

  auto swa = torch::zeros({256, 512}, options);
  auto compressed = torch::zeros_like(swa);
  auto current = torch::zeros({512}, options);
  auto swa_table = torch::tensor({0}, options.dtype(torch::kInt32));
  auto compressed_table = torch::tensor({1}, swa_table.options());
  compressed[128].fill_(4);
  compressed[129].fill_(8);
  // Page 1 makes the buggy address for logical -1 equal 1 * 128 - 1 = 127.
  // It is inside this allocation: reproduce silent stale-data reads without
  // depending on an illegal access or the device allocator's contents.
  compressed[127].fill_(64);

  constexpr int64_t kCapacity = 6;
  auto selected = torch::full({4 * kCapacity, 1, 1, 512},
                              std::numeric_limits<float>::quiet_NaN(),
                              options);
  auto lengths = torch::full({4}, -1, swa_table.options());
  for (int64_t row = 0; row < 4; ++row) {
    kernel::mlu::bangc::deepseek_v41_decode_gather(
        swa,
        compressed,
        current,
        swa_table,
        compressed_table,
        shared.topk[0][row].flatten(),
        /*position=*/row,
        /*window=*/4,
        selected.slice(0, row * kCapacity, (row + 1) * kCapacity),
        lengths.slice(0, row, row + 1));
  }

  // Zero queries and sink logits give uniform weights over the visible KV
  // rows plus one zero-valued sink. The expected outputs are hand-computed.
  auto q = torch::zeros({4, 1, 8, 512}, options);
  auto sink = torch::zeros({8}, query.options());
  auto table =
      torch::arange(4 * kCapacity, swa_table.options()).view({4, 1, kCapacity});
  auto output = torch::empty_like(q);
  std::optional<torch::Tensor> lse;
  kernel::mlu::batch_decode(q,
                            selected,
                            output,
                            table,
                            lengths,
                            std::nullopt,
                            lse,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            "half",
                            /*max_seq_len=*/kCapacity,
                            /*window_size_left=*/-1,
                            /*window_size_right=*/-1,
                            /*scale=*/1.0 / std::sqrt(512),
                            /*return_lse=*/false,
                            /*kv_cache_quant_bit_size=*/-1,
                            std::nullopt,
                            /*max_seq_q=*/-1,
                            sink);
  torch_mlu::synchronize();

  const std::array<int32_t, 4> expected_lengths = {1, 3, 4, 6};
  const std::array<float, 4> expected_values = {0, 1, 0.8f, 12.0f / 7};
  auto host_lengths = lengths.cpu();
  auto host_output = output.cpu().to(torch::kFloat32);
  for (int64_t row = 0; row < 4; ++row) {
    SCOPED_TRACE(::testing::Message() << "query_position=" << row);
    EXPECT_EQ(host_lengths[row].item<int32_t>(), expected_lengths[row]);
    auto expected = torch::full_like(host_output[row], expected_values[row]);
    EXPECT_TRUE(torch::allclose(host_output[row], expected, 0.01, 0.01))
        << "expected attention=" << expected_values[row]
        << ", actual first component="
        << host_output[row].flatten()[0].item<float>();
  }
}

TEST(DeepseekV41IndexerTest, CandidateSelectionKeepsOnlyVisibleRows) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  for (int32_t ratio : {1, 2}) {
    SCOPED_TRACE(ratio);
    ModelArgs args;
    args.compress_ratios({ratio, ratio})
        .index_topk(8)
        .candidate_source_layer_id(0)
        .candidate_block_size(4)
        .candidate_topk_blocks(1);
    const ModelContext context(parallel, args, QuantArgs(), options);
    DeepseekV41Indexer source(context, /*layer_id=*/0);
    DeepseekV41Indexer consumer(context, /*layer_id=*/1);
    Csa2Context shared;
    shared.topk.resize(1);
    shared.candidates.resize(1);
    auto query = torch::ones({1, 1, 1}, options.dtype(torch::kFloat32));
    auto weights = torch::ones({1, 1}, query.options());
    auto keys = torch::arange(6, query.options()).view({6, 1});
    source->select_indices(query,
                           weights,
                           keys,
                           /*begin=*/6 * ratio - 1,
                           /*seq=*/0,
                           shared);
    EXPECT_TRUE(torch::equal(shared.candidates[0].cpu(),
                             torch::tensor({{1}}, torch::kInt64)));
    auto cache = torch::zeros({256, 1}, options);
    cache.slice(0, 128, 134).copy_(keys.to(options));
    auto table = torch::tensor({1}, options.dtype(torch::kInt32));
    consumer->select_candidate_indices(query[0],
                                       weights[0],
                                       cache,
                                       table,
                                       /*length=*/6,
                                       /*seq=*/0,
                                       shared);
    EXPECT_TRUE(torch::equal(std::get<0>(shared.topk[0].cpu().sort()),
                             torch::tensor({{-1, -1, 4, 5}}, torch::kInt64)));
    shared.candidates[0] =
        torch::tensor({{-1, 1}}, options.dtype(torch::kInt64));
    consumer->select_candidate_indices(query[0],
                                       weights[0],
                                       cache,
                                       table,
                                       /*length=*/6,
                                       /*seq=*/0,
                                       shared);
    auto valid = shared.topk[0].masked_select(shared.topk[0].ge(0));
    EXPECT_TRUE(torch::equal(std::get<0>(valid.cpu().sort()),
                             torch::tensor({4, 5}, torch::kInt64)));
    source->select_indices(query,
                           weights,
                           keys.slice(0, 0, 0),
                           /*begin=*/0,
                           /*seq=*/0,
                           shared);
    EXPECT_EQ(shared.candidates[0].numel(), 0);
  }
}

TEST(DeepseekV41IndexerTest, PagedSelectionMatchesScoreBoundaryAndRepeatsTies) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.compress_ratios({2}).index_topk(512).index_n_heads(32).index_head_dim(
      128);
  DeepseekV41Indexer indexer(ModelContext(parallel, args, QuantArgs(), options),
                             /*layer_id=*/0);
  Csa2Context shared;
  shared.topk.resize(1);
  auto query = torch::ones({32, 128}, options.dtype(torch::kFloat32));
  auto weights = torch::ones({32}, query.options()) / 32;
  auto table = torch::tensor({5, 3, 1, 0, 4, 2}, options.dtype(torch::kInt32));
  auto cache = torch::zeros({6, 1, 128, 128}, options);
  auto logical = torch::zeros({768, 128}, options);
  logical.slice(0, 0, 512).fill_(2);
  cache.index_copy_(0, table.to(torch::kInt64), logical.view({6, 1, 128, 128}));
  indexer->select_decode_indices(query,
                                 weights,
                                 cache,
                                 table,
                                 /*length=*/700,
                                 /*seq=*/0,
                                 shared);
  auto paged = shared.topk[0].cpu();
  indexer->select_indices(query.unsqueeze(0),
                          weights.unsqueeze(0),
                          logical.slice(0, 0, 700).to(torch::kFloat32),
                          /*begin=*/1399,
                          /*seq=*/0,
                          shared);
  EXPECT_TRUE(torch::equal(std::get<0>(paged.sort()),
                           std::get<0>(shared.topk[0].cpu().sort())));
  EXPECT_TRUE(torch::equal(std::get<0>(paged.sort()).flatten(),
                           torch::arange(512, torch::kInt64)));
  // At an exactly tied boundary, retain the paged kernel's choice. It must be
  // unique, in range and reproducible; general top-k need not order ties alike.
  cache.fill_(1);
  indexer->select_decode_indices(query,
                                 weights,
                                 cache,
                                 table,
                                 /*length=*/700,
                                 /*seq=*/0,
                                 shared);
  auto tied = shared.topk[0].clone();
  for (int32_t repeat = 0; repeat < 3; ++repeat) {
    indexer->select_decode_indices(query,
                                   weights,
                                   cache,
                                   table,
                                   /*length=*/700,
                                   /*seq=*/0,
                                   shared);
    EXPECT_TRUE(torch::equal(shared.topk[0], tied));
  }
  auto indices = tied.cpu().flatten();
  EXPECT_GE(indices.min().item<int64_t>(), 0);
  EXPECT_LT(indices.max().item<int64_t>(), 700);
  auto membership = torch::zeros({700}, torch::kBool);
  membership.index_fill_(0, indices, true);
  EXPECT_EQ(membership.sum().item<int64_t>(), 512);
}

class DeepseekV41MaskedGatherTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(DeepseekV41MaskedGatherTest, ReplayCompactsLiveIndicesAndUpdatesLength) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const int64_t count = GetParam();
  auto swa = torch::arange(256, options).unsqueeze(1).repeat({1, 512});
  auto compressed = torch::arange(512, options).unsqueeze(1).repeat({1, 512});
  auto current = torch::full({512}, -8, options);
  auto swa_table = torch::tensor({1, 0}, options.dtype(torch::kInt32));
  auto compressed_table = torch::tensor({2, 1, 3}, swa_table.options());
  auto host_indices = torch::arange(count, torch::kInt64).remainder(384);
  auto topk = host_indices.to(device);
  auto output = torch::empty({4 + count, 512}, options);
  auto length = torch::full({1}, -1, swa_table.options());
  const auto gather = [&]() {
    kernel::mlu::bangc::deepseek_v41_decode_gather(swa,
                                                   compressed,
                                                   current,
                                                   swa_table,
                                                   compressed_table,
                                                   topk,
                                                   /*position=*/129,
                                                   /*window=*/4,
                                                   output,
                                                   length);
  };

  // Position 129 wraps the SWA ring: physical rows 254, 255, 0, then current.
  auto expected_window = torch::cat(
      {swa.cpu().index_select(0, torch::tensor({254, 255, 0}, torch::kInt64)),
       current.cpu().unsqueeze(0)},
      0);
  auto logical_compressed =
      compressed.cpu()
          .view({4, 128, 512})
          .index_select(0, torch::tensor({2, 1, 3}, torch::kInt64))
          .reshape({384, 512});
  gather();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    graph.capture_begin();
    gather();
    graph.capture_end();
  }

  auto mixed = host_indices.clone();
  mixed.masked_fill_(torch::arange(count, torch::kInt64).remainder(3).eq(1),
                     -1);
  // A negative index with a negative page quotient must also be rejected
  // before any page-table access. Exercise leading and trailing holes too.
  if (count > 0) {
    mixed[0].fill_(-129);
    mixed[count - 1].fill_(-1);
  }
  const std::vector<torch::Tensor> patterns = {
      host_indices, mixed, torch::full_like(host_indices, -1), host_indices};
  for (size_t pattern = 0; pattern < patterns.size(); ++pattern) {
    SCOPED_TRACE(::testing::Message()
                 << "count=" << count << " pattern=" << pattern);
    topk.copy_(patterns[pattern]);
    output.fill_(std::numeric_limits<float>::quiet_NaN());
    length.fill_(-1);
    // Verify the kernel still publishes the current token to the SWA cache.
    swa[1].fill_(99);
    graph.replay();
    torch_mlu::synchronize();

    auto valid = patterns[pattern].masked_select(patterns[pattern].ge(0));
    auto expected = torch::cat(
        {expected_window, logical_compressed.index_select(0, valid)}, 0);
    ASSERT_EQ(length.cpu().item<int32_t>(), expected.size(0));
    EXPECT_TRUE(
        torch::equal(output.cpu().slice(0, 0, expected.size(0)), expected));
    EXPECT_TRUE(torch::equal(swa[1].cpu(), current.cpu()));
  }
}

INSTANTIATE_TEST_SUITE_P(
    Boundaries,
    DeepseekV41MaskedGatherTest,
    ::testing::Values(0, 1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1025));

}  // namespace
}  // namespace xllm::layer
