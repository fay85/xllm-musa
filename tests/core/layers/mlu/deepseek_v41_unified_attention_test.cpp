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

#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

#include "kernels/mlu/piecewise_graph.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_attention.h"
#include "layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {

TEST(DeepseekV41MetadataTest,
     PositionsDefineChunksAndLogicalRingSurvivesCapacity) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto ints = torch::TensorOptions().device(device).dtype(torch::kInt32);
  ModelInputParams input;
  input.meta.num_sequences = 4;
  input.attention.host.q_seq_lens = {0, 1, 1, 4, 6};
  // Deliberately use unrelated KV lengths: model positions are authoritative.
  input.attention.host.kv_seq_lens = {0, 999, 999, 1998, 2997};
  input.multi_block_tables = {
      torch::tensor({2, 0, 1}, torch::kInt32).repeat({4, 1})};
  auto positions =
      torch::tensor({383, 1, 2, 3, 0, 0}, ints.dtype(torch::kInt64));
  Csa2Context shared;
  deepseek_v41::prepare_metadata(shared,
                                 input,
                                 positions,
                                 /*num_sequences=*/3,
                                 /*kv_seq_len_capacity=*/512);
  EXPECT_EQ(shared.num_sequences, 3);
  EXPECT_EQ(shared.host_query_prefix, (std::vector<int32_t>{0, 1, 1, 4, 4}));
  EXPECT_EQ(shared.host_chunk_begins, (std::vector<int32_t>{383, 0, 1, 0}));
  EXPECT_TRUE(torch::equal(shared.query_prefix.cpu(),
                           torch::tensor({0, 1, 1, 4, 4}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(shared.compressed[0].prefix.cpu(),
                           torch::tensor({0, 1, 1, 4, 4}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(shared.compressed[1].prefix.cpu(),
                           torch::tensor({0, 1, 1, 3, 3}, torch::kInt32)));
  EXPECT_EQ(shared.num_valid_input_rows.item<int32_t>(), 4);
  EXPECT_EQ(shared.block_tables[0].size(1), 4);
  EXPECT_EQ(shared.swa_block_table.size(1), 3);
  EXPECT_TRUE(
      torch::equal(shared.swa_block_table.cpu(), input.multi_block_tables[0]));
  const void* table = shared.block_tables[0].data_ptr();
  const void* latent = shared.compressed[1].outputs.latent.data_ptr();
  const void* prefix = shared.query_prefix.data_ptr();
  // Changing ring width updates the callback view, while captured compression
  // retains a complete absolute-page mapping at the original address.
  input.multi_block_tables[0] =
      torch::tensor({1, 2}, torch::kInt32).repeat({4, 1});
  input.enable_graph = true;
  deepseek_v41::prepare_metadata(shared,
                                 input,
                                 positions,
                                 /*num_sequences=*/0,
                                 /*kv_seq_len_capacity=*/512);
  EXPECT_EQ(shared.num_sequences, 0);
  EXPECT_TRUE(shared.query_prefix.eq(0).all().item<bool>());
  EXPECT_TRUE(shared.compressed[1].prefix.eq(0).all().item<bool>());
  EXPECT_EQ(shared.num_valid_input_rows.item<int32_t>(), 0);
  EXPECT_EQ(shared.swa_block_table.size(1), 2);
  EXPECT_EQ(shared.block_tables[0].data_ptr(), table);
  EXPECT_EQ(shared.compressed[1].outputs.latent.data_ptr(), latent);
  EXPECT_EQ(shared.query_prefix.data_ptr(), prefix);
}

TEST(DeepseekV41MetadataTest, HostPositionsKeepQueuedUploadsIndependent) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto ints = torch::TensorOptions().device(device).dtype(torch::kInt32);
  torch_mlu::mlu::MLUStreamGuard stream(
      torch_mlu::getStreamFromPool(false, device.index()));
  ModelInputParams input;
  input.meta.num_sequences = 2;
  input.attention.host.q_seq_lens = {0, 2, 4};
  input.attention.host.kv_seq_lens = {0, 999, 1998};
  input.multi_block_tables = {torch::zeros({2, 2}, ints)};
  // Only shape/device come from this tensor when the host positions exist.
  // Poisoned contents catch an accidental device readback, including padding.
  const auto positions = torch::full({4}, -1, ints);
  Csa2Context shared;
  std::vector<torch::Tensor> snapshots;
  snapshots.reserve(3);
  const void* prefix_address = nullptr;
  for (int32_t begin : {1, 5, 9}) {
    input.attention.host.positions =
        torch::tensor({begin, begin + 1}, torch::kInt32);
    deepseek_v41::prepare_metadata(shared,
                                   input,
                                   positions,
                                   /*num_sequences=*/1,
                                   /*kv_seq_len_capacity=*/256);
    snapshots.emplace_back(shared.chunk_begins.clone());
    if (prefix_address == nullptr) {
      prefix_address = shared.query_prefix.data_ptr();
    }
    EXPECT_EQ(shared.query_prefix.data_ptr(), prefix_address);
  }
  // Empty active prefixes are valid even when no host token rows remain.
  input.attention.host.positions = torch::empty({0}, torch::kInt64);
  deepseek_v41::prepare_metadata(shared,
                                 input,
                                 positions,
                                 /*num_sequences=*/0,
                                 /*kv_seq_len_capacity=*/256);
  torch_mlu::synchronize();
  for (int32_t index = 0; index < 3; ++index) {
    EXPECT_TRUE(torch::equal(snapshots[index].cpu(),
                             torch::tensor({1 + index * 4, 0}, torch::kInt32)));
  }
  EXPECT_TRUE(shared.query_prefix.eq(0).all().item<bool>());
  EXPECT_TRUE(shared.compressed[1].prefix.eq(0).all().item<bool>());
  EXPECT_EQ(shared.num_valid_input_rows.item<int32_t>(), 0);
  EXPECT_EQ(shared.query_prefix.data_ptr(), prefix_address);
}

// Constant KV rows and zero queries make the independent attention reference
// an arithmetic mean including one zero-valued sink. Identity output weights
// expose that result through the real layer, without using production helpers.
class DeepseekV41UnifiedAttentionTest
    : public ::testing::TestWithParam<std::tuple<int32_t, int32_t, bool>> {};

TEST_P(DeepseekV41UnifiedAttentionTest,
       ReplayAndWorkloadsMatchIndependentReference) {
  torch::InferenceMode inference;
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  const int32_t ratio = std::get<0>(GetParam());
  const int32_t count = std::get<1>(GetParam());
  const bool proposal = std::get<2>(GetParam());
  const int32_t begin = count == 1 ? 0 : 380;
  const int32_t num_valid_input_rows = count + 3;
  const int32_t capacity = num_valid_input_rows + 2;
  ModelArgs args;
  args.hidden_size(32)
      .n_heads(8)
      .head_dim(512)
      .q_lora_rank(16)
      .qk_rope_head_dim(64)
      .o_groups(1)
      .o_lora_rank(8)
      .window_size(128)
      .rms_norm_eps(1e-5)
      .index_topk(4)
      .compress_ratios({ratio, ratio})
      .kv_source_layer_ids({0})
      .dspark_block_size(5);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  DeepseekV41Attention attention(
      ModelContext(parallel, args, QuantArgs(), options),
      /*layer_id=*/1);
  auto wkv = torch::zeros({512, 32}, options);
  wkv.select(1, 0).fill_(1);
  auto wo_a = torch::zeros({1, 8, 4096}, options);
  wo_a[0].slice(1, 0, 8).copy_(torch::eye(8, options));
  auto wo_b = torch::zeros({32, 8}, options);
  wo_b.slice(0, 0, 8).copy_(torch::eye(8, options));
  attention->load_state_dict(StateDict({
      {"wq_a.weight", torch::zeros({16, 32}, options)},
      {"wq_b.weight", torch::zeros({4096, 16}, options)},
      {"wkv.weight", wkv},
      {"q_norm.weight", torch::ones({16}, options)},
      {"kv_norm.weight", torch::ones({512}, options)},
      {"wo_a.weight", wo_a},
      {"wo_b.weight", wo_b},
      {"attn_sink", torch::zeros({8}, options)},
  }));
  const auto make_cache = [&]() {
    std::vector<KVCache> caches;
    caches.reserve(2);
    for (int32_t layer = 0; layer < 2; ++layer) {
      DeepSeekV4KVCacheTensors tensors;
      tensors.swa_cache = torch::arange(6 * 128, options.dtype(torch::kFloat32))
                              .remainder(13)
                              .div(8)
                              .view({6, 128, 1, 1})
                              .expand({6, 128, 1, 512})
                              .to(options)
                              .contiguous();
      tensors.key_cache =
          torch::arange(16 * 128, options.dtype(torch::kFloat32))
              .remainder(7)
              .div(8)
              .view({16, 128, 1, 1})
              .expand({16, 128, 1, 512})
              .to(options)
              .contiguous();
      tensors.index_cache = torch::zeros({16, 128, 1, 128}, options);
      caches.emplace_back(tensors);
    }
    return caches;
  };
  auto graph_caches = make_cache();
  auto eager_caches = make_cache();
  auto prefill_caches = make_cache();
  const auto original_swa =
      graph_caches[1].get_swa_cache().cpu().view({-1, 512});
  const auto original_main = graph_caches[0].get_k_cache().clone();
  auto hidden = torch::zeros({capacity, 32}, options);
  hidden.select(1, 0).copy_(
      torch::arange(capacity, ints).remainder(2).mul(2).sub(1));
  auto positions = torch::cat({torch::arange(begin, begin + count, ints),
                               torch::arange(1, 4, ints),
                               torch::zeros({2}, ints)})
                       .to(torch::kInt64);
  auto rope =
      torch::cat({torch::ones({1024, 64}, options.dtype(torch::kFloat32)),
                  torch::zeros({1024, 64}, options.dtype(torch::kFloat32))},
                 1);
  ModelInputParams input;
  input.meta.batch_forward_type = BatchForwardType::DECODE;
  input.is_spec_verify = count > 1 && !proposal;
  input.use_dflash_proposal_attention = proposal;
  input.meta.num_sequences = 3;
  input.meta.q_max_seq_len = count;
  input.meta.kv_max_seq_len = begin + count;
  input.attention.host.q_seq_lens = {0, count, num_valid_input_rows, capacity};
  input.multi_block_tables = {
      torch::tensor({2, 0, 1, 5, 3, 4}, torch::kInt32).view({2, 3}),
      torch::arange(16, torch::kInt32).view({2, 8}),
      torch::arange(16, torch::kInt32).view({2, 8})};
  Csa2Context graph_shared, eager_shared, prefill_shared;
  const auto prepare = [&](Csa2Context& shared,
                           const ModelInputParams& params,
                           int32_t active,
                           int64_t extent) {
    deepseek_v41::prepare_metadata(shared, params, positions, active, extent);
    deepseek_v41::prepare_workspace(shared, params, args, capacity, options);
    shared.topk.resize(3);
    shared.candidates.resize(3);
    for (int32_t seq = 0; seq < 2; ++seq) {
      const int32_t rows = seq == 0 ? count : 3;
      auto topk = torch::tensor({-1, 0, 1, 2}, options.dtype(torch::kInt64))
                      .repeat({rows, 1});
      if (ratio > 0) {
        auto lengths = torch::floor_divide(
            positions.slice(0,
                            seq == 0 ? 0 : count,
                            seq == 0 ? count : num_valid_input_rows) +
                1,
            ratio);
        topk.masked_fill_(topk >= lengths.unsqueeze(1), -1);
      }
      shared.topk[seq] = topk;
    }
  };
  prepare(graph_shared, input, /*active=*/2, /*extent=*/1024);
  prepare(eager_shared, input, /*active=*/2, /*extent=*/0);
  torch::Tensor expected_prefill;
  if (!proposal) {
    ModelInputParams prefill_input = input;
    prefill_input.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    prefill_input.is_spec_verify = false;
    prepare(prefill_shared, prefill_input, /*active=*/2, /*extent=*/0);
    EXPECT_FALSE(prefill_shared.selected_kv.defined());
    expected_prefill = attention
                           ->forward(hidden,
                                     positions,
                                     rope,
                                     prefill_caches,
                                     prefill_input,
                                     prefill_shared)
                           .clone();
  }
  const auto eager =
      attention
          ->forward(hidden, positions, rope, eager_caches, input, eager_shared)
          .clone();
  // Warm up lazy kernels. Capture itself must leave caches unchanged.
  attention->forward(
      hidden, positions, rope, graph_caches, input, graph_shared);
  const auto before_capture = graph_caches[1].get_swa_cache().clone();
  kernel::mlu::PiecewiseGraph graph(/*kv_max_seq_len=*/1024);
  input.enable_graph = true;
  input.graph.eager_context = &graph;
  torch::Tensor actual;
  torch_mlu::synchronize();
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin(torch_mlu::graph_pool_handle());
    actual = attention->forward(
        hidden, positions, rope, graph_caches, input, graph_shared);
    graph.capture_end();
  }
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(graph_caches[1].get_swa_cache(), before_capture));
  ASSERT_EQ(graph.num_eager_segments(), 1);
  auto reference = torch::zeros({capacity, 32}, torch::kFloat32);
  auto expected_cache = original_swa.clone();
  auto expected_lengths = torch::zeros({capacity}, torch::kInt32);
  const auto main = original_main.cpu().view({-1, 512});
  const auto signs = hidden.cpu().select(1, 0).to(torch::kFloat32);
  const auto table = input.multi_block_tables[0].accessor<int32_t, 2>();
  for (int32_t seq = 0; seq < 2; ++seq) {
    const int32_t offset = seq == 0 ? 0 : count;
    const int32_t start = seq == 0 ? begin : 1;
    const int32_t rows = seq == 0 ? count : 3;
    const auto selections = graph_shared.topk[seq].cpu();
    for (int32_t row = 0; row < rows; ++row) {
      const int32_t position = start + row;
      const int32_t first =
          std::max(0, proposal ? start - 128 : position - 127);
      const int32_t last = proposal ? start + rows - 1 : position;
      float sum = 0;
      int32_t length = last - first + 1;
      for (int32_t token = first; token <= last; ++token) {
        const int32_t physical =
            table[seq][(token / 128) % 3] * 128 + token % 128;
        sum += token >= start ? signs[offset + token - start].item<float>()
                              : original_swa[physical][0].item<float>();
      }
      if (ratio > 0) {
        for (int32_t col = 0; col < 4; ++col) {
          const int64_t logical = selections[row][col].item<int64_t>();
          if (logical < 0) {
            continue;
          }
          sum += main[seq * 8 * 128 + logical][0].item<float>();
          ++length;
        }
      }
      reference[offset + row].slice(0, 0, 8).fill_(sum / (length + 1));
      expected_lengths[offset + row].fill_(length);
      const int32_t target =
          table[seq][(position / 128) % 3] * 128 + position % 128;
      if (!proposal) {
        expected_cache[target].fill_(signs[offset + row].item<float>());
      }
    }
  }
  reference = reference.to(torch::kBFloat16).to(torch::kFloat32);
  EXPECT_TRUE(
      torch::allclose(eager.cpu().to(torch::kFloat32), reference, 0.01, 0.01));
  if (!proposal) {
    EXPECT_TRUE(torch::allclose(
        expected_prefill.cpu().to(torch::kFloat32), reference, 0.01, 0.01));
    EXPECT_TRUE(torch::equal(prefill_caches[1].get_swa_cache(),
                             eager_caches[1].get_swa_cache()));
  }
  EXPECT_TRUE(
      torch::equal(eager_shared.selected_lengths.cpu(), expected_lengths));
  EXPECT_TRUE(torch::equal(
      eager_caches[1].get_swa_cache().cpu().view({-1, 512}), expected_cache));
  for (int32_t active : {2, 0, 2}) {
    SCOPED_TRACE(active);
    graph_caches[1].get_swa_cache().copy_(original_swa.view({6, 128, 1, 512}));
    graph_shared.selected_lengths.fill_(-1);
    graph_shared.selected_kv.fill_(std::numeric_limits<float>::quiet_NaN());
    prepare(graph_shared, input, active, /*extent=*/1024);
    graph.replay();
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
    if (active == 0) {
      EXPECT_TRUE(graph_shared.selected_lengths.eq(0).all().item<bool>());
      EXPECT_TRUE(torch::equal(
          graph_caches[1].get_swa_cache().cpu().view({-1, 512}), original_swa));
      continue;
    }
    EXPECT_TRUE(torch::equal(actual, eager));
    EXPECT_TRUE(
        torch::equal(graph_shared.selected_lengths.cpu(), expected_lengths));
    EXPECT_TRUE(torch::equal(graph_caches[1].get_swa_cache(),
                             eager_caches[1].get_swa_cache()));
    EXPECT_TRUE(torch::equal(graph_caches[0].get_k_cache(), original_main));
  }
}

INSTANTIATE_TEST_SUITE_P(RatiosAndChunks,
                         DeepseekV41UnifiedAttentionTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),
                                            ::testing::Values(1, 5, 400),
                                            ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(ProposalWindow,
                         DeepseekV41UnifiedAttentionTest,
                         ::testing::Values(std::make_tuple(0, 5, true)));

}  // namespace
}  // namespace xllm::layer
