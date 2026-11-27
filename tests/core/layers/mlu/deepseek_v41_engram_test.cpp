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

#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"

#include <ATen/Parallel.h>
#include <framework/core/MLUStream.h>
#include <framework/core/caching_allocator.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "framework/config/model_config.h"
#include "layers/mlu/tests_utils.h"
#include "platform/platform.h"

namespace xllm::layer {
namespace {
TEST(DeepseekV41EngramCpuGatherTest, ParallelBackendUsesMultipleThreads) {
  const int32_t saved_threads = torch::get_num_threads();
  torch::set_num_threads(4);
  std::atomic<uint64_t> participants{0};
  std::atomic<int64_t> visited_rows{0};
  torch::parallel_for(0, 16384, 1024, [&](int64_t begin, int64_t end) {
    participants.fetch_or(uint64_t{1} << torch::get_thread_num(),
                          std::memory_order_relaxed);
    visited_rows.fetch_add(end - begin, std::memory_order_relaxed);
  });
  torch::set_num_threads(saved_threads);
  const uint64_t mask = participants.load(std::memory_order_relaxed);
  EXPECT_NE(mask & (mask - 1), 0)
      << "ATen CPU gather parallel_for must use multiple worker threads";
  EXPECT_EQ(visited_rows.load(std::memory_order_relaxed), 16384);
}

class DeepseekV41EngramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_config_ = ModelConfig::get_instance().engram_config();
    saved_death_test_style_ = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "MLU device is required";
    }
    args_.hidden_size(8)
        .vocab_size(8)
        .hc_mult(2)
        .engram_layer_ids({0})
        .engram_num_embeddings({18})
        .engram_head_dim(32)
        .engram_n_heads(1)
        .engram_max_ngram_size(3)
        .engram_pad_token_id(0)
        .engram_compressed_vocab_size(8)
        .rms_norm_eps(1e-5);
    weights_ = {{"embed.weight",
                 torch::arange(18 * 32).remainder(96).reshape({18, 32}).to(
                     torch::kUInt8)},
                {"embed.scale",
                 torch::arange(18).remainder(4).add(125).reshape({18, 1}).to(
                     torch::kUInt8)},
                {"wkv.weight",
                 torch::linspace(-0.1, 0.1, 24 * 64)
                     .reshape({24, 64})
                     .to(torch::kBFloat16)},
                {"q_weight", torch::ones({2, 8}, torch::kBFloat16)},
                {"k_weight", torch::ones({2, 8}, torch::kBFloat16)}};
    hash_weights_ = {{"token_map", torch::arange(8, torch::kInt64)},
                     {"multipliers", torch::tensor({{1, 3, 5}}, torch::kInt64)},
                     {"primes", torch::tensor({{7, 11}}, torch::kInt64)},
                     {"offsets", torch::tensor({{0, 7}}, torch::kInt64)}};
    input_.meta.num_sequences = 2;
    input_.meta.actual_num_sequences = 2;
    input_.attention.host.q_seq_lens = {0, 2, 4};
    input_.embedding.engram_lookback_token_ids = {1, 2, 3, 4};
  }

  void TearDown() override {
    ModelConfig::get_instance().engram_config() = saved_config_;
    GTEST_FLAG_SET(death_test_style, saved_death_test_style_);
  }

  DeepseekV41Engram create(const std::string& config) {
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group_);
    parallel.tp_group_ = &group_;
    return create(config, ModelContext(parallel, args_, QuantArgs(), options_));
  }

  DeepseekV41Engram create(const std::string& config,
                           const ModelContext& context) {
    ModelConfig::get_instance().engram_config() = config;
    DeepseekV41Engram layer(context, /*engram_id=*/0);
    layer->load_state_dict(
        StateDict(weights_,
                  "layers." + std::to_string(args_.engram_layer_ids().front()) +
                      ".engram."));
    layer->load_hash_state(StateDict(hash_weights_));
    layer->verify_loaded_weights();
    return layer;
  }

  torch::TensorOptions options_ = torch::TensorOptions()
                                      .device(torch::kPrivateUse1, 0)
                                      .dtype(torch::kBFloat16);
  test::MockProcessGroup group_{options_.device()};
  ModelArgs args_;
  ModelInputParams input_;
  std::unordered_map<std::string, torch::Tensor> weights_;
  std::unordered_map<std::string, torch::Tensor> hash_weights_;
  std::string saved_config_;
  std::string saved_death_test_style_;
};

TEST_F(DeepseekV41EngramTest, QuantizedWkvUsesCheckpointLayerId) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  args_.engram_layer_ids({14});
  auto smooth = torch::linspace(0.5, 1.5, 64).to(torch::kBFloat16);
  auto scale =
      torch::linspace(0.001, 0.01, 24).to(torch::kBFloat16).view({24, 1});
  auto weight = torch::arange(24 * 64)
                    .remainder(17)
                    .sub(8)
                    .to(torch::kInt8)
                    .view({24, 64});
  weights_["wkv.weight"] =
      (weight.to(torch::kFloat32) * scale * smooth).to(torch::kBFloat16);
  ParallelArgs parallel(0, 1, &group_);
  parallel.tp_group_ = &group_;
  QuantArgs quant;
  quant.quant_method() = "compressed-tensors";
  quant.is_compressed_tensors_w8a8_dynamic() = true;
  quant.compressed_groups() = {{{"layers.14.engram.wkv"}, 8, 0, true}};
  const ModelContext context(parallel, args_, quant, options_);
  auto reference = create("{}", context);
  DeepseekV41Engram layer(context, /*engram_id=*/0);
  layer->load_state_dict(
      StateDict({{"wkv.smooth", smooth}, {"wkv.weight_scale", scale}},
                "layers.14.engram."));
  weights_["wkv.weight"] = weight;
  layer->load_state_dict(StateDict(weights_, "layers.14.engram."));
  layer->load_hash_state(StateDict(hash_weights_));
  layer->verify_loaded_weights();
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden = torch::linspace(-1, 1, 64).reshape({4, 2, 8}).to(options_);
  auto actual =
      layer->forward(hidden, tokens, input_).cpu().to(torch::kFloat32);
  auto expected =
      reference->forward(hidden, tokens, input_).cpu().to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(actual, expected, 0.035, 0.035));
}

TEST_F(DeepseekV41EngramTest, DefaultHostMatchesDeviceAndSelectedRows) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto device = create(R"({"placement":"device"})");
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  auto actual = host->forward(hidden, tokens, input_);
  auto expected = device->forward(hidden, tokens, input_);
  EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()));
  auto indices = torch::tensor({3, 0}, torch::kInt64).to(options_.device());
  EXPECT_TRUE(torch::equal(
      host->forward(
              hidden.index_select(0, indices), tokens, input_, {}, indices)
          .cpu(),
      expected.index_select(0, indices).cpu()));
}

TEST_F(DeepseekV41EngramTest, ImageAndPaddingRowsPassThroughIncludingCpShards) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto layer = create("{}");
  torch::Tensor tokens = torch::tensor({129264, 5, 129265, 7}, torch::kInt64)
                             .to(options_.device());
  torch::Tensor hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  torch::Tensor actual = layer->forward(hidden, tokens, input_);
  EXPECT_TRUE(torch::equal(actual[0], hidden[0]));
  EXPECT_TRUE(torch::equal(actual[2], hidden[2]));
  EXPECT_FALSE(torch::equal(actual[1], hidden[1]));
  torch::Tensor indices =
      torch::tensor({2, 1, 0}, torch::kInt64).to(options_.device());
  torch::Tensor local = layer->forward(
      hidden.index_select(0, indices), tokens, input_, {}, indices);
  EXPECT_TRUE(torch::equal(local, actual.index_select(0, indices)));
}

TEST_F(DeepseekV41EngramTest, HostPrefetchOwnsTheProvidedTokenSnapshot) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto reference = create(R"({"placement":"device"})");
  auto state = host->create_hash_state(/*rows=*/4);
  const auto device_placeholder =
      torch::full({4}, -1, options_.dtype(torch::kInt64));
  const auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  for (int32_t round = 0; round < 3; ++round) {
    auto host_tokens =
        torch::tensor({2, 5, 1, 7}, torch::kInt32).add(round).remainder(8);
    const auto expected =
        reference->forward(hidden, host_tokens.to(options_.device()), input_)
            .cpu();
    state->prepare(input_);
    // The host snapshot can be consumed before the device input upload has
    // finished. Its lifetime must not extend into the queued CPU callback.
    state->prefetch(device_placeholder, host_tokens);
    host_tokens.fill_(-1);
    EXPECT_TRUE(torch::equal(
        host->forward(hidden, device_placeholder, input_, {state.get()}).cpu(),
        expected));
  }
}

TEST_F(DeepseekV41EngramTest,
       FirstHostUploadWaitsForAllocationStreamBeforeReusingCachedStorage) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  constexpr int64_t kRows = 4096;
  constexpr int64_t kArenaBytes = 1024 * 1024;
  const auto host_tokens = torch::arange(kRows, torch::kInt64).remainder(8);
  const auto tokens = host_tokens.to(options_.device());
  const auto hidden =
      torch::linspace(-1, 1, kRows * 2 * 8).reshape({kRows, 2, 8}).to(options_);
  input_.meta.num_sequences = 1;
  input_.meta.actual_num_sequences = 1;
  input_.attention.host.q_seq_lens = {0, kRows};
  input_.embedding.engram_lookback_token_ids = {1, 2};
  auto reference = create(R"({"placement":"device"})");
  const auto expected = reference->forward(hidden, tokens, input_).cpu();
  ASSERT_TRUE(torch::isfinite(expected).all().item<bool>());
  // Device-only producer work keeps the allocation stream busy without
  // blocking the runtime's host-callback worker used for CPU hash/lookup.
  // Warm up the GEMM algorithm and workspace before arranging the race.
  constexpr int64_t kMatrixSize = 8192;
  const auto lhs = torch::full({kMatrixSize, kMatrixSize}, 0.001, options_);
  const auto rhs = torch::full({kMatrixSize, kMatrixSize}, 0.001, options_);
  auto product = torch::empty({kMatrixSize, kMatrixSize}, options_);
  torch::mm_out(product, lhs, rhs);
  torch_mlu::synchronize();
  for (const bool external_prefetch : {false, true}) {
    SCOPED_TRACE(external_prefetch ? "host snapshot prefetch" : "eager upload");
    auto host = create(R"({"placement":"cpu"})");
    // Initialize the callback thread, pool streams and library kernels before
    // arranging the race. Releasing this state also warms the pinned host
    // allocator, so the new state does not need a fresh host allocation.
    auto warmup = host->create_hash_state(kRows);
    warmup->prepare_eager(host_tokens, input_);
    host->forward(hidden, tokens, input_, {warmup.get()}).cpu();
    torch_mlu::synchronize();
    warmup.reset();
    torch_mlu::MLUCachingAllocator::emptyCache();
    const auto bytes = options_.dtype(torch::kUInt8);
    auto arena = torch::empty({kArenaBytes}, bytes);
    const uintptr_t arena_begin = reinterpret_cast<uintptr_t>(arena.data_ptr());
    const uintptr_t arena_end = arena_begin + kArenaBytes;
    const auto allocation_stream = torch_mlu::getCurrentMLUStream();
    // Occupy every other free block in this stream's small pool. This makes
    // the arena the only reusable allocation when the state requests storage;
    // consuming exact free-block sizes does not create new segments.
    std::vector<torch::Tensor> ballast;
    for (const auto& segment :
         torch_mlu::MLUCachingAllocator::snapshot().segments) {
      if (segment.device != options_.device().index() || segment.is_large ||
          segment.stream != allocation_stream.stream()) {
        continue;
      }
      for (const auto& block : segment.blocks) {
        if (block.allocated || block.active) {
          continue;
        }
        int64_t remaining = static_cast<int64_t>(block.size);
        while (remaining > 0) {
          const int64_t count = std::min(remaining, kArenaBytes);
          ballast.emplace_back(torch::empty({count}, bytes));
          remaining -= count;
        }
      }
    }
    for (int32_t iteration = 0; iteration < 64; ++iteration) {
      torch::mm_out(product, lhs, rhs);
    }
    // 0xff poisons both FP8 data and scale bytes. Without the constructor's
    // dependency, the queued GEMMs delay this write until after the new
    // side-stream upload, overwriting its result.
    arena.fill_(255);
    arena = torch::Tensor();
    auto state = host->create_hash_state(kRows);
    const auto packed = state->lookup_packed_rows();
    const uintptr_t packed_begin =
        reinterpret_cast<uintptr_t>(packed.data_ptr());
    ASSERT_GE(packed_begin, arena_begin);
    ASSERT_LE(packed_begin + packed.numel(), arena_end)
        << "Fixture must actually reuse the pending poison allocation";
    if (external_prefetch) {
      state->prepare(input_, /*device_history=*/false);
      state->prefetch(tokens, host_tokens);
    } else {
      state->prepare_eager(host_tokens, input_);
    }
    const auto actual =
        host->forward(hidden, tokens, input_, {state.get()}).cpu();
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
    EXPECT_TRUE(torch::equal(actual, expected));
  }
}

TEST_F(DeepseekV41EngramTest, DeviceHashGraphRefreshesTokensAndGroupedHistory) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto device = create(R"({"placement":"device"})");
  auto host = create("{}");
  const auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  for (const int32_t width : {1, 2}) {
    auto state = device->create_hash_state(/*rows=*/4, width);
    auto tokens =
        torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
    input_.meta.num_sequences = 4 / width;
    input_.meta.actual_num_sequences = 4 / width;
    input_.attention.host.q_seq_lens = {0};
    for (int32_t seq = 1; seq <= 4 / width; ++seq) {
      input_.attention.host.q_seq_lens.emplace_back(seq * width);
    }
    input_.embedding.engram_lookback_token_ids.assign(8 / width, 1);
    state->prepare(input_);
    auto warmup = device->forward(hidden, tokens, input_, {state.get()});
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    torch::Tensor output;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
      graph.capture_begin();
      output = device->forward(hidden, tokens, input_, {state.get()});
      graph.capture_end();
    }
    for (int32_t round = 0; round < 3; ++round) {
      tokens.copy_(
          torch::tensor({round + 1, 7 - round, 4, round}, torch::kInt64));
      input_.embedding.engram_lookback_token_ids.assign(8 / width, round + 2);
      state->prepare(input_);
      const auto expected = host->forward(hidden, tokens, input_).cpu();
      graph.replay();
      state->mark_consumed();
      EXPECT_TRUE(torch::equal(output.cpu(), expected));
    }
  }
}

TEST_F(DeepseekV41EngramTest,
       FusedLookupAndTmoWkvMatchIndependentCpuReference) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  const auto host_tokens = torch::tensor({2, 5, 1, 7}, torch::kInt64);
  const auto tokens = host_tokens.to(options_.device());
  const auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  // Golden hashes for the fixture's two separate sequences and lookbacks.
  const auto indices = torch::tensor({1, 7, 3, 13, 1, 13, 4, 7}, torch::kInt64);
  const auto decoded = weights_.at("embed.weight")
                           .index_select(0, indices)
                           .view(torch::kFloat8_e4m3fn)
                           .to(torch::kFloat32);
  const auto scales = torch::exp2(weights_.at("embed.scale")
                                      .index_select(0, indices)
                                      .to(torch::kFloat32) -
                                  127)
                          .repeat_interleave(32, -1);
  const auto rows = (decoded * scales).to(torch::kBFloat16).reshape({4, 64});
  const auto kv =
      torch::matmul(rows, weights_.at("wkv.weight").t()).to(torch::kFloat32);
  const auto h = hidden.cpu().to(torch::kFloat32);
  const auto keys = kv.slice(1, 0, 16).reshape({4, 2, 8});
  const auto values = kv.slice(1, 16);
  auto dot = (h * keys * weights_.at("q_weight").to(torch::kFloat32) *
              weights_.at("k_weight").to(torch::kFloat32))
                 .sum(-1);
  dot *= torch::rsqrt(h.square().mean(-1) + args_.rms_norm_eps()) *
         torch::rsqrt(keys.square().mean(-1) + args_.rms_norm_eps()) /
         std::sqrt(8.0);
  const auto gate = torch::sigmoid(torch::where(dot < 0, -1.0, 1.0) *
                                   dot.abs().clamp_min(1e-6).sqrt());
  const auto expected =
      (h + gate.unsqueeze(-1) * values.unsqueeze(1)).to(torch::kBFloat16);
  for (const std::string& config :
       {std::string("{}"),
        std::string(R"({"placement":"cpu","replication":"per_node"})"),
        std::string(R"({"placement":"device"})")}) {
    auto layer = create(config);
    EXPECT_TRUE(torch::allclose(
        layer->forward(hidden, tokens, input_).cpu(), expected, 0.008, 1e-4))
        << "config=" << config;
  }
}

TEST_F(DeepseekV41EngramTest, ParallelPrefillGatherMatchesDeviceAcrossChunks) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  constexpr int64_t kTokens = 1031;
  auto host = create("{}");
  auto device = create(R"({"placement":"device"})");
  input_.meta.num_sequences = 1;
  input_.meta.actual_num_sequences = 1;
  input_.attention.host.q_seq_lens = {0, kTokens};
  auto hidden = torch::linspace(-1, 1, kTokens * 2 * 8)
                    .reshape({kTokens, 2, 8})
                    .to(options_);
  for (int32_t round = 0; round < 3; ++round) {
    auto host_tokens =
        (torch::arange(kTokens, torch::kInt64) + round).remainder(8);
    auto tokens = host_tokens.to(options_.device());
    input_.embedding.engram_lookback_token_ids = {round, round + 1};
    host->prepare_embeddings(host_tokens, input_);
    auto actual = host->forward(hidden, tokens, input_);
    auto expected = device->forward(hidden, tokens, input_);
    EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()));
  }
}

TEST_F(DeepseekV41EngramTest, RejectsEagerRowsLeftForAnotherHashSource) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto host_tokens = torch::tensor({2, 5, 1, 7}, torch::kInt64);
  auto tokens = host_tokens.to(options_.device());
  auto hidden = torch::ones({4, 2, 8}, options_);
  auto state = host->create_hash_state(/*rows=*/4);
  host->prepare_embeddings(host_tokens, input_);
  EXPECT_DEATH(host->forward(hidden, tokens, input_, {state.get()}),
               "prepared but not consumed");
}

TEST_F(DeepseekV41EngramTest, RejectsGraphInputPreparedWithoutPrefetch) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden = torch::ones({4, 2, 8}, options_);
  auto state = host->create_hash_state(/*rows=*/4);
  // Without the device history only prefetch() can supply the hashes. The
  // inline path would otherwise hash against a history it never uploaded.
  state->prepare(input_, /*device_history=*/false);
  EXPECT_DEATH(host->forward(hidden, tokens, input_, {state.get()}),
               "prefetch\\(\\) must run before forward");
  EXPECT_DEATH(state->mark_consumed(), "no prefetch supplied its hashes");
  state->prefetch(tokens);
  const auto actual = host->forward(hidden, tokens, input_, {state.get()});
  EXPECT_TRUE(torch::equal(actual.cpu(),
                           create(R"({"placement":"device"})")
                               ->forward(hidden, tokens, input_)
                               .cpu()));
}

TEST_F(DeepseekV41EngramTest, HostTokensAreRequiredOnlyForCpuHashing) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto device = create(R"({"placement":"device"})");
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  // Prefill hashes host-table rows on MLU, so no host snapshot is needed.
  input_.meta.batch_forward_type = BatchForwardType::PREFILL;
  ASSERT_TRUE(host->prefill_hashes_on_device(input_));
  host->prepare_embeddings(/*host_tokens=*/{}, input_, tokens);
  EXPECT_TRUE(torch::equal(host->forward(hidden, tokens, input_).cpu(),
                           device->forward(hidden, tokens, input_).cpu()));
  // Decode hashes on the CPU and must be given the tokens it reads.
  input_.meta.batch_forward_type = BatchForwardType::DECODE;
  ASSERT_FALSE(host->prefill_hashes_on_device(input_));
  EXPECT_DEATH(host->prepare_embeddings(/*host_tokens=*/{}, input_, tokens),
               "CPU hashing requires host tokens");
}

TEST_F(DeepseekV41EngramTest, ChunkedPrefillMatchesWholeSequenceAtBoundaries) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  constexpr int64_t kTokens = 8203;
  auto host = create(R"({"placement":"cpu","prefill_hash":"device"})");
  auto reference = create(R"({"placement":"device"})");
  auto host_tokens =
      torch::arange(kTokens, torch::kInt64).mul(3).add(1).remainder(7).add(1);
  auto tokens = host_tokens.to(options_.device());
  auto hidden = torch::linspace(-1, 1, kTokens * 2 * 8)
                    .reshape({kTokens, 2, 8})
                    .to(options_);
  input_.meta.num_sequences = 1;
  input_.meta.actual_num_sequences = 1;
  input_.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  input_.attention.host.q_seq_lens = {0, kTokens};
  input_.embedding.engram_lookback_token_ids = {6, 7};
  auto expected = reference->forward(hidden, tokens, input_).cpu();
  for (const int64_t chunk_size : {int64_t{1}, int64_t{127}, int64_t{8192}}) {
    // A one-token boundary validates suffix history; avoid thousands of MLU
    // launches by using it only for the first two chunks of this sequence.
    int64_t begin = 0;
    while (begin < kTokens) {
      const int64_t active_chunk =
          chunk_size == 1 && begin >= 2 ? int64_t{8192} : chunk_size;
      const int64_t end = std::min(begin + active_chunk, kTokens);
      input_.attention.host.q_seq_lens = {0, end - begin};
      input_.embedding.engram_lookback_token_ids = {
          begin >= 1 ? host_tokens[begin - 1].item<int32_t>() : 6,
          begin >= 2 ? host_tokens[begin - 2].item<int32_t>()
                     : (begin == 1 ? 6 : 7)};
      host->prepare_embeddings(host_tokens.slice(0, begin, end),
                               input_,
                               tokens.slice(0, begin, end));
      auto actual = host->forward(
          hidden.slice(0, begin, end), tokens.slice(0, begin, end), input_);
      EXPECT_TRUE(torch::equal(actual.cpu(), expected.slice(0, begin, end)))
          << "chunk_size=" << chunk_size << " begin=" << begin;
      begin = end;
    }
  }
}

TEST_F(DeepseekV41EngramTest, PriorityPrefillMatrixPreservesChunkHistory) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create(R"({"placement":"cpu","prefill_hash":"device"})");
  auto reference = create(R"({"placement":"device"})");
  input_.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  constexpr int64_t kChunkSize = 32768;
  const std::vector<std::pair<int64_t, int32_t>> cases = {{8192, 1},
                                                          {8192, 2},
                                                          {8192, 4},
                                                          {32768, 1},
                                                          {32768, 2},
                                                          {131072, 1},
                                                          {131089, 1}};
  for (const auto& [sequence_length, batch] : cases) {
    const int64_t rows = sequence_length * batch;
    auto host_tokens =
        torch::arange(rows, torch::kInt64).mul(3).add(1).remainder(7).add(1);
    auto tokens = host_tokens.to(options_.device());
    auto hidden =
        torch::linspace(-1, 1, rows * 2 * 8).reshape({rows, 2, 8}).to(options_);
    input_.meta.num_sequences = batch;
    input_.meta.actual_num_sequences = batch;
    input_.attention.host.q_seq_lens.resize(batch + 1);
    input_.embedding.engram_lookback_token_ids.clear();
    input_.embedding.engram_lookback_token_ids.reserve(batch * 2);
    for (int32_t seq = 0; seq <= batch; ++seq) {
      input_.attention.host.q_seq_lens[seq] =
          static_cast<int32_t>(seq * sequence_length);
    }
    for (int32_t seq = 0; seq < batch; ++seq) {
      input_.embedding.engram_lookback_token_ids.emplace_back((seq + 1) % 8);
      input_.embedding.engram_lookback_token_ids.emplace_back((seq + 2) % 8);
    }
    const auto expected = reference->forward(hidden, tokens, input_).cpu();
    const int64_t* ids = host_tokens.data_ptr<int64_t>();
    for (int64_t begin = 0; begin < rows; begin += kChunkSize) {
      const int64_t end = std::min(begin + kChunkSize, rows);
      input_.attention.host.q_seq_lens = {0};
      input_.embedding.engram_lookback_token_ids.clear();
      int64_t position = begin;
      while (position < end) {
        const int64_t seq = position / sequence_length;
        const int64_t relative = position % sequence_length;
        const int64_t part_end = std::min((seq + 1) * sequence_length, end);
        input_.attention.host.q_seq_lens.emplace_back(static_cast<int32_t>(
            input_.attention.host.q_seq_lens.back() + part_end - position));
        input_.embedding.engram_lookback_token_ids.emplace_back(
            relative > 0 ? ids[position - 1] : (seq + 1) % 8);
        input_.embedding.engram_lookback_token_ids.emplace_back(
            relative > 1 ? ids[position - 2]
                         : (relative == 1 ? (seq + 1) % 8 : (seq + 2) % 8));
        position = part_end;
      }
      input_.meta.num_sequences =
          static_cast<int32_t>(input_.attention.host.q_seq_lens.size() - 1);
      input_.meta.actual_num_sequences = input_.meta.num_sequences;
      host->prepare_embeddings(host_tokens.slice(0, begin, end),
                               input_,
                               tokens.slice(0, begin, end));
      const auto actual = host->forward(
          hidden.slice(0, begin, end), tokens.slice(0, begin, end), input_);
      EXPECT_TRUE(torch::equal(actual.cpu(), expected.slice(0, begin, end)))
          << "sequence_length=" << sequence_length << " batch=" << batch
          << " chunk_begin=" << begin;
    }
  }
}

TEST_F(DeepseekV41EngramTest, DecodeAndDraftSuffixShapesMatchDevice) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto reference = create(R"({"placement":"device"})");
  for (const int32_t batch : {8, 16, 32, 64, 128}) {
    for (const int32_t step : {1, 3, 4, 5, 6}) {
      const int64_t rows = batch * step;
      auto host_tokens = torch::arange(rows, torch::kInt64).remainder(8);
      auto tokens = host_tokens.to(options_.device());
      auto hidden = torch::linspace(-1, 1, rows * 2 * 8)
                        .reshape({rows, 2, 8})
                        .to(options_);
      input_.meta.num_sequences = batch;
      input_.meta.actual_num_sequences = batch;
      input_.attention.host.q_seq_lens.resize(batch + 1);
      for (int32_t seq = 0; seq <= batch; ++seq) {
        input_.attention.host.q_seq_lens[seq] = seq * step;
      }
      for (const int32_t accepted_suffix : {0, 3}) {
        // Host history is authoritative after acceptance or rollback. This
        // tests the Engram suffix contract, not full-model MTP support.
        input_.embedding.engram_lookback_token_ids.assign(batch * 2,
                                                          accepted_suffix);
        host->prepare_embeddings(host_tokens, input_);
        auto actual = host->forward(hidden, tokens, input_);
        auto expected = reference->forward(hidden, tokens, input_);
        EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()))
            << "batch=" << batch << " step=" << step
            << "accepted_suffix=" << accepted_suffix;
      }
    }
  }
}

TEST_F(DeepseekV41EngramTest, SharedHostMatchesDeviceWithStridedFp8Checkpoint) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto device = create(R"({"placement":"device"})");
  weights_["embed.weight"] = weights_.at("embed.weight")
                                 .view(torch::kFloat8_e4m3fn)
                                 .t()
                                 .contiguous()
                                 .t();
  weights_["embed.scale"] =
      weights_.at("embed.scale").expand({18, 2}).contiguous().slice(1, 0, 1);
  auto host = create(R"({"placement":"cpu","replication":"per_node"})");
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  EXPECT_TRUE(torch::equal(host->forward(hidden, tokens, input_).cpu(),
                           device->forward(hidden, tokens, input_).cpu()));
}

TEST_F(DeepseekV41EngramTest, HostLookupSupportsStridedCheckpointRows) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto device = create(R"({"placement":"device"})");
  weights_["embed.weight"] = weights_.at("embed.weight").t().contiguous().t();
  weights_["embed.scale"] =
      weights_.at("embed.scale").expand({18, 2}).contiguous().slice(1, 0, 1);
  ASSERT_FALSE(weights_.at("embed.weight").is_contiguous());
  ASSERT_FALSE(weights_.at("embed.scale").is_contiguous());
  auto host = create("{}");
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  EXPECT_TRUE(torch::equal(host->forward(hidden, tokens, input_).cpu(),
                           device->forward(hidden, tokens, input_).cpu()));
}

TEST_F(DeepseekV41EngramTest, HashClampsLargeTokensAndBreaksOnImageOrPadding) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  // Image hash sentinels are fixed by the Python Engram contract, regardless
  // of the model's configurable multimodal image token.
  args_.image_token_id(6);
  auto layer = create("{}");
  auto state = layer->create_hash_state(/*rows=*/5);
  input_.meta.actual_num_sequences = 5;
  input_.attention.host.q_seq_lens = {0, 1, 2, 3, 4, 5};
  input_.embedding.engram_lookback_token_ids = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  state->prepare(input_);
  auto tokens =
      torch::tensor(
          std::vector<int64_t>{
              8, std::numeric_limits<int64_t>::max(), 129264, 129265, -1},
          torch::kInt64)
          .to(options_.device());
  auto actual = state->forward(tokens).cpu();
  auto expected =
      torch::tensor({{4, 10}, {4, 10}, {0, 7}, {0, 7}, {0, 7}}, torch::kInt64);
  EXPECT_TRUE(torch::equal(actual, expected));
}

class DeepseekV41EngramGraphTest : public DeepseekV41EngramTest,
                                   public ::testing::WithParamInterface<bool> {
};

TEST_P(DeepseekV41EngramGraphTest,
       ReplaySupportsDynamicSequenceLengthsAndPerSequenceHistory) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  constexpr int64_t kCapacity = 24;
  auto reference = create(R"({"placement":"device"})");
  for (const bool external_prefetch : {false, true}) {
    auto host = create(
        GetParam()
            ? R"({"placement":"cpu","replication":"per_node","prefill_hash":"device"})"
            : R"({"placement":"cpu","replication":"none","prefill_hash":"device"})");
    auto state = host->create_hash_state(kCapacity);
    auto tokens = torch::arange(kCapacity, torch::kInt64)
                      .remainder(8)
                      .to(options_.device());
    auto hidden = torch::linspace(-1, 1, kCapacity * 2 * 8)
                      .reshape({kCapacity, 2, 8})
                      .to(options_);
    input_.meta.num_sequences = 4;
    input_.meta.actual_num_sequences = 4;
    input_.attention.host.q_seq_lens = {0, 6, 12, 18, 24};
    input_.embedding.engram_lookback_token_ids.assign(8, 2);
    input_.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    input_.enable_graph = false;
    state->prepare_eager(tokens.cpu(), input_, tokens);
    const auto prefill_output =
        host->forward(hidden, tokens, input_, {state.get()});
    EXPECT_TRUE(torch::equal(prefill_output.cpu(),
                             reference->forward(hidden, tokens, input_).cpu()));
    // Reuse the exact eager-prefill state for CPU graph preparation. Device
    // hashing must not change replay's persistent buffers or callback path.
    input_.meta.batch_forward_type = BatchForwardType::DECODE;
    input_.enable_graph = true;
    auto prepare = [&]() {
      state->prepare(input_, !external_prefetch);
      if (external_prefetch) {
        state->prefetch(tokens);
      }
    };
    prepare();
    auto warmup = host->forward(hidden, tokens, input_, {state.get()});
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    torch::Tensor output;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      output = host->forward(hidden, tokens, input_, {state.get()});
      graph.capture_end();
    }
    for (const int32_t batch : {4, 2, 1, 0, 24}) {
      const int32_t width = batch == 4 ? 6 : (batch == 24 ? 1 : 5);
      input_.meta.num_sequences = batch;
      input_.meta.actual_num_sequences = batch;
      input_.attention.host.q_seq_lens.resize(batch + 1);
      for (int32_t seq = 0; seq <= batch; ++seq) {
        input_.attention.host.q_seq_lens[seq] =
            batch == 24 ? std::max(0, seq - 1) : seq * width;
      }
      const int32_t active_rows = input_.attention.host.q_seq_lens.back();
      input_.embedding.engram_lookback_token_ids.assign(batch * 2, batch);
      tokens.copy_(torch::arange(kCapacity, torch::kInt64).remainder(8));
      tokens.slice(0, active_rows).fill_(-1);
      prepare();
      auto expected_input = input_;
      // The reference eagerly processes the complete capacity, including the
      // explicit padding sequence added below, rather than graph metadata.
      expected_input.enable_graph = false;
      if (active_rows < kCapacity) {
        ++expected_input.meta.num_sequences;
        expected_input.attention.host.q_seq_lens.emplace_back(kCapacity);
        expected_input.embedding.engram_lookback_token_ids.emplace_back(-1);
        expected_input.embedding.engram_lookback_token_ids.emplace_back(-1);
      }
      auto expected = reference->forward(hidden, tokens, expected_input);
      graph.replay();
      state->mark_consumed();
      EXPECT_TRUE(torch::equal(output.cpu(), expected.cpu()))
          << "external_prefetch=" << external_prefetch << " batch=" << batch;
    }
  }
}

TEST_P(DeepseekV41EngramGraphTest, GraphReplayRefreshesHostRowsAndHistory) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host =
      create(GetParam() ? R"({"placement":"cpu","replication":"per_node"})"
                        : R"({"placement":"cpu","replication":"none"})");
  auto device = create(R"({"placement":"device"})");
  for (const int32_t query_width : {1, 2}) {
    auto state = host->create_hash_state(/*rows=*/4, query_width);
    auto tokens =
        torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
    auto hidden =
        torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
    input_.meta.num_sequences = 4 / query_width;
    input_.meta.actual_num_sequences = 4 / query_width;
    input_.attention.host.q_seq_lens.resize(4 / query_width + 1);
    for (int32_t seq = 0; seq <= 4 / query_width; ++seq) {
      input_.attention.host.q_seq_lens[seq] = seq * query_width;
    }
    input_.embedding.engram_lookback_token_ids.assign(8 / query_width, 1);
    state->prepare(input_);
    auto warmup = host->forward(hidden, tokens, input_, {state.get()});
    EXPECT_TRUE(torch::equal(warmup.cpu(),
                             device->forward(hidden, tokens, input_).cpu()));
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    torch::Tensor output;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
      graph.capture_begin();
      output = host->forward(hidden, tokens, input_, {state.get()});
      graph.capture_end();
    }
    for (int32_t round = 0; round < 3; ++round) {
      tokens.copy_(
          torch::tensor({round + 1, 7 - round, 4, round}, torch::kInt64));
      input_.embedding.engram_lookback_token_ids.assign(8 / query_width,
                                                        round + 1);
      state->prepare(input_);
      auto expected = device->forward(hidden, tokens, input_);
      graph.replay();
      state->mark_consumed();
      EXPECT_TRUE(torch::equal(output.cpu(), expected.cpu()));
    }
  }
}

TEST_P(DeepseekV41EngramGraphTest,
       PrefetchedReplayRefreshesTokensHistoryAndPadding) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host =
      create(GetParam() ? R"({"placement":"cpu","replication":"per_node"})"
                        : R"({"placement":"cpu","replication":"none"})");
  auto reference = create(R"({"placement":"device"})");
  auto state = host->create_hash_state(/*rows=*/4);
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  input_.meta.num_sequences = 4;
  input_.meta.actual_num_sequences = 4;
  input_.attention.host.q_seq_lens = {0, 1, 2, 3, 4};
  input_.embedding.engram_lookback_token_ids.assign(8, 1);
  state->prepare(input_);
  state->prefetch(tokens);
  auto warmup = host->forward(hidden, tokens, input_, {state.get()});
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor output;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    output = host->forward(hidden, tokens, input_, {state.get()});
    graph.capture_end();
  }
  for (int32_t round = 0; round < 4; ++round) {
    // Change only history, then only tokens, then reduce the real batch size.
    if (round == 1) {
      input_.embedding.engram_lookback_token_ids.assign(8, 4);
    } else if (round == 2) {
      tokens.copy_(torch::tensor({6, 3, 4, 2}, torch::kInt64));
    } else if (round == 3) {
      input_.meta.actual_num_sequences = 2;
      input_.embedding.engram_lookback_token_ids = {5, 6, 7, 1};
      tokens.slice(0, 2).fill_(-1);
    }
    state->prepare(input_);
    state->prefetch(tokens);
    auto expected_input = input_;
    expected_input.embedding.engram_lookback_token_ids.resize(8, -1);
    auto expected = reference->forward(hidden, tokens, expected_input);
    graph.replay();
    state->mark_consumed();
    EXPECT_TRUE(torch::equal(output.cpu(), expected.cpu()))
        << "round=" << round;
  }
}

TEST_P(DeepseekV41EngramGraphTest,
       CaptureDiscardDoesNotLeaveAnUnexecutedConsumerFence) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto layer =
      create(GetParam() ? R"({"placement":"cpu","replication":"per_node"})"
                        : R"({"placement":"cpu","replication":"none"})");
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden = torch::ones({4, 2, 8}, options_);
  input_.meta.num_sequences = 4;
  input_.meta.actual_num_sequences = 4;
  input_.attention.host.q_seq_lens = {0, 1, 2, 3, 4};
  input_.embedding.engram_lookback_token_ids.assign(8, 1);
  // The first state is discarded without replay. A fresh state then captures
  // and executes normally, including reuse after a changed history.
  for (int32_t attempt = 0; attempt < 2; ++attempt) {
    auto state = layer->create_hash_state(4);
    state->prepare(input_, false);
    state->prefetch(tokens);
    // Warm up lazy library state without recording any consumer fence on this
    // HashState, so capture-only destruction exercises an unrecorded notifier.
    layer->forward(hidden, tokens, input_).cpu();
    {
      torch_mlu::MLUGraph graph;
      torch::Tensor output;
      {
        torch_mlu::mlu::MLUStreamGuard stream(
            torch_mlu::getStreamFromPool(false, 0));
        graph.capture_begin();
        output = layer->forward(hidden, tokens, input_, {state.get()});
        graph.capture_end();
      }
      if (attempt != 0) {
        graph.replay();
        state->mark_consumed();
        EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
        input_.embedding.engram_lookback_token_ids.assign(8, 4);
        state->prepare(input_, false);
        state->prefetch(tokens);
        graph.replay();
        state->mark_consumed();
        EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
      }
    }
    state.reset();
  }
}

TEST_F(DeepseekV41EngramTest, EarlyEagerStagingRefreshesChunkHistory) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto reference = create(R"({"placement":"device"})");
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  for (int32_t round = 0; round < 4; ++round) {
    auto cpu_tokens =
        torch::tensor({round + 1, 5, 7 - round, 2}, torch::kInt64);
    auto tokens = cpu_tokens.to(options_.device());
    input_.embedding.engram_lookback_token_ids = {round, 2, 3, 7 - round};
    host->prepare_embeddings(cpu_tokens, input_);
    auto output = host->forward(hidden, tokens, input_);
    EXPECT_TRUE(torch::equal(output.cpu(),
                             reference->forward(hidden, tokens, input_).cpu()));
  }
}

TEST_F(DeepseekV41EngramTest,
       EagerHashModesPreserveHashesAndOutputsAcrossBatchTransitions) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  for (const std::string config :
       {"{}",
        R"({"placement":"cpu"})",
        R"({"placement":"cpu","prefill_hash":"cpu"})",
        R"({"placement":"cpu","prefill_hash":"device"})"}) {
    auto host = create(config);
    auto cpu_state = host->create_hash_state(/*rows=*/4);
    auto candidate_state = host->create_hash_state(/*rows=*/4);
    const auto hidden =
        torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
    std::vector<torch::Tensor> actual_hashes, expected_hashes;
    std::vector<torch::Tensor> actual_outputs, expected_outputs;
    constexpr int32_t kRounds = 10;
    std::vector<torch::Tensor> host_inputs, device_inputs;
    host_inputs.reserve(kRounds);
    device_inputs.reserve(kRounds);
    for (int32_t round = 0; round < kRounds; ++round) {
      host_inputs.emplace_back(torch::tensor(
          std::vector<int64_t>{round % 8,
                               round == 2 ? 129265 : 5,
                               std::numeric_limits<int64_t>::max(),
                               -1},
          torch::kInt64));
      device_inputs.emplace_back(host_inputs.back().to(options_.device()));
    }
    actual_hashes.reserve(kRounds);
    expected_hashes.reserve(kRounds);
    actual_outputs.reserve(kRounds);
    expected_outputs.reserve(kRounds);
    for (int32_t round = 0; round < kRounds; ++round) {
      input_.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
      input_.enable_graph = round == 5;
      input_.is_spec_verify = round == 6;
      input_.meta.num_speculative_tokens = round == 7 ? 5 : 0;
      input_.use_dflash_proposal_attention = round == 8;
      if (round == 0) {
        input_.meta.batch_forward_type = BatchForwardType::PREFILL;
      }
      if (round == 3) {
        input_.meta.batch_forward_type = BatchForwardType::DECODE;
      }
      if (round == 4) {
        input_.meta.batch_forward_type = BatchForwardType::MIXED;
      }
      input_.embedding.engram_lookback_token_ids = {
          round % 8, -1, 129264, (round + 2) % 8};
      const auto& host_tokens = host_inputs[round];
      const auto& tokens = device_inputs[round];
      cpu_state->prepare_eager(host_tokens, input_);
      // The legacy call explicitly selects CPU hashing. Reentering it after
      // Device prefill must not retain any previous choice or stale history.
      candidate_state->prepare_eager(
          host_tokens, input_, round == 1 ? torch::Tensor{} : tokens);
      expected_hashes.emplace_back(cpu_state->forward(tokens).clone());
      actual_hashes.emplace_back(candidate_state->forward(tokens).clone());
      expected_outputs.emplace_back(
          host->forward(hidden, tokens, input_, {cpu_state.get()}));
      actual_outputs.emplace_back(
          host->forward(hidden, tokens, input_, {candidate_state.get()}));
      // Queue the next prepare before reading outputs back. Its side stream
      // must respect the preceding forward's consumed dependency.
    }
    for (int32_t round = 0; round < kRounds; ++round) {
      EXPECT_TRUE(torch::equal(actual_hashes[round].cpu(),
                               expected_hashes[round].cpu()))
          << "config=" << config << " round=" << round;
      EXPECT_TRUE(torch::equal(actual_outputs[round].cpu(),
                               expected_outputs[round].cpu()))
          << "config=" << config << " round=" << round;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(HostPlacement,
                         DeepseekV41EngramGraphTest,
                         ::testing::Bool());

TEST_F(DeepseekV41EngramTest, HostLookupMasksUnownedShardRows) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  for (int32_t rank = 0; rank < 2; ++rank) {
    test::MockProcessGroup shard_group(options_.device(), rank, 2);
    ParallelArgs parallel(rank, 2, &shard_group);
    parallel.tp_group_ = &shard_group;
    ModelContext context(parallel, args_, QuantArgs(), options_);
    // Construct with the sharded group; the mock reduction leaves this rank's
    // masked contribution intact, so compare it with the same device shard.
    auto host_shard = create("{}", context);
    auto device_shard = create(R"({"placement":"device"})", context);
    auto tokens =
        torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
    auto hidden = torch::ones({4, 2, 8}, options_);
    EXPECT_TRUE(
        torch::equal(host_shard->forward(hidden, tokens, input_).cpu(),
                     device_shard->forward(hidden, tokens, input_).cpu()));
  }
}

TEST_F(DeepseekV41EngramTest, RejectsUnknownOrUnsupportedConfiguration) {
  EXPECT_DEATH(create("true"), "must be a JSON object");
  EXPECT_DEATH(create("false"), "must be a JSON object");
  for (const std::string key : {"cpu_offload",
                                "tp_shared_memory",
                                "node_shared_memory",
                                "dp_shared_memory"}) {
    EXPECT_DEATH(create("{\"" + key + "\":true}"), "Unknown Engram");
  }
  EXPECT_DEATH(create(R"({"placement":true})"), "must be a string");
  EXPECT_DEATH(create(R"({"replication":true})"), "must be a string");
  EXPECT_DEATH(create(R"({"prefill_hash":true})"), "must be a string");
  EXPECT_DEATH(create(R"({"prefill_hash":"automatic"})"),
               "prefill_hash must be cpu or device");
  EXPECT_DEATH(create(R"({"placement":"disk"})"),
               "placement must be cpu or device");
  EXPECT_DEATH(create(R"({"replication":"per_rank"})"), "replication must be");
  EXPECT_DEATH(create(R"({"placement":"device","replication":"per_node"})"),
               "requires placement=cpu");
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, nullptr);
  parallel.tp_group_ = &group_;
  ModelContext context(parallel, args_, QuantArgs(), options_);
  EXPECT_DEATH(
      create(R"({"placement":"cpu","replication":"per_node"})", context),
      "requires the model world process group");
}

TEST_F(DeepseekV41EngramTest, RejectsInvalidPadAndLayerConfiguration) {
  args_.engram_pad_token_id(-1);
  EXPECT_DEATH(create("{}"), "engram_pad_token_id must be in");
  args_.engram_pad_token_id(8);
  EXPECT_DEATH(create("{}"), "engram_pad_token_id must be in");
  args_.engram_pad_token_id(0).engram_layer_ids({});
  EXPECT_DEATH(create("{}"), "must have the same length");
  args_.engram_num_embeddings({});
  EXPECT_DEATH(create("{}"), "Invalid Engram layer index");
}

TEST_F(DeepseekV41EngramTest, RejectsInvalidHashStateBeforeHashing) {
  hash_weights_["token_map"] = torch::arange(7, torch::kInt64);
  EXPECT_DEATH(create("{}"), "token_map must cover vocab_size");
  hash_weights_["token_map"] = torch::arange(8, torch::kInt64);
  hash_weights_["token_map"][0] = -1;
  EXPECT_DEATH(create("{}"), "negative compressed id");
  hash_weights_["token_map"][0] = 8;
  EXPECT_DEATH(create("{}"), "exceeds compressed_vocab_size");
  hash_weights_["token_map"][0] = 0;
  hash_weights_["multipliers"][0][0] = std::numeric_limits<int64_t>::max();
  EXPECT_DEATH(create("{}"), "multiplier can overflow");
}
}  // namespace
}  // namespace xllm::layer
