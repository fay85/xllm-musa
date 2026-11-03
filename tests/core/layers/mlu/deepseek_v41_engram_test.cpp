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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>

#include <limits>

#include "framework/config/model_config.h"
#include "layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {
class DeepseekV41EngramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_config_ = ModelConfig::get_instance().engram_config();
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
    layer->load_state_dict(StateDict(weights_));
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
};

TEST_F(DeepseekV41EngramTest, DefaultHostMatchesDevice) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create("{}");
  auto device = create(R"({"cpu_offload":false})");
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  auto actual = host->forward(hidden, tokens, input_);
  auto expected = device->forward(hidden, tokens, input_);
  EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()));
}

TEST_F(DeepseekV41EngramTest, HostLookupSupportsStridedCheckpointRows) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto device = create(R"({"cpu_offload":false})");
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

TEST_F(DeepseekV41EngramTest, GraphReplayRefreshesHostRowsAndHistory) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  auto host = create(R"({"cpu_offload":true})");
  auto device = create(R"({"cpu_offload":false})");
  auto state = host->create_hash_state(/*rows=*/4);
  auto tokens =
      torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options_);
  input_.meta.num_sequences = 4;
  input_.meta.actual_num_sequences = 4;
  input_.attention.host.q_seq_lens.resize(4 + 1);
  for (int32_t seq = 0; seq <= 4; ++seq) {
    input_.attention.host.q_seq_lens[seq] = seq;
  }
  input_.embedding.engram_lookback_token_ids.assign(8, 1);
  state->prepare(input_);
  auto warmup = host->forward(hidden, tokens, input_, state.get());
  EXPECT_TRUE(torch::equal(warmup.cpu(),
                           device->forward(hidden, tokens, input_).cpu()));
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor output;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    graph.capture_begin();
    output = host->forward(hidden, tokens, input_, state.get());
    graph.capture_end();
  }
  for (int32_t round = 0; round < 3; ++round) {
    tokens.copy_(
        torch::tensor({round + 1, 7 - round, 4, round}, torch::kInt64));
    input_.embedding.engram_lookback_token_ids.assign(8, round + 1);
    state->prepare(input_);
    auto expected = device->forward(hidden, tokens, input_);
    graph.replay();
    EXPECT_TRUE(torch::equal(output.cpu(), expected.cpu()));
  }
}

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
    auto device_shard = create(R"({"cpu_offload":false})", context);
    auto tokens =
        torch::tensor({2, 5, 1, 7}, torch::kInt64).to(options_.device());
    auto hidden = torch::ones({4, 2, 8}, options_);
    EXPECT_TRUE(
        torch::equal(host_shard->forward(hidden, tokens, input_).cpu(),
                     device_shard->forward(hidden, tokens, input_).cpu()));
  }
}

TEST_F(DeepseekV41EngramTest, RejectsUnknownOrUnsupportedConfiguration) {
  EXPECT_DEATH(create(R"({"cpu_offlaod":true})"), "Unknown Engram");
  EXPECT_DEATH(create(R"({"cpu_offload":"true"})"), "must be boolean");
  EXPECT_DEATH(create(R"({"dp_shared_memory":true})"),
               "does not support dp_shared_memory");
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
