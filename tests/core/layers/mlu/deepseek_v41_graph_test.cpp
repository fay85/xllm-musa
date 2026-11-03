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

#include <framework/core/MLUStream.h>
#include <framework/core/caching_allocator.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/mlu/piecewise_graph.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_attention.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"
#include "layers/mlu/tests_utils.h"
namespace xllm::layer {
namespace {
TEST(DeepseekV41GraphTest,
     NativeHashStateNormalizesTokensAndMatchesTrainingRng) {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("xllm_engram_tokenizer_" + std::to_string(getpid()) + ".json");
  {
    std::ofstream file(path);
    file << R"json({"version":"1.0","truncation":null,"padding":null,
      "added_tokens":[],"normalizer":null,"pre_tokenizer":null,
      "post_processor":null,"decoder":null,"model":{"type":"WordLevel",
      "vocab":{"A":0,"a":1,"É":2,"e":3," ":4," \t":5,"�":6},
      "unk_token":"�"}})json";
  }
  ModelArgs args;
  args.vocab_size() = 7;
  args.engram_compressed_vocab_size() = 4;
  args.engram_layer_ids() = {1, 14};
  args.engram_max_ngram_size() = 4;
  args.engram_n_heads() = 2;
  args.engram_vocab_size() = 10;
  args.engram_num_embeddings() = {200, 300};
  auto state = build_deepseek_v41_hash_state(args, path.string());
  std::filesystem::remove(path);
  EXPECT_TRUE(
      torch::equal(state.get_tensor("token_map"),
                   torch::tensor({0, 0, 1, 1, 2, 2, 3}, torch::kInt32)));
  // Golden NumPy PCG64 draws; offsets/primes are globally unique per head.
  EXPECT_TRUE(
      torch::equal(state.get_tensor("multipliers").flatten(),
                   torch::tensor(std::vector<int64_t>{1898406915353658309LL,
                                                      119898250459655815LL,
                                                      890828962367240933LL,
                                                      1832888310856756327LL,
                                                      1677548552443721761LL,
                                                      1276077216879092079LL,
                                                      766014534253018689LL,
                                                      2046726097727588497LL},
                                 torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      state.get_tensor("primes").flatten(),
      torch::tensor({11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53},
                    torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      state.get_tensor("offsets").flatten(),
      torch::tensor({0, 11, 24, 41, 60, 83, 0, 31, 68, 109, 152, 199},
                    torch::kInt64)));
}

TEST(DeepseekV41GraphTest, NativeHashStateRejectsInvalidTokenizerJson) {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("xllm_invalid_engram_tokenizer_" + std::to_string(getpid()) + ".json");
  ModelArgs args;
  args.vocab_size(7)
      .engram_compressed_vocab_size(4)
      .engram_layer_ids({1})
      .engram_num_embeddings({200})
      .engram_max_ngram_size(4)
      .engram_n_heads(2)
      .engram_vocab_size(10);
  {
    std::ofstream file(path);
    file << "{";
  }
  EXPECT_DEATH(build_deepseek_v41_hash_state(args, path.string()),
               "Cannot parse Engram tokenizer JSON: .*xllm_invalid_engram");
  {
    std::ofstream file(path);
    file << "[]";
  }
  EXPECT_DEATH(build_deepseek_v41_hash_state(args, path.string()),
               "tokenizer JSON must be an object");
  {
    std::ofstream file(path);
    file << "{}";
  }
  EXPECT_DEATH(build_deepseek_v41_hash_state(args, path.string()),
               "tokenizer JSON requires a model object");
  std::filesystem::remove(path);
}

TEST(DeepseekV41GraphTest, NativeHashStateMatchesFullCheckpointReference) {
  const char* model = std::getenv("XLLM_TEST_DEEPSEEK_V41_MODEL");
  const char* reference = std::getenv("XLLM_TEST_DEEPSEEK_V41_HASH_REFERENCE");
  if (model == nullptr || reference == nullptr) {
    GTEST_SKIP()
        << "Set model and hash reference paths for full tokenizer parity";
  }
  std::ifstream file(std::string(model) + "/config.json");
  const auto config = nlohmann::json::parse(file).at("text_config");
  ModelArgs args;
  args.vocab_size() = config.at("vocab_size");
  args.engram_compressed_vocab_size() =
      config.at("engram_compressed_vocab_size");
  args.engram_layer_ids() =
      config.at("engram_layer_ids").get<std::vector<int32_t>>();
  args.engram_max_ngram_size() = config.at("engram_max_ngram_size");
  args.engram_n_heads() = config.at("engram_n_heads");
  args.engram_vocab_size() = config.at("engram_vocab_size");
  args.engram_num_embeddings() =
      config.at("engram_num_embeddings").get<std::vector<int64_t>>();
  auto actual = build_deepseek_v41_hash_state(
      args, std::string(model) + "/tokenizer.json");
  auto expected = StateDictFromSafeTensor::load(reference);
  ASSERT_NE(expected, nullptr);
  for (const std::string name :
       {"token_map", "multipliers", "primes", "offsets"}) {
    EXPECT_TRUE(torch::equal(actual.get_tensor(name).flatten(),
                             expected->get_tensor(name).flatten()))
        << name;
  }
}

TEST(DeepseekV41GraphTest, HashReplayUpdatesTokensHistoryAndPadding) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  auto options = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.vocab_size() = 8;
  args.engram_layer_ids() = {0};
  args.engram_num_embeddings() = {64};
  args.engram_compressed_vocab_size() = 8;
  args.engram_head_dim() = 32;
  args.engram_n_heads() = 2;
  args.engram_max_ngram_size() = 3;
  args.engram_pad_token_id() = 0;
  DeepseekV41Engram engram(ModelContext(parallel, args, QuantArgs(), options),
                           0);
  engram->load_state_dict(StateDict({
      {"embed.weight", torch::zeros({64, 32}, torch::kUInt8)},
      {"embed.scale", torch::full({64, 1}, 127, torch::kUInt8)},
  }));
  engram->load_hash_state(StateDict(
      {{"token_map", torch::arange(8, torch::kInt64)},
       {"multipliers", torch::tensor({3, 5, 7}, torch::kInt64).view({1, 3})},
       {"primes",
        torch::tensor({11, 13, 17, 19}, torch::kInt64).view({1, 2, 2})},
       {"offsets",
        torch::tensor({0, 11, 24, 41}, torch::kInt64).view({1, 4})}}));
  auto state = engram->create_hash_state(2);
  ModelInputParams input;
  input.meta.actual_num_sequences = 2;
  input.embedding.engram_lookback_token_ids = {4, 5, 6, 7};
  state->prepare(input);
  auto tokens = torch::tensor({2, 3}, options.dtype(torch::kInt64));
  auto output = state->forward(tokens);
  torch_mlu::synchronize();
  auto expected =
      torch::tensor({7, 16, 39, 52, 1, 21, 28, 41}, torch::kInt64).view({2, 4});
  EXPECT_TRUE(torch::equal(output.cpu(), expected));
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    output = state->forward(tokens);
    graph.capture_end();
  }
  input.embedding.engram_lookback_token_ids = {6, 7, 4, 5};
  state->prepare(input);
  tokens.copy_(torch::tensor({3, 2}, tokens.options()));
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(output.cpu(), expected.flip({0})));
  input.meta.actual_num_sequences = 1;
  input.embedding.engram_lookback_token_ids = {4, 5};
  state->prepare(input);
  tokens.copy_(torch::tensor({2, 0}, tokens.options()));
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(output[0].cpu(), expected[0]));
  EXPECT_TRUE(torch::equal(output[1].cpu(),
                           torch::tensor({0, 11, 24, 41}, torch::kInt64)));
}

class DeepseekV41PagedIndexerTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(DeepseekV41PagedIndexerTest, ReturnsOnlyLiveUniqueLogicalIndices) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  torch::manual_seed(43);
  const int64_t length = GetParam();
  const int64_t topk = 512;
  const int64_t count = std::min(length, topk);
  auto options = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  auto cache = torch::randn({6, 1, 128, 128}, options);
  auto query = torch::randn({1, 1, 32, 128}, options);
  auto weights = torch::randn({1, 1, 32}, options.dtype(torch::kFloat32)) / 64;
  auto table = torch::tensor({5, 3, 1, 0, 4, 2}, options.dtype(torch::kInt32))
                   .view({1, 6});
  auto identity = torch::arange(6, table.options()).view({1, 6});
  auto lengths = torch::tensor({length}, table.options());
  auto indices = torch::full({1, 1, topk}, -1, table.options());
  auto selected_lengths = torch::zeros_like(lengths);
  kernel::mlu::masked_indexer_select_paged_kv(query,
                                              cache,
                                              weights,
                                              identity,
                                              std::nullopt,
                                              std::nullopt,
                                              lengths,
                                              table,
                                              false,
                                              topk,
                                              128,
                                              1.0,
                                              std::nullopt,
                                              std::nullopt,
                                              indices,
                                              selected_lengths,
                                              true,
                                              1,
                                              std::nullopt);
  torch_mlu::synchronize();
  auto keys = cache.cpu()
                  .index_select(0, table.cpu().flatten().to(torch::kInt64))
                  .reshape({768, 128})
                  .slice(0, 0, length)
                  .to(torch::kFloat32);
  auto scores =
      (torch::matmul(query.cpu().view({32, 128}).to(torch::kFloat32), keys.t())
           .relu() *
       weights.cpu().view({32, 1}))
          .sum(0);
  // Match select_decode_indices: only min(length, topk) entries reach gather.
  // The kernel's remaining output capacity has no initialized-value contract.
  auto selected = indices.cpu().flatten().slice(0, 0, count).to(torch::kInt64);
  ASSERT_EQ(selected_lengths.cpu().item<int32_t>(), count);
  ASSERT_EQ(selected.numel(), count);
  ASSERT_GE(selected.min().item<int64_t>(), 0);
  ASSERT_LT(selected.max().item<int64_t>(), length);
  auto mask = torch::zeros({length}, torch::kBool);
  mask.index_fill_(0, selected, true);
  EXPECT_EQ(mask.sum().item<int64_t>(), count);
  EXPECT_GE(scores.index_select(0, selected).min().item<float>() + 0.02f,
            std::get<0>(scores.topk(count)).min().item<float>());
}

INSTANTIATE_TEST_SUITE_P(
    PageAndTopkBoundaries,
    DeepseekV41PagedIndexerTest,
    ::testing::Values(1, 127, 128, 129, 511, 512, 513, 700));

TEST(DeepseekV41GraphTest, SparseGatherAndSinkAttentionReplayUseLiveLengths) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  torch::manual_seed(41);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  auto swa = torch::randn({384, 512}, options);
  auto comp = torch::randn_like(swa);
  auto current = torch::randn({512}, options);
  auto page = torch::tensor({1, 0}, options.dtype(torch::kInt32));
  auto comp_page = torch::tensor({2, 1}, page.options());
  auto topk = torch::tensor({2, 129}, options.dtype(torch::kInt64));
  auto lengths = torch::zeros({2}, page.options());
  auto cache = torch::full(
      {16, 1, 1, 512}, std::numeric_limits<float>::quiet_NaN(), options);
  kernel::mlu::bangc::deepseek_v41_decode_gather(swa,
                                                 comp,
                                                 current,
                                                 page,
                                                 comp_page,
                                                 topk,
                                                 129,
                                                 4,
                                                 cache.slice(0, 0, 8),
                                                 lengths.slice(0, 0, 1));
  kernel::mlu::bangc::deepseek_v41_decode_gather(swa,
                                                 comp,
                                                 current,
                                                 page,
                                                 comp_page,
                                                 topk.slice(0, 0, 0),
                                                 0,
                                                 4,
                                                 cache.slice(0, 8, 16),
                                                 lengths.slice(0, 1, 2));
  auto q = torch::randn({2, 1, 8, 512}, options);
  auto sink = torch::randn({8}, options.dtype(torch::kFloat32));
  auto table = torch::arange(16, lengths.options()).view({2, 1, 8});
  auto output = torch::empty_like(q);
  auto run = [&]() {
    std::optional<torch::Tensor> lse;
    kernel::mlu::batch_decode(q,
                              cache,
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
                              8,
                              -1,
                              -1,
                              1.0 / std::sqrt(512),
                              false,
                              -1,
                              std::nullopt,
                              -1,
                              sink);
  };
  auto check = [&](int64_t row, torch::Tensor selected) {
    auto query = q[row][0].cpu().to(torch::kFloat32);
    auto keys = selected.cpu().to(torch::kFloat32);
    auto scores = torch::matmul(query, keys.t()) / std::sqrt(512);
    auto probability = torch::cat({scores, sink.cpu().view({8, 1})}, 1)
                           .softmax(1)
                           .slice(1, 0, keys.size(0));
    auto reference = torch::matmul(probability, keys);
    EXPECT_TRUE(torch::allclose(
        output[row][0].cpu().to(torch::kFloat32), reference, 0.02, 0.02));
  };
  run();
  torch_mlu::synchronize();
  auto selected = torch::cat(
      {swa.index_select(0, torch::tensor({254, 255, 0}, topk.options())),
       current.unsqueeze(0),
       comp.index_select(0, torch::tensor({258, 129}, topk.options()))},
      0);
  EXPECT_TRUE(
      torch::equal(cache.view({2, 8, 512})[0].slice(0, 0, 6), selected));
  EXPECT_TRUE(torch::equal(swa[1], current));
  check(0, selected);
  check(1, current.unsqueeze(0));
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    run();
    graph.capture_end();
  }
  lengths.copy_(torch::tensor({1, 0}, lengths.options()));
  kernel::mlu::bangc::deepseek_v41_decode_gather(swa,
                                                 comp,
                                                 current,
                                                 page,
                                                 comp_page,
                                                 topk.slice(0, 0, 0),
                                                 0,
                                                 4,
                                                 cache.slice(0, 0, 8),
                                                 lengths.slice(0, 0, 1));
  output.fill_(std::numeric_limits<float>::quiet_NaN());
  graph.replay();
  torch_mlu::synchronize();
  check(0, current.unsqueeze(0));
  EXPECT_TRUE(torch::isfinite(output[1]).all().item<bool>());
}

ModelArgs gather_test_args() {
  ModelArgs args;
  args.hidden_size(128)
      .n_heads(8)
      .head_dim(512)
      .q_lora_rank(128)
      .qk_rope_head_dim(64)
      .o_groups(1)
      .o_lora_rank(128)
      .rms_norm_eps(1e-5)
      .window_size(128)
      .index_topk(512)
      .compress_ratios({0, 2, 2, 1, 1, 0})
      .kv_source_layer_ids({0});
  return args;
}

std::vector<DeepseekV41Attention> gather_test_layers(
    const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& options = context.get_tensor_options();
  const int64_t hidden = args.hidden_size();
  const int64_t rank = args.q_lora_rank();
  const int64_t dim = args.head_dim();
  const int64_t heads = args.n_heads();
  std::vector<DeepseekV41Attention> layers;
  layers.reserve(args.compress_ratios().size());
  for (int32_t id = 0; id < args.compress_ratios().size(); ++id) {
    layers.emplace_back(context, id);
    layers.back()->load_state_dict(StateDict({
        {"wq_a.weight", torch::randn({rank, hidden}, options) * 0.05},
        {"wq_b.weight", torch::randn({heads * dim, rank}, options) * 0.05},
        {"wkv.weight", torch::randn({dim, hidden}, options) * 0.05},
        {"q_norm.weight", torch::ones({rank}, options)},
        {"kv_norm.weight", torch::ones({dim}, options)},
        {"wo_a.weight",
         torch::randn({args.o_groups(), args.o_lora_rank(), heads * dim},
                      options) *
             0.05},
        {"wo_b.weight",
         torch::randn({hidden, args.o_groups() * args.o_lora_rank()}, options) *
             0.05},
        {"attn_sink", torch::randn({heads}, options.dtype(torch::kFloat32))},
    }));
  }
  return layers;
}

// Capture records callbacks without executing them, so no populated KV caches
// are needed to check retained gather storage through the real forward path.
TEST(DeepseekV41GraphTest, CaptureUsesOneGatherBufferAcrossLayers) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const torch::NoGradGuard no_grad;
  torch::manual_seed(41);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  const ModelArgs args = gather_test_args();
  const ModelContext context(parallel, args, QuantArgs(), options);
  auto attention = gather_test_layers(context);
  const int64_t batch = 4;
  const int64_t window = args.window_size();
  const int64_t capacity = window + args.index_topk();
  const int64_t dim = args.head_dim();
  auto hidden = torch::zeros({batch, args.hidden_size()}, options);
  auto positions = torch::zeros({batch}, options.dtype(torch::kInt64));
  auto rope = torch::ones({1, 128}, options.dtype(torch::kFloat32));
  std::vector<KVCache> caches;
  Csa2Context shared;
  ModelInputParams input;
  input.enable_graph = true;
  const int32_t kv_max_seq_len = 1;
  const auto pool = torch_mlu::graph_pool_handle();
  auto graph = std::make_unique<kernel::mlu::PiecewiseGraph>(kv_max_seq_len);
  input.graph.eager_context = graph.get();
  shared.selected_kv = torch::empty({batch * capacity, 1, 1, dim}, options);
  shared.selected_lengths = torch::empty({batch}, options.dtype(torch::kInt32));
  const void* shared_address = shared.selected_kv.data_ptr();
  const int64_t swa_bytes = batch * window * dim * hidden.element_size();
  const int64_t compressed_bytes =
      batch * capacity * dim * hidden.element_size();
  // Count live allocations, not reserved pool bytes. The projection sizes above
  // are smaller than either gather allocation, so exact requested sizes
  // identify the gather buffers without exposing production internals to the
  // test.
  auto check_buffers = [&](int64_t expected_swa, int64_t expected_compressed) {
    int64_t swa_count = 0;
    int64_t compressed_count = 0;
    for (const auto& segment :
         torch_mlu::MLUCachingAllocator::snapshot().segments) {
      if (segment.device != device.index()) {
        continue;
      }
      size_t address = segment.address;
      for (const auto& block : segment.blocks) {
        if (block.allocated &&
            (segment.owner_private_pool_id == pool ||
             reinterpret_cast<const void*>(address) == shared_address)) {
          swa_count += block.requested_size == swa_bytes;
          compressed_count += block.requested_size == compressed_bytes;
        }
        address += block.size;
      }
    }
    EXPECT_EQ(swa_count, expected_swa);
    EXPECT_EQ(compressed_count, expected_compressed);
  };
  check_buffers(/*expected_swa=*/0, /*expected_compressed=*/1);
  torch_mlu::synchronize();
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    graph->capture_begin(pool);
    for (auto& layer : attention) {
      // Discard each output: no test-owned tensor retains the gather storage.
      layer->forward(hidden, positions, rope, caches, input, shared);
    }
    graph->capture_end();
  }
  torch_mlu::synchronize();
  EXPECT_EQ(graph->num_eager_segments(), attention.size());
  check_buffers(/*expected_swa=*/0, /*expected_compressed=*/1);
  input.graph.eager_context = nullptr;
  graph.reset();
  shared.selected_kv = torch::Tensor();
  shared.selected_lengths = torch::Tensor();
  torch_mlu::synchronize();
  check_buffers(/*expected_swa=*/0, /*expected_compressed=*/0);
}

// Keep every callback dependency alive until the graph has been destroyed.
struct AttentionReplayTrial {
  ModelInputParams input;
  Csa2Context shared;
  std::vector<Csa2Context> independent;
  std::vector<KVCache> caches;
  std::vector<KVCache> reference_caches;
  torch::Tensor hidden;
  torch::Tensor positions;
  torch::Tensor rope;
  std::vector<torch::Tensor> outputs;
  std::unique_ptr<kernel::mlu::PiecewiseGraph> graph;
};

TEST(DeepseekV41GraphTest, SharedGatherReplayMatchesIndependentLayerBuffers) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const torch::NoGradGuard no_grad;
  torch::manual_seed(73);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  const ModelArgs args = gather_test_args();
  // Supply a fixed source cache and explicit selections so the comparison
  // isolates scratch aliasing from compression/indexer numerical differences.
  const ModelContext context(parallel, args, QuantArgs(), options);
  auto layers = gather_test_layers(context);
  const int64_t dim = args.head_dim();
  const int64_t max_capacity = args.window_size() + args.index_topk();
  const int32_t kv_max_seq_len = 256;
  auto make_trial = [&](int64_t batch) {
    auto trial = std::make_unique<AttentionReplayTrial>();
    trial->input.enable_graph = true;
    trial->input.meta.num_sequences = batch;
    trial->shared.selected_kv =
        torch::empty({batch * max_capacity, 1, 1, dim}, options);
    trial->shared.selected_lengths =
        torch::empty({batch}, options.dtype(torch::kInt32));
    trial->shared.topk.resize(batch);
    auto table =
        torch::arange(batch, options.dtype(torch::kInt32)).view({batch, 1});
    trial->shared.block_tables = {table, table, table};
    trial->hidden = torch::randn({batch, args.hidden_size()}, options);
    trial->positions = torch::zeros({batch}, options.dtype(torch::kInt64));
    trial->rope =
        torch::cat({torch::ones({256, 64}, options.dtype(torch::kFloat32)),
                    torch::zeros({256, 64}, options.dtype(torch::kFloat32))},
                   1);
    trial->caches.reserve(layers.size());
    trial->reference_caches.reserve(layers.size());
    trial->independent.reserve(layers.size());
    trial->outputs.reserve(layers.size());
    for (size_t id = 0; id < layers.size(); ++id) {
      DeepSeekV4KVCacheTensors tensors;
      tensors.swa_cache = torch::randn({batch, 1, 128, 512}, options);
      tensors.key_cache = torch::randn({batch, 1, 128, 512}, options);
      tensors.index_cache = torch::zeros({batch, 1, 128, 128}, options);
      trial->caches.emplace_back(tensors);
      tensors.swa_cache = tensors.swa_cache.clone();
      trial->reference_caches.emplace_back(tensors);
      Csa2Context independent;
      independent.block_tables = trial->shared.block_tables;
      independent.topk.resize(batch);
      const int64_t capacity =
          args.window_size() +
          (args.compress_ratios()[id] > 0 ? args.index_topk() : 0);
      independent.selected_kv =
          torch::empty({batch * capacity, 1, 1, dim}, options);
      independent.selected_lengths =
          torch::empty({batch}, options.dtype(torch::kInt32));
      trial->independent.emplace_back(std::move(independent));
    }
    trial->graph =
        std::make_unique<kernel::mlu::PiecewiseGraph>(kv_max_seq_len);
    trial->input.graph.eager_context = trial->graph.get();
    torch_mlu::synchronize();
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
      trial->graph->capture_begin(torch_mlu::graph_pool_handle());
      torch::Tensor hidden = trial->hidden;
      for (auto& layer : layers) {
        hidden = layer->forward(hidden,
                                trial->positions,
                                trial->rope,
                                trial->caches,
                                trial->input,
                                trial->shared);
        trial->outputs.emplace_back(hidden);
      }
      trial->graph->capture_end();
    }
    return trial;
  };
  auto small = make_trial(/*batch=*/2);
  auto large = make_trial(/*batch=*/4);
  ASSERT_NE(small->shared.selected_kv.data_ptr(),
            large->shared.selected_kv.data_ptr());
  // Alternate graph buckets; cross the ring boundary and shrink/grow the live
  // batch. Poison unused scratch so stale lengths or data cannot pass silently.
  for (int32_t step = 0; step < 4; ++step) {
    for (AttentionReplayTrial* trial : {small.get(), large.get()}) {
      const int64_t batch = trial->hidden.size(0);
      const int32_t active = step % 2 == 0 ? batch : batch - 1;
      trial->input.meta.actual_num_sequences = active;
      trial->input.attention.host.q_seq_lens.clear();
      trial->input.attention.host.kv_seq_lens.clear();
      trial->input.attention.host.q_seq_lens.emplace_back(0);
      trial->input.attention.host.kv_seq_lens.emplace_back(0);
      const int32_t position = 126 + step;
      for (int32_t seq = 0; seq < batch; ++seq) {
        trial->input.attention.host.q_seq_lens.emplace_back(seq + 1);
        trial->input.attention.host.kv_seq_lens.emplace_back((seq + 1) *
                                                             (position + 1));
        auto selected =
            torch::arange(1 + step + seq, options.dtype(torch::kInt64))
                .view({1, -1});
        trial->shared.topk[seq] = selected;
        for (auto& independent : trial->independent) {
          independent.topk[seq] = selected;
        }
      }
      trial->hidden.copy_(torch::randn_like(trial->hidden));
      trial->positions.fill_(position);
      trial->shared.selected_kv.fill_(std::numeric_limits<float>::quiet_NaN());
      trial->shared.selected_lengths.fill_(-1);
      ModelInputParams reference_input = trial->input;
      reference_input.graph.eager_context = nullptr;
      std::vector<torch::Tensor> reference;
      reference.reserve(layers.size());
      torch::Tensor hidden = trial->hidden;
      for (size_t id = 0; id < layers.size(); ++id) {
        hidden = layers[id]->forward(hidden,
                                     trial->positions,
                                     trial->rope,
                                     trial->reference_caches,
                                     reference_input,
                                     trial->independent[id]);
        reference.emplace_back(hidden);
      }
      trial->graph->replay();
      torch_mlu::synchronize();
      for (size_t id = 0; id < layers.size(); ++id) {
        SCOPED_TRACE(::testing::Message() << "step=" << step << " batch="
                                          << batch << " layer=" << id);
        auto actual = trial->outputs[id].slice(0, 0, active).cpu();
        auto expected = reference[id].slice(0, 0, active).cpu();
        ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
        EXPECT_TRUE(torch::equal(actual, expected));
        EXPECT_TRUE(torch::equal(trial->caches[id].get_swa_cache(),
                                 trial->reference_caches[id].get_swa_cache()));
      }
    }
  }
}
}  // namespace
}  // namespace xllm::layer
