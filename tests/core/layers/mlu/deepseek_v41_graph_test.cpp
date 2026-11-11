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
#include <utility>
#include <vector>

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

TEST(DeepseekV41GraphTest, IndexerConsumerNeedsOnlyHeadWeights) {
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ModelArgs args;
  args.hidden_size() = 5120;
  args.index_n_heads() = 32;
  args.index_head_dim() = 128;
  args.qk_rope_head_dim() = 64;
  args.compress_ratios() = {1, 1};
  args.kv_source_layer_ids() = {0};
  args.index_source_layer_ids() = {1};
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  DeepseekV41Indexer indexer(ModelContext(parallel, args, QuantArgs(), options),
                             /*layer_id=*/1);
  torch::manual_seed(41);
  const torch::Tensor weight = torch::randn({32, 5120}, options) * 0.02;
  indexer->load_state_dict(StateDict({{"weights_proj.weight", weight}}));
  const torch::Tensor x = torch::randn({17, 5120}, options);
  const torch::Tensor cache = torch::full({2, 128, 1, 128}, -7, options);
  const torch::Tensor actual = indexer->project_indexer(x,
                                                        torch::Tensor(),
                                                        torch::Tensor(),
                                                        torch::Tensor(),
                                                        torch::Tensor(),
                                                        cache);
  const torch::Tensor expected =
      torch::matmul(x, weight.t()).to(torch::kFloat32) / 64;
  EXPECT_TRUE(torch::allclose(actual.cpu(), expected.cpu(), 0.02, 0.001));
  EXPECT_TRUE(cache.cpu().eq(-7).all().item<bool>());
}

TEST(DeepseekV41GraphTest, QueryProjectionKeepsFloat32Contract) {
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ModelArgs args;
  args.hidden_size() = 5120;
  args.q_lora_rank() = 1280;
  args.index_n_heads() = 32;
  args.index_head_dim() = 128;
  args.qk_rope_head_dim() = 64;
  args.compress_ratios() = {1};
  args.index_source_layer_ids() = {0};
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  DeepseekV41Indexer indexer(ModelContext(parallel, args, QuantArgs(), options),
                             0);
  const torch::Tensor weight = torch::ones({4096, 1280}, options) / 1280;
  indexer->load_state_dict(StateDict({{"wq_b.weight", weight}}));
  auto qr = torch::ones({6, 1280}, options);
  auto positions = torch::arange(6, options.dtype(torch::kInt64));
  const torch::Tensor angles =
      torch::arange(6, options.dtype(torch::kFloat32)).unsqueeze(1) * 0.1;
  const torch::Tensor rope = torch::cat(
      {angles.cos().expand({6, 64}), angles.sin().expand({6, 64})}, 1);
  const torch::Tensor output = indexer->project_query(qr, positions, rope);
  EXPECT_EQ(output.scalar_type(), torch::kFloat32);
  EXPECT_EQ(output.sizes(), (torch::IntArrayRef{6, 32, 128}));
  EXPECT_TRUE(torch::equal(
      output, indexer->project_query(qr, positions.to(torch::kInt32), rope)));
  torch_mlu::synchronize();
  torch_mlu::mlu::MLUStreamGuard stream(
      torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
  for (int32_t i = 0; i < 10; ++i) {
    indexer->project_query(qr, positions, rope);
  }
  torch_mlu::MLUGraph graph;
  graph.capture_begin();
  const torch::Tensor captured = indexer->project_query(qr, positions, rope);
  graph.capture_end();
  qr.mul_(0.5);
  positions.fill_(0);
  graph.replay();
  EXPECT_TRUE(
      torch::equal(captured, indexer->project_query(qr, positions, rope)));
  EXPECT_FALSE(torch::equal(captured, output));
}

TEST(DeepseekV41GraphTest, FusedIndexerMatchesIndependentRequests) {
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  const torch::DeviceGuard guard(device);
  const torch::NoGradGuard no_grad;
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  for (const bool decode : {false, true}) {
    SCOPED_TRACE(decode);
    for (const int32_t ratio : {1, 2}) {
      torch::manual_seed(2026);
      ModelArgs args;
      args.hidden_size() = 5120;
      args.n_heads() = 1;
      args.head_dim() = 512;
      args.q_lora_rank() = 1280;
      args.qk_rope_head_dim() = 64;
      args.o_groups() = 1;
      args.o_lora_rank() = 8;
      args.window_size() = 16;
      args.rms_norm_eps() = 1e-20;
      args.compress_ratios() = {ratio};
      args.kv_source_layer_ids() = {0};
      args.index_source_layer_ids() = {0};
      args.index_n_heads() = 32;
      args.index_head_dim() = 128;
      args.index_topk() = 4;
      args.num_speculative_tokens() = 5;
      test::MockProcessGroup group(device);
      ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
      parallel.tp_group_ = &group;
      DeepseekV41Attention attention(
          ModelContext(parallel, args, QuantArgs(), options), 0);
      auto random = [&](std::vector<int64_t> shape) {
        return torch::randn(shape, options) * 0.02;
      };
      StateDict state({
          {"wq_a.weight", random({1280, 5120})},
          {"wq_b.weight", random({512, 1280})},
          {"wkv.weight", random({512, 5120})},
          {"q_norm.weight", torch::ones({1280}, options)},
          {"kv_norm.weight", torch::ones({512}, options)},
          {"wo_a.weight", random({1, 8, 512})},
          {"wo_b.weight", random({5120, 8})},
          {"attn_sink", torch::zeros({1}, options)},
          {"compressor.wkv.weight", random({512, 5120})},
          {"compressor.wgate.weight", random({512, 5120})},
          {"compressor.norm.weight", torch::ones({512}, options)},
          {"indexer.wq_b.weight", random({4096, 1280})},
          {"indexer.weights_proj.weight", random({32, 5120})},
          {"indexer.wk.weight", random({128, 512})},
          {"indexer.k_norm.weight", torch::ones({128}, options)},
      });
      // Single dyadic products keep KV projections exact across GEMM row
      // counts, so batched and independent cache placement can be compared
      // bitwise without depending on the selected GEMM accumulation order.
      for (const std::string name :
           {"wkv.weight", "compressor.wkv.weight", "compressor.wgate.weight"}) {
        torch::Tensor weight = state.get_tensor(name);
        weight.zero_();
        weight.select(/*dim=*/1, /*index=*/0).fill_(/*value=*/0.125);
      }
      attention->load_state_dict(state);
      attention->verify_loaded_weights();
      auto make_cache = [&]() {
        DeepSeekV4KVCacheTensors tensors;
        tensors.swa_cache = torch::zeros({3, 128, 1, 512}, options);
        tensors.key_cache = torch::zeros({3, 128, 1, 512}, options);
        tensors.index_cache = torch::full({3, 128, 1, 128}, -7, options);
        tensors.compress_kv_state =
            torch::zeros({3, 32, 512}, options.dtype(torch::kFloat32));
        tensors.compress_score_state =
            torch::zeros_like(tensors.compress_kv_state);
        std::vector<KVCache> result;
        result.emplace_back(tensors);
        return result;
      };
      auto batched = make_cache();
      auto independent = make_cache();
      const torch::Tensor table =
          torch::tensor({2, 0, 1}, options.dtype(torch::kInt32)).view({3, 1});
      const torch::Tensor angles =
          torch::arange(32, options.dtype(torch::kFloat32))
              .unsqueeze(1)
              .expand({32, 64}) *
          0.017;
      const torch::Tensor rope =
          torch::cat({angles.cos(), angles.sin()}, /*dim=*/1);
      const torch::Tensor hidden = random({12, 5120});
      for (int32_t step = 0; step < 3; ++step) {
        const bool initial = step == 0;
        std::vector<int32_t> counts = initial ? std::vector<int32_t>{1, 0, 3}
                                              : std::vector<int32_t>{3, 0, 5};
        std::vector<int32_t> begins = initial ? std::vector<int32_t>{0, 0, 0}
                                              : std::vector<int32_t>{1, 0, 3};
        if (step == 2) {
          counts = {1, 0, 1};
          begins = {4, 0, 8};
        }
        ModelInputParams input;
        input.meta.batch_forward_type = decode
                                            ? BatchForwardType::DECODE
                                            : BatchForwardType::CHUNKED_PREFILL;
        input.meta.num_sequences = 3;
        input.attention.host.q_seq_lens = {
            0, counts[0], counts[0], counts[0] + counts[2]};
        input.multi_block_tables = {table, table, table};
        Csa2Context shared;
        shared.topk.resize(3);
        shared.candidates.resize(3);
        const torch::Tensor positions =
            torch::cat({torch::arange(begins[0],
                                      begins[0] + counts[0],
                                      options.dtype(torch::kInt64)),
                        torch::arange(begins[2],
                                      begins[2] + counts[2],
                                      options.dtype(torch::kInt64))});
        const torch::Tensor batch_x =
            step == 2 ? hidden.slice(0, 0, 2)
                      : hidden.slice(0, initial ? 0 : 4, initial ? 4 : 12);
        deepseek_v41::prepare_metadata(
            shared, input, positions, input.meta.num_sequences);
        deepseek_v41::prepare_workspace(
            shared, input, args, batch_x.size(0), options);
        const torch::Tensor actual = attention->forward(
            batch_x, positions, rope, batched, input, shared);
        int64_t offset = 0;
        for (const int32_t seq : {0, 2}) {
          ModelInputParams single;
          single.meta.num_sequences = 1;
          single.meta.batch_forward_type = input.meta.batch_forward_type;
          single.attention.host.q_seq_lens = {0, counts[seq]};
          single.multi_block_tables = {table.slice(0, seq, seq + 1),
                                       table.slice(0, seq, seq + 1),
                                       table.slice(0, seq, seq + 1)};
          Csa2Context reference_shared;
          reference_shared.topk.resize(1);
          reference_shared.candidates.resize(1);
          deepseek_v41::prepare_metadata(
              reference_shared,
              single,
              positions.slice(0, offset, offset + counts[seq]),
              single.meta.num_sequences);
          deepseek_v41::prepare_workspace(
              reference_shared, single, args, counts[seq], options);
          const torch::Tensor expected = attention->forward(
              batch_x.slice(0, offset, offset + counts[seq]),
              positions.slice(0, offset, offset + counts[seq]),
              rope,
              independent,
              single,
              reference_shared);
          EXPECT_TRUE(
              torch::allclose(actual.slice(0, offset, offset + counts[seq]),
                              expected,
                              0.02,
                              0.02));
          EXPECT_TRUE(torch::equal(shared.topk[seq].cpu(),
                                   reference_shared.topk[0].cpu()));
          offset += counts[seq];
        }
        EXPECT_TRUE(torch::equal(batched[0].get_swa_cache().cpu(),
                                 independent[0].get_swa_cache().cpu()));
        EXPECT_TRUE(torch::equal(batched[0].get_k_cache().cpu(),
                                 independent[0].get_k_cache().cpu()));
        EXPECT_TRUE(torch::allclose(batched[0].get_index_cache().cpu(),
                                    independent[0].get_index_cache().cpu(),
                                    0.02,
                                    0.02));
      }
    }
  }
}

TEST(DeepseekV41GraphTest, VerifyRejectionRetainsInteriorCompressorRows) {
  torch::manual_seed(91);
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ModelArgs args;
  args.hidden_size() = 32;
  args.n_heads() = 8;
  args.head_dim() = 512;
  args.q_lora_rank() = 16;
  args.qk_rope_head_dim() = 64;
  args.o_groups() = 1;
  args.o_lora_rank() = 8;
  args.window_size() = 16;
  args.rms_norm_eps() = 1e-20;
  args.compress_ratios() = {2};
  args.kv_source_layer_ids() = {0};
  args.index_head_dim() = 128;
  args.num_speculative_tokens() = 5;
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  DeepseekV41Attention attention(
      ModelContext(parallel, args, QuantArgs(), options), 0);
  auto random = [&](std::vector<int64_t> shape) {
    return torch::randn(shape, options) * 0.1;
  };
  attention->load_state_dict(StateDict({
      {"wq_a.weight", random({16, 32})},
      {"wkv.weight", random({512, 32})},
      {"q_norm.weight", torch::ones({16}, options)},
      {"kv_norm.weight", torch::ones({512}, options)},
      {"wq_b.weight", random({4096, 16})},
      {"wo_a.weight", random({1, 8, 4096})},
      {"wo_b.weight", random({32, 8})},
      {"attn_sink", torch::zeros({8}, options)},
      {"compressor.wkv.weight", random({512, 32})},
      {"compressor.wgate.weight", random({512, 32})},
      {"compressor.norm.weight", torch::ones({512}, options)},
      {"indexer.wk.weight", random({128, 512})},
      {"indexer.k_norm.weight", torch::ones({128}, options)},
  }));
  auto make_cache = [&](bool packed) {
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache = torch::zeros({2, 128, 1, 512}, options);
    tensors.key_cache = torch::zeros({2, 128, 1, 512}, options);
    tensors.index_cache = torch::zeros({2, 128, 1, 128}, options);
    if (packed) {
      tensors.compress_state =
          torch::zeros({2, 32, 1024}, options.dtype(torch::kFloat32));
    } else {
      tensors.compress_kv_state =
          torch::zeros({2, 32, 512}, options.dtype(torch::kFloat32));
      tensors.compress_score_state =
          torch::zeros_like(tensors.compress_kv_state);
    }
    std::vector<KVCache> caches;
    caches.emplace_back(tensors);
    return caches;
  };
  // PD exports a packed owner; attention writes through its strided views.
  // Compare against independent contiguous planes through verify/rejection.
  auto actual = make_cache(/*packed=*/true);
  auto reference = make_cache(/*packed=*/false);
  auto hidden = random({20, 32});
  auto rope =
      torch::cat({torch::ones({32, 64}, options.dtype(torch::kFloat32)),
                  torch::zeros({32, 64}, options.dtype(torch::kFloat32))},
                 -1);
  auto run = [&](std::vector<KVCache>& caches, int64_t begin, int64_t count) {
    ModelInputParams input;
    input.meta.num_sequences = 1;
    input.meta.batch_forward_type = BatchForwardType::DECODE;
    input.is_spec_verify = count > 1;
    input.attention.host.q_seq_lens = {0, static_cast<int32_t>(count)};
    auto table =
        torch::tensor({0, 1}, options.dtype(torch::kInt32)).view({1, 2});
    input.multi_block_tables = {table, table};
    Csa2Context shared;
    shared.topk.emplace_back(
        torch::empty({count, 0}, options.dtype(torch::kInt64)));
    shared.candidates.resize(1);
    deepseek_v41::prepare_metadata(
        shared,
        input,
        torch::arange(begin, begin + count, options.dtype(torch::kInt64)),
        input.meta.num_sequences);
    deepseek_v41::prepare_workspace(shared, input, args, count, options);
    return attention->forward(
        hidden.slice(0, begin, begin + count),
        torch::arange(begin, begin + count, options.dtype(torch::kInt64)),
        rope,
        caches,
        input,
        shared);
  };
  run(actual, 0, 1);
  run(actual, 1, 6);
  // Reject the proposals after position two, then cross the odd C2 boundary.
  run(reference, 0, 3);
  auto output = run(actual, 3, 1);
  auto expected = run(reference, 3, 1);
  torch_mlu::synchronize();
  EXPECT_TRUE(
      torch::equal(actual[0].get_k_cache().flatten(0, 2).slice(0, 0, 2),
                   reference[0].get_k_cache().flatten(0, 2).slice(0, 0, 2)));
  EXPECT_TRUE(torch::allclose(output, expected, 0.02, 0.02));

  // Flat DSpark verify is regrouped into contiguous request chunks before
  // attention. Replay must refresh the chunk's positions and host KV lengths,
  // including a rejection that resumes inside the previous chunk.
  auto graph_caches = make_cache(/*packed=*/true);
  auto eager_caches = make_cache(/*packed=*/false);
  run(graph_caches, 0, 1);
  run(eager_caches, 0, 1);
  ModelInputParams input;
  input.enable_graph = true;
  input.meta.batch_forward_type = BatchForwardType::DECODE;
  input.meta.actual_num_sequences = 1;
  input.meta.num_sequences = 1;
  input.meta.kv_max_seq_len = 7;
  input.attention.host.q_seq_lens = {0, 6};
  input.attention.host.kv_seq_lens = {0, 7};
  Csa2Context shared;
  shared.selected_kv = torch::empty(
      {6 * (args.window_size() + args.index_topk()), 1, 1, args.head_dim()},
      options);
  shared.selected_lengths = torch::empty({6}, options.dtype(torch::kInt32));
  shared.topk.emplace_back(torch::empty({6, 0}, options.dtype(torch::kInt64)));
  shared.candidates.resize(1);
  auto table = torch::tensor({0, 1}, options.dtype(torch::kInt32)).view({1, 2});
  input.multi_block_tables = {table, table};
  auto graph_hidden = hidden.slice(0, 1, 7).clone();
  auto positions = torch::arange(1, 7, options.dtype(torch::kInt64));
  deepseek_v41::prepare_metadata(
      shared, input, positions, input.meta.actual_num_sequences);
  auto graph_forward = [&]() {
    return attention->forward(
        graph_hidden, positions, rope, graph_caches, input, shared);
  };
  // Compile lazy attention kernels before capture; cache writes are confined
  // to the callback, which PiecewiseGraph records without executing.
  graph_forward();
  kernel::mlu::PiecewiseGraph graph(input.meta.kv_max_seq_len);
  input.graph.eager_context = &graph;
  torch::Tensor graph_output;
  torch_mlu::synchronize();
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin(torch_mlu::graph_pool_handle());
    graph_output = graph_forward();
    graph.capture_end();
  }
  torch_mlu::synchronize();
  ASSERT_GT(graph.num_graphs(), 0);
  ASSERT_EQ(graph.num_eager_segments(), 1);
  for (int32_t begin : {1, 3, 7}) {
    graph_hidden.copy_(hidden.slice(0, begin, begin + 6));
    positions.copy_(torch::arange(begin, begin + 6, positions.options()));
    input.meta.kv_max_seq_len = begin + 6;
    input.attention.host.kv_seq_lens = {0, begin + 6};
    auto eager_output = run(eager_caches, begin, 6);
    deepseek_v41::prepare_metadata(
        shared, input, positions, input.meta.actual_num_sequences);
    graph.replay();
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(graph_output, eager_output, 0.02, 0.02));
    EXPECT_TRUE(torch::equal(graph_caches[0].get_swa_cache(),
                             eager_caches[0].get_swa_cache()));
    EXPECT_TRUE(torch::equal(graph_caches[0].get_k_cache(),
                             eager_caches[0].get_k_cache()));
    EXPECT_TRUE(torch::equal(graph_caches[0].get_index_cache(),
                             eager_caches[0].get_index_cache()));
  }
  std::vector<torch::Tensor> snapshots;
  for (const auto& tensor : {graph_caches[0].get_k_cache(),
                             graph_caches[0].get_index_cache(),
                             graph_caches[0].get_swa_cache(),
                             graph_caches[0].get_compress_kv_state(),
                             graph_caches[0].get_compress_score_state()}) {
    snapshots.emplace_back(tensor.clone());
  }
  input.meta.actual_num_sequences = 0;
  ModelInputParams empty_input = input;
  empty_input.enable_graph = false;
  empty_input.graph.eager_context = nullptr;
  Csa2Context empty_shared;
  deepseek_v41::prepare_metadata(empty_shared,
                                 empty_input,
                                 positions,
                                 /*num_sequences=*/0);
  deepseek_v41::prepare_workspace(
      empty_shared, empty_input, args, positions.numel(), options);
  const auto empty_output = attention->forward(
      graph_hidden, positions, rope, graph_caches, empty_input, empty_shared);
  EXPECT_TRUE(torch::isfinite(empty_output).all().item<bool>());
  EXPECT_TRUE(empty_shared.selected_lengths.eq(0).all().item<bool>());
  EXPECT_EQ(empty_shared.compressed[1].outputs.num_valid_rows.item<int32_t>(),
            0);
  deepseek_v41::prepare_metadata(
      shared, input, positions, input.meta.actual_num_sequences);
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_EQ(shared.compressed[1].outputs.num_valid_rows.item<int32_t>(), 0);
  EXPECT_TRUE(shared.compressed[1].outputs.slots.eq(-1).all().item<bool>());
  EXPECT_TRUE(shared.compressed[1].outputs.latent.eq(0).all().item<bool>());
  int32_t index = 0;
  for (const auto& tensor : {graph_caches[0].get_k_cache(),
                             graph_caches[0].get_index_cache(),
                             graph_caches[0].get_swa_cache(),
                             graph_caches[0].get_compress_kv_state(),
                             graph_caches[0].get_compress_score_state()}) {
    EXPECT_TRUE(torch::equal(tensor, snapshots[index++]));
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

  // Verify blocks are grouped by request: later rows read earlier tokens in
  // the same block, and only the first rows use the request's lookback.
  auto grouped = engram->create_hash_state(3, /*query_width=*/3);
  auto reference = engram->create_hash_state(3);
  auto block = torch::tensor({2, 3, 1}, tokens.options());
  input.meta.actual_num_sequences = 1;
  input.embedding.engram_lookback_token_ids = {4, 5};
  grouped->prepare(input);
  input.meta.actual_num_sequences = 3;
  input.embedding.engram_lookback_token_ids = {4, 5, 2, 4, 3, 2};
  reference->prepare(input);
  auto grouped_output = grouped->forward(block);
  auto reference_output = reference->forward(block);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(grouped_output, reference_output));
  torch_mlu::MLUGraph verify_graph;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    verify_graph.capture_begin();
    grouped_output = grouped->forward(block);
    verify_graph.capture_end();
  }
  block.copy_(torch::tensor({3, 2, 1}, block.options()));
  input.meta.actual_num_sequences = 1;
  input.embedding.engram_lookback_token_ids = {6, 7};
  grouped->prepare(input);
  input.meta.actual_num_sequences = 3;
  input.embedding.engram_lookback_token_ids = {6, 7, 3, 6, 2, 3};
  reference->prepare(input);
  reference_output = reference->forward(block);
  verify_graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(grouped_output, reference_output));
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

TEST(DeepseekV41GatherDeathTest, RejectsCpuBuffersBeforeKernelLaunch) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const int64_t index : {1, 2, 6}) {
    SCOPED_TRACE(index);
    EXPECT_DEATH(
        {
          const torch::Device device(torch::kPrivateUse1, 0);
          const auto options =
              torch::TensorOptions().device(device).dtype(torch::kBFloat16);
          std::vector<torch::Tensor> arguments(
              {torch::zeros({128, 512}, options),
               torch::zeros({128, 512}, options),
               torch::ones({512}, options),
               torch::zeros({1}, options.dtype(torch::kInt32)),
               torch::zeros({1}, options.dtype(torch::kInt32)),
               torch::zeros({1}, options.dtype(torch::kInt64)),
               torch::empty({2, 512}, options),
               torch::zeros({1}, options.dtype(torch::kInt32))});
          arguments[index] = arguments[index].cpu();
          kernel::mlu::bangc::deepseek_v41_decode_gather(arguments[0],
                                                         arguments[1],
                                                         arguments[2],
                                                         arguments[3],
                                                         arguments[4],
                                                         arguments[5],
                                                         /*position=*/0,
                                                         /*window=*/1,
                                                         arguments[6],
                                                         arguments[7]);
        },
        "same device as swa");
  }
}

TEST(DeepseekV41GraphTest, SparseGatherCompactsMaskedCompressedIndices) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto integers = options.dtype(torch::kInt64);
  auto swa = torch::arange(384 * 512, options.dtype(torch::kFloat32))
                 .view({384, 512})
                 .to(torch::kBFloat16);
  auto compressed =
      torch::full({384, 512}, std::numeric_limits<float>::quiet_NaN(), options);
  compressed[258].fill_(2);
  compressed[129].fill_(3);
  auto current = torch::ones({512}, options);
  auto page = torch::tensor({1, 0}, options.dtype(torch::kInt32));
  auto compressed_page = torch::tensor({2, 1}, page.options());
  struct Selection {
    std::vector<int64_t> indices;
    std::vector<int64_t> physical_rows;
  };
  std::vector<int64_t> tiled(259, -1);
  tiled[0] = 2;
  tiled[128] = 129;
  tiled[258] = 2;
  const std::vector<Selection> selections = {
      {{-1, 2, -1, 129, -1}, {258, 129}},
      {{-1, -1, -1, -1, -1}, {}},
      {std::move(tiled), {258, 129, 258}}};
  for (const auto& selection : selections) {
    SCOPED_TRACE(selection.indices.size());
    auto topk = torch::tensor(selection.indices, integers);
    auto output = torch::full({4 + topk.numel(), 1, 1, 512},
                              std::numeric_limits<float>::quiet_NaN(),
                              options);
    auto length = torch::zeros({1}, page.options());
    kernel::mlu::bangc::deepseek_v41_decode_gather(swa,
                                                   compressed,
                                                   current,
                                                   page,
                                                   compressed_page,
                                                   topk,
                                                   /*position=*/129,
                                                   /*window=*/4,
                                                   output,
                                                   length);
    auto expected =
        torch::cat({swa.index_select(0, torch::tensor({254, 255, 0}, integers)),
                    current.unsqueeze(0),
                    compressed.index_select(
                        0, torch::tensor(selection.physical_rows, integers))},
                   0);
    torch_mlu::synchronize();
    ASSERT_EQ(length.cpu().item<int32_t>(), expected.size(0));
    EXPECT_TRUE(torch::equal(
        output.view({-1, 512}).slice(0, 0, expected.size(0)), expected));
  }
}

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
  input.meta.batch_forward_type = BatchForwardType::DECODE;
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
    trial->input.meta.batch_forward_type = BatchForwardType::DECODE;
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
      reference_input.enable_graph = false;
      trial->input.multi_block_tables = trial->shared.block_tables;
      reference_input.multi_block_tables = trial->shared.block_tables;
      deepseek_v41::prepare_metadata(
          trial->shared, trial->input, trial->positions, active);
      std::vector<torch::Tensor> reference;
      reference.reserve(layers.size());
      torch::Tensor hidden = trial->hidden;
      for (size_t id = 0; id < layers.size(); ++id) {
        deepseek_v41::prepare_metadata(
            trial->independent[id], reference_input, trial->positions, active);
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
