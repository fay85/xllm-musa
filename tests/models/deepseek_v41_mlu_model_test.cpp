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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/attention_metadata.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "core/runtime/options.h"
#include "models/llm/mlu/deepseek_v41.h"
#include "tests/core/layers/mlu/tests_utils.h"

namespace xllm {
namespace {

using Weights = std::unordered_map<std::string, torch::Tensor>;

// Replace checkpoint I/O while exercising the real model and weight loaders.
class InMemoryModelLoader final : public ModelLoader {
 public:
  InMemoryModelLoader(ModelArgs args,
                      std::string path,
                      std::vector<Weights> shards)
      : path_(std::move(path)) {
    args_ = std::move(args);
    states_.reserve(shards.size());
    for (auto& shard : shards) {
      states_.emplace_back(std::make_unique<StateDict>(std::move(shard)));
    }
  }

  std::unique_ptr<Tokenizer> tokenizer() const override {
    LOG(FATAL) << "Tokenizer construction is not used by model loading";
    return nullptr;
  }

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return states_;
  }

  std::string model_weights_path() const override { return path_; }

 private:
  std::string path_;
  std::vector<std::unique_ptr<StateDict>> states_;
};

class DeepseekV41MluModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "xllm_v41_loading_XXXXXX")
            .string();
    const char* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    path_ = directory;
    std::ofstream tokenizer(path_ / "tokenizer.json");
    tokenizer << R"json({"version":"1.0","truncation":null,"padding":null,
      "added_tokens":[],"normalizer":null,"pre_tokenizer":null,
      "post_processor":null,"decoder":null,"model":{"type":"WordLevel",
      "vocab":{"a":0,"b":1,"c":2,"unk":3},"unk_token":"unk"}})json";
    ASSERT_TRUE(tokenizer.good());
    args_.model_type("deepseek_v41")
        .n_layers(1)
        .hidden_size(5120)
        .vocab_size(4)
        .n_heads(1)
        .head_dim(512)
        .q_lora_rank(128)
        .qk_rope_head_dim(64)
        .index_n_heads(1)
        .index_head_dim(128)
        .o_groups(1)
        .o_lora_rank(128)
        .hc_mult(4)
        .hc_sinkhorn_iters(4)
        .hc_eps(1e-6)
        .rms_norm_eps(1e-5)
        .rope_theta(10000)
        .compress_rope_theta(160000)
        .max_position_embeddings(32)
        .factor(1)
        .compress_ratios({0})
        .n_routed_experts(2)
        .num_experts_per_tok(1)
        .n_shared_experts(0)
        .moe_intermediate_size(128)
        .hidden_act("silu")
        .scoring_func("softmax")
        .n_group(1)
        .topk_group(1)
        .engram_max_ngram_size(2)
        .engram_n_heads(1)
        .engram_compressed_vocab_size(4);
  }

  void TearDown() override {
    if (!path_.empty()) {
      std::filesystem::remove_all(path_);
    }
  }

  Weights make_weights() const {
    const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
    const int64_t hidden = args_.hidden_size();
    const int64_t hc = args_.hc_mult();
    const int64_t qr = args_.q_lora_rank();
    const int64_t heads = args_.n_heads();
    const int64_t index_heads = args_.index_n_heads();
    const int64_t hc_mixes = hc * hc + 2 * hc;
    Weights weights{
        {"embed.weight", torch::full({4, hidden}, 0.25, options)},
        {"norm.weight", torch::ones({hidden}, options)},
        {"layers.0.attn.wq_a.weight", torch::zeros({qr, hidden}, options)},
        {"layers.0.attn.wq_b.weight", torch::zeros({heads * 512, qr}, options)},
        {"layers.0.attn.wkv.weight", torch::zeros({512, hidden}, options)},
        {"layers.0.attn.q_norm.weight", torch::ones({qr}, options)},
        {"layers.0.attn.kv_norm.weight", torch::ones({512}, options)},
        {"layers.0.attn.wo_a.weight",
         torch::zeros({1, 128, heads * 512}, options)},
        {"layers.0.attn.wo_b.weight", torch::zeros({hidden, 128}, options)},
        {"layers.0.attn.attn_sink", torch::zeros({heads}, torch::kFloat32)},
        {"layers.0.attn_norm.weight", torch::ones({hidden}, options)},
        {"layers.0.ffn_norm.weight", torch::ones({hidden}, options)},
        {"layers.0.ffn.gate.weight", torch::zeros({2, hidden}, options)},
    };
    for (const std::string prefix :
         {"layers.0.hc_attn_pre.", "layers.0.hc_ffn_pre."}) {
      weights.emplace(prefix + "hc_fn",
                      torch::zeros({hc_mixes, hc * hidden}, options));
      weights.emplace(prefix + "hc_base",
                      torch::zeros({hc_mixes}, torch::kFloat32));
      weights.emplace(prefix + "hc_scale", torch::ones({3}, torch::kFloat32));
    }
    for (int32_t expert = 0; expert < 2; ++expert) {
      const std::string prefix =
          "layers.0.ffn.experts." + std::to_string(expert) + ".";
      for (const std::string projection : {"w1", "w2", "w3"}) {
        weights.emplace(
            prefix + projection + ".weight",
            torch::zeros(projection == "w2" ? std::vector<int64_t>{hidden, 128}
                                            : std::vector<int64_t>{128, hidden},
                         options));
      }
    }
    const int32_t ratio = args_.compress_ratios().front();
    if (ratio > 0) {
      weights.emplace("layers.0.attn.compressor.wkv.weight",
                      torch::zeros({512, hidden}, options));
      weights.emplace("layers.0.attn.compressor.norm.weight",
                      torch::ones({512}, options));
      weights.emplace("layers.0.attn.indexer.wq_b.weight",
                      torch::zeros({index_heads * 128, qr}, options));
      weights.emplace("layers.0.attn.indexer.weights_proj.weight",
                      torch::zeros({index_heads, hidden}, options));
      weights.emplace("layers.0.attn.indexer.wk.weight",
                      torch::zeros({128, 512}, options));
      weights.emplace("layers.0.attn.indexer.k_norm.weight",
                      torch::ones({128}, options));
    }
    if (ratio == 2) {
      weights.emplace("layers.0.attn.compressor.wgate.weight",
                      torch::zeros({512, hidden}, options));
    }
    return weights;
  }

  std::unique_ptr<ModelLoader> make_loader(Weights weights) const {
    std::vector<Weights> shards;
    shards.reserve(1);
    shards.emplace_back(std::move(weights));
    return make_loader(std::move(shards));
  }

  std::unique_ptr<ModelLoader> make_loader(std::vector<Weights> shards) const {
    return std::make_unique<InMemoryModelLoader>(
        args_, path_.string(), std::move(shards));
  }

  ModelContext make_context() {
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group_);
    parallel.tp_group_ = &group_;
    parallel.single_rank_group_ = &group_;
    return ModelContext(parallel, args_, QuantArgs(), options_);
  }

  void set_cache_role(int32_t ratio) {
    if (ratio > 0) {
      args_.q_lora_rank(1280).index_n_heads(32).rms_norm_eps(1e-20);
    }
    args_.compress_ratios({ratio});
    args_.kv_source_layer_ids(ratio == 0 ? std::vector<int32_t>{}
                                         : std::vector<int32_t>{0});
    args_.index_source_layer_ids(args_.kv_source_layer_ids());
  }

  void expect_missing_weight(const std::string& name) {
    SCOPED_TRACE(name);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto weights = make_weights();
    weights.emplace(
        "head.weight",
        torch::full({4, args_.hidden_size()}, 0.5, torch::kBFloat16));
    ASSERT_EQ(weights.erase(name), 1);
    EXPECT_DEATH(
        {
          mlu::model::DeepseekV41ForCausalLM model(make_context());
          model->load_model(make_loader(std::move(weights)));
        },
        name);
  }

  torch::TensorOptions options_ = torch::TensorOptions()
                                      .device(torch::kPrivateUse1, 0)
                                      .dtype(torch::kBFloat16);
  layer::test::MockProcessGroup group_{options_.device()};
  ModelArgs args_;
  std::filesystem::path path_;
};

// Count host model entries without replacing any V4.1 forward/metadata logic.
class ExecutorV41Model final : public CausalLM {
 public:
  ExecutorV41Model(const ModelContext& context,
                   const StateDict& weights,
                   const StateDict& hashes)
      : model_(context), options_(context.get_tensor_options()) {
    model_->load_state_dict(weights);
    model_->load_hash_state(hashes);
    model_->verify_loaded_weights();
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& input) override {
    ++forward_count_;
    return model_->forward(tokens, positions, caches, input);
  }
  bool requires_graph_forward_metadata() override { return true; }
  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      override {
    return std::make_unique<mlu::model::DeepseekV41GraphState>();
  }
  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input) override {
    model_->prepare_graph_forward_metadata(state, positions, input);
    graph_state_ = dynamic_cast<mlu::model::DeepseekV41GraphState*>(state);
  }
  layer::WordEmbedding get_word_embedding() override {
    return model_->get_word_embedding();
  }
  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& /*selected*/) override {
    return hidden;
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  int32_t forward_count() const { return forward_count_; }
  const mlu::model::DeepseekV41GraphState* graph_state() const {
    return graph_state_;
  }

 private:
  mlu::model::DeepseekV41Model model_;
  torch::TensorOptions options_;
  int32_t forward_count_ = 0;
  const mlu::model::DeepseekV41GraphState* graph_state_ = nullptr;
};

// Observe the same public aux_hidden_states tensor that the draft consumes.
// Zero attention/FFN weights preserve the BF16 embedding across layers, so
// entering-residual capture has an independent oracle at materialized
// boundaries.
class DeepseekV41CaptureTest : public DeepseekV41MluModelTest {
 protected:
  void SetUp() override {
    DeepseekV41MluModelTest::SetUp();
    saved_engram_config_ = ModelConfig::get_instance().engram_config();
    ModelConfig::get_instance().engram_config() = R"({"placement":"gpu"})";
  }

  void TearDown() override {
    ModelConfig::get_instance().engram_config() = saved_engram_config_;
    DeepseekV41MluModelTest::TearDown();
  }

  void expect_entering_residual(bool with_engram,
                                double value_scale,
                                int32_t capture_layer = 0,
                                int64_t model_hidden_size = 5120) {
    torch::InferenceMode inference;
    torch::DeviceGuard guard(options_.device());
    // Fused MHC runs for hc_mult=4 and hidden_size=5120; other hidden sizes
    // take the unfused boundaries.
    args_.layers_to_capture({capture_layer})
        .n_layers(capture_layer + 1)
        .compress_ratios(std::vector<int32_t>(capture_layer + 1, 0))
        .window_size(16)
        .hc_mult(4)
        .hidden_size(model_hidden_size);
    const int64_t hidden_size = args_.hidden_size();
    auto weights = make_weights();
    // Zero attention/FFN branches and uniform Sinkhorn preserve the BF16
    // embedding across layer 0, giving an independent upper-layer oracle.
    const auto first_layer_weights = weights;
    for (int32_t layer_id = 1; layer_id <= capture_layer; ++layer_id) {
      for (const auto& [name, tensor] : first_layer_weights) {
        if (name.rfind("layers.0.", 0) == 0) {
          weights.emplace("layers." + std::to_string(layer_id) + "." +
                              name.substr(std::string("layers.0.").size()),
                          tensor);
        }
      }
    }
    const std::string engram_prefix =
        "layers." + std::to_string(capture_layer) + ".engram.";
    weights["embed.weight"] = torch::arange(4 * hidden_size, torch::kFloat32)
                                  .remainder(31)
                                  .sub(15)
                                  .div(16)
                                  .reshape({4, hidden_size})
                                  .to(torch::kBFloat16);
    const StateDict hashes({
        {"token_map", torch::arange(4, torch::kInt64)},
        {"multipliers", torch::tensor({{1, 3}}, torch::kInt64)},
        {"primes", torch::tensor({{7}}, torch::kInt64)},
        {"offsets", torch::tensor({{0}}, torch::kInt64)},
    });
    if (with_engram) {
      args_.engram_layer_ids({capture_layer})
          .engram_num_embeddings({7})
          .engram_head_dim(32)
          .engram_pad_token_id(0);
      // E4M3 byte 0x38 and E8M0 byte 127 decode to 1. Zero keys make the
      // gate ~0.5; a signed value vector produces a non-collinear residual
      // change. All values are deterministic and exactly BF16-representable.
      weights.emplace(engram_prefix + "embed.weight",
                      torch::full({7, 32}, 0x38, torch::kUInt8));
      weights.emplace(engram_prefix + "embed.scale",
                      torch::full({7, 1}, 127, torch::kUInt8));
      auto wkv = torch::zeros({5 * hidden_size, 32}, torch::kBFloat16);
      auto values = torch::arange(hidden_size, torch::kFloat32)
                        .remainder(2)
                        .mul(2)
                        .sub(1)
                        .mul(value_scale / 32);
      wkv.slice(0, 4 * hidden_size)
          .copy_(values.unsqueeze(1).expand({hidden_size, 32}));
      weights.emplace(engram_prefix + "wkv.weight", wkv);
      weights.emplace(engram_prefix + "q_weight",
                      torch::ones({4, hidden_size}, torch::kBFloat16));
      weights.emplace(engram_prefix + "k_weight",
                      torch::ones({4, hidden_size}, torch::kBFloat16));
    }
    const auto context = make_context();
    mlu::model::DeepseekV41Model model(context);
    const StateDict state(weights);
    model->load_state_dict(state);
    model->load_hash_state(hashes);
    model->verify_loaded_weights();

    const auto host_tokens = torch::tensor({0, 2}, torch::kInt64);
    const auto tokens = host_tokens.to(options_.device());
    const auto positions = torch::arange(2, tokens.options());
    const auto expected = weights.at("embed.weight")
                              .index_select(0, host_tokens)
                              .to(torch::kFloat32);
    ModelInputParams input;
    input.meta.batch_forward_type = BatchForwardType::PREFILL;
    input.meta.num_sequences = 1;
    input.meta.actual_num_sequences = 1;
    input.meta.q_max_seq_len = 2;
    input.meta.kv_max_seq_len = 2;
    input.attention.host.q_seq_lens = {0, 2};
    input.attention.host.kv_seq_lens = {0, 2};
    input.embedding.engram_lookback_token_ids = {0};
    input.multi_block_tables = {torch::zeros({1, 1}, torch::kInt32)};
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache = torch::zeros({1, 1, 128, 512}, options_);
    std::vector<KVCache> caches;
    caches.reserve(capture_layer + 1);
    for (int32_t layer_id = 0; layer_id <= capture_layer; ++layer_id) {
      DeepSeekV4KVCacheTensors layer_tensors;
      layer_tensors.swa_cache = torch::zeros_like(tensors.swa_cache);
      caches.emplace_back(layer_tensors);
    }

    // This independent Engram call is only diagnostic. The correctness
    // assertion below always uses the entering embedding, even after a fix.
    torch::Tensor after_engram = expected;
    if (with_engram) {
      layer::DeepseekV41Engram engram(context, /*engram_id=*/0);
      engram->load_state_dict(state.get_dict_with_prefix(engram_prefix));
      engram->load_hash_state(hashes);
      auto hidden = expected.to(options_).unsqueeze(1).repeat({1, 4, 1});
      after_engram = engram->forward(hidden, tokens, input)
                         .to(torch::kFloat32)
                         .mean(1)
                         .cpu();
      if (value_scale != 0) {
        ASSERT_GT((after_engram - expected).abs().max().item<float>(), 0.1f)
            << "Fixture must inject a visible Engram contribution";
      }
    }
    const auto result = model->forward(tokens, positions, caches, input);
    ASSERT_TRUE(result.aux_hidden_states.defined());
    const auto actual = result.aux_hidden_states.cpu().to(torch::kFloat32);
    ASSERT_EQ(actual.sizes(), expected.sizes());
    ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
    const auto diff = actual - expected;
    const double max_abs = diff.abs().max().item<double>();
    const double mean_abs = diff.abs().mean().item<double>();
    const double relative_l2 = (diff.norm() / expected.norm()).item<double>();
    const double cosine =
        ((actual * expected).sum() / (actual.norm() * expected.norm()))
            .item<double>();
    const double post_engram_max_abs =
        (actual - after_engram).abs().max().item<double>();
    RecordProperty("max_abs", std::to_string(max_abs));
    RecordProperty("mean_abs", std::to_string(mean_abs));
    RecordProperty("relative_l2", std::to_string(relative_l2));
    RecordProperty("cosine", std::to_string(cosine));
    RecordProperty("post_engram_max_abs", std::to_string(post_engram_max_abs));
    LOG(INFO) << "Capture difference: with_engram=" << with_engram
              << " value_scale=" << value_scale << " max_abs=" << max_abs
              << " mean_abs=" << mean_abs << " relative_l2=" << relative_l2
              << " cosine=" << cosine
              << " post_engram_max_abs=" << post_engram_max_abs;
    // No tolerance is needed: capture only averages four identical HC lanes.
    EXPECT_TRUE(torch::equal(actual, expected))
        << "DSpark must receive the entering residual before Engram. "
        << "max_abs=" << max_abs << ", relative_l2=" << relative_l2
        << ", distance to post-Engram features=" << post_engram_max_abs;
  }

 private:
  std::string saved_engram_config_;
};

TEST_F(DeepseekV41CaptureTest, CapturesEnteringResidualWithoutEngram) {
  expect_entering_residual(/*with_engram=*/false, /*value_scale=*/0);
}

TEST_F(DeepseekV41CaptureTest, CapturesEnteringResidualWithZeroEngramValue) {
  expect_entering_residual(/*with_engram=*/true, /*value_scale=*/0);
}

TEST_F(DeepseekV41CaptureTest, CapturesEnteringResidualBeforeEngramInjection) {
  expect_entering_residual(/*with_engram=*/true, /*value_scale=*/0.5);
}

TEST_F(DeepseekV41CaptureTest, MaterializesPostBeforeUpperLayerCapture) {
  expect_entering_residual(/*with_engram=*/false,
                           /*value_scale=*/0,
                           /*capture_layer=*/1);
}

TEST_F(DeepseekV41CaptureTest,
       MaterializesPostBeforeUpperLayerEngramAndCapture) {
  expect_entering_residual(/*with_engram=*/true,
                           /*value_scale=*/0.5,
                           /*capture_layer=*/1);
}

// Hidden sizes other than 5120 have no fused MHC kernel: the layers run the
// unfused boundaries and never defer their post.
TEST_F(DeepseekV41CaptureTest, UnsupportedHiddenSizeUsesUnfusedBoundaries) {
  expect_entering_residual(/*with_engram=*/false,
                           /*value_scale=*/0,
                           /*capture_layer=*/1,
                           /*model_hidden_size=*/256);
}

TEST_F(DeepseekV41CaptureTest,
       UnsupportedHiddenSizeUsesUnfusedBoundariesWithEngram) {
  expect_entering_residual(/*with_engram=*/true,
                           /*value_scale=*/0.5,
                           /*capture_layer=*/1,
                           /*model_hidden_size=*/256);
}

TEST_F(DeepseekV41CaptureTest,
       DSparkExecutorRegroupsRealModelAndReplaysAfterCandidateRejection) {
  torch::InferenceMode inference;
  torch::DeviceGuard guard(options_.device());
  ModelConfig::get_instance().engram_config() = R"({"placement":"cpu"})";
  constexpr int32_t kWidth = 6;
  int32_t requests = 3;
  args_.dtype("bfloat16")
      .hidden_size(5120)
      .hc_mult(4)
      .n_heads(8)
      .vocab_size(8)
      .window_size(16)
      .index_topk(4)
      .num_speculative_tokens(kWidth - 1)
      .max_position_embeddings(512)
      .layers_to_capture({0})
      .engram_layer_ids({0})
      .engram_num_embeddings({18})
      .engram_head_dim(32)
      .engram_max_ngram_size(3)
      .engram_compressed_vocab_size(8)
      .engram_pad_token_id(0);
  set_cache_role(2);
  auto weights = make_weights();
  // Nonzero, nonuniform projections make stale tokens, histories and cache
  // pages visible. MoE remains zero so attention/cache failures stay isolated.
  const auto patterned = [](std::vector<int64_t> shape, double scale) {
    const int64_t elements = std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    return (torch::arange(elements, torch::kFloat32).mul(0.17).sin() * scale)
        .reshape(shape)
        .to(torch::kBFloat16);
  };
  const int64_t hidden_size = args_.hidden_size();
  weights["embed.weight"] = patterned({8, hidden_size}, 0.5);
  weights["layers.0.attn.wq_b.weight"] =
      patterned({4096, args_.q_lora_rank()}, 0.01);
  weights["layers.0.attn.wo_a.weight"] = patterned({1, 128, 4096}, 0.01);
  weights["layers.0.attn.attn_sink"] = torch::zeros({8}, torch::kFloat32);
  for (const std::string name : {"wq_a",
                                 "wkv",
                                 "wo_b",
                                 "compressor.wkv",
                                 "compressor.wgate",
                                 "indexer.wq_b",
                                 "indexer.weights_proj",
                                 "indexer.wk"}) {
    auto& weight = weights.at("layers.0.attn." + name + ".weight");
    weight = patterned(weight.sizes().vec(), 0.02);
  }
  weights.emplace(
      "layers.0.engram.embed.weight",
      torch::arange(18).add(0x30).unsqueeze(1).expand({18, 32}).contiguous().to(
          torch::kUInt8));
  weights.emplace("layers.0.engram.embed.scale",
                  torch::full({18, 1}, 127, torch::kUInt8));
  // Zero keys fix the gate near 0.5. Distinct signed value directions for
  // the two hash columns make a history-only change visible after final norm.
  auto engram_wkv = torch::zeros({5 * hidden_size, 64}, torch::kBFloat16);
  auto value_rows = engram_wkv.slice(0, 4 * hidden_size);
  auto channels = torch::arange(hidden_size, torch::kFloat32);
  value_rows.slice(1, 0, 32).copy_(
      channels.remainder(2).mul(2).sub(1).div(32).unsqueeze(1));
  value_rows.slice(1, 32, 64).copy_(channels.remainder(4)
                                        .ge(2)
                                        .to(torch::kFloat32)
                                        .mul(2)
                                        .sub(1)
                                        .div(32)
                                        .unsqueeze(1));
  weights.emplace("layers.0.engram.wkv.weight", engram_wkv);
  weights.emplace("layers.0.engram.q_weight",
                  torch::ones({4, hidden_size}, torch::kBFloat16));
  weights.emplace("layers.0.engram.k_weight",
                  torch::ones({4, hidden_size}, torch::kBFloat16));
  const StateDict hashes({
      {"token_map", torch::arange(8, torch::kInt64)},
      {"multipliers", torch::tensor({{1, 3, 5}}, torch::kInt64)},
      {"primes", torch::tensor({{7, 11}}, torch::kInt64)},
      {"offsets", torch::tensor({{0, 7}}, torch::kInt64)},
  });
  const auto context = make_context();
  ExecutorV41Model graph_model(context, StateDict(weights), hashes);
  ExecutorV41Model eager_model(context, StateDict(weights), hashes);
  runtime::Options executor_options;
  executor_options.block_size(128)
      .dp_size(1)
      .num_decoding_tokens(kWidth)
      .num_speculative_tokens(kWidth - 1)
      .enable_speculative_decode(true)
      .speculative_algorithm("DSpark")
      .enable_graph_aux_hidden_states(true);
  mlu::MluGraphExecutorImpl executor(
      &graph_model, args_, options_.device(), executor_options);
  const auto ints = options_.dtype(torch::kInt32);
  const auto longs = options_.dtype(torch::kInt64);
  const auto make_cache = [&]() {
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache = torch::zeros({8, 128, 1, 512}, options_);
    tensors.key_cache = torch::zeros_like(tensors.swa_cache);
    tensors.index_cache = torch::zeros({8, 128, 1, 128}, options_);
    tensors.compress_kv_state =
        torch::zeros({8, 32, 512}, options_.dtype(torch::kFloat32));
    tensors.compress_score_state = torch::zeros_like(tensors.compress_kv_state);
    std::vector<KVCache> caches;
    caches.reserve(1);
    caches.emplace_back(tensors);
    return caches;
  };
  const auto cache_tensors = [](const std::vector<KVCache>& caches) {
    return std::vector<torch::Tensor>{caches[0].get_swa_cache(),
                                      caches[0].get_k_cache(),
                                      caches[0].get_index_cache(),
                                      caches[0].get_compress_kv_state(),
                                      caches[0].get_compress_score_state()};
  };
  const auto copy_cache = [&](std::vector<KVCache>& to,
                              const std::vector<KVCache>& from) {
    auto destination = cache_tensors(to);
    const auto source = cache_tensors(from);
    for (size_t index = 0; index < source.size(); ++index) {
      destination[index].copy_(source[index]);
    }
  };
  const auto expect_cache = [&](const std::vector<KVCache>& actual,
                                const std::vector<KVCache>& expected) {
    const auto left = cache_tensors(actual);
    const auto right = cache_tensors(expected);
    for (size_t index = 0; index < left.size(); ++index) {
      EXPECT_TRUE(torch::allclose(left[index], right[index], 0.02, 0.02))
          << "Cache tensor " << index;
    }
  };
  // Eager input keeps real request chunks; it never calls group_verify_input.
  const auto make_input = [&](const std::vector<int32_t>& lengths,
                              const std::vector<int32_t>& begins,
                              const std::vector<int32_t>& blocks,
                              std::vector<int32_t> history) {
    ModelInputParams input;
    input.meta.batch_forward_type = BatchForwardType::DECODE;
    const int32_t count = lengths.size();
    input.meta.num_sequences = count;
    input.meta.actual_num_sequences = count;
    input.meta.q_max_seq_len =
        *std::max_element(lengths.begin(), lengths.end());
    input.attention.host.q_seq_lens = {0};
    input.attention.host.kv_seq_lens = {0};
    input.attention.host.q_seq_lens.reserve(count + 1);
    input.attention.host.kv_seq_lens.reserve(count + 1);
    for (int32_t seq = 0; seq < count; ++seq) {
      input.attention.host.q_seq_lens.emplace_back(
          input.attention.host.q_seq_lens.back() + lengths[seq]);
      const int32_t end = begins[seq] + lengths[seq];
      input.meta.kv_max_seq_len = std::max(input.meta.kv_max_seq_len, end);
      input.attention.host.kv_seq_lens.emplace_back(
          input.attention.host.kv_seq_lens.back() + end);
    }
    input.attention.host.q_cu_seq_lens = input.attention.host.q_seq_lens;
    input.embedding.engram_lookback_token_ids = std::move(history);
    auto table = torch::tensor(blocks, torch::kInt32).view({count, 1});
    input.multi_block_tables = {table, table.clone()};
    return input;
  };
  const auto make_positions = [&](const std::vector<int32_t>& begins,
                                  int32_t count) {
    std::vector<torch::Tensor> parts;
    parts.reserve(begins.size());
    for (int32_t begin : begins) {
      parts.emplace_back(torch::arange(begin, begin + count, longs));
    }
    return torch::cat(parts);
  };
  const auto flatten_verify = [&](const ModelInputParams& grouped,
                                  const std::vector<int32_t>& begins) {
    ModelInputParams flat = grouped;
    const int32_t count = begins.size();
    flat.meta.num_sequences = count * kWidth;
    flat.meta.actual_num_sequences = count * kWidth;
    flat.meta.spec_verify_width = kWidth;
    flat.meta.q_max_seq_len = 1;
    // Metadata-owning models use one planner state ID per logical request.
    flat.embedding.linear_state_ids.resize(count);
    std::iota(flat.embedding.linear_state_ids.begin(),
              flat.embedding.linear_state_ids.end(),
              1);
    flat.embedding.linear_state_indices =
        torch::tensor(flat.embedding.linear_state_ids, ints);
    flat.attention.host.q_seq_lens = {0};
    flat.attention.host.kv_seq_lens = {0};
    flat.attention.host.q_seq_lens.reserve(count * kWidth + 1);
    flat.attention.host.kv_seq_lens.reserve(count * kWidth + 1);
    for (int32_t seq = 0; seq < count; ++seq) {
      for (int32_t row = 0; row < kWidth; ++row) {
        flat.attention.host.q_seq_lens.emplace_back(
            flat.attention.host.q_seq_lens.back() + 1);
        flat.attention.host.kv_seq_lens.emplace_back(
            flat.attention.host.kv_seq_lens.back() + begins[seq] + row + 1);
      }
    }
    flat.attention.host.q_cu_seq_lens = flat.attention.host.q_seq_lens;
    flat.attention.device.q_seq_lens =
        torch::tensor(flat.attention.host.q_seq_lens, ints);
    flat.attention.device.kv_seq_lens =
        torch::tensor(flat.attention.host.kv_seq_lens, ints);
    for (auto& table : flat.multi_block_tables) {
      table = table.repeat_interleave(kWidth, 0);
    }
    const auto device_table = flat.multi_block_tables[0].to(options_.device());
    flat.attention.device.block_tables = device_table;
    flat.attention.device.new_cache_slots =
        device_table.select(1, 0) * 128 +
        make_positions(begins, kWidth).to(torch::kInt32);
    return flat;
  };
  auto graph_cache = make_cache();
  auto eager_cache = make_cache();
  auto prefix_cache = make_cache();
  int32_t captured_forward_count = 0;
  torch::Tensor previous_output;
  torch::Tensor previous_expected;
  std::vector<int32_t> blocks;
  std::vector<int32_t> history;
  torch::Tensor last_tokens;
  std::vector<int32_t> begins;
  for (int32_t round = 0; round < 3; ++round) {
    SCOPED_TRACE(round);
    requests = round == 1 ? 4 : 3;
    begins.clear();
    blocks.clear();
    history.clear();
    std::vector<int32_t> prefix_tokens, prefix_positions;
    for (int32_t seq = 0; seq < requests; ++seq) {
      begins.emplace_back(2 * seq + 1);
      blocks.emplace_back((seq + round) % 8);
      history.emplace_back((round + seq + 1) % 8);
      history.emplace_back((7 - round + seq) % 8);
      for (int32_t row = 0; row < begins.back(); ++row) {
        prefix_tokens.emplace_back((seq + row) % 7 + 1);
        prefix_positions.emplace_back(row);
      }
    }
    for (auto& tensor : cache_tensors(prefix_cache)) {
      tensor.zero_();
    }
    auto prefix_input = make_input(begins,
                                   std::vector<int32_t>(requests, 0),
                                   blocks,
                                   std::vector<int32_t>(2 * requests, -1));
    eager_model.forward(torch::tensor(prefix_tokens, longs),
                        torch::tensor(prefix_positions, longs),
                        prefix_cache,
                        prefix_input);
    copy_cache(graph_cache, prefix_cache);
    copy_cache(eager_cache, prefix_cache);
    auto grouped = make_input(
        std::vector<int32_t>(requests, kWidth), begins, blocks, history);
    if (round == 1) {
      for (auto& table : grouped.multi_block_tables) {
        table = torch::cat({table, (table + 1).remainder(8)}, 1);
      }
    }
    auto flat = flatten_verify(grouped, begins);
    // The first replay changes history and physical pages only; the second
    // also changes tokens. Both must be visible at the comparison tolerance.
    const int32_t token_shift = round == 1 ? 0 : round;
    last_tokens = (torch::arange(requests * kWidth, longs) + token_shift)
                      .remainder(7)
                      .add(1);
    const auto positions = make_positions(begins, kWidth);
    const auto expected =
        eager_model.forward(last_tokens, positions, eager_cache, grouped);
    const auto actual = executor.run(last_tokens, positions, graph_cache, flat);
    EXPECT_TRUE(torch::allclose(
        actual.hidden_states, expected.hidden_states, 0.02, 0.02));
    EXPECT_TRUE(
        torch::equal(actual.aux_hidden_states, expected.aux_hidden_states));
    expect_cache(graph_cache, eager_cache);
    ASSERT_NE(graph_model.graph_state(), nullptr);
    const auto& state = *graph_model.graph_state();
    ASSERT_TRUE(state.verify_input.has_value());
    EXPECT_EQ(state.verify_input->meta.num_sequences, 4);
    EXPECT_EQ(state.verify_input->meta.actual_num_sequences, requests);
    EXPECT_EQ(state.verify_input->attention.host.q_seq_lens,
              (std::vector<int32_t>{0, 6, 12, 18, 24}));
    const auto& compressed = state.shared.compressed[1];
    EXPECT_EQ(compressed.outputs.num_valid_rows.item<int32_t>(), requests * 3);
    EXPECT_TRUE(torch::equal(
        state.shared.query_prefix.cpu(),
        torch::tensor({0, 6, 12, 18, requests * 6}, torch::kInt32)));
    EXPECT_EQ(state.shared.block_tables[1].size(1), 2);
    if (round == 2) {
      EXPECT_TRUE(
          state.shared.block_tables[1].select(1, 1).eq(0).all().item<bool>());
    }
    ASSERT_EQ(state.hashes.size(), 1);
    ASSERT_NE(state.hashes[0], nullptr);
    // This also observes query_width=6: independent decode hashes would have
    // needed 24 history IDs, while the real grouped HashState accepts four.
    EXPECT_EQ(state.verify_input->embedding.engram_lookback_token_ids, history);
    if (round == 0) {
      captured_forward_count = graph_model.forward_count();
      ASSERT_GT(captured_forward_count, 0);
    } else {
      EXPECT_EQ(graph_model.forward_count(), captured_forward_count)
          << "Expected graph replay, not eager execution or recapture";
      EXPECT_FALSE(torch::allclose(actual.hidden_states.slice(0, 0, 18),
                                   previous_output.slice(0, 0, 18),
                                   0.02,
                                   0.02));
      EXPECT_FALSE(torch::allclose(expected.hidden_states.slice(0, 0, 18),
                                   previous_expected.slice(0, 0, 18),
                                   0.02,
                                   0.02))
          << "Fixture must expose stale history at the verification tolerance";
    }
    previous_output = actual.hidden_states.clone();
    previous_expected = expected.hidden_states.clone();
  }
  // Keep only the first two verified tokens in an independent reference.
  // The graph cache still contains all rejected proposals, including the raw
  // compressor rows that must survive an interior rejection at an odd boundary.
  copy_cache(eager_cache, prefix_cache);
  const auto accepted = last_tokens.view({requests, kWidth})
                            .slice(1, 0, 2)
                            .contiguous()
                            .flatten();
  auto accepted_input =
      make_input(std::vector<int32_t>(requests, 2), begins, blocks, history);
  eager_model.forward(
      accepted, make_positions(begins, 2), eager_cache, accepted_input);
  std::vector<int32_t> resumed_begins;
  resumed_begins.reserve(requests);
  history.clear();
  for (int32_t seq = 0; seq < requests; ++seq) {
    resumed_begins.emplace_back(begins[seq] + 2);
    history.emplace_back(accepted[seq * 2 + 1].item<int32_t>());
    history.emplace_back(accepted[seq * 2].item<int32_t>());
  }
  auto resumed = make_input(
      std::vector<int32_t>(requests, kWidth), resumed_begins, blocks, history);
  auto flat = flatten_verify(resumed, resumed_begins);
  auto tokens = (last_tokens + 2).remainder(7).add(1);
  const auto positions = make_positions(resumed_begins, kWidth);
  const auto expected =
      eager_model.forward(tokens, positions, eager_cache, resumed);
  const auto actual = executor.run(tokens, positions, graph_cache, flat);
  EXPECT_EQ(graph_model.forward_count(), captured_forward_count);
  EXPECT_TRUE(torch::allclose(
      actual.hidden_states, expected.hidden_states, 0.02, 0.02));
  EXPECT_TRUE(
      torch::equal(actual.aux_hidden_states, expected.aux_hidden_states));
  expect_cache(graph_cache, eager_cache);
  const auto* large_state = graph_model.graph_state();
  auto large_latent = large_state->shared.compressed[1].outputs.latent.clone();
  auto small = make_input({kWidth}, {3}, {blocks[0]}, {history[0], history[1]});
  auto small_flat = flatten_verify(small, {3});
  auto small_tokens = tokens.slice(0, 0, kWidth);
  auto small_positions = make_positions({3}, kWidth);
  auto small_expected =
      eager_model.forward(small_tokens, small_positions, eager_cache, small);
  auto small_actual =
      executor.run(small_tokens, small_positions, graph_cache, small_flat);
  EXPECT_TRUE(torch::allclose(
      small_actual.hidden_states, small_expected.hidden_states, 0.02, 0.02));
  EXPECT_NE(graph_model.graph_state(), large_state);
  EXPECT_NE(
      graph_model.graph_state()->shared.compressed[1].outputs.latent.data_ptr(),
      large_state->shared.compressed[1].outputs.latent.data_ptr());
  EXPECT_TRUE(torch::equal(large_latent,
                           large_state->shared.compressed[1].outputs.latent));
  const int32_t after_small_capture = graph_model.forward_count();
  auto again_expected =
      eager_model.forward(tokens, positions, eager_cache, resumed);
  auto again_actual = executor.run(tokens, positions, graph_cache, flat);
  EXPECT_EQ(graph_model.forward_count(), after_small_capture);
  EXPECT_EQ(graph_model.graph_state(), large_state);
  EXPECT_TRUE(torch::allclose(
      again_actual.hidden_states, again_expected.hidden_states, 0.02, 0.02));
  expect_cache(graph_cache, eager_cache);
  std::vector<torch::Tensor> idle_cache;
  for (const auto& tensor : cache_tensors(eager_cache)) {
    idle_cache.emplace_back(tensor.clone());
  }
  ModelInputParams idle;
  idle.meta.num_sequences = 0;
  idle.meta.q_max_seq_len = 0;
  idle.parallel.dp_global_token_nums = {0};
  idle.attention.host.q_seq_lens = {0};
  idle.attention.host.kv_seq_lens = {0};
  auto empty_table = torch::empty({0, 1}, ints);
  idle.multi_block_tables = {empty_table, empty_table};
  eager_model.forward(
      torch::empty({0}, longs), torch::empty({0}, longs), eager_cache, idle);
  int32_t tensor_index = 0;
  for (const auto& tensor : cache_tensors(eager_cache)) {
    EXPECT_TRUE(torch::equal(tensor, idle_cache[tensor_index++]));
  }
}

TEST_F(DeepseekV41MluModelTest, CpuBlockTablesSupportEagerPrefillAndDecode) {
  torch::InferenceMode inference;
  const torch::DeviceGuard guard(options_.device());
  args_.n_heads(8).window_size(16).index_topk(4).num_speculative_tokens(5);
  for (int32_t ratio : {1, 2}) {
    SCOPED_TRACE(ratio);
    set_cache_role(ratio);
    auto weights = make_weights();
    weights.at("layers.0.attn.compressor.wkv.weight")
        .select(/*dim=*/1, /*index=*/0)
        .fill_(0.125);
    weights.at("layers.0.attn.indexer.wk.weight").fill_(0.125);
    mlu::model::DeepseekV41Model model(make_context());
    model->load_state_dict(StateDict(weights));
    model->verify_loaded_weights();
    const auto make_cache = [&]() {
      DeepSeekV4KVCacheTensors tensors;
      tensors.swa_cache = torch::zeros({4, 128, 1, 512}, options_);
      tensors.key_cache = torch::full_like(tensors.swa_cache, -7);
      tensors.index_cache = torch::full({4, 128, 1, 128}, -7, options_);
      if (ratio == 2) {
        tensors.compress_kv_state =
            torch::zeros({4, 32, 512}, options_.dtype(torch::kFloat32));
        tensors.compress_score_state =
            torch::zeros_like(tensors.compress_kv_state);
      }
      std::vector<KVCache> caches;
      caches.reserve(1);
      caches.emplace_back(tensors);
      return caches;
    };
    auto actual_cache = make_cache();
    auto expected_cache = make_cache();
    for (bool prefill : {true, false}) {
      SCOPED_TRACE(prefill);
      ModelInputParams host_input;
      host_input.meta.batch_forward_type =
          prefill ? BatchForwardType::PREFILL : BatchForwardType::DECODE;
      host_input.meta.num_sequences = 2;
      host_input.meta.actual_num_sequences = 2;
      host_input.meta.q_max_seq_len = prefill ? 3 : 1;
      host_input.meta.kv_max_seq_len = prefill ? 3 : 4;
      host_input.attention.host.q_seq_lens =
          prefill ? std::vector<int32_t>{0, 3, 4}
                  : std::vector<int32_t>{0, 1, 2};
      host_input.attention.host.kv_seq_lens =
          prefill ? std::vector<int32_t>{0, 3, 4}
                  : std::vector<int32_t>{0, 4, 6};
      host_input.multi_block_tables = {
          torch::tensor({1, 0}, torch::kInt32).view({2, 1}),
          torch::tensor({3, 2}, torch::kInt32).view({2, 1}),
          torch::tensor({3, 2}, torch::kInt32).view({2, 1})};
      const auto input = host_input.to(options_.device());
      for (const auto& table : input.multi_block_tables) {
        ASSERT_TRUE(table.is_cpu());
      }
      ModelInputParams device_input = input;
      for (auto& table : device_input.multi_block_tables) {
        table = table.to(options_.device());
      }
      const auto longs = options_.dtype(torch::kInt64);
      const auto positions =
          (prefill ? torch::tensor({0, 1, 2, 0}, torch::kInt64)
                   : torch::tensor({3, 1}, torch::kInt64))
              .to(longs);
      const auto tokens = torch::arange(positions.numel(), longs);
      const auto expected =
          model->forward(tokens, positions, expected_cache, device_input);
      const auto actual =
          model->forward(tokens, positions, actual_cache, input);
      EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
      EXPECT_TRUE(torch::equal(actual_cache[0].get_swa_cache(),
                               expected_cache[0].get_swa_cache()));
      EXPECT_TRUE(torch::equal(actual_cache[0].get_k_cache(),
                               expected_cache[0].get_k_cache()));
      EXPECT_TRUE(torch::equal(actual_cache[0].get_index_cache(),
                               expected_cache[0].get_index_cache()));
      EXPECT_TRUE(
          actual_cache[0].get_k_cache()[3][0].ne(-7).any().item<bool>());
      if (ratio == 2) {
        EXPECT_TRUE(torch::equal(actual_cache[0].get_compress_kv_state(),
                                 expected_cache[0].get_compress_kv_state()));
        EXPECT_TRUE(torch::equal(actual_cache[0].get_compress_score_state(),
                                 expected_cache[0].get_compress_score_state()));
      }
    }
  }
}

TEST_F(DeepseekV41MluModelTest, SelectsFusedMhcOnlyForSupportedShapes) {
  {
    layer::DeepseekV41DecoderLayer layer(make_context(), 0);
    EXPECT_TRUE(layer->uses_fused_mhc());
  }
  {
    args_.hidden_size(128);
    layer::DeepseekV41DecoderLayer layer(make_context(), 0);
    EXPECT_FALSE(layer->uses_fused_mhc());
  }
  {
    args_.hidden_size(5120).hc_mult(2);
    layer::DeepseekV41DecoderLayer layer(make_context(), 0);
    EXPECT_FALSE(layer->uses_fused_mhc());
  }
}

TEST_F(DeepseekV41MluModelTest, RejectsInvalidSinkhornIterations) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        args_.hc_sinkhorn_iters(0);
        layer::DeepseekV41DecoderLayer layer(make_context(), 0);
      },
      "args.hc_sinkhorn_iters");
}

TEST_F(DeepseekV41MluModelTest, EmbeddingModeLoadsWithoutLanguageModelHead) {
  args_.embedding_mode(true);
  mlu::model::DeepseekV41ForCausalLM model(make_context());
  model->load_model(make_loader(make_weights()));
  auto tokens = torch::tensor({0, 2}, options_.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(
      model->get_input_embeddings(tokens).cpu(),
      torch::full({2, args_.hidden_size()}, 0.25, torch::kBFloat16)));
}

TEST_F(DeepseekV41MluModelTest, TiedHeadLoadsFromEmbeddingWeights) {
  args_.tie_word_embeddings(true);
  mlu::model::DeepseekV41ForCausalLM model(make_context());
  model->load_model(make_loader(make_weights()));
  ASSERT_TRUE(model->has_loaded_vocab_weights());
  auto hidden = torch::ones({2, args_.hidden_size()}, options_);
  EXPECT_TRUE(torch::equal(
      model->logits(hidden, torch::Tensor()).cpu(),
      torch::full({2, 4}, args_.hidden_size() * 0.25, torch::kBFloat16)));
}

TEST_F(DeepseekV41MluModelTest,
       RejectsMissingVocabNormAndHyperConnectionWeights) {
  for (const std::string name : {"embed.weight",
                                 "head.weight",
                                 "norm.weight",
                                 "layers.0.attn_norm.weight",
                                 "layers.0.ffn_norm.weight",
                                 "layers.0.hc_attn_pre.hc_fn",
                                 "layers.0.hc_attn_pre.hc_base",
                                 "layers.0.hc_attn_pre.hc_scale",
                                 "layers.0.hc_ffn_pre.hc_fn",
                                 "layers.0.hc_ffn_pre.hc_base",
                                 "layers.0.hc_ffn_pre.hc_scale"}) {
    args_.embedding_mode(name != "head.weight");
    expect_missing_weight(name);
  }
}

TEST_F(DeepseekV41MluModelTest,
       RejectsMissingAttentionProjectionAndSelectionWeights) {
  args_.embedding_mode(true);
  for (int32_t ratio : {0, 1, 2}) {
    SCOPED_TRACE(ratio);
    set_cache_role(ratio);
    std::vector<std::string> fields{"wq_b.weight", "wo_b.weight"};
    fields.reserve(9);
    if (ratio > 0) {
      fields.insert(fields.end(),
                    {"compressor.wkv.weight",
                     "compressor.norm.weight",
                     "indexer.wq_b.weight",
                     "indexer.weights_proj.weight",
                     "indexer.wk.weight",
                     "indexer.k_norm.weight"});
    }
    if (ratio == 2) {
      fields.emplace_back("compressor.wgate.weight");
    }
    for (const auto& field : fields) {
      expect_missing_weight("layers.0.attn." + field);
    }
  }
}

TEST_F(DeepseekV41MluModelTest,
       CompleteCheckpointLoadsAcrossShardsAndCacheRoles) {
  for (int32_t ratio : {0, 1, 2}) {
    SCOPED_TRACE(ratio);
    set_cache_role(ratio);
    auto weights = make_weights();
    weights.emplace(
        "head.weight",
        torch::full({4, args_.hidden_size()}, 0.5, torch::kBFloat16));
    std::vector<Weights> shards(2);
    for (auto& [name, value] : weights) {
      const bool decoder = name.rfind("layers.", 0) == 0;
      shards[decoder ? 0 : 1].emplace("model." + name, std::move(value));
    }
    mlu::model::DeepseekV41ForCausalLM model(make_context());
    model->load_model(make_loader(std::move(shards)));
    ASSERT_TRUE(model->has_loaded_vocab_weights());
    auto hidden = torch::ones({2, args_.hidden_size()}, options_);
    EXPECT_TRUE(torch::equal(
        model->logits(hidden, torch::Tensor()).cpu(),
        torch::full({2, 4}, args_.hidden_size() * 0.5, torch::kBFloat16)));
  }
}

TEST_F(DeepseekV41MluModelTest,
       VerifyMetadataGroupsRequestsAndReusesGatherStorage) {
  args_.window_size(16).index_topk(4);
  set_cache_role(2);
  mlu::model::DeepseekV41Model model(make_context());
  mlu::model::DeepseekV41GraphState state;
  ModelInputParams input;
  input.meta.batch_forward_type = BatchForwardType::DECODE;
  input.meta.num_sequences = 6;
  input.meta.actual_num_sequences = 3;
  input.meta.spec_verify_width = 3;
  input.attention.host.q_seq_lens = {0, 1, 2, 3, 4, 5, 6};
  input.attention.host.kv_seq_lens = {0, 4, 9, 15, 16, 18, 21};
  input.multi_block_tables = {
      torch::arange(12, options_.dtype(torch::kInt32)).view({6, 2})};
  auto positions = torch::arange(6, options_.dtype(torch::kInt64));
  input.attention.host.positions = torch::arange(6, torch::kInt64);

  model->prepare_graph_forward_metadata(&state, positions, input);
  ASSERT_TRUE(state.verify_input.has_value());
  EXPECT_EQ(state.verify_input->meta.num_sequences, 2);
  EXPECT_EQ(state.verify_input->meta.actual_num_sequences, 1);
  EXPECT_EQ(state.verify_input->attention.host.q_seq_lens,
            (std::vector<int32_t>{0, 3, 6}));
  EXPECT_EQ(state.verify_input->attention.host.kv_seq_lens,
            (std::vector<int32_t>{0, 6, 9}));
  EXPECT_EQ(state.shared.topk.size(), 2);
  EXPECT_EQ(state.shared.candidates.size(), 2);
  EXPECT_EQ(state.shared.selected_lengths.numel(), 6);
  EXPECT_EQ(state.shared.selected_kv.size(0), 6 * (16 + 4));
  EXPECT_TRUE(torch::equal(state.shared.block_tables[0],
                           input.multi_block_tables[0].slice(0, 0, 6, 3)));

  const void* gather_address = state.shared.selected_kv.data_ptr();
  input.attention.host.kv_seq_lens = {0, 6, 13, 21, 22, 24, 27};
  model->prepare_graph_forward_metadata(&state, positions, input);
  EXPECT_EQ(state.shared.selected_kv.data_ptr(), gather_address);
  EXPECT_EQ(state.verify_input->attention.host.kv_seq_lens,
            (std::vector<int32_t>{0, 8, 11}));
}

TEST_F(DeepseekV41MluModelTest,
       SharedKvConsumerDoesNotRequireSourceOnlyWeights) {
  args_.n_layers(2)
      .compress_ratios({1, 1})
      .kv_source_layer_ids({0})
      .index_source_layer_ids({0});
  auto state = StateDict(make_weights()).get_dict_with_prefix("layers.0.attn.");
  Weights weights;
  for (auto it = state.begin(); it != state.end(); ++it) {
    if (it->first.rfind("compressor.", 0) == 0 ||
        it->first.rfind("indexer.", 0) == 0) {
      continue;
    }
    weights.emplace(it->first, it->second);
  }
  layer::DeepseekV41Attention attention(make_context(), /*layer_id=*/1);
  attention->load_state_dict(StateDict(std::move(weights)));
  attention->verify_loaded_weights();
}

TEST_F(DeepseekV41MluModelTest,
       CompressedGraphMetadataOwnsCapacityAndClearsPadding) {
  set_cache_role(2);
  mlu::model::DeepseekV41Model model(make_context());
  mlu::model::DeepseekV41GraphState first, second;
  ModelInputParams input;
  input.enable_graph = true;
  input.meta.batch_forward_type = BatchForwardType::DECODE;
  input.meta.num_sequences = 24;
  input.meta.actual_num_sequences = 18;
  input.meta.spec_verify_width = 6;
  input.graph.kv_seq_len_capacity = 512;
  for (int32_t row = 0; row <= 24; ++row) {
    input.attention.host.q_seq_lens.emplace_back(row);
    input.attention.host.kv_seq_lens.emplace_back(row * 8);
  }
  auto positions =
      torch::arange(2, 8, options_.dtype(torch::kInt64)).repeat({4});
  input.attention.host.positions =
      torch::arange(2, 8, torch::kInt64).repeat({4});
  input.multi_block_tables = {torch::arange(24, torch::kInt32).view({24, 1})};
  model->prepare_graph_forward_metadata(&first, positions, input);
  EXPECT_TRUE(torch::equal(first.shared.query_prefix.cpu(),
                           torch::tensor({0, 6, 12, 18, 18}, torch::kInt32)));
  EXPECT_EQ(first.shared.num_valid_input_rows.item<int32_t>(), 18);
  const void* latent = first.shared.compressed[1].outputs.latent.data_ptr();
  const void* table = first.shared.block_tables[0].data_ptr();
  input.multi_block_tables[0] =
      torch::tensor({2, 1}, torch::kInt32).view({1, 2}).expand({24, 2});
  input.meta.actual_num_sequences = 24;
  model->prepare_graph_forward_metadata(&first, positions, input);
  EXPECT_TRUE(torch::equal(first.shared.block_tables[0][0].cpu(),
                           torch::tensor({2, 1, 2, 1}, torch::kInt32)));
  EXPECT_EQ(first.shared.num_valid_input_rows.item<int32_t>(), 24);
  model->prepare_graph_forward_metadata(&second, positions, input);
  EXPECT_NE(second.shared.compressed[1].outputs.latent.data_ptr(), latent);
  EXPECT_NE(second.shared.block_tables[0].data_ptr(), table);
  input.meta.actual_num_sequences = 0;
  model->prepare_graph_forward_metadata(&first, positions, input);
  EXPECT_EQ(first.shared.num_valid_input_rows.item<int32_t>(), 0);
  EXPECT_TRUE(first.shared.query_prefix.eq(0).all().item<bool>());
  EXPECT_TRUE(first.shared.compressed[1].prefix.eq(0).all().item<bool>());
  EXPECT_EQ(first.shared.compressed[1].outputs.latent.data_ptr(), latent);
  EXPECT_EQ(first.shared.block_tables[0].data_ptr(), table);
  EXPECT_EQ(second.shared.num_valid_input_rows.item<int32_t>(), 24);
}

}  // namespace
}  // namespace xllm
