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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model_loader.h"
#include "core/layers/common/attention_metadata.h"
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
        .hidden_size(128)
        .vocab_size(4)
        .n_heads(1)
        .head_dim(512)
        .q_lora_rank(128)
        .qk_rope_head_dim(64)
        .index_n_heads(1)
        .index_head_dim(128)
        .o_groups(1)
        .o_lora_rank(128)
        .hc_mult(2)
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
    Weights weights{
        {"embed.weight", torch::full({4, 128}, 0.25, options)},
        {"norm.weight", torch::ones({128}, options)},
        {"layers.0.attn.wq_a.weight", torch::zeros({128, 128}, options)},
        {"layers.0.attn.wq_b.weight", torch::zeros({512, 128}, options)},
        {"layers.0.attn.wkv.weight", torch::zeros({512, 128}, options)},
        {"layers.0.attn.q_norm.weight", torch::ones({128}, options)},
        {"layers.0.attn.kv_norm.weight", torch::ones({512}, options)},
        {"layers.0.attn.wo_a.weight", torch::zeros({1, 128, 512}, options)},
        {"layers.0.attn.wo_b.weight", torch::zeros({128, 128}, options)},
        {"layers.0.attn.attn_sink", torch::zeros({1}, torch::kFloat32)},
        {"layers.0.attn_norm.weight", torch::ones({128}, options)},
        {"layers.0.ffn_norm.weight", torch::ones({128}, options)},
        {"layers.0.ffn.gate.weight", torch::zeros({2, 128}, options)},
    };
    for (const std::string prefix :
         {"layers.0.hc_attn_pre.", "layers.0.hc_ffn_pre."}) {
      weights.emplace(prefix + "hc_fn", torch::zeros({8, 256}, options));
      weights.emplace(prefix + "hc_base", torch::zeros({8}, torch::kFloat32));
      weights.emplace(prefix + "hc_scale", torch::ones({3}, torch::kFloat32));
    }
    for (int32_t expert = 0; expert < 2; ++expert) {
      const std::string prefix =
          "layers.0.ffn.experts." + std::to_string(expert) + ".";
      for (const std::string projection : {"w1", "w2", "w3"}) {
        weights.emplace(prefix + projection + ".weight",
                        torch::zeros({128, 128}, options));
      }
    }
    const int32_t ratio = args_.compress_ratios().front();
    if (ratio > 0) {
      weights.emplace("layers.0.attn.compressor.wkv.weight",
                      torch::zeros({512, 128}, options));
      weights.emplace("layers.0.attn.compressor.norm.weight",
                      torch::ones({512}, options));
      weights.emplace("layers.0.attn.indexer.wq_b.weight",
                      torch::zeros({128, 128}, options));
      weights.emplace("layers.0.attn.indexer.weights_proj.weight",
                      torch::zeros({1, 128}, options));
      weights.emplace("layers.0.attn.indexer.wk.weight",
                      torch::zeros({128, 512}, options));
      weights.emplace("layers.0.attn.indexer.k_norm.weight",
                      torch::ones({128}, options));
    }
    if (ratio == 2) {
      weights.emplace("layers.0.attn.compressor.wgate.weight",
                      torch::zeros({512, 128}, options));
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
    args_.compress_ratios({ratio});
    args_.kv_source_layer_ids(ratio == 0 ? std::vector<int32_t>{}
                                         : std::vector<int32_t>{0});
    args_.index_source_layer_ids(args_.kv_source_layer_ids());
  }

  void expect_missing_weight(const std::string& name) {
    SCOPED_TRACE(name);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto weights = make_weights();
    weights.emplace("head.weight",
                    torch::full({4, 128}, 0.5, torch::kBFloat16));
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

TEST_F(DeepseekV41MluModelTest, EmbeddingModeLoadsWithoutLanguageModelHead) {
  args_.embedding_mode(true);
  mlu::model::DeepseekV41ForCausalLM model(make_context());
  model->load_model(make_loader(make_weights()));
  auto tokens = torch::tensor({0, 2}, options_.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(model->get_input_embeddings(tokens).cpu(),
                           torch::full({2, 128}, 0.25, torch::kBFloat16)));
}

TEST_F(DeepseekV41MluModelTest, TiedHeadLoadsFromEmbeddingWeights) {
  args_.tie_word_embeddings(true);
  mlu::model::DeepseekV41ForCausalLM model(make_context());
  model->load_model(make_loader(make_weights()));
  ASSERT_TRUE(model->has_loaded_vocab_weights());
  auto hidden = torch::ones({2, 128}, options_);
  EXPECT_TRUE(torch::equal(model->logits(hidden, torch::Tensor()).cpu(),
                           torch::full({2, 4}, 32, torch::kBFloat16)));
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
    weights.emplace("head.weight",
                    torch::full({4, 128}, 0.5, torch::kBFloat16));
    std::vector<Weights> shards(2);
    for (auto& [name, value] : weights) {
      const bool decoder = name.rfind("layers.", 0) == 0;
      shards[decoder ? 0 : 1].emplace("model." + name, std::move(value));
    }
    mlu::model::DeepseekV41ForCausalLM model(make_context());
    model->load_model(make_loader(std::move(shards)));
    ASSERT_TRUE(model->has_loaded_vocab_weights());
    auto hidden = torch::ones({2, 128}, options_);
    EXPECT_TRUE(torch::equal(model->logits(hidden, torch::Tensor()).cpu(),
                             torch::full({2, 4}, 64, torch::kBFloat16)));
  }
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

}  // namespace
}  // namespace xllm
