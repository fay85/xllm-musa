/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <gtest/gtest.h>

#include "core/framework/config/execution_config.h"
#include "core/runtime/vlm_executor_impl.h"

namespace xllm {
namespace {

class RecordingLanguageModel final : public CausalLM {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*caches*/,
                      const ModelInputParams& /*params*/) override {
    positions_ = positions.clone();
    ++calls_;
    return ModelOutput(tokens.to(torch::kFloat32));
  }
  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& /*indices*/) override {
    return hidden;
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  void prepare_expert_weight(int32_t /*layer_id*/,
                             const std::vector<int32_t>& /*ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  const torch::TensorOptions& options() const override { return options_; }
  const torch::Tensor& positions() const { return positions_; }
  int32_t calls() const { return calls_; }

 private:
  torch::TensorOptions options_ = torch::TensorOptions().device(torch::kCPU);
  torch::Tensor positions_;
  int32_t calls_ = 0;
};

class VlmExecutorEagerDraftTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_graph_ = ExecutionConfig::get_instance().enable_graph();
    ExecutionConfig::get_instance().enable_graph(false);
    args_.rope_scaling_rope_type("mrope");
    options_.backend("vlm");
    options_.enable_graph(false);
  }
  void TearDown() override {
    ExecutionConfig::get_instance().enable_graph(previous_graph_);
  }
  bool previous_graph_ = false;
  RecordingLanguageModel model_;
  ModelArgs args_;
  runtime::Options options_;
};

TEST_F(VlmExecutorEagerDraftTest, RoutesDecodeThroughLanguageModel) {
  VlmExecutorImpl executor(
      &model_, args_, torch::Device(torch::kCPU), options_);
  auto tokens = torch::tensor({1, 2}, torch::kInt64);
  auto positions = torch::tensor({{3, 4}, {3, 4}, {3, 4}}, torch::kInt64);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  std::vector<KVCache> caches;
  const auto output = executor.run(tokens, positions, caches, params);
  EXPECT_EQ(model_.calls(), 1);
  EXPECT_TRUE(torch::equal(model_.positions(), positions[0]));
  EXPECT_TRUE(torch::equal(output.hidden_states, tokens.to(torch::kFloat32)));
}

TEST_F(VlmExecutorEagerDraftTest, PreservesPrefillMropeRows) {
  VlmExecutorImpl executor(
      &model_, args_, torch::Device(torch::kCPU), options_);
  auto tokens = torch::tensor({1, 2}, torch::kInt64);
  auto positions = torch::tensor({{3, 4}, {5, 6}, {7, 8}}, torch::kInt64);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  std::vector<KVCache> caches;
  executor.run(tokens, positions, caches, params);
  EXPECT_EQ(model_.calls(), 1);
  EXPECT_TRUE(torch::equal(model_.positions(), positions));
}

TEST_F(VlmExecutorEagerDraftTest, RejectsImageEncodingOnLanguageOnlyDraft) {
  VlmExecutorImpl executor(
      &model_, args_, torch::Device(torch::kCPU), options_);
  ModelInputParams params;
  EXPECT_DEATH(executor.encode(params),
               "language-only model cannot encode images");
}

}  // namespace
}  // namespace xllm
