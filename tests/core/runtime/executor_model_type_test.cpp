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

#include <string>
#include <typeinfo>

#include "core/framework/config/execution_config.h"
#include "core/framework/model/causal_vlm.h"
#include "core/runtime/executor_impl_factory.h"

namespace xllm {
namespace {

template <typename Base>
class EchoModel : public Base {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& /*params*/) override {
    return ModelOutput(tokens + positions);
  }
  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& /*indices*/) override {
    return hidden;
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_ = torch::TensorOptions().device(torch::kCPU);
};

class EchoVlm final : public EchoModel<CausalVLM> {
 public:
  MMDict encode(const ModelInputParams& /*params*/) override { return {}; }
  torch::Tensor get_input_embeddings(
      const torch::Tensor& tokens,
      const ModelInputParams& /*params*/) override {
    return tokens;
  }
};

class ExecutorModelTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_graph_ = ExecutionConfig::get_instance().enable_graph();
    ExecutionConfig::get_instance().enable_graph(false);
    options_.backend("llm").enable_graph(false);
  }
  void TearDown() override {
    ExecutionConfig::get_instance().enable_graph(previous_graph_);
  }
  runtime::Options options_;
  ModelArgs args_;
  torch::Device device_{torch::kCPU};

 private:
  bool previous_graph_ = false;
};

TEST_F(ExecutorModelTypeTest, LlmAndRecCreateBaseExecutorWithoutGraph) {
  EchoModel<CausalLM> model;
  for (const std::string backend : {"llm", "rec"}) {
    auto executor = ExecutorImplFactory::get_instance().create_executor_impl(
        &model, args_, device_, options_, backend);
    // Do not include the executor implementation headers: they register
    // creators and could mask a broken production registry during static init.
    const std::string type_name = typeid(*executor).name();
    EXPECT_NE(type_name.find("BaseExecutorImpl"), std::string::npos);
    EXPECT_EQ(type_name.find("VlmExecutorImpl"), std::string::npos);
  }
}

TEST_F(ExecutorModelTypeTest, TextModelInVlmExecutorForwardsPrefillAndDecode) {
  EchoModel<CausalLM> model;
  auto executor = ExecutorImplFactory::get_instance().create_executor_impl(
      &model, args_, device_, options_, "vlm");
  std::vector<KVCache> caches;
  for (const auto phase :
       {BatchForwardType::PREFILL, BatchForwardType::DECODE}) {
    ModelInputParams params;
    params.meta.batch_forward_type = phase;
    const torch::Tensor tokens = torch::tensor({2, 5, 7});
    const torch::Tensor positions = torch::tensor({11, 12, 13});
    const ModelOutput result = executor->run(tokens, positions, caches, params);
    EXPECT_TRUE(
        torch::equal(result.hidden_states, torch::tensor({13, 17, 20})));
    EXPECT_FALSE(params.embedding.input_embedding.defined());
  }
}

TEST_F(ExecutorModelTypeTest, VlmRegistrationRetainsVlmDecodeBehavior) {
  EchoVlm model;
  options_.backend("vlm");
  args_.rope_scaling_rope_type("mrope");
  auto executor = ExecutorImplFactory::get_instance().create_executor_impl(
      &model, args_, device_, options_, "vlm");
  ASSERT_NE(std::string(typeid(*executor).name()).find("VlmExecutorImpl"),
            std::string::npos);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  std::vector<KVCache> caches;
  const torch::Tensor tokens = torch::tensor({2, 5});
  const torch::Tensor positions = torch::tensor({{11, 12}, {11, 12}, {11, 12}});
  const ModelOutput result = executor->run(tokens, positions, caches, params);
  EXPECT_TRUE(torch::equal(result.hidden_states, torch::tensor({13, 17})));
}

}  // namespace
}  // namespace xllm
