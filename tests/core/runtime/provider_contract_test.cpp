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

#include "core/runtime/provider_contract.h"

#include <gtest/gtest.h>

#include "core/common/types.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

class DescriptorEngine final : public Engine {
 public:
  explicit DescriptorEngine(const std::string& model_type,
                            int32_t compress_ratio = 1) {
    args_.model_type(model_type)
        .dtype("bfloat16")
        .n_layers(2)
        .n_heads(8)
        .n_kv_heads(8)
        .head_dim(64)
        .compress_ratios({compress_ratio, compress_ratio});
    tokenizer_args_.chat_template("{{ messages }}");
  }

  ForwardOutput step(std::vector<Batch>&) override { return {}; }
  void update_last_step_result(std::vector<Batch>&) override {}
  std::vector<int64_t> get_active_activation_memory() const override {
    return {};
  }
};

TEST(ProviderContractTest, RegistersActualCacheGroupsForFlatAndGroupedPools) {
  auto& model = ModelConfig::get_instance();
  auto& parallel = ParallelConfig::get_instance();
  auto& disagg = DisaggPDConfig::get_instance();
  const ModelConfig saved_model = model;
  const ParallelConfig saved_parallel = parallel;
  const DisaggPDConfig saved_disagg = disagg;
  const ScopeGuard restore_config([&] {
    model = saved_model;
    parallel = saved_parallel;
    disagg = saved_disagg;
  });
  model.backend("llm").model_id("test-model");
  parallel.tp_size(1).dp_size(1).cp_size(1).ep_size(1).kv_split_size(1);
  disagg.enable_disagg_pd(true).kv_cache_transfer_mode("PUSH");

  InstanceInfo instance;
  instance.name = "127.0.0.1:7442";
  instance.incarnation_id = "test-incarnation";
  instance.type = "PREFILL";
  instance.cluster_ids = {1};

  const std::vector<std::string> model_types = {
      "", "deepseek_v4", "deepseek_v41"};
  const std::vector<std::vector<std::string>> expected = {
      {"group:0"}, {"group:0", "group:1"}, {"group:0", "group:1"}};
  for (size_t index = 0; index < model_types.size(); ++index) {
    SCOPED_TRACE(index);
    const DescriptorEngine engine(model_types[index]);
    BlockManagerPool::Options options;
    options.num_blocks(4)
        .swa_block_size(128)
        .enable_prefix_cache(false)
        .sliding_window_size(128)
        .swa_blocks_per_seq(2)
        .swa_num_blocks(8)
        .block_size(128)
        .max_seqs_per_batch(1)
        .max_tokens_per_batch(128);
    if (!model_types[index].empty()) {
      options.manager_types({1, 0});
    }
    const BlockManagerPool pool(options);
    std::string error;
    const auto config = collect_native_provider_descriptor_config(
        instance, engine, pool, &error);
    EXPECT_TRUE(config.has_value()) << error;
    if (!config.has_value()) {
      continue;
    }
    EXPECT_EQ(config->cache_groups, expected[index]);
    const auto descriptor = build_native_provider_descriptor(*config, &error);
    ASSERT_TRUE(descriptor.has_value()) << error;
    EXPECT_EQ(descriptor->contract_version(), 1u);
    EXPECT_EQ(descriptor->identity().protocol_version(), 1u);
    EXPECT_FALSE(config->kv_layout_contract.empty());
    EXPECT_TRUE(config->kv_transfer_layout_contract.empty());
    EXPECT_EQ(descriptor->kv().kv_layout_digest().size(), 64u);
    EXPECT_TRUE(descriptor->kv().transfer_compatibility_digest().empty());

    const DescriptorEngine different_layout(model_types[index],
                                            /*compress_ratio=*/2);
    const auto different_config = collect_native_provider_descriptor_config(
        instance, different_layout, pool, &error);
    ASSERT_TRUE(different_config.has_value()) << error;
    EXPECT_EQ(different_config->cache_groups, config->cache_groups);
    EXPECT_NE(different_config->kv_layout_contract, config->kv_layout_contract);
    EXPECT_TRUE(different_config->kv_transfer_layout_contract.empty());
    const auto different_descriptor =
        build_native_provider_descriptor(*different_config, &error);
    ASSERT_TRUE(different_descriptor.has_value()) << error;
    EXPECT_NE(different_descriptor->kv().kv_layout_digest(),
              descriptor->kv().kv_layout_digest());
    EXPECT_TRUE(
        different_descriptor->kv().transfer_compatibility_digest().empty());
  }
}

TEST(ProviderContractTest, TaskOnlyDescriptorPublishesStaticNonKvState) {
  ProviderDescriptorConfig config{
      .task_only = true,
      .engine_uid = "127.0.0.1:8010",
      .instance_pool_id = "image-pool",
      .api_address = "127.0.0.1:8010",
      .role = proto::ENGINE_ROLE_AGGREGATED,
      .runtime_version = "test-runtime",
      .hardware_runtime_version = "test-hardware",
      .model_revision = "dit-model",
      .tokenizer_revision = "dit-model",
      .chat_template_contract = "dit-renderer",
      .renderer_contract = "dit-renderer",
      .quantization = "model-dtype=bfloat16",
      .soc = "test-soc",
      .device_count = 4,
      .tp = 1,
      .dp = 1,
      .pp = 1,
      .ep = 1,
      .cp = 1,
      .kv_layout_contract = "task-no-kv-v1",
      .kv_cache_dtype = "none",
      .block_size = 1,
      .hash_version = 0,
      .hash_seed = 0,
      .cache_isolation_domain = "",
      .cache_groups = {},
      .head_shard_mapping_contract = "task-no-kv-v1",
      .connector = "none",
      .connector_version = "builtin-test-runtime",
      .scheduler_class = "dit_one_shot",
      .max_num_seqs = 1,
      .max_num_batched_tokens = 1,
      .scheduler_policy_contract = "dit-one-shot-v1",
      .api_features = {"image_generation"},
  };
  std::string error;
  const auto descriptor = build_native_provider_descriptor(config, &error);
  ASSERT_TRUE(descriptor.has_value()) << error;
  EXPECT_TRUE(descriptor->task_only());
  EXPECT_EQ(descriptor->endpoint().data_transport(), "brpc");
  EXPECT_TRUE(descriptor->has_kv());
  EXPECT_EQ(descriptor->kv().cache_dtype(), "none");
  EXPECT_EQ(descriptor->kv().transfer_modes_size(), 1);
  EXPECT_EQ(descriptor->kv().transfer_modes(0), proto::TRANSFER_MODE_NONE);
  EXPECT_EQ(descriptor->capabilities_size(), 1);

  EngineStateSnapshot snapshot;
  EngineStateBuilder builder(*descriptor);
  const auto state = builder.build(snapshot, &error);
  ASSERT_TRUE(state.has_value()) << error;
  EXPECT_EQ(state->per_dp_size(), 0);
}

}  // namespace
}  // namespace xllm
