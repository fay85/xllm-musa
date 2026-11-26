/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include "common/options.h"
#include "distributed_runtime/master.h"
#include "framework/config/parallel_config.h"
#include "models/cp_capability.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

Options make_cp_options(int32_t world_size) {
  Options options;
  options.cp_size(2)
      .dp_size(1)
      .ep_size(world_size)
      .task_type("generate")
      .instance_role(InstanceRole::PREFILL)
      .speculative_algorithm("Suffix");
  ParallelConfig::get_instance().kv_split_size(1);
  return options;
}

TEST(MluCpCapabilityTest, RegistersSupportedModels) {
  EXPECT_TRUE(is_mlu_model_cp_capable("deepseek_v4"));
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v4"),
            CpShardingMode::MODEL);
  EXPECT_FALSE(is_mlu_model_cp_capable("deepseek_v32"));
  EXPECT_TRUE(is_mlu_model_cp_capable("glm_moe_dsa"));
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v32"),
            CpShardingMode::NONE);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("glm_moe_dsa"),
            CpShardingMode::MODEL);
}

TEST(MluCpCapabilityTest, RejectsUnsupportedModels) {
  EXPECT_FALSE(is_mlu_model_cp_capable("deepseek_v4_mtp"));
  EXPECT_FALSE(is_mlu_model_cp_capable("deepseek_v3"));
  EXPECT_FALSE(is_mlu_model_cp_capable("qwen3"));
  EXPECT_FALSE(is_mlu_model_cp_capable("definitely_not_a_model"));
}

TEST(MluCpCapabilityTest, SharedRegistryDoesNotLeakBackendCapability) {
  ModelRegistry::register_cp_sharding_mode("npu_only_fixture",
                                           CpShardingMode::MODEL);
  EXPECT_FALSE(is_mlu_model_cp_capable("npu_only_fixture"));
}

TEST(MluCpCapabilityTest, AcceptsOrthogonalDeepseekV4TargetAndSuffix) {
  constexpr int32_t kWorldSize = 8;
  const Options options = make_cp_options(kWorldSize);

  EXPECT_FALSE(
      validate_model_cp(options, EngineType::LLM, "deepseek_v4", kWorldSize)
          .has_value());
  EXPECT_FALSE(
      validate_model_cp(options, EngineType::SSM, "deepseek_v4", kWorldSize)
          .has_value());
}

TEST(MluCpCapabilityTest, ValidatesDeepseekV41VlmCp) {
  constexpr int32_t kWorldSize = 8;
  Options options = make_cp_options(kWorldSize);
  options.backend("vlm")
      .model_impl("native")
      .cp_size(4)
      .instance_role(InstanceRole::DEFAULT)
      .speculative_algorithm("DSpark");
  for (const EngineType type : {EngineType::VLM, EngineType::VLMSSM}) {
    SCOPED_TRACE(type.to_string());
    EXPECT_FALSE(validate_model_cp(options, type, "deepseek_v41", kWorldSize)
                     .has_value());
    EXPECT_TRUE(
        validate_model_cp(options, type, "qwen3_vl", kWorldSize).has_value());
  }
  options.ep_size(4);
  EXPECT_TRUE(
      validate_model_cp(options, EngineType::VLM, "deepseek_v41", kWorldSize)
          .has_value());
}

TEST(MluCpCapabilityTest, AcceptsGlm5NextVlmCp) {
  constexpr int32_t kWorldSize = 8;
  Options options = make_cp_options(kWorldSize);
  options.backend("vlm")
      .model_impl("native")
      .instance_role(InstanceRole::DEFAULT)
      .num_speculative_tokens(0);
  EXPECT_FALSE(
      validate_model_cp(options, EngineType::VLM, "glm5_next", kWorldSize)
          .has_value());
  options.instance_role(InstanceRole::PREFILL).enable_disagg_pd(true);
  EXPECT_FALSE(
      validate_model_cp(options, EngineType::VLM, "glm5_next", kWorldSize)
          .has_value());
}

TEST(MluCpCapabilityTest, AcceptsGlm5NextVlmSpeculativeCp) {
  constexpr int32_t kWorldSize = 8;
  Options options = make_cp_options(kWorldSize);
  options.backend("vlm");
  options.speculative_algorithm("MTP").num_speculative_tokens(2);
  EXPECT_FALSE(
      validate_model_cp(options, EngineType::VLMSSM, "glm5_next", kWorldSize)
          .has_value());
  options.speculative_algorithm("DFlash2").num_speculative_tokens(7);
  EXPECT_FALSE(
      validate_model_cp(options, EngineType::VLMSSM, "glm5_next", kWorldSize)
          .has_value());
  options.speculative_algorithm("DSpark");
  EXPECT_FALSE(
      validate_model_cp(options, EngineType::VLMSSM, "glm5_next", kWorldSize)
          .has_value());
}

TEST(MluCpCapabilityTest, RequiresPushForDisaggregatedGlm5NextVlmMtp) {
  Options options;
  options.speculative_algorithm("MTP")
      .enable_disagg_pd(true)
      .kv_cache_transfer_mode("PULL");
  EXPECT_TRUE(validate_vlm_speculative_pd_transfer_mode(options, "glm5_next")
                  .has_value());
  options.kv_cache_transfer_mode("PUSH");
  EXPECT_FALSE(validate_vlm_speculative_pd_transfer_mode(options, "glm5_next")
                   .has_value());
  options.enable_disagg_pd(false)
      .enable_service_routing(true)
      .kv_cache_transfer_mode("PULL");
  EXPECT_FALSE(validate_vlm_speculative_pd_transfer_mode(options, "glm5_next")
                   .has_value());
}

TEST(MluCpCapabilityTest, ValidatesQwenVlmSpeculativePdTransferMode) {
  Options options;
  options.enable_disagg_pd(true);
  for (const char* algorithm : {"MTP", "DFlash2"}) {
    SCOPED_TRACE(algorithm);
    options.speculative_algorithm(algorithm).kv_cache_transfer_mode("PULL");
    EXPECT_TRUE(validate_vlm_speculative_pd_transfer_mode(options, "qwen3_5")
                    .has_value());
    options.kv_cache_transfer_mode("PUSH");
    EXPECT_FALSE(validate_vlm_speculative_pd_transfer_mode(options, "qwen3_5")
                     .has_value());
  }
  options.speculative_algorithm("Eagle3").kv_cache_transfer_mode("PULL");
  EXPECT_FALSE(validate_vlm_speculative_pd_transfer_mode(options, "qwen3_vl")
                   .has_value());
}

TEST(MluCpCapabilityTest, SupportsGlmPcpDcpAndRejectsLegacyDraftTargets) {
  constexpr int32_t kWorldSize = 8;
  const Options options = make_cp_options(kWorldSize);
  Options glm_options = options;
  glm_options.enable_schedule_overlap(false);

  EXPECT_TRUE(
      validate_model_cp(options, EngineType::LLM, "deepseek_v32", kWorldSize)
          .has_value());
  EXPECT_FALSE(
      validate_model_cp(glm_options, EngineType::LLM, "glm_moe_dsa", kWorldSize)
          .has_value());

  ParallelConfig::get_instance().kv_split_size(kWorldSize);
  EXPECT_FALSE(
      validate_model_cp(glm_options, EngineType::LLM, "glm_moe_dsa", kWorldSize)
          .has_value());
  ParallelConfig::get_instance().kv_split_size(2);
  EXPECT_FALSE(
      validate_model_cp(glm_options, EngineType::LLM, "glm_moe_dsa", kWorldSize)
          .has_value());
  ParallelConfig::get_instance().kv_split_size(4);
  EXPECT_TRUE(
      validate_model_cp(glm_options, EngineType::LLM, "glm_moe_dsa", kWorldSize)
          .has_value());
  ParallelConfig::get_instance().kv_split_size(1);

  EXPECT_TRUE(
      validate_model_cp(options, EngineType::LLM, "deepseek_v4_mtp", kWorldSize)
          .has_value());
}

TEST(MluCpCapabilityTest, RejectsUnsupportedTopologyAndAcceptsMtp) {
  constexpr int32_t kWorldSize = 8;

  Options invalid_world = make_cp_options(kWorldSize);
  EXPECT_TRUE(validate_model_cp(invalid_world,
                                EngineType::LLM,
                                "deepseek_v4",
                                /*global_world_size=*/7)
                  .has_value());

  Options invalid_ep = make_cp_options(kWorldSize);
  invalid_ep.ep_size(4);
  EXPECT_TRUE(
      validate_model_cp(invalid_ep, EngineType::LLM, "deepseek_v4", kWorldSize)
          .has_value());

  Options mtp_options = make_cp_options(kWorldSize);
  mtp_options.speculative_algorithm("MTP");
  EXPECT_FALSE(
      validate_model_cp(mtp_options, EngineType::SSM, "glm_moe_dsa", kWorldSize)
          .has_value());

  Options invalid_kv_split = make_cp_options(kWorldSize);
  ParallelConfig::get_instance().kv_split_size(2);
  EXPECT_TRUE(validate_model_cp(
                  invalid_kv_split, EngineType::LLM, "deepseek_v4", kWorldSize)
                  .has_value());
  ParallelConfig::get_instance().kv_split_size(1);
}

}  // namespace
}  // namespace xllm
