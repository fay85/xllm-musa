/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <optional>
#include <string>

#include "core/distributed_runtime/master.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/util/scope_guard.h"
#include "models/cp_capability.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

TEST(NpuDcpTopologyTest, KvOwnerIsLocalToDpCohort) {
  for (int32_t rank = 0; rank < 8; ++rank) {
    ParallelArgs args(rank,
                      /*world_size=*/8,
                      /*dp_size=*/2,
                      /*cp_size=*/1,
                      /*process_group=*/nullptr,
                      /*ep_size=*/8);
    args.kv_split_size(2);
    EXPECT_EQ(args.kv_split_rank(), (rank % 4) / 2);
    args.cp_size(2);
    EXPECT_EQ(args.kv_split_rank(), (rank % 4) / 2);
  }
}

TEST(NpuCpCapabilityTest, RegisteredCpCapableModels) {
  // The models that opt into NPU model-side CP. deepseek_v32 / glm_moe_dsa
  // drive it through the ATB NpuCpPlan pipeline; deepseek_v4 owns its split
  // inside the model on the TORCH backend. Both are advertised here because
  // this is the master-side startup gate, not the worker-side sharding switch.
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32_mtp"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v4"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v4_mtp"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa_mtp"));
  // The registry must advertise MODEL for these and NONE for the rest.
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v32"),
            CpShardingMode::MODEL);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("glm_moe_dsa_mtp"),
            CpShardingMode::MODEL);
}

TEST(NpuCpCapabilityTest, UnregisteredModelsAreNotCapable) {
  // deepseek_v3_mtp uses the DeepSeekV2 decoder without the V3.2 ATB CP
  // metadata/TP contract; it must NOT be advertised as CP-capable so that
  // validate_model_cp rejects deepseek_v3_mtp + cp_size>1 at startup.
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3_mtp"));
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3"));
  // Unrelated NPU models are not CP-capable.
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3"));
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3_atb"));
  // Hybrid linear attention models are the only ones the graph executor takes
  // through spec-verify chunked prefill; none of them is CP-capable, which is
  // what keeps that capture path CP-free.
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3_next"));
  // Unknown model names default to NONE.
  EXPECT_FALSE(is_npu_model_cp_capable("definitely_not_a_model"));
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v3_mtp"),
            CpShardingMode::NONE);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("definitely_not_a_model"),
            CpShardingMode::NONE);
}

TEST(NpuCpCapabilityTest, PythonQwen3VlAdmitsFullPrefillAndRejectsUnsafeModes) {
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  KVCacheConfig& cache_config = KVCacheConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  const bool original_chunked_prefill =
      scheduler_config.enable_chunked_prefill();
  const bool original_prefix_cache = cache_config.enable_prefix_cache();
  ScopeGuard config_guard([&] {
    model_config.model_impl(original_model_impl);
    parallel_config.kv_split_size(original_kv_split_size);
    scheduler_config.enable_chunked_prefill(original_chunked_prefill);
    cache_config.enable_prefix_cache(original_prefix_cache);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);
  scheduler_config.enable_chunked_prefill(false);
  cache_config.enable_prefix_cache(false);

  Options options;
  options.task_type("generate")
      .cp_size(2)
      .dp_size(1)
      .enable_chunked_prefill(false)
      .enable_prefix_cache(false)
      .enable_schedule_overlap(false)
      .enable_graph(false);
  const auto validate = [&](const Options& candidate, EngineType engine_type) {
    return validate_model_cp(candidate,
                             engine_type,
                             "qwen3_vl",
                             /*global_world_size=*/4);
  };
  // CP2 x TP2 and CP4 x TP1 both keep the full prefill and decode contract.
  EXPECT_FALSE(validate(options, EngineType::VLM).has_value());
  options.cp_size(4);
  EXPECT_FALSE(validate(options, EngineType::VLM).has_value());
  options.cp_size(2);

  EXPECT_TRUE(validate(options, EngineType::VLMSSM).has_value());
  EXPECT_TRUE(validate(options, EngineType::LLM).has_value());
  Options unsupported = options;
  unsupported.enable_chunked_prefill(true);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  unsupported = options;
  unsupported.priority_strategy("multi_slo_and_prio");
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  unsupported = options;
  unsupported.enable_prefix_cache(true);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  // Global configs feed Python metadata too; refuse a conflicting effective
  // configuration even when the engine options alone appear safe.
  scheduler_config.enable_chunked_prefill(true);
  EXPECT_TRUE(validate(options, EngineType::VLM).has_value());
  scheduler_config.enable_chunked_prefill(false);
  cache_config.enable_prefix_cache(true);
  EXPECT_TRUE(validate(options, EngineType::VLM).has_value());
  cache_config.enable_prefix_cache(false);
  unsupported = options;
  unsupported.enable_schedule_overlap(true);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  unsupported = options;
  unsupported.enable_graph(true);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  unsupported = options;
  unsupported.enable_disagg_pd(true).instance_role(InstanceRole::PREFILL);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  unsupported = options;
  unsupported.num_speculative_tokens(1);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  unsupported = options;
  unsupported.dp_size(2);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  unsupported = options;
  unsupported.cp_size(3);
  EXPECT_TRUE(validate(unsupported, EngineType::VLM).has_value());
  parallel_config.kv_split_size(2);
  EXPECT_TRUE(validate(options, EngineType::VLM).has_value());
  parallel_config.kv_split_size(1);
  model_config.model_impl("native");
  EXPECT_TRUE(validate(options, EngineType::VLM).has_value());
  model_config.model_impl("python");
  EXPECT_TRUE(validate_model_cp(options,
                                EngineType::VLM,
                                "qwen2_5_vl",
                                /*global_world_size=*/4)
                  .has_value());
  options.cp_size(1).enable_chunked_prefill(true).enable_prefix_cache(true);
  EXPECT_FALSE(validate(options, EngineType::VLM).has_value());
}

TEST(NpuCpCapabilityTest, PythonCpAllowsGraphEnabledAndDisabled) {
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  options.enable_graph(false);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
}

TEST(NpuCpCapabilityTest, PythonCpPreservesQwenAndRestrictsGlm) {
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(false)
      .speculative_algorithm("MTP");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python model-side CP does not support MTP speculative "
                "verification; run MTP on a cp_size=1 Decode instance"));

  options.speculative_algorithm("DSpark");
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Current model-side CP does not support aux-hidden-capture "
                "speculative algorithms (Eagle3/DFlash/DSpark); run "
                "speculative decoding on a cp_size=1 Decode instance."));

  // glm_moe_dsa + DFlash2 is admitted for the python impl: glm5_2.py
  // restores the aux-hidden buffer to global row order under CP.
  options.speculative_algorithm("DFlash2");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
  options.speculative_algorithm("dflash2");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
  // The native/ATB impl keeps the rejection: it has no aux capture.
  model_config.model_impl("atb");
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Current model-side CP does not support aux-hidden-capture "
                "speculative algorithms (Eagle3/DFlash/DSpark); run "
                "speculative decoding on a cp_size=1 Decode instance."));
  model_config.model_impl("python");
  options.speculative_algorithm("DSpark");
  // The exemption is DFlash2-only: with the impl restored to python, the
  // non-DFlash2 aux-capture algorithm is still rejected.
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Current model-side CP does not support aux-hidden-capture "
                "speculative algorithms (Eagle3/DFlash/DSpark); run "
                "speculative decoding on a cp_size=1 Decode instance."));

  options.cp_size(1).instance_role(InstanceRole::DECODE).enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
  options.cp_size(4)
      .instance_role(InstanceRole::PREFILL)
      .enable_disagg_pd(false)
      .speculative_algorithm("MTP");

  // EP topology validation belongs to model construction, not this Python CP
  // capability gate.
  options.ep_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  parallel_config.kv_split_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python GLM CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));

  options.enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python GLM CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));
  options.instance_role(InstanceRole::PREFILL);

  parallel_config.kv_split_size(1);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa_mtp",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python model-side CP does not support "
                "model_type=glm_moe_dsa_mtp; supported models are qwen3, "
                "qwen3_vl, glm_moe_dsa, and glm5_next."));

  parallel_config.kv_split_size(3);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python CP requires kv_split_size effective value to be a "
                "positive divisor of cp_size"));
}

TEST(NpuCpCapabilityTest, PythonGlm5NextCapabilityGate) {
  KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  const bool original_chunked_prefill =
      scheduler_config.enable_chunked_prefill();
  const bool original_mix_batch = scheduler_config.enable_mix_batch();
  const bool original_prefix_cache = kv_cache_config.enable_prefix_cache();
  const bool original_schedule_overlap =
      scheduler_config.enable_schedule_overlap();
  ScopeGuard config_guard([&] {
    scheduler_config.enable_schedule_overlap(original_schedule_overlap);
    kv_cache_config.enable_prefix_cache(original_prefix_cache);
    scheduler_config.enable_mix_batch(original_mix_batch);
    scheduler_config.enable_chunked_prefill(original_chunked_prefill);
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);
  scheduler_config.enable_chunked_prefill(false);
  scheduler_config.enable_mix_batch(false);
  kv_cache_config.enable_prefix_cache(false);
  scheduler_config.enable_schedule_overlap(false);

  Options options;
  options.task_type("generate")
      .cp_size(2)
      .dp_size(1)
      .ep_size(1)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(false);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.instance_role(InstanceRole::PREFILL);

  options.enable_graph(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP does not support this target-side "
                "speculative algorithm; run speculation on a cp_size=1 "
                "Decode instance"));

  options.speculative_algorithm("DFlash2");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.speculative_algorithm("dflash2");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  options.speculative_algorithm("Eagle3");
  EXPECT_EQ(
      validate_model_cp(options,
                        EngineType::SSM,
                        "glm5_next",
                        /*global_world_size=*/8),
      std::optional<std::string>(
          "Current model-side CP does not support aux-hidden-capture "
          "speculative algorithms (Eagle3/DFlash/DSpark); run speculative "
          "decoding on a cp_size=1 Decode instance."));

  options.speculative_algorithm("MTP");

  options.enable_graph(false);
  options.ep_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.dp_size(2).ep_size(8);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.dp_size(0);
  EXPECT_TRUE(validate_model_cp(options,
                                EngineType::LLM,
                                "glm5_next",
                                /*global_world_size=*/8)
                  .has_value());
  options.dp_size(uint32_t{1} << 31);
  EXPECT_TRUE(validate_model_cp(options,
                                EngineType::LLM,
                                "glm5_next",
                                /*global_world_size=*/8)
                  .has_value());
  options.dp_size(2);
  for (uint32_t ep_size : {0u, 3u}) {
    options.ep_size(ep_size);
    EXPECT_TRUE(validate_model_cp(options,
                                  EngineType::LLM,
                                  "glm5_next",
                                  /*global_world_size=*/8)
                    .has_value());
  }
  options.dp_size(1);
  options.ep_size(1);

  options.expert_parallel_degree(2).ep_size(8);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.ep_size(4);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next EPLv2 with PCP requires ep_size equal to "
                "world_size"));
  options.expert_parallel_degree(1).ep_size(1);

  options.cp_size(4);
  parallel_config.kv_split_size(2);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));
  options.enable_disagg_pd(true);
  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));
  options.instance_role(InstanceRole::PREFILL);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  parallel_config.kv_split_size(4);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  parallel_config.kv_split_size(1);
  options.enable_disagg_pd(false);
  options.cp_size(2);

  scheduler_config.enable_chunked_prefill(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  scheduler_config.enable_chunked_prefill(false);

  scheduler_config.enable_mix_batch(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  scheduler_config.enable_mix_batch(false);

  kv_cache_config.enable_prefix_cache(true);
  EXPECT_EQ(
      validate_model_cp(options,
                        EngineType::LLM,
                        "glm5_next",
                        /*global_world_size=*/8),
      std::optional<std::string>(
          "Python GLM-5 Next CP with enable_prefix_cache=true requires "
          "disaggregated PD with the PREFILL role; a standalone CP instance "
          "must set enable_prefix_cache=false"));
  kv_cache_config.enable_prefix_cache(false);

  scheduler_config.enable_schedule_overlap(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  scheduler_config.enable_mix_batch(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  scheduler_config.enable_mix_batch(false);

  kv_cache_config.enable_prefix_cache(true);
  EXPECT_EQ(
      validate_model_cp(options,
                        EngineType::LLM,
                        "glm5_next",
                        /*global_world_size=*/8),
      std::optional<std::string>(
          "Python GLM-5 Next CP with enable_prefix_cache=true requires "
          "disaggregated PD with the PREFILL role; a standalone CP instance "
          "must set enable_prefix_cache=false"));
  kv_cache_config.enable_prefix_cache(false);

  options.enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  kv_cache_config.enable_prefix_cache(true);
  scheduler_config.enable_chunked_prefill(true);
  options.cp_size(4);
  parallel_config.kv_split_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.cp_size(2);
  parallel_config.kv_split_size(1);
  scheduler_config.enable_chunked_prefill(false);
  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with enable_prefix_cache=true requires "
                "disaggregated PD with the PREFILL role; a standalone CP "
                "instance must set enable_prefix_cache=false"));
  kv_cache_config.enable_prefix_cache(false);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with disaggregated PD requires the "
                "PREFILL role"));
}

TEST(NpuCpCapabilityTest, PythonGlm5NextShardedKvIsAdmittedAtCpSizeOne) {
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(2);

  Options options;
  options.task_type("generate");
  options.cp_size(1);
  options.dp_size(1);
  options.ep_size(16);
  options.enable_disagg_pd(true);
  options.instance_role(InstanceRole::DECODE);

  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/16)
                   .has_value());

  EXPECT_FALSE(validate_qwen_dcp_topology(
                   /*global_world_size=*/16,
                   /*dp_size=*/1,
                   /*kv_split_size=*/2)
                   .has_value())
      << "kv_split_size=2 must divide the TP size (16) within the DP replica";
}

TEST(NpuCpCapabilityTest, PythonDcpIsAdmittedAtDpGreaterThanOneAndCpSizeOne) {
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(2);

  Options options;
  options.task_type("generate");
  options.cp_size(1);
  options.dp_size(2);
  options.ep_size(2);
  options.enable_disagg_pd(true);
  options.instance_role(InstanceRole::DECODE);

  const std::optional<std::string> dp_error =
      validate_model_cp(options,
                        EngineType::LLM,
                        "glm5_next",
                        /*global_world_size=*/4);
  EXPECT_FALSE(dp_error.has_value())
      << "dp_size > 1 with kv_split_size > 1 at cp_size == 1 must be "
         "admissible, got: "
      << dp_error.value_or("");

  EXPECT_FALSE(validate_qwen_dcp_topology(
                   /*global_world_size=*/4,
                   /*dp_size=*/2,
                   /*kv_split_size=*/2)
                   .has_value());
  const std::optional<std::string> topology_error = validate_qwen_dcp_topology(
      /*global_world_size=*/4,
      /*dp_size=*/2,
      /*kv_split_size=*/4);
  ASSERT_TRUE(topology_error.has_value())
      << "kv_split_size=4 exceeds the TP size (2) and must be refused";
  EXPECT_EQ(topology_error.value(),
            "Qwen DCP kv_split_size must divide the TP size within each DP "
            "replica");
}

TEST(NpuDcpTopologyTest, AcceptsOnlyTpLocalFactors) {
  const auto is_valid = [](int32_t kv_split_size) {
    return !validate_qwen_dcp_topology(/*global_world_size=*/8,
                                       /*dp_size=*/2,
                                       kv_split_size)
                .has_value();
  };

  EXPECT_TRUE(is_valid(2));
  EXPECT_TRUE(is_valid(4));
  EXPECT_FALSE(is_valid(3));
  EXPECT_FALSE(is_valid(8));
}

}  // namespace
}  // namespace xllm
