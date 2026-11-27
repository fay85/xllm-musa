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

#include "runtime/prefill_layerwise_eligibility.h"

#include <gtest/gtest.h>

#include <cstdlib>

#include "core/framework/config/parallel_config.h"
#include "core/util/scope_guard.h"
#include "runtime/options.h"
#include "tests/core/common/scoped_environment_variable.h"

namespace xllm {
namespace {

PrefillLayerwiseEligibility baseline_eligibility() {
  PrefillLayerwiseEligibility eligibility;
  eligibility.npu = true;
  eligibility.torch_kernel_backend = true;
  eligibility.llm_backend = true;
  eligibility.generate_task = true;
  eligibility.pd_prefill_only = true;
  eligibility.push_transfer = true;
  eligibility.kv_cache_dtype = "auto";
  eligibility.model_type = "glm_moe_dsa";
  return eligibility;
}

TEST(PrefillLayerwiseEligibilityTest, AdmitsBaselineEagerPrefill) {
  EXPECT_TRUE(baseline_eligibility().supported());
}

TEST(PrefillLayerwiseEligibilityTest, AdmitsContextParallelWithReplicatedKv) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  for (int32_t cp_size : {1, 2, 4}) {
    eligibility.cp_size = cp_size;
    eligibility.layerwise_split_size = 1;
    eligibility.kv_split_size_effective = 1;
    EXPECT_TRUE(eligibility.supported());
  }

  eligibility.cp_size = 4;
  eligibility.layerwise_split_size = 2;
  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("layerwise_split_size=1"),
            std::string::npos);

  eligibility.layerwise_split_size = 1;
  eligibility.cp_size = 0;
  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("CP >= 1"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest,
     AdmitsCompatibleKvShardsOnPythonCpPrefill) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.cp_size = 4;
  eligibility.kv_split_size_effective = 2;
  EXPECT_FALSE(eligibility.supported());

  eligibility.python_model = true;
  eligibility.replicated_index = true;
  EXPECT_TRUE(eligibility.supported());

  eligibility.replicated_index = false;
  EXPECT_FALSE(eligibility.supported());
  eligibility.replicated_index = true;
  for (int32_t cp_size : {1, 3}) {
    eligibility.cp_size = cp_size;
    EXPECT_FALSE(eligibility.supported());
  }
  eligibility.cp_size = 4;
  eligibility.kv_split_size_effective = 4;
  EXPECT_TRUE(eligibility.supported());
  eligibility.cp_size = 8;
  EXPECT_TRUE(eligibility.supported());
  eligibility.cp_size = 6;
  EXPECT_FALSE(eligibility.supported());
  eligibility.cp_size = 2;
  EXPECT_FALSE(eligibility.supported());
  eligibility.kv_split_size_effective = 0;
  EXPECT_FALSE(eligibility.supported());
}

TEST(PrefillLayerwiseEligibilityTest, FactoryReadsIndexModeForFourKvShards) {
  auto& parallel = ParallelConfig::get_instance();
  const int32_t previous_split = parallel.kv_split_size();
  const ScopeGuard restore([&] { parallel.kv_split_size(previous_split); });
  parallel.kv_split_size(4);

  runtime::Options options;
  options.cp_size(8);
  constexpr char kIndexMode[] = "XLLM_CP_INDEX_WRITE_MODE";
  const tests::ScopedEnvironmentVariable env_guard(kIndexMode);
  ASSERT_EQ(::setenv(kIndexMode, "replicated", 1), 0);
  const PrefillLayerwiseEligibility replicated =
      build_prefill_layerwise_eligibility(options, "glm_moe_dsa", "python");
  EXPECT_EQ(replicated.kv_split_size_effective, 4);
  EXPECT_TRUE(replicated.replicated_index);

  ASSERT_EQ(::setenv(kIndexMode, "sharded", 1), 0);
  EXPECT_FALSE(
      build_prefill_layerwise_eligibility(options, "glm_moe_dsa", "python")
          .replicated_index);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsAtbKernelBackend) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.torch_kernel_backend = false;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("TORCH"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest,
     AdmitsDflash2TargetWithCompatibleKvShards) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.has_speculative_decode = true;
  eligibility.supported_speculative_target = true;

  EXPECT_TRUE(eligibility.supported());

  eligibility.cp_size = 8;
  eligibility.python_model = true;
  eligibility.replicated_index = true;
  for (int32_t split_size : {2, 4}) {
    SCOPED_TRACE(split_size);
    eligibility.kv_split_size_effective = split_size;
    EXPECT_TRUE(eligibility.supported());
  }
}

TEST(PrefillLayerwiseEligibilityTest, RejectsDraftEngine) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.has_speculative_decode = true;
  eligibility.supported_speculative_target = true;
  eligibility.is_draft_engine = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("draft engine"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsOtherSpeculativeAlgorithms) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.has_speculative_decode = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("speculative decoding"),
            std::string::npos);

  eligibility.cp_size = 4;
  eligibility.kv_split_size_effective = 4;
  eligibility.python_model = true;
  eligibility.replicated_index = true;
  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("speculative decoding"),
            std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, AdmitsDecodeGraphFlagOnPurePrefill) {
  runtime::Options options;
  options.backend("llm")
      .task_type("generate")
      .npu_kernel_backend("TORCH")
      .enable_disagg_pd(true)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(true);
  PrefillLayerwiseEligibility eligibility =
      build_prefill_layerwise_eligibility(options, "glm_moe_dsa", "python");
  eligibility.npu = true;

  EXPECT_EQ(eligibility.error(), std::nullopt);

  for (const char* algorithm : {"MTP", "DFlash"}) {
    SCOPED_TRACE(algorithm);
    options.speculative_algorithm(algorithm).num_speculative_tokens(2);
    eligibility =
        build_prefill_layerwise_eligibility(options, "glm_moe_dsa", "python");
    eligibility.npu = true;
    EXPECT_EQ(eligibility.error(), std::nullopt);
  }
}

TEST(PrefillLayerwiseEligibilityTest, RejectsPiecewisePrefillGraph) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.has_speculative_decode = true;
  eligibility.supported_speculative_target = true;
  eligibility.enable_prefill_piecewise_graph = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("piecewise"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsDecodeCapablePdInstance) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.pd_prefill_only = false;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("PREFILL"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsHostTierCombination) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.host_blocks_factor = 2.0;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("Host/Store"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsEmbeddingWorker) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.generate_task = false;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("task=generate"), std::string::npos);
}

// The factory, not the value object, is the seam to the config system: a
// DFlash2 deployment sets only num_speculative_tokens, and the three-argument
// predicate must see model_type, algorithm and model_impl together.
TEST(PrefillLayerwiseEligibilityTest, FactoryTreatsSpeculativeTokensAsIntent) {
  runtime::Options options;
  options.backend("llm");
  options.task_type("generate");
  options.num_speculative_tokens(2);
  options.enable_speculative_decode(false);
  options.speculative_algorithm("DFlash2");

  const PrefillLayerwiseEligibility eligibility =
      build_prefill_layerwise_eligibility(
          options, "glm_moe_dsa", /*model_impl=*/"python");

  EXPECT_TRUE(eligibility.has_speculative_decode);
  EXPECT_TRUE(eligibility.supported_speculative_target);
  EXPECT_TRUE(eligibility.llm_backend);
  EXPECT_TRUE(eligibility.generate_task);
}

TEST(PrefillLayerwiseEligibilityTest, FactoryScopesSupportedSpeculation) {
  runtime::Options options;
  options.num_speculative_tokens(2);
  options.speculative_algorithm("DFlash2");

  EXPECT_FALSE(build_prefill_layerwise_eligibility(
                   options, "glm_moe_dsa", /*model_impl=*/"native")
                   .supported_speculative_target);
  EXPECT_FALSE(build_prefill_layerwise_eligibility(
                   options, "glm5_next", /*model_impl=*/"python")
                   .supported_speculative_target);
  options.speculative_algorithm("Suffix");
  EXPECT_FALSE(build_prefill_layerwise_eligibility(
                   options, "glm_moe_dsa", /*model_impl=*/"python")
                   .supported_speculative_target);
}

}  // namespace
}  // namespace xllm
