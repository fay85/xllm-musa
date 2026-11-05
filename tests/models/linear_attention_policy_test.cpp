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

#include "core/platform/platform.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

TEST(LinearAttentionPolicyTest, BackendConstraintAppliesBeyondGlmNext) {
  ModelArgs args;
  args.layer_types(/*layer_types=*/{"linear_attention"});
  for (const std::string& model_type :
       {"glm5_next", "qwen3_5", "qwen3_next", "unregistered_linear_model"}) {
    const LinearAttentionPolicy policy =
        ModelRegistry::get_linear_attention_policy(model_type);
    EXPECT_EQ(validate_linear_attention_kv_split(
                  args, policy, /*effective_kv_split_size=*/2)
                  .has_value(),
              Platform::is_mlu())
        << model_type;
  }
}

TEST(LinearAttentionPolicyTest, OnlyMluGlmNextRegistersCpHeadSharding) {
  const LinearAttentionPolicy policy =
      ModelRegistry::get_linear_attention_policy("glm5_next");
  EXPECT_EQ(linear_head_tp_size(policy, /*attention_tp_size=*/2, /*cp_size=*/4),
            Platform::is_mlu() ? 8 : 2);
  for (const std::string& model_type :
       {"qwen3_5", "glm5_next_mtp", "unregistered_linear_model"}) {
    const LinearAttentionPolicy other_policy =
        ModelRegistry::get_linear_attention_policy(model_type);
    EXPECT_EQ(linear_head_tp_size(
                  other_policy, /*attention_tp_size=*/2, /*cp_size=*/4),
              2)
        << model_type;
  }
}

TEST(LinearAttentionPolicyTest, ModelRegistrationKeepsBackendConstraint) {
  const std::string model_type = "linear_head_sharding_fixture";
  ModelRegistry::register_linear_head_sharding(model_type,
                                               LinearHeadSharding::CP_TP);
  const LinearAttentionPolicy policy =
      ModelRegistry::get_linear_attention_policy(model_type);
  EXPECT_EQ(linear_head_tp_size(policy, /*attention_tp_size=*/2, /*cp_size=*/4),
            8);
  EXPECT_EQ(policy.requires_unsplit_kv, Platform::is_mlu());

  const LinearAttentionPolicy unknown_policy =
      ModelRegistry::get_linear_attention_policy("other_linear_model");
  EXPECT_EQ(linear_head_tp_size(
                unknown_policy, /*attention_tp_size=*/2, /*cp_size=*/4),
            2);
}

}  // namespace
}  // namespace xllm
