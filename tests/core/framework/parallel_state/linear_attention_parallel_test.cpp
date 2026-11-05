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

#include "core/framework/parallel_state/linear_attention_parallel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include "core/framework/parallel_state/parallel_args.h"

namespace xllm {
namespace {

TEST(LinearAttentionParallelTest, AcceptsUnsplitLinearStateWithAndWithoutCp) {
  LinearAttentionPolicy policy;
  policy.requires_unsplit_kv = true;
  ModelArgs args;
  args.layer_types(/*layer_types=*/{"linear_attention", "full_attention"});
  for (const int32_t cp_size : {1, 2, 4}) {
    ParallelArgs parallel(/*rank=*/0,
                          /*world_size=*/8,
                          /*dp_size=*/1,
                          cp_size,
                          /*process_group=*/nullptr,
                          /*ep_size=*/8);
    parallel.kv_split_size(/*kv_split_size=*/1);
    EXPECT_FALSE(validate_linear_attention_kv_split(
                     args, policy, parallel.kv_split_size_effective())
                     .has_value())
        << "cp_size=" << cp_size;
  }
}

TEST(LinearAttentionParallelTest, DoesNotRestrictFullAttentionKvSplit) {
  LinearAttentionPolicy policy;
  policy.requires_unsplit_kv = true;
  ModelArgs args;
  args.layer_types(/*layer_types=*/
                   {"full_attention",
                    "attention",
                    "deepseek_sparse_attention"});
  EXPECT_FALSE(validate_linear_attention_kv_split(
                   args, policy, /*effective_kv_split_size=*/4)
                   .has_value());
}

TEST(LinearAttentionParallelTest, RejectsSplitWhenPolicyRequiresUnsplitState) {
  LinearAttentionPolicy policy;
  policy.requires_unsplit_kv = true;
  ModelArgs args;
  args.layer_types(
      /*layer_types=*/{"linear_attention", "deepseek_sparse_attention"});
  for (const int32_t cp_size : {1, 2}) {
    ParallelArgs parallel(/*rank=*/0,
                          /*world_size=*/8,
                          /*dp_size=*/1,
                          cp_size,
                          /*process_group=*/nullptr,
                          /*ep_size=*/8);
    parallel.kv_split_size(/*kv_split_size=*/2);
    const std::optional<std::string> error = validate_linear_attention_kv_split(
        args, policy, parallel.kv_split_size_effective());
    ASSERT_TRUE(error.has_value()) << "cp_size=" << cp_size;
    EXPECT_NE(error->find("Linear attention"), std::string::npos);
    EXPECT_NE(error->find("--kv_split_size=1"), std::string::npos);
  }
}

TEST(LinearAttentionParallelTest, ValidatesZeroKvSplitAfterResolvingCp) {
  LinearAttentionPolicy policy;
  policy.requires_unsplit_kv = true;
  ModelArgs args;
  args.layer_types(/*layer_types=*/{"linear_attention"});
  ParallelArgs parallel(/*rank=*/0,
                        /*world_size=*/8,
                        /*dp_size=*/1,
                        /*cp_size=*/1,
                        /*process_group=*/nullptr,
                        /*ep_size=*/8);
  parallel.kv_split_size(/*kv_split_size=*/0);
  EXPECT_FALSE(validate_linear_attention_kv_split(
                   args, policy, parallel.kv_split_size_effective())
                   .has_value());
  parallel.cp_size(/*cp_size=*/2);
  EXPECT_TRUE(validate_linear_attention_kv_split(
                  args, policy, parallel.kv_split_size_effective())
                  .has_value());
}

TEST(LinearAttentionParallelTest, RecognizesImplicitLinearLayers) {
  LinearAttentionPolicy policy;
  policy.requires_unsplit_kv = true;
  ModelArgs args;
  args.full_attention_interval(/*full_attention_interval=*/4);
  EXPECT_TRUE(validate_linear_attention_kv_split(
                  args, policy, /*effective_kv_split_size=*/2)
                  .has_value());
}
TEST(LinearAttentionParallelTest, LeavesKvSplitUnrestrictedByDefault) {
  const LinearAttentionPolicy policy;
  for (const std::string& model_type : {"qwen3_5", "glm5_next"}) {
    ModelArgs args;
    args.model_type(model_type)
        .layer_types(/*layer_types=*/{"linear_attention"});
    for (const int32_t cp_size : {1, 2}) {
      ParallelArgs parallel(/*rank=*/0,
                            /*world_size=*/8,
                            /*dp_size=*/1,
                            cp_size,
                            /*process_group=*/nullptr,
                            /*ep_size=*/8);
      for (const int32_t kv_split_size : {0, 2}) {
        parallel.kv_split_size(kv_split_size);
        EXPECT_FALSE(validate_linear_attention_kv_split(
                         args, policy, parallel.kv_split_size_effective())
                         .has_value())
            << "model_type=" << model_type << ", cp_size=" << cp_size
            << ", kv_split_size=" << kv_split_size;
      }
    }
  }
}

TEST(LinearAttentionParallelTest, HeadShardingUsesAttentionTpByDefault) {
  const LinearAttentionPolicy policy;
  for (const int64_t cp_size : {1, 2, 4}) {
    EXPECT_EQ(linear_head_tp_size(policy, /*attention_tp_size=*/2, cp_size), 2);
  }
}

TEST(LinearAttentionParallelTest,
     HeadShardingCanIncludeCpIndependentlyOfKvSplit) {
  for (const bool requires_unsplit_kv : {false, true}) {
    const LinearAttentionPolicy policy{LinearHeadSharding::CP_TP,
                                       requires_unsplit_kv};
    for (const int64_t cp_size : {1, 2, 4, 8}) {
      const int64_t attention_tp_size = 8 / cp_size;
      EXPECT_EQ(linear_head_tp_size(policy, attention_tp_size, cp_size), 8);
    }
  }
}

}  // namespace
}  // namespace xllm
