/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <string>

#include "framework/kv_cache/cache_layout_builder.h"

namespace xllm {
namespace {

TEST(CacheLayoutBuilderTest, HybridMlaKeepsRecurrentStateSharded) {
  CacheTensorLayoutContext context;
  context.enable_mla = true;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_key_head_count = 4;
  context.linear_value_head_count = 4;
  context.linear_key_head_dim = 3;
  context.linear_ssm_checkpoint_stride = 4;
  KVCacheTensor conv{KVCacheTensorRole::CONV,
                     torch::zeros({3, 6, 18}),
                     cache_group_id(BlockType::LINEAR),
                     /*sequence_scoped=*/true};
  KVCacheTensor ssm{KVCacheTensorRole::SSM,
                    torch::zeros({12, 2, 3, 3}),
                    cache_group_id(BlockType::LINEAR),
                    /*sequence_scoped=*/true};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &conv, &error)) << error;
  ASSERT_TRUE(describe_cache_tensor(context, &ssm, &error)) << error;
  const auto& conv_layout = *conv.shard_descriptor;
  const auto& ssm_layout = *ssm.shard_descriptor;
  EXPECT_EQ(conv_layout.kind, LogicalShardKind::COMPOSITE);
  EXPECT_EQ(ssm_layout.kind, LogicalShardKind::SHARDED);
  EXPECT_EQ(conv_layout.resource_scope, CacheResourceScope::SEQUENCE);
  EXPECT_EQ(ssm_layout.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(conv_layout.spans.size(), 6U);
  ASSERT_EQ(ssm_layout.spans.size(), 2U);
  EXPECT_EQ(conv_layout.spans[0].owner_tp_rank, 1);
  EXPECT_EQ(conv_layout.spans[0].logical_offset_bytes, 2U * 3U * 4U);
  EXPECT_EQ(ssm_layout.spans[0].owner_tp_rank, 1);
  EXPECT_EQ(ssm_layout.spans[0].logical_offset_bytes, 2U * 3U * 3U * 4U);
  EXPECT_EQ(ssm_layout.spans[0].repeat_count, 4U);
}

TEST(CacheLayoutBuilderTest, HybridMlaUsesFullHeadTpForRecurrentOffsets) {
  for (int32_t rank = 0; rank < 8; ++rank) {
    CacheTensorLayoutContext context;
    context.enable_mla = true;
    context.tp_size = 2;
    context.tp_rank = rank % 2;
    context.linear_head_tp_size = 8;
    context.linear_head_tp_rank = rank;
    context.linear_key_head_count = 8;
    context.linear_value_head_count = 8;
    context.linear_key_head_dim = 3;
    context.linear_ssm_checkpoint_stride = 4;
    KVCacheTensor conv{KVCacheTensorRole::CONV,
                       torch::zeros({3, 6, 9}),
                       cache_group_id(BlockType::LINEAR),
                       true};
    KVCacheTensor ssm{KVCacheTensorRole::SSM,
                      torch::zeros({12, 1, 3, 3}),
                      cache_group_id(BlockType::LINEAR),
                      true};
    std::string error;
    ASSERT_TRUE(describe_cache_tensor(context, &conv, &error)) << error;
    ASSERT_TRUE(describe_cache_tensor(context, &ssm, &error)) << error;
    EXPECT_TRUE(conv.shard_descriptor->partitioned_across_cp);
    EXPECT_TRUE(ssm.shard_descriptor->partitioned_across_cp);
    ASSERT_EQ(conv.shard_descriptor->spans.size(), 3U);
    ASSERT_EQ(ssm.shard_descriptor->spans.size(), 1U);
    EXPECT_EQ(conv.shard_descriptor->spans[0].logical_offset_bytes,
              static_cast<uint64_t>(rank) * 3 * 4);
    EXPECT_EQ(ssm.shard_descriptor->spans[0].logical_offset_bytes,
              static_cast<uint64_t>(rank) * 3 * 3 * 4);
    EXPECT_EQ(ssm.shard_descriptor->spans[0].owner_tp_rank, rank % 2);
  }
}

}  // namespace
}  // namespace xllm
