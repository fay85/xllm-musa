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

#include "framework/kv_cache/cache_layout_builder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/kv_shard_layout.h"
#include "layers/common/kv_shard_batch_metadata.h"
#if defined(USE_MUSA)
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache_transfer/reshard_planner.h"
#endif

namespace xllm {

namespace {

TEST(CacheLayoutBuilderTest, DescribesTokenMajorGqaHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 3;
  context.block_token_capacity = 3;
  context.kv_head_count = 6;
  KVCacheTensor tensor{KVCacheTensorRole::KEY,
                       torch::zeros({2, 3, 2, 4}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::SHARDED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::BLOCK);
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 2U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[1].logical_offset_bytes, 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].logical_stride_bytes, 6U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].physical_stride_bytes, 2U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 1);
}

TEST(CacheLayoutBuilderTest, SelectsStableOwnerForMqaReplicas) {
  CacheTensorLayoutContext context;
  context.tp_rank = 3;
  context.tp_size = 4;
  context.block_token_capacity = 2;
  context.kv_head_count = 1;
  KVCacheTensor tensor{KVCacheTensorRole::VALUE,
                       torch::zeros({2, 2, 1, 8}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, SelectsContiguousReplicaGroupForGqaHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 2;
  context.tp_size = 4;
  context.block_token_capacity = 2;
  context.kv_head_count = 2;
  KVCacheTensor tensor{KVCacheTensorRole::VALUE,
                       torch::zeros({2, 2, 1, 8}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 8U * 4U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 2);
}

TEST(CacheLayoutBuilderTest, DescribesMlaAsWholeResourceReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 2;
  context.tp_size = 4;
  context.enable_mla = true;
  KVCacheTensor tensor{KVCacheTensorRole::KEY,
                       torch::zeros({2, 3, 16}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region,
            static_cast<uint64_t>(tensor.tensor.nbytes() / 2));
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSharedIndexerKeyAsReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 7;
  context.tp_size = 8;
  context.block_token_capacity = 3;
  context.index_head_count = 64;
  KVCacheTensor tensor{KVCacheTensorRole::INDEX,
                       torch::zeros({2, 3, 1, 4}),
                       cache_group_id(BlockType::C4)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSharedIndexerScaleAsReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 3;
  context.tp_size = 4;
  context.block_token_capacity = 3;
  context.index_head_count = 64;
  KVCacheTensor tensor{KVCacheTensorRole::INDEX_SCALE,
                       torch::zeros({2, 3, 1}),
                       cache_group_id(BlockType::C4)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSequenceScopedSsmHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
#if defined(USE_NPU)
  context.enable_mla = true;
#endif
  context.linear_value_head_count = 4;
  KVCacheTensor tensor{KVCacheTensorRole::SSM,
                       torch::zeros({2, 2, 3, 4}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::SHARDED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 2U * 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 1U);
}

TEST(CacheLayoutBuilderTest, DescribesCheckpointedSsmRowsPerLogicalSlot) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_value_head_count = 4;
  context.linear_ssm_checkpoint_stride = 3;
  KVCacheTensor tensor{KVCacheTensorRole::SSM,
                       torch::zeros({6, 2, 3, 4}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].logical_stride_bytes, 4U * 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].physical_stride_bytes, 2U * 3U * 4U * 4U);
}

TEST(CacheLayoutBuilderTest, DescribesCompositeConvState) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
#if defined(USE_NPU)
  context.enable_mla = true;
#endif
  context.linear_key_head_count = 4;
  context.linear_value_head_count = 2;
  context.linear_key_head_dim = 3;
  KVCacheTensor tensor{KVCacheTensorRole::CONV,
#if defined(USE_MUSA)
                       torch::zeros({2, 15, 5}),
#else
                       torch::zeros({2, 5, 15}),
#endif
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::COMPOSITE);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
#if defined(USE_MUSA)
  ASSERT_EQ(descriptor.spans.size(), 25U);
  EXPECT_EQ(descriptor.spans[0].logical_tensor, "conv_key_a");
  EXPECT_EQ(descriptor.spans[10].logical_tensor, "conv_key_b");
  EXPECT_EQ(descriptor.spans[20].logical_tensor, "conv_value");
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 24U);
  EXPECT_EQ(descriptor.spans[0].physical_stride_bytes, 20U);
  EXPECT_EQ(descriptor.spans[5].physical_offset_bytes, 60U);
#else
  ASSERT_EQ(descriptor.spans.size(), 5U);
  EXPECT_EQ(descriptor.spans[0].logical_tensor, "conv_key_a");
  EXPECT_EQ(descriptor.spans[2].logical_tensor, "conv_key_b");
  EXPECT_EQ(descriptor.spans[4].logical_tensor, "conv_value");
  EXPECT_EQ(descriptor.spans[0].repeat_count, 5U);
#endif
}

#if defined(USE_MUSA)
TEST(CacheLayoutBuilderTest, RejectsInvalidMusaConvStorage) {
  CacheTensorLayoutContext context;
  context.linear_key_head_count = 2;
  context.linear_value_head_count = 2;
  context.linear_key_head_dim = 2;
  const std::vector<torch::Tensor> invalid_tensors = {
      torch::zeros({4, 3, 12}),
      torch::zeros({4, 11, 3}),
      torch::zeros({4, 3, 12}).transpose(1, 2)};
  for (const torch::Tensor& value : invalid_tensors) {
    KVCacheTensor tensor{KVCacheTensorRole::CONV,
                         value,
                         cache_group_id(BlockType::LINEAR),
                         /*sequence_scoped=*/true};
    std::string error;
    EXPECT_FALSE(describe_cache_tensor(context, &tensor, &error));
    EXPECT_FALSE(error.empty());
  }
}
#endif

TEST(CacheLayoutBuilderTest, HybridMlaKeepsKPoolTailSequenceScopedReplica) {
  CacheTensorLayoutContext context;
  context.enable_mla = true;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_value_head_count = 4;
  KVCacheTensor tail{KVCacheTensorRole::KPOOL_TAIL,
                     torch::zeros({3, 2, 8, 4}, torch::kBFloat16),
                     cache_group_id(BlockType::LINEAR),
                     /*sequence_scoped=*/true};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &tail, &error)) << error;
  const auto& descriptor = *tail.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 2U * 8U * 4U * 2U);
}

TEST(KVShardLayoutTest, MapsTokensAndSlots) {
  const int64_t dcp4_lengths[] = {128, 128, 1, 0};
  for (int32_t rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(KVShardLayout(128, 4, rank).local_token_count(257),
              dcp4_lengths[rank]);
  }
  EXPECT_EQ(KVShardLayout(128, 2, 0).local_token_count(300), 172);
  EXPECT_EQ(KVShardLayout(128, 2, 1).local_token_count(300), 128);

  const torch::Tensor slots =
      torch::tensor({-1, 0, 127, 128, 255, 256}, torch::kInt32);
  EXPECT_TRUE(torch::equal(
      layer::localize_kv_shard_slots(slots, KVShardLayout(128, 2, 0)),
      torch::tensor({-1, 0, 127, -1, -1, 128}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      layer::localize_kv_shard_slots(slots, KVShardLayout(128, 2, 1)),
      torch::tensor({-1, -1, -1, 0, 127, -1}, torch::kInt32)));
  EXPECT_EQ(layer::localize_kv_shard_slots(torch::empty({0}),
                                           KVShardLayout(128, 2, 0))
                .numel(),
            0);
  EXPECT_FALSE(
      layer::localize_kv_shard_slots(torch::Tensor(), KVShardLayout(128, 2, 0))
          .defined());
}

TEST(CacheLayoutBuilderTest, RecordsPerPoolSpansFromTensors) {
  const Dsv4CacheGeometry geometry;
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 128;
  context.kv_head_count = 1;

  std::vector<KVCache> caches;
  DeepSeekV4KVCacheTensors swa_tensors;
  swa_tensors.swa_cache = torch::zeros({2, 128, 1, 8}, torch::kBFloat16);
  caches.emplace_back(swa_tensors);
  DeepSeekV4KVCacheTensors c4_tensors;
  c4_tensors.compressed_block_type = BlockType::C4;
  c4_tensors.key_cache =
      torch::zeros({4, geometry.c4_physical_dim(), 1, 8}, torch::kBFloat16);
  // INDEX gets a narrower token axis on purpose: compressed KPool layouts
  // shape it differently from the pool's K/V tensors, so recording it into
  // the per-pool span map would abort with a span conflict.
  c4_tensors.index_cache = torch::zeros({4, 128, 1, 8}, torch::kInt8);
  c4_tensors.swa_cache = torch::zeros({2, 128, 1, 8}, torch::kBFloat16);
  caches.emplace_back(c4_tensors);
  DeepSeekV4KVCacheTensors c128_tensors;
  c128_tensors.compressed_block_type = BlockType::C128;
  c128_tensors.key_cache =
      torch::zeros({8, geometry.c128_physical_dim(), 1, 8}, torch::kBFloat16);
  c128_tensors.swa_cache = torch::zeros({2, 128, 1, 8}, torch::kBFloat16);
  caches.emplace_back(c128_tensors);

  record_group_block_capacities(caches, &context);

  // Only the compressed K/V pools record a span: WINDOW, INDEX and the state
  // tensors do not define their pool's token axis, so the SWA group keeps the
  // global block_token_capacity fallback.
  ASSERT_EQ(context.group_block_capacities.size(), 2U);
  EXPECT_EQ(context.group_block_capacities.at(cache_group_id(BlockType::C4)),
            geometry.c4_physical_dim());
  EXPECT_EQ(context.group_block_capacities.at(cache_group_id(BlockType::C128)),
            geometry.c128_physical_dim());
  EXPECT_EQ(
      context.group_block_capacities.count(cache_group_id(BlockType::SWA)), 0U);
}

TEST(CacheLayoutBuilderTest, DescribesTypedPoolAgainstRecordedSpan) {
  const Dsv4CacheGeometry geometry;
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 128;
  context.kv_head_count = 1;

  std::vector<KVCache> caches;
  DeepSeekV4KVCacheTensors c4_tensors;
  c4_tensors.compressed_block_type = BlockType::C4;
  c4_tensors.key_cache =
      torch::zeros({4, geometry.c4_physical_dim(), 1, 8}, torch::kBFloat16);
  DeepSeekV4KVCacheTensors c128_tensors;
  c128_tensors.compressed_block_type = BlockType::C128;
  c128_tensors.key_cache =
      torch::zeros({8, geometry.c128_physical_dim(), 1, 8}, torch::kBFloat16);
  caches.emplace_back(c4_tensors);
  caches.emplace_back(c128_tensors);
  record_group_block_capacities(caches, &context);

  // Each typed pool validates against its own recorded span instead of the
  // uniform scheduler block size.
  KVCacheTensor c4_key{
      KVCacheTensorRole::KEY,
      torch::zeros({4, geometry.c4_physical_dim(), 1, 8}, torch::kBFloat16),
      cache_group_id(BlockType::C4)};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &c4_key, &error)) << error;

  KVCacheTensor c128_key{
      KVCacheTensorRole::KEY,
      torch::zeros({4, geometry.c128_physical_dim(), 1, 8}, torch::kBFloat16),
      cache_group_id(BlockType::C128)};
  error.clear();
  ASSERT_TRUE(describe_cache_tensor(context, &c128_key, &error)) << error;

  // Pools without a recorded span keep the global capacity.
  KVCacheTensor uniform_key{KVCacheTensorRole::KEY,
                            torch::zeros({4, 128, 1, 8}, torch::kBFloat16),
                            cache_group_id(BlockType::KV)};
  error.clear();
  EXPECT_TRUE(describe_cache_tensor(context, &uniform_key, &error)) << error;

  // A C4 tensor with the uniform capacity is rejected: the pool's recorded
  // span is the authority.
  KVCacheTensor mismatched_key{KVCacheTensorRole::KEY,
                               torch::zeros({4, 128, 1, 8}, torch::kBFloat16),
                               cache_group_id(BlockType::C4)};
  error.clear();
  EXPECT_FALSE(describe_cache_tensor(context, &mismatched_key, &error));
  EXPECT_NE(error.find("token capacity differs"), std::string::npos);
}

TEST(CacheLayoutBuilderTest, FailsWhenPoolSpansConflict) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;

  // Both spans deviate from the uniform capacity: pools running at the
  // uniform capacity record nothing, so a conflict needs two distinct
  // deviating spans inside one pool.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 32, 1, 8}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 64, 1, 8}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  EXPECT_DEATH(record_group_block_capacities(caches, &context),
               "Cache tensors of one pool must share the token-axis span");
}

TEST(CacheLayoutBuilderTest, SkipsSpanRecordingForMlaLayouts) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;
  context.enable_mla = true;

  // MLA pools legitimately mix K/V token-axis spans: the NPU NZ layout
  // shapes the KEY axis after kv_lora_rank and the VALUE axis after
  // qk_rope_head_dim, and describe routes MLA tensors through the
  // replicated-tensor contract, which never reads a pool span.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 32, 16, 16}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 4, 16, 16}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  record_group_block_capacities(caches, &context);
  EXPECT_EQ(context.group_block_capacities.size(), 0U);
}

TEST(CacheLayoutBuilderTest, SkipsSpanRecordingForUniformPools) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;

  // Uniform pools run at the declared block capacity: recording their span
  // would make describe validate the tensor against its own size and mute
  // the producer-capacity check. They stay on the global fallback.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 16, 1, 8}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 16, 1, 8}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  record_group_block_capacities(caches, &context);
  EXPECT_EQ(context.group_block_capacities.size(), 0U);

  // The producer-declared capacity stays the authority: a uniform-pool
  // tensor whose token axis diverges from it is rejected at describe
  // instead of silently redefining the pool's published span.
  KVCacheTensor diverging_key{KVCacheTensorRole::KEY,
                              torch::zeros({4, 32, 1, 8}, torch::kBFloat16),
                              cache_group_id(BlockType::KV)};
  std::string error;
  EXPECT_FALSE(describe_cache_tensor(context, &diverging_key, &error));
  EXPECT_NE(error.find("token capacity differs"), std::string::npos);
}

TEST(CacheLayoutBuilderTest, SkipsSpanRecordingForTensorsWithoutTokenAxis) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;

  // A tensor without the token axis defines no pool span: recording it
  // would read a nonexistent axis. It is skipped here and left to
  // describe_cache_tensor, which rejects it with a precise error. The
  // value cache runs at a deviating span so recording still happens for
  // the well-formed tensor.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 32, 1, 8}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  record_group_block_capacities(caches, &context);

  ASSERT_EQ(context.group_block_capacities.size(), 1U);
  EXPECT_EQ(context.group_block_capacities.at(cache_group_id(BlockType::KV)),
            32);
}

TEST(CacheLayoutBuilderTest, RejectsOrdinaryNoncontiguousAttentionStorage) {
  CacheTensorLayoutContext context;
  context.tp_size = 8;
  context.kv_head_count = 1;
  context.block_token_capacity = 128;
  const torch::Tensor storage = torch::zeros({3, 128, 1, 16});
  for (KVCacheTensorRole role :
       {KVCacheTensorRole::KEY, KVCacheTensorRole::VALUE}) {
    SCOPED_TRACE(role.to_string());
    KVCacheTensor tensor{role,
                         storage.narrow(/*dim=*/3, /*start=*/0, /*length=*/8),
                         cache_group_id(BlockType::KV)};
    ASSERT_FALSE(tensor.tensor.is_contiguous());
    ASSERT_FALSE(tensor.shard_descriptor.has_value());
    std::string error;
    EXPECT_FALSE(describe_cache_tensor(context, &tensor, &error));
    EXPECT_NE(error.find("must be contiguous"), std::string::npos);
  }
}

#if defined(USE_MUSA)
std::vector<KVCache> make_musa_grouped_caches() {
  KVCacheCapacity capacity;
  capacity.block_size(128).swa_count(3).c4_count(96).c128_count(3);
  ModelArgs model_args;
  model_args.model_type("deepseek_v4");
  const KVCacheShape shape(capacity, model_args, /*world_size=*/8);
  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kBFloat16)
      .num_layers(3)
      .model_type("deepseek_v4")
      .block_size(128)
      .head_dim(512)
      .index_head_dim(128)
      .window_size(128)
      .compress_ratios({1, 4, 128});
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  return caches;
}

WorkerCacheLayoutManifest make_musa_page_manifest(
    const KVCacheTensor& cache_tensor,
    const std::string& address,
    int32_t tp_rank) {
  WorkerCacheLayoutManifest manifest;
  manifest.incarnation_id = address + "-incarnation";
  manifest.layout_generation = 1;
  manifest.fingerprint = "musa-dsv4-model1-padding-test";
  manifest.backend = "musa";
  manifest.layout_family = "grouped";
  manifest.addr = address;
  manifest.coordinates.tp_rank = tp_rank;
  manifest.coordinates.tp_size = 8;
  const torch::Tensor& tensor = cache_tensor.tensor;
  CacheTensorManifest description;
  description.role = static_cast<int32_t>(KVCacheTensorRole::VALUE);
  description.group_id = cache_tensor.group_id;
  description.scalar_type = static_cast<int32_t>(tensor.scalar_type());
  description.element_bytes = tensor.element_size();
  description.shape = tensor.sizes().vec();
  description.stride = tensor.strides().vec();
  description.contiguous = tensor.is_contiguous();
  description.resource_count = static_cast<uint64_t>(tensor.size(0));
  description.resource_stride_bytes =
      static_cast<uint64_t>(tensor.stride(0)) * tensor.element_size();
  description.buffer_bytes = tensor.nbytes();
  description.block_token_capacity = 128;
  description.shard = cache_tensor.shard_descriptor.value();
  manifest.tensors.reserve(1);
  manifest.tensors.emplace_back(std::move(description));
  return manifest;
}

TEST(CacheLayoutBuilderTest, MusaGroupedModel1BytesDoNotRedefineTokenCapacity) {
  const std::vector<KVCache> caches = make_musa_grouped_caches();
  ASSERT_EQ(caches.size(), 3U);
  CacheTensorLayoutContext context;
  context.tp_rank = 7;
  context.tp_size = 8;
  context.kv_head_count = 1;
  context.index_head_count = 64;
  context.block_token_capacity = 128;
  record_group_block_capacities(caches, &context);
  EXPECT_TRUE(context.group_block_capacities.empty());
  for (BlockType type : {BlockType::SWA, BlockType::C4, BlockType::C128}) {
    EXPECT_EQ(effective_block_token_capacity(context, cache_group_id(type)),
              128);
  }
  int64_t model1_tensor_count = 0;
  for (const KVCache& cache : caches) {
    for (KVCacheTensor tensor : cache.get_cache_tensors()) {
      SCOPED_TRACE(tensor.role.to_string());
      std::string error;
      ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
      ASSERT_TRUE(tensor.shard_descriptor.has_value());
      ASSERT_TRUE(tensor.tensor.is_contiguous());
      model1_tensor_count += tensor.role == KVCacheTensorRole::VALUE ? 1 : 0;
    }
  }
  EXPECT_EQ(model1_tensor_count, 4);
}

TEST(CacheLayoutBuilderTest, MusaGroupedModel1TransfersWholeFinalPage) {
  constexpr int64_t kTokenBytes = 584;
  constexpr int64_t kPayloadBytes = 128 * kTokenBytes;
  constexpr int64_t kPageBytes = 74880;
  const std::vector<KVCache> sources = make_musa_grouped_caches();
  const std::vector<KVCache> destinations = make_musa_grouped_caches();
  ASSERT_EQ(sources.size(), 3U);
  ASSERT_EQ(destinations.size(), 3U);
  for (int64_t layer_id : {1, 2}) {
    const BlockType compressed_type =
        layer_id == 1 ? BlockType::C4 : BlockType::C128;
    const auto source_tensors = sources[layer_id].get_cache_tensors();
    const auto destination_tensors = destinations[layer_id].get_cache_tensors();
    for (BlockType type : {BlockType::SWA, compressed_type}) {
      SCOPED_TRACE("layer=" + std::to_string(layer_id) +
                   ", group=" + std::to_string(cache_group_id(type)));
      const auto matches = [type](const KVCacheTensor& tensor) {
        return tensor.role == KVCacheTensorRole::VALUE &&
               tensor.group_id == cache_group_id(type);
      };
      const auto source =
          std::find_if(source_tensors.begin(), source_tensors.end(), matches);
      const auto destination = std::find_if(
          destination_tensors.begin(), destination_tensors.end(), matches);
      ASSERT_NE(source, source_tensors.end());
      ASSERT_NE(destination, destination_tensors.end());
      const torch::Tensor source_kernel =
          sources[layer_id].get_block_type_tensors(type).at(
              KVCacheTensorRole::VALUE);
      const torch::Tensor destination_kernel =
          destinations[layer_id].get_block_type_tensors(type).at(
              KVCacheTensorRole::VALUE);
      const int64_t block_count = source_kernel.size(0);
      ASSERT_GT(block_count, 1);
      EXPECT_EQ(source_kernel.sizes().vec(),
                (std::vector<int64_t>{block_count, 128, 1, kTokenBytes}));
      EXPECT_EQ(
          source_kernel.strides().vec(),
          (std::vector<int64_t>{kPageBytes, kTokenBytes, kTokenBytes, 1}));
      EXPECT_FALSE(source_kernel.is_contiguous());
      EXPECT_EQ(source->tensor.sizes().vec(),
                (std::vector<int64_t>{block_count, kPageBytes}));
      EXPECT_EQ(source->tensor.data_ptr(), source_kernel.data_ptr());
      EXPECT_EQ(source->tensor.storage().unsafeGetStorageImpl(),
                source_kernel.storage().unsafeGetStorageImpl());
      EXPECT_EQ(source->tensor.nbytes(),
                static_cast<size_t>(block_count * kPageBytes));
      ASSERT_TRUE(source->shard_descriptor.has_value());
      ASSERT_TRUE(destination->shard_descriptor.has_value());
      const LogicalShardDescriptor& descriptor = *source->shard_descriptor;
      EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
      EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::BLOCK);
      ASSERT_EQ(descriptor.spans.size(), 1U);
      EXPECT_EQ(descriptor.spans[0].bytes_per_region, kPageBytes);
      EXPECT_EQ(descriptor.spans[0].repeat_count, 1U);
      EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);

      source->tensor.fill_(17);
      source->tensor[block_count - 1]
          .narrow(/*dim=*/0, /*start=*/kPayloadBytes, /*length=*/128)
          .fill_(85);
      source_kernel[block_count - 1].fill_(31);
      destination->tensor.fill_(239);
      // Instance coverage validates the complete TP topology even when only
      // rank 0 owns the replicated page's outgoing bytes.
      std::vector<WorkerCacheLayoutManifest> source_manifests;
      source_manifests.reserve(8);
      for (int32_t tp_rank = 0; tp_rank < 8; ++tp_rank) {
        source_manifests.emplace_back(make_musa_page_manifest(
            *source, "source-" + std::to_string(tp_rank), tp_rank));
      }
      const WorkerCacheLayoutManifest& source_manifest =
          source_manifests.front();
      WorkerCacheLayoutManifest destination_manifest =
          make_musa_page_manifest(*destination, "destination", /*tp_rank=*/7);
      const Status source_status =
          validate_worker_cache_layout(source_manifest);
      ASSERT_TRUE(source_status.ok()) << source_status.message();
      const Status destination_status =
          validate_worker_cache_layout(destination_manifest);
      ASSERT_TRUE(destination_status.ok()) << destination_status.message();
      ReshardPlanner planner;
      const Status coverage_status = planner.validate_destination_coverage(
          source_manifests, destination_manifest);
      ASSERT_TRUE(coverage_status.ok()) << coverage_status.message();
      ReshardPlanTemplate plan;
      const Status plan_status = planner.build_outgoing_plan(
          source_manifest, destination_manifest, &plan);
      ASSERT_TRUE(plan_status.ok()) << plan_status.message();
      KVTransferMapping mapping;
      mapping.group_id = cache_group_id(type);
      mapping.local_ids = {static_cast<uint64_t>(block_count - 1)};
      mapping.remote_ids = {static_cast<uint64_t>(block_count - 1)};
      std::vector<ByteRegion> regions;
      const Status bind_status =
          RequestRegionBinder().bind_pull(plan,
                                          {mapping},
                                          CacheNamespace::MAIN,
                                          /*layer_id=*/0,
                                          &regions);
      ASSERT_TRUE(bind_status.ok()) << bind_status.message();
      ASSERT_EQ(regions.size(), 1U);
      const ByteRegion& region = regions.front();
      EXPECT_EQ(region.length, kPageBytes);
      ASSERT_EQ(region.remote_offset + region.length, source->tensor.nbytes());
      ASSERT_EQ(region.local_offset + region.length,
                destination->tensor.nbytes());
      std::memcpy(destination->tensor.data_ptr<uint8_t>() + region.local_offset,
                  source->tensor.data_ptr<uint8_t>() + region.remote_offset,
                  static_cast<size_t>(region.length));
      EXPECT_TRUE(torch::equal(destination_kernel[block_count - 1],
                               source_kernel[block_count - 1]));
      EXPECT_TRUE(torch::equal(destination->tensor[block_count - 1],
                               source->tensor[block_count - 1]));
      EXPECT_TRUE(
          torch::equal(destination->tensor[block_count - 1].narrow(
                           /*dim=*/0, /*start=*/kPayloadBytes, /*length=*/128),
                       torch::full({128}, 85, torch::kUInt8)));
      EXPECT_EQ(
          destination->tensor[block_count - 1][kPageBytes - 1].item<uint8_t>(),
          85);
      EXPECT_TRUE(torch::equal(
          destination->tensor.narrow(
              /*dim=*/0, /*start=*/0, /*length=*/block_count - 1),
          torch::full({block_count - 1, kPageBytes}, 239, torch::kUInt8)));
      EXPECT_EQ(destination_kernel.sizes().vec(), source_kernel.sizes().vec());
      EXPECT_EQ(destination_kernel.strides().vec(),
                source_kernel.strides().vec());
    }
  }
}

TEST(CacheLayoutBuilderTest, RejectsMusaModel1StorageMissingFinalPadding) {
  constexpr int64_t kPageBytes = 74880;
  const torch::Tensor storage =
      torch::empty({3 * kPageBytes - 128}, torch::kUInt8);
  DeepSeekV4KVCacheTensors tensors;
  tensors.compressed_block_type = BlockType::C4;
  tensors.mate_key_cache =
      storage.as_strided({3, 128, 1, 584}, {kPageBytes, 584, 584, 1});
  KVCache cache(tensors);
  EXPECT_DEATH(cache.get_cache_tensors(),
               "must include the final page padding");
}
#endif

}  // namespace

}  // namespace xllm
