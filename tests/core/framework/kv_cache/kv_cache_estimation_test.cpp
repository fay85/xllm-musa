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

#include "framework/kv_cache/kv_cache_estimation.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/speculative_config.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"

namespace xllm {
namespace {

class DcpIndexerEstimationTest : public ::testing::TestWithParam<int32_t> {
 protected:
  void SetUp() override {
    ParallelConfig::get_instance().kv_split_size(GetParam()).cp_size(4);
    // These cases describe the replicated indexer geometry, which the Python
    // executor still requires. The ATB/native graph allocates a single owner
    // page per logical block instead (indexer_pages_per_block()).
    saved_model_impl_ = ModelConfig::get_instance().model_impl();
    ModelConfig::get_instance().model_impl("python");
  }

  void TearDown() override {
    ModelConfig::get_instance().model_impl(saved_model_impl_);
    ParallelConfig::get_instance() = saved_config_;
  }

  int64_t expected_factor() const {
#if defined(USE_NPU)
    return GetParam() == 0 ? 4 : GetParam();
#else
    return 1;
#endif
  }

 private:
  ParallelConfig saved_config_ = ParallelConfig::get_instance();
  std::string saved_model_impl_;
};

ModelArgs make_standard_args() {
  ModelArgs model_args;
  model_args.n_layers(4).head_dim(16);
  return model_args;
}

KVCacheEstimateOptions make_estimate_options() {
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 16;
  options.world_size = 1;
  options.n_local_kv_heads = 2;
  options.max_seqs_per_batch = 8;
  return options;
}

}  // namespace

TEST(KVCacheEstimationTest, EstimatesStandardAttentionBlocks) {
  ModelArgs model_args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.cache_size_in_bytes(), 1024 * 1024);
  EXPECT_EQ(capacity.block_size(), 16);
  EXPECT_EQ(capacity.slot_size(), 128);
  EXPECT_EQ(capacity.n_layers(), 4);
  EXPECT_EQ(capacity.num_full_attention_layers(), 4);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.n_blocks(), 128);
}

TEST(KVCacheEstimationTest,
     PrefillLayerwiseCapacityUsesBothDeviceAndHostBudgets) {
  const ModelArgs args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();
  options.prefill_layerwise_num_buffers = 2;
  options.prefill_layerwise_host_memory_bytes = 64 * 1024 * 1024;
  EXPECT_EQ(estimate_kv_cache_capacity(args, options).n_blocks(), 170);
  options.prefill_layerwise_num_buffers = 1;
  EXPECT_EQ(estimate_kv_cache_capacity(args, options).n_blocks(), 256);
  options.prefill_layerwise_host_memory_bytes =
      4 * host_page_size() * args.n_layers() + 512 * 1024;
  const KVCacheCapacity host_limited =
      estimate_kv_cache_capacity(args, options);
  EXPECT_EQ(host_limited.n_blocks(), 64);
  // At B=N the shared pool is capped at N-1. A one-layer model still needs
  // its complete Host mirror, whose budget can limit resident block count.
  options.prefill_layerwise_num_buffers = 4;
  options.prefill_layerwise_host_memory_bytes = 64 * 1024 * 1024;
  EXPECT_EQ(estimate_kv_cache_capacity(args, options).n_blocks(), 128);
  ModelArgs single_layer_args = args;
  single_layer_args.n_layers(1);
  options.prefill_layerwise_num_buffers = 1;
  options.prefill_layerwise_host_memory_bytes =
      4 * host_page_size() + 128 * 1024;
  EXPECT_EQ(estimate_kv_cache_capacity(single_layer_args, options).n_blocks(),
            64);
  options.prefill_layerwise_num_buffers = 0;
  EXPECT_EQ(estimate_kv_cache_capacity(args, options).n_blocks(), 128);
  options.dp_size = 2;
  options.world_size = 2;
  EXPECT_EQ(estimate_kv_cache_capacity(args, options).n_blocks(), 128);
  options.prefill_layerwise_num_buffers = 1;
  EXPECT_DEATH(estimate_kv_cache_capacity(args, options), "requires DP=1");
  EXPECT_EQ(prefill_layerwise_buffer_index(/*layer=*/0, /*buffers=*/0), 0);
  EXPECT_DEATH(prefill_layerwise_buffer_index(/*layer=*/1, /*buffers=*/0),
               "layer=1, buffers=0");
}

TEST(KVCacheEstimationTest, PrefillLayerwiseCapacityCountsPhysicalIndexSlots) {
  const ModelArgs no_index_args = make_standard_args();
  ModelArgs index_args = no_index_args;
  index_args.index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.prefill_layerwise_num_buffers = 2;
  options.prefill_layerwise_host_memory_bytes = 64 * 1024 * 1024;

  const KVCacheCapacity without_index =
      estimate_kv_cache_capacity(no_index_args, options);
  const KVCacheCapacity with_index =
      estimate_kv_cache_capacity(index_args, options);

  EXPECT_EQ(without_index.index_slot_size(), 0);
  EXPECT_GT(with_index.index_slot_size(), 0);
  EXPECT_GT(without_index.n_blocks(), with_index.n_blocks());
  EXPECT_EQ(prefill_layerwise_device_block_size_in_bytes(
                index_args, with_index, /*num_buffers=*/2),
            3 * (with_index.block_size() * with_index.slot_size() +
                 with_index.index_block_size()));
#if defined(USE_MLU) || defined(USE_NPU)
  // FSFS maps index data to the resident slot and only the second shared
  // slot. The first shared slot holds layers 1 and 3, which reuse top-k.
  index_args.model_type("glm_moe_dsa").index_topk(8).index_topk_pattern("FSFS");
  const KVCacheCapacity shared_index =
      estimate_kv_cache_capacity(index_args, options);
  EXPECT_EQ(prefill_layerwise_device_block_size_in_bytes(
                index_args, shared_index, /*num_buffers=*/2),
            3 * shared_index.block_size() * shared_index.slot_size() +
                2 * shared_index.index_block_size());
  options.indexer_cache_dtype = "int8";
  const KVCacheCapacity quantized_index =
      estimate_kv_cache_capacity(index_args, options);
  EXPECT_TRUE(quantized_index.enable_indexer_cache_quant());
  EXPECT_EQ(quantized_index.scale_slot_size(), 0);
  // Quantized index block bytes already include the per-token scale. The
  // FSFS layout needs this complete footprint in two physical index slots.
  EXPECT_EQ(quantized_index.index_block_size(),
            options.block_size * (index_args.index_head_dim() +
                                  static_cast<int64_t>(sizeof(float))));
  EXPECT_EQ(prefill_layerwise_device_block_size_in_bytes(
                index_args, quantized_index, /*num_buffers=*/2),
            3 * quantized_index.block_size() * quantized_index.slot_size() +
                2 * quantized_index.index_block_size());
#endif
}

TEST(KVCacheEstimationTest,
     PrefillLayerwiseSpeculativeCapacityReservesDraftDeviceCache) {
  const ModelArgs args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();
  options.prefill_layerwise_num_buffers = 2;
  options.prefill_layerwise_host_memory_bytes = 64 * 1024 * 1024;
  const KVCacheCapacity target = estimate_kv_cache_capacity(args, options);
  ASSERT_EQ(target.n_blocks(), 170);
  EXPECT_EQ(prefill_layerwise_device_block_size_in_bytes(args, target, 2),
            3 * 16 * 128);

  KVCacheCapacity draft;
  draft.n_blocks(1000)
      .cache_size_in_bytes(1024 * 1024)
      .block_size(16)
      .slot_size(32)
      .n_layers(1)
      .num_full_attention_layers(1);
  const int64_t combined_block_bytes = 3 * 16 * 128 + 16 * 32;
  const PrefillLayerwiseSpecPlan replicated_once =
      prefill_layerwise_blocks_with_draft(args,
                                          target,
                                          draft,
                                          /*num_buffers=*/2,
                                          /*draft_replicated_block_pages=*/1,
                                          /*draft_body_uses_tp1=*/false,
                                          /*draft_manages_own_kv_shape=*/true);
  EXPECT_EQ(replicated_once.n_blocks, (1024 * 1024) / combined_block_bytes);
  EXPECT_EQ(replicated_once.budget_bytes,
            replicated_once.n_blocks * combined_block_bytes);

  const int64_t replicated_twice_block_bytes = 3 * 16 * 128 + 2 * 16 * 32;
  const PrefillLayerwiseSpecPlan replicated_twice =
      prefill_layerwise_blocks_with_draft(args,
                                          target,
                                          draft,
                                          /*num_buffers=*/2,
                                          /*draft_replicated_block_pages=*/2,
                                          /*draft_body_uses_tp1=*/false,
                                          /*draft_manages_own_kv_shape=*/true);
  EXPECT_EQ(replicated_twice.n_blocks,
            (1024 * 1024) / replicated_twice_block_bytes);
  EXPECT_EQ(replicated_twice.budget_bytes,
            replicated_twice.n_blocks * replicated_twice_block_bytes);
}

TEST(KVCacheEstimationTest, PrefillLayerwiseMtpBudgetUsesTargetKvShape) {
  ModelArgs args = make_standard_args();
  args.index_n_heads(1).index_head_dim(16);
  KVCacheCapacity target;
  target.n_blocks(1000)
      .cache_size_in_bytes(1024 * 1024)
      .block_size(16)
      .slot_size(128)
      .index_cache_sizes(8, 128)
      .n_layers(4)
      .num_full_attention_layers(4)
      .num_indexer_layers(4);
  KVCacheCapacity draft = target;
  draft.slot_size(32)
      .n_layers(1)
      .num_full_attention_layers(1)
      .num_indexer_layers(1);

  const PrefillLayerwiseSpecPlan plan =
      prefill_layerwise_blocks_with_draft(args,
                                          target,
                                          draft,
                                          /*num_buffers=*/2,
                                          /*draft_replicated_block_pages=*/1,
                                          /*draft_body_uses_tp1=*/false,
                                          /*draft_manages_own_kv_shape=*/false);
  const int64_t combined_block_bytes = 4 * (16 * 128 + 128);
  EXPECT_EQ(plan.n_blocks, (1024 * 1024) / combined_block_bytes);
  EXPECT_EQ(plan.budget_bytes, plan.n_blocks * combined_block_bytes);
}

TEST(KVCacheEstimationTest, IgnoresLinearStateSlotsWithoutLinearAttention) {
  ModelArgs model_args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();
  options.max_linear_state_cache_slots = 32;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.n_blocks(), 128);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 0);
}

TEST(KVCacheEstimationTest, IndexCacheSizesMustBeConfiguredTogether) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        KVCacheCapacity capacity;
        capacity.index_cache_sizes(/*slot_size=*/8, /*block_size=*/0);
      },
      "slot and block sizes must both be zero or positive");
}

TEST(KVCacheEstimationTest, UserIndexerCacheDtypeDirectlyControlsQuantization) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("unsupported_model")
      .index_n_heads(1)
      .index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();

  options.indexer_cache_dtype = "auto";
  KVCacheCapacity auto_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(auto_capacity.index_slot_size(), 32);
  EXPECT_FALSE(auto_capacity.enable_indexer_cache_quant());

  options.indexer_cache_dtype = "int8";
  KVCacheCapacity int8_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(int8_capacity.index_slot_size(), 20);
  EXPECT_TRUE(int8_capacity.enable_indexer_cache_quant());

#if defined(USE_NPU)
  ModelArgs kpool_args = make_standard_args();
  kpool_args.model_type("glm5_next")
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(4)
      .index_kpool_compress(true);
  KVCacheEstimateOptions kpool_options = make_estimate_options();
  kpool_options.block_size = 128;
  kpool_options.dtype = torch::kBFloat16;
  kpool_options.kpool_layout = KPoolCacheLayout::COMPRESSED_WITH_TAIL;

  const KVCacheCapacity packed_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(packed_capacity.kpool_layout(), KPoolCacheLayout::PACKED);
  EXPECT_EQ(packed_capacity.index_block_size(),
            packed_capacity.index_slot_size() * kpool_options.block_size);

  kpool_args.index_kpool_always_select_tail(true);
  kpool_options.num_speculative_tokens = 3;
  kpool_options.kpool_layout = KPoolCacheLayout::PACKED;
  const KVCacheCapacity compressed_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(compressed_capacity.kpool_layout(),
            KPoolCacheLayout::COMPRESSED_WITH_TAIL);
  EXPECT_EQ(compressed_capacity.kpool_tail_len(), 7);
  EXPECT_EQ(compressed_capacity.kpool_tail_slot_size(),
            2 * static_cast<int64_t>(sizeof(float)) * 7 * 128);

  kpool_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  const KVCacheCapacity non_target_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(non_target_capacity.kpool_layout(), KPoolCacheLayout::PACKED);

  kpool_args.model_type("glm5_next")
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true);
  kpool_options.indexer_cache_dtype = "int8";

  EXPECT_DEATH((void)estimate_kv_cache_capacity(kpool_args, kpool_options),
               "KPool cache layouts do not support.*int8");
#endif
}

TEST(KVCacheEstimationTest, IndexerScaleUsesLogicalCacheCapacity) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "int8";
  options.cache_size_in_bytes = 10 * 1024 * 1024;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.n_blocks(), 1107);
}

#if defined(USE_MLU) || defined(USE_NPU)
TEST(KVCacheEstimationTest, SharedDsaLayersDoNotConsumeIndexerCacheBudget) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm_moe_dsa")
      .index_n_heads(1)
      .index_head_dim(16)
      .index_topk(8)
      .index_topk_pattern("FSFS");
  KVCacheEstimateOptions options = make_estimate_options();

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_full_attention_layers(), 4);
  EXPECT_EQ(capacity.num_indexer_layers(), 2);
  EXPECT_EQ(capacity.n_blocks(), 113);
}

TEST(KVCacheEstimationTest, GlmSharedIndexerFreqAllocatesOnlyFullLayers) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm_moe_dsa")
      .n_layers(78)
      .index_n_heads(1)
      .index_head_dim(16)
      .index_topk(8)
      .index_topk_freq(4)
      .index_skip_topk_offset(3);

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, make_estimate_options());

  EXPECT_EQ(capacity.num_full_attention_layers(), 78);
  EXPECT_EQ(capacity.num_indexer_layers(), 21);
}

#endif

namespace {

ModelArgs make_linear_attention_args(int64_t head_dim = 16) {
  ModelArgs model_args = make_standard_args();
  model_args.head_dim(head_dim)
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  return model_args;
}

KVCacheEstimateOptions make_linear_attention_options() {
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  return options;
}

}  // namespace

TEST(KVCacheEstimationTest, ReservesLinearAttentionState) {
  ModelArgs model_args = make_linear_attention_args();
  KVCacheEstimateOptions options = make_linear_attention_options();

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_full_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 229);
  EXPECT_EQ(capacity.linear_slot_size(), 256);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 117248);
  EXPECT_EQ(capacity.n_blocks(), 227);
}

TEST(KVCacheEstimationTest, LinearStateCapacityDoesNotDependOnPrefixCache) {
  const ModelArgs model_args = make_linear_attention_args(1);
  for (const int64_t explicit_slots : {0, 32}) {
    KVCacheEstimateOptions options = make_linear_attention_options();
    options.block_size = 1;
    options.n_local_kv_heads = 1;
    options.max_concurrent_requests = 1;
    options.max_linear_state_cache_slots = explicit_slots;
    options.enable_prefix_cache = false;
    const KVCacheCapacity disabled =
        estimate_kv_cache_capacity(model_args, options);
    options.enable_prefix_cache = true;
    const KVCacheCapacity enabled =
        estimate_kv_cache_capacity(model_args, options);
    EXPECT_EQ(disabled.num_linear_state_blocks(),
              enabled.num_linear_state_blocks());
    EXPECT_EQ(disabled.n_blocks(), enabled.n_blocks());
    EXPECT_EQ(disabled.linear_cache_size_in_bytes(),
              enabled.linear_cache_size_in_bytes());
    EXPECT_EQ(disabled.num_linear_state_blocks(),
              explicit_slots > 0 ? 34 : 970);
    EXPECT_GT(disabled.n_blocks(), 0);
    EXPECT_LE(disabled.linear_cache_size_in_bytes(),
              options.cache_size_in_bytes);
  }
}

TEST(KVCacheEstimationTest, DecodePrefixCacheResolvesLinearStateSlots) {
  ModelArgs model_args = make_linear_attention_args(1);
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.instance_role = InstanceRole::DECODE;
  options.enable_prefix_cache = true;
  options.max_linear_state_cache_slots = 0;
  // 64 GiB keeps the proportional share above the 1.5x requests below, so
  // the batch cap is the binding constraint.
  options.cache_size_in_bytes = 64LL << 30;

  for (const auto& [max_seqs, expected_blocks] :
       std::vector<std::pair<int64_t, int64_t>>{{8, 14}, {9, 16}, {128, 194}}) {
    SCOPED_TRACE(max_seqs);
    options.max_seqs_per_batch = max_seqs;
    const KVCacheCapacity default_capacity =
        estimate_kv_cache_capacity(model_args, options);
    EXPECT_EQ(default_capacity.num_linear_state_blocks(), expected_blocks);
  }

  options.max_linear_state_cache_slots = 5;
  const KVCacheCapacity explicit_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(explicit_capacity.num_linear_state_blocks(), 7);
}

TEST(KVCacheEstimationTest, DecodePrefixCacheTakesMinOfBatchAndRatio) {
  ModelArgs model_args = make_linear_attention_args(1);
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.instance_role = InstanceRole::DECODE;
  options.enable_prefix_cache = true;
  options.max_linear_state_cache_slots = 0;
  options.cache_size_in_bytes = 1024 * 1024;

  options.max_seqs_per_batch = 0;
  const KVCacheCapacity ratio_capacity =
      estimate_kv_cache_capacity(model_args, options);

  // ceil(1.5 * 8) + 2 = 14, below the 1 MiB proportional share.
  options.max_seqs_per_batch = 8;
  const KVCacheCapacity small_batch =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(small_batch.num_linear_state_blocks(), 14);
  EXPECT_LT(small_batch.num_linear_state_blocks(),
            ratio_capacity.num_linear_state_blocks());

  // ceil(1.5 * 100000) + 2 exceeds the proportional share, so the ratio wins.
  options.max_seqs_per_batch = 100000;
  const KVCacheCapacity large_batch =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(large_batch.num_linear_state_blocks(),
            ratio_capacity.num_linear_state_blocks());
  EXPECT_GT(large_batch.n_blocks(), 0);
}

#if defined(USE_MLU)
TEST(KVCacheEstimationTest, Glm5NextSpecVerifyExpandsHybridState) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm5_next")
      .enable_mla(true)
      .kv_lora_rank(8)
      .qk_rope_head_dim(0)
      .index_n_heads(1)
      .index_head_dim(16)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(4)
      .linear_conv_kernel_dim(4)
      .layer_types({"linear_attention",
                    "deepseek_sparse_attention",
                    "linear_attention",
                    "deepseek_sparse_attention"});
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 2;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 5);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 3);
}

TEST(KVCacheEstimationTest, KPoolReservesSpeculativeTailWithRequestState) {
  ModelArgs args = make_linear_attention_args();
  args.index_n_heads(1).index_head_dim(16).index_kpool(4).index_kpool_compress(
      true);
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.dtype = torch::kBFloat16;
  options.num_speculative_tokens = 5;
  // Fix the request-state capacity; automatic sizing follows the memory budget.
  options.max_linear_state_cache_slots = 8;
  const KVCacheCapacity capacity = estimate_kv_cache_capacity(args, options);
  // Two recurrent layers and two indexer layers. Each tail keeps K + W
  // BF16 key/gate rows; W includes the verify base token and prelaunch room.
  EXPECT_EQ(capacity.num_linear_state_blocks(), 10);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 25600);
  EXPECT_EQ(
      capacity.num_linear_attention_layers() * capacity.linear_slot_size(),
      512);
  EXPECT_EQ(capacity.num_indexer_layers() * capacity.kpool_tail_slot_size(),
            2048);
  EXPECT_EQ(capacity.kpool_tail_len(), 16);
  EXPECT_LE(capacity.n_blocks() * 16 * (2 * 128 + 2 * 8) +
                capacity.linear_cache_size_in_bytes(),
            options.cache_size_in_bytes);
  args.num_nextn_predict_layers(2);
  const KVCacheCapacity speculative = estimate_kv_cache_capacity(args, options);
  EXPECT_EQ(speculative.num_full_attention_layers(), 2);
  EXPECT_EQ(speculative.num_indexer_layers(), 2);
  EXPECT_EQ(speculative.linear_cache_size_in_bytes(),
            capacity.linear_cache_size_in_bytes());
  EXPECT_EQ(speculative.n_blocks(), capacity.n_blocks());
}
#endif

TEST(KVCacheEstimationTest, LinearStateCapacityVariants) {
  struct TestCase {
    const char* name;
    int64_t head_dim;
    int64_t cache_size_in_bytes;
    int64_t block_size;
    int64_t n_local_kv_heads;
    int64_t max_seqs_per_batch;
    bool enable_prefix_cache;
    int64_t max_linear_state_cache_slots;
    int64_t expected_num_linear_state_blocks;
    int64_t min_num_linear_state_blocks;
  };

  const std::vector<TestCase> test_cases = {
      {"PrefixCacheGrowsLinearStateCheckpointPool",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/-1,
       /*min_num_linear_state_blocks=*/202},
      {"PrefixCacheUsesLinearStateMemoryRatio",
       /*head_dim=*/1,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/1,
       /*n_local_kv_heads=*/1,
       /*max_seqs_per_batch=*/8,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/970,
       /*min_num_linear_state_blocks=*/-1},
      {"NoPrefixCacheCapsLinearStateBlocksByBudget",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/100000,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/229,
       /*min_num_linear_state_blocks=*/-1},
      {"UnlimitedConcurrencyUsesLinearStateMemoryRatio",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/0,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/229,
       /*min_num_linear_state_blocks=*/-1},
      {"ExplicitLinearStateSlotsOverrideAutoSizing",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/32,
       /*expected_num_linear_state_blocks=*/34,
       /*min_num_linear_state_blocks=*/-1},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    ModelArgs model_args = make_linear_attention_args(test_case.head_dim);
    KVCacheEstimateOptions options = make_linear_attention_options();
    options.cache_size_in_bytes = test_case.cache_size_in_bytes;
    options.block_size = test_case.block_size;
    options.n_local_kv_heads = test_case.n_local_kv_heads;
    options.max_seqs_per_batch = test_case.max_seqs_per_batch;
    options.enable_prefix_cache = test_case.enable_prefix_cache;
    options.max_linear_state_cache_slots =
        test_case.max_linear_state_cache_slots;

    KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

    if (test_case.expected_num_linear_state_blocks >= 0) {
      EXPECT_EQ(capacity.num_linear_state_blocks(),
                test_case.expected_num_linear_state_blocks);
    }
    if (test_case.min_num_linear_state_blocks >= 0) {
      EXPECT_GT(capacity.num_linear_state_blocks(),
                test_case.min_num_linear_state_blocks);
    }
  }
}

#if defined(USE_NPU)
TEST(KVCacheEstimationTest, CapsQwen35PhysicalLinearStateSlotsForMegaGdn) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.model_type("qwen3_5");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.cache_size_in_bytes = 64LL << 30;
  options.enable_prefix_cache = true;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_linear_state_blocks(), 1024);
}

TEST(KVCacheEstimationTest, ClampsDerivedQwen35SlotsToMegaGdnLimit) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.model_type("qwen3_5");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.cache_size_in_bytes = 64LL << 30;
  options.enable_prefix_cache = true;
  options.instance_role = InstanceRole::DECODE;
  options.max_linear_state_cache_slots = 0;
  options.max_seqs_per_batch = 682;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_linear_state_blocks(), 1024);
}

TEST(KVCacheEstimationTest, ClampsDerivedLinearStateSlotsToCacheBudget) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/16);
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.instance_role = InstanceRole::DECODE;
  options.enable_prefix_cache = true;
  options.max_linear_state_cache_slots = 0;
  options.max_seqs_per_batch = 100000;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  options.max_seqs_per_batch = 0;
  const KVCacheCapacity ratio_capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_linear_state_blocks(),
            ratio_capacity.num_linear_state_blocks());
  EXPECT_EQ(capacity.num_linear_state_blocks(), 229);
  EXPECT_GT(capacity.n_blocks(), 0);
}

TEST(KVCacheEstimationTest, RejectsQwen35ExplicitSlotsBeyondMegaGdnLimit) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.model_type("qwen3_5");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.cache_size_in_bytes = 64LL << 30;
  options.enable_prefix_cache = true;
  options.max_linear_state_cache_slots = 1023;

  EXPECT_DEATH((void)estimate_kv_cache_capacity(model_args, options),
               "MegaGdn supports at most 1024 physical linear-state slots");
}
#endif

TEST(KVCacheEstimationTest, Qwen35MtpExpandsConvStateLen) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 1;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 3);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 2);
  EXPECT_EQ(capacity.linear_slot_size(), 448);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 189056);
}

TEST(KVCacheEstimationTest, Qwen35TextMtpUsesSsmCheckpointStride) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5_text")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 1;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 3);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 2);
  EXPECT_EQ(capacity.linear_slot_size(), 448);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 189056);
}

#if defined(USE_NPU)
TEST(KVCacheEstimationTest, Glm5MtpPreallocatesSpeculativeCheckpoints) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.model_type("glm5_next");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.num_speculative_tokens = 3;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 5);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 4);
  EXPECT_EQ(capacity.linear_slot_size(), 832);
}

TEST(KVCacheEstimationTest, Glm5PrefillDoesNotReserveMtpCheckpoints) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.model_type("glm5_next");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.num_speculative_tokens = 3;
  options.instance_role = InstanceRole::PREFILL;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 2);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 1);
  EXPECT_LT(capacity.linear_slot_size(), 832);
}
#endif

TEST(KVCacheEstimationTest, EstimatesDeepSeekV4Pools) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes =
      2818048 + /*sixteen_additional_swa_blocks=*/16 * 90112;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 35);
#if defined(USE_MLU)
  constexpr int64_t kKvBlockSize = kDsv4FixedBlockSize;
#else
  constexpr int64_t kKvBlockSize = kDsv4NpuBlockSize;
#endif
  EXPECT_EQ(capacity.block_size(), kKvBlockSize);
  constexpr int64_t kSwaBytesPerBlock =
      128 * 16 * 4 + 128 * (16 * 4 + 2 * 16 * 4 * 2 + 2 * 8 * 4 * 2) +
      128 * (16 * 4 + 16 * 4 * 2);
#if defined(USE_MLU)
  constexpr int64_t kIndexBytes = 4;
  constexpr int64_t kScaleBytes = 0;
  constexpr int64_t kExpectedUnits = 352;
#else
  constexpr int64_t kIndexBytes = 1;
  constexpr int64_t kScaleBytes = 2;
  constexpr int64_t kExpectedUnits = 28;
#endif
  const int64_t token_unit_bytes =
      dsv4_physical_dim(kKvBlockSize, kDsv4C4CompressRatio) *
          (16 * 4 + 8 * kIndexBytes + kScaleBytes) +
      dsv4_physical_dim(kKvBlockSize, kDsv4C128CompressRatio) * 16 * 4;
  const int64_t expected_units =
      (options.cache_size_in_bytes - 35 * kSwaBytesPerBlock) / token_unit_bytes;
  EXPECT_EQ(expected_units, kExpectedUnits);
  EXPECT_EQ(capacity.n_blocks(), expected_units);
}

TEST(KVCacheEstimationTest, DeepSeekV4RejectsBudgetWithoutCompressedCacheUnit) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({1, 4, 128});

  constexpr int64_t kSwaCount = 7;
  constexpr int64_t kSwaBytesPerBlock =
      /*c1=*/128 * 16 * 4 +
      /*c4=*/128 * (16 * 4 + 2 * 16 * 4 * 2 + 2 * 8 * 4 * 2) +
      /*c128=*/128 * (16 * 4 + 16 * 4 * 2);
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes =
      kSwaCount * kSwaBytesPerBlock + /*remaining_bytes=*/1;
  options.block_size = 128;
  options.max_seqs_per_batch = 1;
  options.max_tokens_per_batch = 384;
  options.max_tokens_per_chunk_for_prefill = 128;

  EXPECT_DEATH(
      estimate_kv_cache_capacity(model_args, options),
      "minimum DSV4 SWA cache leaves insufficient memory for one compressed "
      "cache unit");
}

TEST(KVCacheEstimationTest, DeepSeekV4PdPrefillUsesBatchTokenCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Four sequences retain three window blocks each, plus four burst blocks,
  // four per-sequence tail blocks, and two guard blocks.
  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest, DeepSeekV4MixUsesBatchTokenCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 16384;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.instance_role = InstanceRole::MIX;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 146);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4PdPrefillWithoutChunkingUsesBatchCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 129;
  options.enable_chunked_prefill = false;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Chunking does not affect SWA sizing.
  EXPECT_EQ(capacity.swa_count(), 22);
}

// The SWA ring backs both the live prefill burst and the rows of a published
// prefix-cache checkpoint. The fair per-DP token budget only bounds the
// former, so a ring sized from the per-DP share alone recycles checkpoint rows
// before the next request can probe them and turns every cross-request prefix
// hit into a full re-prefill. The ring must keep one publish unit of rows.
TEST(KVCacheEstimationTest,
     DeepSeekV4PdPrefillFairBudgetKeepsPrefixCachePublishUnit) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 64 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 4096;
  options.max_tokens_per_chunk_for_prefill = 1024;
  options.enable_prefix_cache = true;
  options.enable_dp_fair_token_budget = true;
  options.dp_size = 4;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // The fair budget funds eight SWA blocks. A checkpoint can require more,
  // depending on the platform's logical KV block size.
  switch (capacity.block_size()) {
    case kDsv4NpuBlockSize:
      EXPECT_EQ(capacity.swa_count(), 34);  // 18 + 16 blocks for publication.
      break;
    case kDsv4FixedBlockSize:
      EXPECT_EQ(capacity.swa_count(), 26);  // 18 + 8 blocks for the burst.
      break;
    default:
      FAIL() << "Missing SWA capacity expectation for KV block size "
             << capacity.block_size();
  }

  // Without a published KV checkpoint there is nothing to retain, so the
  // per-DP share still sizes the ring.
  KVCacheEstimateOptions no_prefix_options = options;
  no_prefix_options.enable_prefix_cache = false;
  const KVCacheCapacity no_prefix_capacity =
      estimate_kv_cache_capacity(model_args, no_prefix_options);
  EXPECT_EQ(no_prefix_capacity.swa_count(), 26);
}

// Same guard, replayed with the shipped DeepSeek-V4-Flash-0731-w8a8 shape and
// the exact serving parameters of
// deepseek_v4_xllm_perf_prefix_i30000_o1024_bs1_dp4_ep16, so this test predicts
// the startup log line the CICD run prints:
//   Initializing DSV4 kv cache with shape: [swa_count=66, ...]
// swa_count is a function of the window, max_seqs_per_batch, the per-step
// token bound and the logical KV span -- it does not depend on the memory
// budget -- so the two budget cases below must agree.
TEST(KVCacheEstimationTest, DeepSeekV4FlashPrefillSwaRingMatchesShippedShape) {
  ModelArgs model_args;
  // compress_ratios from DeepSeek-V4-Flash-0731-w8a8/config.json (46 entries;
  // only the 43 real layers are read).
  model_args.model_type("deepseek_v4")
      .n_layers(43)
      .head_dim(512)
      .index_head_dim(128)
      .index_n_heads(64)
      .window_size(128)
      .compress_ratios({0, 0,   4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                        4, 128, 4, 128, 4, 128, 4, 0,   0, 0});

  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 64LL * 1024 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 16;
  options.max_tokens_per_batch = 16384;
  options.max_tokens_per_chunk_for_prefill = 4096;
  options.enable_chunked_prefill = true;
  options.enable_prefix_cache = true;
  options.enable_dp_fair_token_budget = true;
  options.dp_size = 4;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // window 128 / block 128 -> one window block per sequence, 16 sequences;
  // 32 burst blocks for the fair share 16384 / 4; 16 tail + 2 guard.
  EXPECT_EQ(capacity.swa_count(), 66);
  EXPECT_GT(capacity.n_blocks(), 0);
}

TEST(KVCacheEstimationTest, DeepSeekV4PdDecodeUsesBatchTokenCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.instance_role = InstanceRole::DECODE;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4PdDecodeSwaCapacityIgnoresScheduleOverlap) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 1;
  options.max_tokens_per_batch = 385;
  options.num_speculative_tokens = 65;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::DECODE;

  options.enable_schedule_overlap = false;
  const KVCacheCapacity non_overlap_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(non_overlap_capacity.swa_count(), 8);

  options.enable_schedule_overlap = true;
  const KVCacheCapacity overlap_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(overlap_capacity.swa_count(), 8);
}

TEST(KVCacheEstimationTest, DeepSeekV4PrefixCacheKeepsOperationalSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 128 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  ASSERT_GT(capacity.n_blocks(), 0);
  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4RealisticMixBudgetRetainsCompressedPools) {
  std::vector<int32_t> compress_ratios{0, 0};
  compress_ratios.reserve(43);
  for (int32_t layer_id = 2; layer_id < 43; ++layer_id) {
    compress_ratios.emplace_back(layer_id % 2 == 0 ? 4 : 128);
  }

  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(43)
      .head_dim(512)
      .index_head_dim(128)
      .window_size(128)
      .max_seq_len(1048576)
      .compress_ratios(std::move(compress_ratios));

  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = int64_t{16} * 1024 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 10;
  options.max_tokens_per_batch = 10240;
  options.max_tokens_per_chunk_for_prefill = 2048;
  options.enable_prefix_cache = true;
  options.instance_role = InstanceRole::MIX;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 102);
  EXPECT_GT(capacity.n_blocks(), 0);
}

TEST(KVCacheEstimationTest, DeepSeekV4DecodeKeepsOperationalSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 128 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::DECODE;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
  EXPECT_GT(capacity.n_blocks(), 0);
}

TEST(KVCacheEstimationTest, DeepSeekV4SwaOnlyPrefixKeepsOperationalSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  constexpr int64_t kSwaBytesPerBlock = 3 * 128 * 16 * 4;
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 43 * kSwaBytesPerBlock;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
  EXPECT_EQ(capacity.n_blocks(), 1);
  EXPECT_FALSE(capacity.has_kv_cache());
}

TEST(KVCacheEstimationTest, EstimatesDeepSeekV4DSparkSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 2818048;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 35);
  EXPECT_EQ(capacity.n_blocks(), 1);
  EXPECT_FALSE(capacity.has_kv_cache());
}

TEST(KVCacheEstimationTest,
     DeepSeekV4SpeculativeBudgetIncludesDSparkDraftSwaPool) {
  ModelArgs target_args;
  target_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});
  ModelArgs draft_args;
  draft_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  KVCacheEstimateOptions target_options;
  target_options.dtype = torch::kFloat32;
  target_options.kv_cache_dtype = "auto";
  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(target_options.dtype);
  const int64_t scale_bytes =
      cache_policy.has_indexer_cache_scale ? cache_policy.scale_dtype_size : 0;
#if defined(USE_MLU)
  constexpr int64_t kKvBlockSize = kDsv4FixedBlockSize;
#else
  constexpr int64_t kKvBlockSize = kDsv4NpuBlockSize;
#endif
  const int64_t c4_physical_dim =
      dsv4_physical_dim(kKvBlockSize, kDsv4C4CompressRatio);
  const int64_t c128_physical_dim =
      dsv4_physical_dim(kKvBlockSize, kDsv4C128CompressRatio);
  const int64_t c4_block_bytes =
      c4_physical_dim *
      (16 * 4 + 8 * cache_policy.index_dtype_size + scale_bytes);
  const int64_t c128_block_bytes = c128_physical_dim * 16 * 4;
  const int64_t compressed_unit_bytes = c4_block_bytes + c128_block_bytes;
  constexpr int64_t kTargetSwaBytes = 35 * 90112;
  constexpr int64_t kDraftSwaBytes =
      /*layers=*/3 * /*swa_count=*/35 * /*block_size=*/128 *
      /*head_dim=*/16 * /*float32_bytes=*/4;
  // Reserve both SWA pools and exactly two compressed units for either
  // the model-dtype indexer cache or the quantized indexer cache with scales.
  target_options.cache_size_in_bytes =
      kTargetSwaBytes + kDraftSwaBytes + 2 * compressed_unit_bytes;
  target_options.block_size = 128;
  target_options.max_seqs_per_batch = 4;
  target_options.max_tokens_per_batch = 2176;
  KVCacheEstimateOptions draft_options = target_options;
  draft_options.is_draft_engine = true;

  const KVCacheCapacity target_only_capacity =
      estimate_kv_cache_capacity(target_args, target_options);

  target_options.draft_model_args = &draft_args;
  target_options.draft_options = &draft_options;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(target_args, target_options);

  EXPECT_LE(capacity.cache_size_in_bytes() + kDraftSwaBytes,
            target_options.cache_size_in_bytes);
  EXPECT_EQ(capacity.swa_count(), 35);
  EXPECT_EQ(capacity.n_blocks(), 2);
  EXPECT_EQ(capacity.swa_count(), target_only_capacity.swa_count());
  EXPECT_LT(capacity.n_blocks(), target_only_capacity.n_blocks());
}

TEST_P(DcpIndexerEstimationTest, IncludesAllIndexerBytes) {
  ModelArgs model_args = make_standard_args();
  model_args.index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "auto";
  const KVCacheCapacity ordinary =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(ordinary.index_slot_size(), 32 * expected_factor());

  options.indexer_cache_dtype = "int8";
  const KVCacheCapacity quantized =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(quantized.index_slot_size(), 20 * expected_factor());
  EXPECT_EQ(quantized.slot_size(), ordinary.slot_size());

  model_args.index_n_heads(0);
  const KVCacheCapacity absent =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(absent.index_slot_size(), 0);
}

INSTANTIATE_TEST_SUITE_P(KvSplits,
                         DcpIndexerEstimationTest,
                         ::testing::Values(1, 2, 4, 0));

TEST(KVCacheEstimationTest, EstimatesDeepSeekV41Pools) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v41")
      .n_layers(6)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      .kv_source_layer_ids({2, 4});

  // Cost model (see estimate_dsv41_kv_cache_cost): every layer pays
  // 128 * 16 * 4 = 8192 B of SWA window per block and the single ratio-2
  // kv source adds 2 * 32 * 16 * 4 = 4096 B of fp32 kv/score state, so
  // swa_bytes_per_block = 5 * 8192 + (8192 + 4096) = 53248 B. Each logical
  // compressed unit owns one C2 key block (64 rows), one C1 key block
  // (128 rows); index and scale use the same row count as each key plane.
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";

  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(options.dtype);
  const int64_t index_scale_bytes =
      cache_policy.has_indexer_cache_scale ? cache_policy.scale_dtype_size : 0;
  const int64_t token_unit_bytes =
      (64 + 128) *
      (16 * 4 + 8 * cache_policy.index_dtype_size + index_scale_bytes);
#if defined(USE_MLU)
  EXPECT_FALSE(cache_policy.has_indexer_cache_scale);
  EXPECT_EQ(token_unit_bytes, 18432);
#else
  EXPECT_TRUE(cache_policy.has_indexer_cache_scale);
  EXPECT_EQ(cache_policy.index_dtype, torch::kInt8);
  EXPECT_EQ(cache_policy.scale_dtype, torch::kFloat16);
  EXPECT_EQ(token_unit_bytes, 14208);
#endif

  options.cache_size_in_bytes =
      /*swa=*/27 * 53248 + /*ten units=*/10 * token_unit_bytes;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 27);
  EXPECT_EQ(capacity.n_blocks(), 10);
  EXPECT_EQ(capacity.block_size(), 128);
}

TEST(KVCacheEstimationTest, DeepSeekV41PdPrefillFairBudgetFitsSameTwoGiBLimit) {
  // Minimal C1-source fixture, not a copy of the deployed model. Keep the
  // reported serving limits fixed; only the forwarded DP budget fields vary.
  ModelArgs model_args;
  model_args.model_type("deepseek_v41")
      .n_layers(43)
      .head_dim(512)
      .index_head_dim(128)
      .window_size(128)
      .compress_ratios(std::vector<int32_t>(/*count=*/43, /*value=*/1))
      .kv_source_layer_ids({0});
  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.cache_size_in_bytes = 2LL * 1024 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 64;
  options.max_tokens_per_batch = 32768;
  options.max_tokens_per_chunk_for_prefill = 8192;
  options.enable_chunked_prefill = true;
  options.enable_prefix_cache = true;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;
  options.dp_size = 4;
  options.enable_dp_fair_token_budget = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  // 64 window blocks + 64 per-DP burst blocks + 64 tails + 2 guards.
  EXPECT_EQ(capacity.swa_count(), 194);
  EXPECT_GT(capacity.n_blocks(), 0);

  auto missing_dp_fields = options;
  missing_dp_fields.dp_size = 1;
  missing_dp_fields.enable_dp_fair_token_budget = false;
  // The same inputs without the DP fields reserve 386 SWA blocks. This
  // fixture needs 2,175,533,056 bytes, exceeding the unchanged 2 GiB limit.
  EXPECT_DEATH(estimate_kv_cache_capacity(model_args, missing_dp_fields),
               "minimum DSV4 SWA cache");
}

TEST(KVCacheEstimationTest, DeepSeekV41CountsOnlyKvSourceLayersForTokenGroups) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v41")
      .n_layers(6)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      // Layer 3 (ratio 2) and layer 5 (ratio 1) are reuse layers: they must
      // not grow the TOKEN-group budget.
      .kv_source_layer_ids({2});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Only layer 2 owns history and compressor state. Reuse layers still own
  // windows, but neither a ratio-1 nor a second ratio-2 history allocation.
  const DeepSeekV4CachePolicy policy = get_dsv4_cache_policy(options.dtype);
  const int64_t scale_bytes =
      policy.has_indexer_cache_scale ? policy.scale_dtype_size : 0;
  const int64_t history_block_bytes =
      64 * (16 * 4 + 8 * policy.index_dtype_size + scale_bytes);
  constexpr int64_t kWindowBlockBytes = 6 * 128 * 16 * 4 + 2 * 32 * 16 * 4;
  const int64_t expected_blocks =
      (options.cache_size_in_bytes - capacity.swa_count() * kWindowBlockBytes) /
      history_block_bytes;
  EXPECT_EQ(capacity.n_blocks(), expected_blocks);
}

TEST(KVCacheEstimationTest, DeepSeekV41RejectsBudgetWithoutTokenCacheUnit) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v41")
      .n_layers(6)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      .kv_source_layer_ids({2, 4});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = /*swa=*/27 * 53248 + /*remaining_bytes=*/1;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  EXPECT_DEATH(
      estimate_kv_cache_capacity(model_args, options),
      "minimum DSV4 SWA cache leaves insufficient memory for one compressed "
      "cache unit");
}

TEST(KVCacheEstimationTest, SpeculativeDecodeUsesDraftShapeForDFlash2) {
  // A DFlash2 draft whose per-slot KV is wider than the target's (a hybrid
  // target with narrow full-attention layers) must skip the draft-slot<=
  // target-slot guard and be sized by the draft's own geometry, matching
  // DFlash2WorkerImpl::draft_kv_cache_shape allocations. Asserted through
  // the cost estimator the engine calls; the block count is the engine's
  // own division of the same numbers (cache - linear) / combined cost.
  KVCacheCapacity target_capacity;
  target_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(2)
      .scale_slot_size(1)
      .index_cache_sizes(3, 30)
      .num_full_attention_layers(2)
      .num_indexer_layers(1);
  KVCacheCapacity draft_capacity;
  draft_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(4)
      .scale_slot_size(1)
      .index_cache_sizes(2, 20)
      .num_indexer_layers(1)
      .n_layers(2);

  const SpeculativeFullAttentionBlockCost dflash2_cost =
      estimate_speculative_full_attention_block_cost(
          target_capacity,
          draft_capacity,
          /*draft_body_uses_tp1=*/false,
          /*draft_manages_own_kv_shape=*/true);
  const SpeculativeFullAttentionBlockCost tp1_cost =
      estimate_speculative_full_attention_block_cost(
          target_capacity,
          draft_capacity,
          /*draft_body_uses_tp1=*/true,
          /*draft_manages_own_kv_shape=*/false);
  EXPECT_EQ(dflash2_cost.draft_full_attention_block_size_in_bytes,
            tp1_cost.draft_full_attention_block_size_in_bytes);
  const int64_t blocks =
      (target_capacity.cache_size_in_bytes() -
       target_capacity.linear_cache_size_in_bytes()) /
      (dflash2_cost.target_full_attention_block_size_in_bytes +
       dflash2_cost.draft_full_attention_block_size_in_bytes);
  EXPECT_EQ(blocks, 476);
}

TEST(KVCacheEstimationTest, SpeculativeDecodeSizesIndexCacheByBlockBytes) {
  // The index term of the shared speculative block cost must use
  // index_block_size() (block units), matching
  // standard_full_cache_block_size_in_bytes and the pre-extraction engine
  // formula. Folding it into block_size * index_slot_size() overestimates
  // compressed-tail layouts, where the block value is
  // block_size/kpool*pool_bytes (floor) while the slot value is
  // ceil(pool_bytes/kpool): here index_block_size()=25 differs from
  // block_size * index_slot_size()=30, and the slot-folded form would
  // report 90 instead of 85 bytes per target block.
  KVCacheCapacity target_capacity;
  target_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(2)
      .scale_slot_size(1)
      .index_cache_sizes(3, 25)
      .num_full_attention_layers(2)
      .num_indexer_layers(1);
  KVCacheCapacity draft_capacity;
  draft_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(4)
      .scale_slot_size(1)
      .index_cache_sizes(2, 20)
      .num_indexer_layers(1)
      .n_layers(2);

  const SpeculativeFullAttentionBlockCost cost =
      estimate_speculative_full_attention_block_cost(
          target_capacity,
          draft_capacity,
          /*draft_body_uses_tp1=*/true,
          /*draft_manages_own_kv_shape=*/false);
  EXPECT_EQ(cost.target_full_attention_block_size_in_bytes,
            10 * 2 * (2 + 1) + 1 * 25);
  EXPECT_EQ(cost.draft_full_attention_block_size_in_bytes,
            10 * 2 * (4 + 1) + 1 * 20);
}

TEST(KVCacheEstimationTest, DraftOwnShapeCoversBlockDiffusionWorkers) {
  // The own-shape predicate must match the worker dispatch: DFlash, DFlash2,
  // and DSpark all inherit DraftModelSpecWorkerImpl and build the draft pool
  // through build_draft_kv_cache_shape, so a full-attention draft on a
  // narrow-slot target must neither trip the estimator's draft-slot CHECK
  // nor be sized by the target slot. Workers that reuse the target
  // KVCacheShape (MTP, Eagle3) must stay excluded so the CHECK still guards
  // them.
  EXPECT_TRUE(SpeculativeConfig::draft_manages_own_kv_shape("DFlash"));
  EXPECT_TRUE(SpeculativeConfig::draft_manages_own_kv_shape("DFlash2"));
  EXPECT_TRUE(SpeculativeConfig::draft_manages_own_kv_shape("DSpark"));
  EXPECT_FALSE(SpeculativeConfig::draft_manages_own_kv_shape("MTP"));
  EXPECT_FALSE(SpeculativeConfig::draft_manages_own_kv_shape("Eagle3"));
}

}  // namespace xllm

namespace xllm {
TEST(KVCacheEstimationTest, CompressedPoolsUseExactBlockBytes) {
  // NPU selects this layout from model configuration, not the option alone.
  ModelArgs args;
  args.model_type("glm5_next")
      .n_layers(1)
      .head_dim(128)
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(3)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true);
  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 24;
  options.world_size = 1;
  options.n_local_kv_heads = 1;
  options.max_seqs_per_batch = 1;
  options.kpool_layout = KPoolCacheLayout::COMPRESSED_WITH_TAIL;
  const auto capacity = estimate_kv_cache_capacity(args, options);
  ASSERT_EQ(capacity.kpool_layout(), KPoolCacheLayout::COMPRESSED_WITH_TAIL);
  const auto actual = torch::empty({1, 1, 8, 128}, torch::kBFloat16);
  EXPECT_EQ(capacity.index_block_size(), actual.nbytes());
  const int64_t block_bytes = 24 * capacity.slot_size() + actual.nbytes();
  const int64_t available =
      capacity.cache_size_in_bytes() - capacity.linear_cache_size_in_bytes();
  EXPECT_EQ(capacity.n_blocks(), available / block_bytes);
}
}  // namespace xllm
