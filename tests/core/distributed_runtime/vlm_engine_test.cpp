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

#include "core/distributed_runtime/vlm_engine.h"

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <utility>

#include "core/distributed_runtime/engine_utils.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "framework/block/block_utils.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "runtime/worker_client.h"

namespace xllm {
class VLMEngineCacheTestPeer final {
 public:
  template <typename... Args>
  static auto make_kv_cache_manager(Args&&... args) {
    return VLMEngine::make_kv_cache_manager(std::forward<Args>(args)...);
  }
};

namespace {

BlockManagerPool::HostBlockCounts make_grouped_host_cache_capacities(
    const ModelArgs& args,
    const KVCacheCapacity& capacity,
    double host_blocks_factor) {
  runtime::Options runtime_options;
  runtime_options.host_blocks_factor(host_blocks_factor);
  BlockManagerPool::Options block_options;
  block_options.enable_prefix_cache(true);
  configure_dsv4_block_manager_options(
      args, block_options, capacity, /*max_tokens_per_batch=*/512);
  auto pool = create_kv_cache_manager(block_options,
                                      runtime_options,
                                      capacity,
                                      /*has_gdn_state=*/false,
                                      /*engine=*/nullptr,
                                      /*dp_size=*/1);
  return block_options.host_num_blocks_by_type();
}

TEST(VLMEngineTest, BuildsTypedV4HostCacheCapacities) {
  ModelArgs args;
  args.model_type("deepseek_v4").window_size(512);
  KVCacheCapacity capacity;
  capacity.n_blocks(8)
      .block_size(dsv4_kv_block_size(Platform::is_npu()))
      .swa_count(4)
      .swa_block_size(kDsv4SwaBlockSize);

  const BlockManagerPool::HostBlockCounts host_capacities =
      make_grouped_host_cache_capacities(args, capacity, 2.0);

  ASSERT_EQ(host_capacities.size(), 2u);
  EXPECT_EQ(host_capacities.at(BlockType::KV), 16u);
  EXPECT_EQ(host_capacities.at(BlockType::SWA), Platform::is_npu() ? 64u : 16u);
}

TEST(VLMEngineTest, BuildsTypedV41HostCacheCapacities) {
  ModelArgs args;
  args.model_type("deepseek_v41").window_size(512);
  KVCacheCapacity capacity;
  capacity.n_blocks(8)
      .block_size(kDsv4FixedBlockSize)
      .swa_count(4)
      .swa_block_size(kDsv4SwaBlockSize);

  const BlockManagerPool::HostBlockCounts host_capacities =
      make_grouped_host_cache_capacities(args, capacity, 2.0);

  ASSERT_EQ(host_capacities.size(), 2u);
  EXPECT_EQ(host_capacities.at(BlockType::SWA), 16u);
  EXPECT_EQ(host_capacities.at(BlockType::KV), 16u);
}

TEST(VLMEngineTest, RejectsInvalidTypedHostCapacityArguments) {
  ModelArgs args;
  args.model_type("deepseek_v4").window_size(512);
  KVCacheCapacity capacity;
  capacity.n_blocks(8)
      .block_size(dsv4_kv_block_size(Platform::is_npu()))
      .swa_count(4)
      .swa_block_size(kDsv4SwaBlockSize);
  EXPECT_TRUE(make_grouped_host_cache_capacities(args, capacity, 1.0).empty());
  capacity.block_size(0);
  EXPECT_DEATH(make_grouped_host_cache_capacities(args, capacity, 2.0),
               "kv_cache_cap.block_size\\(\\) > 0");
  capacity.block_size(dsv4_kv_block_size(Platform::is_npu()))
      .n_blocks(0)
      .has_kv_cache(false);
  EXPECT_DEATH(make_grouped_host_cache_capacities(args, capacity, 2.0),
               "Standalone window-only cache does not support Host offload");
}

TEST(VLMEngineTest, GroupedHostCapacityUsesLogicalKvAndSwaSpans) {
  ModelArgs args;
  args.model_type("deepseek_v4").window_size(512).max_seq_len(1024);
  KVCacheCapacity capacity;
  capacity.n_blocks(8)
      .block_size(dsv4_kv_block_size(Platform::is_npu()))
      .swa_count(4)
      .swa_block_size(kDsv4SwaBlockSize);
  runtime::Options runtime_options;
  runtime_options.block_size(128).host_blocks_factor(2.0);
  BlockManagerPool::Options block_options;
  block_options.num_blocks(16)
      .block_size(
          fold_kv_cache_block_size(/*block_size=*/128, /*kv_split_size=*/2))
      .enable_prefix_cache(true);
  configure_dsv4_block_manager_options(args,
                                       block_options,
                                       capacity,
                                       /*max_tokens_per_batch=*/512);
  EXPECT_EQ(block_options.swa_blocks_per_seq(), 4u);
  auto pool =
      VLMEngineCacheTestPeer::make_kv_cache_manager(block_options,
                                                    runtime_options,
                                                    capacity,
                                                    /*has_gdn_state=*/false,
                                                    /*engine=*/nullptr,
                                                    /*dp_size=*/1);
  ASSERT_NE(pool, nullptr);
  const int64_t worker_host_swa_count =
      dsv4_host_swa_block_count(capacity.n_blocks(),
                                runtime_options.host_blocks_factor(),
                                block_options.sliding_window_size(),
                                capacity.swa_block_size(),
                                capacity.block_size());
  EXPECT_EQ(block_options.host_num_blocks_by_type().at(BlockType::SWA),
            worker_host_swa_count);
  EXPECT_EQ(block_options.block_size(), capacity.block_size());
  EXPECT_EQ(worker_host_swa_count, Platform::is_npu() ? 64 : 16);
  EXPECT_EQ(prefill_host_linear_block_count(
                /*host_kv_block_count=*/16,
                /*worker_block_size=*/128,
                /*chunk_size=*/512),
            4);
  EXPECT_DEATH(prefill_host_linear_block_count(
                   /*host_kv_block_count=*/1,
                   /*worker_block_size=*/128,
                   /*chunk_size=*/512),
               "prefill Host linear pool needs more than one slot");
}

TEST(VLMEngineTest, SelectsFlatHostPoolOnlyWhenEnabled) {
  KVCacheCapacity capacity;
  capacity.n_blocks(16).block_size(128);
  runtime::Options runtime_options;
  runtime_options.host_blocks_factor(1.0);
  BlockManagerPool::Options block_options;
  block_options.num_blocks(16).block_size(128).enable_prefix_cache(true);
  auto device_pool =
      VLMEngineCacheTestPeer::make_kv_cache_manager(block_options,
                                                    runtime_options,
                                                    capacity,
                                                    /*has_gdn_state=*/false,
                                                    /*engine=*/nullptr,
                                                    /*dp_size=*/1);
  ASSERT_NE(device_pool, nullptr);
  EXPECT_FALSE(block_options.enable_host_offload());
  EXPECT_TRUE(block_options.host_num_blocks_by_type().empty());

  runtime_options.host_blocks_factor(2.0);
  auto host_pool =
      VLMEngineCacheTestPeer::make_kv_cache_manager(block_options,
                                                    runtime_options,
                                                    capacity,
                                                    /*has_gdn_state=*/false,
                                                    /*engine=*/nullptr,
                                                    /*dp_size=*/1);
  ASSERT_NE(host_pool, nullptr);
  EXPECT_TRUE(block_options.enable_host_offload());
  EXPECT_EQ(block_options.host_num_blocks_by_type().at(BlockType::KV), 32u);
  EXPECT_EQ(block_options.host_num_blocks(), 0u);
  // An empty flat Host map used to build null Host managers on release.
  host_pool->reset_prefix_cache();
}

TEST(VLMEngineTest, VlmHostLinearCapacityUsesPhysicalWorkerBlockSize) {
  auto& scheduler_config = SchedulerConfig::get_instance();
  const int32_t previous_chunk_size =
      scheduler_config.max_tokens_per_chunk_for_prefill();
  auto restore = folly::makeGuard([&scheduler_config, previous_chunk_size]() {
    scheduler_config.max_tokens_per_chunk_for_prefill(previous_chunk_size);
  });
  scheduler_config.max_tokens_per_chunk_for_prefill(512);

  KVCacheCapacity capacity;
  capacity.n_blocks(8).block_size(128).num_linear_state_blocks(16);
  runtime::Options runtime_options;
  runtime_options.backend("vlm").block_size(128).host_blocks_factor(2.0);
  BlockManagerPool::Options block_options;
  block_options.num_blocks(8)
      .block_size(256)
      .host_num_blocks(7)
      .enable_prefix_cache(true)
      .enable_linear_state(true)
      .linear_state_num_slots(16);
  auto pool =
      VLMEngineCacheTestPeer::make_kv_cache_manager(block_options,
                                                    runtime_options,
                                                    capacity,
                                                    /*has_gdn_state=*/true,
                                                    /*engine=*/nullptr,
                                                    /*dp_size=*/1);
  ASSERT_NE(pool, nullptr);
  // Hybrid layouts install typed KV even when a scalar was supplied.
  EXPECT_EQ(block_options.host_num_blocks(), 7u);
  EXPECT_EQ(block_options.host_num_blocks_by_type().at(BlockType::KV), 16u);
  // WorkerImpl passes runtime_options.block_size() to the Host tensor's
  // KVCacheCreateOptions. LINEAR rows must use that physical 128-token width,
  // not the block manager's kv_split-folded 256-token width (which gives 8).
  EXPECT_EQ(block_options.host_num_blocks_by_type().at(BlockType::LINEAR), 4u);
  pool.reset();
  capacity.block_size(256);
  EXPECT_DEATH(
      VLMEngineCacheTestPeer::make_kv_cache_manager(block_options,
                                                    runtime_options,
                                                    capacity,
                                                    /*has_gdn_state=*/true,
                                                    /*engine=*/nullptr,
                                                    /*dp_size=*/1),
      "Host LINEAR physical block size must match worker allocation");
  // LLM and direct callers must reject the same mismatch without relying on
  // the VLM wrapper to validate the worker's physical block width.
  EXPECT_DEATH(create_kv_cache_manager(block_options,
                                       runtime_options,
                                       capacity,
                                       /*has_gdn_state=*/true,
                                       /*engine=*/nullptr,
                                       /*dp_size=*/1),
               "Host LINEAR physical block size must match worker allocation");
}

TEST(EngineUtilsTest, PreservesPreparedFlatHostCapacity) {
  KVCacheCapacity capacity;
  capacity.n_blocks(16).block_size(128);
  runtime::Options runtime_options;
  runtime_options.host_blocks_factor(2.0);
  BlockManagerPool::Options block_options;
  block_options.num_blocks(16)
      .block_size(128)
      .host_num_blocks(32)
      .enable_prefix_cache(true);
  auto pool = create_kv_cache_manager(block_options,
                                      runtime_options,
                                      capacity,
                                      /*has_gdn_state=*/false,
                                      /*engine=*/nullptr,
                                      /*dp_size=*/1);
  ASSERT_NE(pool, nullptr);
  EXPECT_TRUE(block_options.enable_host_offload());
  EXPECT_EQ(block_options.host_num_blocks(), 32u);
  EXPECT_TRUE(block_options.host_num_blocks_by_type().empty());
  pool->reset_prefix_cache();
}

TEST(EngineUtilsTest, MtpDraftRetainsExistingDevicePoolPolicy) {
  KVCacheCapacity capacity;
  capacity.n_blocks(16).block_size(128);
  runtime::Options runtime_options;
  runtime_options.host_blocks_factor(2.0)
      .enable_kvcache_store(true)
      .is_draft_engine(true)
      .speculative_algorithm("MTP");
  BlockManagerPool::Options block_options;
  block_options.num_blocks(16)
      .block_size(128)
      .host_num_blocks(32)
      .enable_prefix_cache(true);
  auto pool = create_kv_cache_manager(block_options,
                                      runtime_options,
                                      capacity,
                                      /*has_gdn_state=*/false,
                                      /*engine=*/nullptr,
                                      /*dp_size=*/1);
  ASSERT_NE(pool, nullptr);
  EXPECT_FALSE(block_options.enable_prefix_cache());
  EXPECT_FALSE(block_options.enable_host_offload());
  EXPECT_TRUE(block_options.host_num_blocks_by_type().empty());
}

TEST(EngineUtilsTest, HostValidationOptionsIncludeRuntimeConfiguration) {
  ModelArgs args;
  args.model_type("deepseek_v41");
  KVCacheCapacity capacity;
  capacity.n_blocks(16);
  KVCacheShape shape;
  runtime::Options runtime_options;
  runtime_options.host_blocks_factor(2.0)
      .enable_prefix_cache(false)
      .enable_disagg_pd(true)
      .enable_kvcache_store(true)
      .instance_role(InstanceRole::DECODE)
      .kv_cache_dtype("auto");
  const auto options = build_host_cache_validation_options(
      runtime_options, args, shape, capacity, /*kv_split_size_eff=*/1);
  EXPECT_EQ(options.host_blocks_factor, 2.0);
  EXPECT_EQ(options.device_block_count, 16);
  EXPECT_FALSE(options.enable_prefix_cache);
  EXPECT_TRUE(options.enable_disagg_pd);
  EXPECT_TRUE(options.enable_kvcache_store);
  EXPECT_EQ(options.instance_role, InstanceRole::DECODE);
  EXPECT_EQ(options.has_key_cache_shape, shape.has_key_cache_shape());
  EXPECT_TRUE(options.supports_grouped_cache_offload);
  EXPECT_EQ(options.indexer_cache_dtype,
            KVCacheConfig::get_instance().indexer_cache_dtype());
}

class RejectPrematurePrefetchWorker final : public WorkerClient {
 public:
  void prefetch_from_storage(
      const std::shared_ptr<const StoragePrefetchRequest>& /*request*/,
      std::shared_ptr<PrefetchResult> /*result*/,
      size_t /*worker_index*/) override {
    LOG(FATAL) << "Prefetch dispatched before bounds validation";
  }
};

TEST(EngineUtilsTest, RejectsPrefetchIndicesBeforeAnyWorkerDispatch) {
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.emplace_back(std::make_shared<RejectPrematurePrefetchWorker>());
  auto request = std::make_shared<StoragePrefetchRequest>();
  PrefetchUnit unit;
  BlockTransferInfo info(/*src_block_id=*/0, /*dst_block_id=*/1);
  info.transfer_type = TransferType::G2H;
  unit.gated_blocks.emplace_back(info);
  request->units.emplace_back(std::move(unit));
  // Index 0 is valid: the sentinel must not be contacted before index 1 fails.
  EXPECT_DEATH(dispatch_storage_prefetch(
                   clients,
                   std::vector<uint32_t>{0, 1},
                   request,
                   /*batch_size=*/1,
                   /*configured_timeout_ms=*/0,
                   [] { return false; },
                   [](PrefetchSummary /*unused*/) {}),
               "Prefetch worker index exceeds client count");
}

TEST(VLMEngineTest, DpGroupWorkerIndicesPartitionWorkersExactlyOnce) {
  constexpr uint32_t kCpSize = 2;
  constexpr uint32_t kTpSize = 2;
  constexpr uint32_t kLocalSize = kCpSize * kTpSize;
  const auto dp0 = dp_group_worker_indices(0, 2, kLocalSize, 8);
  const auto dp1 = dp_group_worker_indices(1, 2, kLocalSize, 8);
  EXPECT_EQ(dp0, (std::vector<uint32_t>{0, 1, 2, 3}));
  EXPECT_EQ(dp1, (std::vector<uint32_t>{4, 5, 6, 7}));
  auto combined = dp0;
  combined.insert(combined.end(), dp1.begin(), dp1.end());
  std::sort(combined.begin(), combined.end());
  EXPECT_EQ(combined, (std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_DEATH(dp_group_worker_indices(
                   /*dp_rank=*/2,
                   /*dp_size=*/2,
                   /*local_worker_count=*/4,
                   /*worker_count=*/8),
               "dp_rank < dp_size");
  EXPECT_DEATH(dp_group_worker_indices(
                   /*dp_rank=*/0,
                   /*dp_size=*/2,
                   /*local_worker_count=*/4,
                   /*worker_count=*/7),
               "DP-local worker width does not partition the worker group");
}

}  // namespace
}  // namespace xllm
