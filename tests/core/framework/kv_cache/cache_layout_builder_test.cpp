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
#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/util/scope_guard.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/host_cache_plan.h"
#include "framework/kv_cache/kv_cache_estimation.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_shard_layout.h"
#include "framework/kv_cache/prefill_layerwise_kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/process_group.h"
#include "layers/common/kv_shard_batch_metadata.h"
#include "platform/platform.h"
#include "worker.pb.h"

namespace xllm {

namespace {

class PreparationProcessGroup final : public ProcessGroup {
 public:
  PreparationProcessGroup(int32_t rank, int32_t world_size, bool peer_prepared)
      : ProcessGroup(rank, world_size, torch::Device(torch::kCPU)),
        peer_prepared_(peer_prepared) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++allgather_count_;
    torch::Tensor prepared = torch::ones({world_size()}, input.options());
    if (!peer_prepared_) {
      prepared[0] = 0;
    }
    prepared[rank()] = input[0];
    return prepared;
  }

  void peer_prepared(bool value) { peer_prepared_ = value; }
  int32_t allgather_count() const { return allgather_count_; }

 private:
  bool peer_prepared_;
  int32_t allgather_count_ = 0;
};

class HostPoolProcessGroup final : public ProcessGroup {
 public:
  class Exchange final {
   public:
    explicit Exchange(int32_t world_size = 2)
        : world_size(world_size), ready(world_size), inputs(world_size) {}

    const int32_t world_size;
    std::barrier<> ready;
    std::vector<std::array<torch::Tensor, 3>> inputs;
  };

  HostPoolProcessGroup(int32_t rank, Exchange& exchange)
      : ProcessGroup(rank, exchange.world_size, torch::Device(torch::kCPU)),
        exchange_(exchange) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    const size_t phase = phase_++;
    exchange_.inputs.at(rank()).at(phase) = input;
    exchange_.ready.arrive_and_wait();
    std::vector<torch::Tensor> gathered;
    gathered.reserve(world_size());
    for (const auto& rank_inputs : exchange_.inputs) {
      gathered.emplace_back(rank_inputs.at(phase));
    }
    return torch::stack(gathered);
  }

 private:
  Exchange& exchange_;
  size_t phase_ = 0;
};

TEST(CacheLayoutBuilderTest,
     PrefillPreparationRejectsEveryRankAndPreservesHostHistory) {
  proto::KVCacheShape proto_shape;
  for (int64_t dim : {4, 2, 1, 8}) {
    proto_shape.add_key_cache_shape(dim);
  }
  const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(2);
  // TP=2, CP=2: rank 0 rejects in the first TP cohort. Both ranks in the
  // other CP cohort must reject too, preserving each worker's Host history.
  constexpr int32_t kWorldSize = 4;
  std::array<std::vector<KVCache>, kWorldSize> caches;
  std::array<std::unique_ptr<PrefillLayerwiseKVCache>, kWorldSize> owners;
  std::array<std::unique_ptr<PreparationProcessGroup>, kWorldSize> groups;
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    owners[rank] =
        std::make_unique<PrefillLayerwiseKVCache>(shape,
                                                  options,
                                                  /*num_buffers=*/1,
                                                  /*host_memory_bytes=*/4096,
                                                  caches[rank]);
    for (int32_t layer = 0; layer < 2; ++layer) {
      owners[rank]->host_caches()[layer].get_k_cache().fill_(rank * 10 + layer);
    }
    // A different Host value proves that preparation and prefix reuse do
    // not reload resident layer 0 from its mirror.
    caches[rank][0].get_k_cache().fill_(rank * 10 + 100);
    groups[rank] = std::make_unique<PreparationProcessGroup>(
        rank, kWorldSize, /*peer_prepared=*/rank == 0);
  }
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.attention.host.block_tables = torch::tensor({{1}}, torch::kInt32);
  params.attention.host.kv_seq_lens = {2};
  params.attention.host.q_seq_lens = {1};
  ModelInputParams invalid = params;
  invalid.attention.host.block_tables = torch::tensor({{-1}}, torch::kInt32);
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    ASSERT_FALSE(owners[rank]->begin(rank == 0 ? invalid : params,
                                     /*block_size=*/2,
                                     /*compute_stream=*/nullptr,
                                     /*expect_push=*/true,
                                     groups[rank].get()));
    EXPECT_EQ(groups[rank]->allgather_count(), 1);
    for (int32_t layer = 0; layer < 2; ++layer) {
      const torch::Tensor& history =
          owners[rank]->host_caches()[layer].get_k_cache();
      EXPECT_TRUE(history.eq(rank * 10 + layer).all().item<bool>());
    }
    EXPECT_TRUE(
        caches[rank][0].get_k_cache().eq(rank * 10 + 100).all().item<bool>());
    groups[rank]->peer_prepared(true);
    ASSERT_TRUE(owners[rank]->begin(params,
                                    /*block_size=*/2,
                                    /*compute_stream=*/nullptr,
                                    /*expect_push=*/false,
                                    groups[rank].get()));
    EXPECT_EQ(groups[rank]->allgather_count(), 2);
    for (int32_t layer = 0; layer < 2; ++layer) {
      ASSERT_TRUE(owners[rank]->synchronize_layer(layer));
      EXPECT_TRUE(caches[rank][layer]
                      .get_k_cache()[1]
                      .eq(layer == 0 ? rank * 10 + 100 : rank * 10 + layer)
                      .all()
                      .item<bool>());
    }
    ASSERT_TRUE(owners[rank]->finish());
    EXPECT_TRUE(owners[rank]
                    ->host_caches()[0]
                    .get_k_cache()[1]
                    .eq(rank * 10 + 100)
                    .all()
                    .item<bool>());
    EXPECT_TRUE(owners[rank]
                    ->host_caches()[0]
                    .get_k_cache()[0]
                    .eq(rank * 10)
                    .all()
                    .item<bool>());
  }
}

TEST(CacheLayoutBuilderTest,
     PrefillLayerwiseBuffersPreservePrefixAndChunkHistory) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // B=1, B=2 and B=N cover one shared slot, pipelined shared slots and the
  // independent-buffer control. Layer 0 stays resident in every layout.
  for (int32_t buffers : {1, 2, 5}) {
    const int32_t shared_buffers = std::min(buffers, 4);
    proto::KVCacheShape proto_shape;
    for (int64_t dim : {6, 4, 1, 8}) {
      proto_shape.add_key_cache_shape(dim);
      proto_shape.add_value_cache_shape(dim);
    }
    for (int64_t dim : {6, 4, 1, 4}) {
      proto_shape.add_index_cache_shape(dim);
    }
    for (int64_t dim : {6, 4, 1, 1}) {
      proto_shape.add_index_cache_scale_shape(dim);
    }
    const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
    KVCacheCreateOptions options;
    options.device(torch::Device(torch::kCPU))
        .dtype(torch::kFloat32)
        .num_layers(5)
        .enable_lighting_indexer(true)
        .enable_indexer_cache_quant(true);
    std::vector<KVCache> caches;
    PrefillLayerwiseKVCache offload(shape,
                                    options,
                                    buffers,
                                    /*host_memory_bytes=*/1024 * 1024,
                                    caches);
    ASSERT_EQ(caches.size(), 5);
    EXPECT_EQ(offload.device_bytes() * 5,
              offload.host_bytes() * (1 + shared_buffers));
    EXPECT_TRUE(offload.host_caches()[0].get_k_cache().defined());
    EXPECT_NE(offload.host_caches()[0].get_k_cache().data_ptr(),
              caches[0].get_k_cache().data_ptr());
    for (int64_t layer = 1; layer < 5; ++layer) {
      EXPECT_NE(caches[0].get_k_cache().data_ptr(),
                caches[layer].get_k_cache().data_ptr());
      EXPECT_NE(caches[0].get_index_cache().data_ptr(),
                caches[layer].get_index_cache().data_ptr());
    }
    if (shared_buffers < 4) {
      EXPECT_EQ(caches[1].get_k_cache().data_ptr(),
                caches[1 + shared_buffers].get_k_cache().data_ptr());
      EXPECT_EQ(caches[1].get_index_cache().data_ptr(),
                caches[1 + shared_buffers].get_index_cache().data_ptr());
      EXPECT_NE(
          offload.host_caches()[1].get_k_cache().data_ptr(),
          offload.host_caches()[1 + shared_buffers].get_k_cache().data_ptr());
    }
    const std::vector<int64_t> blocks{1, 3};
    ModelInputParams params;
    params.meta.batch_forward_type = BatchForwardType::PREFILL;
    params.attention.host.block_tables = torch::tensor({{1, 3}}, torch::kInt32);
    params.attention.host.kv_seq_lens = {6};
    params.attention.host.q_seq_lens = {6};
    for (int64_t layer = 0; layer < 5; ++layer) {
      offload.host_caches()[layer].get_k_cache()[1].fill_(7);
    }
    ASSERT_TRUE(
        offload.begin(params, /*block_size=*/4, /*compute_stream=*/nullptr));
    // A fresh sequence has no prefix rows, so its whole read set is skipped
    // even though every block is a write block.
    for (int64_t layer = 0; layer < 5; ++layer) {
      EXPECT_TRUE(caches[layer].get_k_cache()[1].eq(0).all().item<bool>());
    }
    for (int64_t layer = 0; layer < 5; ++layer) {
      ASSERT_TRUE(offload.synchronize_layer(layer));
      for (const KVCacheTensor& tensor : caches[layer].get_cache_tensors()) {
        tensor.tensor[1].fill_(layer + 1);
        tensor.tensor[3].fill_(layer + 11);
      }
    }
    ASSERT_TRUE(offload.finish());
    // A prefix hit reads layer 0 directly from its resident Device cache and
    // reloads later layers using the same scheduler block ids. The second
    // chunk writes only block 3; its old rows and full block 1 must survive.
    params.attention.host.kv_seq_lens = {8};
    params.attention.host.q_seq_lens = {2};
    ASSERT_TRUE(
        offload.begin(params, /*block_size=*/4, /*compute_stream=*/nullptr));
    for (int64_t layer = 0; layer < 5; ++layer) {
      ASSERT_TRUE(offload.synchronize_layer(layer));
      for (const KVCacheTensor& tensor : caches[layer].get_cache_tensors()) {
        EXPECT_TRUE(tensor.tensor[1].eq(layer + 1).all().item<bool>());
        EXPECT_TRUE(tensor.tensor[3].eq(layer + 11).all().item<bool>());
        tensor.tensor[3].slice(/*dim=*/0, /*start=*/2).fill_(layer + 21);
      }
    }
    ASSERT_TRUE(offload.finish());
    for (int64_t layer = 0; layer < 5; ++layer) {
      const KVCache& history = offload.host_caches()[layer];
      for (const KVCacheTensor& tensor : history.get_cache_tensors()) {
        EXPECT_TRUE(tensor.tensor[1].eq(layer + 1).all().item<bool>());
        EXPECT_TRUE(
            tensor.tensor[3].slice(0, 0, 2).eq(layer + 11).all().item<bool>());
        EXPECT_TRUE(
            tensor.tensor[3].slice(0, 2).eq(layer + 21).all().item<bool>());
        EXPECT_TRUE(tensor.tensor[5].eq(0).all().item<bool>());
      }
    }
    // Invalid ids fail before any DMA or cache-state mutation. A following
    // valid batch can still run and read the previous chunks intact.
    EXPECT_FALSE(offload.begin({6}, {6}, /*compute_stream=*/nullptr));
    EXPECT_FALSE(offload.begin({1}, {-1}, /*compute_stream=*/nullptr));
    params.block_copy.swap_blocks.emplace_back(1, 3);
    EXPECT_FALSE(
        offload.begin(params, /*block_size=*/4, /*compute_stream=*/nullptr));
    params.block_copy.swap_blocks.clear();
    params.attention.host.kv_seq_lens = {9};
    EXPECT_FALSE(
        offload.begin(params, /*block_size=*/4, /*compute_stream=*/nullptr));
    params.attention.host.kv_seq_lens = {8};
    ASSERT_TRUE(offload.begin(blocks, {}, /*compute_stream=*/nullptr));
    for (int64_t layer = 0; layer < 5; ++layer) {
      ASSERT_TRUE(offload.synchronize_layer(layer));
      EXPECT_TRUE(caches[layer]
                      .get_k_cache()[3]
                      .slice(0, 2)
                      .eq(layer + 21)
                      .all()
                      .item<bool>());
    }
    ASSERT_TRUE(offload.finish());
    // A PUSH reader keeps old slots alive. Already-loaded layers must still
    // advance; only the layer that needs an occupied slot waits for its send.
    ASSERT_TRUE(offload.begin(blocks,
                              {},
                              /*compute_stream=*/nullptr,
                              /*expect_push=*/true));
    for (int64_t layer = 0; layer <= shared_buffers; ++layer) {
      ASSERT_TRUE(offload.synchronize_layer(layer));
    }
    if (shared_buffers < 4) {
      auto next_layer = std::async(std::launch::async, [&]() {
        return offload.synchronize_layer(1 + shared_buffers);
      });
      EXPECT_EQ(next_layer.wait_for(std::chrono::milliseconds(100)),
                std::future_status::timeout);
      EXPECT_TRUE(caches[0].get_k_cache()[1].eq(1).all().item<bool>());
      // Completion of the resident layer cannot release a shared slot.
      offload.notify_push_completion(0, true);
      EXPECT_EQ(next_layer.wait_for(std::chrono::milliseconds(100)),
                std::future_status::timeout);
      offload.notify_push_completion(1, true);
      ASSERT_TRUE(next_layer.get());
      EXPECT_TRUE(caches[1 + shared_buffers]
                      .get_k_cache()[1]
                      .eq(shared_buffers + 2)
                      .all()
                      .item<bool>());
      for (int64_t layer = shared_buffers + 2; layer < 5; ++layer) {
        offload.notify_push_completion(layer - shared_buffers, true);
        ASSERT_TRUE(offload.synchronize_layer(layer));
      }
    }
    auto pushed =
        std::async(std::launch::async, [&]() { return offload.finish(); });
    EXPECT_EQ(pushed.wait_for(std::chrono::milliseconds(100)),
              std::future_status::timeout);
    offload.notify_push_completion(std::nullopt, true);
    ASSERT_TRUE(pushed.get());

    // Sender failure must release the waiter without allowing any slot reuse
    // or another batch on this owner, even if local D2H has already completed.
    ASSERT_TRUE(offload.begin(blocks,
                              {},
                              /*compute_stream=*/nullptr,
                              /*expect_push=*/true));
    for (int64_t layer = 0; layer <= shared_buffers; ++layer) {
      ASSERT_TRUE(offload.synchronize_layer(layer));
    }
    auto failed_push = std::async(std::launch::async, [&]() {
      return shared_buffers < 4 ? offload.synchronize_layer(1 + shared_buffers)
                                : offload.finish();
    });
    EXPECT_EQ(failed_push.wait_for(std::chrono::milliseconds(100)),
              std::future_status::timeout);
    offload.notify_push_completion(std::nullopt, false);
    EXPECT_FALSE(failed_push.get());
    EXPECT_DEATH(offload.begin(blocks, {}, /*compute_stream=*/nullptr),
                 "failed and cannot be reused");
  }
}

}  // namespace

TEST(CacheLayoutBuilderTest, Dsv4OwnedTensorAllocationsMatchPlannedBudget) {
  for (const bool is_v41 : {false, true}) {
    SCOPED_TRACE(is_v41);
    ModelArgs args;
    args.model_type(is_v41 ? "deepseek_v41" : "deepseek_v4")
        .n_layers(4)
        .head_dim(16)
        .index_head_dim(8)
        .window_size(256)
        .compress_ratios(is_v41 ? std::vector<int32_t>{2, 1, 2, 1}
                                : std::vector<int32_t>{1, 4, 128, 4});
    if (is_v41) {
      // Reuse layers keep their window but do not own shared kv/state.
      args.kv_source_layer_ids({0, 3});
    }
    KVCacheEstimateOptions estimate;
    estimate.block_size = 512;
    estimate.dtype = torch::kBFloat16;
    estimate.cache_size_in_bytes = 4 * 1024 * 1024;
    estimate.max_seqs_per_batch = 2;
    estimate.max_tokens_per_batch = 256;
    const KVCacheCapacity capacity = estimate_kv_cache_capacity(args, estimate);
    EXPECT_EQ(capacity.block_size(),
              dsv4_kv_block_size(!is_v41 && Platform::is_npu()));
    const KVCacheShape shape(capacity, args, /*world_size=*/1);
    proto::KVCacheShape serialized;
    shape.to_proto(&serialized);
    const KVCacheShape worker_shape = KVCacheShape::from_proto(serialized);
    EXPECT_TRUE(worker_shape.has_grouped_cache_layout());
    EXPECT_EQ(
        worker_shape.key_cache_shape(),
        (std::vector<int64_t>{capacity.swa_count(), capacity.n_blocks()}));
    if (!is_v41) {
      proto::KVCacheShape legacy = serialized;
      legacy.add_key_cache_shape(capacity.n_blocks());
      EXPECT_DEATH(KVCacheShape::from_proto(legacy), "Grouped cache requires ");
    }
    KVCacheCreateOptions create;
    create.device(torch::Device(torch::kCPU))
        .dtype(estimate.dtype)
        .model_type(args.model_type())
        .num_layers(args.n_layers())
        .block_size(estimate.block_size)
        .head_dim(args.head_dim())
        .index_head_dim(args.index_head_dim())
        .window_size(args.window_size())
        .compress_ratios(args.compress_ratios())
        .kv_source_layer_ids(args.kv_source_layer_ids());
    std::vector<KVCache> caches;
    allocate_kv_caches(caches, worker_shape, create);
    const HostCachePlan host_plan =
        build_host_cache_plan(caches,
                              shape,
                              create,
                              /*host_blocks_factor=*/2.0,
                              InstanceRole::PREFILL,
                              /*enable_mla=*/true);
    EXPECT_EQ(host_plan.block_size, capacity.block_size());
    EXPECT_EQ(host_plan.window_size, args.window_size());
    EXPECT_EQ(host_plan.window_block_size, dsv4_swa_block_size());
    EXPECT_FALSE(host_plan.kv_tp_replicated);
    EXPECT_EQ(host_plan.block_counts.at(BlockType::KV),
              capacity.n_blocks() * 2);
    EXPECT_EQ(host_plan.block_counts.at(BlockType::SWA),
              dsv4_host_swa_block_count(capacity.n_blocks(),
                                        2.0,
                                        args.window_size(),
                                        dsv4_swa_block_size(),
                                        capacity.block_size()));
    int64_t allocated_bytes = 0;
    int64_t kv_block_bytes = 0;
    for (const KVCache& cache : caches) {
      const BlockTypeTensorMap window_roles =
          cache.get_block_type_tensors(BlockType::SWA);
      const BlockTypeTensorMap kv_roles =
          cache.get_block_type_tensors(BlockType::KV);
      ASSERT_EQ(window_roles.size(), 1u);
      EXPECT_EQ(window_roles.at(KVCacheTensorRole::SWA).data_ptr(),
                cache.get_swa_cache().data_ptr());
      size_t expected_kv_planes = cache.get_k_cache().defined() ? 1 : 0;
      expected_kv_planes += cache.get_index_cache().defined() ? 1 : 0;
      expected_kv_planes += cache.get_indexer_cache_scale().has_value() ? 1 : 0;
      ASSERT_EQ(kv_roles.size(), expected_kv_planes);
      for (const auto& [role, tensor] : kv_roles) {
        EXPECT_TRUE(tensor.defined());
        EXPECT_TRUE(role == KVCacheTensorRole::KEY ||
                    role == KVCacheTensorRole::INDEX ||
                    role == KVCacheTensorRole::INDEX_SCALE);
      }
      for (const KVCacheTensor& owner : cache.get_cache_tensors()) {
        const int64_t bytes = owner.tensor.numel() *
                              static_cast<int64_t>(owner.tensor.element_size());
        allocated_bytes += bytes;
        if (owner.group_id != cache_group_id(BlockType::SWA)) {
          ASSERT_EQ(owner.tensor.size(0), capacity.n_blocks());
          kv_block_bytes += bytes / owner.tensor.size(0);
        }
      }
    }
    ASSERT_GT(kv_block_bytes, 0);
    const int64_t unused_bytes = estimate.cache_size_in_bytes - allocated_bytes;
    EXPECT_GE(unused_bytes, 0);
    EXPECT_LT(unused_bytes, kv_block_bytes);
    if (is_v41) {
      EXPECT_FALSE(caches[1].get_k_cache().defined());
      EXPECT_FALSE(caches[2].get_k_cache().defined());
    }
  }
}

TEST(CacheLayoutBuilderTest, SharedKvCountMatchesActualDraftAllocation) {
  for (const bool draft_has_kv : {false, true}) {
    SCOPED_TRACE(draft_has_kv);
    ModelArgs target_args;
    target_args.model_type("deepseek_v4")
        .n_layers(2)
        .head_dim(16)
        .index_head_dim(8)
        .window_size(256)
        .compress_ratios({4, 128});
    ModelArgs draft_args;
    draft_args.model_type("deepseek_v4_mtp")
        .n_layers(1)
        .head_dim(8)
        .index_head_dim(4)
        .window_size(128)
        .compress_ratios({draft_has_kv ? 4 : 1});
    KVCacheEstimateOptions estimate;
    estimate.block_size = 128;
    estimate.dtype = torch::kBFloat16;
    estimate.cache_size_in_bytes = 4 * 1024 * 1024;
    estimate.max_seqs_per_batch = 2;
    estimate.max_tokens_per_batch = 256;
    const KVCacheEstimateOptions draft_estimate = estimate;
    estimate.draft_model_args = &draft_args;
    estimate.draft_options = &draft_estimate;
    const KVCacheCapacity target =
        estimate_kv_cache_capacity(target_args, estimate);
    const KVCacheCapacity original_draft =
        estimate_kv_cache_capacity(draft_args, draft_estimate);
    KVCacheCapacity draft = original_draft;
    draft.n_blocks() = target.n_blocks();
    ASSERT_GT(target.n_blocks(), 1);
    EXPECT_EQ(draft.n_blocks(), target.n_blocks());
    EXPECT_EQ(draft.has_kv_cache(), draft_has_kv);
    EXPECT_EQ(draft.swa_count(), original_draft.swa_count());
    EXPECT_NE(target.swa_count(), draft.swa_count());
    int64_t allocated_bytes = 0;
    int64_t block_bytes = 0;
    auto allocate = [&](const KVCacheCapacity& capacity,
                        const ModelArgs& args) {
      const KVCacheShape shape(capacity, args, /*world_size=*/1);
      proto::KVCacheShape serialized;
      shape.to_proto(&serialized);
      KVCacheCreateOptions create;
      create.device(torch::Device(torch::kCPU))
          .dtype(estimate.dtype)
          .model_type(args.model_type())
          .num_layers(args.n_layers())
          .block_size(estimate.block_size)
          .head_dim(args.head_dim())
          .index_head_dim(args.index_head_dim())
          .window_size(args.window_size())
          .compress_ratios(args.compress_ratios());
      std::vector<KVCache> caches;
      allocate_kv_caches(caches, KVCacheShape::from_proto(serialized), create);
      bool found_kv = false;
      for (const KVCache& cache : caches) {
        for (const KVCacheTensor& owner : cache.get_cache_tensors()) {
          const int64_t bytes =
              owner.tensor.numel() *
              static_cast<int64_t>(owner.tensor.element_size());
          allocated_bytes += bytes;
          if (owner.group_id != cache_group_id(BlockType::KV)) {
            EXPECT_EQ(owner.tensor.size(0), capacity.swa_count());
            continue;
          }
          found_kv = true;
          ASSERT_EQ(owner.tensor.size(0), target.n_blocks());
          block_bytes += bytes / owner.tensor.size(0);
        }
      }
      EXPECT_EQ(found_kv, capacity.has_kv_cache());
    };
    allocate(target, target_args);
    allocate(draft, draft_args);
    ASSERT_GT(block_bytes, 0);
    const int64_t unused_bytes = estimate.cache_size_in_bytes - allocated_bytes;
    EXPECT_GE(unused_bytes, 0);
    EXPECT_LT(unused_bytes, block_bytes);
  }
}

TEST(CacheLayoutBuilderTest, DraftSwaFollowsTargetSwaOrKvCount) {
  auto make_dsv4_cache = [](int64_t kv_blocks, int64_t swa_blocks) {
    DeepSeekV4KVCacheTensors tensors;
    tensors.key_cache = torch::zeros({kv_blocks, 2, 1, 8});
    tensors.swa_cache = torch::zeros({swa_blocks, 2, 1, 8});
    return KVCache(tensors);
  };
  KVCacheCreateOptions options;
  options.block_size(128).window_size(256);
  std::vector<KVCache> target_caches;
  target_caches.emplace_back(
      make_dsv4_cache(/*kv_blocks=*/4, /*swa_blocks=*/6));
  const HostCachePlan target = build_host_cache_plan(target_caches,
                                                     KVCacheShape{},
                                                     options,
                                                     /*host_blocks_factor=*/1.0,
                                                     InstanceRole::PREFILL,
                                                     /*enable_mla=*/false);
  ASSERT_EQ(target.block_counts.at(BlockType::SWA), 6);
  ASSERT_EQ(target.block_counts.at(BlockType::KV), 4);

  std::vector<KVCache> draft_caches;
  draft_caches.emplace_back(make_dsv4_cache(/*kv_blocks=*/8, /*swa_blocks=*/3));
  const HostCachePlan draft_with_target_swa =
      build_host_cache_plan(draft_caches,
                            KVCacheShape{},
                            options,
                            /*host_blocks_factor=*/1.0,
                            InstanceRole::PREFILL,
                            /*enable_mla=*/false,
                            &target);
  EXPECT_EQ(draft_with_target_swa.block_counts.at(BlockType::SWA),
            target.block_counts.at(BlockType::SWA));
  EXPECT_EQ(draft_with_target_swa.block_counts.at(BlockType::KV), 8);

  std::vector<KVCache> kv_only_caches;
  kv_only_caches.emplace_back(
      KVCacheTensors{torch::zeros({5, 2, 4}), torch::zeros({5, 2, 4})});
  const HostCachePlan kv_only_target =
      build_host_cache_plan(kv_only_caches,
                            KVCacheShape{},
                            options,
                            /*host_blocks_factor=*/1.0,
                            InstanceRole::PREFILL,
                            /*enable_mla=*/false);
  ASSERT_EQ(kv_only_target.block_counts.count(BlockType::SWA), 0u);
  ASSERT_EQ(kv_only_target.block_counts.at(BlockType::KV), 5);
  const HostCachePlan draft_without_target_swa =
      build_host_cache_plan(draft_caches,
                            KVCacheShape{},
                            options,
                            /*host_blocks_factor=*/1.0,
                            InstanceRole::PREFILL,
                            /*enable_mla=*/false,
                            &kv_only_target);
  EXPECT_EQ(draft_without_target_swa.block_counts.at(BlockType::SWA),
            kv_only_target.block_counts.at(BlockType::KV));
}

TEST(CacheLayoutBuilderTest, DraftKvRequiresCompatibleLogicalBlocks) {
  std::vector<KVCache> caches;
  caches.reserve(1);
  caches.emplace_back(KVCacheTensors{torch::zeros({3, 2, 4}), {}});
  KVCacheCreateOptions options;
  options.block_size(128);
  const HostCachePlan target = build_host_cache_plan(caches,
                                                     KVCacheShape{},
                                                     options,
                                                     /*host_blocks_factor=*/2.0,
                                                     InstanceRole::PREFILL,
                                                     /*enable_mla=*/false);
  const HostCachePlan draft = build_host_cache_plan(caches,
                                                    KVCacheShape{},
                                                    options,
                                                    /*host_blocks_factor=*/2.0,
                                                    InstanceRole::PREFILL,
                                                    /*enable_mla=*/false,
                                                    &target);
  EXPECT_EQ(target.block_counts, draft.block_counts);
  options.block_size(64);
  EXPECT_DEATH(build_host_cache_plan(caches,
                                     KVCacheShape{},
                                     options,
                                     /*host_blocks_factor=*/2.0,
                                     InstanceRole::PREFILL,
                                     /*enable_mla=*/false,
                                     &target),
               "share the block size");
  options.block_size(128);
  caches[0] = KVCache(KVCacheTensors{torch::zeros({2, 2, 4}), {}});
  EXPECT_DEATH(build_host_cache_plan(caches,
                                     KVCacheShape{},
                                     options,
                                     /*host_blocks_factor=*/2.0,
                                     InstanceRole::PREFILL,
                                     /*enable_mla=*/false,
                                     &target),
               "every target logical block ID");
}

TEST(CacheLayoutBuilderTest, HostCheckpointKeepsScalesAndOmitsKPoolTail) {
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors.key_cache = torch::ones({2, 4, 1, 8});
  tensors.index_cache = torch::ones({2, 4, 1, 4}, torch::kInt8);
  tensors.index_cache_scale = torch::ones({2, 4, 1}, torch::kFloat16);
  tensors.kpool_tail = torch::ones({3, 2, 4, 8});
  tensors.key_cache_scale = torch::ones({2, 4, 1}, torch::kFloat32);
  tensors.value_cache_scale = torch::ones({2, 4, 1}, torch::kFloat32);
  KVCache cache(tensors);
  const BlockTypeTensorMap persistent =
      cache.get_block_type_tensors(BlockType::KV, /*checkpoint_row=*/0);
  ASSERT_EQ(persistent.size(), 5U);
  EXPECT_EQ(persistent.at(KVCacheTensorRole::INDEX_SCALE).data_ptr(),
            tensors.index_cache_scale->data_ptr());
  EXPECT_EQ(persistent.at(KVCacheTensorRole::KEY_SCALE).data_ptr(),
            tensors.key_cache_scale->data_ptr());
  EXPECT_EQ(persistent.at(KVCacheTensorRole::VALUE_SCALE).data_ptr(),
            tensors.value_cache_scale->data_ptr());
  EXPECT_TRUE(cache
                  .get_block_type_tensors(BlockType::LINEAR,
                                          /*checkpoint_row=*/0)
                  .empty());
  const auto pd_tensors = cache.get_cache_tensors();
  ASSERT_EQ(pd_tensors.size(), 6U);
  const auto tail = std::find_if(
      pd_tensors.begin(), pd_tensors.end(), [](const KVCacheTensor& tensor) {
        return tensor.role == KVCacheTensorRole::KPOOL_TAIL;
      });
  ASSERT_NE(tail, pd_tensors.end());
  EXPECT_TRUE(tail->sequence_scoped);
  EXPECT_EQ(tail->tensor.data_ptr(), tensors.kpool_tail.data_ptr());
}

TEST(CacheLayoutBuilderTest, PersistentCheckpointViewsRetainLogicalSlots) {
  LinearAttentionKVCacheTensors tensors;
  std::vector<int64_t> conv_shape = {2, 5, 4};
  if (KVCacheShape::kConvHistoryAxis == 2) {
    conv_shape = {2, 4, 5};
  }
  tensors.conv_cache = torch::arange(40, torch::kFloat32).reshape(conv_shape);
  tensors.ssm_cache = torch::arange(48, torch::kFloat32).reshape({6, 2, 2, 2});
  KVCache cache(tensors);
  const BlockTypeTensorMap persistent =
      cache.get_block_type_tensors(BlockType::LINEAR, /*checkpoint_row=*/1);
  ASSERT_EQ(persistent.size(), 2U);
  for (const auto& [role, tensor] : persistent) {
    EXPECT_EQ(tensor.size(0), 2);
    if (role == KVCacheTensorRole::CONV) {
      EXPECT_TRUE(torch::equal(
          tensor,
          tensors.conv_cache.narrow(KVCacheShape::kConvHistoryAxis, 1, 3)));
    } else {
      EXPECT_EQ(role, KVCacheTensorRole::SSM);
      EXPECT_EQ(tensor.size(1), 1);
      EXPECT_TRUE(torch::equal(tensor[1][0], tensors.ssm_cache[4]));
      EXPECT_FALSE(tensor.is_contiguous());
    }
  }
}

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

TEST(CacheLayoutBuilderTest, PrefillHostPoolSharesHistoryAcrossBatches) {
  proto::KVCacheShape proto_shape;
  for (int64_t dim : {2, 2, 1, 8}) {
    proto_shape.add_key_cache_shape(dim);
    proto_shape.add_value_cache_shape(dim);
  }
  for (int64_t dim : {2, 2, 1, 4}) {
    proto_shape.add_index_cache_shape(dim);
  }
  for (int64_t dim : {2, 2, 1, 1}) {
    proto_shape.add_index_cache_scale_shape(dim);
  }
  const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(2)
      .enable_lighting_indexer(true)
      .enable_indexer_cache_quant(true);
  HostPoolProcessGroup::Exchange exchange;
  HostPoolProcessGroup first_group(0, exchange);
  HostPoolProcessGroup second_group(1, exchange);
  std::array<std::vector<KVCache>, 2> caches;
  auto first_setup = std::async(std::launch::async, [&] {
    return std::make_unique<PrefillLayerwiseKVCache>(
        shape, options, 1, 1024 * 1024, caches[0], &first_group);
  });
  PrefillLayerwiseKVCache second(
      shape, options, 1, 1024 * 1024, caches[1], &second_group);
  const auto first_owner = first_setup.get();
  PrefillLayerwiseKVCache& first = *first_owner;
  ASSERT_NE(first.host_caches()[0].get_k_cache().data_ptr(),
            second.host_caches()[0].get_k_cache().data_ptr());
  for (const KVCacheTensor& tensor :
       first.host_caches()[0].get_cache_tensors()) {
    tensor.tensor[0].fill_(9);
  }
  for (const KVCacheTensor& tensor :
       first.host_caches()[1].get_cache_tensors()) {
    tensor.tensor[0].fill_(3);
  }
  EXPECT_TRUE(
      second.host_caches()[0].get_k_cache()[0].eq(9).all().item<bool>());
  ModelInputParams params;
  params.meta.batch_id = 1;
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.attention.host.block_tables = torch::tensor({{0}}, torch::kInt32);
  params.attention.host.kv_seq_lens = {2};
  params.attention.host.q_seq_lens = {1};
  auto first_begin = std::async(std::launch::async, [&] {
    return first.begin(params, /*block_size=*/2, /*compute_stream=*/nullptr);
  });
  ASSERT_TRUE(
      second.begin(params, /*block_size=*/2, /*compute_stream=*/nullptr));
  ASSERT_TRUE(first_begin.get());
  ASSERT_TRUE(first.synchronize_layer(0));
  ASSERT_TRUE(second.synchronize_layer(0));
  for (size_t rank = 0; rank < caches.size(); ++rank) {
    for (const KVCacheTensor& tensor : caches[rank][0].get_cache_tensors()) {
      EXPECT_TRUE(tensor.tensor[0].eq(0).all().item<bool>());
      tensor.tensor[0].fill_(rank + 1);
    }
  }
  ASSERT_TRUE(first.synchronize_layer(1));
  for (const KVCacheTensor& tensor :
       first.host_caches()[0].get_cache_tensors()) {
    EXPECT_TRUE(tensor.tensor[0].eq(1).all().item<bool>());
  }
  ASSERT_TRUE(second.synchronize_layer(1));
  for (size_t rank = 0; rank < caches.size(); ++rank) {
    for (const KVCacheTensor& tensor : caches[rank][1].get_cache_tensors()) {
      EXPECT_TRUE(tensor.tensor[0].eq(3).all().item<bool>());
      tensor.tensor[0][1].fill_(rank + 4);
    }
  }

  auto writer_done =
      std::async(std::launch::async, [&] { return second.finish(); });
  EXPECT_EQ(writer_done.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  EXPECT_TRUE(first.finish());
  ASSERT_TRUE(writer_done.get());
  caches[0][1].get_k_cache()[0].fill_(8);
  params.meta.batch_id = 2;
  params.attention.host.q_seq_lens = {0};
  auto rejected_begin = std::async(std::launch::async, [&] {
    return first.begin(params, /*block_size=*/2, /*compute_stream=*/nullptr);
  });
  ModelInputParams mismatch = params;
  mismatch.attention.host.block_tables = torch::tensor({{1}}, torch::kInt32);
  EXPECT_FALSE(
      second.begin(mismatch, /*block_size=*/2, /*compute_stream=*/nullptr));
  EXPECT_FALSE(rejected_begin.get());
  EXPECT_TRUE(caches[0][1].get_k_cache()[0].eq(8).all().item<bool>());

  params.meta.batch_id = 3;
  auto next = std::async(std::launch::async, [&] {
    return first.begin(params, /*block_size=*/2, /*compute_stream=*/nullptr);
  });
  ASSERT_TRUE(
      second.begin(params, /*block_size=*/2, /*compute_stream=*/nullptr));
  ASSERT_TRUE(next.get());
  for (int64_t layer = 0; layer < 2; ++layer) {
    ASSERT_TRUE(first.synchronize_layer(layer));
    ASSERT_TRUE(second.synchronize_layer(layer));
  }
  for (size_t rank = 0; rank < caches.size(); ++rank) {
    for (const KVCacheTensor& tensor : caches[rank][1].get_cache_tensors()) {
      EXPECT_TRUE(tensor.tensor[0][0].eq(3).all().item<bool>());
      EXPECT_TRUE(tensor.tensor[0][1].eq(5).all().item<bool>());
    }
  }
  EXPECT_TRUE(first.finish());
  EXPECT_TRUE(second.finish());
}

namespace {

template <int32_t kKvSplitSize>
void check_prefill_split_sharing_and_index_copies() {
  SCOPED_TRACE(kKvSplitSize);
  auto& model = ModelConfig::get_instance();
  auto& parallel = ParallelConfig::get_instance();
  const std::string old_model_impl = model.model_impl();
  const int32_t old_kv_split_size = parallel.kv_split_size();
  const ScopeGuard restore([&] {
    model.model_impl(old_model_impl);
    parallel.kv_split_size(old_kv_split_size);
  });
  model.model_impl("python");
  parallel.kv_split_size(kKvSplitSize);

  constexpr int32_t kPhysicalBlockSize = 2;
  proto::KVCacheShape proto_shape;
  for (int64_t dim : {2, kPhysicalBlockSize, 1, 8}) {
    proto_shape.add_key_cache_shape(dim);
    proto_shape.add_value_cache_shape(dim);
  }
  for (int64_t dim : {2 * kKvSplitSize, kPhysicalBlockSize, 1, 4}) {
    proto_shape.add_index_cache_shape(dim);
  }
  for (int64_t dim : {2 * kKvSplitSize, kPhysicalBlockSize, 1}) {
    proto_shape.add_index_cache_scale_shape(dim);
  }
  const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(2)
      .enable_indexer_cache_quant(true);
  constexpr int32_t kRanksPerShard = 2;
  constexpr int32_t kWorldSize = kKvSplitSize * kRanksPerShard;
  HostPoolProcessGroup::Exchange exchange(kWorldSize);
  std::array<std::vector<KVCache>, kWorldSize> caches;
  std::array<std::unique_ptr<HostPoolProcessGroup>, kWorldSize> groups;
  std::array<std::future<std::unique_ptr<PrefillLayerwiseKVCache>>, kWorldSize>
      allocations;
  std::array<std::unique_ptr<PrefillLayerwiseKVCache>, kWorldSize> owners;
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    groups[rank] = std::make_unique<HostPoolProcessGroup>(rank, exchange);
    allocations[rank] = std::async(std::launch::async, [&, rank] {
      return std::make_unique<PrefillLayerwiseKVCache>(
          shape,
          options,
          /*num_buffers=*/1,
          /*host_memory_bytes=*/1024 * 1024,
          caches[rank],
          groups[rank].get(),
          kKvSplitSize,
          /*kv_split_rank=*/rank / kRanksPerShard);
    });
  }
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    owners[rank] = allocations[rank].get();
  }
  constexpr int64_t kBlock = 1;
  constexpr int64_t kIndexPage = kKvSplitSize * kBlock;
  constexpr int64_t kLastIndexPage = kIndexPage + kKvSplitSize - 1;
  for (int32_t shard = 0; shard < kKvSplitSize; ++shard) {
    const KVCache& history = owners[kRanksPerShard * shard]->host_caches()[1];
    history.get_k_cache()[kBlock].fill_(10 + shard);
    for (int64_t page = 0; page < kKvSplitSize; ++page) {
      history.get_index_cache()[kIndexPage + page].fill_(21 + page);
      (*history.get_indexer_cache_scale())[kIndexPage + page].fill_(31 + page);
    }
  }
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    const KVCache& history = owners[rank]->host_caches()[1];
    EXPECT_TRUE(history.get_k_cache()[kBlock]
                    .eq(10 + rank / kRanksPerShard)
                    .all()
                    .item<bool>());
  }

  ModelInputParams params;
  params.meta.batch_id = 1;
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.attention.host.block_tables = torch::tensor({{kBlock}}, torch::kInt32);
  params.attention.host.kv_seq_lens = {kPhysicalBlockSize * kKvSplitSize};
  params.attention.host.q_seq_lens = {1};
  std::array<std::future<bool>, kWorldSize> steps;
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    steps[rank] = std::async(std::launch::async, [&, rank] {
      return owners[rank]->begin(params, kPhysicalBlockSize, nullptr);
    });
  }
  for (auto& step : steps) {
    ASSERT_TRUE(step.get());
  }
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    ASSERT_TRUE(owners[rank]->synchronize_layer(0));
    const KVCache& resident = caches[rank][0];
    EXPECT_TRUE(resident.get_k_cache()[kBlock].eq(0).all().item<bool>());
    ASSERT_TRUE(owners[rank]->synchronize_layer(1));
    const KVCache& device = caches[rank][1];
    EXPECT_TRUE(device.get_k_cache()[kBlock]
                    .eq(10 + rank / kRanksPerShard)
                    .all()
                    .item<bool>());
    const torch::Tensor scales = *device.get_indexer_cache_scale();
    for (int64_t page = 0; page < kKvSplitSize; ++page) {
      EXPECT_TRUE(device.get_index_cache()[kIndexPage + page]
                      .eq(21 + page)
                      .all()
                      .item<bool>());
      EXPECT_TRUE(scales[kIndexPage + page].eq(31 + page).all().item<bool>());
    }
    device.get_k_cache()[kBlock].fill_(40 + rank);
    device.get_index_cache()[kLastIndexPage].fill_(50 + rank);
    scales[kLastIndexPage].fill_(60 + rank);
  }
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    steps[rank] = std::async(std::launch::async,
                             [&, rank] { return owners[rank]->finish(); });
  }
  for (auto& step : steps) {
    ASSERT_TRUE(step.get());
  }
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    const int32_t writer = (rank / kRanksPerShard) * kRanksPerShard + 1;
    const KVCache& history = owners[rank]->host_caches()[1];
    EXPECT_TRUE(
        history.get_k_cache()[kBlock].eq(40 + writer).all().item<bool>());
    EXPECT_TRUE(
        history.get_index_cache()[kIndexPage].eq(21).all().item<bool>());
    EXPECT_TRUE(history.get_index_cache()[kLastIndexPage]
                    .eq(50 + writer)
                    .all()
                    .item<bool>());
    EXPECT_TRUE((*history.get_indexer_cache_scale())[kLastIndexPage]
                    .eq(60 + writer)
                    .all()
                    .item<bool>());
  }
}

}  // namespace

TEST(CacheLayoutBuilderTest,
     PrefillKvSplitSharesOnlySameShardAndCopiesIndexPages) {
  if (!Platform::is_npu()) {
    GTEST_SKIP() << "Replicated DSA Index pages are an NPU layout";
  }
  ASSERT_NO_FATAL_FAILURE(check_prefill_split_sharing_and_index_copies<2>());
  check_prefill_split_sharing_and_index_copies<4>();
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
                       cache_group_id(BlockType::KV)};
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
                       cache_group_id(BlockType::KV)};
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
                       torch::zeros({2, 5, 15}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::COMPOSITE);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 5U);
  EXPECT_EQ(descriptor.spans[0].logical_tensor, "conv_key_a");
  EXPECT_EQ(descriptor.spans[2].logical_tensor, "conv_key_b");
  EXPECT_EQ(descriptor.spans[4].logical_tensor, "conv_value");
  EXPECT_EQ(descriptor.spans[0].repeat_count, 5U);
}

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

TEST(CacheLayoutBuilderTest, DescribesHeterogeneousKvShapes) {
  CacheTensorLayoutContext context;
  proto::KVCacheShape serialized;
  serialized.set_grouped_cache_layout(true);
  serialized.add_key_cache_shape(4);
  serialized.add_key_cache_shape(4);
  apply_cache_shape_layout(KVCacheShape::from_proto(serialized), &context);
  context.block_token_capacity = 128;
  context.kv_head_count = 1;
  context.index_block_capacity = 4;
  for (const int64_t rows : {256, 8}) {
    KVCacheTensor tensor{KVCacheTensorRole::KEY,
                         torch::zeros({4, rows, 1, 8}),
                         cache_group_id(BlockType::KV)};
    std::string error;
    ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
    EXPECT_EQ(effective_block_token_capacity(context, tensor), rows);
    ASSERT_TRUE(tensor.shard_descriptor.has_value());
    ASSERT_EQ(tensor.shard_descriptor->spans.size(), 1U);
    EXPECT_EQ(tensor.shard_descriptor->spans[0].repeat_count,
              static_cast<uint64_t>(rows));
  }
  KVCacheTensor index{KVCacheTensorRole::INDEX,
                      torch::zeros({4, 32, 1, 8}),
                      cache_group_id(BlockType::KV)};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &index, &error)) << error;
  EXPECT_EQ(effective_block_token_capacity(context, index), 32);
  EXPECT_EQ(index.shard_descriptor->spans[0].repeat_count, 32U);
  KVCacheTensor window{KVCacheTensorRole::SWA,
                       torch::zeros({4, 64, 1, 8}),
                       cache_group_id(BlockType::SWA)};
  EXPECT_EQ(effective_block_token_capacity(context, window), 64);
}

TEST(CacheLayoutBuilderTest, DescribesHeadMajorGroupedShapes) {
  CacheTensorLayoutContext context;
  context.has_grouped_cache_layout = true;
  context.head_major_layout = true;
  context.block_token_capacity = 128;
  context.kv_head_count = 1;
  KVCacheTensor tensor{KVCacheTensorRole::KEY,
                       torch::zeros({4, 1, 32, 8}),
                       cache_group_id(BlockType::KV)};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  EXPECT_EQ(effective_block_token_capacity(context, tensor), 32);
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  EXPECT_EQ(tensor.shard_descriptor->spans[0].repeat_count, 32U);
  KVCacheTensor scale{KVCacheTensorRole::INDEX_SCALE,
                      torch::zeros({4, 32, 1}),
                      cache_group_id(BlockType::KV)};
  EXPECT_EQ(effective_block_token_capacity(context, scale), 32);
}

TEST(CacheLayoutBuilderTest, PreservesUniformProducerValidation) {
  CacheTensorLayoutContext context;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;
  KVCacheTensor uniform{KVCacheTensorRole::KEY,
                        torch::zeros({4, 16, 1, 8}),
                        cache_group_id(BlockType::KV)};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &uniform, &error)) << error;
  KVCacheTensor diverging{KVCacheTensorRole::KEY,
                          torch::zeros({4, 32, 1, 8}),
                          cache_group_id(BlockType::KV)};
  EXPECT_FALSE(describe_cache_tensor(context, &diverging, &error));
  EXPECT_NE(error.find("token capacity differs"), std::string::npos);
  KVCacheTensor malformed{
      KVCacheTensorRole::KEY, torch::zeros({4}), cache_group_id(BlockType::KV)};
  EXPECT_FALSE(describe_cache_tensor(context, &malformed, &error));
}

TEST(CacheLayoutBuilderTest, PreservesMlaPhysicalLayout) {
  CacheTensorLayoutContext context;
  context.block_token_capacity = 16;
  context.enable_mla = true;
  for (const int64_t width : {32, 4}) {
    KVCacheTensor tensor{KVCacheTensorRole::KEY,
                         torch::zeros({4, width, 16, 16}),
                         cache_group_id(BlockType::KV)};
    std::string error;
    ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
    ASSERT_TRUE(tensor.shard_descriptor.has_value());
    EXPECT_EQ(tensor.shard_descriptor->kind, LogicalShardKind::REPLICATED);
    EXPECT_EQ(tensor.shard_descriptor->spans[0].bytes_per_region,
              static_cast<uint64_t>(tensor.tensor[0].nbytes()));
  }
}

}  // namespace

}  // namespace xllm
