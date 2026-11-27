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

#include "framework/block/hierarchy_block_manager_pool.h"

#include <folly/ScopeGuard.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/util/env_var.h"
#include "core/util/hash_util.h"
#include "framework/batch/batch.h"
#include "framework/block/block_utils.h"
#include "framework/block/linear_state_block_manager.h"
#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "tests/core/framework/block/mm_test_utils.h"

namespace xllm {

// Peer that reaches inside HierarchyBlockManagerPool for verification of host
// leaf construction. The pool is heavy to spin up; we exercise the plumbing
// with an Engine stub set to nullptr because the constructor only touches the
// engine during allocate / transfer paths, not at build time.
class HierarchyPoolTestPeer final {
 public:
  static const std::vector<std::unique_ptr<CompositeBlockManager>>&
  host_block_managers(const HierarchyBlockManagerPool& pool) {
    return pool.host_block_managers_;
  }

  static const CompositeBlockManager::LeafMap& host_leaves(
      const HierarchyBlockManagerPool& pool,
      int32_t dp_rank = 0) {
    return pool.host_block_managers_[dp_rank]->leaf_entries();
  }

  static std::vector<BlockTransferInfo> pending_load_infos(
      const HierarchyBlockManagerPool& pool) {
    std::vector<BlockTransferInfo> infos;
    for (const auto& per_dp : pool.load_block_transfer_infos_) {
      infos.insert(infos.end(), per_dp.begin(), per_dp.end());
    }
    return infos;
  }

  static const std::vector<BlockTransferInfo>& pending_load_infos(
      const HierarchyBlockManagerPool& pool,
      int32_t dp_rank) {
    return pool.load_block_transfer_infos_.at(dp_rank);
  }

  static CompositeBlockManager* device_composite(
      HierarchyBlockManagerPool& pool) {
    return static_cast<CompositeBlockManager*>(
        pool.block_managers_.front().get());
  }

  static void dispatch_pending_h2d(HierarchyBlockManagerPool& pool) {
    for (auto& per_dp : pool.load_block_transfer_infos_) {
      per_dp.clear();
    }
  }

  static void collect_offload_pairs(HierarchyBlockManagerPool& pool,
                                    Sequence* sequence) {
    pool.collect_offload_pairs(sequence);
  }

  static size_t pending_offload_pair_count(
      const HierarchyBlockManagerPool& pool) {
    size_t count = 0;
    for (const auto& queue : pool.offload_block_pair_queues_) {
      count += queue.size_approx();
    }
    return count;
  }

  static size_t pending_offload_pair_count(
      const HierarchyBlockManagerPool& pool,
      int32_t dp_rank) {
    return pool.offload_block_pair_queues_.at(dp_rank).size_approx();
  }

  static void release_pending_offload_pairs(HierarchyBlockManagerPool& pool) {
    for (auto& queue : pool.offload_block_pair_queues_) {
      std::shared_ptr<OffloadBlockPair> pair;
      while (queue.try_dequeue(pair)) {
        pair.reset();
      }
    }
  }

  static std::shared_ptr<OffloadBlockPair> take_pending_offload_pair(
      HierarchyBlockManagerPool& pool,
      int32_t dp_rank = 0) {
    std::shared_ptr<OffloadBlockPair> pair;
    pool.offload_block_pair_queues_.at(dp_rank).try_dequeue(pair);
    return pair;
  }

  // Decode that crosses a KV block also queues the completed KV block. Pull
  // the LINEAR pair out and put the other pairs back in their original order.
  static std::shared_ptr<OffloadBlockPair> take_linear_offload_pair(
      HierarchyBlockManagerPool& pool,
      int32_t dp_rank = 0) {
    std::vector<std::shared_ptr<OffloadBlockPair>> held;
    std::shared_ptr<OffloadBlockPair> linear_pair;
    std::shared_ptr<OffloadBlockPair> pair;
    while (pool.offload_block_pair_queues_.at(dp_rank).try_dequeue(pair)) {
      if (pair != nullptr && pair->block_type == BlockType::LINEAR &&
          linear_pair == nullptr) {
        linear_pair = std::move(pair);
        continue;
      }
      held.emplace_back(std::move(pair));
    }
    for (std::shared_ptr<OffloadBlockPair>& kept : held) {
      pool.offload_block_pair_queues_.at(dp_rank).enqueue(std::move(kept));
    }
    return linear_pair;
  }

  static Block take_host_linear_front(Sequence& sequence) {
    return std::move(
        sequence.host_kv_state().mutable_blocks(BlockType::LINEAR)->front());
  }

  static void enqueue_offload_pair(HierarchyBlockManagerPool& pool,
                                   std::shared_ptr<OffloadBlockPair> pair,
                                   int32_t dp_rank = 0) {
    pool.offload_block_pair_queues_.at(dp_rank).enqueue(std::move(pair));
  }

  static std::set<std::pair<BlockType, std::string>> complete_offload_pairs(
      HierarchyBlockManagerPool& pool) {
    std::set<std::pair<BlockType, std::string>> stored_keys;
    std::map<BlockType, std::vector<Block>> host_blocks;
    std::vector<Block> device_blocks;
    const size_t pair_count = pending_offload_pair_count(pool);
    device_blocks.reserve(pair_count);
    std::shared_ptr<OffloadBlockPair> pair;
    while (pool.offload_block_pair_queues_.front().try_dequeue(pair)) {
      stored_keys.emplace(pair->block_type,
                          std::string(reinterpret_cast<const char*>(
                                          pair->dst.get_immutable_hash_value()),
                                      XXH3_128BITS_HASH_VALUE_LEN));
      auto [blocks, inserted] = host_blocks.try_emplace(pair->block_type);
      if (inserted) {
        blocks->second.reserve(pair_count);
      }
      blocks->second.emplace_back(std::move(pair->dst));
      device_blocks.emplace_back(std::move(pair->src));
      pair.reset();
    }
    device_blocks.clear();
    for (auto& [type, blocks] : host_blocks) {
      auto* leaf =
          pool.host_block_managers_.front()->leaf_entries().at(type).leaf.get();
      leaf->cache(blocks);
      blocks.clear();
    }
    return stored_keys;
  }
};

namespace {

struct CapturedTransfer {
  uint32_t dp_rank = 0;
  uint64_t batch_id = 0;
  std::vector<BlockTransferInfo> infos;
  std::vector<folly::Promise<uint32_t>> promises;
  std::vector<uint8_t> completed;
};

class FakePrefetchEngine final : public Engine {
 public:
  explicit FakePrefetchEngine(size_t worker_count,
                              int64_t timeout_ms = -1,
                              size_t batch_size = 2)
      : worker_count_(worker_count),
        timeout_ms_(timeout_ms),
        batch_size_(batch_size) {}

  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  void prefetch_from_storage(
      uint32_t dp_rank,
      std::shared_ptr<const StoragePrefetchRequest> request,
      PrefetchResult::StopPredicate stop_requested,
      PrefetchResult::DoneCallback done) override {
    CHECK(request != nullptr);
    dp_rank_ = dp_rank;
    ++request_count_;
    workers_ended_.assign(worker_count_, false);
    request_ = std::move(request);
    result_ = std::make_shared<PrefetchResult>(worker_count_,
                                               *request_,
                                               batch_size_,
                                               timeout_ms_,
                                               std::move(stop_requested),
                                               std::move(done));
  }

  uint32_t dp_rank() const { return dp_rank_; }

  const StoragePrefetchRequest& request() const { return *request_; }

  const std::shared_ptr<PrefetchResult>& result() const { return result_; }

  size_t request_count() const { return request_count_; }

  void finish_worker(size_t worker_index,
                     size_t hit_units,
                     bool worker_ok = true) {
    CHECK(result_ != nullptr);
    const size_t batch_size = result_->batch_wire_size();
    size_t remaining_hits = hit_units;
    for (size_t batch_index = 0;
         batch_index < request_->batch_count(batch_size);
         ++batch_index) {
      const size_t batch_units =
          request_->batch_unit_count(batch_index, batch_size);
      const uint8_t batch_hits =
          static_cast<uint8_t>(std::min(remaining_hits, batch_units));
      const std::optional<PrefetchControl> control =
          result_->record_batch_result(worker_index, batch_hits);
      CHECK(control.has_value());
      remaining_hits -= batch_hits;
      if (*control == PrefetchControl::STOP) {
        break;
      }
    }
    CHECK_EQ(remaining_hits, 0u);
    workers_ended_.at(worker_index) = true;
    result_->mark_worker_ended(worker_index, worker_ok);
  }

  void finish_worker_with_logical_hits(size_t worker_index,
                                       const std::vector<uint8_t>& logical_hits,
                                       bool worker_ok = true) {
    CHECK(result_ != nullptr);
    const size_t batch_size = result_->batch_wire_size();
    size_t logical_index = 0;
    for (size_t batch_index = 0;
         batch_index < request_->batch_count(batch_size);
         ++batch_index) {
      const size_t unit_begin =
          request_->batch_unit_begin(batch_index, batch_size);
      const size_t unit_count =
          request_->batch_unit_count(batch_index, batch_size);
      std::vector<uint8_t> gated(batch_size, 0);
      std::vector<uint8_t> non_gated(batch_size, 0);
      for (size_t local = 0; local < unit_count; ++local) {
        const PrefetchUnit& unit = request_->units[unit_begin + local];
        bool gated_hit = true;
        for (size_t i = 0; i < unit.gated_blocks.size(); ++i) {
          CHECK_LT(logical_index, logical_hits.size());
          gated_hit = gated_hit && logical_hits[logical_index++] != 0;
        }
        bool non_gated_hit = !unit.has_non_gated;
        if (unit.has_non_gated) {
          non_gated_hit = !unit.non_gated_blocks.empty();
          for (size_t i = 0; i < unit.non_gated_blocks.size(); ++i) {
            CHECK_LT(logical_index, logical_hits.size());
            non_gated_hit = non_gated_hit && logical_hits[logical_index++] != 0;
          }
        }
        gated[local] = gated_hit ? 1 : 0;
        non_gated[local] = non_gated_hit ? 1 : 0;
      }
      const std::optional<PrefetchControl> control =
          result_->record_batch_result(worker_index, gated, non_gated);
      CHECK(control.has_value());
      if (*control == PrefetchControl::STOP) {
        break;
      }
    }
    CHECK_EQ(logical_index, logical_hits.size());
    workers_ended_.at(worker_index) = true;
    result_->mark_worker_ended(worker_index, worker_ok);
  }

  void drain_workers() {
    if (result_ == nullptr) {
      return;
    }
    for (size_t index = 0; index < worker_count_; ++index) {
      if (!workers_ended_[index]) {
        workers_ended_[index] = true;
        result_->mark_worker_ended(index, /*worker_ok=*/false);
      }
    }
    // Transport completion also releases its stop predicate's Request alias.
    // Otherwise a fixture would keep Host blocks pinned after all workers end.
    result_.reset();
  }

  uint32_t host_transfer_worker_count() const override {
    return static_cast<uint32_t>(worker_count_);
  }

  void transfer_kv_blocks(
      uint32_t dp_rank,
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& infos) override {
    CapturedTransfer capture;
    capture.dp_rank = dp_rank;
    capture.batch_id = batch_id;
    capture.infos = infos;
    loads_.emplace_back(std::move(capture));
  }

  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& infos) override {
    offloads_.emplace_back();
    CapturedTransfer& capture = offloads_.back();
    capture.dp_rank = dp_rank;
    capture.infos = infos;
    const size_t result_count = offload_result_count_.value_or(worker_count_);
    std::vector<folly::SemiFuture<uint32_t>> futures;
    futures.reserve(result_count);
    if (automatic_offload_) {
      for (size_t index = 0; index < result_count; ++index) {
        futures.emplace_back(
            folly::makeSemiFuture(static_cast<uint32_t>(infos.size())));
      }
    } else {
      capture.promises.resize(result_count);
      capture.completed.assign(result_count, 0);
      for (auto& promise : capture.promises) {
        futures.emplace_back(promise.getSemiFuture());
      }
    }
    return futures;
  }

  void hold_offloads(size_t result_count) {
    automatic_offload_ = false;
    offload_result_count_ = result_count;
  }

  void finish_offload_worker(size_t capture_index,
                             size_t worker_index,
                             std::optional<uint32_t> copied = std::nullopt,
                             bool exception = false) {
    CapturedTransfer& capture = offloads_.at(capture_index);
    capture.completed.at(worker_index) = 1;
    if (exception) {
      capture.promises.at(worker_index)
          .setException(folly::make_exception_wrapper<std::runtime_error>(
              "offload failed"));
    } else {
      capture.promises.at(worker_index)
          .setValue(
              copied.value_or(static_cast<uint32_t>(capture.infos.size())));
    }
  }

  void drain_offloads() {
    for (size_t capture = 0; capture < offloads_.size(); ++capture) {
      for (size_t worker = 0; worker < offloads_[capture].completed.size();
           ++worker) {
        if (!offloads_[capture].completed[worker]) {
          finish_offload_worker(capture, worker, /*copied=*/0);
        }
      }
    }
  }

  std::vector<CapturedTransfer> loads_;
  std::vector<CapturedTransfer> offloads_;

 private:
  bool automatic_offload_ = true;
  std::optional<size_t> offload_result_count_;
  size_t worker_count_ = 0;
  int64_t timeout_ms_ = -1;
  size_t batch_size_ = 2;
  uint32_t dp_rank_ = 0;
  size_t request_count_ = 0;
  std::vector<bool> workers_ended_;
  std::shared_ptr<const StoragePrefetchRequest> request_;
  std::shared_ptr<PrefetchResult> result_;
};

constexpr size_t kUnitTokens = kDsv4FixedBlockSize;
constexpr uint32_t kBaseBlockSize = 128;

BlockManagerPool::Options make_flat_kv_options() {
  BlockManagerPool::Options opts;
  opts.num_blocks(64)
      .host_num_blocks(128)
      .block_size(128)
      .enable_prefix_cache(true)
      .enable_host_offload(true);
  return opts;
}

BlockManagerPool::Options make_typed_cache_options() {
  constexpr uint32_t kWindow = 128;
  const uint32_t swa_blocks_per_seq =
      static_cast<uint32_t>(get_swa_blocks_per_seq(kWindow, kBaseBlockSize));

  BlockManagerPool::Options opts;
  opts.num_blocks(4096)
      .block_size(kUnitTokens)
      .swa_block_size(kBaseBlockSize)
      .enable_prefix_cache(true)
      .enable_host_offload(true)
      .sliding_window_size(kWindow)
      .swa_blocks_per_seq(swa_blocks_per_seq)
      .swa_num_blocks(266)
      .max_tokens_per_batch(32768)
      .max_seqs_per_batch(4)
      .manager_types({1u, 0u})
      .host_num_blocks_by_type({{BlockType::SWA, 512}, {BlockType::KV, 512}});
  return opts;
}

Sequence make_test_sequence(size_t index,
                            const std::vector<int32_t>& prompt_token_ids,
                            bool enable_schedule_overlap = false,
                            const MMData& mm_data = MMData()) {
  // Sequence borrows these defaults for its entire lifetime.
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker(/*max_generated_tokens=*/16,
                                          /*max_context_len=*/0,
                                          /*eos_token=*/-1,
                                          /*ignore_eos=*/false,
                                          /*stop_tokens=*/{},
                                          /*stop_sequences=*/{});
  SequenceParams seq_params;
  seq_params.seq_capacity =
      std::max<size_t>(32768, prompt_token_ids.size() + 16);
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = enable_schedule_overlap;

  torch::Tensor input_embedding;
  IncrementalDecoder decoder("", 1, false, false);
  return Sequence(index,
                  prompt_token_ids,
                  input_embedding,
                  mm_data,
                  std::move(decoder),
                  seq_params);
}

std::shared_ptr<Request> make_test_request(
    const std::vector<int32_t>& prompt_token_ids,
    size_t best_of = 1,
    size_t generation_budget = 16,
    MMData mm_data = MMData()) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(generation_budget);
  stopping_checker.set_max_context_len(prompt_token_ids.size() +
                                       generation_budget);
  stopping_checker.set_ignore_eos(true);

  RequestState request_state("test",
                             prompt_token_ids,
                             sampling_param,
                             scheduler_param,
                             stopping_checker,
                             prompt_token_ids.size() + generation_budget,
                             /*n=*/best_of,
                             /*best_of=*/best_of,
                             /*logprobs=*/false,
                             /*stream=*/false,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true,
                             /*enable_schedule_overlap=*/false,
                             /*output_func=*/nullptr,
                             /*outputs_func=*/nullptr);
  request_state.mm_data = std::move(mm_data);
  return std::make_shared<Request>("request",
                                   "x-request",
                                   "time",
                                   std::move(request_state),
                                   "service-request");
}

size_t count_valid_blocks(const Slice<Block>& blocks) {
  return static_cast<size_t>(
      std::count_if(blocks.begin(), blocks.end(), [](const Block& block) {
        return block.is_valid();
      }));
}

Block linear_block(int32_t id) { return Block(id, nullptr); }

void shutdown_pool(HierarchyBlockManagerPool& pool,
                   FakePrefetchEngine& engine) {
  engine.drain_workers();
  engine.drain_offloads();
  // Consume any remaining staged descriptions through the normal public path.
  pool.transfer_blocks();
  engine.drain_offloads();
  pool.reset_prefix_cache();
}

void publish_host_prefix(HierarchyBlockManagerPool& pool,
                         FakePrefetchEngine& engine,
                         const std::vector<int32_t>& tokens,
                         int32_t dp_rank = 0,
                         size_t worker_count = 2) {
  auto request = make_test_request(tokens);
  request->sequences().front()->set_dp_rank(dp_rank);
  auto drain = folly::makeGuard([&engine]() { engine.drain_workers(); });
  bool completed = false;
  pool.prefetch_from_storage(
      request, [&completed](std::shared_ptr<Request>) { completed = true; });
  if (!completed) {
    ASSERT_NE(engine.result(), nullptr);
    const size_t units = engine.request().units.size();
    for (size_t worker = 0; worker < worker_count; ++worker) {
      engine.finish_worker(worker, units);
    }
  }
  ASSERT_TRUE(completed);
}

void dispatch_loads(HierarchyBlockManagerPool& pool,
                    Sequence& sequence,
                    size_t dp_size = 1) {
  std::vector<Batch> batches(dp_size);
  batches.at(static_cast<size_t>(sequence.dp_rank())).add(&sequence);
  pool.transfer_blocks(batches);
}

using tests::make_image_mm_data;

BlockManagerPool::Options make_v41_options(uint32_t block_size = 256) {
  auto options = make_typed_cache_options();
  // Publish and probe hash each leaf with its own span. Keep SWA on the same
  // span as KV so a completed unit is visible to both leaves.
  options.block_size(block_size)
      .swa_block_size(block_size)
      .num_blocks(64)
      .enable_kvcache_store(true)
      .sliding_window_size(512)
      .swa_blocks_per_seq(512 / block_size)
      .manager_types({1, 0})
      .host_num_blocks_by_type({{BlockType::SWA, 128}, {BlockType::KV, 128}});
  return options;
}

size_t transfer_count(const FakePrefetchEngine& engine, BlockType type) {
  size_t count = 0;
  for (const auto& capture : engine.offloads_) {
    count += static_cast<size_t>(
        std::count_if(capture.infos.begin(),
                      capture.infos.end(),
                      [type](const BlockTransferInfo& info) {
                        return info.block_type == type;
                      }));
  }
  return count;
}

TEST(HierarchyBlockManagerPoolTest, V41TokenGroupsOffloadAndRestoreCommonUnit) {
  for (const uint32_t block_size : {128u, 256u}) {
    SCOPED_TRACE(block_size);
    const size_t unit = block_size;
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(make_v41_options(block_size), &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
    const std::vector<int32_t> tokens(unit + 1, 83);
    Sequence source = make_test_sequence(0, tokens);
    ASSERT_TRUE(pool.allocate(&source, unit));
    source.kv_state().set_kv_cache_tokens_num(unit);
    pool.deallocate(&source);
    pool.transfer_blocks();
    ASSERT_EQ(engine.offloads_.size(), 1u);
    EXPECT_EQ(engine.offloads_.front().infos.size(), 2u);
    for (const auto& info : engine.offloads_.front().infos) {
      EXPECT_GT(info.src_block_id, 0);
      EXPECT_GT(info.dst_block_id, 0);
    }
    auto restored = make_test_request(tokens);
    bool completed = false;
    pool.prefetch_from_storage(
        restored, [&completed](std::shared_ptr<Request>) { completed = true; });
    auto drain = folly::makeGuard([&engine]() { engine.drain_workers(); });
    ASSERT_TRUE(completed) << "completed offload must serve the Host prefix";
    EXPECT_EQ(engine.request_count(), 0u);
    Sequence& sequence = *restored->sequences().front();
    EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), unit);
    const auto point = pool.select_host_cache_restore(&sequence, 1);
    EXPECT_EQ(point.restore_target_tokens, unit);
    EXPECT_EQ(point.copy_units, 1u);
  }
}

TEST(HierarchyBlockManagerPoolTest,
     DeepSeekV4HostPrefixCacheUsesImageIdentity) {
  for (const bool is_v41 : {false, true}) {
    SCOPED_TRACE(is_v41 ? "deepseek_v41" : "deepseek_v4");
    const uint32_t unit = is_v41 ? kDsv4FixedBlockSize : kUnitTokens;
    auto options = make_v41_options(unit);
    options.hasher_type(BlockHasherType::MM);
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(options, &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });

    const std::vector<int32_t> tokens(unit + 1, 83);
    const MMData image = make_image_mm_data(
        /*offset=*/0,
        /*length=*/static_cast<int32_t>(unit),
        /*key_byte=*/3);
    Sequence source = make_test_sequence(0, tokens, false, image);
    ASSERT_TRUE(pool.allocate(&source, unit));
    source.kv_state().set_kv_cache_tokens_num(unit);
    pool.deallocate(&source);
    pool.transfer_blocks();
    ASSERT_FALSE(engine.offloads_.empty());

    // Force the following probes through Host rather than the still-warm HBM
    // prefix cache, as happens after the device copy is reclaimed.
    pool.BlockManagerPool::reset_prefix_cache();
    Sequence same_image = make_test_sequence(1, tokens, false, image);
    pool.allocate_shared(&same_image);
    EXPECT_EQ(same_image.host_kv_state().kv_cache_tokens_num(), unit);
    pool.deallocate(&same_image);

    const MMData different_image = make_image_mm_data(
        /*offset=*/0,
        /*length=*/static_cast<int32_t>(unit),
        /*key_byte=*/4);
    Sequence changed_image =
        make_test_sequence(2, tokens, false, different_image);
    pool.allocate_shared(&changed_image);
    EXPECT_EQ(changed_image.host_kv_state().kv_cache_tokens_num(), 0u);
    pool.deallocate(&changed_image);
  }
}

TEST(HierarchyBlockManagerPoolTest, V41NarrowWindowOffloadKeepsSwaTails) {
  auto options = make_v41_options();
  options.sliding_window_size(128).swa_blocks_per_seq(1);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(513, 87);
  Sequence source = make_test_sequence(0, tokens);
  ASSERT_TRUE(pool.allocate(&source, 512));
  source.kv_state().set_kv_cache_tokens_num(512);
  pool.deallocate(&source);
  pool.transfer_blocks();
  EXPECT_EQ(transfer_count(engine, BlockType::SWA), 2u);
  EXPECT_EQ(transfer_count(engine, BlockType::KV), 2u);
  auto restored = make_test_request(tokens);
  bool completed = false;
  pool.prefetch_from_storage(
      restored, [&completed](std::shared_ptr<Request>) { completed = true; });
  auto drain = folly::makeGuard([&engine]() { engine.drain_workers(); });
  ASSERT_TRUE(completed);
  Sequence& sequence = *restored->sequences().front();
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 512u);
  const auto swa = sequence.host_kv_state().blocks(BlockType::SWA);
  ASSERT_EQ(swa.size(), 2u);
  EXPECT_TRUE(swa[0].is_valid());
  EXPECT_TRUE(swa[1].is_valid());
  EXPECT_EQ(pool.select_host_cache_restore(&sequence, 1).restore_target_tokens,
            256u);
}

TEST(HierarchyBlockManagerPoolTest, V41DSparkOffloadsOnlyCommittedCheckpoints) {
  for (const size_t committed : {255u, 256u, 512u}) {
    SCOPED_TRACE(committed);
    auto options = make_v41_options();
    options.num_speculative_tokens(6)
        .num_embedding_blocks(4)
        .sliding_window_size(128)
        .swa_blocks_per_seq(1);
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(options, &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
    Sequence source = make_test_sequence(0, std::vector<int32_t>(519, 31));
    ASSERT_TRUE(pool.allocate(&source, 519));
    source.kv_state().set_kv_cache_tokens_num(committed);
    pool.deallocate(&source);
    pool.transfer_blocks();
    const size_t units = committed / options.block_size();
    EXPECT_EQ(transfer_count(engine, BlockType::KV), units);
    EXPECT_EQ(transfer_count(engine, BlockType::SWA), units);
  }
}

TEST(HierarchyBlockManagerPoolTest,
     AllocateSharedAdvertisesC128AlignedCopyUnits) {
  auto options = make_typed_cache_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(2 * kUnitTokens + 1, 83);
  publish_host_prefix(pool, engine, tokens);
  Sequence sequence = make_test_sequence(0, tokens);
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  ASSERT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 2 * kUnitTokens);
  EXPECT_TRUE(engine.loads_.empty());
  const auto point = pool.select_host_cache_restore(&sequence, 2);
  EXPECT_EQ(point.copy_units, 2u);
  EXPECT_EQ(point.restore_target_tokens, 2 * kUnitTokens);
  pool.trim_host_cache(&sequence, point);
  ASSERT_TRUE(pool.allocate(&sequence, 2 * kUnitTokens + 1));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 2 * kUnitTokens);
  dispatch_loads(pool, sequence);
  ASSERT_EQ(engine.loads_.size(), 1u);
  size_t kv = 0;
  for (const auto& info : engine.loads_.front().infos) {
    EXPECT_EQ(info.transfer_type, TransferType::H2D);
    kv += info.block_type == BlockType::KV;
    EXPECT_GT(info.src_block_id, 0);
    EXPECT_GT(info.dst_block_id, 0);
    const auto host = sequence.host_kv_state().blocks(info.block_type);
    const auto device = sequence.kv_state().blocks(info.block_type);
    const auto source =
        std::find_if(host.begin(), host.end(), [&info](const Block& block) {
          return block.id() == info.src_block_id;
        });
    const auto destination =
        std::find_if(device.begin(), device.end(), [&info](const Block& block) {
          return block.id() == info.dst_block_id;
        });
    ASSERT_NE(source, host.end());
    ASSERT_NE(destination, device.end());
    EXPECT_EQ(XXH3Key(info.hash_key),
              XXH3Key(source->get_immutable_hash_value()));
    EXPECT_EQ(XXH3Key(info.hash_key),
              XXH3Key(destination->get_immutable_hash_value()));
  }
  EXPECT_EQ(kv, 2u);
  pool.deallocate(&sequence);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest, FlatKvRestoreBudgetRemainsBlockLinear) {
  auto options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(1025, 23);
  publish_host_prefix(pool, engine, tokens);
  auto request = make_test_request(tokens);
  Sequence& sequence = *request->sequences().front();
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.host_cache_copy_units(), 8u);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  const auto point = pool.select_host_cache_restore(&sequence, 4);
  pool.trim_host_cache(&sequence, point);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 512u);
  EXPECT_EQ(sequence.num_prefix_cache_tokens(), 512u);
  ASSERT_TRUE(pool.allocate(&sequence, 1025));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 512u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 4u);
  EXPECT_TRUE(engine.loads_.empty());
  dispatch_loads(pool, sequence);
  ASSERT_EQ(engine.loads_.size(), 1u);
  ASSERT_EQ(engine.loads_.front().infos.size(), 4u);
  const auto host = sequence.host_kv_state().blocks(BlockType::KV);
  const auto device = sequence.kv_state().blocks(BlockType::KV);
  for (size_t index = 0; index < 4; ++index) {
    const auto& info = engine.loads_.front().infos[index];
    EXPECT_EQ(info.src_block_id, host[index].id());
    EXPECT_EQ(info.dst_block_id, device[index].id());
    EXPECT_EQ(info.block_type, BlockType::KV);
    EXPECT_EQ(info.transfer_type, TransferType::H2D);
  }
  ASSERT_TRUE(pool.allocate(&sequence, 1025));
  dispatch_loads(pool, sequence);
  EXPECT_EQ(engine.loads_.size(), 1u);
  pool.deallocate(&sequence);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest,
     ZeroCopyRestoreTargetDoesNotReprobeHostCache) {
  auto options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(1025, 67);
  publish_host_prefix(pool, engine, tokens);
  Sequence sequence = make_test_sequence(0, tokens);
  pool.allocate_shared(&sequence);
  const auto point = pool.select_host_cache_restore(&sequence, 0);
  EXPECT_EQ(point.restore_target_tokens, 0u);
  pool.trim_host_cache(&sequence, point);
  ASSERT_TRUE(pool.allocate(&sequence, 1));
  dispatch_loads(pool, sequence);
  EXPECT_TRUE(engine.loads_.empty());
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.num_prefix_cache_tokens(), 0u);
  pool.deallocate(&sequence);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest, FailedHbmAllocationDoesNotQueueH2d) {
  auto options = make_flat_kv_options();
  options.num_blocks(8).enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(1025, 23);
  publish_host_prefix(pool, engine, tokens);
  Sequence sequence = make_test_sequence(0, tokens);
  pool.allocate_shared(&sequence);
  const auto point = pool.select_host_cache_restore(&sequence, 4);
  pool.trim_host_cache(&sequence, point);
  const auto before = pool.num_used_blocks();
  EXPECT_FALSE(pool.allocate(&sequence, 1025));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_TRUE(sequence.host_kv_state().has_any_blocks());
  dispatch_loads(pool, sequence);
  EXPECT_TRUE(engine.loads_.empty());
  EXPECT_EQ(pool.num_used_blocks(), before);
  // Cancellation before H2D must not copy the larger logical Host match as
  // though it were completed HBM work.
  pool.deallocate(&sequence);
  pool.transfer_blocks();
  EXPECT_TRUE(engine.offloads_.empty());
  pool.allocate_shared(&sequence);
  const auto retry_point = pool.select_host_cache_restore(&sequence, 4);
  pool.trim_host_cache(&sequence, retry_point);
  ASSERT_TRUE(pool.allocate(&sequence, 512));
  dispatch_loads(pool, sequence);
  EXPECT_EQ(engine.loads_.front().infos.size(), 4u);
  pool.deallocate(&sequence);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest, HostAllocationFailureRejectsHbmAllocation) {
  auto options = make_flat_kv_options();
  options.host_num_blocks(3).enable_prefix_cache(false);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const auto before = pool.num_free_blocks();
  Sequence sequence = make_test_sequence(0, std::vector<int32_t>(385, 23));
  EXPECT_FALSE(pool.allocate(&sequence, 384));
  EXPECT_FALSE(sequence.has_any_blocks());
  EXPECT_EQ(pool.num_free_blocks(), before);
  EXPECT_TRUE(engine.loads_.empty());
  EXPECT_TRUE(engine.offloads_.empty());
  ASSERT_TRUE(pool.allocate(&sequence, 256));
  pool.deallocate(&sequence);
  pool.transfer_blocks();
  EXPECT_EQ(pool.num_free_blocks(), before);
}
TEST(HierarchyBlockManagerPoolTest,
     PrefixRetryKeepsMountedStateAfterHostGrowthFailure) {
  auto options = make_flat_kv_options();
  options.host_num_blocks(9).enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(1537, 45);
  Sequence source = make_test_sequence(0, tokens);
  ASSERT_TRUE(pool.allocate(&source, 512));
  source.kv_state().set_kv_cache_tokens_num(512);
  ASSERT_TRUE(pool.allocate(&source, 1024));
  pool.transfer_blocks();
  // A real Host-prefix consumer pins the completed first chunk.
  auto pin = make_test_request(std::vector<int32_t>(513, 45));
  bool pin_ready = false;
  pool.prefetch_from_storage(
      pin, [&pin_ready](std::shared_ptr<Request>) { pin_ready = true; });
  auto finish_pin = folly::makeGuard([&engine]() { engine.drain_workers(); });
  ASSERT_TRUE(pin_ready);
  ASSERT_EQ(pin->sequences().front()->host_kv_state().kv_cache_tokens_num(),
            512u);
  const int32_t first_id = source.kv_state().blocks(BlockType::KV).front().id();
  source.kv_state().set_kv_cache_tokens_num(1024);
  EXPECT_FALSE(pool.allocate(&source, 1536));
  EXPECT_EQ(source.kv_state().kv_cache_tokens_num(), 1024u);
  EXPECT_EQ(source.kv_state().blocks(BlockType::KV).front().id(), first_id);
  EXPECT_EQ(source.host_kv_state().num_blocks(BlockType::KV), 8u);
  EXPECT_EQ(engine.offloads_.size(), 1u);
  pool.deallocate(pin->sequences().front().get());
  ASSERT_TRUE(pool.allocate(&source, 1536));
  EXPECT_EQ(source.kv_state().blocks(BlockType::KV).front().id(), first_id);
  EXPECT_EQ(source.host_kv_state().num_blocks(BlockType::KV), 12u);
  pool.transfer_blocks();
  pool.deallocate(&source);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest,
     PrefixRetryKeepsMountedStateAfterHbmGrowthFailure) {
  auto options = make_flat_kv_options();
  options.num_blocks(13);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  Sequence source = make_test_sequence(0, std::vector<int32_t>(1537, 47));
  ASSERT_TRUE(pool.allocate(&source, 512));
  source.kv_state().set_kv_cache_tokens_num(512);
  ASSERT_TRUE(pool.allocate(&source, 1024));
  pool.transfer_blocks();
  Sequence pressure = make_test_sequence(1, std::vector<int32_t>(513, 99));
  ASSERT_TRUE(pool.allocate(&pressure, 512));
  source.kv_state().set_kv_cache_tokens_num(1024);
  const int32_t first_id = source.kv_state().blocks(BlockType::KV).front().id();
  EXPECT_FALSE(pool.allocate(&source, 1536));
  EXPECT_EQ(source.kv_state().num_blocks(BlockType::KV), 8u);
  EXPECT_EQ(source.kv_state().kv_cache_tokens_num(), 1024u);
  EXPECT_EQ(engine.offloads_.size(), 1u);
  pool.deallocate(&pressure);
  ASSERT_TRUE(pool.allocate(&source, 1536));
  EXPECT_EQ(source.kv_state().blocks(BlockType::KV).front().id(), first_id);
  pool.transfer_blocks();
  pool.deallocate(&source);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest, LastRequestDeallocateDropsFailedGrowth) {
  auto options = make_flat_kv_options();
  options.num_blocks(10).host_num_blocks(32);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const auto free_before = pool.num_free_blocks();
  Sequence sequence = make_test_sequence(0, std::vector<int32_t>(1537, 45));
  ASSERT_TRUE(pool.allocate(&sequence, 512));
  sequence.kv_state().set_kv_cache_tokens_num(512);
  ASSERT_TRUE(pool.allocate(&sequence, 1024));
  pool.transfer_blocks();
  sequence.kv_state().set_kv_cache_tokens_num(1024);
  EXPECT_FALSE(pool.allocate(&sequence, 1536));
  pool.deallocate(&sequence);
  pool.transfer_blocks();
  pool.reset_prefix_cache();
  EXPECT_FALSE(sequence.has_any_blocks());
  EXPECT_EQ(pool.num_free_blocks(), free_before);
  Sequence replacement = make_test_sequence(1, std::vector<int32_t>(1025, 91));
  ASSERT_TRUE(pool.allocate(&replacement, 1024));
  pool.deallocate(&replacement);
  pool.transfer_blocks();
}

struct OffloadOutcome {
  size_t result_count;
  int32_t failed_worker;
  bool exception;
  bool publish;
};

class HostOffloadCompletionTest
    : public ::testing::TestWithParam<OffloadOutcome> {};

TEST_P(HostOffloadCompletionTest,
       PublishesOnlyCompleteCopiesAfterAllWorkersEnd) {
  const auto& outcome = GetParam();
  FakePrefetchEngine engine(8);
  engine.hold_offloads(outcome.result_count);
  auto options = make_flat_kv_options();
  options.host_num_blocks(3);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain = folly::makeGuard([&engine]() { engine.drain_offloads(); });
  const size_t free_before = pool.num_free_blocks().front();
  const std::vector<int32_t> tokens(257, 73);
  Sequence source = make_test_sequence(0, tokens);
  ASSERT_TRUE(pool.allocate(&source, 128));
  source.kv_state().set_kv_cache_tokens_num(128);
  ASSERT_TRUE(pool.allocate(&source, 256));
  pool.transfer_blocks();
  ASSERT_EQ(engine.offloads_.size(), 1u);
  ASSERT_EQ(engine.offloads_.front().infos.size(), 1u);
  pool.deallocate(&source);
  pool.reset_prefix_cache();
  Sequence pressure = make_test_sequence(1, std::vector<int32_t>(257, 91));
  for (size_t worker = 0; worker < outcome.result_count; ++worker) {
    EXPECT_TRUE(pool.has_pending_async_block_release());
    EXPECT_EQ(pool.num_free_blocks().front(), free_before - 1);
    // Two Host slots are required; one is still owned by the in-flight copy.
    EXPECT_FALSE(pool.allocate(&pressure, 256));
    EXPECT_FALSE(pressure.has_any_blocks());
    if (static_cast<int32_t>(worker) == outcome.failed_worker) {
      engine.finish_offload_worker(0, worker, /*copied=*/0, outcome.exception);
    } else {
      engine.finish_offload_worker(0, worker);
    }
  }
  EXPECT_FALSE(pool.has_pending_async_block_release());
  pool.BlockManagerPool::reset_prefix_cache();
  EXPECT_EQ(pool.num_free_blocks().front(), free_before);
  Sequence restored = make_test_sequence(2, std::vector<int32_t>(129, 73));
  pool.allocate_shared(&restored);
  EXPECT_EQ(restored.host_kv_state().kv_cache_tokens_num(),
            outcome.publish ? 128u : 0u);
  pool.deallocate(&restored);
  ASSERT_TRUE(pool.allocate(&pressure, 256));
  pool.deallocate(&pressure);
}

INSTANTIATE_TEST_SUITE_P(
    Workers,
    HostOffloadCompletionTest,
    ::testing::Values(OffloadOutcome{8, -1, false, true},
                      OffloadOutcome{8, 2, false, false},
                      OffloadOutcome{8, 2, true, false},
                      OffloadOutcome{2, -1, false, false}));

class TypedHostOffloadTest : public ::testing::TestWithParam<bool> {};

TEST_P(TypedHostOffloadTest, PreservesTypedReservationsUntilAllCopiesEnd) {
  FakePrefetchEngine engine(8);
  engine.hold_offloads(8);
  HierarchyBlockManagerPool pool(make_typed_cache_options(), &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain = folly::makeGuard([&engine]() { engine.drain_offloads(); });
  Sequence source =
      make_test_sequence(0, std::vector<int32_t>(kUnitTokens + 129, 73));
  ASSERT_TRUE(pool.allocate(&source, kUnitTokens));
  source.kv_state().set_kv_cache_tokens_num(kUnitTokens);
  ASSERT_TRUE(pool.allocate(&source, kUnitTokens + 128));
  pool.transfer_blocks();
  ASSERT_EQ(engine.offloads_.size(), 1u);
  const uint32_t copies =
      static_cast<uint32_t>(engine.offloads_.front().infos.size());
  ASSERT_EQ(copies, 2u);
  EXPECT_EQ(transfer_count(engine, BlockType::KV), 1u);
  EXPECT_EQ(transfer_count(engine, BlockType::SWA), 1u);
  pool.deallocate(&source);
  pool.reset_prefix_cache();
  for (size_t worker = 0; worker < 8; ++worker) {
    EXPECT_TRUE(pool.has_pending_async_block_release());
    const auto usage = pool.block_utilization_by_type_per_dp();
    EXPECT_GT(usage.front().at(BlockType::KV), 0.0);
    Sequence early =
        make_test_sequence(1, std::vector<int32_t>(kUnitTokens + 1, 73));
    pool.allocate_shared(&early);
    EXPECT_EQ(early.host_kv_state().kv_cache_tokens_num(), 0u);
    pool.deallocate(&early);
    engine.finish_offload_worker(0,
                                 worker,
                                 !GetParam() && worker == 2
                                     ? std::optional<uint32_t>(copies - 1)
                                     : std::nullopt);
  }
  EXPECT_FALSE(pool.has_pending_async_block_release());
  const auto usage = pool.block_utilization_by_type_per_dp();
  EXPECT_DOUBLE_EQ(usage.front().at(BlockType::KV), 0.0);
  Sequence restored =
      make_test_sequence(2, std::vector<int32_t>(kUnitTokens + 1, 73));
  pool.allocate_shared(&restored);
  EXPECT_EQ(restored.host_kv_state().kv_cache_tokens_num(),
            GetParam() ? kUnitTokens : 0u);
  pool.deallocate(&restored);
}

INSTANTIATE_TEST_SUITE_P(Completion, TypedHostOffloadTest, ::testing::Bool());

TEST(HierarchyBlockManagerPoolTest, D2hUsesSequenceDpRank) {
  auto options = make_flat_kv_options();
  options.block_size(32);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 2);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  Sequence source = make_test_sequence(0, std::vector<int32_t>(97, 83));
  source.set_dp_rank(1);
  ASSERT_TRUE(pool.allocate(&source, 32));
  source.kv_state().set_kv_cache_tokens_num(32);
  ASSERT_TRUE(pool.allocate(&source, 64));
  const int32_t source_id =
      source.kv_state().blocks(BlockType::KV).front().id();
  pool.transfer_blocks();
  ASSERT_EQ(engine.offloads_.size(), 1u);
  EXPECT_EQ(engine.offloads_.front().dp_rank, 1u);
  ASSERT_EQ(engine.offloads_.front().infos.size(), 1u);
  EXPECT_EQ(engine.offloads_.front().infos.front().src_block_id, source_id);
  EXPECT_EQ(XXH3Key(engine.offloads_.front().infos.front().hash_key),
            source.block_hashes(32).front());
  pool.deallocate(&source);
  pool.transfer_blocks();
}
TEST(HierarchyBlockManagerPoolTest, H2dUsesSequenceDpRank) {
  auto options = make_flat_kv_options();
  options.block_size(32).enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 2);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(129, 83);
  publish_host_prefix(pool, engine, tokens, 1);
  Sequence restored = make_test_sequence(0, tokens);
  restored.set_dp_rank(1);
  pool.allocate_shared(&restored);
  const auto point = pool.select_host_cache_restore(&restored, 2);
  pool.trim_host_cache(&restored, point);
  ASSERT_TRUE(pool.allocate(&restored, 65));
  dispatch_loads(pool, restored, 2);
  ASSERT_EQ(engine.loads_.size(), 1u);
  EXPECT_EQ(engine.loads_.front().dp_rank, 1u);
  ASSERT_EQ(engine.loads_.front().infos.size(), 2u);
  EXPECT_NE(engine.loads_.front().batch_id, 0u);
  pool.deallocate(&restored);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest, Dsv4LargeChunksPersistEveryC128Checkpoint) {
  for (const size_t units : {1u, 3u}) {
    SCOPED_TRACE(units);
    auto options = make_typed_cache_options();
    options.enable_kvcache_store(true);
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(options, &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
    const size_t completed_tokens = units * kUnitTokens;
    const std::vector<int32_t> tokens(completed_tokens + kUnitTokens + 1, 97);
    Sequence source = make_test_sequence(0, tokens);
    ASSERT_TRUE(pool.allocate(&source, completed_tokens));
    source.kv_state().set_kv_cache_tokens_num(completed_tokens);
    ASSERT_TRUE(pool.allocate(&source, completed_tokens + kUnitTokens));
    pool.transfer_blocks();
    EXPECT_EQ(transfer_count(engine, BlockType::KV), units);
    EXPECT_EQ(transfer_count(engine, BlockType::SWA), 1u);
    pool.deallocate(&source);
    pool.transfer_blocks();
    auto restored =
        make_test_request(std::vector<int32_t>(completed_tokens + 1, 97));
    bool done = false;
    pool.prefetch_from_storage(
        restored, [&done](std::shared_ptr<Request>) { done = true; });
    auto drain = folly::makeGuard([&engine]() { engine.drain_workers(); });
    ASSERT_TRUE(done);
    EXPECT_EQ(engine.request_count(), 0u);
    EXPECT_EQ(
        restored->sequences().front()->host_kv_state().kv_cache_tokens_num(),
        completed_tokens);
  }
}

TEST(HierarchyBlockManagerPoolTest, Dsv4OffloadWaitsForCompleteC128Unit) {
  for (const size_t committed_tokens : {kUnitTokens - 1, kUnitTokens}) {
    SCOPED_TRACE(committed_tokens);
    auto options = make_typed_cache_options();
    options.enable_kvcache_store(true);
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(options, &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
    Sequence source =
        make_test_sequence(0, std::vector<int32_t>(kUnitTokens + 1, 101));
    ASSERT_TRUE(pool.allocate(&source, kUnitTokens));
    source.kv_state().set_kv_cache_tokens_num(committed_tokens);
    pool.deallocate(&source);
    EXPECT_TRUE(engine.offloads_.empty());
    pool.transfer_blocks();

    const size_t expected_units = committed_tokens / kUnitTokens;
    EXPECT_EQ(transfer_count(engine, BlockType::KV), expected_units);
    EXPECT_EQ(transfer_count(engine, BlockType::SWA), expected_units);

    Sequence restored =
        make_test_sequence(1, std::vector<int32_t>(kUnitTokens + 1, 101));
    pool.allocate_shared(&restored);
    EXPECT_EQ(restored.host_kv_state().kv_cache_tokens_num(),
              expected_units * kUnitTokens);
    pool.deallocate(&restored);
    pool.transfer_blocks();
  }
}

TEST(HierarchyBlockManagerPoolTest,
     Dsv4OneMillionTokensReuseHostBlocksWithSmallChunks) {
  constexpr size_t kContextTokens = 1024 * 1024;
  auto options = make_typed_cache_options();
  options.num_blocks(kContextTokens / kUnitTokens + 2)
      .swa_num_blocks(40)
      .max_tokens_per_batch(kUnitTokens)
      .host_num_blocks_by_type({{BlockType::SWA, 34}, {BlockType::KV, 3}});
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  Sequence source =
      make_test_sequence(0, std::vector<int32_t>(kContextTokens + 1, 97));
  for (size_t target = kUnitTokens; target <= kContextTokens;
       target += kUnitTokens) {
    ASSERT_TRUE(pool.allocate(&source, target)) << target;
    pool.transfer_blocks();
    EXPECT_FALSE(pool.has_pending_async_block_release());
    source.kv_state().set_kv_cache_tokens_num(target);
  }
  pool.deallocate(&source);
  pool.transfer_blocks();
  const size_t units = kContextTokens / kUnitTokens;
  EXPECT_EQ(transfer_count(engine, BlockType::KV), units);
  EXPECT_EQ(transfer_count(engine, BlockType::SWA), units);
  EXPECT_FALSE(source.has_any_blocks());
  pool.reset_prefix_cache();
  EXPECT_EQ(pool.num_used_blocks(), (std::vector<size_t>{0}));
}

TEST(HierarchyBlockManagerPoolTest,
     H2dRestoreIsPublishedToDevicePrefixCacheDuringAllocation) {
  auto options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(513, 89);
  publish_host_prefix(pool, engine, tokens);
  Sequence first = make_test_sequence(0, tokens);
  pool.allocate_shared(&first);
  const auto point = pool.select_host_cache_restore(&first, 4);
  pool.trim_host_cache(&first, point);
  ASSERT_TRUE(pool.allocate(&first, 513));
  dispatch_loads(pool, first);
  ASSERT_EQ(engine.loads_.size(), 1u);
  ASSERT_EQ(engine.loads_.front().infos.size(), 4u);
  Sequence second = make_test_sequence(1, tokens);
  pool.allocate_shared(&second);
  EXPECT_EQ(second.kv_state().kv_cache_tokens_num(), 512u);
  ASSERT_TRUE(pool.allocate(&second, 513));
  dispatch_loads(pool, second);
  EXPECT_EQ(engine.loads_.size(), 1u);
  const auto a = first.kv_state().blocks(BlockType::KV);
  const auto b = second.kv_state().blocks(BlockType::KV);
  ASSERT_EQ(a.size(), 5u);
  ASSERT_EQ(b.size(), 5u);
  for (size_t index = 0; index < 4; ++index) {
    EXPECT_EQ(a[index].id(), b[index].id());
  }
  EXPECT_NE(a.back().id(), b.back().id());
  pool.deallocate(&first);
  pool.deallocate(&second);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest,
     SharedHostPrefixCopiesToEachPrivateDeviceAllocation) {
  constexpr size_t kPromptTokens = 65;
  constexpr size_t kBlockSize = 16;
  constexpr size_t kPrefixBlocks = 4;
  for (const size_t hbm_prefix_tokens : {0u, 16u, 17u, 32u, 48u, 64u}) {
    SCOPED_TRACE(hbm_prefix_tokens);
    auto options = make_flat_kv_options();
    options.block_size(kBlockSize)
        .hasher_type(BlockHasherType::MTP_TEXT)
        .enable_kvcache_store(true);
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(options, &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
    const std::vector<int32_t> tokens(kPromptTokens, 41);
    std::unique_ptr<Sequence> device_seed;
    if (hbm_prefix_tokens > 0) {
      device_seed = std::make_unique<Sequence>(make_test_sequence(
          /*index=*/2, std::vector<int32_t>(hbm_prefix_tokens + 1, 41)));
      ASSERT_TRUE(pool.allocate(device_seed.get(), hbm_prefix_tokens + 1));
      device_seed->kv_state().set_kv_cache_tokens_num(hbm_prefix_tokens);
      pool.cache(device_seed.get(), hbm_prefix_tokens + 1);
    }
    publish_host_prefix(pool, engine, tokens);

    Sequence first = make_test_sequence(/*index=*/0, tokens);
    Sequence second = make_test_sequence(/*index=*/1, tokens);
    pool.allocate_shared(&first);
    pool.allocate_shared(&second);
    const size_t hbm_prefix_blocks = hbm_prefix_tokens / kBlockSize;
    for (Sequence* sequence : {&first, &second}) {
      EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(),
                kPrefixBlocks * kBlockSize);
      EXPECT_EQ(sequence->kv_state().num_cached_blocks(BlockType::KV),
                hbm_prefix_blocks);
      // A live cursor may extend past its last complete cached block.
      sequence->kv_state().set_kv_cache_tokens_num(hbm_prefix_tokens);
      ASSERT_TRUE(pool.allocate(sequence, kPromptTokens));
      EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(),
                kPrefixBlocks * kBlockSize);
    }
    std::vector<Batch> batches(1);
    batches.front().add(&first);
    batches.front().add(&second);
    pool.transfer_blocks(batches);

    const size_t expected_transfers = 2 * (kPrefixBlocks - hbm_prefix_blocks);
    ASSERT_EQ(engine.loads_.size(), expected_transfers == 0 ? 0u : 1u);
    if (expected_transfers > 0) {
      ASSERT_EQ(engine.loads_.front().infos.size(), expected_transfers);
    }
    const auto first_host = first.host_kv_state().blocks(BlockType::KV);
    const auto second_host = second.host_kv_state().blocks(BlockType::KV);
    const auto first_device = first.kv_state().blocks(BlockType::KV);
    const auto second_device = second.kv_state().blocks(BlockType::KV);
    ASSERT_EQ(first_host.size(), kPromptTokens / kBlockSize + 1);
    ASSERT_EQ(second_host.size(), first_host.size());
    ASSERT_EQ(first_device.size(), kPromptTokens / kBlockSize + 1);
    ASSERT_EQ(second_device.size(), first_device.size());
    for (size_t block = 0; block < kPrefixBlocks; ++block) {
      EXPECT_EQ(first_host[block].id(), second_host[block].id());
      if (block < hbm_prefix_blocks) {
        EXPECT_EQ(first_device[block].id(), second_device[block].id());
        continue;
      }
      EXPECT_NE(first_device[block].id(), second_device[block].id());
      auto copies_to = [&](int32_t destination_id) {
        size_t copies = 0;
        for (const CapturedTransfer& capture : engine.loads_) {
          copies += std::count_if(
              capture.infos.begin(),
              capture.infos.end(),
              [&](const BlockTransferInfo& info) {
                return info.transfer_type == TransferType::H2D &&
                       info.block_type == BlockType::KV &&
                       info.src_block_id == first_host[block].id() &&
                       info.dst_block_id == destination_id;
              });
        }
        return copies;
      };
      EXPECT_EQ(copies_to(first_device[block].id()), 1);
      EXPECT_EQ(copies_to(second_device[block].id()), 1);
    }

    ASSERT_TRUE(pool.allocate(&first, kPromptTokens));
    ASSERT_TRUE(pool.allocate(&second, kPromptTokens));
    pool.transfer_blocks(batches);
    EXPECT_EQ(engine.loads_.size(), expected_transfers == 0 ? 0u : 1u);
    pool.deallocate(&first);
    pool.deallocate(&second);
    if (device_seed != nullptr) {
      pool.deallocate(device_seed.get());
    }
    pool.transfer_blocks();
  }
}

TEST(HierarchyBlockManagerPoolTest,
     CancelledStoragePrefetchReleasesReservedBlocks) {
  auto options = make_flat_kv_options();
  options.host_num_blocks(5).enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto request = make_test_request(std::vector<int32_t>(513, 83));
  bool done = false;
  pool.prefetch_from_storage(
      request, [&done](std::shared_ptr<Request>) { done = true; });
  auto drain = folly::makeGuard([&engine]() { engine.drain_workers(); });
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_EQ(engine.request().units.size(), 4u);
  request->set_cancel();
  engine.finish_worker(0, 2);
  EXPECT_FALSE(done);
  EXPECT_TRUE(request->sequences().front()->host_kv_state().has_any_blocks());
  engine.finish_worker(1, 2);
  EXPECT_TRUE(done);
  EXPECT_FALSE(request->sequences().front()->host_kv_state().has_any_blocks());
  const size_t calls = engine.request_count();
  // Full-capacity replacement is the observable reclaim probe, not a private
  // Host free-list getter. Cancellation must not leave a discoverable prefix.
  publish_host_prefix(pool, engine, std::vector<int32_t>(513, 83));
  EXPECT_EQ(engine.request_count(), calls + 1);
}

TEST(HierarchyBlockManagerPoolTest,
     FailedStoragePrefetchDoesNotPublishHostPrefix) {
  auto options = make_flat_kv_options();
  options.host_num_blocks(5).enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto request = make_test_request(std::vector<int32_t>(513, 83));
  bool done = false;
  pool.prefetch_from_storage(
      request, [&done](std::shared_ptr<Request>) { done = true; });
  auto drain = folly::makeGuard([&engine]() { engine.drain_workers(); });
  ASSERT_NE(engine.result(), nullptr);
  engine.finish_worker(0, 2, /*worker_ok=*/false);
  EXPECT_FALSE(done);
  engine.finish_worker(1, 2);
  EXPECT_TRUE(done);
  EXPECT_FALSE(request->sequences().front()->host_kv_state().has_any_blocks());
  const size_t calls = engine.request_count();
  publish_host_prefix(pool, engine, std::vector<int32_t>(513, 83));
  EXPECT_EQ(engine.request_count(), calls + 1);
}

TEST(HierarchyBlockManagerPoolTest,
     MultiSequenceStoragePrefetchUsesOnlyFirstSequence) {
  auto options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto request = make_test_request(std::vector<int32_t>(513, 83), 2);
  ASSERT_TRUE(request->expand_sequences(false));
  ASSERT_EQ(request->sequences().size(), 2u);
  bool done = false;
  pool.prefetch_from_storage(
      request, [&done](std::shared_ptr<Request>) { done = true; });
  auto drain = folly::makeGuard([&engine]() { engine.drain_workers(); });
  ASSERT_EQ(engine.request_count(), 1u);
  const size_t units = engine.request().units.size();
  engine.finish_worker(0, units);
  engine.finish_worker(1, units);
  ASSERT_TRUE(done);
  EXPECT_TRUE(request->sequences()[0]->host_kv_state().has_any_blocks());
  EXPECT_FALSE(request->sequences()[1]->host_kv_state().has_any_blocks());
}

TEST(HierarchyBlockManagerPoolTest,
     LinearTailHostOffloadBecomesVisibleAfterAllWorkersComplete) {
  const int32_t old_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  auto restore = folly::makeGuard([old_stride]() {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        old_stride;
  });
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 8;
  for (const bool warmup : {false, true}) {
    SCOPED_TRACE(warmup);
    auto options = make_flat_kv_options();
    options.block_size(4)
        .enable_linear_state(true)
        .linear_state_num_slots(64)
        .enable_kvcache_store(true)
        .host_num_blocks_by_type(
            {{BlockType::KV, 128}, {BlockType::LINEAR, 64}});
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(options, &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
    const std::vector<int32_t> shorter_tokens(9, 19);
    if (warmup) {
      // Seed the 8-token checkpoint as the tail of a different prompt, not
      // as an intermediate chunk of the 13-token prompt below.
      Sequence shorter_source = make_test_sequence(/*index=*/0, shorter_tokens);
      const auto cleanup = folly::makeGuard([&pool, &shorter_source] {
        pool.deallocate(&shorter_source);
        pool.transfer_blocks();
      });
      ASSERT_TRUE(pool.allocate(&shorter_source, shorter_tokens.size()));
      shorter_source.kv_state().set_kv_cache_tokens_num(8);
      ASSERT_TRUE(pool.allocate(&shorter_source, shorter_tokens.size()));
    }
    const std::vector<int32_t> tokens(13, 19);
    Sequence source = make_test_sequence(0, tokens);
    ASSERT_TRUE(pool.allocate(&source, 12));
    pool.transfer_blocks();
    engine.hold_offloads(2);
    source.kv_state().set_kv_cache_tokens_num(12);
    ASSERT_TRUE(pool.allocate(&source, 13));
    const int32_t source_id =
        source.kv_state().blocks(BlockType::LINEAR).front().id();
    pool.transfer_blocks();
    ASSERT_EQ(engine.offloads_.size(), warmup ? 2u : 1u);
    const size_t copy_index = engine.offloads_.size() - 1;
    const auto& infos = engine.offloads_.back().infos;
    const auto linear = std::find_if(
        infos.begin(), infos.end(), [](const BlockTransferInfo& info) {
          return info.block_type == BlockType::LINEAR;
        });
    ASSERT_NE(linear, infos.end());
    EXPECT_EQ(linear->src_block_id, source_id);
    EXPECT_EQ(XXH3Key(linear->hash_key), source.block_hashes(4)[2]);
    const int32_t destination = linear->dst_block_id;
    pool.deallocate(&source);
    EXPECT_TRUE(pool.has_pending_async_block_release());
    engine.finish_offload_worker(copy_index, 0);
    EXPECT_TRUE(pool.has_pending_async_block_release());
    engine.finish_offload_worker(copy_index, 1);
    EXPECT_FALSE(pool.has_pending_async_block_release());
    auto restored = make_test_request(tokens);
    bool done = false;
    pool.prefetch_from_storage(
        restored, [&done](std::shared_ptr<Request>) { done = true; });
    ASSERT_TRUE(done);
    const auto checkpoint =
        restored->sequences().front()->host_kv_state().blocks(
            BlockType::LINEAR);
    ASSERT_EQ(checkpoint.size(), 1u);
    EXPECT_EQ(checkpoint.front().id(), destination);
    EXPECT_EQ(
        restored->sequences().front()->host_kv_state().kv_cache_tokens_num(),
        12u);

    // The tail's actual hash must not be published under a regular boundary.
    // With no 8-token checkpoint, a shorter prompt cannot restore this tail.
    auto shorter = make_test_request(shorter_tokens);
    bool shorter_done = false;
    pool.prefetch_from_storage(
        shorter,
        [&shorter_done](std::shared_ptr<Request>) { shorter_done = true; });
    EXPECT_EQ(
        shorter->sequences().front()->host_kv_state().kv_cache_tokens_num(),
        warmup ? 8u : 0u);
    if (warmup) {
      EXPECT_TRUE(shorter_done);
      EXPECT_NE(shorter->sequences()
                    .front()
                    ->host_kv_state()
                    .blocks(BlockType::LINEAR)
                    .front()
                    .id(),
                destination);
    } else {
      // A short tail may have no complete transferable checkpoint unit.
      // Either immediate completion or closing the workers must keep a miss.
      if (!shorter_done) {
        engine.drain_workers();
      }
      EXPECT_TRUE(shorter_done);
      EXPECT_EQ(
          shorter->sequences().front()->host_kv_state().kv_cache_tokens_num(),
          0u);
    }
  }
}

TEST(HierarchyBlockManagerPoolTest, TryAllocateMatchesHbmAndFloorsHostCursor) {
  auto options = make_typed_cache_options();
  options.instance_is_decode(true).enable_disagg_pd(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  Sequence sequence =
      make_test_sequence(0, std::vector<int32_t>(3 * kUnitTokens + 1, 97));
  ASSERT_TRUE(pool.try_allocate(&sequence));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 3 * kUnitTokens + 1);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 3 * kUnitTokens);
  pool.transfer_blocks();
  EXPECT_TRUE(engine.offloads_.empty());
  EXPECT_TRUE(engine.loads_.empty());
  pool.deallocate(&sequence);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest, SwaTailBlocksClampToAvailableHistory) {
  for (size_t available = 1; available <= 4; ++available) {
    for (size_t window = 1; window <= 4; ++window) {
      const size_t retained = swa_tail_blocks(available, window);
      size_t expected = 0;
      for (size_t index = 0; index < available; ++index) {
        if (index + window >= available) {
          ++expected;
        }
      }
      EXPECT_EQ(retained, expected);
      EXPECT_LE(retained, available);
      EXPECT_LE(retained, window);
    }
  }
  EXPECT_EQ(swa_tail_blocks(0, 4), 0u);
}

TEST(HierarchyBlockManagerPoolTest, V41StorePrefetchGatesBothTokenGroups) {
  auto options = make_typed_cache_options();
  options.block_size(256)
      .swa_block_size(256)
      .num_blocks(64)
      .sliding_window_size(512)
      .swa_blocks_per_seq(2)
      .enable_kvcache_store(true)
      .prefetch_batch_size(2)
      .host_num_blocks_by_type({{BlockType::SWA, 128}, {BlockType::KV, 128}});
  FakePrefetchEngine engine(/*worker_count=*/1);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  auto request = make_test_request(std::vector<int32_t>(513, 87));
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_EQ(engine.request().units.size(), 2u);
  for (const auto& unit : engine.request().units) {
    ASSERT_EQ(unit.gated_blocks.size(), 1u);
    EXPECT_EQ(unit.gated_blocks.front().block_type, BlockType::KV);
    EXPECT_TRUE(unit.has_non_gated);
  }
  EXPECT_EQ(engine.request().units[0].non_gated_blocks.size(), 1u);
  EXPECT_EQ(engine.request().units[1].non_gated_blocks.size(), 2u);
  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/2);
  EXPECT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(), 512u);
  const HostCacheRestorePoint partial =
      pool.select_host_cache_restore(sequence, /*max_copy_units=*/1);
  EXPECT_EQ(partial.restore_target_tokens, 256u);
  EXPECT_EQ(partial.copy_units, 1u);
  // Missing the last block rejects checkpoint 2, but checkpoint 1's
  // complete partial window remains a valid restore point.
  sequence->host_kv_state().mutable_blocks(BlockType::SWA)->back() = Block{};
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(sequence, /*max_copy_units=*/2);
  EXPECT_EQ(selected.restore_target_tokens, 256u);
  EXPECT_EQ(selected.copy_units, 1u);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest, V41StorePrefetchFitsCheckpointSwaCapacity) {
  auto options = make_typed_cache_options();
  options.block_size(256)
      .swa_block_size(256)
      .num_blocks(64)
      .sliding_window_size(256)
      .swa_blocks_per_seq(1)
      .enable_kvcache_store(true)
      .prefetch_batch_size(2)
      // Two physical SWA checkpoints plus the leaf's reserved padding block.
      .host_num_blocks_by_type({{BlockType::SWA, 3}, {BlockType::KV, 128}});
  FakePrefetchEngine engine(/*worker_count=*/1);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  auto request = make_test_request(std::vector<int32_t>(513, 87));
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  EXPECT_EQ(engine.request().units.size(), 2u);
  for (const auto& unit : engine.request().units) {
    EXPECT_EQ(unit.gated_blocks.size(), 1u);
    EXPECT_EQ(unit.non_gated_blocks.size(), 1u);
  }
  // A one-block SWA window keeps only each checkpoint tail.
  const Slice<Block> swa = sequence->host_kv_state().blocks(BlockType::SWA);
  ASSERT_EQ(swa.size(), 2u);
  EXPECT_TRUE(swa[0].is_valid());
  EXPECT_TRUE(swa[1].is_valid());
  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/2);
  EXPECT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(), 512u);
  const HostCacheRestorePoint complete =
      pool.select_host_cache_restore(sequence, /*max_copy_units=*/2);
  EXPECT_EQ(complete.restore_target_tokens, 512u);
  EXPECT_EQ(complete.copy_units, 2u);
  // Finalization retains only the last SWA window, as in V4. Losing it
  // must reject restoration; the earlier tail was already released.
  sequence->host_kv_state().mutable_blocks(BlockType::SWA)->back() = Block{};
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(sequence, /*max_copy_units=*/2);
  EXPECT_EQ(selected.restore_target_tokens, 0u);
  EXPECT_EQ(selected.copy_units, 0u);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     Dsv4WideWindowPrefetchRetainsFullRestoreTail) {
  constexpr size_t kWindowTokens = 2048;
  const size_t unit_tokens = kDsv4FixedBlockSize;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.sliding_window_size(kWindowTokens)
      .swa_blocks_per_seq(kWindowTokens / options.swa_block_size())
      .enable_kvcache_store(true)
      .prefetch_batch_size(64);
  FakePrefetchEngine engine(/*worker_count=*/1,
                            /*timeout_ms=*/-1,
                            /*batch_size=*/64);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  std::shared_ptr<Request> request = make_test_request(
      std::vector<int32_t>(kWindowTokens + unit_tokens + 1, 23));
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  const size_t window_blocks = kWindowTokens / options.swa_block_size();
  const size_t unit_blocks = unit_tokens / options.swa_block_size();
  ASSERT_EQ(engine.request().units.size(),
            (kWindowTokens + unit_tokens) / unit_tokens);
  const PrefetchUnit& last_unit = engine.request().units.back();
  EXPECT_EQ(last_unit.non_gated_blocks.size(), window_blocks);
  const Slice<Block> swa_blocks =
      sequence->host_kv_state().blocks(BlockType::SWA);
  const size_t last_unit_end =
      kWindowTokens / options.swa_block_size() + unit_blocks;
  for (size_t index = last_unit_end - window_blocks; index < last_unit_end;
       ++index) {
    EXPECT_TRUE(swa_blocks[index].is_valid()) << index;
  }
  engine.finish_worker_with_logical_hits(
      /*worker_index=*/0,
      std::vector<uint8_t>(engine.request().batch_transfer_count(
                               0, engine.request().units.size()),
                           1));
  EXPECT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(),
            kWindowTokens + unit_tokens);
  EXPECT_EQ(
      count_valid_blocks(sequence->host_kv_state().blocks(BlockType::SWA)),
      window_blocks);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest, Dsv4PrefetchClampsWindowToCheckpoint) {
  const size_t unit_tokens = kDsv4FixedBlockSize;
  auto options = make_typed_cache_options();
  options.sliding_window_size(unit_tokens * 2)
      .swa_blocks_per_seq(unit_tokens * 2 / options.swa_block_size())
      .enable_kvcache_store(true)
      .prefetch_batch_size(2);
  FakePrefetchEngine engine(1, -1, 2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  auto request =
      make_test_request(std::vector<int32_t>(unit_tokens * 2 + 1, 23));
  Sequence* sequence = request->sequences().front().get();
  const auto cleanup = folly::makeGuard([&pool, &engine, sequence] {
    engine.drain_workers();
    pool.deallocate(sequence);
  });
  bool completed = false;
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_EQ(engine.request().units.size(), 2u);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::SWA),
            unit_tokens * 2 / options.swa_block_size());
  EXPECT_EQ(
      count_valid_blocks(sequence->host_kv_state().blocks(BlockType::SWA)),
      unit_tokens * 2 / options.swa_block_size());
  engine.finish_worker_with_logical_hits(
      0, std::vector<uint8_t>(engine.request().batch_transfer_count(0, 2), 1));
  EXPECT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(), unit_tokens * 2);
}

TEST(PrefetchResultTest, UsesMinimumContiguousUnitPrefixAcrossWorkers) {
  size_t common_hit_units = std::numeric_limits<size_t>::max();
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 5; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo info(/*src_block_id=*/-1,
                           /*dst_block_id=*/static_cast<int32_t>(unit_index));
    info.transfer_type = TransferType::G2H;
    info.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(info);
    request.units.emplace_back(std::move(unit));
  }
  PrefetchResult result(
      /*worker_count=*/2,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&common_hit_units](PrefetchSummary summary) {
        common_hit_units = 0;
        while (common_hit_units < summary.gated_hits.size() &&
               summary.gated_hits[common_hit_units] == summary.rank_count) {
          ++common_hit_units;
        }
      });

  std::optional<PrefetchControl> control =
      result.record_batch_result(/*worker_index=*/0, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);
  control =
      result.record_batch_result(/*worker_index=*/1, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);

  control =
      result.record_batch_result(/*worker_index=*/0, /*prefix_hit_units=*/1);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  EXPECT_EQ(common_hit_units, std::numeric_limits<size_t>::max());

  control =
      result.record_batch_result(/*worker_index=*/1, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  // Worker 0 has already observed the shared gate miss, so worker 1 closes
  // after its current batch even though that batch itself is a hit.
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/1, /*worker_ok=*/true);

  EXPECT_EQ(common_hit_units, 3u);
}

TEST(PrefetchResultTest, WorkersAdvanceIndependently) {
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 4; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo info(/*src_block_id=*/-1,
                           /*dst_block_id=*/static_cast<int32_t>(unit_index));
    info.transfer_type = TransferType::G2H;
    info.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(info);
    request.units.emplace_back(std::move(unit));
  }
  PrefetchResult result(
      /*worker_count=*/2,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [](PrefetchSummary /*summary*/) {});
  const std::vector<uint8_t> hits = {1, 1};

  std::optional<PrefetchControl> control =
      result.record_batch_result(/*worker_index=*/0, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);
  control = result.record_batch_result(/*worker_index=*/0, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);

  control = result.record_batch_result(/*worker_index=*/1, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);
  control = result.record_batch_result(/*worker_index=*/1, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  result.mark_worker_ended(/*worker_index=*/1, /*worker_ok=*/true);
}

TEST(PrefetchResultTest, OptionalMissDoesNotStopGatePrefix) {
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo gated(/*src_block_id=*/-1,
                            /*dst_block_id=*/static_cast<int32_t>(unit_index));
    gated.transfer_type = TransferType::G2H;
    gated.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(gated);
    BlockTransferInfo non_gated(
        /*src_block_id=*/-1,
        /*dst_block_id=*/static_cast<int32_t>(unit_index + 2));
    non_gated.transfer_type = TransferType::G2H;
    non_gated.block_type = BlockType::LINEAR;
    unit.non_gated_blocks.emplace_back(non_gated);
    unit.has_non_gated = true;
    request.units.emplace_back(std::move(unit));
  }

  PrefetchSummary summary;
  PrefetchResult result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  const std::optional<PrefetchControl> control = result.record_batch_result(
      /*worker_index=*/0,
      /*gated_hits=*/std::vector<uint8_t>{1, 1},
      /*non_gated_hits=*/std::vector<uint8_t>{0, 0});
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);

  EXPECT_EQ(summary.rank_count, 1u);
  EXPECT_EQ(summary.gated_hits, (std::vector<uint8_t>{1, 1}));
  EXPECT_EQ(summary.non_gated_hits, (std::vector<uint8_t>{0, 0}));
}

TEST(PrefetchResultTest, LogicalHitsAreSummedAcrossWorkers) {
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo gated(/*src_block_id=*/-1,
                            /*dst_block_id=*/static_cast<int32_t>(unit_index));
    gated.transfer_type = TransferType::G2H;
    gated.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(gated);
    BlockTransferInfo non_gated(
        /*src_block_id=*/-1,
        /*dst_block_id=*/static_cast<int32_t>(unit_index + 2));
    non_gated.transfer_type = TransferType::G2H;
    non_gated.block_type = BlockType::LINEAR;
    unit.non_gated_blocks.emplace_back(non_gated);
    unit.has_non_gated = true;
    request.units.emplace_back(std::move(unit));
  }

  PrefetchSummary summary;
  PrefetchResult result(
      /*worker_count=*/2,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  const std::vector<uint8_t> worker_zero_gated = {1, 1};
  const std::vector<uint8_t> worker_zero_non_gated = {0, 1};
  const std::vector<uint8_t> worker_one_gated = {1, 1};
  const std::vector<uint8_t> worker_one_non_gated = {1, 0};
  ASSERT_TRUE(
      result.record_batch_result(0, worker_zero_gated, worker_zero_non_gated)
          .has_value());
  result.mark_worker_ended(0, /*worker_ok=*/true);
  ASSERT_TRUE(
      result.record_batch_result(1, worker_one_gated, worker_one_non_gated)
          .has_value());
  result.mark_worker_ended(1, /*worker_ok=*/true);

  EXPECT_EQ(summary.rank_count, 2u);
  EXPECT_EQ(summary.gated_hits, (std::vector<uint8_t>{2, 2}));
  EXPECT_EQ(summary.non_gated_hits, (std::vector<uint8_t>{1, 1}));
}

TEST(PrefetchResultTest, WorkerFailureMarksSummaryUnsuccessful) {
  StoragePrefetchRequest request;
  BlockTransferInfo info(/*src_block_id=*/-1, /*dst_block_id=*/0);
  info.transfer_type = TransferType::G2H;
  info.block_type = BlockType::KV;
  PrefetchUnit unit;
  unit.gated_blocks.emplace_back(info);
  request.units.emplace_back(std::move(unit));

  PrefetchSummary summary;
  PrefetchResult result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/1,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  ASSERT_TRUE(result
                  .record_batch_result(
                      0, std::vector<uint8_t>{1}, std::vector<uint8_t>{1})
                  .has_value());
  result.mark_worker_ended(0, /*worker_ok=*/false);

  EXPECT_TRUE(summary.gated_hits.empty());
}

TEST(StoragePrefetchRequestTest, SwaMissPreservesCompressedGatePrefix) {
  const std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash = {};
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
    PrefetchUnit unit;
    for (BlockType block_type : {BlockType::SWA, BlockType::KV}) {
      BlockTransferInfo info(-1,
                             static_cast<int32_t>(unit_index),
                             hash.data(),
                             TransferType::G2H,
                             block_type);
      if (block_type == BlockType::SWA) {
        unit.non_gated_blocks.emplace_back(std::move(info));
        unit.has_non_gated = true;
      } else {
        unit.gated_blocks.emplace_back(std::move(info));
      }
    }
    request.units.emplace_back(std::move(unit));
  }
  ASSERT_TRUE(request.valid());
  PrefetchSummary summary;
  PrefetchResult result(
      1,
      request,
      2,
      -1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  ASSERT_TRUE(result
                  .record_batch_result(
                      0, std::vector<uint8_t>{1, 1}, std::vector<uint8_t>{0, 0})
                  .has_value());
  result.mark_worker_ended(0, true);
  EXPECT_EQ(summary.rank_count, 1u);
  EXPECT_EQ(summary.gated_hits, (std::vector<uint8_t>{1, 1}));
  EXPECT_EQ(summary.non_gated_hits, (std::vector<uint8_t>{0, 0}));
}

TEST(PrefetchResultTest, StopsAfterCancellationOrTimeout) {
  bool cancelled = false;
  size_t common_hit_units = 0;
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 4; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo info(/*src_block_id=*/-1,
                           /*dst_block_id=*/static_cast<int32_t>(unit_index));
    info.transfer_type = TransferType::G2H;
    info.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(info);
    request.units.emplace_back(std::move(unit));
  }
  PrefetchResult cancelled_result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [&cancelled] { return cancelled; },
      [&common_hit_units](PrefetchSummary summary) {
        common_hit_units = 0;
        while (common_hit_units < summary.gated_hits.size() &&
               summary.gated_hits[common_hit_units] == summary.rank_count) {
          ++common_hit_units;
        }
      });
  cancelled = true;
  std::optional<PrefetchControl> control = cancelled_result.record_batch_result(
      /*worker_index=*/0, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  EXPECT_FALSE(cancelled_result
                   .record_batch_result(/*worker_index=*/0,
                                        /*prefix_hit_units=*/2)
                   .has_value());
  cancelled_result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  EXPECT_EQ(common_hit_units, 2u);

  PrefetchResult timed_out_result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/1,
      [] { return false; },
      [](PrefetchSummary /*summary*/) {});
  EXPECT_EQ(timed_out_result.stream_idle_timeout_ms(), 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  control = timed_out_result.record_batch_result(
      /*worker_index=*/0, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
}

TEST(HierarchyBlockManagerPoolTest,
     StoragePrefetchTimeoutWaitsForWorkersToEnd) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/2, /*timeout_ms=*/1);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  std::vector<int32_t> tokens(1025, 83);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_EQ(engine.request().units.size(), 8u);
  EXPECT_EQ(engine.request().batch_count(2), 4u);
  EXPECT_TRUE(sequence->host_kv_state().has_any_blocks());
  // Host admission marks the state matched when it reserves the prefix.
  // Publish still waits until every worker ends.
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::optional<PrefetchControl> control =
      engine.result()->record_batch_result(/*worker_index=*/0,
                                           /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  engine.result()->mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  EXPECT_FALSE(completed);
  control = engine.result()->record_batch_result(/*worker_index=*/1,
                                                 /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  engine.result()->mark_worker_ended(/*worker_index=*/1, /*worker_ok=*/true);

  EXPECT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 256u);
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     StoragePrefetchPublishesOnlyAfterAllTpWorkersEnd) {
  const double fetched_before =
      COUNTER_VALUE(store_prefetch_master_fetched_tokens_total);
  const double sessions_before =
      MULTI_COUNTER_store_prefetch_master_sessions_total
          .get_stats(/*labels=*/{"complete"})
          ->get_value();
  const int64_t samples_before =
      HISTOGRAM_store_prefetch_master_latency_microseconds.count();
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  std::vector<int32_t> tokens(1025, 73);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_EQ(engine.dp_rank(), 0u);
  ASSERT_EQ(engine.request().units.size(), 8u);
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_TRUE(sequence->host_kv_state().has_any_blocks());
  // The matched flag is set at reservation. The callback and the published
  // block count still wait for every TP worker.
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());

  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/8);
  EXPECT_FALSE(completed);
  EXPECT_EQ(HISTOGRAM_store_prefetch_master_latency_microseconds.count(),
            samples_before);
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  engine.finish_worker(/*worker_index=*/1, /*hit_units=*/8);

  EXPECT_TRUE(completed);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_TRUE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->kv_state().prefix_cache_matched());
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  pool.allocate_shared(sequence);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::KV), 8u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 1024u);
  EXPECT_EQ(COUNTER_VALUE(store_prefetch_master_fetched_tokens_total),
            fetched_before + 1024);
  EXPECT_EQ(MULTI_COUNTER_store_prefetch_master_sessions_total
                .get_stats(/*labels=*/{"complete"})
                ->get_value(),
            sessions_before + 1);
  EXPECT_EQ(HISTOGRAM_store_prefetch_master_latency_microseconds.count(),
            samples_before + 1);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     TypedStoragePrefetchRetainsLongestPrefixWithinHostCapacity) {
  constexpr size_t kPromptTokens = 2 * kUnitTokens + 1;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true)
      .prefetch_batch_size(2)
      .host_num_blocks_by_type({{BlockType::SWA, 129}, {BlockType::KV, 2}});
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  std::vector<int32_t> tokens(kPromptTokens, 79);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_EQ(engine.request().units.size(), 1u);
  EXPECT_EQ(engine.request().batch_transfer_count(0, 2), 2u);

  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/1);
  EXPECT_FALSE(completed);
  engine.finish_worker(/*worker_index=*/1, /*hit_units=*/1);
  ASSERT_TRUE(completed);

  pool.allocate_shared(sequence);
  const Slice<Block> swa_blocks =
      sequence->host_kv_state().blocks(BlockType::SWA);
  ASSERT_EQ(swa_blocks.size(), kUnitTokens / kBaseBlockSize);
  for (size_t i = 0; i + 1 < swa_blocks.size(); ++i) {
    EXPECT_FALSE(swa_blocks[i].is_valid());
  }
  EXPECT_TRUE(swa_blocks.back().is_valid());
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::KV), 1u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), kUnitTokens);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     TypedStoragePrefetchStopsAtFirstIncompleteUnit) {
  constexpr size_t kPromptTokens = 2 * kUnitTokens + 1;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  std::vector<int32_t> tokens(kPromptTokens, 79);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_EQ(engine.request().units.size(), 2u);
  EXPECT_EQ(engine.request().batch_transfer_count(0, 2), 4u);

  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/0);
  EXPECT_FALSE(completed);
  engine.finish_worker(/*worker_index=*/1, /*hit_units=*/2);

  EXPECT_TRUE(completed);
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 0u);
}

TEST(HierarchyBlockManagerPoolTest,
     LinearStorePrefetchDropsKvWhenOptionalCheckpointMisses) {
  constexpr int32_t kChunkStride = 2048;
  constexpr int32_t kKvBlockSize = 128;
  constexpr size_t kPromptTokens = 4228;
  constexpr size_t kCachedTokens = 4224;

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  auto restore_chunk_stride = folly::makeGuard([original_chunk_stride]() {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride;
  });
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kChunkStride;

  BlockManagerPool::Options options = make_flat_kv_options();
  options.block_size(kKvBlockSize)
      .enable_linear_state(true)
      .linear_state_num_slots(80)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 80}})
      .enable_kvcache_store(true)
      .prefetch_batch_size(8)
      .prefetch_valid_blocks_per_batch(8);
  FakePrefetchEngine engine(/*worker_count=*/1,
                            /*timeout_ms=*/-1,
                            /*batch_size=*/8);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto drain_callbacks = folly::makeGuard([&engine]() {
    engine.drain_workers();
    engine.drain_offloads();
  });
  std::shared_ptr<Request> request =
      make_test_request(std::vector<int32_t>(kPromptTokens, 17));
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });

  ASSERT_NE(engine.result(), nullptr);
  bool prefetch_finished = false;
  auto finish_prefetch = folly::makeGuard([&engine, &prefetch_finished]() {
    if (prefetch_finished || engine.result() == nullptr ||
        engine.request().units.empty()) {
      return;
    }
    size_t transfer_count = 0;
    for (const PrefetchUnit& unit : engine.request().units) {
      transfer_count += unit.gated_blocks.size();
      transfer_count += unit.non_gated_blocks.size();
    }
    std::vector<uint8_t> hits(transfer_count, 1);
    size_t cursor = 0;
    for (const PrefetchUnit& unit : engine.request().units) {
      cursor += unit.gated_blocks.size();
      for (size_t i = 0; i < unit.non_gated_blocks.size(); ++i) {
        hits[cursor++] = 0;
      }
    }
    engine.finish_worker_with_logical_hits(/*worker_index=*/0, hits);
  });

  const size_t expected_units = kCachedTokens / kKvBlockSize;
  ASSERT_EQ(engine.request().units.size(), expected_units);
  for (const PrefetchUnit& unit : engine.request().units) {
    EXPECT_EQ(unit.gated_blocks.size(), 1u);
    EXPECT_EQ(unit.non_gated_blocks.size(), 1u);
    EXPECT_TRUE(unit.has_non_gated);
  }

  size_t transfer_count = 0;
  std::vector<uint8_t> hits;
  hits.reserve(expected_units * 2);
  for (const PrefetchUnit& unit : engine.request().units) {
    transfer_count += unit.gated_blocks.size();
    transfer_count += unit.non_gated_blocks.size();
    hits.push_back(1);
    hits.insert(hits.end(), unit.non_gated_blocks.size(), 0);
  }
  ASSERT_EQ(hits.size(), transfer_count);
  engine.finish_worker_with_logical_hits(/*worker_index=*/0, hits);
  prefetch_finished = true;

  ASSERT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence->host_kv_state().shared_blocks_num(BlockType::KV), 0u);
  pool.deallocate(sequence);
}

// Host hit length is the cursor after the prefetch Host probe and before the
// Store GET. A pure Store admission logs host_hit_tokens=0.
TEST(HierarchyBlockManagerPoolTest, PrefetchLogsHostHitBeforeStoreGet) {
  constexpr int32_t kKvBlockSize = 128;
  constexpr size_t kPromptTokens = 4228;
  constexpr size_t kCachedTokens = 4224;

  BlockManagerPool::Options options = make_flat_kv_options();
  options.block_size(kKvBlockSize)
      .enable_kvcache_store(true)
      .prefetch_batch_size(64);
  FakePrefetchEngine engine(/*worker_count=*/1,
                            /*timeout_ms=*/-1,
                            /*batch_size=*/64);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::shared_ptr<Request> request =
      make_test_request(std::vector<int32_t>(kPromptTokens, 19));
  Sequence* sequence = request->sequences().front().get();
  google::SetVLOGLevel("hierarchy_block_manager_pool", 1);
  testing::internal::CaptureStderr();
  bool completed = false;
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  const std::string logged = testing::internal::GetCapturedStderr();
  const std::string expected =
      "[HostCache][PrefetchHit] sequence_id=-1 host_hit_tokens=0";
  EXPECT_NE(logged.find(expected), std::string::npos) << logged;
  ASSERT_NE(engine.result(), nullptr);
  const size_t expected_units = kCachedTokens / kKvBlockSize;
  ASSERT_EQ(engine.request().units.size(), expected_units);
  engine.finish_worker(/*worker_index=*/0, expected_units);
  ASSERT_TRUE(completed);
  EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(), kCachedTokens);
  pool.deallocate(sequence);
}

namespace {

void seed_host_prefix(BlockManager* leaf, const std::vector<int32_t>& tokens) {
  ASSERT_NE(leaf, nullptr);
  const size_t block_count = tokens.size() / leaf->block_size();
  std::vector<Block> blocks = leaf->allocate(block_count);
  ASSERT_EQ(blocks.size(), block_count);
  leaf->cache(tokens, blocks);
  blocks.clear();
}

struct LinearTailPrefetchCase {
  int32_t chunk_stride;
  int32_t kv_block_size;
  size_t prompt_tokens;
  bool seed_regular_checkpoints;
};

class LinearTailHostPrefetchTest
    : public ::testing::TestWithParam<LinearTailPrefetchCase> {
 protected:
  void SetUp() override {
    int32_t& chunk_stride =
        SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
    original_chunk_stride_ = chunk_stride;
    chunk_stride = GetParam().chunk_stride;
  }

  void TearDown() override {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride_;
  }

 private:
  int32_t original_chunk_stride_ = 0;
};

std::vector<Block> exhaust_free_blocks(BlockManager* leaf, size_t keep_free) {
  const size_t free_blocks = leaf->num_free_blocks();
  CHECK_GE(free_blocks, keep_free);
  return leaf->allocate(free_blocks - keep_free);
}

}  // namespace

TEST(HierarchyBlockManagerPoolTest,
     LinearStorePrefetchDoesNotRepeatMatchedPartialUnit) {
  constexpr int32_t kChunkStride = 2048;
  constexpr int32_t kKvBlockSize = 128;
  constexpr size_t kPromptTokens = 10195;
  constexpr size_t kMatchedTailTokens = 9984;
  constexpr size_t kTargetTailTokens = 10112;

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  auto restore_chunk_stride = folly::makeGuard([original_chunk_stride]() {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride;
  });
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kChunkStride;

  BlockManagerPool::Options options = make_flat_kv_options();
  options.block_size(kKvBlockSize)
      .enable_linear_state(true)
      .linear_state_num_slots(80)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 80}})
      .enable_kvcache_store(true)
      .prefetch_batch_size(2)
      .prefetch_valid_blocks_per_batch(1);
  FakePrefetchEngine engine(/*worker_count=*/1,
                            /*timeout_ms=*/-1,
                            /*batch_size=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  const auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  const std::vector<int32_t> cached_tokens(kMatchedTailTokens, 17);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), cached_tokens);

  Sequence hash_sequence = make_test_sequence(
      /*index=*/0, std::vector<int32_t>(kPromptTokens, 17));
  hash_sequence.update_block_hashes(kKvBlockSize, BlockHasherType::TEXT);
  std::vector<Block> tail_blocks = host.at(BlockType::LINEAR).leaf->allocate(1);
  ASSERT_EQ(tail_blocks.size(), 1u);
  tail_blocks.front().set_hash_value(
      hash_sequence
          .block_hashes(kKvBlockSize)[kMatchedTailTokens / kKvBlockSize - 1]
          .data);
  host.at(BlockType::LINEAR).leaf->cache(tail_blocks);
  tail_blocks.clear();

  std::shared_ptr<Request> request =
      make_test_request(std::vector<int32_t>(kPromptTokens, 17));
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;
  const auto cleanup = folly::makeGuard([&pool, &engine, sequence] {
    // Fatal assertions must not leave the pool holding unfinished callbacks.
    // Unfinished workers are failed, never reported as successful transfers.
    engine.drain_workers();
    pool.deallocate(sequence);
    HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
  });
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });

  ASSERT_NE(engine.result(), nullptr);
  EXPECT_EQ(engine.result()->batch_wire_size(), 1u);
  ASSERT_EQ(engine.request().units.size(), 1u);
  EXPECT_EQ(engine.request().units.front().gated_blocks.size() * kKvBlockSize,
            kTargetTailTokens - kMatchedTailTokens);
  ASSERT_EQ(engine.request().units.front().non_gated_blocks.size(), 1u);

  const size_t transfer_count =
      engine.request().batch_transfer_count(0, engine.request().units.size());
  engine.finish_worker_with_logical_hits(
      /*worker_index=*/0, std::vector<uint8_t>(transfer_count, 1));

  ASSERT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(), kTargetTailTokens);
  pool.deallocate(sequence);
}

TEST_P(LinearTailHostPrefetchTest,
       RetainsTailBoundaryWithoutPublishingRegularCheckpoint) {
  const LinearTailPrefetchCase& test_case = GetParam();
  const size_t kv_block_size = static_cast<size_t>(test_case.kv_block_size);
  const size_t chunk_stride = static_cast<size_t>(test_case.chunk_stride);
  const size_t tail_tokens =
      test_case.prompt_tokens / kv_block_size * kv_block_size;
  const size_t checkpoint_count = tail_tokens / chunk_stride;
  ASSERT_GT(tail_tokens, chunk_stride);
  ASSERT_NE(tail_tokens % chunk_stride, 0u);

  BlockManagerPool::Options options = make_flat_kv_options();
  options.block_size(test_case.kv_block_size)
      .enable_linear_state(true)
      .linear_state_num_slots(80)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 80}})
      .enable_kvcache_store(true);
  FakePrefetchEngine engine(/*worker_count=*/1);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  const auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  BlockManager* linear_leaf = host.at(BlockType::LINEAR).leaf.get();
  const std::vector<int32_t> tokens(test_case.prompt_tokens, 7);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence hash_sequence = make_test_sequence(/*index=*/0, tokens);
  hash_sequence.update_block_hashes(test_case.kv_block_size,
                                    BlockHasherType::TEXT);
  std::vector<int32_t> regular_block_ids;
  regular_block_ids.reserve(checkpoint_count);
  if (test_case.seed_regular_checkpoints) {
    // The cache API keeps the final slot live and publishes preceding slots.
    std::vector<Block> regular_blocks =
        linear_leaf->allocate(checkpoint_count + 1);
    ASSERT_EQ(regular_blocks.size(), checkpoint_count + 1);
    for (size_t checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
      regular_block_ids.emplace_back(regular_blocks[checkpoint].id());
    }
    linear_leaf->cache(tokens,
                       regular_blocks,
                       /*existed_shared_blocks_num=*/0,
                       hash_sequence.mm_data(),
                       hash_sequence.block_hashes(test_case.kv_block_size));
    regular_blocks.clear();
  }

  std::vector<Block> tail_blocks = linear_leaf->allocate(1);
  ASSERT_EQ(tail_blocks.size(), 1u);
  const int32_t tail_block_id = tail_blocks.front().id();
  tail_blocks.front().set_hash_value(
      hash_sequence
          .block_hashes(
              test_case.kv_block_size)[tail_tokens / kv_block_size - 1]
          .data);
  linear_leaf->cache(tail_blocks);
  tail_blocks.clear();

  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;
  auto cleanup = folly::makeGuard([&pool, &engine, sequence] {
    // Fatal assertions must not leave the pool holding unfinished callbacks.
    // Unfinished workers are failed, never reported as successful transfers.
    engine.drain_workers();
    pool.deallocate(sequence);
    HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
  });
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });

  ASSERT_TRUE(completed);
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(), tail_tokens);
  const Slice<Block> retained =
      sequence->host_kv_state().blocks(BlockType::LINEAR);
  // Prefetch trim keeps one restore source, the deepest checkpoint.
  ASSERT_EQ(retained.size(), 1u);
  ASSERT_TRUE(retained.front().is_valid());
  EXPECT_EQ(retained.front().id(), tail_block_id);
  EXPECT_TRUE(XXH3Key(retained.front().get_immutable_hash_value()) ==
              hash_sequence.block_hashes(
                  test_case.kv_block_size)[tail_tokens / kv_block_size - 1]);
  pool.deallocate(sequence);
  cleanup.dismiss();

  // Each shorter prompt ends one token beyond a regular boundary, so a hit
  // must restore that exact checkpoint rather than the deeper tail state.
  CompositeBlockManager* host_manager =
      HierarchyPoolTestPeer::host_block_managers(pool).front().get();
  for (size_t checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
    const size_t boundary = (checkpoint + 1) * chunk_stride;
    SCOPED_TRACE(boundary);
    const std::vector<int32_t> shorter_tokens(tokens.begin(),
                                              tokens.begin() + boundary + 1);
    Sequence shorter = make_test_sequence(/*index=*/1, shorter_tokens);
    const auto cleanup_shorter = folly::makeGuard([&pool, &shorter] {
      pool.deallocate(&shorter);
      HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
    });
    host_manager->allocate_shared_for_sequence(&shorter,
                                               shorter.host_kv_state());
    const Slice<Block> source =
        shorter.host_kv_state().blocks(BlockType::LINEAR);
    if (test_case.seed_regular_checkpoints) {
      // A seeded regular checkpoint restores one source. The matched length
      // is a positive multiple of the chunk stride and stops short of the
      // deeper tail that belongs to the long prompt.
      const size_t matched = shorter.host_kv_state().kv_cache_tokens_num();
      EXPECT_GT(matched, 0u);
      EXPECT_EQ(matched % kv_block_size, 0u);
      EXPECT_LE(matched, tail_tokens);
      ASSERT_EQ(source.size(), 1u);
      EXPECT_TRUE(source.front().is_valid());
      EXPECT_NE(source.front().id(), tail_block_id);
    } else {
      EXPECT_EQ(shorter.host_kv_state().kv_cache_tokens_num(), 0u);
      EXPECT_TRUE(source.empty());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultiChunkTail,
    LinearTailHostPrefetchTest,
    ::testing::Values(LinearTailPrefetchCase{8, 4, 13, false},
                      LinearTailPrefetchCase{8, 4, 13, true},
                      LinearTailPrefetchCase{8, 4, 21, false},
                      LinearTailPrefetchCase{8, 4, 21, true},
                      LinearTailPrefetchCase{8192, 128, 9000, false},
                      LinearTailPrefetchCase{8192, 128, 9000, true}));

TEST(HierarchyBlockManagerPoolTest, PrefillOffloadsOnlyTailLinearCheckpoint) {
  constexpr int32_t kChunkSize = 4096;
  constexpr uint32_t kBlockSize = 128;
  constexpr size_t kTailCheckpoint = 2 * kChunkSize + 1024;
  constexpr size_t kPromptTokens = kTailCheckpoint + 1;
  int32_t& chunk_size =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const auto restore_chunk_size = folly::makeGuard(
      [&chunk_size, original = chunk_size] { chunk_size = original; });
  chunk_size = kChunkSize;

  BlockManagerPool::Options options = make_flat_kv_options();
  options.block_size(kBlockSize)
      .num_blocks(128)
      .enable_linear_state(true)
      .linear_state_num_slots(16)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 16}});
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  Sequence sequence =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(kPromptTokens, 7));
  const auto cleanup = folly::makeGuard([&pool, &sequence] {
    pool.deallocate(&sequence);
    HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
  });
  ASSERT_TRUE(pool.allocate(&sequence, kPromptTokens));

  size_t previous_kv_blocks = 0;
  for (const size_t cached_tokens : {size_t{kChunkSize},
                                     size_t{2 * kChunkSize},
                                     kTailCheckpoint - 1,
                                     kTailCheckpoint}) {
    SCOPED_TRACE(cached_tokens);
    sequence.kv_state().set_kv_cache_tokens_num(cached_tokens);
    ASSERT_TRUE(pool.allocate(&sequence, kPromptTokens));
    const auto stored = HierarchyPoolTestPeer::complete_offload_pairs(pool);

    const auto hashes = sequence.block_hashes(kBlockSize);
    const size_t kv_blocks = cached_tokens / kBlockSize;
    const bool offloads_linear = cached_tokens == kTailCheckpoint;
    EXPECT_EQ(stored.size(), kv_blocks - previous_kv_blocks + offloads_linear);
    for (size_t i = previous_kv_blocks; i < kv_blocks; ++i) {
      const std::string key(reinterpret_cast<const char*>(hashes[i].data),
                            XXH3_128BITS_HASH_VALUE_LEN);
      EXPECT_EQ(stored.count({BlockType::KV, key}), 1u);
      EXPECT_EQ(stored.count({BlockType::LINEAR, key}),
                offloads_linear && i == kv_blocks - 1 ? 1u : 0u);
    }
    previous_kv_blocks = kv_blocks;
  }
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeLinearOffloadKeepsLiveSlotWithInflightReferences) {
  constexpr int32_t kKvBlockSize = 4;
  constexpr size_t kAcceptedLength = 1;
  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const auto restore_chunk_stride = folly::makeGuard([original_chunk_stride] {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride;
  });
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kKvBlockSize;

  for (const bool overlap : {false, true}) {
    SCOPED_TRACE(overlap);
    BlockManagerPool::Options options = make_flat_kv_options();
    options.enable_linear_state(true)
        .block_size(kKvBlockSize)
        .linear_state_num_slots(64)
        .host_num_blocks_by_type(
            {{BlockType::KV, 128}, {BlockType::LINEAR, 64}})
        .enable_prefix_cache(false)
        .enable_disagg_pd(true)
        .instance_is_decode(true)
        .enable_kvcache_store(true)
        .num_embedding_blocks(64)
        .num_speculative_tokens(kAcceptedLength)
        .enable_schedule_overlap(overlap);
    HierarchyBlockManagerPool pool(options,
                                   /*engine=*/nullptr,
                                   /*dp_size=*/1);
    CompositeBlockManager* device =
        HierarchyPoolTestPeer::device_composite(pool);
    const auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
    std::map<BlockType, size_t> device_free_before;
    std::map<BlockType, size_t> host_free_before;
    for (const BlockType type : {BlockType::KV, BlockType::LINEAR}) {
      device_free_before.emplace(
          type, device->leaf_entries().at(type).leaf->num_free_blocks());
      host_free_before.emplace(type,
                               host_leaves.at(type).leaf->num_free_blocks());
    }

    Sequence sequence = make_test_sequence(
        /*index=*/0, std::vector<int32_t>(9, 103), overlap);
    auto cleanup = folly::makeGuard([&pool, &sequence] {
      pool.deallocate(&sequence);
      HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
    });
    ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/9));
    sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
    ASSERT_EQ(sequence.stage(), SequenceStage::DECODE);
    auto append_generated_token = [&sequence, overlap](int32_t token_id) {
      if (overlap) {
        sequence.append_token(/*fake placeholder=*/-1);
        sequence.update_last_step_token(Token(token_id), /*token_offset=*/0);
      } else {
        sequence.append_token(token_id);
      }
    };
    append_generated_token(104);
    append_generated_token(105);
    append_generated_token(106);
    ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), 11u);
    sequence.set_accepted_length(0);
    ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
    // 11 cached tokens complete two ordinary KV chunks (block_size 4), which
    // collect_offload_pairs legitimately queues for host offload regardless
    // of LINEAR state. The absence of a LINEAR pair is what this step is
    // actually verifying: no checkpoint has completed yet.
    ASSERT_EQ(HierarchyPoolTestPeer::take_linear_offload_pair(pool), nullptr);

    const Slice<Block> hbm_before =
        sequence.kv_state().blocks(BlockType::LINEAR);
    const Slice<Block> host_before =
        sequence.host_kv_state().blocks(BlockType::LINEAR);
    ASSERT_EQ(hbm_before.size(), 2u);
    ASSERT_EQ(host_before.size(), 2u);
    const int32_t hbm_checkpoint_id = hbm_before.back().id();
    const int32_t hbm_old_front_id = hbm_before.front().id();
    const int32_t host_checkpoint_id = host_before.back().id();
    const int32_t host_old_front_id = host_before.front().id();
    // allocate_decode keeps the checkpoint at front under overlap (the
    // pipeline lag already landed it there) and at back otherwise; match
    // whichever this sequence actually preserved.
    const int32_t expected_hbm_checkpoint_id =
        overlap ? hbm_old_front_id : hbm_checkpoint_id;
    const int32_t expected_host_checkpoint_id =
        overlap ? host_old_front_id : host_checkpoint_id;

    std::vector<Block> inflight_blocks;
    inflight_blocks.reserve(hbm_before.size());
    inflight_blocks.emplace_back(hbm_before.front());
    inflight_blocks.emplace_back(hbm_before.back());
    ASSERT_GT(hbm_before.front().ref_count(), 1u);
    ASSERT_GT(hbm_before.back().ref_count(), 1u);
    ASSERT_EQ(host_before.front().ref_count(), 1u);
    ASSERT_EQ(host_before.back().ref_count(), 1u);

    append_generated_token(107);
    ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), 12u);
    sequence.set_accepted_length(kAcceptedLength);
    ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));

    const Slice<Block> hbm_after =
        sequence.kv_state().blocks(BlockType::LINEAR);
    const Slice<Block> host_after =
        sequence.host_kv_state().blocks(BlockType::LINEAR);
    ASSERT_EQ(hbm_after.size(), 1u);
    ASSERT_EQ(host_after.size(), 1u);
    if (overlap) {
      EXPECT_EQ(hbm_after.back().id(), hbm_checkpoint_id);
    } else {
      EXPECT_NE(hbm_after.back().id(), hbm_old_front_id);
      EXPECT_NE(hbm_after.back().id(), hbm_checkpoint_id);
    }
    EXPECT_EQ(host_after.back().id(),
              overlap ? host_checkpoint_id : host_old_front_id);
    EXPECT_EQ(hbm_after.back().ref_count(), overlap ? 2u : 1u);
    EXPECT_EQ(host_after.back().ref_count(), 1u);

    std::shared_ptr<OffloadBlockPair> pair =
        HierarchyPoolTestPeer::take_linear_offload_pair(pool);
    ASSERT_NE(pair, nullptr);
    EXPECT_EQ(pair->block_type, BlockType::LINEAR);
    EXPECT_EQ(pair->src.id(), expected_hbm_checkpoint_id);
    EXPECT_NE(pair->src.id(), hbm_after.back().id());
    EXPECT_EQ(pair->dst.id(), expected_host_checkpoint_id);
    EXPECT_NE(pair->dst.id(), host_after.back().id());
    EXPECT_EQ(pair->checkpoint_row, kAcceptedLength);
    EXPECT_EQ(pair->src.ref_count(), 2u);
    EXPECT_EQ(pair->dst.ref_count(), 1u);
    HierarchyPoolTestPeer::enqueue_offload_pair(pool, std::move(pair));

    // Release the sequence, in-flight references, and queue; KV and LINEAR
    // blocks in both tiers should be reclaimed.
    pool.deallocate(&sequence);
    inflight_blocks.clear();
    HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
    cleanup.dismiss();
    device->reset_prefix_cache();
    HierarchyPoolTestPeer::host_block_managers(pool)
        .front()
        ->reset_prefix_cache();
    EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);
    EXPECT_FALSE(pool.has_pending_async_block_release());
    for (const BlockType type : {BlockType::KV, BlockType::LINEAR}) {
      EXPECT_EQ(device->leaf_entries().at(type).leaf->num_free_blocks(),
                device_free_before.at(type));
      EXPECT_EQ(host_leaves.at(type).leaf->num_free_blocks(),
                host_free_before.at(type));
    }
  }
}

// Decode LINEAR window around a checkpoint boundary.
//
// Prefix cache is off here, so the HBM stride is max(KV block, chunk). This
// test pins the chunk to the KV block. Host LINEAR keeps prefix cache and
// uses the KV block, so both tiers share the same grid.
// Decode retains the latest output as the read source and allocates a write
// slot when needed. The completed span is [cached - accepted - 1, cached). At a
// boundary, collect removes the checkpoint front and retains the live back in
// both tiers. An exclusive single slot advances in place below the next
// boundary.
TEST(HierarchyBlockManagerPoolTest, DecodeLinearCheckpointWindowFollowsCursor) {
  constexpr int32_t kKvBlockSize = 8;
  constexpr int32_t kPrefillChunkSize = 8;
  constexpr size_t kSpeculativeTokens = 3;
  BlockManagerPool::Options options = make_flat_kv_options();
  options.block_size(kKvBlockSize)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 64}})
      .enable_prefix_cache(false)
      .enable_disagg_pd(true)
      .instance_is_decode(true)
      .enable_kvcache_store(true)
      .num_embedding_blocks(64)
      .num_speculative_tokens(kSpeculativeTokens)
      .enable_schedule_overlap(false);

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  auto restore_stride = folly::makeGuard([original_chunk_stride]() {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride;
  });
  // Chunk equals the KV block, so prefix-cache-off HBM and prefix-cache-on
  // Host use the same checkpoint grid.
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kPrefillChunkSize;

  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);

  auto linear_ids = [](const KVCacheState& state) {
    const Slice<Block> blocks = state.blocks(BlockType::LINEAR);
    std::vector<int32_t> ids;
    ids.reserve(blocks.size());
    for (const Block& block : blocks) {
      ids.emplace_back(block.id());
    }
    return ids;
  };

  // Prompt already covers checkpoint 8. Decode starts with one live slot on
  // each tier. The next boundary under test is 16.
  Sequence overlap_off =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(10, 103));
  ASSERT_TRUE(pool.allocate(&overlap_off, /*num_tokens=*/10));
  overlap_off.kv_state().set_kv_cache_tokens_num(10);
  overlap_off.host_kv_state().set_kv_cache_tokens_num(8);
  ASSERT_EQ(linear_ids(overlap_off.kv_state()).size(), 1u);
  ASSERT_EQ(linear_ids(overlap_off.host_kv_state()).size(), 1u);
  const int32_t hbm_live = linear_ids(overlap_off.kv_state()).front();
  const int32_t host_live = linear_ids(overlap_off.host_kv_state()).front();

  // Overlap off writes the token index. append_token records the index before
  // the increment, so four generated tokens leave HBM cached tokens at 13.
  // 13 plus a 4-token speculative step reaches 17 and crosses 16.
  // cached - accepted is still below 16, so nothing is offloaded.
  overlap_off.kv_state().set_kv_cache_tokens_num(17);
  overlap_off.append_token(104);
  overlap_off.append_token(105);
  overlap_off.append_token(106);
  overlap_off.append_token(107);
  ASSERT_EQ(overlap_off.num_tokens(), 14u);
  ASSERT_EQ(overlap_off.kv_state().kv_cache_tokens_num(), 13u);
  overlap_off.set_accepted_length(0);
  ASSERT_TRUE(pool.allocate(&overlap_off, /*num_tokens=*/17));
  const std::vector<int32_t> hbm_window = linear_ids(overlap_off.kv_state());
  const std::vector<int32_t> host_window =
      linear_ids(overlap_off.host_kv_state());
  ASSERT_EQ(hbm_window.size(), 2u);
  ASSERT_EQ(host_window.size(), 2u);
  // Host offload appends the fresh write. The filled slot stays at the front
  // so collect can move it.
  EXPECT_EQ(hbm_window.front(), hbm_live);
  EXPECT_EQ(host_window.front(), host_live);
  EXPECT_NE(hbm_window.back(), hbm_live);
  EXPECT_NE(host_window.back(), host_live);
  // 13 and 8 share a KV block, so the window grows without an offload pair.
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);
  const int32_t hbm_next = hbm_window.back();
  const int32_t host_next = host_window.back();

  // The new cursor has not crossed a boundary. Host offload preserves the
  // completed result at the front and the exclusive write at the back.
  overlap_off.kv_state().set_kv_cache_tokens_num(14);
  overlap_off.set_accepted_length(0);
  ASSERT_EQ(overlap_off.kv_state().kv_cache_tokens_num() -
                overlap_off.accepted_length(),
            14u);
  ASSERT_TRUE(pool.allocate(&overlap_off, /*num_tokens=*/18));
  ASSERT_EQ(linear_ids(overlap_off.kv_state()).size(), 2u);
  ASSERT_EQ(linear_ids(overlap_off.host_kv_state()).size(), 2u);
  // The exclusive window swaps. The previous write becomes the read source
  // and the other exclusive slot becomes the live write.
  EXPECT_EQ(linear_ids(overlap_off.kv_state()).front(), hbm_next);
  EXPECT_EQ(linear_ids(overlap_off.host_kv_state()).front(), host_next);
  EXPECT_EQ(linear_ids(overlap_off.kv_state()).back(), hbm_live);
  EXPECT_EQ(linear_ids(overlap_off.host_kv_state()).back(), host_live);
  const int32_t hbm_completed = linear_ids(overlap_off.kv_state()).front();
  const int32_t host_completed =
      linear_ids(overlap_off.host_kv_state()).front();
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  // n-1 result. append_token records the index before the increment, so two
  // generated tokens leave the cursor at 15. The completed span crosses 16
  // only when the cursor itself is the boundary.
  // The completed result stays front. This step predicts a checkpoint, so
  // host offload releases the stale front and appends a fresh write. Collect
  // then moves that completed front.
  overlap_off.append_token(108);
  overlap_off.append_token(109);
  overlap_off.kv_state().set_kv_cache_tokens_num(16);
  overlap_off.set_accepted_length(0);
  ASSERT_EQ(overlap_off.kv_state().kv_cache_tokens_num(), 16u);
  ASSERT_TRUE(pool.allocate(&overlap_off, /*num_tokens=*/20));
  ASSERT_EQ(linear_ids(overlap_off.kv_state()).size(), 1u);
  ASSERT_EQ(linear_ids(overlap_off.host_kv_state()).size(), 1u);
  EXPECT_NE(linear_ids(overlap_off.kv_state()).back(), hbm_live);
  EXPECT_NE(linear_ids(overlap_off.host_kv_state()).back(), host_live);
  EXPECT_EQ(
      overlap_off.kv_state().blocks(BlockType::LINEAR).front().ref_count(), 1u);
  EXPECT_EQ(overlap_off.kv_state().blocks(BlockType::LINEAR).back().ref_count(),
            1u);
  EXPECT_EQ(
      overlap_off.host_kv_state().blocks(BlockType::LINEAR).front().ref_count(),
      1u);
  EXPECT_EQ(
      overlap_off.host_kv_state().blocks(BlockType::LINEAR).back().ref_count(),
      1u);
  {
    const std::shared_ptr<OffloadBlockPair> pair =
        HierarchyPoolTestPeer::take_linear_offload_pair(pool);
    ASSERT_NE(pair, nullptr);
    EXPECT_EQ(pair->block_type, BlockType::LINEAR);
    EXPECT_EQ(pair->src.id(), hbm_live);
    EXPECT_EQ(pair->dst.id(), host_live);
    EXPECT_NE(pair->src.id(),
              overlap_off.kv_state().blocks(BlockType::LINEAR).back().id());
    HierarchyPoolTestPeer::enqueue_offload_pair(pool, pair);
  }
  HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
  EXPECT_EQ(
      overlap_off.kv_state().blocks(BlockType::LINEAR).front().ref_count(), 1u);
  EXPECT_EQ(
      overlap_off.host_kv_state().blocks(BlockType::LINEAR).front().ref_count(),
      1u);

  // Overlap on writes the n-2 cursor. update_last_step_token records
  // cur_generated_token_idx_ before incrementing it. After four commits the
  // HBM cache cursor is 13 while num_tokens is already 14. The manager reads
  // overlap from its options, not from the sequence.
  options.enable_schedule_overlap(true);
  HierarchyBlockManagerPool overlap_pool(options,
                                         /*engine=*/nullptr,
                                         /*dp_size=*/1);
  Sequence overlap_on = make_test_sequence(
      /*index=*/1,
      std::vector<int32_t>(10, 203),
      /*enable_schedule_overlap=*/true);
  ASSERT_TRUE(overlap_pool.allocate(&overlap_on, /*num_tokens=*/10));
  overlap_on.kv_state().set_kv_cache_tokens_num(18);
  overlap_on.host_kv_state().set_kv_cache_tokens_num(8);
  ASSERT_EQ(linear_ids(overlap_on.kv_state()).size(), 1u);
  const int32_t overlap_hbm_live = linear_ids(overlap_on.kv_state()).front();
  const int32_t overlap_host_live =
      linear_ids(overlap_on.host_kv_state()).front();

  overlap_on.append_token(/*fake placeholder=*/-1);
  overlap_on.update_last_step_token(Token(204), /*token_offset=*/0);
  overlap_on.append_token(/*fake placeholder=*/-1);
  overlap_on.update_last_step_token(Token(205), /*token_offset=*/0);
  overlap_on.append_token(/*fake placeholder=*/-1);
  overlap_on.update_last_step_token(Token(206), /*token_offset=*/0);
  overlap_on.append_token(/*fake placeholder=*/-1);
  overlap_on.update_last_step_token(Token(207), /*token_offset=*/0);
  ASSERT_EQ(overlap_on.num_tokens(), 14u);
  ASSERT_EQ(overlap_on.kv_state().kv_cache_tokens_num(), 13u);
  ASSERT_EQ(overlap_on.accepted_length(), 0u);
  ASSERT_TRUE(overlap_pool.allocate(&overlap_on, /*num_tokens=*/18));
  const std::vector<int32_t> overlap_hbm = linear_ids(overlap_on.kv_state());
  const std::vector<int32_t> overlap_host =
      linear_ids(overlap_on.host_kv_state());
  ASSERT_EQ(overlap_hbm.size(), 2u);
  ASSERT_EQ(overlap_host.size(), 2u);
  EXPECT_EQ(overlap_hbm.front(), overlap_hbm_live);
  EXPECT_EQ(overlap_host.front(), overlap_host_live);
  EXPECT_NE(overlap_hbm.back(), overlap_hbm_live);
  EXPECT_NE(overlap_host.back(), overlap_host_live);
  // 13 and 8 share a KV block, so the overlap window grows without a pair.
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(overlap_pool),
            0u);

  // Three commits advance the overlap cursor to 16. Allocate preserves the
  // completed front. Collect moves that front into the checkpoint pair and
  // leaves the exclusive write as the live slot.
  const int32_t overlap_hbm_next = overlap_hbm.back();
  const int32_t overlap_host_next = overlap_host.back();
  overlap_on.append_token(/*fake placeholder=*/-1);
  overlap_on.update_last_step_token(Token(208), /*token_offset=*/0);
  overlap_on.append_token(/*fake placeholder=*/-1);
  overlap_on.update_last_step_token(Token(209), /*token_offset=*/0);
  overlap_on.append_token(/*fake placeholder=*/-1);
  overlap_on.update_last_step_token(Token(210), /*token_offset=*/0);
  ASSERT_EQ(overlap_on.num_tokens(), 17u);
  ASSERT_EQ(overlap_on.kv_state().kv_cache_tokens_num(), 16u);
  ASSERT_EQ(overlap_on.accepted_length(), 0u);
  ASSERT_TRUE(overlap_pool.allocate(&overlap_on, /*num_tokens=*/20));
  const std::vector<int32_t> overlap_hbm_after =
      linear_ids(overlap_on.kv_state());
  const std::vector<int32_t> overlap_host_after =
      linear_ids(overlap_on.host_kv_state());
  ASSERT_EQ(overlap_hbm_after.size(), 1u);
  ASSERT_EQ(overlap_host_after.size(), 1u);
  EXPECT_EQ(overlap_hbm_after.back(), overlap_hbm_next);
  EXPECT_EQ(overlap_host_after.back(), overlap_host_next);
  EXPECT_NE(overlap_hbm_after.back(), overlap_hbm_live);
  EXPECT_EQ(overlap_on.kv_state().blocks(BlockType::LINEAR).front().ref_count(),
            1u);
  EXPECT_EQ(overlap_on.kv_state().blocks(BlockType::LINEAR).back().ref_count(),
            1u);
  EXPECT_EQ(
      overlap_on.host_kv_state().blocks(BlockType::LINEAR).front().ref_count(),
      1u);
  EXPECT_EQ(
      overlap_on.host_kv_state().blocks(BlockType::LINEAR).back().ref_count(),
      1u);
  {
    const std::shared_ptr<OffloadBlockPair> pair =
        HierarchyPoolTestPeer::take_linear_offload_pair(overlap_pool);
    ASSERT_NE(pair, nullptr);
    EXPECT_EQ(pair->block_type, BlockType::LINEAR);
    EXPECT_EQ(pair->src.id(), overlap_hbm_live);
    EXPECT_EQ(pair->dst.id(), overlap_host_live);
    EXPECT_NE(pair->src.id(), overlap_hbm_after.back());
    HierarchyPoolTestPeer::enqueue_offload_pair(overlap_pool, pair);
  }
  HierarchyPoolTestPeer::release_pending_offload_pairs(overlap_pool);
}

TEST(HierarchyBlockManagerPoolTest,
     OverlapDecodeRetiresInvalidPrefillHostFront) {
  constexpr int32_t kKvBlockSize = 4;
  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const auto restore_stride = folly::makeGuard([original_chunk_stride] {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride;
  });
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kKvBlockSize;

  for (const bool host_has_capacity : {false, true}) {
    SCOPED_TRACE(host_has_capacity);
    BlockManagerPool::Options options = make_flat_kv_options();
    options.enable_linear_state(true)
        .block_size(kKvBlockSize)
        .linear_state_num_slots(16)
        .host_num_blocks_by_type(
            {{BlockType::KV, 128}, {BlockType::LINEAR, 16}})
        .enable_prefix_cache(false)
        .enable_disagg_pd(true)
        .instance_is_decode(true)
        .enable_kvcache_store(true)
        .num_embedding_blocks(64)
        .num_speculative_tokens(1)
        .enable_schedule_overlap(true);
    HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
    BlockManager* host_leaf = HierarchyPoolTestPeer::host_leaves(pool)
                                  .at(BlockType::LINEAR)
                                  .leaf.get();
    const size_t host_free_before = host_leaf->num_free_blocks();
    Sequence sequence = make_test_sequence(
        /*index=*/0,
        std::vector<int32_t>(9, 103),
        /*enable_schedule_overlap=*/true);
    auto cleanup = folly::makeGuard([&pool, &sequence] {
      pool.deallocate(&sequence);
      HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
    });
    ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/9));
    sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
    auto append_generated_token = [&sequence](int32_t token_id) {
      sequence.append_token(/*fake placeholder=*/-1);
      sequence.update_last_step_token(Token(token_id), /*token_offset=*/0);
    };
    append_generated_token(104);
    append_generated_token(105);
    append_generated_token(106);
    sequence.set_accepted_length(0);
    ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
    ASSERT_EQ(sequence.host_kv_state().num_blocks(BlockType::LINEAR), 2u);
    const int32_t host_read_id =
        sequence.host_kv_state().blocks(BlockType::LINEAR).back().id();

    // Actual prefill collect removes the host front, leaving [-1, live];
    // simulate its in-flight ownership.
    Block prefill_destination =
        HierarchyPoolTestPeer::take_host_linear_front(sequence);
    ASSERT_TRUE(prefill_destination.is_valid());
    ASSERT_FALSE(
        sequence.host_kv_state().blocks(BlockType::LINEAR).front().is_valid());
    ASSERT_EQ(sequence.host_kv_state().blocks(BlockType::LINEAR).front().id(),
              -1);
    std::vector<Block> host_held;
    if (!host_has_capacity) {
      host_held = exhaust_free_blocks(host_leaf, /*keep_free=*/0);
      ASSERT_EQ(host_leaf->num_free_blocks(), 0u);
    }
    append_generated_token(107);
    sequence.set_accepted_length(1);
    ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), 12u);
    ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
    const Slice<Block> repaired =
        sequence.host_kv_state().blocks(BlockType::LINEAR);
    // Overlap keeps the invalid front for collect. The live slot stays.
    ASSERT_EQ(repaired.size(), 2u);
    EXPECT_FALSE(repaired.front().is_valid());
    EXPECT_TRUE(repaired.back().is_valid());
    EXPECT_EQ(repaired.back().id(), host_read_id);
    EXPECT_NE(repaired.back().id(), prefill_destination.id());
    EXPECT_EQ(repaired.back().ref_count(), 1u);
    std::shared_ptr<OffloadBlockPair> pair =
        HierarchyPoolTestPeer::take_linear_offload_pair(pool);
    // The front is invalid, so collect does not queue a LINEAR pair.
    EXPECT_EQ(pair, nullptr);
    pair.reset();
    host_held.clear();
    prefill_destination = Block();
    pool.deallocate(&sequence);
    HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
    cleanup.dismiss();
    host_leaf->reset_prefix_cache();
    EXPECT_EQ(host_leaf->num_free_blocks(), host_free_before);
  }
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeLinearCheckpointRowTracksBoundaryOvershoot) {
  constexpr size_t kCheckpointStride = 8;
  constexpr size_t kAcceptedLength = 3;
  Sequence sequence =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(16, 103));
  sequence.set_accepted_length(kAcceptedLength);

  for (size_t hbm_tokens = 8; hbm_tokens <= 10; ++hbm_tokens) {
    sequence.kv_state().set_kv_cache_tokens_num(hbm_tokens);
    const std::optional<size_t> checkpoint_row =
        last_decode_step_checkpoint_row(sequence, kCheckpointStride);
    ASSERT_TRUE(checkpoint_row.has_value());
    // begin = hbm cached tokens - accepted_length - 1. The row is the
    // accepted offset of the token that crossed the boundary.
    EXPECT_EQ(checkpoint_row.value(), kAcceptedLength);
    EXPECT_TRUE(
        last_decode_step_crosses_checkpoint(sequence, kCheckpointStride));
  }

  sequence.kv_state().set_kv_cache_tokens_num(12);
  EXPECT_FALSE(
      last_decode_step_checkpoint_row(sequence, kCheckpointStride).has_value());
  EXPECT_FALSE(
      last_decode_step_crosses_checkpoint(sequence, kCheckpointStride));

  sequence.kv_state().set_kv_cache_tokens_num(2049);
  EXPECT_EQ(
      last_decode_step_checkpoint_row(sequence, /*checkpoint_stride=*/512),
      kAcceptedLength);
}

TEST(LinearPrefetchSelectionTest, KeepsLatestUnwrittenFullHit) {
  std::vector<Block> blocks = {linear_block(1),
                               linear_block(2),
                               linear_block(3),
                               linear_block(4),
                               linear_block(1),
                               linear_block(2)};
  const std::vector<uint8_t> hits = {2, 2, 2, 2, 0, 2};
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/4, hits, 2),
            4u);
}

TEST(LinearPrefetchSelectionTest, CompletedNextGroupOverwritesEarlierAlias) {
  std::vector<Block> blocks(4);
  blocks[0] = linear_block(1);
  blocks[1] = linear_block(2);
  blocks[2] = linear_block(1);
  blocks[3] = linear_block(2);
  const std::vector<uint8_t> hits = {2, 2, 2, 2};
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/4, hits, 2),
            4u);
  // The second batch rewrites both pages and sits past the KV prefix.
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/2, hits, 2),
            0u);
}

TEST(LinearPrefetchSelectionTest, HitPastKvPrefixOverwritesEarlierAlias) {
  std::vector<Block> blocks(2, linear_block(1));
  const std::vector<uint8_t> hits = {2, 2};
  // The second batch rewrites the only page and sits past the KV prefix.
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/1, hits, 2),
            0u);
}

TEST(LinearPrefetchSelectionTest, HostPrefixIsNotAPrefetchHit) {
  std::vector<Block> blocks = {linear_block(7), linear_block(1)};
  const std::vector<uint8_t> hits = {0};
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/1, /*kv_common_blocks=*/1, hits, 2),
            1u);
}

TEST(LinearPrefetchSelectionTest, HostPrefixIsNotInspected) {
  std::vector<Block> blocks = {
      linear_block(1), linear_block(9), linear_block(1)};
  const std::vector<uint8_t> hits = {2, 2};
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/1, /*kv_common_blocks=*/2, hits, 2),
            2u);
}

TEST(LinearPrefetchSelectionTest, PartialGroupReuseKeepsEarlierBatch) {
  std::vector<Block> blocks(16);
  blocks[3] = linear_block(1);
  blocks[7] = linear_block(2);
  blocks[11] = linear_block(1);
  blocks[15] = linear_block(3);
  std::vector<uint8_t> hits(16, 2);
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/16, hits, 2),
            16u);
}

TEST(LinearPrefetchSelectionTest, TailCheckpointInsidePrefixIsKept) {
  std::vector<Block> blocks(15);
  blocks[3] = linear_block(1);
  blocks[7] = linear_block(2);
  blocks[11] = linear_block(3);
  blocks[14] = linear_block(1);
  std::vector<uint8_t> hits(15, 2);
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/15, hits, 2),
            15u);
}

TEST(LinearPrefetchSelectionTest, KeepsDeepestFullHitInsideKvPrefix) {
  std::vector<Block> blocks(15);
  blocks[3] = linear_block(1);
  blocks[7] = linear_block(2);
  blocks[11] = linear_block(1);
  std::vector<uint8_t> hits(15, 2);
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/15, hits, 2),
            12u);
}

TEST(LinearPrefetchSelectionTest, PartialHitInsidePrefixOverwritesSharedPage) {
  std::vector<Block> blocks = {linear_block(1),
                               linear_block(2),
                               linear_block(3),
                               linear_block(4),
                               linear_block(1),
                               linear_block(2),
                               linear_block(3),
                               linear_block(4)};
  const std::vector<uint8_t> hits = {4, 4, 4, 4, 0, 0, 0, 1};
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/8, hits, 4),
            3u);
}

TEST(LinearPrefetchSelectionTest, PartialHitInsidePrefixDropsDirtyPage) {
  std::vector<Block> blocks(8);
  blocks[3] = linear_block(4);
  blocks[7] = linear_block(4);
  const std::vector<uint8_t> hits = {0, 0, 0, 4, 0, 0, 0, 1};
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/8, hits, 4),
            0u);
}

TEST(LinearPrefetchSelectionTest, MissPastKvPrefixDoesNotOverwrite) {
  std::vector<Block> blocks(6);
  blocks[0] = linear_block(1);
  blocks[1] = linear_block(2);
  blocks[4] = linear_block(1);
  const std::vector<uint8_t> hits = {2, 2, 0, 0, 2, 0};
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/5, hits, 2),
            5u);
}

TEST(LinearPrefetchSelectionTest, InvalidPastKvPrefixDoesNotOverwrite) {
  std::vector<Block> blocks(20);
  blocks[3] = linear_block(1);
  blocks[7] = linear_block(2);
  blocks[11] = linear_block(1);
  // Positions 16-19 stay default-invalid.
  std::vector<uint8_t> hits(20, 2);
  EXPECT_EQ(last_reusable_linear_checkpoint(
                blocks, /*base_blocks=*/0, /*kv_common_blocks=*/17, hits, 2),
            12u);
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeLinearOffloadUsesCanonicalMtpCheckpointHash) {
  // Prefix cache is off, so HBM LINEAR uses max(KV block, chunk). Host LINEAR
  // keeps prefix cache and uses the KV block. Pin the chunk to the KV block
  // so both tiers share this checkpoint grid.
  constexpr int32_t kPrefillChunkSize = 4;
  constexpr int32_t kKvBlockSize = 4;
  // One accepted draft token. The completed span is
  // [cached - accepted - 1, cached) and crosses the KV block at 12.
  constexpr size_t kAcceptedLength = 1;
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_linear_state(true)
      .block_size(kKvBlockSize)
      .linear_state_num_slots(64)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 64}})
      .enable_prefix_cache(false)
      .hasher_type(BlockHasherType::MTP_TEXT)
      .enable_disagg_pd(true)
      .instance_is_decode(true)
      .enable_kvcache_store(true)
      .num_embedding_blocks(64)
      .num_speculative_tokens(kAcceptedLength)
      .enable_schedule_overlap(false);

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const auto restore_chunk_stride = folly::makeGuard([original_chunk_stride] {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride;
  });
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kPrefillChunkSize;

  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  Sequence sequence =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(9, 103));
  const auto cleanup = folly::makeGuard([&pool, &sequence] {
    pool.deallocate(&sequence);
    pool.transfer_blocks();
  });
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/9));
  // Cover the prompt before appending. A second prefill allocate would move
  // the host front into an offload pair and leave that slot invalid.
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  ASSERT_EQ(sequence.stage(), SequenceStage::DECODE);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);

  // append_token records the index before the increment. Three generated
  // tokens leave the cursor at 11. The next speculative step reaches 13 and
  // crosses 12, while [10, 11) does not, so the window grows and stays.
  sequence.append_token(104);
  sequence.append_token(105);
  sequence.append_token(106);
  ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), 11u);
  sequence.set_accepted_length(0);
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  pool.transfer_blocks();
  EXPECT_EQ(transfer_count(engine, BlockType::LINEAR), 0u);

  const Slice<Block> hbm_before = sequence.kv_state().blocks(BlockType::LINEAR);
  const Slice<Block> host_before =
      sequence.host_kv_state().blocks(BlockType::LINEAR);
  ASSERT_EQ(hbm_before.size(), 2u);
  ASSERT_EQ(host_before.size(), 2u);
  const int32_t hbm_checkpoint_id = hbm_before.front().id();
  const int32_t hbm_live_id = hbm_before.back().id();
  const int32_t host_checkpoint_id = host_before.front().id();
  const int32_t host_live_id = host_before.back().id();

  // One more generated token lands the cursor on 12. accepted_length 1 makes
  // the completed span [10, 12), which crosses that boundary. Host offload
  // keeps the completed checkpoint at the front and appends the next write.
  // collect moves that front.
  sequence.append_token(107);
  ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), 12u);
  sequence.set_accepted_length(kAcceptedLength);
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  // Overlap retains completed KV blocks after collect has queued the pair.
  // Holding them across transfer_blocks checks that the queued copy still
  // completes while those keepalives are alive.
  const std::vector<Block> inflight = sequence.get_inflight_blocks();

  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            hbm_checkpoint_id);
  ASSERT_EQ(sequence.host_kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.host_kv_state().blocks(BlockType::LINEAR).front().id(),
            host_checkpoint_id);
  pool.transfer_blocks();
  ASSERT_FALSE(engine.offloads_.empty());
  const auto& infos = engine.offloads_.back().infos;
  const auto checkpoint_kv = std::find_if(
      infos.begin(), infos.end(), [](const BlockTransferInfo& info) {
        return info.block_type == BlockType::KV;
      });
  ASSERT_NE(checkpoint_kv, infos.end());
  EXPECT_EQ(XXH3Key(checkpoint_kv->hash_key),
            sequence.block_hashes(kKvBlockSize)[2]);
  const auto linear = std::find_if(
      infos.begin(), infos.end(), [](const BlockTransferInfo& info) {
        return info.block_type == BlockType::LINEAR;
      });
  ASSERT_NE(linear, infos.end());
  EXPECT_EQ(linear->src_block_id, hbm_live_id);
  EXPECT_EQ(linear->dst_block_id, host_live_id);
  EXPECT_EQ(linear->checkpoint_row, kAcceptedLength);
  EXPECT_EQ(XXH3Key(linear->hash_key), sequence.block_hashes(kKvBlockSize)[2]);

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(HierarchyBlockManagerPoolTest, DecodeOffloadsKvWithExtraHbmReference) {
  constexpr int32_t kBlockSize = 4;
  constexpr size_t kPreviousBoundary = 8;
  constexpr size_t kCheckpoint = 12;
  BlockManagerPool::Options options = make_flat_kv_options();
  options.block_size(kBlockSize)
      .enable_linear_state(true)
      .linear_state_num_slots(16)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 16}})
      .hasher_type(BlockHasherType::MTP_TEXT)
      .enable_disagg_pd(true)
      .instance_is_decode(true)
      .enable_kvcache_store(true)
      .num_embedding_blocks(16)
      .num_speculative_tokens(1);
  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const auto restore_stride = folly::makeGuard([original_chunk_stride] {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_chunk_stride;
  });
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kBlockSize;

  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  const auto shutdown =
      folly::makeGuard([&pool, &engine] { shutdown_pool(pool, engine); });
  Sequence sequence =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(9, 103));
  const auto cleanup = folly::makeGuard([&pool, &sequence] {
    pool.deallocate(&sequence);
    pool.transfer_blocks();
  });
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.host_kv_state().set_kv_cache_tokens_num(kPreviousBoundary);
  sequence.append_token(104);
  sequence.append_token(105);
  sequence.append_token(106);
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.append_token(107);
  sequence.set_accepted_length(1);
  ASSERT_EQ(sequence.kv_cache_tokens_num(), kCheckpoint);
  const Block& kv_block = sequence.kv_state().blocks(BlockType::KV)[2];
  const int32_t kv_id = kv_block.id();
  {
    const Block keepalive = kv_block;
    ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
    EXPECT_EQ(keepalive.ref_count(), 4u);
    pool.transfer_blocks();
    EXPECT_EQ(transfer_count(engine, BlockType::LINEAR), 1u);
    ASSERT_EQ(transfer_count(engine, BlockType::KV), 1u);
    ASSERT_FALSE(engine.offloads_.empty());
    const auto& infos = engine.offloads_.back().infos;
    const auto kv = std::find_if(
        infos.begin(), infos.end(), [](const BlockTransferInfo& info) {
          return info.block_type == BlockType::KV;
        });
    ASSERT_NE(kv, infos.end());
    EXPECT_EQ(kv->src_block_id, kv_id);
    EXPECT_EQ(XXH3Key(kv->hash_key), sequence.block_hashes(kBlockSize)[2]);
    EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), kCheckpoint);
  }
  HierarchyPoolTestPeer::collect_offload_pairs(pool, &sequence);
  pool.transfer_blocks();
  EXPECT_EQ(transfer_count(engine, BlockType::KV), 1u);
  EXPECT_EQ(transfer_count(engine, BlockType::LINEAR), 1u);
}

TEST(HierarchyBlockManagerPoolTest, V41DecodeOffloadsCommittedTokenUnit) {
  constexpr size_t kUnit = 256;
  auto options = make_v41_options();
  options.instance_is_decode(true).enable_disagg_pd(true).swa_blocks_per_seq(
      512 / options.block_size() + 1);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  auto request = make_test_request(std::vector<int32_t>(kUnit, 83),
                                   /*best_of=*/1,
                                   /*generation_budget=*/kUnit + 2);
  Sequence& source = *request->sequences().front();
  ASSERT_TRUE(pool.try_allocate(&source));
  ASSERT_EQ(source.stage(), SequenceStage::DECODE);
  EXPECT_EQ(source.host_kv_state().kv_cache_tokens_num(), kUnit);
  pool.transfer_blocks();
  EXPECT_TRUE(engine.offloads_.empty());
  for (size_t token = 0; token <= kUnit; ++token) {
    source.append_token(83);
    ASSERT_TRUE(pool.allocate(&source, source.num_tokens()));
    pool.transfer_blocks();
  }
  EXPECT_EQ(transfer_count(engine, BlockType::KV), 1u);
  EXPECT_GT(transfer_count(engine, BlockType::SWA), 0u);
  EXPECT_EQ(source.host_kv_state().kv_cache_tokens_num(), 2 * kUnit);
  pool.deallocate(&source);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest,
     V41MultimodalDecodeOffloadsCommittedTokenUnit) {
  constexpr size_t kUnit = 256;
  for (const bool enable_disagg_pd : {false, true}) {
    SCOPED_TRACE(enable_disagg_pd ? "PD Decode" : "standalone decode");
    auto options = make_v41_options();
    options.hasher_type(BlockHasherType::MM)
        .instance_is_decode(enable_disagg_pd)
        .enable_disagg_pd(enable_disagg_pd)
        .swa_blocks_per_seq(5);
    FakePrefetchEngine engine(2);
    HierarchyBlockManagerPool pool(options, &engine, 1);
    auto shutdown =
        folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
    auto request = make_test_request(
        std::vector<int32_t>(kUnit, 83),
        /*best_of=*/1,
        /*generation_budget=*/kUnit + 2,
        make_image_mm_data(/*offset=*/0,
                           /*length=*/static_cast<int32_t>(kUnit),
                           /*key_byte=*/3));
    Sequence& source = *request->sequences().front();
    if (enable_disagg_pd) {
      ASSERT_TRUE(pool.try_allocate(&source));
    } else {
      ASSERT_TRUE(pool.allocate(&source, source.num_tokens()));
      source.kv_state().set_kv_cache_tokens_num(kUnit);
      ASSERT_TRUE(pool.allocate(&source, source.num_tokens()));
    }
    pool.transfer_blocks();
    ASSERT_EQ(source.stage(), SequenceStage::DECODE);
    ASSERT_EQ(source.host_kv_state().kv_cache_tokens_num(), kUnit);
    if (enable_disagg_pd) {
      EXPECT_TRUE(engine.offloads_.empty());
    }
    const size_t kv_before = transfer_count(engine, BlockType::KV);
    const size_t swa_before = transfer_count(engine, BlockType::SWA);
    for (size_t token = 0; token <= kUnit; ++token) {
      source.append_token(83);
      ASSERT_TRUE(pool.allocate(&source, source.num_tokens()));
      pool.transfer_blocks();
    }
    EXPECT_EQ(transfer_count(engine, BlockType::KV) - kv_before, 1u);
    EXPECT_GT(transfer_count(engine, BlockType::SWA), swa_before);
    EXPECT_EQ(source.host_kv_state().kv_cache_tokens_num(), 2 * kUnit);
    pool.deallocate(&source);
    pool.transfer_blocks();
  }
}

TEST(HierarchyBlockManagerPoolTest, EmptyHostConfigurationRejectsAllocation) {
  auto options = make_flat_kv_options();
  options.host_num_blocks(0).host_num_blocks_by_type({});
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  Sequence source = make_test_sequence(0, std::vector<int32_t>(257, 23));
  EXPECT_DEATH(pool.allocate(&source, 257), "host_manager");
}

TEST(HierarchyBlockManagerPoolTest, ExistingHostBlocksSkipPrefixRematch) {
  auto options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  publish_host_prefix(pool, engine, std::vector<int32_t>(129, 43));
  Sequence source = make_test_sequence(0, std::vector<int32_t>(1025, 43));
  pool.allocate_shared(&source);
  ASSERT_EQ(source.host_kv_state().kv_cache_tokens_num(), 128u);
  ASSERT_EQ(source.host_kv_state().num_blocks(BlockType::KV), 1u);
  publish_host_prefix(pool, engine, std::vector<int32_t>(1025, 43));
  pool.allocate_shared(&source);
  EXPECT_EQ(source.host_kv_state().kv_cache_tokens_num(), 128u);
  EXPECT_EQ(source.host_kv_state().num_blocks(BlockType::KV), 1u);
  EXPECT_EQ(source.kv_state().kv_cache_tokens_num(), 0u);
  pool.deallocate(&source);
  pool.transfer_blocks();
}

TEST(HierarchyBlockManagerPoolTest,
     CopyBudgetTrimFallsBackPastMissingSwaWindow) {
  auto options = make_v41_options();
  // Four usable Host SWA pages hold the pinned first unit plus the final
  // one-page window. Normal growth must evict the unpinned middle unit.
  options.host_num_blocks_by_type({{BlockType::SWA, 5}, {BlockType::KV, 128}});
  FakePrefetchEngine engine(2);
  HierarchyBlockManagerPool pool(options, &engine, 1);
  auto shutdown =
      folly::makeGuard([&pool, &engine]() { shutdown_pool(pool, engine); });
  const std::vector<int32_t> tokens(1025, 83);
  Sequence source = make_test_sequence(0, tokens);
  ASSERT_TRUE(pool.allocate(&source, 256));
  source.kv_state().set_kv_cache_tokens_num(256);
  ASSERT_TRUE(pool.allocate(&source, 512));
  pool.transfer_blocks();
  auto pin = make_test_request(std::vector<int32_t>(257, 83));
  bool pin_ready = false;
  pool.prefetch_from_storage(
      pin, [&pin_ready](std::shared_ptr<Request>) { pin_ready = true; });
  ASSERT_TRUE(pin_ready);
  for (const size_t target : {768u, 1024u}) {
    source.kv_state().set_kv_cache_tokens_num(target - 256);
    ASSERT_TRUE(pool.allocate(&source, target));
    pool.transfer_blocks();
  }
  source.kv_state().set_kv_cache_tokens_num(1024);
  pool.deallocate(&source);
  pool.transfer_blocks();
  auto restored = make_test_request(tokens);
  bool ready = false;
  pool.prefetch_from_storage(
      restored, [&ready](std::shared_ptr<Request>) { ready = true; });
  ASSERT_TRUE(ready);
  Sequence& sequence = *restored->sequences().front();
  ASSERT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 1024u);
  const auto swa = sequence.host_kv_state().blocks(BlockType::SWA);
  ASSERT_EQ(swa.size(), 4u);
  EXPECT_TRUE(swa[1].is_valid());
  EXPECT_TRUE(swa[2].is_valid());
  const auto point = pool.select_host_cache_restore(&sequence, 3);
  EXPECT_EQ(point.restore_target_tokens, 768u);
  EXPECT_EQ(point.copy_units, 3u);
}

}  // namespace
}  // namespace xllm
