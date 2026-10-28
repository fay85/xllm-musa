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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/distributed_runtime/llm_engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/stopping_checker.h"

namespace xllm {
namespace {

class RecordingWorker final : public WorkerClient {
 public:
  void prefetch_from_storage(
      const std::shared_ptr<const StoragePrefetchRequest>& request,
      std::shared_ptr<PrefetchResult> result,
      size_t worker_index) override {
    prefetch_request_ = request;
    prefetch_result_ = std::move(result);
    prefetch_index_ = worker_index;
  }

  folly::SemiFuture<uint32_t> transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& infos) override {
    ++offload_count_;
    offload_infos_ = infos;
    return folly::makeSemiFuture(static_cast<uint32_t>(infos.size()));
  }

  void transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& infos) override {
    ++load_count_;
    load_batch_id_ = batch_id;
    load_infos_ = infos;
  }

  uint32_t offload_count_ = 0;
  uint32_t load_count_ = 0;
  uint64_t load_batch_id_ = 0;
  std::vector<BlockTransferInfo> offload_infos_;
  std::vector<BlockTransferInfo> load_infos_;
  std::shared_ptr<const StoragePrefetchRequest> prefetch_request_;
  std::shared_ptr<PrefetchResult> prefetch_result_;
  size_t prefetch_index_ = 0;
};

class ClientEngine final : public LLMEngine {
 public:
  ClientEngine(runtime::Options options,
               std::vector<std::shared_ptr<WorkerClient>> clients)
      : LLMEngine(std::move(options), std::move(clients)) {}
};

#if defined(USE_MUSA)
class RecordingPullWorker final : public WorkerClient {
 public:
  struct PullCall {
    uint64_t cluster_id;
    std::string address;
    std::vector<KVTransferMapping> mappings;
  };

  bool pull_kv_blocks(uint64_t cluster_id,
                      const std::string& address,
                      const std::vector<KVTransferMapping>& mappings) override {
    record_call(cluster_id, address, mappings);
    return address != failed_address_;
  }

  folly::SemiFuture<bool> pull_kv_blocks_async(
      uint64_t cluster_id,
      const std::string& address,
      const std::vector<KVTransferMapping>& mappings) override {
    record_call(cluster_id, address, mappings);
    if (!defer_) {
      return folly::makeSemiFuture(address != failed_address_);
    }
    folly::Promise<bool> promise;
    folly::SemiFuture<bool> future = promise.getSemiFuture();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_promises_.emplace_back(std::move(promise));
    }
    condition_.notify_all();
    return future;
  }

  void wait_for_calls(size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this, count] { return call_count_ >= count; });
  }

  void resolve_all(bool value) {
    std::vector<folly::Promise<bool>> promises;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      promises.swap(pending_promises_);
    }
    for (folly::Promise<bool>& promise : promises) {
      promise.setValue(value);
    }
  }

  size_t call_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return call_count_;
  }

  bool defer_ = false;
  std::vector<PullCall> calls_;
  std::string failed_address_;

 private:
  void record_call(uint64_t cluster_id,
                   const std::string& address,
                   const std::vector<KVTransferMapping>& mappings) {
    std::lock_guard<std::mutex> lock(mutex_);
    calls_.push_back({cluster_id, address, mappings});
    ++call_count_;
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<folly::Promise<bool>> pending_promises_;
  size_t call_count_ = 0;
};

class LLMEnginePullRouteTest : public ::testing::Test {
 protected:
  void create_engine(int32_t world_size, int32_t dp_size) {
    std::vector<std::shared_ptr<WorkerClient>> clients;
    clients.reserve(world_size);
    workers_.reserve(world_size);
    for (int32_t rank = 0; rank < world_size; ++rank) {
      auto worker = std::make_shared<RecordingPullWorker>();
      clients.emplace_back(worker);
      workers_.emplace_back(std::move(worker));
    }
    runtime::Options options;
    options.world_size(world_size).nnodes(world_size).dp_size(dp_size);
    engine_ =
        std::make_unique<ClientEngine>(std::move(options), std::move(clients));
  }

  std::vector<std::shared_ptr<RecordingPullWorker>> workers_;
  std::unique_ptr<ClientEngine> engine_;
};

TEST_F(LLMEnginePullRouteTest, VisitsAllSourcesInOnlyTheRequestedDpGroup) {
  create_engine(/*world_size=*/16, /*dp_size=*/2);
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> addresses;
  cluster_ids.reserve(16);
  addresses.reserve(16);
  for (int32_t rank = 0; rank < 16; ++rank) {
    cluster_ids.emplace_back(100 + rank);
    addresses.emplace_back("source-" + std::to_string(rank));
  }
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::C4);
  mapping.local_ids = {3};
  mapping.remote_ids = {7};
  mapping.remote_shared_num = 1;

  ASSERT_TRUE(engine_->pull_kv_blocks(/*src_dp_size=*/2,
                                      /*src_dp_rank=*/1,
                                      cluster_ids,
                                      addresses,
                                      /*dst_dp_rank=*/1,
                                      {mapping}));
  for (size_t rank = 0; rank < 8; ++rank) {
    EXPECT_TRUE(workers_[rank]->calls_.empty());
  }
  for (size_t rank = 8; rank < 16; ++rank) {
    const auto& calls = workers_[rank]->calls_;
    ASSERT_EQ(calls.size(), 8U) << "destination rank " << rank;
    for (size_t index = 0; index < calls.size(); ++index) {
      EXPECT_EQ(calls[index].cluster_id, cluster_ids[8 + index]);
      EXPECT_EQ(calls[index].address, addresses[8 + index]);
      ASSERT_EQ(calls[index].mappings.size(), 1U);
      EXPECT_EQ(calls[index].mappings[0].group_id, mapping.group_id);
      EXPECT_EQ(calls[index].mappings[0].local_ids, mapping.local_ids);
      EXPECT_EQ(calls[index].mappings[0].remote_ids, mapping.remote_ids);
      EXPECT_EQ(calls[index].mappings[0].remote_shared_num,
                mapping.remote_shared_num);
    }
  }
}

TEST_F(LLMEnginePullRouteTest, PropagatesFailureFromANonmatchingSourceRank) {
  create_engine(/*world_size=*/2, /*dp_size=*/1);
  workers_[1]->failed_address_ = "source-0";
  EXPECT_FALSE(engine_->pull_kv_blocks(/*src_dp_size=*/1,
                                       /*src_dp_rank=*/0,
                                       {10, 11},
                                       {"source-0", "source-1"},
                                       /*dst_dp_rank=*/0,
                                       {}));
}

TEST_F(LLMEnginePullRouteTest, RejectsInvalidTopologyBeforeDispatch) {
  create_engine(/*world_size=*/2, /*dp_size=*/1);
  EXPECT_FALSE(engine_->pull_kv_blocks(/*src_dp_size=*/0,
                                       /*src_dp_rank=*/0,
                                       {10, 11},
                                       {"source-0", "source-1"},
                                       /*dst_dp_rank=*/0,
                                       {}));
  EXPECT_FALSE(engine_->pull_kv_blocks(/*src_dp_size=*/1,
                                       /*src_dp_rank=*/0,
                                       {10, 11},
                                       {"source-0"},
                                       /*dst_dp_rank=*/0,
                                       {}));
  EXPECT_FALSE(engine_->pull_kv_blocks(/*src_dp_size=*/1,
                                       /*src_dp_rank=*/1,
                                       {10, 11},
                                       {"source-0", "source-1"},
                                       /*dst_dp_rank=*/0,
                                       {}));
  EXPECT_FALSE(engine_->pull_kv_blocks(/*src_dp_size=*/1,
                                       /*src_dp_rank=*/0,
                                       {10, 11},
                                       {"source-0", "source-1"},
                                       /*dst_dp_rank=*/1,
                                       {}));
  for (const auto& worker : workers_) {
    EXPECT_TRUE(worker->calls_.empty());
  }
}
#endif

#if defined(USE_MUSA)
TEST_F(LLMEnginePullRouteTest, DispatchesAllDestinationsBeforeWaiting) {
  create_engine(/*world_size=*/2, /*dp_size=*/1);
  workers_[0]->defer_ = true;
  workers_[1]->defer_ = true;

  std::atomic<bool> result = false;
  std::thread caller([this, &result] {
    result = engine_->pull_kv_blocks(
        /*src_dp_size=*/1,
        /*src_dp_rank=*/0,
        {10},
        {"source-0"},
        /*dst_dp_rank=*/0,
        {});
  });
  workers_[0]->wait_for_calls(1);
  workers_[1]->wait_for_calls(1);
  workers_[0]->resolve_all(true);
  workers_[1]->resolve_all(true);
  caller.join();
  EXPECT_TRUE(result);
  EXPECT_EQ(workers_[0]->call_count(), 1U);
  EXPECT_EQ(workers_[1]->call_count(), 1U);
}

TEST_F(LLMEnginePullRouteTest, FailureDrainsEverySubmittedDestinationFuture) {
  create_engine(/*world_size=*/2, /*dp_size=*/1);
  workers_[0]->defer_ = true;
  workers_[1]->defer_ = true;

  std::atomic<bool> result = true;
  std::thread caller([this, &result] {
    result = engine_->pull_kv_blocks(
        /*src_dp_size=*/1,
        /*src_dp_rank=*/0,
        {10},
        {"source-0"},
        /*dst_dp_rank=*/0,
        {});
  });
  workers_[0]->wait_for_calls(1);
  workers_[1]->wait_for_calls(1);
  workers_[0]->resolve_all(false);
  workers_[1]->resolve_all(true);
  caller.join();
  EXPECT_FALSE(result);
  EXPECT_EQ(workers_[0]->call_count(), 1U);
  EXPECT_EQ(workers_[1]->call_count(), 1U);
}
#endif

struct TransferTopology {
  uint32_t workers_;
  uint32_t dp_;
  uint32_t cp_;
  uint32_t rank_;
  std::vector<uint32_t> expected_;
};

class EngineHostTransferTest
    : public ::testing::TestWithParam<TransferTopology> {};

TEST_P(EngineHostTransferTest, TransfersEveryShardInOnlyTheRequestedDpGroup) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.reserve(topology.workers_);
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_));
  ClientEngine engine(std::move(options), std::move(clients));
  EXPECT_EQ(engine.host_transfer_worker_count(), topology.expected_.size());
  BlockTransferInfo info(/*src_block_id=*/3, /*dst_block_id=*/7);
  info.transfer_type = TransferType::D2H2G;
  info.block_type = BlockType::C4;
  std::fill(std::begin(info.hash_key), std::end(info.hash_key), 42);
  auto futures = engine.transfer_kv_blocks(topology.rank_, {info});
  EXPECT_EQ(futures.size(), topology.expected_.size());
  for (auto& future : futures) {
    EXPECT_EQ(std::move(future).get(), 1U);
  }
  info.transfer_type = TransferType::H2D;
  engine.transfer_kv_blocks(topology.rank_, /*batch_id=*/123, {info});
  std::vector<uint32_t> offloaded;
  std::vector<uint32_t> loaded;
  offloaded.reserve(topology.workers_);
  loaded.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    const auto& worker = *workers[i];
    EXPECT_LE(worker.offload_count_, 1U);
    EXPECT_LE(worker.load_count_, 1U);
    if (worker.offload_count_ != 0) {
      offloaded.emplace_back(i);
      ASSERT_EQ(worker.offload_infos_.size(), 1U);
      EXPECT_EQ(worker.offload_infos_[0].src_block_id, 3);
      EXPECT_EQ(worker.offload_infos_[0].dst_block_id, 7);
      EXPECT_EQ(worker.offload_infos_[0].transfer_type, TransferType::D2H2G);
    }
    if (worker.load_count_ == 0) {
      continue;
    }
    loaded.emplace_back(i);
    EXPECT_EQ(worker.load_batch_id_, 123U);
    ASSERT_EQ(worker.load_infos_.size(), 1U);
    EXPECT_EQ(worker.load_infos_[0].src_block_id, 3);
    EXPECT_EQ(worker.load_infos_[0].dst_block_id, 7);
    EXPECT_EQ(worker.load_infos_[0].block_type, BlockType::C4);
    EXPECT_EQ(worker.load_infos_[0].transfer_type, TransferType::H2D);
    EXPECT_TRUE(std::equal(std::begin(info.hash_key),
                           std::end(info.hash_key),
                           std::begin(worker.load_infos_[0].hash_key)));
  }
  EXPECT_EQ(offloaded, topology.expected_);
  EXPECT_EQ(loaded, topology.expected_);
}

TEST_P(EngineHostTransferTest, PrefetchWaitsForEveryShardAndStopsAtFirstMiss) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  clients.reserve(topology.workers_);
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_))
      .prefetch_timeout(0)
      .prefetch_batch_size(5);
  ClientEngine engine(std::move(options), std::move(clients));
  auto request = std::make_shared<StoragePrefetchRequest>();
  for (int32_t i = 0; i < 5; ++i) {
    BlockTransferInfo info(/*src_block_id=*/i, /*dst_block_id=*/i);
    info.transfer_type = TransferType::G2H;
    PrefetchUnit unit;
    unit.gated_blocks.emplace_back(info);
    request->units.emplace_back(std::move(unit));
  }
  size_t callbacks = 0;
  size_t common_prefix = 0;
  engine.prefetch_from_storage(
      topology.rank_,
      request,
      [] { return false; },
      [&](PrefetchSummary summary) {
        ++callbacks;
        common_prefix = 0;
        while (common_prefix < summary.gated_hits.size() &&
               summary.gated_hits[common_prefix] != 0) {
          ++common_prefix;
        }
      });
  std::vector<uint32_t> prefetched;
  prefetched.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    if (workers[i]->prefetch_request_ != nullptr) {
      prefetched.emplace_back(i);
    }
  }
  ASSERT_EQ(prefetched, topology.expected_);
  for (size_t i = 0; i < topology.expected_.size(); ++i) {
    const auto& worker = *workers[topology.expected_[i]];
    EXPECT_EQ(worker.prefetch_request_, request);
    EXPECT_EQ(worker.prefetch_index_, i);
    EXPECT_EQ(worker.prefetch_result_->worker_count(),
              topology.expected_.size());
    EXPECT_EQ(callbacks, 0U);
    const bool last = i + 1 == topology.expected_.size();
    const std::vector<uint8_t> gated_hits =
        last ? std::vector<uint8_t>{1, 1, 1, 0, 1}
             : std::vector<uint8_t>{1, 1, 1, 1, 1};
    const std::vector<uint8_t> non_gated_hits(5, 1);
    EXPECT_EQ(worker.prefetch_result_->record_batch_result(
                  i, gated_hits, non_gated_hits),
              PrefetchControl::STOP);
    EXPECT_EQ(callbacks, 0U);
    worker.prefetch_result_->mark_worker_ended(i, /*worker_ok=*/true);
  }
  EXPECT_EQ(callbacks, 1U);
  EXPECT_EQ(common_prefix, 3U);
}

INSTANTIATE_TEST_SUITE_P(
    DpCp,
    EngineHostTransferTest,
    ::testing::Values(
        TransferTopology{8, 1, 1, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{8, 1, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 1, 1, {8, 9, 10, 11, 12, 13, 14, 15}},
        TransferTopology{16, 2, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 4, 1, {8, 9, 10, 11, 12, 13, 14, 15}}));

// Overlap batches borrow the sequence cache. Serial worker execution does not
// require additional Block owners in the batch.

class StepCaptureWorker final : public WorkerClient {
 public:
  folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const ForwardInput& inputs) override {
    ++step_count_;
    captured_token_ids_ = inputs.token_ids.clone();
    RawForwardOutput output;
    RawSampleOutput sample;
    RawToken token;
    token.id = 5;
    sample.tokens.emplace_back(std::move(token));
    output.outputs.emplace_back(std::move(sample));
    return folly::makeSemiFuture(
        std::optional<RawForwardOutput>(std::move(output)));
  }

  uint32_t step_count() const { return step_count_; }
  const torch::Tensor& captured_token_ids() const {
    return captured_token_ids_;
  }

 private:
  uint32_t step_count_ = 0;
  torch::Tensor captured_token_ids_;
};

std::shared_ptr<Request> make_overlap_step_request() {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(32);
  stopping_checker.set_max_context_len(30000);
  stopping_checker.set_ignore_eos(true);
  RequestState req_state("x",
                         prompt,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt.size() + 30000,
                         /*n=*/1,
                         /*best_of=*/1,
                         /*logprobs=*/false,
                         /*stream=*/false,
                         /*echo=*/false,
                         /*skip_special_tokens=*/false,
                         /*enable_schedule_overlap=*/false,
                         /*output_func=*/nullptr,
                         /*outputs_func=*/nullptr);
  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

TEST(LLMEngineOverlapOwnershipTest, MtpOverlapStepDoesNotCopyCpPrefillBlocks) {
  // Two workers form one DP group with cp_size=2: the engine must dispatch
  // the same forward_inputs[dp_rank] to both CP ranks (llm_engine.cpp: "Engine
  // sends full global tokens; model-side CP shards inside the worker").
  std::vector<std::shared_ptr<StepCaptureWorker>> workers;
  std::vector<std::shared_ptr<WorkerClient>> clients;
  workers.reserve(2);
  clients.reserve(2);
  for (int32_t worker_rank = 0; worker_rank < 2; ++worker_rank) {
    auto worker = std::make_shared<StepCaptureWorker>();
    workers.emplace_back(worker);
    clients.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(2).dp_size(1).cp_size(2).enable_schedule_overlap(true);
  options.speculative_algorithm("mtp");
  ClientEngine engine(std::move(options), std::move(clients));

  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 32;
  pool_options.block_size_ = 4;
  pool_options.max_seqs_per_batch_ = 1024;
  BlockManagerPool pool(pool_options, /*dp_size=*/1);
  auto request = make_overlap_step_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(pool.try_allocate(sequence));
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  // A prefill batch processes every prompt token: no KV tokens yet.
  sequence->kv_state().set_kv_cache_tokens_num(0);

  std::vector<Batch> batches;
  batches.emplace_back(sequence);
  engine.step(batches);

  // Both CP ranks ran the same batch with byte-identical token inputs.
  for (const auto& worker : workers) {
    ASSERT_EQ(worker->step_count(), 1u);
  }
  EXPECT_TRUE(torch::equal(workers[0]->captured_token_ids(),
                           workers[1]->captured_token_ids()));

  // Serial worker execution needs only the sequence's own reference.
  const auto kv_blocks = sequence->kv_state().blocks(BlockType::KV);
  ASSERT_FALSE(kv_blocks.empty());
  EXPECT_EQ(kv_blocks.front().ref_count(), 1u);
}

TEST(LLMEngineOverlapOwnershipTest,
     DFlash2OverlapStepDoesNotCopyCpPrefillBlocks) {
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.reserve(2);
  for (int32_t worker_rank = 0; worker_rank < 2; ++worker_rank) {
    clients.emplace_back(std::make_shared<StepCaptureWorker>());
  }
  runtime::Options options;
  options.world_size(2).dp_size(1).cp_size(2).enable_schedule_overlap(true);
  options.speculative_algorithm("DFlash2");
  ClientEngine engine(std::move(options), std::move(clients));

  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 32;
  pool_options.block_size_ = 4;
  pool_options.max_seqs_per_batch_ = 1024;
  BlockManagerPool pool(pool_options, /*dp_size=*/1);
  auto request = make_overlap_step_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(pool.try_allocate(sequence));
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  sequence->kv_state().set_kv_cache_tokens_num(0);

  std::vector<Batch> batches;
  batches.emplace_back(sequence);
  engine.step(batches);

  // DFlash2 follows the same ownership rule: Batch does not copy Blocks.
  const auto kv_blocks = sequence->kv_state().blocks(BlockType::KV);
  ASSERT_FALSE(kv_blocks.empty());
  EXPECT_EQ(kv_blocks.front().ref_count(), 1u);
  // Dropping the batch preserves the sequence's owner.
  batches.clear();
  EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().ref_count(), 1u);
}

}  // namespace
}  // namespace xllm
