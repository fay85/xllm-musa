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

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "core/distributed_runtime/engine_utils.h"
#include "core/distributed_runtime/llm_engine.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/platform/platform.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/config/execution_config.h"
#include "framework/request/request.h"
#include "framework/request/stopping_checker.h"

namespace xllm {
namespace {

TEST(DraftEngineCacheTest, WorkerRanksBuildWithoutControllerClients) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        runtime::Options target;
        target.node_rank(1)
            .dp_size(1)
            .cp_size(8)
            .tp_size(1)
            .speculative_algorithm("DFlash2");
        const runtime::Options draft =
            build_draft_engine_options(target, /*world_size=*/0);
        std::_Exit(draft.node_rank() == 1 && draft.cp_size() == 8 &&
                           draft.is_draft_engine()
                       ? 0
                       : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(DraftEngineCacheTest, CpPrefillBudgetMatchesDraftTensorParallelCache) {
  if (!Platform::uses_cp_independent_tp_for_block_diffusion_draft()) {
    GTEST_SKIP() << "Requires CP-independent draft tensor parallelism";
  }
  runtime::Options target;
  target.dp_size(1).cp_size(8).tp_size(1).speculative_algorithm("DFlash2");
  const runtime::Options draft =
      build_draft_engine_options(target, /*world_size=*/8);
  ModelArgs args;
  args.model_type("qwen3").n_layers(5).n_heads(32).n_kv_heads(8).head_dim(128);
  KVCacheEstimateOptions estimate;
  estimate.dtype = torch::kBFloat16;
  estimate.cache_size_in_bytes = 409600;
  estimate.block_size = 16;
  estimate.world_size = draft.world_size() / draft.dp_size() / draft.cp_size();
  estimate.n_local_kv_heads =
      std::max<int64_t>(1, args.n_kv_heads().value() / estimate.world_size);
  estimate.is_draft_engine = draft.is_draft_engine();
  const KVCacheCapacity capacity = estimate_kv_cache_capacity(args, estimate);

  // Five BF16 layers, K+V, 16 tokens and one local 128-wide KV head:
  // each TP8 rank pays 40,960 bytes per block, so this budget fits ten.
  EXPECT_EQ(capacity.slot_size(), 512);
  EXPECT_EQ(capacity.n_blocks(), 10);
  const KVCacheShape worker_shape(capacity, args, /*world_size=*/8);
  EXPECT_EQ(worker_shape.key_cache_shape(),
            (std::vector<int64_t>{10, 1, 16, 128}));
  EXPECT_EQ(worker_shape.value_cache_shape(),
            (std::vector<int64_t>{10, 1, 16, 128}));
  EXPECT_EQ(target.cp_size(), 8);
  EXPECT_EQ(target.tp_size(), 1);
}

TEST(DraftEngineCacheTest, CpDraftRetainsDataParallelOwnership) {
  if (!Platform::uses_cp_independent_tp_for_block_diffusion_draft()) {
    GTEST_SKIP() << "Requires CP-independent draft tensor parallelism";
  }
  runtime::Options target;
  target.world_size(16).dp_size(2).cp_size(4).tp_size(2).speculative_algorithm(
      "DFlash2");
  const runtime::Options draft =
      build_draft_engine_options(target, /*world_size=*/16);
  EXPECT_EQ(draft.world_size(), 16);
  EXPECT_EQ(draft.dp_size(), 2);
  EXPECT_EQ(draft.cp_size(), 1);
  EXPECT_EQ(draft.tp_size(), 8);
}

TEST(DraftEngineCacheTest, PreservesDecodeAndNonBlockDiffusionLayouts) {
  runtime::Options target;
  target.world_size(8).dp_size(1).cp_size(1).tp_size(8).speculative_algorithm(
      "DFlash2");
  const runtime::Options decode =
      build_draft_engine_options(target, /*world_size=*/8);
  EXPECT_EQ(decode.cp_size(), 1);
  EXPECT_EQ(decode.tp_size(), 8);

  target.cp_size(8).tp_size(1).speculative_algorithm("MTP");
  const runtime::Options mtp =
      build_draft_engine_options(target, /*world_size=*/8);
  EXPECT_EQ(mtp.cp_size(), 8);
  EXPECT_EQ(mtp.tp_size(), 1);
}

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

class RestoreAckWorker final : public WorkerClient {
 public:
  explicit RestoreAckWorker(std::future<void> ack = {})
      : ack_(std::move(ack)) {}

  void transfer_kv_blocks(
      uint64_t /*unused*/,
      const std::vector<BlockTransferInfo>& /*unused*/) override {
    entered.set_value();
    if (ack_.valid()) {
      ack_.get();
    }
  }

  std::promise<void> entered;

 private:
  std::future<void> ack_;
};

TEST(EngineHostRestoreTest, SubmitsInParallelAndWaitsForEveryAck) {
  using namespace std::chrono_literals;
  std::promise<void> slow_ack;
  auto slow_worker = std::make_shared<RestoreAckWorker>(slow_ack.get_future());
  auto fast_worker = std::make_shared<RestoreAckWorker>();
  runtime::Options options;
  options.world_size(2).dp_size(1).cp_size(1).host_blocks_factor(2.0);
  ClientEngine engine(std::move(options), {slow_worker, fast_worker});
  BlockTransferInfo info(/*src_block_id=*/3, /*dst_block_id=*/7);
  info.transfer_type = TransferType::H2D;
  auto registration = std::async(std::launch::async, [&] {
    engine.transfer_kv_blocks(/*dp_rank=*/0, /*batch_id=*/123, {info});
  });
  auto release_ack = folly::makeGuard([&] { slow_ack.set_value(); });

  ASSERT_EQ(slow_worker->entered.get_future().wait_for(5s),
            std::future_status::ready);
  ASSERT_EQ(fast_worker->entered.get_future().wait_for(5s),
            std::future_status::ready);
  EXPECT_EQ(registration.wait_for(20ms), std::future_status::timeout);

  slow_ack.set_value();
  release_ack.dismiss();
  EXPECT_EQ(registration.wait_for(5s), std::future_status::ready);
  EXPECT_NO_THROW(registration.get());
}

TEST(EngineHostRestoreTest, RegistrationWaitHasItsOwnDeadline) {
  using namespace std::chrono_literals;
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        ExecutionConfig::get_instance().forward_execution_timeout_ms(5000);
        auto worker = std::make_shared<RestoreAckWorker>(std::async(
            std::launch::async, [] { std::this_thread::sleep_for(2s); }));
        runtime::Options options;
        options.world_size(1).dp_size(1).cp_size(1).host_blocks_factor(2.0);
        ClientEngine engine(std::move(options), {worker});
        BlockTransferInfo info(/*src_block_id=*/3, /*dst_block_id=*/7);
        info.transfer_type = TransferType::H2D;
        engine.transfer_kv_blocks(/*dp_rank=*/0, /*batch_id=*/123, {info});
      },
      "H2D registration wait failed for batch_id=123");
}

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
      .cp_size(static_cast<int32_t>(topology.cp_))
      .host_blocks_factor(2.0);
  ClientEngine engine(std::move(options), std::move(clients));
  EXPECT_EQ(engine.host_transfer_worker_count(), topology.expected_.size());
  BlockTransferInfo info(/*src_block_id=*/3, /*dst_block_id=*/7);
  info.transfer_type = TransferType::D2H2G;
  info.block_type = BlockType::KV;
  std::fill(std::begin(info.hash_key), std::end(info.hash_key), 42);
  auto futures = engine.transfer_kv_blocks(topology.rank_, {info});
  EXPECT_EQ(futures.size(), topology.expected_.size());
  for (auto& future : futures) {
    EXPECT_EQ(std::move(future).get(), 1U);
  }
  info.transfer_type = TransferType::H2D;
  engine.transfer_kv_blocks(topology.rank_, /*batch_id=*/123, {info});
  engine.transfer_kv_blocks(topology.rank_, /*batch_id=*/124, {});
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
    EXPECT_EQ(worker.load_infos_[0].block_type, BlockType::KV);
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
               summary.gated_hits[common_prefix] == summary.rank_count) {
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

class StepCaptureWorker final : public WorkerClient {
 public:
  folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const ForwardInput& inputs) override {
    ++step_count_;
    captured_token_ids_ =
        inputs.token_ids.defined() ? inputs.token_ids.clone() : torch::Tensor();
    captured_is_graph_warmup_ = inputs.input_params.meta.is_graph_warmup;
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
  bool captured_is_graph_warmup() const { return captured_is_graph_warmup_; }
  const torch::Tensor& captured_token_ids() const {
    return captured_token_ids_;
  }

 private:
  uint32_t step_count_ = 0;
  bool captured_is_graph_warmup_ = false;
  torch::Tensor captured_token_ids_;
};

std::shared_ptr<Request> make_overlap_step_request(
    bool is_graph_warmup = false) {
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
  req_state.is_graph_warmup = is_graph_warmup;
  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

class EngineWarmupInputTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void TearDown() override {
    ExecutionConfig::get_instance() = original_execution_;
  }

 private:
  ExecutionConfig original_execution_ = ExecutionConfig::get_instance();
};

TEST_P(EngineWarmupInputTest, MarksEmptyDpOnlyForStaticKernelWarmup) {
  const bool static_kernel = std::get<0>(GetParam());
  const bool warmup = std::get<1>(GetParam());
  ExecutionConfig::get_instance().enable_static_kernel() = static_kernel;
  ExecutionConfig::get_instance().enable_graph() = true;
  auto active = std::make_shared<StepCaptureWorker>();
  auto empty = std::make_shared<StepCaptureWorker>();
  runtime::Options options;
  options.world_size(2).dp_size(2).cp_size(1);
  ClientEngine engine(std::move(options), {active, empty});
  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 32;
  pool_options.block_size_ = 4;
  pool_options.max_seqs_per_batch_ = 4;
  BlockManagerPool pool(pool_options, /*dp_size=*/2);
  auto request = make_overlap_step_request(warmup);
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(pool.try_allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(0);
  std::vector<Batch> batches(2);
  batches[0] = Batch(sequence);
  engine.step(batches);
  EXPECT_EQ(active->step_count(), 1U);
  EXPECT_EQ(empty->step_count(), 1U);
  EXPECT_EQ(active->captured_is_graph_warmup(), warmup);
  EXPECT_EQ(empty->captured_is_graph_warmup(), static_kernel && warmup);
  EXPECT_TRUE(active->captured_token_ids().defined());
  EXPECT_FALSE(empty->captured_token_ids().defined());
}

INSTANTIATE_TEST_SUITE_P(StaticKernelAndWarmup,
                         EngineWarmupInputTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

TEST(LLMEngineOverlapOwnershipTest, CpDispatchPreservesBatchPinsUntilRelease) {
  // CP ranks receive global tokens; the batch retains its inflight pins.
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
  const auto kv_blocks = sequence->kv_state().blocks(BlockType::KV);
  ASSERT_FALSE(kv_blocks.empty());
  EXPECT_EQ(kv_blocks.front().ref_count(), 2u);
  engine.step(batches);

  // Both CP ranks ran the same batch with byte-identical token inputs.
  for (const auto& worker : workers) {
    ASSERT_EQ(worker->step_count(), 1u);
    EXPECT_TRUE(torch::equal(worker->captured_token_ids(),
                             torch::tensor({1, 2, 3, 4}, torch::kInt32)));
  }

  EXPECT_EQ(kv_blocks.front().ref_count(), 2u);
  batches.clear();
  EXPECT_EQ(kv_blocks.front().ref_count(), 1u);
}

}  // namespace
}  // namespace xllm
