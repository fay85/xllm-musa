/* Copyright 2026 The xLLM Authors.

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

#include <folly/executors/InlineExecutor.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "core/runtime/dflash2_worker_impl.h"
#include "core/runtime/dflash_worker_impl.h"
#include "core/runtime/worker_impl.h"
#include "platform/device.h"
#include "runtime/dspark_worker_impl.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"

namespace xllm {

class DFlashWorkerImplTestAccess final {
 public:
  static WorkerImpl* target_worker(DFlashWorkerImpl& worker) {
    return worker.impl_.get();
  }
};

namespace {

class DFlashWorkerTestAdapter final : public DFlashWorkerImpl {
 public:
  using DFlashWorkerImpl::DFlashWorkerImpl;

  void set_draft_prelaunch_enabled(bool enabled) {
    draft_prelaunch_enabled_for_init_ = enabled;
  }

  void finish_idle_decode(const ForwardInput& input,
                          ForwardOutput& output,
                          std::optional<ForwardOutput>& draft_output) {
    finish_empty_decode(input, output, draft_output);
  }

  size_t idle_retained_input_count() const {
    return idle_decode_output_.retained_inputs.size();
  }

  std::vector<int64_t> pending_rows(const ForwardInput& input) const {
    return build_batch_to_pending_row_mapping(input);
  }

  void install_pending_context(std::shared_ptr<EmbeddingCache> cache,
                               ForwardInput identity,
                               const torch::Tensor& accepted_device,
                               const torch::Tensor& accepted_host,
                               const torch::Tensor& embeddings,
                               const torch::Tensor& base_positions,
                               StreamEventPtr ready) {
    embedding_cache_ = std::move(cache);
    PendingContext pending;
    pending.identity = std::move(identity);
    pending.accepted_device = accepted_device;
    pending.accepted_host = accepted_host;
    pending.embeddings = embeddings;
    pending.base_positions = base_positions;
    pending.d2h_ready = ready;
    pending.context_ready = std::move(ready);
    pending_context_ = std::move(pending);
  }

  void commit_pending_context_for_test() { commit_pending_context(); }
};

class DeviceTailGate final {
 public:
  ~DeviceTailGate() { release(); }

  aclError enqueue(aclrtStream stream) const {
    auto callback_state = std::make_unique<std::shared_ptr<State>>(state_);
    const aclError result = aclrtLaunchHostFunc(
        stream,
        [](void* user_data) {
          std::unique_ptr<std::shared_ptr<State>> callback_state(
              static_cast<std::shared_ptr<State>*>(user_data));
          State& state = **callback_state;
          std::unique_lock<std::mutex> lock(state.mutex);
          state.entered = true;
          state.condition.notify_all();
          state.condition.wait(lock, [&state] { return state.released; });
        },
        callback_state.get());
    if (result == ACL_SUCCESS) {
      callback_state.release();
    }
    return result;
  }

  bool wait_until_entered() const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->condition.wait_for(
        lock, std::chrono::seconds(10), [this] { return state_->entered; });
  }

  void release() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->released = true;
    state_->condition.notify_all();
  }

 private:
  struct State {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool released = false;
  };

  std::shared_ptr<State> state_ = std::make_shared<State>();
};

TEST(DFlash2PrelaunchLifecycleTest, VlmTargetBuildsMultimodalWorker) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .speculative_algorithm("DFlash2")
      .num_speculative_tokens(7);
  DFlash2WorkerImpl worker(
      parallel_args, device.unwrap(), options, WorkerType::VLM);
  WorkerImpl* target = DFlashWorkerImplTestAccess::target_worker(worker);
  ASSERT_NE(target, nullptr);
  EXPECT_NE(dynamic_cast<VLMWorkerImpl*>(target), nullptr);
  EXPECT_EQ(dynamic_cast<LLMWorkerImpl*>(target), nullptr);

  options.speculative_algorithm("DSpark");
  DSparkWorkerImpl dspark_worker(
      parallel_args, device.unwrap(), options, WorkerType::VLM);
  WorkerImpl* dspark_target =
      DFlashWorkerImplTestAccess::target_worker(dspark_worker);
  ASSERT_NE(dspark_target, nullptr);
  EXPECT_NE(dynamic_cast<VLMWorkerImpl*>(dspark_target), nullptr);
  EXPECT_EQ(dynamic_cast<LLMWorkerImpl*>(dspark_target), nullptr);
}

TEST(DFlash2PrelaunchLifecycleTest,
     IdleDecodeRetainsInputsUntilTheNextSubmissionCompletesPriorEvent) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .speculative_algorithm("DFlash2")
      .num_speculative_tokens(5);
  DFlashWorkerTestAdapter worker(
      parallel_args, device.unwrap(), options, WorkerType::LLM);
  worker.set_draft_prelaunch_enabled(true);

  auto gate = std::make_shared<DeviceTailGate>();
  auto compute = device.current_stream();
  {
    auto guard = compute->set_stream_guard();
    ASSERT_EQ(gate->enqueue(compute->get_stream()->stream()), ACL_SUCCESS);
  }
  const bool tail_entered = gate->wait_until_entered();

  auto target_input = std::make_shared<ForwardInput>();
  auto draft_input = std::make_shared<ForwardInput>();
  const std::weak_ptr<ForwardInput> first_target = target_input;
  const std::weak_ptr<ForwardInput> first_draft = draft_input;
  ForwardOutput output;
  output.retained_inputs.emplace_back(std::move(target_input));
  std::optional<ForwardOutput> draft_output = ForwardOutput();
  draft_output->retained_inputs.emplace_back(std::move(draft_input));
  worker.finish_idle_decode(ForwardInput{}, output, draft_output);

  EXPECT_TRUE(tail_entered);
  EXPECT_EQ(output.ready_event, nullptr);
  EXPECT_TRUE(output.retained_inputs.empty());
  ASSERT_TRUE(draft_output.has_value());
  EXPECT_TRUE(draft_output->retained_inputs.empty());
  EXPECT_FALSE(first_target.expired());
  EXPECT_FALSE(first_draft.expired());
  EXPECT_EQ(worker.idle_retained_input_count(), 2);

  auto next_input = std::make_shared<ForwardInput>();
  const std::weak_ptr<ForwardInput> final_input = next_input;
  ForwardOutput next_output;
  next_output.retained_inputs.emplace_back(std::move(next_input));
  draft_output.reset();
  std::promise<void> second_submission_started;
  auto second_submission_started_future =
      second_submission_started.get_future();
  auto second_submission = std::async(
      std::launch::async,
      [&device,
       &worker,
       &next_output,
       &draft_output,
       &second_submission_started]() {
        device.set_device();
        second_submission_started.set_value();
        worker.finish_idle_decode(ForwardInput{}, next_output, draft_output);
      });
  const auto started_status =
      second_submission_started_future.wait_for(std::chrono::seconds(10));
  const auto retirement_status =
      second_submission.wait_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(first_target.expired());
  EXPECT_FALSE(first_draft.expired());
  gate->release();
  second_submission.get();

  EXPECT_EQ(started_status, std::future_status::ready);
  EXPECT_EQ(retirement_status, std::future_status::timeout);
  EXPECT_TRUE(first_target.expired());
  EXPECT_TRUE(first_draft.expired());
  EXPECT_FALSE(final_input.expired());
  EXPECT_EQ(worker.idle_retained_input_count(), 1);
}

TEST(DFlash2PrelaunchLifecycleTest,
     PendingRowsFollowThreeToTwoBatchAndFailClosedOnIdentityMismatch) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .speculative_algorithm("DFlash2")
      .num_speculative_tokens(7);
  DFlashWorkerTestAdapter worker(
      parallel_args, device.unwrap(), options, WorkerType::LLM);

  ForwardInput previous;
  previous.input_params.embedding.embedding_ids = {4, 5, 6};
  previous.input_params.embedding.request_ids = {"first", "second", "third"};
  previous.sample_sequence_ids = {"sequence-0", "sequence-1", "sequence-2"};
  previous.input_params.embedding.linear_state_ids = {40, 50, 60};
  previous.input_params.embedding.linear_state_read_ids = {41, 51, 61};
  auto compute = device.current_stream();
  auto ready = compute->record_event();
  auto cache = std::make_shared<EmbeddingCache>(1, true);
  const auto device_options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kLong);
  const auto host_options =
      torch::TensorOptions().dtype(torch::kLong).pinned_memory(true);
  worker.install_pending_context(cache,
                                 previous,
                                 torch::empty({0}, device_options),
                                 torch::empty({0}, host_options),
                                 torch::empty({0}, device_options),
                                 torch::empty({0}, device_options),
                                 ready);

  ForwardInput current;
  current.input_params.embedding.embedding_ids = {6, 4};
  current.input_params.embedding.request_ids = {"third", "first"};
  current.sample_sequence_ids = {"sequence-2", "sequence-0"};
  current.input_params.embedding.linear_state_ids = {99, 98};
  current.input_params.embedding.linear_state_read_ids = {97, 96};
  EXPECT_EQ(worker.pending_rows(current), (std::vector<int64_t>{2, 0}));

  current.input_params.embedding.embedding_ids[0] = 9;
  EXPECT_TRUE(worker.pending_rows(current).empty());
  current.input_params.embedding.embedding_ids[0] = 6;
  current.sample_sequence_ids[0] = "replacement-sequence";
  EXPECT_TRUE(worker.pending_rows(current).empty());
}

TEST(DFlash2PrelaunchLifecycleTest,
     DestructionRetiresLastContextBeforeBlockReuse) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .speculative_algorithm("DFlash2")
      .num_speculative_tokens(2);
  BlockManager::Options block_options;
  block_options.num_blocks(2).block_size(1).enable_prefix_cache(false);
  BlockManagerImpl manager(block_options);
  auto blocks = manager.allocate(1);
  ASSERT_EQ(blocks.size(), 1u);
  const int32_t block_id = blocks.front().id();
  ASSERT_EQ(manager.num_free_blocks(), 0u);

  auto worker = std::make_unique<DFlashWorkerTestAdapter>(
      parallel_args, device.unwrap(), options, WorkerType::LLM);
  auto compute = device.current_stream();
  const auto device_options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kInt);
  torch::Tensor cache = torch::zeros({2, 1}, device_options);
  auto gate = std::make_shared<DeviceTailGate>();
  StreamEventPtr ready;
  {
    auto guard = compute->set_stream_guard();
    ASSERT_EQ(gate->enqueue(compute->get_stream()->stream()), ACL_SUCCESS);
    cache.select(0, block_id).fill_(11);
    ready = compute->record_event();
  }
  ASSERT_TRUE(gate->wait_until_entered());
  auto embedding_cache = std::make_shared<EmbeddingCache>(1, true);
  worker->install_pending_context(
      embedding_cache,
      ForwardInput{},
      torch::zeros({1}, torch::TensorOptions().device(device.unwrap())),
      torch::zeros({1}, torch::TensorOptions().pinned_memory(true)),
      torch::zeros({1}, torch::TensorOptions().device(device.unwrap())),
      torch::zeros({1}, torch::TensorOptions().device(device.unwrap())),
      ready);

  std::promise<void> destruction_started;
  auto destruction_started_future = destruction_started.get_future();
  auto destruction = std::async(std::launch::async,
                                [&device, &worker, &destruction_started]() {
                                  device.set_device();
                                  destruction_started.set_value();
                                  worker.reset();
                                });
  const auto started_status =
      destruction_started_future.wait_for(std::chrono::seconds(10));
  const auto retirement_status =
      destruction.wait_for(std::chrono::milliseconds(100));
  gate->release();
  destruction.get();
  EXPECT_EQ(started_status, std::future_status::ready);
  EXPECT_EQ(retirement_status, std::future_status::timeout);

  aclrtEventRecordedStatus status;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &status), ACL_SUCCESS);
  EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
  EXPECT_TRUE(cache.select(0, block_id).cpu().eq(11).all().item<bool>());
  EXPECT_EQ(manager.num_free_blocks(), 0u);

  blocks.clear();
  ASSERT_EQ(manager.num_free_blocks(), 1u);
  auto reused_blocks = manager.allocate(1);
  ASSERT_EQ(reused_blocks.size(), 1u);
  EXPECT_EQ(reused_blocks.front().id(), block_id);
  auto reuse_stream = device.get_stream_from_pool();
  {
    auto guard = reuse_stream->set_stream_guard();
    cache.select(0, reused_blocks.front().id()).fill_(23);
  }
  ASSERT_EQ(reuse_stream->synchronize(), 0);
  EXPECT_TRUE(cache.select(0, block_id).cpu().eq(23).all().item<bool>());
}

TEST(DFlash2PrelaunchLifecycleTest, IdleDecodeKeepsSynchronousFallback) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .speculative_algorithm("DFlash2")
      .num_speculative_tokens(5);
  DFlashWorkerTestAdapter worker(
      parallel_args, device.unwrap(), options, WorkerType::LLM);

  for (const auto [prelaunch_enabled, graph_warmup] :
       {std::pair(false, false), std::pair(true, true)}) {
    worker.set_draft_prelaunch_enabled(prelaunch_enabled);
    ForwardInput input;
    input.input_params.meta.is_graph_warmup = graph_warmup;
    ForwardOutput output;
    output.retained_inputs.emplace_back(std::make_shared<ForwardInput>());
    std::optional<ForwardOutput> draft_output = ForwardOutput();
    auto gate = std::make_shared<DeviceTailGate>();
    auto compute = device.current_stream();
    {
      auto guard = compute->set_stream_guard();
      ASSERT_EQ(gate->enqueue(compute->get_stream()->stream()), ACL_SUCCESS);
    }
    const bool tail_entered = gate->wait_until_entered();
    std::promise<void> fallback_started;
    auto fallback_started_future = fallback_started.get_future();
    auto fallback =
        std::async(std::launch::async,
                   [&device,
                    &worker,
                    &input,
                    &output,
                    &draft_output,
                    &fallback_started]() {
                     device.set_device();
                     fallback_started.set_value();
                     worker.finish_idle_decode(input, output, draft_output);
                   });
    const auto started_status =
        fallback_started_future.wait_for(std::chrono::seconds(10));
    const auto fallback_status =
        fallback.wait_for(std::chrono::milliseconds(100));
    gate->release();
    fallback.get();

    EXPECT_TRUE(tail_entered);
    EXPECT_EQ(started_status, std::future_status::ready);
    EXPECT_EQ(fallback_status, std::future_status::timeout);
    EXPECT_EQ(output.ready_event, nullptr);
    EXPECT_EQ(output.retained_inputs.size(), 1);
    EXPECT_EQ(worker.idle_retained_input_count(), 0);
  }
}

TEST(DFlash2PrelaunchLifecycleTest,
     CommitWaitsForDeviceCopyAndRetainsContextUntilDeviceWorkCompletes) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .speculative_algorithm("DFlash2")
      .num_speculative_tokens(2);
  auto worker = std::make_unique<DFlashWorkerTestAdapter>(
      parallel_args, device.unwrap(), options, WorkerType::LLM);
  auto cache = std::make_shared<EmbeddingCache>(1, true);

  ForwardInput identity;
  identity.input_params.embedding.embedding_ids = {0};
  identity.input_params.embedding.request_ids = {"request"};
  const torch::Tensor expected = torch::tensor({{17, 23, -1}}, torch::kLong);
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kLong).pinned_memory(true);
  torch::Tensor accepted_host = torch::full({1, 3}, -99, host_options);
  torch::Tensor accepted_device = expected.to(device.unwrap());
  torch::Tensor embeddings =
      torch::arange(6, torch::kFloat).view({1, 3, 2}).to(device.unwrap());
  torch::Tensor base_positions =
      torch::tensor({101}, torch::kLong).to(device.unwrap());

  auto compute = device.current_stream();
  ASSERT_EQ(compute->synchronize(), 0);
  auto copy_gate = std::make_shared<DeviceTailGate>();
  StreamEventPtr copy_ready;
  {
    auto guard = compute->set_stream_guard();
    ASSERT_EQ(copy_gate->enqueue(compute->get_stream()->stream()), ACL_SUCCESS);
    accepted_host.copy_(accepted_device, /*non_blocking=*/true);
    copy_ready = compute->record_event();
  }
  ASSERT_TRUE(copy_gate->wait_until_entered());
  worker->install_pending_context(cache,
                                  identity,
                                  accepted_device,
                                  accepted_host,
                                  embeddings,
                                  base_positions,
                                  copy_ready);

  std::promise<void> service_wait_enqueued;
  auto service_wait_ready = service_wait_enqueued.get_future();
  auto service_reader = std::async(
      std::launch::async,
      [&device, copy_ready, accepted_host, &service_wait_enqueued]() {
        device.set_device();
        auto service_stream = device.get_stream_from_pool();
        const bool wait_enqueued = service_stream->wait_event(copy_ready);
        service_wait_enqueued.set_value();
        if (!wait_enqueued || service_stream->synchronize() != 0) {
          return torch::Tensor();
        }
        return accepted_host.clone();
      });
  service_wait_ready.wait();
  std::promise<void> commit_started;
  auto commit_started_future = commit_started.get_future();
  auto commit =
      std::async(std::launch::async, [&device, &worker, &commit_started]() {
        device.set_device();
        commit_started.set_value();
        worker->commit_pending_context_for_test();
      });
  const auto commit_started_status =
      commit_started_future.wait_for(std::chrono::seconds(10));
  const auto commit_status = commit.wait_for(std::chrono::milliseconds(100));
  copy_gate->release();
  commit.get();
  const torch::Tensor service_tokens = service_reader.get();

  EXPECT_EQ(commit_started_status, std::future_status::ready);
  EXPECT_EQ(commit_status, std::future_status::timeout);
  EXPECT_TRUE(torch::equal(service_tokens, expected));
  EXPECT_TRUE(torch::equal(accepted_host, expected));
  const auto states = cache->read_decode_states({0}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_EQ(states.front().token_id, 23);
  EXPECT_EQ(states.front().position_offset, 1);
  EXPECT_EQ(states.front().replay_token_ids, (std::vector<int32_t>{17, 23}));
  EXPECT_TRUE(
      torch::equal(states.front().embedding.cpu(), embeddings[0][1].cpu()));
  EXPECT_TRUE(torch::equal(states.front().prev_embedding.cpu(),
                           embeddings[0][0].cpu()));

  auto retirement_gate = std::make_shared<DeviceTailGate>();
  StreamEventPtr device_tail;
  {
    auto guard = compute->set_stream_guard();
    ASSERT_EQ(retirement_gate->enqueue(compute->get_stream()->stream()),
              ACL_SUCCESS);
    device_tail = compute->record_event();
  }
  ASSERT_TRUE(retirement_gate->wait_until_entered());
  std::promise<void> retirement_started;
  auto retirement_started_future = retirement_started.get_future();
  auto retirement =
      std::async(std::launch::async, [&device, &worker, &retirement_started]() {
        device.set_device();
        retirement_started.set_value();
        worker.reset();
      });
  const auto retirement_started_status =
      retirement_started_future.wait_for(std::chrono::seconds(10));
  const auto destruction_status =
      retirement.wait_for(std::chrono::milliseconds(100));
  retirement_gate->release();
  retirement.get();
  EXPECT_EQ(retirement_started_status, std::future_status::ready);
  EXPECT_EQ(destruction_status, std::future_status::timeout);
  aclrtEventRecordedStatus tail_status;
  ASSERT_EQ(aclrtQueryEventStatus(device_tail->npu_event(), &tail_status),
            ACL_SUCCESS);
  EXPECT_EQ(tail_status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
}

class OverlapResultWorker final : public WorkerImpl {
 public:
  OverlapResultWorker(const ParallelArgs& parallel_args,
                      const torch::Device& device,
                      const runtime::Options& options)
      : WorkerImpl(parallel_args, device, options) {}

  bool init_model(ModelContext& context) override { return false; }

  void prepare_work_before_execute(const ForwardInput& input,
                                   ForwardInput& prepared) override {
    prepared = input;
  }

  std::optional<ForwardOutput> step(const ForwardInput& input) override {
    ADD_FAILURE() << "synchronous execution is outside this fixture";
    return std::nullopt;
  }

  bool wait_for_submission(int64_t step_id) const {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (submitted_step_.load(std::memory_order_acquire) != step_id) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
  }

  bool has_stored_output() const { return last_step_output_valid_; }

  bool has_recorded_output() const { return is_recorded_; }

  std::shared_ptr<DeviceTailGate> hold_next_tail() {
    std::lock_guard<std::mutex> lock(gate_mutex_);
    CHECK(next_gate_ == nullptr);
    next_gate_ = std::make_shared<DeviceTailGate>();
    return next_gate_;
  }

 protected:
  std::optional<ForwardOutput> step_for_schedule_overlap(
      ForwardInput& input) override {
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      submitted_step_.store(-1, std::memory_order_release);
      return std::nullopt;
    }
    const int64_t step_id = input.positions_host.item<int64_t>();
    auto guard = compute_stream_->set_stream_guard();
    const torch::TensorOptions token_options =
        torch::TensorOptions().device(device_.unwrap()).dtype(torch::kLong);
    torch::Tensor tokens = torch::full({1, 1}, -777, token_options);
    torch::Tensor host_tokens = torch::full(
        {1, 1},
        -777,
        torch::TensorOptions().dtype(torch::kLong).pinned_memory(true));
    auto retained = std::make_shared<ForwardInput>(input);
    retained->token_ids = tokens;
    retained->token_ids_host = host_tokens;
    std::shared_ptr<DeviceTailGate> gate;
    {
      std::lock_guard<std::mutex> lock(gate_mutex_);
      gate = std::exchange(next_gate_, nullptr);
    }
    if (gate != nullptr) {
      EXPECT_EQ(gate->enqueue(compute_stream_->get_stream()->stream()),
                ACL_SUCCESS);
    }
    tokens.fill_(step_id);
    host_tokens.copy_(tokens, true);
    ForwardOutput output;
    output.sample_output.next_tokens = std::move(tokens);
    output.retained_inputs.push_back(std::move(retained));
    output.ready_event = compute_stream_->record_event();
    output.requires_host_retirement = true;
    submitted_step_.store(step_id, std::memory_order_release);
    return output;
  }

 private:
  std::mutex gate_mutex_;
  std::shared_ptr<DeviceTailGate> next_gate_;
  std::atomic<int64_t> submitted_step_{-1};
};

ForwardInput make_overlap_producer_input(int64_t step_id) {
  ForwardInput input;
  input.token_ids = torch::tensor({step_id}, torch::kLong);
  input.token_ids_host = input.token_ids;
  input.positions = torch::zeros({1}, torch::kLong);
  input.positions_host = torch::tensor({step_id}, torch::kLong);
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input.input_params.meta.num_sequences = 1;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request-" +
                                              std::to_string(step_id)};
  input.sample_sequence_ids = {"sequence-" + std::to_string(step_id)};
  return input;
}

ForwardInput make_overlap_empty_input() {
  ForwardInput input;
  input.token_ids = torch::empty({0}, torch::kLong);
  input.input_params.meta.batch_forward_type = BatchForwardType::EMPTY;
  return input;
}

class WorkerOverlapResultSlotTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(WorkerOverlapResultSlotTest,
       FullSlotKeepsCurrentStepUntilConsumerDequeues) {
  Device device(GetParam());
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 2, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_speculative_decode(true)
      .num_speculative_tokens(15);
  OverlapResultWorker worker(parallel_args, device.unwrap(), options);
  int32_t full_slot_windows = 0;
  int32_t dequeued_not_ready_windows = 0;
  for (int64_t iteration = 0; iteration < 8; ++iteration) {
    const int64_t step_id = iteration * 2;
    auto gate = worker.hold_next_tail();
    auto current = worker.step_async(make_overlap_producer_input(step_id));
    ASSERT_TRUE(std::move(current).get().has_value());
    ASSERT_TRUE(gate->wait_until_entered());
    auto next = worker.step_async(make_overlap_producer_input(step_id + 1))
                    .via(&folly::InlineExecutor::instance());
    EXPECT_TRUE(worker.wait_for_submission(step_id + 1));
    next.wait(std::chrono::milliseconds(1));
    if (!next.isReady()) {
      ++full_slot_windows;
    }
    EXPECT_FALSE(next.isReady());
    ForwardOutput result = worker.get_last_step_result();
    ASSERT_NE(result.ready_event, nullptr);
    ASSERT_EQ(result.retained_inputs.size(), 1);
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(result.ready_event->npu_event(), &status),
              ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++dequeued_not_ready_windows;
    }
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_NOT_READY);
    EXPECT_EQ(
        result.retained_inputs.front()->input_params.embedding.request_ids,
        std::vector<std::string>{"request-" + std::to_string(step_id)});
    EXPECT_EQ(result.retained_inputs.front()->sample_sequence_ids,
              std::vector<std::string>{"sequence-" + std::to_string(step_id)});
    gate->release();
    EXPECT_TRUE(result.ready_event->synchronize());
    EXPECT_EQ(result.retained_inputs.front()->token_ids_host.item<int64_t>(),
              step_id);
    ASSERT_TRUE(std::move(next).get().has_value());
    ForwardOutput next_result = worker.get_last_step_result();
    ASSERT_NE(next_result.ready_event, nullptr);
    EXPECT_TRUE(next_result.ready_event->synchronize());
    EXPECT_EQ(
        next_result.retained_inputs.front()->token_ids_host.item<int64_t>(),
        step_id + 1);
  }
  EXPECT_EQ(full_slot_windows, 8);
  EXPECT_EQ(dequeued_not_ready_windows, 8);
  ::testing::Test::RecordProperty("full_slot_windows", full_slot_windows);
  ::testing::Test::RecordProperty("dequeued_not_ready_windows",
                                  dequeued_not_ready_windows);
  ::testing::Test::RecordProperty("logical_rank", 0);
  ::testing::Test::RecordProperty("local_device_index", GetParam());
  ::testing::Test::RecordProperty("scope",
                                  "real WorkerImpl publication/dequeue; "
                                  "bounded producer, no Service RPC or model");
}

TEST_P(WorkerOverlapResultSlotTest,
       NonDriverOverwritesAndClearsAfterOwnDeviceTail) {
  Device device(GetParam());
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(1, 2, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_speculative_decode(true)
      .num_speculative_tokens(15);
  OverlapResultWorker worker(parallel_args, device.unwrap(), options);
  auto gate = worker.hold_next_tail();
  auto previous =
      std::move(worker.step_async(make_overlap_producer_input(0))).get();
  ASSERT_TRUE(previous.has_value());
  int32_t overwrite_not_ready_windows = 0;
  int32_t clear_not_ready_windows = 0;
  for (int64_t step_id = 0; step_id < 8; ++step_id) {
    ASSERT_TRUE(gate->wait_until_entered());
    const StreamEventPtr previous_ready = previous->ready_event;
    const std::weak_ptr<ForwardInput> previous_owner =
        previous->retained_inputs.front();
    const torch::Tensor previous_host =
        previous->retained_inputs.front()->token_ids_host;
    previous.reset();
    EXPECT_FALSE(previous_owner.expired());
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++overwrite_not_ready_windows;
    }
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_NOT_READY);
    auto next_gate = worker.hold_next_tail();
    auto next = worker.step_async(make_overlap_producer_input(step_id + 1))
                    .via(&folly::InlineExecutor::instance());
    EXPECT_TRUE(worker.wait_for_submission(step_id + 1));
    next.wait(std::chrono::milliseconds(1));
    EXPECT_FALSE(next.isReady());
    EXPECT_FALSE(previous_owner.expired());
    gate->release();
    previous = std::move(next).get();
    gate = std::move(next_gate);
    ASSERT_TRUE(previous.has_value());
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
    EXPECT_TRUE(previous_owner.expired());
    EXPECT_EQ(previous_host.item<int64_t>(), step_id);
    EXPECT_FALSE(worker.has_recorded_output());
  }
  for (int64_t step_id = 8; step_id < 16; ++step_id) {
    ASSERT_TRUE(gate->wait_until_entered());
    const StreamEventPtr previous_ready = previous->ready_event;
    const std::weak_ptr<ForwardInput> previous_owner =
        previous->retained_inputs.front();
    const torch::Tensor previous_host =
        previous->retained_inputs.front()->token_ids_host;
    previous.reset();
    EXPECT_FALSE(previous_owner.expired());
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++clear_not_ready_windows;
    }
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_NOT_READY);
    auto clearing = worker.step_async(make_overlap_empty_input())
                        .via(&folly::InlineExecutor::instance());
    EXPECT_TRUE(worker.wait_for_submission(-1));
    clearing.wait(std::chrono::milliseconds(1));
    EXPECT_FALSE(clearing.isReady());
    EXPECT_FALSE(previous_owner.expired());
    gate->release();
    const auto cleared = std::move(clearing).get();
    EXPECT_FALSE(cleared.has_value());
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
    EXPECT_TRUE(previous_owner.expired());
    EXPECT_EQ(previous_host.item<int64_t>(), step_id);
    EXPECT_FALSE(worker.has_stored_output());
    EXPECT_FALSE(worker.has_recorded_output());
    if (step_id < 15) {
      gate = worker.hold_next_tail();
      previous =
          std::move(worker.step_async(make_overlap_producer_input(step_id + 1)))
              .get();
      ASSERT_TRUE(previous.has_value());
    }
  }
  EXPECT_EQ(overwrite_not_ready_windows, 8);
  EXPECT_EQ(clear_not_ready_windows, 8);
  ::testing::Test::RecordProperty("overwrite_not_ready_windows",
                                  overwrite_not_ready_windows);
  ::testing::Test::RecordProperty("clear_not_ready_windows",
                                  clear_not_ready_windows);
  ::testing::Test::RecordProperty("logical_rank", 1);
  ::testing::Test::RecordProperty("local_device_index", GetParam());
  ::testing::Test::RecordProperty(
      "scope",
      "real WorkerImpl non-driver replacement/empty retirement; no HCCL or "
      "model");
}

TEST_P(WorkerOverlapResultSlotTest,
       NonDriverIdleTailKeepsOwnershipWithoutAnotherModelStep) {
  Device device(GetParam());
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(1, 2, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_speculative_decode(true)
      .num_speculative_tokens(15);
  OverlapResultWorker worker(parallel_args, device.unwrap(), options);
  int32_t idle_not_ready_windows = 0;
  for (int64_t step_id = 0; step_id < 8; ++step_id) {
    auto gate = worker.hold_next_tail();
    auto output =
        std::move(worker.step_async(make_overlap_producer_input(step_id)))
            .get();
    ASSERT_TRUE(output.has_value());
    ASSERT_TRUE(gate->wait_until_entered());
    const StreamEventPtr ready = output->ready_event;
    const std::weak_ptr<ForwardInput> owner = output->retained_inputs.front();
    const torch::Tensor host = output->retained_inputs.front()->token_ids_host;
    output.reset();
    EXPECT_FALSE(owner.expired());
    EXPECT_TRUE(worker.has_stored_output());
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &status), ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++idle_not_ready_windows;
    }
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_NOT_READY);
    gate->release();
    EXPECT_TRUE(ready->synchronize());
    EXPECT_EQ(host.item<int64_t>(), step_id);
    EXPECT_FALSE(owner.expired());
    const auto cleared =
        std::move(worker.step_async(make_overlap_empty_input())).get();
    EXPECT_FALSE(cleared.has_value());
    EXPECT_TRUE(owner.expired());
    EXPECT_FALSE(worker.has_stored_output());
  }
  EXPECT_EQ(idle_not_ready_windows, 8);
  ::testing::Test::RecordProperty("idle_not_ready_windows",
                                  idle_not_ready_windows);
  ::testing::Test::RecordProperty("logical_rank", 1);
  ::testing::Test::RecordProperty("local_device_index", GetParam());
  ::testing::Test::RecordProperty(
      "scope",
      "idle tail completes without a next model step; retirement uses real "
      "empty control call");
}

INSTANTIATE_TEST_SUITE_P(BothLocalDevices,
                         WorkerOverlapResultSlotTest,
                         ::testing::Values(0, 1));

TEST(DSparkPrelaunchDeviceInputTest, EagerUsesCurrentDeviceAcceptanceAndPages) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  torch::Tensor accepted_host = torch::full({2, 16}, -1, torch::kLong);
  accepted_host[0][0] = 21;
  accepted_host[1].copy_(torch::arange(30, 46, torch::kLong));
  torch::Tensor base_host = torch::tensor({113, 112}, torch::kLong);
  torch::Tensor table_host = torch::tensor({{3, 1}, {0, 2}}, torch::kInt32);
  torch::Tensor accepted = accepted_host.to(device.unwrap());
  torch::Tensor base = base_host.to(device.unwrap());
  torch::Tensor table = table_host.to(device.unwrap());
  auto stream = device.get_stream_from_pool();
  stream->wait_stream(*device.current_stream());
  auto guard = stream->set_stream_guard();
  const auto initial_result =
      dflash_async::prepare_next_draft(accepted, base, table, 999, 128, 15);
  accepted_host[0].copy_(torch::arange(50, 66, torch::kLong));
  accepted_host[1].fill_(-1);
  accepted_host[1][0] = 71;
  base_host = torch::tensor({112, 113}, torch::kLong);
  table_host = torch::tensor({{2, 0}, {1, 3}}, torch::kInt32);
  accepted.copy_(accepted_host.to(device.unwrap()));
  base.copy_(base_host.to(device.unwrap()));
  table.copy_(table_host.to(device.unwrap()));
  const auto result =
      dflash_async::prepare_next_draft(accepted, base, table, 999, 128, 15);
  ASSERT_EQ(stream->synchronize(), 0);
  EXPECT_TRUE(torch::equal(initial_result.anchor_tokens.cpu(),
                           torch::tensor({21, 45}, torch::kLong)));
  const auto expected = dflash_async::prepare_next_draft(
      accepted_host, base_host, table_host, 999, 128, 15);
  EXPECT_TRUE(
      torch::equal(result.accepted_counts.cpu(), expected.accepted_counts));
  EXPECT_TRUE(torch::equal(result.anchor_tokens.cpu(), expected.anchor_tokens));
  EXPECT_TRUE(torch::equal(result.query_tokens.cpu(), expected.query_tokens));
  EXPECT_TRUE(
      torch::equal(result.query_positions.cpu(), expected.query_positions));
  EXPECT_TRUE(torch::equal(result.query_slots.cpu(), expected.query_slots));
  EXPECT_TRUE(torch::equal(result.kv_lengths.cpu(), expected.kv_lengths));
}

TEST(DFlash2PrelaunchTransferTest, NonSendingPushCompletionIsGraphCapturable) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/2, device.unwrap());
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
  parallel_args.tp_group_ = &tp_group;
  auto marker = torch::zeros(
      {1}, torch::TensorOptions().device(device.unwrap()).dtype(torch::kInt));
  auto stream = device.get_stream_from_pool();
  stream->wait_stream(*device.current_stream());
  auto guard = stream->set_stream_guard();
  marker.add_(1);
  stream->synchronize();
  for (const InstanceRole role :
       {InstanceRole::DEFAULT, InstanceRole::DECODE}) {
    KVTransferCompletion completion;
    c10_npu::NPUGraph graph;
    graph.capture_begin(/*pool=*/{0, 0},
                        ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    const auto failures = finalize_kv_push_failures(
        completion, {}, "PUSH", role, parallel_args, device);
    marker.add_(1);
    graph.capture_end();
    EXPECT_TRUE(failures.failed_request_ids.empty());
    EXPECT_TRUE(failures.unsafe_request_ids.empty());
    marker.zero_();
    graph.replay();
    stream->synchronize();
    EXPECT_EQ(marker.cpu().item<int32_t>(), 1);
  }
}

}  // namespace
}  // namespace xllm
