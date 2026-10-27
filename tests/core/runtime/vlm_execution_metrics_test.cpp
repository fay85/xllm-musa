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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/common/device_monitor.h"
#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/multimodal_config.h"
#include "core/framework/model/causal_vlm.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/runtime/embed_vlm_worker_impl.h"
#include "core/runtime/mm_embed_vlm_worker_impl.h"
#include "core/runtime/vlm_executor_impl.h"
#include "core/runtime/vlm_worker_impl.h"

namespace xllm {
namespace {

constexpr int64_t kPreviousActivation = 13;
constexpr int64_t kForwardActivation = 4096;

struct MetricsSnapshot {
  double eager = COUNTER_VALUE(num_model_execution_total_eager);
  double model = COUNTER_VALUE(execution_latency_seconds_model);
  double sampling = COUNTER_VALUE(execution_latency_seconds_sampling);
};

void expect_metrics(const MetricsSnapshot& before, double eager, bool model) {
  const MetricsSnapshot after;
  EXPECT_DOUBLE_EQ(after.eager - before.eager, eager);
  if (model) {
    EXPECT_GT(after.model, before.model);
  } else {
    EXPECT_DOUBLE_EQ(after.model, before.model);
  }
}

torch::Tensor expected_hidden() { return torch::tensor({{0.0f, 2.0f, 1.0f}}); }

class MetricsVlm final : public CausalVLM {
 public:
  explicit MetricsVlm(const torch::Device& device)
      : options_(torch::TensorOptions().device(device).dtype(torch::kFloat)) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& params) override {
    ++forward_calls_;
    sampling_at_forward_ = COUNTER_VALUE(execution_latency_seconds_sampling);
    if (!device().is_cpu()) {
      DeviceMonitor::get_instance().monitor_buffer(device().index(),
                                                   kForwardActivation);
    }
    return ModelOutput(get_input_embeddings(tokens, params));
  }

  MMDict encode(const ModelInputParams& params) override {
    ++encode_calls_;
    const auto pixels =
        params.multimodal.mm_data.get<torch::Tensor>("pixel_values");
    if (!pixels.has_value()) {
      return {};
    }
    return {{"image|embedding", pixels.value().split(/*split_size=*/1)}};
  }
  torch::Tensor get_input_embeddings(const torch::Tensor& tokens,
                                     const ModelInputParams& params) override {
    const auto embedding =
        params.multimodal.mm_data.get<torch::Tensor>("image|embedding");
    if (embedding.has_value()) {
      return embedding.value();
    }
    return expected_hidden().to(options_) +
           (tokens.to(options_) - 2).unsqueeze(/*dim=*/1);
  }
  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    return hidden.index_select(/*dim=*/0, selected);
  }
  torch::Tensor pooler(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    ++pooler_calls_;
    return CausalLM::pooler(hidden, selected);
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  int32_t forward_calls() const { return forward_calls_; }
  int32_t encode_calls() const { return encode_calls_; }
  int32_t pooler_calls() const { return pooler_calls_; }
  double sampling_at_forward() const { return sampling_at_forward_; }

 private:
  torch::TensorOptions options_;
  int32_t forward_calls_ = 0;
  int32_t encode_calls_ = 0;
  int32_t pooler_calls_ = 0;
  double sampling_at_forward_ = 0;
};

runtime::Options worker_options() {
  runtime::Options options;
  options.backend("vlm").enable_graph(false).enable_schedule_overlap(false);
  return options;
}

template <typename Worker>
class MetricsWorker final : public Worker {
 public:
  MetricsWorker(const torch::Device& device, int32_t rank)
      : Worker(ParallelArgs(rank, /*world_size=*/2, nullptr),
               device,
               worker_options()) {
    this->dtype_ = torch::kFloat;
    this->model_ = std::make_unique<MetricsVlm>(device);
    this->context_ = ModelContext(this->parallel_args_,
                                  ModelArgs(),
                                  QuantArgs(),
                                  this->model_->options(),
                                  /*context=*/nullptr);
    this->model_executor_ = std::make_unique<Executor>(
        this->model_.get(), ModelArgs(), device, this->options_);
  }
  const MetricsVlm& model() const {
    return static_cast<const MetricsVlm&>(*this->model_);
  }
};

ForwardInput make_input(const torch::Device& device) {
  ForwardInput input;
  input.token_ids = torch::tensor({2}, torch::kLong).to(device);
  input.positions = torch::tensor({0}, torch::kLong).to(device);
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.attention.host.q_seq_lens = {1};
  input.input_params.attention.host.kv_seq_lens = {1};
  input.sampling_params.selected_token_idxes = input.positions;
  input.sampling_params.sample_idxes = input.positions;
  input.sampling_params.all_greedy_sample = true;
  input.sampling_params.do_sample = torch::tensor({false}).to(device);
  return input;
}

ForwardInput make_mm_input(const torch::Device& device,
                           bool preembedded = false) {
  ForwardInput input = make_input(device);
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  MMDataItem item(MMType::IMAGE);
  item.add(preembedded ? "image|embedding" : "pixel_values",
           expected_hidden().to(device));
  auto& state = item.mutable_state();
  state.mutable_token_pos() = {0, 1};
  state.mutable_mm_token_mask() = torch::tensor({true});
  state.mutable_mm_token_num() = 1;
  MMData data(MMType::IMAGE, MMItemVec{item});
  UpdateMMItemScheduleStateVisitor visitor(
      /*computed_token_num=*/0, /*q_seq_len=*/1, /*seq_idx=*/0);
  EXPECT_TRUE(data.foreach (visitor));
  input.input_params.multimodal.mm_data =
      MMBatchData({MMData(visitor.scheduled_type_, visitor.mm_data_items_)});
  return input;
}

class VlmExecutionMetricsTest : public ::testing::Test {
 protected:
  VlmExecutionMetricsTest() {
    ExecutionConfig::get_instance().enable_graph(false);
    LoadConfig::get_instance().enable_manual_loader(false);
    ModelConfig::get_instance().model_impl("auto").capture_hidden_state_layer(
        -1);
    MultimodalConfig::get_instance().enable_return_mm_full_embeddings(false);
  }
  ~VlmExecutionMetricsTest() override {
    ExecutionConfig::get_instance() = execution_;
    LoadConfig::get_instance() = load_;
    ModelConfig::get_instance() = model_;
    MultimodalConfig::get_instance() = multimodal_;
  }

 private:
  ExecutionConfig execution_ = ExecutionConfig::get_instance();
  LoadConfig load_ = LoadConfig::get_instance();
  ModelConfig model_ = ModelConfig::get_instance();
  MultimodalConfig multimodal_ = MultimodalConfig::get_instance();
};

TEST_F(VlmExecutionMetricsTest, DecodeAndPrefillCountOnce) {
  const torch::Device cpu(torch::kCPU);
  MetricsVlm model(cpu);
  VlmExecutorImpl executor(&model, ModelArgs(), cpu, worker_options());
  std::vector<KVCache> caches;
  for (BatchForwardType type :
       {BatchForwardType::DECODE, BatchForwardType::PREFILL}) {
    SCOPED_TRACE(type.to_string());
    ForwardInput input = make_input(cpu);
    input.input_params.meta.batch_forward_type = type;
    const MetricsSnapshot before;
    auto output = executor.run(
        input.token_ids, input.positions, caches, input.input_params);
    expect_metrics(before, /*eager=*/1, /*model=*/false);
    EXPECT_TRUE(torch::equal(output.hidden_states, expected_hidden()));
    EXPECT_EQ(input.input_params.embedding.input_embedding.defined(),
              type.is_prefill());
  }
  EXPECT_EQ(model.forward_calls(), 2);
}

TEST_F(VlmExecutionMetricsTest, MultimodalInputsPreserveEmbeddings) {
  const torch::Device cpu(torch::kCPU);
  MetricsVlm model(cpu);
  VlmExecutorImpl executor(&model, ModelArgs(), cpu, worker_options());
  std::vector<KVCache> caches;
  for (bool preembedded : {false, true}) {
    SCOPED_TRACE(preembedded);
    ForwardInput input = make_mm_input(cpu, preembedded);
    auto output = executor.run(
        input.token_ids, input.positions, caches, input.input_params);
    EXPECT_TRUE(torch::equal(output.hidden_states, expected_hidden()));
    EXPECT_TRUE(torch::equal(input.input_params.embedding.input_embedding,
                             expected_hidden()));
  }
  EXPECT_EQ(model.encode_calls(), 2);
  EXPECT_EQ(model.forward_calls(), 2);
}

TEST_F(VlmExecutionMetricsTest, MultimodalWarmupExecutesModel) {
  const torch::Device cpu(torch::kCPU);
  MetricsVlm model(cpu);
  VlmExecutorImpl executor(&model, ModelArgs(), cpu, worker_options());
  std::vector<KVCache> caches;
  ForwardInput input = make_mm_input(cpu);
  input.input_params.meta.is_graph_warmup = true;
  auto output = executor.run(
      input.token_ids, input.positions, caches, input.input_params);
  EXPECT_EQ(model.encode_calls(), 1);
  EXPECT_EQ(model.forward_calls(), 1);
  EXPECT_TRUE(torch::equal(output.hidden_states, expected_hidden()));
}

class NpuVlmExecutionMetricsTest : public VlmExecutionMetricsTest {
 protected:
  void SetUp() override {
    device_.set_device();
    device_.init_device_context();
    auto& monitor = DeviceMonitor::get_instance();
    monitor.initialize({device_.unwrap()});
    monitor.monitor_buffer(device_.index(), kPreviousActivation);
    monitor.update_active_activation_memory(device_.index());
  }
  void TearDown() override { device_.synchronize_default_stream(); }
  int64_t activation() const {
    return DeviceMonitor::get_instance()
        .get_device_stats(device_.index())
        .active_activation_memory;
  }
  Device device_{/*index=*/0};
};

TEST_F(NpuVlmExecutionMetricsTest, SyncCountsAndRefreshesMemory) {
  MetricsWorker<VLMWorkerImpl> worker(device_.unwrap(), /*rank=*/0);
  const ForwardInput input = make_input(device_.unwrap());
  const MetricsSnapshot before;
  auto output = worker.step(input);
  ASSERT_TRUE(output.has_value());
  expect_metrics(before, /*eager=*/1, /*model=*/true);
  EXPECT_EQ(activation(), kForwardActivation);
  EXPECT_TRUE(torch::equal(output->logits.cpu(), expected_hidden()));
  EXPECT_TRUE(torch::equal(output->sample_output.next_tokens.cpu(),
                           torch::tensor({1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(output->do_sample.cpu(), torch::tensor({false})));
}

TEST_F(NpuVlmExecutionMetricsTest, LogitsOnlyReturnsLogitsAndCountsModel) {
  MetricsWorker<VLMWorkerImpl> worker(device_.unwrap(), /*rank=*/0);
  ForwardInput input = make_input(device_.unwrap());
  input.skip_sampling_for_logits_only = true;
  const MetricsSnapshot before;
  auto output = worker.step(input);
  ASSERT_TRUE(output.has_value());
  expect_metrics(before, /*eager=*/1, /*model=*/true);
  EXPECT_TRUE(torch::equal(output->logits.cpu(), expected_hidden()));
  EXPECT_FALSE(output->sample_output.next_tokens.defined());
}

TEST_F(NpuVlmExecutionMetricsTest, EmptyShardDoesNotExecuteOrChangeMetrics) {
  MetricsWorker<VLMWorkerImpl> worker(device_.unwrap(), /*rank=*/0);
  const MetricsSnapshot before;
  auto output = worker.step(ForwardInput{});
  ASSERT_TRUE(output.has_value());
  expect_metrics(before, /*eager=*/0, /*model=*/false);
  EXPECT_EQ(worker.model().encode_calls(), 0);
  EXPECT_EQ(worker.model().forward_calls(), 0);
  EXPECT_FALSE(output->logits.defined());
  EXPECT_FALSE(output->sample_output.next_tokens.defined());
  EXPECT_EQ(activation(), kPreviousActivation);
}

TEST_F(NpuVlmExecutionMetricsTest, NoSyncSkipsLatencyAndMemory) {
  MetricsWorker<VLMWorkerImpl> worker(device_.unwrap(), /*rank=*/0);
  const ForwardInput input = make_input(device_.unwrap());
  auto stream = device_.current_stream();
  const MetricsSnapshot before;
  // Call the existing public base hook; the VLM override is private.
  auto output = static_cast<WorkerImpl&>(worker).execute_no_sync_on_stream(
      input, *stream);
  ASSERT_TRUE(output.has_value());
  expect_metrics(before, /*eager=*/1, /*model=*/false);
  EXPECT_EQ(activation(), kPreviousActivation);
  ASSERT_EQ(output->retained_inputs.size(), 1);
  ASSERT_EQ(stream->synchronize(), 0);
  EXPECT_TRUE(torch::equal(output->sample_output.next_tokens.cpu(),
                           torch::tensor({1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(output->logits.cpu(), expected_hidden()));
  EXPECT_EQ(activation(), kPreviousActivation);

  const MetricsSnapshot sync_before;
  ASSERT_TRUE(worker.step(input).has_value());
  expect_metrics(sync_before, /*eager=*/1, /*model=*/true);
  EXPECT_EQ(activation(), kForwardActivation);
}

TEST_F(NpuVlmExecutionMetricsTest, NonDriverSkipsLatencyAndMemory) {
  MetricsWorker<VLMWorkerImpl> worker(device_.unwrap(), /*rank=*/1);
  const ForwardInput input = make_input(device_.unwrap());
  const MetricsSnapshot before;
  EXPECT_FALSE(worker.step(input).has_value());
  expect_metrics(before, /*eager=*/1, /*model=*/false);
  EXPECT_EQ(worker.model().forward_calls(), 1);
  EXPECT_EQ(activation(), kPreviousActivation);
}

TEST_F(NpuVlmExecutionMetricsTest, EmbeddingOutputCountsSampling) {
  MetricsWorker<EmbedVLMWorkerImpl> worker(device_.unwrap(), /*rank=*/0);
  ForwardInput input = make_input(device_.unwrap());
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input.sampling_params.is_embeddings = true;
  const MetricsSnapshot before;
  auto pooled = worker.step(input);
  ASSERT_TRUE(pooled.has_value());
  expect_metrics(before, /*eager=*/1, /*model=*/true);
  EXPECT_GT(COUNTER_VALUE(execution_latency_seconds_sampling), before.sampling);
  EXPECT_DOUBLE_EQ(worker.model().sampling_at_forward(), before.sampling);
  EXPECT_EQ(worker.model().pooler_calls(), 1);
  EXPECT_TRUE(
      torch::equal(pooled->sample_output.embeddings.cpu(), expected_hidden()));
  EXPECT_TRUE(pooled->sample_output.mm_embeddings.empty());

  MultimodalConfig::get_instance().enable_return_mm_full_embeddings(true);
  input.token_ids = torch::tensor({2, 3}, torch::kLong).to(device_.unwrap());
  input.positions = input.positions.repeat({2});
  input.input_params.meta.num_sequences = 2;
  input.input_params.attention.host.q_seq_lens = {1, 1};
  input.input_params.attention.host.kv_seq_lens = {1, 1};
  const MetricsSnapshot full_before;
  auto full = worker.step(input);
  ASSERT_TRUE(full.has_value());
  expect_metrics(full_before, /*eager=*/1, /*model=*/true);
  EXPECT_GT(COUNTER_VALUE(execution_latency_seconds_sampling),
            full_before.sampling);
  EXPECT_EQ(worker.model().pooler_calls(), 1);
  EXPECT_FALSE(full->sample_output.embeddings.defined());
  ASSERT_EQ(full->sample_output.mm_embeddings.size(), 2);
  for (int64_t row = 0; row < 2; ++row) {
    const auto& sequence = full->sample_output.mm_embeddings[row];
    ASSERT_EQ(sequence.size(), 1);
    EXPECT_TRUE(torch::equal(sequence.front().cpu(), expected_hidden() + row));
  }
}

TEST_F(NpuVlmExecutionMetricsTest, EmbeddingBypassSkipsSampling) {
  MetricsWorker<EmbedVLMWorkerImpl> worker(device_.unwrap(), /*rank=*/0);
  for (bool is_embeddings : {false, true}) {
    SCOPED_TRACE(is_embeddings);
    ForwardInput input = make_input(device_.unwrap());
    input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
    input.sampling_params.is_embeddings = is_embeddings;
    if (is_embeddings) {
      input.sampling_params.selected_token_idxes = torch::Tensor();
    }
    const int32_t forward_before = worker.model().forward_calls();
    const MetricsSnapshot before;

    auto output = worker.step(input);

    ASSERT_TRUE(output.has_value());
    expect_metrics(before, /*eager=*/1, /*model=*/true);
    EXPECT_DOUBLE_EQ(COUNTER_VALUE(execution_latency_seconds_sampling),
                     before.sampling);
    EXPECT_EQ(worker.model().forward_calls(), forward_before + 1);
    EXPECT_EQ(worker.model().pooler_calls(), 0);
    EXPECT_FALSE(output->sample_output.embeddings.defined());
    EXPECT_TRUE(output->sample_output.mm_embeddings.empty());
  }
}

TEST_F(NpuVlmExecutionMetricsTest, MultimodalExecutesOnAllRanks) {
  for (int32_t rank : {0, 1}) {
    SCOPED_TRACE(rank);
    MetricsWorker<VLMWorkerImpl> worker(device_.unwrap(), rank);
    const ForwardInput input = make_mm_input(device_.unwrap());
    auto output = worker.step(input);
    EXPECT_EQ(worker.model().encode_calls(), 1);
    EXPECT_EQ(worker.model().forward_calls(), 1);
    EXPECT_EQ(output.has_value(), rank == 0);
    if (output.has_value()) {
      EXPECT_TRUE(torch::equal(output->logits.cpu(), expected_hidden()));
    }
  }
}

TEST_F(NpuVlmExecutionMetricsTest, EmbeddingWorkerPreservesMultimodalOutput) {
  for (int32_t rank : {0, 1}) {
    SCOPED_TRACE(rank);
    MetricsWorker<EmbedVLMWorkerImpl> worker(device_.unwrap(), rank);
    ForwardInput input = make_mm_input(device_.unwrap());
    input.sampling_params.is_embeddings = true;
    const MetricsSnapshot before;
    auto output = worker.step(input);
    EXPECT_EQ(worker.model().encode_calls(), 1);
    EXPECT_EQ(worker.model().pooler_calls(), rank == 0 ? 1 : 0);
    EXPECT_EQ(output.has_value(), rank == 0);
    if (rank == 0) {
      EXPECT_GT(COUNTER_VALUE(execution_latency_seconds_sampling),
                before.sampling);
    } else {
      EXPECT_DOUBLE_EQ(COUNTER_VALUE(execution_latency_seconds_sampling),
                       before.sampling);
    }
    if (output.has_value()) {
      EXPECT_TRUE(torch::equal(output->sample_output.embeddings.cpu(),
                               expected_hidden()));
    }
  }
}

TEST_F(NpuVlmExecutionMetricsTest, MmEmbedOnlyEncodes) {
  for (int32_t rank : {0, 1}) {
    SCOPED_TRACE(rank);
    MetricsWorker<MMEmbedVLMWorkerImpl> worker(device_.unwrap(), rank);
    ForwardInput input = make_mm_input(device_.unwrap());
    input.sampling_params.is_embeddings = true;
    auto output = worker.step(input);
    EXPECT_EQ(worker.model().encode_calls(), 1);
    EXPECT_EQ(worker.model().forward_calls(), 0);
    EXPECT_EQ(output.has_value(), rank == 0);
    if (output.has_value()) {
      ASSERT_EQ(output->sample_output.mm_embeddings.size(), 1u);
      ASSERT_EQ(output->sample_output.mm_embeddings.front().size(), 1u);
      EXPECT_TRUE(torch::equal(
          output->sample_output.mm_embeddings.front().front().cpu(),
          expected_hidden()));
    }
  }
}

TEST_F(NpuVlmExecutionMetricsTest, MmEmbedAlwaysCallsEncode) {
  MetricsWorker<MMEmbedVLMWorkerImpl> worker(device_.unwrap(), /*rank=*/1);
  for (int32_t scenario = 0; scenario < 4; ++scenario) {
    SCOPED_TRACE(scenario);
    ForwardInput input = make_input(device_.unwrap());
    if (scenario == 1) {
      input = make_mm_input(device_.unwrap(), /*preembedded=*/true);
    } else if (scenario == 2) {
      input = make_mm_input(device_.unwrap());
      input.input_params.meta.is_graph_warmup = true;
    } else if (scenario == 3) {
      input.input_params.multimodal.mm_data = MMBatchData(
          MMType::IMAGE,
          MMDict{{"pixel_values", expected_hidden().to(device_.unwrap())}});
    }
    input.sampling_params.is_embeddings = true;
    EXPECT_FALSE(worker.step(input).has_value());
    EXPECT_EQ(worker.model().encode_calls(), scenario + 1);
  }
  EXPECT_EQ(worker.model().encode_calls(), 4);
}

}  // namespace
}  // namespace xllm
