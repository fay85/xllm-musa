/* Copyright 2026 The xLLM Authors.

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

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "framework/model/causal_lm.h"
#include "platform/device.h"
#include "runtime/llm_worker_impl.h"
#if defined(USE_MUSA)
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#endif

namespace xllm {
namespace {

class OutputTestModel final : public CausalLM {
 public:
  OutputTestModel(ModelOutput output,
                  torch::TensorOptions options,
                  std::function<void()> on_forward = {})
      : output_(std::move(output)),
        options_(std::move(options)),
        on_forward_(std::move(on_forward)) {}

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams& params) override {
    received_sampling_ = params.sampling.has_value();
    if (on_forward_) {
      on_forward_();
    }
    return output_;
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& indices) override {
    ++logits_calls_;
    return hidden.index_select(0, indices.to(torch::kLong));
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& indices,
                       torch::Tensor& selected_hidden) override {
    selected_hidden = logits(hidden, indices);
    return selected_hidden;
  }

  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}

  int32_t logits_calls() const { return logits_calls_; }
  bool received_sampling() const { return received_sampling_; }

 private:
  ModelOutput output_;
  torch::TensorOptions options_;
  std::function<void()> on_forward_;
  int32_t logits_calls_ = 0;
  bool received_sampling_ = false;
};

class OutputTestWorker final : public LLMWorkerImpl {
 public:
  OutputTestWorker(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   std::unique_ptr<OutputTestModel> model)
      : LLMWorkerImpl(parallel_args, device, options) {
    ModelArgs args;
    args.model_type("llm_output_test").n_layers(1).hidden_size(4);
    context_ = ModelContext(parallel_args, args, QuantArgs(), model->options());
    model_ = std::move(model);
    model_executor_ =
        std::make_unique<Executor>(model_.get(), args, device, options);
    is_spec_draft_ = options.is_draft_engine();
  }

  std::optional<ForwardOutput> run(const ForwardInput& input,
                                   bool record_ready_event) {
    return execute_no_sync_on_stream(
        input, *compute_stream_, record_ready_event);
  }

#if defined(USE_MUSA)
  void set_kv_transfer(std::shared_ptr<KVCacheTransfer> transfer) {
    kv_cache_transfer_ = std::move(transfer);
  }
#endif
};

class LLMWorkerOutputTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    saved_model_impl_ = ModelConfig::get_instance().model_impl();
    saved_manual_loader_ = LoadConfig::get_instance().enable_manual_loader();
    ModelConfig::get_instance().model_impl("native");
    LoadConfig::get_instance().enable_manual_loader(false);
    device_.set_device();
    device_.init_device_context();
    options_.enable_graph(false)
        .backend("llm")
        .enable_schedule_overlap(true)
        .enable_speculative_decode(true)
        .is_draft_engine(true)
        .num_speculative_tokens(3);
  }

  void TearDown() override {
    EXPECT_EQ(device_.synchronize_default_stream(), 0);
    ModelConfig::get_instance().model_impl(saved_model_impl_);
    LoadConfig::get_instance().enable_manual_loader(saved_manual_loader_);
  }

  ForwardInput make_input() {
    ForwardInput input;
    input.token_ids =
        torch::tensor({1, 0, 0}, torch::kInt).to(device_.unwrap());
    input.positions = torch::arange(3, input.token_ids.options());
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.sampling.emplace();
    input.sampling_params.selected_token_idxes =
        torch::arange(3, input.token_ids.options().dtype(torch::kLong));
    input.return_selected_hidden = true;
    auto stream = device_.current_stream();
    input.metadata_ready_event = stream->record_event();
    return input;
  }

  void expect_retained_input(const ForwardOutput& output,
                             const ForwardInput& input) {
    ASSERT_FALSE(output.retained_inputs.empty());
    EXPECT_EQ(output.retained_inputs.front()->token_ids.data_ptr(),
              input.token_ids.data_ptr());
    EXPECT_TRUE(output.failed_request_ids.empty());
    EXPECT_EQ(output.ready_event != nullptr, GetParam());
    if (output.ready_event != nullptr) {
      EXPECT_TRUE(output.ready_event->synchronize());
    }
  }

  Device device_{0};
  runtime::Options options_;
  std::string saved_model_impl_;
  bool saved_manual_loader_ = false;
};

#if defined(USE_MUSA)
// Use the real synchronizer and completion path, replacing only network I/O.
class WaitingPushTransfer final : public KVCacheTransfer {
 public:
  std::promise<void> started_;
  std::promise<bool> layer_result_;
  std::promise<void> release_;

  void get_cache_info(uint64_t&, std::string&) override {}
  bool link_clusters(const std::vector<uint64_t>&,
                     const std::vector<std::string>&,
                     const std::vector<uint16_t>&) override {
    return false;
  }
  bool unlink_cluster(const uint64_t&,
                      const std::string&,
                      uint16_t,
                      bool) override {
    return false;
  }
  bool pull_kv_blocks(uint64_t,
                      const std::string&,
                      const std::vector<KVTransferMapping>&) override {
    return false;
  }
  std::vector<KVTransferTaskResult> push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>&,
      std::shared_ptr<KVPushSynchronizerImpl>&,
      bool,
      int32_t,
      int32_t) override {
    return {};
  }

  folly::SemiFuture<std::vector<KVTransferTaskResult>> push_kv_blocks_async(
      const std::vector<TransferKVInfo>& infos,
      const ParallelArgs&,
      std::shared_ptr<KVPushSynchronizerImpl> synchronizer,
      bool) override {
    folly::Promise<std::vector<KVTransferTaskResult>> completion;
    auto future = completion.getSemiFuture();
    auto released = release_.get_future();
    threadpool_.schedule([this,
                          synchronizer = std::move(synchronizer),
                          ids = unique_transfer_request_ids(infos),
                          completion = std::move(completion),
                          released = std::move(released)]() mutable {
      started_.set_value();
      const bool ready = synchronizer->synchronize_layer(/*layer_index=*/0);
      layer_result_.set_value(ready);
      // A canceled sender may still own source buffers. Hold its completion
      // until the test verifies that the Worker has not returned yet.
      released.wait();
      KVTransferTaskResult result;
      result.request_ids = std::move(ids);
      result.error_code =
          ready ? KVTransferErrorCode::NONE : KVTransferErrorCode::FAILED;
      completion.setValue(std::vector<KVTransferTaskResult>{std::move(result)});
    });
    return future;
  }
};

class LLMWorkerPushFailureTest : public LLMWorkerOutputTest {};

TEST_P(LLMWorkerPushFailureTest, ForwardFailureCancelsAndDrainsStartedPush) {
  options_.enable_speculative_decode(false)
      .is_draft_engine(false)
      .num_speculative_tokens(0)
      .kv_cache_transfer_mode("PUSH")
      .instance_role(InstanceRole::PREFILL);
  auto transfer = std::make_shared<WaitingPushTransfer>();
  auto started = transfer->started_.get_future();
  auto layer_result = transfer->layer_result_.get_future();
  const bool throw_on_forward = GetParam();
  const auto tensor_options =
      torch::TensorOptions().device(device_.unwrap()).dtype(torch::kFloat);
  auto model =
      std::make_unique<OutputTestModel>(ModelOutput(), tensor_options, [&] {
        EXPECT_EQ(started.wait_for(std::chrono::seconds(5)),
                  std::future_status::ready);
        if (throw_on_forward) {
          throw std::runtime_error("injected forward failure");
        }
      });
  OutputTestModel* model_ptr = model.get();
  const ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, nullptr);
  OutputTestWorker worker(
      parallel_args, device_.unwrap(), options_, std::move(model));
  worker.set_kv_transfer(transfer);
  ForwardInput input = make_input();
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input.transfer_kv_infos = {TransferKVInfo{.request_id = "request-b"},
                             TransferKVInfo{.request_id = "request-a"},
                             TransferKVInfo{.request_id = "request-b"}};
  auto output = std::async(std::launch::async, [&] {
    device_.set_device();
    return worker.run(input, /*record_ready_event=*/false);
  });
  const std::future_status cancellation =
      layer_result.wait_for(std::chrono::seconds(5));
  EXPECT_EQ(cancellation, std::future_status::ready);
  // Release even on regression: don't turn a failed assertion into a hang.
  if (cancellation != std::future_status::ready &&
      started.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    input.input_params.parallel.layer_synchronizer->abort();
  }
  EXPECT_EQ(output.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  transfer->release_.set_value();
  ASSERT_EQ(layer_result.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_FALSE(layer_result.get());
  if (throw_on_forward) {
    try {
      output.get();
      FAIL() << "forward exception was swallowed";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "injected forward failure");
    }
  } else {
    const auto result = output.get();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->failed_request_ids,
              (std::vector<std::string>{"request-a", "request-b"}));
    EXPECT_TRUE(result->unsafe_request_ids.empty());
    EXPECT_FALSE(result->logits.defined());
    EXPECT_FALSE(result->sample_output.next_tokens.defined());
  }
  EXPECT_EQ(model_ptr->logits_calls(), 0);
}

INSTANTIATE_TEST_SUITE_P(EmptyOutputOrException,
                         LLMWorkerPushFailureTest,
                         ::testing::Values(false, true));
#endif

TEST_P(LLMWorkerOutputTest,
       CompletedProposalSkipsLogitsAndKeepsCommonRetirement) {
  const auto tensor_options =
      torch::TensorOptions().device(device_.unwrap()).dtype(torch::kFloat);
  torch::Tensor token_ids =
      torch::tensor({{2, 3, 4}}, torch::kLong).to(device_.unwrap());
  torch::Tensor draft_logits = torch::zeros(/*size=*/{1, 3, 5}, tensor_options);
  torch::Tensor confidence =
      torch::full(/*size=*/{1, 3}, /*fill_value=*/0.75, tensor_options);
  ModelOutput model_output;
  model_output.draft_token_ids = token_ids;
  model_output.draft_logits = draft_logits;
  model_output.draft_confidence_probs = confidence;
  auto model = std::make_unique<OutputTestModel>(std::move(model_output),
                                                 tensor_options);
  OutputTestModel* model_ptr = model.get();
  const ParallelArgs parallel_args(0, 1, nullptr);
  OutputTestWorker worker(
      parallel_args, device_.unwrap(), options_, std::move(model));
  ForwardInput input = make_input();

  std::optional<ForwardOutput> output = worker.run(input, GetParam());

  ASSERT_TRUE(output.has_value());
  ASSERT_TRUE(output->draft_token_ids.defined());
  EXPECT_EQ(model_ptr->logits_calls(), 0);
  EXPECT_TRUE(model_ptr->received_sampling());
  EXPECT_FALSE(output->logits.defined());
  EXPECT_FALSE(output->selected_hidden.defined());
  EXPECT_FALSE(output->sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(output->draft_token_ids, token_ids));
  EXPECT_TRUE(torch::equal(output->draft_logits, draft_logits));
  EXPECT_TRUE(torch::equal(output->draft_confidence_probs, confidence));
  expect_retained_input(*output, input);
}

TEST_P(LLMWorkerOutputTest, HiddenStatesKeepTheExistingLogitsPath) {
  const auto tensor_options =
      torch::TensorOptions().device(device_.unwrap()).dtype(torch::kFloat);
  torch::Tensor hidden = torch::arange(12, tensor_options).view({3, 4});
  auto model =
      std::make_unique<OutputTestModel>(ModelOutput(hidden), tensor_options);
  OutputTestModel* model_ptr = model.get();
  const ParallelArgs parallel_args(0, 1, nullptr);
  OutputTestWorker worker(
      parallel_args, device_.unwrap(), options_, std::move(model));
  ForwardInput input = make_input();
  input.skip_sampling_for_logits_only = true;
  input.sampling_params.selected_token_idxes =
      torch::tensor({0, 2}, torch::kLong).to(device_.unwrap());

  std::optional<ForwardOutput> output = worker.run(input, GetParam());

  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(output->draft_token_ids.defined());
  EXPECT_EQ(model_ptr->logits_calls(), 1);
  EXPECT_TRUE(model_ptr->received_sampling());
  torch::Tensor expected =
      hidden.index_select(0, input.sampling_params.selected_token_idxes);
  EXPECT_TRUE(torch::equal(output->logits, expected));
  EXPECT_TRUE(torch::equal(output->selected_hidden, expected));
  EXPECT_TRUE(torch::equal(output->sample_output.embeddings, expected));
  EXPECT_FALSE(output->sample_output.next_tokens.defined());
  expect_retained_input(*output, input);
}

TEST_P(LLMWorkerOutputTest, OrdinaryHiddenStatesStillRunSampling) {
  const auto tensor_options =
      torch::TensorOptions().device(device_.unwrap()).dtype(torch::kFloat);
  torch::Tensor hidden = torch::arange(12, tensor_options).view({3, 4});
  auto model =
      std::make_unique<OutputTestModel>(ModelOutput(hidden), tensor_options);
  OutputTestModel* model_ptr = model.get();
  const ParallelArgs parallel_args(0, 1, nullptr);
  options_.enable_speculative_decode(false)
      .is_draft_engine(false)
      .num_speculative_tokens(0);
  OutputTestWorker worker(
      parallel_args, device_.unwrap(), options_, std::move(model));
  ForwardInput input = make_input();
  input.input_params.sampling.reset();
  input.return_selected_hidden = false;
  input.sampling_params.sample_idxes =
      input.sampling_params.selected_token_idxes;

  std::optional<ForwardOutput> output = worker.run(input, GetParam());

  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(output->draft_token_ids.defined());
  EXPECT_EQ(model_ptr->logits_calls(), 1);
  EXPECT_FALSE(model_ptr->received_sampling());
  EXPECT_TRUE(torch::equal(output->logits, hidden));
  EXPECT_TRUE(
      torch::equal(output->sample_output.next_tokens, hidden.argmax(-1)));
  expect_retained_input(*output, input);
}

INSTANTIATE_TEST_SUITE_P(Readiness,
                         LLMWorkerOutputTest,
                         ::testing::Values(false, true));

}  // namespace
}  // namespace xllm
