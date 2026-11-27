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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/causal_vlm.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampler.h"
#include "core/runtime/vlm_worker_impl.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests/core/forked_rank_session.h"
#include "util/net.h"

namespace xllm {
namespace {

constexpr int32_t kWorldSize = 4;
constexpr int32_t kTpSize = 2;
constexpr int64_t kBatchSize = 32;
constexpr int64_t kVocabSize = 64;

class SamplingVlm final : public CausalVLM {
 public:
  explicit SamplingVlm(const torch::Device& device)
      : options_(torch::TensorOptions().device(device).dtype(torch::kFloat32)) {
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& /*params*/) override {
    last_tokens_ = tokens.clone();
    return ModelOutput(torch::zeros({tokens.numel(), kVocabSize}, options_));
  }
  MMDict encode(const ModelInputParams& /*params*/) override { return {}; }
  torch::Tensor get_input_embeddings(
      const torch::Tensor& tokens,
      const ModelInputParams& /*params*/) override {
    return torch::zeros({tokens.numel(), kVocabSize}, options_);
  }
  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    return hidden.index_select(/*dim=*/0, selected);
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  const torch::Tensor& last_tokens() const { return last_tokens_; }

 private:
  torch::TensorOptions options_;
  torch::Tensor last_tokens_;
};

class SamplingVlmWorker final : public VLMWorkerImpl {
 public:
  SamplingVlmWorker(const ParallelArgs& parallel,
                    const torch::Device& device,
                    const runtime::Options& options)
      : VLMWorkerImpl(parallel, device, options) {
    dtype_ = torch::kFloat32;
    model_ = std::make_unique<SamplingVlm>(device);
    context_ =
        ModelContext(parallel, ModelArgs(), QuantArgs(), model_->options());
    model_executor_ = std::make_unique<Executor>(
        model_.get(), context_.get_model_args(), device, options);
  }

  using VLMWorkerImpl::step_for_schedule_overlap;
  using VLMWorkerImpl::update_input_by_last_step_output_for_schedule_overlap;
  using WorkerImpl::update_last_step_output;

  const torch::Tensor& last_model_tokens() const {
    return static_cast<const SamplingVlm&>(*model_).last_tokens();
  }
};

ForwardInput make_input(const torch::Device& device) {
  ForwardInput input;
  const auto ints = torch::TensorOptions().device(device).dtype(torch::kInt64);
  input.token_ids = torch::ones({kBatchSize}, ints);
  input.positions = torch::zeros({kBatchSize}, ints);
  input.input_params.meta.num_sequences = kBatchSize;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.sampling_params.selected_token_idxes = torch::arange(kBatchSize, ints);
  input.sampling_params.sample_idxes =
      input.sampling_params.selected_token_idxes;
  input.sampling_params.do_sample =
      torch::ones({kBatchSize}, ints.dtype(torch::kBool));
  input.sampling_params.sampling_mode = SamplingMode::RANDOM;

  return input;
}

void perturb_rng(const Device& device, int32_t rank) {
  device.set_seed(2026);
  if (rank > 0) {
    // Different offsets exercise consensus along both the TP and CP axes.
    torch::rand({rank * 4096}, torch::TensorOptions().device(device.unwrap()));
  }
}

int32_t run_sampling_rank(int32_t rank, const std::array<int32_t, 5>& ports) {
  if (Platform::device_count() < kWorldSize) {
    return 77;
  }
  Device device(rank);
  device.set_device();
  ExecutionConfig::get_instance().enable_graph(false);
  LoadConfig::get_instance().enable_manual_loader(false);
  ModelConfig::get_instance().model_impl("native");
  const torch::NoGradGuard no_grad;
  auto global = create_process_group(rank,
                                     kWorldSize,
                                     kWorldSize,
                                     ports[0],
                                     /*trans=*/false,
                                     /*host=*/"127.0.0.1",
                                     /*group_name=*/"vlm_sampling_global",
                                     device);
  auto tp = create_process_group(rank,
                                 kWorldSize,
                                 kTpSize,
                                 ports[1 + rank / kTpSize],
                                 /*trans=*/false,
                                 /*host=*/"127.0.0.1",
                                 /*group_name=*/"vlm_sampling_tp",
                                 device);
  auto cp = create_process_group(rank,
                                 kWorldSize,
                                 /*rank_size=*/2,
                                 ports[3 + rank % kTpSize],
                                 /*trans=*/true,
                                 /*host=*/"127.0.0.1",
                                 /*group_name=*/"vlm_sampling_cp",
                                 device);
  ParallelArgs parallel(rank, kWorldSize, global.get());
  parallel.cp_size(2);
  parallel.tp_size(kTpSize);
  parallel.tp_group_ = tp.get();
  parallel.cp_group_ = cp.get();
  runtime::Options options;
  options.backend("vlm").cp_size(2).enable_graph(false).enable_schedule_overlap(
      true);
  SamplingVlmWorker worker(parallel, device, options);
  ForwardInput input = make_input(device);

  // Prove that the RNG perturbation changes real sampler results before
  // checking the worker's consensus, rather than relying on different seeds.
  perturb_rng(device, rank);
  torch::Tensor logits = torch::zeros({kBatchSize, kVocabSize},
                                      torch::TensorOptions().device(device));
  const SampleOutput local_sample = sample(logits, input.sampling_params);
  std::vector<torch::Tensor> local_samples;
  local_samples.reserve(kWorldSize);
  for (int32_t peer = 0; peer < kWorldSize; ++peer) {
    local_samples.emplace_back(torch::empty_like(local_sample.next_tokens));
  }
  global->allgather(local_sample.next_tokens, local_samples);
  CHECK(!torch::equal(local_samples[0], local_samples[1]));
  CHECK(!torch::equal(local_samples[0], local_samples[2]));
  const torch::Tensor expected = local_samples[0].clone();

  perturb_rng(device, rank);
  auto output = worker.step_for_schedule_overlap(input);
  CHECK(output.has_value());
  CHECK(output->ready_event != nullptr);
  CHECK(output->ready_event->synchronize());
  CHECK(torch::equal(output->sample_output.next_tokens, expected));

  worker.update_last_step_output(
      output, /*request_ids=*/{}, /*sample_sequence_ids=*/{});
  ForwardInput next = make_input(device);
  // Negative placeholders map to the previous step's sample slots.
  next.token_ids = -torch::arange(1, kBatchSize + 1, next.token_ids.options());
  next = worker.update_input_by_last_step_output_for_schedule_overlap(next);
  CHECK(torch::equal(next.token_ids, expected));
  const auto next_output = worker.step_for_schedule_overlap(next);
  CHECK(next_output.has_value());
  CHECK(next_output->ready_event->synchronize());
  CHECK(torch::equal(worker.last_model_tokens(), expected));
  return 0;
}

TEST(VlmWorkerSamplingTest, OverlapTokensAgreeAcrossTpAndCpAfterRngDiverges) {
  std::array<int32_t, 5> ports{};
  for (int32_t& port : ports) {
    int32_t candidate = net::get_local_free_port();
    while (std::find(ports.begin(), ports.end(), candidate) != ports.end()) {
      candidate = net::get_local_free_port();
    }
    ASSERT_GT(candidate, 0);
    port = candidate;
  }
  test::ForkedRankSession session(kWorldSize,
                                  ports[0],
                                  {{"sampling",
                                    [&ports](int32_t rank, int32_t /*unused*/) {
                                      return run_sampling_rank(rank, ports);
                                    }}},
                                  /*cleanup=*/{},
                                  /*timeout=*/std::chrono::seconds(90));
  test::expect_rank_result(session.run("sampling"));
  test::expect_rank_result(session.close());
}

TEST(VlmWorkerSamplingTest, GreedyOverlapAndNonOverlapNeedNoSamplingGroups) {
  test::ForkedRankSession session(
      /*ranks=*/1,
      /*port=*/0,
      {{"no_consensus", [](int32_t /*rank*/, int32_t /*unused*/) {
          if (Platform::device_count() == 0) {
            return int32_t{77};
          }
          Device device(/*device_index=*/0);
          device.set_device();
          ExecutionConfig::get_instance().enable_graph(false);
          LoadConfig::get_instance().enable_manual_loader(false);
          ModelConfig::get_instance().model_impl("native");
          ParallelArgs parallel(/*rank=*/0, /*world_size=*/2, nullptr);
          parallel.cp_size(2);
          for (const bool overlap : {true, false}) {
            runtime::Options options;
            options.backend("vlm")
                .cp_size(2)
                .enable_graph(false)
                .enable_schedule_overlap(overlap);
            SamplingVlmWorker worker(parallel, device, options);
            ForwardInput input = make_input(device);
            input.sampling_params.sampling_mode =
                overlap ? SamplingMode::GREEDY : SamplingMode::RANDOM;
            input.sampling_params.do_sample.fill_(!overlap);
            const auto output = worker.step(input);
            CHECK(output.has_value());
            CHECK(output->sample_output.next_tokens.defined());
            if (overlap) {
              CHECK(torch::equal(output->sample_output.next_tokens,
                                 torch::zeros_like(input.token_ids)));
            }
          }
          return int32_t{0};
        }}});
  test::expect_rank_result(session.run("no_consensus"));
  test::expect_rank_result(session.close());
}

}  // namespace
}  // namespace xllm
