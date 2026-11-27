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
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <utility>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/model_loader.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/layers/npu_torch/tests_utils.h"
#include "core/runtime/dflash2_worker_impl.h"
#include "models/py_model_helper.h"
#include "platform/device.h"
#include "runtime/dspark_worker_impl.h"
#include "runtime/mtp_worker_impl.h"
#include "runtime/params_utils.h"

namespace py = pybind11;

namespace xllm {

namespace {

class DecodeInputMTPWorker final : public MTPWorkerImpl {
 public:
  using MTPWorkerImpl::fill_validate_input_from_draft_outputs;
  using MTPWorkerImpl::MTPWorkerImpl;
  using MTPWorkerImpl::update_decode_step_input;
  using WorkerImpl::compute_stream_;
};

// Exposes inherited protected runtime seams only from the test binary; no
// production friend or test-only getter is needed.
TEST(MTPDecodeStepTest, CacheCorrectionSurvivesPackedInputMaterialization) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_graph(true)
      .speculative_algorithm("mtp")
      .num_speculative_tokens(3)
      .block_size(16);
  DecodeInputMTPWorker worker(
      parallel_args, device.unwrap(), options, /*worker_type=*/WorkerType::LLM);
  ForwardInput scheduler;
  scheduler.input_params.meta.num_sequences = 3;
  scheduler.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  scheduler.token_ids_host = torch::tensor({-1, 17, -1}, torch::kInt32);
  scheduler.positions_host = torch::tensor({10, 20, 30}, torch::kInt32);
  scheduler.input_params.attention.host.q_seq_lens = {1, 1, 1};
  scheduler.input_params.attention.host.q_cu_seq_lens = {0, 1, 2, 3};
  scheduler.input_params.attention.host.kv_seq_lens = {11, 21, 31};
  scheduler.input_params.embedding.engram_lookback_token_ids = {
      1, 2, 3, 4, 5, 6};
  scheduler.input_params.execution_batch.num_reqs = 3;
  scheduler.input_params.execution_batch.num_tokens = 3;
  scheduler.input_params.execution_batch.num_scheduled_tokens = {1, 1, 1};
  scheduler.input_params.execution_batch.num_computed_tokens = {10, 20, 30};
  scheduler.input_params.execution_batch.is_prefilling = {0, 0, 0};
  scheduler.input_params.execution_batch.query_start_loc = {0, 1, 2, 3};
  proto::PackedForwardInput packed;
  ASSERT_TRUE(forward_input_to_packed_proto(scheduler, &packed));
  ForwardInput input;
  packed_proto_to_forward_input(packed, input, device.unwrap(), nullptr);
  input = input.to(device.unwrap(), torch::kFloat32);

  std::vector<EmbeddingCache::DecodeState> states(3);
  states[0].valid = true;
  states[0].token_id = 101;
  states[0].position_offset = 3;
  states[1].valid = true;
  states[1].token_id = 102;
  states[1].position_offset = 9;
  states[2].token_id = 0;
  input.sampling_params.logprob_token_ids =
      torch::tensor({{7, -1}, {11, 13}, {19, 23}}, torch::kInt32);
  input.decoder_sampling_params.logprob_token_ids =
      torch::tensor({{17, -1}, {19, 23}, {29, 31}}, torch::kInt32);
  worker.update_decode_step_input(input, states);
  ForwardInput materialized = input.to(device.unwrap(), torch::kFloat32);
  ForwardInput host_input =
      input.to(torch::Device(torch::kCPU), torch::kFloat32);
  ForwardInput contiguous;
  ASSERT_TRUE(
      host_input.to_contiguous_input_buffer(device.unwrap(), contiguous));
  EXPECT_TRUE(torch::equal(contiguous.token_ids, materialized.token_ids));
  EXPECT_TRUE(torch::equal(contiguous.positions, materialized.positions));
  EXPECT_TRUE(torch::equal(contiguous.sampling_params.logprob_token_ids.cpu(),
                           input.sampling_params.logprob_token_ids));
  EXPECT_TRUE(
      torch::equal(contiguous.decoder_sampling_params.logprob_token_ids.cpu(),
                   input.decoder_sampling_params.logprob_token_ids));
  EXPECT_TRUE(torch::equal(materialized.token_ids.cpu(),
                           torch::tensor({101, 17, 0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(materialized.positions.cpu(),
                           torch::tensor({13, 20, 30}, torch::kInt32)));
  EXPECT_EQ(materialized.input_params.attention.host.kv_seq_lens,
            (std::vector<int32_t>{14, 21, 31}));
  EXPECT_EQ(materialized.input_params.embedding.engram_lookback_token_ids,
            scheduler.input_params.embedding.engram_lookback_token_ids);
  EXPECT_EQ(materialized.input_params.execution_batch.num_computed_tokens,
            scheduler.input_params.execution_batch.num_computed_tokens);
}

TEST(MTPDecodeStepTest, FillsTargetWithRequestMajorDraftTokens) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(false)
      .enable_graph(false)
      .speculative_algorithm("mtp")
      .num_speculative_tokens(2)
      .block_size(16);
  DecodeInputMTPWorker worker(
      parallel_args, device.unwrap(), options, /*worker_type=*/WorkerType::LLM);

  ForwardInput input;
  input.input_params.meta.num_sequences = 2;

  ForwardInput validate_input;
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device.unwrap());
  validate_input.token_ids = torch::full({6}, /*fill_value=*/5, token_options);

  std::vector<ForwardOutput> draft_outputs(2);
  const torch::TensorOptions draft_token_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device.unwrap());
  draft_outputs[0].sample_output.next_tokens =
      torch::tensor({11, 12}, draft_token_options);
  draft_outputs[1].sample_output.next_tokens =
      torch::tensor(/*values=*/{21, 22}, draft_token_options);

  worker.fill_validate_input_from_draft_outputs(
      input, draft_outputs, validate_input, *worker.compute_stream_);

  EXPECT_TRUE(torch::equal(validate_input.token_ids.cpu(),
                           torch::tensor(/*values=*/{5, 11, 21, 5, 12, 22},
                                         /*options=*/torch::kInt32)));
}

class BlockDiffusionDecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_model_impl_ = ModelConfig::get_instance().model_impl();
    saved_python_model_path_ = ModelConfig::get_instance().python_model_path();
    saved_manual_loader_ = LoadConfig::get_instance().enable_manual_loader();
    saved_npu_kernel_backend_ =
        KernelConfig::get_instance().npu_kernel_backend();
    saved_chunked_prefill_ =
        SchedulerConfig::get_instance().enable_chunked_prefill();
    KernelConfig::get_instance().npu_kernel_backend("TORCH");
    const std::filesystem::path source_dir(XLLM_RUNTIME_TEST_SOURCE_DIR);
    ModelConfig::get_instance().model_impl("python");
    ModelConfig::get_instance().python_model_path(
        source_dir.parent_path().parent_path().parent_path().string());
    LoadConfig::get_instance().enable_manual_loader(false);
    SchedulerConfig::get_instance().enable_chunked_prefill(true);
    device_.set_device();
    device_.init_device_context();
    process_group_ =
        std::make_unique<layer::test::MockProcessGroup>(device_.unwrap());
    c10d::TCPStoreOptions store_options;
    store_options.port = 0;
    store_options.isServer = true;
    store_options.numWorkers = 1;
    rendezvous_ =
        c10::make_intrusive<c10d::TCPStore>("127.0.0.1", store_options);
    ensure_python_interpreter();
    {
      py::gil_scoped_acquire gil;
      py::dict globals;
      globals["model_file"] = (source_dir / "runtime_test_models.py").string();
      py::exec(R"(
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("xllm_runtime_test_models", model_file)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
fixture = module._create_fixture()
)",
               globals);
      registered_models_ = globals["fixture"];
    }
    char path[] = "/tmp/xllm_decode_step_XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    model_path_ = path;
    std::ofstream(model_path_ / "config.json") << R"json({
      "model_type": "qwen3", "torch_dtype": "float32",
      "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 1,
      "num_attention_heads": 1, "num_key_value_heads": 1,
      "head_dim": 4, "vocab_size": 64, "max_position_embeddings": 128,
      "eos_token_id": 63, "mask_token_id": 31,
      "dflash_config": {"block_size": 3, "selector_top_k": 1,
                         "conv_kernel_size": 1, "target_layer_ids": [0]}
    })json";
    std::ofstream(model_path_ / "model.safetensors", std::ios::binary)
        .write("\x08\0\0\0\0\0\0\0{}      ", 16);
    nlohmann::json vocabulary = nlohmann::json::object();
    for (int32_t token = 0; token < 64; ++token) {
      vocabulary[std::to_string(token)] = token;
    }
    // XGrammar constraint-vocabulary rebuilds require a ByteLevel decoder
    // (see FastTokenizer's get_constraint_vocab); the digit strings keep the
    // CrossProcessGrammarRestoresCommittedHistory regex assertions intact.
    nlohmann::json tokenizer = {{"version", "1.0"},
                                {"added_tokens", nlohmann::json::array()},
                                {"normalizer", nullptr},
                                {"pre_tokenizer",
                                 {{"type", "ByteLevel"},
                                  {"add_prefix_space", false},
                                  {"trim_offsets", true},
                                  {"use_regex", false}}},
                                {"post_processor", nullptr},
                                {"decoder",
                                 {{"type", "ByteLevel"},
                                  {"add_prefix_space", true},
                                  {"trim_offsets", true},
                                  {"use_regex", true}}},
                                {"truncation", nullptr},
                                {"padding", nullptr},
                                {"model",
                                 {{"type", "BPE"},
                                  {"dropout", nullptr},
                                  {"unk_token", nullptr},
                                  {"continuing_subword_prefix", nullptr},
                                  {"end_of_word_suffix", nullptr},
                                  {"fuse_unk", false},
                                  {"byte_fallback", false},
                                  {"vocab", std::move(vocabulary)},
                                  {"merges", nlohmann::json::array()}}}};
    std::ofstream(model_path_ / "tokenizer.json") << tokenizer;
  }

  void TearDown() override {
    EXPECT_EQ(device_.synchronize_default_stream(), 0);
    if (registered_models_) {
      py::gil_scoped_acquire gil;
      registered_models_.attr("close")();
      registered_models_ = py::object();
    }
    ModelConfig::get_instance().model_impl(saved_model_impl_);
    ModelConfig::get_instance().python_model_path(saved_python_model_path_);
    LoadConfig::get_instance().enable_manual_loader(saved_manual_loader_);
    KernelConfig::get_instance().npu_kernel_backend(saved_npu_kernel_backend_);
    SchedulerConfig::get_instance().enable_chunked_prefill(
        saved_chunked_prefill_);
    if (!model_path_.empty()) {
      std::filesystem::remove_all(model_path_);
    }
    process_group_.reset();
    rendezvous_.reset();
  }

  ForwardInput make_input(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& positions,
                          bool reordered = false) {
    ForwardInput input;
    input.token_ids_host = torch::tensor(tokens, torch::kInt32);
    input.positions_host = torch::tensor(positions, torch::kInt32);
    input.token_ids = input.token_ids_host.to(device_.unwrap());
    input.positions = input.positions_host.to(device_.unwrap());
    auto& params = input.input_params;
    params.meta.num_sequences = 2;
    params.meta.batch_forward_type = BatchForwardType::DECODE;
    params.meta.kv_max_seq_len = std::max(positions[0], positions[1]) + 1;
    params.embedding.embedding_ids =
        reordered ? std::vector<int32_t>{1, 0} : std::vector<int32_t>{0, 1};
    params.embedding.request_ids =
        reordered ? std::vector<std::string>{"second", "first"}
                  : std::vector<std::string>{"first", "second"};
    input.sample_sequence_ids = params.embedding.request_ids;
    params.attention.host.q_seq_lens = {1, 1};
    params.attention.host.kv_seq_lens = {positions[0] + 1, positions[1] + 1};
    params.attention.host.block_tables =
        torch::arange(16, torch::kInt32).view({2, 8});
    if (reordered) {
      params.attention.host.block_tables =
          params.attention.host.block_tables.flip({0});
    }
    params.attention.device.block_tables =
        params.attention.host.block_tables.to(device_.unwrap());
    auto& execution = params.execution_batch;
    execution.num_reqs = 2;
    execution.num_tokens = 2;
    execution.num_scheduled_tokens = {1, 1};
    execution.num_computed_tokens = {positions[0], positions[1]};
    execution.query_start_loc = {0, 1, 2};
    execution.is_prefilling = {0, 0};
    input.sampling_params.sampling_mode = SamplingMode::GREEDY;
    input.sampling_params.do_sample =
        torch::zeros({2}, input.token_ids.options().dtype(torch::kBool));
    input.metadata_ready_event = device_.current_stream()->record_event();
    return input;
  }

  std::unique_ptr<DFlashWorkerImpl> make_worker(
      const std::string& speculative_algorithm,
      bool enable_probabilistic_draft = false) {
    ParallelArgs parallel_args(0, 1, process_group_.get());
    parallel_args.tp_group_ = process_group_.get();
    parallel_args.moe_tp_group_ = process_group_.get();
    parallel_args.single_rank_group_ = process_group_.get();
    parallel_args.python_rendezvous_host_ = "127.0.0.1";
    parallel_args.python_rendezvous_port_ = rendezvous_->getPort();
    runtime::Options options;
    options.enable_graph(false)
        .backend("llm")
        .model_path(model_path_.string())
        .draft_model_path(model_path_.string())
        .enable_schedule_overlap(true)
        .enable_speculative_decode(true)
        .speculative_algorithm(speculative_algorithm)
        .enable_probabilistic_draft(enable_probabilistic_draft)
        .num_speculative_tokens(2)
        .block_size(4);
    std::unique_ptr<DFlashWorkerImpl> worker;
    if (speculative_algorithm == "DSpark") {
      worker = std::make_unique<DSparkWorkerImpl>(
          parallel_args, device_.unwrap(), options, WorkerType::LLM);
    } else if (speculative_algorithm == "DFlash2") {
      worker = std::make_unique<DFlash2WorkerImpl>(
          parallel_args, device_.unwrap(), options, WorkerType::LLM);
    } else {
      worker = std::make_unique<DFlashWorkerImpl>(
          parallel_args, device_.unwrap(), options, WorkerType::LLM);
    }
    CHECK(worker->init_model(
        model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
    CHECK(worker->init_model(
        model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
    KVCacheCapacity capacity;
    capacity.n_blocks(32).block_size(4);
    ModelArgs cache_args;
    cache_args.n_layers(1).n_heads(1).n_kv_heads(1).hidden_size(4).head_dim(4);
    CHECK(worker->allocate_kv_cache(
        KVCacheShape(capacity, cache_args, /*world_size=*/1)));

    return worker;
  }

  torch::Tensor context_cache() const {
    py::gil_scoped_acquire gil;
    return registered_models_.attr("trace")
        .attr("context_cache")
        .cast<torch::Tensor>();
  }

  std::vector<torch::Tensor> trace_tensors(const char* field) const {
    py::gil_scoped_acquire gil;
    return registered_models_.attr("trace")
        .attr(field)
        .cast<std::vector<torch::Tensor>>();
  }

  Device device_{0};
  std::unique_ptr<layer::test::MockProcessGroup> process_group_;
  c10::intrusive_ptr<c10d::TCPStore> rendezvous_;
  std::filesystem::path model_path_;
  std::string saved_model_impl_;
  std::string saved_python_model_path_;
  py::object registered_models_;
  bool saved_manual_loader_ = false;
  std::string saved_npu_kernel_backend_;
  bool saved_chunked_prefill_ = false;
};

class BlockDiffusionDecodeExecutionTest
    : public BlockDiffusionDecodeTest,
      public ::testing::WithParamInterface<std::string> {};

TEST_F(BlockDiffusionDecodeTest, MissingConstraintSnapshotFailsClosed) {
  auto worker = make_worker("DSpark");
  ForwardInput input;
  input.sample_sequence_ids = {"cross-process"};
  input.json_object_state_snapshots.resize(1);
  input.json_object_state_snapshots[0].constrained_without_snapshot = true;
  worker->restore_json_object_states(input);
  ASSERT_EQ(input.json_object_errors.size(), 1);
  EXPECT_EQ(input.json_object_errors.front().sample_sequence_id,
            "cross-process");
  ASSERT_EQ(input.json_object_states.size(), 1);
  EXPECT_TRUE(input.json_object_states.front().initialized());
  EXPECT_FALSE(input.json_object_states.front().is_valid());
  const torch::Tensor mask = build_filter_bitmask(input.json_object_states);
  EXPECT_TRUE(torch::equal(mask, torch::zeros_like(mask)));
}

TEST_F(BlockDiffusionDecodeTest, ExpandedValidationStatesSkipSequenceRestore) {
  // Target verification expands each sequence-level snapshot into
  // num_speculative_tokens + 1 per-position states; those rows are already
  // restored, so the sequence snapshot must be skipped instead of replayed
  // over them.
  auto worker = make_worker("DSpark");
  ForwardInput input;
  input.sample_sequence_ids = {"expanded-validation"};
  input.json_object_state_snapshots.resize(1);
  input.json_object_states.resize(3);  // 1 snapshot x (2 speculative + 1).
  worker->restore_json_object_states(input);

  EXPECT_TRUE(input.json_object_state_snapshots.empty());
  EXPECT_EQ(input.json_object_states.size(), 3u);
  EXPECT_TRUE(input.json_object_errors.empty());
}

TEST_F(BlockDiffusionDecodeTest, AsyncOutputDoesNotRequireDraftPrelaunch) {
  // Probabilistic draft sampling disables device-input prelaunch at init,
  // while schedule overlap still enables asynchronous target output.
  auto worker =
      make_worker(/*algorithm=*/"DSpark", /*enable_probabilistic_draft=*/true);
  ForwardInput input = make_input({2, 12}, {2, 10});
  auto output = worker->step(input);
  ASSERT_TRUE(output.has_value());
  ASSERT_NE(output->ready_event, nullptr);
  EXPECT_TRUE(output->requires_host_retirement);
  EXPECT_TRUE(output->ready_event->synchronize());
}

TEST_F(BlockDiffusionDecodeTest, PendingRowIdentityMismatchUsesCurrentBatch) {
  auto worker = make_worker("DSpark");
  ForwardInput first_input = make_input({2, 12}, {2, 10});
  auto first_output = worker->step(first_input);
  ASSERT_TRUE(first_output.has_value());
  ASSERT_NE(first_output->ready_event, nullptr);
  ASSERT_TRUE(first_output->ready_event->synchronize());

  // Keep the sequence labels but change both cache identities. The pending
  // row map must reject this partial identity match and build a fresh query
  // from the new batch's tokens instead of reusing the previous acceptance.
  ForwardInput mismatched_input = make_input({30, 40}, {6, 14});
  mismatched_input.input_params.embedding.embedding_ids = {10, 11};
  mismatched_input.input_params.embedding.request_ids = {"new-first",
                                                         "new-second"};
  mismatched_input.sample_sequence_ids = {"new-sequence-0", "new-sequence-1"};
  auto output = worker->step(mismatched_input);
  ASSERT_TRUE(output.has_value());
  ASSERT_NE(output->ready_event, nullptr);
  ASSERT_TRUE(output->ready_event->synchronize());

  const auto draft_tokens = trace_tensors("draft_tokens");
  ASSERT_EQ(draft_tokens.size(), 2u);
  const auto actual_anchors =
      draft_tokens.back().cpu().view({2, -1}).select(1, 0).to(torch::kLong);
  EXPECT_TRUE(
      torch::equal(actual_anchors, torch::tensor({30, 40}, torch::kLong)));
}

TEST_F(BlockDiffusionDecodeTest,
       ConstrainedDecodeKeepsAsyncOutputAndEnforcesMask) {
  auto worker = make_worker("DSpark");
  ForwardInput input = make_input({2, 12}, {2, 10});
  input.sampling_params.filter_mask = torch::full(
      {2, 64}, -1.0e9F, torch::TensorOptions().dtype(torch::kFloat));
  input.sampling_params.filter_mask.select(1, 50).zero_();

  auto output = worker->step(input);
  ASSERT_TRUE(output.has_value());
  ASSERT_NE(output->ready_event, nullptr);
  EXPECT_TRUE(output->requires_host_retirement);
  ASSERT_TRUE(output->ready_event->synchronize());
  const torch::Tensor first_tokens =
      output->sample_output.next_tokens.cpu().view({2, -1}).select(1, 0).to(
          torch::kLong);
  EXPECT_TRUE(torch::equal(first_tokens, torch::full({2}, 50, torch::kLong)));
}

#if defined(USE_XGRAMMAR)
TEST_F(BlockDiffusionDecodeTest, CrossProcessGrammarRestoresCommittedHistory) {
  auto worker = make_worker("DSpark");
  ForwardInput input;
  input.sample_sequence_ids = {"cross-process-xgrammar"};
  input.json_object_state_snapshots.resize(1);
  auto& snapshot = input.json_object_state_snapshots.front();
  snapshot.enabled = true;
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::REGEX;
  spec.source = "[0-9]{3}";
  snapshot.grammar_spec = std::make_shared<const GrammarSpec>(std::move(spec));
  snapshot.token_ids = {0};

  auto grammar_spec = snapshot.grammar_spec;
  worker->restore_json_object_states(input);
  EXPECT_TRUE(input.json_object_errors.empty());
  ASSERT_EQ(input.json_object_states.size(), 1u);
  ASSERT_TRUE(input.json_object_states.front().is_valid());
  EXPECT_TRUE(input.json_object_states.front().machine()->can_accept(
      input.json_object_states.front(), 1));

  input.json_object_state_snapshots.resize(1);
  auto& next_snapshot = input.json_object_state_snapshots.front();
  next_snapshot.enabled = true;
  next_snapshot.grammar_spec = std::move(grammar_spec);
  next_snapshot.token_ids = {0, 1};
  worker->restore_json_object_states(input);
  EXPECT_TRUE(input.json_object_errors.empty());
  ASSERT_EQ(input.json_object_states.size(), 1u);
  ASSERT_TRUE(input.json_object_states.front().is_valid());
  EXPECT_TRUE(input.json_object_states.front().machine()->can_accept(
      input.json_object_states.front(), 2));
}
#endif

TEST_P(BlockDiffusionDecodeExecutionTest,
       AcceptedHistorySurvivesFallbackAndReordering) {
  const std::string speculative_algorithm = GetParam();
  if (speculative_algorithm == "DFlash2") {
    // DFlash2 startup requires this field through the full HF loader path.
    auto loader = ModelLoader::create(model_path_.string());
    ASSERT_EQ(loader->model_args().num_speculative_tokens(), 2)
        << "generated config must provide DFlash2 block size";
  }
  auto worker = make_worker(speculative_algorithm);
  torch::Tensor reused_cache = context_cache();

  const std::vector<std::vector<int32_t>> scheduler_positions = {
      {2, 10}, {3, 11}, {6, 14}, {17, 9}};
  const std::vector<std::vector<int32_t>> expected_anchors = {
      {2, 12}, {5, 15}, {8, 18}, {21, 11}};
  std::vector<ForwardOutput> outputs;
  outputs.reserve(scheduler_positions.size());
  std::vector<ForwardInput> inputs;
  inputs.reserve(scheduler_positions.size());
  std::promise<void> service_reader_started;
  auto service_reader_ready = service_reader_started.get_future();
  std::future<torch::Tensor> service_acceptance;
  for (size_t step = 0; step < scheduler_positions.size(); ++step) {
    ForwardInput input = make_input(
        step == 0 ? std::vector<int32_t>{2, 12} : std::vector<int32_t>{-1, -1},
        scheduler_positions[step],
        /*reordered=*/step == 3);
    input.input_params.meta.batch_id = step;
    if (step == 2) {
      input.sampling_params.repetition_penalties = torch::tensor({1.1F, 1.1F});
      input.sampling_params.unique_token_ids =
          torch::tensor({{6, 7, 8}, {16, 17, 18}}, torch::kLong);
      input.sampling_params.unique_token_counts =
          torch::ones({2, 3}, torch::kInt32);
      input.sampling_params.unique_token_ids_lens =
          torch::full({2}, 3, torch::kInt32);
    }
    auto output = worker->step(input);
    ASSERT_TRUE(output.has_value());
    ASSERT_NE(output->ready_event, nullptr);
    EXPECT_TRUE(output->requires_host_retirement);
    if (step == 0) {
      const StreamEventPtr ready = output->ready_event;
      const torch::Tensor accepted_host = output->sample_output.next_tokens;
      service_acceptance = std::async(
          std::launch::async,
          [this, ready, accepted_host, &service_reader_started]() {
            device_.set_device();
            auto service_stream = device_.get_stream_from_pool();
            const bool wait_enqueued = service_stream->wait_event(ready);
            service_reader_started.set_value();
            if (!wait_enqueued || service_stream->synchronize() != 0) {
              return torch::Tensor();
            }
            return accepted_host.clone();
          });
      service_reader_ready.wait();
    }
    inputs.emplace_back(std::move(input));
    outputs.emplace_back(std::move(*output));
  }
  ASSERT_TRUE(service_acceptance.valid());
  EXPECT_TRUE(torch::equal(service_acceptance.get(),
                           outputs.front().sample_output.next_tokens));
  // Destroy the final worker without another decode submission. Its last
  // context write must retire before the cache pool can be released.
  aclrtEventRecordedStatus status;
  ASSERT_EQ(
      aclrtQueryEventStatus(outputs.back().ready_event->npu_event(), &status),
      ACL_SUCCESS);
  ASSERT_EQ(status, ACL_EVENT_RECORDED_STATUS_NOT_READY);
  worker.reset();
  auto reuse_stream = device_.get_stream_from_pool();
  {
    auto guard = reuse_stream->set_stream_guard();
    reused_cache.fill_(23);
  }
  ASSERT_EQ(reuse_stream->synchronize(), 0);
  EXPECT_TRUE(reused_cache.cpu().eq(23).all().item<bool>());
  const auto draft_tokens = trace_tensors("draft_tokens");
  const auto context_slots = trace_tensors("context_slots");
  ASSERT_EQ(draft_tokens.size(), outputs.size());
  ASSERT_EQ(context_slots.size(), outputs.size());
  for (size_t step = 0; step < outputs.size(); ++step) {
    SCOPED_TRACE(step);
    ASSERT_TRUE(outputs[step].ready_event->synchronize());
    auto anchors = torch::tensor(expected_anchors[step], torch::kLong);
    EXPECT_TRUE(torch::equal(
        draft_tokens[step].cpu().view({2, -1}).select(1, 0).to(torch::kLong),
        anchors));
    EXPECT_TRUE(
        torch::equal(outputs[step].sample_output.next_tokens.to(torch::kLong),
                     anchors.unsqueeze(1) + torch::arange(1, 4, torch::kLong)));
    auto positions = torch::tensor(scheduler_positions[step], torch::kInt32);
    if (step != 0) {
      positions += 2;
    }
    auto context_positions =
        positions.unsqueeze(1) + torch::arange(3, torch::kInt32).unsqueeze(0);
    auto tables = inputs[step].input_params.attention.host.block_tables;
    auto expected_slots =
        tables.gather(
            1, torch::floor_divide(context_positions, 4).to(torch::kLong)) *
            4 +
        context_positions.remainder(4);
    EXPECT_TRUE(
        torch::equal(context_slots[step].cpu().view({2, 3}), expected_slots));
  }
}

INSTANTIATE_TEST_SUITE_P(BlockDiffusionAlgorithms,
                         BlockDiffusionDecodeExecutionTest,
                         ::testing::Values(std::string("DSpark"),
                                           std::string("DFlash"),
                                           std::string("DFlash2")));

}  // namespace
}  // namespace xllm
