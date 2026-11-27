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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/kernels/musa/gdn_ops.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/musa_graph_executor_impl.h"
#include "tests/core/framework/constraint/builtin_state_test_utils.h"

namespace xllm::runtime::musa {
namespace {

class VerifyGraphModel final : public CausalLM {
 public:
  explicit VerifyGraphModel(const torch::TensorOptions& options)
      : options_(options),
        embedding_weight_(torch::arange(16, options)
                              .unsqueeze(1)
                              .expand({16, 256})
                              .contiguous()) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    ++forward_count_;
    if (params.embedding.input_embedding.defined()) {
      if (params.attn_metadata) {
        EXPECT_FALSE(params.attn_metadata->attn_mask.defined());
        EXPECT_EQ(params.attn_metadata->is_spec_verify, params.is_spec_verify);
        EXPECT_EQ(params.attn_metadata->max_query_len,
                  params.meta.q_max_seq_len);
      }
      const torch::Tensor embedding = torch::nn::functional::embedding(
          tokens.clamp(/*min=*/0, /*max=*/15), embedding_weight_);
      const torch::Tensor position_values =
          positions.dim() == 2 ? positions.sum(/*dim=*/0) : positions;
      return ModelOutput(embedding + params.embedding.input_embedding +
                         embedding.sum() +
                         params.embedding.input_embedding.sum() +
                         position_values.to(embedding.options()).unsqueeze(1));
    }
    torch::Tensor hidden = tokens + positions + params.meta.num_sequences;
    const auto linear_cache = std::find_if(
        kv_caches.begin(), kv_caches.end(), [](const KVCache& cache) {
          return cache.get_conv_cache().defined();
        });
    if (linear_cache != kv_caches.end()) {
      torch::Tensor conv = linear_cache->get_conv_cache();
      torch::Tensor ssm = linear_cache->get_ssm_cache();
      const int64_t stride = ssm.size(0) / conv.size(0);
      const torch::Tensor increment = tokens[0].to(torch::kFloat32);
      for (int32_t id : params.embedding.linear_state_ids) {
        conv.select(/*dim=*/0, id).add_(increment);
        ssm.narrow(/*dim=*/0, /*start=*/id * stride, stride).add_(increment);
      }
      hidden = hidden + conv.sum() + ssm.sum();
    }
    return ModelOutput(
        hidden.to(options_).unsqueeze(1).expand({tokens.size(0), 256}));
  }
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& /*selected_indices*/) override {
    return hidden_states;
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  const torch::TensorOptions& options() const override { return options_; }
  layer::WordEmbedding get_word_embedding() override { return nullptr; }
  int32_t forward_count() const { return forward_count_; }

 private:
  torch::TensorOptions options_;
  torch::Tensor embedding_weight_;
  int32_t forward_count_ = 0;
};

ModelInputParams make_decode_params(const std::vector<int32_t>& ids,
                                    const torch::Device& device) {
  const int64_t batch = static_cast<int64_t>(ids.size());
  const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = batch;
  params.meta.actual_num_sequences = batch;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 10;
  params.attention.device.q_seq_lens = torch::arange(batch + 1, ints);
  params.attention.device.q_cu_seq_lens = torch::arange(batch + 1, ints);
  params.attention.device.kv_seq_lens = torch::arange(batch + 1, ints) * 10;
  params.attention.device.new_cache_slots = torch::arange(batch, ints) * 64;
  params.attention.device.block_tables =
      torch::arange(batch, ints).unsqueeze(1);
  params.attention.device.paged_kv_indptr = torch::arange(batch + 1, ints);
  params.attention.device.paged_kv_indices = torch::arange(batch, ints);
  params.attention.device.paged_kv_last_page_len =
      torch::full({batch}, 10, ints);
  params.attention.device.kv_cache_tokens_nums = torch::full({batch}, 9, ints);
  params.embedding.linear_state_ids = ids;
  params.embedding.linear_state_indices = torch::tensor(ids, ints);
  params.attention.host.q_seq_lens.reserve(batch + 1);
  params.attention.host.kv_seq_lens.reserve(batch + 1);
  for (int32_t i = 0; i <= batch; ++i) {
    params.attention.host.q_seq_lens.emplace_back(i);
    params.attention.host.kv_seq_lens.emplace_back(i * 10);
  }
  return params;
}

ModelArgs make_verify_args() {
  ModelArgs args;
  args.model_type("qwen3_5_text");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.head_dim(256);
  args.n_heads(12);
  args.n_kv_heads(2);
  args.linear_num_key_heads(1);
  args.linear_num_value_heads(2);
  args.linear_key_head_dim(128);
  args.linear_value_head_dim(128);
  args.linear_conv_kernel_dim(4);
  args.max_position_embeddings(128);
  return args;
}

ModelInputParams make_verify_params(const torch::Device& device,
                                    int32_t batch = 2,
                                    int32_t tokens = 4) {
  auto params = make_decode_params(std::vector<int32_t>(batch, 0), device);
  params.is_spec_verify = true;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.q_max_seq_len = 2;
  params.attention.host.q_seq_lens.clear();
  for (int32_t row = 0; row <= batch; ++row) {
    params.attention.host.q_seq_lens.emplace_back(
        row + std::min(row, tokens - batch));
  }
  params.attention.device.q_seq_lens =
      torch::tensor(params.attention.host.q_seq_lens,
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  params.attention.device.q_cu_seq_lens = params.attention.device.q_seq_lens;
  params.attention.device.new_cache_slots =
      torch::arange(tokens, params.attention.device.q_seq_lens.options());
  params.linear_state_validity_mask.assign(batch, 1);
  params.num_accepted_tokens_host.assign(batch, 1);
  params.num_accepted_tokens =
      torch::ones({batch}, params.attention.device.q_seq_lens.options());
  return params;
}

class MusaMtpGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "No MUSA device available";
    }
    device_.set_device();
    device_.init_device_context();
    bf16_ =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device_.unwrap());
    args_ = make_verify_args();
    options_.max_seqs_per_batch(16);
    options_.max_tokens_per_batch(32);
    options_.block_size(64);
    caches_.emplace_back(KVCacheTensors{torch::zeros({16, 64, 2, 256}, bf16_),
                                        torch::zeros({16, 64, 2, 256}, bf16_)});
    auto& execution = ExecutionConfig::get_instance();
    saved_vmm_ = execution.enable_graph_vmm_pool();
    execution.enable_graph_vmm_pool(false);
  }
  void TearDown() override {
    if (saved_vmm_.has_value()) {
      ExecutionConfig::get_instance().enable_graph_vmm_pool(*saved_vmm_);
    }
  }
  Device device_{/*device_index=*/0};
  torch::TensorOptions bf16_;
  ModelArgs args_;
  runtime::Options options_;
  std::vector<KVCache> caches_;

 private:
  torch::NoGradGuard no_grad_;
  std::optional<bool> saved_vmm_;
};
using MusaMtpGraphDeathTest = MusaMtpGraphTest;

class MtpVerifyInputWorker final : public MTPWorkerImpl {
 public:
  using MTPWorkerImpl::fill_validate_input_from_draft_outputs;
  using MTPWorkerImpl::MTPWorkerImpl;
};

TEST_F(MusaMtpGraphTest, JsonVerifyMasksUnconstrainedDraftTokens) {
  options_.num_speculative_tokens(3);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  MtpVerifyInputWorker worker(
      parallel_args, device_.unwrap(), options_, WorkerType::LLM);
  std::unique_ptr<Stream> stream = device_.current_stream();
  c10::StreamGuard stream_guard = stream->set_stream_guard();
  const auto ints = bf16_.dtype(torch::kInt32);
  JsonObjectGrammar grammar({"{", "\"a\"", ":", "1", "}", "stop"},
                            /*stop_token_ids=*/{5});
  ForwardInput input;
  input.json_object_states = {make_builtin_test_state(grammar),
                              ConstraintState()};
  input.input_params.meta.num_sequences = 2;
  std::vector<ForwardOutput> draft_outputs(3);
  draft_outputs[0].sample_output.next_tokens =
      torch::tensor({4, 1}, ints.dtype(torch::kInt64));
  draft_outputs[1].sample_output.next_tokens =
      torch::tensor(/*values=*/{1, 2}, ints.dtype(/*dtype=*/torch::kInt64));
  draft_outputs[2].sample_output.next_tokens =
      torch::tensor(/*values=*/{2, 3}, ints.dtype(/*dtype=*/torch::kInt64));
  ForwardInput verify;
  verify.token_ids = torch::tensor({2, -1, -2, -3, 3, -1, -2, -3}, ints);
  worker.fill_validate_input_from_draft_outputs(
      input, draft_outputs, verify, *stream);
  ASSERT_TRUE(
      torch::equal(verify.token_ids,
                   torch::tensor(/*values=*/{2, 4, 1, 2, 3, 1, 2, 3}, ints)));
  EXPECT_EQ(verify.json_object_invalid_draft,
            std::vector<uint8_t>({1, 1, 1, 0, 0, 0}));
}

TEST(MusaGraphExecutorTest, ShrinkingDecodeReplayDoesNotUpdatePaddingState) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::NoGradGuard no_grad;
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  const auto ints = bf16.dtype(torch::kInt32);
  ModelArgs args;
  args.model_type("qwen3_5_text");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.head_dim(256);
  args.n_heads(12);
  args.n_kv_heads(2);
  args.max_position_embeddings(128);
  runtime::Options options;
  options.max_seqs_per_batch(32);
  options.max_tokens_per_batch(16);
  options.block_size(64);
  options.world_size(1);
  MusaGraphPersistentParam persistent(args, device.unwrap(), options);
  const torch::Tensor k_cache = torch::zeros({16, 64, 2, 256}, bf16);
  const torch::Tensor v_cache = torch::zeros_like(k_cache);
  const auto update = [&](const ModelInputParams& input) {
    const int64_t batch = input.meta.actual_num_sequences;
    return persistent.update(torch::ones({batch}, ints),
                             k_cache,
                             v_cache,
                             torch::ones({batch}, ints),
                             input,
                             /*padded_num_tokens=*/8,
                             /*return_capture_params=*/true);
  };
  const auto full =
      make_decode_params({0, 1, 2, 3, 4, 5, 6, 7}, device.unwrap());
  const auto captured = update(full);
  ASSERT_TRUE(captured.has_value());
  const torch::Tensor captured_indices =
      captured->embedding.linear_state_indices;

  kernel::musa::MateGatedDeltaRuleDecodeParams params;
  params.num_k_heads = 1;
  params.num_v_heads = 2;
  params.head_k_dim = 16;
  params.head_v_dim = 16;
  params.scale = 0.25;
  params.use_qk_l2norm = true;
  params.mixed_qkv = torch::full({8, 64}, 0.125, bf16);
  params.mixed_qkv.narrow(0, 5, 3).zero_();
  params.A_log = torch::zeros({2}, bf16.dtype(torch::kFloat32));
  params.dt_bias = torch::zeros_like(params.A_log);
  params.a = torch::zeros({8, 2}, bf16);
  params.b = torch::zeros({8, 2}, bf16);
  const torch::Tensor initial =
      torch::ones({16, 2, 16, 16}, bf16.dtype(torch::kFloat32));
  params.state = initial.clone();
  params.state_indices = captured_indices;
  params.decode_output = torch::empty({8, 2, 16}, bf16);
  kernel::musa::fused_gated_delta_rule_decode(params);
  params.state.copy_(initial);
  device.synchronize_default_stream();
  at::musa::MUSAGraph graph;
  const c10::musa::MUSAStream capture_stream = c10::musa::getStreamFromPool();
  {
    c10::StreamGuard stream_guard(capture_stream.unwrap());
    graph.capture_begin({0, 0}, musaStreamCaptureModeThreadLocal);
    kernel::musa::fused_gated_delta_rule_decode(params);
    graph.capture_end();
  }
  capture_stream.synchronize();
  auto small = make_decode_params({0, 1, 2, 3, 7}, device.unwrap());
  auto reference = params;
  reference.state = initial.clone();
  reference.state_indices = small.embedding.linear_state_indices;
  reference.mixed_qkv = params.mixed_qkv.narrow(0, 0, 5);
  reference.a = params.a.narrow(0, 0, 5);
  reference.b = params.b.narrow(0, 0, 5);
  reference.decode_output = torch::empty({5, 2, 16}, bf16);
  kernel::musa::fused_gated_delta_rule_decode(reference);
  device.synchronize_default_stream();

  // Exercise both sources with stale capacity rows beyond the live prefix.
  small.meta.num_sequences = 8;
  small.embedding.linear_state_ids = {0, 1, 2, 3, 7, 5, 6, 7};
  small.embedding.linear_state_indices =
      torch::tensor(small.embedding.linear_state_ids, ints);
  const torch::Tensor expected_indices =
      torch::tensor({0, 1, 2, 3, 7, -1, -1, -1}, torch::kInt32);
  for (const bool tensor_source : {true, false}) {
    SCOPED_TRACE(tensor_source);
    update(full);
    if (!tensor_source) {
      small.embedding.linear_state_indices = torch::Tensor();
    }
    const auto next = update(small);
    ASSERT_TRUE(next.has_value());
    ASSERT_EQ(next->embedding.linear_state_indices.numel(), 8);
    EXPECT_TRUE(torch::equal(captured_indices.cpu(), expected_indices));
    params.state.copy_(initial);
    graph.replay();
    device.synchronize_default_stream();
    EXPECT_TRUE(torch::equal(params.state.cpu(), reference.state.cpu()))
        << "state max_abs="
        << (params.state - reference.state).abs().max().item<float>();
  }
}

TEST_F(MusaMtpGraphTest, MtpReplayRefreshesTokenEmbeddingAndPadding) {
  const auto ints = bf16_.dtype(torch::kInt32);
  args_.model_type("qwen3_5_mtp");
  options_.max_seqs_per_batch(3);
  options_.max_tokens_per_batch(8);
  VerifyGraphModel model(bf16_);
  VerifyGraphModel reference_model(bf16_);
  MusaGraphExecutorImpl executor(&model, args_, device_.unwrap(), options_);
  const auto run = [&](int64_t rows, int32_t token_id) {
    auto input =
        make_decode_params(std::vector<int32_t>(rows, 0), device_.unwrap());
    input.clear_linear_attention_state();
    input.embedding.input_embedding = torch::full({rows, 256}, 3, bf16_);
    const torch::Tensor tokens = torch::full({rows}, token_id, ints);
    const torch::Tensor positions = torch::ones_like(tokens);
    const ModelOutput expected =
        reference_model.forward(tokens, positions, caches_, input);
    // Incoming graph metadata is stale; the executor must rebuild this batch.
    input.attn_metadata = std::make_shared<layer::AttentionMetadata>();
    input.attn_metadata->enable_cuda_graph = true;
    input.attn_metadata->is_spec_verify = true;
    input.attn_metadata->max_query_len = 2;
    const ModelOutput output = executor.run(tokens, positions, caches_, input);
    device_.synchronize_default_stream();
    EXPECT_TRUE(
        torch::equal(output.hidden_states.cpu(), expected.hidden_states.cpu()));
  };
  // Three requests produce six repair/current attention rows.
  run(/*rows=*/6, /*token_id=*/2);
  const int32_t capture_count = model.forward_count();
  EXPECT_GT(capture_count, 1);
  run(/*rows=*/6, /*token_id=*/7);
  EXPECT_EQ(model.forward_count(), capture_count);
  run(/*rows=*/6, /*token_id=*/-1);
  EXPECT_EQ(model.forward_count(), capture_count);
  run(/*rows=*/5, /*token_id=*/4);
  EXPECT_EQ(model.forward_count(), capture_count);
  // A smaller bucket is captured independently, then the larger bucket is
  // reused with refreshed inputs and zeroed padding.
  run(/*rows=*/2, /*token_id=*/4);
  const int32_t after_shrink = model.forward_count();
  EXPECT_GT(after_shrink, capture_count);
  run(/*rows=*/6, /*token_id=*/9);
  EXPECT_EQ(model.forward_count(), after_shrink);

  args_.model_type("qwen3_5_text");
  args_.rope_scaling_mrope_section({2, 1, 1});
  MusaGraphExecutorImpl target(&model, args_, device_.unwrap(), options_);
  auto input = make_verify_params(device_.unwrap());
  input.clear_linear_attention_state();
  const torch::Tensor tokens = torch::arange(4, ints);
  const torch::Tensor positions = torch::arange(12, ints).view({3, 4});
  for (int32_t offset : {3, 7}) {
    // Preserve unequal mRoPE axes and refresh the VLM-prepared embeddings.
    input.embedding.input_embedding = torch::full({4, 256}, offset, bf16_);
    const ModelOutput expected =
        reference_model.forward(tokens, positions + offset, caches_, input);
    const int32_t before = model.forward_count();
    const ModelOutput output =
        target.run(tokens, positions + offset, caches_, input);
    device_.synchronize_default_stream();
    EXPECT_TRUE(
        torch::equal(output.hidden_states.cpu(), expected.hidden_states.cpu()));
    if (offset == 3) {
      EXPECT_GT(model.forward_count(), before + 1);
    } else {
      EXPECT_EQ(model.forward_count(), before);
    }
  }
}

TEST_F(MusaMtpGraphTest, VerifyCacheEvictsColdShapesAndRecapturesCorrectly) {
  const auto ints = bf16_.dtype(torch::kInt32);
  VerifyGraphModel model(bf16_);
  MusaGraphExecutorImpl executor(&model, args_, device_.unwrap(), options_);
  std::vector<std::pair<torch::Tensor, torch::Tensor>> outputs;
  outputs.reserve(26);
  const auto run = [&](int32_t token_count,
                       int32_t offset,
                       bool capture,
                       int32_t batch = 0) {
    if (batch == 0) {
      batch = (token_count + 1) / 2;
    }
    // Odd counts exercise ragged verification without padding.
    auto params = make_verify_params(device_.unwrap(), batch, token_count);
    params.clear_linear_attention_state();
    const torch::Tensor tokens = torch::arange(token_count, ints) + offset;
    const torch::Tensor positions = torch::ones_like(tokens);
    const int32_t before = model.forward_count();
    outputs.emplace_back(
        executor.run(tokens, positions, caches_, params).hidden_states.clone(),
        (tokens + positions + batch)
            .to(torch::kBFloat16)
            .unsqueeze(1)
            .expand({token_count, 256}));
    if (capture) {
      EXPECT_GT(model.forward_count(), before);
    } else {
      EXPECT_EQ(model.forward_count(), before);
    }
  };
  for (int32_t token_count = 2; token_count <= 17; ++token_count) {
    run(token_count, /*offset=*/0, /*capture=*/true);
  }
  run(/*token_count=*/2, /*offset=*/10, /*capture=*/false);
  device_.synchronize_default_stream();
  const c10::musa::MUSAStream replay_stream = c10::musa::getStreamFromPool();
  {
    c10::StreamGuard stream_guard(replay_stream.unwrap());
    run(/*token_count=*/2, /*offset=*/15, /*capture=*/false);
    replay_stream.synchronize();
  }
  run(/*token_count=*/18, /*offset=*/20, /*capture=*/true);
  run(/*token_count=*/2, /*offset=*/30, /*capture=*/false);
  run(/*token_count=*/3, /*offset=*/40, /*capture=*/true);
  // Distinct batch shapes both across and within a scheduler metadata bucket.
  run(/*token_count=*/8, /*offset=*/50, /*capture=*/false, /*batch=*/4);
  for (int32_t batch : {5, 6}) {
    run(/*token_count=*/8, /*offset=*/60, /*capture=*/true, batch);
    run(/*token_count=*/8, /*offset=*/70, /*capture=*/false, batch);
  }
  // Keep replay outputs in flight across eviction, then compare all results.
  for (size_t index = 0; index < outputs.size(); ++index) {
    EXPECT_TRUE(
        torch::equal(outputs[index].first.cpu(), outputs[index].second.cpu()))
        << "forward " << index;
  }

  args_.linear_key_head_dim(64);
  MusaGraphExecutorImpl eager_executor(
      &model, args_, device_.unwrap(), options_);
  auto unsupported = make_verify_params(device_.unwrap());
  unsupported.clear_linear_attention_state();
  const torch::Tensor tokens = torch::arange(4, ints);
  const torch::Tensor positions = torch::ones_like(tokens);
  const int32_t before_eager = model.forward_count();
  for (int32_t index = 0; index < 2; ++index) {
    const ModelOutput result =
        eager_executor.run(tokens, positions, caches_, unsupported);
    EXPECT_TRUE(torch::equal(result.hidden_states.cpu(),
                             (tokens + positions + 2)
                                 .to(torch::kBFloat16)
                                 .unsqueeze(1)
                                 .expand({4, 256})
                                 .cpu()));
  }
  EXPECT_EQ(model.forward_count(), before_eager + 2);

  args_.linear_key_head_dim(128);
  args_.n_heads(60);
  args_.n_kv_heads(10);
  options_.world_size(5);
  // Truncation produces valid-looking local 3/9 heads in both invalid cases.
  for (const auto& [key_heads, value_heads] :
       {std::pair{16, 45}, std::pair{15, 48}, std::pair{15, 45}}) {
    SCOPED_TRACE(::testing::Message() << key_heads << "/" << value_heads);
    args_.linear_num_key_heads(key_heads);
    args_.linear_num_value_heads(value_heads);
    MusaGraphExecutorImpl tp_executor(
        &model, args_, device_.unwrap(), options_);
    const bool graph_supported = key_heads == 15 && value_heads == 45;
    const int32_t before = model.forward_count();
    int32_t previous = before;
    for (int32_t index = 0; index < 2; ++index) {
      const ModelOutput result =
          tp_executor.run(tokens, positions, caches_, unsupported);
      EXPECT_TRUE(torch::equal(result.hidden_states.cpu(),
                               (tokens + positions + 2)
                                   .to(bf16_)
                                   .unsqueeze(1)
                                   .expand({4, 256})
                                   .cpu()));
      if (graph_supported && index == 0) {
        EXPECT_GT(model.forward_count(), before + 1);
      } else {
        EXPECT_EQ(model.forward_count(), previous + (graph_supported ? 0 : 1));
      }
      previous = model.forward_count();
    }
  }
}

TEST_F(MusaMtpGraphTest, CaptureRestoresAllSelectedLinearCheckpoints) {
  const auto floats = bf16_.dtype(torch::kFloat32);
  const auto ints = bf16_.dtype(torch::kInt32);
  options_.max_seqs_per_batch(8);
  options_.max_tokens_per_batch(8);
  const torch::Tensor conv = torch::arange(6 * 8 * 6, floats).view({6, 8, 6});
  const torch::Tensor ssm =
      torch::arange(6 * 4 * 2 * 2, floats).view({24, 2, 2});
  caches_.emplace_back(
      LinearAttentionKVCacheTensors{conv.clone(), ssm.clone()});
  std::vector<KVCache> reference_caches;
  reference_caches.reserve(1);
  reference_caches.emplace_back(
      LinearAttentionKVCacheTensors{conv.clone(), ssm.clone()});
  // K=3 owns four SSM checkpoint rows per logical convolution slot.
  const auto params = make_decode_params({1, 3}, device_.unwrap());
  const torch::Tensor positions = torch::ones({2}, ints);
  VerifyGraphModel model(bf16_);
  VerifyGraphModel reference_model(bf16_);
  MusaGraphExecutorImpl executor(&model, args_, device_.unwrap(), options_);
  for (int32_t value : {2, 5}) {
    const torch::Tensor tokens = torch::full({2}, value, ints);
    const ModelOutput expected =
        reference_model.forward(tokens, positions, reference_caches, params);
    // The first run prepares and captures before replay; only one forward's
    // state updates may survive. Subsequent runs must replay with fresh inputs.
    const ModelOutput output = executor.run(tokens, positions, caches_, params);
    device_.synchronize_default_stream();
    EXPECT_TRUE(torch::equal(caches_.back().get_conv_cache().cpu(),
                             reference_caches.back().get_conv_cache().cpu()));
    EXPECT_TRUE(torch::equal(caches_.back().get_ssm_cache().cpu(),
                             reference_caches.back().get_ssm_cache().cpu()));
    EXPECT_TRUE(
        torch::equal(output.hidden_states.cpu(), expected.hidden_states.cpu()));
  }
  EXPECT_GT(model.forward_count(), reference_model.forward_count());
}

TEST_F(MusaMtpGraphDeathTest, RejectsInvalidVerifyAcceptedCountsOnReplay) {
  // Re-exec instead of forking initialized MUSA and embedded Go runtimes.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto ints = bf16_.dtype(torch::kInt32);
  options_.max_seqs_per_batch(2);
  options_.max_tokens_per_batch(4);
  VerifyGraphModel model(bf16_);
  MusaGraphExecutorImpl executor(&model, args_, device_.unwrap(), options_);
  auto params = make_verify_params(device_.unwrap());
  const torch::Tensor tokens = torch::arange(4, ints);
  const torch::Tensor positions = torch::ones_like(tokens);
  executor.run(tokens, positions, caches_, params);
  const int32_t capture_count = model.forward_count();
  for (int32_t count : {0, 3}) {
    params.num_accepted_tokens_host = {count, 1};
    params.num_accepted_tokens = torch::tensor({count, 1}, ints);
    EXPECT_DEATH(executor.run(tokens, positions, caches_, params),
                 "spec verify accepted prefix length");
  }
  params.num_accepted_tokens_host = {1, 1};
  params.num_accepted_tokens = torch::ones({2}, ints);
  executor.run(tokens, positions, caches_, params);
  EXPECT_EQ(model.forward_count(), capture_count);
}

}  // namespace
}  // namespace xllm::runtime::musa
