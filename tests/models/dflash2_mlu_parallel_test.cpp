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

#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/parallel_state/collective_communicator.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/layers/mlu/deepseek_v32/multi_device_test_utils.h"
#include "core/layers/mlu/dflash2_decoder_layer.h"
#include "core/platform/device.h"
#include "core/runtime/dflash2_worker_impl.h"
#include "models/model_registry.h"

namespace xllm {

namespace {

ModelArgs make_args() {
  ModelArgs args;
  args.model_type("DFlash2DraftModel")
      .n_layers(1)
      .hidden_size(128)
      .intermediate_size(256)
      .hidden_act("silu")
      .head_dim(128)
      .n_heads(8)
      .n_kv_heads(2)
      .attention_bias(false)
      .max_position_embeddings(32)
      .rope_theta(10000)
      .rms_norm_eps(1e-5)
      .sliding_window(16)
      .num_speculative_tokens(/*value=*/3)
      .dflash2_conv_group_size(16)
      .dflash2_conv_kernel_size(2);
  return args;
}

torch::Tensor make_weight(int64_t rows, int64_t cols, int64_t offset) {
  return ((torch::arange(rows * cols, torch::kFloat32) + offset).remainder(31) -
          15)
             .reshape({rows, cols}) /
         256;
}

StateDict make_weights(const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  weights.emplace("self_attn.q_proj.weight", make_weight(1024, 128, 1));
  weights.emplace("self_attn.k_proj.weight", make_weight(256, 128, 3));
  weights.emplace("self_attn.v_proj.weight", make_weight(256, 128, 7));
  weights.emplace("self_attn.o_proj.weight", make_weight(128, 1024, 11));
  weights.emplace("mlp.gate_proj.weight", make_weight(256, 128, 13));
  weights.emplace("mlp.up_proj.weight", make_weight(256, 128, 17));
  weights.emplace("mlp.down_proj.weight", make_weight(128, 256, 19));
  for (const std::string& norm : {"self_attn.q_norm",
                                  "self_attn.k_norm",
                                  "input_layernorm",
                                  "post_attention_layernorm"}) {
    weights.emplace(norm + ".weight", torch::ones({128}, options));
  }
  for (const std::string& conv : {"attention_conv", "mlp_conv"}) {
    weights.emplace(conv + ".base_kernel",
                    torch::full({2, 2, 128}, 0.25, options));
    weights.emplace(conv + ".kernel_projection.weight",
                    make_weight(32, 128, 23));
  }
  for (auto& [name, weight] : weights) {
    weight = weight.to(options);
  }
  return StateDict(weights);
}

void compare_decoder(int32_t rank, int32_t port, int32_t cp_size) {
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  CollectiveCommunicator communicator(rank,
                                      /*world_size=*/8,
                                      /*dp_size=*/1,
                                      /*ep_size=*/1,
                                      cp_size,
                                      /*buffer_offset=*/0,
                                      /*needs_tp_cp_group=*/true);
  communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                     device.unwrap());
  const ParallelArgs& target = *communicator.parallel_args();
  const ParallelArgs draft =
      parallel_state::get_cp_independent_tp_parallel_args(target);
  ASSERT_EQ(target.cp_size(), cp_size);
  ASSERT_EQ(target.tp_group_->world_size(), 8 / cp_size);
  ASSERT_EQ(draft.cp_size(), 1);
  ASSERT_EQ(draft.tp_group_->world_size(), 8);
  ASSERT_EQ(draft.tp_group_->rank(), rank);

  const ModelArgs args = make_args();
  ModelContext context(draft, args, QuantArgs(), options);
  layer::mlu::DFlash2DecoderLayer decoder(context, /*layer_id=*/0);
  const StateDict weights = make_weights(options);
  decoder->load_state_dict(weights);
  decoder->verify_loaded_weights("layers.0.");

  ProcessGroup single(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs reference_parallel(/*rank=*/0, /*world_size=*/1, &single);
  reference_parallel.tp_group_ = &single;
  ModelContext reference_context(
      reference_parallel, args, QuantArgs(), options);
  layer::mlu::DFlash2DecoderLayer reference(reference_context, /*layer_id=*/0);
  reference->load_state_dict(weights);
  reference->verify_loaded_weights("layers.0.");

  const auto ints = options.dtype(torch::kInt32);
  layer::AttentionMetadata metadata{};
  metadata.is_dummy = false;
  metadata.is_chunked_prefill = true;
  metadata.max_query_len = 4;
  metadata.max_seq_len = 4;
  metadata.q_cu_seq_lens = torch::tensor({0, 4}, ints);
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.block_table = torch::tensor({0}, ints).reshape({1, 1});
  metadata.slot_mapping = torch::arange(4, ints);
  metadata.is_causal = false;
  metadata.non_causal_window_right = 3;
  KVCache cache(KVCacheTensors{torch::zeros({1, 1, 16, 128}, options),
                               torch::zeros({1, 1, 16, 128}, options)});
  KVCache reference_cache(
      KVCacheTensors{torch::zeros({1, 2, 16, 128}, options),
                     torch::zeros({1, 2, 16, 128}, options)});
  torch::Tensor positions = torch::arange(4, ints);
  torch::Tensor actual = make_weight(4, 128, 29).to(options);
  torch::Tensor expected = actual.clone();
  std::optional<torch::Tensor> actual_residual;
  std::optional<torch::Tensor> expected_residual;
  ModelInputParams input;
  actual = decoder->forward(
      actual, actual_residual, positions, metadata, cache, input);
  expected = reference->forward(
      expected, expected_residual, positions, metadata, reference_cache, input);
  ASSERT_TRUE(actual_residual.has_value());
  ASSERT_TRUE(expected_residual.has_value());
  EXPECT_TRUE(torch::allclose(actual, expected, 0.03, 0.003));
  // BF16 rounds the eight partial o-projection matmuls before their sum.
  // Bound both aggregate and peak error relative to the reference scale;
  // elementwise relative error is unstable near residual cancellations.
  const torch::Tensor reference_residual =
      expected_residual->to(torch::kFloat32);
  const torch::Tensor residual_error =
      actual_residual->to(torch::kFloat32) - reference_residual;
  const float reference_norm = reference_residual.norm().item<float>();
  ASSERT_GT(reference_norm, 0.0f);
  EXPECT_LT(residual_error.norm().item<float>() / reference_norm, 0.01f);
  EXPECT_LT(residual_error.abs().max().item<float>(),
            0.01f * reference_residual.abs().max().item<float>());
  // TP8 replicates each of the two KV heads across four adjacent ranks.
  EXPECT_TRUE(torch::allclose(
      cache.get_k_cache(),
      reference_cache.get_k_cache().slice(/*dim=*/1, rank / 4, rank / 4 + 1),
      0.03,
      0.003));
  EXPECT_TRUE(torch::allclose(
      cache.get_v_cache(),
      reference_cache.get_v_cache().slice(/*dim=*/1, rank / 4, rank / 4 + 1),
      0.03,
      0.003));

  // The draft sampler must share noise over the whole draft TP domain.
  torch::Tensor noise = torch::full({2, 3, 4}, rank, options);
  draft.tp_group_->broadcast(noise, /*root_rank=*/0);
  EXPECT_EQ(noise.abs().max().item<float>(), 0.0f);
}

void check_cp_dp_groups(int32_t rank, int32_t port, bool needs_tp_cp_group) {
  Device device(rank);
  device.set_device();
  CollectiveCommunicator communicator(rank,
                                      /*world_size=*/4,
                                      /*dp_size=*/2,
                                      /*ep_size=*/4,
                                      /*cp_size=*/2,
                                      /*buffer_offset=*/0,
                                      needs_tp_cp_group);
  communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                     device.unwrap());
  const ParallelArgs& target = *communicator.parallel_args();
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  if (needs_tp_cp_group) {
    const ParallelArgs draft =
        parallel_state::get_cp_independent_tp_parallel_args(target);
    EXPECT_EQ(draft.tp_size(), 2);
    EXPECT_EQ(draft.cp_size(), 1);
    torch::Tensor value = torch::full({1}, rank + 1, options);
    draft.tp_group_->allreduce(value);
    EXPECT_EQ(value.item<float>(), rank < 2 ? 3.0f : 7.0f);
  } else {
    EXPECT_EQ(target.tp_cp_group_, nullptr);
  }
  // Both port layouts must leave DP and the subsequent EP group usable.
  torch::Tensor dp_value = torch::ones({1}, options);
  target.dp_local_process_group_->allreduce(dp_value);
  EXPECT_EQ(dp_value.item<float>(), 2.0f);
  torch::Tensor ep_value = torch::full({1}, rank + 1, options);
  target.moe_ep_group_->allreduce(ep_value);
  EXPECT_EQ(ep_value.item<float>(), 10.0f);
}

TEST(DFlash2MluParallelTest, CpDpTargetWithoutDraftSkipsDraftTpGroup) {
  test::run_multi_device(/*ranks=*/4, [](int32_t rank, int32_t port) {
    check_cp_dp_groups(rank, port, /*needs_tp_cp_group=*/false);
  });
}

TEST(DFlash2MluParallelTest, CpDpDraftTpGroupKeepsDpBatchesIsolated) {
  test::run_multi_device(/*ranks=*/4, [](int32_t rank, int32_t port) {
    check_cp_dp_groups(rank, port, /*needs_tp_cp_group=*/true);
  });
}

// Full worker integration requires real, compatible GLM5-Next and DFlash2
// checkpoints. No worker state, model registration or private method is
// patched.
TEST(DFlash2MluParallelTest, ValidationTokensAgreeAcrossFormerCpRanks) {
  const char* target_path = std::getenv("XLLM_DFLASH2_CP_TARGET_MODEL");
  const char* draft_path = std::getenv("XLLM_DFLASH2_CP_DRAFT_MODEL");
  if (target_path == nullptr || draft_path == nullptr) {
    GTEST_SKIP() << "Set XLLM_DFLASH2_CP_TARGET_MODEL and "
                    "XLLM_DFLASH2_CP_DRAFT_MODEL to compatible checkpoints";
  }
  test::run_multi_device(/*ranks=*/8, [=](int32_t rank, int32_t port) {
    Device device(rank);
    device.set_device();
    device.init_device_context();
    torch::InferenceMode inference;
    CollectiveCommunicator communicator(rank,
                                        /*world_size=*/8,
                                        /*dp_size=*/1,
                                        /*ep_size=*/8,
                                        /*cp_size=*/4,
                                        /*buffer_offset=*/0,
                                        /*needs_tp_cp_group=*/true);
    communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                       device.unwrap());
    const ParallelArgs& target = *communicator.parallel_args();
    SchedulerConfig::get_instance().enable_chunked_prefill(true);
    KVCacheConfig::get_instance().block_size(16);
    runtime::Options options;
    options.model_path(target_path)
        .draft_model_path(std::string(draft_path))
        .backend("llm")
        .tp_size(2)
        .cp_size(4)
        .dp_size(1)
        .ep_size(8)
        .speculative_algorithm("DFlash2")
        .num_speculative_tokens(3)
        .enable_speculative_decode(true)
        .block_size(16)
        .max_seqs_per_batch(2)
        .max_tokens_per_batch(1024)
        .enable_graph(false)
        .enable_schedule_overlap(true);
    DFlash2WorkerImpl worker(target, device.unwrap(), options, WorkerType::LLM);
    ASSERT_TRUE(worker.init_model(
        target_path, /*random_seed=*/21, MasterStatus::WAKEUP));
    ASSERT_TRUE(worker.init_model(
        draft_path, /*random_seed=*/21, MasterStatus::WAKEUP));
    const ModelArgs& args = worker.context_.get_model_args();
    ASSERT_EQ(args.model_type(), "glm5_next");
    ASSERT_GT(args.n_layers(), 0);
    ASSERT_GT(args.vocab_size(), 4);
    ASSERT_TRUE(worker.get_optimization_config().enable_spec_token_broadcast);
    KVCacheCapacity capacity;
    capacity.n_blocks(160)
        .block_size(16)
        .num_linear_state_blocks(3)
        .linear_conv_state_len(args.linear_conv_kernel_dim() - 1 + 3)
        .linear_ssm_checkpoint_stride(4)
        .kpool_tail_len(args.index_kpool() + 8);
    const LinearAttentionPolicy policy =
        ModelRegistry::get_linear_attention_policy(args.model_type());
    const KVCacheShape shape(
        capacity,
        args,
        /*world_size=*/2,
        linear_head_tp_size(policy, /*attention_tp_size=*/2, /*cp_size=*/4));
    ASSERT_TRUE(worker.allocate_kv_cache(shape));
    const auto make_input = [&](bool prefill) {
      constexpr int32_t kPrefix = 512;
      const int32_t width = prefill ? kPrefix : 1;
      const int32_t begin = prefill ? 0 : kPrefix;
      ForwardInput input;
      input.token_ids_host =
          torch::arange(2 * width, torch::kInt32).remainder(4).add(1);
      input.positions_host =
          torch::arange(begin, begin + width, torch::kInt32).repeat({2});
      input.token_ids = input.token_ids_host;
      input.positions = input.positions_host;
      auto& params = input.input_params;
      params.meta.batch_forward_type =
          prefill ? BatchForwardType::PREFILL : BatchForwardType::DECODE;
      params.meta.num_sequences = 2;
      params.meta.actual_num_sequences = 2;
      params.meta.q_max_seq_len = width;
      params.meta.kv_max_seq_len = begin + width;
      params.parallel.dp_global_token_nums = {2 * width};
      params.parallel.dp_is_decode = {prefill ? 0 : 1};
      params.attention.host.q_seq_lens = {0, width, 2 * width};
      params.attention.host.q_cu_seq_lens = params.attention.host.q_seq_lens;
      params.attention.host.kv_seq_lens = {
          0, begin + width, 2 * (begin + width)};
      params.attention.host.kv_cache_tokens_nums = {begin, begin};
      params.attention.host.block_tables =
          torch::arange(1, 129, torch::kInt32).view({2, 64});
      params.attention.host.positions = input.positions_host;
      params.attention.device.q_seq_lens =
          torch::tensor(params.attention.host.q_seq_lens, torch::kInt32);
      params.attention.device.q_cu_seq_lens =
          params.attention.device.q_seq_lens;
      params.attention.device.kv_seq_lens =
          torch::tensor(params.attention.host.kv_seq_lens, torch::kInt32);
      params.attention.device.kv_cache_tokens_nums =
          torch::tensor({begin, begin}, torch::kInt32);
      params.attention.device.block_tables = params.attention.host.block_tables;
      params.attention.device.new_cache_slots = torch::cat(
          {torch::arange(16 + begin, 16 + begin + width, torch::kInt32),
           torch::arange(
               65 * 16 + begin, 65 * 16 + begin + width, torch::kInt32)});
      params.embedding.linear_state_ids = {1, 2};
      params.embedding.embedding_ids = {1, 65};
      params.embedding.request_ids = {"cp-request-0", "cp-request-1"};
      params.linear_state_validity_mask = {prefill ? 0 : 1, prefill ? 0 : 1};
      params.execution_batch.num_reqs = 2;
      params.execution_batch.num_tokens = 2 * width;
      params.execution_batch.num_scheduled_tokens = {width, width};
      params.execution_batch.num_computed_tokens = {begin, begin};
      params.execution_batch.is_prefilling = {prefill ? 1 : 0, prefill ? 1 : 0};
      params.execution_batch.query_start_loc = params.attention.host.q_seq_lens;
      RequestSamplingParam sampling;
      sampling.do_sample = !prefill;
      sampling.temperature = prefill ? 0.0f : 0.8f;
      input.sampling_params.init({&sampling, &sampling},
                                 {width - 1, 2 * width - 1},
                                 {0, 1},
                                 {{}, {}},
                                 {{}, {}},
                                 {0, 0});
      input.decoder_sampling_params = input.sampling_params;
      input.sample_sequence_ids = params.embedding.request_ids;
      return input;
    };
    // Use the production upload/prepare path and public step dispatch. Overlap
    // mode retains accepted output on non-driver ranks, without changing flags
    // or injecting workers after construction.
    WorkerImpl& public_worker = worker;
    ASSERT_TRUE(public_worker.step_async(make_input(/*prefill=*/true))
                    .get()
                    .has_value());
    if (public_worker.is_driver()) {
      public_worker.get_last_step_result();
    }
    for (int32_t round = 0; round < 3; ++round) {
      torch::manual_seed(1234 + 31 * rank + round);
      auto output =
          public_worker.step_async(make_input(/*prefill=*/false)).get();
      ASSERT_TRUE(output.has_value());
      ASSERT_TRUE(output->failed_request_ids.empty());
      if (output->ready_event != nullptr) {
        ASSERT_TRUE(output->ready_event->synchronize());
      }
      if (public_worker.is_driver()) {
        public_worker.get_last_step_result();
      }
      const torch::Tensor accepted =
          output->sample_output.next_tokens.to(device.unwrap(), torch::kInt64);
      ASSERT_EQ(accepted.sizes(), torch::IntArrayRef({2, 4}));
      std::vector<torch::Tensor> all_ranks;
      all_ranks.reserve(8);
      for (int32_t peer = 0; peer < 8; ++peer) {
        all_ranks.emplace_back(torch::empty_like(accepted));
      }
      target.tp_cp_group_->allgather(accepted, all_ranks);
      for (const auto& tokens : all_ranks) {
        EXPECT_TRUE(torch::equal(tokens, all_ranks.front()))
            << "accepted length or bonus differs across former CP ranks";
      }
      EXPECT_TRUE(accepted.select(1, 0).ge(0).all().item<bool>());
    }
  });
}

TEST(DFlash2MluParallelTest, TargetCp8DraftTp8MatchesUnshardedDecoder) {
  test::run_multi_device(/*ranks=*/8, [](int32_t rank, int32_t port) {
    compare_decoder(rank, port, /*cp_size=*/8);
  });
}

TEST(DFlash2MluParallelTest, TargetCp4Tp2DraftTp8MatchesUnshardedDecoder) {
  test::run_multi_device(/*ranks=*/8, [](int32_t rank, int32_t port) {
    compare_decoder(rank, port, /*cp_size=*/4);
  });
}

}  // namespace
}  // namespace xllm
