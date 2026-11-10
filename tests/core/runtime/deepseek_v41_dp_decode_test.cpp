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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/runtime/draft_model_config.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "models/model_registry.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "platform/stream.h"
#include "tests/core/forked_rank_session.h"
#include "tests/core/runtime/deepseek_v41_test_utils.h"
#include "util/env_var.h"
#include "util/net.h"

namespace xllm {
namespace {
constexpr int32_t kWorldSize = 8;
constexpr int32_t kTpSize = 4;
constexpr int32_t kLayers = 3;
constexpr int32_t kExperts = 128;
constexpr int32_t kSkip = 77;
class DraftConfigDirectory final {
 public:
  DraftConfigDirectory() {
    char pattern[] = "/tmp/xllm-dp-dspark-config-XXXXXX";
    const char* path = mkdtemp(pattern);
    CHECK(path != nullptr);
    path_ = path;
    std::ofstream config(path_ + "/config.json");
    config << R"({"dspark_block_size":5})";
    CHECK(config.good());
  }
  ~DraftConfigDirectory() { std::filesystem::remove_all(path_); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

int32_t run_rank(int32_t rank,
                 const std::array<int32_t, 15>& ports,
                 const std::string& config_path) {
  if (Platform::device_count() < kWorldSize) {
    return kSkip;
  }
  Device runtime_device(rank);
  runtime_device.set_device();
  const auto compute_stream =
      util::get_bool_env("XLLM_DP_REPRO_DEFAULT_STREAM", true)
          ? runtime_device.current_stream()
          : runtime_device.get_stream_from_pool();
  const auto compute_guard = compute_stream->set_stream_guard();
  const torch::Device device = runtime_device.unwrap();
  const torch::NoGradGuard no_grad;
  torch::manual_seed(41);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  auto global = create_process_group(rank,
                                     kWorldSize,
                                     kWorldSize,
                                     ports[0],
                                     /*trans=*/false,
                                     /*host=*/"127.0.0.1",
                                     /*group_name=*/"v41_ep",
                                     device);
  auto tp = create_process_group(rank,
                                 kWorldSize,
                                 kTpSize,
                                 ports[1 + rank / kTpSize],
                                 /*trans=*/false,
                                 /*host=*/"127.0.0.1",
                                 /*group_name=*/"v41_tp",
                                 device);
  auto dp = create_process_group(rank,
                                 kWorldSize,
                                 2,
                                 ports[3 + rank % kTpSize],
                                 /*trans=*/true,
                                 /*host=*/"127.0.0.1",
                                 /*group_name=*/"v41_dp",
                                 device);
  auto single = create_process_group(rank,
                                     kWorldSize,
                                     1,
                                     ports[7 + rank],
                                     /*trans=*/false,
                                     /*host=*/"127.0.0.1",
                                     /*group_name=*/"v41_single",
                                     device);
  ParallelArgs parallel(rank, kWorldSize, /*dp_size=*/2, global.get());
  parallel.ep_size(kWorldSize);
  parallel.tp_group_ = tp.get();
  parallel.dp_local_process_group_ = dp.get();
  parallel.moe_ep_group_ = global.get();
  parallel.moe_tp_group_ = single.get();
  parallel.single_rank_group_ = single.get();
  const bool full_geometry =
      util::get_bool_env("XLLM_DP_REPRO_FULL_GEOMETRY", false);
  const int64_t hidden_size = full_geometry ? 5120 : 256;
  const int64_t q_rank = full_geometry ? 1280 : 16;
  const int64_t heads = full_geometry ? 64 : 8;
  const int64_t groups = full_geometry ? 8 : 4;
  const int64_t o_rank = full_geometry ? 1024 : 8;
  ModelArgs args;
  args.model_type("deepseek_v41")
      .dtype("bfloat16")
      .hidden_size(hidden_size)
      .vocab_size(128)
      .n_layers(kLayers)
      .dspark_num_layers(kLayers)
      .dspark_block_size(5)
      .markov_rank(8)
      .n_heads(heads)
      .n_kv_heads(1)
      .head_dim(512)
      .q_lora_rank(q_rank)
      .qk_rope_head_dim(64)
      .o_groups(groups)
      .o_lora_rank(o_rank)
      .window_size(16)
      .compress_ratios({0, 0, 0})
      .max_position_embeddings(512)
      .compress_rope_theta(10000)
      .factor(1)
      .beta_fast(32)
      .beta_slow(1)
      .hc_mult(4)
      .hc_sinkhorn_iters(20)
      .hc_eps(1e-6)
      .rms_norm_eps(1e-6)
      .n_shared_experts(1)
      .n_routed_experts(kExperts)
      .num_experts_per_tok(3)
      .n_group(1)
      .topk_group(1)
      .routed_scaling_factor(1)
      .norm_topk_prob(true)
      .moe_intermediate_size(128)
      .hidden_act("silu")
      .scoring_func("sqrtsoftplus")
      .topk_method("noaux_tc")
      .swiglu_limit(10);

  runtime::Options runtime_options;
  runtime_options.is_draft_engine(true)
      .speculative_algorithm("DSpark")
      .num_speculative_tokens(5)
      .block_size(128)
      .dp_size(2)
      .world_size(kWorldSize)
      .model_path(config_path);
  ModelConfig::get_instance().model_impl("native");
  configure_block_diffusion_model(args, runtime_options);
  CHECK_EQ(args.dummy_token_count(), 5);
  const ModelContext context(parallel, args, QuantArgs(), options);
  auto model = create_llm_model(context);
  CHECK(model != nullptr);
  model->load_model(std::make_unique<test::TensorModelLoader>(
      test::make_deepseek_v41_dspark_weights(args, options)));
  DeepSeekV4KVCacheTensors cache_tensors;
  cache_tensors.swa_cache = torch::zeros({8, 128, 1, 512}, options);
  std::vector<KVCache> caches;
  caches.reserve(kLayers);
  std::vector<torch::Tensor> original_caches;
  original_caches.reserve(kLayers);
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache = cache_tensors.swa_cache.clone();
    caches.emplace_back(std::move(tensors));
    original_caches.emplace_back(caches.back().get_swa_cache().clone());
  }
  mlu::MluGraphExecutorImpl executor(
      model.get(), args, device, runtime_options);
  executor.set_dp_process_group(dp.get());
  torch::Tensor expected_eager;
  for (int32_t step = 0; step < 5; ++step) {
    const bool peer_idle = step == 1 || step == 3;
    const bool first_shard_idle = step == 2;
    const bool idle =
        (peer_idle && rank >= kTpSize) || (first_shard_idle && rank < kTpSize);
    const bool empty_input = step == 3 && idle;
    const int64_t idle_rows = args.dummy_token_count();
    ModelInputParams input;
    input.clear_linear_attention_state();
    input.use_dflash_proposal_attention = true;
    input.meta.batch_forward_type = BatchForwardType::DECODE;
    input.meta.num_sequences = idle ? 0 : 1;
    input.meta.q_max_seq_len = idle ? (idle_rows == 1 ? 0 : 1) : 5;
    input.meta.kv_max_seq_len = idle ? 0 : 10;
    input.parallel.dp_global_token_nums = {first_shard_idle ? 0 : 5,
                                           peer_idle ? 0 : 5};
    input.parallel.raw_dp_global_token_nums =
        input.parallel.dp_global_token_nums;
    input.parallel.dp_is_decode = {1, 1};
    input.attention.host.q_seq_lens =
        idle ? (idle_rows == 1 ? std::vector<int32_t>{0}
                               : std::vector<int32_t>(idle_rows, 1))
             : std::vector<int32_t>{0, 5};
    input.attention.host.kv_seq_lens =
        idle ? (idle_rows == 1 ? std::vector<int32_t>{0}
                               : std::vector<int32_t>(idle_rows, 1))
             : std::vector<int32_t>{0, 10};
    input.attention.device.q_seq_lens =
        torch::tensor(input.attention.host.q_seq_lens, ints);
    input.attention.device.kv_seq_lens =
        torch::tensor(input.attention.host.kv_seq_lens, ints);
    input.attention.device.new_cache_slots =
        torch::full({idle ? 0 : 5}, -1, ints);
    input.multi_block_tables = {
        torch::zeros({idle ? 0 : 1, 2}, ints.device(torch::kCPU))};
    const torch::Tensor tokens = empty_input
                                     ? torch::empty({0}, ints)
                                     : (idle ? torch::ones({idle_rows}, ints)
                                             : torch::arange(3, 8, ints));
    const torch::Tensor positions = empty_input
                                        ? torch::empty({0}, ints)
                                        : (idle ? torch::arange(idle_rows, ints)
                                                : torch::arange(5, 10, ints));
    LOG(INFO) << "V4.1 DP proposal step=" << step << " rank=" << rank
              << " idle=" << idle;
    if (step == 0) {
      expected_eager = model->forward(tokens, positions, caches, input)
                           .hidden_states.to(torch::kCPU);
    }
    // Exercise the model's own empty-input materialization without graph
    // preprocessing; the other idle steps use Worker-shaped five-row inputs.
    const ModelOutput output =
        step == 3 ? model->forward(tokens, positions, caches, input)
                  : executor.run(tokens, positions, caches, input);
    if (!idle) {
      const torch::Tensor actual = output.hidden_states.to(torch::kCPU);
      CHECK(torch::isfinite(actual).all().item<bool>());
      const torch::Tensor error =
          (actual.to(torch::kFloat32) - expected_eager.to(torch::kFloat32))
              .abs();
      LOG(INFO) << "Graph/peer DP error step=" << step << " rank=" << rank
                << " max=" << error.max().item<float>()
                << " mean=" << error.mean().item<float>();
      const float relative_error =
          error.norm().item<float>() /
          expected_eager.to(torch::kFloat32).norm().item<float>();
      // Capture and replay must agree with eager, including after idle shards
      // rejoin. Comparing replay only with capture can hide a persistent error.
      CHECK(step == 0 || step == 4
                ? torch::allclose(actual, expected_eager, 0.02, 0.02)
                : relative_error < 0.02)
          << "Graph execution and peer DP activity must preserve the eager "
             "output: step="
          << step << " rank=" << rank << " relative error=" << relative_error;
    }
    for (int32_t layer = 0; layer < kLayers; ++layer) {
      CHECK(
          torch::equal(caches[layer].get_swa_cache(), original_caches[layer]));
    }
    torch::Tensor barrier = torch::ones({1}, ints);
    global->allreduce(barrier);
    CHECK_EQ(barrier.to(torch::kCPU).item<int32_t>(), kWorldSize);
  }
  return 0;
}

TEST(DeepseekV41DpDecodeTest, IdleShardPreservesProposalAndGraphReplay) {
  const DraftConfigDirectory config;
  std::array<int32_t, 15> ports{};
  for (int32_t& port : ports) {
    int32_t candidate = net::get_local_free_port();
    while (std::find(ports.begin(), ports.end(), candidate) != ports.end()) {
      candidate = net::get_local_free_port();
    }
    ASSERT_GT(candidate, 0);
    port = candidate;
  }
  test::ForkedRankSession session(
      kWorldSize,
      ports[0],
      {{"IdleShardPreservesProposalAndGraphReplay",
        [&ports, &config](int32_t rank, int32_t /*unused*/) {
          return run_rank(rank, ports, config.path());
        }}},
      /*cleanup=*/{},
      /*timeout=*/std::chrono::seconds(90));
  const auto result = session.run("IdleShardPreservesProposalAndGraphReplay");
  test::expect_rank_result(result);
  test::expect_rank_result(session.close());
}

}  // namespace
}  // namespace xllm
