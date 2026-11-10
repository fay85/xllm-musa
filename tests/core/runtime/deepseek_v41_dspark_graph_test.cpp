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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "core/framework/model_context.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "models/model_registry.h"
#include "tests/core/layers/mlu/tests_utils.h"
#include "tests/core/runtime/deepseek_v41_test_utils.h"

namespace xllm {
namespace {

TEST(DeepseekV41DSparkGraphTest,
     ReplaysProposalWithFreshRowsPositionsAndPages) {
  const torch::NoGradGuard no_grad;
  torch::manual_seed(41);
  const torch::Device device("mlu:0");
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  ModelArgs args;
  args.model_type("deepseek_v41_dspark")
      .dtype("bfloat16")
      .hidden_size(256)
      .vocab_size(128)
      .n_layers(1)
      .dspark_num_layers(1)
      .dspark_block_size(5)
      .markov_rank(8)
      .n_heads(8)
      .n_kv_heads(1)
      .head_dim(512)
      .q_lora_rank(16)
      .qk_rope_head_dim(64)
      .o_groups(1)
      .o_lora_rank(8)
      .window_size(16)
      .compress_ratios({0})
      .max_position_embeddings(512)
      .compress_rope_theta(10000)
      .factor(1)
      .beta_fast(32)
      .beta_slow(1)
      .hc_mult(4)
      .hc_sinkhorn_iters(20)
      .hc_eps(1e-6)
      .rms_norm_eps(1e-6)
      .n_routed_experts(4)
      .num_experts_per_tok(2)
      .n_group(2)
      .topk_group(2)
      .routed_scaling_factor(1)
      .norm_topk_prob(true)
      .moe_intermediate_size(128)
      .hidden_act("silu")
      .scoring_func("sqrtsoftplus")
      .topk_method("noaux_tc")
      .swiglu_limit(10);
  layer::test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  const ModelContext context(parallel, args, QuantArgs(), options);
  auto model = create_llm_model(context);
  ASSERT_NE(model, nullptr);
  ASSERT_TRUE(model->requires_graph_forward_metadata());

  model->load_model(std::make_unique<test::TensorModelLoader>(
      test::make_deepseek_v41_dspark_weights(args, options)));
  DeepSeekV4KVCacheTensors cache_tensors;
  cache_tensors.swa_cache = torch::randn({8, 128, 1, 512}, options) * 0.05;
  std::vector<KVCache> caches;
  caches.reserve(1);
  caches.emplace_back(cache_tensors);
  const torch::Tensor original_cache = caches[0].get_swa_cache().clone();

  runtime::Options runtime_options;
  runtime_options.is_draft_engine(true)
      .speculative_algorithm("DSpark")
      .num_speculative_tokens(5)
      .block_size(128);
  mlu::GraphPlannerConfig config;
  config.options = runtime_options;
  config.capabilities = ModelRegistry::get_mtp_capabilities(args.model_type());
  config.state_required = model->requires_graph_forward_metadata();
  config.linear_state_required = has_linear_attention_layers(args);
  config.max_tokens = 64;
  const mlu::MluGraphPlanner planner(config);
  std::unique_ptr<mlu::MluGraph> graph;
  mlu::GraphKey captured_key;
  for (int32_t step = 0; step < 4; ++step) {
    // Warm up the padded shape, then capture three requests in a four-request
    // bucket. Replay grows and shrinks that batch while the SWA ring wraps.
    const int32_t requests = step % 2 == 0 ? 4 : 3;
    const int32_t begin = step == 0 ? 2 : (step == 1 ? 7 : 128 * step + 1);
    ModelInputParams input;
    input.clear_linear_attention_state();
    input.use_dflash_proposal_attention = true;
    input.meta.batch_forward_type = BatchForwardType::DECODE;
    input.meta.num_sequences = requests;
    input.meta.q_max_seq_len = 5;
    input.meta.kv_max_seq_len = begin + 5;
    input.parallel.dp_global_token_nums = {requests * 5};
    input.parallel.dp_is_decode = {1};
    input.attention.host.q_seq_lens.reserve(requests + 1);
    input.attention.host.kv_seq_lens.reserve(requests + 1);
    for (int32_t row = 0; row <= requests; ++row) {
      input.attention.host.q_seq_lens.emplace_back(row * 5);
      input.attention.host.kv_seq_lens.emplace_back(row * (begin + 5));
    }
    input.attention.device.q_seq_lens =
        torch::tensor(input.attention.host.q_seq_lens, ints);
    input.attention.device.kv_seq_lens =
        torch::tensor(input.attention.host.kv_seq_lens, ints);
    input.attention.device.new_cache_slots =
        torch::full({requests * 5}, -1, ints);
    // Remap every request onto a different physical ring on each replay.
    input.multi_block_tables = {(torch::arange(requests * 2, ints) + step * 2)
                                    .remainder(8)
                                    .view({requests, 2})};
    const torch::Tensor tokens = torch::randint(128, {requests * 5}, ints);
    const torch::Tensor positions =
        torch::arange(begin, begin + 5, options.dtype(torch::kInt64))
            .repeat({requests});
    const ModelOutput eager = model->forward(tokens, positions, caches, input);
    const torch::Tensor expected = eager.hidden_states.clone();
    if (step == 0) {
      continue;
    }
    const auto decision = planner.plan({tokens.numel(), input});
    ASSERT_TRUE(std::holds_alternative<mlu::GraphPlan>(decision));
    const auto& plan = std::get<mlu::GraphPlan>(decision);
    ASSERT_EQ(plan.layout.padded_num_tokens, 20);
    ModelOutput output;
    if (graph == nullptr) {
      auto buffers = std::make_unique<mlu::GraphPersistentParam>(
          tokens,
          positions,
          input,
          plan.layout,
          /*main_block_table_columns=*/4,
          config.capabilities.graph_dynamic_multi_block_tables);
      graph = std::make_unique<mlu::MluGraph>(
          std::move(buffers), config.capabilities.enable_piecewise_graph);
      graph->update_input_buffer(
          model.get(), tokens, positions, input, plan.layout);
      output = graph->capture(model.get(),
                              caches,
                              torch_mlu::graph_pool_handle(),
                              torch_mlu::getStreamFromPool(false, 0),
                              runtime_options);
      ASSERT_GT(graph->num_graphs(), 0);
      ASSERT_EQ(graph->num_eager_segments(), 1);
      captured_key = plan.key;
    } else {
      ASSERT_TRUE(captured_key == plan.key);
      graph->update_input_buffer(
          model.get(), tokens, positions, input, plan.layout);
      output = graph->replay();
    }
    torch_mlu::synchronize();
    EXPECT_TRUE(
        torch::allclose(output.hidden_states.narrow(0, 0, tokens.numel()),
                        expected,
                        0.02,
                        0.02));
    EXPECT_TRUE(torch::equal(caches[0].get_swa_cache(), original_cache));
  }
}

}  // namespace
}  // namespace xllm
