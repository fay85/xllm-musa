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

#include "core/runtime/mlu_graph_planner.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "models/llm/mlu/mtp_topk_state.h"

namespace xllm::mlu {
namespace {

GraphPlannerConfig qwen_config() {
  GraphPlannerConfig config;
  config.options.block_size(/*value=*/16);
  config.capabilities.graph_cache_pad_slot = -1;
  config.max_tokens = 64;
  return config;
}

// A draft engine whose model supports both grouped graphs and native index
// sharing is the only configuration that admits top-k reuse steps.
GraphPlannerConfig topk_reuse_config() {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true);
  config.capabilities.supports_grouped_mtp_graph = true;
  config.capabilities.supports_native_index_share_for_iteration = true;
  return config;
}

ModelInputParams decode_input(int64_t history, int64_t rows = 3) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = rows;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = history;
  params.parallel.dp_global_token_nums = {static_cast<int32_t>(rows)};
  params.parallel.dp_is_decode = {1};
  for (int64_t row = 0; row <= rows; ++row) {
    params.attention.host.q_seq_lens.push_back(static_cast<int32_t>(row));
  }
  params.attention.device.block_tables = torch::zeros({rows, 2}, torch::kInt32);
  return params;
}

ModelInputParams grouped_input(int64_t rows = 3) {
  ModelInputParams params = decode_input(/*history=*/16, rows);
  params.attention.device.q_seq_lens = torch::zeros({rows + 1}, torch::kInt32);
  params.attention.device.kv_seq_lens = torch::zeros({rows + 1}, torch::kInt32);
  params.attention.device.new_cache_slots = torch::zeros({rows}, torch::kInt32);
  return params;
}

TEST(MluGraphPlannerTest, ModelManagedCacheGraphDoesNotRequireSlots) {
  GraphPlannerConfig config = qwen_config();
  config.capabilities.supports_grouped_mtp_graph = true;
  config.capabilities.graph_requires_cache_slots = false;
  config.state_required = true;
  ModelInputParams params = grouped_input();
  params.multi_block_tables = {params.attention.device.block_tables};
  params.attention.device.block_tables = torch::Tensor();
  params.attention.device.new_cache_slots = torch::Tensor();
  const MluGraphPlanner planner(config);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));

  config.capabilities.graph_requires_cache_slots = true;
  auto decision = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<EagerReason>(decision));
  EXPECT_EQ(std::get<EagerReason>(decision), EagerReason::UNSUPPORTED_INPUT);
  params.attention.device.new_cache_slots = torch::empty({0}, torch::kInt32);
  decision = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<EagerReason>(decision));
  EXPECT_EQ(std::get<EagerReason>(decision), EagerReason::INVALID_LAYOUT);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));

  params.multi_block_tables[0] = params.multi_block_tables[0].slice(0, 0, 2);
  decision = planner.plan({3, params});
  ASSERT_TRUE(std::holds_alternative<EagerReason>(decision));
  EXPECT_EQ(std::get<EagerReason>(decision), EagerReason::INVALID_LAYOUT);
}

void expect_eager(const MluGraphPlanner& planner,
                  int64_t tokens,
                  const ModelInputParams& params,
                  EagerReason reason) {
  const GraphDecision decision = planner.plan({tokens, params});
  ASSERT_TRUE(std::holds_alternative<EagerReason>(decision));
  EXPECT_EQ(std::get<EagerReason>(decision), reason);
}

class PlannerTopkState final : public MtpTopkState {
 public:
  int64_t num_rows() const override { return 1; }
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  MtpTopkStatePtr to(const torch::Device& /*device*/) const override {
    return nullptr;
  }
  MtpTopkStatePtr index_select_rows(
      const torch::Tensor& /*index*/) const override {
    return nullptr;
  }
};

// A non-native implementation proves admission depends on the graph contract,
// not the MLU model's concrete carrier or per-layer tensor layout.
class PlannerGraphTopkState final : public MtpTopkGraphState {
 public:
  explicit PlannerGraphTopkState(std::vector<int64_t> schema = {7, 11})
      : schema_(std::move(schema)) {}

  int64_t num_rows() const override { return 3; }
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  MtpTopkStatePtr to(const torch::Device& /*device*/) const override {
    return std::make_shared<PlannerGraphTopkState>(schema_);
  }
  MtpTopkStatePtr index_select_rows(
      const torch::Tensor& /*index*/) const override {
    return nullptr;
  }
  std::vector<int64_t> graph_schema() const override { return schema_; }
  std::unique_ptr<MtpTopkGraphBuffer> make_graph_buffer(
      int64_t /*rows*/) const override {
    return nullptr;
  }

 private:
  std::vector<int64_t> schema_;
};

TEST(MluGraphPlannerTest, AcceptsBackendIndependentTopkGraphContract) {
  GraphPlannerConfig config = topk_reuse_config();
  ModelInputParams params = grouped_input();
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  const auto decision = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  EXPECT_EQ(std::get<GraphPlan>(decision).key.mtp_topk_schema,
            (std::vector<int64_t>{7, 11}));
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, DFlash2AdmitsOnlyCompleteProposalBlocks) {
  GraphPlannerConfig config = topk_reuse_config();
  config.options.speculative_algorithm(/*value=*/"DFlash2")
      .num_speculative_tokens(
          /*num_speculative_tokens=*/5);
  ModelInputParams params = grouped_input();
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.q_max_seq_len = 6;
  params.attention.host.q_seq_lens = {0, 6, 12, 18};
  params.attention.device.new_cache_slots = torch::zeros({18}, torch::kInt32);
  params.parallel.dp_global_token_nums = {18};
  const auto decision = MluGraphPlanner(config).plan({18, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  const auto& layout = std::get<GraphPlan>(decision).layout;
  EXPECT_EQ(layout.tokens_per_request, 6);
  EXPECT_EQ(layout.padded_num_reqs, 4);
  EXPECT_EQ(layout.padded_num_tokens, 24);

  config.options.kv_cache_dtype("int8");
  expect_eager(MluGraphPlanner(config), 18, params, EagerReason::NON_DECODE);
  config.options.kv_cache_dtype("auto").dp_size(2);
  expect_eager(MluGraphPlanner(config), 18, params, EagerReason::NON_DECODE);
  config.options.dp_size(1).cp_size(2);
  expect_eager(MluGraphPlanner(config), 18, params, EagerReason::NON_DECODE);
  config.options.cp_size(1).world_size(1);
  params.prefill_without_cache = true;
  expect_eager(
      MluGraphPlanner(config), 18, params, EagerReason::UNSUPPORTED_INPUT);
  params.prefill_without_cache = false;
  params.meta.q_max_seq_len = 5;
  expect_eager(MluGraphPlanner(config), 18, params, EagerReason::NON_DECODE);
  params.meta.q_max_seq_len = 6;
  config.capabilities.supports_grouped_mtp_graph = false;
  expect_eager(MluGraphPlanner(config), 18, params, EagerReason::NON_DECODE);
  config.capabilities.supports_grouped_mtp_graph = true;
  config.options.speculative_algorithm("DFlash");
  expect_eager(MluGraphPlanner(config), 18, params, EagerReason::NON_DECODE);
}

// Data-parallel ranks compare the whole key in one fixed-size packet, so a key
// that cannot fit is rejected here instead of being dropped at agreement time.
TEST(MluGraphPlannerTest, KeyLayoutBeyondAgreementPacketStaysEager) {
  const MluGraphPlanner planner(topk_reuse_config());
  ModelInputParams params = grouped_input();
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>(
      std::vector<int64_t>(kMaxGraphKeyLayoutFields, /*value=*/8));
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>(
      std::vector<int64_t>(kMaxGraphKeyLayoutFields + 1, /*value=*/8));
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, MixedKeyFieldsKeepAgreementBudget) {
  const MluGraphPlanner planner(topk_reuse_config());
  ModelInputParams params = grouped_input();
  params.multi_block_tables = {torch::zeros({3, 2}, torch::kInt32)};
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>(
      std::vector<int64_t>(kMaxGraphKeyLayoutFields - 1, /*value=*/8));
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.multi_block_tables.emplace_back(torch::zeros({3, 4}, torch::kInt32));
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

// The empty schema keys producer graphs, so a reuse step reporting it must not
// be admitted into the producer's cache entry.
TEST(MluGraphPlannerTest, ProducerDoesNotRequireTopkReuseCapability) {
  GraphPlannerConfig config = topk_reuse_config();
  config.capabilities.supports_native_index_share_for_iteration = false;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input();
  const auto producer = planner.plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(producer));
  EXPECT_TRUE(std::get<GraphPlan>(producer).key.mtp_topk_schema.empty());
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, StatefulTargetGroupingDoesNotEnableTopkReuse) {
  GraphPlannerConfig config = qwen_config();
  config.state_required = true;
  config.linear_state_required = true;
  config.capabilities.supports_grouped_mtp_graph = true;
  config.capabilities.supports_native_index_share_for_iteration = true;
  ModelInputParams params = grouped_input();
  params.embedding.linear_state_ids = {1, 2, 3};
  const MluGraphPlanner target(config);
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(target.plan({3, params})));
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  expect_eager(target, 3, params, EagerReason::UNSUPPORTED_INPUT);
  config.options.is_draft_engine(true);
  const auto reuse = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(reuse));
  EXPECT_EQ(std::get<GraphPlan>(reuse).key.mtp_topk_schema,
            (std::vector<int64_t>{7, 11}));
  config.capabilities.supports_grouped_mtp_graph = false;
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, OrdinaryDpAdmissionChecksCountsAndDecodeFlags) {
  GraphPlannerConfig config = qwen_config();
  config.options.dp_size(2).world_size(8);
  const MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 1};
  const GraphDecision equal = planner.plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(equal));
  EXPECT_EQ(std::get<GraphPlan>(equal).layout.padded_num_tokens, 4);

  params.parallel.dp_global_token_nums = {2, 3};
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.meta.q_max_seq_len = 2;
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.meta.q_max_seq_len = 1;
  params.parallel.dp_global_token_nums = {3, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1, 1};
  params.meta.q_max_seq_len = 0;
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, OrdinaryDpRejectsCountsOutsideConfiguredGroup) {
  GraphPlannerConfig config = qwen_config();
  config.options.dp_size(2).world_size(2);
  const MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  // The former RunMode skipped DP validation for a singleton count and never
  // compared a longer count vector with dp_size. Both are malformed for DP=2.
  params.parallel.dp_global_token_nums = {3};
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3, 3};
  params.parallel.dp_is_decode = {1, 1, 1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, GroupedStagesAndRestrictionsStayDistinct) {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true);
  config.capabilities.supports_grouped_mtp_graph = true;
  ModelInputParams params = grouped_input();
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  config.options.cp_size(4).world_size(4);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));
  config.options.cp_size(1).world_size(1);
  params.prefill_without_cache = true;
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.prefill_without_cache = false;
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state.reset();
  params.attention.device.new_cache_slots = torch::Tensor();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.attention.device.new_cache_slots = torch::zeros({3}, torch::kInt32);
  params.embedding.linear_state_ids = {1, 2};
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, GroupedVerifyChecksStageAndDpMetadata) {
  GraphPlannerConfig config = qwen_config();
  config.options.enable_speculative_decode(true).speculative_algorithm("MTP");
  config.options.dp_size(2).world_size(2);
  config.capabilities.supports_grouped_mtp_graph = true;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input();
  params.is_spec_verify = true;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.num_accepted_tokens = torch::ones({3}, torch::kInt32);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 1};
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  expect_eager(planner, 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.parallel.dp_global_token_nums = {3, 2};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3};
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1, 1};
  params.num_accepted_tokens = torch::ones({2}, torch::kInt32);
  expect_eager(planner, 3, params, EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, FlatVerifyBucketsPreserveWholeRequests) {
  GraphPlannerConfig config = qwen_config();
  config.options.enable_speculative_decode(true).speculative_algorithm(
      "DSpark");
  config.capabilities.supports_grouped_mtp_graph = true;
  config.capabilities.graph_dynamic_multi_block_tables = true;
  const auto model_managed_input = [](int64_t rows) {
    auto params = grouped_input(rows);
    params.multi_block_tables = {params.attention.device.block_tables};
    params.attention.device.block_tables = torch::Tensor();
    return params;
  };
  for (int32_t cp : {1, 4}) {
    config.options.cp_size(cp).world_size(cp);
    const MluGraphPlanner planner(config);
    for (const int64_t requests : {3, 8}) {
      ModelInputParams params = model_managed_input(requests * 6);
      params.meta.num_speculative_tokens = 5;
      const auto decision = planner.plan({requests * 6, params});
      ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
      const auto& plan = std::get<GraphPlan>(decision);
      EXPECT_EQ(plan.layout.tokens_per_request, 6);
      EXPECT_EQ(plan.layout.attention_row_query_len, 1);
      EXPECT_EQ(plan.layout.padded_num_tokens % 6, 0);
      EXPECT_GE(plan.layout.padded_num_reqs, requests);
      const auto decode =
          planner.plan({plan.layout.padded_num_tokens,
                        model_managed_input(plan.layout.padded_num_tokens)});
      ASSERT_TRUE(std::holds_alternative<GraphPlan>(decode));
      EXPECT_NE(get_key_fields(plan.key),
                get_key_fields(std::get<GraphPlan>(decode).key));
    }
  }
  const MluGraphPlanner planner(config);
  ModelInputParams partial = model_managed_input(6);
  partial.meta.num_speculative_tokens = 5;
  partial.multi_block_tables[0] = partial.multi_block_tables[0].slice(0, 0, 5);
  expect_eager(planner, 6, partial, EagerReason::INVALID_LAYOUT);
  partial = model_managed_input(5);
  partial.meta.num_speculative_tokens = 5;
  expect_eager(planner, 5, partial, EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, GroupedPaddingKeepsRequestsAndTpAlignment) {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true).dp_size(2).world_size(12);
  config.capabilities.supports_grouped_mtp_graph = true;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input(/*rows=*/12);
  params.meta.num_sequences = 3;
  params.attention.host.kpool_query_lens = {4, 4, 4};
  params.parallel.dp_global_token_nums = {12, 12};
  params.parallel.dp_is_decode = {1, 1};
  const GraphDecision decision = planner.plan({12, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  const GraphPlan& plan = std::get<GraphPlan>(decision);
  EXPECT_EQ(plan.layout.tokens_per_request, 4);
  EXPECT_EQ(plan.layout.num_reqs, 3);
  EXPECT_EQ(plan.layout.padded_num_reqs, 6);
  EXPECT_EQ(plan.layout.padded_num_tokens, 24);

  config.options.enable_graph_mode_decode_no_padding(true);
  const GraphDecision no_padding = MluGraphPlanner(config).plan({12, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(no_padding));
  EXPECT_EQ(std::get<GraphPlan>(no_padding).layout.padded_num_reqs, 3);
  EXPECT_EQ(std::get<GraphPlan>(no_padding).layout.padded_num_tokens, 12);
}

TEST(MluGraphPlannerTest, NonDecodeAndTokenLimitReturnReasons) {
  GraphPlannerConfig config = qwen_config();
  config.max_tokens = 3;
  MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, params})),
            EagerReason::TOKEN_LIMIT);
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, params})),
            EagerReason::NON_DECODE);
}

TEST(MluGraphPlannerTest, DraftGraphMetadataDoesNotRequireLinearState) {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true).speculative_algorithm("DSpark");
  config.capabilities.supports_grouped_mtp_graph = true;
  config.state_required = true;
  ModelInputParams params = grouped_input(/*rows=*/3);
  params.use_dflash_proposal_attention = true;
  params.meta.q_max_seq_len = 5;
  params.attention.host.q_seq_lens = {0, 5, 10, 15};
  params.attention.device.new_cache_slots = torch::zeros({15}, torch::kInt32);
  params.multi_block_tables = {params.attention.device.block_tables};
  params.attention.device.block_tables = torch::Tensor();
  const auto decision = MluGraphPlanner(config).plan({15, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  const auto& plan = std::get<GraphPlan>(decision);
  EXPECT_EQ(plan.layout.tokens_per_request, 5);
  EXPECT_EQ(plan.layout.padded_num_reqs, 4);
  EXPECT_EQ(plan.layout.padded_num_tokens, 20);

  config.linear_state_required = true;
  expect_eager(
      MluGraphPlanner(config), 15, params, EagerReason::INVALID_LAYOUT);
  params.embedding.linear_state_ids = {1, 2, 3};
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({15, params})));
}

TEST(MluGraphPlannerTest, NativeDraftSharingSeparatesProducerAndReuseGraphs) {
  GraphPlannerConfig config = topk_reuse_config();
  ModelInputParams params = grouped_input();
  const auto producer = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(producer));
  const auto make_state = [](int64_t rows, int64_t columns) {
    model::MluMtpTopkState::LayerStates layers;
    layers.reserve(2);
    layers.emplace_back(
        layer::DsaTopkState(torch::zeros({rows, columns}, torch::kInt32),
                            torch::ones({rows}, torch::kInt32)));
    layers.emplace_back(std::nullopt);
    return std::make_shared<model::MluMtpTopkState>(std::move(layers));
  };
  params.mtp_topk_state = make_state(/*rows=*/3, /*columns=*/8);
  const auto reuse = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(reuse));
  EXPECT_FALSE(std::get<GraphPlan>(producer).key ==
               std::get<GraphPlan>(reuse).key);
  params.mtp_topk_state = make_state(/*rows=*/3, /*columns=*/16);
  const auto changed_width = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(changed_width));
  EXPECT_FALSE(std::get<GraphPlan>(reuse).key ==
               std::get<GraphPlan>(changed_width).key);
  params.mtp_topk_state = make_state(/*rows=*/2, /*columns=*/8);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state = make_state(/*rows=*/3, /*columns=*/8);
  config.capabilities.supports_native_index_share_for_iteration = false;
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  config.capabilities.supports_native_index_share_for_iteration = true;
  config.options.is_draft_engine(false);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, KvSplitHistoryReusesKeyAcrossLogicalBlocks) {
  auto config = qwen_config();
  config.capabilities.graph_block_table_extra_columns = 0;
  MluGraphPlanner planner(config);
  std::optional<GraphKey> first_key;
  for (int64_t history : {46, 64, 65, 2049, 4097, 32768, 16}) {
    auto params = decode_input(history);
    params.attention.device.block_tables =
        torch::zeros({3, (history + 63) / 64}, torch::kInt32);
    const auto decision = planner.plan({3, params});
    ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
    const auto& plan = std::get<GraphPlan>(decision);
    if (first_key.has_value()) {
      EXPECT_TRUE(*first_key == plan.key);
    } else {
      first_key = plan.key;
    }
  }
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      planner.plan({3, decode_input(32769)})));
}

TEST(MluGraphPlannerTest, Dflash2VerifyPreservesB16GroupsAt64kHistory) {
  GraphPlannerConfig config = qwen_config();
  config.options.enable_speculative_decode(true)
      .speculative_algorithm("DFlash2")
      .num_speculative_tokens(/*num_speculative_tokens=*/3)
      .cp_size(4)
      .world_size(8);
  config.capabilities.supports_grouped_mtp_graph = true;
  config.capabilities.replay_family = MtpReplayFamily::GLM5;
  config.state_required = true;
  config.linear_state_required = true;
  config.max_tokens = 64;
  ModelInputParams params = grouped_input(/*rows=*/16);
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.q_max_seq_len = 4;
  params.meta.kv_max_seq_len = 65536;
  params.is_spec_verify = true;
  params.attention.host.q_seq_lens.clear();
  for (int32_t row = 0; row <= 16; ++row) {
    params.attention.host.q_seq_lens.emplace_back(row * 4);
  }
  params.attention.host.kpool_query_lens.assign(16, 4);
  params.attention.device.q_seq_lens =
      torch::tensor(params.attention.host.q_seq_lens, torch::kInt32);
  params.attention.device.kv_seq_lens = torch::zeros({17}, torch::kInt32);
  params.attention.device.new_cache_slots = torch::zeros({64}, torch::kInt32);
  params.attention.device.block_tables =
      torch::zeros({16, 4608}, torch::kInt32);
  params.embedding.linear_state_ids.assign(16, 1);
  params.num_accepted_tokens = torch::ones({16}, torch::kInt32);
  params.parallel.dp_global_token_nums = {64};
  const MluGraphPlanner planner(config);
  const GraphDecision first = planner.plan({64, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(first));
  const GraphPlan& plan = std::get<GraphPlan>(first);
  EXPECT_EQ(plan.layout.num_reqs, 16);
  EXPECT_EQ(plan.layout.tokens_per_request, 4);
  EXPECT_EQ(plan.layout.padded_num_tokens, 64);
  params.meta.kv_max_seq_len = 73728;
  const GraphDecision boundary = planner.plan({64, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(boundary));
  EXPECT_TRUE(plan.key == std::get<GraphPlan>(boundary).key);
  params.meta.kv_max_seq_len = 73729;
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({64, params})));
  params.meta.kv_max_seq_len = 65536;
  params.num_accepted_tokens = torch::Tensor();
  expect_eager(planner, 64, params, EagerReason::INVALID_LAYOUT);
  params.num_accepted_tokens = torch::ones({16}, torch::kInt32);
  params.embedding.linear_state_ids.clear();
  expect_eager(planner, 64, params, EagerReason::INVALID_LAYOUT);
  params.embedding.linear_state_ids.assign(16, 1);
  params.attention.host.kpool_query_lens.assign(16, 1);
  expect_eager(planner, 64, params, EagerReason::INVALID_LAYOUT);
  params.attention.host.kpool_query_lens.assign(16, 4);
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  expect_eager(planner, 64, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  config.capabilities.replay_family = MtpReplayFamily::QWEN35;
  expect_eager(MluGraphPlanner(config), 64, params, EagerReason::NON_DECODE);
  config.capabilities.replay_family = MtpReplayFamily::GLM5;
  config.options.speculative_algorithm("DFlash");
  expect_eager(MluGraphPlanner(config), 64, params, EagerReason::NON_DECODE);
  config.options.speculative_algorithm("DFlash2").is_draft_engine(true);
  params.is_spec_verify = false;
  expect_eager(MluGraphPlanner(config), 64, params, EagerReason::NON_DECODE);
}

}  // namespace
}  // namespace xllm::mlu
