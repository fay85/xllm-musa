/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/speculative/rejection_sampler.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>

#include "core/framework/speculative/spec_verify.h"
#include "core/runtime/forward_params.h"

namespace xllm {

TEST(RejectionSamplerTest, LogitsOnlyVerifyMasksEachDraftAndBonusPosition) {
  SamplingParameters sampling_params;
  sampling_params.do_sample =
      torch::zeros(/*size=*/{2}, /*options=*/torch::kBool);

  sampling_params.sampling_mode = SamplingMode::GREEDY;

  SamplingParameters verify_sampling_params = sampling_params;
  verify_sampling_params.selected_token_idxes = torch::arange(/*end=*/6);
  verify_sampling_params.sample_idxes =
      verify_sampling_params.selected_token_idxes;
  const float blocked = -std::numeric_limits<float>::infinity();
  verify_sampling_params.filter_mask =
      torch::tensor({{0.0f, blocked, blocked, blocked},
                     {blocked, 0.0f, blocked, blocked},
                     {blocked, 0.0f, blocked, blocked},
                     {blocked, blocked, 0.0f, blocked},
                     {blocked, 0.0f, blocked, blocked},
                     {blocked, blocked, blocked, 0.0f}});

  ForwardOutput target_output;
  target_output.logits = torch::tensor({{0.0f, 1.0f, 9.0f, 0.0f},
                                        {0.0f, 1.0f, 9.0f, 0.0f},
                                        {0.0f, 1.0f, 9.0f, 0.0f},
                                        {0.0f, 1.0f, 9.0f, 0.0f},
                                        {0.0f, 1.0f, 9.0f, 0.0f},
                                        {0.0f, 1.0f, 9.0f, 0.0f}});
  const auto draft_token_ids =
      torch::tensor({{0, 2}, {2, 1}}, /*options=*/torch::kInt64);

  const SampleOutput output = run_rejection_sampling(
      verify_sampling_params, draft_token_ids, {}, target_output.logits, {});
  EXPECT_TRUE(torch::equal(
      output.next_tokens,
      torch::tensor({{0, 1, -1}, {2, 1, 3}}, /*options=*/torch::kInt64)));
}

TEST(RejectionSamplerTest, UsesLogitsForAcceptanceRecoveryAndBonus) {
  const float blocked = -std::numeric_limits<float>::infinity();
  SamplingParameters params;

  params.sampling_mode = SamplingMode::RANDOM;
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.top_k = torch::ones(/*size=*/{6}, /*options=*/torch::kInt64);
  params.selected_token_idxes = torch::arange(/*end=*/6);
  params.sample_idxes = params.selected_token_idxes;
  const auto draft_token_ids =
      torch::tensor({{0, 1}, {2, 0}}, /*options=*/torch::kInt64);
  const auto draft_logits =
      torch::tensor({{{10.0f, blocked, blocked}, {blocked, 20.0f, blocked}},
                     {{blocked, blocked, 2.0f}, {1.0f, blocked, blocked}}});
  const auto target_logits = torch::tensor({{5.0f, 0.0f, 0.0f},
                                            {0.0f, 0.0f, 6.0f},
                                            {0.0f, 3.0f, 0.0f},
                                            {0.0f, 0.0f, 7.0f},
                                            {8.0f, 0.0f, 0.0f},
                                            {0.0f, 4.0f, 0.0f}});
  const auto output =
      rejection_sample(params, draft_token_ids, draft_logits, target_logits);
  EXPECT_TRUE(torch::equal(
      output.next_tokens,
      torch::tensor({{0, 2, -1}, {2, 0, 1}}, /*options=*/torch::kInt64)));
  EXPECT_TRUE(output.logprobs.eq(/*other=*/0).all().item<bool>());
  EXPECT_TRUE(torch::isneginf(output.top_logprobs.select(/*dim=*/-1,
                                                         /*index=*/1))
                  .all()
                  .item<bool>());
}

TEST(RejectionSamplerTest, ReturnsBonusWhenDraftIsEmpty) {
  SamplingParameters params;
  params.sampling_mode = SamplingMode::GREEDY;
  params.selected_token_idxes = torch::arange(/*end=*/2);
  params.sample_idxes = params.selected_token_idxes;
  const auto draft_token_ids =
      torch::empty(/*size=*/{2, 0}, /*options=*/torch::kInt64);
  const auto target_logits = torch::tensor({{0.0f, 1.0f}, {1.0f, 0.0f}});
  const auto output =
      rejection_sample(params, draft_token_ids, {}, target_logits);
  EXPECT_TRUE(
      torch::equal(output.next_tokens,
                   torch::tensor({{1}, {0}}, /*options=*/torch::kInt64)));
}

TEST(RejectionSamplerTest, GrammarInvalidDraftUsesFullTargetDistribution) {
  SamplingParameters params;

  params.sampling_mode = SamplingMode::RANDOM;
  params.selected_token_idxes = torch::arange(/*end=*/2);
  params.sample_idxes = params.selected_token_idxes;
  const float blocked = -std::numeric_limits<float>::infinity();
  ForwardOutput target;
  target.logits = torch::tensor({{blocked, 10.0f}, {1.0f, blocked}});
  const auto draft_token_ids = torch::tensor({{-1}}, /*options=*/torch::kInt64);
  const auto draft_logits = torch::tensor({{{blocked, 3.0f}}});
  const auto output = run_rejection_sampling(
      params, draft_token_ids, draft_logits, target.logits, {});
  EXPECT_TRUE(torch::equal(
      output.next_tokens, torch::tensor({{1, -1}}, /*options=*/torch::kInt64)));
}

}  // namespace xllm
