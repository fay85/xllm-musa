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

#include "core/framework/speculative/rejection_sampler.h"

#include <glog/logging.h>

#include <limits>
#include <tuple>
#include <utility>

#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

torch::Tensor commit_random_prefix(const torch::Tensor& target_logits,
                                   const torch::Tensor& draft_token_ids,
                                   const torch::Tensor& draft_logits,
                                   const torch::Tensor& bonus_token_ids) {
  const int64_t num_draft_tokens = draft_token_ids.size(/*dim=*/1);
  const auto draft_indices = draft_token_ids.clamp_min(/*min=*/0);
  const auto valid_draft = draft_token_ids.ge(/*other=*/0);
  const auto gather_logits = [&draft_indices](const torch::Tensor& logits) {
    return logits.gather(/*dim=*/-1, draft_indices.unsqueeze(/*dim=*/-1))
        .squeeze(/*dim=*/-1)
        .to(/*dtype=*/torch::kFloat32);
  };
  const auto target_normalizers =
      torch::logsumexp(target_logits.to(/*dtype=*/torch::kFloat32), /*dim=*/-1);
  auto log_acceptance = gather_logits(target_logits) - target_normalizers;
  torch::Tensor draft_normalizers;
  if (draft_logits.defined()) {
    draft_normalizers = torch::logsumexp(
        draft_logits.to(/*dtype=*/torch::kFloat32), /*dim=*/-1);
    log_acceptance.sub_(gather_logits(draft_logits) - draft_normalizers);
  }
  const auto accepted =
      torch::rand(draft_token_ids.sizes(), log_acceptance.options())
          .log_()
          .lt(log_acceptance)
          .logical_and(valid_draft);

  // Recovery reads only the first rejected position of each request.
  const auto committed_prefix =
      accepted.to(/*dtype=*/torch::kInt32).cumprod(/*dim=*/1).gt(/*other=*/0);
  const auto accepted_count = committed_prefix.sum(/*dim=*/1, /*keepdim=*/true);
  const auto recovery_indices = accepted_count.clamp_max(num_draft_tokens - 1);
  const auto gather_recovery_log_distribution =
      [&](const torch::Tensor& logits, const torch::Tensor& normalizers) {
        return logits
            .gather(/*dim=*/1,
                    recovery_indices.unsqueeze(/*dim=*/-1)
                        .expand({-1, 1, logits.size(/*dim=*/-1)}))
            .squeeze(/*dim=*/1)
            .to(/*dtype=*/torch::kFloat32)
            .sub_(normalizers.gather(/*dim=*/1, recovery_indices));
      };
  auto recovery_logits =
      gather_recovery_log_distribution(target_logits, target_normalizers);
  const auto valid_recovery = valid_draft.gather(/*dim=*/1, recovery_indices);
  const auto rejected_token = draft_indices.gather(/*dim=*/1, recovery_indices);
  constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
  if (draft_logits.defined()) {
    auto recovery_draft =
        gather_recovery_log_distribution(draft_logits, draft_normalizers);
    recovery_draft.sub_(recovery_logits)
        .nan_to_num_(/*nan=*/0.0)
        .clamp_max_(/*max=*/0)
        .exp_()
        .neg_()
        .log1p_();
    recovery_draft.masked_fill_(valid_recovery.logical_not(), /*value=*/0);
    recovery_logits.add_(recovery_draft);
  } else {
    auto rejected_logit = recovery_logits.gather(/*dim=*/1, rejected_token);
    rejected_logit.masked_fill_(valid_recovery, kNegativeInfinity);
    recovery_logits.scatter_(/*dim=*/1, rejected_token, rejected_logit);
  }
  const auto recovered_ids =
      gumbel_sample(recovery_logits).unsqueeze(/*dim=*/1);
  const auto rejected_tail =
      torch::full_like(draft_token_ids, /*fill_value=*/-1);
  const auto committed =
      torch::where(committed_prefix, draft_token_ids, rejected_tail);
  auto output = torch::cat(
      {committed,
       torch::where(accepted_count.eq(num_draft_tokens),
                    bonus_token_ids,
                    torch::full_like(bonus_token_ids, /*fill_value=*/-1))},
      /*dim=*/1);
  output.scatter_(
      /*dim=*/1,
      accepted_count,
      torch::where(
          accepted_count.lt(num_draft_tokens), recovered_ids, bonus_token_ids));
  return output;
}

torch::Tensor commit_greedy_prefix(const torch::Tensor& draft_token_ids,
                                   const torch::Tensor& target_token_ids) {
  const int64_t num_draft_tokens = draft_token_ids.size(/*dim=*/1);
  if (num_draft_tokens == 0) {
    return target_token_ids;
  }
  const auto target_drafts =
      target_token_ids.narrow(/*dim=*/1, /*start=*/0, num_draft_tokens);
  const auto accepted = draft_token_ids.eq(target_drafts)
                            .to(/*dtype=*/torch::kInt32)
                            .cumprod(/*dim=*/1)
                            .gt(/*other=*/0);
  const auto keep = torch::cat(
      {torch::ones_like(accepted.narrow(/*dim=*/1, /*start=*/0, /*length=*/1)),
       accepted},
      /*dim=*/1);
  return target_token_ids.masked_fill(keep.logical_not(), /*value=*/-1);
}

}  // namespace

SampleOutput rejection_sample(const SamplingParameters& params,
                              const torch::Tensor& draft_token_ids,
                              const torch::Tensor& draft_logits,
                              torch::Tensor target_logits) {
  CHECK_EQ(draft_token_ids.dim(), 2);
  const int64_t num_sequences = draft_token_ids.size(/*dim=*/0);
  const int64_t num_draft_tokens = draft_token_ids.size(/*dim=*/1);
  const int64_t verify_width = num_draft_tokens + 1;
  CHECK_EQ(target_logits.dim(), 2);
  CHECK_EQ(target_logits.size(/*dim=*/0), num_sequences * verify_width);
  const int64_t vocab_size = target_logits.size(/*dim=*/1);
  if (draft_logits.defined()) {
    CHECK_EQ(draft_logits.sizes(),
             torch::IntArrayRef({num_sequences, num_draft_tokens, vocab_size}));
  }

  // The ordinary sampler's transform contract is two-dimensional.
  auto flat_logits = process_logits(target_logits, params);
  if (params.sampling_mode != SamplingMode::GREEDY) {
    apply_logits_transforms(
        flat_logits, params.temperatures, params.top_k, params.top_p);
  }
  const auto grouped_logits =
      flat_logits.view({num_sequences, verify_width, vocab_size});
  const auto greedy_ids =
      params.sampling_mode == SamplingMode::RANDOM
          ? torch::Tensor()
          : greedy_sample(grouped_logits).to(/*dtype=*/torch::kInt64);
  const auto draft_tokens =
      draft_token_ids.to(flat_logits.options().dtype(/*dtype=*/torch::kInt64));

  SampleOutput output;
  if (params.sampling_mode == SamplingMode::GREEDY) {
    output.next_tokens = commit_greedy_prefix(draft_tokens, greedy_ids);
  } else {
    const auto bonus_ids =
        gumbel_sample(grouped_logits.select(/*dim=*/1, num_draft_tokens))
            .unsqueeze(/*dim=*/1);
    output.next_tokens =
        num_draft_tokens == 0
            ? bonus_ids
            : commit_random_prefix(
                  grouped_logits.narrow(
                      /*dim=*/1, /*start=*/0, num_draft_tokens),
                  draft_tokens,
                  draft_logits,
                  bonus_ids);
    if (params.sampling_mode != SamplingMode::RANDOM) {
      output.next_tokens =
          torch::where(params.do_sample.view(/*size=*/{-1, 1}),
                       output.next_tokens,
                       commit_greedy_prefix(draft_tokens, greedy_ids));
    }
  }
  if (params.logprobs) {
    auto logprobs = torch::log_softmax(grouped_logits,
                                       /*dim=*/-1,
                                       /*dtype=*/torch::kFloat32);
    const auto token_indices = output.next_tokens.clamp_min(/*min=*/0);
    output.logprobs =
        logprobs.gather(/*dim=*/-1, token_indices.unsqueeze(/*dim=*/-1))
            .squeeze(/*dim=*/-1)
            .masked_fill(output.next_tokens.lt(/*other=*/0), /*value=*/0);
    if (params.max_top_logprobs > 0) {
      std::tie(output.top_logprobs, output.top_tokens) =
          logprobs.topk(params.max_top_logprobs, /*dim=*/-1);
    }
  }
  return output;
}

}  // namespace xllm
