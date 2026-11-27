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

#include "core/framework/speculative/speculative_token_stats.h"

#include <glog/logging.h>

#include "core/common/metrics.h"

namespace xllm {
namespace {

template <typename Token>
void accumulate_speculative_acceptance(const torch::Tensor& tokens,
                                       std::span<const char> constrained_mask,
                                       SpeculativeOutputStats& stats) {
  const Token* data = tokens.const_data_ptr<Token>();
  const int64_t width = tokens.size(/*dim=*/1);
  const int64_t proposals = width - 1;
  for (size_t row = 0; row < stats.sequence_stats.size(); ++row) {
    const Token* sequence_token_ids = data + row * width;
    int64_t committed = 0;
    while (committed < width && sequence_token_ids[committed] >= 0) {
      ++committed;
    }
    const int64_t accepted = committed > 0 ? committed - 1 : 0;
    stats.sequence_stats[row].proposed_tokens = proposals;
    stats.sequence_stats[row].accepted_tokens = accepted;
    stats.committed_tokens += committed;
    stats.accepted_tokens += accepted;
    for (int64_t position = 0; position < accepted; ++position) {
      ++stats.accepted_per_position[static_cast<size_t>(position)];
    }
    if (!constrained_mask.empty() && constrained_mask[row] != 0) {
      stats.constrained_draft_tokens += proposals;
      stats.constrained_accepted_tokens += accepted;
    }
  }
}

}  // namespace

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    std::span<const char> constrained_mask) {
  CHECK(tokens.defined() && tokens.device().is_cpu() && tokens.dim() == 2);
  const int64_t batch_size = tokens.size(/*dim=*/0);
  const int64_t width = tokens.size(/*dim=*/1);
  CHECK_GT(width, 0);
  CHECK(constrained_mask.empty() ||
        constrained_mask.size() == static_cast<size_t>(batch_size));

  SpeculativeOutputStats stats;
  stats.accepted_per_position.resize(static_cast<size_t>(width - 1));
  stats.sequence_stats.resize(static_cast<size_t>(batch_size));
  stats.draft_tokens = batch_size * (width - 1);

  if (tokens.scalar_type() == torch::kInt32) {
    accumulate_speculative_acceptance<int32_t>(
        tokens.contiguous(), constrained_mask, stats);
  } else {
    accumulate_speculative_acceptance<int64_t>(
        tokens.to(/*dtype=*/torch::kInt64).contiguous(),
        constrained_mask,
        stats);
  }
  return stats;
}

void record_speculative_output_stats(
    const SpeculativeOutputStats& stats,
    std::span<const std::string> position_labels) {
  CHECK_EQ(stats.accepted_per_position.size(), position_labels.size());
  COUNTER_ADD(speculative_num_drafts_total, stats.sequence_stats.size());
  COUNTER_ADD(speculative_num_committed_tokens_total, stats.committed_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_total, stats.draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total, stats.accepted_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_constrained_total,
              stats.constrained_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_constrained_total,
              stats.constrained_accepted_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_plain_total,
              stats.draft_tokens - stats.constrained_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_plain_total,
              stats.accepted_tokens - stats.constrained_accepted_tokens);
  for (size_t position = 0; position < position_labels.size(); ++position) {
    MULTI_COUNTER_ADD(speculative_num_accepted_tokens_per_pos,
                      position_labels[position],
                      stats.accepted_per_position[position]);
  }

  const double total_drafts = COUNTER_VALUE(speculative_num_drafts_total);
  if (total_drafts > 0) {
    GAUGE_SET(
        speculative_mean_acceptance_length,
        COUNTER_VALUE(speculative_num_committed_tokens_total) / total_drafts);
  }
}

}  // namespace xllm
