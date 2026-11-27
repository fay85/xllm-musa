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

#include "core/framework/speculative/draft_vocab_mapping.h"

#include <glog/logging.h>

#include <limits>
#include <tuple>
#include <utility>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

DraftVocabMapping::DraftVocabMapping(
    const torch::Tensor& draft_token_id_offsets,
    int64_t draft_vocab_size,
    int64_t target_vocab_size,
    const torch::Device& device)
    : target_vocab_size_(target_vocab_size) {
  if (!draft_token_id_offsets.defined()) {
    CHECK(draft_vocab_size == 0 || draft_vocab_size == target_vocab_size)
        << "Different draft and target vocabulary sizes require a d2t tensor.";
    return;
  }
  CHECK_EQ(draft_token_id_offsets.dim(), 1) << "Draft d2t table must be 1D.";
  const int64_t num_draft_ids = draft_token_id_offsets.numel();
  if (draft_vocab_size != 0) {
    CHECK_EQ(num_draft_ids, draft_vocab_size)
        << "Draft d2t table size must match draft_vocab_size.";
  }
  auto mapped_ids =
      draft_token_id_offsets.to(/*dtype=*/torch::kLong) +
      torch::arange(
          num_draft_ids,
          draft_token_id_offsets.options().dtype(/*dtype=*/torch::kLong));
  CHECK_EQ(std::get<0>(torch::_unique2(mapped_ids)).numel(), num_draft_ids)
      << "Draft d2t target ids must be unique.";
  const auto [min_id, max_id] = torch::aminmax(mapped_ids);
  CHECK_GE(min_id.item<int64_t>(), 0);
  CHECK_LT(max_id.item<int64_t>(), target_vocab_size);
  draft_to_target_ids_ = mapped_ids.to(device);
}

void DraftVocabMapping::remap(SampleOutput& output) const {
  if (!draft_to_target_ids_.defined() || !output.next_tokens.defined() ||
      output.next_tokens.numel() == 0) {
    return;
  }
  if (output.draft_logits.defined()) {
    const int64_t num_proposed_tokens = output.draft_logits.size(/*dim=*/0);
    torch::Tensor target_logits =
        torch::full({num_proposed_tokens, target_vocab_size_},
                    -std::numeric_limits<float>::infinity(),
                    output.draft_logits.options());
    target_logits.index_copy_(
        /*dim=*/1, draft_to_target_ids_, output.draft_logits);
    output.draft_logits = std::move(target_logits);
  }
  output.next_tokens =
      draft_to_target_ids_.index_select(/*dim=*/0, output.next_tokens);
}

}  // namespace xllm
