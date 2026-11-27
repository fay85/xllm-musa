/* Copyright 2026-2026 The xLLM Authors.

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

#include "core/runtime/json_object_decode_adapter.h"

#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/constraint/draft_validation.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/runtime/forward_params.h"

namespace xllm {

void JsonObjectDecodeAdapter::prepare_verify_mask(
    const ForwardInput& input,
    const detail::JsonDraftValidationScratch& draft_states,
    ForwardInput& verify_input,
    const torch::Device& device) {
  CHECK_EQ(input.json_object_states.size(),
           static_cast<size_t>(input.input_params.meta.num_sequences));
  verify_input.json_object_states = detail::build_json_validation_states(
      input.json_object_states,
      draft_states,
      verify_input.json_object_invalid_draft);
  // The verify batch owns one constraint state per target logit. Its states
  // have already been restored and replayed from the sequence snapshots.
  verify_input.json_object_state_snapshots.clear();
  install_filter_bitmask(verify_input.sampling_params,
                         verify_input.json_object_states,
                         device,
                         /*phase=*/JsonObjectMaskBuildPhase::TARGET);
}

void JsonObjectDecodeAdapter::prepare_block_verify_mask(
    const ForwardInput& input,
    const torch::Tensor& draft_token_ids,
    ForwardInput& verify_input,
    const torch::Device& device) {
  CHECK(draft_token_ids.device().is_cpu() && draft_token_ids.dim() == 2);
  const int64_t num_sequences = draft_token_ids.size(/*dim=*/0);
  const int64_t num_draft_tokens = draft_token_ids.size(/*dim=*/1);
  CHECK_EQ(input.json_object_states.size(), static_cast<size_t>(num_sequences));
  CHECK_GT(num_draft_tokens, 0);

  const torch::Tensor token_ids =
      draft_token_ids.to(/*dtype=*/torch::kInt32).contiguous();
  const int32_t* draft_token_data = token_ids.const_data_ptr<int32_t>();
  std::vector<ConstraintState> states = input.json_object_states;
  std::vector<uint8_t> invalid_suffix(static_cast<size_t>(num_sequences), 0);
  std::vector<int32_t> token_column(static_cast<size_t>(num_sequences));
  detail::JsonDraftValidationScratch scratch;
  scratch.states_after.reserve(static_cast<size_t>(num_draft_tokens));
  scratch.invalid_draft_step_major.reserve(
      static_cast<size_t>(num_sequences * num_draft_tokens));
  for (int64_t position = 0; position < num_draft_tokens; ++position) {
    for (int64_t seq = 0; seq < num_sequences; ++seq) {
      token_column[static_cast<size_t>(seq)] =
          draft_token_data[seq * num_draft_tokens + position];
    }
    detail::advance_json_draft_step(
        states, invalid_suffix, token_column, scratch);
  }

  prepare_verify_mask(input, scratch, verify_input, device);
}

torch::Tensor JsonObjectDecodeAdapter::mask_invalid_draft_tokens(
    const torch::Tensor& draft_token_ids,
    const std::vector<uint8_t>& invalid_draft) {
  if (invalid_draft.empty()) {
    return draft_token_ids;
  }
  CHECK_EQ(invalid_draft.size(), static_cast<size_t>(draft_token_ids.numel()));
  const auto invalid = torch::tensor(invalid_draft, /*options=*/torch::kUInt8)
                           .to(draft_token_ids.device())
                           .view(draft_token_ids.sizes())
                           .to(/*dtype=*/torch::kBool);
  return draft_token_ids.masked_fill(invalid, /*value=*/-1);
}

std::vector<size_t> JsonObjectDecodeAdapter::check_verified_tokens(
    const ForwardInput& input,
    const torch::Tensor& accepted_token_ids,
    ForwardOutput& output) {
  CHECK(accepted_token_ids.device().is_cpu() && accepted_token_ids.dim() == 2);
  const int64_t num_sequences = accepted_token_ids.size(/*dim=*/0);
  const int64_t width = accepted_token_ids.size(/*dim=*/1);
  CHECK_EQ(input.sample_sequence_ids.size(),
           static_cast<size_t>(num_sequences));
  const torch::Tensor token_ids =
      accepted_token_ids.to(/*dtype=*/torch::kInt64).contiguous();
  const auto mismatches =
      detail::verify_json_accepted_rows(input.json_object_states,
                                        token_ids.const_data_ptr<int64_t>(),
                                        static_cast<size_t>(num_sequences),
                                        static_cast<size_t>(width));
  std::vector<size_t> failed_sequences;
  failed_sequences.reserve(mismatches.size());
  for (const detail::JsonAcceptedTokenMismatch& mismatch : mismatches) {
    const std::string& sequence_id =
        input.sample_sequence_ids[mismatch.sequence_index];
    LOG(ERROR) << "Verified token violates json_object grammar: sequence_id="
               << sequence_id << ", token_offset=" << mismatch.token_offset
               << ", token_id=" << mismatch.token_id
               << ", committed_tokens=" << mismatch.committed_tokens
               << ", state_fingerprint=" << mismatch.state_fingerprint;
    output.json_object_errors.push_back(
        {sequence_id,
         "accepted output violates json_object grammar, token_id=" +
             std::to_string(mismatch.token_id)});
    failed_sequences.emplace_back(mismatch.sequence_index);
  }
  return failed_sequences;
}

}  // namespace xllm
