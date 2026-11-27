/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/constraint/draft_validation.h"

#include <gtest/gtest.h>

#include <vector>

#include "core/framework/constraint/builtin_json_machine.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "tests/core/framework/constraint/builtin_state_test_utils.h"

namespace xllm {
namespace {

JsonObjectGrammar make_mtp_grammar() {
  return JsonObjectGrammar({"{", "\"a\"", ":", "1", "}", "stop"},
                           /*stop_token_ids=*/{5});
}

TEST(MtpJsonObjectStateTest, BuildsSequenceMajorRowsForMixedAcceptedDrafts) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states = {
      make_builtin_test_state(grammar), ConstraintState()};
  std::vector<ConstraintState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_FALSE(detail::advance_json_draft_step(
      current_states, invalid_suffix, {0, 4}, scratch));
  EXPECT_FALSE(detail::advance_json_draft_step(
      current_states, invalid_suffix, {1, 4}, scratch));
  EXPECT_FALSE(detail::advance_json_draft_step(
      current_states, invalid_suffix, {2, 4}, scratch));

  std::vector<uint8_t> invalid_draft;
  const auto validation_states = detail::build_json_validation_states(
      initial_states, scratch, invalid_draft);

  ASSERT_EQ(validation_states.size(), 8u);
  EXPECT_EQ(validation_states[0].fingerprint(),
            initial_states[0].fingerprint());
  EXPECT_EQ(validation_states[1].fingerprint(),
            scratch.states_after[0][0].fingerprint());
  EXPECT_EQ(validation_states[2].fingerprint(),
            scratch.states_after[1][0].fingerprint());
  EXPECT_EQ(validation_states[3].fingerprint(),
            scratch.states_after[2][0].fingerprint());
  for (size_t row = 4; row < validation_states.size(); ++row) {
    EXPECT_FALSE(validation_states[row].initialized());
  }
  EXPECT_EQ(invalid_draft, std::vector<uint8_t>({0, 0, 0, 0, 0, 0}));
  EXPECT_TRUE(initial_states[0].snapshot().token_ids.empty());

  // Keep one sequence invalid across all draft columns while the uninitialized
  // second sequence remains valid; invalid flags are flattened sequence-major.
  std::vector<ConstraintState> invalid_states = initial_states;
  std::vector<uint8_t> mixed_invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch invalid_scratch;
  EXPECT_TRUE(detail::advance_json_draft_step(
      invalid_states, mixed_invalid_suffix, {4, 0}, invalid_scratch));
  EXPECT_FALSE(detail::advance_json_draft_step(
      invalid_states, mixed_invalid_suffix, {-1, 0}, invalid_scratch));
  EXPECT_FALSE(detail::advance_json_draft_step(
      invalid_states, mixed_invalid_suffix, {-1, 0}, invalid_scratch));
  std::vector<uint8_t> mixed_invalid_draft;
  const auto invalid_validation_states = detail::build_json_validation_states(
      initial_states, invalid_scratch, mixed_invalid_draft);
  EXPECT_EQ(mixed_invalid_draft, std::vector<uint8_t>({1, 1, 1, 0, 0, 0}));
  for (size_t row = 0; row < 4; ++row) {
    EXPECT_EQ(invalid_validation_states[row].fingerprint(),
              initial_states[0].fingerprint());
  }
}

TEST(MtpJsonObjectStateTest, PrelaunchedStaleFirstTokenIsRejectedNotCommitted) {
  // The MTP first-draft prelaunch seeds the consume branch from this step's
  // fresh states, but the draft sampled under the submit-time (one-commit-
  // stale) mask. This test pins that a stale first draft token is rejected by
  // the current mask (can_accept on the seeded state) and never advances the
  // live matcher: the invalid row is frozen at the initial fingerprint.
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states = {
      make_builtin_test_states(grammar, /*count=*/1)[0], ConstraintState()};
  std::vector<ConstraintState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  // Token 4 is illegal at the initial position under the current mask, so a
  // stale first draft that proposed it must be flagged and frozen.
  EXPECT_TRUE(detail::advance_json_draft_step(
      current_states, invalid_suffix, {4, 0}, scratch));
  // A legal token afterwards still processes normally on the live row.
  EXPECT_FALSE(detail::advance_json_draft_step(
      current_states, invalid_suffix, {0, 0}, scratch));

  // The invalid first row never advanced: its state fingerprint is still the
  // initial one, and the invalid suffix marks it (and the legal row's flag is
  // untouched).
  EXPECT_EQ(current_states[0].fingerprint(), initial_states[0].fingerprint());
  EXPECT_TRUE(current_states[0].initialized());
  EXPECT_FALSE(current_states[1].initialized());
  EXPECT_EQ(invalid_suffix[0], 1);
  EXPECT_EQ(invalid_suffix[1], 0);
  EXPECT_TRUE(initial_states[0].snapshot().token_ids.empty());
}

TEST(MtpJsonObjectStateTest, FrozenSuffixRowsKeepTheCommittedVerifyMask) {
  // Prelaunch samples the first draft under the submit-time (one commit stale)
  // mask, so validation rejects it and freezes the suffix. The rejection must
  // be durable: every suffix row keeps the committed state instead of
  // advancing, the stale token stays outside every row's mask, and the freeze
  // does not leak into the committed row itself.
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states =
      make_builtin_test_states(grammar, /*count=*/1);
  std::vector<ConstraintState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(1, 0);
  detail::JsonDraftValidationScratch scratch;

  // Step 0: "}" (4) is illegal at the initial position under the current mask,
  // which is how a stale prelaunched first draft surfaces.
  ASSERT_TRUE(detail::advance_json_draft_step(
      current_states, invalid_suffix, {4}, scratch));
  // Steps 1-2: the suffix still carries draft tokens; the freeze must keep them
  // out of verification instead of advancing the matcher with them.
  detail::advance_json_draft_step(current_states, invalid_suffix, {5}, scratch);
  detail::advance_json_draft_step(current_states, invalid_suffix, {0}, scratch);

  std::vector<uint8_t> invalid_draft;
  const auto validation_states = detail::build_json_validation_states(
      initial_states, scratch, invalid_draft);
  EXPECT_EQ(invalid_draft, std::vector<uint8_t>({1, 1, 1}));
  ASSERT_EQ(validation_states.size(), 4u);

  for (size_t row = 0; row < 3; ++row) {
    SCOPED_TRACE(row);
    // No suffix row advanced: each verifies against the committed state.
    EXPECT_EQ(validation_states[row].fingerprint(),
              initial_states[0].fingerprint());
    // The stale token is denied everywhere, so it can never verify.
    EXPECT_FALSE(validation_states[row].can_accept(/*stale=*/4));
  }
  // The freeze is scoped to the draft suffix: the committed row still accepts
  // the legal opening token.
  EXPECT_TRUE(validation_states[0].can_accept(/*open=*/0));
}

TEST(MtpJsonObjectStateTest, PreservesAcceptedPrefixAfterMiddleInvalidDraft) {
  struct Case {
    std::vector<int32_t> draft;
    std::vector<uint8_t> invalid;
    std::vector<int32_t> prefix;
  };
  const std::vector<Case> cases = {
      {{0, 2, -1}, {0, 1, 1}, {0}},
      // DFlash replays this whole-block sequence through the same helper.
      {{0, 1, 5, 3, 4}, {0, 0, 1, 1, 1}, {0, 1}},
  };
  JsonObjectGrammar grammar = make_mtp_grammar();
  for (const Case& test_case : cases) {
    const std::vector<ConstraintState> initial_states =
        make_builtin_test_states(grammar, /*count=*/1);
    std::vector<ConstraintState> current_states = initial_states;
    std::vector<uint8_t> invalid_suffix(1, 0);
    detail::JsonDraftValidationScratch scratch;
    for (size_t index = 0; index < test_case.draft.size(); ++index) {
      const bool halted = detail::advance_json_draft_step(
          current_states, invalid_suffix, {test_case.draft[index]}, scratch);
      EXPECT_EQ(halted, index == test_case.prefix.size());
    }
    std::vector<uint8_t> invalid_draft;
    const auto validation_states = detail::build_json_validation_states(
        initial_states, scratch, invalid_draft);
    EXPECT_EQ(invalid_draft, test_case.invalid);
    ASSERT_EQ(validation_states.size(), test_case.draft.size() + 1);
    ConstraintState expected = initial_states[0].fork();
    for (size_t index = 0; index < test_case.prefix.size(); ++index) {
      ASSERT_TRUE(
          expected.advance(test_case.prefix[index], HaltPolicy::RESPECT));
      EXPECT_EQ(validation_states[index + 1].fingerprint(),
                expected.fingerprint());
    }
    for (size_t row = test_case.prefix.size(); row < validation_states.size();
         ++row) {
      EXPECT_EQ(validation_states[row].fingerprint(), expected.fingerprint());
    }
    EXPECT_TRUE(initial_states[0].can_accept(/*open_object=*/0));
  }
}

TEST(MtpJsonObjectStateTest, FiveDraftRowsReplayFromCommittedState) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states =
      make_builtin_test_states(grammar, 1);
  std::vector<ConstraintState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(1, 0);
  detail::JsonDraftValidationScratch scratch;
  const std::vector<int32_t> draft_tokens = {0, 1, 2, 3, 4};

  for (const int32_t token_id : draft_tokens) {
    EXPECT_FALSE(detail::advance_json_draft_step(
        current_states, invalid_suffix, {token_id}, scratch));
  }

  std::vector<uint8_t> invalid_draft;
  const std::vector<ConstraintState> validation_states =
      detail::build_json_validation_states(
          initial_states, scratch, invalid_draft);

  ASSERT_EQ(validation_states.size(), 6u);
  EXPECT_EQ(invalid_draft, std::vector<uint8_t>({0, 0, 0, 0, 0}));
  ConstraintState committed_state = initial_states[0].fork();
  for (size_t token_idx = 0; token_idx < draft_tokens.size(); ++token_idx) {
    EXPECT_EQ(validation_states[token_idx].fingerprint(),
              committed_state.fingerprint());
    ASSERT_TRUE(
        validation_states[token_idx].can_accept(draft_tokens[token_idx]));
    ASSERT_TRUE(
        committed_state.advance(draft_tokens[token_idx], HaltPolicy::RESPECT));
  }
  EXPECT_EQ(validation_states.back().fingerprint(),
            committed_state.fingerprint());
  EXPECT_TRUE(validation_states.back().can_accept(/*stop=*/5));
  EXPECT_TRUE(committed_state.advance(/*stop=*/5, HaltPolicy::RESPECT));
}

TEST(MtpJsonObjectStateTest,
     ReasoningDisabledValidationRowsRejectReasoningEndMarker) {
  JsonObjectGrammar grammar(
      {"{", "\"a\"", ":", "1", "}", "stop", "<think>", "</think>"},
      /*stop_token_ids=*/{5},
      /*reasoning_end_token_ids=*/{7});
  const std::vector<ConstraintState> initial_states =
      make_builtin_test_states(grammar, 1, /*reasoning_phase=*/false);
  std::vector<ConstraintState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(1, 0);
  detail::JsonDraftValidationScratch scratch;
  const std::vector<int32_t> draft_tokens = {0, 1, 2, 3, 4};

  for (const int32_t token_id : draft_tokens) {
    EXPECT_FALSE(detail::advance_json_draft_step(
        current_states, invalid_suffix, {token_id}, scratch));
  }

  std::vector<uint8_t> invalid_draft;
  const std::vector<ConstraintState> validation_states =
      detail::build_json_validation_states(
          initial_states, scratch, invalid_draft);

  ASSERT_EQ(validation_states.size(), 6u);
  for (const ConstraintState& state : validation_states) {
    EXPECT_FALSE(state.as<JsonObjectGrammarState>().in_reasoning());
    EXPECT_FALSE(state.can_accept(/*reasoning_end=*/7));
  }
}

TEST(MtpJsonObjectStateTest, ReplaysFiveStepMixedAcceptedOutputRows) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states = {
      make_builtin_test_state(grammar), ConstraintState()};
  const int64_t accepted_tokens[] = {
      0,
      1,
      2,
      3,
      4,
      5,
      5,
      4,
      3,
      2,
      1,
      0,
  };

  const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
      detail::verify_json_accepted_rows(
          initial_states, accepted_tokens, /*num_rows=*/2, /*row_width=*/6);

  EXPECT_TRUE(mismatches.empty());
}

TEST(MtpJsonObjectStateTest, ReportsFirstAcceptedOutputMismatchPerSequence) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  std::vector<ConstraintState> machine_states =
      make_builtin_test_states(grammar, 1);
  ConstraintState committed_state = machine_states[0].fork();
  ASSERT_TRUE(committed_state.advance(/*open_object=*/0, HaltPolicy::RESPECT));
  const std::vector<ConstraintState> initial_states = {
      committed_state, ConstraintState(), make_builtin_test_state(grammar)};
  const int64_t accepted_tokens[] = {
      1,
      4,
      3,
      -1,
      -1,
      -1,
      5,
      4,
      3,
      2,
      1,
      0,
      0,
      1,
      5,
      3,
      4,
      -1,
  };

  const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
      detail::verify_json_accepted_rows(
          initial_states, accepted_tokens, /*num_rows=*/3, /*row_width=*/6);

  ASSERT_EQ(mismatches.size(), 2u);
  EXPECT_EQ(mismatches[0].sequence_index, 0u);
  EXPECT_EQ(mismatches[0].token_offset, 1u);
  EXPECT_EQ(mismatches[0].token_id, 4);
  EXPECT_EQ(mismatches[0].committed_tokens, 2u);
  EXPECT_EQ(mismatches[1].sequence_index, 2u);
  EXPECT_EQ(mismatches[1].token_offset, 2u);
  EXPECT_EQ(mismatches[1].token_id, 5);
  EXPECT_EQ(mismatches[1].committed_tokens, 2u);
}

TEST(MtpJsonObjectStateTest, AcceptedOutputOnlyUsesMinusOneAsPadding) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states =
      make_builtin_test_states(grammar, 1);
  const int64_t accepted_tokens[] = {0, -2};

  EXPECT_DEATH(
      detail::verify_json_accepted_rows(
          initial_states, accepted_tokens, /*num_rows=*/1, /*row_width=*/2),
      "padding must use -1");
}

// DFlash drafts the whole block in one forward and replays it position by
// position through these helpers; the tests below pin the invariants the
// DFlash JSON validation relies on.
TEST(DFlashJsonObjectStateTest, MaskedResampleAtRejectionReplaysCleanly) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states =
      make_builtin_test_states(grammar, 1);
  // Rejection sampling under the frozen-state mask accepts the legal prefix,
  // resamples the illegal position from the target's masked distribution
  // (":" here), and pads the rest of the row with -1.
  const int64_t accepted_tokens[] = {0, 1, 2, -1, -1, -1};

  const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
      detail::verify_json_accepted_rows(
          initial_states, accepted_tokens, /*num_rows=*/1, /*row_width=*/6);

  EXPECT_TRUE(mismatches.empty());
}

TEST(DFlashJsonObjectStateTest, ReplayStopsAfterTerminalToken) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<ConstraintState> initial_states =
      make_builtin_test_states(grammar, 1);
  // Slots after the stop token are outside the accepted row, regardless of
  // whether they contain zero padding or a nonzero reused token id.
  const int64_t accepted_tokens[] = {0, 1, 2, 3, 4, 5, 0, 1};

  const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
      detail::verify_json_accepted_rows(
          initial_states, accepted_tokens, /*num_rows=*/1, /*row_width=*/8);

  EXPECT_TRUE(mismatches.empty());
}

}  // namespace
}  // namespace xllm
