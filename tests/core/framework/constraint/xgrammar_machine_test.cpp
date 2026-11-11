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

#include "core/framework/constraint/xgrammar_machine.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "core/framework/constraint/draft_validation.h"
#include "core/framework/sampling/xgrammar_constraint.h"

namespace xllm {
namespace {

constexpr int32_t kStopTokenId = 5;

std::shared_ptr<XGrammarConstraintMachine> make_machine() {
  auto compiler = std::make_unique<XGrammarCompiler>(
      std::vector<std::string>{"A", "B", "-", "1", "x", "", "{", "}"},
      /*model_vocab_size=*/32,
      std::vector<int32_t>{kStopTokenId},
      /*cache_max_memory_bytes=*/0);
  std::string error;
  auto grammar = compiler->compile_regex("[A-Z]{2}-[0-9]{4}", error);
  if (grammar == nullptr) {
    ADD_FAILURE() << error;
    return nullptr;
  }
  return std::make_shared<XGrammarConstraintMachine>(grammar);
}

TEST(XGrammarMachineTest, InitialStateAcceptsRegexPrefix) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  ASSERT_TRUE(static_cast<bool>(state));
  EXPECT_TRUE(machine->is_valid(state));
  EXPECT_FALSE(machine->is_complete(state));
  EXPECT_TRUE(machine->can_accept(state, 0));   // "A"
  EXPECT_FALSE(machine->can_accept(state, 4));  // "x"
}

TEST(XGrammarMachineTest, AdvanceCommitsTokensAndDetectsCompletion) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  for (const int32_t token_id : {0, 1, 2, 3, 3, 3, 3}) {
    EXPECT_TRUE(machine->advance(state, token_id, HaltPolicy::IGNORE));
  }
  EXPECT_TRUE(machine->is_complete(state));
}

TEST(XGrammarMachineTest, HaltRespectFreezesOnFirstRejection) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);

  EXPECT_FALSE(machine->advance(state, 4, HaltPolicy::RESPECT));  // "x"
  EXPECT_FALSE(machine->is_valid(state));
  // Frozen: everything reports rejected, including valid tokens.
  EXPECT_FALSE(machine->can_accept(state, 0));
  EXPECT_FALSE(machine->advance(state, 0, HaltPolicy::RESPECT));
  // allowed_bitmask agrees with can_accept on a frozen row: all-deny, so the
  // frozen mask/can_accept invariant holds (the grammar-legal token is denied).
  const auto frozen_mask = machine->allowed_bitmask(state);
  ASSERT_FALSE(frozen_mask.empty());
  for (const uint32_t word : frozen_mask) {
    EXPECT_EQ(word, 0u);
  }

  // IGNORE keeps the matcher usable after a rejection (target-verify rows).
  ConstraintState verify = machine->initial_state(false);
  EXPECT_FALSE(machine->advance(verify, 4, HaltPolicy::IGNORE));
  EXPECT_TRUE(machine->is_valid(verify));
  EXPECT_TRUE(machine->advance(verify, 0, HaltPolicy::IGNORE));
}

TEST(XGrammarMachineTest, ForkDivergesIndependentlyFromSource) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState base = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(base, 0, HaltPolicy::IGNORE));

  ConstraintState branch = machine->fork(base);
  EXPECT_EQ(machine->fingerprint(branch), machine->fingerprint(base));
  // "x" is not accepted at the second-letter position; RESPECT freezes the
  // branch while the source keeps its position.
  EXPECT_FALSE(machine->advance(branch, 4, HaltPolicy::RESPECT));
  EXPECT_FALSE(machine->is_valid(branch));
  EXPECT_TRUE(machine->is_valid(base));
  EXPECT_NE(machine->fingerprint(branch), machine->fingerprint(base));
}

TEST(XGrammarMachineTest, SnapshotRestoreRoundTripsHistory) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(state, 0, HaltPolicy::IGNORE));
  ASSERT_TRUE(machine->advance(state, 1, HaltPolicy::IGNORE));

  const ConstraintSnapshot snap = machine->snapshot(state);
  ASSERT_TRUE(snap.enabled);
  ASSERT_EQ(snap.token_ids, std::vector<int32_t>({0, 1}));

  ConstraintState restored = machine->restore(snap);
  EXPECT_TRUE(machine->is_valid(restored));
  EXPECT_EQ(machine->fingerprint(restored), machine->fingerprint(state));
  EXPECT_TRUE(machine->can_accept(restored, 2));  // "-" after "AB"

  ConstraintState replayed = machine->replay(snap.token_ids, false);
  EXPECT_EQ(machine->fingerprint(replayed), machine->fingerprint(state));

  const std::vector<int32_t> bad_history = {4};
  ConstraintState bad = machine->replay(bad_history, false);
  EXPECT_FALSE(machine->is_valid(bad));
}

TEST(XGrammarMachineTest, FrozenStateRoundTripsItsInvalidFlag) {
  // A RESPECT-frozen state must stay frozen across snapshot/restore: replay
  // alone would return a valid matcher whose fingerprint differs (the invalid
  // bit is hashed), losing the halt.
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  // Token 4 ("x") is illegal at the first position under RESPECT: it freezes.
  EXPECT_FALSE(machine->advance(state, 4, HaltPolicy::RESPECT));
  ASSERT_FALSE(machine->is_valid(state));

  const ConstraintSnapshot snap = machine->snapshot(state);
  ASSERT_TRUE(snap.invalid);

  ConstraintState restored = machine->restore(snap);
  EXPECT_FALSE(machine->is_valid(restored));
  EXPECT_EQ(machine->fingerprint(restored), machine->fingerprint(state));
  // A frozen matcher rejects everything, even a token the bare grammar allows.
  EXPECT_FALSE(machine->can_accept(restored, 0));
}

TEST(XGrammarMachineTest, MachineLevelSnapshotDoesNotAdvertiseSpecSupport) {
  // The machine-level snapshot/restore above round-trips token history, but
  // the batch channel does not carry xgrammar spec snapshots yet (G9): the
  // false default makes the worker preserve the live state on an empty
  // snapshot instead of resetting the row to unconstrained.
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  EXPECT_FALSE(machine->supports_spec_snapshot());
}

TEST(XGrammarMachineTest, DraftValidationAndVerifyWorkOnXgrammarStates) {
  // The shared speculative helpers must treat xgrammar states exactly like
  // builtin ones: draft rows fork before advancing, verification probes
  // per position.
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  // advance_json_draft_step mutates the states it advances; keep the
  // pristine rows for building validation states.
  const std::vector<ConstraintState> initial = {machine->initial_state(false),
                                                ConstraintState()};
  std::vector<ConstraintState> states = initial;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_FALSE(
      detail::advance_json_draft_step(states, invalid_suffix, {0, 4}, scratch));
  EXPECT_FALSE(
      detail::advance_json_draft_step(states, invalid_suffix, {1, 4}, scratch));
  // Unconstrained row 1 accepts anything, row 0 stays on the regex.
  EXPECT_EQ(invalid_suffix, std::vector<uint8_t>({0, 0}));

  std::vector<uint8_t> invalid_draft;
  const std::vector<ConstraintState> validation_states =
      detail::build_json_validation_states(initial, scratch, invalid_draft);
  ASSERT_EQ(validation_states.size(), 6u);  // 2 sequences x (2 draft + live)

  // Target verify: the legal prefix row reports no mismatch, while a row
  // starting with the illegal "x" reports it at offset 0.
  const int64_t legal[] = {0, 1};
  EXPECT_TRUE(
      detail::verify_json_accepted_rows({validation_states[0]}, legal, 1, 2)
          .empty());
  const int64_t illegal[] = {4, -1};
  const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
      detail::verify_json_accepted_rows({validation_states[0]}, illegal, 1, 2);
  ASSERT_EQ(mismatches.size(), 1u);
  EXPECT_EQ(mismatches[0].sequence_index, 0u);
  EXPECT_EQ(mismatches[0].token_id, 4);
}

TEST(XGrammarMachineTest, AllowedBitmaskMatchesCanAccept) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  const std::vector<uint32_t> bitmask = machine->allowed_bitmask(state);
  ASSERT_EQ(bitmask.size(), 1u);  // 32-token vocab -> one word

  for (int32_t token = 0; token < 8; ++token) {
    const bool allowed = (bitmask[0] & (1U << token)) != 0U;
    EXPECT_EQ(allowed, machine->can_accept(state, token)) << "token " << token;
  }
  EXPECT_EQ(machine->fingerprint(state),
            machine->fingerprint(machine->initial_state(false)));
}

TEST(XGrammarMachineTest, ErrorDetailNamesTheRejectedState) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  const std::string detail = machine->error_detail(state);
  EXPECT_NE(detail.find("state_fingerprint"), std::string::npos);
  EXPECT_NE(detail.find("committed_tokens"), std::string::npos);
}

}  // namespace
}  // namespace xllm
