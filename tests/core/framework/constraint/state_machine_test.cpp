/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <memory>
#include <unordered_set>
#include <vector>

#include "core/framework/constraint/builtin_json_machine.h"
#include "core/framework/sampling/json_object_grammar.h"

namespace xllm {
namespace {

// 0:"{" 1:"}" 2:"\"" 3:"a" 4:"b" 5:":" 6:"," 7:"[" 8:"]" 9:"true" 12:"1"
std::shared_ptr<const JsonObjectGrammar> make_grammar() {
  return std::make_shared<JsonObjectGrammar>(std::vector<std::string>{
      "{", "}", "\"", "a", "b", ":", ",", "[",  "]", "true", "false", "null",
      "1", "-", "0",  ".", "2", "e", " ", "\\", "n", "u",    "0041",  "x"});
}

std::shared_ptr<BuiltinJsonConstraintMachine> make_machine() {
  return wrap_builtin_machine(make_grammar());
}

const std::vector<int32_t> kDocTokens = {0, 2, 3, 2, 5, 12, 1};  // {"a":1}

TEST(ConstraintStateMachineTest, InitialStateAcceptsDocumentPrefix) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState state = machine->initial_state(/*reasoning_phase=*/false);
  ASSERT_TRUE(static_cast<bool>(state));
  EXPECT_TRUE(machine->is_valid(state));
  EXPECT_FALSE(machine->is_complete(state));
  EXPECT_TRUE(machine->can_accept(state, 0));   // "{"
  EXPECT_FALSE(machine->can_accept(state, 1));  // "}"
}

TEST(ConstraintStateMachineTest, AdvanceCommitsTokensAndDetectsCompletion) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState state = machine->initial_state(false);
  for (const int32_t token : kDocTokens) {
    EXPECT_TRUE(machine->advance(state, token, HaltPolicy::RESPECT));
  }
  EXPECT_TRUE(machine->is_complete(state));

  ConstraintState bad = machine->initial_state(false);
  EXPECT_FALSE(machine->advance(bad, 1, HaltPolicy::IGNORE));  // "}" first
  EXPECT_FALSE(machine->is_valid(bad));
}

TEST(ConstraintStateMachineTest, RespectHaltFreezesPositionalRejections) {
  // A stop token offered before the root is complete is a positional
  // rejection (no piece-level violation): under RESPECT the state must
  // freeze exactly like the xgrammar adapter, while IGNORE keeps it usable
  // for the next accepted token.
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", "stop"},
          std::unordered_set<int32_t>{2});
  const std::shared_ptr<BuiltinJsonConstraintMachine> machine =
      wrap_builtin_machine(grammar);

  ConstraintState respect = machine->initial_state(false);
  EXPECT_FALSE(machine->advance(respect, /*stop=*/2, HaltPolicy::RESPECT));
  EXPECT_FALSE(machine->is_valid(respect));
  // The freeze is sticky: even an acceptable token is rejected afterwards.
  EXPECT_FALSE(machine->advance(respect, /*open=*/0, HaltPolicy::RESPECT));

  ConstraintState ignore = machine->initial_state(false);
  EXPECT_FALSE(machine->advance(ignore, /*stop=*/2, HaltPolicy::IGNORE));
  EXPECT_TRUE(machine->is_valid(ignore));
  EXPECT_TRUE(machine->advance(ignore, /*open=*/0, HaltPolicy::IGNORE));
}

TEST(ConstraintStateMachineTest, ReplayClampsTheReasoningPhaseToCapability) {
  std::shared_ptr<const JsonObjectGrammar> plain =
      std::make_shared<JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", "stop"},
          std::unordered_set<int32_t>{2});
  const std::shared_ptr<BuiltinJsonConstraintMachine> machine =
      wrap_builtin_machine(plain);
  // A reasoning-phase request on a grammar without end markers must not
  // enter the phase: it could never terminate, and every token would be
  // accepted, silently disabling the constraint. The replayed state stays
  // JSON-constrained instead.
  const ConstraintState clamped =
      machine->replay(/*committed_tokens=*/{}, /*reasoning_phase=*/true);
  EXPECT_FALSE(machine->can_accept(clamped, /*"}" first*/ 1));

  // A reasoning-capable grammar with the end marker in the committed
  // history exits the phase during replay and enforces JSON afterwards.
  std::shared_ptr<const JsonObjectGrammar> reasoning =
      std::make_shared<JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", "</think", "stop"},
          std::unordered_set<int32_t>{3},
          std::vector<int32_t>{2});
  const std::shared_ptr<BuiltinJsonConstraintMachine> reasoning_machine =
      wrap_builtin_machine(reasoning);
  const ConstraintState exited = reasoning_machine->replay(
      /*committed_tokens=*/{2, 0}, /*reasoning_phase=*/true);
  EXPECT_TRUE(reasoning_machine->can_accept(exited, /*close empty*/ 1));
  EXPECT_FALSE(reasoning_machine->can_accept(exited, /*marker-as-key*/ 2));
}

TEST(ConstraintStateMachineTest, SnapshotRestoresTheFrozenFlag) {
  // A RESPECT positional rejection freezes the state without touching the
  // committed history; the snapshot must round-trip the invalid flag so the
  // restored state stays frozen instead of replaying back to valid.
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", "stop"},
          std::unordered_set<int32_t>{2});
  const std::shared_ptr<BuiltinJsonConstraintMachine> machine =
      wrap_builtin_machine(grammar);

  ConstraintState frozen = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(frozen, /*open=*/0, HaltPolicy::RESPECT));
  EXPECT_FALSE(machine->advance(frozen, /*stop-early=*/2, HaltPolicy::RESPECT));
  ASSERT_FALSE(machine->is_valid(frozen));

  const ConstraintState restored = machine->restore(machine->snapshot(frozen));
  EXPECT_FALSE(machine->is_valid(restored));
  EXPECT_FALSE(machine->can_accept(restored, /*open=*/0));
}

TEST(ConstraintStateMachineTest, HandleAdvanceForksSharedPayloads) {
  // Copy-on-write guard: advancing through a shared handle must not be
  // observed by the other copies, so a channel state that still aliases a
  // live payload cannot double-advance it under schedule overlap.
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState original = machine->initial_state(false);
  const uint64_t fingerprint_before = machine->fingerprint(original);
  const bool accepts_open_before = machine->can_accept(original, 0);

  ConstraintState aliased = original;  // handle copy: shares the payload
  ASSERT_TRUE(aliased.advance(0, HaltPolicy::IGNORE));

  // The guard forked the shared payload before advancing: the original
  // state is untouched and still sits at its pre-advance position.
  EXPECT_EQ(machine->fingerprint(original), fingerprint_before);
  EXPECT_EQ(machine->can_accept(original, 0), accepts_open_before);
  // The advancing handle moved on.
  EXPECT_NE(machine->fingerprint(aliased), fingerprint_before);
}

TEST(ConstraintStateMachineTest, ForkDivergesIndependentlyFromSource) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState base = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(base, 0, HaltPolicy::RESPECT));

  ConstraintState branch = machine->fork(base);
  EXPECT_TRUE(machine->advance(branch, 2, HaltPolicy::RESPECT));  // "\""
  EXPECT_TRUE(machine->can_accept(branch, 3));                    // "a"

  // The source state is unaffected by the branch's mutation.
  EXPECT_FALSE(machine->can_accept(base, 3));
  EXPECT_TRUE(machine->can_accept(base, 2));
}

TEST(ConstraintStateMachineTest, SnapshotRestoreRoundTripsState) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState state = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(state, 0, HaltPolicy::RESPECT));
  ASSERT_TRUE(machine->advance(state, 2, HaltPolicy::RESPECT));

  const ConstraintSnapshot snap = machine->snapshot(state);
  ASSERT_TRUE(snap.enabled);
  ASSERT_EQ(snap.token_ids.size(), 2U);

  ConstraintState restored = machine->restore(snap);
  EXPECT_TRUE(machine->is_valid(restored));
  EXPECT_TRUE(machine->can_accept(restored, 3));  // "a" after "{\""
  EXPECT_EQ(machine->fingerprint(restored), machine->fingerprint(state));
}

TEST(ConstraintStateMachineTest, ReplayRebuildsFromCommittedHistory) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState state = machine->replay(kDocTokens, false);
  EXPECT_TRUE(machine->is_valid(state));
  EXPECT_TRUE(machine->is_complete(state));
  EXPECT_EQ(machine->fingerprint(state),
            machine->fingerprint(machine->replay(kDocTokens, false)));

  const std::vector<int32_t> bad_history = {1, 0};  // "}" before "{"
  ConstraintState bad = machine->replay(bad_history, false);
  EXPECT_FALSE(machine->is_valid(bad));
}

namespace {

void expect_bitmask_matches_can_accept(const ConstraintStateMachine& machine,
                                       const ConstraintState& state) {
  const std::vector<uint32_t> bitmask = machine.allowed_bitmask(state);
  ASSERT_EQ(bitmask.size(), 1U);  // 24 tokens -> one word
  for (int32_t token = 0; token < 24; ++token) {
    const bool allowed = (bitmask[0] & (1U << token)) != 0U;
    EXPECT_EQ(allowed, machine.can_accept(state, token)) << "token " << token;
  }
}

}  // namespace

TEST(ConstraintStateMachineTest, AllowedBitmaskMatchesCanAccept) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState state = machine->initial_state(false);
  expect_bitmask_matches_can_accept(*machine, state);
  EXPECT_EQ(machine->fingerprint(state),
            machine->fingerprint(machine->initial_state(false)));

  // The equivalence must survive advancement: the mask cache is keyed by
  // fingerprint, so a second FSM position must produce a different
  // fingerprint and a matching different mask.
  ASSERT_TRUE(machine->advance(state, 0, HaltPolicy::RESPECT));  // "{"
  ASSERT_TRUE(machine->advance(state, 2, HaltPolicy::RESPECT));  // "\""
  expect_bitmask_matches_can_accept(*machine, state);
  const uint64_t advanced_fingerprint = machine->fingerprint(state);
  EXPECT_NE(advanced_fingerprint,
            machine->fingerprint(machine->initial_state(false)));
}

TEST(ConstraintStateMachineTest, ErrorDetailNamesTheRejectedState) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState state = machine->initial_state(false);
  const std::string detail = machine->error_detail(state);
  EXPECT_NE(detail.find("state_fingerprint"), std::string::npos);
  EXPECT_NE(detail.find("committed_tokens"), std::string::npos);
}

TEST(ConstraintStateMachineTest, EmptyStateIsFalsyAndUnsafe) {
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  ConstraintState empty;
  EXPECT_FALSE(static_cast<bool>(empty));
  ConstraintState state = machine->initial_state(false);
  EXPECT_TRUE(static_cast<bool>(state));
}

TEST(ConstraintStateMachineTest, BuiltinAdvertisesSpecSnapshotSupport) {
  // The worker tells 'genuinely unconstrained' (builtin row, empty snapshot)
  // apart from 'backend without spec snapshots' (xgrammar row, empty
  // snapshot) through this flag, so the builtin override must stay true.
  std::shared_ptr<BuiltinJsonConstraintMachine> machine = make_machine();
  EXPECT_TRUE(machine->supports_spec_snapshot());
}

}  // namespace
}  // namespace xllm
