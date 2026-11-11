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

#include "core/framework/constraint/builtin_json_machine.h"

#include <utility>

namespace xllm {

namespace {

ConstraintState wrap(const BuiltinJsonConstraintMachine& machine,
                     JsonObjectGrammarState state) {
  return ConstraintState(
      std::make_shared<JsonObjectGrammarState>(std::move(state)),
      machine.shared_from_this());
}

}  // namespace

BuiltinJsonConstraintMachine::BuiltinJsonConstraintMachine(
    std::shared_ptr<const JsonObjectGrammar> grammar)
    : grammar_(std::move(grammar)) {
  CHECK(grammar_ != nullptr) << "builtin constraint machine needs a grammar";
}

const JsonObjectGrammarState& BuiltinJsonConstraintMachine::cast(
    const ConstraintState& state) {
  CHECK(state.initialized()) << "builtin constraint op on an empty state";
  return state.as<JsonObjectGrammarState>();
}

JsonObjectGrammarState& BuiltinJsonConstraintMachine::cast(
    ConstraintState& state) {
  CHECK(state.initialized()) << "builtin constraint op on an empty state";
  return state.as<JsonObjectGrammarState>();
}

ConstraintState BuiltinJsonConstraintMachine::initial_state(
    bool reasoning_phase) const {
  return wrap(*this, grammar_->initial_state(reasoning_phase));
}

ConstraintState BuiltinJsonConstraintMachine::restore(
    const ConstraintSnapshot& snapshot) const {
  return wrap(*this,
              grammar_->restore_from_snapshot(to_grammar_snapshot(snapshot)));
}

bool BuiltinJsonConstraintMachine::can_accept(const ConstraintState& state,
                                              int32_t token_id) const {
  const JsonObjectGrammarState& s = cast(state);
  // Explicit validity gate for adapter uniformity: the trial copy inside
  // can_accept_token already propagates valid_ (accept_token rejects an
  // invalid state first), but mirroring the xgrammar adapter's explicit
  // check keeps the freeze contract visible at this layer too.
  return s.is_valid() && s.can_accept_token(token_id);
}

bool BuiltinJsonConstraintMachine::is_valid(
    const ConstraintState& state) const {
  return cast(state).is_valid();
}

bool BuiltinJsonConstraintMachine::is_complete(
    const ConstraintState& state) const {
  return cast(state).is_complete();
}

bool BuiltinJsonConstraintMachine::advance(ConstraintState& state,
                                           int32_t token_id,
                                           HaltPolicy halt) const {
  JsonObjectGrammarState& s = cast(state);
  const bool accepted = s.accept_token(token_id);
  if (!accepted && halt == HaltPolicy::RESPECT) {
    // Same RESPECT freeze the xgrammar adapter applies: a rejected token
    // halts the state, so later positions see a stopped matcher instead of
    // re-trying different tokens at the same grammar position. accept_token
    // on an already-invalid state returns false, so the freeze is sticky.
    s.invalidate();
  }
  return accepted;
}

ConstraintState BuiltinJsonConstraintMachine::fork(
    const ConstraintState& state) const {
  return wrap(*this, cast(state));
}

ConstraintState BuiltinJsonConstraintMachine::replay(
    const std::vector<int32_t>& committed_tokens,
    bool reasoning_phase) const {
  JsonObjectGrammarSnapshot snapshot;
  snapshot.enabled = true;
  // The snapshot field records the reasoning flow (capability), while the
  // argument is the requested initial phase; a phase request on a grammar
  // without reasoning markers must not enter the phase, because it could
  // never terminate and would silently disable the constraint (every token
  // is accepted while in reasoning).
  snapshot.reasoning_enabled = grammar_->has_reasoning() && reasoning_phase;
  snapshot.token_ids = committed_tokens;
  return wrap(*this, grammar_->restore_from_snapshot(snapshot));
}

std::vector<uint32_t> BuiltinJsonConstraintMachine::allowed_bitmask(
    const ConstraintState& state) const {
  return grammar_->allowed_token_bitmask(cast(state));
}

std::shared_ptr<const std::vector<uint32_t>>
BuiltinJsonConstraintMachine::shared_allowed_bitmask(
    const ConstraintState& state) const {
  // Aliasing shared_ptr onto the grammar's cached mask: no deep copy, and the
  // cache entry stays alive as long as any caller holds the view.
  return grammar_->shared_cached_bitmask(cast(state));
}

uint64_t BuiltinJsonConstraintMachine::fingerprint(
    const ConstraintState& state) const {
  return cast(state).fingerprint();
}

ConstraintSnapshot BuiltinJsonConstraintMachine::snapshot(
    const ConstraintState& state) const {
  return to_constraint_snapshot(cast(state).snapshot());
}

std::string BuiltinJsonConstraintMachine::error_detail(
    const ConstraintState& state) const {
  const JsonObjectGrammarState& s = cast(state);
  return "builtin json constraint rejected token; state_fingerprint=" +
         std::to_string(s.fingerprint()) +
         ", committed_tokens=" + std::to_string(s.snapshot().token_ids.size());
}

}  // namespace xllm
