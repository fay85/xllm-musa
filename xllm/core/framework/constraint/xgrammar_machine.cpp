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

#include <glog/logging.h>

#include <utility>

#include "core/framework/sampling/fnv1a.h"

#if defined(USE_XGRAMMAR)
#include <xgrammar/xgrammar.h>
#endif

namespace xllm {

namespace {

ConstraintState wrap(const XGrammarConstraintMachine& machine,
                     XGrammarState state) {
  return ConstraintState(std::make_shared<XGrammarState>(std::move(state)),
                         machine.shared_from_this());
}

}  // namespace

XGrammarConstraintMachine::XGrammarConstraintMachine(
    std::shared_ptr<const CompiledXGrammar> grammar)
    : grammar_(std::move(grammar)) {
  CHECK(grammar_ != nullptr) << "xgrammar constraint machine needs a grammar";
  // Serialize once per grammar: initial/restore/replay construct states on the
  // draft hot path and reuse this cached id instead of re-serializing.
  grammar_id_ = fnv1a(grammar_->SerializeJSON());
}

const XGrammarState& XGrammarConstraintMachine::cast(
    const ConstraintState& state) {
  return state.as<XGrammarState>();
}

XGrammarState& XGrammarConstraintMachine::cast(ConstraintState& state) {
  return state.as<XGrammarState>();
}

ConstraintState XGrammarConstraintMachine::initial_state(
    bool /*reasoning_phase*/) const {
  return wrap(*this, XGrammarState(*grammar_, grammar_id_));
}

ConstraintState XGrammarConstraintMachine::restore(
    const ConstraintSnapshot& snapshot) const {
  if (!snapshot.enabled) {
    // A disabled snapshot means an unconstrained row; do not rebuild a
    // constrained matcher for it (matching the builtin restore semantics).
    return ConstraintState();
  }
  ConstraintState state = replay(snapshot.token_ids, /*reasoning_phase=*/false);
  if (snapshot.invalid) {
    // A frozen state must stay frozen across the snapshot round-trip; replay
    // alone would return a valid matcher with a different fingerprint.
    cast(state).set_invalid();
  }
  return state;
}

bool XGrammarConstraintMachine::can_accept(const ConstraintState& state,
                                           int32_t token_id) const {
  // Logically read-only: the probe refreshes matcher-internal scratch.
  XGrammarState& s = cast(const_cast<ConstraintState&>(state));
  return s.is_valid() && s.can_accept(token_id);
}

bool XGrammarConstraintMachine::is_valid(const ConstraintState& state) const {
  return cast(state).is_valid();
}

bool XGrammarConstraintMachine::is_complete(
    const ConstraintState& state) const {
  const XGrammarState& s = cast(state);
  return s.is_valid() && s.is_completed();
}

bool XGrammarConstraintMachine::advance(ConstraintState& state,
                                        int32_t token_id,
                                        HaltPolicy halt) const {
  XGrammarState& s = cast(state);
  if (!s.is_valid()) {
    return false;
  }
  const bool accepted = s.accept_token(token_id);
  if (!accepted && halt == HaltPolicy::RESPECT) {
    s.set_invalid();
  }
  return accepted;
}

ConstraintState XGrammarConstraintMachine::fork(
    const ConstraintState& state) const {
  return wrap(*this, cast(state).fork());
}

ConstraintState XGrammarConstraintMachine::replay(
    const std::vector<int32_t>& committed_tokens,
    bool /*reasoning_phase*/) const {
  // Reuse the cached grammar id, like initial_state, so replaying each draft
  // position does not re-serialize the grammar.
  XGrammarState state(*grammar_, grammar_id_);
  for (const int32_t token_id : committed_tokens) {
    if (!state.accept_token(token_id)) {
      state.set_invalid();
      break;
    }
  }
  return wrap(*this, std::move(state));
}

std::vector<uint32_t> XGrammarConstraintMachine::allowed_bitmask(
    const ConstraintState& state) const {
  // Delegates to the state's own mask: one implementation shared with the
  // FSM-only fingerprint (frozen = all-deny, unconstrained = all-allow).
  return cast(state).allowed_bitmask();
}

uint64_t XGrammarConstraintMachine::fingerprint(
    const ConstraintState& state) const {
  return cast(state).fingerprint();
}

size_t XGrammarConstraintMachine::bitmask_num_words() const {
  const int64_t vocab_size = grammar_->GetTokenizerInfo().GetVocabSize();
  return static_cast<size_t>((vocab_size + 31) / 32);
}

ConstraintSnapshot XGrammarConstraintMachine::snapshot(
    const ConstraintState& state) const {
  ConstraintSnapshot out;
  out.enabled = true;
  out.reasoning_enabled = false;
  out.token_ids = cast(state).accepted_tokens();
  out.invalid = !cast(state).is_valid();
  return out;
}

std::string XGrammarConstraintMachine::error_detail(
    const ConstraintState& state) const {
  const XGrammarState& s = cast(state);
  return "xgrammar constraint rejected token; state_fingerprint=" +
         std::to_string(s.fingerprint()) +
         ", committed_tokens=" + std::to_string(s.accepted_tokens().size());
}

}  // namespace xllm
