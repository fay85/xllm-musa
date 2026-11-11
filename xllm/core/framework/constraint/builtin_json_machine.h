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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/framework/constraint/state_machine.h"
#include "core/framework/sampling/json_object_grammar.h"

namespace xllm {

// Field-for-field conversion between the builtin and transport snapshot
// forms; kept explicit until the state channels carry ConstraintSnapshot.
inline ConstraintSnapshot to_constraint_snapshot(
    const JsonObjectGrammarSnapshot& s) {
  ConstraintSnapshot out;
  out.enabled = s.enabled;
  out.reasoning_enabled = s.reasoning_enabled;
  out.token_ids = s.token_ids;
  out.invalid = s.invalid;
  return out;
}

// Inverse mapping for channels that carry the builtin snapshot form.
inline JsonObjectGrammarSnapshot to_grammar_snapshot(
    const ConstraintSnapshot& s) {
  JsonObjectGrammarSnapshot out;
  out.enabled = s.enabled;
  out.reasoning_enabled = s.reasoning_enabled;
  out.token_ids = s.token_ids;
  out.invalid = s.invalid;
  return out;
}

// Builtin-JSON adapter: every operation delegates to JsonObjectGrammar with
// no added grammar logic. Piece-level violations already invalidate the
// state in the grammar itself (both policies); advance() adds the missing
// RESPECT half of the HaltPolicy contract so positional rejections (a stop
// token before the root completes, a token past a completed root) freeze
// the state exactly like the xgrammar adapter does.

class BuiltinJsonConstraintMachine final
    : public ConstraintStateMachine,
      public std::enable_shared_from_this<BuiltinJsonConstraintMachine> {
 public:
  // Construct through wrap_builtin_machine only: states embed the machine
  // via shared_from_this, so a stack or directly-owned instance is unusable.
  friend std::shared_ptr<BuiltinJsonConstraintMachine> wrap_builtin_machine(
      std::shared_ptr<const JsonObjectGrammar> grammar);

  ConstraintState initial_state(bool reasoning_phase) const override;
  ConstraintState restore(const ConstraintSnapshot& snapshot) const override;

  bool can_accept(const ConstraintState& state,
                  int32_t token_id) const override;
  bool is_valid(const ConstraintState& state) const override;
  bool is_complete(const ConstraintState& state) const override;

  bool advance(ConstraintState& state,
               int32_t token_id,
               HaltPolicy halt) const override;
  ConstraintState fork(const ConstraintState& state) const override;
  ConstraintState replay(const std::vector<int32_t>& committed_tokens,
                         bool reasoning_phase) const override;

  std::vector<uint32_t> allowed_bitmask(
      const ConstraintState& state) const override;
  // Zero-copy view of the grammar's cached mask (shared ownership).
  std::shared_ptr<const std::vector<uint32_t>> shared_allowed_bitmask(
      const ConstraintState& state) const override;
  size_t bitmask_num_words() const override {
    return grammar_->bitmask_num_words();
  }
  // Builtin serializes real matcher snapshots, so an empty snapshot for a
  // builtin row legitimately means 'unconstrained' rather than 'snapshot
  // support missing'; the worker uses this to tell the two apart.
  bool supports_spec_snapshot() const override { return true; }
  uint64_t fingerprint(const ConstraintState& state) const override;

  ConstraintSnapshot snapshot(const ConstraintState& state) const override;
  std::string error_detail(const ConstraintState& state) const override;

 private:
  explicit BuiltinJsonConstraintMachine(
      std::shared_ptr<const JsonObjectGrammar> grammar);

  static const JsonObjectGrammarState& cast(const ConstraintState& state);
  static JsonObjectGrammarState& cast(ConstraintState& state);

  std::shared_ptr<const JsonObjectGrammar> grammar_;
};

// Centralize the "valid grammar implies machine" invariant; null grammar in,
// null machine out.
inline std::shared_ptr<BuiltinJsonConstraintMachine> wrap_builtin_machine(
    std::shared_ptr<const JsonObjectGrammar> grammar) {
  if (grammar == nullptr) {
    return nullptr;
  }
  // Plain new: make_shared's internal construction is not covered by the
  // friend declaration guarding the private constructor.
  return std::shared_ptr<BuiltinJsonConstraintMachine>(
      new BuiltinJsonConstraintMachine(std::move(grammar)));
}

}  // namespace xllm
