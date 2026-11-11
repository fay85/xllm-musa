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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/framework/constraint/state_machine.h"
#include "core/framework/sampling/xgrammar_constraint.h"

namespace xllm {

// XGrammar adapter over the unified interface. Halt semantics: advance with
// HaltPolicy::RESPECT freezes the matcher on the first rejected token
// (subsequent operations report invalid); IGNORE rejects without freezing,
// which target-verify rows use to keep processing the remaining positions.
class XGrammarConstraintMachine final
    : public ConstraintStateMachine,
      public std::enable_shared_from_this<XGrammarConstraintMachine> {
 public:
  explicit XGrammarConstraintMachine(
      std::shared_ptr<const CompiledXGrammar> grammar);

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
  uint64_t fingerprint(const ConstraintState& state) const override;

  ConstraintSnapshot snapshot(const ConstraintState& state) const override;
  std::string error_detail(const ConstraintState& state) const override;

  // Vocabulary row width from the compiled grammar's tokenizer info.
  size_t bitmask_num_words() const override;

 private:
  static const XGrammarState& cast(const ConstraintState& state);
  static XGrammarState& cast(ConstraintState& state);

  std::shared_ptr<const CompiledXGrammar> grammar_;
  uint64_t grammar_id_ = 0;
};

}  // namespace xllm
