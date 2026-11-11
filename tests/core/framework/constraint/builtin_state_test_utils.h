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
===============================================================================*/

#pragma once

#include <memory>
#include <vector>

#include "core/framework/constraint/builtin_json_machine.h"
#include "core/framework/sampling/json_object_grammar.h"

namespace xllm {

// Shared helpers for tests that need builtin constraint states over a
// caller-owned grammar. The machine aliases (does not own) the grammar, so
// the grammar must outlive every state produced here; centralizing the
// aliasing construction keeps the deliberately non-owning pattern in one
// place, so a change in the machine construction constraints only has to
// follow this helper instead of every test file.
inline std::shared_ptr<BuiltinJsonConstraintMachine> make_builtin_test_machine(
    const JsonObjectGrammar& grammar) {
  return wrap_builtin_machine(std::shared_ptr<const JsonObjectGrammar>(
      std::shared_ptr<const JsonObjectGrammar>(), &grammar));
}

inline std::vector<ConstraintState> make_builtin_test_states(
    const JsonObjectGrammar& grammar,
    size_t count,
    bool reasoning_phase = false) {
  auto machine = make_builtin_test_machine(grammar);
  std::vector<ConstraintState> states;
  states.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    states.push_back(machine->initial_state(reasoning_phase));
  }
  return states;
}

inline ConstraintState make_builtin_test_state(const JsonObjectGrammar& grammar,
                                               bool reasoning_phase = false) {
  return make_builtin_test_machine(grammar)->initial_state(reasoning_phase);
}

}  // namespace xllm
