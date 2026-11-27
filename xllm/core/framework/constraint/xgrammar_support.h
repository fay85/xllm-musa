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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/constraint/snapshot.h"
#include "core/framework/constraint/xgrammar_machine.h"
#include "core/framework/model/model_args.h"
#include "core/framework/sampling/xgrammar_constraint.h"
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {

// Holder for the worker- and service-side xgrammar machine pool: compiles
// per-request grammar specs through a lazily built model-scoped compiler
// (mirroring the request factory's construction) and caches the stateless
// machines by spec, so snapshot restores and PD request migrations rebuild
// the exact grammar cheaply. Thread-safe.
class XGrammarMachineCache final {
 public:
  // `key_of` overrides the cache key for tests that need to force the hash
  // collision path deterministically; production callers omit it and get the
  // canonical fnv1a spec key.
  XGrammarMachineCache(
      const Tokenizer& tokenizer,
      ModelArgs model_args,
      std::function<uint64_t(const GrammarSpec&)> key_of = nullptr);

  // Returns the machine for the spec; nullptr + error on compile failure
  // or in builds without USE_XGRAMMAR.
  //
  // Entries hold weak references: live rows keep their machine alive
  // through the state's aliasing shared_ptr, and once a request finishes
  // its entry expires with the machine, so the compiler's own size-bounded
  // cache (not this map) governs the compiled-grammar memory. The map key
  // is a 64-bit hash of the spec (no per-step source copies); colliding
  // hashes share a bucket whose rows each verify against the stored spec,
  // so a collision can neither serve another request's grammar nor skip
  // caching behind a per-request recompile.
  std::shared_ptr<XGrammarConstraintMachine> get(const GrammarSpec& spec,
                                                 std::string& error);

 private:
  struct CacheEntry final {
    GrammarSpec spec;  // verifies the hash hit
    std::weak_ptr<XGrammarConstraintMachine> machine;
  };

  const Tokenizer& tokenizer_;
  ModelArgs model_args_;
  std::mutex mutex_;
  std::unique_ptr<XGrammarCompiler> compiler_;
  // Key override for tests that force the collision path; empty in
  // production, where the canonical fnv1a spec key applies.
  std::function<uint64_t(const GrammarSpec&)> key_of_;
  // One bucket per 64-bit key; each row is one spec sharing that hash.
  std::unordered_map<uint64_t, std::vector<CacheEntry>> machines_;
};

// Runtime feature switches that gate the xgrammar structured-output backend.
// Mirrors the engine config so the support check is a pure, testable function.
struct XGrammarSupportInputs final {
  bool enable_service_routing = false;
  bool enable_prefix_cache = false;
  int32_t num_speculative_tokens = 0;
  std::string speculative_algorithm = "MTP";
};

// Features the xgrammar backend does not yet support, as human-readable names
// for the startup error. Empty means every listed feature is supported.
// Speculative decoding and schedule overlap were unlocked by the constraint
// state-machine migration (batches 4-6); the rest stay gated until their
// combinations are verified.
std::vector<std::string> xgrammar_unsupported_features(
    const XGrammarSupportInputs& inputs);

}  // namespace xllm
