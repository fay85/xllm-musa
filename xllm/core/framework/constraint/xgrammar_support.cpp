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

#include "core/framework/constraint/xgrammar_support.h"

#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/sampling/fnv1a.h"

namespace xllm {

std::vector<std::string> xgrammar_unsupported_features(
    const XGrammarSupportInputs& inputs) {
  std::vector<std::string> unsupported;
  if (inputs.num_speculative_tokens > 0 &&
      !SpeculativeConfig::drives_constraint_channel(
          inputs.speculative_algorithm)) {
    unsupported.emplace_back("speculative_" +
                             (inputs.speculative_algorithm.empty()
                                  ? std::string("<unnamed>")
                                  : inputs.speculative_algorithm));
  }
  // Each gated feature names the unverified constraint-state interaction that
  // keeps it off (the gate is conservative: unverified, not known-broken).
  //
  // Graph and prefill_piecewise_graph are NOT gated: the constraint mask
  // never enters the captured graph. The graph executors capture only the
  // model forward (acl_graph_executor_impl.cpp builds the graph over
  // model->forward); every mask application site runs eagerly outside the
  // graph on the replayed logits in the target sampler or the speculative
  // rejection sampler. The builtin backend uses the same eager mask path with
  // graph enabled in production, and piecewise graph is the graph
  // implementation of prefill segmentation - the matcher initializes at the
  // first decode token (the same argument that leaves chunked_prefill
  // ungated), so prompt segment boundaries cannot touch it.
  // chunked_prefill is not gated: it only splits the prompt across steps, and
  // the constraint matcher is initialized at the first decode token (after
  // prefill), so the chunk boundary cannot affect the decode-time grammar. It
  // is also a hard requirement of the block-diffusion family, so gating it
  // would make the unlocked speculative algorithms unstartable.
  // Regular disagg PD is open: the real cross-instance multi-token
  // constrained decode verification passed (1P1D on the production-aligned
  // Mooncake pin; the spec rides the KV first generation, the decode side
  // rebuilds through the machine cache and applies the FSM mask per token).
  // PD OOC stays fail-closed: its scheduler does not populate the grammar
  // fields on the allocation request yet, so a migrated request would lose
  // its grammar.
  if (inputs.enable_pd_ooc) {
    unsupported.emplace_back("pd_ooc");
  }
  // Service routing uses the same serialized constraint snapshots as regular
  // disagg PD. XGrammar snapshots carry the grammar identity and committed
  // token history, so a receiving worker rebuilds the exact machine before
  // applying the next mask instead of dropping or resetting the constraint.
  //
  // Prefix caching only reuses prompt KV. The response grammar starts at the
  // first generated token, after prefill, so a cached prompt prefix does not
  // advance or otherwise change the response matcher.
  return unsupported;
}

namespace {
// Dead weak-entry sweep threshold: amortized cleanup while inserting.
constexpr size_t kSweepThreshold = 1024;
}  // namespace

XGrammarMachineCache::XGrammarMachineCache(
    const Tokenizer& tokenizer,
    ModelArgs model_args,
    std::function<uint64_t(const GrammarSpec&)> key_of)
    : tokenizer_(tokenizer),
      model_args_(std::move(model_args)),
      key_of_(std::move(key_of)) {}

std::shared_ptr<XGrammarConstraintMachine> XGrammarMachineCache::get(
    const GrammarSpec& spec,
    std::string& error) {
  if (spec.empty() || !GrammarSpec::source_within_limit(spec.source) ||
      (spec.type != GrammarSpec::Type::JSON_SCHEMA &&
       spec.type != GrammarSpec::Type::REGEX &&
       spec.type != GrammarSpec::Type::EBNF)) {
    error = "invalid grammar spec type or source length";
    return nullptr;
  }
  // Hash key: the lookup allocates nothing regardless of the source size.
  // fnv1a over the spec fields; the stored spec verifies the hit, so a
  // collision cannot silently serve a different grammar.
  const uint64_t key =
      key_of_ ? key_of_(spec)
              : fnv1a_byte(fnv1a_byte(fnv1a(spec.source),
                                      static_cast<uint8_t>(spec.type)),
                           spec.strict ? 1 : 0);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = machines_.find(key);
    if (it != machines_.end()) {
      for (const CacheEntry& entry : it->second) {
        if (entry.spec.type == spec.type && entry.spec.strict == spec.strict &&
            entry.spec.source == spec.source) {
          std::shared_ptr<XGrammarConstraintMachine> machine =
              entry.machine.lock();
          if (machine != nullptr) {
            return machine;
          }
        }
      }
    }
  }
  // Build the compiler and compile OUTSIDE the lock: a cold compile can
  // take milliseconds, and holding the mutex through it would stall every
  // other thread's restore.
#if defined(USE_XGRAMMAR)
  // Publish the compiler once and never replace it: callers can safely use
  // this non-owning pointer after releasing the cache mutex. The compiler
  // serializes its own third-party calls without blocking cache hits.
  XGrammarCompiler* compiler = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    compiler = compiler_.get();
  }
  if (compiler == nullptr) {
    std::unique_ptr<XGrammarCompiler> fresh_compiler = XGrammarCompiler::create(
        tokenizer_,
        model_args_,
        ServiceConfig::get_instance().xgrammar_cache_max_memory_bytes(),
        error);
    if (fresh_compiler == nullptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (compiler_ == nullptr) {
      compiler_ = std::move(fresh_compiler);
    }
    compiler = compiler_.get();
  }
  std::shared_ptr<const CompiledXGrammar> grammar;
  if (spec.type == GrammarSpec::Type::REGEX) {
    grammar = compiler->compile_regex(spec.source, error);
  } else if (spec.type == GrammarSpec::Type::EBNF) {
    grammar = compiler->compile_ebnf(spec.source, error);
  } else {
    grammar = compiler->compile_json_schema(spec.source, spec.strict, error);
  }
  if (grammar == nullptr) {
    return nullptr;
  }
  auto machine = wrap_xgrammar_machine(std::move(grammar), spec);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Sweep expired entries while inserting: dead weak references cost a
    // map node each, and the sweep keeps the structure flat without
    // evicting anything live.
    if (machines_.size() >= kSweepThreshold) {
      for (auto it = machines_.begin(); it != machines_.end();) {
        std::erase_if(it->second, [](const CacheEntry& entry) {
          return entry.machine.expired();
        });
        it = it->second.empty() ? machines_.erase(it) : std::next(it);
      }
    }
    // Colliding specs share one bucket, each spec in its own row: a
    // collision must still cache, otherwise every request with the
    // colliding spec pays a cold compile.
    std::vector<CacheEntry>& bucket = machines_[key];
    for (CacheEntry& entry : bucket) {
      if (entry.spec.type == spec.type && entry.spec.strict == spec.strict &&
          entry.spec.source == spec.source) {
        // A racing thread populated the row first; prefer the winner so
        // all rows share one machine per spec.
        if (auto winner = entry.machine.lock()) {
          return winner;
        }
        entry = CacheEntry{spec, machine};
        return machine;
      }
    }
    bucket.push_back(CacheEntry{spec, machine});
    return machine;
  }
#else
  error = "build with USE_XGRAMMAR=ON to rebuild per-request grammars";
  return nullptr;
#endif
}

}  // namespace xllm
