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

#include <gtest/gtest.h>

#include <algorithm>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace xllm {
namespace {

bool contains(const std::vector<std::string>& names, const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

TEST(XGrammarSupportTest, PlainEagerDecodeIsSupported) {
  EXPECT_TRUE(xgrammar_unsupported_features(XGrammarSupportInputs{}).empty());
}

TEST(XGrammarSupportTest, ChannelMigratedSpeculativeAlgorithmsAreSupported) {
  for (const char* algorithm :
       {"MTP", "dflash", "dflash2", "DSpark", "suffix"}) {
    XGrammarSupportInputs inputs;
    inputs.num_speculative_tokens = 3;
    inputs.speculative_algorithm = algorithm;
    EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty()) << algorithm;
  }
}

TEST(XGrammarSupportTest, Eagle3StaysGated) {
  XGrammarSupportInputs inputs;
  inputs.num_speculative_tokens = 3;
  inputs.speculative_algorithm = "eagle3";
  const auto unsupported = xgrammar_unsupported_features(inputs);
  ASSERT_EQ(unsupported.size(), 1u);
  EXPECT_TRUE(contains(unsupported, /*name=*/"speculative_eagle3"));
}

TEST(XGrammarSupportTest, ZeroSpeculativeTokensNeverGates) {
  XGrammarSupportInputs inputs;
  inputs.num_speculative_tokens = 0;
  inputs.speculative_algorithm = "suffix";
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());
}

TEST(XGrammarSupportTest, DisaggPdIsSupportedAfterCrossInstanceVerification) {
  // Regular disagg PD passed the real 1P1D cross-instance verification
  // (grammar spec on the KV first generation, per-token FSM mask on the
  // decode side), so it no longer contributes an unsupported feature.
  XGrammarSupportInputs inputs;
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());
}

TEST(XGrammarSupportTest, ServiceRoutingUsesSerializedConstraintStates) {
  XGrammarSupportInputs inputs;
  inputs.enable_service_routing = true;
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());
}

TEST(XGrammarSupportTest, PromptPrefixCacheDoesNotAdvanceResponseGrammar) {
  XGrammarSupportInputs inputs;
  inputs.enable_prefix_cache = true;
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());
}

TEST(XGrammarSupportTest, ChunkedPrefillStaysSupportedWithSpeculation) {
  // chunked_prefill is deliberately NOT gated: it only splits the prompt, and
  // the matcher initializes at the first decode token, so the chunk boundary
  // cannot affect the decode-time grammar. It is also a hard requirement of
  // the block-diffusion family, so gating it would make the unlocked
  // speculative algorithms unstartable. Pin that decision here: the previous
  // startup CHECK rejected this combination.
  XGrammarSupportInputs inputs;
  inputs.num_speculative_tokens = 4;
  inputs.speculative_algorithm = "MTP";
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());

  // The draft families the channel migrated must stay accepted too, including
  // DSpark whose block-diffusion path requires chunked prefill.
  for (const char* algorithm : {"DFlash", "DFlash2", "DSpark"}) {
    XGrammarSupportInputs draft;
    draft.num_speculative_tokens = 4;
    draft.speculative_algorithm = algorithm;
    SCOPED_TRACE(algorithm);
    EXPECT_TRUE(xgrammar_unsupported_features(draft).empty());
  }
}

TEST(XGrammarSupportTest, RoutingCacheAndDFlash2StaySupportedTogether) {
  XGrammarSupportInputs inputs;
  inputs.enable_service_routing = true;
  inputs.enable_prefix_cache = true;
  inputs.num_speculative_tokens = 7;
  inputs.speculative_algorithm = "DFlash2";
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());
}

class CacheTestTokenizer final : public Tokenizer {
 public:
  size_t vocab_size() const override { return 6; }
  bool get_constraint_vocab(std::vector<std::string>* pieces) const override {
    *pieces = {"A", "B", "-", "1", "x", ""};
    return true;
  }
};

ModelArgs cache_test_model_args() {
  ModelArgs args;
  args.vocab_size(32).eos_token_id(5);
  return args;
}

TEST(XGrammarSupportTest, InvalidSpecRejectedBeforeCompilation) {
  CacheTestTokenizer tokenizer;
  XGrammarMachineCache cache(tokenizer, cache_test_model_args());
  GrammarSpec spec;
  spec.type = static_cast<GrammarSpec::Type>(-1);
  spec.source = "A";
  std::string error;
  EXPECT_EQ(cache.get(spec, error), nullptr);
  EXPECT_FALSE(error.empty());
}

#if defined(USE_XGRAMMAR)
TEST(XGrammarSupportTest, ConcurrentColdMissesKeepOneCompilerAlive) {
  CacheTestTokenizer tokenizer;
  XGrammarMachineCache cache(tokenizer, cache_test_model_args());
  std::vector<std::future<std::shared_ptr<XGrammarConstraintMachine>>> futures;
  futures.reserve(8);
  for (int32_t i = 0; i < 8; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&cache, i] {
      GrammarSpec spec;
      spec.type = GrammarSpec::Type::REGEX;
      spec.source = i % 2 == 0 ? "A" : "B";
      std::string error;
      return cache.get(spec, error);
    }));
  }
  std::vector<std::shared_ptr<XGrammarConstraintMachine>> machines;
  machines.reserve(futures.size());
  for (auto& future : futures) {
    machines.emplace_back(future.get());
    ASSERT_NE(machines.back(), nullptr);
  }
  for (size_t i = 0; i < machines.size(); ++i) {
    EXPECT_EQ(machines[i], machines[i % 2]);
    ConstraintState state = machines[i]->initial_state(false);
    EXPECT_TRUE(machines[i]->can_accept(state, static_cast<int32_t>(i % 2)));
  }
}

TEST(XGrammarSupportTest, FinishedRowsReleaseCachedMachines) {
  CacheTestTokenizer tokenizer;
  XGrammarMachineCache cache(tokenizer, cache_test_model_args());
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::REGEX;
  spec.source = "A";
  std::string error;
  auto machine = cache.get(spec, error);
  ASSERT_NE(machine, nullptr) << error;
  std::weak_ptr<XGrammarConstraintMachine> weak = machine;
  ConstraintState state = machine->initial_state(false);
  machine.reset();
  EXPECT_FALSE(weak.expired());
  state = ConstraintState();
  EXPECT_TRUE(weak.expired());
  EXPECT_NE(cache.get(spec, error), nullptr);
}

TEST(XGrammarSupportTest, EbnfSpecsRebuildThroughTheMachineCache) {
  // The EBNF spec type rides the same cache as schemas and regexes: a PD
  // rebuild or snapshot restore gets a working machine whose compiled
  // grammar admits exactly the strings the EBNF rules allow.
  CacheTestTokenizer tokenizer;
  XGrammarMachineCache cache(tokenizer, cache_test_model_args());
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::EBNF;
  spec.source = "root ::= \"A\" | \"B\" \"-\" \"1\"";
  std::string error;
  auto machine = cache.get(spec, error);
  ASSERT_NE(machine, nullptr) << error;
  EXPECT_EQ(cache.get(spec, error), machine);

  ConstraintState state = machine->initial_state(false);
  EXPECT_TRUE(machine->can_accept(state, /*token_id=*/0));   // "A"
  EXPECT_TRUE(machine->can_accept(state, /*token_id=*/1));   // "B"
  EXPECT_FALSE(machine->can_accept(state, /*token_id=*/3));  // "1"

  // Dead EBNF grammars fail the cache the same way they fail the request
  // factory: compile-time rejection instead of an all-deny decode stall.
  GrammarSpec dead;
  dead.type = GrammarSpec::Type::EBNF;
  dead.source = "root ::= root";
  error.clear();
  EXPECT_EQ(cache.get(dead, error), nullptr);
  EXPECT_FALSE(error.empty());
}

TEST(XGrammarSupportTest, HashCollidingSpecsAreCachedNotRecompiled) {
  CacheTestTokenizer tokenizer;
  // Force every spec onto one key so the two sources collide the way two
  // distinct grammars would under a real 64-bit fnv1a collision; a real
  // collision cannot be constructed from printable regex sources in a
  // test, so the key override stands in for it.
  XGrammarMachineCache collided_cache(tokenizer,
                                      cache_test_model_args(),
                                      [](const GrammarSpec&) { return 42; });
  GrammarSpec first;
  first.type = GrammarSpec::Type::REGEX;
  first.source = "A";
  GrammarSpec second;
  second.type = GrammarSpec::Type::REGEX;
  second.source = "B";
  std::string error;
  auto first_machine = collided_cache.get(first, error);
  ASSERT_NE(first_machine, nullptr) << error;
  auto second_machine = collided_cache.get(second, error);
  ASSERT_NE(second_machine, nullptr) << error;
  // A collision must not serve another request's grammar...
  EXPECT_NE(first_machine, second_machine);
  // ...and must not fall back to a per-request recompile either.
  EXPECT_EQ(collided_cache.get(first, error), first_machine);
  EXPECT_EQ(collided_cache.get(second, error), second_machine);
}

TEST(XGrammarSupportTest, CompilerRejectsOversizedSourcesAndVocabulary) {
  CacheTestTokenizer tokenizer;
  ModelArgs args = cache_test_model_args();
  args.vocab_size(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) +
                  1);
  std::string error;
  EXPECT_EQ(XGrammarCompiler::create(tokenizer, args, /*cache_bytes=*/0, error),
            nullptr);
  args = cache_test_model_args();
  args.stop_token_ids({static_cast<int32_t>(args.vocab_size())});
  EXPECT_EQ(XGrammarCompiler::create(tokenizer, args, /*cache_bytes=*/0, error),
            nullptr);
  EXPECT_FALSE(error.empty());
  args = cache_test_model_args();
  auto compiler =
      XGrammarCompiler::create(tokenizer, args, /*cache_bytes=*/0, error);
  ASSERT_NE(compiler, nullptr) << error;
  const std::string oversized(GrammarSpec::kMaxSourceBytes + 1, 'A');
  EXPECT_EQ(compiler->compile_regex(oversized, error), nullptr);
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(compiler->compile_json_schema(oversized, /*strict=*/true, error),
            nullptr);
  EXPECT_FALSE(error.empty());
}
#else
TEST(XGrammarSupportTest, CompilerFactoryFailsExplicitlyWithoutXgrammar) {
  CacheTestTokenizer tokenizer;
  std::string error;
  EXPECT_EQ(XGrammarCompiler::create(tokenizer,
                                     cache_test_model_args(),
                                     /*cache_bytes=*/0,
                                     error),
            nullptr);
  EXPECT_FALSE(error.empty());
}
#endif

}  // namespace
}  // namespace xllm
