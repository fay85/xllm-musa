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

#include "core/framework/sampling/xgrammar_constraint.h"

#include <glog/logging.h>

#include <algorithm>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "core/framework/sampling/fnv1a.h"

#if defined(USE_XGRAMMAR)
#include <xgrammar/xgrammar.h>
#else
namespace xgrammar {
class GrammarCompiler final {};
class GrammarMatcher final {};
}  // namespace xgrammar
#endif

namespace xllm {

#if defined(USE_XGRAMMAR)
namespace {
// Multiplier for the state fingerprint's token-history mixing (a prime, so
// the hash spreads well). Named per the project's constant rule.
constexpr uint64_t kFingerprintMultiplier = 1000003ULL;
// Salt mixed into the fingerprint of an invalid (halted) state, so an invalid
// state never collides with a valid one. Named per the project's constant rule.
constexpr uint64_t kInvalidStateSalt = 0x9e3779b97f4a7c15ULL;

template <typename Compile>
std::shared_ptr<const CompiledXGrammar> compile_grammar(Compile compile,
                                                        std::mutex& mutex,
                                                        std::string& error) {
  std::lock_guard<std::mutex> lock(mutex);
  error.clear();
  try {
    return std::make_shared<CompiledXGrammar>(compile());
  } catch (const std::runtime_error& exception) {
    error = std::string("Failed to initialize XGrammar constraint: ") +
            exception.what();
    return nullptr;
  }
}
}  // namespace

XGrammarCompiler::XGrammarCompiler(const std::vector<std::string>& token_pieces,
                                   int64_t model_vocab_size,
                                   const std::vector<int32_t>& stop_token_ids,
                                   int64_t cache_max_memory_bytes)
    : compiler_(std::make_unique<xgrammar::GrammarCompiler>(
          xgrammar::TokenizerInfo(token_pieces,
                                  xgrammar::VocabType::RAW,
                                  static_cast<int32_t>(model_vocab_size),
                                  stop_token_ids,
                                  /*add_prefix_space=*/false),
          /*max_threads=*/1,
          /*cache_enabled=*/cache_max_memory_bytes != 0,
          cache_max_memory_bytes)) {}

std::shared_ptr<const CompiledXGrammar> XGrammarCompiler::compile_json_schema(
    const std::string& schema,
    bool strict_mode,
    std::string& error) {
  return compile_grammar(
      [&] {
        return compiler_->CompileJSONSchema(schema,
                                            /*any_whitespace=*/true,
                                            std::nullopt,
                                            std::nullopt,
                                            strict_mode);
      },
      mutex_,
      error);
}

std::shared_ptr<const CompiledXGrammar> XGrammarCompiler::compile_regex(
    const std::string& regex,
    std::string& error) {
  return compile_grammar(
      [&] { return compiler_->CompileRegex(regex); }, mutex_, error);
}

XGrammarState::XGrammarState(const CompiledXGrammar& grammar,
                             uint64_t grammar_id)
    : matcher_(std::make_unique<xgrammar::GrammarMatcher>(grammar)),
      vocab_size_(grammar.GetTokenizerInfo().GetVocabSize()),
      // Content hash, not the object address: a grammar freed and recompiled
      // at the same address must not collide, and a hash of the serialized
      // grammar stays stable across the grammar's lifetime (and across
      // processes once the fingerprint is persisted for PD spec transfer).
      // The caller (machine/compiler) passes a cached id so the per-state
      // construction on the draft hot path does not re-serialize the grammar.
      grammar_id_(grammar_id != 0 ? grammar_id
                                  : fnv1a(grammar.SerializeJSON())) {}

XGrammarState::XGrammarState(std::unique_ptr<xgrammar::GrammarMatcher> matcher,
                             int64_t vocab_size)
    : matcher_(std::move(matcher)), vocab_size_(vocab_size) {}

XGrammarState XGrammarState::fork() const {
  XGrammarState forked(
      std::make_unique<xgrammar::GrammarMatcher>(matcher_->Fork()),
      vocab_size_);
  forked.grammar_id_ = grammar_id_;
  forked.invalid_ = invalid_;
  forked.accepted_tokens_ = accepted_tokens_;
  return forked;
}

bool XGrammarState::accept_token(int32_t token_id) {
  const bool accepted = matcher_->AcceptToken(token_id);
  if (accepted) {
    accepted_tokens_.push_back(token_id);
  }
  return accepted;
}

bool XGrammarState::can_accept(int32_t token_id) {
  if (token_id < 0 || token_id >= vocab_size_) {
    return false;
  }
  const size_t num_words = static_cast<size_t>((vocab_size_ + 31) / 32);
  // Thread-local probe buffer: can_accept is logically read-only and runs per
  // token, so the scratch is reused within a thread without allocation, while
  // thread isolation keeps it from racing a concurrent allowed_bitmask.
  // resize (not assign): the fill fully writes the row when it returns true,
  // and is not read when false, so a same-size probe does no memset.
  thread_local std::vector<int32_t> probe_bitmask;
  probe_bitmask.resize(num_words);
  if (!fill_next_token_bitmask(probe_bitmask)) {
    return true;  // Unconstrained step: every token is allowed.
  }
  return (probe_bitmask[token_id >> 5] >> (token_id & 31)) & 1;
}

bool XGrammarState::is_completed() const { return matcher_->IsCompleted(); }

std::vector<uint32_t> XGrammarState::allowed_bitmask() const {
  const size_t num_words = static_cast<size_t>((vocab_size_ + 31) / 32);
  // A frozen matcher rejects everything: the all-deny mask matches
  // can_accept instead of the live fill.
  if (!is_valid()) {
    return std::vector<uint32_t>(num_words, 0);
  }
  // Logically read-only: the fill may refresh matcher caches, like the
  // builtin grammar's cached-mask path.
  std::vector<int32_t> words(num_words, 0);
  // A false return means an unconstrained step (all tokens allowed); the
  // zero-initialized buffer would otherwise read as all-deny.
  if (!const_cast<XGrammarState*>(this)->fill_next_token_bitmask(words)) {
    return std::vector<uint32_t>(num_words, ~uint32_t{0});
  }
  return std::vector<uint32_t>(words.begin(), words.end());
}

uint64_t XGrammarState::fingerprint() const {
  // FSM-only fingerprint: hash the current allowed mask (a pure function of the
  // matcher's FSM state), NOT the accepted-token history. Identical FSM states
  // share the same allowed mask regardless of how many tokens were committed,
  // so the fingerprint must not depend on the history.
  uint64_t hash = grammar_id_;
  const std::vector<uint32_t> mask = allowed_bitmask();
  for (const uint32_t word : mask) {
    hash = (hash ^ static_cast<uint64_t>(word)) * kFingerprintMultiplier;
  }
  // A frozen matcher is a distinct state: it rejects everything.
  hash =
      (hash ^ (invalid_ ? kInvalidStateSalt : 0ULL)) * kFingerprintMultiplier;
  return hash;
}

bool XGrammarState::fill_next_token_bitmask(std::span<int32_t> words) {
  int64_t num_words = static_cast<int64_t>(words.size());
  DLTensor bitmask{};
  bitmask.data = words.data();
  bitmask.device = {kDLCPU, 0};
  bitmask.ndim = 1;
  bitmask.dtype = {kDLInt, 32, 1};
  bitmask.shape = &num_words;
  return matcher_->FillNextTokenBitmask(&bitmask);
}
#else
XGrammarCompiler::XGrammarCompiler(const std::vector<std::string>&,
                                   int64_t,
                                   const std::vector<int32_t>&,
                                   int64_t) {
  LOG(FATAL) << "Build with USE_XGRAMMAR=ON to create a grammar compiler";
}
std::shared_ptr<const CompiledXGrammar> XGrammarCompiler::compile_json_schema(
    const std::string&,
    bool,
    std::string& error) {
  error = "XGrammar support was not built; configure with USE_XGRAMMAR=ON";
  return nullptr;
}
std::shared_ptr<const CompiledXGrammar> XGrammarCompiler::compile_regex(
    const std::string&,
    std::string& error) {
  return compile_json_schema({}, false, error);
}
XGrammarState::XGrammarState(const CompiledXGrammar&, uint64_t) {
  LOG(FATAL) << "Cannot construct a matcher without XGrammar support";
}
XGrammarState XGrammarState::fork() const {
  LOG(FATAL) << "Cannot fork a matcher without XGrammar support";
}
uint64_t XGrammarState::fingerprint() const { return 0; }
bool XGrammarState::accept_token(int32_t) { return false; }
bool XGrammarState::can_accept(int32_t) { return false; }
bool XGrammarState::is_completed() const { return false; }
bool XGrammarState::fill_next_token_bitmask(std::span<int32_t>) {
  LOG(FATAL) << "Cannot construct a bitmask without XGrammar support";
}
std::vector<uint32_t> XGrammarState::allowed_bitmask() const {
  LOG(FATAL) << "Cannot produce a bitmask without XGrammar support";
}
#endif

XGrammarCompiler::~XGrammarCompiler() = default;
XGrammarState::~XGrammarState() = default;

torch::Tensor build_xgrammar_filter_bitmask(
    std::span<XGrammarState* const> states) {
  const auto first = std::find_if(
      states.begin(), states.end(), [](const XGrammarState* state) {
        return state != nullptr;
      });
  if (first == states.end()) {
    return {};
  }
  const int64_t vocab_size = (*first)->vocab_size();
  const int64_t num_words = (vocab_size + 31) / 32;
  torch::Tensor bitmask = torch::empty(
      {static_cast<int64_t>(states.size()), num_words},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  int32_t* data = bitmask.data_ptr<int32_t>();
  for (size_t row = 0; row < states.size(); ++row) {
    int32_t* row_data = data + row * static_cast<size_t>(num_words);
    if (states[row] == nullptr) {
      std::fill_n(row_data, num_words, -1);
      continue;
    }
    CHECK_EQ(states[row]->vocab_size(), vocab_size)
        << "mixed XGrammar vocabularies in one batch";
    if (!states[row]->fill_next_token_bitmask(
            {row_data, static_cast<size_t>(num_words)})) {
      // A false fill means an unconstrained step (every token allowed) and
      // leaves the buffer untouched; the torch::empty row would otherwise
      // read as uninitialized garbage. Fill all-allow.
      std::fill_n(row_data, num_words, -1);
    }
  }
  return bitmask;
}

}  // namespace xllm
