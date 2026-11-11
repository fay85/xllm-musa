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

#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace xgrammar {
class CompiledGrammar;
class GrammarCompiler;
class GrammarMatcher;
}  // namespace xgrammar

namespace xllm {

// Upstream grammar handles share immutable compilation data.
using CompiledXGrammar = xgrammar::CompiledGrammar;

// Model-scoped CPU compiler. Owns synchronization at the third-party boundary.
class XGrammarCompiler final {
 public:
  XGrammarCompiler(const std::vector<std::string>& token_pieces,
                   int64_t model_vocab_size,
                   const std::vector<int32_t>& stop_token_ids,
                   int64_t cache_max_memory_bytes);
  ~XGrammarCompiler();

  std::shared_ptr<const CompiledXGrammar> compile_json_schema(
      const std::string& schema,
      bool strict_mode,
      std::string& error);
  std::shared_ptr<const CompiledXGrammar> compile_regex(
      const std::string& regex,
      std::string& error);

 private:
  std::mutex mutex_;
  std::unique_ptr<xgrammar::GrammarCompiler> compiler_;
};

// Sequence-owned matcher; only generated tokens advance its state.
class XGrammarState final {
 public:
  explicit XGrammarState(const CompiledXGrammar& grammar,
                         uint64_t grammar_id = 0);
  ~XGrammarState();
  XGrammarState(XGrammarState&&) = default;
  XGrammarState& operator=(XGrammarState&&) = default;

  // Deep copy for draft/beam/parallel candidates; the compiled grammar and
  // tokenizer info stay shared with the source matcher.
  XGrammarState fork() const;

  bool accept_token(int32_t token_id);
  // Read-only probe: reports whether accept_token would accept, without
  // advancing the matcher.
  bool can_accept(int32_t token_id);
  // The root rule is fully matched by the accepted input (stop token not
  // required).
  bool is_completed() const;
  // A frozen matcher rejects everything; used for halt semantics.
  bool is_valid() const { return !invalid_; }
  void set_invalid() { invalid_ = true; }
  // FSM-only hash of (grammar identity, allowed bitmask, invalid salt):
  // identical FSM states share the fingerprint regardless of how many
  // tokens were committed to reach them; the history must not leak in
  // (mask-cache keys and beam-restore comparisons rely on that).
  uint64_t fingerprint() const;
  // Accepted-token prefix, in commit order; drives snapshot and replay.
  const std::vector<int32_t>& accepted_tokens() const {
    return accepted_tokens_;
  }
  int64_t vocab_size() const { return vocab_size_; }
  // Fully writes a caller-owned CPU row of (vocab_size() + 31) / 32 words.
  // Returns whether the mask needs applying (false means all-true).
  bool fill_next_token_bitmask(std::span<int32_t> words);
  // The same mask as a packed uint32 vector: all-deny when frozen,
  // all-allow when the step is unconstrained. Single source for the
  // machine's allowed_bitmask and the FSM-only fingerprint.
  std::vector<uint32_t> allowed_bitmask() const;

 private:
  XGrammarState(std::unique_ptr<xgrammar::GrammarMatcher> matcher,
                int64_t vocab_size);

  std::unique_ptr<xgrammar::GrammarMatcher> matcher_;
  int64_t vocab_size_ = 0;
  uint64_t grammar_id_ = 0;
  bool invalid_ = false;
  std::vector<int32_t> accepted_tokens_;
};

// Rows follow selected-token order, including unconstrained (nullptr) rows.
torch::Tensor build_xgrammar_filter_bitmask(
    std::span<XGrammarState* const> states);

}  // namespace xllm
