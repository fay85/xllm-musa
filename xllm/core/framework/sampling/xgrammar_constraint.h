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
  explicit XGrammarState(const CompiledXGrammar& grammar);
  ~XGrammarState();

  bool accept_token(int32_t token_id);
  int64_t vocab_size() const { return vocab_size_; }
  // Fully writes a caller-owned CPU row of (vocab_size() + 31) / 32 words.
  void fill_next_token_bitmask(std::span<int32_t> words);

 private:
  std::unique_ptr<xgrammar::GrammarMatcher> matcher_;
  int64_t vocab_size_ = 0;
};

// Rows follow selected-token order, including unconstrained (nullptr) rows.
torch::Tensor build_xgrammar_filter_bitmask(
    std::span<XGrammarState* const> states);

}  // namespace xllm
