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

XGrammarState::XGrammarState(const CompiledXGrammar& grammar)
    : matcher_(std::make_unique<xgrammar::GrammarMatcher>(grammar)),
      vocab_size_(grammar.GetTokenizerInfo().GetVocabSize()) {}

bool XGrammarState::accept_token(int32_t token_id) {
  return matcher_->AcceptToken(token_id);
}

void XGrammarState::fill_next_token_bitmask(std::span<int32_t> words) {
  int64_t num_words = static_cast<int64_t>(words.size());
  DLTensor bitmask{};
  bitmask.data = words.data();
  bitmask.device = {kDLCPU, 0};
  bitmask.ndim = 1;
  bitmask.dtype = {kDLInt, 32, 1};
  bitmask.shape = &num_words;
  matcher_->FillNextTokenBitmask(&bitmask);
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
XGrammarState::XGrammarState(const CompiledXGrammar&) {
  LOG(FATAL) << "Cannot construct a matcher without XGrammar support";
}
bool XGrammarState::accept_token(int32_t) { return false; }
void XGrammarState::fill_next_token_bitmask(std::span<int32_t>) {
  LOG(FATAL) << "Cannot construct a bitmask without XGrammar support";
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
    states[row]->fill_next_token_bitmask(
        {row_data, static_cast<size_t>(num_words)});
  }
  return bitmask;
}

}  // namespace xllm
