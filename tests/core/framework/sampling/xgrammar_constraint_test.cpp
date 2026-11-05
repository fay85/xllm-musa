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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xllm {
namespace {

constexpr int32_t kStopTokenId = 5;

std::unique_ptr<XGrammarCompiler> make_compiler() {
  return std::make_unique<XGrammarCompiler>(
      std::vector<std::string>{"A", "B", "-", "1", "x", "", "{", "}"},
      /*model_vocab_size=*/32,
      std::vector<int32_t>{kStopTokenId},
      /*cache_max_memory_bytes=*/0);
}

TEST(XGrammarConstraintTest, RegexConstrainsTokensUntilStop) {
  auto compiler = make_compiler();
  std::string error;
  auto grammar = compiler->compile_regex("[A-Z]{2}-[0-9]{4}", error);
  ASSERT_NE(grammar, nullptr) << error;
  XGrammarState state(*grammar);
  std::vector<int32_t> mask(1);
  state.fill_next_token_bitmask(mask);
  EXPECT_EQ(mask[0], (1 << 0) | (1 << 1));
  EXPECT_FALSE(state.accept_token(/*token_id=*/4));
  for (const int32_t token_id : {0, 1, 2, 3, 3, 3, 3}) {
    state.fill_next_token_bitmask(mask);
    ASSERT_NE(mask[0] & (1 << token_id), 0);
    ASSERT_TRUE(state.accept_token(token_id));
  }
  state.fill_next_token_bitmask(mask);
  EXPECT_NE(mask[0] & (1 << kStopTokenId), 0);
  EXPECT_EQ(mask[0] & (1 << 3), 0);
  EXPECT_TRUE(state.accept_token(kStopTokenId));
}

TEST(XGrammarConstraintTest, JsonSchemaConstrainsAnObject) {
  auto compiler = make_compiler();
  std::string error;
  auto grammar = compiler->compile_json_schema(
      R"({"type":"object"})", /*strict_mode=*/false, error);
  ASSERT_NE(grammar, nullptr) << error;
  XGrammarState state(*grammar);
  EXPECT_FALSE(state.accept_token(/*token_id=*/0));
  EXPECT_TRUE(state.accept_token(/*token_id=*/6));
  EXPECT_TRUE(state.accept_token(/*token_id=*/7));
  EXPECT_TRUE(state.accept_token(kStopTokenId));
}

TEST(XGrammarConstraintTest, InvalidGrammarReturnsAnError) {
  auto compiler = make_compiler();
  std::string error;
  EXPECT_EQ(compiler->compile_regex("[", error), nullptr);
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(compiler->compile_json_schema("{", /*strict_mode=*/true, error),
            nullptr);
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace xllm
