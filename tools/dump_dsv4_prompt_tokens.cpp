/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
https://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core/framework/chat_template/deepseek_v4_cpp_template.h"
#include "core/framework/tokenizer/fast_tokenizer.h"

// CPU-only probe using the same compiled template/tokenizer objects as xLLM.
// Dataset answers are never read. JSONL output is intended for exact comparison.
int main(int argc, char** argv) {
  CHECK_EQ(argc, 3) << "usage: dump_dsv4_prompt_tokens tokenizer.json dataset.jsonl";
  xllm::TokenizerArgs args;
  args.vocab_file(argv[1]);
  args.bos_token("<｜begin▁of▁sentence｜>");
  args.eos_token("<｜end▁of▁sentence｜>");
  args.add_bos_token(false);
  args.add_eos_token(false);
  xllm::FastTokenizer tokenizer(args);
  xllm::DeepseekV4CppTemplate encoder(args);
  std::ifstream input(argv[2]);
  CHECK(input.good());
  std::string line;
  int32_t index = 0;
  while (std::getline(input, line)) {
    const auto row = nlohmann::ordered_json::parse(line);
    const std::string problem = row.at("problem").get<std::string>();
    const std::string content =
        "Solve the following math problem. Make sure to put the answer "
        "(and only answer) inside \\boxed{}.\n\n" + problem;
    xllm::ChatMessages messages;
    messages.emplace_back("user", content);
    const nlohmann::ordered_json kwargs = {{"thinking", false}};
    const auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
    CHECK(prompt.has_value());
    std::vector<int32_t> ids;
    CHECK(tokenizer.encode(*prompt, &ids, /*add_special_tokens=*/true));
    const nlohmann::ordered_json output = {
        {"id", "gsm8k-" + std::to_string(index)},
        {"rendered_prompt", *prompt}, {"token_ids", ids}};
    std::cout << output.dump() << '\n';
    ++index;
  }
}
