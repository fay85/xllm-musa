/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
==============================================================================*/
#pragma once

#include <glog/logging.h>

#include <cstdlib>
#include <string>
#include <string_view>

namespace xllm {

// KV layout equality alone is insufficient when projections round their
// activations differently. Carry semantic selections into both namespace
// and transfer compatibility digest before either endpoint serves requests.
inline std::string deepseek_v41_execution_contract(
    [[maybe_unused]] bool is_v41) {
#if defined(USE_MUSA)
  if (!is_v41) {
    return {};
  }
  const char* dense_value = std::getenv("XLLM_DSV41_MUSA_DENSE_QUANT");
  const char* expert_value = std::getenv("XLLM_DSV41_MUSA_FP4_LINEAR");
  const char* engram_value = std::getenv("XLLM_DSV41_MUSA_ENGRAM_GATE");
  const std::string_view dense =
      dense_value == nullptr ? "reference" : dense_value;
  const std::string_view expert =
      expert_value == nullptr ? "reference" : expert_value;
  const std::string_view engram =
      engram_value == nullptr ? "reference" : engram_value;
  CHECK(dense == "reference" || dense == "original_qdq");
  CHECK(expert == "reference" || expert == "packed_bf16" ||
        expert == "block32" || expert == "partition32");
  CHECK(engram == "reference" || engram == "original" ||
        engram == "sglang_fused");
  std::string contract;
  if (dense == "original_qdq") {
    contract += "v41-dense-activation-block32-qdq-v1;";
  }
  if (expert == "block32" || expert == "partition32") {
    contract += "v41-expert-activation-block32-qdq-v1;";
  }
  if (engram == "original" || engram == "sglang_fused") {
    contract += "v41-engram-original-product-order-v1;";
  }
  return contract;
#else
  return {};
#endif
}
}  // namespace xllm
