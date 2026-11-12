/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
namespace xllm::layer {
// MUSA supplies cumulative host lengths. Accept only one actual token per
// request; q>1 prefill and speculative verification are different contracts.
inline std::optional<std::vector<int64_t>> parse_deepseek_v41_decode_batch(
    const std::vector<int32_t>& query,
    const std::vector<int32_t>& context,
    int64_t requests,
    int64_t tokens) {
  if (requests < 2 || requests > 8 || tokens != requests ||
      query.size() != static_cast<size_t>(requests + 1) ||
      context.size() != static_cast<size_t>(requests + 1) ||
      query.front() != 0 || context.front() != 0) {
    return std::nullopt;
  }
  std::vector<int64_t> lengths;
  lengths.reserve(requests);
  for (int64_t row = 0; row < requests; ++row) {
    if (query[row] != row || query[row + 1] != row + 1) {
      return std::nullopt;
    }
    const int64_t length =
        static_cast<int64_t>(context[row + 1]) - context[row];
    if (length <= 0) {
      return std::nullopt;
    }
    lengths.emplace_back(length);
  }
  return lengths;
}
}  // namespace xllm::layer
