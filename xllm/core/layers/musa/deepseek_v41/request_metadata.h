/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace xllm::layer {

// Scheduler host metadata can carry per-request [length] or the MUSA
// cumulative [0, length] form. Accept precisely one sequence in either ABI.
inline std::optional<int64_t> parse_deepseek_v41_c1_length(
    const std::vector<int32_t>& values) {
  if (values.size() == 1 && values.front() >= 0) {
    return values.front();
  }
  if (values.size() == 2 && values.front() == 0 && values.back() >= 0) {
    return values.back();
  }
  return std::nullopt;
}

// Framework KV/INDEX/SWA planes retain a singleton KV-head axis for
// transport. The V4.1 layers consume a view of the same physical storage.
inline torch::Tensor deepseek_v41_cache_view(const torch::Tensor& cache) {
  if (!cache.defined() || cache.numel() == 0) {
    return {};
  }
  if (cache.dim() == 4) {
    CHECK_EQ(cache.size(2), 1);
    return cache.squeeze(2);
  }
  CHECK_EQ(cache.dim(), 3);
  return cache;
}

}  // namespace xllm::layer
