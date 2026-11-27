/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

struct SpeculativeOutputStats {
  std::vector<int64_t> accepted_per_position;
  std::vector<SpeculativeTokenStats> sequence_stats;
  int64_t committed_tokens = 0;
  int64_t draft_tokens = 0;
  int64_t accepted_tokens = 0;
  int64_t constrained_draft_tokens = 0;
  int64_t constrained_accepted_tokens = 0;
};

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    std::span<const char> constrained_mask = {});

void record_speculative_output_stats(
    const SpeculativeOutputStats& stats,
    std::span<const std::string> position_labels);

}  // namespace xllm
