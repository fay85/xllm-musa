/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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
#include <vector>

namespace xllm {

// Backend-neutral transport form of a per-sequence constraint state.
struct ConstraintSnapshot final {
  bool enabled = false;
  bool reasoning_enabled = false;
  std::vector<int32_t> token_ids;
  // Frozen (halted) states must round-trip their invalid flag; builtin
  // freezes on RESPECT rejections and piece-level violations, xgrammar on
  // RESPECT rejections.
  bool invalid = false;
};

}  // namespace xllm
