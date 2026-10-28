/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace xllm::kernel::musa {

inline int64_t deepseek_v4_hc_prefill_row_limit(int64_t hidden_dim) {
  const char* enabled = std::getenv("XLLM_DSV4_HC_LARGE_PREFILL");
  // Preserve the model-validated default. The optional 4096-wide route has
  // independent TF32-reference and exact row-partition/Graph coverage.
  return hidden_dim == 4096 && enabled != nullptr &&
                 std::strcmp(enabled, "1") == 0
             ? 1024
             : 320;
}

}  // namespace xllm::kernel::musa
