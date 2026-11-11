/* Copyright 2026 The xLLM Authors.

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
#include <cnrt.h>

#include <cstdint>
void deepseek_v41_gather_launch(cnrtQueue_t queue,
                                const void* swa,
                                const void* compressed,
                                const void* current,
                                const int32_t* swa_table,
                                const int32_t* comp_table,
                                const int64_t* topk,
                                int32_t count,
                                int64_t position,
                                int32_t window,
                                int32_t swa_columns,
                                void* output,
                                int32_t* length);

namespace xllm::kernel::mlu::bangc::indexer_q {
// Shared by the kernel tiling and wrapper row-count overflow guard.
constexpr int32_t kRowsPerBlock = 48;

int32_t launch(cnrtQueue_t queue,
               const void* qr,
               const void* wq,
               const int32_t* positions,
               const float* rope,
               void* out,
               int32_t tokens);
}  // namespace xllm::kernel::mlu::bangc::indexer_q
