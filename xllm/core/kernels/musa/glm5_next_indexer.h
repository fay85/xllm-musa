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

#pragma once

#include <torch/torch.h>

namespace xllm::kernel::musa {

struct Glm5NextIndexerSelection {
  torch::Tensor
      tokens;  // [T,round_up(token_topk+3,128)], logical IDs, -1 holes
  torch::Tensor lengths;  // [T], candidate prefix end, including holes
  torch::Tensor rows;     // [T], request row in block_table; -1 for padding
};

// Rate-4 GLM IndexPool. BF16 query [T,32,128], raw_key/gate [T,128], FP32
// head weights [T,32], APE [4,128]. Query rows are grouped by cu_seqlens;
// positions within a request must be consecutive, increasing logical IDs.
// state_ids/read_state_ids [B] int32 reference [S,2,4,128] BF16 tail slots;
// zero is the reserved padding slot. Position zero resets a request's tail.
// Pool cache has logical shape [P,64,1,132] uint8. Each page physically stores
// 64*128 E4M3 payload bytes followed by 64 FP32 power-of-two scales.
// Each pool page covers 256 original tokens, using the main block_table.
// Missing historical cache/tail must be restored by the caller before use.
// Inputs and outputs stay on MUSA. Eager MP31 only; scoring is chunked to
// bound logits storage. The same update/causal selection handles prefill,
// chunked prefill, decode and multiple sequences.
Glm5NextIndexerSelection glm5_next_indexer(const torch::Tensor& query,
                                           const torch::Tensor& raw_key,
                                           const torch::Tensor& gate,
                                           const torch::Tensor& weights,
                                           const torch::Tensor& ape,
                                           const torch::Tensor& positions,
                                           const torch::Tensor& cu_seqlens,
                                           const torch::Tensor& state_ids,
                                           const torch::Tensor& read_state_ids,
                                           const torch::Tensor& block_table,
                                           torch::Tensor& cache,
                                           torch::Tensor& tail,
                                           int64_t max_context,
                                           int64_t token_topk);

}  // namespace xllm::kernel::musa
