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

// BF16 latent attention with independent candidate lists. Q [T,H,512], cache
// [slots,1,512], indices [T,1,K] int32, prefix lengths [T] int32. K is a
// multiple of 64. Negative/out-of-range slots are ignored. A length bounds a
// prefix (which can contain holes), not the number of valid slots. The caller
// must enforce causal visibility and avoid duplicate valid slots per row.
// Supports MP31 H=8/16/32/64. Explicit scale preserves absorbed MLA semantics.
// Returns [T,H,512]. Precompiled modules are required; no runtime Python/JIT.
torch::Tensor sparse_mla_bf16(const torch::Tensor& query,
                              const torch::Tensor& cache,
                              const torch::Tensor& indices,
                              const torch::Tensor& lengths,
                              double scale);

// GLM inline FP8 NoPE cache: [P,page_size,1,528] bytes, containing 512 E4M3
// values and four FP32 group-128 scales. Q [B,1,H,512] BF16, indices
// [B,1,1,K] int32, lengths [B] int32. Eager MP31 single-query decode only.
torch::Tensor sparse_mla_nope_fp8(const torch::Tensor& query,
                                  const torch::Tensor& cache,
                                  const torch::Tensor& indices,
                                  const torch::Tensor& lengths,
                                  double scale);

// Update token-major BF16 latent cache [pages,page_size,1,512]. Negative slots
// are padding and do not mutate cache. Valid destination slots must be unique.
void sparse_mla_cache_write(const torch::Tensor& latent,
                            const torch::Tensor& slots,
                            torch::Tensor& cache);

void sparse_mla_fp8_cache_write(const torch::Tensor& latent,
                                const torch::Tensor& slots,
                                torch::Tensor& cache);

// Map per-query logical token candidates [T,K] to physical cache slots. Rows
// identify requests in block_table, positions are inclusive causal bounds.
// Invalid requests, pages, future tokens and out-of-cache slots become -1.
torch::Tensor sparse_mla_map_indices(const torch::Tensor& logical,
                                     const torch::Tensor& rows,
                                     const torch::Tensor& positions,
                                     const torch::Tensor& block_table,
                                     int64_t page_size,
                                     int64_t num_pages);

}  // namespace xllm::kernel::musa
