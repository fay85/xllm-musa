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

#include <cstdint>

#include "cnrt.h"

namespace xllm::kernel::mlu::bangc::indexer_k {

constexpr int32_t kHidden = 5120;
constexpr int32_t kHeads = 32;
constexpr int32_t kLatent = 512;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kRopeDim = 64;
constexpr int32_t kBlockSize = 128;

// All pointers are device addresses. A branch with zero capacity carries null
// pointers and is skipped entirely. Valid row count pointers may be null,
// meaning every row within the corresponding branch capacity is valid.
struct Params {
  const void* x;
  const void* wp;
  float* head;
  const int32_t* num_valid_input_rows;
  int64_t x_stride;
  int32_t cap_x;

  const void* latent;
  const void* wk;
  const void* gamma;
  const float* rope;
  const int32_t* positions;
  const int32_t* slots;
  void* cache;
  const int32_t* num_valid_compressed_rows;
  int64_t latent_stride;
  int64_t rope_stride;
  int64_t rope_rows;
  int64_t cache_rows;
  int32_t cap_k;
  int32_t gamma_fp32;

  // Rows per head / K task, chosen by launch() from the capacities.
  int32_t head_rows;
  int32_t key_rows;
  // Small head loads: each cluster owns a block of head_rows rows and its 4
  // cores split K (head_blocks clusters, 4 tasks each).
  int32_t head_splitk;
  int32_t head_blocks;
  // Head loads above the split-K range: feature-major tiles (x as the WRAM
  // conv filter, wp quarters in NRAM).
  int32_t head_fm;
  // Prefill-size dual-branch loads: the memory cores stream the head x into
  // SRAM under the K phase; up to `mfuse` head chunk convs run per K tile
  // (0 = off).
  int32_t mfuse;
  // Feature-major K phase reads wk from the cluster's SRAM copy, staged by
  // the memory core during the head phase (second cluster barrier).
  int32_t wk_sram;
  // Small K loads: each cluster owns a block of key_rows rows and its 4
  // cores split the latent dim (key_blocks clusters, 4 tasks each).
  int32_t key_splitk;
  int32_t key_blocks;
  // RoPE table and cache byte spans both below 4 GiB: K metadata is
  // vectorized with uint32 offsets (else a scalar path with 64-bit offsets).
  int32_t meta_u32;
};

// Returns cnrtSuccess or the launch error.
cnrtRet_t launch(cnrtQueue_t queue, const Params& params);

}  // namespace xllm::kernel::mlu::bangc::indexer_k
