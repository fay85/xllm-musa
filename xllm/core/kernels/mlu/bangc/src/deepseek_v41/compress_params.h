/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

#include <cstdint>

#include "cnrt.h"

namespace xllm::kernel::mlu::bangc {

// Device pointers and static launch configuration of one call. Element
// strides are in elements of the pointed dtype. Every dynamic quantity (row
// counts, prefixes, pages, row map) stays in device memory.
struct FusedCompressKvCacheParams {
  const void* raw;
  const float* scores;
  const void* norm_weight;
  const float* cos_sin_cache;
  const int32_t* query_start_loc;
  const int32_t* chunk_begins;
  const int32_t* compressed_start_loc;
  const int32_t* compressed_block_table;
  void* main_kv_cache;
  void* latent_out;
  int32_t* positions_out;
  int32_t* slots_out;
  int32_t* num_valid_compressed_rows_out;
  const int32_t* state_block_table;
  float* state_raw;
  float* state_score;
  const int32_t* input_row_map;
  int64_t raw_row_stride;
  int64_t score_row_stride;
  int64_t state_raw_page_stride;
  int64_t state_raw_row_stride;
  int64_t state_score_page_stride;
  int64_t state_score_row_stride;
  int64_t compressed_table_stride;
  int64_t state_table_stride;
  int64_t row_map_size;
  int32_t batch;
  int32_t k_capacity;
  int32_t compressed_columns;
  int32_t state_columns;
  int32_t ratio;
  int32_t save_full_tail;
  int32_t norm_weight_bf16;
  // scores == raw + 512 with both row strides 1024 (one [raw | score] owner).
  int32_t packed_projection;
  // state_score == state_raw + 512 with equal page/row strides.
  int32_t packed_state;
  float rms_norm_eps;
};

// Enqueue the fused kernel on `queue`; returns the cnrt launch status.
cnrtRet_t launch_fused_compress_kv_cache(
    cnrtQueue_t queue,
    const FusedCompressKvCacheParams& params);

}  // namespace xllm::kernel::mlu::bangc
