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

#include <array>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::layer {
struct CompressedKvBuffers {
  torch::Tensor prefix;
  kernel::mlu::bangc::CompressKvOutputs outputs;
};

// One forward's cross-layer selection. KV itself remains framework-owned.
struct Csa2Context {
  // Live sequences occupy a contiguous prefix; token rows are packed by the
  // query prefix, including zero-length sequences. Capacity is independent.
  int32_t num_sequences = 0;
  std::vector<int32_t> host_query_prefix;
  std::vector<int32_t> host_chunk_begins;
  torch::Tensor host_positions;
  // Each defined entry has shape {query_rows, k}: one top-k list per query.
  // A selection with no keys keeps the two-dimensional shape {query_rows, 0}.
  std::vector<torch::Tensor> topk;
  std::vector<torch::Tensor> candidates;
  // Device INT32 tables shared by eager and captured attention kernels.
  std::vector<torch::Tensor> block_tables;
  // The logical SWA ring view retains the input width even when block_tables
  // stores an expanded absolute-page mapping for captured compression.
  torch::Tensor swa_block_table;
  // Shared decode/verify scratch. Graph owners retain it across replays;
  // ordinary forwards own it until all layers have consumed it.
  torch::Tensor selected_kv;
  torch::Tensor selected_lengths;
  // Stable device storage for the prefixes, chunk starts and input row count.
  // Each preparation uploads these views together from fresh pinned staging.
  torch::Tensor compression_metadata;
  torch::Tensor query_prefix;
  torch::Tensor chunk_begins;
  // Valid local projection rows; CP's query_prefix describes global KV rows.
  torch::Tensor num_valid_input_rows;
  std::array<CompressedKvBuffers, 2> compressed;
};

namespace deepseek_v41 {
bool uses_decode_attention(const ModelInputParams& input);

void prepare_workspace(Csa2Context& shared,
                       const ModelInputParams& input,
                       const ModelArgs& args,
                       int64_t query_capacity,
                       const torch::TensorOptions& options);

// Prepare once before layers or capture/replay, using global KV positions for
// CP. num_sequences is explicit (zero is valid); padded sequences have no rows.
// A positive KV capacity reserves stable absolute-page tables for graph owners.
// Prefer input.attention.host.positions when supplied; device-only callers
// explicitly use the synchronous positions readback path.
// CP callers update num_valid_input_rows to their local query count after this
// global setup.
void prepare_metadata(Csa2Context& shared,
                      const ModelInputParams& input,
                      const torch::Tensor& positions,
                      int32_t num_sequences,
                      int64_t kv_seq_len_capacity = 0);
}  // namespace deepseek_v41
}  // namespace xllm::layer
