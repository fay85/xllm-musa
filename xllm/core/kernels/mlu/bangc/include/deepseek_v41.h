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
#include <torch/torch.h>

#include <optional>
namespace xllm::kernel::mlu::bangc {
// Contiguous BF16 qr [T,1280], wq [4096,1280], INT32 positions [T],
// FP32 rope [M,128] (cos | sin), 1 <= M <= 1048576, T >= 1. Positions
// must be in [0,M). Returns independent BF16 [T,32,128]; projection and
// tail-64 FP32 RoPE each round to BF16. Inputs are read-only.
torch::Tensor fused_indexer_q(const torch::Tensor& qr,
                              const torch::Tensor& wq,
                              const torch::Tensor& positions,
                              const torch::Tensor& rope);

// Fixed BF16 projections (5120 -> 32 and 512 -> 128), RMSNorm eps=1e-20,
// and tail-64 interleaved FP32 RoPE. Branch capacities may differ or be zero.
// Valid row counts are optional device INT32 [1] tensors. A missing count uses
// the branch capacity; zero is a valid empty prefix. Head padding is zeroed and
// inactive/negative-slot K rows never access RoPE or change cache. Inputs and
// outputs must not alias; valid slots must be unique with valid positions.
// Cache accepts contiguous [blocks,1,128,128] or [blocks,128,1,128].
torch::Tensor fused_indexer_k(
    const std::optional<torch::Tensor>& x,
    const std::optional<torch::Tensor>& latent,
    const std::optional<torch::Tensor>& weights_proj,
    const std::optional<torch::Tensor>& wk,
    const std::optional<torch::Tensor>& gamma,
    const std::optional<torch::Tensor>& rope,
    const std::optional<torch::Tensor>& positions,
    const std::optional<torch::Tensor>& slots,
    torch::Tensor cache,
    torch::Tensor head,
    const std::optional<torch::Tensor>& num_valid_input_rows = std::nullopt,
    const std::optional<torch::Tensor>& num_valid_compressed_rows =
        std::nullopt);

// One decode row: map pages, gather live sparse KV, and publish the SWA token.
void deepseek_v41_decode_gather(const torch::Tensor& swa,
                                const torch::Tensor& compressed,
                                const torch::Tensor& current,
                                const torch::Tensor& swa_table,
                                const torch::Tensor& compressed_table,
                                const torch::Tensor& topk,
                                int64_t position,
                                int32_t window,
                                torch::Tensor output,
                                torch::Tensor length);

struct CompressKvMetadata {
  torch::Tensor query_start_loc;
  torch::Tensor chunk_begins;
  torch::Tensor compressed_start_loc;
  torch::Tensor compressed_block_table;
  std::optional<torch::Tensor> input_row_map;
};

// Present as a complete group for ratio 2; absent for ratio 1.
struct CompressKvPoolingInputs {
  torch::Tensor scores;
  torch::Tensor state_block_table;
  torch::Tensor state_raw;
  torch::Tensor state_score;
  bool save_full_tail = false;
};

struct CompressKvOutputs {
  torch::Tensor latent;
  torch::Tensor positions;
  torch::Tensor slots;
  // Device INT32 [1] count of the valid compressed output prefix.
  torch::Tensor num_valid_rows;
};

struct CompressKvOptions {
  int64_t ratio = 1;
  double rms_norm_eps = 1e-20;
};

// Writes main_kv_cache, pooling states and preallocated outputs in place on
// the current MLU stream. Tensor storage is never allocated or replaced.
// The caller retains these buffers until the queued work completes.
void fused_compress_kv_cache(
    const torch::Tensor& raw,
    const torch::Tensor& norm_weight,
    const torch::Tensor& cos_sin_cache,
    const CompressKvMetadata& metadata,
    const std::optional<CompressKvPoolingInputs>& pooling,
    torch::Tensor main_kv_cache,
    const CompressKvOutputs& outputs,
    const CompressKvOptions& options);

}  // namespace xllm::kernel::mlu::bangc
