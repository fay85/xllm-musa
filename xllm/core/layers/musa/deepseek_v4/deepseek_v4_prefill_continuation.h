// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace xllm::layer {

struct Dsv4PrefillContinuation {
  torch::Tensor kv;
  torch::Tensor score;
  int64_t prepended_rows = 0;
  int64_t output_block_offset = 0;
};

// Reconstruct only the unfinished compression block (and the preceding C4
// overlap block). State scores already include APE; projection scores do not.
// Both the main compressor and Indexer use this same continuation contract.
inline Dsv4PrefillContinuation dsv4_prefill_continuation(
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& ape,
    const torch::Tensor& state_kv,
    const torch::Tensor& state_score,
    int64_t cached_tokens,
    int64_t ratio) {
  TORCH_CHECK(ratio == 4 || ratio == 128);
  TORCH_CHECK(cached_tokens >= 0);
  if (cached_tokens == 0) return {kv, score, 0, 0};
  const int64_t overlap = ratio == 4 ? ratio : 0;
  const int64_t partial = cached_tokens % ratio;
  TORCH_CHECK(state_kv.defined() && state_score.defined(),
              "Cached prefill requires restored compressor state");
  TORCH_CHECK(state_kv.dim() == 2 && state_score.sizes() == state_kv.sizes());
  TORCH_CHECK(state_kv.size(0) == (ratio == 4 ? 2 : 1) * ratio);
  TORCH_CHECK(state_kv.size(1) == kv.size(1));
  TORCH_CHECK(ape.dim() == 2 && ape.size(0) == ratio &&
              ape.size(1) == score.size(1));
  if (overlap + partial == 0) return {kv, score, 0, 0};
  std::vector<torch::Tensor> kv_parts, score_parts;
  if (overlap) {
    kv_parts.push_back(state_kv.narrow(0, 0, overlap));
    score_parts.push_back(state_score.narrow(0, 0, overlap) - ape);
  }
  if (partial) {
    kv_parts.push_back(state_kv.narrow(0, overlap, partial));
    score_parts.push_back(state_score.narrow(0, overlap, partial) -
                          ape.narrow(0, 0, partial));
  }
  kv_parts.push_back(kv);
  score_parts.push_back(score);
  return {torch::cat(kv_parts, 0), torch::cat(score_parts, 0),
          overlap + partial, overlap ? 1 : 0};
}

inline int64_t dsv4_prefill_compressed_rows(int64_t cached_tokens,
                                           int64_t query_tokens,
                                           int64_t ratio) {
  TORCH_CHECK(cached_tokens >= 0 && query_tokens >= 0 && ratio > 0);
  return (cached_tokens + query_tokens) / ratio - cached_tokens / ratio;
}

inline void dsv4_clear_prefill_state_rows(torch::Tensor state,
                                         const torch::Tensor& active_blocks) {
  TORCH_CHECK(state.dim() == 3 && active_blocks.dim() == 1 &&
              active_blocks.scalar_type() == torch::kInt64);
  // Called only from eager prefill. Do not clear unrelated decode requests
  // or prefix snapshots sharing the same persistent state pool.
  state.index_fill_(0, active_blocks, 0);
}

}  // namespace xllm::layer
