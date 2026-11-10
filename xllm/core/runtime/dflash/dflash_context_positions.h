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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm::dflash_detail {

// The target produces [3, T] positions for MRoPE models, while the draft's
// context KV scatter uses one plain-RoPE position per token. Text targets keep
// their original [T] positions and do not need a rebuild.
inline bool is_mrope_context_positions(const torch::Tensor& positions_host,
                                       int64_t num_context_tokens) {
  CHECK(positions_host.defined())
      << "DFlash prefill requires processed positions_host.";
  CHECK_EQ(positions_host.scalar_type(), torch::kInt)
      << "DFlash context positions must use int32.";
  CHECK_GE(num_context_tokens, 0);
  CHECK(positions_host.dim() == 1 || positions_host.dim() == 2)
      << "DFlash context positions must be [T] or MRoPE [3, T], but got "
      << positions_host.sizes() << ".";
  if (positions_host.dim() == 1) {
    CHECK_EQ(positions_host.numel(), num_context_tokens)
        << "DFlash prefill hidden/position count mismatch.";
    return false;
  }

  CHECK_EQ(positions_host.size(0), 3)
      << "DFlash MRoPE context positions must have three axes, but got "
      << positions_host.sizes() << ".";
  CHECK_EQ(positions_host.size(1), num_context_tokens)
      << "DFlash MRoPE position axes must align with context tokens.";
  return true;
}

// Rebuild positions in the KV cache's logical token-index coordinate system.
// This mirrors the decode-side invariant position + 1 == kv_len and keeps the
// draft query and transplanted context K in the same coordinate system.
inline std::vector<int32_t> build_token_aligned_context_positions(
    const std::vector<int32_t>& kv_seq_lens,
    const std::vector<int32_t>& q_seq_lens,
    int64_t num_context_tokens) {
  CHECK_EQ(kv_seq_lens.size(), q_seq_lens.size())
      << "DFlash prefill KV/query sequence count mismatch.";
  CHECK_GE(num_context_tokens, 0);
  std::vector<int32_t> positions;
  positions.reserve(static_cast<size_t>(num_context_tokens));
  for (size_t seq = 0; seq < kv_seq_lens.size(); ++seq) {
    const int32_t kv_len = kv_seq_lens[seq];
    const int32_t q_len = q_seq_lens[seq];
    CHECK_GE(q_len, 0) << "DFlash prefill query length must be non-negative.";
    CHECK_GE(kv_len, q_len)
        << "DFlash prefill kv_len must cover q_len, seq=" << seq;
    const int32_t start = kv_len - q_len;
    for (int32_t offset = 0; offset < q_len; ++offset) {
      positions.emplace_back(start + offset);
    }
  }
  CHECK_EQ(static_cast<int64_t>(positions.size()), num_context_tokens)
      << "DFlash token-aligned positions must match the context token count.";
  return positions;
}

}  // namespace xllm::dflash_detail
