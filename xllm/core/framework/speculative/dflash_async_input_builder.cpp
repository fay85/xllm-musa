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

#include "core/framework/speculative/dflash_async_input_builder.h"

#include <glog/logging.h>

#include "core/framework/speculative/speculative_cache_utils.h"

namespace xllm::dflash_async {
NextDraftInputs prepare_next_draft(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& block_table,
                                   int64_t mask_token_id,
                                   int64_t block_size,
                                   const torch::Tensor& base_cache_positions) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(base_positions.numel(), accepted_tokens.size(0));
  CHECK_EQ(block_table.size(0), accepted_tokens.size(0));
  CHECK_GT(block_size, 0);
  const torch::Tensor cache_base_1d =
      base_cache_positions.defined() ? base_cache_positions : base_positions;
  CHECK_EQ(cache_base_1d.dim(), 1);
  CHECK_EQ(cache_base_1d.numel(), accepted_tokens.size(0));
  const int64_t width = accepted_tokens.size(1);
  torch::Tensor row =
      torch::arange(width, base_positions.options().dtype(torch::kLong))
          .view({1, width});
  torch::Tensor valid = accepted_tokens.ge(0).to(torch::kLong).cumprod(1);
  NextDraftInputs result;
  result.accepted_counts = valid.sum(1);
  // The rejection sampler guarantees at least one token per active row.
  torch::Tensor last = (result.accepted_counts - 1).clamp_min(0).unsqueeze(1);
  result.anchor_tokens = accepted_tokens.gather(1, last).reshape({-1});
  result.query_tokens =
      torch::where(row.eq(0),
                   result.anchor_tokens.unsqueeze(1),
                   torch::full_like(accepted_tokens, mask_token_id))
          .reshape({-1});
  torch::Tensor model_base = base_positions.to(torch::kLong).unsqueeze(1);
  torch::Tensor cache_base = cache_base_1d.to(torch::kLong).unsqueeze(1);
  torch::Tensor next_positions =
      model_base + result.accepted_counts.unsqueeze(1) + row;
  torch::Tensor next_cache_positions =
      cache_base + result.accepted_counts.unsqueeze(1) + row;
  result.query_positions = next_positions.reshape({-1});
  result.query_slots = speculative::map_positions_to_cache_slots(
      block_table, next_cache_positions, block_size);
  result.kv_lengths =
      (cache_base_1d.to(torch::kLong) + result.accepted_counts + width)
          .to(torch::kInt32);
  torch::Tensor context_positions = model_base + row;
  torch::Tensor context_cache_positions = cache_base + row;
  result.context_positions = context_positions.reshape({-1});
  torch::Tensor context_slots = speculative::map_positions_to_cache_slots(
      block_table, context_cache_positions, block_size);
  result.context_slots = torch::where(valid.reshape({-1}).ne(0),
                                      context_slots,
                                      torch::full_like(context_slots, -1));
  return result;
}
}  // namespace xllm::dflash_async
