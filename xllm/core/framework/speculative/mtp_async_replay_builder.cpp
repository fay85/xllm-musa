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

#include "core/framework/speculative/mtp_async_replay_builder.h"

#include <glog/logging.h>

#include <algorithm>

#include "runtime/forward_params.h"

namespace xllm::mtp_async {

DraftReplayInputPlan build_draft_replay_input_plan(
    const DecodeBatchContext& ctx,
    const std::vector<EmbeddingCache::DecodeState>& states,
    const torch::Tensor& placeholder,
    int32_t block_size,
    int32_t uniform_width,
    bool is_graph_warmup,
    int32_t pad_cache_slot) {
  CHECK_EQ(states.size(), static_cast<size_t>(ctx.num_requests));
  CHECK_GE(uniform_width, 0);
  DraftReplayInputPlan result;
  int32_t capacity = 0;
  for (const auto& state : states) {
    capacity +=
        uniform_width > 0
            ? uniform_width
            : std::max<int32_t>(/*__a=*/1, state.replay_token_ids.size());
  }
  prepare_decode_buffers(
      result.rows, ctx, capacity, /*emit_block_tables=*/true);
  result.embeddings.reserve(capacity);
  result.selected_rows.reserve(ctx.num_requests);
  result.source_sequences.reserve(capacity);
  result.valid_rows.reserve(capacity);
  if (ctx.num_rope_axes == 3) {
    result.mrope_positions.reserve(capacity);
  }
  for (int32_t seq_id = 0; seq_id < ctx.num_requests; ++seq_id) {
    const auto& state = states[seq_id];
    const int32_t span = static_cast<int32_t>(state.replay_token_ids.size());
    CHECK(state.valid || is_graph_warmup)
        << "MTP replay requires target context";
    if (state.valid) {
      CHECK_GT(span, 0);
      CHECK_EQ(state.replay_token_ids.back(), ctx.token_ids[seq_id]);
      CHECK(state.replay_embeddings.defined());
      CHECK_EQ(state.replay_embeddings.dim(), 2);
      CHECK_EQ(state.replay_embeddings.size(/*dim=*/0), span);
      CHECK_GE(ctx.positions[seq_id], span);
    }
    const int32_t width =
        uniform_width > 0 ? uniform_width : std::max(/*__a=*/1, span);
    CHECK_GE(width, span);
    const int32_t padding = width - span;
    for (int32_t row_id = 0; row_id < width; ++row_id) {
      const int32_t replay_id = row_id - padding;
      const bool valid = replay_id >= 0;
      DecodeTokenInput row;
      row.req_id = seq_id;
      row.token_id = valid ? state.replay_token_ids[replay_id] : 0;
      row.position_offset = valid ? replay_id - span : -ctx.positions[seq_id];
      row.append_q_len_one = true;
      row.append_block_table = true;
      append_decode_token(ctx, row, block_size, result.rows);
      if (!valid) {
        result.rows.out_new_cache_slots.back() = pad_cache_slot;
      }
      result.embeddings.emplace_back(
          valid ? state.replay_embeddings.select(/*dim=*/0, replay_id)
                : placeholder);
      result.source_sequences.emplace_back(seq_id);
      result.valid_rows.emplace_back(valid ? 1 : 0);
      if (ctx.num_rope_axes == 3) {
        std::array<int32_t, 3> rotary = {0, 0, 0};
        if (valid) {
          if (state.replay_mrope_positions.defined()) {
            CHECK_EQ(state.replay_mrope_positions.sizes(),
                     torch::IntArrayRef({3, span}));
            auto stored = state.replay_mrope_positions.accessor<int32_t, 2>();
            for (int32_t axis = 0; axis < 3; ++axis) {
              rotary[axis] = stored[axis][replay_id];
            }
          } else {
            for (int32_t axis = 0; axis < 3; ++axis) {
              rotary[axis] = ctx.positions[axis * ctx.num_requests + seq_id] +
                             row.position_offset;
            }
          }
        }
        result.mrope_positions.emplace_back(rotary);
      }
    }
    result.selected_rows.emplace_back(result.embeddings.size() - 1);
  }
  result.kpool_query_lens.assign(ctx.num_requests, uniform_width);
  return result;
}

DraftReplayInputPlan build_draft_replay_input_plan(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& states,
    const torch::Tensor& embedding_placeholder,
    int32_t block_size,
    int32_t uniform_width,
    bool graph_warmup,
    int32_t pad_cache_slot) {
  return build_draft_replay_input_plan(make_decode_batch_context(base_input),
                                       states,
                                       embedding_placeholder,
                                       block_size,
                                       uniform_width,
                                       graph_warmup,
                                       pad_cache_slot);
}

}  // namespace xllm::mtp_async
