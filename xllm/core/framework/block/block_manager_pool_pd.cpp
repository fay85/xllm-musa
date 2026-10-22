/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/block/block_manager_pool.h"

#include <algorithm>

#include "core/framework/block/composite_block_manager.h"
#include "core/framework/request/sequence.h"

namespace xllm {

size_t BlockManagerPool::reclaim_completed_pd_prefill_swa(Sequence* sequence) {
#if defined(DSPARK_PD_RECLAIM_CONTROL)
  return reclaim_completed_prefill_swa(sequence);
#else
  if (sequence == nullptr || !options_.enable_disagg_pd() ||
      options_.instance_is_decode() || options_.sliding_window_size() == 0 ||
      options_.enable_host_offload() || options_.enable_kvcache_store() ||
      sequence->check_beam_search() ||
      sequence->kv_state().kv_cache_tokens_num() < sequence->num_prompt_tokens()) {
    return 0;
  }
  const int32_t rank = sequence->dp_rank();
  if (rank < 0 || static_cast<size_t>(rank) >= block_managers_.size()) {
    return 0;
  }
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[rank].get());
  const auto& leaves = composite->leaf_entries();
  if (!leaves.contains(BlockType::SWA) || !leaves.contains(BlockType::C4) ||
      !leaves.contains(BlockType::C128)) {
    return 0;
  }
  const auto count_valid = [&]() -> size_t {
    const auto blocks = sequence->kv_state().blocks(BlockType::SWA);
    return static_cast<size_t>(std::count_if(
        blocks.begin(), blocks.end(),
        [](const Block& block) { return block.is_valid(); }));
  };
  const size_t before = count_valid();
  // Publish completed snapshots before releasing live SWA references. The
  // leaf retains logical placeholders, the window plus speculative margin,
  // and all C4/C128 history. No source mapping has been exposed yet.
  cache(sequence);
  composite->release_out_of_window_for_sequence(sequence);
  return before - count_valid();
#endif
}

}  // namespace xllm
