/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "core/framework/block/block_manager_impl.h"
#include "core/util/hash_util.h"

namespace xllm {

class Sequence;

// Returns true when the latest confirmed Decode result completed a checkpoint.
bool last_decode_step_crosses_checkpoint(const Sequence& sequence,
                                         size_t checkpoint_stride);

// Returns the row of the completed checkpoint within that Decode result.
std::optional<size_t> last_decode_step_checkpoint_row(const Sequence& sequence,
                                                      size_t checkpoint_stride);

class LinearStateBlockManager final : public BlockManagerImpl {
 public:
  explicit LinearStateBlockManager(uint32_t num_slots,
                                   int32_t chunk_stride,
                                   bool enable_prefix_cache = true,
                                   bool instance_is_decode = false,
                                   uint32_t num_speculative_tokens = 0,
                                   bool preserve_decode_checkpoint = false,
                                   uint32_t num_ckpt_slots = 0);
  ~LinearStateBlockManager() override = default;

  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      KVCacheState& kv_state,
      size_t num_tokens) override;
  bool allocate_for_prefetch(Sequence* seq, size_t num_tokens) override;
  void trim_prefetch_blocks(Sequence* seq, size_t max_hit_tokens) override;

  using BlockManagerImpl::allocate;
  Block allocate() override;
  void deallocate(std::vector<Block>&& blocks) override;
  void free(int32_t block_id) override;
  void publish_linear_checkpoint(
      Sequence* seq,
      KVCacheState& kv_state,
      std::vector<BlockTransferInfo>* swap_transfers) override;

  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  using BlockManagerImpl::cache;

 private:
  void cache_read_source(Sequence* seq, KVCacheState& kv_state);
  void retain_read_source(KVCacheState& kv_state);
  std::optional<std::vector<Block>> allocate_prefill(Sequence* seq,
                                                     KVCacheState& kv_state);
  std::optional<std::vector<Block>> allocate_decode(Sequence* seq,
                                                    KVCacheState& kv_state);

  // Host offload consumes a completed Decode checkpoint after both the HBM
  // and Host leaves finish allocation. Keep the source in the rolling vector
  // until HierarchyBlockManagerPool moves it into an OffloadBlockPair.
  int32_t alloc_ckpt_slot_locked();

  bool preserve_decode_checkpoint_ = false;
  uint32_t num_active_slots_ = 0;
  uint32_t num_ckpt_slots_ = 0;
  std::mutex ckpt_mutex_;
  std::vector<int32_t> ckpt_free_ids_;

  friend class BlockManagerPoolTestPeer;
};

}  // namespace xllm
