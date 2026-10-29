/* Copyright 2026 The xLLM Authors.

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

#include "framework/block/musa/deepseek_v4_prefix_cache_policy.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "framework/block/block_manager.h"
#include "framework/block/block_manager_pool.h"
#include "framework/block/composite_block_manager.h"
#include "framework/block/musa/deepseek_v4_prefix_transfer.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "util/hash_util.h"

namespace xllm::musa {

namespace {

constexpr size_t kPrefixRateNumerator = 80;
constexpr size_t kPrefixRateDenominator = 100;
constexpr size_t kMaxPrefixEntries = 64;

using PrefixBlockSnapshot = std::map<BlockType, std::vector<Block>>;

struct PrefixCacheEntry {
  size_t prefix_tokens = 0;
  PrefixBlockSnapshot blocks;
  Block state_block;
  uint64_t last_access = 0;
  bool ready = false;
};

using PrefixCache = std::unordered_map<XXH3Key,
                                       PrefixCacheEntry,
                                       FixedStringKeyHash,
                                       FixedStringKeyEqual>;

struct PendingSequenceState {
  XXH3Key snapshot_key;
  int32_t snapshot_block_id = -1;
  size_t snapshot_tokens = 0;
  int32_t restore_src_block_id = -1;
};

struct PrefixPoolState {
  explicit PrefixPoolState(int32_t dp_size)
      : caches(static_cast<size_t>(dp_size)) {}

  std::vector<PrefixCache> caches;
  std::unordered_map<const Sequence*, PendingSequenceState> pending_sequences;
  uint64_t access_counter = 0;
};

std::mutex prefix_pool_states_mutex;
std::unordered_map<const BlockManagerPool*, std::unique_ptr<PrefixPoolState>>
    prefix_pool_states;

bool has_deepseek_v4_cache_shape(const CompositeBlockManager& composite) {
  const CompositeBlockManager::LeafMap& leaves = composite.leaf_entries();
  return leaves.find(BlockType::SWA) != leaves.end() &&
         leaves.find(BlockType::C4) != leaves.end() &&
         leaves.find(BlockType::C128) != leaves.end() &&
         leaves.find(BlockType::KV) == leaves.end() &&
         leaves.find(BlockType::LINEAR) == leaves.end();
}

bool snapshot_prefix_blocks(const CompositeBlockManager& composite,
                            Sequence& sequence,
                            size_t prefix_tokens,
                            PrefixBlockSnapshot* snapshot) {
  if (snapshot == nullptr || prefix_tokens == 0 ||
      !has_deepseek_v4_cache_shape(composite)) {
    return false;
  }

  PrefixBlockSnapshot candidate;
  for (const auto& [block_type, entry] : composite.leaf_entries()) {
    if (block_type == BlockType::EMBEDDING || block_type == BlockType::LINEAR) {
      continue;
    }
    const size_t leaf_block_size = entry.leaf->block_size();
    if (leaf_block_size == 0) {
      return false;
    }
    const size_t blocks_needed =
        (prefix_tokens + leaf_block_size - 1) / leaf_block_size;
    const Slice<Block> held = sequence.kv_state().blocks(block_type);
    if (blocks_needed == 0 || held.size() < blocks_needed) {
      return false;
    }

    size_t first_required = 0;
    if (block_type == BlockType::SWA) {
      const size_t window =
          std::max<size_t>(entry.leaf->options().sliding_window_size(), 1);
      if (prefix_tokens > window) {
        first_required = (prefix_tokens - window) / leaf_block_size;
      }
    }
    std::vector<Block> prefix_blocks;
    prefix_blocks.reserve(blocks_needed);
    for (size_t block_index = 0; block_index < blocks_needed; ++block_index) {
      if (block_index < first_required) {
        // Preserve absolute logical columns without pinning slid-out SWA
        // pages. Only the tail window and separate recurrent snapshot are
        // needed on restore; C4/C128 still retain the complete prefix.
        prefix_blocks.emplace_back();
      } else {
        prefix_blocks.emplace_back(held[block_index]);
      }
    }
    for (size_t block_index = first_required;
         block_index < prefix_blocks.size();
         ++block_index) {
      if (!prefix_blocks[block_index].is_valid()) {
        return false;
      }
    }
    candidate.emplace(block_type, std::move(prefix_blocks));
  }
  *snapshot = std::move(candidate);
  return true;
}

bool restore_prefix_blocks(CompositeBlockManager* composite,
                           Sequence* sequence,
                           size_t prefix_tokens,
                           const PrefixBlockSnapshot& snapshot,
                           std::vector<BlockTransferInfo>* transfers) {
  if (composite == nullptr || sequence == nullptr || transfers == nullptr ||
      prefix_tokens == 0 || !has_deepseek_v4_cache_shape(*composite) ||
      sequence->kv_state().has_any_blocks()) {
    return false;
  }

  PrefixBlockSnapshot restored;
  std::vector<BlockTransferInfo> pending_copies;
  for (const auto& [block_type, entry] : composite->leaf_entries()) {
    if (block_type == BlockType::EMBEDDING || block_type == BlockType::LINEAR) {
      continue;
    }
    const auto snapshot_it = snapshot.find(block_type);
    if (snapshot_it == snapshot.end()) {
      return false;
    }
    const size_t leaf_block_size = entry.leaf->block_size();
    const size_t blocks_needed =
        (prefix_tokens + leaf_block_size - 1) / leaf_block_size;
    if (blocks_needed == 0 || snapshot_it->second.size() < blocks_needed) {
      return false;
    }
    std::vector<Block> blocks(snapshot_it->second.begin(),
                              snapshot_it->second.begin() + blocks_needed);

    if ((block_type == BlockType::C4 || block_type == BlockType::C128) &&
        prefix_tokens % leaf_block_size != 0) {
      if (!blocks.back().is_valid()) {
        return false;
      }
      std::vector<Block> private_tail = entry.leaf->allocate(1);
      if (private_tail.size() != 1) {
        return false;
      }
      BlockTransferInfo transfer(/*src_block_id=*/0, /*dst_block_id=*/0);
      if (!encode_deepseek_v4_prefix_block_copy(block_type,
                                                blocks.back().id(),
                                                private_tail.front().id(),
                                                &transfer)) {
        return false;
      }
      pending_copies.emplace_back(std::move(transfer));
      blocks.back() = std::move(private_tail.front());
    }
    restored.emplace(block_type, std::move(blocks));
  }

  for (auto& [block_type, blocks] : restored) {
    sequence->kv_state().mount_composite_shared(block_type, std::move(blocks));
  }
  sequence->kv_state().set_kv_cache_tokens_num(prefix_tokens);
  sequence->kv_state().set_prefix_cache_matched();
  transfers->insert(
      transfers->end(), pending_copies.begin(), pending_copies.end());
  return true;
}

bool has_pending_reader(const PrefixCacheEntry& entry,
                        const PrefixPoolState& pool,
                        const std::vector<BlockTransferInfo>& transfers) {
  for (const auto& [sequence, pending] : pool.pending_sequences) {
    if (pending.restore_src_block_id == entry.state_block.id()) return true;
  }
  for (const auto& transfer : transfers) {
    const auto decoded = decode_deepseek_v4_prefix_transfer(transfer);
    if (decoded.kind != DeepseekV4PrefixTransferKind::BLOCK_COPY) continue;
    if (decoded.block_type == BlockType::SWA &&
        decoded.src_block_id == entry.state_block.id())
      return true;
    const auto blocks = entry.blocks.find(decoded.block_type);
    if (blocks == entry.blocks.end()) continue;
    for (const auto& block : blocks->second) {
      if (block.is_valid() && block.id() == decoded.src_block_id) return true;
    }
  }
  return false;
}

bool evict_oldest_ready_entry(PrefixCache* cache,
                              const PrefixPoolState& pool,
                              const std::vector<BlockTransferInfo>& transfers) {
  if (cache == nullptr) {
    return false;
  }
  auto oldest = cache->end();
  for (auto it = cache->begin(); it != cache->end(); ++it) {
    if (!it->second.ready || has_pending_reader(it->second, pool, transfers)) {
      continue;
    }
    if (oldest == cache->end() ||
        it->second.last_access < oldest->second.last_access) {
      oldest = it;
    }
  }
  if (oldest == cache->end()) {
    return false;
  }
  cache->erase(oldest);
  return true;
}

PrefixPoolState* find_pool_state(const BlockManagerPool* owner) {
  const auto it = prefix_pool_states.find(owner);
  return it == prefix_pool_states.end() ? nullptr : it->second.get();
}

}  // namespace

void initialize_deepseek_v4_prefix_cache(const BlockManagerPool* owner,
                                         int32_t dp_size) {
  CHECK(owner != nullptr);
  CHECK_GT(dp_size, 0);
  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  prefix_pool_states.erase(owner);
  // These snapshots are written by the prefill forward at an exact prompt
  // boundary. PD Decode only imports the final prompt state through PULL;
  // it never executes that producer. Publishing its pending snapshot after
  // generation would mark unwritten state as ready, and a repeated prompt
  // could overwrite freshly imported state with that invalid snapshot.
  // Leave ordinary composite C4/C128 prefix participation unchanged.
  if (owner->options().instance_is_decode()) {
    return;
  }
  prefix_pool_states.emplace(owner, std::make_unique<PrefixPoolState>(dp_size));
}

void destroy_deepseek_v4_prefix_cache(const BlockManagerPool* owner) {
  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  prefix_pool_states.erase(owner);
}

bool try_restore_deepseek_v4_prefix(const BlockManagerPool* owner,
                                    Sequence* sequence,
                                    int32_t dp_rank,
                                    CompositeBlockManager* composite,
                                    uint32_t base_block_size,
                                    BlockHasherType hasher_type,
                                    std::vector<BlockTransferInfo>* transfers) {
  if (owner == nullptr || sequence == nullptr || composite == nullptr ||
      transfers == nullptr || base_block_size == 0 ||
      !owner->options().enable_prefix_cache() ||
      !has_deepseek_v4_cache_shape(*composite)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  PrefixPoolState* pool_state = find_pool_state(owner);
  if (pool_state == nullptr || dp_rank < 0 ||
      static_cast<size_t>(dp_rank) >= pool_state->caches.size()) {
    return false;
  }

  sequence->update_block_hashes(base_block_size, hasher_type);
  const size_t full_blocks =
      std::min(sequence->tokens().size() / base_block_size,
               sequence->block_hashes().size());
  PrefixCache& cache = pool_state->caches[static_cast<size_t>(dp_rank)];
  for (size_t block_count = full_blocks; block_count > 0; --block_count) {
    const size_t prefix_tokens = block_count * base_block_size;
    if (prefix_tokens >= sequence->tokens().size()) {
      continue;
    }
    const XXH3Key& key = sequence->block_hashes()[block_count - 1];
    auto entry_it = cache.find(key);
    if (entry_it == cache.end() || !entry_it->second.ready ||
        entry_it->second.prefix_tokens != prefix_tokens ||
        !entry_it->second.state_block.is_valid()) {
      continue;
    }
    const size_t original_transfer_count = transfers->size();
    if (!restore_prefix_blocks(composite,
                               sequence,
                               prefix_tokens,
                               entry_it->second.blocks,
                               transfers)) {
      transfers->erase(transfers->begin() + original_transfer_count,
                       transfers->end());
      continue;
    }
    PendingSequenceState& pending = pool_state->pending_sequences[sequence];
    pending.restore_src_block_id = entry_it->second.state_block.id();
    entry_it->second.last_access = ++pool_state->access_counter;
    return true;
  }
  return false;
}

void prepare_deepseek_v4_prefix_snapshot(
    const BlockManagerPool* owner,
    Sequence* sequence,
    int32_t dp_rank,
    CompositeBlockManager* composite,
    uint32_t base_block_size,
    BlockHasherType hasher_type,
    size_t max_computed_tokens,
    std::vector<BlockTransferInfo>* transfers) {
  if (owner == nullptr || sequence == nullptr || composite == nullptr ||
      transfers == nullptr || base_block_size == 0 ||
      !owner->options().enable_prefix_cache() ||
      !has_deepseek_v4_cache_shape(*composite)) {
    return;
  }

  const size_t prompt_tokens = sequence->num_prompt_tokens();
  if (prompt_tokens == 0) {
    return;
  }
  const size_t unaligned_prefix =
      (prompt_tokens / kPrefixRateDenominator) * kPrefixRateNumerator +
      ((prompt_tokens % kPrefixRateDenominator) * kPrefixRateNumerator) /
          kPrefixRateDenominator;
  const size_t prefix_tokens =
      (unaligned_prefix / base_block_size) * base_block_size;
  const size_t computed_before = sequence->kv_state().kv_cache_tokens_num();
  const char* chunk_env = std::getenv("XLLM_DSV4_CHUNK_PREFIX_SNAPSHOT");
  const bool allow_chunked =
      chunk_env != nullptr && std::string(chunk_env) == "1";
  if (!deepseek_v4_prefix_snapshot_in_chunk(prompt_tokens,
                                            prefix_tokens,
                                            computed_before,
                                            max_computed_tokens,
                                            allow_chunked)) {
    return;
  }

  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  PrefixPoolState* pool_state = find_pool_state(owner);
  if (pool_state == nullptr || dp_rank < 0 ||
      static_cast<size_t>(dp_rank) >= pool_state->caches.size() ||
      pool_state->pending_sequences.find(sequence) !=
          pool_state->pending_sequences.end()) {
    return;
  }

  sequence->update_block_hashes(base_block_size, hasher_type);
  const size_t block_count = prefix_tokens / base_block_size;
  if (block_count == 0 || sequence->block_hashes().size() < block_count) {
    return;
  }
  const XXH3Key key = sequence->block_hashes()[block_count - 1];
  PrefixCache& cache = pool_state->caches[static_cast<size_t>(dp_rank)];
  if (cache.find(key) != cache.end()) {
    return;
  }

  PrefixBlockSnapshot snapshot;
  if (!snapshot_prefix_blocks(
          *composite, *sequence, prefix_tokens, &snapshot)) {
    return;
  }
  while (cache.size() >= kMaxPrefixEntries &&
         evict_oldest_ready_entry(&cache, *pool_state, *transfers)) {
  }
  if (cache.size() >= kMaxPrefixEntries) {
    return;
  }

  std::vector<Block> state_blocks =
      composite->allocate_blocks(BlockType::SWA, /*num_blocks=*/1);
  while (state_blocks.empty() &&
         evict_oldest_ready_entry(&cache, *pool_state, *transfers)) {
    state_blocks = composite->allocate_blocks(BlockType::SWA, /*num_blocks=*/1);
  }
  if (state_blocks.size() != 1) {
    return;
  }

  PrefixCacheEntry entry;
  entry.prefix_tokens = prefix_tokens;
  entry.blocks = std::move(snapshot);
  entry.state_block = std::move(state_blocks.front());
  entry.last_access = ++pool_state->access_counter;
  const int32_t state_block_id = entry.state_block.id();
  cache.emplace(key, std::move(entry));

  PendingSequenceState pending;
  pending.snapshot_key = key;
  pending.snapshot_block_id = state_block_id;
  pending.snapshot_tokens = prefix_tokens;
  pool_state->pending_sequences.emplace(sequence, std::move(pending));

  BlockTransferInfo snapshot_transfer(/*src_block_id=*/0,
                                      /*dst_block_id=*/0);
  CHECK(encode_deepseek_v4_prefix_state_snapshot(
      state_block_id, static_cast<int32_t>(prefix_tokens), &snapshot_transfer));
  transfers->emplace_back(std::move(snapshot_transfer));
  if (allow_chunked) {
    LOG_FIRST_N(INFO, 8) << "[DSV4_CHUNK_PREFIX_SNAPSHOT] before="
                         << computed_before << " end=" << max_computed_tokens
                         << " prefix=" << prefix_tokens
                         << " prompt=" << prompt_tokens;
  }
}

void activate_deepseek_v4_prefix_snapshot(const BlockManagerPool* owner,
                                          Sequence* sequence,
                                          int32_t dp_rank) {
  if (owner == nullptr || sequence == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  PrefixPoolState* pool_state = find_pool_state(owner);
  if (pool_state == nullptr || dp_rank < 0 ||
      static_cast<size_t>(dp_rank) >= pool_state->caches.size()) {
    return;
  }
  auto pending_it = pool_state->pending_sequences.find(sequence);
  if (pending_it == pool_state->pending_sequences.end() ||
      pending_it->second.snapshot_block_id < 0 ||
      pending_it->second.snapshot_tokens == 0 ||
      sequence->kv_state().kv_cache_tokens_num() <
          pending_it->second.snapshot_tokens) {
    return;
  }
  PrefixCache& cache = pool_state->caches[static_cast<size_t>(dp_rank)];
  auto cache_it = cache.find(pending_it->second.snapshot_key);
  if (cache_it == cache.end() || cache_it->second.state_block.id() !=
                                     pending_it->second.snapshot_block_id) {
    pool_state->pending_sequences.erase(pending_it);
    return;
  }
  cache_it->second.ready = true;
  cache_it->second.last_access = ++pool_state->access_counter;
  pool_state->pending_sequences.erase(pending_it);
}

void discard_deepseek_v4_prefix_sequence(const BlockManagerPool* owner,
                                         Sequence* sequence,
                                         int32_t dp_rank) {
  if (owner == nullptr || sequence == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  PrefixPoolState* pool_state = find_pool_state(owner);
  if (pool_state == nullptr || dp_rank < 0 ||
      static_cast<size_t>(dp_rank) >= pool_state->caches.size()) {
    return;
  }
  auto pending_it = pool_state->pending_sequences.find(sequence);
  if (pending_it == pool_state->pending_sequences.end()) {
    return;
  }
  if (pending_it->second.snapshot_block_id >= 0) {
    PrefixCache& cache = pool_state->caches[static_cast<size_t>(dp_rank)];
    auto cache_it = cache.find(pending_it->second.snapshot_key);
    if (cache_it != cache.end() && !cache_it->second.ready &&
        cache_it->second.state_block.id() ==
            pending_it->second.snapshot_block_id) {
      cache.erase(cache_it);
    }
  }
  pool_state->pending_sequences.erase(pending_it);
}

void append_deepseek_v4_prefix_state_restore(
    const BlockManagerPool* owner,
    Sequence* sequence,
    std::vector<BlockTransferInfo>* transfers) {
  if (owner == nullptr || sequence == nullptr || transfers == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  PrefixPoolState* pool_state = find_pool_state(owner);
  if (pool_state == nullptr) {
    return;
  }
  auto pending_it = pool_state->pending_sequences.find(sequence);
  if (pending_it == pool_state->pending_sequences.end() ||
      pending_it->second.restore_src_block_id < 0) {
    return;
  }

  const Slice<Block> swa_blocks = sequence->kv_state().blocks(BlockType::SWA);
  int32_t dst_block_id = -1;
  for (size_t block_index = swa_blocks.size(); block_index > 0; --block_index) {
    if (swa_blocks[block_index - 1].is_valid()) {
      dst_block_id = swa_blocks[block_index - 1].id();
      break;
    }
  }
  CHECK_GE(dst_block_id, 0)
      << "DeepSeek-V4 prefix restore requires an active SWA destination";
  BlockTransferInfo transfer(/*src_block_id=*/0, /*dst_block_id=*/0);
  CHECK(encode_deepseek_v4_prefix_block_copy(
      BlockType::SWA,
      pending_it->second.restore_src_block_id,
      dst_block_id,
      &transfer));
  transfers->emplace_back(std::move(transfer));
  pool_state->pending_sequences.erase(pending_it);
}

void reset_deepseek_v4_prefix_cache(const BlockManagerPool* owner) {
  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  PrefixPoolState* pool_state = find_pool_state(owner);
  if (pool_state == nullptr) {
    return;
  }
  for (PrefixCache& cache : pool_state->caches) {
    cache.clear();
  }
  pool_state->pending_sequences.clear();
  pool_state->access_counter = 0;
}

bool evict_one_deepseek_v4_prefix_snapshot(
    const BlockManagerPool* owner,
    int32_t dp_rank,
    const std::vector<BlockTransferInfo>& queued_transfers) {
  if (!owner || !owner->options().enable_prefix_cache()) return false;
  std::lock_guard<std::mutex> lock(prefix_pool_states_mutex);
  auto* pool = find_pool_state(owner);
  if (!pool || dp_rank < 0 ||
      static_cast<size_t>(dp_rank) >= pool->caches.size()) {
    return false;
  }
  return evict_oldest_ready_entry(
      &pool->caches[dp_rank], *pool, queued_transfers);
}

}  // namespace xllm::musa
