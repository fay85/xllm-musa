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

#include "core/framework/block/linear_state_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "core/framework/block/musa/deepseek_v4_prefix_transfer.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/prefix_cache/linear_state_prefix_cache.h"
#include "core/framework/request/sequence.h"

namespace xllm {

namespace {

BlockManager::Options make_linear_state_options(
    uint32_t num_slots,
    int32_t chunk_stride,
    bool enable_prefix_cache,
    bool instance_is_decode,
    uint32_t num_speculative_tokens) {
  BlockManager::Options options;
  options.num_blocks(num_slots);
  options.block_size(chunk_stride);
  options.enable_prefix_cache(enable_prefix_cache);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::LINEAR);
  options.instance_is_decode(instance_is_decode);
  options.num_speculative_tokens(num_speculative_tokens);
  return options;
}

inline bool is_chat_im_start_token(int32_t token_id) {
  // Qwen3.5 / Qwen3.8 (<|im_start|>=248045) and Qwen2.5 / Qwen3 (151644).
  return token_id == 248045 || token_id == 151644;
}

inline bool is_chat_im_end_token(int32_t token_id) {
  // Qwen3.5 / Qwen3.8 (<|im_end|>=248046) and Qwen2.5 / Qwen3 (151645).
  return token_id == 248046 || token_id == 151645;
}

struct ChatLinearCheckpointBounds {
  bool has_chat_boundaries = false;
  size_t hist_ckpt = 0;
  size_t end_ckpt = 0;
};

ChatLinearCheckpointBounds find_chat_linear_checkpoint_bounds(
    const Slice<int32_t>& tokens,
    size_t total_tokens,
    size_t align_stride) {
  ChatLinearCheckpointBounds out;
  if (align_stride == 0 || total_tokens <= align_stride ||
      tokens.size() < total_tokens) {
    return out;
  }
  const size_t scan_limit = std::min<size_t>(total_tokens, 8192);
  const size_t min_idx = total_tokens - scan_limit;
  constexpr size_t kNpos = std::numeric_limits<size_t>::max();

  size_t last_im_end = kNpos;
  for (size_t i = total_tokens; i > min_idx; --i) {
    if (is_chat_im_end_token(tokens[i - 1])) {
      last_im_end = i - 1;
      break;
    }
  }
  if (last_im_end == kNpos) {
    return out;
  }
  out.has_chat_boundaries = true;
  const size_t safe_end_limit =
      std::min<size_t>(total_tokens - 1, last_im_end + 3);
  if (safe_end_limit >= align_stride) {
    out.end_ckpt = (safe_end_limit / align_stride) * align_stride;
  }

  // Locate the <|im_start|> opening the final message (preceded by <|im_end|>).
  for (size_t i = last_im_end; i > min_idx; --i) {
    const size_t idx = i - 1;
    if (!is_chat_im_start_token(tokens[idx])) {
      continue;
    }
    bool preceded_by_im_end = false;
    const size_t lookback = std::min<size_t>(idx, 4);
    for (size_t d = 1; d <= lookback; ++d) {
      if (is_chat_im_end_token(tokens[idx - d])) {
        preceded_by_im_end = true;
        break;
      }
    }
    if (preceded_by_im_end) {
      if (idx >= align_stride) {
        out.hist_ckpt = (idx / align_stride) * align_stride;
      }
      break;
    }
  }
  return out;
}

bool should_publish_linear_checkpoint(const Sequence& seq,
                                      size_t cached_tokens,
                                      size_t total_tokens,
                                      size_t chunk_stride) {
  if (cached_tokens == 0 || chunk_stride == 0 ||
      cached_tokens % chunk_stride != 0) {
    return false;
  }
  if (total_tokens == 0 || cached_tokens >= total_tokens) {
    return true;
  }
  const int32_t configured_chunk =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const size_t major_stride = std::max<size_t>(
      chunk_stride,
      configured_chunk > 0 ? static_cast<size_t>(configured_chunk) : 8192);
  if (cached_tokens % major_stride == 0) {
    return true;
  }
  const size_t fine_stride =
      std::max<size_t>(chunk_stride, std::min<size_t>(major_stride, 2048));
  const size_t fallback_ckpt =
      ((total_tokens - 1) / fine_stride) * fine_stride;
  if (cached_tokens == fallback_ckpt) {
    return true;
  }
  const ChatLinearCheckpointBounds bounds =
      find_chat_linear_checkpoint_bounds(
          seq.tokens(), total_tokens, chunk_stride);
  if (bounds.has_chat_boundaries &&
      (cached_tokens == bounds.hist_ckpt || cached_tokens == bounds.end_ckpt)) {
    return true;
  }
  return false;
}

bool crosses_checkpoint(size_t begin, size_t end, size_t checkpoint_stride) {
  CHECK_LE(begin, end);
  CHECK_GT(checkpoint_stride, 0u);
  return begin / checkpoint_stride != end / checkpoint_stride;
}

bool singlepass_gdn_ckpt_enabled() {
  const char* env = std::getenv("XLLM_GDN_SINGLEPASS_CKPT");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return std::string(env) != "0" && std::string(env) != "false" &&
         std::string(env) != "FALSE";
}

size_t read_env_size_t(const char* name, size_t default_val) {
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return default_val;
  }
  char* endptr = nullptr;
  const unsigned long long parsed = std::strtoull(env, &endptr, 10);
  if (endptr == env) {
    return default_val;
  }
  return static_cast<size_t>(parsed);
}

struct PendingSinglepassGdnEntry {
  PrefixCache* prefix_cache = nullptr;
  XXH3Key key;
  Block ckpt_block;
};

std::mutex& pending_singlepass_mutex() {
  static std::mutex m;
  return m;
}

std::vector<PendingSinglepassGdnEntry>& pending_singlepass_entries() {
  static std::vector<PendingSinglepassGdnEntry> entries;
  return entries;
}

bool has_pending_singlepass_key(const PrefixCache* cache, const XXH3Key& key) {
  std::lock_guard<std::mutex> lock(pending_singlepass_mutex());
  FixedStringKeyEqual eq;
  for (const auto& entry : pending_singlepass_entries()) {
    if (entry.prefix_cache == cache && eq(entry.key, key)) {
      return true;
    }
  }
  return false;
}

void enqueue_pending_singlepass_entry(PrefixCache* cache,
                                      const XXH3Key& key,
                                      Block&& ckpt_block) {
  std::lock_guard<std::mutex> lock(pending_singlepass_mutex());
  pending_singlepass_entries().push_back(
      PendingSinglepassGdnEntry{cache, key, std::move(ckpt_block)});
}

}  // namespace

namespace musa {

void flush_pending_gdn_singlepass_checkpoints() {
  std::vector<PendingSinglepassGdnEntry> to_flush;
  {
    std::lock_guard<std::mutex> lock(pending_singlepass_mutex());
    if (pending_singlepass_entries().empty()) {
      return;
    }
    to_flush.swap(pending_singlepass_entries());
  }
  for (auto& entry : to_flush) {
    if (entry.prefix_cache == nullptr || !entry.ckpt_block.is_valid()) {
      continue;
    }
    if (entry.prefix_cache->find(entry.key).is_valid()) {
      continue;
    }
    std::vector<Block> single;
    single.emplace_back(std::move(entry.ckpt_block));
    entry.prefix_cache->insert(single);
  }
}

}  // namespace musa

bool last_decode_step_crosses_checkpoint(const Sequence& sequence,
                                         size_t checkpoint_stride) {
  return last_decode_step_checkpoint_row(sequence, checkpoint_stride)
      .has_value();
}

std::optional<size_t> last_decode_step_checkpoint_row(
    const Sequence& sequence,
    size_t checkpoint_stride) {
  CHECK_GT(checkpoint_stride, 0u);
  const size_t confirmed_tokens = sequence.last_confirmed_cached_tokens_num();
  const size_t accepted_length = sequence.accepted_length();
  // accepted_length is the maximum accepted draft offset. The preceding bonus
  // token also advances the recurrent cache, so the confirmed span is one
  // larger than that offset.
  if (confirmed_tokens <= accepted_length) {
    return std::nullopt;
  }
  const size_t confirmed_span = accepted_length + 1;
  const size_t begin = confirmed_tokens - confirmed_span;
  if (!crosses_checkpoint(begin, confirmed_tokens, checkpoint_stride)) {
    return std::nullopt;
  }
  const size_t checkpoint_tokens =
      (confirmed_tokens / checkpoint_stride) * checkpoint_stride;
  CHECK_GT(checkpoint_tokens, begin);
  return checkpoint_tokens - begin - 1;
}

LinearStateBlockManager::LinearStateBlockManager(
    uint32_t num_slots,
    int32_t chunk_stride,
    bool enable_prefix_cache,
    bool instance_is_decode,
    uint32_t num_speculative_tokens,
    bool preserve_decode_checkpoint,
    uint32_t num_ckpt_slots)
    : BlockManagerImpl(make_linear_state_options(num_slots,
                                                 chunk_stride,
                                                 enable_prefix_cache,
                                                 instance_is_decode,
                                                 num_speculative_tokens)),
      preserve_decode_checkpoint_(preserve_decode_checkpoint),
      num_active_slots_(num_slots),
      num_ckpt_slots_(enable_prefix_cache ? num_ckpt_slots : 0),
      num_host_ckpt_slots_(
          (enable_prefix_cache && !instance_is_decode && num_ckpt_slots > 0)
              ? musa::resolve_gdn_host_ckpt_slots(num_ckpt_slots)
              : 0u) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(chunk_stride, 0)
      << "linear-state leaf needs a positive chunk stride";
  if (num_ckpt_slots_ > 0) {
    ckpt_free_ids_.reserve(num_ckpt_slots_);
    for (uint32_t i = num_ckpt_slots_; i > 0; --i) {
      ckpt_free_ids_.push_back(
          static_cast<int32_t>(num_active_slots_ + i - 1));
    }
    hbm_ckpt_keys_.resize(num_ckpt_slots_);
    hbm_ckpt_has_key_.assign(num_ckpt_slots_, 0);
  }
  if (num_host_ckpt_slots_ > 0) {
    host_ckpt_prefix_cache_ = std::make_unique<LinearStatePrefixCache>(
        static_cast<uint32_t>(chunk_stride), options_.hasher_type());
    host_ckpt_free_ids_.reserve(num_host_ckpt_slots_);
    const uint32_t host_base = num_active_slots_ + num_ckpt_slots_;
    for (uint32_t i = num_host_ckpt_slots_; i > 0; --i) {
      host_ckpt_free_ids_.push_back(static_cast<int32_t>(host_base + i - 1));
    }
  }
}

LinearStateBlockManager::~LinearStateBlockManager() {
  musa::flush_pending_gdn_singlepass_checkpoints();
  host_ckpt_prefix_cache_.reset();
  prefix_cache_.reset();
}

void LinearStateBlockManager::reset_prefix_cache() {
  if (host_ckpt_prefix_cache_ != nullptr) {
    host_ckpt_prefix_cache_->evict(host_ckpt_prefix_cache_->num_blocks());
  }
  BlockManagerImpl::reset_prefix_cache();
}

bool LinearStateBlockManager::has_checkpoint_for_sequence_tokens(
    Sequence& seq,
    size_t tokens) {
  if (ckpt_backing_manager_ != nullptr) {
    return ckpt_backing_manager_->has_checkpoint_for_sequence_tokens(
        seq, tokens);
  }
  if (tokens == 0 || block_size() == 0 || (tokens % block_size()) != 0) {
    return false;
  }
  const size_t ckpt_idx = tokens / block_size() - 1;
  seq.update_linear_state_hashes(static_cast<uint32_t>(block_size()));
  const Slice<XXH3Key> hashes = seq.linear_state_hashes();
  if (ckpt_idx >= hashes.size()) {
    return false;
  }
  return find_any_checkpoint(hashes[ckpt_idx]).is_valid();
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               KVCacheState& kv_state,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (seq->is_prefill_stage()) {
    if (!options_.instance_is_decode()) {
      return allocate_prefill(seq, kv_state);
    }
    retain_read_source(kv_state);
    return allocate_decode(seq, kv_state);
  }
  return allocate_decode(seq, kv_state);
}

bool LinearStateBlockManager::allocate_for_prefetch(Sequence* seq,
                                                    size_t num_tokens) {
  if (seq == nullptr || block_size() == 0) {
    return false;
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t target_blocks = num_tokens / block_size();
  const size_t cached_blocks = host_state.num_cached_blocks(BlockType::LINEAR);
  const size_t shared_blocks = host_state.shared_blocks_num(BlockType::LINEAR);
  const size_t cursor = std::min(cached_blocks, target_blocks);
  const bool source_in_target =
      cached_blocks > 0 && cached_blocks <= target_blocks;

  std::vector<Block> old_blocks = host_state.take_blocks(BlockType::LINEAR);
  // The vector is indexed by logical checkpoint, while only the deepest
  // matched checkpoint has a physical source after prefix-cache admission.
  // Keep the earlier positions as invalid placeholders so Store transfers and
  // restore selection continue to use absolute checkpoint indices.
  std::vector<Block> blocks(target_blocks);
  Block source;
  if (source_in_target) {
    for (size_t index = std::min(cached_blocks, old_blocks.size()); index > 0;
         --index) {
      Block& candidate = old_blocks[index - 1];
      if (candidate.is_valid()) {
        source = std::move(candidate);
        break;
      }
    }
  }
  std::vector<Block> dropped;
  dropped.reserve(old_blocks.size());
  for (size_t index = 0; index < old_blocks.size(); ++index) {
    Block& block = old_blocks[index];
    if (!block.is_valid()) {
      continue;
    }
    if (index >= cursor && index < target_blocks) {
      blocks[index] = std::move(block);
    } else {
      dropped.emplace_back(std::move(block));
    }
  }
  if (!dropped.empty()) {
    deallocate(std::move(dropped));
  }

  const bool has_source = source_in_target && source.is_valid();
  if (has_source) {
    blocks[cached_blocks - 1] = std::move(source);
  }

  seq->update_linear_state_hashes(static_cast<uint32_t>(block_size()));
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), target_blocks);

  std::vector<size_t> missing;
  missing.reserve(target_blocks > cursor ? target_blocks - cursor : 0);
  for (size_t index = cursor; index < target_blocks; ++index) {
    if (!blocks[index].is_valid()) {
      missing.emplace_back(index);
    }
  }

  size_t allocatable = std::min(
      missing.size(), num_free_blocks() + num_blocks_in_prefix_cache());
  std::vector<Block> allocated = allocate(allocatable);
  if (allocated.empty() && allocatable > 0) {
    allocatable = std::min(missing.size(), num_free_blocks());
    allocated = allocate(allocatable);
  }
  for (size_t index = 0; index < allocated.size(); ++index) {
    const size_t block_index = missing[index];
    allocated[index].set_hash_value(hashes[block_index].data);
    blocks[block_index] = std::move(allocated[index]);
  }

  if (!blocks.empty()) {
    host_state.replace_composite_blocks(
        BlockType::LINEAR,
        std::move(blocks),
        has_source ? std::min(shared_blocks, target_blocks) : 0,
        cursor);
  }
  return allocated.size() == missing.size();
}

void LinearStateBlockManager::trim_prefetch_blocks(Sequence* seq,
                                                   size_t max_hit_tokens) {
  if (seq == nullptr || block_size() == 0) {
    return;
  }

  KVCacheState& host_state = seq->host_kv_state();
  std::vector<Block> blocks = host_state.take_blocks(BlockType::LINEAR);
  const size_t keep = std::min(max_hit_tokens / block_size(), blocks.size());

  // A Linear checkpoint vector is a logical index space. Keep only the
  // deepest valid checkpoint in the accepted gate range; all other physical
  // aliases, including valid blocks after the range, are temporary Store
  // state and must be released by this leaf.
  size_t source_index = keep;
  for (size_t index = keep; index > 0; --index) {
    if (blocks[index - 1].is_valid()) {
      source_index = index - 1;
      break;
    }
  }

  std::vector<Block> to_release;
  to_release.reserve(blocks.size());
  Block source;
  if (source_index < keep) {
    source = blocks[source_index];
  }
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (!blocks[index].is_valid() || index == source_index) {
      continue;
    }
    to_release.emplace_back(std::move(blocks[index]));
  }
  if (!to_release.empty()) {
    deallocate(std::move(to_release));
  }

  if (!source.is_valid()) {
    host_state.erase_blocks(BlockType::LINEAR);
    return;
  }

  // LinearStatePrefixCache needs the absolute checkpoint index to derive the
  // chained key. Publish through a sparse logical vector, then collapse the
  // sequence-owned view to the single restore source required by Linear
  // runtime allocation.
  seq->update_linear_state_hashes(static_cast<uint32_t>(block_size()));
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GT(hashes.size(), source_index);
  std::vector<Block> publish(source_index + 1);
  publish[source_index] = source;
  BlockManagerImpl::cache(seq->hash_tokens(options_.hasher_type()),
                          publish,
                          /*existed_shared_blocks_num=*/0,
                          seq->mm_data(),
                          hashes);

  std::vector<Block> retained(1);
  retained[0] = std::move(source);
  host_state.replace_composite_blocks(BlockType::LINEAR,
                                      std::move(retained),
                                      /*num_shared_blocks=*/1,
                                      /*cache_publish_cursor=*/1);
}

void LinearStateBlockManager::retain_read_source(KVCacheState& kv_state) {
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  if (blocks->empty()) {
    return;
  }
  CHECK(blocks->back().is_valid());
  const bool shared_source =
      kv_state.shared_blocks_num(BlockType::LINEAR) == blocks->size();
  const size_t keep_begin = blocks->size() - 1;
  std::vector<Block> to_release;
  to_release.reserve(keep_begin);
  for (size_t i = 0; i < keep_begin; ++i) {
    to_release.emplace_back(std::move((*blocks)[i]));
  }
  deallocate(std::move(to_release));
  blocks->erase(blocks->begin(), blocks->begin() + keep_begin);
  kv_state.set_shared_blocks_num(BlockType::LINEAR, shared_source ? 1 : 0);
}

void LinearStateBlockManager::cache_read_source(Sequence* seq,
                                                KVCacheState& kv_state) {
  const Slice<Block> blocks = kv_state.blocks(BlockType::LINEAR);
  if (seq->is_graph_warmup() || prefix_cache_ == nullptr || blocks.empty()) {
    return;
  }
  const size_t chunk_stride = block_size();
  const size_t cached_tokens = seq->kv_cache_tokens_num();
  const size_t total_tokens = seq->num_tokens();
  if (!should_publish_linear_checkpoint(
          *seq, cached_tokens, total_tokens, chunk_stride)) {
    return;
  }
  const size_t checkpoint_index = cached_tokens / chunk_stride;
  if (checkpoint_index <= kv_state.num_cached_blocks(BlockType::LINEAR)) {
    return;
  }
  seq->update_linear_state_hashes(block_size());
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), checkpoint_index);
  auto* source = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK(source->back().is_valid());
  source->back().set_hash_value(hashes[checkpoint_index - 1].data);
  prefix_cache_->insert(
      Slice<Block>(*source).slice(source->size() - 1, source->size()));
  kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_prefill(
    Sequence* seq,
    KVCacheState& kv_state) {
  // If the sequence's effective restore boundary is ahead of this tier's
  // cursor, the existing slot belongs to a shorter cache hit. Drop it without
  // inserting into this tier's prefix cache, then build the two-slot
  // [restore, live] window. The rule applies identically to HBM and Host
  // states; the caller identifies the tier by the KVCacheState reference.
  if (seq->kv_cache_tokens_num() > kv_state.kv_cache_tokens_num()) {
    if (num_ckpt_slots_ > 0 && ckpt_backing_manager_ == nullptr) {
      const size_t target_tokens = seq->kv_cache_tokens_num();
      if (target_tokens > 0 && block_size() > 0 &&
          (target_tokens % block_size()) == 0) {
        const size_t checkpoint_index = target_tokens / block_size() - 1;
        seq->update_linear_state_hashes(static_cast<uint32_t>(block_size()));
        const Slice<XXH3Key> hashes = seq->linear_state_hashes();
        if (checkpoint_index < hashes.size()) {
          Block ckpt_src = find_any_checkpoint(hashes[checkpoint_index]);
          if (ckpt_src.is_valid()) {
            std::vector<Block> old_blocks =
                kv_state.take_blocks(BlockType::LINEAR);
            deallocate(std::move(old_blocks));
            std::vector<Block> live = BlockManagerImpl::allocate(1);
            if (live.empty()) {
              return std::nullopt;
            }
            std::vector<Block> shared_src;
            shared_src.emplace_back(std::move(ckpt_src));
            kv_state.replace_composite_blocks(
                BlockType::LINEAR,
                std::move(shared_src),
                /*num_shared_blocks=*/1,
                /*cache_publish_cursor=*/1);
            kv_state.set_num_cached_blocks(BlockType::LINEAR,
                                           checkpoint_index + 1);
            return live;
          }
        }
      }
    }
    std::vector<Block> old_blocks = kv_state.take_blocks(BlockType::LINEAR);
    deallocate(std::move(old_blocks));
    std::vector<Block> allocated = BlockManagerImpl::allocate(2);
    if (allocated.size() != 2) {
      deallocate(std::move(allocated));
      return std::nullopt;
    }
    return allocated;
  }

  if (num_ckpt_slots_ == 0 && ckpt_backing_manager_ == nullptr) {
    cache_read_source(seq, kv_state);
  }
  retain_read_source(kv_state);
  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    return std::nullopt;
  }
  return allocated;
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_decode(
    Sequence* seq,
    KVCacheState& kv_state) {
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK_LE(blocks->size(), 2u);
  if (seq->is_prefill_stage()) {
    if (!blocks->empty()) {
      return std::vector<Block>{};
    }
    std::vector<Block> allocated = BlockManagerImpl::allocate(1);
    return allocated.empty()
               ? std::nullopt
               : std::optional<std::vector<Block>>(std::move(allocated));
  }

  const size_t confirmed_tokens = seq->last_confirmed_cached_tokens_num();
  const size_t tokens_per_step =
      static_cast<size_t>(options_.num_speculative_tokens()) + 1;
  const size_t checkpoint_stride = block_size();
  const bool completed_checkpoint =
      blocks->size() == 2 &&
      last_decode_step_crosses_checkpoint(*seq, checkpoint_stride);

  if (preserve_decode_checkpoint_ && completed_checkpoint) {
    // HierarchyBlockManagerPool owns the transfer boundary. It moves the
    // HBM and Host front blocks into an OffloadBlockPair after both tiers
    // preserve the same logical checkpoint. Hashing happens once at that
    // ownership boundary.
    return std::vector<Block>{};
  }

  // Prefill may leave its read source in front of the newest write block.
  // Decode starts from the newest state, so discard that historical prefill
  // source before applying the decode rolling rules.  In particular, do not
  // treat the prefill pair as a pending decode window.
  if (blocks->size() == 2 &&
      kv_state.kv_cache_tokens_num() <= seq->num_prompt_tokens()) {
    retain_read_source(kv_state);
  }

  if (blocks->size() == 2) {
    if (completed_checkpoint) {
      std::vector<Block> to_release;
      to_release.emplace_back(std::move((*blocks)[0]));
      deallocate(std::move(to_release));
      blocks->erase(blocks->begin());
      kv_state.set_shared_blocks_num(BlockType::LINEAR, 0);
    } else {
      // The previous decode step has not yet confirmed a checkpoint. Keep
      // both slots alive and alternate their roles so the next out-of-place
      // step reads the slot written by the preceding step. This is also
      // what makes overlap allocation safe: the old read slot cannot be
      // returned to the global pool while its in-flight step may still use it.
      std::iter_swap(blocks->begin(), blocks->begin() + 1);
      return std::vector<Block>{};
    }
  }

  if (blocks->empty()) {
    std::vector<Block> allocated = BlockManagerImpl::allocate(1);
    return allocated.empty()
               ? std::nullopt
               : std::optional<std::vector<Block>>(std::move(allocated));
  }

  // A stride of one is the legacy fallback when chunked prefill is disabled.
  // It does not describe a persistent LINEAR checkpoint window to roll.
  if (checkpoint_stride == 1) {
    return std::vector<Block>{};
  }

  CHECK_LE(tokens_per_step,
           std::numeric_limits<size_t>::max() - confirmed_tokens)
      << "linear-state confirmed-token cursor overflow";
  const size_t predicted_tokens = confirmed_tokens + tokens_per_step;
  if (!crosses_checkpoint(
          confirmed_tokens, predicted_tokens, checkpoint_stride)) {
    return std::vector<Block>{};
  }

  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    // The extra slot only preserves a checkpoint for the future offload path;
    // recurrent decode itself supports reading and writing the live slot in
    // place. Under pressure, keep decoding instead of turning an optional
    // checkpoint copy into a sequence preemption.
    return std::vector<Block>{};
  }
  return allocated;
}

Block LinearStateBlockManager::allocate() {
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  if (blocks.empty()) {
    return Block();
  }
  return std::move(blocks[0]);
}

void LinearStateBlockManager::free(int32_t block_id) {
  if (ckpt_backing_wrapper_ != nullptr &&
      block_id >= static_cast<int32_t>(num_active_slots_)) {
    ckpt_backing_wrapper_->free(block_id);
    return;
  }
  if (num_ckpt_slots_ > 0 &&
      block_id >= static_cast<int32_t>(num_active_slots_)) {
    std::lock_guard<std::mutex> lock(ckpt_mutex_);
    if (block_id <
        static_cast<int32_t>(num_active_slots_ + num_ckpt_slots_)) {
      ckpt_free_ids_.push_back(block_id);
    } else {
      host_ckpt_free_ids_.push_back(block_id);
    }
    return;
  }
  BlockManagerImpl::free(block_id);
}

void LinearStateBlockManager::deallocate(std::vector<Block>&& blocks) {
  if (num_ckpt_slots_ == 0 && ckpt_backing_manager_ == nullptr) {
    BlockManagerImpl::deallocate(std::move(blocks));
    return;
  }
  std::vector<Block> active_blocks;
  active_blocks.reserve(blocks.size());
  for (auto& block : blocks) {
    if (!block.is_valid()) {
      continue;
    }
    if (block.id() < static_cast<int32_t>(num_active_slots_)) {
      active_blocks.emplace_back(std::move(block));
    }
  }
  blocks.clear();
  if (!active_blocks.empty()) {
    BlockManagerImpl::deallocate(std::move(active_blocks));
  }
}

int32_t LinearStateBlockManager::alloc_ckpt_slot_locked() {
  std::lock_guard<std::mutex> lock(ckpt_mutex_);
  if (ckpt_free_ids_.empty()) {
    return -1;
  }
  const int32_t id = ckpt_free_ids_.back();
  ckpt_free_ids_.pop_back();
  return id;
}

int32_t LinearStateBlockManager::alloc_host_ckpt_slot_locked() {
  std::lock_guard<std::mutex> lock(ckpt_mutex_);
  if (host_ckpt_free_ids_.empty()) {
    return -1;
  }
  const int32_t id = host_ckpt_free_ids_.back();
  host_ckpt_free_ids_.pop_back();
  return id;
}

void LinearStateBlockManager::record_hbm_ckpt_key(int32_t hbm_ckpt_id,
                                                  const XXH3Key& key) {
  const int32_t idx = hbm_ckpt_id - static_cast<int32_t>(num_active_slots_);
  if (idx >= 0 && static_cast<size_t>(idx) < hbm_ckpt_keys_.size()) {
    hbm_ckpt_keys_[static_cast<size_t>(idx)] = key;
    hbm_ckpt_has_key_[static_cast<size_t>(idx)] = 1;
  }
}

Block LinearStateBlockManager::find_any_checkpoint(const XXH3Key& key) {
  if (prefix_cache_ != nullptr) {
    Block b = prefix_cache_->find(key);
    if (b.is_valid()) {
      return b;
    }
  }
  if (host_ckpt_prefix_cache_ != nullptr) {
    Block b = host_ckpt_prefix_cache_->find(key);
    if (b.is_valid()) {
      return b;
    }
  }
  return Block();
}

bool LinearStateBlockManager::has_any_checkpoint_or_pending(
    const XXH3Key& key) {
  if (find_any_checkpoint(key).is_valid()) {
    return true;
  }
  if (prefix_cache_ != nullptr &&
      has_pending_singlepass_key(prefix_cache_.get(), key)) {
    return true;
  }
  if (host_ckpt_prefix_cache_ != nullptr &&
      has_pending_singlepass_key(host_ckpt_prefix_cache_.get(), key)) {
    return true;
  }
  return false;
}

int32_t LinearStateBlockManager::acquire_hbm_ckpt_slot_with_demote(
    std::vector<BlockTransferInfo>* swap_transfers) {
  int32_t ckpt_id = alloc_ckpt_slot_locked();
  if (ckpt_id >= 0) {
    return ckpt_id;
  }
  if (prefix_cache_ == nullptr || prefix_cache_->evict(1) == 0) {
    return -1;
  }
  ckpt_id = alloc_ckpt_slot_locked();
  if (ckpt_id < 0) {
    return -1;
  }
  const int32_t hbm_idx =
      ckpt_id - static_cast<int32_t>(num_active_slots_);
  if (swap_transfers != nullptr && num_host_ckpt_slots_ > 0 &&
      host_ckpt_prefix_cache_ != nullptr && hbm_idx >= 0 &&
      static_cast<size_t>(hbm_idx) < hbm_ckpt_has_key_.size() &&
      hbm_ckpt_has_key_[static_cast<size_t>(hbm_idx)] != 0) {
    const XXH3Key victim_key = hbm_ckpt_keys_[static_cast<size_t>(hbm_idx)];
    if (!host_ckpt_prefix_cache_->contains(victim_key) &&
        !has_pending_singlepass_key(host_ckpt_prefix_cache_.get(),
                                    victim_key)) {
      int32_t host_id = alloc_host_ckpt_slot_locked();
      if (host_id < 0) {
        host_ckpt_prefix_cache_->evict(1);
        host_id = alloc_host_ckpt_slot_locked();
      }
      if (host_id >= 0) {
        BlockTransferInfo demote_tx(-1, -1);
        if (musa::encode_deepseek_v4_gdn_host_demote(
                ckpt_id, host_id, &demote_tx)) {
          swap_transfers->emplace_back(std::move(demote_tx));
          Block host_block(host_id, this);
          host_block.set_hash_value(victim_key.data);
          enqueue_pending_singlepass_entry(
              host_ckpt_prefix_cache_.get(),
              victim_key,
              std::move(host_block));
        } else {
          free(host_id);
        }
      }
    }
  }
  return ckpt_id;
}

void LinearStateBlockManager::publish_linear_checkpoint(
    Sequence* seq,
    KVCacheState& kv_state,
    std::vector<BlockTransferInfo>* swap_transfers) {
  if (num_ckpt_slots_ == 0 || seq == nullptr || seq->is_graph_warmup() ||
      prefix_cache_ == nullptr || swap_transfers == nullptr ||
      options_.instance_is_decode()) {
    return;
  }
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  if (blocks == nullptr || blocks->empty()) {
    return;
  }
  if (singlepass_gdn_ckpt_enabled()) {
    Block& live_block = blocks->back();
    if (!live_block.is_valid() || live_block.id() <= 0 ||
        live_block.id() >= static_cast<int32_t>(num_active_slots_)) {
      return;
    }
    const size_t chunk_stride = block_size();
    const size_t start_pos = seq->kv_cache_tokens_num();
    const size_t total_tokens = seq->num_tokens();
    if (chunk_stride == 0 || total_tokens <= chunk_stride ||
        start_pos >= total_tokens - 1) {
      return;
    }
    const size_t step_end_tokens =
        std::min<size_t>(total_tokens, kv_state.current_max_tokens_capacity());
    const size_t max_ckpt_pos =
        std::min<size_t>(step_end_tokens, total_tokens - 1);
    if (max_ckpt_pos <= start_pos || max_ckpt_pos < chunk_stride) {
      return;
    }

    const size_t initial_hit = seq->num_prefix_cache_tokens();
    const bool started_cold = (initial_hit == 0);
    const size_t warm_min_advance = std::max<size_t>(
        chunk_stride,
        read_env_size_t("XLLM_GDN_SINGLEPASS_WARM_MIN_ADVANCE", 512));
    const size_t cold_stride_raw =
        read_env_size_t("XLLM_GDN_SINGLEPASS_COLD_STRIDE", 2048);
    const size_t cold_stride =
        cold_stride_raw > 0
            ? ((cold_stride_raw + chunk_stride - 1) / chunk_stride) *
                  chunk_stride
            : 0;

    const ChatLinearCheckpointBounds bounds =
        find_chat_linear_checkpoint_bounds(
            seq->tokens(), total_tokens, chunk_stride);

    std::vector<size_t> candidate_positions;
    candidate_positions.reserve(8);

    auto near_chat_boundary = [&](size_t pos) -> bool {
      if (!bounds.has_chat_boundaries) {
        return false;
      }
      if (bounds.hist_ckpt > 0) {
        const size_t diff =
            pos > bounds.hist_ckpt ? pos - bounds.hist_ckpt
                                   : bounds.hist_ckpt - pos;
        if (diff < 256) {
          return true;
        }
      }
      if (bounds.end_ckpt > 0) {
        const size_t diff =
            pos > bounds.end_ckpt ? pos - bounds.end_ckpt
                                  : bounds.end_ckpt - pos;
        if (diff < 256) {
          return true;
        }
      }
      return false;
    };

    if (cold_stride > 0 &&
        (started_cold || (step_end_tokens - start_pos) >= 4096)) {
      const size_t first_mult =
          ((start_pos / cold_stride) + 1) * cold_stride;
      for (size_t p = first_mult; p <= max_ckpt_pos; p += cold_stride) {
        if (p == step_end_tokens || !near_chat_boundary(p)) {
          candidate_positions.push_back(p);
        }
      }
    }

    if (step_end_tokens <= max_ckpt_pos &&
        (step_end_tokens % chunk_stride) == 0 &&
        step_end_tokens >= start_pos + 256) {
      candidate_positions.push_back(step_end_tokens);
    }

    if (bounds.has_chat_boundaries) {
      if (bounds.hist_ckpt > start_pos && bounds.hist_ckpt <= max_ckpt_pos) {
        const size_t req_adv =
            (started_cold || bounds.hist_ckpt == step_end_tokens)
                ? std::max<size_t>(chunk_stride * 4, 256)
                : warm_min_advance;
        if (bounds.hist_ckpt >= start_pos + req_adv) {
          candidate_positions.push_back(bounds.hist_ckpt);
        }
      }
      if (bounds.end_ckpt > start_pos && bounds.end_ckpt <= max_ckpt_pos) {
        if (started_cold || bounds.hist_ckpt > initial_hit + 192) {
          const size_t base_pos = std::max<size_t>(start_pos, bounds.hist_ckpt);
          if (bounds.end_ckpt >= base_pos + 512) {
            candidate_positions.push_back(bounds.end_ckpt);
          }
        } else {
          const size_t req_adv =
              (bounds.end_ckpt == step_end_tokens)
                  ? std::max<size_t>(chunk_stride * 4, 256)
                  : warm_min_advance;
          if (bounds.end_ckpt >= start_pos + req_adv) {
            candidate_positions.push_back(bounds.end_ckpt);
          }
        }
      }
    } else {
      const size_t fallback_stride = std::max<size_t>(chunk_stride, 2048);
      const size_t fallback_ckpt =
          ((total_tokens - 1) / fallback_stride) * fallback_stride;
      const size_t req_adv =
          (started_cold || fallback_ckpt == step_end_tokens)
              ? std::max<size_t>(chunk_stride * 4, 256)
              : warm_min_advance;
      if (fallback_ckpt > start_pos && fallback_ckpt <= max_ckpt_pos &&
          fallback_ckpt >= start_pos + req_adv) {
        candidate_positions.push_back(fallback_ckpt);
      }
    }

    if (candidate_positions.empty()) {
      return;
    }
    std::sort(candidate_positions.begin(), candidate_positions.end());
    candidate_positions.erase(
        std::unique(candidate_positions.begin(), candidate_positions.end()),
        candidate_positions.end());

    seq->update_linear_state_hashes(static_cast<uint32_t>(chunk_stride));
    const Slice<XXH3Key> hashes = seq->linear_state_hashes();

    for (size_t target_pos : candidate_positions) {
      if (target_pos <= start_pos || target_pos > max_ckpt_pos ||
          (target_pos % chunk_stride) != 0) {
        continue;
      }
      const size_t checkpoint_index = target_pos / chunk_stride;
      if (checkpoint_index == 0 || checkpoint_index > hashes.size()) {
        continue;
      }
      const XXH3Key& hash_key = hashes[checkpoint_index - 1];
      if (has_any_checkpoint_or_pending(hash_key)) {
        kv_state.set_num_cached_blocks(
            BlockType::LINEAR,
            std::max(kv_state.num_cached_blocks(BlockType::LINEAR),
                     checkpoint_index));
        continue;
      }
      int32_t ckpt_id = acquire_hbm_ckpt_slot_with_demote(swap_transfers);
      if (ckpt_id < 0) {
        break;
      }
      const int32_t offset_tokens =
          static_cast<int32_t>(target_pos - start_pos);
      const bool is_end_of_step = (target_pos == step_end_tokens);
      BlockTransferInfo transfer(-1, -1);
      if (!musa::encode_deepseek_v4_gdn_singlepass_checkpoint(
              live_block.id(),
              ckpt_id,
              offset_tokens,
              is_end_of_step,
              &transfer)) {
        free(ckpt_id);
        continue;
      }
      record_hbm_ckpt_key(ckpt_id, hash_key);
      Block ckpt_block(ckpt_id, this);
      ckpt_block.set_hash_value(hash_key.data);
      enqueue_pending_singlepass_entry(
          prefix_cache_.get(), hash_key, std::move(ckpt_block));
      kv_state.set_num_cached_blocks(
          BlockType::LINEAR,
          std::max(kv_state.num_cached_blocks(BlockType::LINEAR),
                   checkpoint_index));
      swap_transfers->emplace_back(std::move(transfer));
    }
    return;
  }

  if (blocks->size() < 2) {
    return;
  }
  Block& src_block = (*blocks)[blocks->size() - 2];
  if (!src_block.is_valid() ||
      src_block.id() >= static_cast<int32_t>(num_active_slots_)) {
    return;
  }
  const size_t chunk_stride = block_size();
  const size_t cached_tokens = seq->kv_cache_tokens_num();
  const size_t total_tokens = seq->num_tokens();
  if (!should_publish_linear_checkpoint(
          *seq, cached_tokens, total_tokens, chunk_stride)) {
    return;
  }
  const size_t checkpoint_index = cached_tokens / chunk_stride;
  if (checkpoint_index <= kv_state.num_cached_blocks(BlockType::LINEAR)) {
    return;
  }
  seq->update_linear_state_hashes(static_cast<uint32_t>(chunk_stride));
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), checkpoint_index);
  const XXH3Key& hash_key = hashes[checkpoint_index - 1];
  src_block.set_hash_value(hash_key.data);
  if (has_any_checkpoint_or_pending(hash_key)) {
    kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);
    return;
  }
  int32_t ckpt_id = acquire_hbm_ckpt_slot_with_demote(swap_transfers);
  if (ckpt_id < 0) {
    return;
  }
  record_hbm_ckpt_key(ckpt_id, hash_key);
  Block ckpt_block(ckpt_id, this);
  ckpt_block.set_hash_value(hash_key.data);
  std::vector<Block> to_insert;
  to_insert.emplace_back(std::move(ckpt_block));
  prefix_cache_->insert(to_insert);
  kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);

  BlockTransferInfo transfer(-1, -1);
  CHECK(musa::encode_deepseek_v4_prefix_block_copy(
      BlockType::LINEAR, src_block.id(), ckpt_id, &transfer));
  swap_transfers->emplace_back(std::move(transfer));
}

std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  if (ckpt_backing_wrapper_ != nullptr) {
    return ckpt_backing_wrapper_->allocate_shared(
        token_ids, existed_shared_blocks, mm_data, block_hashes);
  }
  if (token_ids.size() == 0 || prefix_cache_ == nullptr) {
    return {};
  }
  const Slice<int32_t> prefix_slice = token_ids.slice(0, token_ids.size() - 1);
  std::vector<Block> blocks =
      prefix_cache_->match(prefix_slice, {}, MMData(), block_hashes);
  if (host_ckpt_prefix_cache_ != nullptr) {
    std::vector<Block> l2_blocks = host_ckpt_prefix_cache_->match(
        prefix_slice, {}, MMData(), block_hashes);
    if (l2_blocks.size() > blocks.size()) {
      blocks.resize(l2_blocks.size());
    }
    std::vector<Block> dup_drop;
    for (size_t i = 0; i < l2_blocks.size(); ++i) {
      if (!l2_blocks[i].is_valid()) {
        continue;
      }
      if (!blocks[i].is_valid()) {
        blocks[i] = std::move(l2_blocks[i]);
      } else {
        dup_drop.emplace_back(std::move(l2_blocks[i]));
      }
    }
    if (!dup_drop.empty()) {
      deallocate(std::move(dup_drop));
    }
  }
  VLOG(1) << "[HostCache][LinearMatch] manager=" << this
          << " decode=" << options_.instance_is_decode()
          << " blocks=" << blocks.size();
  for (const Block& block : blocks) {
    if (block.is_valid() &&
        block.id() < static_cast<int32_t>(num_active_slots_) &&
        mark_used(&usage_accounted_ids_, block.id())) {
      num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return blocks;
}

void LinearStateBlockManager::cache(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    size_t existed_shared_blocks_num,
                                    const MMData& mm_data,
                                    const Slice<XXH3Key>& block_hashes) {
  if (blocks.empty() || prefix_cache_ == nullptr) {
    return;
  }
  const size_t publish_end =
      std::min(token_ids.size() / block_size(), blocks.size() - 1);
  VLOG(1) << "[HostCache][LinearCache] manager=" << this
          << " decode=" << options_.instance_is_decode()
          << " blocks=" << blocks.size() << " publish_end=" << publish_end;
  if (publish_end <= existed_shared_blocks_num) {
    return;
  }
  CHECK_GE(block_hashes.size(), publish_end);
  BlockManagerImpl::cache(token_ids.slice(0, publish_end * block_size()),
                          blocks,
                          existed_shared_blocks_num,
                          mm_data,
                          block_hashes);
}

}  // namespace xllm
