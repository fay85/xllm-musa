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

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/util/hash_util.h"
#include "mm_batch_data.h"
#include "mm_data.h"
#include "mm_input.h"

namespace xllm {

class MMInputGatherVisitor final : public MMInputItem::IVisitor {
 public:
  MMInputGatherVisitor() = default;

  bool visit(const MMInputItem& item) override;
  MMItemVec finish(std::vector<MMDataItem>& image_items,
                   std::vector<MMDataItem>& video_items,
                   std::vector<MMDataItem>& audio_items);

  uint32_t data_type_ = MMType::NONE;
  std::vector<uint32_t> item_types_;
  std::vector<torch::Tensor> images_;
  std::vector<torch::Tensor> videos_;
  std::vector<torch::Tensor> audios_;
  std::vector<VideoMetadata> video_metadata_;
  std::vector<AudioMetadata> audio_metadata_;
  std::vector<MMDataItem> image_embedding_items_;
};

class CollectItemTensorVisitor : public MMData::IItemVisitor {
 public:
  CollectItemTensorVisitor(
      std::unordered_map<MMKey, std::vector<torch::Tensor>>& data,
      const std::unordered_set<MMKey>& black_keys = {},
      const std::unordered_set<MMKey>& white_keys = {})
      : datas_(data), black_keys_(black_keys), white_keys_(white_keys) {};

  CollectItemTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                           const std::unordered_set<MMKey>& white_keys = {})
      : datas_(stub_), black_keys_(black_keys), white_keys_(white_keys) {};

  bool visit(MMDataItem& item) override;
  bool visit(const MMKey& key, MMValue& value) override;

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> stub_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>>& datas_;

  std::unordered_set<MMKey> black_keys_;
  std::unordered_set<MMKey> white_keys_;
};

class CollectMMDataTensorVisitor : public MMData::IVisitor {
 public:
  CollectMMDataTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                             const std::unordered_set<MMKey>& white_keys = {})
      : item_visitor_(datas_, black_keys, white_keys) {};

  bool visit(MMData& data) override;

 public:
  uint32_t type_ = MMType::NONE;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;

  CollectItemTensorVisitor item_visitor_;
};

class MMTokenNumVisitor final : public MMDataItem::IVisitor {
 public:
  explicit MMTokenNumVisitor(MMType type) : type_(type) {}

  bool visit(MMDataItem& item) override;

  const std::vector<int32_t>& token_nums() const { return token_nums_; }

 private:
  MMType type_;
  std::vector<int32_t> token_nums_;
};

class EncoderInputGatherVisitor : public MMDataItem::IVisitor {
 public:
  EncoderInputGatherVisitor() = default;

  bool visit(MMDataItem& item) override;
  bool finish(MMBatchData& mm_data);

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
  std::string filter_prefix_ = "embedding";
};

// Distributes the encoder output dict onto freshly-encoded items. Every
// "<modality>|<field>" key is per-item: a vector<Tensor> contributes its i-th
// element, a Tensor its i-th row; all keys of a modality advance in lockstep.
class EncoderOutputScatterVisitor : public MMDataItem::IVisitor {
 public:
  EncoderOutputScatterVisitor(const MMDict& data) : data_(data) {}

  bool visit(MMDataItem& data) override;
  bool finish() const;
 public:
  const MMDict& data_;
  int32_t image_idx = 0;
  int32_t video_idx = 0;
  int32_t audio_idx = 0;
};

class EncoderEmbeddingGatherVisitor final : public MMDataItem::IVisitor {
 public:
  EncoderEmbeddingGatherVisitor(const std::vector<int32_t>& seq_lens,
                                const std::vector<int32_t>& scheduled_seq_lens);
  bool visit(MMDataItem& data) override;
  bool finish(MMBatchData& mm_data);

 public:
  std::vector<int32_t> per_seq_context_lens_;
  std::vector<int32_t> per_seq_scheduled_lens_;
  std::vector<int32_t> per_seq_scheduled_offsets_;
  int32_t total_scheduled_tokens_ = 0;
  torch::Tensor image_mask_;
  torch::Tensor video_mask_;
  torch::Tensor audio_mask_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
};

class UpdateMMItemScheduleStateVisitor : public MMDataItem::IVisitor {
 public:
  UpdateMMItemScheduleStateVisitor(int32_t computed_token_num = 0,
                                   int32_t q_seq_len = 0,
                                   int32_t seq_idx = 0)
      : computed_token_num_(computed_token_num),
        q_seq_len_(q_seq_len),
        seq_idx_(seq_idx) {}

  bool visit(MMDataItem& item) override;

 public:
  std::vector<MMDataItem> mm_data_items_;
  uint32_t scheduled_type_ = MMType::NONE;
  int32_t computed_token_num_ = 0;
  int32_t q_seq_len_ = 0;
  int32_t seq_idx_ = 0;
};

class EncoderCache;
class MMSideOutputCache;
class ProcessorCache;

class ProcessorCacheLookupVisitor final : public MMInputItem::IVisitor {
 public:
  ProcessorCacheLookupVisitor(ProcessorCache& cache, size_t item_count);
  bool visit(const MMInputItem& input) override;

  std::vector<std::optional<MMDataItem>> cache_hits_;
  std::vector<MMInputItem> miss_inputs_;

 private:
  ProcessorCache& cache_;
};

class ProcessorCacheInsertVisitor final : public MMDataItem::IVisitor {
 public:
  explicit ProcessorCacheInsertVisitor(ProcessorCache& cache);
  bool visit(MMDataItem& item) override;

 private:
  ProcessorCache& cache_;
};

// Item entries through which the engine instructs the encoder caches of its
// workers. Only the key names carry meaning, and workers remove them before
// encoding. Every planned item carries exactly one of the store, reuse and
// skip entries; an evict entry names, in hex, an output to drop.
inline constexpr char kEncoderCacheCommandPrefix[] = "encoder_cache|";
inline constexpr char kEncoderCacheStoreKey[] = "encoder_cache|store";
inline constexpr char kEncoderCacheReuseKey[] = "encoder_cache|reuse";
inline constexpr char kEncoderCacheSkipKey[] = "encoder_cache|skip";
inline constexpr char kEncoderCacheEvictPrefix[] = "encoder_cache|evict|";

// Engine-side record of the encoder outputs its workers cache. Workers store
// the output of every item it marks for storing and drop outputs only when it
// evicts them, so the record is exact and an item whose output is cached
// reaches the workers without its encoder inputs.
class EncoderCacheDirectory final {
 public:
  EncoderCacheDirectory(int64_t max_bytes, int64_t bytes_per_token);

  // Plans one step: items whose output the workers hold reuse it, the others
  // are stored, evicting the least recently used outputs the step does not
  // reference.
  void plan(MMBatchData& batch);

 private:
  using KeySet =
      std::unordered_set<XXH3Key, FixedStringKeyHash, FixedStringKeyEqual>;
  using LruList = std::list<XXH3Key>;

  struct Entry final {
    int64_t size = 0;
    LruList::iterator lru_it;
  };

  using EntryMap = std::
      unordered_map<XXH3Key, Entry, FixedStringKeyHash, FixedStringKeyEqual>;

  // Records `key` unless it cannot fit beside the outputs of this step, and
  // appends the outputs it evicts to `evicted`.
  bool admit(const XXH3Key& key,
             int64_t size,
             const KeySet& referenced,
             std::vector<XXH3Key>& evicted);

  int64_t max_bytes_ = 0;
  int64_t bytes_per_token_ = 0;
  int64_t current_bytes_ = 0;
  LruList lru_keys_;
  EntryMap entries_;
};

// Applies the instructions of an engine-managed encoder cache and removes
// them from the items.
class EncoderCacheCommandVisitor final : public MMDataItem::IVisitor {
 public:
  explicit EncoderCacheCommandVisitor(EncoderCache* cache);
  bool visit(MMDataItem& item) override;

  // Whether the engine relies on the cache holding the output of `key`.
  bool reuses(const XXH3Key& key) const;
  // Whether the output of `key` must stay out of the cache.
  bool skips(const XXH3Key& key) const;

 private:
  using KeySet =
      std::unordered_set<XXH3Key, FixedStringKeyHash, FixedStringKeyEqual>;

  EncoderCache* cache_;
  KeySet reused_keys_;
  KeySet skipped_keys_;
};

class EncoderCacheLookupVisitor : public MMDataItem::IVisitor {
 public:
  EncoderCacheLookupVisitor(EncoderCache* cache,
                            const EncoderCacheCommandVisitor& commands);
  bool visit(MMDataItem& item) override;

 private:
  EncoderCache* cache_;
  const EncoderCacheCommandVisitor& commands_;
};

class EncoderCacheInsertVisitor : public MMDataItem::IVisitor {
 public:
  EncoderCacheInsertVisitor(EncoderCache* cache,
                            const EncoderCacheCommandVisitor& commands);
  bool visit(MMDataItem& item) override;

 private:
  EncoderCache* cache_;
  const EncoderCacheCommandVisitor& commands_;
};

// Restores missing CTC tensors or caches newly computed outputs on each item.
class MMSideOutputCacheVisitor final : public MMData::IVisitor,
                                       public MMDataItem::IVisitor {
 public:
  explicit MMSideOutputCacheVisitor(MMSideOutputCache& cache);
  bool visit(MMData& data) override;
  bool visit(MMDataItem& item) override;

 private:
  MMSideOutputCache& cache_;
};

// Collects CTC tensors by sequence index, preserving full-batch empty slots.
class MMSideOutputCollectVisitor final : public MMData::IVisitor,
                          public MMDataItem::IVisitor {
 public:
  explicit MMSideOutputCollectVisitor(int32_t num_sequences);
  bool visit(MMData& data) override;
  bool visit(MMDataItem& item) override;
  std::vector<MMDict> finish();

 private:
  std::vector<MMDict> outputs_;
};

}  // namespace xllm
