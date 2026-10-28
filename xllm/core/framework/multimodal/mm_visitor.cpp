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

#include "mm_visitor.h"

#include <absl/strings/escaping.h>
#include <absl/strings/match.h>
#include <absl/strings/string_view.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "common/metrics.h"
#include "framework/encoder_cache/encoder_cache.h"
#include "processors/processor_cache.h"

namespace xllm {

namespace {
MMDict build_embedding_data(MMType type, const EmbeddingOutput& embedding) {
  MMDict data;
  data[get_embedding_key(type)] = embedding.embedding;
  for (const auto& [key, value] : embedding.metadata) {
    data[key] = value;
  }
  return data;
}

std::pair<int32_t, int32_t> compute_emb_range(int32_t start_pos,
                                              int32_t end_pos,
                                              const torch::Tensor& mask) {
  if (!mask.defined() || mask.numel() == 0) {
    return {start_pos, end_pos};
  }
  auto mask_cpu = mask.to(torch::kCPU);
  auto upto_start_pos =
      mask_cpu.slice(/*dim*/ 0, /*start*/ 0, /*end*/ start_pos)
          .sum()
          .item<int32_t>();
  auto upto_end_pos = mask_cpu.slice(/*dim*/ 0, /*start*/ 0, /*end*/ end_pos)
                          .sum()
                          .item<int32_t>();
  return {upto_start_pos, upto_end_pos};
}

std::vector<int32_t> normalize_to_per_seq_lens(
    const std::vector<int32_t>& seq_lens) {
#if defined(USE_NPU)
  return seq_lens;
#else
  // Other backends pass cumulative sequence lengths.
  std::vector<int32_t> per_seq_lens;
  per_seq_lens.reserve(seq_lens.empty() ? 0 : seq_lens.size() - 1);
  for (size_t i = 1; i < seq_lens.size(); ++i) {
    per_seq_lens.push_back(seq_lens[i] - seq_lens[i - 1]);
  }
  return per_seq_lens;
#endif
}
}  // namespace

bool MMInputGatherVisitor::visit(const MMInputItem& item) {
  if (item.has_type(MMType::IMAGE)) {
    data_type_ |= MMType::IMAGE;
    if (item.is_embedding()) {
      item_types_.push_back(MMType::IMAGE | MMType::EMBEDDING);
      MMDataItem data_item(MMType::IMAGE);
      data_item.set_data(build_embedding_data(MMType::IMAGE, item.embedding));
      image_embedding_items_.push_back(std::move(data_item));
    } else {
      item_types_.push_back(MMType::IMAGE);
      images_.push_back(item.decode_image);
    }
  }
  if (item.has_type(MMType::VIDEO)) {
    data_type_ |= MMType::VIDEO;
    uint32_t item_type = MMType::VIDEO;
    if (item.has_type(MMType::AUDIO)) {
      item_type |= MMType::AUDIO;
    }
    item_types_.push_back(item_type);
    videos_.push_back(item.decode_video);
    video_metadata_.push_back(item.video_meta);
  }
  if (item.has_type(MMType::AUDIO)) {
    data_type_ |= MMType::AUDIO;
    if (!item.has_type(MMType::VIDEO)) {
      item_types_.push_back(MMType::AUDIO);
    }
    audios_.push_back(item.decode_audio);
    audio_metadata_.push_back(item.audio_meta);
  }
  return true;
}

MMItemVec MMInputGatherVisitor::finish(std::vector<MMDataItem>& image_items,
                                       std::vector<MMDataItem>& video_items,
                                       std::vector<MMDataItem>& audio_items) {
  size_t image_embedding_idx = 0;
  size_t image_idx = 0;
  size_t video_idx = 0;
  size_t audio_idx = 0;
  MMItemVec output_items;

  auto take_next = [&output_items](std::vector<MMDataItem>& items,
                                   size_t& index) {
    CHECK(index < items.size())
        << "Multimodal item count does not match input.";
    output_items.push_back(std::move(items[index]));
    ++index;
  };

  output_items.reserve(item_types_.size());
  for (uint32_t item_type : item_types_) {
    if ((item_type & MMType::IMAGE) && (item_type & MMType::EMBEDDING)) {
      take_next(image_embedding_items_, image_embedding_idx);
    } else if (item_type & MMType::IMAGE) {
      take_next(image_items, image_idx);
    } else if (item_type & MMType::VIDEO) {
      CHECK(video_idx < video_items.size())
          << "Multimodal item count does not match input.";
      MMDataItem video_item = std::move(video_items[video_idx++]);
      if (item_type & MMType::AUDIO) {
        CHECK(audio_idx < audio_items.size())
            << "Multimodal item count does not match input.";
        for (auto& [key, value] : audio_items[audio_idx++].mutable_data()) {
          video_item.mutable_data()[key] = std::move(value);
        }
      }
      output_items.push_back(std::move(video_item));
    } else if (item_type & MMType::AUDIO) {
      take_next(audio_items, audio_idx);
    } else {
      LOG(FATAL) << "Invalid multimodal item type: " << item_type;
    }
  }

  CHECK(image_embedding_idx == image_embedding_items_.size() &&
        image_idx == image_items.size() && video_idx == video_items.size() &&
        audio_idx == audio_items.size())
      << "Multimodal item count does not match input.";
  return output_items;
}

bool CollectItemTensorVisitor::visit(MMDataItem& item) {
  for (const auto& pair : item.data()) {
    const auto& key = pair.first;

    if (!black_keys_.empty() && black_keys_.count(key)) {
      continue;
    }

    if (!white_keys_.empty() && !white_keys_.count(key)) {
      continue;
    }

    auto& tar = datas_[pair.first];
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      tar.emplace_back(std::get<torch::Tensor>(pair.second));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);
      tar.insert(tar.end(), lst.begin(), lst.end());
    }
  }
  return true;
}

bool CollectItemTensorVisitor::visit(const MMKey& key, MMValue& value) {
  if (!black_keys_.empty() && black_keys_.count(key)) {
    return true;
  }

  if (!white_keys_.empty() && !white_keys_.count(key)) {
    return true;
  }

  auto& tar = datas_[key];
  if (std::holds_alternative<torch::Tensor>(value)) {
    tar.push_back(std::get<torch::Tensor>(value));
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
    const auto& lst = std::get<std::vector<torch::Tensor>>(value);
    tar.insert(tar.end(), lst.begin(), lst.end());
  }

  return true;
}

bool CollectMMDataTensorVisitor::visit(MMData& data) {
  type_ |= data.type();
  data.foreach (item_visitor_);
  return true;
}

bool MMTokenNumVisitor::visit(MMDataItem& item) {
  if (item.type() == type_ && !item.is_embedded()) {
    token_nums_.push_back(item.state().mm_token_num());
  }
  return true;
}

bool EncoderInputGatherVisitor::visit(MMDataItem& item) {
  if (item.is_embedded()) return true;

  for (const auto& [key, value] : item.data()) {
    if (absl::StartsWith(key, filter_prefix_)) continue;
    auto& tar = datas_[key];
    if (std::holds_alternative<torch::Tensor>(value)) {
      tar.push_back(std::get<torch::Tensor>(value));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
      const auto& vec = std::get<std::vector<torch::Tensor>>(value);
      tar.insert(tar.end(), vec.begin(), vec.end());
    }
  }
  return true;
}

bool EncoderInputGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict dict;
  for (const auto& pair : datas_) {
    torch::Tensor tar;
    if (safe_concat(pair.second, tar)) {
      dict[pair.first] = tar;
    } else {
      dict[pair.first] = std::move(pair.second);
    }
  }
  mm_data.replace(dict);
  return true;
}

bool EncoderOutputScatterVisitor::visit(MMDataItem& item) {
  if (item.is_embedded()) return true;

  int32_t* idx = nullptr;
  if (item.type() == MMType::IMAGE) {
    idx = &image_idx;
  } else if (item.type() == MMType::VIDEO) {
    idx = &video_idx;
  } else if (item.type() == MMType::AUDIO) {
    idx = &audio_idx;
  } else {
    LOG(FATAL) << " mm data item type invalid, type is " << item.type();
    return true;
  }

  const std::string embedding_key = get_embedding_key(item.type());
  const auto& vec =
      std::get<std::vector<torch::Tensor>>(data_.at(embedding_key));
  item.add(embedding_key, vec[*idx]);

  if (item.type() == MMType::AUDIO && data_.count("audio|ctc_ids") &&
      data_.count("audio|ctc_conf")) {
    item.add(
        "audio|ctc_ids",
        std::get<std::vector<torch::Tensor>>(data_.at("audio|ctc_ids"))[*idx]);
    item.add(
        "audio|ctc_conf",
        std::get<std::vector<torch::Tensor>>(data_.at("audio|ctc_conf"))[*idx]);
  }
  ++(*idx);
  return true;
}

bool EncoderOutputScatterVisitor::finish() const {
  for (const auto& [key, value] : data_) {
    std::string name = key.substr(0, key.find("|"));
    uint32_t idx = 0;
    if (name == "image") {
      idx = image_idx;
    } else if (name == "video") {
      idx = video_idx;
    } else if (name == "audio") {
      idx = audio_idx;
    } else {
      LOG(FATAL) << "invalid modality key: " << key;
    }
    if (idx != std::get<std::vector<torch::Tensor>>(value).size()) {
      return false;
    }
  }
  return true;
}

EncoderEmbeddingGatherVisitor::EncoderEmbeddingGatherVisitor(
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& scheduled_seq_lens)
    : per_seq_context_lens_(normalize_to_per_seq_lens(seq_lens)),
      per_seq_scheduled_lens_(normalize_to_per_seq_lens(scheduled_seq_lens)),
      per_seq_scheduled_offsets_(per_seq_scheduled_lens_.size(), 0),
      total_scheduled_tokens_(std::accumulate(per_seq_scheduled_lens_.begin(),
                                              per_seq_scheduled_lens_.end(),
                                              0)) {
  LOG(INFO) << "MM_TRACE raw_context=" << seq_lens.size()
            << " raw_scheduled=" << scheduled_seq_lens.size()
            << " context=" << per_seq_context_lens_.size()
            << " scheduled=" << per_seq_scheduled_lens_.size()
            << " total_scheduled=" << total_scheduled_tokens_;
  CHECK_EQ(per_seq_context_lens_.size(), per_seq_scheduled_lens_.size())
      << "sequence length counts do not match";
  for (size_t seq_index = 0; seq_index < per_seq_context_lens_.size();
       ++seq_index) {
    CHECK_GE(per_seq_context_lens_[seq_index],
             per_seq_scheduled_lens_[seq_index])
        << "kv length below q length for seq " << seq_index;
  }
  if (per_seq_scheduled_lens_.size() > 1) {
    std::partial_sum(per_seq_scheduled_lens_.begin(),
                     per_seq_scheduled_lens_.end() - 1,
                     per_seq_scheduled_offsets_.begin() + 1);
  }
}

bool EncoderEmbeddingGatherVisitor::visit(MMDataItem& item) {
  const auto& state = item.state();

  int32_t seq_index = state.seq_index();
  CHECK_GE(seq_index, 0);
  CHECK_LT(static_cast<size_t>(seq_index), per_seq_scheduled_lens_.size())
      << "item sequence index outside scheduled batch";

  auto token_pos = item.state().token_pos();
  int32_t start_pos = state.schedule_data().start_pos;
  int32_t end_pos = state.schedule_data().end_pos;
  // start_pos / end_pos select the scheduled subrange inside this multimodal
  // item's token span in the prompt. Not every token in that subrange is a real
  // multimodal token, so mm_token_mask is used to filter out non-multimodal
  // positions when slicing multimodal embeddings.
  auto [emb_start, emb_end] =
      compute_emb_range(start_pos, end_pos, state.mm_token_mask());
  int32_t schedule_tokens_num = per_seq_scheduled_lens_[seq_index];
  int32_t context_tokens_num = per_seq_context_lens_[seq_index];

  int32_t computed_tokens_num = context_tokens_num - schedule_tokens_num;
  int32_t req_start_idx_ = per_seq_scheduled_offsets_[seq_index];
  int32_t req_start_pos =
      req_start_idx_ + token_pos.offset - computed_tokens_num + start_pos;
  int32_t req_end_pos =
      req_start_idx_ + token_pos.offset - computed_tokens_num + end_pos;
  CHECK_GE(req_start_pos, req_start_idx_)
      << "scatter range starts before its sequence window";
  CHECK_LE(req_end_pos, req_start_idx_ + schedule_tokens_num)
      << "scatter range ends past its sequence window";

  const std::string key = get_embedding_key(item.type());
  auto emb = item.get<torch::Tensor>(key);
  if (!emb.has_value()) {
    LOG(ERROR) << "embedding not found for key: " << key;
    return false;
  }
  torch::Tensor embedding = emb.value();
  if (embedding.size(0) != state.mm_token_num()) {
    LOG(ERROR) << "embedding rows do not match the item token count for key: "
               << key;
    return false;
  }
  const auto update_mask = [&](torch::Tensor& mask) {
    if (!mask.defined()) {
      mask = torch::zeros({total_scheduled_tokens_},
                          embedding.options().dtype(torch::kBool));
    }
    mask.slice(/*dim=*/0, /*start=*/req_start_pos, /*end=*/req_end_pos) =
        state.mm_token_mask().slice(
            /*dim=*/0, /*start=*/start_pos, /*end=*/end_pos);
  };
  if (item.type() == MMType::IMAGE) {
    update_mask(image_mask_);
  } else if (item.type() == MMType::VIDEO) {
    update_mask(video_mask_);
  } else if (item.type() == MMType::AUDIO) {
    update_mask(audio_mask_);
  }
  datas_[key].push_back(
      embedding.slice(/*dim*/ 0, /*start*/ emb_start, /*end*/ emb_end));
  return true;
}

bool EncoderEmbeddingGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict data;
  torch::Tensor tar;
  for (auto& [key, value] : datas_) {
    if (safe_concat(value, tar)) {
      data[key] = tar;
    } else {
      LOG(ERROR) << "safe concat failed.";
      return false;
    }
  }
  // mask for merge multimodal embeddings into text.
  if (image_mask_.defined()) {
    data["image|mask"] = image_mask_;
  }
  if (video_mask_.defined()) {
    data["video|mask"] = video_mask_;
  }
  if (audio_mask_.defined()) {
    data["audio|mask"] = audio_mask_;
  }
  mm_data.replace(data);
  return true;
}

bool UpdateMMItemScheduleStateVisitor::visit(MMDataItem& item) {
  auto& schedule_data = item.mutable_state().mutable_schedule_data();
  auto& token_pos = item.state().token_pos();
  int32_t mm_end_idx = token_pos.offset + token_pos.length - 1;
  int32_t schedule_token_num = q_seq_len_;
  if (mm_end_idx < computed_token_num_) {
    return true;
  }
  if (token_pos.offset >= computed_token_num_ + schedule_token_num) {
    return true;
  }
  schedule_data.start_pos = std::max(computed_token_num_ - token_pos.offset, 0);
  schedule_data.end_pos =
      std::min(computed_token_num_ - token_pos.offset + schedule_token_num,
               token_pos.length);
  item.mutable_state().mutable_seq_index() = seq_idx_;
  scheduled_type_ |= item.type();
  mm_data_items_.push_back(item);
  return true;
}

ProcessorCacheLookupVisitor::ProcessorCacheLookupVisitor(ProcessorCache& cache,
                                                         size_t item_count)
    : cache_(cache) {
  cache_hits_.reserve(item_count);
  miss_inputs_.reserve(item_count);
}

bool ProcessorCacheLookupVisitor::visit(const MMInputItem& input) {
  std::optional<MMDataItem> cache_hit;
  if (!input.is_embedding() && input.hash_key.has_value()) {
    cache_hit = cache_.lookup(input.hash_key.value());
  }
  if (!cache_hit.has_value()) {
    miss_inputs_.emplace_back(input);
  }
  cache_hits_.emplace_back(std::move(cache_hit));
  return true;
}

ProcessorCacheInsertVisitor::ProcessorCacheInsertVisitor(ProcessorCache& cache)
    : cache_(cache) {}

bool ProcessorCacheInsertVisitor::visit(MMDataItem& item) {
  if (!item.is_embedded()) {
    cache_.insert(item.state().schedule_data().key, item);
  }
  return true;
}

namespace {

// Encoder inputs an item no longer needs once its output is cached.
constexpr std::array<const char*, 2> kEncoderInputKeys = {
    "pixel_values", "pixel_values_videos"};

class ItemCollector final : public MMDataItem::IVisitor {
 public:
  bool visit(MMDataItem& item) override {
    if (item.valid()) {
      items_.emplace_back(&item);
    }
    return true;
  }

  const std::vector<MMDataItem*>& items() const { return items_; }

 private:
  std::vector<MMDataItem*> items_;
};

bool is_set(const XXH3Key& key) {
  return std::any_of(std::begin(key.data),
                     std::end(key.data),
                     [](uint8_t byte) { return byte != 0; });
}

void add_command(MMDataItem& item, const std::string& command) {
  item.mutable_data()[command] = torch::zeros({1}, torch::kUInt8);
}

}  // namespace

EncoderCacheDirectory::EncoderCacheDirectory(int64_t max_bytes,
                                             int64_t bytes_per_token)
    : max_bytes_(max_bytes), bytes_per_token_(bytes_per_token) {
  CHECK_GT(max_bytes, 0);
  CHECK_GT(bytes_per_token, 0);
}

void EncoderCacheDirectory::plan(MMBatchData& batch) {
  ItemCollector collector;
  CHECK(batch.foreach (collector));
  if (collector.items().empty()) {
    return;
  }
  KeySet referenced;
  for (const MMDataItem* item : collector.items()) {
    referenced.insert(item->state().schedule_data().key);
  }

  // Workers look every item up before storing any output of the step, so an
  // output first stored in this step cannot serve another item of the step.
  KeySet stored;
  std::vector<XXH3Key> evicted;
  for (MMDataItem* item : collector.items()) {
    const XXH3Key& key = item->state().schedule_data().key;
    const int64_t size =
        static_cast<int64_t>(item->state().mm_token_num()) * bytes_per_token_;
    if (item->is_embedded() || !is_set(key) || size <= 0) {
      add_command(*item, kEncoderCacheSkipKey);
      continue;
    }
    EntryMap::iterator it = entries_.find(key);
    if (it != entries_.end() && stored.count(key) == 0) {
      lru_keys_.splice(lru_keys_.end(), lru_keys_, it->second.lru_it);
      for (const char* input_key : kEncoderInputKeys) {
        item->mutable_data().erase(input_key);
      }
      add_command(*item, kEncoderCacheReuseKey);
      continue;
    }
    if (it != entries_.end() || admit(key, size, referenced, evicted)) {
      stored.insert(key);
      add_command(*item, kEncoderCacheStoreKey);
      continue;
    }
    add_command(*item, kEncoderCacheSkipKey);
  }

  MMDataItem& first_item = *collector.items().front();
  for (const XXH3Key& key : evicted) {
    add_command(first_item,
                kEncoderCacheEvictPrefix +
                    absl::BytesToHexString(absl::string_view(
                        reinterpret_cast<const char*>(key.data),
                        sizeof(key.data))));
  }
}

bool EncoderCacheDirectory::admit(const XXH3Key& key,
                                  int64_t size,
                                  const KeySet& referenced,
                                  std::vector<XXH3Key>& evicted) {
  if (size > max_bytes_) {
    return false;
  }
  LruList::iterator candidate = lru_keys_.begin();
  while (current_bytes_ + size > max_bytes_ && candidate != lru_keys_.end()) {
    if (referenced.count(*candidate) != 0) {
      ++candidate;
      continue;
    }
    EntryMap::iterator victim = entries_.find(*candidate);
    CHECK(victim != entries_.end());
    current_bytes_ -= victim->second.size;
    evicted.emplace_back(*candidate);
    entries_.erase(victim);
    candidate = lru_keys_.erase(candidate);
  }
  if (current_bytes_ + size > max_bytes_) {
    return false;
  }
  lru_keys_.push_back(key);
  entries_.emplace(key, Entry{size, std::prev(lru_keys_.end())});
  current_bytes_ += size;
  return true;
}

EncoderCacheCommandVisitor::EncoderCacheCommandVisitor(EncoderCache* cache)
    : cache_(cache) {}

bool EncoderCacheCommandVisitor::visit(MMDataItem& item) {
  const XXH3Key& key = item.state().schedule_data().key;
  MMDict& data = item.mutable_data();
  for (MMDict::iterator it = data.begin(); it != data.end();) {
    const std::string& name = it->first;
    if (!absl::StartsWith(name, kEncoderCacheCommandPrefix)) {
      ++it;
      continue;
    }
    cache_->set_engine_managed();
    if (absl::StartsWith(name, kEncoderCacheEvictPrefix)) {
      const std::string evicted = absl::HexStringToBytes(
          absl::string_view(name).substr(sizeof(kEncoderCacheEvictPrefix) - 1));
      CHECK_EQ(evicted.size(), sizeof(XXH3Key::data))
          << "malformed encoder cache eviction: " << name;
      cache_->erase(XXH3Key(reinterpret_cast<const uint8_t*>(evicted.data())));
    } else if (name == kEncoderCacheReuseKey) {
      reused_keys_.insert(key);
    } else if (name == kEncoderCacheSkipKey) {
      skipped_keys_.insert(key);
    }
    it = data.erase(it);
  }
  return true;
}

bool EncoderCacheCommandVisitor::reuses(const XXH3Key& key) const {
  return reused_keys_.count(key) != 0;
}

bool EncoderCacheCommandVisitor::skips(const XXH3Key& key) const {
  return skipped_keys_.count(key) != 0;
}

EncoderCacheLookupVisitor::EncoderCacheLookupVisitor(
    EncoderCache* cache,
    const EncoderCacheCommandVisitor& commands)
    : cache_(cache), commands_(commands) {}

bool EncoderCacheLookupVisitor::visit(MMDataItem& item) {
  if (item.is_embedded()) {
    return true;
  }
  const XXH3Key& key = item.state().schedule_data().key;
  std::optional<torch::Tensor> cached = cache_->lookup(key);
  HISTOGRAM_OBSERVE(encoder_cache_hit_rate, cached.has_value() ? 100 : 0);
  if (!cached.has_value()) {
    CHECK(!commands_.reuses(key))
        << "encoder cache lost an output the engine reuses without inputs";
    return true;
  }
  item.add(get_embedding_key(item.type()), cached.value());
  return true;
}

EncoderCacheInsertVisitor::EncoderCacheInsertVisitor(
    EncoderCache* cache,
    const EncoderCacheCommandVisitor& commands)
    : cache_(cache), commands_(commands) {}

bool EncoderCacheInsertVisitor::visit(MMDataItem& item) {
  if (!item.is_embedded()) {
    return true;
  }
  const XXH3Key& key = item.state().schedule_data().key;
  if (commands_.skips(key)) {
    return true;
  }
  std::optional<torch::Tensor> embedding =
      item.get<torch::Tensor>(get_embedding_key(item.type()));
  if (!embedding.has_value()) {
    return true;
  }
  cache_->insert(key, std::move(embedding.value()));
  return true;
}

MMSideOutputCacheVisitor::MMSideOutputCacheVisitor(MMSideOutputCache& cache)
    : cache_(cache) {}

bool MMSideOutputCacheVisitor::visit(MMData& data) {
  if (!data.valid() || !data.hold<MMItemVec>()) {
    return true;
  }
  return data.foreach (static_cast<MMDataItem::IVisitor&>(*this));
}

bool MMSideOutputCacheVisitor::visit(MMDataItem& item) {
  if (item.type() != MMType::AUDIO) {
    return true;
  }
  const XXH3Key& key = item.state().schedule_data().key;
  auto ids = item.get<torch::Tensor>("audio|ctc_ids");
  auto conf = item.get<torch::Tensor>("audio|ctc_conf");
  if (ids.has_value() && ids->defined() && conf.has_value() && conf->defined()) {
    cache_.insert(key, std::move(*ids), std::move(*conf));
    return true;
  }
  auto cached = cache_.lookup(key);
  if (cached.has_value()) {
    item.add("audio|ctc_ids", cached->first);
    item.add("audio|ctc_conf", cached->second);
  }
  return true;
}

MMSideOutputCollectVisitor::MMSideOutputCollectVisitor(int32_t num_sequences) {
  CHECK_GE(num_sequences, 0);
  outputs_.resize(num_sequences);
}

bool MMSideOutputCollectVisitor::visit(MMData& data) {
  if (!data.valid() || !data.hold<MMItemVec>()) {
    return true;
  }
  return data.foreach (static_cast<MMDataItem::IVisitor&>(*this));
}

bool MMSideOutputCollectVisitor::visit(MMDataItem& item) {
  if (item.type() != MMType::AUDIO) {
    return true;
  }
  const auto ids = item.get<torch::Tensor>("audio|ctc_ids");
  const auto conf = item.get<torch::Tensor>("audio|ctc_conf");
  if (!ids.has_value() || !ids->defined() || !conf.has_value() ||
      !conf->defined()) {
    return true;
  }
  // Text-only sequences have no items here, so traversal cannot advance past
  // their batch slots. Use seq_index to avoid shifting CTC outputs to them.
  int32_t seq_index = item.state().seq_index();
  CHECK_GE(seq_index, 0);
  CHECK_LT(static_cast<size_t>(seq_index), outputs_.size());
  auto& output = outputs_[seq_index];
  std::get<std::vector<torch::Tensor>>(
      output.try_emplace("audio|ctc_ids", std::vector<torch::Tensor>())
          .first->second)
      .push_back(*ids);
  std::get<std::vector<torch::Tensor>>(
      output.try_emplace("audio|ctc_conf", std::vector<torch::Tensor>())
          .first->second)
      .push_back(*conf);
  return true;
}

std::vector<MMDict> MMSideOutputCollectVisitor::finish() {
  return std::move(outputs_);
}

}  // namespace xllm
