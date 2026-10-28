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

#include "processors/multimodal_processor.h"

#include <utility>

#include "common/metrics.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "models/model_registry.h"
#include "processors/cacheable_multimodal_processor.h"
#include "util/timer.h"

namespace xllm {

namespace {

// Each distinct shared prefix contributes one cached segment per image.
constexpr size_t kMaxCachedSegments = 256;

}  // namespace

MultimodalProcessorBase::MultimodalProcessorBase(
    std::shared_ptr<Tokenizer> tokenizer,
    const TokenizerArgs& tokenizer_args)
    : tokenizer_(std::move(tokenizer)), tokenizer_args_(tokenizer_args) {}

MultimodalProcessorBase::~MultimodalProcessorBase() = default;

bool MultimodalProcessorBase::tokenize(const std::string& prompt,
                                       std::vector<int32_t>& token_ids) const {
  return tokenize(prompt, /*segment_delimiter=*/{}, token_ids);
}

bool MultimodalProcessorBase::tokenize(const std::string& prompt,
                                       std::string_view segment_delimiter,
                                       std::vector<int32_t>& token_ids) const {
  Timer timer;
  const bool reuse_segments =
      !segment_delimiter.empty() && segment_reuse_enabled_.load();
  const bool ok =
      reuse_segments ? tokenize_segments(prompt, segment_delimiter, token_ids)
                     : tokenizer_->encode(prompt, &token_ids);
  if (!ok) {
    LOG(ERROR) << "Failed to encode prompt: " + prompt;
    return false;
  }

  pad_to_max_length(token_ids);
  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());
  return true;
}

bool MultimodalProcessorBase::tokenize_segments(
    const std::string& prompt,
    std::string_view segment_delimiter,
    std::vector<int32_t>& token_ids) const {
  token_ids.clear();
  size_t begin = 0;
  for (size_t end = prompt.find(segment_delimiter, begin);
       end != std::string::npos;
       end = prompt.find(segment_delimiter, begin)) {
    const size_t segment_end = end + segment_delimiter.size();
    if (!append_cached_segment(prompt.substr(begin, segment_end - begin),
                               token_ids)) {
      return false;
    }
    begin = segment_end;
  }
  std::vector<int32_t> tail_ids;
  if (!tokenizer_->encode(prompt.substr(begin), &tail_ids)) {
    return false;
  }
  token_ids.insert(token_ids.end(), tail_ids.begin(), tail_ids.end());

  // Segment independence relies on the tokenizer splitting at special tokens;
  // confirm it once against whole-prompt encoding.
  if (!segment_reuse_verified_.exchange(true)) {
    std::vector<int32_t> whole_ids;
    if (!tokenizer_->encode(prompt, &whole_ids)) {
      return false;
    }
    if (whole_ids != token_ids) {
      LOG(WARNING) << "Segmented prompt encoding differs from whole-prompt "
                      "encoding; disabling segment reuse";
      segment_reuse_enabled_.store(false);
      token_ids = std::move(whole_ids);
    }
  }
  return true;
}

bool MultimodalProcessorBase::append_cached_segment(
    std::string segment,
    std::vector<int32_t>& token_ids) const {
  {
    std::lock_guard<std::mutex> lock(segment_cache_mutex_);
    auto it = segment_cache_.find(segment);
    if (it != segment_cache_.end()) {
      token_ids.insert(token_ids.end(), it->second.begin(), it->second.end());
      return true;
    }
  }
  std::vector<int32_t> segment_ids;
  if (!tokenizer_->encode(segment, &segment_ids)) {
    return false;
  }
  token_ids.insert(token_ids.end(), segment_ids.begin(), segment_ids.end());
  std::lock_guard<std::mutex> lock(segment_cache_mutex_);
  if (segment_cache_.size() >= kMaxCachedSegments) {
    segment_cache_.clear();
  }
  segment_cache_.emplace(std::move(segment), std::move(segment_ids));
  return true;
}

void MultimodalProcessorBase::assign_mm_hash_keys(const MMInput& mm_input,
                                                  MMData& mm_data) const {
  const std::vector<MMInputItem>& input_items = mm_input.items();
  MMItemVec& output_items = mm_data.items<MMItemVec>();
  CHECK_EQ(input_items.size(), output_items.size());
  for (size_t index = 0; index < input_items.size(); ++index) {
    const std::optional<XXH3Key>& hash_key = input_items[index].hash_key;
    if (hash_key.has_value()) {
      output_items[index].mutable_state().mutable_schedule_data().key =
          hash_key.value();
    }
  }
}

void MultimodalProcessorBase::pad_to_max_length(
    std::vector<int32_t>& token_ids) const {
  const int32_t max_sequence_length =
      DiTConfig::get_instance().max_sequence_length();
  if (max_sequence_length <= 0 || tokenizer_args_.pad_token().empty()) {
    return;
  }

  const auto pad_id = tokenizer_->token_to_id(tokenizer_args_.pad_token());
  if (!pad_id.has_value() ||
      static_cast<int32_t>(token_ids.size()) >= max_sequence_length) {
    return;
  }

  const int32_t pad_count =
      max_sequence_length - static_cast<int32_t>(token_ids.size());
  token_ids.insert(token_ids.begin(), pad_count, pad_id.value());
}

std::unique_ptr<MultimodalProcessorBase> create_multimodal_processor(
    const ModelArgs& model_args,
    std::shared_ptr<Tokenizer> tokenizer,
    int64_t max_cache_items,
    const TokenizerArgs& tokenizer_args) {
  const std::string& model_type = model_args.model_type();
  std::string resolved_name;
  std::string error_message;
  CHECK(resolve_model_registration_name(
      model_type, &resolved_name, &error_message))
      << error_message;

  MultimodalProcessorFactory multimodal_processor_factory =
      ModelRegistry::get_multimodal_processor_factory(resolved_name);
  CHECK(multimodal_processor_factory != nullptr)
      << "Missing multimodal processor for model type: " << model_type;
  std::unique_ptr<MultimodalProcessorBase> processor =
      multimodal_processor_factory(
          model_args, std::move(tokenizer), tokenizer_args);
  if (max_cache_items == 0) {
    return processor;
  }
  return std::make_unique<CacheableMultimodalProcessor>(std::move(processor),
                                                        max_cache_items);
}

}  // namespace xllm
