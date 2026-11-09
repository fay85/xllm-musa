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

#include "scheduler/linear_prefix_admission_policy.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <utility>

#include "core/framework/request/sequence.h"

namespace xllm {

LinearPrefixAdmissionPolicy::LinearPrefixAdmissionPolicy(size_t block_size,
                                                         Options options)
    : block_size_(block_size), options_(std::move(options)) {
  CHECK_GT(block_size_, 0u);
  CHECK_GT(options_.chunk_size, 0);
  CHECK_GT(options_.max_entries, 0u);
  CHECK_GE(options_.min_distinct_successors, 2u);
  CHECK_GT(options_.observation_ttl.count(), 0);
  CHECK_NE(block_size_ % static_cast<size_t>(options_.chunk_size), 0u);
}

Slice<XXH3Key> LinearPrefixAdmissionPolicy::prompt_block_hashes(
    Sequence& sequence) const {
  const uint32_t block_size = static_cast<uint32_t>(block_size_);
  sequence.update_block_hashes(block_size, options_.hasher_type);
  const Slice<XXH3Key> hashes = sequence.block_hashes(block_size);
  const size_t prompt_blocks = std::min(
      hashes.size(),
      num_hash_blocks(
          options_.hasher_type, sequence.num_prompt_tokens(), block_size));
  return hashes.slice(0, prompt_blocks);
}

size_t LinearPrefixAdmissionPolicy::plan_checkpoint(Sequence* sequence) {
  if (sequence == nullptr || sequence->is_graph_warmup() ||
      sequence->stage() == SequenceStage::DECODE) {
    return 0;
  }
  const std::optional<size_t> cached_checkpoint =
      sequence->linear_prefix_checkpoint_tokens();
  if (cached_checkpoint.has_value()) {
    return cached_checkpoint.value();
  }
  const Slice<XXH3Key> hashes = prompt_block_hashes(*sequence);
  const size_t checkpoint = find_checkpoint_tokens(hashes);
  sequence->set_linear_prefix_checkpoint_tokens(checkpoint);
  observe(hashes);
  return checkpoint;
}

size_t LinearPrefixAdmissionPolicy::find_checkpoint_tokens(
    const Slice<XXH3Key>& hashes,
    std::optional<Clock::time_point> timestamp) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Clock::time_point now = timestamp.value_or(Clock::now());
  expire(now);
  Node* parent = &root_;
  size_t hash_index = 0;
  size_t checkpoint = 0;
  while (hash_index < hashes.size()) {
    const auto found = parent->children.find(hashes[hash_index]);
    if (found == parent->children.end()) {
      break;
    }
    Node& node = *found->second;
    const size_t matched_hashes = match_edge(node, hashes, hash_index);
    if (matched_hashes != node.hashes.size()) {
      break;
    }
    hash_index += matched_hashes;
    touch(node, now);
    if (node.children.size() >= options_.min_distinct_successors &&
        hash_index < hashes.size() &&
        node.token_count % static_cast<size_t>(options_.chunk_size) != 0) {
      checkpoint = node.token_count;
    }
    parent = &node;
  }
  return checkpoint;
}

void LinearPrefixAdmissionPolicy::observe(
    const Slice<XXH3Key>& hashes,
    std::optional<Clock::time_point> timestamp) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Clock::time_point now = timestamp.value_or(Clock::now());
  expire(now);
  Node* parent = &root_;
  size_t hash_index = 0;
  while (hash_index < hashes.size()) {
    const auto found = parent->children.find(hashes[hash_index]);
    if (found == parent->children.end()) {
      add_child(*parent, hashes, hash_index, now);
      return;
    }
    Node* node = found->second;
    const size_t matched_hashes = match_edge(*node, hashes, hash_index);
    if (matched_hashes < node->hashes.size()) {
      node = &split_node(*node, matched_hashes);
    }
    hash_index += matched_hashes;
    touch(*node, now);
    parent = node;
  }
}

size_t LinearPrefixAdmissionPolicy::match_edge(const Node& node,
                                               const Slice<XXH3Key>& hashes,
                                               size_t hash_index) {
  const size_t available_hashes =
      std::min(node.hashes.size(), hashes.size() - hash_index);
  size_t matched_hashes = 0;
  while (matched_hashes < available_hashes &&
         FixedStringKeyEqual{}(node.hashes[matched_hashes],
                               hashes[hash_index + matched_hashes])) {
    ++matched_hashes;
  }
  return matched_hashes;
}

LinearPrefixAdmissionPolicy::Node& LinearPrefixAdmissionPolicy::split_node(
    Node& node,
    size_t matched_hashes) {
  CHECK_GT(matched_hashes, 0u);
  CHECK_LT(matched_hashes, node.hashes.size());
  Node& prefix = nodes_.emplace_back();
  prefix.hashes.reserve(matched_hashes);
  prefix.hashes.assign(node.hashes.begin(),
                       node.hashes.begin() + matched_hashes);
  prefix.token_count =
      node.token_count - (node.hashes.size() - matched_hashes) * block_size_;
  prefix.last_access = node.last_access;
  prefix.parent = node.parent;
  prefix.storage_iterator = std::prev(nodes_.end());
  prefix.leaf_iterator = leaves_.end();
  node.hashes = std::vector<XXH3Key>(node.hashes.begin() + matched_hashes,
                                     node.hashes.end());
  prefix.children.emplace(node.hashes.front(), &node);
  prefix.parent->children.at(prefix.hashes.front()) = &prefix;
  node.parent = &prefix;
  return prefix;
}

void LinearPrefixAdmissionPolicy::add_child(Node& parent,
                                            const Slice<XXH3Key>& hashes,
                                            size_t hash_index,
                                            Clock::time_point now) {
  const bool extends_leaf = &parent != &root_ && parent.children.empty();
  if (hash_index >= options_.max_entries) {
    return;
  }
  if (extends_leaf) {
    leaves_.erase(parent.leaf_iterator);
    parent.leaf_iterator = leaves_.end();
  }
  while (num_entries_ >= options_.max_entries) {
    CHECK(!leaves_.empty());
    Node& leaf = *leaves_.begin()->second;
    if (leaf.hashes.size() > 1) {
      trim_leaf(leaf);
    } else {
      erase_leaf(leaf, &parent);
    }
  }
  const size_t hash_count =
      std::min(hashes.size() - hash_index, options_.max_entries - num_entries_);
  Node& node = nodes_.emplace_back();
  node.hashes.reserve(hash_count);
  node.hashes.assign(hashes.begin() + hash_index,
                     hashes.begin() + hash_index + hash_count);
  node.token_count = (hash_index + hash_count) * block_size_;
  node.last_access = now;
  node.parent = &parent;
  node.storage_iterator = std::prev(nodes_.end());
  node.leaf_iterator = leaves_.emplace(now, &node);
  parent.children.emplace(node.hashes.front(), &node);
  num_entries_ += hash_count;
}

void LinearPrefixAdmissionPolicy::touch(Node& node, Clock::time_point now) {
  node.last_access = now;
  if (node.children.empty()) {
    leaves_.erase(node.leaf_iterator);
    node.leaf_iterator = leaves_.emplace(now, &node);
  }
}

void LinearPrefixAdmissionPolicy::expire(Clock::time_point now) {
  while (!leaves_.empty()) {
    Node& leaf = *leaves_.begin()->second;
    if (now - leaf.last_access < options_.observation_ttl) {
      break;
    }
    erase_leaf(leaf);
  }
}

void LinearPrefixAdmissionPolicy::erase_leaf(Node& node,
                                             Node* protected_parent) {
  Node* parent = node.parent;
  leaves_.erase(node.leaf_iterator);
  num_entries_ -= node.hashes.size();
  parent->children.erase(node.hashes.front());
  nodes_.erase(node.storage_iterator);
  if (parent != &root_ && parent != protected_parent &&
      parent->children.empty()) {
    parent->leaf_iterator = leaves_.emplace(parent->last_access, parent);
  }
}

void LinearPrefixAdmissionPolicy::trim_leaf(Node& node) {
  const size_t removed_hashes = node.hashes.size() - 1;
  std::vector<XXH3Key> retained_hashes{node.hashes.front()};
  node.hashes.swap(retained_hashes);
  node.token_count -= removed_hashes * block_size_;
  num_entries_ -= removed_hashes;
}

void LinearPrefixAdmissionPolicy::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  root_.children.clear();
  leaves_.clear();
  nodes_.clear();
  num_entries_ = 0;
}

}  // namespace xllm
