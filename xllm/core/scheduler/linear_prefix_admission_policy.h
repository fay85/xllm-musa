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

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/framework/prefix_cache/block_hasher.h"
#include "core/util/hash_util.h"
#include "core/util/slice.h"

namespace xllm {

class Sequence;

class LinearPrefixAdmissionPolicy final {
 public:
  using Clock = std::chrono::steady_clock;

  struct Options {
    int32_t chunk_size = 0;
    BlockHasherType hasher_type = BlockHasherType::TEXT;
    size_t max_entries = 65536;
    size_t min_distinct_successors = 4;
    std::chrono::seconds observation_ttl = std::chrono::minutes(10);
  };

  explicit LinearPrefixAdmissionPolicy(size_t block_size, Options options);

  LinearPrefixAdmissionPolicy(const LinearPrefixAdmissionPolicy&) = delete;
  LinearPrefixAdmissionPolicy& operator=(const LinearPrefixAdmissionPolicy&) =
      delete;

  size_t plan_checkpoint(Sequence* sequence);

  size_t find_checkpoint_tokens(
      const Slice<XXH3Key>& hashes,
      std::optional<Clock::time_point> timestamp = std::nullopt);
  void observe(const Slice<XXH3Key>& hashes,
               std::optional<Clock::time_point> timestamp = std::nullopt);
  void reset();

 private:
  struct Node;
  using NodeList = std::list<Node>;
  using LeafIndex = std::multimap<Clock::time_point, Node*>;

  struct Node {
    std::vector<XXH3Key> hashes;
    size_t token_count = 0;
    Clock::time_point last_access;
    Node* parent = nullptr;
    std::unordered_map<XXH3Key, Node*, FixedStringKeyHash, FixedStringKeyEqual>
        children;
    NodeList::iterator storage_iterator;
    LeafIndex::iterator leaf_iterator;
  };

  Slice<XXH3Key> prompt_block_hashes(Sequence& sequence) const;
  static size_t match_edge(const Node& node,
                           const Slice<XXH3Key>& hashes,
                           size_t hash_index);
  Node& split_node(Node& node, size_t matched_hashes);
  void add_child(Node& parent,
                 const Slice<XXH3Key>& hashes,
                 size_t hash_index,
                 Clock::time_point now);
  void touch(Node& node, Clock::time_point now);
  void expire(Clock::time_point now);
  void erase_leaf(Node& node, Node* protected_parent = nullptr);
  void trim_leaf(Node& node);

  size_t block_size_;
  Options options_;
  std::mutex mutex_;
  Node root_;
  NodeList nodes_;
  LeafIndex leaves_;
  size_t num_entries_ = 0;
};

}  // namespace xllm
