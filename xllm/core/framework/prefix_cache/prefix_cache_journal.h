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

#pragma once

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "core/framework/block/block.h"
#include "core/util/blockingconcurrentqueue.h"
#include "core/util/hash_util.h"

namespace xllm {

// Engine-local prefix-cache records. Identity, contract version, event
// reasons, and RPC failure state are filled by XServiceClient.
enum class PrefixCacheEventKind : int8_t {
  STORED = 1,
  REMOVED = 2,
  CLEARED = 3,
};

struct PrefixCacheEvent {
  PrefixCacheEventKind kind = PrefixCacheEventKind::STORED;
  XXH3Key hash;
};

struct PrefixCacheEvents {
  std::vector<PrefixCacheEvent> events;
  uint64_t commit_version = 0;
};

struct PrefixCacheSnapshot {
  std::vector<XXH3Key> hashes;
  uint64_t commit_version = 0;
};

// Engine-local event source. Host, HBM, and every DP rank publish the same
// hash. publish_stored and publish_removed only enqueue. resident_ counts
// holders and is updated when those events are dequeued: a STORED is reported
// only when the hash is inserted, and a REMOVED only when the holder count
// reaches zero. Snapshot copies the hashes that still have a holder. RPC stays
// outside this object. KV, C1, and C4 never coexist, so one hash map is the
// whole published cache.
class PrefixCacheJournal final {
 public:
  static PrefixCacheJournal* instance();

  // The pending queue is unbounded. Publishing stays off until the KV-event
  // consumer starts, so instances that never drain the journal do not retain
  // every insert and eviction.
  void set_publishing_enabled(bool enabled);

  void publish_stored(const XXH3Key& hash, BlockType type);
  void publish_removed(const XXH3Key& hash, BlockType type);
  void publish_cleared();

  PrefixCacheSnapshot get_snapshot();

  PrefixCacheEvents get_prefixcache_events(size_t max_events);

  size_t pending_event_count() const {
    return pending_events_.load(std::memory_order_acquire);
  }

 private:
  using ResidentMap = absl::
      flat_hash_map<XXH3Key, uint32_t, FixedStringKeyHash, FixedStringKeyEqual>;

  void enqueue_event(PrefixCacheEventKind kind, std::optional<XXH3Key> hash);
  // Applies one dequeued event to resident_. Appends it when the published
  // hash set changes.
  void apply_dequeued_event(const PrefixCacheEvent& event,
                            std::vector<PrefixCacheEvent>* emitted);

  mutable std::mutex mutex_;
  uint64_t commit_version_ = 0;
  std::atomic<bool> publishing_enabled_{false};
  std::atomic<size_t> pending_events_{0};
  moodycamel::BlockingConcurrentQueue<PrefixCacheEvent> pending_;
  // Holder count per published hash. Updated only while draining, in the same
  // critical section that advances commit_version_. KV, C1, and C4 never share
  // a process, so one map is enough.
  ResidentMap resident_;
};

}  // namespace xllm
