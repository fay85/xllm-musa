/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "platform/batch_memcpy.h"
#include "platform/layer_synchronizer.h"

namespace xllm {

struct ModelInputParams;
class ProcessGroup;

class PrefillLayerwiseHostPool final {
 public:
  explicit PrefillLayerwiseHostPool(ProcessGroup* process_group);

 private:
  friend class PrefillLayerwiseKVCache;
  class Mapping;
  struct BatchState;

  void initialize(const std::vector<KVCache>& device_caches,
                  int64_t host_memory_bytes,
                  int32_t kv_split_rank);
  bool shared() const { return rank_count_ > 1; }
  bool is_writer(int64_t layer) const;
  bool enter(uint64_t batch_id,
             const std::vector<int64_t>& read_blocks,
             const std::vector<int64_t>& write_blocks,
             bool valid);
  bool read_complete(int64_t layer);
  void leave();
  void fail();

  int32_t rank_count_ = 1;
  int32_t local_rank_ = 0;
  ProcessGroup* const process_group_;
  std::vector<KVCache> caches_;
  int64_t bytes_ = 0;
  int64_t layers_ = 0;
  int64_t blocks_ = 0;
  std::shared_ptr<Mapping> mapping_;
  BatchState* batch_ = nullptr;
};

class PrefillLayerwiseKVCache final : public LayerSynchronizer {
 public:
  PrefillLayerwiseKVCache(const KVCacheShape& shape,
                          const KVCacheCreateOptions& options,
                          int32_t num_buffers,
                          int64_t host_memory_bytes,
                          std::vector<KVCache>& device_caches,
                          ProcessGroup* host_group = nullptr,
                          int32_t kv_split_size = 1,
                          int32_t kv_split_rank = 0);
  ~PrefillLayerwiseKVCache() override;

  // Validate all block ids before changing cache state. Blocks use the same
  // logical ids as the scheduler's Device pool. Reads cover every block that
  // holds prefix tokens, including a partially written boundary block; blocks
  // written only by this batch are not loaded. `write_blocks` lists the blocks
  // whose current-batch rows are saved back to Host.
  bool begin(std::vector<int64_t> read_blocks,
             std::vector<int64_t> write_blocks,
             const Stream* compute_stream,
             bool expect_push = false);
  bool begin(const ModelInputParams& params,
             int64_t block_size,
             const Stream* compute_stream,
             bool expect_push = false);
  // Every participating rank calls this before model/PUSH execution. A peer
  // rejection cancels only scratch preparation and retains usable Host data.
  bool begin(const ModelInputParams& params,
             int64_t block_size,
             const Stream* compute_stream,
             bool expect_push,
             ProcessGroup* preparation_group);
  void notify_push_completion(std::optional<int64_t> layer, bool safe_to_reuse);
  bool finish();
  bool synchronize_layer(int64_t layer_index) override;
  bool record_stream(int64_t layer_index, Stream* stream) override;
  void abort() override;
  uint32_t size() const override;

  const std::vector<KVCache>& host_caches() const { return host_pool_.caches_; }
  int64_t device_bytes() const { return device_bytes_; }
  int64_t host_bytes() const { return host_pool_.bytes_; }

 private:
  bool prepare(std::vector<int64_t> read_blocks,
               std::vector<int64_t> write_blocks,
               const Stream* compute_stream,
               bool expect_push,
               uint64_t batch_id,
               bool metadata_valid);
  void require_usable() const;
  bool load_layer(int64_t layer_index);
  bool save_layer(int64_t layer_index);
  bool copy_layer(int64_t layer_index, bool load);
  bool push_layer_complete(int64_t layer_index);
  bool wait_push_completion(std::optional<int64_t> layer);
  void cancel_preparation();
  void leave_host_batch();
  void drain();

  std::vector<KVCache>* device_caches_;
  PrefillLayerwiseHostPool host_pool_;
  const int32_t kv_split_size_;
  int32_t num_buffers_;
  int64_t block_count_;
  int64_t device_bytes_ = 0;
  std::unique_ptr<BatchMemcpy> batch_memcpy_;
  std::unique_ptr<Stream> load_stream_;
  std::unique_ptr<Stream> save_stream_;
  const Stream* compute_stream_ = nullptr;
  std::vector<StreamEventPtr> load_ready_;
  std::vector<StreamEventPtr> save_ready_;
  std::vector<int64_t> read_blocks_;
  std::vector<int64_t> write_blocks_;
  int64_t current_layer_ = -1;
  int64_t next_load_layer_ = 0;
  bool in_host_batch_ = false;
  bool host_read_write_overlap_ = false;
  bool active_ = false;
  std::atomic<bool> failed_{false};
  std::mutex push_mutex_;
  std::condition_variable push_cv_;
  std::vector<char> push_completed_layers_;
  bool expect_push_ = false;
  bool push_finished_ = true;
};

}  // namespace xllm
