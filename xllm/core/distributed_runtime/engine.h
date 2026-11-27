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

#include <folly/futures/Future.h>

#include <unordered_map>

#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/config/kv_cache_config.h"
#include "framework/kv_cache_transfer/prefetch_result.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/decode_graph_bucket.h"
#include "runtime/options.h"
#include "util/threadpool.h"

namespace xllm {

class WorkerClient;

class Engine {
 public:
  virtual ~Engine() = default;

  virtual bool init() { return true; };

  virtual bool init(MasterStatus master_status) { return true; };

  // execute model with batch input
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;

  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;

  // return the tokenizer
  virtual const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  // return the block manager
  virtual BlockManagerPool* block_manager_pool() const {
    auto p = reinterpret_cast<BlockManagerPool*>(kv_cache_manager_.get());
    if (!p) {
      LOG(FATAL) << "kv_cache_manager_ is not BlockManagerPool type!";
    }
    return p;
  }

  // return the model args
  virtual const ModelArgs& model_args() const { return args_; }

  virtual runtime::DecodeGraphWarmupConfig decode_graph_warmup_config() const {
    return {};
  }

  // return the tokenizer args
  virtual const TokenizerArgs& tokenizer_args() const {
    return tokenizer_args_;
  }

  // return the active activation memory
  virtual std::vector<int64_t> get_active_activation_memory() const = 0;

  // P/D
  virtual bool pull_kv_blocks(const int32_t src_dp_size,
                              const int32_t src_dp_rank,
                              const std::vector<uint64_t>& src_cluster_ids,
                              const std::vector<std::string>& src_addrs,
                              const int32_t dst_dp_rank,
                              const std::vector<KVTransferMapping>& mappings) {
    NOT_IMPLEMENTED();
    return false;
  };

  // Number of cache shards in one DP group, independent of submitted results.
  virtual uint32_t host_transfer_worker_count() const {
    NOT_IMPLEMENTED();
    return 0;
  }

  virtual std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      const uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    NOT_IMPLEMENTED();
    return {};
  };

  virtual void transfer_kv_blocks(
      const uint32_t dp_rank,
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    NOT_IMPLEMENTED();
  };

  virtual void prefetch_from_storage(
      const uint32_t dp_rank,
      std::shared_ptr<const StoragePrefetchRequest> request,
      PrefetchResult::StopPredicate stop_requested,
      PrefetchResult::DoneCallback done) {
    NOT_IMPLEMENTED();
  };

  virtual void get_cache_info(std::vector<uint64_t>& cluster_ids,
                              std::vector<std::string>& addrs,
                              std::vector<uint16_t>& ports) {
    NOT_IMPLEMENTED();
  };

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports,
                            const int32_t src_dp_size,
                            const int32_t src_kv_split_size = 1) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports,
                              const int32_t src_dp_size,
                              const int32_t src_kv_split_size = 1) {
    NOT_IMPLEMENTED();
    return false;
  };

  // P2P link for weight transfer - each worker links to one remote addr.
  virtual bool sleep(MasterStatus master_status) {
    LOG(FATAL) << " sleep is not implemented!";
    return false;
  };

  virtual bool update_weights(const std::string& weights_path) {
    LOG(FATAL) << " update_weights is not implemented!";
    return false;
  }

  // Multi-tenant LoRA serving: broadcast adapter load / unload to all
  // workers so the CPU->device copy runs on each worker's own
  // threadpool thread (primed device context). Implementations should
  // fan out via RemoteWorker::load_lora_adapter_async and fail if any
  // worker fails. lora_int_id is the API-assigned adapter id every
  // worker must adopt.
  virtual bool load_lora_broadcast(const std::string& lora_name,
                                   const std::string& lora_path,
                                   const std::string& base_model_name,
                                   uint64_t lora_int_id) {
    LOG(FATAL) << " load_lora_broadcast is not implemented!";
    return false;
  }

  virtual bool unload_lora_broadcast(const std::string& lora_name) {
    LOG(FATAL) << " unload_lora_broadcast is not implemented!";
    return false;
  }

  virtual bool wakeup(const WakeupOptions& options) {
    LOG(FATAL) << " wakeup is not implemented!";
    return false;
  };

  // Start/stop online timeline profiling on all workers. CUDA only for now.
  virtual bool start_profile() {
    LOG(ERROR) << "start_profile is not implemented for this engine!";
    return false;
  };

  virtual bool stop_profile() {
    LOG(ERROR) << "stop_profile is not implemented for this engine!";
    return false;
  };

 protected:
  void init_host_restore_threadpool(const runtime::Options& options,
                                    uint32_t dp_group_size);

  void register_host_restore(
      const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
      const std::vector<uint32_t>& worker_indices,
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  void configure_prefix_cache(runtime::Options& options) const {
    if (!options.is_draft_engine()) {
      KVCacheConfig::get_instance().enable_prefix_cache(
          options.enable_prefix_cache());
    }
  }

  // model args
  ModelArgs args_;

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // kv cache manager
  std::unique_ptr<KVCacheManager> kv_cache_manager_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

 private:
  std::unique_ptr<ThreadPool> host_restore_threadpool_;
};
}  // namespace xllm
