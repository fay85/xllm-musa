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

#include "core/distributed_runtime/engine_utils.h"

#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>
#include <vector>

#include "core/common/constants.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/platform/platform.h"
#include "core/runtime/decode_graph_bucket.h"
#include "core/runtime/options.h"
#include "core/util/utils.h"
#include "framework/block/block_utils.h"
#include "framework/block/hierarchy_block_manager_pool.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "runtime/worker_client.h"

namespace xllm {
namespace {

bool uses_shared_window_cache(const runtime::Options& options,
                              const KVCacheCapacity& capacity) {
  return options.is_draft_engine() && !capacity.has_kv_cache() &&
         capacity.swa_count() > 0;
}

}  // namespace

runtime::Options build_draft_engine_options(const runtime::Options& options,
                                            int64_t world_size) {
  runtime::Options draft_options = options;
  draft_options.model_path(options.draft_model_path().value_or(""))
      .devices(options.draft_devices())
      .backend("llm")
      .num_decoding_tokens(1)
      .enable_speculative_decode(/*enable_speculative_decode=*/false)
      .enable_graph(/*enable_graph=*/false)
      .is_draft_engine(true);
  // Only the controller owns worker clients and plans cache capacity.
  if (options.node_rank() == 0 && options.cp_size() > 1 &&
      Platform::uses_cp_independent_tp_for_block_diffusion_draft() &&
      SpeculativeConfig::is_block_diffusion_algorithm(
          options.speculative_algorithm())) {
    CHECK_GT(world_size, 0);
    CHECK_LE(world_size, std::numeric_limits<int32_t>::max());
    CHECK_GT(options.dp_size(), 0);
    CHECK_EQ(world_size % options.dp_size(), 0);
    // Cache estimation and allocation must use the draft worker's TP domain.
    draft_options.world_size(static_cast<int32_t>(world_size))
        .cp_size(1)
        .tp_size(static_cast<int32_t>(world_size / options.dp_size()));
  }
  return draft_options;
}

runtime::DecodeGraphWarmupConfig build_decode_graph_warmup_config(
    const runtime::Options& options) {
  runtime::DecodeGraphWarmupConfig warmup_config;
  warmup_config.num_decoding_tokens = options.num_decoding_tokens();
  warmup_config.num_speculative_tokens = options.num_speculative_tokens();
  warmup_config.enable_graph_mode_decode_no_padding =
      options.enable_graph_mode_decode_no_padding();
  if (!Platform::is_npu()) {
    return warmup_config;
  }

  const auto& execution_config = ExecutionConfig::get_instance();
  warmup_config.max_graph_batch_size =
      execution_config.acl_graph_decode_batch_size_limit();
  warmup_config.graph_capture_sizes = runtime::get_decode_graph_capture_sizes(
      options, options.num_decoding_tokens());
  return warmup_config;
}

void configure_dsv4_block_manager_options(const ModelArgs& args,
                                          BlockManagerPool::Options& options,
                                          const KVCacheCapacity& kv_cache_cap,
                                          int32_t max_tokens_per_batch) {
  if (!util::is_deepseek_v4_model_type(args.model_type())) {
    return;
  }

  constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
  constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

  std::vector<uint32_t> manager_types{kManagerTypeSlidingWindowBlockManager};
  if (kv_cache_cap.has_kv_cache()) {
    manager_types.push_back(kManagerTypeBlockManagerImpl);
  } else {
    // A standalone SWA ring has no history checkpoint to share as a prefix.
    options.enable_prefix_cache(false);
  }

  const int64_t semantic_window = std::max(args.window_size(), 1);
  const int64_t max_model_len = args.max_seq_len();
  const int64_t effective_window =
      max_model_len > 0 ? std::min<int64_t>(semantic_window, max_model_len)
                        : semantic_window;
  // The pool exposes the window as uint32_t; a value beyond that would silently
  // truncate on the cast below, so reject it like the block-count checks.
  CHECK_LE(effective_window,
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 effective_window exceeds uint32_t range: " << effective_window;
  CHECK_GT(kv_cache_cap.swa_block_size(), 0);
  CHECK_LE(kv_cache_cap.swa_block_size(),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t swa_blocks_per_seq =
      get_swa_blocks_per_seq(effective_window, kv_cache_cap.swa_block_size());
  CHECK_LE(swa_blocks_per_seq,
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 swa_blocks_per_seq exceeds uint32_t range: "
      << swa_blocks_per_seq;
  const uint32_t swa_blocks_per_seq_u32 =
      static_cast<uint32_t>(swa_blocks_per_seq);
  CHECK_LE(kv_cache_cap.swa_count(),
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 swa_count exceeds uint32_t range: " << kv_cache_cap.swa_count();
  CHECK_LE(kv_cache_cap.n_blocks(),
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 n_blocks exceeds uint32_t range: " << kv_cache_cap.n_blocks();
  // The pool exposes this as uint32_t; a negative value would wrap to a huge
  // capacity, so reject it up front like the block-count checks above.
  CHECK_GE(max_tokens_per_batch, 0)
      << "DSV4 max_tokens_per_batch must be >= 0, got " << max_tokens_per_batch;
  CHECK_GT(kv_cache_cap.block_size(), 0);
  CHECK_LE(kv_cache_cap.block_size(),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  options.swa_block_size(static_cast<int32_t>(kv_cache_cap.swa_block_size()))
      .sliding_window_size(static_cast<uint32_t>(effective_window))
      .block_size(static_cast<int32_t>(kv_cache_cap.block_size()))
      .swa_blocks_per_seq(swa_blocks_per_seq_u32)
      .swa_num_blocks(static_cast<uint32_t>(kv_cache_cap.swa_count()))
      .num_blocks(static_cast<uint32_t>(kv_cache_cap.n_blocks()))
      .max_tokens_per_batch(static_cast<uint32_t>(max_tokens_per_batch))
      .manager_types(std::move(manager_types));
}

std::vector<uint32_t> dp_group_worker_indices(uint32_t dp_rank,
                                              uint32_t dp_size,
                                              uint32_t local_worker_count,
                                              size_t worker_count) {
  CHECK_LE(worker_count, std::numeric_limits<uint32_t>::max());
  CHECK_GT(dp_size, 0U);
  CHECK_LT(dp_rank, dp_size);
  CHECK_GT(local_worker_count, 0U);
  CHECK_EQ(static_cast<uint64_t>(local_worker_count) * dp_size, worker_count)
      << "DP-local worker width does not partition the worker group";
  const uint64_t begin = static_cast<uint64_t>(dp_rank) * local_worker_count;

  std::vector<uint32_t> worker_indices;
  worker_indices.reserve(local_worker_count);
  for (uint32_t local_rank = 0; local_rank < local_worker_count; ++local_rank) {
    worker_indices.emplace_back(static_cast<uint32_t>(begin + local_rank));
  }
  return worker_indices;
}

void Engine::init_host_restore_threadpool(const runtime::Options& options,
                                          uint32_t dp_group_size) {
  if (options.host_blocks_factor() > 1.0 && dp_group_size > 0) {
    host_restore_threadpool_ =
        std::make_unique<ThreadPool>(/*num_threads=*/dp_group_size,
                                     /*cpu_binding=*/false,
                                     /*pool_name=*/"Engine.host_restore");
  }
}

void Engine::register_host_restore(
    const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
    const std::vector<uint32_t>& worker_indices,
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return;
  }
  CHECK(host_restore_threadpool_) << "Host restore thread pool is unavailable";

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(kHostRestoreRegistrationTimeoutMs);
  const auto blocks = std::make_shared<const std::vector<BlockTransferInfo>>(
      block_transfer_info);
  std::vector<folly::SemiFuture<folly::Unit>> registrations;
  registrations.reserve(worker_indices.size());
  for (const uint32_t worker_rank : worker_indices) {
    auto [promise, future] = folly::makePromiseContract<folly::Unit>();
    registrations.emplace_back(std::move(future));
    host_restore_threadpool_->schedule([client = worker_clients[worker_rank],
                                        batch_id,
                                        blocks,
                                        promise =
                                            std::move(promise)]() mutable {
      promise.setWith([&] { client->transfer_kv_blocks(batch_id, *blocks); });
    });
  }

  // The driver's SHM input wakes the whole group.
  const auto remaining = std::chrono::duration_cast<folly::HighResDuration>(
      deadline - std::chrono::steady_clock::now());
  CHECK(remaining > folly::HighResDuration::zero())
      << "H2D registration submission timed out for batch_id=" << batch_id;
  const auto completion =
      folly::collectAll(registrations).within(remaining).getTry();
  CHECK(completion.hasValue())
      << "H2D registration wait failed for batch_id=" << batch_id << ": "
      << completion.exception().what();
  const auto& results = completion.value();
  for (size_t i = 0; i < results.size(); ++i) {
    CHECK(results[i].hasValue())
        << "H2D registration failed for batch_id=" << batch_id
        << ", worker_rank=" << worker_indices[i] << ": "
        << results[i].exception().what();
  }
}

void dispatch_storage_prefetch(
    const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
    const std::vector<uint32_t>& worker_indices,
    std::shared_ptr<const StoragePrefetchRequest> request,
    uint32_t batch_size,
    uint32_t configured_timeout_ms,
    PrefetchResult::StopPredicate stop_requested,
    PrefetchResult::DoneCallback done) {
  CHECK(request != nullptr);
  CHECK(request->valid());
  CHECK(!worker_indices.empty());
  // Validate every index before creating a result or dispatching any worker.
  CHECK_LT(*std::max_element(worker_indices.begin(), worker_indices.end()),
           worker_clients.size())
      << "Prefetch worker index exceeds client count";
  // The configured timeout is unsigned; 0 means no deadline for both engines.
  const int64_t timeout_ms = configured_timeout_ms == 0
                                 ? -1
                                 : static_cast<int64_t>(configured_timeout_ms);
  auto result = std::make_shared<PrefetchResult>(worker_indices.size(),
                                                 *request,
                                                 batch_size,
                                                 timeout_ms,
                                                 std::move(stop_requested),
                                                 std::move(done));
  for (size_t local_rank = 0; local_rank < worker_indices.size();
       ++local_rank) {
    worker_clients[worker_indices[local_rank]]->prefetch_from_storage(
        request, result, local_rank);
  }
}

int32_t fold_kv_cache_block_size(int32_t block_size, int32_t kv_split_size) {
  CHECK_GT(block_size, 0);
  CHECK_GT(kv_split_size, 0);
  CHECK_LE(block_size, std::numeric_limits<int32_t>::max() / kv_split_size)
      << "Effective KV block size exceeds int32_t range";
  return block_size * kv_split_size;
}

HostCacheValidationOptions build_host_cache_validation_options(
    const runtime::Options& options,
    const ModelArgs& args,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCapacity& kv_cache_cap,
    int32_t kv_split_size_eff) {
  const bool needs_transfer_folding =
      (Platform::requires_dsa_indexer_cache_replication() &&
       kv_split_size_eff > 1) ||
      (SpeculativeConfig::draft_replicated_block_pages(
           options.speculative_algorithm(), kv_split_size_eff) > 1);
  return {
      .host_blocks_factor = uses_shared_window_cache(options, kv_cache_cap)
                                ? 0.0
                                : options.host_blocks_factor(),
      .device_block_count = kv_cache_cap.n_blocks(),
      .supports_host_kv_offload = Platform::supports_host_kv_offload(),
      .enable_prefix_cache = options.enable_prefix_cache(),
      .enable_disagg_pd = options.enable_disagg_pd(),
      .enable_pd_ooc = options.enable_pd_ooc(),
      .enable_kvcache_store = options.enable_kvcache_store(),
      .instance_role = options.instance_role(),
      .has_key_cache_shape = kv_cache_shape.has_key_cache_shape(),
      .has_grouped_cache_layout = kv_cache_shape.has_grouped_cache_layout(),
      .supports_grouped_cache_offload =
          util::is_deepseek_v4_model_type(args.model_type()),
      .has_conv_cache_shape = kv_cache_shape.has_conv_cache_shape(),
      .has_ssm_cache_shape = kv_cache_shape.has_ssm_cache_shape(),
      .kv_cache_dtype = options.kv_cache_dtype(),
      .indexer_cache_dtype =
          KVCacheConfig::get_instance().indexer_cache_dtype(),
      .model_type = args.model_type(),
      .is_npu_platform = Platform::is_npu(),
      .needs_transfer_folding = needs_transfer_folding,
  };
}

std::unique_ptr<KVCacheManager> create_kv_cache_manager(
    BlockManagerPool::Options& block_options,
    const runtime::Options& runtime_options,
    const KVCacheCapacity& kv_cache_cap,
    bool has_gdn_state,
    Engine* engine,
    uint32_t dp_size) {
  if (runtime_options.enable_kvcache_store()) {
    CHECK_GT(runtime_options.host_blocks_factor(), 1.0)
        << "KV cache Store requires Host cache blocks.";
  }
  const bool is_mtp_draft_engine = runtime_options.is_draft_engine() &&
                                   SpeculativeConfig::is_mtp_algorithm(
                                       runtime_options.speculative_algorithm());
  if (is_mtp_draft_engine) {
    block_options.enable_prefix_cache(false);
  }
  if (runtime_options.host_blocks_factor() <= 1.0 || is_mtp_draft_engine ||
      uses_shared_window_cache(runtime_options, kv_cache_cap)) {
    return std::make_unique<BlockManagerPool>(block_options, dp_size);
  }
  CHECK(kv_cache_cap.has_kv_cache())
      << "Standalone window-only cache does not support Host offload: "
         "restoration requires a KV checkpoint.";

  BlockManagerPool::HostBlockCounts host_capacities;
  const int64_t host_kv_block_count = scale_host_block_count(
      kv_cache_cap.n_blocks(), runtime_options.host_blocks_factor());
  // Preserve scalar capacity only for flat KV without LINEAR state. Hybrid
  // LINEAR layouts always install typed KV capacity from host_kv_block_count;
  // their LINEAR count below must use that same installed KV capacity.
  if (!block_options.manager_types().empty() || has_gdn_state ||
      block_options.host_num_blocks() == 0) {
    CHECK_LE(host_kv_block_count,
             static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
        << "Host KV block count exceeds uint32_t range: "
        << host_kv_block_count;
    host_capacities.emplace(BlockType::KV,
                            static_cast<uint32_t>(host_kv_block_count));
  }
  if (has_gdn_state) {
    int64_t host_linear_blocks = 0;
    if (runtime_options.instance_role() == InstanceRole::DECODE) {
      host_linear_blocks = kv_cache_cap.num_linear_state_blocks() * 2;
    } else {
      CHECK_EQ(kv_cache_cap.block_size(), runtime_options.block_size())
          << "Host LINEAR physical block size must match worker allocation";
      const int64_t chunk_size = static_cast<int64_t>(
          SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill());
      // Host LINEAR rows follow the worker's physical KV width, not the
      // folded logical block-manager width used by grouped KV/SWA pools.
      host_linear_blocks = prefill_host_linear_block_count(
          host_kv_block_count, kv_cache_cap.block_size(), chunk_size);
    }
    CHECK_GE(host_linear_blocks, 0)
        << "Host linear block count must be non-negative: "
        << host_linear_blocks;
    CHECK_LE(host_linear_blocks,
             static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
        << "Host linear block count exceeds uint32_t range: "
        << host_linear_blocks;
    host_capacities.emplace(BlockType::LINEAR,
                            static_cast<uint32_t>(host_linear_blocks));
  }
  if (kv_cache_cap.swa_count() > 0) {
    CHECK_GT(host_kv_block_count, 0)
        << "Host SWA checkpoints require positive KV capacity.";
    const int64_t host_swa_count =
        dsv4_host_swa_block_count(kv_cache_cap.n_blocks(),
                                  runtime_options.host_blocks_factor(),
                                  block_options.sliding_window_size(),
                                  kv_cache_cap.swa_block_size(),
                                  kv_cache_cap.block_size());
    CHECK_LE(host_swa_count,
             static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));
    host_capacities.emplace(BlockType::SWA,
                            static_cast<uint32_t>(host_swa_count));
  }

  block_options.host_num_blocks_by_type(std::move(host_capacities));
  block_options.enable_host_offload(true);
  return std::make_unique<HierarchyBlockManagerPool>(
      block_options, engine, dp_size);
}
}  // namespace xllm
