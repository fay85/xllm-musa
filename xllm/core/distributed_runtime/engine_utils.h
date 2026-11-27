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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "framework/block/block_manager_pool.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/prefetch_result.h"
#include "framework/model/model_args.h"

namespace xllm {

class Engine;
class WorkerClient;

namespace runtime {
class Options;
struct DecodeGraphWarmupConfig;
}  // namespace runtime

// Builds draft options from the target options. On the controller
// (node_rank == 0), MLU block-diffusion drafts with CP > 1 merge CP into TP,
// preserving DP: cp_size = 1 and tp_size = world_size / dp_size. Here,
// world_size is the actual worker client count, so cache sizing follows the
// draft worker layout.
runtime::Options build_draft_engine_options(const runtime::Options& options,
                                            int64_t world_size);

// Shared by text and multimodal engines so synthetic warmup uses the same
// NPU decode token shapes as the executor. Other backends keep their policy.
runtime::DecodeGraphWarmupConfig build_decode_graph_warmup_config(
    const runtime::Options& options);

// Configures `options` for the DeepSeek-V4 family (V4 and V4.1): DeepSeek-V4
// serves its compressed KV through a composite block manager (SWA + one unified
// KV manager) plus the sliding-window pool sizing
// derived from `kv_cache_cap`. Without this, a plain single-KV pool is built,
// the sequence never owns composite (SWA/KV) blocks,
// `multi_block_tables` stays empty, and the DSA attention loses its block
// tables -> garbage output even for pure-text requests.
//
// Extracted from LLMEngine's inline KV-block-manager configuration and shared
// by both LLMEngine and VLMEngine (a past VLM-side omission of this block
// caused exactly that drift). No-op for non-DSV4 model types.
void configure_dsv4_block_manager_options(const ModelArgs& args,
                                          BlockManagerPool::Options& options,
                                          const KVCacheCapacity& kv_cache_cap,
                                          int32_t max_tokens_per_batch);

// Full DP-local worker group, checked against the actual client count.
std::vector<uint32_t> dp_group_worker_indices(uint32_t dp_rank,
                                              uint32_t dp_size,
                                              uint32_t local_worker_count,
                                              size_t worker_count);

// Shares request validation, timeout (0 means no deadline), result aggregation,
// and local worker ordinals between text and multimodal engines.
void dispatch_storage_prefetch(
    const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
    const std::vector<uint32_t>& worker_indices,
    std::shared_ptr<const StoragePrefetchRequest> request,
    uint32_t batch_size,
    uint32_t configured_timeout_ms,
    PrefetchResult::StopPredicate stop_requested,
    PrefetchResult::DoneCallback done);

// Logical block-manager width for kv-split caches; tensor allocation keeps
// its physical width. Shared by text and multimodal engine setup.
int32_t fold_kv_cache_block_size(int32_t block_size, int32_t kv_split_size);

HostCacheValidationOptions build_host_cache_validation_options(
    const runtime::Options& options,
    const ModelArgs& args,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCapacity& kv_cache_cap,
    int32_t kv_split_size_eff);

// Keeps caller-prepared flat capacities and the MTP-draft policy.
std::unique_ptr<KVCacheManager> create_kv_cache_manager(
    BlockManagerPool::Options& block_options,
    const runtime::Options& runtime_options,
    const KVCacheCapacity& kv_cache_cap,
    bool has_gdn_state,
    Engine* engine,
    uint32_t dp_size);

}  // namespace xllm
