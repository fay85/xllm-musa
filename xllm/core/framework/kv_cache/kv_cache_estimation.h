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

#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"
#include "framework/kv_cache/cache_layout_types.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

class ModelArgs;

struct KVCacheEstimateOptions {
  KPoolCacheLayout kpool_layout = default_kpool_layout();
  torch::ScalarType dtype = torch::kBFloat16;
  std::string kv_cache_dtype = "auto";
  std::string indexer_cache_dtype = "auto";
  int64_t cache_size_in_bytes = 0;
  int64_t block_size = 0;
  int64_t world_size = 1;
  int64_t n_local_kv_heads = 0;
  int64_t n_local_linear_k_heads = 0;
  int64_t n_local_linear_v_heads = 0;
  int64_t max_seqs_per_batch = 0;
  int64_t max_concurrent_requests = 0;
  int64_t num_speculative_tokens = 0;
  int64_t max_tokens_per_batch = 0;
  int64_t max_tokens_per_chunk_for_prefill = 0;
  int64_t max_linear_state_cache_slots = 0;
  bool is_draft_engine = false;
  bool enable_prefix_cache = false;
  int32_t layerwise_split_size = 1;
  int32_t prefill_layerwise_num_buffers = 0;
  int64_t prefill_layerwise_host_memory_bytes = 0;
  bool enable_chunked_prefill = true;
  bool enable_schedule_overlap = true;
  bool enable_disagg_pd = false;
  InstanceRole instance_role = InstanceRole::DEFAULT;
  // DP size used to bound the per-group prefill burst when
  // enable_dp_fair_token_budget caps each DP group at
  // max_tokens_per_batch / dp_size tokens per scheduling round.
  int64_t dp_size = 1;
  bool enable_dp_fair_token_budget = false;
  const ModelArgs* draft_model_args = nullptr;
  const KVCacheEstimateOptions* draft_options = nullptr;
};

struct Dsv4KVCacheEstimateCost {
  int64_t swa_count = 0;
  int64_t block_size = 0;
  int64_t swa_bytes_per_block = 0;
  int64_t constant_swa_bytes = 0;
  int64_t token_unit_bytes = 0;
};

struct SharedKPoolPlan {
  KVCacheCapacity target;
  KVCacheCapacity draft;
};

std::vector<bool> resolve_indexer_cache_enabled_layers(
    const ModelArgs& model_args,
    int64_t num_cache_layers);

// Linear-attention layers stay owned on every rank. Full-attention layers
// follow LayerwiseSplitLayout.
std::vector<bool> build_layer_cache_owned(const ModelArgs& model_args,
                                          const LayerwiseSplitLayout& layout,
                                          int64_t num_layers);

// Common block count across a layerwise split group. Each rank pays for the
// layers it owns plus one shared scratch layer of the same per-block cost, so
// the group settles on the smallest count any rank can afford.
int64_t estimate_layerwise_split_block_count(
    const ModelArgs& model_args,
    int32_t layerwise_split_size,
    const KVCacheCapacity& kv_cache_cap,
    int64_t available_bytes,
    int64_t additional_block_bytes);

KVCacheCapacity estimate_kv_cache_capacity(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options);

// Device bytes per logical block: one resident layer-0 slot plus at most
// `num_buffers` round-robin slots for later layers. Index tensors are allocated
// only for physical buffers whose layer ranges contain an indexer layer.
int64_t prefill_layerwise_device_block_size_in_bytes(
    const ModelArgs& model_args,
    const KVCacheCapacity& kv_cache_cap,
    int32_t num_buffers);

struct PrefillLayerwiseSpecPlan {
  int64_t n_blocks = 0;
  // Device bytes the joint target/draft allocation actually occupies: the
  // target linear-state reserve plus n_blocks of the combined block. Callers
  // write it back to both capacities so the returned budgets match n_blocks,
  // matching the ordinary speculative and plan_shared_kpool conventions.
  int64_t budget_bytes = 0;
};

// Target blocks affordable by the layerwise Device slots while a co-located
// speculative draft keeps its ordinary full cache. The target capacity has
// already applied its Host budget and its own Device-slot limit, so the result
// is additionally capped by those.
PrefillLayerwiseSpecPlan prefill_layerwise_blocks_with_draft(
    const ModelArgs& target_model_args,
    const KVCacheCapacity& target_capacity,
    const KVCacheCapacity& draft_capacity,
    int32_t num_buffers,
    int32_t draft_replicated_block_pages,
    bool draft_body_uses_tp1,
    bool draft_manages_own_kv_shape);

// Co-located KPool drafts inherit the target's request slots and verify window.
// Each capacity retains its own model's layer counts and page costs.
SharedKPoolPlan plan_shared_kpool(KVCacheCapacity target_cap,
                                  KVCacheCapacity draft_cap,
                                  const ModelArgs& draft_args);

// Per-block full-attention KV cost of a speculative target+draft pair.
struct SpeculativeFullAttentionBlockCost {
  int64_t target_full_attention_block_size_in_bytes = 0;
  int64_t draft_full_attention_block_size_in_bytes = 0;
};

// Shared draft-slot sizing for ordinary and layerwise speculative allocation.
// `draft_body_uses_tp1` marks an MTP draft body
// running TP1; `draft_manages_own_kv_shape` mirrors
// SpeculativeConfig::draft_manages_own_kv_shape. Both draft flavors are
// sized by the draft's own slot geometry; only workers that reuse the
// target KVCacheShape require the draft slot to fit within the target's.
// The caller owns the kv_split replicated-draft-page multiplier and the
// block-count division, so this helper deliberately returns costs only.
SpeculativeFullAttentionBlockCost
estimate_speculative_full_attention_block_cost(
    const KVCacheCapacity& target_kv_cache_cap,
    const KVCacheCapacity& draft_kv_cache_cap,
    bool draft_body_uses_tp1,
    bool draft_manages_own_kv_shape);

}  // namespace xllm
