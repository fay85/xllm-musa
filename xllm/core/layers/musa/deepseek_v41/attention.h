/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/kernels/musa/deepseek_v41_attention_quant.h"
#include "core/layers/musa/deepseek_v41/attention_projection.h"
#include "core/layers/musa/deepseek_v41/compressor.h"
#include "core/layers/musa/deepseek_v41/indexer.h"
#include "core/layers/musa/deepseek_v41/paged_attention.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"
#include "core/layers/musa/deepseek_v41/token_cache.h"

namespace xllm::layer {

enum class DeepseekV41AttentionMode : int8_t { SWA, FULL, REINDEX, REUSE };

struct DeepseekV41AttentionConfig {
  DeepseekV41AttentionMode mode = DeepseekV41AttentionMode::SWA;
  int64_t layer = 0;
  int64_t source_layer = -1;
  int64_t ratio = 0;
  int64_t heads = 8;
  int64_t groups = 2;
  int64_t output_rank = 512;
  int64_t window = 128;
  int64_t index_heads = 32;
  int64_t index_dim = 128;
  int64_t index_topk = 256;
  int64_t candidate_source = 20;
  int64_t candidate_blocks = 8;
  int64_t candidate_block_size = 64;
  double epsilon = 1e-20;
  double scale = 1.0;
  bool original_fp8_activation = true;
  bool decode_cache_read_reuse = false;
  bool decode_fused_swa_read = false;
  std::string sparse_library_path;
  std::string indexer_score_directory;
};

struct DeepseekV41AttentionWeights {
  std::vector<torch::Tensor> projection;
  torch::Tensor sink;
  torch::Tensor compressor_kv;
  torch::Tensor compressor_norm;
  std::optional<torch::Tensor> compressor_gate;
  std::vector<torch::Tensor> indexer;
};

struct DeepseekV41AttentionCache {
  torch::Tensor swa;
  torch::Tensor key;
  torch::Tensor index;
  std::optional<torch::Tensor> compressor_kv_state;
  std::optional<torch::Tensor> compressor_score_state;
};

// Only the CausalLM forward owns this rolling cross-layer handoff. Construct
// a new object for each forward, never retain candidate masks across requests.
// Persistent caches are scheduler owned and are published as tensor handles.
struct DeepseekV41AttentionRuntime {
  int64_t source_layer = -1;
  int64_t ratio = 0;
  torch::Tensor key;
  torch::Tensor key_table;
  torch::Tensor index;
  torch::Tensor index_table;
  std::optional<DeepseekV41TokenCacheReadView> key_read;
  std::optional<DeepseekV41TokenCacheReadView> index_read;
  int64_t materialized_committed = -1;
  int64_t topk_layer = -1;
  torch::Tensor topk;
  int64_t candidates_layer = -1;
  std::optional<torch::Tensor> candidates;
};

// Complete native C1/eager attention, including Full/Reindex/Reuse/SWA,
// compressor state, original quantization, physical caches and shared top-k.
// Returns the local output projection. The Decoder performs its TP reduction
// once; no Python model callback or CPU attention math occurs in this class.
class DeepseekV41Attention final {
 public:
  DeepseekV41Attention(DeepseekV41AttentionConfig config,
                       DeepseekV41AttentionWeights weights)
      : config_(config),
        weights_(std::move(weights)),
        projection_(weights_.projection,
                    config_.heads,
                    config_.groups,
                    config_.output_rank,
                    config_.epsilon,
                    config_.original_fp8_activation) {
    CHECK_GE(config_.layer, 0);
    CHECK_GT(config_.window, 0);
    CHECK(weights_.sink.defined());
    CHECK_EQ(weights_.sink.scalar_type(), torch::kFloat32);
    CHECK_EQ(weights_.sink.dim(), 1);
    CHECK_EQ(weights_.sink.size(0), config_.heads);
    if (config_.mode == DeepseekV41AttentionMode::SWA) {
      CHECK_EQ(config_.ratio, 0);
      CHECK_EQ(config_.source_layer, -1);
      CHECK(weights_.indexer.empty());
      return;
    }
    CHECK(config_.ratio == 1 || config_.ratio == 2);
    CHECK_GE(config_.source_layer, 0);
    CHECK_LE(config_.source_layer, config_.layer);
    CHECK_GT(config_.index_topk, 0);
    if (config_.mode == DeepseekV41AttentionMode::FULL) {
      CHECK_EQ(config_.source_layer, config_.layer);
      CHECK_EQ(weights_.indexer.size(), 4);
    } else {
      CHECK_LT(config_.source_layer, config_.layer);
    }
    if (config_.mode != DeepseekV41AttentionMode::REUSE) {
      indexer_ =
          std::make_unique<DeepseekV41Indexer>(weights_.indexer,
                                               config_.index_heads,
                                               config_.index_dim,
                                               config_.epsilon,
                                               config_.original_fp8_activation,
                                               config_.indexer_score_directory);
    } else {
      CHECK(weights_.indexer.empty());
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& positions,
                        const torch::Tensor& rope_table,
                        const DeepseekV41AttentionCache& cache,
                        const torch::Tensor& swa_table,
                        const torch::Tensor& key_table,
                        const torch::Tensor& index_table,
                        int64_t context_length,
                        DeepseekV41AttentionRuntime& shared) {
    const DeepseekV41ProfileRange range("V41/Attention");
    CHECK_GE(context_length, hidden.size(0));
    auto [query, current_kv, qr] =
        projection_.prepare(hidden, positions, rope_table);
    if (config_.mode == DeepseekV41AttentionMode::FULL) {
      publish(hidden,
              rope_table,
              cache,
              swa_table,
              key_table,
              index_table,
              context_length - hidden.size(0),
              shared);
    }
    std::optional<torch::Tensor> compressed;
    std::optional<torch::Tensor> picks;
    if (config_.mode != DeepseekV41AttentionMode::SWA) {
      CHECK_EQ(shared.source_layer, config_.source_layer);
      CHECK_EQ(shared.ratio, config_.ratio);
      const int64_t committed = context_length / config_.ratio;
      const bool reuse_read =
          config_.decode_cache_read_reuse && hidden.size(0) == 1;
      if (reuse_read) {
        if (!shared.key_read.has_value()) {
          shared.key_read.emplace(
              shared.key, shared.key_table, committed, committed);
          shared.index_read.emplace(
              shared.index, shared.index_table, committed, committed);
          shared.materialized_committed = committed;
        }
        CHECK_EQ(shared.materialized_committed, committed);
      }
      if (indexer_) {
        auto [index_query, head_weights] =
            indexer_->prepare_query(hidden, qr, positions, rope_table);
        const torch::Tensor keys =
            reuse_read ? shared.index_read->materialize()
                       : DeepseekV41TokenCache(shared.index)
                             .gather(shared.index_table, committed, committed);
        const torch::Tensor scores = indexer_->score(index_query,
                                                     keys,
                                                     head_weights,
                                                     positions,
                                                     committed,
                                                     config_.ratio);
        const torch::Tensor visible =
            torch::floor_divide(positions + 1, config_.ratio);
        if (config_.layer == config_.candidate_source) {
          shared.candidates =
              indexer_->candidates(scores,
                                   visible,
                                   config_.candidate_blocks,
                                   config_.candidate_block_size);
          shared.candidates_layer = config_.layer;
        }
        std::optional<torch::Tensor> pool;
        if (config_.layer > config_.candidate_source &&
            config_.candidate_source >= 0) {
          CHECK_EQ(shared.candidates_layer, config_.candidate_source);
          CHECK(shared.candidates.has_value());
          pool = shared.candidates;
        }
        shared.topk =
            indexer_->select(scores, visible, config_.index_topk, pool);
        shared.topk_layer = config_.layer;
      } else {
        CHECK_GE(shared.topk_layer, config_.source_layer);
        CHECK_LT(shared.topk_layer, config_.layer);
        CHECK(shared.topk.defined());
      }
      CHECK_EQ(shared.topk.size(0), hidden.size(0));
      compressed = reuse_read
                       ? shared.key_read->materialize()
                       : DeepseekV41TokenCache(shared.key)
                             .gather(shared.key_table, committed, committed);
      const int64_t padding = config_.index_topk - shared.topk.size(1);
      CHECK_GE(padding, 0);
      // Fixed pick width matches the Python reduction even for zero capacity.
      picks = torch::cat(
          {shared.topk,
           torch::full({hidden.size(0), padding}, -1, shared.topk.options())},
          -1);
    }
    const torch::Tensor attention =
        DeepseekV41PagedAttention(cache.swa,
                                  config_.window,
                                  config_.scale,
                                  config_.sparse_library_path,
                                  /*single_token_scatter=*/true,
                                  config_.decode_fused_swa_read)
            .forward_with_positions(query,
                                    current_kv,
                                    swa_table,
                                    context_length,
                                    weights_.sink,
                                    compressed,
                                    picks,
                                    positions);
    return projection_.project_output(attention, positions, rope_table);
  }

 private:
  void publish(const torch::Tensor& hidden,
               const torch::Tensor& rope_table,
               const DeepseekV41AttentionCache& cache,
               const torch::Tensor& swa_table,
               const torch::Tensor& key_table,
               const torch::Tensor& index_table,
               int64_t start_position,
               DeepseekV41AttentionRuntime& shared) {
    CHECK(indexer_ != nullptr);
    DeepseekV41Compressor compressor(weights_.compressor_kv,
                                     weights_.compressor_norm,
                                     weights_.compressor_gate,
                                     cache.compressor_kv_state,
                                     cache.compressor_score_state,
                                     config_.ratio,
                                     cache.swa.size(1),
                                     config_.epsilon,
                                     /*fused_pair_pool=*/true);
    auto [latent, starts] =
        compressor.forward(hidden, start_position, swa_table);
    const torch::Tensor index_values =
        indexer_->prepare_key(latent, starts, rope_table);
    DeepseekV41TokenCache(cache.index)
        .commit(index_values,
                torch::floor_divide(starts, config_.ratio),
                index_table);
    const torch::Tensor values =
        kernel::musa::deepseek_v41_rope_fp4_quant(latent.contiguous(),
                                                  starts,
                                                  rope_table,
                                                  /*group_size=*/16,
                                                  /*e4m3_scale=*/true);
    DeepseekV41TokenCache(cache.key).commit(
        values, torch::floor_divide(starts, config_.ratio), key_table);
    // Source publication invalidates all materialized read views. Runtime
    // itself is a forward-local object, so no token/request can inherit them.
    shared.key_read.reset();
    shared.index_read.reset();
    shared.materialized_committed = -1;
    shared.key = cache.key;
    shared.key_table = key_table;
    shared.index = cache.index;
    shared.index_table = index_table;
    shared.ratio = config_.ratio;
    shared.source_layer = config_.layer;
    shared.topk = torch::Tensor{};
    shared.topk_layer = -1;
  }

  DeepseekV41AttentionConfig config_;
  DeepseekV41AttentionWeights weights_;
  DeepseekV41AttentionProjection projection_;
  std::unique_ptr<DeepseekV41Indexer> indexer_;
};

}  // namespace xllm::layer
