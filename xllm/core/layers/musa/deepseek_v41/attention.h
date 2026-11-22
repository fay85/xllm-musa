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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/kernels/musa/deepseek_v41_attention_quant.h"
#include "core/layers/musa/deepseek_v41/attention_projection.h"
#include "core/layers/musa/deepseek_v41/compressor.h"
#include "core/layers/musa/deepseek_v41/indexer.h"
#include "core/layers/musa/deepseek_v41/paged_attention.h"
#include "core/layers/musa/deepseek_v41/prefill_tail.h"
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
  bool prefill_cache_read_reuse = false;
  bool prefill_fused_swa_commit = false;
  bool prefill_wide_swa_commit = false;
  bool short_context_indexer = false;
  bool decode_fused_swa_read = false;
  bool verify_fused_swa_read = false;
  bool decode_packed_qkv = false;
  bool direct_rope_table = false;
  bool prefill_direct_rope_table = false;
  bool prefill_sliced_rope = false;
  bool prefill_query_only = false;
  bool prefill_context_kv_partition = false;
  bool prefill_compressor_projection_partition = false;
  bool verify_skinny_projection = false;
  bool draft_small_commit = false;
  std::string sparse_library_path;
  std::string indexer_score_directory;
  int64_t indexer_max_select_width = 8192;
  std::string dense_prequant_directory;
};

struct DeepseekV41AttentionWeights {
  std::vector<torch::Tensor> projection;
  std::vector<torch::Tensor> projection_fp8_scales;
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
  // Nonzero only for fixed-capacity C1 decode. Real visibility is positions+1.
  int64_t decode_capacity = 0;
  int64_t swa_replay_start = 0;
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
                    config_.original_fp8_activation,
                    config_.decode_packed_qkv,
                    config_.verify_skinny_projection,
                    weights_.projection_fp8_scales,
                    config_.dense_prequant_directory,
                    config_.direct_rope_table,
                    config_.prefill_direct_rope_table,
                    config_.prefill_sliced_rope) {
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
    CHECK_GT(config_.candidate_block_size, 0);
    CHECK_GT(config_.candidate_blocks, 0);
    if (config_.mode == DeepseekV41AttentionMode::FULL) {
      CHECK_EQ(config_.source_layer, config_.layer);
      CHECK_EQ(weights_.indexer.size(), 4);
    } else {
      CHECK_LT(config_.source_layer, config_.layer);
    }
    if (config_.mode != DeepseekV41AttentionMode::REUSE) {
      indexer_ = std::make_unique<DeepseekV41Indexer>(
          weights_.indexer,
          config_.index_heads,
          config_.index_dim,
          config_.epsilon,
          config_.original_fp8_activation,
          config_.indexer_score_directory,
          config_.indexer_max_select_width);
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
        projection_.prepare(hidden,
                            positions,
                            rope_table,
                            /*static_verify=*/shared.decode_capacity > 0);
    return projection_.project_output(attend_prepared(hidden,
                                                      positions,
                                                      rope_table,
                                                      cache,
                                                      swa_table,
                                                      key_table,
                                                      index_table,
                                                      context_length,
                                                      shared,
                                                      query,
                                                      current_kv,
                                                      qr),
                                      positions,
                                      rope_table);
  }

  // Query-only Prefill partition: each rank retains complete scheduler-owned
  // KV/compressor state but evaluates a disjoint query slice with H64 weights.
  // The decoder gathers completed output rows before mHC and MoE. This is not
  // a replacement for distributed KV ownership or a change to PD cache layout.
  torch::Tensor forward_query_partition(const torch::Tensor& hidden,
                                        const torch::Tensor& positions,
                                        const torch::Tensor& rope_table,
                                        const DeepseekV41AttentionCache& cache,
                                        const torch::Tensor& swa_table,
                                        const torch::Tensor& key_table,
                                        const torch::Tensor& index_table,
                                        int64_t context_length,
                                        DeepseekV41AttentionRuntime& shared,
                                        int64_t rank,
                                        int64_t world,
                                        ProcessGroup* context_group = nullptr) {
    const DeepseekV41ProfileRange range("V41/Attention/query_partition");
    CHECK_GT(world, 1);
    CHECK_GE(rank, 0);
    CHECK_LT(rank, world);
    CHECK_GT(hidden.size(0), 6);
    CHECK_LE(hidden.size(0), kDeepseekV41PrefillMaxRows);
    CHECK_EQ(hidden.size(0) % world, 0);
    CHECK_EQ(shared.decode_capacity, 0);
    const int64_t count = hidden.size(0) / world;
    const int64_t offset = rank * count;
    const torch::Tensor local_hidden = hidden.narrow(0, offset, count);
    const torch::Tensor local_positions = positions.narrow(0, offset, count);
    torch::Tensor query;
    torch::Tensor qr;
    torch::Tensor unused_kv;
    {
      const DeepseekV41ProfileRange query_range(
          "V41/Attention/local_projection");
      if (config_.prefill_query_only) {
        std::tie(query, qr) = projection_.prepare_query(
            local_hidden, local_positions, rope_table);
      } else {
        std::tie(query, unused_kv, qr) =
            projection_.prepare(local_hidden, local_positions, rope_table);
      }
    }
    torch::Tensor full_kv;
    {
      const DeepseekV41ProfileRange context_range("V41/Attention/context_kv");
      if (config_.prefill_context_kv_partition && context_group != nullptr &&
          hidden.size(0) > (world - 1) * kDeepseekV41PrefillTileRows) {
        CHECK_EQ(context_group->rank(), rank);
        CHECK_EQ(context_group->world_size(), world);
        full_kv = prepare_partitioned_context_kv(
            hidden, positions, rope_table, unused_kv, *context_group);
      } else if (hidden.size(0) <= kDeepseekV41PrefillTileRows) {
        full_kv = projection_.prepare_context_kv(hidden, positions, rope_table);
      } else {
        // Preserve the projection shapes of the original SWA writes.
        std::vector<torch::Tensor> values;
        deepseek_v41_for_each_prefill_tile(
            hidden.size(0), [&](int64_t start, int64_t count) {
              values.emplace_back(projection_.prepare_context_kv(
                  hidden.narrow(0, start, count),
                  positions.narrow(0, start, count),
                  rope_table));
            });
        full_kv = torch::cat(values, 0);
      }
    }
    if (config_.mode == DeepseekV41AttentionMode::FULL) {
      const int64_t previous = context_length - hidden.size(0);
      if (config_.prefill_compressor_projection_partition &&
          context_group != nullptr &&
          hidden.size(0) > (world - 1) * kDeepseekV41PrefillTileRows) {
        CHECK_EQ(context_group->rank(), rank);
        CHECK_EQ(context_group->world_size(), world);
        publish_partitioned_compressor(hidden,
                                       rope_table,
                                       cache,
                                       swa_table,
                                       key_table,
                                       index_table,
                                       previous,
                                       shared,
                                       *context_group);
      } else if (hidden.size(0) <= kDeepseekV41PrefillTileRows) {
        publish(hidden,
                positions,
                rope_table,
                cache,
                swa_table,
                key_table,
                index_table,
                previous,
                shared);
      } else {
        // Pair pooling stores only its current tail. Keep all original
        // write boundaries before evaluating attention over the wide chunk.
        deepseek_v41_for_each_prefill_tile(
            hidden.size(0), [&](int64_t start, int64_t count) {
              publish(hidden.narrow(0, start, count),
                      positions.narrow(0, start, count),
                      rope_table,
                      cache,
                      swa_table,
                      key_table,
                      index_table,
                      previous + start,
                      shared);
            });
      }
    }
    const torch::Tensor attended = attend_prepared(local_hidden,
                                                   local_positions,
                                                   rope_table,
                                                   cache,
                                                   swa_table,
                                                   key_table,
                                                   index_table,
                                                   context_length,
                                                   shared,
                                                   query,
                                                   full_kv,
                                                   qr,
                                                   offset);
    const DeepseekV41ProfileRange output_range(
        "V41/Attention/output_projection");
    return projection_.project_output(attended, local_positions, rope_table);
  }

  // Only true multi-request q1 decode. Projection GEMMs run once over B;
  // physical cache, compressor, indexer and CSA2 runtime stay request-local.
  torch::Tensor forward_batched_decode(
      const torch::Tensor& hidden,
      const torch::Tensor& positions,
      const torch::Tensor& rope_table,
      const DeepseekV41AttentionCache& cache,
      const torch::Tensor& swa_table,
      const torch::Tensor& key_table,
      const torch::Tensor& index_table,
      const std::vector<int64_t>& context_lengths,
      std::vector<DeepseekV41AttentionRuntime>& runtime) {
    const DeepseekV41ProfileRange range("V41/Attention/batched_decode");
    const int64_t batch = hidden.size(0);
    CHECK_GT(batch, 1);
    CHECK_LE(batch, 8);
    CHECK_EQ(positions.numel(), batch);
    CHECK_EQ(swa_table.dim(), 2);
    CHECK_EQ(swa_table.size(0), batch);
    for (const auto& table : {key_table, index_table}) {
      if (table.defined()) {
        CHECK_EQ(table.dim(), 2);
        CHECK_EQ(table.size(0), batch);
      }
    }
    CHECK_EQ(context_lengths.size(), static_cast<size_t>(batch));
    CHECK_EQ(runtime.size(), static_cast<size_t>(batch));
    auto [query, current_kv, qr] =
        projection_.prepare(hidden, positions, rope_table);
    const auto table_row = [](const torch::Tensor& table, int64_t row) {
      return table.defined() ? table.narrow(0, row, 1) : torch::Tensor{};
    };
    std::vector<torch::Tensor> attention;
    attention.reserve(batch);
    for (int64_t row = 0; row < batch; ++row) {
      CHECK_GT(context_lengths[row], 0);
      attention.emplace_back(attend_prepared(hidden.narrow(0, row, 1),
                                             positions.narrow(0, row, 1),
                                             rope_table,
                                             cache,
                                             table_row(swa_table, row),
                                             table_row(key_table, row),
                                             table_row(index_table, row),
                                             context_lengths[row],
                                             runtime[row],
                                             query.narrow(0, row, 1),
                                             current_kv.narrow(0, row, 1),
                                             qr.narrow(0, row, 1)));
    }
    return projection_.project_output(
        torch::cat(attention, 0), positions, rope_table);
  }

  torch::Tensor forward_draft_block(const torch::Tensor& hidden,
                                    const torch::Tensor& positions,
                                    const torch::Tensor& rope_table,
                                    const torch::Tensor& swa_cache,
                                    const torch::Tensor& swa_table,
                                    int64_t previous_context) {
    CHECK(config_.mode == DeepseekV41AttentionMode::SWA);
    CHECK_EQ(config_.ratio, 0);
    auto [query, current_kv, ignored_qr] =
        projection_.prepare(hidden, positions, rope_table);
    const torch::Tensor attention =
        DeepseekV41PagedAttention(swa_cache,
                                  config_.window,
                                  config_.scale,
                                  config_.sparse_library_path)
            .forward_draft_block(
                query, current_kv, swa_table, previous_context, weights_.sink);
    return projection_.project_output(attention, positions, rope_table);
  }

  torch::Tensor forward_draft_block_static(
      const torch::Tensor& hidden,
      const torch::Tensor& positions,
      const torch::Tensor& rope_table,
      const torch::Tensor& swa_cache,
      const torch::Tensor& swa_table,
      const torch::Tensor& previous_context) {
    CHECK(config_.mode == DeepseekV41AttentionMode::SWA);
    CHECK_EQ(config_.ratio, 0);
    auto [query, current_kv, ignored_qr] =
        projection_.prepare(hidden, positions, rope_table);
    const torch::Tensor attention =
        DeepseekV41PagedAttention(swa_cache,
                                  config_.window,
                                  config_.scale,
                                  config_.sparse_library_path)
            .forward_draft_block_static(
                query, current_kv, swa_table, previous_context, weights_.sink);
    return projection_.project_output(attention, positions, rope_table);
  }

  void write_draft_context(const torch::Tensor& hidden,
                           const torch::Tensor& positions,
                           const torch::Tensor& rope_table,
                           const torch::Tensor& swa_cache,
                           const torch::Tensor& cache_slots) {
    CHECK(config_.mode == DeepseekV41AttentionMode::SWA);
    CHECK_EQ(config_.ratio, 0);
    const torch::Tensor values =
        projection_.prepare_context_kv(hidden, positions, rope_table);
    DeepseekV41PagedAttention(
        swa_cache, config_.window, config_.scale, config_.sparse_library_path)
        .commit_draft_context(cache_slots, values, config_.draft_small_commit);
  }

 private:
  torch::Tensor attend_prepared(const torch::Tensor& hidden,
                                const torch::Tensor& positions,
                                const torch::Tensor& rope_table,
                                const DeepseekV41AttentionCache& cache,
                                const torch::Tensor& swa_table,
                                const torch::Tensor& key_table,
                                const torch::Tensor& index_table,
                                int64_t context_length,
                                DeepseekV41AttentionRuntime& shared,
                                const torch::Tensor& query,
                                const torch::Tensor& current_kv,
                                const torch::Tensor& qr,
                                int64_t prefill_query_offset = -1) {
    if (config_.mode == DeepseekV41AttentionMode::FULL &&
        prefill_query_offset < 0) {
      publish(hidden,
              positions,
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
      const bool static_decode = shared.decode_capacity > 0;
      CHECK(!static_decode || (hidden.size(0) > 0 && hidden.size(0) <= 6));
      const int64_t committed =
          (static_decode ? shared.decode_capacity : context_length) /
          config_.ratio;
      const std::optional<torch::Tensor> dynamic_visible =
          static_decode ? std::make_optional(torch::floor_divide(
                              positions.narrow(0, positions.numel() - 1, 1) + 1,
                              config_.ratio))
                        : std::nullopt;
      const bool reuse_read = (config_.decode_cache_read_reuse &&
                               (hidden.size(0) == 1 || static_decode)) ||
                              (config_.prefill_cache_read_reuse &&
                               !static_decode && hidden.size(0) > 1);
      CHECK(!static_decode || reuse_read);
      if (reuse_read) {
        if (!shared.key_read.has_value()) {
          shared.key_read.emplace(shared.key,
                                  shared.key_table,
                                  committed,
                                  committed,
                                  dynamic_visible);
          shared.index_read.emplace(shared.index,
                                    shared.index_table,
                                    committed,
                                    committed,
                                    dynamic_visible);
          shared.materialized_committed = committed;
        }
        CHECK_EQ(shared.materialized_committed, committed);
      }
      if (indexer_) {
        const bool all_fit = config_.short_context_indexer && static_decode &&
                             committed <= config_.index_topk &&
                             (committed + config_.candidate_block_size - 1) /
                                     config_.candidate_block_size <=
                                 config_.candidate_blocks;
        if (all_fit) {
          // Every reachable compressed row fits in both selection budgets.
          // Compression/index-K publication above remains active so a later
          // graph bucket can use the ordinary scored path without lost state.
          const torch::Tensor visible =
              torch::floor_divide(positions + 1, config_.ratio);
          std::optional<torch::Tensor> pool;
          if (config_.layer > config_.candidate_source &&
              config_.candidate_source >= 0) {
            CHECK_EQ(shared.candidates_layer, config_.candidate_source);
            CHECK(shared.candidates.has_value());
            pool = shared.candidates;
          }
          auto [indices, candidates] = DeepseekV41Indexer::select_all_visible(
              visible, committed, config_.candidate_block_size, pool);
          shared.topk = std::move(indices);
          if (config_.layer == config_.candidate_source) {
            shared.candidates = std::move(candidates);
            shared.candidates_layer = config_.layer;
          }
        } else {
          auto [index_query, head_weights] =
              indexer_->prepare_query(hidden, qr, positions, rope_table);
          const torch::Tensor keys =
              reuse_read
                  ? shared.index_read->materialize()
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
        }
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
                                  config_.decode_fused_swa_read,
                                  config_.verify_fused_swa_read,
                                  config_.prefill_fused_swa_commit,
                                  config_.prefill_wide_swa_commit)
            .forward_with_window_floor(
                query,
                current_kv,
                swa_table,
                context_length,
                weights_.sink,
                compressed,
                picks,
                positions,
                shared.swa_replay_start,
                /*static_verify=*/shared.decode_capacity > 0,
                prefill_query_offset);
    return attention;
  }

  static int64_t prefill_projection_rank_rows(int64_t rows, int64_t world) {
    constexpr int64_t tile = kDeepseekV41PrefillTileRows;
    return ((rows + world * tile - 1) / (world * tile)) * tile;
  }

  torch::Tensor prepare_partitioned_context_kv(const torch::Tensor& hidden,
                                               const torch::Tensor& positions,
                                               const torch::Tensor& rope_table,
                                               const torch::Tensor& query_kv,
                                               ProcessGroup& group) const {
    const DeepseekV41ProfileRange range("V41/Attention/context_kv_partitioned");
    constexpr int64_t tile = kDeepseekV41PrefillTileRows;
    const int64_t world = group.world_size();
    const int64_t sequence_rows = hidden.size(0);
    const int64_t rank_rows =
        prefill_projection_rank_rows(sequence_rows, world);
    const int64_t start = group.rank() * rank_rows;
    const int64_t rows =
        std::clamp<int64_t>(sequence_rows - start, 0, rank_rows);
    torch::Tensor local;
    // With one full tile per rank, the query path already computed this
    // rank's context KV with the identical 512-row projection.
    if (sequence_rows == world * tile && query_kv.defined()) {
      local = query_kv;
    } else {
      std::vector<torch::Tensor> values;
      deepseek_v41_for_each_prefill_tile(
          rows, [&](int64_t offset, int64_t count) {
            values.emplace_back(projection_.prepare_context_kv(
                hidden.narrow(0, start + offset, count),
                positions.narrow(0, start + offset, count),
                rope_table));
          });
      if (!values.empty()) {
        local = values.size() == 1 ? values.front() : torch::cat(values, 0);
      }
    }
    const int64_t channels = weights_.projection[3].size(0);
    if (rows < rank_rows) {
      torch::Tensor padded =
          torch::zeros({rank_rows, channels}, hidden.options());
      if (rows != 0) {
        padded.narrow(0, 0, rows).copy_(local);
      }
      local = std::move(padded);
    }
    CHECK_EQ(local.sizes(), torch::IntArrayRef({rank_rows, channels}));
    const DeepseekV41ProfileRange gather("V41/Attention/context_kv_allgather");
    return group.allgather_base_sync(local.contiguous())
        .reshape({world * rank_rows, channels})
        .narrow(0, 0, sequence_rows);
  }

  DeepseekV41Compressor make_compressor(
      const DeepseekV41AttentionCache& cache) {
    return DeepseekV41Compressor(weights_.compressor_kv,
                                 weights_.compressor_norm,
                                 weights_.compressor_gate,
                                 cache.compressor_kv_state,
                                 cache.compressor_score_state,
                                 config_.ratio,
                                 cache.swa.size(1),
                                 config_.epsilon,
                                 /*fused_pair_pool=*/true);
  }

  void publish_partitioned_compressor(const torch::Tensor& hidden,
                                      const torch::Tensor& rope_table,
                                      const DeepseekV41AttentionCache& cache,
                                      const torch::Tensor& swa_table,
                                      const torch::Tensor& key_table,
                                      const torch::Tensor& index_table,
                                      int64_t start_position,
                                      DeepseekV41AttentionRuntime& shared,
                                      ProcessGroup& group) {
    const DeepseekV41ProfileRange range(
        "V41/Attention/compressor_partitioned_publish");
    CHECK(indexer_ != nullptr);
    CHECK_EQ(shared.decode_capacity, 0);
    auto compressor = make_compressor(cache);
    const int64_t world = group.world_size();
    const int64_t total = hidden.size(0);
    const int64_t rank_rows = prefill_projection_rank_rows(total, world);
    const int64_t start = group.rank() * rank_rows;
    const int64_t rows = std::clamp<int64_t>(total - start, 0, rank_rows);
    const int64_t channels = weights_.compressor_kv.size(0);
    const bool paired = config_.ratio == 2;
    const int64_t width = channels * (paired ? 2 : 1);
    torch::Tensor local;
    {
      const DeepseekV41ProfileRange projection_range(
          "V41/Attention/compressor_local_projection");
      std::vector<torch::Tensor> tiles;
      deepseek_v41_for_each_prefill_tile(
          rows, [&](int64_t offset, int64_t count) {
            auto [values, scores] = compressor.project_prefill(
                hidden.narrow(0, start + offset, count));
            tiles.emplace_back(paired ? torch::cat({values, scores}, 1)
                                      : std::move(values));
          });
      if (!tiles.empty()) {
        local =
            tiles.size() == 1 ? std::move(tiles.front()) : torch::cat(tiles, 0);
      }
      if (rows < rank_rows) {
        const auto options =
            hidden.options().dtype(paired ? torch::kFloat32 : torch::kBFloat16);
        auto padded = torch::zeros({rank_rows, width}, options);
        if (rows != 0) {
          padded.narrow(0, 0, rows).copy_(local);
        }
        local = std::move(padded);
      }
    }
    CHECK_EQ(local.sizes(), torch::IntArrayRef({rank_rows, width}));
    torch::Tensor projected;
    {
      const DeepseekV41ProfileRange gather(
          "V41/Attention/compressor_projection_allgather");
      projected = group.allgather_base_sync(local.contiguous())
                      .reshape({world * rank_rows, width})
                      .narrow(0, 0, total);
    }
    // Projection is row independent. Run the original stateful pool and
    // cache commits in the original tile order using exactly the same rows.
    deepseek_v41_for_each_prefill_tile(
        total, [&](int64_t offset, int64_t count) {
          const auto tile = projected.narrow(0, offset, count);
          const auto values = tile.narrow(1, 0, channels).contiguous();
          const auto scores =
              paired ? tile.narrow(1, channels, channels).contiguous()
                     : torch::Tensor{};
          auto [latent, starts] =
              compressor.forward_projected(hidden.narrow(0, offset, count),
                                           values,
                                           scores,
                                           start_position + offset,
                                           swa_table);
          publish_compressed(latent,
                             starts,
                             rope_table,
                             cache,
                             key_table,
                             index_table,
                             shared);
        });
  }

  void publish(const torch::Tensor& hidden,
               const torch::Tensor& positions,
               const torch::Tensor& rope_table,
               const DeepseekV41AttentionCache& cache,
               const torch::Tensor& swa_table,
               const torch::Tensor& key_table,
               const torch::Tensor& index_table,
               int64_t start_position,
               DeepseekV41AttentionRuntime& shared) {
    CHECK(indexer_ != nullptr);
    auto compressor = make_compressor(cache);
    auto [latent, starts] =
        shared.decode_capacity > 0
            ? compressor.forward_static(
                  hidden, positions, start_position % 2 == 1, swa_table)
            : compressor.forward(hidden, start_position, swa_table);
    publish_compressed(
        latent, starts, rope_table, cache, key_table, index_table, shared);
  }

  void publish_compressed(const torch::Tensor& latent,
                          const torch::Tensor& starts,
                          const torch::Tensor& rope_table,
                          const DeepseekV41AttentionCache& cache,
                          const torch::Tensor& key_table,
                          const torch::Tensor& index_table,
                          DeepseekV41AttentionRuntime& shared) {
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
