// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "core/kernels/musa/deepseek_v4_indexer_metadata.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_graph_sequence.h"

namespace xllm::layer {

// Experimental multi-sequence decode implementation. Capture shapes are fixed;
// row identity, valid history and SWA length are all evaluated on device on
// every replay. INT8 keys round to the query dtype before FP32 accumulation,
// matching the paged Indexer. Call only after publishing current-step cache.
inline torch::Tensor dsv4_batched_indexer_gemm(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& scales,
    const torch::Tensor& blocks,
    int64_t query_rows_per_sequence,
    int64_t capacity,
    int64_t topk,
    const torch::Tensor& output,
    const torch::Tensor& compressed_blocks,
    const torch::Tensor& swa_blocks,
    int64_t window,
    int64_t mate_block_size,
    const torch::Tensor& physical_output,
    const torch::Tensor& extra_output,
    const torch::Tensor& lengths) {
  CHECK_EQ(query.dim(), 3);
  CHECK_GT(query_rows_per_sequence, 0);
  CHECK_EQ(query.size(0) % query_rows_per_sequence, 0);
  // Production stores [blocks, page, 1, dim]; standalone cache fixtures also
  // use [blocks, page, dim]. Both have exactly one key vector per cache slot.
  CHECK(cache.dim() == 3 || (cache.dim() == 4 && cache.size(2) == 1));
  CHECK_EQ(cache.scalar_type(), torch::kInt8);
  CHECK_EQ(cache.size(-1), query.size(2));
  CHECK(cache.is_contiguous());
  CHECK_EQ(scales.numel(), cache.size(0) * cache.size(1));
  CHECK_GT(cache.size(0), 0);
  CHECK_EQ(weights.sizes(), query.sizes().slice(0, 2));
  CHECK_EQ(lengths.numel(), query.size(0));
  CHECK_EQ(output.sizes(), torch::IntArrayRef({query.size(0), topk}));
  CHECK_EQ(output.scalar_type(), torch::kInt32);
  CHECK_EQ(physical_output.sizes(), output.sizes());
  CHECK_EQ(extra_output.sizes(), torch::IntArrayRef({query.size(0), 128}));
  CHECK_GE(capacity, topk);
  CHECK_EQ(topk, 512);
  CHECK_GT(mate_block_size, 0);
  const int64_t batch = query.size(0) / query_rows_per_sequence;
  const int64_t heads = query.size(1);
  const int64_t dim = query.size(2);
  const int64_t page = cache.size(1);
  const int64_t cache_slots = cache.size(0) * page;
  CHECK_EQ(blocks.dim(), 2);
  CHECK_GE(blocks.size(0), batch);
  CHECK_GE(blocks.size(1) * page, capacity);
  for (const auto& table : {compressed_blocks, swa_blocks}) {
    CHECK_EQ(table.dim(), 2);
    CHECK_GE(table.size(0), batch);
    CHECK_GT(table.size(1), 0);
  }
  const char* metadata_env = std::getenv("XLLM_DSV4_FUSED_INDEXER_METADATA");
  const bool fused_metadata =
      metadata_env != nullptr && std::string(metadata_env) == "1" &&
      query.scalar_type() == torch::kBFloat16 && dim == 128 &&
      scales.scalar_type() == torch::kFloat16 && scales.is_contiguous() &&
      lengths.scalar_type() == torch::kInt32 && lengths.is_contiguous() &&
      blocks.is_contiguous() && compressed_blocks.is_contiguous() &&
      swa_blocks.is_contiguous();
  const char* single_bf16_env =
      dsv4_single_indexer_bf16_fp32_shape(
          batch, query_rows_per_sequence, heads, dim, capacity)
          ? std::getenv("XLLM_DSV4_SINGLE_INDEXER_BF16_FP32") : nullptr;
  const bool single_bf16 = single_bf16_env != nullptr &&
      single_bf16_env[0] == '1' && single_bf16_env[1] == '\0';
  const char* bf16_fp32_env = batch > 1 || single_bf16
      ? std::getenv("XLLM_DSV4_INDEXER_BF16_FP32") : nullptr;
  const bool bf16_fp32 = fused_metadata &&
      (single_bf16 || dsv4_batched_indexer_bf16_fp32_shape(
          batch, query_rows_per_sequence, heads, dim)) &&
      bf16_fp32_env != nullptr && bf16_fp32_env[0] == '1' &&
      bf16_fp32_env[1] == '\0';
  const char* bf16_keys_env = bf16_fp32
      ? std::getenv("XLLM_DSV4_INDEXER_BF16_KEYS") : nullptr;
  const bool bf16_keys = bf16_keys_env != nullptr &&
      bf16_keys_env[0] == '1' && bf16_keys_env[1] == '\0';
  auto options = query.options().dtype(torch::kInt64);
  torch::Tensor positions;
  torch::Tensor valid_slots;
  torch::Tensor keys;
  if (fused_metadata) {
    keys = torch::empty({batch, capacity, dim},
                        query.options().dtype(bf16_keys ? torch::kBFloat16 : torch::kFloat32));
    valid_slots =
        torch::empty({batch, capacity}, query.options().dtype(torch::kBool));
    if (bf16_keys) {
      kernel::musa::deepseek_v4_indexer_gather_keys_bf16(
          cache, scales, blocks.narrow(0, 0, batch), keys, valid_slots);
      LOG_FIRST_N(INFO, 2) << "[DSV4_INDEXER_BF16_KEYS] sequences=" << batch
                           << " capacity=" << capacity << " unchanged_rounding=1";
    } else {
      kernel::musa::deepseek_v4_indexer_gather_keys_fp32(
          cache, scales, blocks.narrow(0, 0, batch), keys, valid_slots);
    }
    LOG_FIRST_N(INFO, 2) << "[DSV4_FUSED_INDEXER_METADATA] sequences=" << batch
                         << " query_width=" << query_rows_per_sequence
                         << " capacity=" << capacity
                         << " fp32_gemm_unchanged=" << !bf16_fp32;
  } else {
    positions = torch::arange(capacity, options);
    auto table_columns = positions.div(page, "floor");
    auto physical_blocks = blocks.narrow(0, 0, batch)
                               .index_select(1, table_columns)
                               .to(torch::kInt64);
    auto slots = physical_blocks * page + positions.remainder(page);
    // Preserve the existing cache gather's one-based terminal-page
    // compatibility.
    slots =
        torch::where(slots.ge(cache_slots).logical_and(physical_blocks.gt(0)),
                     slots - page,
                     slots);
    valid_slots = slots.ge(0).logical_and(slots.lt(cache_slots));
    auto safe_slots = slots.clamp(0, cache_slots - 1).reshape({-1});
    keys = cache.reshape({cache_slots, dim})
               .index_select(0, safe_slots)
               .to(torch::kFloat32);
    auto key_scales =
        scales.reshape({-1}).index_select(0, safe_slots).to(torch::kFloat32);
    keys.mul_(key_scales.unsqueeze(1));
    keys = keys.to(query.scalar_type()).to(torch::kFloat32);
  }
  torch::Tensor scores;
  if (bf16_fp32) {
    // Keys already contain BF16-rounded values; this cast adds no rounding.
    // Use three-argument bmm_out with FP32 storage. This runtime's separate
    // bmm.dtype overload swaps alpha/beta and must not be used.
    auto query_bf16 =
        query.reshape({batch, query_rows_per_sequence * heads, dim});
    auto keys_bf16 = keys.to(torch::kBFloat16).reshape({batch, capacity, dim});
    scores = torch::empty({batch, query_rows_per_sequence * heads, capacity},
                           query.options().dtype(torch::kFloat32));
    torch::bmm_out(scores, query_bf16, keys_bf16.transpose(1, 2));
    if (single_bf16) {
      LOG_FIRST_N(INFO, 2) << "[DSV4_SINGLE_INDEXER_BF16_FP32] capacity="
                           << capacity << " query_width=" << query_rows_per_sequence
                           << " checked_bmm_out=1 fp32_scores=1";
    }
    LOG_FIRST_N(INFO, 2) << "[DSV4_INDEXER_BF16_FP32] sequences=" << batch
                         << " query_width=" << query_rows_per_sequence
                         << " capacity=" << capacity << " checked_bmm_out=1";
  } else {
    scores = torch::bmm(query.to(torch::kFloat32)
                     .reshape({batch, query_rows_per_sequence * heads, dim}),
                 keys.reshape({batch, capacity, dim}).transpose(1, 2));
  }
  scores = scores.view({batch, query_rows_per_sequence, heads, capacity});
  const char* relu_weight_env = bf16_fp32 &&
      dsv4_indexer_relu_weight_shape(
          batch, query_rows_per_sequence, heads, dim, capacity)
      ? std::getenv("XLLM_DSV4_INDEXER_RELU_WEIGHT") : nullptr;
  const bool fused_relu_weight = relu_weight_env != nullptr &&
      relu_weight_env[0] == '1' && relu_weight_env[1] == '\0';
  if (fused_relu_weight) {
    kernel::musa::deepseek_v4_indexer_relu_weight(
        scores, weights.to(torch::kFloat32).contiguous());
    LOG_FIRST_N(INFO, 2) << "[DSV4_INDEXER_RELU_WEIGHT] sequences=" << batch
                         << " capacity=" << capacity << " native_head_sum=1";
  } else {
    scores.relu_();
    scores.mul_(weights.to(torch::kFloat32)
                    .view({batch, query_rows_per_sequence, heads, 1}));
  }
  scores = scores.sum(2).view({query.size(0), capacity});
  if (fused_metadata) {
    kernel::musa::deepseek_v4_indexer_mask_scores(
        scores, lengths, valid_slots, query_rows_per_sequence);
    auto selected = std::get<1>(scores.topk(topk,
                                            /*dim=*/1,
                                            /*largest=*/true,
                                            /*sorted=*/false));
    kernel::musa::deepseek_v4_indexer_map_outputs(selected,
                                                  lengths,
                                                  compressed_blocks,
                                                  swa_blocks,
                                                  query_rows_per_sequence,
                                                  capacity,
                                                  window,
                                                  mate_block_size,
                                                  output,
                                                  physical_output,
                                                  extra_output);
    return output;
  }
  auto live = lengths.to(torch::kInt64).clamp_min(0);
  auto valid = live.div(4, "floor").clamp_max(capacity);
  auto owners =
      torch::arange(batch, options).repeat_interleave(query_rows_per_sequence);
  auto allowed = positions.unsqueeze(0)
                     .lt(valid.unsqueeze(1))
                     .logical_and(valid_slots.index_select(0, owners));
  scores.masked_fill_(allowed.logical_not(),
                      -std::numeric_limits<float>::infinity());
  auto selected = std::get<1>(scores.topk(topk,
                                          /*dim=*/1,
                                          /*largest=*/true,
                                          /*sorted=*/false));
  auto ranks = torch::arange(topk, options).unsqueeze(0);
  // Match the kernel's all-valid shortcut exactly, including ascending order
  // and -1 padding, independently for each query in a mixed-length batch.
  selected = torch::where(valid.le(topk).unsqueeze(1), ranks, selected);
  selected.masked_fill_(ranks.ge(valid.unsqueeze(1)), -1);
  output.copy_(selected);

  auto map_physical = [&](const torch::Tensor& logical,
                          const torch::Tensor& table) {
    auto columns = logical.clamp_min(0).div(mate_block_size, "floor");
    auto mapped_blocks =
        table.index({owners.unsqueeze(1), columns.clamp_max(table.size(1) - 1)})
            .to(torch::kInt64);
    auto mapped =
        mapped_blocks * mate_block_size + logical.remainder(mate_block_size);
    auto invalid =
        logical.lt(0)
            .logical_or(columns.ge(table.size(1)))
            .logical_or(mapped_blocks.lt(0))
            .logical_or(mapped.gt(std::numeric_limits<int32_t>::max()));
    mapped.masked_fill_(invalid, -1);
    return mapped;
  };
  physical_output.copy_(map_physical(selected, compressed_blocks));
  auto extra_count = live.clamp_max(std::min<int64_t>(128, window));
  auto extra_ranks = torch::arange(128, options).unsqueeze(0);
  auto extra_logical = (live - extra_count).unsqueeze(1) + extra_ranks;
  extra_logical.masked_fill_(extra_ranks.ge(extra_count.unsqueeze(1)), -1);
  extra_output.copy_(map_physical(extra_logical, swa_blocks));
  return output;
}

}  // namespace xllm::layer
