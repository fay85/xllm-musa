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
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>

#include "core/kernels/musa/deepseek_v41_engram_hash.h"

namespace xllm::layer {

// Native eager C1 Engram hashing. Token-map/multipliers/primes/offsets are
// immutable tokenizer-derived assets; history is a receiver-local SWA-slot
// tensor bootstrapped from the framework's existing [slot, token] PD pairs.
// All per-token mapping, int64 rolling hash, and history writes execute GPU.
class DeepseekV41EngramHash final {
 public:
  DeepseekV41EngramHash(torch::Tensor token_map,
                        torch::Tensor multipliers,
                        torch::Tensor primes,
                        torch::Tensor offsets,
                        int64_t pad_id,
                        std::string aot_directory = {})
      : token_map_(std::move(token_map)),
        multipliers_(std::move(multipliers)),
        primes_(std::move(primes)),
        offsets_(std::move(offsets)),
        pad_id_(pad_id),
        aot_directory_(std::move(aot_directory)) {
    CHECK(token_map_.defined());
    CHECK(token_map_.device().type() == torch::kPrivateUse1);
    check_asset(token_map_);
    check_asset(multipliers_);
    check_asset(primes_);
    check_asset(offsets_);
    CHECK_EQ(token_map_.dim(), 1);
    CHECK_GT(token_map_.numel(), 0);
    CHECK_EQ(multipliers_.dim(), 2);
    CHECK_GT(multipliers_.size(0), 0);
    CHECK_GE(multipliers_.size(1), 2);
    CHECK_EQ(primes_.dim(), 3);
    CHECK_EQ(primes_.size(0), multipliers_.size(0));
    CHECK_EQ(primes_.size(1), multipliers_.size(1) - 1);
    CHECK_GT(primes_.size(2), 0);
    CHECK_EQ(offsets_.sizes(),
             torch::IntArrayRef(
                 {primes_.size(0), primes_.size(1) * primes_.size(2)}));
    CHECK_GE(pad_id_, 0);
    if (!aot_directory_.empty()) {
      std::ifstream file(aot_directory_ + "/manifest.json");
      CHECK(file.good());
      nlohmann::json manifest;
      file >> manifest;
      CHECK_EQ(manifest.at("history_slot_dtype").get<std::string>(), "int32");
      CHECK_EQ(manifest.at("mask_store_packed_bytes").get<int32_t>(), 4);
      CHECK_EQ(multipliers_.size(0), 2);
      CHECK_EQ(multipliers_.size(1), 4);
      CHECK_EQ(primes_.size(2), 8);
    }
  }

  void bind_history(torch::Tensor history) {
    CHECK(history.defined());
    CHECK_EQ(history.device(), token_map_.device());
    CHECK_EQ(history.scalar_type(), torch::kInt32);
    CHECK_EQ(history.dim(), 1);
    CHECK_GT(history.numel(), 0);
    CHECK(history.is_contiguous());
    history_ = std::move(history);
  }

  void restore_history(const torch::Tensor& pairs) {
    CHECK(history_.defined());
    check_indices(pairs);
    CHECK_EQ(pairs.dim(), 2);
    CHECK_EQ(pairs.size(1), 2);
    torch::DeviceGuard guard(token_map_.device());
    const torch::Tensor slots = pairs.select(1, 0).to(torch::kInt64);
    const torch::Tensor tokens = pairs.select(1, 1).to(torch::kInt64);
    write(slots, map(tokens).to(torch::kInt32));
  }

  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& input_ids,
      const torch::Tensor& positions,
      const torch::Tensor& swa_table,
      int64_t block_size,
      int64_t start_position,
      const std::optional<torch::Tensor>& lookback_ids) {
    CHECK(history_.defined());
    check_indices(input_ids);
    check_indices(positions);
    check_indices(swa_table);
    CHECK_EQ(input_ids.dim(), 1);
    CHECK_EQ(positions.sizes(), input_ids.sizes());
    CHECK_EQ(swa_table.dim(), 2);
    CHECK_EQ(swa_table.size(0), 1);
    CHECK_GT(swa_table.size(1), 0);
    CHECK_GT(block_size, 0);
    CHECK_GE(start_position, 0);
    if (lookback_ids.has_value()) {
      check_indices(lookback_ids.value());
      CHECK_EQ(lookback_ids->dim(), 2);
      CHECK_EQ(lookback_ids->size(0), 1);
      CHECK_GT(lookback_ids->size(1), 0);
    }
    torch::DeviceGuard guard(token_map_.device());
    if (!aot_directory_.empty()) {
      return kernel::musa::deepseek_v41_paged_engram_hash(input_ids,
                                                          positions,
                                                          swa_table,
                                                          history_,
                                                          token_map_,
                                                          multipliers_,
                                                          primes_,
                                                          offsets_,
                                                          pad_id_,
                                                          block_size,
                                                          start_position,
                                                          lookback_ids,
                                                          aot_directory_);
    }
    const int64_t count = input_ids.size(0);
    const int64_t layers = multipliers_.size(0);
    const int64_t heads = primes_.size(2);
    const int64_t grams = multipliers_.size(1);
    const auto options = input_ids.options().dtype(torch::kInt64);
    const torch::Tensor ids = input_ids.to(torch::kInt64);
    const torch::Tensor pos = positions.to(torch::kInt64);
    const torch::Tensor dead = dead_mask(ids);
    const torch::Tensor compressed = map(ids);
    torch::Tensor output = torch::zeros({count, layers, (grams - 1) * heads},
                                        options.dtype(torch::kInt32));
    if (count == 0) {
      return {output, dead.logical_not()};
    }
    const torch::Tensor rows = torch::arange(count, options);
    torch::Tensor rolling = torch::zeros({count, layers}, options);
    torch::Tensor blocked = torch::zeros({count}, options.dtype(torch::kBool));
    for (int64_t shift = 0; shift < grams; ++shift) {
      const torch::Tensor previous = pos - shift;
      const torch::Tensor in_chunk = previous.ge(start_position);
      torch::Tensor source =
          compressed.index_select(0, (rows - shift).clamp(0, count - 1));
      const torch::Tensor slots =
          physical_slots(previous.clamp_min(0), swa_table, block_size);
      const torch::Tensor cached =
          history_.index_select(0, slots.clamp(0, history_.numel() - 1));
      torch::Tensor fallback =
          torch::where(slots.ge(0), cached.to(torch::kInt64), -1);
      if (lookback_ids.has_value()) {
        const torch::Tensor columns = start_position - 1 - previous;
        const int64_t depth = lookback_ids->size(1);
        const torch::Tensor tokens =
            lookback_ids->select(0, 0)
                .index_select(0, columns.clamp(0, depth - 1))
                .to(torch::kInt64);
        const torch::Tensor known = columns.ge(0)
                                        .logical_and(columns.lt(depth))
                                        .logical_and(tokens.ge(0));
        fallback = torch::where(known, map(tokens), fallback);
      }
      source = torch::where(in_chunk, source, fallback);
      blocked = blocked.logical_or(previous.lt(0)).logical_or(source.eq(-1));
      const torch::Tensor value = torch::where(blocked, pad_id_, source);
      rolling.bitwise_xor_(value.unsqueeze(1) * multipliers_.select(1, shift));
      if (shift == 0) {
        continue;
      }
      const torch::Tensor hashed =
          rolling.unsqueeze(-1).remainder(primes_.select(1, shift - 1)) +
          offsets_.narrow(1, (shift - 1) * heads, heads);
      output.narrow(2, (shift - 1) * heads, heads)
          .copy_(hashed.to(torch::kInt32));
    }
    // C1 target execution needs no early publish for expanded speculative
    // rows. Read the complete old tail first, then retain each slot's last ID.
    write(physical_slots(pos, swa_table, block_size),
          compressed.to(torch::kInt32));
    return {output, dead.logical_not()};
  }

 private:
  void check_asset(const torch::Tensor& value) const {
    CHECK(value.defined());
    CHECK_EQ(value.device(), token_map_.device());
    CHECK_EQ(value.scalar_type(), torch::kInt64);
    CHECK(value.is_contiguous());
  }

  void check_indices(const torch::Tensor& value) const {
    CHECK(value.defined());
    CHECK_EQ(value.device(), token_map_.device());
    CHECK(value.scalar_type() == torch::kInt32 ||
          value.scalar_type() == torch::kInt64);
  }

  torch::Tensor dead_mask(const torch::Tensor& ids) const {
    return ids.eq(129264).logical_or(ids.eq(129265));
  }

  torch::Tensor map(const torch::Tensor& ids) const {
    const torch::Tensor compressed =
        token_map_.index_select(0, ids.clamp(0, token_map_.numel() - 1));
    return torch::where(dead_mask(ids), -1, compressed);
  }

  torch::Tensor physical_slots(const torch::Tensor& positions,
                               const torch::Tensor& table,
                               int64_t block_size) const {
    const torch::Tensor columns =
        torch::floor_divide(positions, block_size).remainder(table.size(1));
    const torch::Tensor blocks =
        table.select(0, 0).to(torch::kInt64).index_select(0, columns);
    return blocks * block_size + positions.remainder(block_size);
  }

  void write(const torch::Tensor& slots, const torch::Tensor& values) {
    if (slots.numel() == 0) {
      return;
    }
    const torch::Tensor order = torch::argsort(slots,
                                               /*stable=*/true,
                                               /*dim=*/0,
                                               /*descending=*/false);
    const torch::Tensor sorted = slots.index_select(0, order);
    torch::Tensor keep = sorted.ge(0);
    if (sorted.numel() > 1) {
      keep.narrow(0, 0, sorted.numel() - 1)
          .logical_and_(sorted.narrow(0, 0, sorted.numel() - 1)
                            .ne(sorted.narrow(0, 1, sorted.numel() - 1)));
    }
    const torch::Tensor latest = order.index({keep});
    history_.index_copy_(
        0, slots.index_select(0, latest), values.index_select(0, latest));
  }

  torch::Tensor token_map_;
  torch::Tensor multipliers_;
  torch::Tensor primes_;
  torch::Tensor offsets_;
  int64_t pad_id_;
  torch::Tensor history_;
  std::string aot_directory_;
};

}  // namespace xllm::layer
