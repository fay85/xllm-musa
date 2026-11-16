/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software is distributed under the License on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <c10/core/StreamGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/layers/musa/deepseek_v41/attention.h"
#include "core/layers/musa/deepseek_v41/state_checkpoint.h"
#include "core/platform/cuda/device_capture_lock.h"
#include "core/platform/musa_graph_handle.h"

namespace xllm::layer {

struct DeepseekV41DecodeInputs {
  torch::Tensor tokens;
  torch::Tensor positions;
  std::vector<DeepseekV41AttentionCache> caches;
  torch::Tensor swa_table;
  torch::Tensor ratio1_table;
  torch::Tensor ratio2_table;
  int64_t context_length = 0;
  std::vector<std::optional<torch::Tensor>> engram_rows;
  std::optional<torch::Tensor> engram_mask;
};

// C1 target decoder only. Engram hashing/lookup and the vocabulary head stay
// outside this graph. No request-private GPU cache copy is used for inference.
// Warmup/capture writes are explicitly rolled back before the one real replay.
class DeepseekV41DecodeGraph final {
 public:
  using Forward =
      std::function<torch::Tensor(const DeepseekV41DecodeInputs&, int64_t)>;

  explicit DeepseekV41DecodeGraph(std::vector<int64_t> ratios)
      : ratios_(std::move(ratios)) {}

  torch::Tensor run(const DeepseekV41DecodeInputs& input,
                    bool replay_graph,
                    const Forward& forward) {
    CHECK_GE(input.tokens.numel(), 1);
    CHECK_LE(input.tokens.numel(), 6);
    CHECK_EQ(input.positions.numel(), input.tokens.numel());
    CHECK_GE(input.context_length, input.tokens.numel());
    CHECK_GT(input.context_length, 0);
    CHECK_LE(input.context_length, 4096);
    CHECK_EQ(input.caches.size(), ratios_.size());
    const int64_t capacity = input.context_length <= 512 ? 512 : 4096;
    const Key key{capacity,
                  (input.context_length - input.tokens.numel()) % 2,
                  input.swa_table.size(1),
                  input.tokens.numel()};
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      auto entry = std::make_unique<Entry>();
      entry->input = allocate(input, capacity);
      it = entries_.emplace(key, std::move(entry)).first;
    }
    Entry& entry = *it->second;
    update(entry.input, input);
    if (!replay_graph) {
      return forward(entry.input, capacity);
    }
    if (!entry.graph) {
      capture(entry, capacity, forward);
    }
    const DeepseekV41ProfileRange range("V41/full_decode_graph_replay");
    entry.graph->replay();
    return entry.output;
  }

 private:
  struct Entry {
    DeepseekV41DecodeInputs input;
    torch::Tensor output;
    // Destroy graph before its referenced inputs/output.
    std::unique_ptr<MusaGraphHandle> graph;
  };
  using Key = std::tuple<int64_t, int64_t, int64_t, int64_t>;

  static torch::Tensor allocate_table(const torch::Tensor& source,
                                      int64_t capacity,
                                      int64_t block_size) {
    if (!source.defined()) {
      return {};
    }
    CHECK_GT(block_size, 0);
    const int64_t columns = (capacity + block_size - 1) / block_size;
    return torch::full({1, columns}, -1, source.options());
  }

  DeepseekV41DecodeInputs allocate(const DeepseekV41DecodeInputs& input,
                                   int64_t capacity) const {
    DeepseekV41DecodeInputs buffers;
    buffers.tokens = torch::empty_like(input.tokens);
    buffers.positions =
        torch::empty({input.positions.numel()},
                     input.positions.options().dtype(torch::kInt64));
    buffers.caches = input.caches;
    buffers.swa_table = torch::empty_like(input.swa_table);
    // Use each group's actual compressed block size, not the SWA block size.
    for (size_t index = 0; index < input.caches.size(); ++index) {
      const auto& cache = input.caches[index];
      if (!cache.key.defined() || cache.key.numel() == 0) {
        continue;
      }
      if (ratios_[index] == 1 && !buffers.ratio1_table.defined()) {
        buffers.ratio1_table =
            allocate_table(input.ratio1_table, capacity, cache.key.size(1));
      }
      if (ratios_[index] == 2 && !buffers.ratio2_table.defined()) {
        buffers.ratio2_table =
            allocate_table(input.ratio2_table, capacity / 2, cache.key.size(1));
      }
    }
    buffers.engram_rows.resize(input.engram_rows.size());
    for (size_t index = 0; index < input.engram_rows.size(); ++index) {
      if (input.engram_rows[index].has_value()) {
        buffers.engram_rows[index] =
            torch::empty_like(input.engram_rows[index].value());
      }
    }
    if (input.engram_mask.has_value()) {
      buffers.engram_mask = torch::empty_like(input.engram_mask.value());
    }
    return buffers;
  }

  static void copy_optional(std::optional<torch::Tensor>& target,
                            const std::optional<torch::Tensor>& source) {
    CHECK_EQ(target.has_value(), source.has_value());
    if (source.has_value()) {
      CHECK_EQ(target->sizes(), source->sizes());
      target->copy_(source.value());
    }
  }

  static void copy_table(torch::Tensor& target, const torch::Tensor& source) {
    CHECK_EQ(target.defined(), source.defined());
    if (!source.defined()) {
      return;
    }
    CHECK_EQ(source.size(0), 1);
    CHECK_EQ(source.scalar_type(), target.scalar_type());
    const int64_t count = std::min(target.size(1), source.size(1));
    target.fill_(-1);
    target.narrow(1, 0, count).copy_(source.narrow(1, 0, count));
  }

  static void check_same_cache(const torch::Tensor& cached,
                               const torch::Tensor& supplied) {
    CHECK_EQ(cached.defined(), supplied.defined());
    if (cached.defined()) {
      CHECK_EQ(cached.data_ptr(), supplied.data_ptr());
      CHECK_EQ(cached.sizes(), supplied.sizes());
    }
  }

  static void update(DeepseekV41DecodeInputs& target,
                     const DeepseekV41DecodeInputs& source) {
    CHECK_EQ(target.caches.size(), source.caches.size());
    for (size_t index = 0; index < target.caches.size(); ++index) {
      const auto& old = target.caches[index];
      const auto& current = source.caches[index];
      check_same_cache(old.swa, current.swa);
      check_same_cache(old.key, current.key);
      check_same_cache(old.index, current.index);
      CHECK_EQ(old.compressor_kv_state.has_value(),
               current.compressor_kv_state.has_value());
      CHECK_EQ(old.compressor_score_state.has_value(),
               current.compressor_score_state.has_value());
      if (old.compressor_kv_state.has_value()) {
        check_same_cache(old.compressor_kv_state.value(),
                         current.compressor_kv_state.value());
        check_same_cache(old.compressor_score_state.value(),
                         current.compressor_score_state.value());
      }
    }
    target.tokens.copy_(source.tokens);
    target.positions.copy_(source.positions);
    target.swa_table.copy_(source.swa_table);
    copy_table(target.ratio1_table, source.ratio1_table);
    copy_table(target.ratio2_table, source.ratio2_table);
    target.context_length = source.context_length;
    CHECK_EQ(target.engram_rows.size(), source.engram_rows.size());
    for (size_t index = 0; index < source.engram_rows.size(); ++index) {
      copy_optional(target.engram_rows[index], source.engram_rows[index]);
    }
    copy_optional(target.engram_mask, source.engram_mask);
  }

  std::unique_ptr<DeepseekV41StateCheckpoint> snapshot(
      const DeepseekV41DecodeInputs& input) const {
    auto state = std::make_unique<DeepseekV41StateCheckpoint>(64 * 1024 * 1024);
    const int64_t tokens = input.tokens.numel();
    const int64_t position = input.context_length - tokens;
    for (size_t index = 0; index < input.caches.size(); ++index) {
      const auto& cache = input.caches[index];
      state->add_paged(
          cache.swa, input.swa_table, position, tokens, /*ring=*/true);
      const int64_t ratio = ratios_[index];
      const torch::Tensor& table =
          ratio == 1 ? input.ratio1_table : input.ratio2_table;
      if (ratio > 0 && cache.key.defined() && cache.key.numel() > 0) {
        state->add_paged(cache.key,
                         table,
                         position / ratio,
                         input.context_length / ratio - position / ratio,
                         /*ring=*/false);
      }
      if (ratio > 0 && cache.index.defined() && cache.index.numel() > 0) {
        state->add_paged(cache.index,
                         table,
                         position / ratio,
                         input.context_length / ratio - position / ratio,
                         /*ring=*/false);
      }
      if (cache.compressor_kv_state.has_value()) {
        const torch::Tensor blocks = input.swa_table.flatten().contiguous();
        state->add_blocks(cache.compressor_kv_state.value(), blocks);
        state->add_blocks(cache.compressor_score_state.value(), blocks);
      }
    }
    return state;
  }

  void capture(Entry& entry, int64_t capacity, const Forward& forward) {
    const DeepseekV41ProfileRange range("V41/full_decode_graph_capture");
    const int32_t device_index = entry.input.tokens.get_device();
    std::unique_lock<std::shared_mutex> lock(
        cuda::DeviceCaptureLock::get_instance().get_write_lock(device_index));
    const auto original = c10::musa::getCurrentMUSAStream(device_index);
    const auto capture_stream =
        c10::musa::getStreamFromPool(false, device_index);
    original.synchronize();
    capture_stream.synchronize();
    {
      c10::StreamGuard guard(capture_stream.unwrap());
      for (int32_t iteration = 0; iteration < 2; ++iteration) {
        auto checkpoint = snapshot(entry.input);
        forward(entry.input, capacity);
        checkpoint->restore();
        capture_stream.synchronize();
      }
      auto checkpoint = snapshot(entry.input);
      capture_stream.synchronize();
      if (!pool_.has_value()) {
        pool_ = at::musa::graph_pool_handle();
      }
      entry.graph = std::make_unique<MusaGraphHandle>();
      entry.graph->capture_begin(pool_.value(),
                                 musaStreamCaptureModeThreadLocal);
      entry.output = forward(entry.input, capacity);
      entry.graph->capture_end();
      capture_stream.synchronize();
      finalize_musa_custom_allreduce_graphs();
      checkpoint->restore();
      capture_stream.synchronize();
    }
    LOG(INFO)
        << "Native V4.1 full decoder Graph captured; capacity=" << capacity
        << ", rows=" << entry.input.tokens.numel() << ", parity="
        << (entry.input.context_length - entry.input.tokens.numel()) % 2
        << "; warmup/capture cache writes restored; Engram lookup/head outside";
  }

  std::vector<int64_t> ratios_;
  std::optional<c10::musa::MempoolId_t> pool_;
  std::map<Key, std::unique_ptr<Entry>> entries_;
};

}  // namespace xllm::layer
