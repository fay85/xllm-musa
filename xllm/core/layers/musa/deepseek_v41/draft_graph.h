/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <c10/core/StreamGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tuple>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"
#include "core/platform/cuda/device_capture_lock.h"
#include "core/platform/musa_graph_handle.h"

namespace xllm::layer {

struct DeepseekV41DraftInputs {
  torch::Tensor tokens;
  torch::Tensor positions;
  torch::Tensor table;
  torch::Tensor previous_context;
  std::vector<torch::Tensor> caches;
};

// Query-only draft forward reads accepted KV but does not commit proposed KV.
// Context projection/writes, logits and Markov bias remain outside the graph.
class DeepseekV41DraftGraph final {
 public:
  using Forward = std::function<torch::Tensor(const DeepseekV41DraftInputs&)>;

  torch::Tensor run(const torch::Tensor& tokens,
                    const torch::Tensor& positions,
                    const std::vector<torch::Tensor>& caches,
                    const torch::Tensor& table,
                    int64_t previous_context,
                    bool replay_graph,
                    const Forward& forward) {
    CHECK_GT(tokens.numel(), 0);
    CHECK_LE(tokens.numel(), 6);
    CHECK_EQ(positions.numel(), tokens.numel());
    CHECK_EQ(caches.size(), 3);
    CHECK_GE(previous_context, 0);
    CHECK_EQ(table.dim(), 2);
    CHECK_EQ(table.size(0), 1);
    const Key key{tokens.numel(), table.size(1)};
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      auto entry = std::make_unique<Entry>();
      entry->input.tokens = torch::empty_like(tokens);
      entry->input.positions = torch::empty_like(positions);
      entry->input.table = torch::empty_like(table);
      entry->input.previous_context =
          torch::empty({1}, tokens.options().dtype(torch::kInt64));
      entry->input.caches = caches;
      it = entries_.emplace(key, std::move(entry)).first;
    }
    Entry& entry = *it->second;
    CHECK_EQ(entry.input.tokens.device(), tokens.device());
    CHECK_EQ(entry.input.tokens.scalar_type(), tokens.scalar_type());
    CHECK_EQ(entry.input.positions.scalar_type(), positions.scalar_type());
    CHECK_EQ(entry.input.table.scalar_type(), table.scalar_type());
    for (size_t index = 0; index < caches.size(); ++index) {
      CHECK_EQ(entry.input.caches[index].data_ptr(), caches[index].data_ptr());
      CHECK_EQ(entry.input.caches[index].sizes(), caches[index].sizes());
      CHECK_EQ(entry.input.caches[index].strides(), caches[index].strides());
    }
    entry.input.tokens.copy_(tokens);
    entry.input.positions.copy_(positions);
    entry.input.table.copy_(table);
    entry.input.previous_context.fill_(previous_context);
    if (!replay_graph) {
      return forward(entry.input);
    }
    if (!entry.graph) {
      capture(entry, forward);
    }
    const DeepseekV41ProfileRange range("V41/DSpark/full_draft_graph_replay");
    entry.graph->replay();
    return entry.output;
  }

 private:
  struct Entry {
    DeepseekV41DraftInputs input;
    torch::Tensor output;
    std::unique_ptr<MusaGraphHandle> graph;
  };
  using Key = std::tuple<int64_t, int64_t>;

  void capture(Entry& entry, const Forward& forward) {
    const DeepseekV41ProfileRange range("V41/DSpark/full_draft_graph_capture");
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
        forward(entry.input);
        capture_stream.synchronize();
      }
      if (!pool_.has_value()) {
        pool_ = at::musa::graph_pool_handle();
      }
      entry.graph = std::make_unique<MusaGraphHandle>();
      entry.graph->capture_begin(pool_.value(),
                                 musaStreamCaptureModeThreadLocal);
      entry.output = forward(entry.input);
      entry.graph->capture_end();
      capture_stream.synchronize();
      finalize_musa_custom_allreduce_graphs();
      capture_stream.synchronize();
    }
    LOG(INFO) << "Native V4.1 read-only draft Graph captured, rows="
              << entry.input.tokens.numel()
              << "; context writes and vocabulary/Markov heads outside";
  }

  std::optional<c10::musa::MempoolId_t> pool_;
  std::map<Key, std::unique_ptr<Entry>> entries_;
};

}  // namespace xllm::layer
