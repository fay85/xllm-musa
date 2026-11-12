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
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/model/aux_hidden_capture.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/layers/musa/deepseek_v41/decoder.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"

namespace xllm::layer {

// Native text backbone and vocabulary head. All decoder iteration and model
// math execute here; no Python forward callback. The framework adapter must
// own checkpoint loading, Engram hash/row preparation, scheduling and the
// process-group lifetime. This domain object alone is not a CausalLM factory.
class DeepseekV41TextModel final {
 public:
  DeepseekV41TextModel(std::vector<DeepseekV41DecoderConfig> configs,
                       std::vector<DeepseekV41DecoderWeights> weights,
                       torch::Tensor embedding,
                       torch::Tensor norm,
                       torch::Tensor head,
                       torch::Tensor plain_rope,
                       torch::Tensor compressed_rope,
                       ProcessGroup* attention_group = nullptr,
                       ProcessGroup* expert_group = nullptr,
                       bool bounded_replay = false)
      : embedding_(std::move(embedding)),
        norm_(std::move(norm)),
        head_(std::move(head)),
        plain_rope_(std::move(plain_rope)),
        compressed_rope_(std::move(compressed_rope)),
        attention_group_(attention_group) {
    CHECK(!configs.empty());
    CHECK_EQ(configs.size(), weights.size());
    CHECK(embedding_.defined() && norm_.defined() && head_.defined());
    CHECK(embedding_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(embedding_.scalar_type(), torch::kBFloat16);
    CHECK_EQ(norm_.device(), embedding_.device());
    CHECK_EQ(head_.device(), embedding_.device());
    CHECK_EQ(norm_.scalar_type(), torch::kBFloat16);
    CHECK_EQ(head_.scalar_type(), torch::kBFloat16);
    CHECK_EQ(embedding_.dim(), 2);
    CHECK_EQ(head_.dim(), 2);
    CHECK_EQ(norm_.dim(), 1);
    const int64_t tp =
        attention_group_ == nullptr ? 1 : attention_group_->world_size();
    CHECK_EQ(embedding_.size(1) * tp, norm_.numel());
    CHECK_EQ(head_.size(1), norm_.numel());
    CHECK_EQ(head_.size(0) * tp, embedding_.size(0));
    norm_epsilon_ = configs[0].attention.epsilon;
    bounded_replay_ = bounded_replay;
    replay_window_ = configs[0].attention.window;
    for (const auto& config : configs) {
      late_layer_start_ =
          std::max(late_layer_start_, config.attention.source_layer + 1);
    }
    if (bounded_replay_) {
      CHECK_GT(replay_window_, 1);
      CHECK_GT(late_layer_start_, 0);
      CHECK_LT(late_layer_start_, configs.size());
      for (size_t index = late_layer_start_; index < configs.size(); ++index) {
        CHECK(configs[index].attention.mode != DeepseekV41AttentionMode::FULL);
        CHECK_LE(configs[index].attention.ratio, 1);
      }
    }
    layers_.reserve(configs.size());
    compressed_layers_.reserve(configs.size());
    source_ratios_.reserve(configs.size());
    for (size_t index = 0; index < configs.size(); ++index) {
      CHECK_EQ(configs[index].attention.layer, index);
      CHECK_EQ(configs[index].attention.epsilon, norm_epsilon_);
      compressed_layers_.emplace_back(configs[index].attention.ratio > 0);
      source_ratios_.emplace_back(configs[index].attention.ratio);
      layers_.emplace_back(
          std::make_unique<DeepseekV41Decoder>(std::move(configs[index]),
                                               std::move(weights[index]),
                                               attention_group_,
                                               expert_group));
    }
  }

  torch::Tensor forward(
      const torch::Tensor& input_ids,
      const torch::Tensor& positions,
      const std::vector<DeepseekV41AttentionCache>& caches,
      const torch::Tensor& swa_table,
      const torch::Tensor& ratio1_table,
      const torch::Tensor& ratio2_table,
      int64_t context_length,
      const std::vector<std::optional<torch::Tensor>>& engram_rows,
      const std::optional<torch::Tensor>& engram_mask) {
    return forward_with_capture(input_ids,
                                positions,
                                caches,
                                swa_table,
                                ratio1_table,
                                ratio2_table,
                                context_length,
                                engram_rows,
                                engram_mask,
                                nullptr);
  }

  torch::Tensor forward_with_capture(
      const torch::Tensor& input_ids,
      const torch::Tensor& positions,
      const std::vector<DeepseekV41AttentionCache>& caches,
      const torch::Tensor& swa_table,
      const torch::Tensor& ratio1_table,
      const torch::Tensor& ratio2_table,
      int64_t context_length,
      const std::vector<std::optional<torch::Tensor>>& engram_rows,
      const std::optional<torch::Tensor>& engram_mask,
      AuxHiddenCapture* auxiliary) {
    CHECK(input_ids.defined() && positions.defined());
    CHECK_EQ(input_ids.device(), embedding_.device());
    CHECK_EQ(positions.device(), embedding_.device());
    CHECK(input_ids.scalar_type() == torch::kInt32 ||
          input_ids.scalar_type() == torch::kInt64);
    CHECK_EQ(input_ids.dim(), 1);
    CHECK_GT(input_ids.numel(), 0)
        << "Native text model currently C1/eager nonempty requests";
    CHECK_EQ(positions.dim(), 1);
    CHECK_EQ(input_ids.size(0), positions.size(0));
    CHECK_EQ(caches.size(), layers_.size());
    CHECK_EQ(engram_rows.size(), layers_.size());
    CHECK_GE(context_length, input_ids.size(0));
    torch::DeviceGuard guard(embedding_.device());
    torch::Tensor hidden =
        torch::embedding(embedding_, input_ids.to(torch::kInt64));
    hidden = parallel_state::gather(hidden, attention_group_, /*dim=*/-1);
    hidden = hidden.unsqueeze(1).expand({-1, 4, -1}).contiguous();
    torch::Tensor pre_mix = torch::zeros(
        {input_ids.size(0), 4}, hidden.options().dtype(torch::kFloat32));
    pre_mix.select(1, 0).fill_(1.0);
    DeepseekV41AttentionRuntime shared;
    last_output_row_offset_ = 0;
    torch::Tensor active_ids = input_ids;
    torch::Tensor active_positions = positions;
    std::optional<torch::Tensor> active_engram_mask = engram_mask;
    const bool diagnostic_progress =
        std::getenv("XLLM_DSV41_DIAGNOSTIC_PROGRESS") != nullptr &&
        input_ids.numel() > 1;
    for (size_t index = 0; index < layers_.size(); ++index) {
      if (bounded_replay_ && index == static_cast<size_t>(late_layer_start_) &&
          input_ids.size(0) > replay_window_) {
        last_output_row_offset_ = input_ids.size(0) - replay_window_;
        hidden = hidden.narrow(0, last_output_row_offset_, replay_window_);
        pre_mix = pre_mix.narrow(0, last_output_row_offset_, replay_window_);
        active_ids =
            input_ids.narrow(0, last_output_row_offset_, replay_window_);
        active_positions =
            positions.narrow(0, last_output_row_offset_, replay_window_);
        shared.swa_replay_start = context_length - replay_window_;
        if (shared.topk.defined()) {
          shared.topk =
              shared.topk.narrow(0, last_output_row_offset_, replay_window_);
        }
        if (shared.candidates.has_value()) {
          shared.candidates = shared.candidates->narrow(
              0, last_output_row_offset_, replay_window_);
        }
        if (active_engram_mask.has_value()) {
          active_engram_mask = active_engram_mask->narrow(
              0, last_output_row_offset_, replay_window_);
        }
      }
      std::optional<torch::Tensor> active_engram = engram_rows[index];
      if (last_output_row_offset_ > 0 && active_engram.has_value()) {
        active_engram =
            active_engram->narrow(0, last_output_row_offset_, replay_window_);
      }

      if (diagnostic_progress) {
        LOG(INFO) << "V41_DIAGNOSTIC layer begin index=" << index
                  << " q=" << hidden.size(0) << " kv=" << context_length;
      }
      const DeepseekV41ProfileRange layer_range("V41/layer",
                                                static_cast<int64_t>(index));
      const torch::Tensor& table =
          source_ratios_[index] == 1 ? ratio1_table : ratio2_table;
      if (auxiliary != nullptr &&
          auxiliary->should_capture(static_cast<int32_t>(index))) {
        // V4.1 names the residual ENTERING the layer, unlike V4 post-layer ids.
        auxiliary->capture_layer(
            static_cast<int32_t>(index), hidden.mean(1), std::nullopt);
      }
      std::tie(hidden, pre_mix) = layers_[index]->forward(
          hidden,
          pre_mix,
          active_positions,
          compressed_layers_[index] ? compressed_rope_ : plain_rope_,
          caches[index],
          swa_table,
          table,
          table,
          context_length,
          shared,
          active_ids,
          active_engram,
          active_engram_mask);
      if (diagnostic_progress) {
        LOG(INFO) << "V41_DIAGNOSTIC layer enqueued index=" << index
                  << " q=" << hidden.size(0) << " kv=" << context_length;
      }
    }
    const torch::Tensor merged = kernel::musa::deepseek_v4_hc_pre_combine(
        hidden.contiguous(), pre_mix.contiguous());
    return deepseek_v41_rms_norm(merged, norm_, norm_epsilon_);
  }

  torch::Tensor forward_batched_decode(
      const torch::Tensor& input_ids,
      const torch::Tensor& positions,
      const std::vector<DeepseekV41AttentionCache>& caches,
      const torch::Tensor& swa_table,
      const torch::Tensor& ratio1_table,
      const torch::Tensor& ratio2_table,
      const std::vector<int64_t>& context_lengths,
      const std::vector<std::optional<torch::Tensor>>& engram_rows,
      const std::optional<torch::Tensor>& engram_mask) {
    CHECK(input_ids.defined() && positions.defined());
    CHECK_EQ(input_ids.device(), embedding_.device());
    CHECK_EQ(positions.device(), embedding_.device());
    CHECK(input_ids.scalar_type() == torch::kInt32 ||
          input_ids.scalar_type() == torch::kInt64);
    CHECK_EQ(input_ids.dim(), 1);
    CHECK_GT(input_ids.numel(), 0)
        << "Native text model currently C1/eager nonempty requests";
    CHECK_EQ(positions.dim(), 1);
    CHECK_EQ(input_ids.size(0), positions.size(0));
    CHECK_EQ(caches.size(), layers_.size());
    CHECK_EQ(engram_rows.size(), layers_.size());
    CHECK_EQ(context_lengths.size(), static_cast<size_t>(input_ids.size(0)));
    CHECK_GT(input_ids.size(0), 1);
    CHECK_LE(input_ids.size(0), 8);
    torch::DeviceGuard guard(embedding_.device());
    torch::Tensor hidden =
        torch::embedding(embedding_, input_ids.to(torch::kInt64));
    hidden = parallel_state::gather(hidden, attention_group_, /*dim=*/-1);
    hidden = hidden.unsqueeze(1).expand({-1, 4, -1}).contiguous();
    torch::Tensor pre_mix = torch::zeros(
        {input_ids.size(0), 4}, hidden.options().dtype(torch::kFloat32));
    pre_mix.select(1, 0).fill_(1.0);
    std::vector<DeepseekV41AttentionRuntime> shared(input_ids.size(0));
    const bool diagnostic_progress = false;
    for (size_t index = 0; index < layers_.size(); ++index) {
      if (diagnostic_progress) {
        LOG(INFO) << "V41_DIAGNOSTIC layer begin index=" << index
                  << " q=" << input_ids.numel() << " kv=" << context_lengths[0];
      }
      const DeepseekV41ProfileRange layer_range("V41/layer",
                                                static_cast<int64_t>(index));
      const torch::Tensor& table =
          source_ratios_[index] == 1 ? ratio1_table : ratio2_table;
      std::tie(hidden, pre_mix) = layers_[index]->forward_batched_decode(
          hidden,
          pre_mix,
          positions,
          compressed_layers_[index] ? compressed_rope_ : plain_rope_,
          caches[index],
          swa_table,
          table,
          table,
          context_lengths,
          shared,
          input_ids,
          engram_rows[index],
          engram_mask);
      if (diagnostic_progress) {
        LOG(INFO) << "V41_DIAGNOSTIC layer enqueued index=" << index
                  << " q=" << input_ids.numel() << " kv=" << context_lengths[0];
      }
    }
    const torch::Tensor merged = kernel::musa::deepseek_v4_hc_pre_combine(
        hidden.contiguous(), pre_mix.contiguous());
    return deepseek_v41_rms_norm(merged, norm_, norm_epsilon_);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const std::optional<torch::Tensor>& selected) const {
    const DeepseekV41ProfileRange range("V41/logits");
    CHECK(hidden.defined());
    CHECK_EQ(hidden.device(), head_.device());
    CHECK_EQ(hidden.scalar_type(), torch::kBFloat16);
    CHECK_EQ(hidden.dim(), 2);
    CHECK_EQ(hidden.size(1), head_.size(1));
    torch::DeviceGuard guard(head_.device());
    return project_logits(select_hidden(hidden, selected));
  }

  torch::Tensor select_hidden(
      const torch::Tensor& hidden,
      const std::optional<torch::Tensor>& selected) const {
    if (last_output_row_offset_ > 0) {
      CHECK(selected.has_value() && selected->defined() &&
            selected->numel() > 0)
          << "Bounded replay does not expose uncomputed prompt hidden rows";
    }
    if (!selected.has_value() || selected->numel() == 0) {
      return hidden;
    }
    CHECK(selected->defined());
    CHECK_EQ(selected->device(), hidden.device());
    CHECK_EQ(selected->dim(), 1);
    CHECK(selected->scalar_type() == torch::kInt32 ||
          selected->scalar_type() == torch::kInt64);
    torch::Tensor rows = selected->to(torch::kInt64);
    if (last_output_row_offset_ > 0) {
      rows = rows - last_output_row_offset_;
      CHECK_GE(rows.min().item<int64_t>(), 0)
          << "Bounded replay cannot return earlier prompt logits";
    }
    return hidden.index_select(0, rows);
  }

  torch::Tensor project_logits(const torch::Tensor& input) const {
    const torch::Tensor local = torch::linear(input, head_);
    return parallel_state::gather(local, attention_group_, /*dim=*/-1);
  }

 private:
  torch::Tensor embedding_;
  torch::Tensor norm_;
  torch::Tensor head_;
  torch::Tensor plain_rope_;
  torch::Tensor compressed_rope_;
  double norm_epsilon_;
  bool bounded_replay_ = false;
  int64_t replay_window_ = 0;
  int64_t late_layer_start_ = 0;
  int64_t last_output_row_offset_ = 0;
  std::vector<std::unique_ptr<DeepseekV41Decoder>> layers_;
  std::vector<bool> compressed_layers_;
  std::vector<int64_t> source_ratios_;
  ProcessGroup* attention_group_;
};

}  // namespace xllm::layer
