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
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

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
                       ProcessGroup* expert_group = nullptr)
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
    const bool diagnostic_progress =
        std::getenv("XLLM_DSV41_DIAGNOSTIC_PROGRESS") != nullptr &&
        input_ids.numel() > 1;
    for (size_t index = 0; index < layers_.size(); ++index) {
      if (diagnostic_progress) {
        LOG(INFO) << "V41_DIAGNOSTIC layer begin index=" << index
                  << " q=" << input_ids.numel() << " kv=" << context_length;
      }
      const DeepseekV41ProfileRange layer_range("V41/layer",
                                                static_cast<int64_t>(index));
      const torch::Tensor& table =
          source_ratios_[index] == 1 ? ratio1_table : ratio2_table;
      std::tie(hidden, pre_mix) = layers_[index]->forward(
          hidden,
          pre_mix,
          positions,
          compressed_layers_[index] ? compressed_rope_ : plain_rope_,
          caches[index],
          swa_table,
          table,
          table,
          context_length,
          shared,
          input_ids,
          engram_rows[index],
          engram_mask);
      if (diagnostic_progress) {
        LOG(INFO) << "V41_DIAGNOSTIC layer enqueued index=" << index
                  << " q=" << input_ids.numel() << " kv=" << context_length;
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
    torch::Tensor input = hidden;
    if (selected.has_value()) {
      CHECK(selected->defined());
      CHECK_EQ(selected->device(), hidden.device());
      CHECK_EQ(selected->dim(), 1);
      CHECK(selected->scalar_type() == torch::kInt32 ||
            selected->scalar_type() == torch::kInt64);
      if (selected->numel() > 0) {
        input = hidden.index_select(0, selected->to(torch::kInt64));
      }
    }
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
  std::vector<std::unique_ptr<DeepseekV41Decoder>> layers_;
  std::vector<bool> compressed_layers_;
  std::vector<int64_t> source_ratios_;
  ProcessGroup* attention_group_;
};

}  // namespace xllm::layer
