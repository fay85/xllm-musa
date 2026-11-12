/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/
#pragma once
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/layers/musa/deepseek_v41/decoder.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"
#include "core/layers/musa/deepseek_v41/rms_norm.h"
namespace xllm::layer {

// Native DSpark domain object. The CausalLM adapter owns checkpoint loading,
// proposal selection, target verification and accepted-length/cache rollback.
// This object never treats speculative query KV as accepted context.
class DeepseekV41DraftModel final {
 public:
  DeepseekV41DraftModel(std::vector<DeepseekV41DecoderConfig> configs,
                        std::vector<DeepseekV41DecoderWeights> weights,
                        torch::Tensor embedding,
                        torch::Tensor main_projection,
                        torch::Tensor main_norm,
                        torch::Tensor output_norm,
                        torch::Tensor head,
                        torch::Tensor rope,
                        ProcessGroup* attention_group = nullptr,
                        ProcessGroup* expert_group = nullptr)
      : embedding_(std::move(embedding)),
        main_projection_(std::move(main_projection),
                         /*original_fp8_activation=*/true),
        main_norm_(std::move(main_norm)),
        output_norm_(std::move(output_norm)),
        head_(std::move(head)),
        rope_(std::move(rope)),
        attention_group_(attention_group) {
    CHECK_EQ(configs.size(), 3);
    CHECK_EQ(weights.size(), configs.size());
    CHECK(embedding_.defined() && main_norm_.defined() &&
          output_norm_.defined());
    CHECK(embedding_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(embedding_.scalar_type(), torch::kBFloat16);
    CHECK_EQ(main_norm_.device(), embedding_.device());
    CHECK_EQ(output_norm_.device(), embedding_.device());
    CHECK_EQ(head_.device(), embedding_.device());
    CHECK_EQ(rope_.device(), embedding_.device());
    CHECK_EQ(embedding_.dim(), 2);
    CHECK_EQ(main_norm_.dim(), 1);
    CHECK_EQ(output_norm_.dim(), 1);
    CHECK_EQ(head_.dim(), 2);
    const int64_t world =
        attention_group_ == nullptr ? 1 : attention_group_->world_size();
    CHECK_EQ(embedding_.size(1) * world, main_norm_.numel());
    CHECK_EQ(output_norm_.numel(), main_norm_.numel());
    CHECK_EQ(head_.size(1), main_norm_.numel());
    CHECK_EQ(head_.size(0) * world, embedding_.size(0));
    epsilon_ = configs.front().attention.epsilon;
    layers_.reserve(configs.size());
    for (size_t index = 0; index < configs.size(); ++index) {
      CHECK_EQ(configs[index].attention.layer, static_cast<int64_t>(index));
      CHECK(configs[index].attention.mode == DeepseekV41AttentionMode::SWA);
      CHECK_EQ(configs[index].attention.ratio, 0);
      CHECK(!configs[index].decode_ffn_tail_graph);
      CHECK(weights[index].engram.empty());
      layers_.emplace_back(
          std::make_unique<DeepseekV41Decoder>(std::move(configs[index]),
                                               std::move(weights[index]),
                                               attention_group_,
                                               expert_group));
    }
  }

  torch::Tensor write_context_kv(const torch::Tensor& captured,
                                 const torch::Tensor& positions,
                                 const torch::Tensor& slots,
                                 const std::vector<torch::Tensor>& swa_caches) {
    CHECK_EQ(captured.dim(), 2);
    CHECK_EQ(captured.size(1), 3 * main_norm_.numel());
    CHECK_EQ(captured.size(0), positions.numel());
    CHECK_EQ(captured.size(0), slots.numel());
    CHECK_EQ(swa_caches.size(), layers_.size());
    torch::DeviceGuard guard(embedding_.device());
    const torch::Tensor projected = deepseek_v41_rms_norm(
        main_projection_.forward(captured), main_norm_, epsilon_);
    for (size_t index = 0; index < layers_.size(); ++index) {
      layers_[index]->write_draft_context(
          projected, positions, rope_, swa_caches[index], slots);
    }
    return projected;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const std::vector<torch::Tensor>& swa_caches,
                        const torch::Tensor& table,
                        int64_t previous_context) {
    CHECK_EQ(tokens.dim(), 1);
    CHECK_GT(tokens.numel(), 0);
    CHECK_LE(tokens.numel(), 8);
    CHECK_EQ(positions.numel(), tokens.numel());
    CHECK_EQ(swa_caches.size(), layers_.size());
    CHECK_EQ(tokens.device(), embedding_.device());
    torch::DeviceGuard guard(embedding_.device());
    torch::Tensor hidden =
        torch::embedding(embedding_, tokens.to(torch::kInt64));
    hidden = parallel_state::gather(hidden, attention_group_, /*dim=*/-1);
    hidden = hidden.unsqueeze(1).expand({-1, 4, -1}).contiguous();
    torch::Tensor pre_mix = torch::zeros(
        {tokens.numel(), 4}, hidden.options().dtype(torch::kFloat32));
    pre_mix.select(1, 0).fill_(1.0);
    for (size_t index = 0; index < layers_.size(); ++index) {
      const DeepseekV41ProfileRange range("V41/DSpark/layer",
                                          static_cast<int64_t>(index));
      std::tie(hidden, pre_mix) =
          layers_[index]->forward_draft_block(hidden,
                                              pre_mix,
                                              positions,
                                              rope_,
                                              swa_caches[index],
                                              table,
                                              previous_context,
                                              tokens);
    }
    const torch::Tensor merged = kernel::musa::deepseek_v4_hc_pre_combine(
        hidden.contiguous(), pre_mix.contiguous());
    return deepseek_v41_rms_norm(merged, output_norm_, epsilon_);
  }

  torch::Tensor logits(const torch::Tensor& hidden) const {
    CHECK_EQ(hidden.dim(), 2);
    CHECK_EQ(hidden.size(1), head_.size(1));
    const torch::Tensor local = torch::nn::functional::linear(hidden, head_);
    return parallel_state::gather(local, attention_group_, /*dim=*/-1);
  }

 private:
  torch::Tensor embedding_;
  DeepseekV41Dense main_projection_;
  torch::Tensor main_norm_;
  torch::Tensor output_norm_;
  torch::Tensor head_;
  torch::Tensor rope_;
  ProcessGroup* attention_group_;
  double epsilon_ = 1e-6;
  std::vector<std::unique_ptr<DeepseekV41Decoder>> layers_;
};
}  // namespace xllm::layer
