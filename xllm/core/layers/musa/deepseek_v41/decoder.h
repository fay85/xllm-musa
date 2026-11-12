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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/layers/musa/deepseek_v41/attention.h"
#include "core/layers/musa/deepseek_v41/engram_injection.h"
#include "core/layers/musa/deepseek_v41/hyper_connection.h"
#include "core/layers/musa/deepseek_v41/moe.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"
#include "core/layers/musa/deepseek_v41/rms_norm.h"

namespace xllm::layer {

struct DeepseekV41DecoderConfig {
  DeepseekV41AttentionConfig attention;
  DeepseekV41MoeConfig moe;
  int64_t sinkhorn_iterations = 20;
  double hc_epsilon = 1e-6;
  std::string mhc_prenorm_directory;
};

struct DeepseekV41DecoderWeights {
  DeepseekV41AttentionWeights attention;
  DeepseekV41MoeWeights moe;
  // attn fn,scale,base; ffn fn,scale,base. All HC weights stay FP32.
  std::array<torch::Tensor, 6> hyper_connection;
  torch::Tensor input_norm;
  torch::Tensor post_attention_norm;
  // wkv, query norm, key norm; an empty vector means this layer has no Engram.
  std::vector<torch::Tensor> engram;
};

// A full native decoder layer: shifted mHC -> norm -> local CSA2 -> TP sum
// -> mHC -> norm -> EP MoE/shared expert -> mHC. The caller supplies gathered
// Engram rows; hash/table ownership remains at the model/executor boundary.
class DeepseekV41Decoder final {
 public:
  DeepseekV41Decoder(DeepseekV41DecoderConfig config,
                     DeepseekV41DecoderWeights weights,
                     ProcessGroup* attention_group = nullptr,
                     ProcessGroup* expert_group = nullptr)
      : config_(std::move(config)),
        weights_(std::move(weights)),
        attention_(config_.attention, weights_.attention),
        moe_(config_.moe, weights_.moe, expert_group),
        attention_hc_(weights_.hyper_connection[0],
                      weights_.hyper_connection[1],
                      weights_.hyper_connection[2],
                      config_.sinkhorn_iterations,
                      config_.attention.epsilon,
                      config_.hc_epsilon,
                      config_.mhc_prenorm_directory),
        ffn_hc_(weights_.hyper_connection[3],
                weights_.hyper_connection[4],
                weights_.hyper_connection[5],
                config_.sinkhorn_iterations,
                config_.attention.epsilon,
                config_.hc_epsilon,
                config_.mhc_prenorm_directory),
        attention_group_(attention_group) {
    check_norm(weights_.input_norm);
    check_norm(weights_.post_attention_norm);
    CHECK_EQ(weights_.input_norm.sizes(), weights_.post_attention_norm.sizes());
    if (!weights_.engram.empty()) {
      CHECK_EQ(weights_.engram.size(), 3);
      engram_ = std::make_unique<DeepseekV41EngramInjection>(
          weights_.engram[0],
          weights_.engram[1],
          weights_.engram[2],
          config_.attention.original_fp8_activation);
    }
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden,
      const torch::Tensor& carried_pre,
      const torch::Tensor& positions,
      const torch::Tensor& rope_table,
      const DeepseekV41AttentionCache& cache,
      const torch::Tensor& swa_table,
      const torch::Tensor& key_table,
      const torch::Tensor& index_table,
      int64_t context_length,
      DeepseekV41AttentionRuntime& shared,
      const std::optional<torch::Tensor>& input_ids,
      const std::optional<torch::Tensor>& engram_embedding,
      const std::optional<torch::Tensor>& engram_mask) {
    CHECK_EQ(engram_embedding.has_value(), engram_ != nullptr);
    torch::Tensor residual = hidden;
    if (engram_) {
      residual =
          engram_->forward(hidden, engram_embedding.value(), engram_mask);
    }
    auto [attn_input, attn_pre, attn_post, attn_comb] =
        attention_hc_.prepare(residual, carried_pre);
    torch::Tensor attn_output =
        attention_.forward(norm(attn_input, weights_.input_norm),
                           positions,
                           rope_table,
                           cache,
                           swa_table,
                           key_table,
                           index_table,
                           context_length,
                           shared);
    if (attention_group_ != nullptr && attention_group_->world_size() > 1) {
      const DeepseekV41ProfileRange reduction_range(
          "V41/Attention/TP_allreduce",
          /*layer=*/-1,
          /*tokens=*/attn_output.size(0),
          /*context=*/-1,
          /*payload_bytes=*/static_cast<int64_t>(attn_output.nbytes()),
          /*group_size=*/attention_group_->world_size());
      attention_group_->allreduce(attn_output);
    }
    residual =
        attention_hc_.combine(attn_output, residual, attn_post, attn_comb);
    // Shifted pre: FFN consumes this layer's ATTENTION pre, and returns its
    // own pre for the NEXT layer's attention. Never collapse with its own pre.
    auto [ffn_input, ffn_pre, ffn_post, ffn_comb] =
        ffn_hc_.prepare(residual, attn_pre);
    const torch::Tensor ffn_output =
        moe_.forward(norm(ffn_input, weights_.post_attention_norm), input_ids);
    return {ffn_hc_.combine(ffn_output, residual, ffn_post, ffn_comb), ffn_pre};
  }

 private:
  void check_norm(const torch::Tensor& value) const {
    CHECK(value.defined());
    CHECK_EQ(value.device(), weights_.hyper_connection[0].device());
    CHECK_EQ(value.scalar_type(), torch::kBFloat16);
    CHECK_EQ(value.dim(), 1);
    CHECK_EQ(value.numel() * 4, weights_.hyper_connection[0].size(1));
  }

  torch::Tensor norm(const torch::Tensor& value,
                     const torch::Tensor& weight) const {
    return deepseek_v41_rms_norm(value, weight, config_.attention.epsilon);
  }

  DeepseekV41DecoderConfig config_;
  DeepseekV41DecoderWeights weights_;
  DeepseekV41Attention attention_;
  DeepseekV41Moe moe_;
  DeepseekV41HyperConnection attention_hc_;
  DeepseekV41HyperConnection ffn_hc_;
  std::unique_ptr<DeepseekV41EngramInjection> engram_;
  ProcessGroup* attention_group_;
};

}  // namespace xllm::layer
