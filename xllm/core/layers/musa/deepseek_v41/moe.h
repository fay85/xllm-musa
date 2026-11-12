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
#include <utility>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"
#include "core/layers/musa/deepseek_v41/w4a8_experts.h"

namespace xllm::layer {

enum class DeepseekV41ExpertFormat : int8_t { ORIGINAL_FP4, W4AFP8_MT };

struct DeepseekV41MoeConfig {
  DeepseekV41ExpertFormat expert_format = DeepseekV41ExpertFormat::ORIGINAL_FP4;
  bool dense = false;
  bool original_fp8_activation = true;
  bool decode_ep_custom_allreduce = false;
  int64_t first_expert = 0;
  int64_t topk = 6;
  int64_t image_token_id = 129264;
  bool normalize = true;
  double scale = 1.5;
  double limit = 10.0;
  std::string bmm_so_path;
  std::string w4a8_so_path;
  int64_t max_partial_bytes = 512 * 1024 * 1024;
};

struct DeepseekV41MoeWeights {
  // Shared/dense w1,w3,w2. Routed packed matrices and scales same ordering.
  std::vector<torch::Tensor> shared;
  std::array<torch::Tensor, 3> expert;
  std::array<torch::Tensor, 3> scale;
  torch::Tensor gate;
  torch::Tensor bias;
  std::optional<torch::Tensor> vision_bias;
};

// Native EP-local experts and replicated shared expert. The framework owns
// the non-owning EP group for longer than the model. All ranks participate
// once even when their local expert list is empty. No Python forward call.
// Large-M retains the current eager segment-metadata D2H boundary, NOT a
// grouped-MegaMoE implementation; all projection/activation math stays GPU.
class DeepseekV41Moe final {
 public:
  DeepseekV41Moe(DeepseekV41MoeConfig config,
                 DeepseekV41MoeWeights weights,
                 ProcessGroup* ep_group = nullptr)
      : config_(std::move(config)),
        weights_(std::move(weights)),
        shared_(shared_weight(0),
                shared_weight(1),
                shared_weight(2),
                config_.limit,
                config_.original_fp8_activation),
        ep_group_(ep_group) {
    if (config_.dense) {
      return;
    }
    CHECK_GE(config_.first_expert, 0);
    CHECK(weights_.gate.defined());
    CHECK_EQ(weights_.gate.dim(), 2);
    CHECK_EQ(weights_.expert[0].dim(), 3);
    const int64_t count = weights_.expert[0].size(0);
    CHECK_GT(count, 0);
    CHECK_LE(config_.first_expert + count, weights_.gate.size(0));
    if (ep_group_ == nullptr) {
      CHECK_EQ(config_.first_expert, 0);
      CHECK_EQ(count, weights_.gate.size(0))
          << "A partial expert slice requires its native EP process group";
    } else {
      CHECK_EQ(count * ep_group_->world_size(), weights_.gate.size(0));
      CHECK_EQ(config_.first_expert, count * ep_group_->rank());
    }
    for (int32_t index = 0; index < 3; ++index) {
      if (config_.expert_format == DeepseekV41ExpertFormat::W4AFP8_MT &&
          index == 2) {
        continue;
      }
      CHECK(weights_.expert[index].defined() &&
            weights_.scale[index].defined());
      CHECK_EQ(weights_.expert[index].dim(), 3);
      CHECK_EQ(weights_.scale[index].dim(), 3);
      CHECK_EQ(weights_.expert[index].size(0), count);
      CHECK_EQ(weights_.scale[index].size(0), count);
    }
    gate_ = std::make_unique<DeepseekV41Gate>(weights_.gate,
                                              weights_.bias,
                                              weights_.vision_bias,
                                              config_.image_token_id,
                                              config_.topk,
                                              config_.normalize,
                                              config_.scale);
    if (config_.expert_format == DeepseekV41ExpertFormat::W4AFP8_MT) {
      w4a8_ = std::make_unique<DeepseekV41W4A8Experts>(weights_.expert[0],
                                                       weights_.scale[0],
                                                       weights_.expert[1],
                                                       weights_.scale[1],
                                                       config_.first_expert,
                                                       config_.limit,
                                                       config_.w4a8_so_path);
      return;
    }
    CHECK(!config_.bmm_so_path.empty());
    decode_ = std::make_unique<DeepseekV41PackedRoutedExperts>(
        weights_.expert[0],
        weights_.scale[0],
        weights_.expert[1],
        weights_.scale[1],
        weights_.expert[2],
        weights_.scale[2],
        config_.first_expert,
        config_.limit,
        /*block32_quant=*/true);
    experts_.reserve(count);
    for (int64_t index = 0; index < count; ++index) {
      experts_.emplace_back(weights_.expert[0].select(0, index),
                            weights_.scale[0].select(0, index),
                            weights_.expert[1].select(0, index),
                            weights_.scale[1].select(0, index),
                            weights_.expert[2].select(0, index),
                            weights_.scale[2].select(0, index),
                            config_.limit,
                            config_.bmm_so_path,
                            config_.max_partial_bytes);
    }
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const std::optional<torch::Tensor>& input_ids) {
    const DeepseekV41ProfileRange range("V41/MoE");
    if (config_.dense) {
      return shared_.forward(input);
    }
    auto [route_weights, indices] = gate_->forward(input, input_ids);
    torch::Tensor routed;
    if (config_.expert_format == DeepseekV41ExpertFormat::W4AFP8_MT) {
      routed = w4a8_->forward(input, indices, route_weights);
    } else if (input.size(0) <= 8) {
      routed = decode_->forward(input, indices, route_weights);
    } else {
      routed = local_experts(input, indices, route_weights);
    }
    if (ep_group_ != nullptr && ep_group_->world_size() > 1) {
      const DeepseekV41ProfileRange reduction_range(
          "V41/MoE/EP_allreduce",
          /*layer=*/-1,
          /*tokens=*/routed.size(0),
          /*context=*/-1,
          /*payload_bytes=*/static_cast<int64_t>(routed.nbytes()),
          /*group_size=*/ep_group_->world_size());
      if (config_.decode_ep_custom_allreduce && input.size(0) == 1) {
        CHECK_EQ(routed.scalar_type(), torch::kFloat32);
        std::optional<torch::Tensor> reduced =
            musa_allreduce_tp_custom(ep_group_, routed);
        CHECK(reduced.has_value())
            << "Selected native V4.1 decode EP custom reduction is unsupported";
        routed = std::move(reduced.value());
        LOG_FIRST_N(INFO, 1)
            << "Native V4.1 decode routed EP backend=custom, group_size="
            << ep_group_->world_size() << ", bytes=" << routed.nbytes();
      } else {
        ep_group_->allreduce(routed);
      }
    }
    // Shared output is replicated in this baseline and must not be included
    // in the EP sum. FP32 routed+shared accumulation rounds to BF16 once.
    routed.add_(shared_.forward(input).to(torch::kFloat32));
    return routed.to(input.scalar_type());
  }

 private:
  torch::Tensor shared_weight(int64_t index) const {
    CHECK_EQ(weights_.shared.size(), 3);
    return weights_.shared[index];
  }

  torch::Tensor local_experts(const torch::Tensor& input,
                              const torch::Tensor& ids,
                              const torch::Tensor& weights) {
    const torch::Tensor flat = ids.reshape({-1});
    const int64_t expert_count = static_cast<int64_t>(experts_.size());
    const torch::Tensor local =
        torch::nonzero(
            flat.ge(config_.first_expert)
                .logical_and(flat.lt(config_.first_expert + expert_count)))
            .flatten();
    const torch::Tensor local_ids = flat.index_select(0, local);
    const torch::Tensor order = torch::argsort(local_ids,
                                               /*stable=*/true,
                                               /*dim=*/0,
                                               /*descending=*/false);
    const torch::Tensor routes = local.index_select(0, order);
    const torch::Tensor sorted = local_ids.index_select(0, order);
    const torch::Tensor rows = torch::floor_divide(routes, config_.topk);
    const torch::Tensor route_weights =
        weights.flatten().index_select(0, routes);
    auto [active, inverse, counts] =
        torch::unique_consecutive(sorted,
                                  /*return_inverse=*/false,
                                  /*return_counts=*/true);
    // Host metadata only, once per layer: no CPU activation/weight math.
    const torch::Tensor segments =
        torch::stack({active - config_.first_expert, counts}, 1)
            .to(torch::kCPU)
            .to(torch::kInt64)
            .contiguous();
    const auto segments_data = segments.accessor<int64_t, 2>();
    torch::Tensor output =
        torch::zeros_like(input, input.options().dtype(torch::kFloat32));
    int64_t offset = 0;
    for (int64_t segment = 0; segment < segments.size(0); ++segment) {
      const int64_t expert = segments_data[segment][0];
      const int64_t count = segments_data[segment][1];
      CHECK_GE(expert, 0);
      CHECK_LT(expert, experts_.size());
      const torch::Tensor token_rows = rows.narrow(0, offset, count);
      const torch::Tensor value =
          experts_[expert].forward(input.index_select(0, token_rows),
                                   route_weights.narrow(0, offset, count));
      output.index_add_(0, token_rows, value.to(torch::kFloat32));
      offset += count;
    }
    return output;
  }

  DeepseekV41MoeConfig config_;
  DeepseekV41MoeWeights weights_;
  DeepseekV41SharedMlp shared_;
  std::unique_ptr<DeepseekV41Gate> gate_;
  std::unique_ptr<DeepseekV41PackedRoutedExperts> decode_;
  std::unique_ptr<DeepseekV41W4A8Experts> w4a8_;
  std::vector<DeepseekV41PartitionRoutedMlp> experts_;
  ProcessGroup* ep_group_;
};

}  // namespace xllm::layer
