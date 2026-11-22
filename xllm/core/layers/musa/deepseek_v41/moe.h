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
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/musa/deepseek_v41/deep_ep_ll_runtime.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"
#include "core/layers/musa/deepseek_v41/prefill_tail.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"
#include "core/layers/musa/deepseek_v41/w4a8_experts.h"
#include "core/util/env_var.h"

namespace xllm::layer {

enum class DeepseekV41ExpertFormat : int8_t { ORIGINAL_FP4, W4AFP8_MT };

struct DeepseekV41MoeConfig {
  std::shared_ptr<DeepseekV41DeepEpLlRuntime> deep_ep_ll;
  DeepseekV41ExpertFormat expert_format = DeepseekV41ExpertFormat::ORIGINAL_FP4;
  bool dense = false;
  bool gate_cache_fp32_weight = false;
  bool prefill_gate_row_partition = false;
  bool prefill_compact_experts = false;
  bool gate_bf16_fp32_projection = false;
  bool gate_simt_projection = false;
  bool prefill_gate_bf16_fp32_projection = false;
  bool original_fp8_activation = true;
  bool decode_ep_custom_allreduce = false;
  bool verify_custom_allreduce = false;
  bool prefill_custom_allreduce = false;
  bool shared_expert_tp_reduce = false;
  bool decode_packed_shared_gate_up = false;
  bool verify_skinny_projection = false;
  bool w4a8_static_activation = false;
  bool w4a8_valid_prefix_swiglu = false;
  bool prefill_valid_prefix_swiglu = false;
  bool prefill_parallel_swiglu = false;
  bool prefill_shared_row_partition = false;
  int64_t first_expert = 0;
  int64_t topk = 6;
  int64_t image_token_id = 129264;
  bool normalize = true;
  double scale = 1.5;
  double limit = 10.0;
  std::string bmm_so_path;
  std::string w4a8_so_path;
  std::string dense_prequant_directory;
  int64_t max_partial_bytes = 512 * 1024 * 1024;
};

struct DeepseekV41MoeWeights {
  // Shared/dense w1,w3,w2. Routed packed matrices and scales same ordering.
  std::vector<torch::Tensor> shared;
  std::vector<torch::Tensor> shared_fp8_scales;
  std::array<torch::Tensor, 3> expert;
  std::array<torch::Tensor, 3> scale;
  std::array<double, 2> expert_input_scale = {0.0, 0.0};
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
                config_.original_fp8_activation,
                config_.decode_packed_shared_gate_up,
                config_.verify_skinny_projection,
                weights_.shared_fp8_scales,
                config_.dense_prequant_directory),
        ep_group_(ep_group) {
    if (config_.dense) {
      return;
    }
    CHECK(!config_.prefill_shared_row_partition ||
          (ep_group_ != nullptr && ep_group_->world_size() == 8 &&
           !config_.shared_expert_tp_reduce && !config_.deep_ep_ll));
    CHECK(!config_.prefill_gate_row_partition ||
          (ep_group_ != nullptr && ep_group_->world_size() == 8 &&
           !config_.deep_ep_ll && !config_.prefill_gate_bf16_fp32_projection));
    CHECK_GE(config_.first_expert, 0);
    CHECK(weights_.gate.defined());
    CHECK_EQ(weights_.gate.dim(), 2);
    if (config_.shared_expert_tp_reduce && config_.verify_custom_allreduce &&
        config_.expert_format == DeepseekV41ExpertFormat::W4AFP8_MT &&
        util::get_bool_env("XLLM_DSV41_VERIFY_SHARED_EXPERT_OVERLAP", false)) {
      verify_shared_overlap_ =
          std::make_unique<DeepseekV41VerifyOverlap>(weights_.gate.device());
    }
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
    gate_ = std::make_unique<DeepseekV41Gate>(
        weights_.gate,
        weights_.bias,
        weights_.vision_bias,
        config_.image_token_id,
        config_.topk,
        config_.normalize,
        config_.scale,
        config_.gate_cache_fp32_weight,
        config_.gate_bf16_fp32_projection,
        config_.gate_simt_projection,
        config_.prefill_gate_bf16_fp32_projection);
    if (config_.expert_format == DeepseekV41ExpertFormat::W4AFP8_MT) {
      w4a8_ = std::make_unique<DeepseekV41W4A8Experts>(
          weights_.expert[0],
          weights_.scale[0],
          weights_.expert[1],
          weights_.scale[1],
          config_.first_expert,
          config_.limit,
          config_.w4a8_so_path,
          weights_.expert_input_scale[0],
          weights_.expert_input_scale[1],
          config_.w4a8_valid_prefix_swiglu,
          config_.prefill_valid_prefix_swiglu,
          config_.prefill_parallel_swiglu);
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
    torch::Tensor shared_local;
    if (verify_shared_overlap_ && input.size(0) > 1 && input.size(0) <= 6 &&
        verify_shared_overlap_->can_fork()) {
      shared_local = verify_shared_overlap_->launch(
          input, [&]() { return shared_.forward(input).to(torch::kFloat32); });
      LOG_FIRST_N(INFO, 1)
          << "Native V4.1 overlaps shared and routed verify experts: rows="
          << input.size(0);
    }
    torch::Tensor route_weights;
    torch::Tensor indices;
    {
      const DeepseekV41ProfileRange gate_range("V41/MoE/gate");
      std::tie(route_weights, indices) = gate_->forward(input, input_ids);
    }
    if (config_.deep_ep_ll) {
      CHECK(!config_.shared_expert_tp_reduce);
      CHECK(w4a8_ != nullptr);
      const int64_t capacity = config_.deep_ep_ll->capacity();
      std::vector<torch::Tensor> chunks;
      chunks.reserve((input.size(0) + capacity - 1) / capacity);
      for (int64_t start = 0; start < input.size(0); start += capacity) {
        const int64_t count = std::min(capacity, input.size(0) - start);
        chunks.emplace_back(forward_dispatched(
            input.narrow(0, start, count).contiguous(),
            indices.narrow(0, start, count).contiguous(),
            route_weights.narrow(0, start, count).contiguous()));
      }
      CHECK(!chunks.empty());
      torch::Tensor output =
          chunks.size() == 1 ? chunks.front() : torch::cat(chunks, 0);
      // LL combine already applies routing weights and merges every expert.
      // Add the full owner shared expert once, with no routed TP/EP sum.
      return (output.to(torch::kFloat32) +
              shared_.forward(input).to(torch::kFloat32))
          .to(input.scalar_type());
    }
    torch::Tensor routed;
    if (config_.expert_format == DeepseekV41ExpertFormat::W4AFP8_MT) {
      routed = forward_w4a8(input, indices, route_weights);
    } else if (input.size(0) <= 8) {
      routed = decode_->forward(input, indices, route_weights);
    } else {
      routed = local_experts(input, indices, route_weights);
    }
    if (config_.shared_expert_tp_reduce) {
      CHECK(ep_group_ != nullptr && ep_group_->world_size() > 1);
      // Local shared slices must join the routed FP32 sum exactly once.
      // The same reduction now combines both sources, with no extra collective.
      if (shared_local.defined()) {
        verify_shared_overlap_->join(shared_local);
      } else {
        shared_local = shared_.forward(input).to(torch::kFloat32);
      }
      routed.add_(shared_local);
    }
    if (ep_group_ != nullptr && ep_group_->world_size() > 1) {
      const DeepseekV41ProfileRange reduction_range(
          "V41/MoE/EP_allreduce",
          /*layer=*/-1,
          /*tokens=*/routed.size(0),
          /*context=*/-1,
          /*payload_bytes=*/static_cast<int64_t>(routed.nbytes()),
          /*group_size=*/ep_group_->world_size());
      const int64_t custom_rows = config_.verify_custom_allreduce ? 6 : 1;
      if ((config_.decode_ep_custom_allreduce && input.size(0) > 0 &&
           input.size(0) <= custom_rows) ||
          (config_.prefill_custom_allreduce && input.size(0) > 6 &&
           input.size(0) <= 512)) {
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
    if (!config_.shared_expert_tp_reduce) {
      // The original shared output is replicated and joins after the EP sum.
      routed.add_(forward_shared(input).to(torch::kFloat32));
    }
    return routed.to(input.scalar_type());
  }

  torch::Tensor forward_partitioned_prefill(
      const torch::Tensor& local,
      const std::optional<torch::Tensor>& input_ids) {
    const DeepseekV41ProfileRange range("V41/MoE/partitioned");
    if (config_.dense) {
      return shared_.forward(local);
    }
    CHECK(w4a8_ != nullptr && ep_group_ != nullptr);
    CHECK(!config_.deep_ep_ll && !config_.shared_expert_tp_reduce &&
          !config_.prefill_custom_allreduce);
    torch::Tensor input =
        ep_group_->allgather_base_sync(local).reshape({-1, local.size(1)});
    torch::Tensor weights;
    torch::Tensor indices;
    {
      const DeepseekV41ProfileRange gate_range("V41/MoE/gate");
      std::tie(weights, indices) =
          config_.prefill_gate_row_partition
              ? forward_gate_partition(local, input_ids)
              : gate_->forward(input, input_ids);
    }
    torch::Tensor routed;
    if (config_.prefill_compact_experts &&
        input.size(0) > kDeepseekV41PrefillTileRows) {
      routed = w4a8_->forward_compact(input, indices, weights);
    }
    if (!routed.defined()) {
      routed = forward_w4a8(input, indices, weights);
    }
    CHECK_EQ(routed.scalar_type(), torch::kFloat32);
    torch::Tensor output = torch::empty(local.sizes(), routed.options());
    {
      const DeepseekV41ProfileRange reduction_range(
          "V41/MoE/EP_reduce_scatter");
      ep_group_->reduce_scatter(routed, output);
    }
    {
      const DeepseekV41ProfileRange shared_range("V41/MoE/shared");
      output.add_(shared_.forward(local).to(torch::kFloat32));
    }
    return output.to(local.scalar_type());
  }

  void forward_empty(int64_t owner_rows, const torch::TensorOptions& options) {
    CHECK(config_.deep_ep_ll != nullptr && w4a8_ != nullptr);
    CHECK_GT(owner_rows, 0);
    const torch::Tensor input =
        torch::empty({0, weights_.gate.size(1)}, options);
    const torch::Tensor ids =
        torch::empty({0, config_.topk}, options.dtype(torch::kInt64));
    const torch::Tensor weights =
        torch::empty({0, config_.topk}, options.dtype(torch::kFloat32));
    for (int64_t start = 0; start < owner_rows;
         start += config_.deep_ep_ll->capacity()) {
      forward_dispatched(input, ids, weights);
    }
  }

 private:
  std::tuple<torch::Tensor, torch::Tensor> forward_gate_partition(
      const torch::Tensor& local,
      const std::optional<torch::Tensor>& input_ids) const {
    const int64_t rows = local.size(0);
    const int64_t total = rows * ep_group_->world_size();
    std::optional<torch::Tensor> local_ids;
    if (input_ids.has_value() && input_ids->defined()) {
      CHECK_EQ(input_ids->numel(), total);
      local_ids =
          input_ids->reshape({-1}).narrow(0, ep_group_->rank() * rows, rows);
    }
    const auto [weights, indices] = gate_->forward(local, local_ids);
    CHECK_EQ(weights.scalar_type(), torch::kFloat32);
    CHECK_EQ(indices.scalar_type(), torch::kInt64);
    const int64_t topk = weights.size(1);
    // Keep FP32 route weights bitwise and carry the bounded expert IDs as i32.
    const torch::Tensor packed =
        torch::cat({weights.view(torch::kInt32), indices.to(torch::kInt32)}, 1);
    const DeepseekV41ProfileRange range("V41/MoE/gate_rows_allgather");
    const torch::Tensor all =
        ep_group_->allgather_base_sync(packed).reshape({total, 2 * topk});
    return {all.narrow(1, 0, topk).contiguous().view(torch::kFloat32),
            all.narrow(1, topk, topk).to(torch::kInt64).contiguous()};
  }

  torch::Tensor forward_w4a8(const torch::Tensor& input,
                             const torch::Tensor& indices,
                             const torch::Tensor& weights) const {
    if (input.size(0) <= kDeepseekV41PrefillTileRows) {
      return w4a8_->forward(input, indices, weights);
    }
    CHECK_LE(input.size(0), kDeepseekV41PrefillMaxRows);
    std::vector<torch::Tensor> expert_tiles;
    deepseek_v41_for_each_prefill_tile(
        input.size(0), [&](int64_t start, int64_t count) {
          expert_tiles.emplace_back(
              w4a8_->forward(input.narrow(0, start, count),
                             indices.narrow(0, start, count),
                             weights.narrow(0, start, count)));
        });
    return torch::cat(expert_tiles, 0);
  }

  torch::Tensor forward_shared(const torch::Tensor& input) {
    const DeepseekV41ProfileRange range("V41/MoE/shared");
    if (!config_.prefill_shared_row_partition || input.size(0) <= 8) {
      return shared_.forward(input);
    }
    CHECK(ep_group_ != nullptr);
    CHECK_LE(input.size(0), 512);
    const int64_t world = ep_group_->world_size();
    CHECK_EQ(input.size(0) % world, 0);
    const int64_t count = input.size(0) / world;
    // Full shared weights stay replicated. Only independent token rows are
    // partitioned, so there is no partial-output sum or duplicate EP reduce.
    const torch::Tensor local =
        shared_.forward(input.narrow(0, ep_group_->rank() * count, count));
    const DeepseekV41ProfileRange gather_range("V41/MoE/shared_allgather");
    return ep_group_->allgather_base_sync(local.contiguous())
        .reshape(input.sizes());
  }

  torch::Tensor forward_dispatched(const torch::Tensor& input,
                                   const torch::Tensor& ids,
                                   const torch::Tensor& weights) {
    const DeepseekV41ProfileRange range("V41/MoE/DeepEP_LL");
    auto& bridge = config_.deep_ep_ll->bridge();
    auto [received, counts] = bridge.dispatch(input, ids);
    return bridge.combine(
        w4a8_->forward_dispatched(received, counts), ids, weights);
  }

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
  std::unique_ptr<DeepseekV41VerifyOverlap> verify_shared_overlap_;
  std::unique_ptr<DeepseekV41W4A8Experts> w4a8_;
  std::vector<DeepseekV41PartitionRoutedMlp> experts_;
  ProcessGroup* ep_group_;
};

}  // namespace xllm::layer
