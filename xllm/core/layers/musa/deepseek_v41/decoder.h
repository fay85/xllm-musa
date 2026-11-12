/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <c10/core/StreamGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/kernels/musa/deepseek_v41_decoder_graph_inputs.h"
#include "core/layers/musa/deepseek_v41/attention.h"
#include "core/layers/musa/deepseek_v41/engram_injection.h"
#include "core/layers/musa/deepseek_v41/hyper_connection.h"
#include "core/layers/musa/deepseek_v41/moe.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"
#include "core/layers/musa/deepseek_v41/rms_norm.h"
#include "core/platform/musa_graph_handle.h"

namespace xllm::layer {

struct DeepseekV41DecoderConfig {
  DeepseekV41AttentionConfig attention;
  DeepseekV41MoeConfig moe;
  int64_t sinkhorn_iterations = 20;
  double hc_epsilon = 1e-6;
  std::string mhc_prenorm_directory;
  bool decode_tp_custom_allreduce = false;
  bool decode_mhc_fused_reduce = false;
  bool decode_ffn_tail_graph = false;
  bool decode_ffn_graph_fused_inputs = false;
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
                      config_.mhc_prenorm_directory,
                      config_.decode_mhc_fused_reduce),
        ffn_hc_(weights_.hyper_connection[3],
                weights_.hyper_connection[4],
                weights_.hyper_connection[5],
                config_.sinkhorn_iterations,
                config_.attention.epsilon,
                config_.hc_epsilon,
                config_.mhc_prenorm_directory,
                config_.decode_mhc_fused_reduce),
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
    return forward_tail(
        attn_output, residual, attn_pre, attn_post, attn_comb, input_ids);
  }

  std::tuple<torch::Tensor, torch::Tensor> forward_batched_decode(
      const torch::Tensor& hidden,
      const torch::Tensor& carried_pre,
      const torch::Tensor& positions,
      const torch::Tensor& rope_table,
      const DeepseekV41AttentionCache& cache,
      const torch::Tensor& swa_table,
      const torch::Tensor& key_table,
      const torch::Tensor& index_table,
      const std::vector<int64_t>& context_lengths,
      std::vector<DeepseekV41AttentionRuntime>& shared,
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
        attention_.forward_batched_decode(norm(attn_input, weights_.input_norm),
                                          positions,
                                          rope_table,
                                          cache,
                                          swa_table,
                                          key_table,
                                          index_table,
                                          context_lengths,
                                          shared);
    return forward_tail(
        attn_output, residual, attn_pre, attn_post, attn_comb, input_ids);
  }

  std::tuple<torch::Tensor, torch::Tensor> forward_draft_block(
      const torch::Tensor& hidden,
      const torch::Tensor& carried_pre,
      const torch::Tensor& positions,
      const torch::Tensor& rope_table,
      const torch::Tensor& swa_cache,
      const torch::Tensor& swa_table,
      int64_t previous_context,
      const torch::Tensor& input_ids) {
    CHECK(engram_ == nullptr);
    CHECK_EQ(config_.attention.ratio, 0);
    CHECK(!config_.decode_ffn_tail_graph)
        << "The first DSpark body only supports explicit eager block execution";
    auto [attn_input, attn_pre, attn_post, attn_comb] =
        attention_hc_.prepare(hidden, carried_pre);
    const torch::Tensor attn_output =
        attention_.forward_draft_block(norm(attn_input, weights_.input_norm),
                                       positions,
                                       rope_table,
                                       swa_cache,
                                       swa_table,
                                       previous_context);
    return forward_tail(
        attn_output, hidden, attn_pre, attn_post, attn_comb, input_ids);
  }

  void write_draft_context(const torch::Tensor& projected,
                           const torch::Tensor& positions,
                           const torch::Tensor& rope_table,
                           const torch::Tensor& swa_cache,
                           const torch::Tensor& slots) {
    CHECK(engram_ == nullptr);
    attention_.write_draft_context(
        projected, positions, rope_table, swa_cache, slots);
  }

  std::tuple<torch::Tensor, torch::Tensor> forward_tail(
      const torch::Tensor& attn_output,
      const torch::Tensor& residual,
      const torch::Tensor& attn_pre,
      const torch::Tensor& attn_post,
      const torch::Tensor& attn_comb,
      const std::optional<torch::Tensor>& input_ids) {
    if (!config_.decode_ffn_tail_graph || attn_output.size(0) != 1) {
      return finish_eager(
          attn_output, residual, attn_pre, attn_post, attn_comb, input_ids);
    }
    const DeepseekV41ProfileRange range("V41/Decoder/FFN_tail_graph_replay");
    CHECK(input_ids.has_value() && input_ids->numel() == 1);
    CHECK(config_.moe.dense ||
          config_.moe.expert_format == DeepseekV41ExpertFormat::W4AFP8_MT);
    CHECK(!kernel::musa::is_stream_capturing())
        << "Nested FFN-tail graph unsupported";
    const std::array<torch::Tensor, 6> inputs = {attn_output,
                                                 residual,
                                                 attn_pre,
                                                 attn_post,
                                                 attn_comb,
                                                 input_ids.value()};
    torch::DeviceGuard guard(attn_output.device());
    if (tail_graph_ == nullptr) {
      capture_tail(inputs);
    }
    if (config_.decode_ffn_graph_fused_inputs) {
      kernel::musa::deepseek_v41_copy_decoder_graph_inputs(inputs,
                                                           tail_inputs_);
    } else {
      for (size_t index = 0; index < inputs.size(); ++index) {
        CHECK_EQ(inputs[index].device(), tail_inputs_[index].device());
        CHECK_EQ(inputs[index].sizes(), tail_inputs_[index].sizes());
        CHECK_EQ(inputs[index].scalar_type(),
                 tail_inputs_[index].scalar_type());
        tail_inputs_[index].copy_(inputs[index]);
      }
    }
    tail_graph_->replay();
    return tail_outputs_;
  }

 private:
  void capture_tail(const std::array<torch::Tensor, 6>& inputs) {
    const DeepseekV41ProfileRange range("V41/Decoder/FFN_tail_graph_capture");
    const int32_t device_index = inputs[0].get_device();
    std::unique_lock<std::shared_mutex> capture_lock(
        cuda::DeviceCaptureLock::get_instance().get_write_lock(device_index));
    for (size_t index = 0; index < inputs.size(); ++index) {
      CHECK_EQ(inputs[index].device(), inputs[0].device());
      tail_inputs_[index] = inputs[index].contiguous().clone();
    }
    const c10::musa::MUSAStream original =
        c10::musa::getCurrentMUSAStream(device_index);
    const c10::musa::MUSAStream capture_stream =
        c10::musa::getStreamFromPool(/*isHighPriority=*/false, device_index);
    original.synchronize();
    capture_stream.synchronize();
    c10::StreamGuard stream_guard(capture_stream.unwrap());
    for (int64_t repeat = 0; repeat < 2; ++repeat) {
      finish_eager(tail_inputs_[0],
                   tail_inputs_[1],
                   tail_inputs_[2],
                   tail_inputs_[3],
                   tail_inputs_[4],
                   tail_inputs_[5]);
    }
    capture_stream.synchronize();
    tail_graph_ = std::make_unique<MusaGraphHandle>();
    static thread_local std::unordered_map<int32_t, c10::musa::MempoolId_t>
        pools;
    auto it = pools.find(device_index);
    if (it == pools.end()) {
      it = pools.emplace(device_index, at::musa::graph_pool_handle()).first;
    }
    tail_graph_->capture_begin(it->second, musaStreamCaptureModeThreadLocal);
    tail_outputs_ = finish_eager(tail_inputs_[0],
                                 tail_inputs_[1],
                                 tail_inputs_[2],
                                 tail_inputs_[3],
                                 tail_inputs_[4],
                                 tail_inputs_[5]);
    tail_graph_->capture_end();
    capture_stream.synchronize();
    // Resolve every captured TP/EP input IPC address before the first replay.
    // Every TP rank takes the same q1 branch and startup policy vote.
    finalize_musa_custom_allreduce_graphs();
    LOG_FIRST_N(INFO, 1) << "Native V4.1 FFN-tail Graph captured, including "
                            "TP, mHC, MoE/shared, EP; "
                            "Attention cache/index/compressor remain eager";
  }

  std::tuple<torch::Tensor, torch::Tensor> finish_eager(
      torch::Tensor attn_output,
      torch::Tensor residual,
      const torch::Tensor& attn_pre,
      const torch::Tensor& attn_post,
      const torch::Tensor& attn_comb,
      const std::optional<torch::Tensor>& input_ids) {
    if (attention_group_ != nullptr && attention_group_->world_size() > 1) {
      const DeepseekV41ProfileRange reduction_range(
          "V41/Attention/TP_allreduce",
          /*layer=*/-1,
          /*tokens=*/attn_output.size(0),
          /*context=*/-1,
          /*payload_bytes=*/static_cast<int64_t>(attn_output.nbytes()),
          /*group_size=*/attention_group_->world_size());
      if (config_.decode_tp_custom_allreduce && attn_output.size(0) == 1) {
        // Selection follows the existing all-rank TP policy. The native
        // custom collective returns separate storage; consume that result.
        // Prefill and the routed-expert FP32 reduction keep their MCCL path.
        std::optional<torch::Tensor> reduced =
            musa_allreduce_tp_custom(attention_group_, attn_output);
        CHECK(reduced.has_value())
            << "Selected native V4.1 decode TP custom reduction is unsupported";
        attn_output = std::move(reduced.value());
        LOG_FIRST_N(INFO, 1)
            << "Native V4.1 decode Attention TP backend=custom, group_size="
            << attention_group_->world_size()
            << ", bytes=" << attn_output.nbytes();
      } else {
        attention_group_->allreduce(attn_output);
      }
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
  std::array<torch::Tensor, 6> tail_inputs_;
  std::tuple<torch::Tensor, torch::Tensor> tail_outputs_;
  // Graph lifetime ends before its persistent inputs/output and model weights.
  std::unique_ptr<MusaGraphHandle> tail_graph_;
};

}  // namespace xllm::layer
