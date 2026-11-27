/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#ifndef XLLM_CORE_FRAMEWORK_MODEL_CAUSAL_LM_H_
#define XLLM_CORE_FRAMEWORK_MODEL_CAUSAL_LM_H_

// clang-format off
#if defined(USE_NPU)
#include "graph/types.h"
#include "layers/npu/npu_lm_head_impl.h"
#include "layers/npu/npu_word_embedding_impl.h"
#endif
#include "layers/common/lm_head.h"
#include "layers/common/word_embedding.h"
// clang-format on
#include <c10/core/Device.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model_loader.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "model_args.h"
#include "model_input_params.h"
#include "model_output.h"
#include "model_traits.h"

namespace xllm {

// The draft's context update protocol after target verification.
enum class DraftContextUpdate : int8_t {
  // Extend the accepted tail using the existing one- or two-row path.
  TAIL_EXTEND,
  // Replay every accepted token with its target hidden state, using the
  // replay path's token/hidden alignment and position conventions.
  ACCEPTED_SPAN_REPLAY,
};

namespace layer {
struct AttentionMetadata;
}

class ModelGraphMetadataState {
 public:
  virtual ~ModelGraphMetadataState() = default;
  // Called outside graph capture, after capture/replay work is enqueued. Input
  // staging may use this boundary to fence device consumers before reuse.
  // Currently invoked by the MLU graph executor only.
  virtual void finish_input_consumption() {}
};

struct DFlash2CandidateOutput {
  // Candidate vocabulary ids [batch, draft_steps, top_k]. Edge logits use
  // [batch, draft_steps, predecessor_top_k, successor_top_k]; at step zero
  // every predecessor entry represents the same anchor token.
  torch::Tensor candidate_ids;
  torch::Tensor edge_logits;
};

class CausalLM : public torch::nn::Module {
 public:
  ~CausalLM() override = default;

  // tokens: [num_tokens]
  // positions: [num_tokens]
  // returns: [num_tokens, hidden_size]
  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& parameters) = 0;

  virtual bool requires_graph_forward_metadata() { return false; }

  virtual bool is_hybrid_linear_attention() { return false; }

  virtual bool supports_mla_graph_kv_bucketing() const { return false; }

  virtual std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return nullptr;
  }

  virtual void prepare_graph_forward_metadata(ModelGraphMetadataState*,
                                              const torch::Tensor&,
                                              ModelInputParams&) {}

  // Start input-dependent work outside capture before graph consumers run.
  // Called right after prepare_graph_forward_metadata for every capture and
  // replay. Currently invoked by the MLU graph executor only.
  virtual void prepare_graph_forward_inputs(ModelGraphMetadataState*,
                                            const torch::Tensor&,
                                            const ModelInputParams&) {}

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    if (seleted_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, seleted_idxes);
    }
    return hidden_states;
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) = 0;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // out_hidden: [num_selected_tokens, hidden_size]
  // returns: [num_selected_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes,
                               torch::Tensor& out_hidden) {
    NOT_IMPLEMENTED();
    return torch::Tensor();
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader) = 0;

  virtual torch::Device device() const = 0;

  virtual void prepare_expert_weight(
      int32_t layer_id,
      const std::vector<int32_t>& expert_ids) = 0;
  virtual void start_expert_weight_transfer(int32_t /*layer_id*/) {}
  virtual void update_expert_weight(int32_t layer_id) = 0;

  // Returns whether the last prepare_expert_weight() call for the given layer
  // succeeded. Default is true so models that do not fail-report keep the old
  // behavior; DSV4 (npu_torch) overrides this to expose real prepare status
  // so EplbExecutor does not advance ready_layer_id on silent failures.
  virtual bool last_prepare_expert_weight_ok(int32_t /*layer_id*/) const {
    return true;
  }

  virtual const torch::TensorOptions& options() const = 0;

#if defined(USE_NPU)
  virtual bool has_npu_lm_head() const { return false; }
  virtual layer::NpuLmHead get_npu_lm_head() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual bool has_restored_npu_word_embedding() { return false; }

  virtual bool init_or_refresh_rolling_runtime(Stream* load_stream,
                                               Stream* compute_stream,
                                               int32_t num_cached_slots,
                                               int32_t requested_rolling_slots,
                                               const std::string& model_id) {
    NOT_IMPLEMENTED();
    return false;
  }
#endif

  virtual layer::LmHead get_lm_head() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual layer::WordEmbedding get_word_embedding() {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  const torch::Tensor& d2t() const { return d2t_; }

  virtual void share_lm_head(CausalLM&) {}
  virtual void share_embed_tokens(CausalLM&) {}

  virtual std::optional<bool> loaded_vocab_weights() { return std::nullopt; }

  // DFlash-specific interface. Attention-free scatter-only pass that projects
  // target hidden into the draft's per-layer KV cache. Not part of forward()
  // because it has no attention and its shape doesn't match forward's decode
  // graph; sits outside the executor to avoid a per-flag branch in the eager
  // gate. Default: NOT_IMPLEMENTED. Overridden by DFlashDraftModel.
  virtual ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                                       const torch::Tensor& positions,
                                       const torch::Tensor& device_cache_slots,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelInputParams& input_params) {
    NOT_IMPLEMENTED();
    return {};
  }

  virtual DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) {
    NOT_IMPLEMENTED();
    return {};
  }

  // DSpark-specific low-rank Markov projection. The draft worker owns the
  // sequential sampling lifecycle; the model owns only the trained weights and
  // bias computation.
  virtual torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) {
    NOT_IMPLEMENTED();
    return {};
  }

  virtual void lazy_load_model(std::unique_ptr<ModelLoader> loader) {
    NOT_IMPLEMENTED();
  }

  virtual void free_model_weights() { NOT_IMPLEMENTED(); }

  virtual void reload_model_weights() { NOT_IMPLEMENTED(); }

  virtual void reload_model_weights_from_device() { NOT_IMPLEMENTED(); }

 protected:
  void load_d2t(const std::vector<std::unique_ptr<StateDict>>& state_dicts) {
    for (const auto& state_dict : state_dicts) {
      const auto offsets = state_dict->get_tensor(/*tensor_name=*/"d2t");
      if (offsets.defined()) {
        d2t_ = offsets.clone();
        return;
      }
    }
    d2t_ = torch::Tensor();
  }

 private:
  torch::Tensor d2t_;
};

template <typename Model>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  bool requires_graph_forward_metadata() override {
    if constexpr (detail::has_requires_graph_forward_metadata<Model>::value) {
      return model_->requires_graph_forward_metadata();
    } else {
      return CausalLM::requires_graph_forward_metadata();
    }
  }

  bool is_hybrid_linear_attention() override {
    if constexpr (detail::has_is_hybrid_linear_attention<Model>::value) {
      return model_->is_hybrid_linear_attention();
    } else {
      return CausalLM::is_hybrid_linear_attention();
    }
  }

  std::optional<bool> loaded_vocab_weights() override {
    if constexpr (detail::has_loaded_vocab_weights<Model>::value &&
                  detail::has_loaded_vocab_weights_capability<Model>::value) {
      if (model_->reports_loaded_vocab_weights()) {
        return model_->has_loaded_vocab_weights();
      }
    }
    return CausalLM::loaded_vocab_weights();
  }

  bool supports_mla_graph_kv_bucketing() const override {
    if constexpr (detail::has_supports_mla_graph_kv_bucketing<Model>::value) {
      return model_->supports_mla_graph_kv_bucketing();
    }
    return CausalLM::supports_mla_graph_kv_bucketing();
  }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      override {
    if constexpr (detail::has_create_graph_forward_metadata_state<
                      Model>::value) {
      return model_->create_graph_forward_metadata_state();
    } else {
      return CausalLM::create_graph_forward_metadata_state();
    }
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& parameters) override {
    if constexpr (detail::has_prepare_graph_forward_metadata<Model>::value) {
      model_->prepare_graph_forward_metadata(state, positions, parameters);
    }
  }

  void prepare_graph_forward_inputs(
      ModelGraphMetadataState* state,
      const torch::Tensor& tokens,
      const ModelInputParams& parameters) override {
    if constexpr (detail::has_prepare_graph_forward_inputs<Model>::value) {
      model_->prepare_graph_forward_inputs(state, tokens, parameters);
    }
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) override {
    if constexpr (detail::has_logits_with_hidden<Model>::value) {
      return model_->logits(hidden_states, seleted_idxes, out_hidden);
    } else {
      // No fused lm_head variant: surface the selected hidden via an explicit
      // gather and project only the selected rows through the 2-arg path,
      // mirroring the CausalVLMImpl fallback so out_hidden stays defined.
      if (seleted_idxes.defined()) {
        torch::Tensor idxes = seleted_idxes.to(
            torch::dtype(torch::kLong).device(hidden_states.device()));
        out_hidden = hidden_states.index_select(/*dim=*/0, idxes);
      } else {
        out_hidden = hidden_states;
      }
      return model_->logits(hidden_states, seleted_idxes);
    }
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    load_d2t(loader->get_state_dicts());
    model_->load_model(std::move(loader));
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) override {
    if constexpr (detail::has_write_context_kv<Model>::value) {
      return model_->write_context_kv(target_hidden,
                                      positions,
                                      device_cache_slots,
                                      kv_caches,
                                      input_params);
    } else {
      return CausalLM::write_context_kv(target_hidden,
                                        positions,
                                        device_cache_slots,
                                        kv_caches,
                                        input_params);
    }
  }

  DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) override {
    if constexpr (detail::has_dflash2_candidates<Model>::value) {
      return model_->dflash2_candidates(
          hidden_states, unary_logits, anchor_token_ids);
    }
    return CausalLM::dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) override {
    if constexpr (detail::has_dspark_markov_bias<Model>::value) {
      return model_->dspark_markov_bias(previous_token_ids);
    }
    return CausalLM::dspark_markov_bias(previous_token_ids);
  }

  void lazy_load_model(std::unique_ptr<ModelLoader> loader) override {
    if constexpr (detail::has_lazy_load_model<Model>::value) {
      load_d2t(loader->get_state_dicts());
      model_->lazy_load_model(std::move(loader));
    } else {
      CausalLM::lazy_load_model(std::move(loader));
    }
  }

  void free_model_weights() override {
    if constexpr (detail::has_free_model_weights<Model>::value) {
      model_->free_model_weights();
    } else {
      CausalLM::free_model_weights();
    }
  }

  void reload_model_weights() override {
    if constexpr (detail::has_reload_model_weights<Model>::value) {
      model_->reload_model_weights();
    } else {
      CausalLM::reload_model_weights();
    }
  }

  void reload_model_weights_from_device() override {
    if constexpr (detail::has_reload_model_weights_from_device<Model>::value) {
      model_->reload_model_weights_from_device();
    } else {
      CausalLM::reload_model_weights_from_device();
    }
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    return model_->prepare_expert_weight(layer_id, expert_ids);
  }

  void start_expert_weight_transfer(int32_t layer_id) override {
    if constexpr (detail::has_start_expert_weight_transfer<Model>::value) {
      model_->start_expert_weight_transfer(layer_id);
    }
  }

  void update_expert_weight(int32_t layer_id) override {
    return model_->update_expert_weight(layer_id);
  }

  bool last_prepare_expert_weight_ok(int32_t layer_id) const override {
    if constexpr (detail::has_last_prepare_expert_weight_ok<Model>::value) {
      return model_->last_prepare_expert_weight_ok(layer_id);
    } else {
      return CausalLM::last_prepare_expert_weight_ok(layer_id);
    }
  }

#if defined(USE_NPU)
  bool has_npu_lm_head() const override {
    return detail::has_get_npu_lm_head<Model>::value;
  }

  layer::NpuLmHead get_npu_lm_head() override {
    if constexpr (detail::has_get_npu_lm_head<Model>::value) {
      return model_->get_npu_lm_head();
    } else {
      return CausalLM::get_npu_lm_head();
    }
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    if constexpr (detail::has_get_npu_word_embedding<Model>::value) {
      return model_->get_npu_word_embedding();
    } else {
      return CausalLM::get_npu_word_embedding();
    }
  }

  bool has_restored_npu_word_embedding() override {
    if constexpr (detail::has_restored_npu_word_embedding<Model>::value) {
      return model_->has_restored_npu_word_embedding();
    } else {
      return CausalLM::has_restored_npu_word_embedding();
    }
  }

  bool init_or_refresh_rolling_runtime(Stream* load_stream,
                                       Stream* compute_stream,
                                       int32_t num_cached_slots,
                                       int32_t requested_rolling_slots,
                                       const std::string& model_id) override {
    if constexpr (detail::has_init_or_refresh_rolling_runtime<Model>::value) {
      return model_->init_or_refresh_rolling_runtime(load_stream,
                                                     compute_stream,
                                                     num_cached_slots,
                                                     requested_rolling_slots,
                                                     model_id);
    }
    return CausalLM::init_or_refresh_rolling_runtime(load_stream,
                                                     compute_stream,
                                                     num_cached_slots,
                                                     requested_rolling_slots,
                                                     model_id);
  }
#endif

  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<Model>::value) {
      return model_->get_lm_head();
    } else {
      return CausalLM::get_lm_head();
    }
  }

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (detail::has_get_word_embedding<Model>::value) {
      return model_->get_word_embedding();
    } else {
      return CausalLM::get_word_embedding();
    }
  }

  void share_lm_head(CausalLM& target_model) override {
#if defined(USE_NPU)
    if constexpr (detail::has_set_npu_lm_head<Model>::value) {
      if (target_model.has_npu_lm_head() &&
          (!detail::has_get_npu_lm_head<Model>::value ||
           KernelConfig::get_instance().npu_kernel_backend() != "TORCH")) {
        auto head = target_model.get_npu_lm_head();
        model_->set_npu_lm_head(head);
        return;
      }
    }
#endif
    if constexpr (detail::has_set_lm_head<Model>::value) {
      auto head = target_model.get_lm_head();
      model_->set_lm_head(head);
    }
  }

  void share_embed_tokens(CausalLM& target_model) override {
#if defined(USE_NPU)
    if constexpr (detail::has_set_npu_word_embedding<Model>::value) {
      if (KernelConfig::get_instance().npu_kernel_backend() != "TORCH") {
        auto embedding = target_model.get_npu_word_embedding();
        if constexpr (detail::has_set_restored_npu_word_embedding<
                          Model>::value) {
          if (target_model.has_restored_npu_word_embedding()) {
            model_->set_restored_npu_word_embedding(embedding);
            return;
          }
        }
        model_->set_npu_word_embedding(embedding);
        return;
      }
    }
#endif
    if constexpr (detail::has_set_word_embedding<Model>::value) {
      auto embedding = target_model.get_word_embedding();
      model_->set_word_embedding(embedding);
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;

  torch::TensorOptions options_;
};

}  // namespace xllm

#endif  // XLLM_CORE_FRAMEWORK_MODEL_CAUSAL_LM_H_
