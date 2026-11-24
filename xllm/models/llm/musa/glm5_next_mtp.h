/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/layers/musa/glm5_next/glm5_next_decoder_layer.h"
#include "core/layers/musa/glm5_next/glm5_next_graph_policy.h"
#include "core/layers/musa/glm5_next/glm5_next_mla_attention.h"
#include "models/llm/glm5_next.h"
#include "models/llm/mtp_model_base.h"
#include "models/model_registry.h"

namespace xllm::musa::model {

struct Glm5NextMtpGraphMetadataState final : ModelGraphMetadataState {
  std::shared_ptr<layer::AttentionMetadata> metadata;
  torch::Tensor slot_mapping;
};

// GLM-5.3 stores its MTP predictor as an appended DSA/MoE decoder layer. The
// predictor receives the target hidden state through the MTP projection before
// entering the ordinary GLM decoder implementation.
class Glm5NextMtpDecoderLayerImpl final
    : public ::xllm::MtpDecoderLayerImplBase<layer::Glm5NextDecoderLayer> {
 public:
  explicit Glm5NextMtpDecoderLayerImpl(const ModelContext& context,
                                       int32_t layer_index)
      : MtpDecoderLayerImplBase<layer::Glm5NextDecoderLayer>(context,
                                                             layer_index) {}
};
TORCH_MODULE(Glm5NextMtpDecoderLayer);

class Glm5NextMtpLayerForwardAdapter final {
 public:
  Glm5NextMtpLayerForwardAdapter(const MtpTopkStatePtr& input_state,
                                 size_t /*num_layers*/,
                                 const torch::Device& /*device*/) {
    CHECK(input_state == nullptr)
        << "GLM5 MTP does not support a chained top-k state";
  }

  torch::Tensor forward_layer(Glm5NextMtpDecoderLayer& decoder_layer,
                              size_t /*layer_id*/,
                              torch::Tensor& hidden_states,
                              std::optional<torch::Tensor>& residual,
                              torch::Tensor& positions,
                              const layer::AttentionMetadata& attn_metadata,
                              KVCache& kv_cache,
                              ModelInputParams& input_params,
                              const std::optional<torch::Tensor>& input_ids) {
    using MtpLayerBase =
        ::xllm::MtpDecoderLayerImplBase<layer::Glm5NextDecoderLayer>;
    auto decoder_forward = [](layer::Glm5NextDecoderLayer& decoder,
                              torch::Tensor& layer_hidden_states,
                              std::optional<torch::Tensor>& layer_residual,
                              torch::Tensor& layer_positions,
                              const layer::AttentionMetadata& metadata,
                              KVCache& layer_kv_cache,
                              ModelInputParams& params,
                              const std::optional<torch::Tensor>& /*unused*/) {
      return decoder->forward(layer_hidden_states,
                              layer_residual,
                              layer_positions,
                              metadata,
                              layer_kv_cache,
                              params,
                              /*pending_mhc=*/nullptr,
                              /*is_last_layer=*/true,
                              /*shared_selection=*/nullptr);
    };
    return static_cast<MtpLayerBase&>(*decoder_layer)
        .forward_with_custom_decoder(hidden_states,
                                     residual,
                                     positions,
                                     attn_metadata,
                                     kv_cache,
                                     input_params,
                                     input_ids,
                                     decoder_forward);
  }

  MtpTopkStatePtr take_output() { return nullptr; }
};

class Glm5NextMtpModelImpl final
    : public ::xllm::MtpModelImplBase<Glm5NextMtpDecoderLayer,
                                      Glm5NextMtpLayerForwardAdapter> {
 public:
  explicit Glm5NextMtpModelImpl(const ModelContext& context)
      : MtpModelImplBase<Glm5NextMtpDecoderLayer,
                         Glm5NextMtpLayerForwardAdapter>(context),
        cp_group_(context.get_parallel_args().cp_group_) {
    CHECK_EQ(context.get_model_args().num_nextn_predict_layers(), 1)
        << "MUSA GLM5 MTP currently supports one predictor layer";
  }
  bool use_cp(const ModelInputParams& params, int64_t rows) const {
    return cp_group_ && cp_group_->world_size() > 1 &&
           rows >= cp_group_->world_size() &&
           params.meta.batch_forward_type.no_decode() &&
           params.attention.host.kpool_query_lens.empty();
  }

  void prepare_cp_layout(layer::AttentionMetadata& metadata,
                         const torch::Tensor& positions) {
    if (!metadata.glm5_prefill_cp_layout) {
      layer::prepare_glm5_next_cp_metadata(
          metadata, *cp_group_, positions.numel(), positions.device());
      metadata.glm5_prefill_cp_layout =
          layer::build_glm5_next_cp_layout(positions, metadata, *cp_group_);
    } else {
      layer::refresh_glm5_next_cp_layout(
          metadata.glm5_prefill_cp_layout, positions, metadata.slot_mapping);
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) {
    using Base = MtpModelImplBase<Glm5NextMtpDecoderLayer,
                                  Glm5NextMtpLayerForwardAdapter>;
    if (!use_cp(params, tokens.numel())) {
      return Base::forward(tokens, positions, caches, params);
    }
    ModelInputParams local = params;
    if (!local.attn_metadata || !local.attn_metadata->glm5_prefill_cp_layout) {
      local.attn_metadata = std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(local,
                                                 /*enable_mla=*/true,
                                                 /*attn_mask=*/std::nullopt,
                                                 positions.device()));
      prepare_cp_layout(*local.attn_metadata, positions);
    }
    const auto& metadata = *local.attn_metadata;
    const int64_t begin = metadata.glm5_prefill_local_row_begin;
    const int64_t end = metadata.glm5_prefill_local_row_end;
    local.embedding.input_embedding =
        params.embedding.input_embedding.slice(0, begin, end);
    if (params.embedding.mtp_token_embedding.defined()) {
      local.embedding.mtp_token_embedding =
          params.embedding.mtp_token_embedding.slice(0, begin, end);
    }
    auto output = Base::forward(tokens.slice(0, begin, end),
                                positions.slice(0, begin, end),
                                caches,
                                local);
    std::vector<int32_t> counts;
    counts.reserve(cp_group_->world_size());
    const auto& offsets = metadata.glm5_prefill_row_offsets;
    for (int32_t rank = 0; rank < cp_group_->world_size(); ++rank) {
      counts.emplace_back(offsets[rank + 1] - offsets[rank]);
    }
    output.hidden_states =
        parallel_state::gather(output.hidden_states, cp_group_, counts);
    return output;
  }

 private:
  ProcessGroup* cp_group_;
};
TORCH_MODULE(Glm5NextMtpModel);

class Glm5NextMtpForCausalLMImpl final
    : public LlmForCausalLMImplBase<Glm5NextMtpModel> {
 public:
  explicit Glm5NextMtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm5NextMtpModel>(context) {}

  bool reports_loaded_vocab_weights() const { return true; }

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<Glm5NextMtpGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    auto& graph = *static_cast<Glm5NextMtpGraphMetadataState*>(state);
    if (!graph.metadata) {
      graph.metadata =
          std::make_shared<layer::AttentionMetadata>(*params.attn_metadata);
      graph.slot_mapping = torch::empty_like(
          params.attention.device.new_cache_slots, torch::kLong);
    }
    const auto& attention = params.attention.device;
    auto& metadata = *graph.metadata;
    metadata.enable_cuda_graph = true;
    metadata.is_prefill = params.meta.batch_forward_type.is_prefill();
    metadata.is_chunked_prefill =
        params.meta.batch_forward_type.is_chunked_prefill();
    metadata.max_query_len = params.meta.q_max_seq_len;
    metadata.max_seq_len = params.meta.batch_forward_type.is_decode()
                               ? layer::glm5_next_graph_context_capacity(
                                     params.meta.kv_max_seq_len)
                               : params.meta.kv_max_seq_len;
    metadata.q_cu_seq_lens = attention.q_seq_lens;
    metadata.kv_cu_seq_lens = attention.kv_seq_lens;
    metadata.block_table = attention.block_tables;
    metadata.linear_state_indices = params.embedding.linear_state_indices;
    graph.slot_mapping.copy_(attention.new_cache_slots, true);
    metadata.slot_mapping = graph.slot_mapping;
    if (model_->use_cp(params, positions.numel())) {
      model_->prepare_cp_layout(metadata, positions);
    }
    params.attn_metadata = graph.metadata;
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      const StateDict model_state = state_dict->get_dict_with_prefix(
          std::vector<std::string>{"model.language_model.", prefix});
      model_->load_state_dict(model_state);
    }
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(Glm5NextMtpForCausalLM);

inline const bool glm5_next_mtp_capabilities_registered = [] {
  MtpModelCapabilities capabilities;
  capabilities.supports_accepted_span_replay = true;
  capabilities.supports_causal_replay_target = true;
  capabilities.replay_family = MtpReplayFamily::GLM5;
  ModelRegistry::register_mtp_capabilities("glm5_next", capabilities);
  ModelRegistry::register_mtp_capabilities("glm5_next_mtp", capabilities);
  return true;
}();

REGISTER_CAUSAL_MODEL(glm5_next_mtp, Glm5NextMtpForCausalLM);

}  // namespace xllm::musa::model
