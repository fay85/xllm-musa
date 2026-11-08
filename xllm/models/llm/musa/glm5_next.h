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
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/kernels/musa/mhc_ops.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/musa/glm5_next/glm5_next_decoder_layer.h"
#include "models/llm/llm_model_base.h"
#include "models/model_registry.h"

namespace xllm::musa::model {

class Glm5NextModelImpl final
    : public LlmModelImplBase<layer::Glm5NextDecoderLayer> {
 public:
  explicit Glm5NextModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Glm5NextDecoderLayer>("glm5_next",
                                                      context.get_model_args()),
        device_(context.get_tensor_options().device()),
        hc_mult_(context.get_model_args().hc_mult()) {
    const ModelArgs& args = context.get_model_args();
    CHECK_GT(hc_mult_, 0);
    CHECK_EQ(args.n_layers(),
             static_cast<int32_t>(args.mlp_layer_types().size()) == 0
                 ? args.n_layers()
                 : static_cast<int32_t>(args.mlp_layer_types().size()));
    for (const std::string& mlp_type : args.mlp_layer_types()) {
      CHECK(mlp_type == "dense" || mlp_type == "sparse")
          << "MUSA GLM5-Next supports only dense or sparse MLP layers, got: "
          << mlp_type;
    }
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(static_cast<size_t>(args.n_layers()));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
    for (int32_t layer_id = 0; layer_id < args.n_layers(); ++layer_id) {
      layer::Glm5NextDecoderLayer decoder_layer(context, layer_id);
      layers_.emplace_back(decoder_layer);
      blocks_->push_back(decoder_layer);
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    CHECK_EQ(kv_caches.size(), layers_.size());
    if (tokens.numel() == 0) {
      tokens = torch::ones(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::zeros(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }
    torch::Tensor hidden_states =
        input_params.embedding.input_embedding.defined()
            ? input_params.embedding.input_embedding
            : embed_tokens_(tokens);
    hidden_states = kernel::musa::mhc_expand(hidden_states, hc_mult_);

    ModelInputParams modified_input_params = input_params;
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    prepare_attention_metadata(modified_input_params);
    const layer::AttentionMetadata& attn_metadata =
        *modified_input_params.attn_metadata;

    std::optional<torch::Tensor> residual;
    std::optional<layer::PendingMHC> pending_mhc;
    std::optional<kernel::musa::Glm5NextIndexerSelection> shared_selection;
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      if (!modified_input_params.synchronize_layer(
              static_cast<uint32_t>(layer_id))) {
        return ModelOutput();
      }
      hidden_states = layers_[layer_id]->forward(
          hidden_states,
          residual,
          positions,
          attn_metadata,
          kv_caches[layer_id],
          modified_input_params,
          &pending_mhc,
          /*is_last_layer=*/layer_id + 1 == layers_.size(),
          shared_selection.has_value() ? &shared_selection.value()
                                       : &shared_selection.emplace());
      if (!modified_input_params.record_layer(static_cast<uint32_t>(layer_id),
                                              hidden_states.device())) {
        return ModelOutput();
      }
    }
    CHECK(!pending_mhc.has_value());
    hidden_states = kernel::musa::mhc_contract_mean(hidden_states);
    return ModelOutput(std::get<0>(norm_->forward(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layers_[layer_id]->load_state_dict(state_dict.get_dict_with_prefix(
          "layers." + std::to_string(layer_id) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights() const {
    for (const auto& layer : layers_) {
      layer->verify_loaded_weights();
    }
  }

 private:
  void prepare_attention_metadata(ModelInputParams& params) const {
    const bool ordinary_prefill =
        params.meta.batch_forward_type.no_decode() && !params.is_spec_verify;
    if (!params.attn_metadata ||
        !params.attn_metadata->q_cu_seq_lens.defined()) {
      params.attn_metadata = std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params,
                                                 model_args_.enable_mla(),
                                                 /*compute_dtype=*/"half",
                                                 /*attn_mask=*/std::nullopt,
                                                 /*device=*/device_));
    } else if (ordinary_prefill ||
               (!params.attn_metadata->is_spec_verify &&
                (params.attn_metadata->is_prefill ||
                 params.attn_metadata->is_chunked_prefill)) ||
               params.attn_metadata->linear_state_read_indices.defined()) {
      params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(*params.attn_metadata);
    }
    auto& metadata = *params.attn_metadata;
    if (metadata.linear_state_read_indices.defined()) {
      metadata.linear_state_read_indices = torch::Tensor();
    }
    params.embedding.linear_state_read_ids = params.embedding.linear_state_ids;
    params.embedding.linear_state_read_indices =
        params.embedding.linear_state_indices;
    if (metadata.is_prefill || metadata.is_chunked_prefill) {
      layer::AttentionMetadataBuilder::build_linear_prefill(metadata,
                                                            /*block_size=*/64);
    }
  }

  torch::Device device_;
  int64_t hc_mult_ = 4;
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(Glm5NextModel);

class Glm5NextForCausalLMImpl final
    : public LlmForCausalLMImplBase<Glm5NextModel> {
 public:
  explicit Glm5NextForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm5NextModel>(context) {}

  bool is_hybrid_linear_attention() { return true; }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    LlmForCausalLMImplBase<Glm5NextModel>::load_model(std::move(loader),
                                                      std::move(prefix));
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(Glm5NextForCausalLM);

REGISTER_CAUSAL_MODEL(glm5_next, Glm5NextForCausalLM);

}  // namespace xllm::musa::model
