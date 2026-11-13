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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/kernels/musa/mhc_ops.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/musa/glm5_next/glm5_next_decoder_layer.h"
#include "core/layers/musa/glm5_next/glm5_next_graph_policy.h"
#include "core/util/env_var.h"
#include "models/llm/llm_model_base.h"
#include "models/model_registry.h"

namespace xllm::musa::model {

struct Glm5NextGraphMetadataState final : ModelGraphMetadataState {
  std::shared_ptr<layer::AttentionMetadata> metadata;
};

class Glm5NextModelImpl final
    : public LlmModelImplBase<layer::Glm5NextDecoderLayer> {
 public:
  explicit Glm5NextModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Glm5NextDecoderLayer>("glm5_next",
                                                      context.get_model_args()),
        device_(context.get_tensor_options().device()),
        hc_mult_(context.get_model_args().hc_mult()),
        cp_group_(context.get_parallel_args().cp_group_),
        cp_size_(context.get_parallel_args().cp_size()),
        ep_size_(context.get_parallel_args().ep_size()),
        prefill_cp_enabled_(util::get_bool_env("XLLM_GLM5_PREFILL_CP", false)),
        prefill_ep_enabled_(util::get_bool_env("XLLM_GLM5_PREFILL_EP", false)) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    CHECK_GT(hc_mult_, 0);
    if (prefill_cp_enabled_) {
      CHECK_EQ(cp_size_, 8)
          << "GLM5 prefill CP0 requires cp_size=8 in the baseline contract.";
      CHECK_EQ(parallel_args.dp_size(), 1)
          << "GLM5 prefill CP0 requires dp_size=1 in the baseline contract.";
      CHECK_EQ(parallel_args.world_size(), 8)
          << "GLM5 prefill CP0 requires an 8-rank world in the baseline "
             "contract.";
      CHECK(cp_group_ != nullptr)
          << "GLM5 prefill CP0 requires the native cp_group.";
      CHECK_EQ(cp_group_->world_size(), 8)
          << "GLM5 prefill CP0 requires an 8-rank cp_group.";
      LOG(INFO) << "GLM5 prefill CP0 contract enabled: cp_rank="
                << cp_group_->rank() << "/" << cp_group_->world_size();
    }
    if (prefill_ep_enabled_) {
      CHECK_EQ(ep_size_, 8)
          << "GLM5 prefill EP0 requires ep_size=8 in the baseline contract.";
      CHECK_EQ(parallel_args.dp_size(), 1)
          << "GLM5 prefill EP0 requires dp_size=1 in the baseline contract.";
      CHECK_EQ(parallel_args.world_size(), 8)
          << "GLM5 prefill EP0 requires an 8-rank world in the baseline "
             "contract.";
      CHECK(parallel_args.moe_ep_group_ != nullptr)
          << "GLM5 prefill EP0 requires the native moe_ep_group.";
      CHECK_EQ(parallel_args.moe_ep_group_->world_size(), 8)
          << "GLM5 prefill EP0 requires an 8-rank moe_ep_group.";
      LOG(INFO) << "GLM5 prefill EP0 contract enabled: ep_rank="
                << parallel_args.moe_ep_group_->rank() << "/"
                << parallel_args.moe_ep_group_->world_size();
    }
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
    if (prefill_cp_enabled_) {
      CHECK(input_params.meta.batch_forward_type.is_prefill() ||
            input_params.meta.batch_forward_type.is_chunked_prefill())
          << "Hybrid CP currently supports eager Prefill only";
      CHECK(!input_params.is_spec_verify);
      CHECK_GT(tokens.numel(), 0) << "CP does not synthesize dummy global rows";
      CHECK(!kernel::musa::is_stream_capturing());
    }
    if (tokens.numel() == 0) {
      tokens = torch::ones(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::zeros(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }
    const int64_t global_rows =
        input_params.embedding.input_embedding.defined()
            ? input_params.embedding.input_embedding.size(0)
            : tokens.numel();
    CHECK_EQ(positions.numel(), global_rows);

    ModelInputParams modified_input_params = input_params;
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    prepare_attention_metadata(modified_input_params);
    if (prefill_cp_enabled_ &&
        modified_input_params.meta.batch_forward_type.no_decode() &&
        !modified_input_params.is_spec_verify) {
      CHECK(!modified_input_params.attn_metadata->enable_cuda_graph)
          << "GLM5 prefill CP0 metadata is eager-only until the CP graph "
             "contract is implemented.";
      prepare_prefill_cp_metadata(modified_input_params, global_rows);
    }
    if (!modified_input_params.attn_metadata->enable_cuda_graph ||
        !modified_input_params.attn_metadata->musa_kda_batch) {
      // Own a fresh batch descriptor even when the scheduler reuses metadata.
      if (!modified_input_params.attn_metadata->enable_cuda_graph) {
        modified_input_params.attn_metadata =
            std::make_shared<layer::AttentionMetadata>(
                *modified_input_params.attn_metadata);
      }
      auto& batch_metadata = *modified_input_params.attn_metadata;
      batch_metadata.musa_kda_batch.reset();
      int64_t state_capacity = std::numeric_limits<int64_t>::max();
      for (auto& cache : kv_caches) {
        const torch::Tensor conv = cache.get_conv_cache();
        if (conv.defined()) {
          state_capacity = std::min(state_capacity, conv.size(0));
        }
      }
      if (state_capacity != std::numeric_limits<int64_t>::max()) {
        const auto& embedding = modified_input_params.embedding;
        const torch::Tensor indices =
            embedding.linear_state_indices.defined()
                ? embedding.linear_state_indices.to(device_, torch::kInt32)
                : torch::tensor(embedding.linear_state_ids,
                                torch::TensorOptions()
                                    .dtype(torch::kInt32)
                                    .device(device_));
        batch_metadata.musa_kda_batch = layer::prepare_musa_kda_batch(
            indices, batch_metadata.q_cu_seq_lens, global_rows, state_capacity);
      }
      kda_state_capacity_ = state_capacity;
    }
    const layer::AttentionMetadata& attn_metadata =
        *modified_input_params.attn_metadata;

    if (prefill_cp_enabled_) {
      cp_global_rows_ = attn_metadata.glm5_prefill_global_rows;
      cp_local_row_begin_ = attn_metadata.glm5_prefill_local_row_begin;
      cp_local_row_end_ = attn_metadata.glm5_prefill_local_row_end;
      cp_row_offsets_ = attn_metadata.glm5_prefill_row_offsets;
    }

    torch::Tensor input_embedding = input_params.embedding.input_embedding;
    if (prefill_cp_enabled_) {
      const int64_t begin = attn_metadata.glm5_prefill_local_row_begin;
      const int64_t count = attn_metadata.glm5_prefill_local_row_end - begin;
      positions = positions.narrow(0, begin, count).contiguous();
      if (input_embedding.defined()) {
        input_embedding = input_embedding.narrow(0, begin, count).contiguous();
      } else {
        tokens = tokens.narrow(0, begin, count).contiguous();
      }
    }
    torch::Tensor hidden_states =
        input_embedding.defined()
            ? input_embedding
            : (tokens.numel() == 0
                   ? torch::empty({0, model_args_.hidden_size()},
                                  norm_->weight().options())
                   : embed_tokens_(tokens));
    hidden_states = kernel::musa::mhc_expand(hidden_states, hc_mult_);
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
    // CP4 restores/selects request outputs in the CausalLM LM Head path; the
    // model itself continues to return Plain local rows to the executor.
    return ModelOutput(hidden_states.numel() == 0
                           ? hidden_states
                           : std::get<0>(norm_->forward(hidden_states)));
  }

  // CP4 restores the Plain local rows after the final local norm.  The
  // all-gather uses a fixed padded row count because CP0 permits S % CP != 0
  // and empty ranks.  Copying is performed on device; row offsets are the
  // metadata contract prepared by the model-level split.
  torch::Tensor restore_prefill_cp_hidden(
      const torch::Tensor& local_hidden_states) const {
    CHECK(prefill_cp_enabled_);
    CHECK(cp_group_ != nullptr);
    CHECK_EQ(cp_row_offsets_.size(),
             static_cast<size_t>(cp_group_->world_size()) + 1);
    CHECK_EQ(local_hidden_states.dim(), 2);
    CHECK_EQ(local_hidden_states.size(0),
             cp_local_row_end_ - cp_local_row_begin_);
    CHECK_EQ(cp_global_rows_, cp_row_offsets_.back());
    int64_t max_rows = 0;
    for (size_t rank = 0; rank + 1 < cp_row_offsets_.size(); ++rank) {
      max_rows = std::max(max_rows,
                          static_cast<int64_t>(cp_row_offsets_[rank + 1] -
                                               cp_row_offsets_[rank]));
    }
    torch::Tensor padded = torch::zeros({max_rows, local_hidden_states.size(1)},
                                        local_hidden_states.options());
    if (local_hidden_states.size(0) > 0) {
      padded.narrow(0, 0, local_hidden_states.size(0))
          .copy_(local_hidden_states);
    }
    torch::Tensor gathered = cp_group_->allgather_base_sync(padded);
    CHECK_EQ(
        gathered.sizes(),
        torch::IntArrayRef(
            {cp_group_->world_size(), max_rows, local_hidden_states.size(1)}));
    torch::Tensor restored =
        torch::empty({cp_global_rows_, local_hidden_states.size(1)},
                     local_hidden_states.options());
    for (int32_t rank = 0; rank < cp_group_->world_size(); ++rank) {
      const int64_t begin = cp_row_offsets_[static_cast<size_t>(rank)];
      const int64_t count =
          cp_row_offsets_[static_cast<size_t>(rank) + 1] - begin;
      if (count > 0) {
        restored.narrow(0, begin, count)
            .copy_(gathered[rank].narrow(0, 0, count));
      }
    }
    return restored;
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    auto* graph_state = dynamic_cast<Glm5NextGraphMetadataState*>(state);
    CHECK(graph_state != nullptr);
    CHECK(!params.is_spec_verify);
    const bool is_decode = params.meta.batch_forward_type.is_decode();
    const bool is_prefill = params.meta.batch_forward_type.is_prefill() ||
                            params.meta.batch_forward_type.is_chunked_prefill();
    CHECK(is_decode || is_prefill);
    const auto& attention = params.attention.device;
    if (!graph_state->metadata) {
      graph_state->metadata =
          std::make_shared<layer::AttentionMetadata>(*params.attn_metadata);
      auto& metadata = *graph_state->metadata;
      metadata.enable_cuda_graph = true;
      metadata.is_prefill =
          is_prefill && !params.meta.batch_forward_type.is_chunked_prefill();
      metadata.is_chunked_prefill =
          params.meta.batch_forward_type.is_chunked_prefill();
      metadata.max_query_len = is_decode ? 1 : params.meta.q_max_seq_len;
      metadata.max_seq_len = is_decode
                                 ? layer::glm5_next_graph_context_capacity(
                                       params.meta.kv_max_seq_len)
                                 : params.meta.kv_max_seq_len;
      metadata.q_cu_seq_lens = attention.q_seq_lens;
      metadata.kv_cu_seq_lens = attention.kv_seq_lens;
      metadata.block_table = attention.block_tables;
      metadata.slot_mapping = attention.new_cache_slots;
      if (params.embedding.linear_state_indices.defined()) {
        metadata.linear_state_indices = params.embedding.linear_state_indices;
      }
      metadata.linear_state_read_indices = torch::Tensor();
      if (params.linear_state_validity_mask_tensor.defined()) {
        metadata.has_initial_states = params.linear_state_validity_mask_tensor;
      }
      if (is_decode) {
        CHECK_GT(kda_state_capacity_, 0);
        metadata.musa_kda_batch =
            layer::prepare_musa_kda_batch(params.embedding.linear_state_indices,
                                          attention.q_seq_lens,
                                          positions.numel(),
                                          kda_state_capacity_);
      } else {
        // This performs the host-side schedule construction before stream
        // capture.  Rebuilding it from inside the captured forward would
        // execute item()/repeat_interleave() on the host and invalidate the
        // graph's recurrent metadata.
        layer::AttentionMetadataBuilder::build_linear_prefill(
            metadata,
            /*block_size=*/64);
      }
    } else {
      auto& metadata = *graph_state->metadata;
      metadata.q_cu_seq_lens = attention.q_seq_lens;
      metadata.kv_cu_seq_lens = attention.kv_seq_lens;
      metadata.block_table = attention.block_tables;
      metadata.slot_mapping = attention.new_cache_slots;
      if (params.linear_state_validity_mask_tensor.defined() &&
          metadata.has_initial_states.defined()) {
        metadata.has_initial_states.copy_(
            params.linear_state_validity_mask_tensor);
      }
      if (metadata.musa_kda_batch) {
        CHECK_GT(kda_state_capacity_, 0);
        const auto batch =
            layer::prepare_musa_kda_batch(params.embedding.linear_state_indices,
                                          attention.q_seq_lens,
                                          positions.numel(),
                                          kda_state_capacity_);
        const auto& captured = metadata.musa_kda_batch;
        CHECK_EQ(captured->num_tokens, batch->num_tokens);
        CHECK_EQ(captured->state_capacity, batch->state_capacity);
        CHECK_EQ(captured->state_indices.data_ptr(),
                 batch->state_indices.data_ptr());
        captured->state_indices_long.copy_(batch->state_indices_long);
      }
    }
    params.attn_metadata = graph_state->metadata;
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
  void prepare_prefill_cp_metadata(ModelInputParams& params,
                                   int64_t global_rows) const {
    CHECK(cp_group_ != nullptr);
    const int32_t cp_rank = cp_group_->rank();
    const int32_t cp_size = cp_group_->world_size();
    CHECK_EQ(cp_size, cp_size_);
    CHECK_GT(cp_size, 1);
    CHECK_GT(global_rows, 0);
    CHECK_LE(global_rows, std::numeric_limits<int32_t>::max());

    const int64_t rows_per_rank = global_rows / cp_size;
    const int64_t remainder = global_rows % cp_size;
    const int64_t local_row_begin =
        cp_rank * rows_per_rank + std::min<int64_t>(cp_rank, remainder);
    const int64_t local_row_count =
        rows_per_rank + (cp_rank < remainder ? static_cast<int64_t>(1) : 0);
    auto& metadata = *params.attn_metadata;
    metadata.glm5_prefill_cp_enabled = true;
    metadata.glm5_prefill_cp_rank = cp_rank;
    metadata.glm5_prefill_cp_size = cp_size;
    metadata.glm5_prefill_global_rows = global_rows;
    metadata.glm5_prefill_local_row_begin = local_row_begin;
    metadata.glm5_prefill_local_row_end = local_row_begin + local_row_count;
    metadata.glm5_prefill_local_row_indices = torch::arange(
        local_row_begin,
        local_row_begin + local_row_count,
        torch::TensorOptions().dtype(torch::kInt32).device(device_));
    metadata.glm5_prefill_row_offsets.resize(static_cast<size_t>(cp_size) + 1);
    for (int32_t rank = 0; rank <= cp_size; ++rank) {
      const int64_t offset =
          rank * rows_per_rank + std::min<int64_t>(rank, remainder);
      metadata.glm5_prefill_row_offsets[static_cast<size_t>(rank)] =
          static_cast<int32_t>(offset);
    }
  }

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
    } else if ((!params.attn_metadata->enable_cuda_graph &&
                (ordinary_prefill ||
                 (!params.attn_metadata->is_spec_verify &&
                  (params.attn_metadata->is_prefill ||
                   params.attn_metadata->is_chunked_prefill)))) ||
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
    // MUSA KDA consumes musa_kda_batch prepared from the global sequence
    // boundaries in forward(). The generic linear-attention chunk schedule is
    // unused here and would introduce redundant device-to-host scalar reads.
  }

  torch::Device device_;
  int64_t hc_mult_ = 4;
  int64_t kda_state_capacity_ = 0;
  ProcessGroup* cp_group_ = nullptr;
  int32_t cp_size_ = 1;
  int32_t ep_size_ = 1;
  bool prefill_cp_enabled_ = false;
  bool prefill_ep_enabled_ = false;
  int64_t cp_global_rows_ = 0;
  int64_t cp_local_row_begin_ = 0;
  int64_t cp_local_row_end_ = 0;
  std::vector<int32_t> cp_row_offsets_;
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(Glm5NextModel);

class Glm5NextForCausalLMImpl final
    : public LlmForCausalLMImplBase<Glm5NextModel> {
 public:
  explicit Glm5NextForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm5NextModel>(context),
        prefill_cp_enabled_(util::get_bool_env("XLLM_GLM5_PREFILL_CP", false)) {
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices) override {
    if (!prefill_cp_enabled_) {
      return LlmForCausalLMImplBase<Glm5NextModel>::logits(hidden_states,
                                                           selected_indices);
    }
    torch::Tensor restored = model_->restore_prefill_cp_hidden(hidden_states);
    return LlmForCausalLMImplBase<Glm5NextModel>::logits(restored,
                                                         selected_indices);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices,
                       torch::Tensor& out_hidden) override {
    if (!prefill_cp_enabled_) {
      return LlmForCausalLMImplBase<Glm5NextModel>::logits(
          hidden_states, selected_indices, out_hidden);
    }
    torch::Tensor restored = model_->restore_prefill_cp_hidden(hidden_states);
    return LlmForCausalLMImplBase<Glm5NextModel>::logits(
        restored, selected_indices, out_hidden);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices) override {
    if (!prefill_cp_enabled_) {
      return LlmForCausalLMImplBase<Glm5NextModel>::pooler(hidden_states,
                                                           selected_indices);
    }
    torch::Tensor restored = model_->restore_prefill_cp_hidden(hidden_states);
    return LlmForCausalLMImplBase<Glm5NextModel>::pooler(restored,
                                                         selected_indices);
  }

  bool is_hybrid_linear_attention() { return true; }

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<Glm5NextGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    model_->prepare_graph_forward_metadata(state, positions, params);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    LlmForCausalLMImplBase<Glm5NextModel>::load_model(std::move(loader),
                                                      std::move(prefix));
    model_->verify_loaded_weights();
  }

 private:
  bool prefill_cp_enabled_ = false;
};
TORCH_MODULE(Glm5NextForCausalLM);

REGISTER_CAUSAL_MODEL(glm5_next, Glm5NextForCausalLM);

}  // namespace xllm::musa::model
