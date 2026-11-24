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

#include <ATen/record_function.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/speculative_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/musa/mhc_ops.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/musa/glm5_next/glm5_next_decoder_layer.h"
#include "core/layers/musa/glm5_next/glm5_next_graph_policy.h"
#include "core/layers/musa/glm5_next/glm5_next_mla_attention.h"
#include "core/util/env_var.h"
#include "models/llm/llm_model_base.h"
#include "models/model_registry.h"

namespace xllm::musa::model {

struct Glm5NextGraphMetadataState final : ModelGraphMetadataState {
  std::shared_ptr<layer::AttentionMetadata> metadata;
  torch::Tensor slot_mapping;
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
      CHECK_GE(ep_size_, 1) << "GLM5 prefill EP0 requires ep_size >= 1.";
      CHECK_EQ(parallel_args.dp_size(), 1)
          << "GLM5 prefill EP0 requires dp_size=1 in the baseline contract.";
      CHECK(parallel_args.moe_tp_group_ == nullptr ||
            parallel_args.moe_tp_group_->world_size() == 1)
          << "GLM5 prefill EP0 requires moe_tp=1.";
      if (ep_size_ > 1) {
        CHECK(parallel_args.moe_ep_group_ != nullptr)
            << "GLM5 prefill EP0 requires the native moe_ep_group when "
               "ep_size > 1.";
        CHECK_EQ(parallel_args.moe_ep_group_->world_size(), ep_size_)
            << "GLM5 prefill EP0 requires moe_ep_group size == ep_size.";
      }
      const int32_t ep_rank = parallel_args.moe_ep_group_ == nullptr
                                  ? 0
                                  : parallel_args.moe_ep_group_->rank();
      LOG(INFO) << "GLM5 prefill EP0 contract enabled: ep_rank=" << ep_rank
                << "/" << ep_size_;
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
    RECORD_USER_SCOPE("nn.Module: Glm5NextModel_0");
    torch::NoGradGuard no_grad;
    CHECK_EQ(kv_caches.size(), layers_.size());
    const bool use_prefill_cp =
        prefill_cp_enabled_ && !input_params.is_spec_verify &&
        (input_params.meta.batch_forward_type.is_prefill() ||
         input_params.meta.batch_forward_type.is_chunked_prefill());
    last_forward_prefill_cp_ = use_prefill_cp;
    if (use_prefill_cp) {
      CHECK(input_params.meta.batch_forward_type.is_prefill() ||
            input_params.meta.batch_forward_type.is_chunked_prefill())
          << "Hybrid CP supports Prefill and Chunked Prefill only";
      CHECK(!input_params.is_spec_verify);
      CHECK_GT(tokens.numel(), 0) << "CP does not synthesize dummy global rows";
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
    if (use_prefill_cp &&
        modified_input_params.meta.batch_forward_type.no_decode() &&
        !modified_input_params.is_spec_verify) {
      layer::prepare_glm5_next_cp_metadata(*modified_input_params.attn_metadata,
                                           *cp_group_,
                                           global_rows,
                                           device_);
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
    }
    const layer::AttentionMetadata& attn_metadata =
        *modified_input_params.attn_metadata;

    if (use_prefill_cp) {
      cp_global_rows_ = attn_metadata.glm5_prefill_global_rows;
      cp_local_row_begin_ = attn_metadata.glm5_prefill_local_row_begin;
      cp_local_row_end_ = attn_metadata.glm5_prefill_local_row_end;
      cp_row_offsets_ = attn_metadata.glm5_prefill_row_offsets;
      // The model already owns the full packed positions before sharding the
      // activations below. Build the CP gather/restore plan from that global
      // tensor once per forward and reuse it in every DSA layer instead of
      // all-gathering positions and repeating D2H/H2D work per layer.
      if (!modified_input_params.attn_metadata->glm5_prefill_cp_layout) {
        RECORD_USER_SCOPE("nn.Module: Glm5NextModel.CPLayoutPlan");
        modified_input_params.attn_metadata->glm5_prefill_cp_layout =
            layer::build_glm5_next_cp_layout(
                positions, attn_metadata, *cp_group_);
      }
    }

    torch::Tensor input_embedding = input_params.embedding.input_embedding;
    if (use_prefill_cp) {
      // DSA, KDA and routed MoE all accept the plain CP shard.  Avoid the
      // round-robin transpose that previously crossed every sparse layer.
      const int64_t begin = attn_metadata.glm5_prefill_local_row_begin;
      const int64_t end = attn_metadata.glm5_prefill_local_row_end;
      positions = positions.slice(/*dim=*/0, begin, end).contiguous();
      if (input_embedding.defined()) {
        input_embedding =
            input_embedding.slice(/*dim=*/0, begin, end).contiguous();
      } else {
        tokens = tokens.slice(/*dim=*/0, begin, end).contiguous();
      }
      modified_input_params.attn_metadata->glm5_prefill_cp_round_robin = false;
      modified_input_params.attn_metadata->glm5_prefill_local_row_indices =
          torch::arange(begin, end, positions.options().dtype(torch::kInt32));
    }
    torch::Tensor hidden_states =
        input_embedding.defined()
            ? input_embedding
            : (tokens.numel() == 0
                   ? torch::empty({0, model_args_.hidden_size()},
                                  norm_->weight().options())
                   : ([&]() {
                       RECORD_USER_SCOPE("nn.Module: VocabParallelEmbedding_0");
                       return embed_tokens_(tokens);
                     })());
    hidden_states = kernel::musa::mhc_expand(hidden_states, hc_mult_);

    // Keep all decoder layers, including the pending mHC post consumed by the
    // next layer, on one concrete compute stream.  Creating this guard inside
    // every layer would return to the default stream at each boundary and
    // enqueue a redundant pair of event handoffs.
    kernel::musa::TvmffiComputeStreamGuard compute_stream_guard(
        hidden_states.device());
    std::optional<torch::Tensor> residual;
    std::optional<layer::PendingMHC> pending_mhc;
    std::optional<kernel::musa::Glm5NextIndexerSelection> shared_selection;
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      if (!modified_input_params.synchronize_layer(
              static_cast<uint32_t>(layer_id))) {
        return ModelOutput();
      }
      const std::string layer_name =
          "nn.Module: Glm5NextDecoderLayer_" + std::to_string(layer_id);
      RECORD_USER_SCOPE(layer_name.c_str());
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
    // Hidden states are already the plain local shard.  The executor restores
    // the global sequence from the recorded row offsets.
    // CP prefill path restores/selects request outputs in the CausalLM LM Head
    // path; the model itself continues to return Plain local rows to the
    // executor.
    if (hidden_states.numel() != 0) {
      RECORD_USER_SCOPE("nn.Module: RMSNorm_22");
      hidden_states = std::get<0>(norm_->forward(hidden_states));
    }
    if (use_prefill_cp &&
        SpeculativeConfig::get_instance().num_speculative_tokens() > 0) {
      hidden_states = restore_prefill_cp_hidden(hidden_states);
      last_forward_prefill_cp_ = false;
    }
    return ModelOutput(hidden_states);
  }

  bool last_forward_used_prefill_cp() const { return last_forward_prefill_cp_; }

  bool can_restore_prefill_cp_hidden(const torch::Tensor& hidden_states) const {
    return last_forward_prefill_cp_ && hidden_states.defined() &&
           hidden_states.dim() == 2 &&
           hidden_states.size(0) == cp_local_row_end_ - cp_local_row_begin_;
  }

  // CP prefill path restores the Plain local rows after the final local norm.
  // Unequal CP row counts use a fixed padded row count because CP0 permits
  // S % CP != 0 and empty ranks.  Equal row counts use a direct rank-major
  // gather.  Row offsets are the metadata contract prepared by the model-level
  // split.
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
    const int64_t local_rows = cp_local_row_end_ - cp_local_row_begin_;
    bool equal_rows = true;
    for (size_t rank = 0; rank < cp_row_offsets_.size() - 1; ++rank) {
      if (cp_row_offsets_[rank + 1] - cp_row_offsets_[rank] != local_rows) {
        equal_rows = false;
        break;
      }
    }
    if (equal_rows) {
      std::vector<int32_t> counts(cp_group_->world_size(),
                                  static_cast<int32_t>(local_rows));
      return parallel_state::gather(
          local_hidden_states.contiguous(), cp_group_, counts);
    }
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

  // Collect only rows requested by the LM head or pooler. Each rank writes
  // the rows it owns into a fixed-size buffer and the CP all-gather restores
  // the requested order. Duplicate and reordered selections are preserved.
  torch::Tensor select_prefill_cp_hidden(
      const torch::Tensor& local_hidden_states,
      const torch::Tensor& selected_indices) const {
    CHECK(prefill_cp_enabled_);
    CHECK(cp_group_ != nullptr);
    CHECK_EQ(local_hidden_states.dim(), 2);
    CHECK_EQ(local_hidden_states.size(0),
             cp_local_row_end_ - cp_local_row_begin_);
    if (!selected_indices.defined()) {
      return restore_prefill_cp_hidden(local_hidden_states);
    }
    CHECK_EQ(selected_indices.dim(), 1);
    if (selected_indices.numel() == 0) {
      return local_hidden_states.index_select(
          0,
          selected_indices.to(
              torch::dtype(torch::kLong).device(local_hidden_states.device())));
    }

    const torch::Tensor selected_cpu =
        selected_indices.to(torch::dtype(torch::kLong).device(torch::kCPU));
    std::vector<int64_t> local_rows;
    std::vector<int64_t> output_rows;
    local_rows.reserve(static_cast<size_t>(selected_indices.numel()));
    output_rows.reserve(static_cast<size_t>(selected_indices.numel()));
    const int32_t rank = cp_group_->rank();
    const int32_t world_size = cp_group_->world_size();
    for (int64_t output = 0; output < selected_cpu.numel(); ++output) {
      const int64_t global_row = selected_cpu[output].item<int64_t>();
      CHECK_GE(global_row, 0);
      CHECK_LT(global_row, cp_global_rows_);
      const auto owner_it = std::upper_bound(
          cp_row_offsets_.begin(), cp_row_offsets_.end(), global_row);
      const int32_t owner =
          static_cast<int32_t>(owner_it - cp_row_offsets_.begin() - 1);
      CHECK_GE(owner, 0);
      CHECK_LT(owner, world_size);
      if (owner == rank) {
        local_rows.emplace_back(global_row - cp_row_offsets_[owner]);
        output_rows.emplace_back(output);
      }
    }

    auto partial =
        torch::zeros({selected_indices.numel(), local_hidden_states.size(1)},
                     local_hidden_states.options());
    if (!local_rows.empty()) {
      const auto local_row_tensor =
          torch::tensor(local_rows,
                        torch::TensorOptions()
                            .dtype(torch::kLong)
                            .device(local_hidden_states.device()));
      const auto output_row_tensor =
          torch::tensor(output_rows,
                        torch::TensorOptions()
                            .dtype(torch::kLong)
                            .device(local_hidden_states.device()));
      partial.index_copy_(
          0,
          output_row_tensor,
          local_hidden_states.index_select(0, local_row_tensor));
    }
    auto gathered = cp_group_->allgather_base_sync(partial);
    return gathered.sum(0);
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    auto* graph_state = dynamic_cast<Glm5NextGraphMetadataState*>(state);
    CHECK(graph_state != nullptr);
    const bool is_decode = params.meta.batch_forward_type.is_decode();
    const bool is_prefill = params.meta.batch_forward_type.is_prefill() ||
                            params.meta.batch_forward_type.is_chunked_prefill();
    CHECK(is_decode || is_prefill);
    if (prefill_cp_enabled_ && is_prefill && !params.is_spec_verify) {
      layer::prepare_glm5_next_cp_metadata(
          *params.attn_metadata, *cp_group_, positions.size(0), device_);
    }
    const auto& attention = params.attention.device;
    const auto copy_cp_metadata = [](const layer::AttentionMetadata& source,
                                     layer::AttentionMetadata& target) {
      target.glm5_prefill_cp_enabled = source.glm5_prefill_cp_enabled;
      target.glm5_prefill_cp_rank = source.glm5_prefill_cp_rank;
      target.glm5_prefill_cp_size = source.glm5_prefill_cp_size;
      target.glm5_prefill_global_rows = source.glm5_prefill_global_rows;
      target.glm5_prefill_local_row_begin = source.glm5_prefill_local_row_begin;
      target.glm5_prefill_local_row_end = source.glm5_prefill_local_row_end;
      target.glm5_prefill_local_row_indices =
          source.glm5_prefill_local_row_indices;
      target.glm5_prefill_row_offsets = source.glm5_prefill_row_offsets;
      target.glm5_prefill_cp_round_robin = source.glm5_prefill_cp_round_robin;
    };
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
      if (params.is_spec_verify) {
        metadata.is_spec_verify = true;
        metadata.max_seq_len = attention.block_tables.size(1) * 256;
      }
      metadata.q_cu_seq_lens = attention.q_seq_lens;
      metadata.kv_cu_seq_lens = attention.kv_seq_lens;
      metadata.block_table = attention.block_tables;
      metadata.slot_mapping = attention.new_cache_slots;
      if (prefill_cp_enabled_ && is_prefill && !params.is_spec_verify) {
        copy_cp_metadata(*params.attn_metadata, metadata);
      }
      if (params.embedding.linear_state_indices.defined()) {
        metadata.linear_state_indices = params.embedding.linear_state_indices;
      }
      metadata.linear_state_read_indices = torch::Tensor();
      if (params.linear_state_validity_mask_tensor.defined()) {
        metadata.has_initial_states = params.linear_state_validity_mask_tensor;
      }
      if (is_decode) {
        CHECK(metadata.musa_kda_batch);
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
      if (params.is_spec_verify) {
        metadata.is_spec_verify = true;
        metadata.max_seq_len = attention.block_tables.size(1) * 256;
      }
      metadata.q_cu_seq_lens = attention.q_seq_lens;
      metadata.kv_cu_seq_lens = attention.kv_seq_lens;
      metadata.block_table = attention.block_tables;
      metadata.slot_mapping = attention.new_cache_slots;
      if (prefill_cp_enabled_ && is_prefill && !params.is_spec_verify) {
        copy_cp_metadata(*params.attn_metadata, metadata);
      }
      if (params.linear_state_validity_mask_tensor.defined() &&
          metadata.has_initial_states.defined()) {
        metadata.has_initial_states.copy_(
            params.linear_state_validity_mask_tensor);
      }
      if (metadata.musa_kda_batch) {
        const auto& captured = metadata.musa_kda_batch;
        const torch::Tensor& slots = params.embedding.linear_state_indices;
        CHECK_EQ(captured->num_tokens, positions.numel());
        CHECK_EQ(captured->state_indices.data_ptr(), slots.data_ptr());
        // Ordinary decode uses the immutable [0, 1, ..., batch] query offsets.
        // Validate the live write slots and refresh the captured int64 buffer.
        const auto& host_ids = params.embedding.linear_state_ids;
        torch::Tensor host_slots;
        const int32_t* ids = host_ids.data();
        if (static_cast<int64_t>(host_ids.size()) != slots.numel()) {
          // Tensor-only and padded inputs may have no matching host mirror.
          host_slots = slots.cpu().contiguous();
          ids = host_slots.data_ptr<int32_t>();
        }
        std::unordered_set<int32_t> unique;
        for (int64_t i = 0; i < slots.numel(); ++i) {
          CHECK_GE(ids[i], 0);
          CHECK_LT(ids[i], captured->state_capacity);
          CHECK(slots.numel() == 1 || unique.insert(ids[i]).second)
              << "KDA write slots must be unique";
        }
        captured->state_indices_long.copy_(slots, /*non_blocking=*/true);
      }
    }
    CHECK(attention.new_cache_slots.defined());
    if (!graph_state->slot_mapping.defined()) {
      graph_state->slot_mapping = torch::empty(
          attention.new_cache_slots.sizes(),
          attention.new_cache_slots.options().dtype(torch::kInt64));
    } else {
      CHECK_EQ(graph_state->slot_mapping.sizes(),
               attention.new_cache_slots.sizes())
          << "GLM5 graph slot mapping shape changed within one graph";
    }
    graph_state->slot_mapping.copy_(attention.new_cache_slots,
                                    /*non_blocking=*/true);
    graph_state->metadata->slot_mapping = graph_state->slot_mapping;
    if (prefill_cp_enabled_ && is_prefill && !params.is_spec_verify) {
      if (!graph_state->metadata->glm5_prefill_cp_layout) {
        graph_state->metadata->glm5_prefill_cp_layout =
            layer::build_glm5_next_cp_layout(
                positions, *graph_state->metadata, *cp_group_);
      } else {
        layer::refresh_glm5_next_cp_layout(
            graph_state->metadata->glm5_prefill_cp_layout,
            positions,
            graph_state->slot_mapping);
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
  ProcessGroup* cp_group_ = nullptr;
  int32_t cp_size_ = 1;
  int32_t ep_size_ = 1;
  bool prefill_cp_enabled_ = false;
  bool prefill_ep_enabled_ = false;
  bool last_forward_prefill_cp_ = false;
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
    if (!model_->can_restore_prefill_cp_hidden(hidden_states)) {
      return LlmForCausalLMImplBase<Glm5NextModel>::logits(hidden_states,
                                                           selected_indices);
    }
    if (!selected_indices.defined()) {
      return LlmForCausalLMImplBase<Glm5NextModel>::logits(
          model_->restore_prefill_cp_hidden(hidden_states), selected_indices);
    }
    torch::Tensor selected =
        model_->select_prefill_cp_hidden(hidden_states, selected_indices);
    return LlmForCausalLMImplBase<Glm5NextModel>::logits(selected,
                                                         torch::Tensor());
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices,
                       torch::Tensor& out_hidden) override {
    if (!model_->can_restore_prefill_cp_hidden(hidden_states)) {
      return LlmForCausalLMImplBase<Glm5NextModel>::logits(
          hidden_states, selected_indices, out_hidden);
    }
    if (!selected_indices.defined()) {
      return LlmForCausalLMImplBase<Glm5NextModel>::logits(
          model_->restore_prefill_cp_hidden(hidden_states),
          selected_indices,
          out_hidden);
    }
    torch::Tensor selected =
        model_->select_prefill_cp_hidden(hidden_states, selected_indices);
    return LlmForCausalLMImplBase<Glm5NextModel>::logits(
        selected, torch::Tensor(), out_hidden);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices) override {
    if (!model_->can_restore_prefill_cp_hidden(hidden_states)) {
      return LlmForCausalLMImplBase<Glm5NextModel>::pooler(hidden_states,
                                                           selected_indices);
    }
    if (!selected_indices.defined()) {
      return LlmForCausalLMImplBase<Glm5NextModel>::pooler(
          model_->restore_prefill_cp_hidden(hidden_states), selected_indices);
    }
    torch::Tensor selected =
        model_->select_prefill_cp_hidden(hidden_states, selected_indices);
    return LlmForCausalLMImplBase<Glm5NextModel>::pooler(selected,
                                                         torch::Tensor());
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
