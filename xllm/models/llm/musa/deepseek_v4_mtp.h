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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_decoder_layer.h"
#include "core/layers/musa/deepseek_v4/dsa_metadata_builder_musa.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mtp_model_base.h"
#include "models/llm/musa/deepseek_v4.h"
#include "models/model_registry.h"

namespace xllm {

class DeepseekV4MultiTokenPredictorLayerImpl final
    : public MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  DeepseekV4MultiTokenPredictorLayerImpl(const ModelContext& context,
                                         int32_t layer_index)
      : MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>(context,
                                                               layer_index) {}

  torch::Tensor forward(torch::Tensor inputs_embeds,
                        torch::Tensor previous_hidden_states,
                        torch::Tensor positions,
                        layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        torch::Tensor tokens,
                        torch::Tensor* aux_hidden_states) {
    ModelInputParams modified_input_params = input_params;
    modified_input_params.embedding.input_embedding = previous_hidden_states;
    std::optional<torch::Tensor> residual;
    return MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>::forward(
        inputs_embeds,
        residual,
        positions,
        attn_metadata,
        kv_cache,
        modified_input_params,
        tokens,
        aux_hidden_states);
  }
};
TORCH_MODULE(DeepseekV4MultiTokenPredictorLayer);

class DeepseekV4MtpModelImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV4MtpModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()) {
    const torch::TensorOptions& options = context.get_tensor_options();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    device_ = options.device();

    const int32_t mtp_n_layers = model_args_.n_layers();
    CHECK_EQ(mtp_n_layers, 1)
        << "DeepSeek-V4 bundled MTP currently supports one predictor layer";

    const int64_t num_heads = model_args_.n_heads();
    const int64_t dp_local_tp_size =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(num_heads % dp_local_tp_size, 0)
        << "DeepSeek-V4 MTP n_heads must be divisible by local TP size";

    window_size_ = model_args_.window_size();
    index_topk_ = model_args_.index_topk();

    const int64_t rope_head_dim = model_args_.rope_head_dim();
    const int64_t max_position_embeddings =
        model_args_.max_position_embeddings();
    if (rope_head_dim > 0 && max_position_embeddings > 0) {
      const int64_t original_max_position_embeddings =
          model_args_.rope_scaling_original_max_position_embeddings() > 0
              ? model_args_.rope_scaling_original_max_position_embeddings()
              : max_position_embeddings;
      dsa_rotary_embedding_ =
          std::make_shared<layer::DeepseekV4RotaryEmbedding>(
              /*rotary_dim=*/rope_head_dim,
              /*max_position_embeddings=*/max_position_embeddings,
              /*interleaved=*/true,
              /*rope_theta=*/model_args_.rope_theta(),
              /*compress_rope_theta=*/model_args_.compress_rope_theta(),
              /*scaling_factor=*/model_args_.factor(),
              /*extrapolation_factor=*/1.0f,
              /*beta_fast=*/model_args_.beta_fast(),
              /*beta_slow=*/model_args_.beta_slow(),
              /*attn_factor=*/model_args_.rope_scaling_attn_factor(),
              /*mscale=*/1.0f,
              /*mscale_all_dim=*/1.0f,
              /*original_max_position_embeddings=*/
              original_max_position_embeddings,
              options);
      dsa_cos_sin_ = dsa_rotary_embedding_->get_cos_sin_cache("default");
    }

    if (model_args_.index_head_dim() > 0) {
      const int64_t hadamard_dim =
          deepseek_v4_next_power_of_two(model_args_.index_head_dim());
      dsa_hadamard_ = deepseek_v4_create_hadamard_matrix(
          hadamard_dim, options.dtype().toScalarType(), options.device());
    }

    deepseek_v4_build_cache_specs(model_args_, caches_info_, group_infos_);
    align_cache_specs_to_dsv4_managers();

    mtp_layers_.reserve(static_cast<size_t>(mtp_n_layers));
    for (int32_t layer_id = 0; layer_id < mtp_n_layers; ++layer_id) {
      mtp_layers_.emplace_back(
          DeepseekV4MultiTokenPredictorLayer(context, layer_id));
      register_module("layer_" + std::to_string(layer_id), mtp_layers_.back());
    }

    final_norm_ = register_module("final_norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  void load_state_dict(const StateDict& state_dict) {
    CHECK_EQ(mtp_layers_.size(), 1U)
        << "DeepSeek-V4 bundled MTP requires exactly one predictor layer";
    const StateDict layer_state_dict = state_dict.get_dict_with_prefix(
        std::vector<std::string>{"0.", "layers.0."});
    CHECK_GT(layer_state_dict.size(), 0U)
        << "Failed to find DeepSeek-V4 bundled MTP layer weights";
    mtp_layers_[0]->load_state_dict(layer_state_dict);

    final_norm_->load_state_dict(state_dict.get_dict_with_prefix(
        std::vector<std::string>{"0.norm.", "layers.0.norm."}));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (const DeepseekV4MultiTokenPredictorLayer& layer : mtp_layers_) {
      layer->verify_loaded_weights();
    }
    final_norm_->verify_loaded_weights(prefix + "mtp.0.norm.");
  }

  void merge_loaded_weights() {}
  void merge_and_move_pinned_host() {}
  void free_weights() {}
  void reload_weights() {}
  void reload_non_decoder_weights() {}
  void reload_weights_from_device() {}
  void refresh_rolling_weights() {}

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;

    // The current runtime keeps enable_graph set on eager prefill requests.
    // Only decode enters the MUSA capture/replay path.
    const bool graph_forward = input_params.enable_graph &&
                               input_params.meta.batch_forward_type.is_decode();
    const bool is_graph_empty_dp_rank =
        graph_forward && input_params.meta.actual_num_sequences == 0 &&
        input_params.meta.num_sequences > 0;
    const bool is_empty_dp_rank = input_params.meta.num_sequences == 0 ||
                                  is_graph_empty_dp_rank || !tokens.defined() ||
                                  tokens.numel() == 0;
    if (is_empty_dp_rank && !graph_forward) {
      tokens = torch::tensor(
          {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::tensor(
          {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    torch::Tensor previous_hidden_states =
        input_params.embedding.input_embedding;
    torch::Tensor hidden_states;
    const int64_t hidden_size = model_args_.hidden_size();
    const int64_t target_context_width = hidden_size * model_args_.hc_mult();
    if (graph_forward && previous_hidden_states.defined() &&
        previous_hidden_states.dim() == 2 &&
        previous_hidden_states.size(1) == target_context_width + hidden_size) {
      hidden_states = previous_hidden_states.slice(
          /*dim=*/1,
          /*start=*/target_context_width,
          /*end=*/target_context_width + hidden_size);
      previous_hidden_states = previous_hidden_states.slice(
          /*dim=*/1, /*start=*/0, /*end=*/target_context_width);
    } else {
      hidden_states = embed_tokens_(tokens);
    }
    if (!previous_hidden_states.defined() && is_empty_dp_rank) {
      previous_hidden_states = hidden_states;
    }
    CHECK(previous_hidden_states.defined())
        << "DeepSeek-V4 MTP requires target pre-HC hidden states";

    const torch::Device runtime_device = hidden_states.device();
    ModelInputParams modified_input_params = input_params;
    modified_input_params.embedding.input_embedding = previous_hidden_states;
    if (is_empty_dp_rank && !graph_forward) {
      fill_empty_dp_rank_input_params(modified_input_params, &kv_caches);
    }

    if (graph_forward) {
      CHECK(tokens.defined() && tokens.device() == runtime_device)
          << "DeepSeek-V4 MTP graph requires tokens on the runtime device";
      CHECK(positions.defined() && positions.device() == runtime_device)
          << "DeepSeek-V4 MTP graph requires positions on the runtime device";
      CHECK(modified_input_params.attention.device.new_cache_slots.defined())
          << "DeepSeek-V4 MTP graph requires persistent new_cache_slots";
      CHECK(modified_input_params.attention.device.block_tables.defined())
          << "DeepSeek-V4 MTP graph requires persistent block_tables";
      normalize_graph_metadata_input_params(modified_input_params);
    } else {
      tokens = maybe_to_device(tokens, runtime_device);
      positions = maybe_to_device(positions, runtime_device);

      torch::Tensor zero_position_mask = positions == 0;
      if (zero_position_mask.any().item<bool>()) {
        hidden_states.index_put_(
            {zero_position_mask},
            torch::zeros_like(hidden_states.index({zero_position_mask})));
      }
    }

    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata ||
        !modified_input_params.attn_metadata->dsa_metadata) {
      CHECK(!graph_forward)
          << "DeepSeek-V4 MTP graph requires prebuilt DSA metadata";
      modified_input_params.attn_metadata =
          build_attention_metadata_for_forward(positions,
                                               modified_input_params);
    }
    CHECK(modified_input_params.attn_metadata != nullptr &&
          modified_input_params.attn_metadata->dsa_metadata != nullptr)
        << "DeepSeek-V4 MTP requires DSA metadata";

    if (!graph_forward) {
      prepare_dsa_metadata_for_forward(*modified_input_params.attn_metadata,
                                       runtime_device);
    }

    CHECK_GE(static_cast<int32_t>(kv_caches.size()),
             static_cast<int32_t>(mtp_layers_.size()))
        << "DeepSeek-V4 MTP requires one KV cache per predictor layer";

    layer::AttentionMetadata& attn_metadata =
        *modified_input_params.attn_metadata;
    const layer::DSAMetadata& dsa = *attn_metadata.dsa_metadata;
    std::unordered_map<int32_t, layer::DeepseekV4RotaryEmbedding::CosSinPair>
        input_rope_by_ratio;
    if (dsa.cos.defined() && dsa.sin.defined()) {
      input_rope_by_ratio[1] = {dsa.cos, dsa.sin};
    }
    if (dsa.c4_input_cos.defined() && dsa.c4_input_sin.defined()) {
      input_rope_by_ratio[4] = {dsa.c4_input_cos, dsa.c4_input_sin};
    }
    if (dsa.c128_input_cos.defined() && dsa.c128_input_sin.defined()) {
      input_rope_by_ratio[128] = {dsa.c128_input_cos, dsa.c128_input_sin};
    }

    torch::Tensor aux_hidden_states;
    for (size_t index = 0; index < mtp_layers_.size(); ++index) {
      const int32_t layer_id = static_cast<int32_t>(index);
      prepare_for_layer(attn_metadata, layer_id, input_rope_by_ratio);
      hidden_states = mtp_layers_[index](hidden_states,
                                         previous_hidden_states,
                                         positions,
                                         attn_metadata,
                                         kv_caches[index],
                                         modified_input_params,
                                         tokens,
                                         &aux_hidden_states);
    }

    std::optional<torch::Tensor> final_norm_output = std::nullopt;
    if (graph_forward) {
      const int64_t num_tokens = hidden_states.size(0);
      torch::Tensor& output_workspace = graph_final_norm_outputs_[num_tokens];
      if (!output_workspace.defined()) {
        output_workspace = torch::empty_like(hidden_states);
      }
      final_norm_output = output_workspace;
    }
    auto [output, residual] =
        final_norm_->forward(hidden_states, std::nullopt, final_norm_output);
    UNUSED_PARAMETER(residual);
    CHECK(aux_hidden_states.defined())
        << "DeepSeek-V4 MTP predictor did not return pre-HC hidden states";
    return ModelOutput(output, torch::Tensor(), aux_hidden_states.flatten(1));
  }

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<DeepseekV4GraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    CHECK(state != nullptr)
        << "DeepSeek-V4 MTP graph metadata state must be initialized";
    DeepseekV4GraphMetadataState* deepseek_v4_state =
        dynamic_cast<DeepseekV4GraphMetadataState*>(state);
    CHECK(deepseek_v4_state != nullptr)
        << "DeepSeek-V4 MTP received incompatible graph metadata state";

    ModelInputParams modified_input_params = input_params;
    if (modified_input_params.meta.actual_num_sequences == 0) {
      if (modified_input_params.meta.num_sequences > 0) {
        fill_empty_dp_rank_graph_metadata_input_params(modified_input_params);
      } else {
        fill_empty_dp_rank_input_params(modified_input_params);
      }
    }
    normalize_graph_metadata_input_params(modified_input_params);
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    std::shared_ptr<layer::AttentionMetadata> attn_metadata =
        std::make_shared<layer::AttentionMetadata>(
            layer::DSAMetadataBuilderMusa::build(modified_input_params,
                                                 positions,
                                                 dsa_cos_sin_,
                                                 caches_info_,
                                                 group_infos_));
    if (attn_metadata->dsa_metadata) {
      layer::DSAMetadata& dsa = *attn_metadata->dsa_metadata;
      if (dsa_hadamard_.defined()) {
        dsa.hadamard = dsa_hadamard_;
      }
      DeepseekV4ModelImpl::copy_to_graph_packed_metadata_buffer(
          dsa, deepseek_v4_state->dsa_metadata_persistent, positions.device());
      prepare_dsa_metadata_for_forward(*attn_metadata,
                                       positions.device(),
                                       /*pack_metadata=*/false,
                                       /*build_rope=*/true);
    }
    input_params.attn_metadata = persist_graph_attention_metadata(
        *deepseek_v4_state, std::move(attn_metadata));
    CHECK(input_params.attn_metadata != nullptr &&
          input_params.attn_metadata->dsa_metadata != nullptr)
        << "DeepSeek-V4 MTP graph requires DSA metadata";
  }

 private:
  void align_cache_specs_to_dsv4_managers() {
    constexpr int32_t kBaseBlockSize = 128;
    const int32_t window_size = static_cast<int32_t>(window_size_);
    if (group_infos_.size() >= 3) {
      return;
    }

    group_infos_ = {{DSACacheType::SLIDING_WINDOW, 1, window_size},
                    {DSACacheType::TOKEN, 4, kBaseBlockSize},
                    {DSACacheType::TOKEN, 128, kBaseBlockSize}};
    caches_info_.assign(static_cast<size_t>(model_args_.n_layers()), {});
    for (int32_t layer_id = 0; layer_id < model_args_.n_layers(); ++layer_id) {
      const int32_t compress_ratio = deepseek_v4_normalize_compress_ratio(
          layer_id < static_cast<int32_t>(model_args_.compress_ratios().size())
              ? model_args_.compress_ratios()[static_cast<size_t>(layer_id)]
              : 1);
      if (compress_ratio == 4) {
        caches_info_[static_cast<size_t>(layer_id)] = {
            {1, DSACacheType::TOKEN, 4, kBaseBlockSize},
            {1, DSACacheType::TOKEN, 4, kBaseBlockSize},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size},
            {1, DSACacheType::TOKEN, 4, kBaseBlockSize}};
      } else if (compress_ratio == 128) {
        caches_info_[static_cast<size_t>(layer_id)] = {
            {2, DSACacheType::TOKEN, 128, kBaseBlockSize},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size},
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size}};
      } else {
        caches_info_[static_cast<size_t>(layer_id)] = {
            {0, DSACacheType::SLIDING_WINDOW, 1, window_size}};
      }
    }
  }

  void normalize_graph_metadata_input_params(ModelInputParams& params) const {
    int64_t actual_metadata_rows =
        std::max<int64_t>(params.meta.actual_num_sequences, 0);
    int64_t padded_metadata_rows = actual_metadata_rows;
    if (params.enable_graph) {
      padded_metadata_rows =
          std::max<int64_t>(padded_metadata_rows, params.meta.num_sequences);
    }
    if (padded_metadata_rows <= 0) {
      padded_metadata_rows = 1;
    }
    actual_metadata_rows =
        std::min<int64_t>(actual_metadata_rows, padded_metadata_rows);

    auto trim_lens = [padded_metadata_rows,
                      actual_metadata_rows](std::vector<int32_t>& lens) {
      const bool cumulative =
          actual_metadata_rows > 0 &&
          lens.size() >= static_cast<size_t>(actual_metadata_rows + 1) &&
          lens.front() == 0 && std::is_sorted(lens.begin(), lens.end());
      if (cumulative) {
        const size_t actual_boundary = static_cast<size_t>(std::min<int64_t>(
            actual_metadata_rows, static_cast<int64_t>(lens.size()) - 1));
        const int32_t cumulative_tail = lens[actual_boundary];
        lens.resize(static_cast<size_t>(padded_metadata_rows + 1),
                    cumulative_tail);
        std::fill(
            lens.begin() + actual_boundary + 1, lens.end(), cumulative_tail);
        return;
      }
      if (lens.empty()) {
        lens.assign(static_cast<size_t>(padded_metadata_rows), 0);
      } else if (static_cast<int64_t>(lens.size()) < padded_metadata_rows) {
        lens.resize(static_cast<size_t>(padded_metadata_rows), 0);
      } else {
        lens.resize(static_cast<size_t>(padded_metadata_rows));
      }
      std::fill(lens.begin() + actual_metadata_rows, lens.end(), 0);
    };

    trim_lens(params.attention.host.kv_seq_lens);
    trim_lens(params.attention.host.q_seq_lens);
    params.meta.num_sequences = static_cast<int32_t>(padded_metadata_rows);
    params.meta.actual_num_sequences =
        static_cast<int32_t>(actual_metadata_rows);
  }

  std::shared_ptr<layer::AttentionMetadata>
  build_attention_metadata_for_forward(const torch::Tensor& positions,
                                       const ModelInputParams& input_params) {
    ModelInputParams modified_input_params = input_params;
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    return std::make_shared<layer::AttentionMetadata>(
        layer::DSAMetadataBuilderMusa::build(modified_input_params,
                                             positions,
                                             dsa_cos_sin_,
                                             caches_info_,
                                             group_infos_));
  }

  std::shared_ptr<layer::AttentionMetadata> persist_graph_attention_metadata(
      DeepseekV4GraphMetadataState& state,
      std::shared_ptr<layer::AttentionMetadata> metadata) const {
    if (!metadata || !metadata->dsa_metadata) {
      return metadata;
    }

    layer::DSAMetadata& dsa = *metadata->dsa_metadata;
    DeepseekV4GraphMetadataState::DSAMetadataPersistent& persistent =
        state.dsa_metadata_persistent;
    dsa.attn_mask = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.attn_mask, persistent.attn_mask);
    dsa.start_pos = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.start_pos, persistent.start_pos);
    dsa.cos =
        DeepseekV4ModelImpl::copy_to_persistent_tensor(dsa.cos, persistent.cos);
    dsa.sin =
        DeepseekV4ModelImpl::copy_to_persistent_tensor(dsa.sin, persistent.sin);
    dsa.c4_cos = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c4_cos, persistent.c4_cos);
    dsa.c4_sin = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c4_sin, persistent.c4_sin);
    dsa.c128_cos = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c128_cos, persistent.c128_cos);
    dsa.c128_sin = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c128_sin, persistent.c128_sin);
    dsa.c4_input_cos = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c4_input_cos, persistent.c4_input_cos);
    dsa.c4_input_sin = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c4_input_sin, persistent.c4_input_sin);
    dsa.c128_input_cos = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c128_input_cos, persistent.c128_input_cos);
    dsa.c128_input_sin = DeepseekV4ModelImpl::copy_to_persistent_tensor(
        dsa.c128_input_sin, persistent.c128_input_sin);
    return metadata;
  }

  void fill_empty_dp_rank_input_params(
      ModelInputParams& params,
      const std::vector<KVCache>* kv_caches = nullptr) const {
    const bool is_chunked_prefill =
        params.meta.batch_forward_type.is_chunked_prefill();
    params.attn_metadata = nullptr;
    const torch::TensorOptions cpu_int_options = torch::TensorOptions()
                                                     .dtype(torch::kInt32)
                                                     .device(torch::kCPU)
                                                     .pinned_memory(true);
    const int32_t dummy_kv_len =
        static_cast<int32_t>(std::max<int64_t>({index_topk_, window_size_, 1}));
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.kv_max_seq_len =
        std::max<int32_t>(params.meta.kv_max_seq_len, dummy_kv_len);
    params.meta.q_max_seq_len = 1;
    params.meta.batch_forward_type = is_chunked_prefill
                                         ? BatchForwardType::CHUNKED_PREFILL
                                         : BatchForwardType::DECODE;
    params.attention.host.kv_seq_lens = {dummy_kv_len};
    params.attention.host.q_seq_lens = {1};
    params.attention.host.q_cu_seq_lens = {1};
    params.attention.device.kv_seq_lens =
        torch::tensor(params.attention.host.kv_seq_lens, cpu_int_options);
    params.attention.device.q_seq_lens =
        torch::tensor(params.attention.host.q_seq_lens, cpu_int_options);
    params.attention.device.q_cu_seq_lens =
        torch::tensor(params.attention.host.q_cu_seq_lens, cpu_int_options);
    params.attention.device.kv_cache_tokens_nums =
        torch::tensor({1}, cpu_int_options);
    params.attention.host.kv_cache_tokens_nums = {1};
    params.attention.device.new_cache_slots =
        torch::tensor({0}, cpu_int_options);
    params.attention.device.block_tables =
        torch::zeros({1, 1}, cpu_int_options);

    if (!params.multi_block_tables.empty()) {
      return;
    }
    const int32_t manager_num = static_cast<int32_t>(group_infos_.size());
    params.multi_block_tables.reserve(static_cast<size_t>(manager_num));
    for (int32_t manager_id = 0; manager_id < manager_num; ++manager_id) {
      int64_t block_num = 1;
      if (kv_caches != nullptr && !kv_caches->empty()) {
        block_num = empty_dp_rank_cache_blocks_for_group(manager_id,
                                                         kv_caches->front());
      }
      params.multi_block_tables.emplace_back(
          torch::zeros({1, std::max<int64_t>(block_num, 1)}, cpu_int_options));
    }
  }

  void fill_empty_dp_rank_graph_metadata_input_params(
      ModelInputParams& params) const {
    params.attn_metadata = nullptr;
    const int64_t metadata_batch_size =
        std::max<int64_t>(params.meta.num_sequences, 1);
    const int32_t dummy_kv_len =
        static_cast<int32_t>(std::max<int64_t>({index_topk_, window_size_, 1}));
    params.meta.num_sequences = static_cast<int32_t>(metadata_batch_size);
    params.meta.kv_max_seq_len =
        std::max<int32_t>(params.meta.kv_max_seq_len, dummy_kv_len);
    params.meta.q_max_seq_len = 1;
    params.meta.batch_forward_type = BatchForwardType::DECODE;

    auto pad_lens = [metadata_batch_size](std::vector<int32_t>& lens,
                                          int32_t fill_value) {
      lens.resize(static_cast<size_t>(metadata_batch_size), fill_value);
      for (int32_t& length : lens) {
        length = std::max<int32_t>(length, fill_value);
      }
    };
    pad_lens(params.attention.host.kv_seq_lens, dummy_kv_len);
    pad_lens(params.attention.host.q_seq_lens, 1);

    const int32_t manager_num = static_cast<int32_t>(group_infos_.size());
    bool has_full_multi_block_tables =
        static_cast<int32_t>(params.multi_block_tables.size()) == manager_num;
    if (has_full_multi_block_tables) {
      for (const torch::Tensor& block_table : params.multi_block_tables) {
        if (!block_table.defined() || block_table.dim() != 2 ||
            block_table.size(0) < metadata_batch_size) {
          has_full_multi_block_tables = false;
          break;
        }
      }
    }
    if (has_full_multi_block_tables) {
      return;
    }

    const torch::TensorOptions cpu_int_options = torch::TensorOptions()
                                                     .dtype(torch::kInt32)
                                                     .device(torch::kCPU)
                                                     .pinned_memory(true);
    params.multi_block_tables.clear();
    params.multi_block_tables.reserve(static_cast<size_t>(manager_num));
    for (int32_t manager_id = 0; manager_id < manager_num; ++manager_id) {
      UNUSED_PARAMETER(manager_id);
      params.multi_block_tables.emplace_back(
          torch::zeros({metadata_batch_size, 1}, cpu_int_options));
    }
  }

  int64_t empty_dp_rank_cache_blocks_for_group(int32_t group_id,
                                               const KVCache& kv_cache) const {
    if (group_id < 0 || group_id >= static_cast<int32_t>(group_infos_.size())) {
      return 1;
    }
    const DSAGroupInfo& group_info =
        group_infos_[static_cast<size_t>(group_id)];
    const int64_t block_size = std::max<int64_t>(group_info.block_size, 1);
    torch::Tensor cache;
    if (group_info.type == DSACacheType::SLIDING_WINDOW) {
      cache = kv_cache.get_swa_cache();
    } else if (group_info.type == DSACacheType::TOKEN) {
      cache = kv_cache.get_k_cache();
    }
    if (!cache.defined() || cache.dim() == 0) {
      return 1;
    }
    return std::max<int64_t>((cache.size(0) + block_size - 1) / block_size, 1);
  }

  void prepare_dsa_metadata_for_forward(layer::AttentionMetadata& metadata,
                                        const torch::Device& runtime_device,
                                        bool pack_metadata = true,
                                        bool build_rope = true) const {
    CHECK(metadata.dsa_metadata != nullptr)
        << "DeepSeek-V4 MTP requires populated DSA metadata";
    layer::DSAMetadata& dsa = *metadata.dsa_metadata;
    if (dsa_hadamard_.defined()) {
      dsa.hadamard = dsa_hadamard_;
    }
    if (pack_metadata) {
      deepseek_v4_pack_dsa_metadata_to_device(dsa, runtime_device);
    }
    if (build_rope) {
      build_dsa_rope_metadata(dsa);
    }
    if (dsa.actual_seq_lengths_kv.defined() && dsa.seq_lens_q.defined()) {
      dsa.start_pos =
          (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch::kInt32);
    }
    dsa.c1_metadata = torch::Tensor();
    dsa.c4_metadata = torch::Tensor();
    dsa.c128_metadata = torch::Tensor();
    dsa.qli_metadata = torch::Tensor();
  }

  void build_dsa_rope_metadata(layer::DSAMetadata& dsa) const {
    if (!dsa_rotary_embedding_) {
      return;
    }

    std::unordered_map<std::string, torch::Tensor> positions_by_group;
    dsa.cos = torch::Tensor();
    dsa.sin = torch::Tensor();
    dsa.c4_cos = torch::Tensor();
    dsa.c4_sin = torch::Tensor();
    dsa.c128_cos = torch::Tensor();
    dsa.c128_sin = torch::Tensor();
    dsa.c4_input_cos = torch::Tensor();
    dsa.c4_input_sin = torch::Tensor();
    dsa.c128_input_cos = torch::Tensor();
    dsa.c128_input_sin = torch::Tensor();

    auto append_positions = [&positions_by_group](
                                const std::string& group,
                                const torch::Tensor& positions) {
      if (!positions.defined() || positions.numel() == 0) {
        return;
      }
      torch::Tensor group_positions = positions;
      if (group_positions.scalar_type() != torch::kInt64) {
        group_positions = group_positions.to(torch::kInt64);
      }
      positions_by_group[group] = group_positions;
    };
    append_positions("default", dsa.input_positions);
    append_positions("c4", dsa.c4_pad_positions);
    append_positions("c128", dsa.c128_pad_positions);

    if (!positions_by_group.empty()) {
      const auto cos_sin_by_group =
          dsa_rotary_embedding_->build(positions_by_group);
      const auto default_it = cos_sin_by_group.find("default");
      if (default_it != cos_sin_by_group.end()) {
        dsa.cos = default_it->second.first;
        dsa.sin = default_it->second.second;
      }
      const auto c4_it = cos_sin_by_group.find("c4");
      if (c4_it != cos_sin_by_group.end()) {
        dsa.c4_cos = c4_it->second.first;
        dsa.c4_sin = c4_it->second.second;
      }
      const auto c128_it = cos_sin_by_group.find("c128");
      if (c128_it != cos_sin_by_group.end()) {
        dsa.c128_cos = c128_it->second.first;
        dsa.c128_sin = c128_it->second.second;
      }
    }

    if (!dsa.input_positions.defined() || dsa.input_positions.numel() == 0) {
      return;
    }
    torch::Tensor input_positions = dsa.input_positions;
    if (input_positions.scalar_type() != torch::kInt64) {
      input_positions = input_positions.to(torch::kInt64);
    }
    const auto input_cos_sin_by_group = dsa_rotary_embedding_->build(
        {{"c4", input_positions}, {"c128", input_positions}});
    const auto c4_input_it = input_cos_sin_by_group.find("c4");
    if (c4_input_it != input_cos_sin_by_group.end()) {
      dsa.c4_input_cos = c4_input_it->second.first;
      dsa.c4_input_sin = c4_input_it->second.second;
    }
    const auto c128_input_it = input_cos_sin_by_group.find("c128");
    if (c128_input_it != input_cos_sin_by_group.end()) {
      dsa.c128_input_cos = c128_input_it->second.first;
      dsa.c128_input_sin = c128_input_it->second.second;
    }
  }

  void prepare_for_layer(
      layer::AttentionMetadata& metadata,
      int32_t layer_id,
      const std::unordered_map<int32_t,
                               layer::DeepseekV4RotaryEmbedding::CosSinPair>&
          input_rope_by_ratio) const {
    CHECK(metadata.dsa_metadata != nullptr)
        << "DeepSeek-V4 MTP requires populated DSA metadata";
    layer::DSAMetadata& dsa = *metadata.dsa_metadata;
    dsa.layer_id = layer_id;
    const int32_t compress_ratio = deepseek_v4_normalize_compress_ratio(
        layer_id < static_cast<int32_t>(model_args_.compress_ratios().size())
            ? model_args_.compress_ratios()[static_cast<size_t>(layer_id)]
            : 1);
    auto rope_it = input_rope_by_ratio.find(compress_ratio);
    if (rope_it == input_rope_by_ratio.end()) {
      rope_it = input_rope_by_ratio.find(1);
    }
    if (rope_it != input_rope_by_ratio.end()) {
      dsa.cos = rope_it->second.first;
      dsa.sin = rope_it->second.second;
    }

    if (layer_id >= static_cast<int32_t>(dsa.block_tables.size()) ||
        layer_id >= static_cast<int32_t>(dsa.slot_mappings.size()) ||
        dsa.block_tables[static_cast<size_t>(layer_id)].empty() ||
        dsa.slot_mappings[static_cast<size_t>(layer_id)].empty()) {
      return;
    }

    size_t attention_cache_index = 0;
    if (layer_id < static_cast<int32_t>(caches_info_.size())) {
      const std::vector<DSACacheInfo>& layer_caches =
          caches_info_[static_cast<size_t>(layer_id)];
      for (size_t cache_index = 0; cache_index < layer_caches.size();
           ++cache_index) {
        if (layer_caches[cache_index].type == DSACacheType::SLIDING_WINDOW) {
          attention_cache_index = cache_index;
          break;
        }
      }
    }

    const std::vector<torch::Tensor>& layer_block_tables =
        dsa.block_tables[static_cast<size_t>(layer_id)];
    const std::vector<torch::Tensor>& layer_slot_mappings =
        dsa.slot_mappings[static_cast<size_t>(layer_id)];
    if (attention_cache_index < layer_block_tables.size() &&
        layer_block_tables[attention_cache_index].defined()) {
      metadata.block_table = layer_block_tables[attention_cache_index];
    }
    if (attention_cache_index < layer_slot_mappings.size() &&
        layer_slot_mappings[attention_cache_index].defined()) {
      metadata.slot_mapping = layer_slot_mappings[attention_cache_index];
    }
  }

  ModelArgs model_args_;
  torch::Device device_{torch::kCPU};
  int64_t window_size_ = 128;
  int64_t index_topk_ = 512;

  std::shared_ptr<layer::DeepseekV4RotaryEmbedding> dsa_rotary_embedding_;
  torch::Tensor dsa_cos_sin_;
  torch::Tensor dsa_hadamard_;
  std::vector<std::vector<DSACacheInfo>> caches_info_;
  std::vector<DSAGroupInfo> group_infos_;
  std::unordered_map<int64_t, torch::Tensor> graph_final_norm_outputs_;

  layer::RMSNorm final_norm_{nullptr};
  layer::WordEmbedding embed_tokens_{nullptr};
  std::vector<DeepseekV4MultiTokenPredictorLayer> mtp_layers_;
};
TORCH_MODULE(DeepseekV4MtpModel);

class DeepseekV4MtpForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV4MtpModel> {
 public:
  explicit DeepseekV4MtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4MtpModel>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    bool mtp_weights_loaded = false;
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      const StateDict mtp_state_dict = state_dict->get_dict_with_prefix(
          std::vector<std::string>{"mtp.", "model.mtp."});
      if (mtp_state_dict.size() == 0U) {
        continue;
      }
      model_->load_state_dict(mtp_state_dict);
      mtp_weights_loaded = true;
    }
    CHECK(mtp_weights_loaded)
        << "Failed to find DeepSeek-V4 bundled MTP weights under mtp.* or "
           "model.mtp.*";
    model_->verify_loaded_weights(prefix);
  }

  bool requires_graph_forward_metadata() {
    return model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    model_->prepare_graph_forward_metadata(state, positions, input_params);
  }
};
TORCH_MODULE(DeepseekV4MtpForCausalLM);

inline void load_deepseek_v4_mtp_model_args(const JsonReader& json,
                                            ModelArgs* args) {
  load_deepseek_v4_model_args(json, args);
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4_mtp");
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);
  SET_ARG(n_hash_layers, 0);
}

REGISTER_CAUSAL_MODEL(deepseek_v4_mtp, DeepseekV4MtpForCausalLM);

REGISTER_MODEL_ARGS(deepseek_v4_mtp, [&] {
  constexpr DeepseekV4PolicyPreset preset = DeepseekV4PolicyPreset::kDefault;
  const DeepseekV4ArgsPolicy args_policy =
      build_deepseek_v4_args_policy(preset);
  load_deepseek_v4_mtp_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
  normalize_deepseek_v4_args(args);
});

}  // namespace xllm
