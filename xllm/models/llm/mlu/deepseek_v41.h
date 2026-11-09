/* Copyright 2026 The xLLM Authors.

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

#include <algorithm>

#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/model/aux_hidden_capture.h"
#include "framework/model/causal_lm.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/common/deepseek_v4_rotary_embedding.h"
#include "layers/common/word_embedding.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_decoder_layer.h"
#include "models/llm/deepseek_v41.h"
#include "models/llm/llm_model_base.h"

namespace xllm::mlu::model {
struct DeepseekV41GraphState final : ModelGraphMetadataState {
  layer::Csa2Context shared;
  std::optional<ModelInputParams> verify_input;
  std::vector<std::unique_ptr<layer::DeepseekV41EngramImpl::HashState>> hashes;
};

class DeepseekV41ModelImpl : public torch::nn::Module {
 public:
  explicit DeepseekV41ModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()),
        aux_capture_(context.get_model_args(),
                     context.get_tensor_options(),
                     SchedulerConfig::get_instance().max_tokens_per_batch()) {
    const auto& args = context.get_model_args();
    const auto& options = context.get_tensor_options();
    const auto rope_options = options.dtype(torch::kFloat32);
    const auto& parallel = context.get_parallel_args();
    cp_size_ = parallel.cp_size();
    cp_group_ = parallel.cp_group_;
    if (cp_size_ > 1) {
      CHECK_EQ(parallel.kv_split_size_effective(), 1);
      CHECK_EQ(parallel.layerwise_split_size(), 1);
      CHECK_EQ(parallel.ep_size(), cp_size_ * parallel.tp_group_->world_size());
    }
    CHECK_EQ(context.get_parallel_args().dp_size(), 1);
    CHECK_LE(args.num_speculative_tokens() + 1, kDsv41CompressStateRows);
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
    // The plain SWA layers do not use YaRN; every CSA2 layer does.
    layer::DeepseekV4RotaryEmbedding plain(args.qk_rope_head_dim(),
                                           args.max_position_embeddings(),
                                           true,
                                           args.rope_theta(),
                                           args.compress_rope_theta(),
                                           1,
                                           1,
                                           args.beta_fast(),
                                           args.beta_slow(),
                                           1,
                                           0,
                                           0,
                                           0,
                                           rope_options);
    layer::DeepseekV4RotaryEmbedding compressed(
        args.qk_rope_head_dim(),
        args.max_position_embeddings(),
        true,
        args.rope_theta(),
        args.compress_rope_theta(),
        args.factor(),
        1,
        args.beta_fast(),
        args.beta_slow(),
        1,
        0,
        0,
        args.rope_scaling_original_max_position_embeddings(),
        rope_options);
    plain_rope_ = plain.get_cos_sin_cache("default");
    compressed_rope_ = compressed.get_cos_sin_cache("c4");
    layers_.reserve(args.n_layers());
    for (int32_t id = 0; id < args.n_layers(); ++id) {
      layers_.emplace_back(layer::DeepseekV41DecoderLayer(context, id));
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor tokens) {
    return embed_tokens_(tokens);
  }
  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }
  void set_word_embedding(layer::WordEmbedding& embedding) {
    embed_tokens_ = embedding;
  }

  void load_hash_state(const StateDict& state) {
    for (auto& layer : layers_) {
      layer->load_hash_state(state);
    }
  }

  void load_state_dict(const StateDict& state) {
    embed_tokens_->load_state_dict(state.get_dict_with_prefix(
        std::vector<std::string>{"embed.", "embed_tokens."}));
    norm_->load_state_dict(state.get_dict_with_prefix("norm."));
    for (size_t id = 0; id < layers_.size(); ++id) {
      layers_[id]->load_state_dict(
          state.get_dict_with_prefix("layers." + std::to_string(id) + "."));
    }
  }

  void verify_loaded_weights() const {
    CHECK(embed_tokens_->is_weight_loaded()) << "Missing weight embed.weight";
    norm_->verify_loaded_weights(/*prefix=*/"norm.");
    for (size_t id = 0; id < layers_.size(); ++id) {
      layers_[id]->verify_loaded_weights("layers." + std::to_string(id) + ".");
    }
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input) {
    graph_state_ = dynamic_cast<DeepseekV41GraphState*>(state);
    CHECK(graph_state_ != nullptr);
    CHECK(input.meta.batch_forward_type.is_decode());
    const ModelInputParams* grouped = &input;
    if (input.meta.spec_verify_width > 1) {
      graph_state_->verify_input = group_verify_input(input);
      grouped = &*graph_state_->verify_input;
    }
    const auto& metadata = *grouped;

    if (graph_state_->hashes.empty()) {
      const bool has_compressed_layers = std::any_of(
          model_args_.compress_ratios().begin(),
          model_args_.compress_ratios().begin() + model_args_.n_layers(),
          [](int32_t ratio) { return ratio > 0; });
      const int64_t capacity =
          model_args_.window_size() +
          (has_compressed_layers ? model_args_.index_topk() : 0);
      graph_state_->shared.selected_kv = torch::empty(
          {positions.numel() * capacity, 1, 1, model_args_.head_dim()},
          options_);
      graph_state_->shared.selected_lengths =
          torch::empty({positions.numel()}, options_.dtype(torch::kInt32));
      graph_state_->shared.topk.resize(metadata.meta.num_sequences);
      graph_state_->shared.candidates.resize(metadata.meta.num_sequences);
      graph_state_->hashes.reserve(layers_.size());
      for (const auto& layer : layers_) {
        graph_state_->hashes.emplace_back(layer->create_hash_state(
            positions.numel(),
            positions.numel() / metadata.meta.num_sequences));
      }
    }
    auto& tables = graph_state_->shared.block_tables;
    tables.clear();
    tables.reserve(metadata.multi_block_tables.size());
    for (const auto& table : metadata.multi_block_tables) {
      tables.emplace_back(
          table.to(positions.device(), torch::kInt32, /*non_blocking=*/true));
    }
    for (auto& hash : graph_state_->hashes) {
      if (hash) {
        hash->prepare(metadata);
      }
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& raw_input) {
    torch::NoGradGuard guard;
    std::optional<ModelInputParams> grouped_input;
    if (!raw_input.enable_graph && raw_input.meta.spec_verify_width > 1) {
      grouped_input = group_verify_input(raw_input);
    }
    const auto& input = raw_input.enable_graph && graph_state_->verify_input
                            ? *graph_state_->verify_input
                            : (grouped_input ? *grouped_input : raw_input);
    std::optional<layer::mlu_v4_cp::DeepseekV4CpContext> cp_context;
    if (cp_size_ > 1 && !input.enable_graph &&
        !input.use_dflash_proposal_attention && !input.is_spec_verify &&
        input.meta.spec_verify_width <= 1 &&
        input.meta.batch_forward_type.no_decode()) {
      cp_context = layer::mlu_v4_cp::build_cp_context(
          cp_size_,
          cp_group_->rank(),
          cp_group_,
          layer::mlu_v4_cp::query_lengths_from_cumulative(
              input.attention.host.q_seq_lens, positions.numel()),
          positions);
    }
    // V4 geometry inserts dummy rows for empty halves. Keep V4.1 on the
    // replicated path for those batches so no dummy enters expert routing.
    if (cp_context) {
      const auto& geometry = cp_context->geometry;
      for (const auto& rows : geometry.scatter_rows_by_rank) {
        if (std::find(rows.begin(), rows.end(), geometry.total_tokens) !=
            rows.end()) {
          cp_context.reset();
          break;
        }
      }
    }
    torch::Tensor local_tokens = tokens;
    if (cp_context) {
      local_tokens = layer::mlu_v4_cp::shard_rows(tokens, *cp_context);
      positions = cp_context->local_positions;
    }
    torch::Tensor hidden = embed_tokens_(local_tokens)
                               .unsqueeze(1)
                               .repeat({1, model_args_.hc_mult(), 1});
    torch::Tensor pre =
        torch::zeros({local_tokens.numel(), model_args_.hc_mult()},
                     hidden.options().dtype(torch::kFloat32));
    pre.select(1, 0).fill_(1);
    layer::Csa2Context eager_shared;
    auto& shared = input.enable_graph ? graph_state_->shared : eager_shared;
    shared.topk.resize(input.meta.num_sequences * (cp_context ? 2 : 1));
    shared.candidates.resize(input.meta.num_sequences * (cp_context ? 2 : 1));
    std::optional<layer::PendingMHC> pending;
    for (size_t id = 0; id < layers_.size(); ++id) {
      if (!input.synchronize_layer(id)) {
        return ModelOutput();
      }
      torch::Tensor captured;
      const bool capture =
          aux_capture_.should_capture(static_cast<int32_t>(id));
      hidden = layers_[id]->forward(
          hidden,
          pre,
          tokens,
          positions,
          model_args_.compress_ratios()[id] == 0 ? plain_rope_
                                                 : compressed_rope_,
          caches,
          input,
          shared,
          pending,
          can_defer_post(id),
          input.enable_graph ? graph_state_->hashes[id].get() : nullptr,
          cp_context ? &*cp_context : nullptr,
          capture ? &captured : nullptr);
      if (capture) {
        aux_capture_.capture_layer(
            static_cast<int32_t>(id), captured, std::nullopt);
      }
      if (!input.record_layer(id, hidden.device())) {
        return ModelOutput();
      }
    }
    CHECK(!pending.has_value());
    hidden = kernel::mlu::fused_mul_reduce_sum(hidden, pre);
    auto [output, residual] = norm_(hidden, std::nullopt);
    // Capture still uses local query rows. Restore both outputs before the
    // worker selects global rows or passes target features to the draft.
    ModelOutput result = aux_capture_.finalize(output, residual);
    if (cp_context) {
      result.hidden_states =
          layer::mlu_v4_cp::gather_restore(output, *cp_context);
      if (result.aux_hidden_states.defined()) {
        result.aux_hidden_states = layer::mlu_v4_cp::gather_restore(
            result.aux_hidden_states, *cp_context);
      }
    }
    return result;
  }

 private:
  // Layer `id` may leave its post to the next layer's fused delayed-pre
  // boundary only when both layers run the fused MHC kernels and the next
  // layer neither injects Engram nor captures its entering residual.
  bool can_defer_post(size_t id) const {
    if (id + 1 >= layers_.size()) {
      return false;
    }
    const auto& next = layers_[id + 1];
    return layers_[id]->uses_fused_mhc() && next->uses_fused_mhc() &&
           !next->has_engram() &&
           !aux_capture_.should_capture(static_cast<int32_t>(id + 1));
  }

  // Generic verify expands each request into independent decode rows. V4.1
  // compression and Engram need the original contiguous request chunks.
  static ModelInputParams group_verify_input(const ModelInputParams& input) {
    ModelInputParams grouped = input;
    const int32_t width = input.meta.spec_verify_width;
    CHECK_EQ(input.meta.num_sequences % width, 0);
    CHECK_EQ(input.meta.actual_num_sequences % width, 0);
    grouped.meta.num_sequences /= width;
    grouped.meta.actual_num_sequences /= width;
    grouped.meta.q_max_seq_len = width;
    auto& host = grouped.attention.host;
    host.q_seq_lens = {0};
    host.kv_seq_lens = {0};
    host.q_seq_lens.reserve(grouped.meta.num_sequences + 1);
    host.kv_seq_lens.reserve(grouped.meta.num_sequences + 1);
    for (int32_t seq = 0; seq < grouped.meta.num_sequences; ++seq) {
      const int32_t last = (seq + 1) * width;
      host.q_seq_lens.emplace_back(input.attention.host.q_seq_lens[last]);
      host.kv_seq_lens.emplace_back(host.kv_seq_lens.back() +
                                    input.attention.host.kv_seq_lens[last] -
                                    input.attention.host.kv_seq_lens[last - 1]);
    }
    host.q_cu_seq_lens = host.q_seq_lens;
    for (auto& table : grouped.multi_block_tables) {
      table = table.slice(0, 0, table.size(0), width);
    }
    return grouped;
  }

  DeepseekV41GraphState* graph_state_ = nullptr;  // Owned by the active graph.
  int32_t cp_size_ = 1;
  ProcessGroup* cp_group_ = nullptr;

 protected:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
  std::vector<layer::DeepseekV41DecoderLayer> layers_;
  AuxHiddenCapture aux_capture_;
  torch::Tensor plain_rope_, compressed_rope_;
};
TORCH_MODULE(DeepseekV41Model);

class DeepseekV41ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV41Model> {
 public:
  explicit DeepseekV41ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV41Model>(context) {}
  bool requires_graph_forward_metadata() { return true; }
  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<DeepseekV41GraphState>();
  }
  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input) {
    model_->prepare_graph_forward_metadata(state, positions, input);
  }
  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    model_->load_hash_state(layer::build_deepseek_v41_hash_state(
        loader->model_args(),
        loader->model_weights_path() + "/tokenizer.json"));
    for (const auto& state : loader->get_state_dicts()) {
      // Both released checkpoints and HF exports are supported by stripping
      // their optional model. prefix without altering layer parameter names.
      std::unordered_map<std::string, torch::Tensor> normalized;
      for (auto it = state->begin(); it != state->end(); ++it) {
        const std::string name = it->first.rfind(prefix, 0) == 0
                                     ? it->first.substr(prefix.size())
                                     : it->first;
        if (loader->quant_args().is_compressed_tensors_w8a8_dynamic() &&
            (name == "layers.0.attn.wq_b.weight" ||
             name == "layers.0.ffn.experts.0.w1.weight")) {
          const auto& args = loader->model_args();
          const bool expert = name == "layers.0.ffn.experts.0.w1.weight";
          const std::vector<int64_t> shape =
              expert ? std::vector<int64_t>{args.moe_intermediate_size(),
                                            args.hidden_size() / 2}
                     : std::vector<int64_t>{args.n_heads() * args.head_dim(),
                                            args.q_lora_rank()};
          CHECK(it->second.scalar_type() == torch::kInt8 &&
                it->second.sizes().vec() == shape)
              << "Expected packed MLU integer weight: " << name;
        }
        normalized.emplace(name, it->second);
      }
      StateDict weights(std::move(normalized));
      model_->load_state_dict(weights);
      if (!embedding_mode_) {
        lm_head_->load_state_dict(weights.get_dict_with_prefix(
            tie_word_embeddings
                ? std::vector<std::string>{"embed.", "embed_tokens."}
                : std::vector<std::string>{"head.", "lm_head."}));
      }
    }
    model_->verify_loaded_weights();
    if (!embedding_mode_) {
      CHECK(lm_head_->is_weight_loaded()) << "Missing weight head.weight";
    }
  }
};
TORCH_MODULE(DeepseekV41ForCausalLM);
const bool deepseek_v41_graph_capabilities_registered = []() {
  MtpModelCapabilities capabilities;
  capabilities.enable_piecewise_graph = true;
  capabilities.supports_grouped_mtp_graph = true;
  capabilities.graph_dynamic_multi_block_tables = true;
  ModelRegistry::register_mtp_capabilities("deepseek_v41", capabilities);
  return true;
}();
REGISTER_CAUSAL_MODEL(deepseek_v41, DeepseekV41ForCausalLM);
}  // namespace xllm::mlu::model
