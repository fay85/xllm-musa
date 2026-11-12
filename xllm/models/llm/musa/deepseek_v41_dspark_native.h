/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/
#pragma once
#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model_context.h"
#include "core/layers/musa/deepseek_v41/draft_model.h"
#include "core/layers/musa/deepseek_v41/request_metadata.h"
#include "core/platform/musa_model_compute_stream.h"
#include "core/util/env_var.h"
#include "models/llm/musa/deepseek_v41_checkpoint.h"
#include "models/model_registry.h"
namespace xllm {

class DeepseekV41NativeDraftCausalLM final : public CausalLM {
 public:
  explicit DeepseekV41NativeDraftCausalLM(const ModelContext& context)
      : args_(context.get_model_args()),
        options_(context.get_tensor_options()),
        tp_group_(context.get_parallel_args().tp_group_),
        ep_group_(context.get_parallel_args().moe_ep_group_) {
    CHECK(util::get_bool_env("XLLM_DSV41_DSPARK_NATIVE", false));
    CHECK_EQ(args_.model_type(), "deepseek_v41_dspark");
    CHECK(context.get_speculative_runtime_config().is_draft_engine);
    CHECK_EQ(args_.n_layers(), 3);
    CHECK_EQ(args_.n_routed_experts(), args_.dspark_n_routed_experts());
    CHECK_EQ(args_.num_experts_per_tok(), args_.dspark_num_experts_per_tok());
    CHECK_EQ(args_.n_routed_experts() % ep_size(), 0);
    CHECK_EQ(args_.n_heads() / tp_size(), 8);
    CHECK_EQ(context.get_parallel_args().cp_size(), 1);
    CHECK_EQ(context.get_parallel_args().dp_size(), 1);
    CHECK_EQ(args_.hc_mult(), 4);
    CHECK_EQ(args_.markov_rank(), 256);
    CHECK(options_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(options_.dtype().toScalarType(), torch::kBFloat16);
    w4a8_path_ = required_path("XLLM_DSV41_MATE_W4A8_SO", false);
    mhc_path_ = required_path("XLLM_DSV41_MATE_MHC_DIR", true);
    sparse_path_ = required_path("XLLM_DSV41_MATE_SPARSE_SO", false);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    CHECK(loader != nullptr && !model_);
    torch::DeviceGuard guard(device());
    torch::NoGradGuard no_grad;
    DeepseekV41Checkpoint checkpoint(
        loader->get_state_dicts(),
        options_,
        tp_group_ == nullptr ? 0 : tp_group_->rank(),
        tp_size());
    std::vector<layer::DeepseekV41DecoderConfig> configs;
    std::vector<layer::DeepseekV41DecoderWeights> weights;
    configs.reserve(3);
    weights.reserve(3);
    for (int64_t index = 0; index < 3; ++index) {
      auto config = layer_config(index);
      weights.emplace_back(layer_weights(checkpoint, config));
      configs.emplace_back(std::move(config));
    }
    const std::string embedding = checkpoint.has("mtp.0.embed.weight")
                                      ? "mtp.0.embed.weight"
                                      : "embed.weight";
    const std::string head = checkpoint.has("mtp.2.head.weight")
                                 ? "mtp.2.head.weight"
                                 : "head.weight";
    const int64_t rotary = args_.qk_rope_head_dim();
    const int64_t length = args_.max_position_embeddings();
    const auto rope_options = options_.dtype(torch::kFloat32);
    const torch::Tensor exponents =
        torch::arange(0, rotary, 2, rope_options) / rotary;
    const torch::Tensor frequencies =
        torch::pow(args_.rope_theta(), -exponents);
    const torch::Tensor angles =
        torch::arange(length, rope_options).unsqueeze(1) *
        frequencies.unsqueeze(0);
    const torch::Tensor rope =
        torch::cat({torch::cos(angles), torch::sin(angles)}, 1).contiguous();
    model_ = std::make_unique<layer::DeepseekV41DraftModel>(
        std::move(configs),
        std::move(weights),
        checkpoint.copy(embedding, torch::kBFloat16, /*shard_dim=*/1),
        checkpoint.dense("mtp.0.main_proj"),
        checkpoint.copy("mtp.0.main_norm.weight", torch::kBFloat16),
        checkpoint.copy("mtp.2.norm.weight", torch::kBFloat16),
        checkpoint.copy(head, torch::kBFloat16, /*shard_dim=*/0),
        rope,
        tp_group_,
        ep_group_);
    markov_embed_ =
        checkpoint.copy("mtp.2.markov_head.embed.weight", torch::kBFloat16);
    markov_head_ =
        checkpoint.copy("mtp.2.markov_head.head.weight", torch::kBFloat16);
    CHECK_EQ(markov_embed_.sizes(),
             torch::IntArrayRef({args_.vocab_size(), args_.markov_rank()}));
    CHECK_EQ(markov_head_.sizes(), markov_embed_.sizes());
    if (args_.enable_confidence_head()) {
      confidence_weight_ = checkpoint.copy("mtp.2.confidence_head.proj.weight",
                                           torch::kBFloat16);
      CHECK_EQ(
          confidence_weight_.sizes(),
          torch::IntArrayRef({1, args_.hidden_size() + args_.markov_rank()}));
    }
    LOG(INFO)
        << "Native V4.1 DSpark checkpoint loaded:3 pure-SWA stages, experts="
        << args_.n_routed_experts() << ", topk=" << args_.num_experts_per_tok()
        << ", tp=" << tp_size() << ", ep=" << ep_size();
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) override {
    CHECK(model_);
    CHECK_EQ(params.execution_batch.num_reqs, 1);
    CHECK(!params.enable_graph);
    CHECK(params.is_dflash_proposal);
    const auto q =
        layer::parse_deepseek_v41_c1_length(params.attention.host.q_seq_lens);
    const auto kv =
        layer::parse_deepseek_v41_c1_length(params.attention.host.kv_seq_lens);
    CHECK(q.has_value() && kv.has_value());
    CHECK_EQ(q.value(), tokens.numel());
    CHECK_GE(kv.value(), q.value());
    // The runtime builds proposal inputs from target inputs. V4.1 target
    // tables are [SWA,C1,C2], while a projected draft input can carry just
    // [SWA]. This pure-SWA drafter consumes only that same leading table;
    // compressed target tables remain target-owned and are never modified.
    const int64_t table_count =
        static_cast<int64_t>(params.multi_block_tables.size());
    CHECK(table_count == 1 || table_count == 3)
        << "Native V4.1 draft expects [SWA] or target [SWA,C1,C2] tables";
    CHECK_EQ(params.multi_block_tables[0].dim(), 2);
    CHECK_EQ(params.multi_block_tables[0].size(0), 1);
    torch::DeviceGuard guard(device());
    std::optional<MusaModelComputeStream> stream;
    if (util::get_bool_env("XLLM_DSV41_MODEL_NONDEFAULT_STREAM", false)) {
      stream.emplace(device());
    }
    const torch::Tensor table =
        params.multi_block_tables[0].to(device(), /*non_blocking=*/true);
    torch::Tensor hidden = model_->forward(
        tokens, positions, swa_caches(caches), table, kv.value() - q.value());
    if (stream.has_value()) {
      stream->record_output(hidden);
      stream->finish();
    }
    return ModelOutput(hidden);
  }

  ModelOutput write_context_kv(const torch::Tensor& hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& slots,
                               std::vector<KVCache>& caches,
                               const ModelInputParams& /*params*/) override {
    CHECK(model_);
    torch::DeviceGuard guard(device());
    std::optional<MusaModelComputeStream> stream;
    if (util::get_bool_env("XLLM_DSV41_MODEL_NONDEFAULT_STREAM", false)) {
      stream.emplace(device());
    }
    torch::Tensor projected =
        model_->write_context_kv(hidden, positions, slots, swa_caches(caches));
    if (stream.has_value()) {
      stream->record_output(projected);
      stream->finish();
    }
    return ModelOutput(projected);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    CHECK(model_);
    const torch::Tensor rows =
        selected.defined() && selected.numel() > 0
            ? hidden.index_select(0, selected.to(torch::kInt64))
            : hidden;
    return model_->logits(rows);
  }
  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected,
                       torch::Tensor& output) override {
    output = selected.defined() && selected.numel() > 0
                 ? hidden.index_select(0, selected.to(torch::kInt64))
                 : hidden;
    return model_->logits(output);
  }
  torch::Tensor dspark_markov_bias(const torch::Tensor& previous) override {
    return torch::nn::functional::linear(
        torch::embedding(markov_embed_, previous.to(torch::kInt64)),
        markov_head_);
  }
  bool has_dspark_confidence_head() const override {
    return confidence_weight_.defined();
  }
  torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden,
      const torch::Tensor& previous) override {
    CHECK(confidence_weight_.defined());
    CHECK_EQ(hidden.dim(), 3);
    CHECK_EQ(previous.dim(), 2);
    const torch::Tensor features =
        torch::embedding(markov_embed_, previous.to(torch::kInt64));
    return torch::sigmoid(
        torch::nn::functional::linear(torch::cat({hidden, features}, -1),
                                      confidence_weight_)
            .squeeze(-1)
            .to(torch::kFloat32));
  }
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {
    LOG(FATAL) << "Native DSpark fixed expert ownership";
  }
  void update_expert_weight(int32_t) override {
    LOG(FATAL) << "Native DSpark fixed expert ownership";
  }

 private:
  std::string required_path(const char* name, bool directory) const {
    const char* value = std::getenv(name);
    CHECK(value != nullptr);
    CHECK(directory ? std::filesystem::is_directory(value)
                    : std::filesystem::is_regular_file(value));
    return value;
  }
  int64_t tp_size() const {
    return tp_group_ == nullptr ? 1 : tp_group_->world_size();
  }
  int64_t ep_size() const {
    return ep_group_ == nullptr ? 1 : ep_group_->world_size();
  }
  std::vector<torch::Tensor> swa_caches(
      const std::vector<KVCache>& caches) const {
    CHECK_EQ(caches.size(), 3);
    std::vector<torch::Tensor> result;
    result.reserve(3);
    for (const auto& cache : caches) {
      result.emplace_back(
          layer::deepseek_v41_cache_view(cache.get_swa_cache()));
    }
    return result;
  }
  layer::DeepseekV41DecoderConfig layer_config(int64_t index) const {
    layer::DeepseekV41DecoderConfig result;
    auto& attention = result.attention;
    attention.layer = index;
    attention.mode = layer::DeepseekV41AttentionMode::SWA;
    attention.ratio = 0;
    attention.source_layer = -1;
    attention.heads = args_.n_heads() / tp_size();
    attention.groups = args_.o_groups() / tp_size();
    attention.output_rank = args_.o_lora_rank();
    attention.window = args_.window_size();
    attention.epsilon = args_.rms_norm_eps();
    attention.scale = std::pow(static_cast<double>(args_.head_dim()), -.5);
    attention.sparse_library_path = sparse_path_;
    result.sinkhorn_iterations = args_.hc_sinkhorn_iters();
    result.hc_epsilon = args_.hc_eps();
    result.mhc_prenorm_directory = mhc_path_;
    auto& moe = result.moe;
    moe.expert_format = layer::DeepseekV41ExpertFormat::W4AFP8_MT;
    moe.w4a8_so_path = w4a8_path_;
    moe.dense = false;
    moe.first_expert = (ep_group_ == nullptr ? 0 : ep_group_->rank()) *
                       (args_.n_routed_experts() / ep_size());
    moe.topk = args_.num_experts_per_tok();
    moe.normalize = args_.norm_topk_prob();
    moe.scale = args_.routed_scaling_factor();
    moe.limit = args_.swiglu_limit();
    moe.image_token_id = args_.image_token_id();
    // First draft integration retains replicated shared expert and normal
    // TP/EP collectives at q>1; accepted target policies stay independent.
    return result;
  }
  layer::DeepseekV41DecoderWeights layer_weights(
      const DeepseekV41Checkpoint& checkpoint,
      const layer::DeepseekV41DecoderConfig& config) const {
    const int64_t index = config.attention.layer;
    const std::string prefix = "mtp." + std::to_string(index) + ".";
    const std::string attention = prefix + "attn.";
    layer::DeepseekV41DecoderWeights weights;
    auto& attn = weights.attention;
    attn.projection = {
        checkpoint.dense(attention + "wq_a"),
        checkpoint.copy(attention + "q_norm.weight", torch::kBFloat16),
        checkpoint.dense(attention + "wq_b", /*shard_dim=*/0),
        checkpoint.dense(attention + "wkv"),
        checkpoint.copy(attention + "kv_norm.weight", torch::kBFloat16),
        checkpoint.dense(attention + "wo_a", /*shard_dim=*/0),
        checkpoint.dense(attention + "wo_b", /*shard_dim=*/1)};
    const std::string sink_name = checkpoint.has(attention + "attn_sink")
                                      ? attention + "attn_sink"
                                      : attention + "attn_sink.weight";
    attn.sink = checkpoint.copy(sink_name, torch::kFloat32, /*shard_dim=*/0);
    const bool full =
        config.attention.mode == layer::DeepseekV41AttentionMode::FULL;
    if (full ||
        config.attention.mode == layer::DeepseekV41AttentionMode::REINDEX) {
      const std::string idx = attention + "indexer.";
      attn.indexer = {
          checkpoint.dense(idx + "wq_b"),
          checkpoint.copy(idx + "weights_proj.weight", torch::kBFloat16)};
      if (full) {
        attn.indexer.reserve(4);
        attn.indexer.emplace_back(
            checkpoint.copy(idx + "wk.weight", torch::kBFloat16));
        attn.indexer.emplace_back(
            checkpoint.copy(idx + "k_norm.weight", torch::kBFloat16));
        const bool pooled = config.attention.ratio == 2;
        attn.compressor_kv =
            checkpoint.copy(attention + "compressor.wkv.weight",
                            pooled ? torch::kFloat32 : torch::kBFloat16);
        attn.compressor_norm = checkpoint.copy(
            attention + "compressor.norm.weight", torch::kBFloat16);
        if (pooled) {
          attn.compressor_gate = checkpoint.copy(
              attention + "compressor.wgate.weight", torch::kFloat32);
        }
      }
    }
    weights.input_norm =
        checkpoint.copy(prefix + "attn_norm.weight", torch::kBFloat16);
    weights.post_attention_norm =
        checkpoint.copy(prefix + "ffn_norm.weight", torch::kBFloat16);
    int32_t hc_index = 0;
    for (const char* part : {"attn", "ffn"}) {
      for (const char* term : {"fn", "scale", "base"}) {
        weights.hyper_connection[hc_index++] = checkpoint.copy(
            prefix + "hc_" + part + "_" + term, torch::kFloat32);
      }
    }
    const std::string ffn = prefix + "ffn.";
    const std::string shared = config.moe.dense ? ffn : ffn + "shared_experts.";
    auto& moe = weights.moe;
    moe.shared = {checkpoint.dense(shared + "w1"),
                  checkpoint.dense(shared + "w3"),
                  checkpoint.dense(shared + "w2")};
    if (config.moe.shared_expert_tp_reduce) {
      const int64_t intermediate = moe.shared[0].size(0);
      CHECK_EQ(intermediate % tp_size(), 0);
      const int64_t local_intermediate = intermediate / tp_size();
      CHECK_EQ(local_intermediate % 32, 0)
          << "Shared-expert TP slices must align activation quant groups";
      const int64_t start = tp_group_->rank() * local_intermediate;
      moe.shared[0] =
          moe.shared[0].narrow(0, start, local_intermediate).contiguous();
      moe.shared[1] =
          moe.shared[1].narrow(0, start, local_intermediate).contiguous();
      moe.shared[2] =
          moe.shared[2].narrow(1, start, local_intermediate).contiguous();
    }
    if (!config.moe.dense) {
      moe.gate = checkpoint.copy(ffn + "gate.weight", torch::kBFloat16);
      moe.bias = checkpoint.copy(ffn + "gate.bias", torch::kFloat32);
      if (checkpoint.has(ffn + "gate.bias_vl")) {
        moe.vision_bias =
            checkpoint.copy(ffn + "gate.bias_vl", torch::kFloat32);
      }
      if (config.moe.expert_format ==
          layer::DeepseekV41ExpertFormat::W4AFP8_MT) {
        std::tie(moe.expert[0], moe.scale[0]) =
            checkpoint.int4_experts(ffn + "experts",
                                    {"w1", "w3"},
                                    config.moe.first_expert,
                                    args_.n_routed_experts() / ep_size());
        std::tie(moe.expert[1], moe.scale[1]) =
            checkpoint.int4_experts(ffn + "experts",
                                    {"w2"},
                                    config.moe.first_expert,
                                    args_.n_routed_experts() / ep_size());
      } else {
        constexpr std::array<const char*, 3> kExpertProjection = {
            "w1", "w3", "w2"};
        for (int32_t i = 0; i < 3; ++i) {
          std::tie(moe.expert[i], moe.scale[i]) =
              checkpoint.experts(ffn + "experts",
                                 kExpertProjection[i],
                                 config.moe.first_expert,
                                 args_.n_routed_experts() / ep_size());
        }
      }
    }
    return weights;
  }

  ModelArgs args_;
  torch::TensorOptions options_;
  ProcessGroup* tp_group_;
  ProcessGroup* ep_group_;
  std::string w4a8_path_;
  std::string mhc_path_;
  std::string sparse_path_;
  torch::Tensor markov_embed_;
  torch::Tensor markov_head_;
  torch::Tensor confidence_weight_;
  std::unique_ptr<layer::DeepseekV41DraftModel> model_;
};
inline const bool kDeepseekV41NativeDraftRegistered = []() {
  ModelRegistry::register_causallm_factory(
      "deepseek_v41_dspark", [](const ModelContext& context) {
        auto model = std::make_unique<DeepseekV41NativeDraftCausalLM>(context);
        model->eval();
        return model;
      });
  return true;
}();
}  // namespace xllm
