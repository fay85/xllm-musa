/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/
#pragma once
#include <glog/logging.h>
#include <torch/torch.h>

#include <array>
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
#include "core/layers/musa/deepseek_v41/draft_graph.h"
#include "core/layers/musa/deepseek_v41/draft_model.h"
#include "core/layers/musa/deepseek_v41/markov_head.h"
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
    const char* dense_prequant = std::getenv("XLLM_DSV41_DENSE_PREQUANT_DIR");
    if (dense_prequant != nullptr) {
      dense_prequant_path_ = dense_prequant;
      CHECK(!dense_prequant_path_.empty());
      CHECK(std::filesystem::is_regular_file(
          std::filesystem::path(dense_prequant_path_) / "export_summary.json"));
      LOG(INFO) << "Native V4.1 preserves FP8 dense weights/scales; AOT="
                << dense_prequant_path_;
    }

    mhc_path_ = required_path("XLLM_DSV41_MATE_MHC_DIR", true);
    sparse_path_ = required_path("XLLM_DSV41_MATE_SPARSE_SO", false);
    draft_custom_allreduce_ =
        util::get_bool_env("XLLM_DSV41_DRAFT_CUSTOM_AR", false);
    CHECK(!draft_custom_allreduce_ ||
          util::get_bool_env("XLLM_MUSA_ENABLE_TP_CUSTOM_AR", true));
    gather_bf16_bits_ =
        util::get_bool_env("XLLM_DSV41_GATHER_BF16_BITS", false);
    direct_rope_table_ =
        util::get_bool_env("XLLM_DSV41_DIRECT_ROPE_TABLE", false);
    CHECK(!gather_bf16_bits_ ||
          (tp_size() == 8 && musa_custom_allreduce_enabled(tp_group_)))
        << "BF16 bit gather requires the initialized TP8 custom communicator";
    static_draft_ = util::get_bool_env("XLLM_DSV41_STATIC_DRAFT", false);
    full_draft_graph_ =
        util::get_bool_env("XLLM_DSV41_FULL_DRAFT_GRAPH", false);
    CHECK(!full_draft_graph_ || (static_draft_ && draft_custom_allreduce_));
    markov_fp32_logits_ =
        util::get_bool_env("XLLM_DSV41_MARKOV_FP32_LOGITS", false);
    draft_small_commit_ =
        util::get_bool_env("XLLM_DSV41_DRAFT_SMALL_COMMIT", false);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    CHECK(loader != nullptr && !model_);
    torch::DeviceGuard guard(device());
    torch::NoGradGuard no_grad;
    initialize_draft_collective_policy();
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
        draft_expert_reduction_group(),
        gather_bf16_bits_);
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
    if (static_draft_) {
      draft_graph_ = std::make_unique<layer::DeepseekV41DraftGraph>();
    }
    LOG(INFO)
        << "Native V4.1 DSpark checkpoint loaded:3 pure-SWA stages, experts="
        << args_.n_routed_experts() << ", topk=" << args_.num_experts_per_tok()
        << ", tp=" << tp_size() << ", ep=" << ep_size();
    LOG(INFO) << "Native V4.1 DSpark Markov BF16 GEMM, FP32 logits="
              << markov_fp32_logits_;
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
    torch::Tensor hidden;
    if (static_draft_) {
      hidden =
          draft_graph_->run(tokens,
                            positions,
                            swa_caches(caches),
                            table,
                            kv.value() - q.value(),
                            full_draft_graph_,
                            [this](const layer::DeepseekV41DraftInputs& input) {
                              return model_->forward(input.tokens,
                                                     input.positions,
                                                     input.caches,
                                                     input.table,
                                                     /*previous_context=*/0,
                                                     input.previous_context);
                            });
    } else {
      hidden = model_->forward(
          tokens, positions, swa_caches(caches), table, kv.value() - q.value());
    }
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
    return layer::deepseek_v41_markov_bias(
        previous, markov_embed_, markov_head_, markov_fp32_logits_);
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
  ProcessGroup* draft_expert_reduction_group() const {
    if (!draft_custom_allreduce_) {
      return ep_group_;
    }
    // This adapter requires CP1/DP1, with TP8 and EP8 in custom mode.
    // Both groups span the same eight ranks in the same order. The existing
    // custom-reduce entry point belongs to tp_group; expert ownership still
    // uses ep_group_, and the replicated shared expert joins after this sum.
    CHECK_EQ(tp_size(), 8);
    CHECK_EQ(ep_size(), tp_size());
    CHECK_EQ(tp_group_->rank(), ep_group_->rank());
    return tp_group_;
  }

  void initialize_draft_collective_policy() const {
    const int32_t policy = static_cast<int32_t>(draft_custom_allreduce_) |
                           (static_cast<int32_t>(static_draft_) << 1) |
                           (static_cast<int32_t>(full_draft_graph_) << 2) |
                           (static_cast<int32_t>(gather_bf16_bits_) << 3) |
                           (static_cast<int32_t>(direct_rope_table_) << 4) |
                           (static_cast<int32_t>(draft_small_commit_) << 5);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local =
          torch::tensor({policy}, options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK_EQ(peers.numel(), tp_size());
      const int32_t* policies = peers.data_ptr<int32_t>();
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], policy)
            << "Native DSpark draft collective policy differs across TP ranks";
      }
    }
    if (draft_custom_allreduce_) {
      CHECK_EQ(tp_size(), 8);
      CHECK_EQ(ep_size(), tp_size());
      CHECK_EQ(tp_group_->rank(), ep_group_->rank());
    }
    LOG(INFO) << "Native V4.1 draft TP/EP custom allreduce="
              << draft_custom_allreduce_ << ", static_inputs=" << static_draft_
              << ", full_draft_graph=" << full_draft_graph_
              << ", small_context_commit=" << draft_small_commit_
              << "; shared expert remains replicated";
  }

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
    attention.dense_prequant_directory = dense_prequant_path_;
    attention.direct_rope_table = direct_rope_table_;
    attention.draft_small_commit = draft_small_commit_;
    attention.sparse_library_path = sparse_path_;
    result.sinkhorn_iterations = args_.hc_sinkhorn_iters();
    result.hc_epsilon = args_.hc_eps();
    result.mhc_prenorm_directory = mhc_path_;
    auto& moe = result.moe;
    moe.expert_format = layer::DeepseekV41ExpertFormat::W4AFP8_MT;
    moe.w4a8_so_path = w4a8_path_;
    moe.dense_prequant_directory = dense_prequant_path_;
    moe.w4a8_static_activation =
        util::get_bool_env("XLLM_DSV41_W4A8_STATIC_ACTIVATION", false);
    moe.w4a8_valid_prefix_swiglu =
        util::get_bool_env("XLLM_DSV41_W4A8_VALID_PREFIX_SWIGLU", false);
    moe.dense = false;
    moe.first_expert = (ep_group_ == nullptr ? 0 : ep_group_->rank()) *
                       (args_.n_routed_experts() / ep_size());
    moe.topk = args_.num_experts_per_tok();
    moe.normalize = args_.norm_topk_prob();
    moe.scale = args_.routed_scaling_factor();
    moe.limit = args_.swiglu_limit();
    moe.image_token_id = args_.image_token_id();
    // Reuse the already checked 1..6-row BF16 attention / FP32 expert
    // reductions. Keep the replicated shared expert outside the EP sum.
    result.decode_tp_custom_allreduce = draft_custom_allreduce_;
    result.verify_custom_allreduce = draft_custom_allreduce_;
    moe.decode_ep_custom_allreduce = draft_custom_allreduce_;
    moe.verify_custom_allreduce = draft_custom_allreduce_;
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
    if (!dense_prequant_path_.empty()) {
      attn.projection_fp8_scales.resize(7);
    }
    const auto projection = [&](const std::string& name,
                                size_t index,
                                std::optional<int64_t> dim = std::nullopt) {
      if (dense_prequant_path_.empty()) {
        return checkpoint.dense(name, dim);
      }
      auto [weight, scale] = checkpoint.dense_fp8(name, dim);
      attn.projection_fp8_scales[index] = std::move(scale);
      return weight;
    };
    attn.projection = {
        projection(attention + "wq_a", 0),
        checkpoint.copy(attention + "q_norm.weight", torch::kBFloat16),
        projection(attention + "wq_b", 2, /*shard_dim=*/0),
        projection(attention + "wkv", 3),
        checkpoint.copy(attention + "kv_norm.weight", torch::kBFloat16),
        checkpoint.dense(attention + "wo_a", /*shard_dim=*/0),
        projection(attention + "wo_b", 6, /*shard_dim=*/1)};
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
    if (dense_prequant_path_.empty()) {
      moe.shared = {checkpoint.dense(shared + "w1"),
                    checkpoint.dense(shared + "w3"),
                    checkpoint.dense(shared + "w2")};
    } else {
      moe.shared.reserve(3);
      moe.shared_fp8_scales.reserve(3);
      const std::array<const char*, 3> names = {"w1", "w3", "w2"};
      for (size_t projection = 0; projection < names.size(); ++projection) {
        const std::optional<int64_t> dim =
            config.moe.shared_expert_tp_reduce
                ? std::optional<int64_t>{projection == 2 ? 1 : 0}
                : std::nullopt;
        auto [weight, scale] =
            checkpoint.dense_fp8(shared + names[projection], dim);
        moe.shared.emplace_back(std::move(weight));
        moe.shared_fp8_scales.emplace_back(std::move(scale));
      }
    }
    if (config.moe.shared_expert_tp_reduce && dense_prequant_path_.empty()) {
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
        if (config.moe.w4a8_static_activation) {
          moe.expert_input_scale = {
              checkpoint.int4_input_scale(ffn + "experts",
                                          {"w1", "w3"},
                                          config.moe.first_expert,
                                          args_.n_routed_experts() / ep_size()),
              checkpoint.int4_input_scale(
                  ffn + "experts",
                  {"w2"},
                  config.moe.first_expert,
                  args_.n_routed_experts() / ep_size())};
        }
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
  std::string dense_prequant_path_;
  std::string mhc_path_;
  std::string sparse_path_;
  bool draft_custom_allreduce_ = false;
  bool static_draft_ = false;
  bool full_draft_graph_ = false;
  bool markov_fp32_logits_ = false;
  bool gather_bf16_bits_ = false;
  bool direct_rope_table_ = false;
  bool draft_small_commit_ = false;
  torch::Tensor markov_embed_;
  torch::Tensor markov_head_;
  torch::Tensor confidence_weight_;
  std::unique_ptr<layer::DeepseekV41DraftModel> model_;
  std::unique_ptr<layer::DeepseekV41DraftGraph> draft_graph_;
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
