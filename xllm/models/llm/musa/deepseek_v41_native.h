/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model_context.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/musa/deepseek_v41/engram_embedding.h"
#include "core/layers/musa/deepseek_v41/engram_hash.h"
#include "core/layers/musa/deepseek_v41/request_metadata.h"
#include "core/layers/musa/deepseek_v41/text_model.h"
#include "core/util/env_var.h"
#include "models/llm/musa/deepseek_v41_checkpoint.h"
#include "models/model_registry.h"

namespace xllm {

// Original/W4AFP8-mt 1017 text target, eager C1, with the same C++
// scheduler/physical KV cache/Mooncake contract as the Python route. This model
// owns its loader, embedding, decoder iteration, Engram state and LM head. No
// Python model callback is used. Offline tokenizer-derived metadata is a small
// sidecar.
class DeepseekV41NativeCausalLM final : public CausalLM {
 public:
  explicit DeepseekV41NativeCausalLM(const ModelContext& context)
      : args_(context.get_model_args()),
        options_(context.get_tensor_options()),
        tp_group_(context.get_parallel_args().tp_group_),
        ep_group_(context.get_parallel_args().moe_ep_group_) {
    CHECK_EQ(args_.model_type(), "deepseek_v41");
    CHECK(options_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(options_.dtype().toScalarType(), torch::kBFloat16);
    CHECK_EQ(context.get_parallel_args().cp_size(), 1);
    CHECK_EQ(context.get_parallel_args().dp_size(), 1);
    CHECK(!ExecutionConfig::get_instance().enable_graph());
    CHECK_EQ(SpeculativeConfig::get_instance().num_speculative_tokens(), 0);
    CHECK(!context.get_speculative_runtime_config().is_draft_engine);
    CHECK_EQ(args_.hc_mult(), 4);
    CHECK_EQ(args_.n_shared_experts(), 1);
    CHECK_EQ(args_.scoring_func(), "sqrtsoftplus");
    CHECK_GT(args_.n_layers(), 0);
    CHECK_GE(args_.compress_ratios().size(),
             static_cast<size_t>(args_.n_layers()));
    CHECK_GT(args_.head_dim(), 0);
    CHECK_EQ(args_.n_heads() % tp_size(), 0);
    CHECK_EQ(args_.o_groups() % tp_size(), 0);
    CHECK_EQ(args_.n_routed_experts() % ep_size(), 0);
    CHECK_EQ(ep_size(), context.get_parallel_args().ep_size());
    CHECK_EQ(tp_size(), context.get_parallel_args().world_size());
    const char* cache_mode = std::getenv("XLLM_DSV41_MUSA_ATTENTION_QUANT");
    CHECK(cache_mode != nullptr &&
          std::string_view(cache_mode) == "original_qdq")
        << "Native V4.1 requires the original-QDQ BF16 index cache carrier";
    const char* score_directory = std::getenv("XLLM_DSV41_INDEXER_AOT_DIR");
    if (score_directory != nullptr) {
      indexer_aot_path_ = score_directory;
      CHECK(std::filesystem::is_regular_file(
          std::filesystem::path(indexer_aot_path_) / "manifest.json"));
      CHECK_EQ(args_.index_n_heads(), 32);
      CHECK_EQ(args_.index_head_dim(), 128);
    }
    const char* engram_directory = std::getenv("XLLM_DSV41_ENGRAM_AOT_DIR");
    const char* hash_directory = std::getenv("XLLM_DSV41_ENGRAM_HASH_AOT_DIR");
    if (hash_directory != nullptr) {
      engram_hash_aot_path_ = hash_directory;
      CHECK(std::filesystem::is_regular_file(
          std::filesystem::path(engram_hash_aot_path_) / "manifest.json"));
      CHECK_EQ(args_.engram_layer_ids().size(), 2);
      CHECK_EQ(args_.engram_max_ngram_size(), 4);
      CHECK_EQ(args_.engram_n_heads(), 8);
    }
    if (engram_directory != nullptr) {
      engram_aot_path_ = engram_directory;
      CHECK(std::filesystem::is_regular_file(
          std::filesystem::path(engram_aot_path_) / "manifest.json"));
      CHECK_EQ(args_.engram_head_dim(), 256);
    }
    const char* sparse_library = std::getenv("XLLM_DSV41_MATE_SPARSE_SO");
    if (sparse_library != nullptr) {
      sparse_path_ = sparse_library;
      CHECK(std::filesystem::is_regular_file(sparse_path_));
      CHECK_EQ(args_.n_heads() / tp_size(), 8);
      CHECK_EQ(args_.head_dim(), 512);
    }
    const char* mhc_directory = std::getenv("XLLM_DSV41_MATE_MHC_DIR");
    if (mhc_directory != nullptr) {
      mhc_path_ = mhc_directory;
      for (const char* filename :
           {"libdeepseek_v41_mhc_m32.so", "libdeepseek_v41_mhc_m64.so"}) {
        CHECK(std::filesystem::is_regular_file(
            std::filesystem::path(mhc_path_) / filename));
      }
    }
    const char* expert_format = std::getenv("XLLM_DSV41_EXPERT_FORMAT");
    const std::string_view format =
        expert_format == nullptr ? "original_fp4" : expert_format;
    CHECK(format == "original_fp4" || format == "w4afp8_mt");
    if (format == "w4afp8_mt") {
      expert_format_ = layer::DeepseekV41ExpertFormat::W4AFP8_MT;
      const char* gemm = std::getenv("XLLM_DSV41_MATE_W4A8_SO");
      CHECK(gemm != nullptr && std::filesystem::is_regular_file(gemm))
          << "W4AFP8-mt requires the frozen corrected MATE S4FP8 adapter";
      w4a8_path_ = gemm;
    } else {
      const char* bmm = std::getenv("XLLM_DSV41_MATE_BMM_SO");
      CHECK(bmm != nullptr && std::filesystem::is_regular_file(bmm))
          << "Original FP4 requires the frozen MATE BF16 BMM adapter";
      bmm_path_ = bmm;
    }
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    CHECK(loader != nullptr);
    CHECK(!model_);
    torch::DeviceGuard guard(device());
    torch::NoGradGuard no_grad;
    initialize_decode_ep_policy();
    DeepseekV41Checkpoint checkpoint(
        loader->get_state_dicts(),
        options_,
        tp_group_ == nullptr ? 0 : tp_group_->rank(),
        tp_size());
    const bool has_int4_experts =
        checkpoint.has("layers.0.ffn.experts.0.w1.weight_scale_inv");
    CHECK_EQ(has_int4_experts,
             expert_format_ == layer::DeepseekV41ExpertFormat::W4AFP8_MT)
        << "Checkpoint expert encoding does not match XLLM_DSV41_EXPERT_FORMAT";
    std::vector<layer::DeepseekV41DecoderConfig> configs;
    std::vector<layer::DeepseekV41DecoderWeights> weights;
    configs.reserve(args_.n_layers());
    weights.reserve(args_.n_layers());
    engram_tables_.resize(args_.n_layers());
    engram_indices_.resize(args_.n_layers(), -1);
    prepare_engram_assets(loader->model_weights_path(), checkpoint);
    int64_t source = -1;
    for (int64_t index = 0; index < args_.n_layers(); ++index) {
      if (contains(args_.kv_source_layer_ids(), index)) {
        source = index;
      }
      auto config = layer_config(index, source);
      weights.emplace_back(layer_weights(checkpoint, config));
      configs.emplace_back(std::move(config));
      LOG(INFO) << "Native V4.1 loaded decoder layer " << index;
    }
    std::string head_name;
    for (const char* name : {"lm_head.weight",
                             "model.lm_head.weight",
                             "model.head.weight",
                             "head.weight"}) {
      if (checkpoint.has(name)) {
        head_name = name;
        break;
      }
    }
    CHECK(!head_name.empty());
    auto [plain_rope, compressed_rope] = rope_tables();
    model_ = std::make_unique<layer::DeepseekV41TextModel>(
        std::move(configs),
        std::move(weights),
        checkpoint.copy("embed.weight", torch::kBFloat16, /*shard_dim=*/1),
        checkpoint.copy("norm.weight", torch::kBFloat16),
        checkpoint.copy(head_name, torch::kBFloat16, /*shard_dim=*/0),
        std::move(plain_rope),
        std::move(compressed_rope),
        tp_group_,
        expert_reduction_group());
    LOG(INFO) << "Native V4.1 weights ready, expert_format="
              << (has_int4_experts ? "w4afp8_mt" : "original_fp4")
              << ", layers=" << args_.n_layers() << ", tp=" << tp_size()
              << ", ep=" << ep_size()
              << ", engram_layers=" << args_.engram_layer_ids().size();
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    CHECK(model_);
    CHECK_EQ(params.execution_batch.num_reqs, 1) << "Native V4.1 currently C1";
    CHECK(!params.enable_graph && !params.is_spec_verify);
    CHECK(!params.embedding.input_embedding.defined())
        << "Native V4.1 text only";
    const auto q_length =
        layer::parse_deepseek_v41_c1_length(params.attention.host.q_seq_lens);
    const auto kv_length =
        layer::parse_deepseek_v41_c1_length(params.attention.host.kv_seq_lens);
    CHECK(q_length.has_value() && kv_length.has_value())
        << "Invalid C1 query/context host length metadata";
    CHECK_EQ(q_length.value(), tokens.numel());
    const int64_t context_length = kv_length.value();
    const layer::DeepseekV41ProfileRange range(
        "V41/step", /*layer=*/-1, tokens.numel(), context_length);
    CHECK_GE(context_length, tokens.numel());
    CHECK_GT(tokens.numel(), 0);
    CHECK_EQ(kv_caches.size(), static_cast<size_t>(args_.n_layers()));
    const int64_t table_count = 1 + static_cast<int64_t>(ratio1_group_ >= 0) +
                                static_cast<int64_t>(ratio2_group_ >= 0);
    CHECK_EQ(params.multi_block_tables.size(), table_count);
    if (!params.synchronize_all_layers()) {
      return ModelOutput();
    }
    torch::DeviceGuard guard(device());
    std::optional<kernel::musa::TvmffiPreparationSyncGuard> ffi_sync;
    if (tokens.numel() > 1 &&
        util::get_bool_env("XLLM_DSV41_PREFILL_FFI_SYNC", false)) {
      ffi_sync.emplace();
      LOG_FIRST_N(INFO, 1)
          << "Native V4.1 prefill uses explicit FFI preparation sync";
    }

    CHECK_EQ(tokens.device(), device());
    CHECK_EQ(positions.device(), device());
    // ModelInputParams::to intentionally keeps these page-ID tables on CPU
    // for the DSA metadata builder. This model consumes them in GPU indexing
    // in both Engram and Attention, so upload each table once per request step.
    const auto device_page_table = [&](int64_t group) {
      const auto& host_table = params.multi_block_tables[group];
      CHECK(host_table.defined());
      CHECK_EQ(host_table.dim(), 2);
      CHECK_EQ(host_table.size(0), 1);
      return host_table.to(device(), /*non_blocking=*/true);
    };
    const torch::Tensor swa_table = device_page_table(0);
    const torch::Tensor ratio1_table =
        ratio1_group_ < 0 ? torch::Tensor{} : device_page_table(ratio1_group_);
    const torch::Tensor ratio2_table =
        ratio2_group_ < 0 ? torch::Tensor{} : device_page_table(ratio2_group_);
    std::vector<layer::DeepseekV41AttentionCache> caches;
    caches.reserve(kv_caches.size());
    for (const KVCache& cache : kv_caches) {
      caches.emplace_back(layer::DeepseekV41AttentionCache{
          layer::deepseek_v41_cache_view(cache.get_swa_cache()),
          layer::deepseek_v41_cache_view(cache.get_k_cache()),
          layer::deepseek_v41_cache_view(cache.get_index_cache()),
          defined(cache.get_compress_kv_state()),
          defined(cache.get_compress_score_state())});
    }
    std::vector<std::optional<torch::Tensor>> engram_rows(kv_caches.size());
    std::optional<torch::Tensor> mask;
    if (engram_hash_) {
      const layer::DeepseekV41ProfileRange engram_range(
          "V41/Engram/hash_lookup");
      CHECK_LE(tokens.numel(), 512)
          << "Native Engram gate supports at most 512 rows";
      const torch::Tensor& swa = caches[0].swa;
      CHECK_EQ(swa.dim(), 3);
      if (!history_swa_.defined() ||
          history_swa_.data_ptr() != swa.data_ptr() ||
          history_swa_.sizes() != swa.sizes()) {
        engram_hash_->bind_history(torch::zeros({swa.size(0) * swa.size(1)},
                                                options_.dtype(torch::kInt32)));
        history_swa_ = swa;
      }
      const auto& history = params.embedding.engram_history;
      if (!history.empty()) {
        CHECK_EQ(history.size() % 2, 0);
        engram_hash_->restore_history(torch::tensor(history, torch::kInt32)
                                          .reshape({-1, 2})
                                          .to(device()));
      }
      const auto& lookback = params.embedding.engram_lookback_token_ids;
      std::optional<torch::Tensor> lookback_tensor;
      if (!lookback.empty()) {
        lookback_tensor = torch::tensor(lookback, torch::kInt32)
                              .reshape({1, -1})
                              .to(device());
      }
      auto [hashes, gate_mask] =
          engram_hash_->forward(tokens,
                                positions,
                                swa_table,
                                swa.size(1),
                                context_length - tokens.numel(),
                                lookback_tensor);
      mask = std::move(gate_mask);
      for (int64_t layer : args_.engram_layer_ids()) {
        engram_rows[layer] =
            engram_tables_[layer]
                ->lookup(hashes.select(1, engram_indices_[layer]))
                .flatten(1);
      }
    }
    return ModelOutput(model_->forward(tokens,
                                       positions,
                                       caches,
                                       swa_table,
                                       ratio1_table,
                                       ratio2_table,
                                       context_length,
                                       engram_rows,
                                       mask));
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    CHECK(model_);
    return model_->logits(hidden, defined(selected));
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected,
                       torch::Tensor& out_hidden) override {
    CHECK(model_);
    out_hidden = selected.defined() && selected.numel() > 0
                     ? hidden.index_select(0, selected.to(torch::kInt64))
                     : hidden;
    return model_->logits(out_hidden, std::nullopt);
  }

  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }

  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {
    LOG(FATAL)
        << "Native V4.1 uses fixed expert ownership; EPLB is unsupported";
  }

  void update_expert_weight(int32_t /*layer_id*/) override {
    LOG(FATAL)
        << "Native V4.1 uses fixed expert ownership; EPLB is unsupported";
  }

 private:
  void initialize_decode_ep_policy() {
    const bool enabled =
        util::get_bool_env("XLLM_DSV41_DECODE_EP_CUSTOM_AR", false);
    // Every TP rank votes, including disabled ranks. A mismatched selector
    // must fail before mixing MCCL with custom collectives during forward.
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local = torch::tensor({static_cast<int32_t>(enabled)},
                                                options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK_EQ(peers.numel(), tp_size());
      const int32_t* policies = peers.data_ptr<int32_t>();
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], static_cast<int32_t>(enabled))
            << "Native V4.1 decode EP policy differs across TP ranks";
      }
    }
    CHECK(!enabled ||
          (tp_size() > 1 && ep_size() == tp_size() &&
           util::get_bool_env("XLLM_MUSA_ENABLE_TP_CUSTOM_AR", true)))
        << "Decode EP custom requires the enabled full-world TP communicator";
    decode_ep_custom_allreduce_ = enabled;
  }

  ProcessGroup* expert_reduction_group() const {
    // This C1/DP1/CP1 model already requires TP to span the complete world.
    // When EP also spans that world, both reductions have identical members
    // and rank order. Reuse one ordered communicator instead of interleaving
    // TP and EP kernels on two MCCL streams across successive prefill chunks.
    // Expert ownership and FP32 routed/shared accumulation remain unchanged.
    if (tp_size() > 1 && ep_size() == tp_size()) {
      CHECK_EQ(tp_group_->rank(), ep_group_->rank());
      LOG(INFO) << "Native V4.1 full-world EP reduction shares TP communicator";
      return tp_group_;
    }
    return ep_group_;
  }

  int64_t tp_size() const {
    return tp_group_ == nullptr ? 1 : tp_group_->world_size();
  }
  int64_t ep_size() const {
    return ep_group_ == nullptr ? 1 : ep_group_->world_size();
  }

  template <typename T>
  static bool contains(const std::vector<T>& values, int64_t index) {
    return std::find(values.begin(), values.end(), index) != values.end();
  }

  static std::optional<torch::Tensor> defined(torch::Tensor value) {
    return value.defined() ? std::optional<torch::Tensor>(std::move(value))
                           : std::nullopt;
  }

  layer::DeepseekV41DecoderConfig layer_config(int64_t index, int64_t source) {
    layer::DeepseekV41DecoderConfig config;
    auto& attention = config.attention;
    attention.layer = index;
    const int64_t ratio = args_.compress_ratios()[index];
    if (source >= 0 && ratio > 0) {
      CHECK(ratio == 1 || ratio == 2);
      CHECK_EQ(args_.compress_ratios()[source], ratio);
      attention.ratio = ratio;
      attention.source_layer = source;
      if (source == index) {
        attention.mode = layer::DeepseekV41AttentionMode::FULL;
      } else if (contains(args_.index_source_layer_ids(), index)) {
        attention.mode = layer::DeepseekV41AttentionMode::REINDEX;
      } else {
        attention.mode = layer::DeepseekV41AttentionMode::REUSE;
      }
    }
    attention.heads = args_.n_heads() / tp_size();
    attention.groups = args_.o_groups() / tp_size();
    attention.output_rank = args_.o_lora_rank();
    attention.window = args_.window_size();
    attention.index_heads = args_.index_n_heads();
    attention.index_dim = args_.index_head_dim();
    attention.index_topk = args_.index_topk();
    attention.candidate_source = args_.candidate_source_layer_id();
    attention.candidate_blocks = args_.candidate_topk_blocks();
    attention.candidate_block_size = args_.candidate_block_size();
    attention.epsilon = args_.rms_norm_eps();
    attention.scale = std::pow(static_cast<double>(args_.head_dim()), -0.5);
    attention.decode_fused_swa_read =
        util::get_bool_env("XLLM_DSV41_DECODE_FUSED_SWA_READ", false);
    attention.decode_cache_read_reuse =
        util::get_bool_env("XLLM_DSV41_DECODE_CACHE_READ_REUSE", false);
    config.sinkhorn_iterations = args_.hc_sinkhorn_iters();
    config.hc_epsilon = args_.hc_eps();
    config.mhc_prenorm_directory = mhc_path_;
    // MusaCustomAllReduce checks this policy across all TP ranks during
    // communicator construction. Keep one consistent selector rather than
    // adding an independent model flag that could disagree between ranks.
    config.decode_tp_custom_allreduce =
        util::get_bool_env("XLLM_MUSA_ENABLE_TP_CUSTOM_AR", true);
    config.attention.sparse_library_path = sparse_path_;
    config.attention.indexer_score_directory = indexer_aot_path_;
    auto& moe = config.moe;
    moe.decode_ep_custom_allreduce = decode_ep_custom_allreduce_;
    moe.expert_format = expert_format_;
    moe.w4a8_so_path = w4a8_path_;
    moe.dense =
        index < args_.first_k_dense_replace() ||
        (args_.moe_layer_freq() > 1 && index % args_.moe_layer_freq() != 0);
    moe.first_expert = (ep_group_ == nullptr ? 0 : ep_group_->rank()) *
                       (args_.n_routed_experts() / ep_size());
    moe.topk = args_.num_experts_per_tok();
    moe.normalize = args_.norm_topk_prob();
    moe.scale = args_.routed_scaling_factor();
    moe.limit = args_.swiglu_limit();
    moe.image_token_id = args_.image_token_id();
    moe.bmm_so_path = bmm_path_;
    return config;
  }

  layer::DeepseekV41DecoderWeights layer_weights(
      const DeepseekV41Checkpoint& checkpoint,
      const layer::DeepseekV41DecoderConfig& config) const {
    const int64_t index = config.attention.layer;
    const std::string prefix = "layers." + std::to_string(index) + ".";
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
    if (engram_indices_[index] >= 0) {
      const std::string engram = prefix + "engram.";
      weights.engram = {checkpoint.dense(engram + "wkv"),
                        checkpoint.copy(engram + "q_weight", torch::kBFloat16),
                        checkpoint.copy(engram + "k_weight", torch::kBFloat16)};
    }
    return weights;
  }

  void prepare_engram_assets(const std::string& model_path,
                             const DeepseekV41Checkpoint& checkpoint) {
    const auto& sources = args_.kv_source_layer_ids();
    bool ratio1 = false;
    bool ratio2 = false;
    for (int64_t index : sources) {
      CHECK_GE(index, 0);
      CHECK_LT(index, args_.n_layers());
      const int64_t ratio = args_.compress_ratios()[index];
      CHECK(ratio == 1 || ratio == 2);
      ratio1 |= ratio == 1;
      ratio2 |= ratio == 2;
    }
    ratio2_group_ = ratio2 ? 1 : -1;
    ratio1_group_ = ratio1 ? (ratio2 ? 2 : 1) : -1;
    const auto& layers = args_.engram_layer_ids();
    if (layers.empty()) {
      return;
    }
    const std::filesystem::path file =
        std::filesystem::path(model_path) / "engram_hash_state.native";
    CHECK(std::filesystem::is_regular_file(file))
        << "Generate the native Engram metadata beside the read-only model "
           "view: "
        << file;
    const auto state = StateDictFromSafeTensor::load(file.string());
    CHECK(state != nullptr);
    const torch::Tensor token_map = state->get_tensor("token_map");
    const torch::Tensor primes = state->get_tensor("primes");
    const torch::Tensor layer_ids = state->get_tensor("layer_ids");
    CHECK_EQ(token_map.numel(), args_.vocab_size());
    CHECK_EQ(layer_ids.numel(), layers.size());
    CHECK_EQ(primes.dim(), 3);
    CHECK_EQ(primes.size(0), layers.size());
    CHECK_EQ(primes.size(1), args_.engram_max_ngram_size() - 1);
    CHECK_EQ(primes.size(2), args_.engram_n_heads());
    CHECK_EQ(token_map.max().item<int64_t>() + 1,
             args_.engram_compressed_vocab_size());
    CHECK_GE(args_.engram_pad_token_id(), 0);
    CHECK_LT(args_.engram_pad_token_id(), token_map.numel());
    engram_hash_ = std::make_unique<layer::DeepseekV41EngramHash>(
        token_map.to(device()),
        state->get_tensor("multipliers").to(device()),
        primes.to(device()),
        state->get_tensor("offsets").to(device()),
        token_map[args_.engram_pad_token_id()].item<int64_t>(),
        engram_hash_aot_path_);
    const int64_t columns = primes.size(1) * primes.size(2);
    CHECK_EQ(columns % tp_size(), 0);
    const int64_t local_heads = columns / tp_size();
    const int64_t head_start =
        (tp_group_ == nullptr ? 0 : tp_group_->rank()) * local_heads;
    for (size_t i = 0; i < layers.size(); ++i) {
      const int64_t index = layers[i];
      CHECK_GE(index, 0);
      CHECK_LT(index, args_.n_layers());
      CHECK_EQ(layer_ids[i].item<int64_t>(), index);
      engram_indices_[index] = static_cast<int64_t>(i);
      const torch::Tensor buckets = primes[static_cast<int64_t>(i)].flatten();
      const int64_t first =
          buckets.narrow(0, 0, head_start).sum().item<int64_t>();
      const int64_t count =
          buckets.narrow(0, head_start, local_heads).sum().item<int64_t>();
      const std::string prefix =
          "layers." + std::to_string(index) + ".engram.embed.";
      const torch::Tensor weight = checkpoint.tensor(prefix + "weight");
      const torch::Tensor scale = checkpoint.tensor(prefix + "scale");
      CHECK_EQ(weight.element_size(), 1);
      CHECK_EQ(scale.element_size(), 1);
      CHECK_EQ(weight.size(1), args_.engram_head_dim());
      CHECK_LE(first + count, weight.size(0));
      CHECK_LE(first + count, scale.size(0));
      engram_tables_[index] =
          std::make_unique<layer::DeepseekV41EngramEmbedding>(
              weight.narrow(0, first, count).view(torch::kUInt8).to(device()),
              scale.narrow(0, first, count).view(torch::kUInt8).to(device()),
              first,
              head_start,
              local_heads,
              tp_group_,
              engram_aot_path_);
    }
  }

  std::pair<torch::Tensor, torch::Tensor> rope_tables() const {
    const int64_t dim = args_.qk_rope_head_dim();
    CHECK_GT(dim, 0);
    CHECK_EQ(dim % 2, 0);
    CHECK_GT(args_.max_position_embeddings(), 0);
    const auto cpu =
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
    const torch::Tensor positions =
        torch::arange(args_.max_position_embeddings(), cpu);
    const torch::Tensor powers = torch::arange(0, dim, 2, cpu) / dim;
    const torch::Tensor plain = torch::outer(
        positions, torch::pow(args_.rope_theta(), powers).reciprocal());
    const torch::Tensor frequency =
        layer::rotary::apply_deepseek_yarn_rope_scaling(
            args_.rope_scaling_factor(),
            /*extrapolation_factor=*/1,
            static_cast<int64_t>(args_.beta_fast()),
            static_cast<int64_t>(args_.beta_slow()),
            dim,
            args_.compress_rope_theta(),
            args_.rope_scaling_original_max_position_embeddings());
    const torch::Tensor compressed = torch::outer(positions, frequency);
    return {torch::cat({plain.cos(), plain.sin()}, -1).to(device()),
            torch::cat({compressed.cos(), compressed.sin()}, -1).to(device())};
  }

  ModelArgs args_;
  torch::TensorOptions options_;
  // Framework-owned groups live through the model worker.
  ProcessGroup* tp_group_;
  ProcessGroup* ep_group_;
  bool decode_ep_custom_allreduce_ = false;
  std::string bmm_path_;
  std::string w4a8_path_;
  std::string mhc_path_;
  std::string sparse_path_;
  std::string indexer_aot_path_;
  std::string engram_aot_path_;
  std::string engram_hash_aot_path_;
  layer::DeepseekV41ExpertFormat expert_format_ =
      layer::DeepseekV41ExpertFormat::ORIGINAL_FP4;
  int64_t ratio1_group_ = -1;
  int64_t ratio2_group_ = -1;
  std::vector<int64_t> engram_indices_;
  std::vector<std::unique_ptr<layer::DeepseekV41EngramEmbedding>>
      engram_tables_;
  std::unique_ptr<layer::DeepseekV41EngramHash> engram_hash_;
  torch::Tensor history_swa_;
  std::unique_ptr<layer::DeepseekV41TextModel> model_;
};

inline const bool kDeepseekV41NativeRegistered = []() {
  ModelRegistry::register_causallm_factory(
      "deepseek_v41", [](const ModelContext& context) {
        auto model = std::make_unique<DeepseekV41NativeCausalLM>(context);
        model->eval();
        return model;
      });
  return true;
}();

}  // namespace xllm
