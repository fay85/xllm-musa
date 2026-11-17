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
#include <fstream>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model_context.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/musa/deepseek_v41/batch_metadata.h"
#include "core/layers/musa/deepseek_v41/decode_graph.h"
#include "core/layers/musa/deepseek_v41/engram_embedding.h"
#include "core/layers/musa/deepseek_v41/engram_hash.h"
#include "core/layers/musa/deepseek_v41/prefix_checkpoint.h"
#include "core/layers/musa/deepseek_v41/request_metadata.h"
#include "core/layers/musa/deepseek_v41/state_checkpoint.h"
#include "core/layers/musa/deepseek_v41/text_model.h"
#include "core/platform/musa_model_compute_stream.h"
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
    replicated_attention_ =
        util::get_bool_env("XLLM_DSV41_REPLICATED_ATTENTION", false);
    owner_attention_ = util::get_bool_env("XLLM_DSV41_OWNER_DEEPEP_LL", false);
    owner_gather_outside_graph_ =
        util::get_bool_env("XLLM_DSV41_OWNER_GATHER_OUTSIDE_GRAPH", false);
    CHECK(!owner_gather_outside_graph_ || owner_attention_);
    CHECK(!owner_attention_ || replicated_attention_);
    native_dspark_ = util::get_bool_env("XLLM_DSV41_DSPARK_NATIVE", false);
    gather_bf16_bits_ =
        util::get_bool_env("XLLM_DSV41_GATHER_BF16_BITS", false);
    direct_rope_table_ =
        util::get_bool_env("XLLM_DSV41_DIRECT_ROPE_TABLE", false);
    CHECK(!gather_bf16_bits_ ||
          (tp_size() == 8 && musa_custom_allreduce_enabled(tp_group_)))
        << "BF16 bit gather requires the initialized TP8 custom communicator";

    const char* dense_prequant = std::getenv("XLLM_DSV41_DENSE_PREQUANT_DIR");
    if (dense_prequant != nullptr) {
      dense_prequant_path_ = dense_prequant;
      CHECK(!dense_prequant_path_.empty());
      CHECK(std::filesystem::is_regular_file(
          std::filesystem::path(dense_prequant_path_) / "export_summary.json"));
      LOG(INFO) << "Native V4.1 preserves FP8 dense weights/scales; AOT="
                << dense_prequant_path_;
    }

    selective_spec_commit_ =
        util::get_bool_env("XLLM_DSV41_SPEC_SELECTIVE_COMMIT", false);
    CHECK(!selective_spec_commit_ || native_dspark_);
    fused_spec_snapshot_ =
        util::get_bool_env("XLLM_DSV41_FUSED_SPEC_SNAPSHOT", false);
    CHECK(!fused_spec_snapshot_ || selective_spec_commit_);
    verify_custom_allreduce_ =
        util::get_bool_env("XLLM_DSV41_VERIFY_CUSTOM_AR", false);
    CHECK(!verify_custom_allreduce_ ||
          (native_dspark_ &&
           util::get_bool_env("XLLM_MUSA_ENABLE_TP_CUSTOM_AR", true) &&
           util::get_bool_env("XLLM_DSV41_DECODE_EP_CUSTOM_AR", false)))
        << "Verification custom reduction requires native DSpark and both "
           "validated TP/EP custom policies";

    full_decode_graph_ =
        util::get_bool_env("XLLM_DSV41_FULL_DECODE_GRAPH", false);
    static_decode_ = full_decode_graph_ ||
                     util::get_bool_env("XLLM_DSV41_STATIC_DECODE", false);
    if (static_decode_) {
      CHECK(!native_dspark_ ||
            (selective_spec_commit_ && verify_custom_allreduce_))
          << "Static verification requires selective cache commit and "
             "the validated multirow TP/EP custom communicator";
      CHECK(util::get_bool_env("XLLM_DSV41_DECODE_FUSED_SWA_READ", false));
      CHECK(util::get_bool_env("XLLM_DSV41_DECODE_CACHE_READ_REUSE", false));
      CHECK(!util::get_bool_env("XLLM_DSV41_DECODE_FFN_TAIL_GRAPH", false));
      CHECK(util::get_bool_env("XLLM_DSV41_MODEL_NONDEFAULT_STREAM", false));
    }
    const int64_t speculative_tokens =
        SpeculativeConfig::get_instance().num_speculative_tokens();
    CHECK(speculative_tokens == 0 ||
          (native_dspark_ && speculative_tokens <= 5));
    if (speculative_tokens > 0) {
      CHECK(args_.layers_to_capture() == std::vector<int32_t>({37, 38, 39}));
      CHECK_EQ(args_.n_layers(), 40);
    }
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
    // Target and draft can have different head counts. Keep the draft's H8
    // library unchanged when the target uses replicated H64 attention.
    const char* target_sparse_library =
        std::getenv("XLLM_DSV41_TARGET_MATE_SPARSE_SO");
    CHECK(!replicated_attention_ || !native_dspark_ ||
          (selective_spec_commit_ && verify_custom_allreduce_ &&
           target_sparse_library != nullptr))
        << "Replicated speculative target requires selective cache commit, "
           "verification reduction, and a target-only H64 sparse library; "
           "the draft keeps its independent H8 attention";
    if (target_sparse_library != nullptr) {
      CHECK_NE(target_sparse_library[0], '\0');
      sparse_library = target_sparse_library;
    }
    if (sparse_library != nullptr) {
      sparse_path_ = sparse_library;
      CHECK(std::filesystem::is_regular_file(sparse_path_));
      CHECK_EQ(args_.n_heads() / attention_tp_size(),
               replicated_attention_ ? 64 : 8);
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
    mhc_dot_sinkhorn_path_ =
        util::get_optional_string_env("XLLM_DSV41_MHC_DOT_SINKHORN_DIR")
            .value_or("");
    if (!mhc_dot_sinkhorn_path_.empty()) {
      for (const char* filename :
           {"libsg_mhc_dot160.so", "reduce.mubin", "manifest.json"}) {
        CHECK(std::filesystem::is_regular_file(
            std::filesystem::path(mhc_dot_sinkhorn_path_) / filename));
      }
      std::ifstream manifest_file(mhc_dot_sinkhorn_path_ + "/manifest.json");
      CHECK(manifest_file.good());
      const auto manifest = nlohmann::json::parse(manifest_file);
      const auto& constants = manifest.at("reduce_constants");
      CHECK_EQ(constants.at("EPS").get<float>(), args_.rms_norm_eps());
      CHECK_EQ(constants.at("HC_EPS").get<float>(), args_.hc_eps());
      CHECK_EQ(constants.at("SINKHORN_ITERS").get<int64_t>(),
               args_.hc_sinkhorn_iters());
      CHECK_EQ(constants.at("NUM_SLICES").get<int64_t>(), 160);
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
    initialize_speculative_commit_policy();
    initialize_attention_replication_policy();
    initialize_decode_ep_policy();
    initialize_ffn_tail_graph_policy();
    initialize_shared_expert_tp_policy();
    initialize_gate_cache_policy();
    initialize_decode_batch_policy();
    initialize_bounded_replay_policy();
    initialize_owner_deepep_policy();
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
        expert_reduction_group(),
        bounded_replay_,
        replicated_attention_,
        owner_attention_,
        gather_bf16_bits_);
    if (static_decode_) {
      std::vector<int64_t> ratios;
      ratios.reserve(args_.n_layers());
      for (int64_t layer = 0; layer < args_.n_layers(); ++layer) {
        ratios.emplace_back(args_.compress_ratios()[layer]);
      }
      decode_graph_ = std::make_unique<layer::DeepseekV41DecodeGraph>(
          std::move(ratios),
          owner_attention_ ? std::function<void()>(
                                 [this]() { deep_ep_ll_->reset_for_capture(); })
                           : std::function<void()>{});
      CHECK(native_dspark_ || args_.layers_to_capture().empty());
      LOG(INFO) << "Native V4.1 fixed-capacity C1 Decode/verify enabled, graph="
                << full_decode_graph_
                << ", rows=1..6, capacities=512/4096; Prefill stays eager";
    }
    if (!args_.layers_to_capture().empty()) {
      auxiliary_ = std::make_unique<AuxHiddenCapture>(
          args_, options_, /*max_tokens_per_batch=*/512);
    }
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
    const int64_t requests = params.execution_batch.num_reqs;
    const bool batched = requests > 1;
    CHECK(!batched || batched_decode_)
        << "Native V4.1 multi-request decode requires explicit batch policy";
    CHECK(!params.enable_graph);
    CHECK(!params.is_spec_verify || native_dspark_);
    CHECK(!params.embedding.input_embedding.defined())
        << "Native V4.1 text only";
    std::optional<std::vector<int64_t>> context_lengths;
    std::optional<int64_t> q_length;
    std::optional<int64_t> kv_length;
    if (batched) {
      context_lengths = layer::parse_deepseek_v41_decode_batch(
          params.attention.host.q_seq_lens,
          params.attention.host.kv_seq_lens,
          requests,
          tokens.numel());
      CHECK(context_lengths.has_value())
          << "Only B2..B8 all-q1 decode is supported; mixed prefill rejected";
      q_length = tokens.numel();
      kv_length =
          *std::max_element(context_lengths->begin(), context_lengths->end());
    } else {
      CHECK_EQ(requests, 1);
      q_length =
          layer::parse_deepseek_v41_c1_length(params.attention.host.q_seq_lens);
      kv_length = layer::parse_deepseek_v41_c1_length(
          params.attention.host.kv_seq_lens);
    }
    CHECK(q_length.has_value() && kv_length.has_value())
        << "Invalid C1 query/context host length metadata";
    CHECK_EQ(q_length.value(), tokens.numel());
    const int64_t context_length = kv_length.value();
    const layer::DeepseekV41ProfileRange range(
        "V41/step", /*layer=*/-1, tokens.numel(), context_length);
    CHECK(batched || context_length >= tokens.numel());
    CHECK_GT(tokens.numel(), 0);
    CHECK_EQ(kv_caches.size(), static_cast<size_t>(args_.n_layers()));
    const int64_t table_count = 1 + static_cast<int64_t>(ratio1_group_ >= 0) +
                                static_cast<int64_t>(ratio2_group_ >= 0);
    CHECK_EQ(params.multi_block_tables.size(), table_count);
    if (!replaying_ && !params.synchronize_all_layers()) {
      return ModelOutput();
    }
    torch::DeviceGuard guard(device());
    std::optional<MusaModelComputeStream> model_stream;
    if (!replaying_ &&
        util::get_bool_env("XLLM_DSV41_MODEL_NONDEFAULT_STREAM", false)) {
      model_stream.emplace(device());
      LOG_FIRST_N(INFO, 1)
          << "Native V4.1 model forward on a valid nondefault stream, "
             "with explicit input/output fences and output storage "
             "registration";
    }
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
      CHECK_EQ(host_table.size(0), requests);
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
    if (native_dspark_ && !replaying_) {
      prepare_speculative_state(tokens,
                                positions,
                                kv_caches,
                                params,
                                caches,
                                swa_table,
                                ratio1_table,
                                ratio2_table,
                                context_length);
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
                              .reshape({requests, -1})
                              .to(device());
      }
      torch::Tensor hashes;
      if (batched) {
        std::vector<torch::Tensor> per_request_hashes;
        std::vector<torch::Tensor> per_request_mask;
        per_request_hashes.reserve(requests);
        per_request_mask.reserve(requests);
        for (int64_t row = 0; row < requests; ++row) {
          const std::optional<torch::Tensor> row_lookback =
              lookback_tensor.has_value()
                  ? std::make_optional(lookback_tensor->narrow(0, row, 1))
                  : std::nullopt;
          auto [row_hashes, row_mask] =
              engram_hash_->forward(tokens.narrow(0, row, 1),
                                    positions.narrow(0, row, 1),
                                    swa_table.narrow(0, row, 1),
                                    swa.size(1),
                                    context_lengths.value()[row] - 1,
                                    row_lookback);
          per_request_hashes.emplace_back(std::move(row_hashes));
          per_request_mask.emplace_back(std::move(row_mask));
        }
        hashes = torch::cat(per_request_hashes, 0);
        mask = torch::cat(per_request_mask, 0);
      } else {
        auto [single_hashes, gate_mask] =
            engram_hash_->forward(tokens,
                                  positions,
                                  swa_table,
                                  swa.size(1),
                                  context_length - tokens.numel(),
                                  lookback_tensor);
        hashes = std::move(single_hashes);
        mask = std::move(gate_mask);
      }
      for (int64_t layer : args_.engram_layer_ids()) {
        engram_rows[layer] =
            engram_tables_[layer]
                ->lookup(hashes.select(1, engram_indices_[layer]))
                .flatten(1);
      }
    }
    std::optional<torch::Tensor> prepared_embedding;
    if (owner_gather_outside_graph_) {
      prepared_embedding = model_->prepare_embeddings(tokens);
    }
    torch::Tensor hidden;
    if (batched) {
      CHECK(!auxiliary_) << "DSpark capture currently requires one request";
      hidden = model_->forward_batched_decode(tokens,
                                              positions,
                                              caches,
                                              swa_table,
                                              ratio1_table,
                                              ratio2_table,
                                              context_lengths.value(),
                                              engram_rows,
                                              mask);
    } else if (decode_graph_ && context_length <= 4096 &&
               (tokens.numel() == 1 ||
                (params.is_spec_verify && tokens.numel() <= 6))) {
      layer::DeepseekV41DecodeInputs input{tokens,
                                           positions,
                                           caches,
                                           swa_table,
                                           ratio1_table,
                                           ratio2_table,
                                           context_length,
                                           engram_rows,
                                           mask,
                                           prepared_embedding};
      hidden = decode_graph_->run(
          input,
          full_decode_graph_,
          [this](const layer::DeepseekV41DecodeInputs& value,
                 int64_t capacity) {
            return model_->forward_with_capture(value.tokens,
                                                value.positions,
                                                value.caches,
                                                value.swa_table,
                                                value.ratio1_table,
                                                value.ratio2_table,
                                                value.context_length,
                                                value.engram_rows,
                                                value.engram_mask,
                                                auxiliary_.get(),
                                                capacity,
                                                value.prepared_embedding);
          });
    } else {
      hidden = model_->forward_with_capture(tokens,
                                            positions,
                                            caches,
                                            swa_table,
                                            ratio1_table,
                                            ratio2_table,
                                            context_length,
                                            engram_rows,
                                            mask,
                                            auxiliary_.get(),
                                            /*decode_capacity=*/0,
                                            prepared_embedding);
    }
    ModelOutput output =
        auxiliary_ ? auxiliary_->finalize(hidden) : ModelOutput(hidden);
    if (owner_attention_) {
      // Target head and draft are still TP8. Only owner computed these values;
      // broadcast once at the model boundary, never sum an LL-combined output.
      tp_group_->broadcast(hidden, /*root_rank=*/0);
      if (output.aux_hidden_states.defined()) {
        tp_group_->broadcast(output.aux_hidden_states, /*root_rank=*/0);
      }
    }
    if (model_stream.has_value()) {
      model_stream->record_output(hidden);
      if (output.aux_hidden_states.defined()) {
        model_stream->record_output(output.aux_hidden_states);
      }
      model_stream->finish();
    }
    return output;
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
    out_hidden = model_->select_hidden(hidden, defined(selected));
    return model_->project_logits(out_hidden);
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
  struct PendingVerification {
    std::unique_ptr<layer::DeepseekV41StateCheckpoint> checkpoint;
    std::unique_ptr<layer::DeepseekV41PrefixCheckpoint> prefix_checkpoint;
    torch::Tensor tokens;
    torch::Tensor positions;
    ModelInputParams parameters;
    std::string request_id;
    int64_t begin = 0;
  };

  void prepare_speculative_state(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      std::vector<KVCache>& kv_caches,
      const ModelInputParams& params,
      const std::vector<layer::DeepseekV41AttentionCache>& caches,
      const torch::Tensor& swa_table,
      const torch::Tensor& c1_table,
      const torch::Tensor& c2_table,
      int64_t context) {
    CHECK_EQ(params.execution_batch.num_reqs, 1);
    if (!params.is_spec_verify && !pending_verification_) {
      return;
    }
    if (params.meta.is_graph_warmup) {
      CHECK(!pending_verification_);
      return;
    }
    CHECK_EQ(params.embedding.request_ids.size(), 1)
        << "Native DSpark requires explicit request identity for cache "
           "rollback";
    const std::string& request = params.embedding.request_ids.front();
    const int64_t begin = context - tokens.numel();
    if (pending_verification_) {
      if (request != pending_verification_->request_id) {
        // A P/D import or a new prefill may already own these physical pages.
        // Never restore a previous request's snapshot over another request.
        pending_verification_.reset();
      } else {
        CHECK(params.is_spec_verify)
            << "Native DSpark does not support changing to eager mid-request";
        const int64_t accepted = begin - pending_verification_->begin;
        CHECK_GT(accepted, 0);
        CHECK_LE(accepted, pending_verification_->tokens.numel());
        CHECK_EQ(params.num_accepted_tokens_host.size(), 1);
        CHECK_EQ(accepted, params.num_accepted_tokens_host.front())
            << "Accepted scheduler length disagrees with next verification "
               "position";
        auto previous = std::move(pending_verification_);
        if (previous->prefix_checkpoint) {
          const layer::DeepseekV41ProfileRange range(
              "V41/DSpark/restore_rejected_suffix");
          previous->prefix_checkpoint->restore_suffix(accepted);
        } else {
          previous->checkpoint->restore();
          ModelInputParams replay = previous->parameters;
          replay.is_spec_verify = false;
          replay.attention.host.q_seq_lens = {0,
                                              static_cast<int32_t>(accepted)};
          replay.attention.host.kv_seq_lens = {0, static_cast<int32_t>(begin)};
          replay.attention.host.q_cu_seq_lens =
              replay.attention.host.q_seq_lens;
          replay.num_accepted_tokens_host.clear();
          replay.num_accepted_tokens = torch::Tensor{};
          replaying_ = true;
          forward(previous->tokens.narrow(0, 0, accepted),
                  previous->positions.narrow(0, 0, accepted),
                  kv_caches,
                  replay);
          replaying_ = false;
        }
      }
    }
    if (!params.is_spec_verify) {
      return;
    }
    CHECK_GT(tokens.numel(), 0);
    CHECK_LE(tokens.numel(), 6);
    CHECK_EQ(caches.size(), static_cast<size_t>(args_.n_layers()));
    // History uses the same physical SWA slots. It must be bound before the
    // first checkpoint, and must be restored along with every cache write.
    if (engram_hash_ && (!history_swa_.defined() ||
                         history_swa_.data_ptr() != caches[0].swa.data_ptr() ||
                         history_swa_.sizes() != caches[0].swa.sizes())) {
      const torch::Tensor& swa = caches[0].swa;
      engram_hash_->bind_history(torch::zeros({swa.size(0) * swa.size(1)},
                                              options_.dtype(torch::kInt32)));
      history_swa_ = swa;
    }
    auto pending = std::make_unique<PendingVerification>();
    if (selective_spec_commit_) {
      pending->prefix_checkpoint =
          std::make_unique<layer::DeepseekV41PrefixCheckpoint>(
              begin,
              tokens.numel(),
              /*max_bytes=*/64 * 1024 * 1024,
              fused_spec_snapshot_);
    } else {
      pending->checkpoint = std::make_unique<layer::DeepseekV41StateCheckpoint>(
          /*max_bytes=*/64 * 1024 * 1024);
    }
    const layer::DeepseekV41ProfileRange snapshot_range(
        "V41/DSpark/snapshot_target_state");
    std::unordered_set<uintptr_t> saved;
    const auto unique_tensor = [&](const torch::Tensor& value) {
      return value.defined() && value.numel() > 0 &&
             saved.insert(reinterpret_cast<uintptr_t>(value.data_ptr())).second;
    };
    const auto save_paged = [&](const torch::Tensor& cache,
                                const torch::Tensor& table,
                                int64_t ratio,
                                bool ring) {
      if (pending->prefix_checkpoint) {
        pending->prefix_checkpoint->add_paged(cache, table, ratio, ring);
      } else {
        const int64_t first = begin / ratio;
        const int64_t count = context / ratio - first;
        pending->checkpoint->add_paged(cache, table, first, count, ring);
      }
    };
    torch::Tensor state_pages;
    if (!selective_spec_commit_) {
      const torch::Tensor rows =
          torch::arange(tokens.numel(),
                        swa_table.options().dtype(torch::kInt64)) +
          begin;
      const torch::Tensor columns =
          torch::floor_divide(rows, caches[0].swa.size(1))
              .remainder(swa_table.size(1));
      state_pages =
          swa_table.select(0, 0).index_select(0, columns).contiguous();
    }
    const auto save_recurrent = [&](const torch::Tensor& state) {
      if (pending->prefix_checkpoint) {
        pending->prefix_checkpoint->add_recurrent(
            state, swa_table, caches[0].swa.size(1));
      } else {
        pending->checkpoint->add_blocks(state, state_pages);
      }
    };
    for (size_t index = 0; index < caches.size(); ++index) {
      const auto& cache = caches[index];
      if (unique_tensor(cache.swa)) {
        save_paged(cache.swa, swa_table, /*ratio=*/1, /*ring=*/true);
      }
      const int64_t ratio = args_.compress_ratios()[index];
      if (ratio > 0) {
        CHECK(ratio == 1 || ratio == 2);
        const torch::Tensor& table = ratio == 1 ? c1_table : c2_table;
        if (unique_tensor(cache.key)) {
          save_paged(cache.key, table, ratio, /*ring=*/false);
        }
        if (unique_tensor(cache.index)) {
          save_paged(cache.index, table, ratio, /*ring=*/false);
        }
      }
      if (cache.compressor_kv_state.has_value() &&
          unique_tensor(cache.compressor_kv_state.value())) {
        save_recurrent(cache.compressor_kv_state.value());
      }
      if (cache.compressor_score_state.has_value() &&
          unique_tensor(cache.compressor_score_state.value())) {
        save_recurrent(cache.compressor_score_state.value());
      }
    }
    if (engram_hash_) {
      const torch::Tensor& swa = caches[0].swa;
      save_paged(engram_hash_->history().view({swa.size(0), swa.size(1), 1}),
                 swa_table,
                 /*ratio=*/1,
                 /*ring=*/true);
    }
    pending->tokens = tokens.clone();
    pending->positions = positions.clone();
    pending->parameters = params;
    for (auto& table : pending->parameters.multi_block_tables) {
      table = table.clone();
    }
    pending->request_id = request;
    pending->begin = begin;
    LOG_FIRST_N(INFO, 1) << "Native DSpark target checkpoint bytes="
                         << (pending->prefix_checkpoint
                                 ? pending->prefix_checkpoint->saved_bytes()
                                 : pending->checkpoint->saved_bytes())
                         << "; selective_commit=" << selective_spec_commit_
                         << "; rejected writes restored, accepted prefix "
                         << (selective_spec_commit_ ? "retained"
                                                    : "replayed eagerly");
    pending_verification_ = std::move(pending);
  }
  void initialize_speculative_commit_policy() {
    const int32_t policy =
        static_cast<int32_t>(native_dspark_) |
        (static_cast<int32_t>(selective_spec_commit_) << 1) |
        (static_cast<int32_t>(verify_custom_allreduce_) << 2) |
        (static_cast<int32_t>(fused_spec_snapshot_) << 3) |
        (static_cast<int32_t>(gather_bf16_bits_) << 4) |
        (static_cast<int32_t>(direct_rope_table_) << 5);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local =
          torch::tensor({policy}, options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK_EQ(peers.numel(), tp_size());
      const int32_t* policies = peers.data_ptr<int32_t>();
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], policy)
            << "Native DSpark cache commit policy differs across TP ranks";
      }
    }
  }

  void initialize_ffn_tail_graph_policy() {
    const bool enabled =
        util::get_bool_env("XLLM_DSV41_DECODE_FFN_TAIL_GRAPH", false);
    const bool fused_inputs =
        util::get_bool_env("XLLM_DSV41_DECODE_FFN_GRAPH_FUSED_INPUTS", false);
    CHECK(!fused_inputs || enabled)
        << "Fused Graph inputs require the FFN-tail Graph";
    const int32_t policy = static_cast<int32_t>(enabled) |
                           (static_cast<int32_t>(fused_inputs) << 1);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local =
          torch::tensor({policy}, options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK_EQ(peers.numel(), tp_size());
      const int32_t* policies = peers.data_ptr<int32_t>();
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], policy)
            << "FFN-tail Graph policy differs across TP ranks";
      }
    }
    CHECK(!enabled ||
          (decode_ep_custom_allreduce_ &&
           util::get_bool_env("XLLM_MUSA_ENABLE_TP_CUSTOM_AR", true)))
        << "Native FFN-tail Graph currently requires the validated TP/EP "
           "custom path";
    decode_ffn_tail_graph_ = enabled;
    decode_ffn_graph_fused_inputs_ = fused_inputs;
  }

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

  void initialize_owner_deepep_policy() {
    const int64_t policy =
        static_cast<int64_t>(owner_attention_) |
        (static_cast<int64_t>(owner_gather_outside_graph_) << 1);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local =
          torch::tensor({policy}, options_.dtype(torch::kInt64));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK(peers.eq(policy).all().item<bool>())
          << "Owner/empty-rank DeepEP policy must agree on every rank";
    }
    if (!owner_attention_) {
      return;
    }
    CHECK_EQ(tp_size(), 8);
    CHECK_EQ(ep_size(), 8);
    CHECK(!shared_expert_tp_reduce_ && !batched_decode_);
    CHECK(!dense_prequant_path_.empty());
    CHECK(expert_format_ == layer::DeepseekV41ExpertFormat::W4AFP8_MT);
    CHECK_EQ(args_.n_layers(), 40);
    const char* endpoint = std::getenv("XLLM_DSV41_DEEPEP_INIT_METHOD");
    CHECK(endpoint != nullptr && endpoint[0] != '\0');
    deep_ep_ll_ = std::make_shared<layer::DeepseekV41DeepEpLlRuntime>(
        tp_group_->rank(),
        tp_size(),
        device().index(),
        /*capacity=*/16,
        args_.hidden_size(),
        args_.n_routed_experts(),
        endpoint);
    LOG(INFO)
        << "V4.1 C1 owner=0 Attention/shared; empty ranks execute EP8 "
           "experts via DeepEP LL; head/embedding/Engram/draft remain TP8";
  }

  void initialize_shared_expert_tp_policy() {
    const bool enabled =
        util::get_bool_env("XLLM_DSV41_SHARED_EXPERT_TP", false);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local = torch::tensor({static_cast<int32_t>(enabled)},
                                                options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK_EQ(peers.numel(), tp_size());
      const int32_t* policies = peers.data_ptr<int32_t>();
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], static_cast<int32_t>(enabled))
            << "Native V4.1 shared expert TP policy differs across ranks";
      }
    }
    CHECK(!enabled || (tp_size() > 1 && ep_size() == tp_size()))
        << "Shared expert TP requires matching full-world TP and EP groups";
    shared_expert_tp_reduce_ = enabled;
  }

  void initialize_decode_batch_policy() {
    const bool enabled = util::get_bool_env("XLLM_DSV41_BATCHED_DECODE", false);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local = torch::tensor({static_cast<int32_t>(enabled)},
                                                options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK_EQ(peers.numel(), tp_size());
      const int32_t* policies = peers.data_ptr<int32_t>();
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], static_cast<int32_t>(enabled))
            << "Native V4.1 batch policy differs across ranks";
      }
    }
    batched_decode_ = enabled;
  }

  void initialize_bounded_replay_policy() {
    const bool enabled =
        util::get_bool_env("XLLM_DSV41_BOUNDED_PREFILL_REPLAY", false);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local = torch::tensor({static_cast<int32_t>(enabled)},
                                                options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      const int32_t* policies = peers.data_ptr<int32_t>();
      CHECK_EQ(peers.numel(), tp_size());
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], static_cast<int32_t>(enabled))
            << "Bounded prefill replay policy differs across TP ranks";
      }
    }
    if (enabled) {
      CHECK(args_.layers_to_capture().empty())
          << "Bounded prefill replay does not expose uncaptured prompt aux "
             "rows";
    }
    bounded_replay_ = enabled;
  }

  void initialize_gate_cache_policy() {
    const bool enabled =
        util::get_bool_env("XLLM_DSV41_GATE_CACHE_FP32_WEIGHT", false);
    if (tp_group_ != nullptr && tp_size() > 1) {
      const torch::Tensor local = torch::tensor({static_cast<int32_t>(enabled)},
                                                options_.dtype(torch::kInt32));
      const torch::Tensor peers =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      CHECK_EQ(peers.numel(), tp_size());
      const int32_t* policies = peers.data_ptr<int32_t>();
      for (int64_t rank = 0; rank < tp_size(); ++rank) {
        CHECK_EQ(policies[rank], static_cast<int32_t>(enabled))
            << "Native V4.1 gate FP32 cache policy differs across ranks";
      }
    }
    gate_cache_fp32_weight_ = enabled;
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

  int64_t attention_tp_size() const {
    return replicated_attention_ ? 1 : tp_size();
  }

  void initialize_attention_replication_policy() {
    if (tp_size() > 1) {
      const torch::Tensor local =
          torch::tensor({replicated_attention_ ? int64_t{1} : int64_t{0}},
                        options_.dtype(torch::kInt64));
      const torch::Tensor choices =
          tp_group_->allgather_base_sync(local).to(torch::kCPU).contiguous();
      const int64_t* values = choices.data_ptr<int64_t>();
      for (int64_t index = 0; index < choices.numel(); ++index) {
        CHECK_EQ(values[index], replicated_attention_ ? 1 : 0)
            << "Attention replication policy must match all world ranks";
      }
    }
    if (replicated_attention_) {
      CHECK_EQ(tp_size(), 8);
      CHECK_EQ(ep_size(), 8);
      CHECK_EQ(args_.n_heads(), 64);
      CHECK(expert_format_ == layer::DeepseekV41ExpertFormat::W4AFP8_MT);
      CHECK(!sparse_path_.empty());
      LOG(INFO) << "V4.1 replicated Attention TP1 on all 8 ranks; "
                   "Engram/embedding/head remain TP8, experts EP8. "
                   "This is replicated C1 execution, not DP8 scheduling.";
    }
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
    attention.heads = args_.n_heads() / attention_tp_size();
    attention.groups = args_.o_groups() / attention_tp_size();
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
    attention.dense_prequant_directory = dense_prequant_path_;
    attention.direct_rope_table = direct_rope_table_;
    attention.verify_skinny_projection =
        util::get_bool_env("XLLM_DSV41_VERIFY_SKINNY_PROJECTION", false);
    attention.decode_packed_qkv =
        util::get_bool_env("XLLM_DSV41_DECODE_PACKED_QKV", false);
    attention.decode_fused_swa_read =
        util::get_bool_env("XLLM_DSV41_DECODE_FUSED_SWA_READ", false);
    attention.verify_fused_swa_read =
        util::get_bool_env("XLLM_DSV41_VERIFY_FUSED_SWA_READ", false);
    attention.decode_cache_read_reuse =
        util::get_bool_env("XLLM_DSV41_DECODE_CACHE_READ_REUSE", false);
    attention.short_context_indexer =
        util::get_bool_env("XLLM_DSV41_SHORT_CONTEXT_INDEXER", false);
    config.sinkhorn_iterations = args_.hc_sinkhorn_iters();
    config.hc_epsilon = args_.hc_eps();
    config.mhc_prenorm_directory = mhc_path_;
    config.mhc_dot_sinkhorn_directory = mhc_dot_sinkhorn_path_;
    config.decode_mhc_fused_reduce =
        util::get_bool_env("XLLM_DSV41_DECODE_MHC_FUSED_REDUCE", false);
    config.decode_ffn_tail_graph = decode_ffn_tail_graph_;
    config.decode_ffn_graph_fused_inputs = decode_ffn_graph_fused_inputs_;
    // MusaCustomAllReduce checks this policy across all TP ranks during
    // communicator construction. Keep one consistent selector rather than
    // adding an independent model flag that could disagree between ranks.
    config.decode_tp_custom_allreduce =
        util::get_bool_env("XLLM_MUSA_ENABLE_TP_CUSTOM_AR", true);
    config.verify_custom_allreduce = verify_custom_allreduce_;
    config.attention.sparse_library_path = sparse_path_;
    config.attention.indexer_score_directory = indexer_aot_path_;
    auto& moe = config.moe;
    moe.decode_ep_custom_allreduce = decode_ep_custom_allreduce_;
    moe.verify_custom_allreduce = verify_custom_allreduce_;
    moe.expert_format = expert_format_;
    moe.w4a8_so_path = w4a8_path_;
    moe.dense_prequant_directory = dense_prequant_path_;
    moe.deep_ep_ll = deep_ep_ll_;
    moe.w4a8_static_activation =
        util::get_bool_env("XLLM_DSV41_W4A8_STATIC_ACTIVATION", false);
    moe.w4a8_valid_prefix_swiglu =
        util::get_bool_env("XLLM_DSV41_W4A8_VALID_PREFIX_SWIGLU", false);
    moe.verify_skinny_projection = attention.verify_skinny_projection;
    moe.decode_packed_shared_gate_up =
        util::get_bool_env("XLLM_DSV41_DECODE_PACKED_SHARED_GATE_UP", false);
    moe.dense =
        index < args_.first_k_dense_replace() ||
        (args_.moe_layer_freq() > 1 && index % args_.moe_layer_freq() != 0);
    moe.shared_expert_tp_reduce = !moe.dense && shared_expert_tp_reduce_;
    moe.gate_cache_fp32_weight = !moe.dense && gate_cache_fp32_weight_;
    moe.gate_bf16_fp32_projection =
        !moe.dense && util::get_bool_env("XLLM_DSV41_GATE_BF16_FP32", false);
    moe.gate_simt_projection =
        !moe.dense && util::get_bool_env("XLLM_DSV41_GATE_SIMT", false);
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
    const std::optional<int64_t> head_shard =
        replicated_attention_ ? std::nullopt : std::optional<int64_t>{0};
    const std::optional<int64_t> output_shard =
        replicated_attention_ ? std::nullopt : std::optional<int64_t>{1};
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
        projection(attention + "wq_b", 2, head_shard),
        projection(attention + "wkv", 3),
        checkpoint.copy(attention + "kv_norm.weight", torch::kBFloat16),
        checkpoint.dense(attention + "wo_a", head_shard),
        projection(attention + "wo_b", 6, output_shard)};
    const std::string sink_name = checkpoint.has(attention + "attn_sink")
                                      ? attention + "attn_sink"
                                      : attention + "attn_sink.weight";
    attn.sink = checkpoint.copy(sink_name, torch::kFloat32, head_shard);
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
  bool verify_custom_allreduce_ = false;
  bool decode_ffn_tail_graph_ = false;
  bool decode_ffn_graph_fused_inputs_ = false;
  bool shared_expert_tp_reduce_ = false;
  bool bounded_replay_ = false;
  bool gate_cache_fp32_weight_ = false;
  bool batched_decode_ = false;
  std::string bmm_path_;
  std::string w4a8_path_;
  std::string dense_prequant_path_;
  std::string mhc_path_;
  std::string mhc_dot_sinkhorn_path_;
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
  bool native_dspark_ = false;
  bool selective_spec_commit_ = false;
  bool fused_spec_snapshot_ = false;
  bool replaying_ = false;
  std::unique_ptr<PendingVerification> pending_verification_;
  std::unique_ptr<AuxHiddenCapture> auxiliary_;
  std::unique_ptr<layer::DeepseekV41TextModel> model_;
  bool static_decode_ = false;
  bool full_decode_graph_ = false;
  bool replicated_attention_ = false;
  bool owner_attention_ = false;
  bool owner_gather_outside_graph_ = false;
  bool gather_bf16_bits_ = false;
  bool direct_rope_table_ = false;
  std::shared_ptr<layer::DeepseekV41DeepEpLlRuntime> deep_ep_ll_;
  // Graphs retain weight/cache pointers; destroy before model and Engram
  // assets.
  std::unique_ptr<layer::DeepseekV41DecodeGraph> decode_graph_;
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
