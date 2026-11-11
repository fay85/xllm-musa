/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/

#include "core/layers/musa/deepseek_v41/attention.h"
#include "core/layers/musa/deepseek_v41/attention_projection.h"
#include "core/layers/musa/deepseek_v41/compressor.h"
#include "core/layers/musa/deepseek_v41/csa2_attention.h"
#include "core/layers/musa/deepseek_v41/decoder.h"
#include "core/layers/musa/deepseek_v41/engram_embedding.h"
#include "core/layers/musa/deepseek_v41/engram_hash.h"
#include "core/layers/musa/deepseek_v41/engram_injection.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"
#include "core/layers/musa/deepseek_v41/hyper_connection.h"
#include "core/layers/musa/deepseek_v41/indexer.h"
#include "core/layers/musa/deepseek_v41/paged_attention.h"
#include "core/layers/musa/deepseek_v41/request_metadata.h"
#include "core/layers/musa/deepseek_v41/text_model.h"
#include "core/layers/musa/deepseek_v41/token_cache.h"
#include "core/runtime/py_musa_kernel_bindings.h"
#include "models/llm/musa/deepseek_v41_checkpoint.h"

// Exposes exactly the production binding for tests without starting a server.
PYBIND11_MODULE(musa_python_kernel_test_module, module) {
  xllm::register_musa_kernel_bindings(module);
  using NativeDense = xllm::layer::DeepseekV41Dense;
  pybind11::class_<NativeDense>(module, "NativeV41Dense")
      .def(pybind11::init<torch::Tensor, bool>())
      .def("forward",
           &NativeDense::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeSparseAttention = xllm::layer::DeepseekV41SparseAttention;
  pybind11::class_<NativeSparseAttention>(module, "NativeV41SparseAttention")
      .def(pybind11::init<double, int64_t>())
      .def("forward",
           &NativeSparseAttention::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeHyperConnection = xllm::layer::DeepseekV41HyperConnection;
  pybind11::class_<NativeHyperConnection>(module, "NativeV41HyperConnection")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          int64_t,
                          double,
                          double>())
      .def("prepare",
           &NativeHyperConnection::prepare,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("combine",
           &NativeHyperConnection::combine,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeSharedMlp = xllm::layer::DeepseekV41SharedMlp;
  pybind11::class_<NativeSharedMlp>(module, "NativeV41SharedMlp")
      .def(
          pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, double>())
      .def(
          pybind11::
              init<torch::Tensor, torch::Tensor, torch::Tensor, double, bool>())
      .def("forward",
           &NativeSharedMlp::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativePackedMlp = xllm::layer::DeepseekV41PackedRoutedMlp;
  pybind11::class_<NativePackedMlp>(module, "NativeV41PackedRoutedMlp")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          double,
                          bool>())
      .def("forward",
           &NativePackedMlp::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeRoutedExperts = xllm::layer::DeepseekV41PackedRoutedExperts;
  pybind11::class_<NativeRoutedExperts>(module, "NativeV41PackedRoutedExperts")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          int64_t,
                          double,
                          bool>())
      .def("forward",
           &NativeRoutedExperts::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativePartitionMlp = xllm::layer::DeepseekV41PartitionRoutedMlp;
  pybind11::class_<NativePartitionMlp>(module, "NativeV41PartitionRoutedMlp")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          double,
                          std::string,
                          int64_t>())
      .def("forward",
           &NativePartitionMlp::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeGate = xllm::layer::DeepseekV41Gate;
  pybind11::class_<NativeGate>(module, "NativeV41Gate")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          std::optional<torch::Tensor>,
                          int64_t,
                          int64_t,
                          bool,
                          double>())
      .def("forward",
           &NativeGate::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeEngramInjection = xllm::layer::DeepseekV41EngramInjection;
  pybind11::class_<NativeEngramInjection>(module, "NativeV41EngramInjection")
      .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, bool>())
      .def("forward",
           &NativeEngramInjection::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeAttentionProjection = xllm::layer::DeepseekV41AttentionProjection;
  pybind11::class_<NativeAttentionProjection>(module,
                                              "NativeV41AttentionProjection")
      .def(pybind11::init<std::vector<torch::Tensor>,
                          int64_t,
                          int64_t,
                          int64_t,
                          double,
                          bool>())
      .def("prepare",
           &NativeAttentionProjection::prepare,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("project_output",
           &NativeAttentionProjection::project_output,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativePagedAttention = xllm::layer::DeepseekV41PagedAttention;
  pybind11::class_<NativePagedAttention>(module, "NativeV41PagedAttention")
      .def(pybind11::init<torch::Tensor, int64_t, double>())
      .def("forward",
           &NativePagedAttention::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeCompressor = xllm::layer::DeepseekV41Compressor;
  pybind11::class_<NativeCompressor>(module, "NativeV41Compressor")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          int64_t,
                          int64_t,
                          double>())
      .def("forward",
           &NativeCompressor::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeIndexer = xllm::layer::DeepseekV41Indexer;
  pybind11::class_<NativeIndexer>(module, "NativeV41Indexer")
      .def(pybind11::init<std::vector<torch::Tensor>,
                          int64_t,
                          int64_t,
                          double,
                          bool>())
      .def("prepare_query",
           &NativeIndexer::prepare_query,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("prepare_key",
           &NativeIndexer::prepare_key,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("score",
           &NativeIndexer::score,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("select",
           &NativeIndexer::select,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("candidates",
           &NativeIndexer::candidates,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeTokenCache = xllm::layer::DeepseekV41TokenCache;
  pybind11::class_<NativeTokenCache>(module, "NativeV41TokenCache")
      .def(pybind11::init<torch::Tensor>())
      .def("commit",
           &NativeTokenCache::commit,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("gather",
           &NativeTokenCache::gather,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using AttentionMode = xllm::layer::DeepseekV41AttentionMode;
  pybind11::enum_<AttentionMode>(module, "NativeV41AttentionMode")
      .value("SWA", AttentionMode::SWA)
      .value("FULL", AttentionMode::FULL)
      .value("REINDEX", AttentionMode::REINDEX)
      .value("REUSE", AttentionMode::REUSE);
  using NativeAttentionConfig = xllm::layer::DeepseekV41AttentionConfig;
  pybind11::class_<NativeAttentionConfig>(module, "NativeV41AttentionConfig")
      .def(pybind11::init<>())
      .def_readwrite("mode", &NativeAttentionConfig::mode)
      .def_readwrite("layer", &NativeAttentionConfig::layer)
      .def_readwrite("source_layer", &NativeAttentionConfig::source_layer)
      .def_readwrite("ratio", &NativeAttentionConfig::ratio)
      .def_readwrite("heads", &NativeAttentionConfig::heads)
      .def_readwrite("groups", &NativeAttentionConfig::groups)
      .def_readwrite("output_rank", &NativeAttentionConfig::output_rank)
      .def_readwrite("window", &NativeAttentionConfig::window)
      .def_readwrite("index_heads", &NativeAttentionConfig::index_heads)
      .def_readwrite("index_dim", &NativeAttentionConfig::index_dim)
      .def_readwrite("index_topk", &NativeAttentionConfig::index_topk)
      .def_readwrite("candidate_source",
                     &NativeAttentionConfig::candidate_source)
      .def_readwrite("candidate_blocks",
                     &NativeAttentionConfig::candidate_blocks)
      .def_readwrite("candidate_block_size",
                     &NativeAttentionConfig::candidate_block_size)
      .def_readwrite("epsilon", &NativeAttentionConfig::epsilon)
      .def_readwrite("scale", &NativeAttentionConfig::scale)
      .def_readwrite("original_fp8_activation",
                     &NativeAttentionConfig::original_fp8_activation);
  using NativeAttentionWeights = xllm::layer::DeepseekV41AttentionWeights;
  pybind11::class_<NativeAttentionWeights>(module, "NativeV41AttentionWeights")
      .def(pybind11::init<>())
      .def_readwrite("projection", &NativeAttentionWeights::projection)
      .def_readwrite("sink", &NativeAttentionWeights::sink)
      .def_readwrite("compressor_kv", &NativeAttentionWeights::compressor_kv)
      .def_readwrite("compressor_norm",
                     &NativeAttentionWeights::compressor_norm)
      .def_readwrite("compressor_gate",
                     &NativeAttentionWeights::compressor_gate)
      .def_readwrite("indexer", &NativeAttentionWeights::indexer);
  using NativeAttentionCache = xllm::layer::DeepseekV41AttentionCache;
  pybind11::class_<NativeAttentionCache>(module, "NativeV41AttentionCache")
      .def(pybind11::init<>())
      .def_readwrite("swa", &NativeAttentionCache::swa)
      .def_readwrite("key", &NativeAttentionCache::key)
      .def_readwrite("index", &NativeAttentionCache::index)
      .def_readwrite("compressor_kv_state",
                     &NativeAttentionCache::compressor_kv_state)
      .def_readwrite("compressor_score_state",
                     &NativeAttentionCache::compressor_score_state);
  using NativeAttentionRuntime = xllm::layer::DeepseekV41AttentionRuntime;
  pybind11::class_<NativeAttentionRuntime>(module, "NativeV41AttentionRuntime")
      .def(pybind11::init<>())
      .def_readwrite("source_layer", &NativeAttentionRuntime::source_layer)
      .def_readwrite("ratio", &NativeAttentionRuntime::ratio)
      .def_readwrite("key", &NativeAttentionRuntime::key)
      .def_readwrite("key_table", &NativeAttentionRuntime::key_table)
      .def_readwrite("index", &NativeAttentionRuntime::index)
      .def_readwrite("index_table", &NativeAttentionRuntime::index_table)
      .def_readwrite("topk_layer", &NativeAttentionRuntime::topk_layer)
      .def_readwrite("topk", &NativeAttentionRuntime::topk)
      .def_readwrite("candidates_layer",
                     &NativeAttentionRuntime::candidates_layer)
      .def_readwrite("candidates", &NativeAttentionRuntime::candidates);
  using NativeAttention = xllm::layer::DeepseekV41Attention;
  pybind11::class_<NativeAttention>(module, "NativeV41Attention")
      .def(pybind11::init<NativeAttentionConfig, NativeAttentionWeights>())
      .def("forward",
           &NativeAttention::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeMoeConfig = xllm::layer::DeepseekV41MoeConfig;
  pybind11::class_<NativeMoeConfig>(module, "NativeV41MoeConfig")
      .def(pybind11::init<>())
      .def_readwrite("dense", &NativeMoeConfig::dense)
      .def_readwrite("original_fp8_activation",
                     &NativeMoeConfig::original_fp8_activation)
      .def_readwrite("first_expert", &NativeMoeConfig::first_expert)
      .def_readwrite("topk", &NativeMoeConfig::topk)
      .def_readwrite("image_token_id", &NativeMoeConfig::image_token_id)
      .def_readwrite("normalize", &NativeMoeConfig::normalize)
      .def_readwrite("scale", &NativeMoeConfig::scale)
      .def_readwrite("limit", &NativeMoeConfig::limit)
      .def_readwrite("bmm_so_path", &NativeMoeConfig::bmm_so_path)
      .def_readwrite("max_partial_bytes", &NativeMoeConfig::max_partial_bytes);
  using NativeMoeWeights = xllm::layer::DeepseekV41MoeWeights;
  pybind11::class_<NativeMoeWeights>(module, "NativeV41MoeWeights")
      .def(pybind11::init<>())
      .def_readwrite("shared", &NativeMoeWeights::shared)
      .def_readwrite("expert", &NativeMoeWeights::expert)
      .def_readwrite("scale", &NativeMoeWeights::scale)
      .def_readwrite("gate", &NativeMoeWeights::gate)
      .def_readwrite("bias", &NativeMoeWeights::bias)
      .def_readwrite("vision_bias", &NativeMoeWeights::vision_bias);
  using NativeDecoderConfig = xllm::layer::DeepseekV41DecoderConfig;
  pybind11::class_<NativeDecoderConfig>(module, "NativeV41DecoderConfig")
      .def(pybind11::init<>())
      .def_readwrite("attention", &NativeDecoderConfig::attention)
      .def_readwrite("moe", &NativeDecoderConfig::moe)
      .def_readwrite("sinkhorn_iterations",
                     &NativeDecoderConfig::sinkhorn_iterations)
      .def_readwrite("hc_epsilon", &NativeDecoderConfig::hc_epsilon);
  using NativeDecoderWeights = xllm::layer::DeepseekV41DecoderWeights;
  pybind11::class_<NativeDecoderWeights>(module, "NativeV41DecoderWeights")
      .def(pybind11::init<>())
      .def_readwrite("attention", &NativeDecoderWeights::attention)
      .def_readwrite("moe", &NativeDecoderWeights::moe)
      .def_readwrite("hyper_connection",
                     &NativeDecoderWeights::hyper_connection)
      .def_readwrite("input_norm", &NativeDecoderWeights::input_norm)
      .def_readwrite("post_attention_norm",
                     &NativeDecoderWeights::post_attention_norm)
      .def_readwrite("engram", &NativeDecoderWeights::engram);
  using NativeMoe = xllm::layer::DeepseekV41Moe;
  pybind11::class_<NativeMoe>(module, "NativeV41Moe")
      .def(pybind11::init<NativeMoeConfig, NativeMoeWeights>())
      .def("forward",
           &NativeMoe::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeDecoder = xllm::layer::DeepseekV41Decoder;
  pybind11::class_<NativeDecoder>(module, "NativeV41Decoder")
      .def(pybind11::init<NativeDecoderConfig, NativeDecoderWeights>())
      .def("forward",
           &NativeDecoder::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeTextModel = xllm::layer::DeepseekV41TextModel;
  pybind11::class_<NativeTextModel>(module, "NativeV41TextModel")
      .def(pybind11::init<std::vector<NativeDecoderConfig>,
                          std::vector<NativeDecoderWeights>,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor>())
      .def("forward",
           &NativeTextModel::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("logits",
           &NativeTextModel::logits,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeEngramHash = xllm::layer::DeepseekV41EngramHash;
  pybind11::class_<NativeEngramHash>(module, "NativeV41EngramHash")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          int64_t>())
      .def("bind_history",
           &NativeEngramHash::bind_history,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("restore_history",
           &NativeEngramHash::restore_history,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("forward",
           &NativeEngramHash::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("native_v41_checkpoint_dequant_fp8",
             &xllm::DeepseekV41Checkpoint::dequant_fp8,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "native_v41_checkpoint_fixture_load",
      [](std::unordered_map<std::string, torch::Tensor> tensors,
         int64_t rank,
         int64_t world_size) {
        std::vector<std::unique_ptr<xllm::StateDict>> shards;
        shards.reserve(1);
        shards.emplace_back(
            std::make_unique<xllm::StateDict>(std::move(tensors)));
        const torch::TensorOptions options =
            torch::TensorOptions()
                .device(torch::Device(torch::kPrivateUse1, 0))
                .dtype(torch::kBFloat16);
        xllm::DeepseekV41Checkpoint checkpoint(
            shards, options, rank, world_size);
        const int64_t count =
            checkpoint.tensor("expert_count").item<int64_t>() / world_size;
        auto [expert, scale] =
            checkpoint.experts("experts", "w1", rank * count, count);
        return std::unordered_map<std::string, torch::Tensor>{
            {"embedding", checkpoint.copy("embed.weight", torch::kBFloat16, 1)},
            {"dense", checkpoint.dense("dense")},
            {"column", checkpoint.dense("dense", 0)},
            {"row", checkpoint.dense("dense", 1)},
            {"norm", checkpoint.copy("norm", torch::kFloat32)},
            {"expert", expert},
            {"scale", scale}};
      },
      pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeEngramEmbedding = xllm::layer::DeepseekV41EngramEmbedding;
  pybind11::class_<NativeEngramEmbedding>(module, "NativeV41EngramEmbedding")
      .def(pybind11::
               init<torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t>())
      .def("local",
           &NativeEngramEmbedding::local,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("lookup",
           &NativeEngramEmbedding::lookup,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "native_v41_checkpoint_mapped_dense",
      [](const std::string& path, const std::string& prefix) {
        const auto shard = xllm::StateDictFromSafeTensor::load(path);
        CHECK(shard != nullptr);
        return xllm::DeepseekV41Checkpoint::dequant_fp8(
            shard->get_tensor(prefix + ".weight"),
            shard->get_tensor(prefix + ".scale"));
      },
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("parse_native_v41_c1_length",
             &xllm::layer::parse_deepseek_v41_c1_length);
  module.def("native_v41_cache_view", &xllm::layer::deepseek_v41_cache_view);
}
