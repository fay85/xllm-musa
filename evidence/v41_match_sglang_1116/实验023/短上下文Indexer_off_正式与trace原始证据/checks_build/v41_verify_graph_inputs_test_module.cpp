/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/layers/musa/deepseek_v41/attention.h"
#include "core/layers/musa/deepseek_v41/engram_hash.h"
#include "core/layers/musa/deepseek_v41/prefix_checkpoint.h"
PYBIND11_MODULE(v41_verify_graph_inputs_test_module, module) {
  using NativeIndexer = xllm::layer::DeepseekV41Indexer;
  module.def("v41_select_all_visible", &NativeIndexer::select_all_visible);
  pybind11::class_<NativeIndexer>(module, "NativeV41Indexer")
      .def(pybind11::init<std::vector<torch::Tensor>,
                          int64_t,
                          int64_t,
                          double,
                          bool,
                          std::string>())
      .def("select", &NativeIndexer::select)
      .def("candidates", &NativeIndexer::candidates);

  using Checkpoint = xllm::layer::DeepseekV41PrefixCheckpoint;
  pybind11::class_<Checkpoint>(module, "PrefixCheckpoint")
      .def(pybind11::init<int64_t, int64_t, int64_t>())
      .def("add_paged", &Checkpoint::add_paged)
      .def("add_recurrent", &Checkpoint::add_recurrent)
      .def("restore_suffix", &Checkpoint::restore_suffix)
      .def("saved_bytes", &Checkpoint::saved_bytes);
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
      .def_readwrite("sparse_library_path",
                     &NativeAttentionConfig::sparse_library_path)
      .def_readwrite("indexer_score_directory",
                     &NativeAttentionConfig::indexer_score_directory)
      .def_readwrite("decode_cache_read_reuse",
                     &NativeAttentionConfig::decode_cache_read_reuse)
      .def_readwrite("short_context_indexer",
                     &NativeAttentionConfig::short_context_indexer)
      .def_readwrite("decode_fused_swa_read",
                     &NativeAttentionConfig::decode_fused_swa_read)
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
      .def_readwrite("decode_capacity",
                     &NativeAttentionRuntime::decode_capacity)
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
      .def("forward_draft_block",
           &NativeAttention::forward_draft_block,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("write_draft_context",
           &NativeAttention::write_draft_context,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("forward",
           &NativeAttention::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using NativeEngramHash = xllm::layer::DeepseekV41EngramHash;
  pybind11::class_<NativeEngramHash>(module, "NativeV41EngramHash")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          int64_t>())
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          int64_t,
                          std::string>())
      .def("bind_history",
           &NativeEngramHash::bind_history,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("restore_history",
           &NativeEngramHash::restore_history,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("forward",
           &NativeEngramHash::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
}
