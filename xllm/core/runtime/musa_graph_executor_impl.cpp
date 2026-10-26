/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/runtime/musa_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <shared_mutex>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/kernels/musa/attention_runner.h"
#include "core/kernels/musa/global_capture_instance.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/linear.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_graph_policy.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_graph_sequence.h"
#include "core/layers/musa/flashinfer_planinfo.h"
#include "core/platform/cuda/device_capture_lock.h"
#include "core/platform/device.h"
#include "core/platform/musa_vmm_capture_pool.h"
#include "core/runtime/decode_graph_bucket.h"
#include "core/runtime/musa/graph_memory_policy.h"
#include "core/util/env_var.h"
#include "core/util/rec_model_utils.h"
#include "core/util/utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace xllm::runtime::musa {

namespace {

std::atomic<uint32_t> s_completed_musa_graph_captures{0};

bool s_enable_graph_timing() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_GRAPH_TIMING");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
}

// Text decode and MTP verify pass positions as [seq]. M-RoPE treats that as
// (p, p, p), the same expansion rotary_embedding_util applies at runtime.
torch::Tensor positions_for_persistent_copy(const torch::Tensor& positions,
                                            int64_t persistent_dim) {
  if (persistent_dim == 2 && positions.dim() == 1) {
    return positions.reshape({1, -1}).expand({3, positions.numel()});
  }
  return positions;
}

const char* s_graph_dump_directory() {
  const char* directory = std::getenv("XLLM_GRAPH_DUMP_DIR");
  return directory != nullptr && directory[0] != '\0' ? directory : nullptr;
}

bool s_use_fa3_decode(int64_t gqa_ratio) {
  static const int32_t setting = [] {
    const char* decode_env = std::getenv("XLLM_USE_FA3_DECODE");
    if (decode_env != nullptr) {
      return std::string(decode_env) == "1" ? int32_t{1} : int32_t{0};
    }
    const char* env = std::getenv("XLLM_USE_FA3");
    if (env == nullptr) {
      return int32_t{-1};
    }
    return std::string(env) == "1" ? int32_t{1} : int32_t{0};
  }();
  return setting < 0 ? gqa_ratio == 8 : setting == 1;
}

// Phase D: wall+device time for packed/eager pure-prefill forwards.
bool s_enable_prefill_fwd_timing() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_PREFILL_FWD_TIMING");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
}

bool s_enable_prefill_empty_cache() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_PREFILL_EMPTY_CACHE");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
}

bool s_enable_dsv4_dynamic_graph_state() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_DSV4_DYNAMIC_GRAPH_STATE");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
}

bool s_enable_dsv4_graph_persistent_reset() {
  static const bool val = [] {
    const char* env =
        std::getenv("XLLM_DSV4_CLEAR_GRAPH_PERSISTENT_ON_REQUEST");
    if (env != nullptr) {
      return std::string(env) == "1";
    }
    // Persistent graph metadata carries accepted-token and block-table state;
    // retaining it across a recycled request can change the next request's
    // q6 target logits. Clearing the small metadata carriers at prefill has no
    // decode cost and is the safe DeepSeek default.
    return true;
  }();
  return val;
}

bool s_enable_dsv4_graph_short_fullkv_eager() {
  const char* env = std::getenv("XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER");
  // Short compact verification uses the dense full-KV attention contract. The
  // Graph paged-dense fallback has different cache/position semantics for
  // these rows; preserve q2 Eager correctness while keeping long q6 Graph.
  return env == nullptr || std::string(env) == "1";
}

uint32_t dsv4_graph_shape_key(uint32_t bucket_num_tokens,
                              uint32_t actual_num_tokens,
                              const ModelArgs& args,
                              const runtime::Options& options,
                              const ModelInputParams& params,
                              bool emit_trace = true) {
  if (!s_enable_dsv4_dynamic_graph_state() ||
      args.model_type() != "deepseek_v4" || !params.is_spec_verify ||
      actual_num_tokens <= 2 || options.num_decoding_tokens() <= 1) {
    return bucket_num_tokens;
  }
  const auto& kv_lens = params.attention.host.kv_seq_lens;
  if (kv_lens.size() < 2 || kv_lens.front() != 0) {
    return bucket_num_tokens;
  }
  const int64_t query_width = options.num_decoding_tokens();
  const char* stable_sequences_env =
      std::getenv("XLLM_DSV4_STABLE_MULTI_GRAPH_SEQUENCES");
  const bool stable_sequences =
      stable_sequences_env != nullptr &&
      std::string(stable_sequences_env) == "1" &&
      layer::dsv4_reusable_graph_capacity(kv_lens, query_width) > 0;
  if (layer::dsv4_graph_sequence_views_enabled() &&
      actual_num_tokens > query_width && !stable_sequences) {
    CHECK_EQ(actual_num_tokens % query_width, 0);
    const int64_t sequences = actual_num_tokens / query_width;
    CHECK_EQ(kv_lens.size(), sequences + 1)
        << "Multi-sequence Graph key requires per-request cumulative KV "
           "lengths";
    const auto enabled = [](const char* name) {
      const char* value = std::getenv(name);
      return value != nullptr && std::string(value) == "1";
    };
    const bool dynamic_short = enabled("XLLM_DSV4_DYNAMIC_SHORT_GRAPH_KEYS");
    const bool global_short_family =
        enabled("XLLM_DSV4_GLOBAL_SHORT_GRAPH_FAMILY");
    CHECK(!global_short_family || dynamic_short)
        << "Global short Graph family requires validated dynamic short keys";
    if (dynamic_short) {
      CHECK(query_width == 7 && args.window_size() == 128)
          << "Dynamic short Graph keys require N6/q7 and a128-token SWA window";
      for (const char* name : {"XLLM_DSV4_DYNAMIC_GRAPH_STATE",
                               "XLLM_DSV4_GRAPH_CAUSAL_SHORT_C4",
                               "XLLM_DSV4_C128_MATE_VERIFY",
                               "XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY",
                               "XLLM_DSV4_MATE_GROUPED_METADATA",
                               "XLLM_DSV4_BATCH_MATE_GROUPED_METADATA",
                               "XLLM_DSV4_BATCH_OUTPUT_PROJECTION",
                               "XLLM_DSV4_BATCH_ATTENTION_PROJECTIONS",
                               "XLLM_DSV4_BATCH_INDEXER_ACROSS_SEQUENCES",
                               "XLLM_DSV4_BATCH_INDEXER_GEMM",
                               "XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE"}) {
        CHECK(enabled(name)) << "Dynamic short Graph key requires " << name;
      }
      for (const char* name : {"XLLM_DSV4_DISABLE_PAGED_DENSE_DECODE",
                               "XLLM_DSV4_DISABLE_FUSED_DENSE_DECODE",
                               "XLLM_DSV4_DISABLE_MATE_SPARSE_DECODE",
                               "XLLM_DSV4_MATE_BATCHED_STATIC_LENGTHS",
                               "XLLM_DSV4_DISABLE_GRAPH_COMPACT_INDEXER"}) {
        CHECK(!enabled(name))
            << "Dynamic short Graph key is incompatible with " << name;
      }
      if (emit_trace) {
        LOG_FIRST_N(INFO, 1) << "[DSV4_SHORT_GRAPH_KEYS] "
                                "exact_short_kv_in_key=0 live_causal_lengths=1";
      }
    }
    const int64_t short_batch_max =
        global_short_family
            ? layer::dsv4_graph_short_batch_max_kv(kv_lens, query_width)
            : 0;
    if (short_batch_max > 0 && emit_trace) {
      LOG_FIRST_N(INFO, 1)
          << "[DSV4_GLOBAL_SHORT_GRAPH_FAMILY] max_kv_capacity=2048 "
             "per_row_validity=live batch_wide_attention_branch=1";
    }
    // Exact structural signatures, not a lossy packed/hash key. Fixed
    // per-request compressor capacities make C4/C128 emission phases dynamic.
    // The guarded direct short route uses a batch-wide attention branch.
    // Other families retain per-row branches and indexer capacities.
    std::vector<int64_t> shape{bucket_num_tokens,
                               query_width,
                               sequences,
                               args.window_size(),
                               layer::dsv4_full_kv_threshold(),
                               short_batch_max > 0};
    for (int64_t seq = 0; seq < sequences; ++seq) {
      const int64_t len = kv_lens[seq + 1] - kv_lens[seq];
      CHECK_GE(len, query_width);
      const bool full =
          layer::dsv4_uses_full_kv(short_batch_max > 0 ? short_batch_max : len,
                                   layer::dsv4_full_kv_threshold(),
                                   args.window_size());
      shape.push_back(full);
      shape.push_back(
          layer::dsv4_graph_indexer_capacity(len / 4, /*multi_sequence=*/true) /
          512);
      shape.push_back(
          layer::dsv4_graph_baked_kv_length(len, full, dynamic_short));
    }
    static thread_local std::map<std::vector<int64_t>, uint32_t> signatures;
    auto [entry, inserted] = signatures.try_emplace(shape, 0);
    if (inserted) {
      CHECK_LT(signatures.size(), uint32_t{1} << 30);
      entry->second = (uint32_t{1} << 30) | signatures.size();
      if (emit_trace) {
        LOG(INFO) << "[DSV4_MULTI_GRAPH_KEY] key=" << entry->second
                  << " sequences=" << sequences
                  << " query_width=" << query_width;
      }
    }
    const char* trace = std::getenv("XLLM_DSV4_GRAPH_KEY_TRACE");
    if (emit_trace && trace != nullptr && std::string(trace) == "1") {
      LOG(INFO) << "[DSV4_MULTI_GRAPH_ROUTE] key=" << entry->second
                << " actual_tokens=" << actual_num_tokens
                << " bucket_tokens=" << bucket_num_tokens
                << " sequences=" << sequences << " query_width=" << query_width;
    }
    return entry->second;
  }
  if (stable_sequences && layer::dsv4_graph_sequence_views_enabled() &&
      actual_num_tokens > query_width && actual_num_tokens % query_width == 0) {
    const int64_t active_sequences = actual_num_tokens / query_width;
    const int64_t bucket_sequences = bucket_num_tokens / query_width;
    if (bucket_sequences >= active_sequences &&
        bucket_sequences <= options.max_seqs_per_batch()) {
      int64_t max_kv_len = 0;
      for (size_t index = 1; index < kv_lens.size(); ++index) {
        max_kv_len =
            std::max<int64_t>(max_kv_len, kv_lens[index] - kv_lens[index - 1]);
      }
      const bool full = layer::dsv4_uses_full_kv(
          max_kv_len, layer::dsv4_full_kv_threshold(), args.window_size());
      std::vector<int64_t> shape{
          bucket_num_tokens,
          query_width,
          bucket_sequences,
          args.window_size(),
          layer::dsv4_full_kv_threshold(),
          layer::dsv4_graph_indexer_capacity(max_kv_len / 4,
                                             /*multi_sequence=*/true) /
              512,
          full || max_kv_len / 128 > 512 ? max_kv_len : 0};
      static thread_local std::map<std::vector<int64_t>, uint32_t>
          stable_signatures;
      auto [entry, inserted] = stable_signatures.try_emplace(shape, 0);
      if (inserted) {
        CHECK_LT(stable_signatures.size(), uint32_t{1} << 30);
        entry->second = (uint32_t{1} << 29) | stable_signatures.size();
        if (emit_trace) {
          LOG(INFO) << "[DSV4_STABLE_MULTI_GRAPH_KEY] key=" << entry->second
                    << " active_sequences=" << active_sequences
                    << " graph_sequences=" << bucket_sequences;
        }
      }
      const char* trace = std::getenv("XLLM_DSV4_GRAPH_KEY_TRACE");
      if (emit_trace && trace != nullptr && std::string(trace) == "1") {
        LOG(INFO) << "[DSV4_STABLE_MULTI_GRAPH_ROUTE] key=" << entry->second
                  << " actual_tokens=" << actual_num_tokens
                  << " bucket_tokens=" << bucket_num_tokens
                  << " active_sequences=" << active_sequences
                  << " graph_sequences=" << bucket_sequences;
        // Report real active sequences for the throughput auditor. Padded
        // graph rows must never count as a physically active B16 request set.
        LOG(INFO) << "[DSV4_MULTI_GRAPH_ROUTE] key=" << entry->second
                  << " actual_tokens=" << actual_num_tokens
                  << " bucket_tokens=" << bucket_num_tokens
                  << " sequences=" << active_sequences
                  << " query_width=" << query_width;
      }
      return entry->second;
    }
  }
  const int64_t start_position = static_cast<int64_t>(kv_lens[1]) - query_width;
  if (start_position < 0) {
    return bucket_num_tokens;
  }
  constexpr int64_t kCompressRatio = 4;
  const int64_t first_boundary =
      (kCompressRatio - ((start_position + 1) % kCompressRatio)) %
      kCompressRatio;
  const int64_t boundary_count =
      first_boundary < query_width
          ? 1 + (query_width - 1 - first_boundary) / kCompressRatio
          : 0;
  // The fused indexer TopK path now uses the replay-time device KV length,
  // but its score workspace is captured at a bounded capacity. Partition
  // Graphs when that capacity grows so a long request does not exceed the
  // first short-prefix capture's fixed score buffer.
  const int64_t indexer_rows = std::max<int64_t>(kv_lens[1] / 4, 1);
  const int64_t indexer_capacity_bucket = (indexer_rows + 511) / 512;
  const char* dynamic_indexer_env =
      std::getenv("XLLM_DSV4_GRAPH_DYNAMIC_INDEXER");
  const bool dynamic_indexer =
      dynamic_indexer_env != nullptr && std::string(dynamic_indexer_env) == "1";
  // Keep graph storage keyed by both query bucket and the fixed C4 output
  // capacity. The captured graph cannot change tensor shapes on replay, while
  // the absolute position itself is supplied through persistent device input.
  CHECK_GE(boundary_count, 0);
  CHECK_LT(boundary_count, 4);
  const uint32_t c4_outputs = static_cast<uint32_t>(boundary_count);
  const int64_t c128_count = layer::dsv4_compression_boundaries(
      start_position, query_width, /*ratio=*/128);
  CHECK_GE(c128_count, 0);
  CHECK_LT(c128_count, 4);
  const bool full_kv = layer::dsv4_uses_full_kv(
      kv_lens[1], layer::dsv4_full_kv_threshold(), args.window_size());
  // Separate namespace from ordinary decode buckets. Never silently clamp a
  // capacity: that would permit replay beyond the captured score workspace.
  CHECK(!dynamic_indexer ||
        (bucket_num_tokens < 4096 && indexer_capacity_bucket < 1024));
  const uint32_t key = layer::dsv4_compact_graph_key(
      bucket_num_tokens,
      dynamic_indexer ? static_cast<uint32_t>(indexer_capacity_bucket) : 0,
      c4_outputs,
      static_cast<uint32_t>(c128_count),
      full_kv);
  const char* trace_env = std::getenv("XLLM_DSV4_GRAPH_KEY_TRACE");
  if (trace_env != nullptr && std::string(trace_env) == "1") {
    const int32_t kv0 = kv_lens.empty() ? -1 : kv_lens.front();
    const int32_t kv1 = kv_lens.size() < 2 ? -1 : kv_lens[1];
    LOG(INFO) << "[DSV4_GRAPH_KEY] bucket=" << bucket_num_tokens
              << " actual_tokens=" << actual_num_tokens
              << " query_width=" << query_width << " kv0=" << kv0
              << " kv1=" << kv1 << " start=" << start_position
              << " first_boundary=" << first_boundary
              << " boundary_count=" << boundary_count << " key=" << key
              << " c128_outputs=" << c128_count << " full_kv=" << full_kv
              << " num_sequences=" << params.meta.num_sequences
              << " actual_sequences=" << params.meta.actual_num_sequences;
  }
  return key;
}

void maybe_empty_prefill_cache() {
  // The caching allocator only releases already-free blocks here. Reclaiming
  // them unconditionally adds driver work to the first-token path without
  // affecting the lifetime of the forward result or live KV-cache tensors.
  // Keep reclamation as an explicit operator/recovery policy.
  if (!s_enable_prefill_empty_cache()) {
    return;
  }
  if (!s_enable_prefill_fwd_timing()) {
    Device::empty_cache(/*device_index=*/-1);
    return;
  }

  const auto start = std::chrono::steady_clock::now();
  Device::empty_cache(/*device_index=*/-1);
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  LOG(INFO) << "[PREFILL_POST] empty_cache_ms=" << elapsed_ms;
}

// Host lengths are either per sequence or cumulative with a leading zero.
// Returns the last sequence length after appending padding to it.
int32_t extend_last_sequence_length(std::vector<int32_t>& lengths,
                                    int32_t num_sequences,
                                    int32_t padding) {
  if (lengths.empty()) {
    return 0;
  }
  lengths.back() += padding;
  const bool cumulative =
      static_cast<int32_t>(lengths.size()) == num_sequences + 1 &&
      lengths.front() == 0;
  if (cumulative) {
    return lengths.back() - lengths[lengths.size() - 2];
  }
  return lengths.back();
}

struct GraphPoolMemoryUsage {
  size_t reserved_bytes = 0;
  size_t allocated_bytes = 0;
  size_t active_bytes = 0;
};

GraphPoolMemoryUsage get_graph_pool_memory_usage(
    c10::DeviceIndex device_index,
    const c10::musa::MempoolId_t& pool_id) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = c10::musa::MUSACachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index ||
        segment.owner_private_pool_id != pool_id) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
  }
  return usage;
}

GraphPoolMemoryUsage get_private_pools_memory_usage(
    c10::DeviceIndex device_index) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = c10::musa::MUSACachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index) {
      continue;
    }
    if (segment.owner_private_pool_id == c10::musa::MempoolId_t{0, 0}) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
  }
  return usage;
}

size_t get_allocator_reserved_bytes(c10::DeviceIndex device_index) {
  const auto device_stats =
      c10::musa::MUSACachingAllocator::getDeviceStats(device_index);
  const size_t stat_index =
      static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return static_cast<size_t>(device_stats.reserved_bytes[stat_index].current);
}

bool is_contiguous_int_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_contiguous()) {
    return false;
  }
  const auto sc = tensor.scalar_type();
  if (sc != torch::kInt32 && sc != torch::kInt64) {
    return false;
  }
  return tensor.device().type() == at::musa::kMUSA;
}

bool is_cpu_int32_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || !tensor.device().is_cpu() || tensor.numel() == 0 ||
      !tensor.is_contiguous()) {
    return false;
  }
  return tensor.scalar_type() == torch::kInt32;
}

bool has_llm_decode_host_metadata(const AttentionHostInput& host) {
  return !host.kv_seq_lens.empty() &&
         is_cpu_int32_tensor(host.paged_kv_indptr) &&
         is_cpu_int32_tensor(host.paged_kv_indices) &&
         is_cpu_int32_tensor(host.paged_kv_last_page_len);
}

// Device-resident decode metadata (cumulative kv_seq_lens with B+1 entries
// plus paged-KV tensors). Produced by the MTP device-target-context draft,
// whose host metadata is only a conservative fixed-shape template.
bool has_llm_decode_device_metadata(const AttentionDeviceInput& device,
                                    const AttentionHostInput& host,
                                    int64_t actual_batch_size) {
  if (actual_batch_size < 0 || !is_contiguous_int_tensor(device.kv_seq_lens) ||
      !is_contiguous_int_tensor(device.paged_kv_indptr) ||
      !is_contiguous_int_tensor(device.paged_kv_indices) ||
      !is_contiguous_int_tensor(device.paged_kv_last_page_len)) {
    return false;
  }
  const int64_t cumulative_size = actual_batch_size + 1;
  if (device.kv_seq_lens.numel() != cumulative_size ||
      device.paged_kv_indptr.numel() != cumulative_size ||
      device.paged_kv_last_page_len.numel() != actual_batch_size) {
    return false;
  }
  if (static_cast<int64_t>(host.kv_seq_lens.size()) != cumulative_size ||
      host.kv_seq_lens.front() != 0 || !has_llm_decode_host_metadata(host) ||
      host.paged_kv_indptr.numel() != cumulative_size) {
    return false;
  }
  const int32_t indices_size =
      host.paged_kv_indptr.data_ptr<int32_t>()[actual_batch_size];
  return indices_size >= 0 && indices_size <= host.paged_kv_indices.numel() &&
         indices_size <= device.paged_kv_indices.numel();
}

struct IndexedTensorSnapshot {
  torch::Tensor target;
  torch::Tensor rows;
  torch::Tensor indices;
};

torch::Tensor get_linear_state_snapshot_indices(const ModelInputParams& params,
                                                const torch::Device& device) {
  if (params.embedding.linear_state_indices.defined()) {
    return params.embedding.linear_state_indices.to(
        torch::TensorOptions().dtype(torch::kLong).device(device));
  }
  CHECK(!params.embedding.linear_state_ids.empty())
      << "MUSA graph capture requires linear state ids";
  return torch::tensor(
      params.embedding.linear_state_ids,
      torch::TensorOptions().dtype(torch::kLong).device(device));
}

std::vector<IndexedTensorSnapshot> snapshot_linear_attention_state(
    std::vector<KVCache>& kv_caches,
    const torch::Tensor& indices) {
  std::vector<IndexedTensorSnapshot> snapshots;
  snapshots.reserve(kv_caches.size() * 2);
  for (KVCache& kv_cache : kv_caches) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    if (conv_cache.defined() && conv_cache.numel() > 0) {
      snapshots.emplace_back(IndexedTensorSnapshot{
          conv_cache,
          torch::index_select(conv_cache, /*dim=*/0, indices),
          indices});
    }
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (ssm_cache.defined() && ssm_cache.numel() > 0) {
      // MTP keeps K+1 SSM checkpoints per conv row; the spec-verify forward
      // rewrites all of them, so snapshot every checkpoint slot of the row.
      torch::Tensor ssm_indices = indices;
      if (conv_cache.defined() && conv_cache.size(0) > 0 &&
          ssm_cache.size(0) > conv_cache.size(0) &&
          ssm_cache.size(0) % conv_cache.size(0) == 0) {
        const int64_t stride = ssm_cache.size(0) / conv_cache.size(0);
        ssm_indices = (indices.unsqueeze(1) * stride +
                       torch::arange(stride, indices.options()).unsqueeze(0))
                          .reshape({-1})
                          .contiguous();
      }
      snapshots.emplace_back(IndexedTensorSnapshot{
          ssm_cache,
          torch::index_select(ssm_cache, /*dim=*/0, ssm_indices),
          ssm_indices});
    }
  }
  return snapshots;
}

void restore_linear_attention_state(
    std::vector<IndexedTensorSnapshot>& snapshots,
    const torch::Tensor& indices) {
  for (IndexedTensorSnapshot& snapshot : snapshots) {
    (void)indices;
    snapshot.target.index_copy_(/*dim=*/0, snapshot.indices, snapshot.rows);
  }
}

bool should_capture_graph_aux_hidden_states(const ModelArgs& args,
                                            const runtime::Options& options) {
  // The DeepSeek-V4 target graph exports pre-HC lanes to MTP, while its draft
  // graph consumes those lanes and does not feed an auxiliary result back.
  // Keep the generic option intact for other models/backends.
  return options.enable_graph_aux_hidden_states() &&
         !(options.is_draft_engine() && args.model_type() == "deepseek_v4_mtp");
}

}  // namespace

// MusaGraphPersistentParam implementation
MusaGraphPersistentParam::MusaGraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : args_(args), device_(device), options_(options) {
  // Use max_tokens_per_batch for first dimension size
  const int64_t max_tokens_per_batch = options.max_tokens_per_batch();
  // Round the sequence capacity to the same bucket used by graph execution so
  // the largest configured decode batch can be padded safely.
  int64_t max_seqs_per_batch;
  if (is_rec_multi_round_mode()) {
    // max_seqs_per_batch is the max sequence count per Batch in a scheduler
    // group.
    // When is_rec_multi_round_mode() == true, multiply by beam_width to account
    // for beam search.
    max_seqs_per_batch = options.max_seqs_per_batch() * options_.beam_width();
  } else {
    max_seqs_per_batch = options.max_seqs_per_batch();
  }
  // Qwen3.5 MTP draft can emit two attention rows per live request (repair +
  // current) after a fully accepted draft. Size persistent per-sequence
  // metadata for both rows (feihu-1016-dev). Without this, graph capture for
  // bucket>max_seqs silently truncates block_tables to max_seqs rows while
  // the forward still runs on all token rows, collapsing MTP accept at C>1.
  if (args_.model_type() == "qwen3_5_mtp" ||
      args_.model_type() == "qwen3_5_moe_mtp") {
    max_seqs_per_batch *= 2;
  }
  max_seqs_per_batch = ::xllm::runtime::get_decode_graph_token_bucket(
      max_seqs_per_batch, options.enable_graph_mode_decode_no_padding());
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  // Qwen VL M-RoPE positions are [3, seq] only in captured prefill graphs.
  // Decode batches carry the [seq] positions of text, which the fused decode
  // metadata update requires.
  const auto& execution_config = ::xllm::ExecutionConfig::get_instance();
  const bool mrope_prefill_graphs =
      execution_config.enable_prefill_piecewise_graph() &&
      (options_.mrope_positions() ||
       (options_.backend() == "vlm" &&
        !args_.rope_scaling_mrope_section().empty()));
  if (mrope_prefill_graphs) {
    persistent_positions_ = torch::zeros(
        {3, max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  } else {
    persistent_positions_ = torch::zeros(
        {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  }
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_linear_state_indices_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_kv_cache_tokens_nums_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_num_accepted_tokens_ = torch::ones(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // q_seq_lens is q_cu_seq_lens in GPU Model.
  // kv_seq_lens is kv_cu_seq_lens in GPU Model.
  q_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::full({max_seqs_per_batch, max_block_table_len},
                  -1,
                  torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "MUSA graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  if (should_capture_graph_aux_hidden_states(args_, options_) &&
      args_.model_type() == "deepseek_v4" && args_.hc_mult() > 1) {
    // MTP consumes pre-HC lanes; DSpark consumes selected layer outputs.
    // Allocate the matching auxiliary width before capture. Allocate the
    // decode-only output before capture: lazily calling torch::zeros from
    // set_aux_hidden_states() is not permitted by the MUSA graph runtime.
    const int64_t max_aux_tokens =
        std::max<int64_t>(1, max_seqs_per_batch) *
        std::max<int64_t>(1, options_.num_speculative_tokens() + 1);
    const int64_t auxiliary_width =
        args_.layers_to_capture().empty()
            ? args_.hc_mult()
            : static_cast<int64_t>(args_.layers_to_capture().size());
    aux_hidden_states_ =
        torch::zeros({max_aux_tokens, args_.hidden_size() * auxiliary_width},
                     torch::dtype(dtype).device(device));
  }

  // FlashInfer decode mode parameters
  // paged_kv_indptr: shape [max_seqs_per_batch + 1]
  persistent_paged_kv_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

  // paged_kv_indices: maximum size based on max blocks
  // Estimate max blocks: max_seqs_per_batch * max_block_table_len
  const int64_t max_paged_kv_indices_size =
      max_seqs_per_batch * max_block_table_len;
  persistent_paged_kv_indices_ = torch::zeros(
      {max_paged_kv_indices_size}, torch::dtype(torch::kInt).device(device));

  // paged_kv_last_page_len: shape [max_seqs_per_batch]
  persistent_paged_kv_last_page_len_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // For decode mode, each sequence has 1 token, so qo_indptr = [0, 1, 2, ...,
  // max_seqs_per_batch]
  persistent_decode_qo_indptr_ = torch::arange(
      0, max_seqs_per_batch + 1, torch::dtype(torch::kInt).device(device));
  persistent_kv_seq_lens_delta_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));
  {
    // Expanded spec-verify rows: up to (K+1) validate tokens per sequence.
    const int64_t expanded_rows = max_seqs_per_batch * 8;
    auto int_opts = torch::dtype(torch::kInt).device(device);
    persistent_expanded_kv_seq_lens_ = torch::ones({expanded_rows}, int_opts);
    persistent_expanded_block_tables_ =
        torch::zeros({expanded_rows, max_block_table_len}, int_opts);
    persistent_expanded_paged_kv_indptr_ =
        torch::zeros({expanded_rows + 1}, int_opts);
    persistent_expanded_paged_kv_indices_ =
        torch::zeros({expanded_rows * max_block_table_len}, int_opts);
    persistent_expanded_paged_kv_last_page_len_ =
        torch::ones({expanded_rows}, int_opts);
    persistent_expanded_qo_indptr_ =
        torch::arange(0, expanded_rows + 1, int_opts);
    persistent_linear_state_validity_ = torch::ones(
        {max_seqs_per_batch}, torch::dtype(torch::kBool).device(device));
  }
  // aux_hidden_states will be lazily initialized when needed
}

std::optional<ModelInputParams> MusaGraphPersistentParam::update_prefill(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params) {
  const uint32_t actual_num_tokens = tokens.size(/*dim=*/0);
  const int64_t actual_batch_size = params.meta.actual_num_sequences > 0
                                        ? params.meta.actual_num_sequences
                                        : params.meta.num_sequences;
  CHECK_GE(padded_num_tokens, actual_num_tokens)
      << "prefill graph bucket is smaller than the live token count";
  CHECK_LE(padded_num_tokens, persistent_tokens_.size(0))
      << "prefill graph tokens exceed persistent capacity";
  CHECK_LT(actual_batch_size, q_seq_lens_.size(0))
      << "prefill graph sequences exceed persistent capacity";

  // Runners read this metadata on replay. It is built from the live inputs.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params, args_.enable_mla()));
  CHECK(attn_metadata) << "attn_metadata should not be null";
  attn_metadata->fa3_metadata.share_fa3_scheduler_metadata = true;

  persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(tokens, /*non_blocking=*/true);
  const torch::Tensor positions_to_copy =
      positions_for_persistent_copy(positions, persistent_positions_.dim());
  const int64_t position_token_dim = persistent_positions_.dim() - 1;
  CHECK_EQ(positions_to_copy.dim(), persistent_positions_.dim());
  persistent_positions_
      .slice(position_token_dim, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(positions_to_copy, /*non_blocking=*/true);

  const AttentionDeviceInput& device_input = params.attention.device;
  const int64_t cu_len = actual_batch_size + 1;
  q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/cu_len)
      .copy_(device_input.q_seq_lens.slice(/*dim=*/0, /*start=*/0, cu_len),
             /*non_blocking=*/true);
  kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/cu_len)
      .copy_(device_input.kv_seq_lens.slice(/*dim=*/0, /*start=*/0, cu_len),
             /*non_blocking=*/true);
  persistent_new_cache_slots_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(device_input.new_cache_slots, /*non_blocking=*/true);
  if (padded_num_tokens > actual_num_tokens) {
    // Padded rows run through the captured forward. A negative slot makes the
    // KV cache write skip them, and zero tokens keep their activations finite.
    persistent_tokens_
        .slice(
            /*dim=*/0, /*start=*/actual_num_tokens, /*end=*/padded_num_tokens)
        .zero_();
    persistent_positions_
        .slice(position_token_dim,
               /*start=*/actual_num_tokens,
               /*end=*/padded_num_tokens)
        .zero_();
    persistent_new_cache_slots_
        .slice(
            /*dim=*/0, /*start=*/actual_num_tokens, /*end=*/padded_num_tokens)
        .fill_(-1);
  }

  torch::Tensor block_tables;
  if (device_input.block_tables.defined()) {
    const int64_t block_table_len = device_input.block_tables.size(1);
    CHECK_LE(block_table_len, persistent_block_tables_.size(1))
        << "prefill graph block table exceeds persistent capacity";
    block_tables = persistent_block_tables_
                       .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
                       .slice(/*dim=*/1, /*start=*/0, /*end=*/block_table_len);
    block_tables.copy_(device_input.block_tables.slice(
                           /*dim=*/0, /*start=*/0, actual_batch_size),
                       /*non_blocking=*/true);
  }
  if (device_input.kv_cache_tokens_nums.defined()) {
    persistent_kv_cache_tokens_nums_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
        .copy_(device_input.kv_cache_tokens_nums.slice(
                   /*dim=*/0, /*start=*/0, actual_batch_size),
               /*non_blocking=*/true);
  }

  const int64_t linear_state_rows =
      static_cast<int64_t>(params.embedding.linear_state_ids.size());
  if (linear_state_rows > 0) {
    CHECK_LE(linear_state_rows, persistent_linear_state_indices_.numel());
    torch::Tensor linear_state_indices = persistent_linear_state_indices_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/linear_state_rows);
    if (params.embedding.linear_state_indices.defined()) {
      linear_state_indices.copy_(params.embedding.linear_state_indices.slice(
                                     /*dim=*/0, /*start=*/0, linear_state_rows),
                                 /*non_blocking=*/true);
    } else {
      linear_state_indices.copy_(
          torch::tensor(params.embedding.linear_state_ids, torch::kInt)
              .to(device_),
          /*non_blocking=*/true);
    }
  }
  const int64_t validity_rows =
      static_cast<int64_t>(params.linear_state_validity_mask.size());
  if (validity_rows > 0) {
    CHECK_LE(validity_rows, persistent_linear_state_validity_.numel());
    persistent_linear_state_validity_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/validity_rows)
        .copy_(torch::tensor(params.linear_state_validity_mask, torch::kLong)
                   .to(torch::kBool),
               /*non_blocking=*/true);
  }

  const torch::Tensor& embedding = params.embedding.input_embedding;
  if (embedding.defined()) {
    if (persistent_embedding_.numel() == 0) {
      persistent_embedding_ =
          torch::zeros({options_.max_tokens_per_batch(), embedding.size(1)},
                       embedding.options().device(device_));
    }
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding.size(0))
        .copy_(embedding, /*non_blocking=*/true);
    if (static_cast<int64_t>(padded_num_tokens) > embedding.size(0)) {
      persistent_embedding_
          .slice(/*dim=*/0,
                 /*start=*/embedding.size(0),
                 /*end=*/padded_num_tokens)
          .zero_();
    }
  }

  // The Qwen3.5 MTP draft prefill reads its token embedding from outside the
  // captured region, as in the decode update.
  const torch::Tensor& mtp_token_embedding =
      params.embedding.mtp_token_embedding;
  if (mtp_token_embedding.defined()) {
    if (!persistent_mtp_token_embedding_.defined() ||
        persistent_mtp_token_embedding_.numel() == 0) {
      persistent_mtp_token_embedding_ = torch::zeros(
          {options_.max_tokens_per_batch(), mtp_token_embedding.size(1)},
          mtp_token_embedding.options().device(device_));
    }
    const int64_t mtp_rows = mtp_token_embedding.size(0);
    persistent_mtp_token_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/mtp_rows)
        .copy_(mtp_token_embedding, /*non_blocking=*/true);
    if (static_cast<int64_t>(padded_num_tokens) > mtp_rows) {
      persistent_mtp_token_embedding_
          .slice(/*dim=*/0, /*start=*/mtp_rows, /*end=*/padded_num_tokens)
          .zero_();
    }
  }

  if (!return_capture_params) {
    return std::nullopt;
  }

  ModelInputParams params_for_capture = params;
  params_for_capture.attn_metadata = attn_metadata;
  if (padded_num_tokens > actual_num_tokens) {
    // Captured launches size rows and host scalars from the padded count,
    // while the persistent cumulative lengths keep the live rows.
    const int32_t padding =
        static_cast<int32_t>(padded_num_tokens - actual_num_tokens);
    const int32_t num_sequences = static_cast<int32_t>(actual_batch_size);
    AttentionHostInput& capture_host = params_for_capture.attention.host;
    const int32_t last_q_len = extend_last_sequence_length(
        capture_host.q_seq_lens, num_sequences, padding);
    const int32_t last_kv_len = extend_last_sequence_length(
        capture_host.kv_seq_lens, num_sequences, padding);
    params_for_capture.meta.q_max_seq_len =
        std::max(params_for_capture.meta.q_max_seq_len, last_q_len);
    params_for_capture.meta.kv_max_seq_len =
        std::max(params_for_capture.meta.kv_max_seq_len, last_kv_len);
    if (!capture_host.q_cu_seq_lens.empty()) {
      capture_host.q_cu_seq_lens.back() += padding;
    }
    if (!capture_host.kv_cu_seq_lens.empty()) {
      capture_host.kv_cu_seq_lens.back() += padding;
    }
  }
  AttentionDeviceInput& capture_device = params_for_capture.attention.device;
  capture_device.q_seq_lens = q_seq_lens(static_cast<uint32_t>(cu_len));
  capture_device.kv_seq_lens = kv_seq_lens(static_cast<uint32_t>(cu_len));
  capture_device.new_cache_slots =
      persistent_new_cache_slots(padded_num_tokens);
  // Captured paged attention replays every later history length, so it reads
  // the full-width table; launches only visit the blocks of the live keys.
  if (block_tables.defined()) {
    capture_device.block_tables = persistent_block_tables_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
  }
  if (device_input.kv_cache_tokens_nums.defined()) {
    capture_device.kv_cache_tokens_nums = persistent_kv_cache_tokens_nums(
        static_cast<uint32_t>(actual_batch_size));
  }
  // Prefill attention runs outside the captured segments. Leaving the paged
  // index tensors undefined keeps the in-forward metadata build free of
  // device-to-host copies.
  capture_device.paged_kv_indptr = torch::Tensor();
  capture_device.paged_kv_indices = torch::Tensor();
  capture_device.paged_kv_last_page_len = torch::Tensor();
  if (linear_state_rows > 0) {
    params_for_capture.embedding.linear_state_indices =
        persistent_linear_state_indices(
            static_cast<uint32_t>(linear_state_rows));
  }
  if (validity_rows > 0) {
    params_for_capture.linear_state_validity_mask_tensor =
        persistent_linear_state_validity_.slice(
            /*dim=*/0, /*start=*/0, /*end=*/validity_rows);
  }
  if (embedding.defined()) {
    params_for_capture.embedding.input_embedding =
        persistent_embedding(padded_num_tokens);
  }
  if (mtp_token_embedding.defined()) {
    params_for_capture.embedding.mtp_token_embedding =
        persistent_mtp_token_embedding(padded_num_tokens);
  }
  return params_for_capture;
}

bool MusaGraphPersistentParam::can_use_llm_decode_fast_path(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params) const {
  // The flat int32 update kernel only matches text positions. VLM M-RoPE
  // positions are [3, seq] and use the rank-aware copy below.
  if (positions.dim() != 1 || persistent_positions_.dim() != 1 ||
      !params.meta.batch_forward_type.is_decode() ||
      is_rec_multi_round_mode() || params.has_llmrec_params()) {
    return false;
  }
  const bool device_token_metadata_ok =
      is_contiguous_int_tensor(tokens) && is_contiguous_int_tensor(positions) &&
      is_contiguous_int_tensor(params.attention.device.new_cache_slots);
  if (!device_token_metadata_ok) {
    return false;
  }
  if (has_llm_decode_host_metadata(params.attention.host)) {
    return true;
  }
  return is_contiguous_int_tensor(params.attention.device.kv_seq_lens) &&
         is_contiguous_int_tensor(params.attention.device.paged_kv_indptr) &&
         is_contiguous_int_tensor(params.attention.device.paged_kv_indices) &&
         is_contiguous_int_tensor(
             params.attention.device.paged_kv_last_page_len);
}

void MusaGraphPersistentParam::update_llm_decode_metadata_fast_path(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    int64_t actual_batch_size,
    int64_t actual_num_tokens) {
  CHECK_GE(actual_batch_size, 0) << "actual_batch_size must be >= 0";
  CHECK_GE(actual_num_tokens, 0) << "actual_num_tokens must be >= 0";

  auto to_int32 = [](const torch::Tensor& t) -> torch::Tensor {
    if (!t.defined() || t.scalar_type() == torch::kInt32) {
      return t;
    }
    return t.to(torch::kInt32);
  };
  const torch::Tensor tokens_i32 = to_int32(tokens);
  const torch::Tensor positions_i32 = to_int32(positions);
  const torch::Tensor new_cache_slots_i32 =
      to_int32(params.attention.device.new_cache_slots);
  const torch::Tensor kv_seq_lens_i32 =
      to_int32(params.attention.device.kv_seq_lens);

  const musaStream_t stream = c10::musa::getCurrentMUSAStream(device_.index());
  const bool use_device_attention_metadata = has_llm_decode_device_metadata(
      params.attention.device, params.attention.host, actual_batch_size);
  if (!use_device_attention_metadata &&
      has_llm_decode_host_metadata(params.attention.host)) {
    const auto& host = params.attention.host;
    CHECK_GE(static_cast<int64_t>(host.kv_seq_lens.size()),
             actual_batch_size + 1)
        << "host kv_seq_lens too small for batch";
    const int64_t actual_indices_size =
        host.paged_kv_indices.defined() ? host.paged_kv_indices.numel() : 0;
    xllm::kernel::musa::LlmDecodeMetadataHostUpdateParams host_update_params{
        .src_tokens = tokens_i32.data_ptr<int32_t>(),
        .src_positions = positions_i32.data_ptr<int32_t>(),
        .src_new_cache_slots = new_cache_slots_i32.data_ptr<int32_t>(),
        .host_kv_seq_lens = host.kv_seq_lens.data(),
        .host_paged_kv_indptr = host.paged_kv_indptr.data_ptr<int32_t>(),
        .host_paged_kv_indices = host.paged_kv_indices.data_ptr<int32_t>(),
        .host_paged_kv_last_page_len =
            host.paged_kv_last_page_len.data_ptr<int32_t>(),
        .dst_tokens = persistent_tokens_.data_ptr<int32_t>(),
        .dst_positions = persistent_positions_.data_ptr<int32_t>(),
        .dst_new_cache_slots = persistent_new_cache_slots_.data_ptr<int32_t>(),
        .dst_kv_seq_lens = kv_seq_lens_.data_ptr<int32_t>(),
        .dst_kv_seq_lens_delta =
            persistent_kv_seq_lens_delta_.data_ptr<int32_t>(),
        .dst_paged_kv_indptr = persistent_paged_kv_indptr_.data_ptr<int32_t>(),
        .dst_paged_kv_indices =
            persistent_paged_kv_indices_.data_ptr<int32_t>(),
        .dst_paged_kv_last_page_len =
            persistent_paged_kv_last_page_len_.data_ptr<int32_t>(),
        .actual_num_tokens = actual_num_tokens,
        .padded_num_tokens = static_cast<int64_t>(padded_num_tokens),
        .actual_batch_size = actual_batch_size,
        .actual_indices_size = actual_indices_size,
    };
    xllm::kernel::musa::update_llm_decode_metadata_from_host(host_update_params,
                                                             stream);
    return;
  }

  const torch::Tensor paged_kv_indptr_i32 =
      to_int32(params.attention.device.paged_kv_indptr);
  const torch::Tensor paged_kv_indices_i32 =
      to_int32(params.attention.device.paged_kv_indices);
  const torch::Tensor paged_kv_last_page_len_i32 =
      to_int32(params.attention.device.paged_kv_last_page_len);
  const int64_t actual_indices_size = paged_kv_indices_i32.size(0);
  xllm::kernel::musa::LlmDecodeMetadataUpdateParams update_params{
      .src_tokens = tokens_i32.data_ptr<int32_t>(),
      .src_positions = positions_i32.data_ptr<int32_t>(),
      .src_new_cache_slots = new_cache_slots_i32.data_ptr<int32_t>(),
      .src_kv_seq_lens = kv_seq_lens_i32.data_ptr<int32_t>(),
      .src_paged_kv_indptr = paged_kv_indptr_i32.data_ptr<int32_t>(),
      .src_paged_kv_indices = paged_kv_indices_i32.data_ptr<int32_t>(),
      .src_paged_kv_last_page_len =
          paged_kv_last_page_len_i32.data_ptr<int32_t>(),
      .dst_tokens = persistent_tokens_.data_ptr<int32_t>(),
      .dst_positions = persistent_positions_.data_ptr<int32_t>(),
      .dst_new_cache_slots = persistent_new_cache_slots_.data_ptr<int32_t>(),
      .dst_kv_seq_lens = kv_seq_lens_.data_ptr<int32_t>(),
      .dst_kv_seq_lens_delta =
          persistent_kv_seq_lens_delta_.data_ptr<int32_t>(),
      .dst_paged_kv_indptr = persistent_paged_kv_indptr_.data_ptr<int32_t>(),
      .dst_paged_kv_indices = persistent_paged_kv_indices_.data_ptr<int32_t>(),
      .dst_paged_kv_last_page_len =
          persistent_paged_kv_last_page_len_.data_ptr<int32_t>(),
      .actual_num_tokens = actual_num_tokens,
      .padded_num_tokens = static_cast<int64_t>(padded_num_tokens),
      .actual_batch_size = actual_batch_size,
      .actual_indices_size = actual_indices_size,
      .max_indices_size_for_graph_capacity =
          persistent_paged_kv_indices_.numel(),
  };
  xllm::kernel::musa::update_llm_decode_metadata(update_params, stream);
}

void MusaGraphPersistentParam::set_aux_hidden_states(
    const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }
  const uint32_t result_tokens = value.size(0);
  if (aux_hidden_states_.numel() == 0) {
    CHECK(c10::musa::currentStreamCaptureStatusMayInitCtx() ==
          c10::musa::CaptureStatus::None)
        << "Persistent auxiliary buffers must be prepared before graph capture";
    // Lazy initialization: create aux_hidden_states tensor if not already
    // created
    const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
    auto shape = value.sizes().vec();
    shape[0] = max_tokens_per_batch;
    torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
    if (args_.dtype() == "float" || args_.dtype() == "float32") {
      dtype = torch::kFloat32;
    }
    aux_hidden_states_ =
        torch::zeros(shape, torch::dtype(dtype).device(device_));
  }
  // Slice to match the actual shape
  auto slice =
      aux_hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens);
  // Reshape slice if needed to match value shape
  if (slice.sizes() == value.sizes()) {
    slice.copy_(value, /*non_blocking=*/true);
  }
}

size_t MusaGraphPersistentParam::get_persistent_tensor_bytes() const {
  auto bytes = [](const torch::Tensor& t) {
    return t.defined() ? static_cast<size_t>(t.numel()) * t.element_size() : 0;
  };
  size_t total = 0;
  total += bytes(persistent_tokens_);
  total += bytes(persistent_positions_);
  total += bytes(persistent_new_cache_slots_);
  total += bytes(persistent_linear_state_indices_);
  total += bytes(persistent_kv_cache_tokens_nums_);
  total += bytes(persistent_num_accepted_tokens_);
  total += bytes(persistent_block_tables_);
  total += bytes(hidden_states_);
  total += bytes(q_seq_lens_);
  total += bytes(kv_seq_lens_);
  total += bytes(persistent_embedding_);
  total += bytes(persistent_mtp_token_embedding_);
  total += bytes(aux_hidden_states_);
  total += bytes(persistent_paged_kv_indptr_);
  total += bytes(persistent_paged_kv_indices_);
  total += bytes(persistent_paged_kv_last_page_len_);
  total += bytes(persistent_decode_qo_indptr_);
  total += bytes(persistent_kv_seq_lens_delta_);
  total += bytes(persistent_expanded_kv_seq_lens_);
  total += bytes(persistent_expanded_block_tables_);
  total += bytes(persistent_expanded_paged_kv_indptr_);
  total += bytes(persistent_expanded_paged_kv_indices_);
  total += bytes(persistent_expanded_paged_kv_last_page_len_);
  total += bytes(persistent_expanded_qo_indptr_);
  return total;
}

torch::Tensor MusaGraphPersistentParam::refresh_linear_state_validity(
    const LinearStateValidityMask& mask) {
  const int64_t rows = static_cast<int64_t>(mask.size());
  CHECK_LE(rows, persistent_linear_state_validity_.numel())
      << "linear-state validity rows exceed persistent capacity";
  torch::Tensor host_rows = torch::empty(
      {rows}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
  bool* host_data = host_rows.data_ptr<bool>();
  for (int64_t i = 0; i < rows; ++i) {
    host_data[i] = mask[i] != 0;
  }
  torch::Tensor device_rows = persistent_linear_state_validity_.slice(
      /*dim=*/0, /*start=*/0, /*end=*/rows);
  device_rows.copy_(host_rows, /*non_blocking=*/true);
  return device_rows;
}

torch::Tensor MusaGraphPersistentParam::fa3_scheduler_metadata(
    int32_t batch_size,
    int32_t num_heads_q,
    int32_t num_heads_kv,
    int32_t head_dim,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t window_size_left,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& seqused_k) {
  const auto generate = [&]() {
    return xllm::kernel::musa::fa3_decode_scheduler_metadata(
        device_,
        batch_size,
        num_heads_q,
        num_heads_kv,
        head_dim,
        head_dim,
        max_seqlen_q,
        max_seqlen_k,
        window_size_left,
        /*window_size_right=*/0,
        cu_seqlens_q,
        seqused_k);
  };
  if (batch_size != 1) {
    return generate();
  }
  const int64_t kv_tile = xllm::kernel::musa::fa3_decode_kv_tile_size();
  const int64_t kv_blocks = (max_seqlen_k + kv_tile - 1) / kv_tile;
  const int64_t key = (static_cast<int64_t>(max_seqlen_q) << 32) | kv_blocks;
  auto it = single_sequence_fa3_metadata_.find(key);
  if (it != single_sequence_fa3_metadata_.end()) {
    return it->second;
  }
  torch::Tensor metadata = generate();
  single_sequence_fa3_metadata_.emplace(key, metadata);
  return metadata;
}

std::optional<ModelInputParams> MusaGraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params,
    bool update_attention_plan) {
  std::optional<ModelInputParams> params_for_capture;
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    params_for_capture = std::make_optional<ModelInputParams>(params);
  }
  // A pageable H2D copy on MUSA waits for the compute stream to drain, so the
  // spec-verify validity rows are refreshed through pinned memory instead of
  // being materialized by the metadata builder.
  torch::Tensor linear_state_validity;
  if (params.is_spec_verify && !params.linear_state_validity_mask.empty() &&
      !params.linear_state_validity_mask_tensor.defined()) {
    linear_state_validity =
        refresh_linear_state_validity(params.linear_state_validity_mask);
  }
  layer::AttentionMetadataBuildOptions metadata_build_options;
  metadata_build_options.materialize_linear_state_validity =
      !linear_state_validity.defined();
  // Build attn_metadata with original model_input_params. So we can set actual
  // batch size in plan_info.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params,
                                                 args_.enable_mla(),
                                                 /*attn_mask=*/std::nullopt,
                                                 /*device=*/std::nullopt,
                                                 metadata_build_options));
  CHECK(attn_metadata) << "attn_metadata should not be null";
  if (linear_state_validity.defined()) {
    attn_metadata->has_initial_states = linear_state_validity;
  }
  attn_metadata->enable_cuda_graph = true;

  const uint32_t actual_num_tokens = tokens.size(0);
  const int64_t linear_state_batch_size =
      params.embedding.linear_state_indices.defined()
          ? std::min<int64_t>(params.embedding.linear_state_indices.numel(),
                              persistent_linear_state_indices_.numel())
          : std::min<int64_t>(params.embedding.linear_state_ids.size(),
                              persistent_linear_state_indices_.numel());
  // The generic MTP input builder expands target verification into one row
  // per validation token. DeepSeek-V4 owns recurrent/DSA state per request,
  // so the target and draft graph carriers must retain one logical sequence
  // with a compact query width (k + 1), rather than treating proposal rows as
  // independent requests.
  int64_t compact_decode_width =
      std::max<int64_t>(options_.num_decoding_tokens(), 1);
  const bool compact_dsv4_target_verify =
      params.meta.batch_forward_type.is_decode() &&
      args_.model_type() == "deepseek_v4" &&
      options_.enable_speculative_decode() && !options_.is_draft_engine() &&
      compact_decode_width > 1 && actual_num_tokens > 0 &&
      actual_num_tokens % compact_decode_width == 0;
  const bool compact_dsv4_draft_extend =
      params.meta.batch_forward_type.is_decode() &&
      args_.model_type() == "deepseek_v4_mtp" && options_.is_draft_engine() &&
      linear_state_batch_size > 0 &&
      actual_num_tokens > linear_state_batch_size &&
      actual_num_tokens % linear_state_batch_size == 0;
  if (compact_dsv4_draft_extend) {
    compact_decode_width =
        static_cast<int64_t>(actual_num_tokens) / linear_state_batch_size;
  }
  const bool compact_dsv4_mtp_decode =
      compact_dsv4_target_verify || compact_dsv4_draft_extend;
  if (s_enable_graph_timing()) {
    LOG(INFO) << "GRAPH_TIMING compact_decode model_type=" << args_.model_type()
              << " is_draft=" << options_.is_draft_engine()
              << " actual_num_tokens=" << actual_num_tokens
              << " compact_width=" << compact_decode_width
              << " compact=" << compact_dsv4_mtp_decode;
  }
  // Decode has exactly one query token per attention row.  Schedule-overlap
  // forward buffers can retain a bucket-capacity num_sequences value (for
  // example 8 for a live C=5 batch), while tokens and all logical outputs
  // contain only the active rows.  Use the token count for ordinary decode so
  // persistent graph tensors never copy padded buffer rows as live requests.
  int64_t actual_batch_size = params.meta.num_sequences;
  if (params.meta.actual_num_sequences > 0) {
    actual_batch_size = params.meta.actual_num_sequences;
  } else if (params.meta.batch_forward_type.is_decode() &&
             !is_rec_multi_round_mode()) {
    actual_batch_size = static_cast<int64_t>(actual_num_tokens);
  }
  if (compact_dsv4_mtp_decode) {
    actual_batch_size =
        static_cast<int64_t>(actual_num_tokens) / compact_decode_width;
  }
  const int64_t accepted_batch_size =
      params.num_accepted_tokens.defined()
          ? std::min<int64_t>(params.num_accepted_tokens.numel(),
                              persistent_num_accepted_tokens_.numel())
          : 0;
  const bool dsv4_request_start =
      args_.model_type() == "deepseek_v4" &&
      (attn_metadata->is_prefill || attn_metadata->is_chunked_prefill) &&
      s_enable_dsv4_graph_persistent_reset();
  if (dsv4_request_start) {
    // These carriers are not necessarily present in a prefill input (for
    // example num_accepted_tokens is decode-only), so a recycled Graph can
    // otherwise retain the previous request's accepted-prefix/state rows.
    // Reset only persistent metadata; model KV/cache tensors are populated by
    // the normal prefill path and are intentionally left untouched.
    persistent_new_cache_slots_.fill_(-1);
    persistent_linear_state_indices_.fill_(-1);
    persistent_kv_cache_tokens_nums_.zero_();
    persistent_num_accepted_tokens_.fill_(1);
    q_seq_lens_.zero_();
    kv_seq_lens_.zero_();
    persistent_block_tables_.fill_(-1);
    persistent_paged_kv_indptr_.zero_();
    persistent_paged_kv_indices_.fill_(-1);
    persistent_paged_kv_last_page_len_.zero_();
    persistent_kv_seq_lens_delta_.zero_();
    LOG_FIRST_N(INFO, 1)
        << "DeepSeek-V4 MUSA graph persistent metadata reset at request "
           "boundary.";
  }
  const bool use_llm_decode_fast_path =
      can_use_llm_decode_fast_path(tokens, positions, params);

  // Cheap when the input builder already pre-staged (just a shared_ptr ref);
  // a single per-step D2H per index tensor in the fallback case (3 D2H total,
  // batch-sized, runs once before capture begin).
  const bool decode_path =
      !attn_metadata->is_prefill && !attn_metadata->is_chunked_prefill;
  const bool use_expanded_spec_verify =
      params.is_spec_verify && attn_metadata->is_chunked_prefill &&
      params.graph.use_expanded_decode_for_spec_verify_attention &&
      params.graph.expanded_paged_kv_indptr.defined();
  const int64_t expanded_rows =
      use_expanded_spec_verify ? params.graph.expanded_kv_seq_lens.numel() : 0;
  if (decode_path) {
    auto ensure_host_mirror = [](torch::Tensor& host_field,
                                 const torch::Tensor& device_field) {
      if (host_field.defined()) {
        return;
      }
      if (!device_field.defined()) {
        return;
      }
      host_field = device_field.to(torch::kCPU);
    };
    if (s_enable_graph_timing()) {
      LOG(INFO)
          << "GRAPH_TIMING ensure_host_mirror: indptr_host_defined="
          << attn_metadata->fa3_metadata.paged_kv_indptr_host.defined()
          << " indices_host_defined="
          << attn_metadata->fa3_metadata.paged_kv_indices_host.defined()
          << " last_page_len_host_defined="
          << attn_metadata->fa3_metadata.paged_kv_last_page_len_host.defined();
    }
    ensure_host_mirror(attn_metadata->fa3_metadata.paged_kv_indptr_host,
                       attn_metadata->paged_kv_indptr);
    ensure_host_mirror(attn_metadata->fa3_metadata.paged_kv_indices_host,
                       attn_metadata->paged_kv_indices);
    ensure_host_mirror(attn_metadata->fa3_metadata.paged_kv_last_page_len_host,
                       attn_metadata->paged_kv_last_page_len);
  }
  const char* stable_multi_graph_env =
      std::getenv("XLLM_DSV4_STABLE_MULTI_GRAPH_SEQUENCES");
  const bool stable_multi_graph_sequences =
      stable_multi_graph_env != nullptr &&
      std::string(stable_multi_graph_env) == "1";
  auto build_capture_params_if_needed =
      [&]() -> std::optional<ModelInputParams> {
    if (!return_capture_params) {
      return std::nullopt;
    }
    CHECK(params_for_capture.has_value())
        << "params_for_capture should be initialized when "
           "return_capture_params "
           "is true";
    if (params.embedding.input_embedding.defined()) {
      params_for_capture->embedding.input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    if (params.embedding.mtp_token_embedding.defined()) {
      params_for_capture->embedding.mtp_token_embedding =
          persistent_mtp_token_embedding(padded_num_tokens);
    }
    if (!params.embedding.linear_state_ids.empty()) {
      params_for_capture->embedding.linear_state_ids =
          params.embedding.linear_state_ids;
      params_for_capture->embedding.linear_state_indices =
          persistent_linear_state_indices(
              static_cast<uint32_t>(linear_state_batch_size));
    }
    if (params.attention.device.kv_cache_tokens_nums.defined()) {
      params_for_capture->attention.device.kv_cache_tokens_nums =
          persistent_kv_cache_tokens_nums(
              static_cast<uint32_t>(actual_batch_size));
    }
    if (params.num_accepted_tokens.defined()) {
      params_for_capture->num_accepted_tokens = persistent_num_accepted_tokens(
          static_cast<uint32_t>(accepted_batch_size));
      // Spec verify already carries the host copy; avoid a per-step D2H sync.
      torch::Tensor nat_host = params.num_accepted_tokens_host.empty()
                                   ? params.num_accepted_tokens.to(torch::kCPU)
                                         .to(torch::kLong)
                                         .contiguous()
                                   : torch::Tensor();
      if (params.num_accepted_tokens_host.empty()) {
        const int64_t* data = nat_host.data_ptr<int64_t>();
        params_for_capture->num_accepted_tokens_host.assign(
            data, data + accepted_batch_size);
      }
    }
    if (use_expanded_spec_verify) {
      params_for_capture->graph.expanded_kv_seq_lens =
          persistent_expanded_kv_seq_lens_.slice(0, 0, expanded_rows);
      params_for_capture->graph.expanded_block_tables =
          persistent_expanded_block_tables_.slice(0, 0, expanded_rows);
      params_for_capture->graph.expanded_paged_kv_indptr =
          persistent_expanded_paged_kv_indptr_.slice(0, 0, expanded_rows + 1);
      params_for_capture->graph.expanded_paged_kv_indices =
          persistent_expanded_paged_kv_indices_;
      params_for_capture->graph.expanded_paged_kv_last_page_len =
          persistent_expanded_paged_kv_last_page_len_.slice(
              0, 0, expanded_rows);
    }
    if (linear_state_validity.defined()) {
      params_for_capture->linear_state_validity_mask_tensor =
          linear_state_validity;
    }
    if (params.is_spec_verify && attn_metadata->is_chunked_prefill) {
      // The model rebuilds attention metadata from attention.device inside
      // forward; point it at the persistent copies so replay reads fresh
      // per-step values from capture-time addresses.
      auto& dev = params_for_capture->attention.device;
      dev.q_seq_lens = q_seq_lens(static_cast<uint32_t>(actual_batch_size + 1));
      dev.kv_seq_lens =
          kv_seq_lens(static_cast<uint32_t>(actual_batch_size + 1));
      dev.new_cache_slots = persistent_new_cache_slots(actual_num_tokens);
      dev.block_tables =
          persistent_block_tables(static_cast<uint32_t>(actual_batch_size));
      dev.paged_kv_indptr =
          persistent_paged_kv_indptr(static_cast<uint32_t>(actual_batch_size));
      dev.paged_kv_indices = persistent_paged_kv_indices_;
      dev.paged_kv_last_page_len = persistent_paged_kv_last_page_len(
          static_cast<uint32_t>(actual_batch_size));
    } else if (decode_path && !params.is_spec_verify &&
               !is_rec_multi_round_mode()) {
      // Same contract for ordinary decode. Rows are sized to the padded
      // bucket, matching the attn_metadata built above, and the block table
      // keeps its full width so FA3 sizes split-KV for any live length.
      auto& dev = params_for_capture->attention.device;
      dev.q_seq_lens = persistent_decode_qo_indptr(padded_num_tokens);
      dev.kv_seq_lens = kv_seq_lens(padded_num_tokens + 1);
      dev.new_cache_slots = persistent_new_cache_slots(padded_num_tokens);
      dev.block_tables = persistent_block_tables(padded_num_tokens);
      dev.paged_kv_indptr = persistent_paged_kv_indptr(padded_num_tokens);
      dev.paged_kv_indices = persistent_paged_kv_indices_;
      dev.paged_kv_last_page_len =
          persistent_paged_kv_last_page_len(padded_num_tokens);
    }
    params_for_capture->enable_graph = true;
    const bool graph_decode =
        !update_attention_plan || params.meta.batch_forward_type.is_decode();
    const int32_t graph_num_sequences =
        compact_dsv4_mtp_decode
            ? static_cast<int32_t>(
                  (padded_num_tokens + compact_decode_width - 1) /
                  compact_decode_width)
            : (graph_decode ? static_cast<int32_t>(padded_num_tokens)
                            : params.meta.num_sequences);
    params_for_capture->meta.num_sequences = graph_num_sequences;
    params_for_capture->meta.actual_num_sequences =
        compact_dsv4_mtp_decode
            ? static_cast<int32_t>(actual_batch_size)
            : (graph_decode ? static_cast<int32_t>(actual_num_tokens)
                            : params.meta.num_sequences);
    if (compact_dsv4_mtp_decode) {
      const int32_t verify_width = static_cast<int32_t>(compact_decode_width);
      const int32_t active_sequences = std::min<int32_t>(
          static_cast<int32_t>(actual_batch_size), graph_num_sequences);

      auto& host = params_for_capture->attention.host;
      host.q_seq_lens.assign(static_cast<size_t>(graph_num_sequences + 1), 0);
      host.q_cu_seq_lens.assign(static_cast<size_t>(graph_num_sequences), 0);
      const auto graph_q_lengths =
          layer::dsv4_graph_query_lengths(active_sequences,
                                          graph_num_sequences,
                                          verify_width,
                                          stable_multi_graph_sequences);
      for (int32_t seq = 0; seq < graph_num_sequences; ++seq) {
        const int32_t q_len = graph_q_lengths[seq];
        host.q_seq_lens[static_cast<size_t>(seq + 1)] =
            host.q_seq_lens[static_cast<size_t>(seq)] + q_len;
        host.q_cu_seq_lens[static_cast<size_t>(seq)] =
            host.q_seq_lens[static_cast<size_t>(seq + 1)];
      }

      const std::vector<int32_t> expanded_kv_lens =
          params.attention.host.kv_seq_lens;
      auto compact_kv_lens =
          layer::dsv4_compact_graph_kv_lengths(expanded_kv_lens,
                                               active_sequences,
                                               graph_num_sequences,
                                               verify_width,
                                               actual_num_tokens,
                                               params.meta.kv_max_seq_len);
      int32_t compact_kv_max = 0;
      for (int32_t seq = 0; seq < graph_num_sequences; ++seq) {
        compact_kv_max = std::max<int32_t>(
            compact_kv_max, compact_kv_lens[seq + 1] - compact_kv_lens[seq]);
      }
      host.kv_seq_lens = std::move(compact_kv_lens);
      params_for_capture->meta.q_max_seq_len = verify_width;
      params_for_capture->meta.kv_max_seq_len =
          std::max(params.meta.kv_max_seq_len, compact_kv_max);
      params_for_capture->is_spec_verify = true;

      for (torch::Tensor& block_table :
           params_for_capture->multi_block_tables) {
        if (block_table.defined() && block_table.dim() == 2 &&
            block_table.size(0) >= actual_num_tokens) {
          block_table = block_table
                            .slice(/*dim=*/0,
                                   /*start=*/0,
                                   /*end=*/actual_num_tokens,
                                   /*step=*/compact_decode_width)
                            .contiguous();
        }
        if (stable_multi_graph_sequences && block_table.defined() &&
            block_table.dim() == 2 &&
            block_table.size(0) < graph_num_sequences) {
          const int64_t missing_rows =
              graph_num_sequences - block_table.size(0);
          auto padding = torch::full(
              {missing_rows, block_table.size(1)}, -1, block_table.options());
          block_table =
              torch::cat({block_table, padding}, /*dim=*/0).contiguous();
        }
      }
    }
    params_for_capture->attention.device.q_seq_lens =
        q_seq_lens(static_cast<uint32_t>(graph_num_sequences + 1));
    params_for_capture->attention.device.kv_seq_lens =
        kv_seq_lens(static_cast<uint32_t>(graph_num_sequences + 1));
    params_for_capture->attention.device.new_cache_slots =
        persistent_new_cache_slots(padded_num_tokens);
    params_for_capture->attention.device.block_tables =
        persistent_block_tables(static_cast<uint32_t>(graph_num_sequences));
    const char* compact_tables_env =
        std::getenv("XLLM_DSV4_COMPACT_GRAPH_BLOCK_TABLES");
    if (compact_dsv4_target_verify && actual_batch_size > 1 &&
        layer::dsv4_graph_sequence_views_enabled() &&
        compact_tables_env != nullptr &&
        std::string(compact_tables_env) == "1") {
      const auto& padded_kv = params_for_capture->attention.host.kv_seq_lens;
      CHECK_GE(padded_kv.size(), static_cast<size_t>(actual_batch_size + 1));
      const std::vector<int32_t> live_kv(
          padded_kv.begin(), padded_kv.begin() + actual_batch_size + 1);
      const int64_t columns = layer::dsv4_graph_block_table_columns(
          live_kv, compact_decode_width, options_.block_size());
      if (columns > 0) {
        auto& carrier = params_for_capture->attention.device.block_tables;
        CHECK_LE(columns, carrier.size(1));
        for (const auto& table : params_for_capture->multi_block_tables) {
          CHECK(table.defined() && table.dim() == 2);
          CHECK_LE(table.size(1), columns)
              << "Raw DSV4 table exceeds its graph-key capacity; cannot trim";
        }
        carrier = carrier.narrow(/*dim=*/1, /*start=*/0, columns);
        LOG_FIRST_N(INFO, 1)
            << "[DSV4_COMPACT_GRAPH_BLOCK_TABLES] sequences="
            << graph_num_sequences << " columns=" << columns
            << " live_kv_max=" << params_for_capture->meta.kv_max_seq_len;
      }
    }
    params_for_capture->attn_metadata = attn_metadata;
    params_for_capture->is_spec_verify =
        compact_dsv4_mtp_decode || params.is_spec_verify;
    return params_for_capture;
  };

  // Copy data from input parameters to persistent graph tensors
  if (update_attention_plan && use_llm_decode_fast_path) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "use fast path for LLM decode metadata update";
    update_llm_decode_metadata_fast_path(tokens,
                                         positions,
                                         params,
                                         padded_num_tokens,
                                         actual_batch_size,
                                         actual_num_tokens);
  } else {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ tokens: src shape=" << tokens.sizes() << ", dst slice shape=["
        << actual_num_tokens << "]";
    persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(tokens, /*non_blocking=*/true);

    if (padded_num_tokens > actual_num_tokens) {
      persistent_tokens_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(0);
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ positions: src shape=" << positions.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    const torch::Tensor positions_to_copy =
        positions_for_persistent_copy(positions, persistent_positions_.dim());
    const int64_t position_token_dim = persistent_positions_.dim() - 1;
    CHECK_EQ(positions_to_copy.dim(), persistent_positions_.dim());
    persistent_positions_
        .slice(position_token_dim, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(positions_to_copy, /*non_blocking=*/true);
    if (padded_num_tokens > actual_num_tokens) {
      persistent_positions_
          .slice(position_token_dim,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(0);
    }
  }

  if (!is_rec_multi_round_mode() && !use_llm_decode_fast_path &&
      update_attention_plan) {
    // q_seq_lens is q_cu_seq_lens in GPU Model.
    // kv_seq_lens is kv_cu_seq_lens in GPU Model.
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ q_seq_lens: src shape="
        << params.attention.device.q_seq_lens.sizes() << ", dst slice shape=["
        << actual_batch_size + 1 << "]";
    CHECK_GE(params.attention.device.q_seq_lens.numel(), actual_batch_size + 1);
    q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.q_seq_lens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1),
               /*non_blocking=*/true);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ kv_seq_lens: src shape="
        << params.attention.device.kv_seq_lens.sizes() << ", dst slice shape=["
        << actual_batch_size + 1 << "]";
    CHECK_GE(params.attention.device.kv_seq_lens.numel(),
             actual_batch_size + 1);
    kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.kv_seq_lens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1),
               /*non_blocking=*/true);
    if (params.meta.batch_forward_type.is_decode() &&
        padded_num_tokens > actual_num_tokens) {
      kv_seq_lens_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size + 1,
                 /*end=*/padded_num_tokens + 1)
          .copy_(params.attention.device.kv_seq_lens
                     .slice(/*dim=*/0,
                            /*start=*/actual_batch_size,
                            /*end=*/actual_batch_size + 1)
                     .expand({static_cast<int64_t>(padded_num_tokens) -
                              actual_batch_size}),
                 /*non_blocking=*/true);
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ new_cache_slots: src shape="
        << params.attention.device.new_cache_slots.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    persistent_new_cache_slots_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(params.attention.device.new_cache_slots, /*non_blocking=*/true);
    if (padded_num_tokens > actual_num_tokens) {
      persistent_new_cache_slots_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(params.meta.batch_forward_type.is_decode() ? -1 : 0);
    }

    if (params.attention.device.kv_cache_tokens_nums.defined()) {
      CHECK_GE(params.attention.device.kv_cache_tokens_nums.numel(),
               actual_batch_size)
          << "kv_cache_tokens_nums must contain one value per sequence";
      persistent_kv_cache_tokens_nums_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .copy_(params.attention.device.kv_cache_tokens_nums.slice(
                     /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size),
                 /*non_blocking=*/true);
    }

    // Keep metadata tensors pointing to persistent buffers used by graph
    // capture/replay so their addresses are stable and shapes match padded
    // tensors in capture path.
    attn_metadata->q_cu_seq_lens = q_seq_lens(/*actual_batch_size=*/
                                              actual_batch_size + 1);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens(/*actual_batch_size=*/
                                                actual_batch_size + 1);
    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
  } else if (!update_attention_plan) {
    // DeepSeek-V4 rebuilds its per-cache slot mappings in the model-private
    // DSA metadata hook. Keep only stable graph-facing carrier tensors here.
    const uint32_t graph_num_sequences =
        compact_dsv4_mtp_decode
            ? static_cast<uint32_t>(
                  (padded_num_tokens + compact_decode_width - 1) /
                  compact_decode_width)
            : (padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens);
    const uint32_t graph_slot_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    persistent_new_cache_slots_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/graph_slot_tokens)
        .fill_(-1);
    auto graph_q_seq_lens = q_seq_lens_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/graph_num_sequences + 1);
    graph_q_seq_lens.copy_(persistent_decode_qo_indptr_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/graph_num_sequences + 1));
    if (compact_dsv4_mtp_decode) {
      graph_q_seq_lens.mul_(compact_decode_width);
    }
    kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/graph_num_sequences + 1)
        .zero_();
  }

  if (!is_rec_multi_round_mode() &&
      !params.embedding.linear_state_ids.empty()) {
    if (params.embedding.linear_state_indices.defined()) {
      persistent_linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_state_batch_size)
          .copy_(params.embedding.linear_state_indices.slice(
                     /*dim=*/0, /*start=*/0, /*end=*/linear_state_batch_size),
                 /*non_blocking=*/true);
    } else {
      persistent_linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_state_batch_size)
          .copy_(torch::tensor(params.embedding.linear_state_ids, torch::kInt)
                     .to(device_),
                 /*non_blocking=*/true);
    }
  }

  if (params.num_accepted_tokens.defined()) {
    persistent_num_accepted_tokens_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/accepted_batch_size)
        .copy_(params.num_accepted_tokens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/accepted_batch_size),
               /*non_blocking=*/true);
  }

  if (update_attention_plan) {
    // Copy block table data. In rec multi-round, block_tables may already be
    // expanded to batch_size * beam_width rows while num_sequences still tracks
    // the logical request count. Use the tensor's real row count here.
    CHECK_GE(persistent_block_tables_.size(0), actual_batch_size)
        << "MTP/decode graph persistent block_tables rows ("
        << persistent_block_tables_.size(0) << ") < live attention rows ("
        << actual_batch_size
        << "); for qwen3_5_mtp the constructor must size max_seqs_per_batch*2";
    const int64_t actual_block_table_batch =
        is_rec_multi_round_mode() ? params.attention.device.block_tables.size(0)
                                  : actual_batch_size;
    CHECK_GE(params.attention.device.block_tables.size(0),
             actual_block_table_batch)
        << "block_tables has fewer rows than the logical graph batch";
    const int64_t actual_block_table_len =
        params.attention.device.block_tables.size(1);
    const torch::Tensor source_block_tables =
        params.attention.device.block_tables
            .slice(/*dim=*/0,
                   /*start=*/0,
                   /*end=*/actual_block_table_batch)
            .slice(/*dim=*/1,
                   /*start=*/0,
                   /*end=*/actual_block_table_len);
    torch::Tensor slice_persistent_block_tables =
        persistent_block_tables_
            .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_block_table_batch)
            .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ block_tables: src shape=" << source_block_tables.sizes()
        << ", dst slice shape=" << slice_persistent_block_tables.sizes();
    slice_persistent_block_tables.copy_(source_block_tables,
                                        /*non_blocking=*/true);
    const bool pad_decode_block_tables =
        params.meta.batch_forward_type.is_decode() &&
        padded_num_tokens > static_cast<uint32_t>(actual_block_table_batch);
    if (pad_decode_block_tables) {
      persistent_block_tables_
          .slice(/*dim=*/0,
                 /*start=*/actual_block_table_batch,
                 /*end=*/padded_num_tokens)
          .fill_(-1);
    }
    if (!attn_metadata->is_prefill || args_.enable_mla()) {
      const uint32_t graph_block_table_batch =
          pad_decode_block_tables
              ? padded_num_tokens
              : static_cast<uint32_t>(actual_block_table_batch);
      attn_metadata->block_table =
          persistent_block_tables(graph_block_table_batch);
    }
  } else {
    const uint32_t graph_num_sequences =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    persistent_block_tables_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/graph_num_sequences)
        .fill_(-1);
  }

  if (use_expanded_spec_verify) {
    const auto& g = params.graph;
    CHECK(g.expanded_kv_seq_lens.defined() &&
          g.expanded_block_tables.defined() &&
          g.expanded_paged_kv_indices.defined() &&
          g.expanded_paged_kv_last_page_len.defined())
        << "expanded spec-verify graph input incomplete";
    CHECK_LE(expanded_rows, persistent_expanded_kv_seq_lens_.numel())
        << "expanded spec-verify rows exceed persistent capacity";
    CHECK_EQ(g.expanded_paged_kv_indptr.numel(), expanded_rows + 1);
    CHECK_EQ(g.expanded_block_tables.size(0), expanded_rows);
    CHECK_LE(g.expanded_block_tables.size(1),
             persistent_expanded_block_tables_.size(1));
    CHECK_LE(g.expanded_paged_kv_indices.numel(),
             persistent_expanded_paged_kv_indices_.numel());
    persistent_expanded_kv_seq_lens_.slice(0, 0, expanded_rows)
        .copy_(g.expanded_kv_seq_lens, /*non_blocking=*/true);
    persistent_expanded_block_tables_.slice(0, 0, expanded_rows)
        .slice(1, 0, g.expanded_block_tables.size(1))
        .copy_(g.expanded_block_tables, /*non_blocking=*/true);
    persistent_expanded_paged_kv_indptr_.slice(0, 0, expanded_rows + 1)
        .copy_(g.expanded_paged_kv_indptr, /*non_blocking=*/true);
    persistent_expanded_paged_kv_indices_
        .slice(0, 0, g.expanded_paged_kv_indices.numel())
        .copy_(g.expanded_paged_kv_indices, /*non_blocking=*/true);
    persistent_expanded_paged_kv_last_page_len_.slice(0, 0, expanded_rows)
        .copy_(g.expanded_paged_kv_last_page_len, /*non_blocking=*/true);
    layer::ExpandedDecodeMetadata& ed = attn_metadata->expanded_decode;
    ed.enabled = true;
    ed.kv_seq_lens =
        persistent_expanded_kv_seq_lens_.slice(0, 0, expanded_rows);
    ed.block_table =
        persistent_expanded_block_tables_.slice(0, 0, expanded_rows);
    ed.paged_kv_indptr =
        persistent_expanded_paged_kv_indptr_.slice(0, 0, expanded_rows + 1);
    ed.paged_kv_indices = persistent_expanded_paged_kv_indices_;
    ed.paged_kv_last_page_len =
        persistent_expanded_paged_kv_last_page_len_.slice(0, 0, expanded_rows);
  }

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.embedding.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      auto embedding_shape = embedding.sizes().vec();
      CHECK(!embedding_shape.empty())
          << "input_embedding must have a token dimension";
      embedding_shape[0] = max_tokens_per_batch;
      torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros(embedding_shape, torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ embedding: src shape=" << embedding.sizes()
        << ", dst slice shape="
        << persistent_embedding_
               .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
               .sizes();
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }

  // Qwen3.5 MTP draft token embedding. The draft forward consumes it instead
  // of running embed_tokens_ (IndexSelect) inside the captured graph, which
  // is not capture-safe on MUSA and yields NaN/garbage draft hidden states.
  const torch::Tensor& mtp_token_embedding =
      params.embedding.mtp_token_embedding;
  if (mtp_token_embedding.defined()) {
    const int64_t mtp_embedding_tokens = mtp_token_embedding.size(0);
    if (!persistent_mtp_token_embedding_.defined() ||
        persistent_mtp_token_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      persistent_mtp_token_embedding_ =
          torch::zeros({max_tokens_per_batch, mtp_token_embedding.size(1)},
                       mtp_token_embedding.options().device(device_));
    }
    persistent_mtp_token_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/mtp_embedding_tokens)
        .copy_(mtp_token_embedding, /*non_blocking=*/true);
  }

  // DeepSeek-V4 owns composite DSA caches and supplies graph metadata through
  // the model hook. It does not consume the generic FlashInfer plan below.
  if (!update_attention_plan) {
    return build_capture_params_if_needed();
  }

  const bool is_decode_with_llmrec =
      params.meta.batch_forward_type.is_decode() && params.has_llmrec_params();
  const bool use_two_stage_decode =
      !::xllm::RecConfig::get_instance().enable_xattention_one_stage() &&
      is_decode_with_llmrec;
  const int32_t head_dim = args_.head_dim();
  const int64_t tp_size =
      options_.world_size() /
      (std::max(options_.dp_size(), 1) * std::max(options_.cp_size(), 1));
  const int64_t n_heads = args_.n_heads() / std::max(tp_size, int64_t{1});
  const int64_t total_kv_heads = args_.n_kv_heads().value_or(args_.n_heads());
  const int64_t n_kv_heads =
      (total_kv_heads >= tp_size)
          ? (total_kv_heads / std::max(tp_size, int64_t{1}))
          : 1;
  const int64_t block_size = options_.block_size();

  // ModelArgs stores the window length, while the attention kernels consume
  // the inclusive left-window offset. Preserve -1 as the disabled sentinel.
  const int32_t model_sliding_window = args_.sliding_window();
  const int32_t sliding_window =
      args_.use_sliding_window() && model_sliding_window > 0
          ? model_sliding_window - 1
          : -1;

  // Get dtype from k_cache
  const auto dtype = k_cache.scalar_type();

  bool use_tensor_core =
      xllm::kernel::musa::should_use_tensor_core(dtype, n_heads, n_kv_heads);
  // Keep in sync with BaseAttentionImpl::decode_use_tensor_core_ on MUSA:
  // the Mate FFI ships the dedicated `batch_decode_*` kernel (exporting the
  // "run" symbol) for our paged-KV layouts, while the chunked-prefill
  // `batch_prefill_*` kernel only exports "paged_run". When this graph-mode
  // planner picks the chunked-prefill URI but `FlashInferAttentionImpl::
  // decoder_forward` calls `batch_decode(..., decode_use_tensor_core_=false)`,
  // `batch_decode` falls into the else branch and tries to look up
  // get_function(prefill_uri, "run") which doesn't exist in the .so. Force
  // the same value here so the planner and the runtime caller agree on the
  // URI scheme.
  use_tensor_core = false;
  if (use_two_stage_decode) {
    LOG(FATAL) << "two-stage xattention decode is not supported in "
                  "MUSA builds.";
  }
  if (use_llm_decode_fast_path) {
    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    // Decode has one query token per sequence, so its graph-facing metadata
    // must describe the padded token bucket rather than only the real batch.
    const uint32_t metadata_batch =
        padded_num_tokens > 0 ? padded_num_tokens
                              : static_cast<uint32_t>(actual_batch_size);
    attn_metadata->q_cu_seq_lens = persistent_decode_qo_indptr(metadata_batch);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens(metadata_batch + 1);
    attn_metadata->kv_seq_lens = persistent_kv_seq_lens_delta(metadata_batch);
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
    attn_metadata->paged_kv_indptr = persistent_paged_kv_indptr(metadata_batch);
    attn_metadata->paged_kv_indices = persistent_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len =
        persistent_paged_kv_last_page_len(metadata_batch);
    attn_metadata->qo_indptr = persistent_decode_qo_indptr(metadata_batch);
  } else {
    CHECK(params.attention.device.paged_kv_indptr.defined())
        << "paged_kv_indptr should not be null";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indptr: src shape="
        << params.attention.device.paged_kv_indptr.sizes()
        << ", dst slice shape=[" << (actual_batch_size + 1) << "]";
    if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
      torch::Tensor paged_kv_indptr_cpu =
          params.attention.device.paged_kv_indptr.to(torch::kCPU);
      VLOG(kGraphExecutorLogVerboseLevel)
          << "copy_ paged_kv_indptr: src values=" << paged_kv_indptr_cpu;
    }
    persistent_paged_kv_indptr_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.paged_kv_indptr, /*non_blocking=*/true);
    const bool pad_decode_metadata =
        params.meta.batch_forward_type.is_decode() &&
        padded_num_tokens > actual_num_tokens;
    if (pad_decode_metadata) {
      persistent_paged_kv_indptr_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size + 1,
                 /*end=*/padded_num_tokens + 1)
          .copy_(params.attention.device.paged_kv_indptr
                     .slice(/*dim=*/0,
                            /*start=*/actual_batch_size,
                            /*end=*/actual_batch_size + 1)
                     .expand({static_cast<int64_t>(padded_num_tokens) -
                              actual_batch_size}),
                 /*non_blocking=*/true);
    }
    CHECK(params.attention.device.paged_kv_indices.defined())
        << "paged_kv_indices should not be null";
    const int64_t actual_indices_size =
        params.attention.device.paged_kv_indices.size(0);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indices: src shape="
        << params.attention.device.paged_kv_indices.sizes()
        << ", dst slice shape=[" << actual_indices_size << "]";
    persistent_paged_kv_indices_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_indices_size)
        .copy_(params.attention.device.paged_kv_indices, /*non_blocking=*/true);
    CHECK(params.attention.device.paged_kv_last_page_len.defined())
        << "paged_kv_last_page_len should not be null";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_last_page_len: src shape="
        << params.attention.device.paged_kv_last_page_len.sizes()
        << ", dst slice shape=[" << actual_batch_size << "]";
    persistent_paged_kv_last_page_len_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_batch_size)
        .copy_(params.attention.device.paged_kv_last_page_len,
               /*non_blocking=*/true);
    if (pad_decode_metadata) {
      persistent_paged_kv_last_page_len_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size,
                 /*end=*/padded_num_tokens)
          .fill_(1);
    }
    const int64_t metadata_batch =
        params.meta.batch_forward_type.is_decode() && padded_num_tokens > 0
            ? padded_num_tokens
            : actual_batch_size;
    if (params.meta.batch_forward_type.is_decode()) {
      attn_metadata->q_cu_seq_lens =
          persistent_decode_qo_indptr(metadata_batch);
      attn_metadata->kv_cu_seq_lens = kv_seq_lens(metadata_batch + 1);
      torch::Tensor kv_seq_lens_delta =
          persistent_kv_seq_lens_delta(metadata_batch);
      kv_seq_lens_delta.copy_(torch::diff(attn_metadata->kv_cu_seq_lens),
                              /*non_blocking=*/true);
      attn_metadata->kv_seq_lens = kv_seq_lens_delta;
    } else if (params.is_spec_verify) {
      // Spec-verify graph: per-sequence KV lengths must live at a fixed
      // address that the captured FA3 decode reads on every replay.
      torch::Tensor kv_seq_lens_delta =
          persistent_kv_seq_lens_delta(metadata_batch);
      kv_seq_lens_delta.copy_(
          torch::diff(kv_seq_lens(/*actual_batch_size=*/metadata_batch + 1)),
          /*non_blocking=*/true);
      attn_metadata->kv_seq_lens = kv_seq_lens_delta;
    } else {
      attn_metadata->kv_seq_lens =
          torch::diff(kv_seq_lens(/*actual_batch_size=*/metadata_batch + 1));
    }
    attn_metadata->paged_kv_indptr = persistent_paged_kv_indptr(metadata_batch);
    attn_metadata->paged_kv_indices = persistent_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len =
        persistent_paged_kv_last_page_len(metadata_batch);
    if (attn_metadata->is_chunked_prefill) {
      attn_metadata->qo_indptr =
          q_seq_lens(/*actual_batch_size=*/actual_batch_size + 1);
    } else {
      attn_metadata->qo_indptr = persistent_decode_qo_indptr(metadata_batch);
    }
  }
  const int64_t gqa_ratio = n_kv_heads > 0 ? n_heads / n_kv_heads : int64_t{0};
  // MUSA Mate 0.2.7 provides the FA3 paged decode contract used by the
  // cached_ops tree, not the legacy FlashInfer FA2 batch_decode module.
  // Avoid creating an FA2 plan for supported BF16 graph shapes; the FA3
  // scheduler metadata below is prepared before capture instead.
  const bool fa3_graph_decode =
      s_use_fa3_decode(gqa_ratio) && dtype == torch::ScalarType::BFloat16 &&
      head_dim == 256 && (gqa_ratio == 6 || gqa_ratio == 8) &&
      !attn_metadata->is_prefill && !attn_metadata->is_chunked_prefill &&
      attn_metadata->block_table.defined();

  // Update plan_info if attn_metadata exists and device-graph mode is on
  // This ensures plan_info is updated before MUSA graph capture/replay.
  {
    // Update plan_info
    // Note: plan_info is only updated at layer 0, so we set layer_id to 0
    attn_metadata->plan_info->layer_id = 0;
    CHECK_EQ(dtype, torch::ScalarType::BFloat16)
        << "only support bf16 kvcache for now";

    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphPersistentParam::update() calling update_plan_info: "
        << "is_prefill=" << attn_metadata->is_prefill
        << ", is_chunked_prefill=" << attn_metadata->is_chunked_prefill
        << ", enable_cuda_graph=" << attn_metadata->enable_cuda_graph;

    if (!attn_metadata->is_prefill && !attn_metadata->is_chunked_prefill &&
        !fa3_graph_decode) {
      // Regular decode uses the paged batch-decode plan.
      const int32_t max_kv_blocks_per_seq_for_capture =
          block_size > 0
              ? static_cast<int32_t>(
                    (args_.max_position_embeddings() + block_size - 1) /
                    block_size)
              : 0;
      layer::musa::flashinfer::update_decode_plan_info(
          attn_metadata->plan_info,
          /*backend=*/"fa2",  // flashinfer paged fa3 is slow, use fa2 instead
          *attn_metadata,
          dtype,                             // query_dtype
          dtype,                             // key_dtype
          dtype,                             // output_dtype
          head_dim,                          // head_dim_qk
          head_dim,                          // head_dim_vo
          static_cast<int32_t>(n_heads),     // num_qo_heads
          static_cast<int32_t>(n_kv_heads),  // num_kv_heads
          static_cast<int32_t>(block_size),  // block_size
          sliding_window,                    // window_size_left
          /*enable_cuda_graph=*/true,
          use_tensor_core,
          max_kv_blocks_per_seq_for_capture);
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphPersistentParam::update() plan_info state: uri="
        << attn_metadata->plan_info->uri << ", plan_info.defined="
        << attn_metadata->plan_info->plan_info.defined() << ", plan_info.size="
        << (attn_metadata->plan_info->plan_info.defined()
                ? attn_metadata->plan_info->plan_info.size()
                : 0);
  }

  // MUSA FA3 graph replay must not capture scheduler-metadata generation.
  // The optimal split count depends on the live KV length, while decode
  // graphs are initially captured with a short synthetic sequence. Generate
  // fresh metadata here, outside capture/replay, and let MusaGraph::replay()
  // copy it into the tensor address retained by the captured attention calls.
  if (fa3_graph_decode && !attn_metadata->is_chunked_prefill &&
      attn_metadata->block_table.defined()) {
    const int64_t batch_size = attn_metadata->block_table.size(0);
    if (batch_size > 0 && (gqa_ratio == 6 || gqa_ratio == 8)) {
      const torch::Tensor cu_seqlens_q =
          attn_metadata->qo_indptr.has_value() &&
                  attn_metadata->qo_indptr->defined()
              ? *attn_metadata->qo_indptr
              : attn_metadata->q_cu_seq_lens;
      CHECK(attn_metadata->kv_seq_lens.defined())
          << "FA3 graph decode requires per-sequence KV lengths";
      attn_metadata->fa3_metadata.share_fa3_scheduler_metadata = true;
      attn_metadata->fa3_metadata.fa3_scheduler_metadata =
          fa3_scheduler_metadata(
              static_cast<int32_t>(batch_size),
              static_cast<int32_t>(n_heads),
              static_cast<int32_t>(n_kv_heads),
              head_dim,
              std::max<int32_t>(attn_metadata->max_query_len, 1),
              std::max<int32_t>(attn_metadata->max_seq_len, 1),
              sliding_window,
              cu_seqlens_q,
              attn_metadata->kv_seq_lens);
    }
  }

  // Spec-verify expanded decode: one FA3 scheduler metadata shared by all
  // layers, generated outside capture and copied into the captured address on
  // replay (same contract as ordinary decode above).
  if (use_expanded_spec_verify && s_use_fa3_decode(gqa_ratio) &&
      dtype == torch::ScalarType::BFloat16 && head_dim == 256 &&
      (gqa_ratio == 6 || gqa_ratio == 8) && expanded_rows > 0) {
    attn_metadata->fa3_metadata.share_fa3_scheduler_metadata = true;
    attn_metadata->fa3_metadata.fa3_scheduler_metadata =
        xllm::kernel::musa::fa3_decode_scheduler_metadata(
            device_,
            static_cast<int32_t>(expanded_rows),
            static_cast<int32_t>(n_heads),
            static_cast<int32_t>(n_kv_heads),
            head_dim,
            head_dim,
            /*max_seqlen_q=*/1,
            std::max<int32_t>(attn_metadata->max_seq_len, 1),
            sliding_window,
            /*window_size_right=*/0,
            persistent_expanded_qo_indptr_.slice(0, 0, expanded_rows + 1),
            attn_metadata->expanded_decode.kv_seq_lens);
  } else if (params.is_spec_verify && attn_metadata->is_chunked_prefill &&
             s_use_fa3_decode(gqa_ratio) &&
             dtype == torch::ScalarType::BFloat16 && head_dim == 256 &&
             (gqa_ratio == 6 || gqa_ratio == 8) &&
             attn_metadata->block_table.defined() &&
             attn_metadata->kv_seq_lens.defined()) {
    // MUSA spec verify (no expanded rows): FA3 decode with q_len=K+1 per
    // sequence. Generate the shared scheduler metadata outside capture.
    const torch::Tensor cu_seqlens_q =
        attn_metadata->qo_indptr.has_value() &&
                attn_metadata->qo_indptr->defined()
            ? *attn_metadata->qo_indptr
            : attn_metadata->q_cu_seq_lens;
    attn_metadata->fa3_metadata.share_fa3_scheduler_metadata = true;
    attn_metadata->fa3_metadata.fa3_scheduler_metadata = fa3_scheduler_metadata(
        static_cast<int32_t>(attn_metadata->block_table.size(0)),
        static_cast<int32_t>(n_heads),
        static_cast<int32_t>(n_kv_heads),
        head_dim,
        std::max<int32_t>(attn_metadata->max_query_len, 1),
        std::max<int32_t>(attn_metadata->max_seq_len, 1),
        sliding_window,
        cu_seqlens_q,
        attn_metadata->kv_seq_lens);
  }

  // Return ModelInputParams with persistent buffer references if requested
  return build_capture_params_if_needed();
}

void MusaGraph::refresh_persistent_paged_kv_host_mirrors(
    const std::shared_ptr<layer::AttentionMetadata>& attn_metadata,
    const AttentionHostInput& host_src) {
  // Only applies to the Mate FFI decode path. Prefill/chunked-prefill and MLA
  // attention do not pass host pointers through the FFI run() boundary, so
  // there is nothing to stabilize there.
  if (!attn_metadata) {
    return;
  }
  if (attn_metadata->is_prefill || attn_metadata->is_chunked_prefill) {
    return;
  }

  // Helper: lazily allocate / grow a pinned host buffer, copy fresh metadata
  // into it, then re-point `host_field` at the owning slice. The buffer's
  // underlying storage pointer must be STABLE across the lifetime of this
  // MusaGraph (captured FFI run() bakes it into the graph).
  //
  // Prefer copying from attention.host CPU mirrors (batch_input_builder path):
  // the values are already on host, so a CPU->pinned memcpy avoids the extra
  // device round-trip and the musaStreamSync that blocking D2H would insert.
  // Fall back to blocking D2H from persistent device tensors for callers that
  // did not pre-stage host mirrors (profile / warmup paths).
  //
  // CRITICAL: when allocating for the first time, size to max(numel,
  // min_alloc_numel). The captured graph cannot tolerate a later realloc:
  // if the KV cache crosses a block boundary mid-replay (e.g., decode 38 of
  // a question with prefill=27, block_size=64), the device-side
  // paged_kv_indices numel grows from 1 to 2 entries. With min_alloc_numel
  // set to the worst case at capture time, the first allocation is already
  // large enough and subsequent refresh calls hit the same storage.
  // Otherwise the realloc returns fresh memory and the captured kernel
  // dereferences a stale (freed) pointer, producing silently-wrong
  // attention outputs from L3 onward. See refresh-call-site comment.
  auto refresh_one = [](torch::Tensor& host_buf,
                        torch::Tensor& host_field,
                        const torch::Tensor& device_src,
                        const torch::Tensor& cpu_src,
                        int64_t min_alloc_numel) {
    if (!device_src.defined()) {
      return;
    }
    const int64_t numel = device_src.numel();
    const torch::ScalarType src_dtype = device_src.scalar_type();
    const bool needs_alloc =
        !host_buf.defined() || host_buf.scalar_type() != src_dtype ||
        host_buf.numel() < numel || host_buf.numel() < min_alloc_numel;
    if (needs_alloc) {
      // Pinned so musaMemcpyAsync from device is a real async copy that can
      // be captured into the graph (the Mate FFI submits H2D internally on
      // some shapes; pinning ensures the captured operation refreshes the
      // device buffer from our stable host pointer on every replay). Sized
      // to max(numel, min_alloc_numel) so the first allocation already
      // covers the worst-case KV cache layout for this MusaGraph instance.
      auto opts = torch::TensorOptions()
                      .dtype(src_dtype)
                      .device(torch::kCPU)
                      .pinned_memory(true);
      const int64_t alloc_numel = std::max<int64_t>(numel, min_alloc_numel);
      host_buf = torch::empty({alloc_numel}, opts);
    }
    auto dst = host_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/numel);
    // device_src may have a non-1D shape (e.g., [bs+1]); view as flat for the
    // copy and let host_field carry the original shape via a view-back.
    //
    // The Mate FFI batch_decode wrapper reads paged_kv_* host tensors on the
    // CPU at submit time, so destination bytes must be valid before replay.
    // CPU->pinned and blocking D2H both satisfy that; async D2H would not.
    const bool use_cpu_src = cpu_src.defined() && cpu_src.device().is_cpu() &&
                             cpu_src.numel() == numel;
    if (use_cpu_src) {
      torch::Tensor cpu_flat = cpu_src.contiguous().view({numel});
      if (cpu_flat.scalar_type() != src_dtype) {
        cpu_flat = cpu_flat.to(src_dtype);
      }
      dst.copy_(cpu_flat);
    } else {
      dst.copy_(device_src.contiguous().view({numel}), /*non_blocking=*/false);
    }
    // Re-point host_field at the persistent storage; preserve the original
    // logical shape so downstream callers that interrogate sizes still see
    // the same view they did before this rewrite.
    host_field = dst.view(device_src.sizes());
  };

  refresh_one(paged_kv_indptr_host_buf_,
              attn_metadata->fa3_metadata.paged_kv_indptr_host,
              attn_metadata->paged_kv_indptr,
              host_src.paged_kv_indptr,
              paged_kv_indptr_host_max_numel_);
  refresh_one(paged_kv_indices_host_buf_,
              attn_metadata->fa3_metadata.paged_kv_indices_host,
              attn_metadata->paged_kv_indices,
              host_src.paged_kv_indices,
              paged_kv_indices_host_max_numel_);
  refresh_one(paged_kv_last_page_len_host_buf_,
              attn_metadata->fa3_metadata.paged_kv_last_page_len_host,
              attn_metadata->paged_kv_last_page_len,
              host_src.paged_kv_last_page_len,
              paged_kv_last_page_len_host_max_numel_);
}

// MusaGraph implementation
void MusaGraph::prepare_model_graph_metadata(CausalLM* model,
                                             const torch::Tensor& positions,
                                             ModelInputParams& params) {
  CHECK(model != nullptr) << "MUSA graph model must not be null";
  if (!model->requires_graph_forward_metadata()) {
    return;
  }
  if (!model_graph_metadata_state_) {
    model_graph_metadata_state_ = model->create_graph_forward_metadata_state();
    CHECK(model_graph_metadata_state_)
        << "MUSA graph metadata state must be initialized during capture";
  }
  model->prepare_graph_forward_metadata(
      model_graph_metadata_state_.get(), positions, params);
  CHECK(params.attn_metadata)
      << "MUSA graph metadata preparation did not populate attn_metadata";
}

bool MusaGraph::capture(CausalLM* model,
                        const ModelArgs& args,
                        const runtime::Options& options,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params,
                        std::vector<KVCache>& kv_cache,
                        uint32_t bucket_num_tokens,
                        const c10::musa::MempoolId_t& pool,
                        std::unique_ptr<MusaMemPool> pool_owner) {
  model_ = model;
  CHECK(pool_owner_ == nullptr);
  pool_owner_ = std::move(pool_owner);
  MusaMemPool* pool_ptr = pool_owner_.get();
  padded_num_tokens_ = bucket_num_tokens;
  const uint32_t actual_num_tokens = tokens.size(0);
  captured_actual_tokens_ = actual_num_tokens;
  CHECK_GE(padded_num_tokens_, actual_num_tokens)
      << "bucket_num_tokens >= actual_num_tokens";

  // Compute worst-case pinned-host-mirror sizes for paged-KV metadata so the
  // FIRST allocation inside refresh_persistent_paged_kv_host_mirrors already
  // covers the largest layout this MusaGraph instance can ever see. Without
  // this, the captured graph bakes in a host pointer whose underlying buffer
  // gets reallocated mid-replay when the KV cache crosses a block boundary
  // (e.g., decode step 38 of a 27-token-prefill question with block_size=64
  // grows paged_kv_indices from 1 -> 2 entries), causing the captured Mate
  // FFI batch_decode kernel to dereference stale memory and silently corrupt
  // attention output. See refresh_persistent_paged_kv_host_mirrors comment.
  //
  // For a decode bucket of N input tokens, at most N sequences are active and
  // each can hold up to ceil(max_position_embeddings / block_size) blocks.
  {
    const int64_t block_size = options.block_size();
    const int64_t max_pos = args.max_position_embeddings();
    const int64_t max_blocks_per_seq =
        block_size > 0 ? (max_pos + block_size - 1) / block_size : 0;
    const int64_t max_seqs = static_cast<int64_t>(bucket_num_tokens);
    paged_kv_indptr_host_max_numel_ = max_seqs + 1;
    paged_kv_indices_host_max_numel_ = max_seqs * max_blocks_per_seq;
    paged_kv_last_page_len_host_max_numel_ = max_seqs;
  }

  // Guard MUSA graph capture region with a device-level exclusive lock to
  // prevent conflicting GPU work from other streams (e.g., prepare streams) on
  // the same device when using musaStreamCaptureModeGlobal. Capture requires
  // exclusive access, so we use write lock.
  std::optional<std::unique_lock<std::shared_mutex>> capture_lock_guard;
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    auto& capture_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_write_lock(
            device_index_);
    capture_lock_guard.emplace(capture_lock);
  }
  // Use the returned ModelInputParams for graph capture
  // Always use capture stream for plan/update + capture + forward.
  c10::musa::MUSAStream original_stream =
      c10::musa::getCurrentMUSAStream(device_index_);
  c10::musa::MUSAStream capture_stream = capture_stream_;
  if (original_stream != capture_stream) {
    original_stream.synchronize();
    capture_stream.synchronize();
  }
  std::optional<c10::StreamGuard> stream_guard;
  stream_guard.emplace(capture_stream.unwrap());

  // auto& tensor_options = model->options();

  // Update persistent parameters with input data before capture (includes
  // FlashInfer plan/update).
  const bool needs_model_graph_metadata =
      model->requires_graph_forward_metadata();
  auto full_attention_cache =
      MusaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(needs_model_graph_metadata || full_attention_cache.has_value())
      << "MUSA graph capture requires at least one full-attention KV cache "
         "unless the model supplies graph-forward metadata";
  const torch::Tensor k_cache =
      full_attention_cache ? full_attention_cache->first : torch::Tensor();
  const torch::Tensor v_cache =
      full_attention_cache ? full_attention_cache->second : torch::Tensor();
  auto graph_params_opt =
      is_piecewise_
          ? persistent_param_.update_prefill(tokens,
                                             positions,
                                             params,
                                             padded_num_tokens_,
                                             /*return_capture_params=*/true)
          : persistent_param_.update(
                tokens,
                k_cache,
                v_cache,
                positions,
                params,
                padded_num_tokens_,
                /*return_capture_params=*/true,
                /*update_attention_plan=*/!needs_model_graph_metadata);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params_opt.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";

  prepare_model_graph_metadata(
      model,
      persistent_param_.persistent_positions(padded_num_tokens_),
      graph_params_opt.value());

  torch::Tensor& capture_fa3_scheduler_metadata =
      graph_params_opt.value()
          .attn_metadata->fa3_metadata.fa3_scheduler_metadata;
  if (capture_fa3_scheduler_metadata.defined()) {
    // Replay overwrites this captured address, so it must not alias the
    // shared metadata cache.
    capture_fa3_scheduler_metadata = capture_fa3_scheduler_metadata.clone();
  }
  captured_fa3_scheduler_metadata_ = capture_fa3_scheduler_metadata;
  replayed_fa3_scheduler_metadata_ = torch::Tensor();

  // Graph preparation executes eager warmup and FFI-record forwards before
  // the real replay. Each forward mutates GDN convolution and recurrent state,
  // including ordinary decode capture. Snapshot the live sequence rows so
  // every preparation pass starts from the same state and the first replay
  // applies the current token or chunk exactly once.
  const bool snapshot_linear_state =
      params.embedding.linear_state_indices.defined() ||
      !params.embedding.linear_state_ids.empty();
  torch::Tensor linear_state_snapshot_indices;
  std::vector<IndexedTensorSnapshot> linear_state_snapshots;
  if (snapshot_linear_state) {
    linear_state_snapshot_indices =
        get_linear_state_snapshot_indices(params, persistent_param_.device());
    linear_state_snapshots = snapshot_linear_attention_state(
        kv_cache, linear_state_snapshot_indices);
  }

  if (!is_piecewise_) {
    refresh_persistent_paged_kv_host_mirrors(
        graph_params_opt.value().attn_metadata, params.attention.host);
  }
  // Piecewise warmups run on the same bucket rows as the capture.
  std::optional<xllm::kernel::musa::PiecewisePreparationScope>
      piecewise_preparation_scope;
  if (is_piecewise_) {
    piecewise_preparation_scope.emplace();
  }

  LOG(INFO) << "MUSA graph capture begin, bucket_num_tokens: "
            << bucket_num_tokens
            << ", actual_num_tokens: " << actual_num_tokens;

  {
    // Normal capture mode for decode.
    // Reuses the outer `capture_stream` (set up at the top of this function
    // and active via stream_guard) so the warmup runs on the exact stream
    // about to be captured -- syncing a different stream would leave the
    // capture stream with stale pending work.
    const char* skip_graph_warmup_env =
        std::getenv("XLLM_MUSA_GRAPH_SKIP_WARMUP");
    const bool skip_graph_warmup = skip_graph_warmup_env != nullptr &&
                                   std::string(skip_graph_warmup_env) == "1";
    // Warmup and FFI recording run eagerly before capture and must not advance
    // DSV4 compressor/indexer state. The real captured forward below is the
    // first preparation pass allowed to mutate that state.
    xllm::kernel::musa::deepseek_v4_set_graph_capture_warmup(true);
    for (int warmup_iter = 0; warmup_iter < (skip_graph_warmup ? 0 : 2);
         ++warmup_iter) {
      capture_stream.synchronize();
      {
        xllm::kernel::musa::TvmffiPreparationSyncGuard ffi_sync_guard;
        auto warmup_output = model->forward(
            persistent_param_.persistent_tokens(padded_num_tokens_),
            persistent_param_.persistent_positions(padded_num_tokens_),
            kv_cache,
            graph_params_opt.value());
        if (should_capture_graph_aux_hidden_states(args, options) &&
            warmup_output.aux_hidden_states.defined()) {
          persistent_param_.set_aux_hidden_states(
              warmup_output.aux_hidden_states);
        }
      }
      xllm::kernel::musa::sync_ffi_stream(persistent_param_.device());
      if (snapshot_linear_state) {
        // The warmup is an eager forward and therefore mutates the live
        // recurrent state. Restore it before the next warmup so every warmup
        // observes the same input state as the captured replay.
        capture_stream.synchronize();
      }
      if (snapshot_linear_state) {
        restore_linear_attention_state(linear_state_snapshots,
                                       linear_state_snapshot_indices);
      }
    }
    capture_stream.synchronize();

    // Record Mate FFI internal scratch allocations on one extra eager forward.
    // The Mate decode .so allocates via TVM-FFI's DLPackManagedTensorAllocator
    // hook (torch::empty), which MUSA rejects under stream capture. We capture
    // the exact sequence of tensors here and replay them during capture_begin.
    // Piecewise prefill graphs take Mate scratch from the shared graph pool
    // instead, because recorded scratch would stay pinned by every exact-shape
    // prefill graph.
    recorded_ffi_allocs_.clear();
    if (!is_piecewise_ && !skip_graph_warmup) {
      xllm::kernel::musa::begin_ffi_alloc_record();
      {
        xllm::kernel::musa::TvmffiPreparationSyncGuard ffi_sync_guard;
        model->forward(
            persistent_param_.persistent_tokens(padded_num_tokens_),
            persistent_param_.persistent_positions(padded_num_tokens_),
            kv_cache,
            graph_params_opt.value());
      }
      xllm::kernel::musa::sync_ffi_stream(persistent_param_.device());
      if (snapshot_linear_state) {
        // The FFI recording pass is eager as well; leave the live cache at the
        // pre-capture state before beginning graph capture.
        capture_stream.synchronize();
      }
      if (snapshot_linear_state) {
        restore_linear_attention_state(linear_state_snapshots,
                                       linear_state_snapshot_indices);
      }
      recorded_ffi_allocs_ = xllm::kernel::musa::end_ffi_alloc_record();
      capture_stream.synchronize();
      LOG(INFO) << "Recorded " << recorded_ffi_allocs_.size()
                << " Mate FFI scratch tensors for decode graph capture, "
                   "bucket_num_tokens="
                << bucket_num_tokens;
    }

    // Allocations made during capture can only be served by the graph pool,
    // and the caching allocator cannot release default-pool blocks until
    // capture ends. Return them to the device before a lazy capture.
    Device::empty_cache(/*device_index=*/-1);

    // MemPoolContext has been deprecated in torch >= 2.8
#if TORCH_VERSION_MAJOR <= 2 && TORCH_VERSION_MINOR <= 7
    // Activate VMM mempool only for the actual capture to keep plan_info
    // allocations out of the shared physical memory pool.
    std::optional<c10::musa::MemPoolContext> mempool_ctx;
    if (pool_ptr != nullptr) {
      mempool_ctx.emplace(pool_ptr);
    }
#endif

    // Begin graph capture (capture_mode defaults to
    // musaStreamCaptureModeGlobal)
    // graph_.capture_begin(pool);
    void* const capture_stream_handle =
        reinterpret_cast<void*>(capture_stream.stream());
    std::optional<xllm::kernel::musa::TvmffiStreamOverrideGuard>
        ffi_capture_stream_guard;
    ffi_capture_stream_guard.emplace(persistent_param_.device(),
                                     capture_stream_handle);
    xllm::kernel::musa::deepseek_v4_set_graph_capture_warmup(false);
    GlobalCaptureInstance& piecewise_capture =
        GlobalCaptureInstance::get_instance();
    if (is_piecewise_) {
      piecewise_capture.begin_capture(pool, pool_ptr != nullptr);
    } else {
      if (s_graph_dump_directory() != nullptr) {
        graph_.enable_debug_mode();
      }
      graph_.capture_begin(
          pool, musaStreamCaptureModeThreadLocal, pool_ptr != nullptr);
    }

    if (!is_piecewise_) {
      xllm::kernel::musa::begin_ffi_alloc_replay(&recorded_ffi_allocs_);
    }
    // Execute forward pass; the MUSA graph captures this work.
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (should_capture_graph_aux_hidden_states(args, options) &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }

    // End graph capture
    if (is_piecewise_) {
      piecewise_graph_ = piecewise_capture.end_capture();
      CHECK(piecewise_graph_ != nullptr)
          << "MUSA piecewise prefill capture returned null";
      LOG(INFO) << "MUSA piecewise prefill capture: graphs="
                << piecewise_graph_->size()
                << ", runners=" << piecewise_graph_->num_runners();
    } else {
      graph_.capture_end();
      xllm::kernel::musa::deepseek_v4_set_graph_capture_warmup(false);
      if (const char* dump_directory = s_graph_dump_directory()) {
        const uint32_t capture_index =
            s_completed_musa_graph_captures.load(std::memory_order_relaxed) + 1;
        const std::string dump_path =
            std::string(dump_directory) + "/target_device" +
            std::to_string(device_index_) + "_capture" +
            std::to_string(capture_index) + ".dot";
        graph_.debug_dump(dump_path);
        LOG(INFO) << "MUSA graph diagnostic dump: " << dump_path;
      }
    }
    ffi_capture_stream_guard.reset();
    if (!is_piecewise_) {
      xllm::kernel::musa::end_ffi_alloc_replay();
    }
    if (snapshot_linear_state) {
      capture_stream.synchronize();
    }
    if (snapshot_linear_state) {
      // The captured forward executes once while recording and mutates the
      // live convolution/SSM cache. Restore the pre-capture snapshot so the
      // first replay applies the current decode input exactly once.
      restore_linear_attention_state(linear_state_snapshots,
                                     linear_state_snapshot_indices);
    }
  }

  if (params.is_spec_verify && params.gdn_mtp_verify_cache != nullptr &&
      params.gdn_mtp_verify_cache->enabled) {
    is_spec_verify_graph_ = true;
    captured_verify_states_ = params.gdn_mtp_verify_cache->layer_states;
    CHECK(!captured_verify_states_.empty())
        << "spec-verify graph capture produced no GDN verify checkpoints";
  }

  // Synchronize to ensure graph capture is completed.
  capture_stream.synchronize();

  // The external custom-allreduce backend records graph buffer addresses
  // during capture. Exchange and register their IPC metadata before replay.
  finalize_musa_custom_allreduce_graphs();

  // Explicitly restore stream after capture before replay logic.
  stream_guard.reset();

  // Replay is unified in MusaGraphExecutorImpl::run() after capture success.

  s_completed_musa_graph_captures.fetch_add(1, std::memory_order_release);

  LOG(INFO) << "MUSA graph capture end, bucket_num_tokens: "
            << bucket_num_tokens;
  return true;
}

ModelOutput MusaGraph::replay(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_cache,
                              const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, padded_num_tokens_)
      << "num_tokens mismatch: expected <= " << padded_num_tokens_ << ", got "
      << actual_num_tokens;

  // Guard MUSA graph replay with a device-level shared lock to allow multiple
  // replay operations to run concurrently while preventing conflicts with
  // capture operations. Replay can share the lock with other replay/prepare
  // operations.
  std::optional<std::shared_lock<std::shared_mutex>> replay_lock_guard;
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    auto& replay_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_read_lock(
            device_index_);
    replay_lock_guard.emplace(replay_lock);
  }

  // Update persistent parameters with new input data
  const bool needs_model_graph_metadata =
      model_ != nullptr && model_->requires_graph_forward_metadata();
  auto full_attention_cache =
      MusaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(needs_model_graph_metadata || full_attention_cache.has_value())
      << "MUSA graph replay requires at least one full-attention KV cache "
         "unless the model supplies graph-forward metadata";
  const torch::Tensor k_cache =
      full_attention_cache ? full_attention_cache->first : torch::Tensor();
  const torch::Tensor v_cache =
      full_attention_cache ? full_attention_cache->second : torch::Tensor();

  if (is_piecewise_) {
    auto replay_params_opt =
        persistent_param_.update_prefill(tokens,
                                         positions,
                                         params,
                                         padded_num_tokens_,
                                         /*return_capture_params=*/true);
    CHECK(replay_params_opt.has_value())
        << "update_prefill() should return ModelInputParams for replay";
    CHECK(piecewise_graph_ != nullptr)
        << "piecewise prefill replay requires a captured graph";
    ::xllm::kernel::musa::AttentionReplayParams runner_params;
    runner_params.actual_num_tokens = actual_num_tokens;
    runner_params.attn_metadata = replay_params_opt->attn_metadata;
    piecewise_graph_->replay(runner_params);
    return ModelOutput(get_hidden_states(actual_num_tokens));
  }

  {
    // Normal replay mode (for decode).
    //
    // Request the metadata back from update() so we can refresh the
    // per-MusaGraph persistent host mirrors of paged_kv_* before replaying.
    // The captured graph holds (stable) pointers to our pinned host buffers
    // from capture time; the data inside those buffers must reflect the
    // current step's paged-KV layout, which is what update() just materialized
    // on the corresponding persistent *device* tensors. The returned
    // ModelInputParams is otherwise unused in the replay branch -- it is a
    // single small per-step allocation that pays for itself by avoiding the
    // page fault inside the captured Mate decode kernel (see .mudmp under
    // repro logs and refresh_persistent_paged_kv_host_mirrors() for the
    // pointer-stability rationale).
    auto replay_params_opt =
        persistent_param_.update(tokens,
                                 k_cache,
                                 v_cache,
                                 positions,
                                 params,
                                 padded_num_tokens_,
                                 /*return_capture_params=*/true,
                                 /*update_attention_plan=*/
                                 !needs_model_graph_metadata);
    CHECK(replay_params_opt.has_value())
        << "update() should return ModelInputParams for decode replay";

    prepare_model_graph_metadata(
        model_,
        persistent_param_.persistent_positions(padded_num_tokens_),
        replay_params_opt.value());

    // persistent_param_.update() and model-private DSA metadata preparation
    // run on the caller stream, while the captured graph replays on the
    // dedicated capture stream.  A host-side synchronize of the caller is
    // insufficient: the capture stream may start before the metadata copies
    // become visible.  Add an explicit producer->capture event for the
    // DeepSeek diagnostic first; this is opt-in until its accuracy/perf impact
    // is measured.
    static const bool s_sync_dsv4_graph_metadata = [] {
      const char* env = std::getenv("XLLM_DSV4_GRAPH_METADATA_EVENT");
      return env != nullptr && std::string(env) == "1";
    }();
    if (s_sync_dsv4_graph_metadata) {
      c10::musa::MUSAStream producer_stream =
          c10::musa::getCurrentMUSAStream(device_index_);
      c10::Event metadata_ready(producer_stream.device_type());
      metadata_ready.record(producer_stream);
      metadata_ready.block(capture_stream_);
      LOG_FIRST_N(INFO, 1)
          << "[DSV4_GRAPH_ROUTE] metadata producer event waits on capture "
             "stream";
    }

    const torch::Tensor& fresh_fa3_scheduler_metadata =
        replay_params_opt.value()
            .attn_metadata->fa3_metadata.fa3_scheduler_metadata;
    if (captured_fa3_scheduler_metadata_.defined()) {
      CHECK(fresh_fa3_scheduler_metadata.defined())
          << "FA3 scheduler metadata disappeared after graph capture";
      CHECK_EQ(captured_fa3_scheduler_metadata_.sizes(),
               fresh_fa3_scheduler_metadata.sizes())
          << "FA3 scheduler metadata shape changed after graph capture";
      if (!fresh_fa3_scheduler_metadata.is_same(
              replayed_fa3_scheduler_metadata_)) {
        captured_fa3_scheduler_metadata_.copy_(fresh_fa3_scheduler_metadata,
                                               /*non_blocking=*/true);
        replayed_fa3_scheduler_metadata_ = fresh_fa3_scheduler_metadata;
      }
    }

    // During graph replay, the captured graph reads paged-KV metadata from
    // persistent *device* tensors (updated by update_llm_decode_metadata_fast
    // _path() above). The pinned host mirrors were set up during capture
    // (see capture() path) and their pointers are baked into the graph, but
    // the graph's FFI batch_decode() call casts paged_kv_*_host to (void) —
    // they are NOT read during replay. The plan_info that consumed the host
    // mirror is cached after first creation and never recomputed on replay.
    //
    // Skipping refresh_persistent_paged_kv_host_mirrors() here avoids 3
    // blocking D2H copies (paged_kv_indptr/indices/last_page_len) that would
    // otherwise stall the CPU for ~48 ms waiting for the previous graph to
    // complete on the same stream. Set XLLM_KEEP_HOST_MIRROR_REFRESH=1 to
    // re-enable for debugging.
    static const bool s_keep_host_mirror_refresh = [] {
      const char* env = std::getenv("XLLM_KEEP_HOST_MIRROR_REFRESH");
      return env && std::string(env) == "1";
    }();
    if (s_keep_host_mirror_refresh) {
      refresh_persistent_paged_kv_host_mirrors(
          replay_params_opt.value().attn_metadata, params.attention.host);
    }

    // Optional diagnostic barrier for persistent input copies. The update path
    // runs on the caller stream while the captured graph is launched on
    // capture_stream_; a barrier here distinguishes a stale-input race from
    // kernel arithmetic differences. Keep it opt-in because the synchronous
    // version is intentionally not the performance default.
    static const bool s_sync_graph_inputs = [] {
      const char* env = std::getenv("XLLM_MUSA_GRAPH_SYNC_INPUTS");
      return env && std::string(env) == "1";
    }();
    if (s_sync_graph_inputs) {
      c10::musa::getCurrentMUSAStream(device_index_).synchronize();
    }

    // Timing is diagnostic-only. Do not synchronize the first q1 replay: the
    // target q2 and draft q2 graphs have not been captured yet, and a device
    // synchronization from that replay is illegal while another executor is
    // entering capture. For the TP8/MTP validation shape all three captures
    // complete before steady-state replay begins.
    const char* timing_all_env = std::getenv("XLLM_GRAPH_TIMING_ALL");
    const bool time_graph_replay =
        s_enable_graph_timing() &&
        ((timing_all_env != nullptr && std::string(timing_all_env) == "1") ||
         s_completed_musa_graph_captures.load(std::memory_order_acquire) >= 3);
    if (time_graph_replay) {
      CHECK_EQ(musaDeviceSynchronize(), musaSuccess)
          << "failed to synchronize before MUSA graph timing";
      const auto replay_start = std::chrono::steady_clock::now();
      graph_.replay();
      CHECK_EQ(musaDeviceSynchronize(), musaSuccess)
          << "failed to synchronize after MUSA graph timing";
      const auto replay_end = std::chrono::steady_clock::now();
      const auto replay_ms =
          std::chrono::duration<double, std::milli>(replay_end - replay_start)
              .count();
      LOG(INFO) << "GRAPH_TIMING actual_num_tokens=" << actual_num_tokens
                << " padded_num_tokens=" << padded_num_tokens_
                << " spec_verify=" << is_spec_verify_graph_
                << " replay_ms=" << replay_ms;
    } else {
      graph_.replay();
    }

    // Optional completion barrier for cross-request isolation diagnostics.
    // The graph runs on capture_stream_, while the next scheduler request may
    // arrive on another worker stream. Waiting here verifies whether a prior
    // replay is still mutating shared KV/output storage when the next request
    // starts. Keep opt-in until the performance cost is measured.
    static const bool s_sync_graph_outputs = [] {
      const char* env = std::getenv("XLLM_MUSA_GRAPH_SYNC_OUTPUTS");
      return env && std::string(env) == "1";
    }();
    if (s_sync_graph_outputs) {
      capture_stream_.synchronize();
    }
  }

  if (is_spec_verify_graph_) {
    CHECK(params.gdn_mtp_verify_cache != nullptr &&
          params.gdn_mtp_verify_cache->enabled)
        << "spec-verify graph replay requires the GDN verify cache";
    params.gdn_mtp_verify_cache->layer_states = captured_verify_states_;
  }

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in MusaGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  ModelOutput output(get_hidden_states(actual_num_tokens));
  return output;
}

MusaGraphExecutorImpl::MusaGraphExecutorImpl(CausalLM* model,
                                             const ModelArgs& args,
                                             const torch::Device& device,
                                             const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  // Keep one pool per executor instance so all captured graphs can reuse it,
  // while avoiding cross-instance stale-handle reuse.
  graph_pool_ = at::musa::graph_pool_handle();
  max_tokens_for_graph_mode_ =
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode();
  if (max_tokens_for_graph_mode_ < options_.max_seqs_per_batch()) {
    max_tokens_for_graph_mode_ = options_.max_seqs_per_batch();
  }
  // Create single persistent parameter object shared by all MusaGraph instances
  persistent_param_ =
      std::make_unique<MusaGraphPersistentParam>(args_, device_, options_);
  const size_t persistent_bytes =
      persistent_param_->get_persistent_tensor_bytes();
  LOG(INFO) << "Persistent input tensor total size: " << persistent_bytes
            << " bytes (" << (persistent_bytes / (1024 * 1024)) << " MB)";

  const auto private_pool_usage =
      get_private_pools_memory_usage(device_.index());
  baseline_private_pool_reserved_bytes_ = private_pool_usage.reserved_bytes;
  baseline_private_pool_allocated_bytes_ = private_pool_usage.allocated_bytes;
  baseline_private_pool_active_bytes_ = private_pool_usage.active_bytes;
  baseline_allocator_reserved_bytes_ =
      get_allocator_reserved_bytes(device_.index());
}

std::optional<std::pair<torch::Tensor, torch::Tensor>>
MusaGraphExecutorImpl::find_first_full_attention_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    if (cache.empty()) {
      continue;
    }
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return std::make_pair(std::move(k_cache), std::move(v_cache));
    }
  }
  return std::nullopt;
}

namespace {
constexpr uint32_t kPhysicalPoolIdDecode = 1;
constexpr uint32_t kPhysicalPoolIdPrefill = 2;
}  // namespace

struct MusaGraphExecutorImpl::VmmPoolState {
  std::unique_ptr<::xllm::MusaVmmCapturePool> capture_pool;
};

MusaGraphExecutorImpl::~MusaGraphExecutorImpl() {
  // Drain submitted work before graph-owned pools and their physical pages
  // are released. Each graph destroys its handles before its logical pool.
  std::optional<c10::musa::MUSAGuard> device_guard;
  if (!vmm_pools_.empty()) {
    device_guard.emplace(device_.index());
    CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  }
  prefill_graphs_.clear();
  graphs_.clear();
  spec_verify_graphs_.clear();
  if (!vmm_pools_.empty()) {
    // Persistent outputs can retain a block from a user pool. Drop those
    // tensors and flush freeable pool blocks while their allocator is alive.
    persistent_param_.reset();
    Device::empty_cache(/*device_index=*/-1);
  }
  vmm_pools_.clear();
}

MusaGraphExecutorImpl::VmmPoolState&
MusaGraphExecutorImpl::get_or_create_vmm_pool_state(uint32_t physical_pool_id) {
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto& slot = vmm_pools_[physical_pool_id];
  if (!slot) {
    auto state = std::make_unique<VmmPoolState>();
    state->capture_pool =
        std::make_unique<::xllm::MusaVmmCapturePool>(device_.index());
    slot = std::move(state);
    LOG(INFO) << "Created VMM pool state for executor " << this << ", device "
              << device_.index() << ", physical_pool_id: " << physical_pool_id;
  }
  return *slot;
}

std::unique_ptr<MusaMemPool> MusaGraphExecutorImpl::create_vmm_mempool(
    uint32_t physical_pool_id) {
  if (!supports_graph_vmm_pool() ||
      !::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
    return nullptr;
  }
  VmmPoolState& state = get_or_create_vmm_pool_state(physical_pool_id);
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto pool = state.capture_pool->create_pool();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Created VMM MemPool for executor " << this
      << ", physical_pool_id: " << physical_pool_id << ", pool_id: {"
      << pool->id().first << ", " << pool->id().second << "}";
  return pool;
}

MusaGraphExecutorImpl::GraphMemoryUsageStats
MusaGraphExecutorImpl::get_graph_memory_usage_stats() {
  GraphMemoryUsageStats stats;

  const bool vmm_enabled =
      supports_graph_vmm_pool() &&
      ::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool();

  if (!vmm_enabled) {
    const auto usage =
        get_graph_pool_memory_usage(device_.index(), graph_pool_);
    stats.executor_total_bytes = usage.reserved_bytes;
    stats.allocated_pool_bytes = usage.allocated_bytes;
    stats.active_pool_bytes = usage.active_bytes;
    stats.pool_high_water_mark_bytes = usage.allocated_bytes;

    if (stats.executor_total_bytes == 0) {
      const auto private_pool_usage =
          get_private_pools_memory_usage(device_.index());
      if (private_pool_usage.reserved_bytes >
          baseline_private_pool_reserved_bytes_) {
        stats.executor_total_bytes = private_pool_usage.reserved_bytes -
                                     baseline_private_pool_reserved_bytes_;
      }
      if (private_pool_usage.allocated_bytes >
          baseline_private_pool_allocated_bytes_) {
        stats.allocated_pool_bytes = private_pool_usage.allocated_bytes -
                                     baseline_private_pool_allocated_bytes_;
      }
      if (private_pool_usage.active_bytes >
          baseline_private_pool_active_bytes_) {
        stats.active_pool_bytes = private_pool_usage.active_bytes -
                                  baseline_private_pool_active_bytes_;
      }
      stats.pool_high_water_mark_bytes = stats.allocated_pool_bytes;
    }

    if (stats.executor_total_bytes == 0) {
      const size_t allocator_reserved_bytes =
          get_allocator_reserved_bytes(device_.index());
      if (allocator_reserved_bytes > baseline_allocator_reserved_bytes_) {
        stats.executor_total_bytes =
            allocator_reserved_bytes - baseline_allocator_reserved_bytes_;
      }
    }
  } else {
    std::lock_guard<std::mutex> lock(vmm_mutex_);
    for (const auto& kv : vmm_pools_) {
      const VmmPoolState& pool_state = *kv.second;
      stats.executor_total_bytes += pool_state.capture_pool->mapped_size();
      stats.allocated_pool_bytes += pool_state.capture_pool->high_water_mark();
      stats.active_pool_bytes += pool_state.capture_pool->current_offset();
    }
    stats.pool_high_water_mark_bytes = stats.allocated_pool_bytes;
  }

  stats.persistent_param_bytes =
      persistent_param_ ? persistent_param_->get_persistent_tensor_bytes() : 0;
  stats.executor_total_bytes += stats.persistent_param_bytes;

  return stats;
}

size_t MusaGraphExecutorImpl::get_graph_memory_usage_bytes() {
  return get_graph_memory_usage_stats().executor_total_bytes;
}

void MusaGraphExecutorImpl::log_graph_memory_after_capture() {
  const auto stats = get_graph_memory_usage_stats();
  if (stats.executor_total_bytes <= last_logged_executor_total_bytes_) {
    return;
  }
  last_logged_executor_total_bytes_ = stats.executor_total_bytes;

  const bool vmm_enabled =
      supports_graph_vmm_pool() &&
      ::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool();
  auto to_mb = [](size_t bytes) { return bytes / (1024 * 1024); };
  LOG(INFO) << "GraphExecutorMemory Usage:"
            << " enable_graph_vmm_pool=" << vmm_enabled
            << " executor_total_memory=" << to_mb(stats.executor_total_bytes)
            << " MB persistent_param=" << to_mb(stats.persistent_param_bytes)
            << " MB allocated_pool_memory=" << to_mb(stats.allocated_pool_bytes)
            << " MB active_pool_memory=" << to_mb(stats.active_pool_bytes)
            << " MB pool_high_water_mark="
            << to_mb(stats.pool_high_water_mark_bytes) << " MB";
}

// Static method to get the MUSA capture stream for the current thread.
// Each thread gets its own high-priority capture stream
c10::musa::MUSAStream MusaGraphExecutorImpl::get_capture_stream(
    c10::DeviceIndex device_index) {
  // Use thread_local to ensure each thread has its own capture stream
  // This is required because MUSA graphs must be captured on a non-default
  // stream. We use high-priority streams for better performance.
  thread_local c10::musa::MUSAStream thread_capture_stream =
      c10::musa::getStreamFromPool(/*isHighPriority=*/true, device_index);

  // Thread-local counter to log initialization only once per thread
  thread_local bool initialized = false;
  if (!initialized) {
    LOG(INFO) << "Initialized capture_stream for thread: "
              << std::this_thread::get_id()
              << ", stream: " << thread_capture_stream
              << ", device_index: " << device_index;
    initialized = true;
  }

  return thread_capture_stream;
}

ForwardInput MusaGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  uint32_t min_decoding_batch_size = 0;
  const char* min_batch_env = std::getenv("XLLM_DSV4_MIN_DECODE_BATCH");
  if (min_batch_env != nullptr && *min_batch_env != '\0') {
    const long parsed = std::strtol(min_batch_env, nullptr, 10);
    if (parsed > 0 && parsed <= std::numeric_limits<uint32_t>::max()) {
      min_decoding_batch_size = static_cast<uint32_t>(parsed);
    }
  }
  return batch.prepare_forward_input(options_.num_decoding_tokens(),
                                     min_decoding_batch_size,
                                     args_,
                                     options_.cp_size());
}

ModelOutput MusaGraphExecutorImpl::attach_aux_hidden_states_if_needed(
    const torch::Tensor& hidden_states,
    uint32_t n_tokens) const {
  if (should_capture_graph_aux_hidden_states(args_, options_)) {
    auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
    if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
  }
  return ModelOutput(hidden_states);
}

ModelInputParams MusaGraphExecutorImpl::maybe_precompute_embedding_for_graph(
    const torch::Tensor& tokens,
    const ModelInputParams& params) const {
  const auto& forward_type = params.meta.batch_forward_type;
  if (!forward_type.is_decode() && !forward_type.is_prefill() &&
      !forward_type.is_chunked_prefill()) {
    return params;
  }

  auto embed_layer = model_->get_word_embedding();
  if (embed_layer.is_empty()) {
    return params;
  }

  // Qwen3.5 MTP draft: input_embedding carries the target hidden state, so
  // the token embedding must also be precomputed outside the captured region
  // and fed through the persistent MTP token-embedding buffer.
  const bool is_qwen35_mtp = args_.model_type() == "qwen3_5_mtp" ||
                             args_.model_type() == "qwen3_5_moe_mtp";
  if (is_qwen35_mtp && params.embedding.input_embedding.defined() &&
      !params.embedding.mtp_token_embedding.defined()) {
    const int64_t vocab = embed_layer->weight().size(0);
    // Placeholder (-1) rows are padding/rejected slots; clamp without a host
    // sync so the lookup stays in-bounds.
    torch::Tensor safe_tokens = tokens.clamp(/*min=*/0, /*max=*/vocab - 1);
    ModelInputParams new_params = params;
    new_params.embedding.mtp_token_embedding = embed_layer(safe_tokens);
    return new_params;
  }

  if (args_.model_type() == "deepseek_v4_mtp") {
    // The MTP graph already carries the target model's pre-HC context in
    // input_embedding. Append the current token embedding outside capture;
    // the MTP forward splits this stable packed tensor into target context and
    // draft-token lanes.
    CHECK(params.embedding.input_embedding.defined())
        << "DeepSeek-V4 MTP Graph decode requires target context";
    CHECK_EQ(params.embedding.input_embedding.dim(), 2)
        << "DeepSeek-V4 MTP target context must be rank 2";
    ModelInputParams new_params = params;
    torch::Tensor token_embedding = embed_layer(tokens);
    new_params.embedding.input_embedding = torch::cat(
        {params.embedding.input_embedding, token_embedding}, /*dim=*/-1);
    return new_params;
  }

  if (params.embedding.input_embedding.defined()) {
    return params;
  }

  // The downstream wiring is already in place:
  //   * MusaGraphPersistentParam::update() copies `params.embedding
  //     .input_embedding` into `persistent_embedding_` (see the update path
  //     under "Update persistent embedding from input_embedding if
  //     available").
  //   * For capture, `build_capture_params_if_needed` rewrites
  //     `params_for_capture->embedding.input_embedding` to a view of
  //     `persistent_embedding_`, so the captured forward references the
  //     persistent buffer's stable address.
  //   * For replay, the captured graph already references that same
  //     persistent address; refreshing the buffer contents here is sufficient
  //     to feed each step with the correct per-token embeddings.
  //
  // Qwen3Next-family models (including Qwen3.5) already honour
  // `input_params.embedding.input_embedding` in their forward, branching
  // around the in-graph `embed_tokens_(tokens)` call when the field is
  // defined (see xllm/models/llm/qwen3_next_hybrid_base.h).
  ModelInputParams new_params = params;
  torch::Tensor input_embedding = embed_layer(tokens);
  if (args_.model_type() == "deepseek_v4" && input_embedding.dim() == 2) {
    // TorchMUSA repeat() allocates its output and is illegal while a stream is
    // capturing.  Materialize the HC lanes here; update() copies them into a
    // stable persistent tensor referenced by both capture and replay.
    input_embedding =
        input_embedding.unsqueeze(1).repeat({1, args_.hc_mult(), 1});
  }
  new_params.embedding.input_embedding = std::move(input_embedding);
  return new_params;
}

ModelOutput MusaGraphExecutorImpl::run(const torch::Tensor& tokens,
                                       const torch::Tensor& positions,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  const bool is_prefill = params.meta.batch_forward_type.is_prefill();
  const bool is_decode = params.meta.batch_forward_type.is_decode();

  // Diagnostic correctness mode: a captured DeepSeek graph owns model-private
  // graph metadata/state carriers. Re-capturing at each new prefill isolates
  // whether cross-request graph reuse, rather than the attention kernel, is
  // responsible for request-order-dependent logits. It is opt-in because
  // capture cost is intentionally excluded from the production fast path.
  if (is_prefill && args_.model_type() == "deepseek_v4") {
    const char* recapture_env =
        std::getenv("XLLM_DSV4_RECAPTURE_GRAPH_ON_PREFILL");
    if (recapture_env != nullptr && std::string(recapture_env) == "1" &&
        !graphs_.empty()) {
      graphs_.clear();
      LOG(INFO) << "[DSV4_GRAPH_ROUTE] recaptured Graph cache at prefill "
                   "request boundary.";
    }
  }

  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens.size(/*dim=*/0);
  const int64_t effective_num_sequences = params.meta.actual_num_sequences > 0
                                              ? params.meta.actual_num_sequences
                                              : params.meta.num_sequences;
  const bool multi_sequence_prefill = is_prefill && effective_num_sequences > 1;

  // Prefill and chunked prefill replay piecewise graphs; launches that need
  // host-side batch layout run as runners between the captured segments.
  if (use_prefill_piecewise_graph(n_tokens, params)) {
    const uint32_t graph_num_tokens =
        get_prefill_graph_num_tokens(n_tokens, params);
    const std::vector<int32_t> graph_key =
        get_prefill_graph_key(n_tokens, params);
    const ModelInputParams graph_params =
        maybe_precompute_embedding_for_graph(tokens, params);
    auto it = prefill_graphs_.find(graph_key);
    if (it != prefill_graphs_.end()) {
      auto result =
          it->second->replay(tokens, positions, kv_caches, graph_params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }
    auto graph =
        std::make_unique<MusaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()),
                                    /*is_piecewise=*/true);
    auto pool_owner = create_vmm_mempool(kPhysicalPoolIdPrefill);
    const c10::musa::MempoolId_t mem_pool =
        pool_owner != nullptr ? pool_owner->id() : graph_pool_;
    const bool capture_success = graph->capture(model_,
                                                args_,
                                                options_,
                                                tokens,
                                                positions,
                                                graph_params,
                                                kv_caches,
                                                graph_num_tokens,
                                                mem_pool,
                                                std::move(pool_owner));
    CHECK(capture_success)
        << "Failed to capture MUSA piecewise prefill graph, num_tokens="
        << graph_num_tokens << " (actual num_tokens: " << n_tokens << ")";
    LOG(INFO) << "Lazy capturing MUSA piecewise prefill graph num_tokens="
              << graph_num_tokens << " (actual num_tokens: " << n_tokens
              << ") num_sequences=" << effective_num_sequences << " chunked="
              << params.meta.batch_forward_type.is_chunked_prefill() << " done";
    log_graph_memory_after_capture();
    auto& stored = prefill_graphs_[graph_key];
    stored = std::move(graph);
    auto result = stored->replay(tokens, positions, kv_caches, graph_params);
    return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
  }

  // Prefill without a piecewise graph runs eager.
  if (is_prefill) {
    COUNTER_INC(num_model_execution_total_eager);
    const bool time_fwd = s_enable_prefill_fwd_timing();
    if (time_fwd) {
      c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
      const auto t0 = std::chrono::steady_clock::now();
      auto result = model_->forward(tokens, positions, kv_caches, params);
      c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
      const auto t1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (s_enable_prefill_fwd_timing()) {
        LOG(INFO) << "[PREFILL_FWD] n_tokens=" << n_tokens
                  << " batch_bs=" << params.meta.num_sequences
                  << " packed_prefill=" << multi_sequence_prefill
                  << " mode=eager"
                  << " fwd_ms=" << ms;
      }
      maybe_empty_prefill_cache();
      return result;
    }
    auto result = model_->forward(tokens, positions, kv_caches, params);
    maybe_empty_prefill_cache();
    return result;
  }

  // MTP target validate (Qwen3.5 hybrid): expanded-decode attention + GPU
  // spec-verify conv / Mate GDN checkpoint kernel, captured per
  // (num_tokens, q_len). No padding: num_tokens = batch * (K+1).
  const bool spec_verify_graph = [&]() {
    if (!(params.is_spec_verify &&
          params.meta.batch_forward_type.is_chunked_prefill() &&
          params.gdn_mtp_verify_cache != nullptr &&
          params.gdn_mtp_verify_cache->enabled &&
          params.meta.kv_max_seq_len <= args_.max_position_embeddings())) {
      return false;
    }
    // Diagnostic only: XLLM_FORCE_EAGER_SPEC_VERIFY=1 skips verify-graph
    // capture so Kineto can see per-kernel times. Do not use in production.
    const char* env = std::getenv("XLLM_FORCE_EAGER_SPEC_VERIFY");
    if (env != nullptr && std::string(env) == "1") {
      LOG_FIRST_N(WARNING, 1)
          << "XLLM_FORCE_EAGER_SPEC_VERIFY=1: verify runs eager for profiling";
      return false;
    }
    return true;
  }();
  if (params.meta.batch_forward_type.is_chunked_prefill() &&
      params.is_spec_verify) {
    LOG_FIRST_N(INFO, 3)
        << "[SPEC_VERIFY_GRAPH] eligible=" << spec_verify_graph << " expanded="
        << params.graph.use_expanded_decode_for_spec_verify_attention
        << " verify_cache=" << (params.gdn_mtp_verify_cache != nullptr)
        << " cache_enabled="
        << (params.gdn_mtp_verify_cache != nullptr &&
            params.gdn_mtp_verify_cache->enabled)
        << " kv_max=" << params.meta.kv_max_seq_len
        << " q_max=" << params.meta.q_max_seq_len << " n_tokens=" << n_tokens;
  }
  if (spec_verify_graph) {
    const uint64_t q_len =
        static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1));
    const uint64_t key = static_cast<uint64_t>(n_tokens) | (q_len << 32);
    auto it = spec_verify_graphs_.find(key);
    if (it != spec_verify_graphs_.end()) {
      auto result = it->second->replay(tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }
    auto graph =
        std::make_unique<MusaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    auto pool_owner = create_vmm_mempool(kPhysicalPoolIdDecode);
    const c10::musa::MempoolId_t mem_pool =
        pool_owner != nullptr ? pool_owner->id() : graph_pool_;
    const bool ok = graph->capture(model_,
                                   args_,
                                   options_,
                                   tokens,
                                   positions,
                                   params,
                                   kv_caches,
                                   n_tokens,
                                   mem_pool,
                                   std::move(pool_owner));
    CHECK(ok) << "Failed to capture MUSA spec-verify graph, num_tokens="
              << n_tokens << " q_len=" << q_len;
    LOG(INFO) << "Lazy capturing MUSA spec-verify graph num_tokens=" << n_tokens
              << " q_len=" << q_len << " done";
    log_graph_memory_after_capture();
    auto& stored = spec_verify_graphs_[key];
    stored = std::move(graph);
    auto result = stored->replay(tokens, positions, kv_caches, params);
    return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
  }

  if (params.meta.batch_forward_type.is_chunked_prefill() ||
      params.meta.batch_forward_type.is_mixed()) {
    COUNTER_INC(num_model_execution_total_eager);
    auto result = model_->forward(tokens, positions, kv_caches, params);
    maybe_empty_prefill_cache();
    return result;
  }

  uint32_t bucket_num_tokens = get_bucket_num_tokens(n_tokens);
  const char* stable_multi_graph_env =
      std::getenv("XLLM_DSV4_STABLE_MULTI_GRAPH_SEQUENCES");
  const bool stable_multi_graph_sequences =
      stable_multi_graph_env != nullptr &&
      std::string(stable_multi_graph_env) == "1";
  const bool reuse_existing_graph =
      stable_multi_graph_sequences && is_decode &&
      args_.model_type() == "deepseek_v4" && params.is_spec_verify &&
      options_.num_decoding_tokens() > 1 && options_.max_seqs_per_batch() > 1 &&
      layer::dsv4_graph_sequence_views_enabled() &&
      layer::dsv4_reusable_graph_capacity(params.attention.host.kv_seq_lens,
                                          options_.num_decoding_tokens()) > 0;
  if (reuse_existing_graph) {
    // First capture only fully populated real rows. Larger shapes are eligible
    // for reuse only after they have been captured by an actual larger batch.
    bucket_num_tokens = n_tokens;
  }

  // Honor the public graph token limit on MUSA as the CUDA executor does.
  // Previously every decode shape entered capture/replay regardless of
  // max_tokens_for_graph_mode, making an explicit eager accuracy comparison
  // impossible and allowing unsupported buckets to retain stale graph state.
  const int32_t max_graph_tokens = options_.max_tokens_for_graph_mode();
  if (max_graph_tokens == 0 ||
      (max_graph_tokens > 0 &&
       static_cast<int64_t>(bucket_num_tokens) > max_graph_tokens)) {
    LOG(INFO) << "MUSA graph bucket " << bucket_num_tokens
              << " exceeds max_tokens_for_graph_mode (" << max_graph_tokens
              << "), falling back to eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Decode phase with full graph
  if (is_decode) {
    // Diagnostic control for DeepSeek-V4 DSpark compact verification. The
    // target q2 Graph path owns a different persistent DSA state carrier than
    // draft q1; keep eager target verification opt-in until that carrier is
    // made numerically equivalent.
    const char* force_eager_compact_env =
        std::getenv("XLLM_DSV4_FORCE_EAGER_COMPACT_VERIFY");
    const char* enable_graph_compact_env =
        std::getenv("XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY");
    const bool graph_compact_opt_in =
        enable_graph_compact_env != nullptr &&
        std::string(enable_graph_compact_env) == "1";
    const bool force_eager_compact_verify =
        force_eager_compact_env != nullptr &&
        std::string(force_eager_compact_env) == "1" &&
        args_.model_type() == "deepseek_v4" && n_tokens > 1 &&
        params.is_spec_verify;
    const bool default_eager_compact_verify =
        !graph_compact_opt_in && args_.model_type() == "deepseek_v4" &&
        n_tokens > 1 && params.is_spec_verify;
    int64_t graph_kv_len_hint = params.meta.kv_max_seq_len;
    if (graph_kv_len_hint <= 0 && !params.attention.host.kv_seq_lens.empty()) {
      const auto& host_kv = params.attention.host.kv_seq_lens;
      graph_kv_len_hint = *std::max_element(host_kv.begin(), host_kv.end());
      if (host_kv.size() > 1 && host_kv.front() == 0) {
        graph_kv_len_hint = host_kv.back();
      }
    }
    const bool short_fullkv_eager =
        graph_compact_opt_in && s_enable_dsv4_graph_short_fullkv_eager() &&
        args_.model_type() == "deepseek_v4" && n_tokens > 1 &&
        params.is_spec_verify && graph_kv_len_hint > 0 &&
        graph_kv_len_hint <= 2048;
    const char* multi_sequence_eager_env =
        std::getenv("XLLM_DSV4_MULTI_SEQUENCE_FORCE_EAGER");
    const bool multi_sequence_eager =
        multi_sequence_eager_env != nullptr &&
        std::string(multi_sequence_eager_env) == "1" &&
        args_.model_type() == "deepseek_v4" && n_tokens > 1 &&
        params.is_spec_verify && params.meta.num_sequences > 1;
    if (force_eager_compact_verify || default_eager_compact_verify ||
        multi_sequence_eager || short_fullkv_eager) {
      LOG(INFO) << "[DSV4_GRAPH_ROUTE] target compact verify forced eager; "
                   "tokens="
                << n_tokens << " reason="
                << (short_fullkv_eager
                        ? "short_fullkv"
                        : (multi_sequence_eager
                               ? "multi_sequence"
                               : (force_eager_compact_verify ? "forced"
                                                             : "default")));
      ModelInputParams eager_params = params;
      eager_params.enable_graph = false;
      eager_params.attn_metadata.reset();
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, eager_params);
    }

    // Check if conditions are suitable for graph execution (replay or capture)
    const auto max_seq_len = args_.max_position_embeddings();
    const bool seq_len_supported = params.meta.kv_max_seq_len <= max_seq_len;

    // Early return if conditions are not suitable for graph operations
    if (!seq_len_supported) {
      LOG(WARNING) << "Not suitable for MUSA graph operations, falling back to "
                      "eager mode.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    // On MUSA the in-graph IndexSelect (embedding lookup) is the only known
    // capture blocker. Compute the embedding here, outside the captured
    // stream region, and pass it through `params.embedding.input_embedding`
    // so both capture and replay paths read from the persistent embedding
    // buffer. No-op on other platforms / non-decode forwards. See
    // maybe_precompute_embedding_for_graph() for the full rationale.
    const ModelInputParams graph_params =
        maybe_precompute_embedding_for_graph(tokens, params);
    if (reuse_existing_graph) {
      const uint32_t width =
          static_cast<uint32_t>(options_.num_decoding_tokens());
      CHECK_EQ(n_tokens % width, 0);
      const uint32_t maximum_sequences = std::min<uint32_t>(
          static_cast<uint32_t>(options_.max_seqs_per_batch()),
          max_graph_tokens > 0
              ? static_cast<uint32_t>(max_graph_tokens) / width
              : static_cast<uint32_t>(options_.max_seqs_per_batch()));
      const uint32_t sequences = layer::dsv4_select_existing_graph_sequences(
          n_tokens / width,
          maximum_sequences,
          [&](uint32_t candidate_sequences) {
            const uint32_t candidate_tokens = candidate_sequences * width;
            const uint32_t key = dsv4_graph_shape_key(candidate_tokens,
                                                      n_tokens,
                                                      args_,
                                                      options_,
                                                      graph_params,
                                                      /*emit_trace=*/false);
            const auto candidate = graphs_.find(key);
            return candidate != graphs_.end() &&
                   candidate->second->captured_without_padding();
          });
      bucket_num_tokens = sequences * width;
    }
    const uint32_t graph_shape_key = dsv4_graph_shape_key(
        bucket_num_tokens, n_tokens, args_, options_, graph_params);

    // Check if captured graph exists for this bucket num_tokens
    auto it = graphs_.find(graph_shape_key);
    if (it != graphs_.end()) {
      CHECK(!reuse_existing_graph || it->second->captured_without_padding());
      // Replay the existing graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "MusaGraphExecutorImpl::run() in decode replay mode";
      auto result =
          it->second->replay(tokens, positions, kv_caches, graph_params);
      auto output =
          attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
      return output;
    }

    // Graph doesn't exist for this bucket num_tokens, try to create it lazily
    auto graph =
        std::make_unique<MusaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphExecutorImpl::run() in decode capture mode";

    auto pool_owner = create_vmm_mempool(kPhysicalPoolIdDecode);
    const c10::musa::MempoolId_t mem_pool =
        pool_owner != nullptr ? pool_owner->id() : graph_pool_;

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          graph_params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          std::move(pool_owner));

    if (capture_success) {
      LOG(INFO) << "Lazy capturing MUSA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      log_graph_memory_after_capture();

      // Save the graph for future reuse
      graphs_[graph_shape_key] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests. Recompute the embedding so the persistent buffer
      // reflects the current token batch (the capture-time embedding above
      // would otherwise be reused unchanged, which is only correct in the
      // unlikely case the post-capture request happens to match exactly).
      const ModelInputParams replay_params =
          maybe_precompute_embedding_for_graph(tokens, params);
      auto result = graphs_[graph_shape_key]->replay(
          tokens, positions, kv_caches, replay_params);
      auto output =
          attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
      return output;
    }

    // Keep graph-mode behavior explicit instead of silently switching
    // execution semantics after a capture failure.
    LOG(FATAL) << "Failed to capture MUSA graph for bucket num_tokens: "
               << bucket_num_tokens << " (actual num_tokens: " << n_tokens
               << ")";
  }

  // Defensive fallback for unsupported forward types (should be unreachable for
  // normal prefill/decode paths).
  LOG(ERROR) << "Failed to capture MUSA graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

uint32_t MusaGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  return static_cast<uint32_t>(::xllm::runtime::get_decode_graph_token_bucket(
      num_tokens, options_.enable_graph_mode_decode_no_padding()));
}

bool MusaGraphExecutorImpl::use_prefill_piecewise_graph(
    uint32_t num_tokens,
    const ModelInputParams& params) const {
  const auto& execution_config = ::xllm::ExecutionConfig::get_instance();
  if (!execution_config.enable_graph() ||
      !execution_config.enable_prefill_piecewise_graph()) {
    return false;
  }
  const auto& forward_type = params.meta.batch_forward_type;
  if (!forward_type.is_prefill() && !forward_type.is_chunked_prefill()) {
    return false;
  }
  // Spec verify has its own graph.
  if (params.is_spec_verify || params.has_llmrec_params()) {
    return false;
  }
  // Multi-sequence GDN chunked prefill runs its recurrence inside the forward
  // rather than in a Mate runner.
  const int64_t num_sequences = params.meta.actual_num_sequences > 0
                                    ? params.meta.actual_num_sequences
                                    : params.meta.num_sequences;
  if (forward_type.is_chunked_prefill() && num_sequences > 1 &&
      !params.embedding.linear_state_ids.empty()) {
    return false;
  }
  return static_cast<int64_t>(num_tokens) <= max_tokens_for_graph_mode_ &&
         static_cast<int64_t>(get_prefill_graph_num_tokens(
             num_tokens, params)) <= options_.max_tokens_per_batch();
}

uint32_t MusaGraphExecutorImpl::get_prefill_graph_num_tokens(
    uint32_t num_tokens,
    const ModelInputParams& params) const {
  const int64_t num_sequences = params.meta.actual_num_sequences > 0
                                    ? params.meta.actual_num_sequences
                                    : params.meta.num_sequences;
  if (num_sequences > 1 && !params.embedding.linear_state_ids.empty()) {
    return num_tokens;
  }
  return static_cast<uint32_t>(::xllm::runtime::get_decode_graph_token_bucket(
      num_tokens, /*enable_no_padding=*/false));
}

std::vector<int32_t> MusaGraphExecutorImpl::get_prefill_graph_key(
    uint32_t num_tokens,
    const ModelInputParams& params) const {
  const std::vector<int32_t>& q_seq_lens = params.attention.host.q_seq_lens;
  std::vector<int32_t> key;
  key.reserve(q_seq_lens.size() + 3);
  key.emplace_back(
      static_cast<int32_t>(get_prefill_graph_num_tokens(num_tokens, params)));
  key.emplace_back(params.meta.batch_forward_type.is_chunked_prefill() ? 1 : 0);
  // A single sequence captures attention and GDN inside the graph, while a
  // multi-sequence batch runs them between the captured segments.
  const int64_t num_sequences = params.meta.actual_num_sequences > 0
                                    ? params.meta.actual_num_sequences
                                    : params.meta.num_sequences;
  key.emplace_back(num_sequences == 1 ? 1 : 0);
  // Multi-sequence Qwen3.5 GDN lays out its padded per-sequence inputs from
  // the host query lengths, so each length layout is captured separately.
  if (num_sequences > 1 && !params.embedding.linear_state_ids.empty()) {
    key.insert(key.end(), q_seq_lens.begin(), q_seq_lens.end());
  }
  return key;
}

// NOTE: REGISTER_EXECUTOR for MusaGraphExecutorImpl lives in
// musa_graph_executor_impl.h. Keeping it in this .cpp meant the static
// initializer's TU was referenced only via runtime factory lookup, and the
// linker dropped libmusa_graph_executor.a's only .o as unused. Putting the
// macro in the header matches base/vlm/acl/mlu/dcu graph executors.

}  // namespace xllm::runtime::musa
