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

#include "core/runtime/musa/musa_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/kernels/musa/global_capture_instance.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/linear.h"
#include "core/layers/musa/flashinfer_planinfo.h"
#include "core/platform/device.h"
#include "core/platform/musa/device_capture_lock.h"
#include "core/util/env_var.h"
#include "core/util/prefill_breakdown.h"
#include "core/util/rec_model_utils.h"
#include "core/util/utils.h"

namespace xllm::runtime::musa {

namespace {

bool s_enable_graph_timing() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_GRAPH_TIMING");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
}

bool s_enable_piecewise_profile() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_PIECEWISE_PROFILE");
    return env != nullptr && std::string(env) == "1";
  }();
  return enabled;
}

bool s_use_musa_fa3_decode(int64_t gqa_ratio, int32_t head_dim) {
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
  const bool default_to_fa3 =
      (head_dim == 128 && (gqa_ratio == 2 || gqa_ratio == 4)) ||
      (head_dim == 256 && (gqa_ratio == 6 || gqa_ratio == 8));
  return default_to_fa3 && setting != 0;
}

constexpr int64_t kDefaultFa3GraphNumSplits = 4;
constexpr int64_t kQwen35Gqa8Fa3GraphSplitBudget = 26;
constexpr int64_t kMaxFa3GraphNumSplits = 64;

int64_t s_fa3_graph_num_splits(uint32_t graph_num_tokens,
                               int64_t gqa_ratio,
                               int32_t head_dim) {
  CHECK_GT(graph_num_tokens, 0);
  int64_t default_num_splits = kDefaultFa3GraphNumSplits;
  if (gqa_ratio == 8 && head_dim == 256) {
    default_num_splits =
        (kQwen35Gqa8Fa3GraphSplitBudget + graph_num_tokens - 1) /
        graph_num_tokens;
  }

  const std::string global_key = "XLLM_FA3_GRAPH_NUM_SPLITS";
  const std::string bucket_key =
      global_key + "_B" + std::to_string(graph_num_tokens);
  const std::string& selected_key =
      std::getenv(global_key.c_str()) != nullptr ? global_key : bucket_key;
  const int64_t num_splits =
      util::get_int_env(selected_key, default_num_splits);
  CHECK_GT(num_splits, 0)
      << selected_key << " must be a positive integer";
  CHECK_LE(num_splits, kMaxFa3GraphNumSplits)
      << selected_key << " exceeds deployed combine capacity";
  return num_splits;
}

// Phase D: wall+device time for packed/eager pure-prefill forwards.
bool s_enable_prefill_fwd_timing() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_PREFILL_FWD_TIMING");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
}

int64_t s_qwen35_c1_piecewise_max_padding_tokens() {
  // Negative disables the adaptive eager fallback for rollback/A-B.
  static const int64_t value =
      util::get_int_env("XLLM_QWEN35_C1_PIECEWISE_MAX_PADDING_TOKENS", 32);
  return value;
}

bool s_enable_prefill_empty_cache() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_PREFILL_EMPTY_CACHE");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
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

// Run pure-prefill piecewise graphs even under enable_packed_prefill. Packed
// multi-sequence batches use the fixed token ladder; live CU metadata keeps the
// bucket tail out of recurrent and attention state updates.
bool s_enable_packed_prefill_piecewise() {
  static const bool val = [] {
    const char* env = std::getenv("XLLM_PACKED_PREFILL_PIECEWISE");
    return env != nullptr && std::string(env) == "1";
  }();
  return val;
}

// Time a prefill graph replay when Phase D/E timing envs are on. Breakdown
// scopes are eager-only (events during MUSA graph replay are not meaningful).
template <typename ReplayFn>
auto timed_prefill_graph_replay(int device_index,
                                uint32_t n_tokens,
                                int32_t batch_bs,
                                bool packed_prefill,
                                const char* mode,
                                ReplayFn&& replay_fn) {
  const bool time_fwd = s_enable_prefill_fwd_timing();
  if (!time_fwd) {
    return replay_fn();
  }
  c10::musa::getCurrentMUSAStream(device_index).synchronize();
  const auto t0 = std::chrono::steady_clock::now();
  auto result = replay_fn();
  c10::musa::getCurrentMUSAStream(device_index).synchronize();
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  LOG(INFO) << "[PREFILL_FWD] n_tokens=" << n_tokens << " batch_bs=" << batch_bs
            << " packed_prefill=" << packed_prefill << " mode=" << mode
            << " fwd_ms=" << ms;
  return result;
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

bool is_musa_contiguous_int_tensor(const torch::Tensor& tensor) {
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
      CHECK(conv_cache.defined() && conv_cache.numel() > 0)
          << "SSM snapshot requires a convolution cache";
      CHECK_GT(conv_cache.size(0), 0)
          << "SSM snapshot requires a non-empty convolution cache";
      CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
          << "SSM cache checkpoint layout mismatch, ssm_rows="
          << ssm_cache.size(0) << ", conv_rows=" << conv_cache.size(0);
      const int64_t checkpoint_stride =
          ssm_cache.size(0) / conv_cache.size(0);
      const torch::Tensor checkpoint_offsets = torch::arange(
          checkpoint_stride,
          indices.options().dtype(torch::kLong));
      const torch::Tensor expanded_indices =
          (indices.unsqueeze(/*dim=*/1) * checkpoint_stride +
           checkpoint_offsets)
              .reshape({-1});
      snapshots.emplace_back(IndexedTensorSnapshot{
          ssm_cache,
          torch::index_select(ssm_cache, /*dim=*/0, expanded_indices),
          expanded_indices});
    }
  }
  return snapshots;
}

void restore_linear_attention_state(
    std::vector<IndexedTensorSnapshot>& snapshots) {
  for (IndexedTensorSnapshot& snapshot : snapshots) {
    snapshot.target.index_copy_(/*dim=*/0, snapshot.indices, snapshot.rows);
  }
}

int64_t get_decode_graph_bucket_num_tokens(int64_t num_tokens) {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  }
  if (num_tokens <= 2) {
    return 2;
  }
  if (num_tokens <= 4) {
    return 4;
  }
  if (num_tokens <= 8) {
    return 8;
  }
  return ((num_tokens + 15) / 16) * 16;
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
  const bool is_qwen35_mtp = args_.model_type() == "qwen3_5_mtp" ||
                             args_.model_type() == "qwen3_5_moe_mtp";
  if (is_qwen35_mtp) {
    // prepare_draft_extend_inputs can emit the previous and current token for
    // every live request after a fully accepted draft.  Give the independent
    // draft graph enough persistent attention metadata for both rows.
    max_seqs_per_batch *= 2;
  }
  max_seqs_per_batch = get_decode_graph_bucket_num_tokens(max_seqs_per_batch);
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  persistent_positions_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
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
  persistent_gdn_cu_seq_lens_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));
  persistent_gdn_kkt_cu_seq_lens_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

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

  // FlashInfer decode mode parameters
  // paged_kv_indptr: shape [max_seqs_per_batch + 1]
  persistent_paged_kv_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

  // paged_kv_indices: maximum size based on max blocks. Expanded MTP
  // validation has one row per query token, so its capacity is sized from
  // max_tokens_per_batch rather than max_seqs_per_batch.
  const int64_t max_paged_kv_indices_size =
      max_seqs_per_batch * max_block_table_len;
  persistent_paged_kv_indices_ = torch::zeros(
      {max_paged_kv_indices_size}, torch::dtype(torch::kInt).device(device));

  // paged_kv_last_page_len: shape [max_seqs_per_batch]
  persistent_paged_kv_last_page_len_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // For decode mode, each sequence has 1 token, so qo_indptr = [0, 1, 2, ...].
  // Expanded MTP verification has one attention row per validation token,
  // which can exceed max_seqs_per_batch (for example C=1, K=2 => 3 rows).
  // Keep this graph-stable buffer sized by token capacity; ordinary decode
  // continues to take only the live batch-sized view.
  persistent_decode_qo_indptr_ = torch::arange(
      0, max_tokens_per_batch + 1, torch::dtype(torch::kInt).device(device));
  persistent_kv_seq_lens_delta_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));
  // will be updated by q_cu_seq_lens, q_cu_seq_lens is the cumulative sum of
  // q_seq_lens
  persistent_chunked_prefill_qo_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));
  // aux_hidden_states will be lazily initialized when needed

  expanded_kv_seq_lens_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_expanded_block_tables_ =
      torch::full({max_tokens_per_batch, max_block_table_len},
                  -1,
                  torch::dtype(torch::kInt).device(device));
  persistent_expanded_paged_kv_indptr_ = torch::zeros(
      {max_tokens_per_batch + 1}, torch::dtype(torch::kInt).device(device));
  persistent_expanded_paged_kv_indices_ = torch::zeros(
      {max_tokens_per_batch * max_block_table_len},
      torch::dtype(torch::kInt).device(device));
  persistent_expanded_paged_kv_last_page_len_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
}

bool MusaGraphPersistentParam::can_use_llm_decode_fast_path(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params) const {
  if (!params.meta.batch_forward_type.is_decode() ||
      is_rec_multi_round_mode() || params.has_llmrec_params()) {
    return false;
  }
  const bool device_token_metadata_ok =
      is_musa_contiguous_int_tensor(tokens) &&
      is_musa_contiguous_int_tensor(positions) &&
      is_musa_contiguous_int_tensor(params.attention.device.new_cache_slots);
  if (!device_token_metadata_ok) {
    return false;
  }
  if (has_llm_decode_host_metadata(params.attention.host)) {
    return true;
  }
  return is_musa_contiguous_int_tensor(params.attention.device.kv_seq_lens) &&
         is_musa_contiguous_int_tensor(
             params.attention.device.paged_kv_indptr) &&
         is_musa_contiguous_int_tensor(
             params.attention.device.paged_kv_indices) &&
         is_musa_contiguous_int_tensor(
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
  if (has_llm_decode_host_metadata(params.attention.host)) {
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
  total += bytes(persistent_gdn_cu_seq_lens_);
  total += bytes(persistent_gdn_kkt_cu_seq_lens_);
  total += bytes(persistent_embedding_);
  total += bytes(persistent_mtp_token_embedding_);
  total += bytes(aux_hidden_states_);
  total += bytes(persistent_paged_kv_indptr_);
  total += bytes(persistent_paged_kv_indices_);
  total += bytes(persistent_paged_kv_last_page_len_);
  total += bytes(persistent_decode_qo_indptr_);
  total += bytes(persistent_kv_seq_lens_delta_);
  total += bytes(persistent_chunked_prefill_qo_indptr_);
  total += bytes(expanded_kv_seq_lens_);
  total += bytes(persistent_expanded_block_tables_);
  total += bytes(persistent_expanded_paged_kv_indptr_);
  total += bytes(persistent_expanded_paged_kv_indices_);
  total += bytes(persistent_expanded_paged_kv_last_page_len_);
  return total;
}

std::vector<int32_t>
MusaGraphPersistentParam::update_expanded_spec_decode_attention(
    const ModelInputParams& input_params,
    uint32_t actual_num_tokens,
    uint32_t padded_num_tokens) {
  CHECK(input_params.is_spec_verify)
      << "expanded spec decode attention is only for spec verify";
  CHECK(input_params.meta.batch_forward_type.is_chunked_prefill())
      << "expanded spec decode attention expects chunked prefill";
  CHECK(input_params.graph.use_expanded_decode_for_spec_verify_attention)
      << "MTP worker must prepare expanded spec-verify graph input";
  CHECK(input_params.graph.expanded_kv_seq_lens.defined())
      << "expanded spec-verify kv seq lens must be defined";
  CHECK(input_params.graph.expanded_block_tables.defined())
      << "expanded spec-verify block tables must be defined";
  CHECK_EQ(input_params.graph.expanded_kv_seq_lens_vec.size(),
           static_cast<size_t>(actual_num_tokens))
      << "expanded kv seq lens size must match validate tokens";
  CHECK_EQ(input_params.graph.expanded_block_tables.size(0),
           static_cast<int64_t>(actual_num_tokens))
      << "expanded block table rows must match validate tokens";

  std::vector<int32_t> expanded_kv_seq_lens_vec =
      input_params.graph.expanded_kv_seq_lens_vec;
  expanded_kv_seq_lens_vec.reserve(padded_num_tokens);
  if (padded_num_tokens > actual_num_tokens) {
    const int64_t pad_count = padded_num_tokens - actual_num_tokens;
    for (int64_t i = 0; i < pad_count; ++i) {
      expanded_kv_seq_lens_vec.emplace_back(1);
    }
  }

  torch::Tensor expanded_kv_tensor =
      torch::tensor(expanded_kv_seq_lens_vec, torch::kInt).to(device_);
  expanded_kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens)
      .copy_(expanded_kv_tensor, /*non_blocking=*/true);

  const int64_t block_table_len =
      input_params.graph.expanded_block_tables.size(1);
  persistent_expanded_block_tables_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens)
      .zero_();
  persistent_expanded_block_tables_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .slice(/*dim=*/1, /*start=*/0, /*end=*/block_table_len)
      .copy_(input_params.graph.expanded_block_tables, /*non_blocking=*/true);

  CHECK(input_params.graph.expanded_paged_kv_indptr.defined())
      << "expanded spec-verify paged_kv_indptr must be defined";
  CHECK(input_params.graph.expanded_paged_kv_indices.defined())
      << "expanded spec-verify paged_kv_indices must be defined";
  CHECK(input_params.graph.expanded_paged_kv_last_page_len.defined())
      << "expanded spec-verify paged_kv_last_page_len must be defined";

  const int64_t actual_indptr_size =
      input_params.graph.expanded_paged_kv_indptr.size(0);
  CHECK_LE(actual_indptr_size,
           persistent_expanded_paged_kv_indptr_.numel())
      << "expanded paged-KV indptr exceeds persistent capacity";
  persistent_expanded_paged_kv_indptr_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_indptr_size)
      .copy_(input_params.graph.expanded_paged_kv_indptr,
             /*non_blocking=*/true);
  const int64_t padded_indptr_size =
      static_cast<int64_t>(padded_num_tokens) + 1;
  if (padded_indptr_size > actual_indptr_size) {
    const int64_t tail_size = padded_indptr_size - actual_indptr_size;
    torch::Tensor tail = persistent_expanded_paged_kv_indptr_.slice(
        /*dim=*/0, /*start=*/actual_indptr_size, /*end=*/padded_indptr_size);
    torch::Tensor last_value = persistent_expanded_paged_kv_indptr_.slice(
        /*dim=*/0,
        /*start=*/actual_indptr_size - 1,
        /*end=*/actual_indptr_size);
    tail.copy_(last_value.expand({tail_size}), /*non_blocking=*/true);
  }

  const int64_t actual_indices_size =
      input_params.graph.expanded_paged_kv_indices.size(0);
  CHECK_LE(actual_indices_size,
           persistent_expanded_paged_kv_indices_.numel())
      << "expanded paged-KV indices exceeds persistent capacity";
  persistent_expanded_paged_kv_indices_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_indices_size)
      .copy_(input_params.graph.expanded_paged_kv_indices,
             /*non_blocking=*/true);

  CHECK_LE(actual_num_tokens,
           static_cast<uint32_t>(
               persistent_expanded_paged_kv_last_page_len_.numel()))
      << "expanded paged-KV last-page lengths exceed persistent capacity";
  persistent_expanded_paged_kv_last_page_len_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(input_params.graph.expanded_paged_kv_last_page_len,
             /*non_blocking=*/true);
  if (padded_num_tokens > actual_num_tokens) {
    persistent_expanded_paged_kv_last_page_len_
        .slice(/*dim=*/0,
               /*start=*/actual_num_tokens,
               /*end=*/static_cast<int64_t>(padded_num_tokens))
        .fill_(1);
  }

  return expanded_kv_seq_lens_vec;
}

std::optional<ModelInputParams> MusaGraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params,
    bool update_prefill_plan) {
  std::optional<ModelInputParams> params_for_capture;
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    params_for_capture = std::make_optional<ModelInputParams>(params);
  }
  // Build attn_metadata with original model_input_params. So we can set actual
  // batch size in plan_info.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params, args_.enable_mla()));
  CHECK(attn_metadata) << "attn_metadata should not be null";
  attn_metadata->enable_cuda_graph = true;

  const uint32_t actual_num_tokens = tokens.size(0);
  // Decode has exactly one query token per attention row.  Schedule-overlap
  // forward buffers can retain a bucket-capacity num_sequences value (for
  // example 8 for a live C=5 batch), while tokens and all logical outputs
  // contain only the active rows.  Use the token count for ordinary decode so
  // persistent graph tensors never copy padded buffer rows as live requests.
  int64_t actual_batch_size = params.meta.num_sequences;
  if (params.meta.actual_num_sequences > 0) {
    actual_batch_size = params.meta.actual_num_sequences;
  } else if (params.is_spec_verify && params.meta.q_max_seq_len > 0) {
    actual_batch_size = (static_cast<int64_t>(actual_num_tokens) +
                         params.meta.q_max_seq_len - 1) /
                        params.meta.q_max_seq_len;
  } else if (params.meta.batch_forward_type.is_decode() &&
             !is_rec_multi_round_mode()) {
    actual_batch_size = static_cast<int64_t>(actual_num_tokens);
  }
  const int64_t accepted_batch_size =
      params.num_accepted_tokens.defined()
          ? std::min<int64_t>(params.num_accepted_tokens.numel(),
                              persistent_num_accepted_tokens_.numel())
          : 0;
  const int64_t linear_state_batch_size =
      params.embedding.linear_state_indices.defined()
          ? std::min<int64_t>(params.embedding.linear_state_indices.numel(),
                              persistent_linear_state_indices_.numel())
          : std::min<int64_t>(params.embedding.linear_state_ids.size(),
                              persistent_linear_state_indices_.numel());

  // Single-sequence piecewise graphs retain the established logical-padding
  // path. Packed multi-sequence prefill keeps live ragged CU endpoints: the
  // bucket tail is capacity-only and is ignored by the GDN/attention runners.
  // Keep this predicate in lockstep with run()'s packed graph routing so a
  // disabled packed path never receives the live-CU metadata contract.
  const bool packed_piecewise_enabled =
      ::xllm::ExecutionConfig::get_instance().enable_packed_prefill() &&
      s_enable_packed_prefill_piecewise();
  const bool packed_piecewise_prefill_pad =
      return_capture_params && packed_piecewise_enabled &&
      attn_metadata->is_prefill && !attn_metadata->is_chunked_prefill &&
      !params.is_spec_verify && actual_batch_size > 1 &&
      padded_num_tokens > actual_num_tokens;
  const bool piecewise_prefill_pad =
      return_capture_params && attn_metadata->is_prefill &&
      !packed_piecewise_prefill_pad && padded_num_tokens > actual_num_tokens;
  if (piecewise_prefill_pad) {
    const uint32_t padding = padded_num_tokens - actual_num_tokens;
    if (attn_metadata->q_seq_lens_vec.size() == 1) {
      attn_metadata->max_query_len = padded_num_tokens;
      attn_metadata->q_seq_lens_vec[0] =
          static_cast<int32_t>(padded_num_tokens);
      if (attn_metadata->kv_seq_lens_vec.size() == 1) {
        attn_metadata->max_seq_len = padded_num_tokens;
        attn_metadata->kv_seq_lens_vec[0] =
            static_cast<int32_t>(padded_num_tokens);
      }
    } else {
      const int32_t last =
          static_cast<int32_t>(attn_metadata->q_seq_lens_vec.size()) - 1;
      attn_metadata->q_seq_lens_vec[last] += static_cast<int32_t>(padding);
      if (last < static_cast<int32_t>(attn_metadata->kv_seq_lens_vec.size())) {
        attn_metadata->kv_seq_lens_vec[last] += static_cast<int32_t>(padding);
      }
      attn_metadata->max_query_len =
          *std::max_element(attn_metadata->q_seq_lens_vec.begin(),
                            attn_metadata->q_seq_lens_vec.end());
    }
  }
  // Match ACL graph persistent param: the expanded-decode graph input is
  // authoritative once MTP worker prepared it; do not additionally require
  // is_spec_verify here (capture/replay params may arrive without it set).
  const bool use_expanded_spec_decode_attention =
      params.graph.use_expanded_decode_for_spec_verify_attention &&
      (params.meta.batch_forward_type.is_chunked_prefill() ||
       attn_metadata->is_chunked_prefill);
  std::vector<int32_t> expanded_kv_seq_lens_vec;
  if (use_expanded_spec_decode_attention) {
    expanded_kv_seq_lens_vec = update_expanded_spec_decode_attention(
        params, actual_num_tokens, padded_num_tokens);
  }
  const bool use_llm_decode_fast_path =
      !use_expanded_spec_decode_attention &&
      can_use_llm_decode_fast_path(tokens, positions, params);

  // Cheap when the input builder already pre-staged (just a shared_ptr ref);
  // a single per-step D2H per index tensor in the fallback case (3 D2H total,
  // batch-sized, runs once before capture begin).
  const bool decode_path =
      !attn_metadata->is_prefill && !attn_metadata->is_chunked_prefill;
  const bool expanded_decode_attention_path =
      use_expanded_spec_decode_attention;
  if (decode_path || expanded_decode_attention_path) {
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
      LOG(INFO) << "GRAPH_TIMING ensure_host_mirror: indptr_host_defined="
                << attn_metadata->musa.paged_kv_indptr_host.defined()
                << " indices_host_defined="
                << attn_metadata->musa.paged_kv_indices_host.defined()
                << " last_page_len_host_defined="
                << attn_metadata->musa.paged_kv_last_page_len_host.defined();
    }
    ensure_host_mirror(attn_metadata->musa.paged_kv_indptr_host,
                       attn_metadata->paged_kv_indptr);
    ensure_host_mirror(attn_metadata->musa.paged_kv_indices_host,
                       attn_metadata->paged_kv_indices);
    ensure_host_mirror(attn_metadata->musa.paged_kv_last_page_len_host,
                       attn_metadata->paged_kv_last_page_len);
  }
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
      if (!params.num_accepted_tokens_host.empty()) {
        CHECK_GE(params.num_accepted_tokens_host.size(),
                 static_cast<size_t>(accepted_batch_size))
            << "num_accepted_tokens host mirror is smaller than its device "
               "tensor";
        params_for_capture->num_accepted_tokens_host.assign(
            params.num_accepted_tokens_host.begin(),
            params.num_accepted_tokens_host.begin() + accepted_batch_size);
      } else {
        torch::Tensor nat_host = params.num_accepted_tokens.to(torch::kCPU)
                                     .to(torch::kLong)
                                     .contiguous();
        const int64_t* data = nat_host.data_ptr<int64_t>();
        params_for_capture->num_accepted_tokens_host.assign(
            data, data + accepted_batch_size);
      }
    }
    params_for_capture->attn_metadata = attn_metadata;
    params_for_capture->is_spec_verify = params.is_spec_verify;
    if (use_expanded_spec_decode_attention) {
      params_for_capture->graph.use_expanded_decode_for_spec_verify_attention =
          true;
      params_for_capture->graph.expanded_kv_seq_lens =
          expanded_kv_seq_lens(padded_num_tokens);
      params_for_capture->graph.expanded_block_tables =
          persistent_expanded_block_tables(padded_num_tokens);
      params_for_capture->graph.expanded_kv_seq_lens_vec =
          expanded_kv_seq_lens_vec;
      params_for_capture->graph.expanded_paged_kv_indptr =
          persistent_expanded_paged_kv_indptr(padded_num_tokens);
      params_for_capture->graph.expanded_paged_kv_indices =
          persistent_expanded_paged_kv_indices_;
      params_for_capture->graph.expanded_paged_kv_last_page_len =
          persistent_expanded_paged_kv_last_page_len(padded_num_tokens);
    }
    return params_for_capture;
  };

  // Copy data from input parameters to persistent graph tensors
  if (use_llm_decode_fast_path) {
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
      if (piecewise_prefill_pad) {
        // Fill padding tokens with last actual token's value so their
        // embedding/QKV matches — prevents KV cache corruption when
        // FlashInfer processes all padded tokens as one sequence.
        persistent_tokens_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .copy_(persistent_tokens_
                       .slice(/*dim=*/0,
                              /*start=*/actual_num_tokens - 1,
                              /*end=*/actual_num_tokens)
                       .expand({static_cast<int64_t>(padded_num_tokens -
                                                     actual_num_tokens)}));
      } else {
        persistent_tokens_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .fill_(0);
      }
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ positions: src shape=" << positions.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    persistent_positions_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(positions, /*non_blocking=*/true);
    if (piecewise_prefill_pad) {
      // Fill padding positions with last actual token's position so RoPE
      // produces the same output for padding tokens as the last actual
      // token — consistent with the token padding fill above.
      persistent_positions_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .copy_(persistent_positions_
                     .slice(/*dim=*/0,
                            /*start=*/actual_num_tokens - 1,
                            /*end=*/actual_num_tokens)
                     .expand({static_cast<int64_t>(padded_num_tokens -
                                                   actual_num_tokens)}));
    } else if (padded_num_tokens > actual_num_tokens) {
      persistent_positions_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(0);
    }
  }

  if (!is_rec_multi_round_mode() && !use_llm_decode_fast_path) {
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
      if (piecewise_prefill_pad) {
        // Fill padding cache slots with last actual token's slot so the
        // KV cache write for padding tokens overwrites the last actual
        // token's slot with the same value (no corruption).
        persistent_new_cache_slots_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .copy_(persistent_new_cache_slots_
                       .slice(/*dim=*/0,
                              /*start=*/actual_num_tokens - 1,
                              /*end=*/actual_num_tokens)
                       .expand({static_cast<int64_t>(padded_num_tokens -
                                                     actual_num_tokens)}));
      } else {
        persistent_new_cache_slots_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .fill_(params.meta.batch_forward_type.is_decode() ||
                           packed_piecewise_prefill_pad
                       ? -1
                       : 0);
      }
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

    // GDN keeps a separate live CU buffer. The attention CU may be changed to
    // the bucket endpoint for legacy single-sequence padding, but packed GDN
    // runners must always see the real ragged endpoints.
    CHECK_GE(params.attention.device.q_seq_lens.numel(), actual_batch_size + 1)
        << "q_seq_lens must contain a B+1 cumulative endpoint";
    persistent_gdn_cu_seq_lens_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.q_seq_lens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1),
               /*non_blocking=*/true);
    persistent_gdn_kkt_cu_seq_lens_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.q_seq_lens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1),
               /*non_blocking=*/true);
    if (packed_piecewise_prefill_pad) {
      persistent_gdn_kkt_cu_seq_lens_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size,
                 /*end=*/actual_batch_size + 1)
          .fill_(static_cast<int32_t>(padded_num_tokens));
    }

    // Keep metadata tensors pointing to persistent buffers used by graph
    // capture/replay so their addresses are stable and shapes match padded
    // tensors in capture path.
    attn_metadata->q_cu_seq_lens = q_seq_lens(/*actual_batch_size=*/
                                              actual_batch_size + 1);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens(/*actual_batch_size=*/
                                                actual_batch_size + 1);
    attn_metadata->musa.gdn_cu_seq_lens =
        gdn_cu_seq_lens(static_cast<uint32_t>(actual_batch_size + 1));
    attn_metadata->musa.gdn_kkt_cu_seq_lens =
        gdn_kkt_cu_seq_lens(static_cast<uint32_t>(actual_batch_size + 1));

    // For piecewise prefill: override the last element of q_cu_seq_lens and
    // kv_cu_seq_lens to padded_num_tokens so FlashInfer processes all padded
    // tokens.  For single-request (bs=1) this yields [0, padded]; for
    // multi-request (bs=N) it yields [0, q0, ..., padded].  The persistent
    // buffers are updated in-place so the eager FlashInfer plan/attention
    // sees the padded values.
    if (piecewise_prefill_pad) {
      q_seq_lens_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size,
                 /*end=*/actual_batch_size + 1)
          .fill_(static_cast<int32_t>(padded_num_tokens));
      kv_seq_lens_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size,
                 /*end=*/actual_batch_size + 1)
          .fill_(static_cast<int32_t>(padded_num_tokens));
    }

    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
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

  // Copy block table data. In rec multi-round, block_tables may already be
  // expanded to batch_size * beam_width rows while num_sequences still tracks
  // the logical request count. Use the tensor's real row count here.
  const int64_t actual_block_table_batch =
      is_rec_multi_round_mode()
          ? params.attention.device.block_tables.size(0)
          : std::min<int64_t>(actual_batch_size,
                              persistent_block_tables_.size(0));
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

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.embedding.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      const int64_t embedding_dim = embedding.size(1);
      torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ embedding: src shape=" << embedding.sizes()
        << ", dst slice shape=[" << embedding_tokens << ", "
        << embedding.size(1) << "]";
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }

  const torch::Tensor& mtp_token_embedding =
      params.embedding.mtp_token_embedding;
  if (mtp_token_embedding.defined()) {
    const int64_t embedding_tokens = mtp_token_embedding.size(0);
    if (!persistent_mtp_token_embedding_.defined()) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      const int64_t embedding_dim = mtp_token_embedding.size(1);
      persistent_mtp_token_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       mtp_token_embedding.options().device(device_));
    }
    persistent_mtp_token_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(mtp_token_embedding, /*non_blocking=*/true);
  }

  const bool is_decode_with_llmrec =
      params.meta.batch_forward_type.is_decode() && params.has_llmrec_params();
  const bool use_two_stage_decode =
      !::xllm::RecConfig::get_instance().enable_xattention_one_stage() &&
      is_decode_with_llmrec;
  const int32_t head_dim = args_.head_dim();
  const int64_t tp_size =
      options_.world_size() / std::max(options_.dp_size(), 1);
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
  // Determine backend
  const std::string backend = xllm::kernel::musa::determine_attention_backend(
      /*pos_encoding_mode=*/0,
      /*use_fp16_qk_reduction=*/false,
      /*use_custom_mask=*/false);

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
  if (use_expanded_spec_decode_attention) {
    const int64_t expanded_batch = static_cast<int64_t>(padded_num_tokens);
    attn_metadata->use_expanded_decode_for_spec_verify_attention = true;
    attn_metadata->expanded_kv_seq_lens =
        expanded_kv_seq_lens(static_cast<uint32_t>(expanded_batch));
    attn_metadata->expanded_block_table =
        persistent_expanded_block_tables(static_cast<uint32_t>(expanded_batch));
    if (!expanded_kv_seq_lens_vec.empty()) {
      attn_metadata->expanded_kv_seq_lens_host =
          torch::tensor(expanded_kv_seq_lens_vec, torch::kInt);
      attn_metadata->max_seq_len = std::max<int32_t>(
          *std::max_element(expanded_kv_seq_lens_vec.begin(),
                            expanded_kv_seq_lens_vec.end()),
          1);
    }
    attn_metadata->paged_kv_indptr = persistent_expanded_paged_kv_indptr(
        static_cast<uint32_t>(expanded_batch));
    attn_metadata->paged_kv_indices = persistent_expanded_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len =
        persistent_expanded_paged_kv_last_page_len(
            static_cast<uint32_t>(expanded_batch));
    attn_metadata->qo_indptr =
        persistent_decode_qo_indptr(static_cast<uint32_t>(expanded_batch));
    attn_metadata->musa.expanded_paged_kv_indptr =
        persistent_expanded_paged_kv_indptr(
            static_cast<uint32_t>(expanded_batch));
    attn_metadata->musa.expanded_paged_kv_indices =
        persistent_expanded_paged_kv_indices_;
    attn_metadata->musa.expanded_paged_kv_last_page_len =
        persistent_expanded_paged_kv_last_page_len(
            static_cast<uint32_t>(expanded_batch));
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
    ensure_host_mirror(attn_metadata->musa.paged_kv_indptr_host,
                       attn_metadata->paged_kv_indptr);
    ensure_host_mirror(attn_metadata->musa.paged_kv_indices_host,
                       attn_metadata->paged_kv_indices);
    ensure_host_mirror(attn_metadata->musa.paged_kv_last_page_len_host,
                       attn_metadata->paged_kv_last_page_len);
  } else if (use_llm_decode_fast_path) {
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
  // Update plan_info if attn_metadata exists and enable_cuda_graph is true
  // This ensures plan_info is updated before MUSA graph capture/replay
  {
    // Determine if causal (prefill mode)
    const bool causal =
        attn_metadata->is_prefill || attn_metadata->is_chunked_prefill;

    // Update plan_info
    // Note: plan_info is only updated at layer 0, so we set layer_id to 0
    attn_metadata->plan_info->layer_id = 0;
    CHECK_EQ(dtype, torch::ScalarType::BFloat16)
        << "only support bf16 kvcache for now";

    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphPersistentParam::update() calling update_plan_info: "
        << "is_prefill=" << attn_metadata->is_prefill
        << ", is_chunked_prefill=" << attn_metadata->is_chunked_prefill
        << ", causal=" << causal << ", backend=" << backend
        << ", enable_cuda_graph=" << attn_metadata->enable_cuda_graph;

    if (attn_metadata->is_prefill) {
      if (update_prefill_plan) {
        layer::flashinfer::update_prefill_plan_info(
            attn_metadata->plan_info,
            backend,
            *attn_metadata,
            dtype,                             // query_dtype
            dtype,                             // key_dtype
            dtype,                             // output_dtype
            head_dim,                          // head_dim_qk
            head_dim,                          // head_dim_vo
            static_cast<int32_t>(n_heads),     // num_qo_heads
            static_cast<int32_t>(n_kv_heads),  // num_kv_heads
            /*enable_cuda_graph=*/true);
      } else {
        VLOG(kGraphExecutorLogVerboseLevel)
            << "Skipping replay-time prefill plan update: piecewise graph "
               "contains no plan-dependent attention runner";
      }
    } else if (use_expanded_spec_decode_attention ||
               !attn_metadata->is_chunked_prefill) {
      // Spec-verify validate uses per-token batch_decode for full-attention
      // layers; regular decode uses the same decode plan path.
      const int32_t max_kv_blocks_per_seq_for_capture =
          block_size > 0
              ? static_cast<int32_t>(
                    (args_.max_position_embeddings() + block_size - 1) /
                    block_size)
              : 0;
      layer::flashinfer::update_decode_plan_info(
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
    } else if (attn_metadata->is_chunked_prefill) {
      // Worst-case KV blocks per sequence for graph capture: plan_info is
      // computed once (cached on PlanInfo) and reused for all replays. Make
      // sure the cached plan covers any future block count by computing it
      // against ceil(max_position_embeddings / block_size) blocks per
      // sequence. See update_chunked_prefill_plan_info comment.
      const int32_t max_kv_blocks_per_seq_for_capture =
          block_size > 0
              ? static_cast<int32_t>(
                    (args_.max_position_embeddings() + block_size - 1) /
                    block_size)
              : 0;
      layer::flashinfer::update_chunked_prefill_plan_info(
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
          /*causal=*/true,
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
  // Keep one fixed split topology per graph token bucket while refreshing the
  // live KV work metadata outside capture. MusaGraph::replay() copies it into
  // the tensor address retained by the captured attention calls.
  const bool is_qwen3 = args_.model_type() == "qwen3";
  const bool is_qwen3_5 = args_.model_type() == "qwen3_5_text" ||
                          args_.model_type() == "qwen3_5_moe_text" ||
                          args_.model_type() == "qwen3_5_mtp" ||
                          args_.model_type() == "qwen3_5_moe_mtp";
  const bool is_qwen3_family = is_qwen3 || is_qwen3_5;
  const int64_t gqa_ratio = n_kv_heads > 0 ? n_heads / n_kv_heads : int64_t{0};
  const bool expanded_fa3_decode = use_expanded_spec_decode_attention;
  const torch::Tensor scheduler_block_table =
      expanded_fa3_decode ? attn_metadata->expanded_block_table
                           : attn_metadata->block_table;
  if (s_use_musa_fa3_decode(gqa_ratio, head_dim) && is_qwen3_family &&
      (expanded_fa3_decode ||
       (!attn_metadata->is_prefill && !attn_metadata->is_chunked_prefill)) &&
      scheduler_block_table.defined()) {
    // Expanded MTP verification is routed through decoder_forward even though
    // its outer metadata is chunked-prefill.  It has one query token per
    // expanded row and must use the expanded page table/KV lengths here;
    // generating metadata from the outer prefill request would bake the first
    // request's split schedule into every later graph replay.
    const int64_t batch_size = scheduler_block_table.size(0);
    if (batch_size > 0 && (gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 6 ||
                           gqa_ratio == 8)) {
      const torch::Tensor cu_seqlens_q =
          attn_metadata->qo_indptr.has_value() &&
                  attn_metadata->qo_indptr->defined()
              ? *attn_metadata->qo_indptr
              : attn_metadata->q_cu_seq_lens;
      const torch::Tensor scheduler_kv_seq_lens =
          expanded_fa3_decode ? attn_metadata->expanded_kv_seq_lens
                              : attn_metadata->kv_seq_lens;
      CHECK(scheduler_kv_seq_lens.defined())
          << "FA3 graph decode requires per-sequence KV lengths";
      CHECK_EQ(scheduler_kv_seq_lens.numel(), batch_size)
          << "FA3 scheduler KV lengths must match the page-table batch";
      CHECK_EQ(cu_seqlens_q.numel(), batch_size + 1)
          << "FA3 scheduler query indptr must have batch+1 entries";
      CHECK_EQ(cu_seqlens_q.scalar_type(), torch::kInt32)
          << "FA3 scheduler query indptr must be int32";
      CHECK_EQ(scheduler_kv_seq_lens.scalar_type(), torch::kInt32)
          << "FA3 scheduler KV lengths must be int32";
      CHECK(cu_seqlens_q.is_contiguous())
          << "FA3 scheduler query indptr must be contiguous";
      CHECK(scheduler_kv_seq_lens.is_contiguous())
          << "FA3 scheduler KV lengths must be contiguous";
      const int32_t max_seqlen_q = expanded_fa3_decode
                                       ? 1
                                       : std::max<int32_t>(
                                             attn_metadata->max_query_len, 1);
      int32_t max_seqlen_k =
          std::max<int32_t>(attn_metadata->max_seq_len, 1);
      if (expanded_fa3_decode) {
        CHECK_EQ(expanded_kv_seq_lens_vec.size(),
                 static_cast<size_t>(batch_size))
            << "expanded FA3 scheduler metadata requires one KV length per "
               "expanded row";
        max_seqlen_k = std::max<int32_t>(
            *std::max_element(expanded_kv_seq_lens_vec.begin(),
                              expanded_kv_seq_lens_vec.end()),
            1);
      }
      attn_metadata->musa.share_fa3_scheduler_metadata = true;
      attn_metadata->musa.fa3_num_splits = s_fa3_graph_num_splits(
          padded_num_tokens, gqa_ratio, head_dim);
      attn_metadata->musa.fa3_scheduler_metadata =
          xllm::kernel::musa::fa3_decode_scheduler_metadata(
              device_,
              static_cast<int32_t>(batch_size),
              static_cast<int32_t>(n_heads),
              static_cast<int32_t>(n_kv_heads),
              head_dim,
              head_dim,
              max_seqlen_q,
              max_seqlen_k,
              sliding_window,
              /*window_size_right=*/0,
              cu_seqlens_q,
              scheduler_kv_seq_lens,
              attn_metadata->musa.fa3_num_splits);
    }
  }

  // Return ModelInputParams with persistent buffer references if requested
  return build_capture_params_if_needed();
}

void MusaGraph::refresh_persistent_paged_kv_host_mirrors(
    const std::shared_ptr<layer::AttentionMetadata>& attn_metadata,
    const ModelInputParams& params) {
  // Only applies to the Mate FFI decode path. Prefill/chunked-prefill and MLA
  // attention do not pass host pointers through the FFI run() boundary, so
  // there is nothing to stabilize there.
  if (!attn_metadata) {
    return;
  }
  if (attn_metadata->is_prefill ||
      (attn_metadata->is_chunked_prefill &&
       !attn_metadata->use_expanded_decode_for_spec_verify_attention)) {
    return;
  }

  AttentionHostInput host_src = params.attention.host;
  if (params.graph.use_expanded_decode_for_spec_verify_attention) {
    // Expanded MTP verification has a different logical row count and block
    // layout from the ordinary request metadata.  Use its pre-staged CPU
    // mirrors when available; otherwise retain the ordinary mirrors and let
    // the helper's device fallback handle legacy/profile callers.
    if (params.graph.expanded_paged_kv_indptr_host.defined()) {
      host_src.paged_kv_indptr = params.graph.expanded_paged_kv_indptr_host;
    }
    if (params.graph.expanded_paged_kv_indices_host.defined()) {
      host_src.paged_kv_indices = params.graph.expanded_paged_kv_indices_host;
    }
    if (params.graph.expanded_paged_kv_last_page_len_host.defined()) {
      host_src.paged_kv_last_page_len =
          params.graph.expanded_paged_kv_last_page_len_host;
    }
  }

  // Helper: lazily allocate / grow a pinned host buffer, copy fresh logical
  // metadata into it, then re-point `host_field` at the graph-stable view.
  // The buffer's underlying storage pointer must be STABLE across the
  // lifetime of this MusaGraph (captured FFI run() bakes it into the graph).
  // The persistent device tensor can be capacity-sized, whereas `cpu_src` is
  // logical-sized; using the latter for the copy avoids a needless D2H.
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
                        int64_t min_alloc_numel,
                        int32_t tail_value,
                        bool tail_from_last_value,
                        bool view_device_shape,
                        bool view_allocated_shape) {
    if (!device_src.defined()) {
      return;
    }
    const bool use_cpu_src = cpu_src.defined() && cpu_src.device().is_cpu();
    // persistent expanded indices use a global capacity tensor.  If a legacy
    // caller omitted the CPU mirror, limit the blocking fallback to this
    // graph's precomputed capacity instead of copying the global buffer.
    const int64_t device_fallback_numel =
        min_alloc_numel > 0
            ? std::min<int64_t>(device_src.numel(), min_alloc_numel)
            : device_src.numel();
    const int64_t logical_numel =
        use_cpu_src ? cpu_src.numel() : device_fallback_numel;
    CHECK_LE(logical_numel, device_src.numel())
        << "host paged-KV mirror is larger than its persistent device "
           "capacity: host_numel="
        << logical_numel << " device_numel=" << device_src.numel();
    const torch::ScalarType src_dtype = device_src.scalar_type();
    // Indptr and last_page_len describe every padded graph row, so retain the
    // persistent device shape for those fields after filling their logical
    // prefix. Indices expose the per-graph allocated shape because the FFI
    // TensorView shape and captured H2D byte count must remain fixed.
    const int64_t view_numel =
        view_device_shape
            ? device_src.numel()
            : (view_allocated_shape
                   ? std::max<int64_t>(logical_numel, min_alloc_numel)
                   : logical_numel);
    const bool needs_alloc =
        !host_buf.defined() || host_buf.scalar_type() != src_dtype ||
        host_buf.numel() < logical_numel || host_buf.numel() < view_numel ||
        host_buf.numel() < min_alloc_numel;
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
      const int64_t alloc_numel = std::max<int64_t>(
          view_numel, std::max<int64_t>(logical_numel, min_alloc_numel));
      host_buf = torch::empty({alloc_numel}, opts);
    }
    auto dst = host_buf.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/logical_numel);
    // device_src may have a non-1D shape (e.g., [bs+1]); view as flat for the
    // copy and let host_field carry the original shape via a view-back.
    //
    // The Mate FFI batch_decode wrapper reads paged_kv_* host tensors on the
    // CPU at submit time, so destination bytes must be valid before replay.
    // CPU->pinned and blocking D2H both satisfy that; async D2H would not.
    if (use_cpu_src) {
      torch::Tensor cpu_flat = cpu_src.contiguous().view({logical_numel});
      if (cpu_flat.scalar_type() != src_dtype) {
        cpu_flat = cpu_flat.to(src_dtype);
      }
      dst.copy_(cpu_flat);
    } else {
      dst.copy_(device_src.contiguous().view({logical_numel}),
                /*non_blocking=*/false);
    }
    // Clear/pad the unused capacity on every refresh.  Normally the FFI sees
    // the logical view above, but keeping a deterministic tail protects any
    // planner/kernel that inspects the capacity-sized backing storage.  The
    // indptr tail must remain cumulative; indices are harmlessly zero-filled;
    // last_page_len uses a valid full-page value for synthetic rows.
    const int64_t tail_numel = host_buf.numel() - logical_numel;
    if (tail_numel > 0) {
      int64_t fill_value = tail_value;
      if (tail_from_last_value && logical_numel > 0) {
        fill_value = dst.select(/*dim=*/0, logical_numel - 1).item<int64_t>();
      }
      host_buf
          .narrow(/*dim=*/0,
                  /*start=*/logical_numel,
                  /*length=*/tail_numel)
          .fill_(fill_value);
    }
    // Re-point host_field at the persistent storage using the per-field graph
    // ABI shape selected above.
    if (view_device_shape) {
      host_field = host_buf
                       .narrow(/*dim=*/0,
                               /*start=*/0,
                               /*length=*/view_numel)
                       .view(device_src.sizes());
    } else if (view_allocated_shape) {
      // FFI TensorView shapes and captured H2D byte counts are fixed during
      // graph capture. Expose the per-graph maximum indices view from the
      // first capture so crossing a KV block boundary changes only contents,
      // never the pointer, shape, or captured copy size.
      host_field = host_buf.narrow(/*dim=*/0,
                                   /*start=*/0,
                                   /*length=*/view_numel);
    } else {
      const auto logical_sizes =
          use_cpu_src ? cpu_src.sizes() : device_src.sizes();
      host_field = dst.view(logical_sizes);
    }
  };

  refresh_one(paged_kv_indptr_host_buf_,
              attn_metadata->musa.paged_kv_indptr_host,
              attn_metadata->paged_kv_indptr,
              host_src.paged_kv_indptr,
              paged_kv_indptr_host_max_numel_,
              /*tail_value=*/0,
              /*tail_from_last_value=*/true,
              /*view_device_shape=*/true,
              /*view_allocated_shape=*/false);
  refresh_one(paged_kv_indices_host_buf_,
              attn_metadata->musa.paged_kv_indices_host,
              attn_metadata->paged_kv_indices,
              host_src.paged_kv_indices,
              paged_kv_indices_host_max_numel_,
              /*tail_value=*/0,
              /*tail_from_last_value=*/false,
              /*view_device_shape=*/false,
              /*view_allocated_shape=*/true);
  refresh_one(paged_kv_last_page_len_host_buf_,
              attn_metadata->musa.paged_kv_last_page_len_host,
              attn_metadata->paged_kv_last_page_len,
              host_src.paged_kv_last_page_len,
              paged_kv_last_page_len_host_max_numel_,
              /*tail_value=*/1,
              /*tail_from_last_value=*/false,
              /*view_device_shape=*/true,
              /*view_allocated_shape=*/false);
}

// MusaGraph implementation
bool MusaGraph::capture(CausalLM* model,
                        const ModelArgs& args,
                        const runtime::Options& options,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params,
                        std::vector<KVCache>& kv_cache,
                        uint32_t bucket_num_tokens,
                        const c10::musa::MempoolId_t& pool,
                        MusaMemPool* pool_ptr) {
  padded_num_tokens_ = bucket_num_tokens;
  const uint32_t actual_num_tokens = tokens.size(0);
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
        ::xllm::musa::DeviceCaptureLock::get_instance().get_write_lock(
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
  auto full_attention_cache =
      MusaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(full_attention_cache.has_value())
      << "MUSA graph capture requires at least one full-attention KV cache";
  const torch::Tensor& k_cache = full_attention_cache->first;
  const torch::Tensor& v_cache = full_attention_cache->second;
  auto graph_params_opt =
      persistent_param_.update(tokens,
                               k_cache,
                               v_cache,
                               positions,
                               params,
                               padded_num_tokens_,
                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params_opt.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";

  captured_fa3_scheduler_metadata_ =
      graph_params_opt.value().attn_metadata->musa.fa3_scheduler_metadata;

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

  refresh_persistent_paged_kv_host_mirrors(
      graph_params_opt.value().attn_metadata, graph_params_opt.value());

  LOG(INFO) << "MUSA graph capture begin, bucket_num_tokens: "
            << bucket_num_tokens << ", actual_num_tokens: " << actual_num_tokens
            << ", is_piecewise: " << is_piecewise_;

  if (is_piecewise_) {
    // Piecewise capture mode (for prefill)
    if (!prefill_matmul_buffer_pool_) {
      prefill_matmul_buffer_pool_ =
          std::make_shared<layer::musa::PiecewiseGraphMatmulBufferPool>();
    }
    layer::musa::PiecewiseGraphMatmulBufferScope buffer_pool_scope(
        prefill_matmul_buffer_pool_.get());

    // Warm up once without capture because MUSA BLAS handles and runtime
    // resources cannot be created during graph capture.
    {
      xllm::kernel::musa::MusaTvmffiPreparationSyncGuard ffi_sync_guard;
      model->forward(persistent_param_.persistent_tokens(padded_num_tokens_),
                     persistent_param_.persistent_positions(padded_num_tokens_),
                     kv_cache,
                     graph_params_opt.value());
    }
    // MUSA TVM-FFI/AOT operators may finish initialization work on their
    // thread-local pool stream even when the model forward runs on the capture
    // stream. Drain that stream before restoring state or beginning capture;
    // synchronizing only capture_stream leaves initialization racing capture.
    xllm::kernel::musa::sync_musa_ffi_stream(persistent_param_.device());
    if (snapshot_linear_state) {
      restore_linear_attention_state(linear_state_snapshots);
      capture_stream.synchronize();
    }
    // All tensor shapes used by the graph-owned linear/GDN buffers must have
    // been observed by the eager warmup.  A frozen pool turns any accidental
    // allocation during capture into an immediate, actionable failure.
    prefill_matmul_buffer_pool_->freeze();
    // Warmup and capture share one BufferScope; warmup advances ring
    // cursors to the end. Restart slots so capture binds the same
    // buffer sequence that replay will use after reset_forward_slots().
    prefill_matmul_buffer_pool_->reset_forward_slots();

    // MemPoolContext has been deprecated in torch >= 2.8
#if TORCH_VERSION_MAJOR <= 2 && TORCH_VERSION_MINOR <= 7
    // Activate VMM mempool only for the actual capture to keep plan_info
    // allocations out of the shared physical memory pool.
    std::optional<c10::musa::MemPoolContext> mempool_ctx;
    if (pool_ptr != nullptr) {
      mempool_ctx.emplace(pool_ptr);
    }
#endif

    // Begin piecewise capture via GlobalCaptureInstance.
    void* const capture_stream_handle =
        reinterpret_cast<void*>(capture_stream.stream());
    std::optional<xllm::kernel::musa::MusaTvmffiStreamOverrideGuard>
        ffi_capture_stream_guard;
    ffi_capture_stream_guard.emplace(persistent_param_.device(),
                                     capture_stream_handle);
    GlobalCaptureInstance::get_instance().begin_capture(pool);

    // Execute forward pass - attention operations will be captured separately
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    // Only capture aux_hidden_states when enable_graph_aux_hidden_states is on
    // (e.g. main worker in EAGLE-3); draft worker has this option false.
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    VLOG(kGraphExecutorLogVerboseLevel)
        << "Piecewise capture forward_result shape: "
        << forward_result.hidden_states.sizes();

    // End capture and get piecewise graphs
    auto piecewise_graphs = GlobalCaptureInstance::get_instance().end_capture();
    ffi_capture_stream_guard.reset();
    if (!piecewise_graphs || piecewise_graphs->empty()) {
      LOG(WARNING) << "Failed to capture piecewise graph: no graphs captured";
      return false;
    }

    // Move piecewise graphs to member
    piecewise_graph_ = std::move(*piecewise_graphs);

    LOG(INFO) << "Piecewise graph capture end, bucket_num_tokens: "
              << bucket_num_tokens
              << ", num_graphs: " << piecewise_graph_.num_graphs()
              << ", num_runners: " << piecewise_graph_.num_runners()
              << ", num_instructions: " << piecewise_graph_.size();
    if (snapshot_linear_state) {
      restore_linear_attention_state(linear_state_snapshots);
    }
  } else {
    // Normal capture mode (for decode)
    // Reuses the outer `capture_stream` (set up at the top of this function
    // and active via stream_guard) so the warmup runs on the exact stream
    // about to be captured -- syncing a different stream would leave the
    // capture stream with stale pending work.
    static const bool capture_logits_enabled =
        std::getenv("XLLM_GRAPH_CAPTURE_LOGITS") != nullptr;
    capture_logits_ = capture_logits_enabled && bucket_num_tokens == 1 &&
                      !options.enable_speculative_decode();
    if (capture_logits_) {
      persistent_param_.ensure_logits_buffer(
          args.vocab_size(), torch::kBFloat16, persistent_param_.device());
      LOG(INFO) << "D1: capturing lm_head into decode graph (bucket_num_tokens="
                << bucket_num_tokens << ")";
    }
    for (int warmup_iter = 0; warmup_iter < 2; ++warmup_iter) {
      capture_stream.synchronize();
      {
        xllm::kernel::musa::MusaTvmffiPreparationSyncGuard ffi_sync_guard;
        auto warmup_result = model->forward(
            persistent_param_.persistent_tokens(padded_num_tokens_),
            persistent_param_.persistent_positions(padded_num_tokens_),
            kv_cache,
            graph_params_opt.value());
        if (capture_logits_) {
          // Pre-warm lm_head output_buf_ so no torch::empty fires under
          // capture.
          auto warmup_logits =
              model->logits(warmup_result.hidden_states, torch::Tensor());
          persistent_param_.set_logits(warmup_logits);
        }
      }
      xllm::kernel::musa::sync_musa_ffi_stream(persistent_param_.device());
      if (snapshot_linear_state) {
        // The warmup is an eager forward and therefore mutates the live
        // recurrent state. Restore it before the next warmup so every warmup
        // observes the same input state as the captured replay.
        capture_stream.synchronize();
      }
      if (snapshot_linear_state) {
        restore_linear_attention_state(linear_state_snapshots);
      }
    }
    capture_stream.synchronize();

    // Record Mate FFI internal scratch allocations on one extra eager forward.
    // The Mate decode .so allocates via TVM-FFI's DLPackManagedTensorAllocator
    // hook (torch::empty), which MUSA rejects under stream capture. We capture
    // the exact sequence of tensors here and replay them during capture_begin.
    recorded_ffi_allocs_.clear();
    xllm::kernel::musa::begin_ffi_alloc_record();
    {
      xllm::kernel::musa::MusaTvmffiPreparationSyncGuard ffi_sync_guard;
      auto ffi_forward = model->forward(
          persistent_param_.persistent_tokens(padded_num_tokens_),
          persistent_param_.persistent_positions(padded_num_tokens_),
          kv_cache,
          graph_params_opt.value());
      if (capture_logits_) {
        auto ffi_logits =
            model->logits(ffi_forward.hidden_states, torch::Tensor());
        persistent_param_.set_logits(ffi_logits);
      }
    }
    xllm::kernel::musa::sync_musa_ffi_stream(persistent_param_.device());
    if (snapshot_linear_state) {
      // The FFI recording pass is eager as well; leave the live cache at the
      // pre-capture state before beginning graph capture.
      capture_stream.synchronize();
    }
    if (snapshot_linear_state) {
      restore_linear_attention_state(linear_state_snapshots);
    }
    recorded_ffi_allocs_ = xllm::kernel::musa::end_ffi_alloc_record();
    capture_stream.synchronize();
    LOG(INFO) << "Recorded " << recorded_ffi_allocs_.size()
              << " Mate FFI scratch tensors for decode graph capture, "
                 "bucket_num_tokens="
              << bucket_num_tokens;

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
    std::optional<xllm::kernel::musa::MusaTvmffiStreamOverrideGuard>
        ffi_capture_stream_guard;
    ffi_capture_stream_guard.emplace(persistent_param_.device(),
                                     capture_stream_handle);
    graph_.capture_begin(pool, musaStreamCaptureModeThreadLocal);

    xllm::kernel::musa::begin_ffi_alloc_replay(&recorded_ffi_allocs_);
    // Execute forward pass - MUSA graph will capture this
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }

    // D1: capture lm_head GEMM inside the graph. For B=1 decode we skip
    // index_select (the only known MUSA capture blocker) by passing an
    // undefined selected_token_idxes - hidden_states is already [1, hidden].
    if (capture_logits_) {
      auto captured_logits =
          model->logits(forward_result.hidden_states, torch::Tensor());
      persistent_param_.set_logits(captured_logits);
    }

    // End graph capture
    graph_.capture_end();
    ffi_capture_stream_guard.reset();
    xllm::kernel::musa::end_ffi_alloc_replay();
    if (snapshot_linear_state) {
      capture_stream.synchronize();
    }
    if (snapshot_linear_state) {
      // The captured forward executes once while recording and mutates the
      // live convolution/SSM cache. Restore the pre-capture snapshot so the
      // first replay applies the current decode input exactly once.
      restore_linear_attention_state(linear_state_snapshots);
    }
  }

  // Synchronize to ensure graph capture is completed.
  capture_stream.synchronize();

  // Explicitly restore stream after capture before replay logic.
  stream_guard.reset();

  // Replay is unified in MusaGraphExecutorImpl::run() after capture success
  // for both prefill and decode.

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
        ::xllm::musa::DeviceCaptureLock::get_instance().get_read_lock(
            device_index_);
    replay_lock_guard.emplace(replay_lock);
  }

  // Update persistent parameters with new input data
  auto full_attention_cache =
      MusaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(full_attention_cache.has_value())
      << "MUSA graph replay requires at least one full-attention KV cache";
  const torch::Tensor& k_cache = full_attention_cache->first;
  const torch::Tensor& v_cache = full_attention_cache->second;

  if (is_piecewise_) {
    // Piecewise replay mode (for prefill)
    // Match capture: MoE/linear buffer rings must restart each forward so
    // same-shaped live values do not alias.
    std::optional<layer::musa::PiecewiseGraphMatmulBufferScope>
        replay_buffer_pool_scope;
    if (prefill_matmul_buffer_pool_) {
      replay_buffer_pool_scope.emplace(prefill_matmul_buffer_pool_.get());
    }
    const bool profile = s_enable_piecewise_profile();
    std::chrono::steady_clock::time_point update_start;
    if (profile) {
      c10::musa::getCurrentMUSAStream(device_index_).synchronize();
      update_start = std::chrono::steady_clock::now();
    }

    // Need to get updated params with attn_metadata for attention replay
    auto updated_params_opt =
        persistent_param_.update(tokens,
                                 k_cache,
                                 v_cache,
                                 positions,
                                 params,
                                 padded_num_tokens_,
                                 /*return_capture_params=*/true,
                                 /*update_prefill_plan=*/
                                 piecewise_graph_.requires_plan_info());
    double update_ms = 0.0;
    std::chrono::steady_clock::time_point replay_param_start;
    if (profile) {
      c10::musa::getCurrentMUSAStream(device_index_).synchronize();
      const std::chrono::steady_clock::time_point update_end =
          std::chrono::steady_clock::now();
      update_ms =
          std::chrono::duration<double, std::milli>(update_end - update_start)
              .count();
      replay_param_start = update_end;
    }
    CHECK(updated_params_opt.has_value())
        << "update() should return ModelInputParams for piecewise replay";

    const auto& updated_params = updated_params_opt.value();
    CHECK(!piecewise_graph_.empty()) << "Piecewise graph must not be empty";
    CHECK(updated_params.attn_metadata)
        << "attn_metadata is required for piecewise replay";
    if (piecewise_graph_.requires_plan_info()) {
      CHECK(updated_params.attn_metadata->plan_info)
          << "plan_info is required for piecewise replay";
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraph::replay() piecewise replay with uri="
        << updated_params.attn_metadata->plan_info->uri
        << ", plan_info.defined="
        << updated_params.attn_metadata->plan_info->plan_info.defined();

    // Build AttentionReplayParams from updated attn_metadata
    ::xllm::kernel::musa::AttentionReplayParams replay_params;
    replay_params.actual_num_tokens = actual_num_tokens;
    if (updated_params.attn_metadata->plan_info) {
      replay_params.plan_info =
          updated_params.attn_metadata->plan_info->plan_info;
    }
    replay_params.q_cu_seq_lens = updated_params.attn_metadata->q_cu_seq_lens;
    replay_params.gdn_cu_seq_lens =
        updated_params.attn_metadata->musa.gdn_cu_seq_lens;
    replay_params.gdn_kkt_cu_seq_lens =
        updated_params.attn_metadata->musa.gdn_kkt_cu_seq_lens;
    replay_params.kv_cu_seq_lens = updated_params.attn_metadata->kv_cu_seq_lens;
    replay_params.q_cu_seq_lens_host =
        updated_params.attn_metadata->musa.q_cu_seq_lens_host_vec;
    replay_params.max_seqlen_q = updated_params.attn_metadata->max_query_len;
    replay_params.max_seqlen_k = updated_params.attn_metadata->max_seq_len;
    replay_params.paged_kv_indptr =
        updated_params.attn_metadata->paged_kv_indptr;
    replay_params.paged_kv_indices =
        updated_params.attn_metadata->paged_kv_indices;
    replay_params.paged_kv_last_page_len =
        updated_params.attn_metadata->paged_kv_last_page_len;
    replay_params.paged_kv_indptr_host =
        updated_params.attn_metadata->musa.paged_kv_indptr_host;
    replay_params.paged_kv_indices_host =
        updated_params.attn_metadata->musa.paged_kv_indices_host;
    replay_params.paged_kv_last_page_len_host =
        updated_params.attn_metadata->musa.paged_kv_last_page_len_host;
    replay_params.qo_indptr = updated_params.attn_metadata->qo_indptr;

    double replay_param_ms = 0.0;
    if (profile) {
      const std::chrono::steady_clock::time_point replay_param_end =
          std::chrono::steady_clock::now();
      replay_param_ms = std::chrono::duration<double, std::milli>(
                            replay_param_end - replay_param_start)
                            .count();
    }

    // Replay piecewise graphs and attention runners
    piecewise_graph_.replay(replay_params);
    if (profile) {
      LOG(INFO) << "[PIECEWISE_REPLAY_PROFILE] actual_num_tokens="
                << actual_num_tokens
                << " padded_num_tokens=" << padded_num_tokens_
                << " update_ms=" << update_ms
                << " replay_param_ms=" << replay_param_ms;
    }
  } else {
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
                                 /*return_capture_params=*/true);
    CHECK(replay_params_opt.has_value())
        << "update() should return ModelInputParams for decode replay";

    const torch::Tensor& fresh_fa3_scheduler_metadata =
        replay_params_opt.value().attn_metadata->musa.fa3_scheduler_metadata;
    if (captured_fa3_scheduler_metadata_.defined()) {
      CHECK(fresh_fa3_scheduler_metadata.defined())
          << "FA3 scheduler metadata disappeared after graph capture";
      CHECK_EQ(captured_fa3_scheduler_metadata_.sizes(),
               fresh_fa3_scheduler_metadata.sizes())
          << "FA3 scheduler metadata shape changed after graph capture";
      captured_fa3_scheduler_metadata_.copy_(fresh_fa3_scheduler_metadata,
                                             /*non_blocking=*/true);
    }

    // Mate batch_decode receives CPU paged-KV tensors on every invocation.
    // Refresh the stable backing buffers before replay so captured host
    // pointers describe the current request. The normal path copies logical
    // CPU mirrors and performs no device-to-host transfer or stream sync;
    // legacy callers without mirrors retain the explicit blocking fallback.
    refresh_persistent_paged_kv_host_mirrors(
        replay_params_opt.value().attn_metadata, replay_params_opt.value());

    if (s_enable_graph_timing()) {
      auto stream = c10::musa::getCurrentMUSAStream(device_index_);
      stream.synchronize();
      const auto replay_start = std::chrono::steady_clock::now();
      graph_.replay();
      stream.synchronize();
      const auto replay_end = std::chrono::steady_clock::now();
      const auto replay_ms =
          std::chrono::duration<double, std::milli>(replay_end - replay_start)
              .count();
      LOG(INFO) << "GRAPH_TIMING actual_num_tokens=" << actual_num_tokens
                << " padded_num_tokens=" << padded_num_tokens_
                << " replay_ms=" << replay_ms;
    } else {
      graph_.replay();
    }
  }

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in MusaGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  ModelOutput output(get_hidden_states(actual_num_tokens));
  if (capture_logits_) {
    output.logits = persistent_param_.logits(actual_num_tokens);
  }
  return output;
}

// MusaGraphExecutorImpl implementation
namespace {

// Match SGLang ServerArgs._generate_piecewise_cuda_graph_tokens so pure-prefill
// batches reuse a small fixed set of pad sizes instead of ceil-to-16 buckets.
std::vector<uint32_t> generate_piecewise_prefill_graph_tokens(
    uint32_t max_tokens) {
  std::vector<uint32_t> capture_sizes;
  auto append_range =
      [&](uint32_t start, uint32_t end_inclusive, uint32_t step) {
        for (uint32_t size = start; size <= end_inclusive && size <= max_tokens;
             size += step) {
          capture_sizes.push_back(size);
        }
      };
  append_range(/*start=*/4, /*end_inclusive=*/32, /*step=*/4);
  append_range(/*start=*/48, /*end_inclusive=*/256, /*step=*/16);
  append_range(/*start=*/288, /*end_inclusive=*/512, /*step=*/32);
  append_range(/*start=*/576, /*end_inclusive=*/1024, /*step=*/64);
  append_range(/*start=*/1280, /*end_inclusive=*/4096, /*step=*/256);
  append_range(/*start=*/4608, /*end_inclusive=*/max_tokens, /*step=*/512);
  // The coarse SGLang ladder leaves increasingly large padding gaps for
  // nominal 2k prompts and packed multi-sequence batches. Keep the extra
  // shoulders aligned to 64 tokens, the Mate GDN/KKT granularity. The 2112
  // shoulder prevents a one-token crossing of 2048 from selecting 2304; the
  // remaining entries tighten B=2..7 without introducing a shape for every
  // exact host length.
  constexpr std::array<uint32_t, 8> kPrefillShoulders = {
      2080, 2112, 4352, 6400, 8448, 10496, 12544, 14592};
  for (uint32_t shoulder : kPrefillShoulders) {
    if (shoulder <= max_tokens) {
      capture_sizes.push_back(shoulder);
    }
  }
  std::sort(capture_sizes.begin(), capture_sizes.end());
  capture_sizes.erase(std::unique(capture_sizes.begin(), capture_sizes.end()),
                      capture_sizes.end());
  if (max_tokens > 0 &&
      (capture_sizes.empty() || capture_sizes.back() < max_tokens)) {
    capture_sizes.push_back(max_tokens);
  }
  return capture_sizes;
}

bool should_enable_prefill_piecewise_graph(const runtime::Options& options) {
  const bool configured =
      ::xllm::ExecutionConfig::get_instance().enable_prefill_piecewise_graph();
  // The one-layer MTP draft model uses FlashInfer prefill plan updates that
  // allocate/copy metadata.  Those operations are intentionally outside the
  // independent draft-decode and target-verify graphs and are not capturable
  // on MUSA.  Keep piecewise prefill graph capture for the target model only.
  return configured && !options.is_draft_engine();
}

}  // namespace

MusaGraphExecutorImpl::MusaGraphExecutorImpl(CausalLM* model,
                                             const ModelArgs& args,
                                             const torch::Device& device,
                                             const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      enable_prefill_piecewise_graph_(
          should_enable_prefill_piecewise_graph(options)),
      enable_packed_prefill_(
          ::xllm::ExecutionConfig::get_instance().enable_packed_prefill()) {
  max_tokens_for_graph_mode_ =
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode();
  if (max_tokens_for_graph_mode_ < options_.max_seqs_per_batch()) {
    max_tokens_for_graph_mode_ = options_.max_seqs_per_batch();
  }
  prefill_graph_token_buckets_ = generate_piecewise_prefill_graph_tokens(
      static_cast<uint32_t>(max_tokens_for_graph_mode_));
  {
    std::ostringstream oss;
    for (size_t i = 0; i < prefill_graph_token_buckets_.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << prefill_graph_token_buckets_[i];
    }
    LOG(INFO) << "Prefill piecewise graph token buckets ("
              << prefill_graph_token_buckets_.size() << "): [" << oss.str()
              << "]";
  }
  // Keep one pool per executor instance so all captured graphs can reuse it,
  // while avoiding cross-instance stale-handle reuse.
  graph_pool_ = at::musa::graph_pool_handle();
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
// Physical pool id: same id => reuse across different shapes (prefill vs decode
// are different physical pools).
constexpr uint32_t kPhysicalPoolIdPrefill = 0;
constexpr uint32_t kPhysicalPoolIdDecode = 1;
}  // namespace

struct MusaGraphExecutorImpl::VmmPoolState {};

MusaGraphExecutorImpl::~MusaGraphExecutorImpl() {
  chunked_prefill_graphs_.clear();
  packed_prefill_graphs_.clear();
  spec_verify_graphs_.clear();
  prefill_graphs_.clear();
  graphs_.clear();
}

MusaGraphExecutorImpl::VmmPoolState&
MusaGraphExecutorImpl::get_or_create_vmm_pool_state(uint32_t physical_pool_id) {
  LOG(FATAL) << "Graph VMM pool is not enabled for MUSA builds";
}

MusaMemPool* MusaGraphExecutorImpl::get_or_create_vmm_mempool(
    uint32_t physical_pool_id,
    uint32_t shape_id) {
  (void)physical_pool_id;
  (void)shape_id;
  LOG(FATAL) << "Graph VMM pool is not enabled for MUSA builds";
  return nullptr;
}

MusaMemPool* MusaGraphExecutorImpl::get_vmm_mempool(uint32_t physical_pool_id,
                                                    uint32_t shape_id) {
  (void)physical_pool_id;
  (void)shape_id;
  return nullptr;
}

void MusaGraphExecutorImpl::reset_vmm_allocator_offset(
    uint32_t physical_pool_id) {
  (void)physical_pool_id;
}

MusaGraphExecutorImpl::GraphMemoryUsageStats
MusaGraphExecutorImpl::get_graph_memory_usage_stats() {
  return GraphMemoryUsageStats{};
}

size_t MusaGraphExecutorImpl::get_graph_memory_usage_bytes() { return 0; }

void MusaGraphExecutorImpl::log_graph_memory_after_capture() {}

// Get graph memory pool id for capture. When VMM is enabled, uses per-shape
// MemPool under (physical_pool_id, shape_id).
c10::musa::MempoolId_t MusaGraphExecutorImpl::get_mem_pool(
    uint32_t physical_pool_id,
    uint32_t shape_id) {
  if (!::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
    // Non-VMM mode intentionally uses one pool per executor instance.
    // Rationale: this executor is designed for single-threaded invocation, and
    // concurrent run() on the same executor instance is not allowed.
    // Under this contract, a per-instance pool is safe and keeps graph memory
    // ownership tied to the executor lifecycle.
    return graph_pool_;
  }
  // Per-shape VMM MemPool: look up pool for (physical_pool_id, shape_id).
  MusaMemPool* pool = get_vmm_mempool(physical_pool_id, shape_id);
  CHECK(pool != nullptr)
      << "VMM MemPool for shape_id=" << shape_id
      << ", physical_pool_id=" << physical_pool_id
      << " not found; get_or_create_vmm_mempool must be called before capture";
  return pool->id();
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
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput MusaGraphExecutorImpl::attach_aux_hidden_states_if_needed(
    const torch::Tensor& hidden_states,
    uint32_t n_tokens) const {
  if (options_.enable_graph_aux_hidden_states()) {
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
  // Only intervene on decode or MTP spec-verify validate graph paths.
  // Piecewise prefill already breaks the graph at attention boundaries.
  const bool in_spec_verify_embedding_phase =
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  if (!params.meta.batch_forward_type.is_decode() &&
      !in_spec_verify_embedding_phase) {
    return params;
  }

  auto embed_layer = model_->get_word_embedding();
  if (embed_layer.is_empty()) {
    return params;
  }

  const bool is_qwen35_mtp = args_.model_type() == "qwen3_5_mtp" ||
                             args_.model_type() == "qwen3_5_moe_mtp";
  if (is_qwen35_mtp && params.embedding.input_embedding.defined()) {
    ModelInputParams new_params = params;
    new_params.embedding.mtp_token_embedding = embed_layer(tokens);
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
  new_params.embedding.input_embedding = embed_layer(tokens);
  return new_params;
}

ModelOutput MusaGraphExecutorImpl::run(const torch::Tensor& tokens,
                                       const torch::Tensor& positions,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  const bool is_prefill = params.meta.batch_forward_type.is_prefill();
  const bool is_chunked_prefill =
      params.meta.batch_forward_type.is_chunked_prefill();
  const bool is_mixed = params.meta.batch_forward_type.is_mixed();
  const bool is_decode = params.meta.batch_forward_type.is_decode();

  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens.size(/*dim=*/0);
  const int64_t effective_num_sequences = params.meta.actual_num_sequences > 0
                                              ? params.meta.actual_num_sequences
                                              : params.meta.num_sequences;
  // Packed multi-seq piecewise uses a token bucket.  The live ragged CU
  // metadata (and the separate KKT sentinel) prevents the bucket tail from
  // mutating recurrent state, so exact host lengths must not force a recapture.
  const bool packed_multi_piecewise = is_prefill && enable_packed_prefill_ &&
                                      s_enable_packed_prefill_piecewise() &&
                                      effective_num_sequences > 1 &&
                                      !params.is_spec_verify;
  const bool fixed_qwen35_mtp_draft_decode =
      is_decode && (args_.model_type() == "qwen3_5_mtp" ||
                    args_.model_type() == "qwen3_5_moe_mtp");
  // Chunked/mixed GDN control flow is specialized to the exact packed query
  // shape. Unlike pure prefill and packed pure-prefill, it cannot safely
  // append padding tokens because those tokens would mutate recurrent state
  // and paged-KV state.
  uint32_t bucket_num_tokens =
      (is_chunked_prefill || is_mixed)
          ? n_tokens
          : get_bucket_num_tokens(n_tokens, is_prefill);
  // The Mate varlen KKT kernel pads its token dimension to 64 internally. Keep
  // the graph-owned packed buffers at that same granularity so an externally
  // supplied KKT/output buffer never changes shape between warmup and replay.
  const bool qwen35_prefill =
      is_prefill && args_.model_type().find("qwen3_5") != std::string::npos;
  bool prefill_bucket_shape_supported = true;
  const bool qwen35_c1_partial_bucket = qwen35_prefill &&
                                        effective_num_sequences == 1 &&
                                        bucket_num_tokens == 2080;
  if (qwen35_prefill && !qwen35_c1_partial_bucket &&
      bucket_num_tokens % 64 != 0) {
    const uint32_t aligned_bucket = ((bucket_num_tokens + 63) / 64) * 64;
    if (aligned_bucket <= max_tokens_for_graph_mode_) {
      bucket_num_tokens = aligned_bucket;
    } else {
      prefill_bucket_shape_supported = false;
    }
  }
  // The B>1 Qwen3.5 Mate/GDN piecewise runner is not correct for the first
  // 64-token bucket: a short arithmetic prompt (B=2, 54 live tokens) leaves
  // the packed recurrent output corrupted even though the HTTP request
  // succeeds.  Keep only that tiny bucket on eager prefill; production-sized
  // buckets (including the 2k-prompt 4352 bucket) retain graph replay.
  constexpr uint32_t kMinQwen35PackedPiecewiseGraphTokens = 128;
  const bool qwen35_packed_bucket_too_small =
      qwen35_prefill && packed_multi_piecewise &&
      bucket_num_tokens < kMinQwen35PackedPiecewiseGraphTokens;
  const int64_t qwen35_c1_max_padding =
      s_qwen35_c1_piecewise_max_padding_tokens();
  const uint32_t prefill_padding_tokens =
      bucket_num_tokens >= n_tokens ? bucket_num_tokens - n_tokens : 0;
  const bool qwen35_c1_padding_too_large =
      qwen35_prefill && !packed_multi_piecewise &&
      effective_num_sequences == 1 && qwen35_c1_max_padding >= 0 &&
      static_cast<int64_t>(prefill_padding_tokens) > qwen35_c1_max_padding;
  bool force_eager_prefill = !prefill_bucket_shape_supported ||
                             qwen35_packed_bucket_too_small ||
                             qwen35_c1_padding_too_large ||
                             (packed_multi_piecewise &&
                              bucket_num_tokens > max_tokens_for_graph_mode_);
  if (qwen35_c1_padding_too_large) {
    LOG_FIRST_N(INFO, 10) << "Qwen3.5 C1 prefill padding "
                          << prefill_padding_tokens
                          << " exceeds piecewise threshold "
                          << qwen35_c1_max_padding
                          << "; using eager for actual=" << n_tokens
                          << " bucket=" << bucket_num_tokens;
  }
  // A packed graph currently contains B-shaped GDN state/scatter tensors.
  // Reusing a graph captured for another B would be incorrect, but capturing
  // another graph for every scheduler partition defeats the bucket win. If a
  // bucket already has a graph for a different effective B, take the bounded
  // eager path for this alternate shape instead of paying another lazy capture
  // (and keep the first graph available for its stable shape).
  if (packed_multi_piecewise && enable_prefill_piecewise_graph_ &&
      prefill_bucket_shape_supported) {
    const uint64_t current_key =
        get_packed_prefill_graph_key(bucket_num_tokens, params);
    const uint32_t bucket_key = static_cast<uint32_t>(current_key);
    const bool current_graph_exists =
        packed_prefill_graphs_.find(current_key) !=
        packed_prefill_graphs_.end();
    const bool bucket_has_other_batch_shape =
        std::any_of(packed_prefill_graphs_.begin(),
                    packed_prefill_graphs_.end(),
                    [bucket_key, current_key](const auto& item) {
                      return static_cast<uint32_t>(item.first) == bucket_key &&
                             item.first != current_key;
                    });
    if (!current_graph_exists && bucket_has_other_batch_shape) {
      force_eager_prefill = true;
      VLOG(kGraphExecutorLogVerboseLevel)
          << "Packed prefill bucket " << bucket_num_tokens
          << " already has a graph for another batch shape; using eager for B="
          << effective_num_sequences
          << " instead of recapturing a second graph";
    }
  }
  if (fixed_qwen35_mtp_draft_decode) {
    bucket_num_tokens =
        static_cast<uint32_t>(options_.max_seqs_per_batch() * 2);
    CHECK_LE(n_tokens, bucket_num_tokens)
        << "Qwen3.5 MTP draft rows exceed fixed graph capacity";
  }

  // Prefill phase with piecewise graph. Single-sequence Qwen3.5 prefill uses
  // the ladder only while its bucket padding stays within the adaptive
  // threshold above; packed multi-sequence prefill is enabled by
  // XLLM_PACKED_PREFILL_PIECEWISE=1 and uses the bucket path below unchanged.
  if (is_prefill && enable_prefill_piecewise_graph_ &&
      (!enable_packed_prefill_ || s_enable_packed_prefill_piecewise()) &&
      !packed_multi_piecewise && !force_eager_prefill) {
    // Check if token count is within limit
    const bool graph_mode_supported =
        prefill_bucket_shape_supported &&
        bucket_num_tokens <= max_tokens_for_graph_mode_;

    if (!graph_mode_supported) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "Token count " << n_tokens
          << " exceeds max_tokens_for_graph_mode ("
          << max_tokens_for_graph_mode_ << "), falling back to eager mode";
      COUNTER_INC(num_model_execution_total_eager);
      auto result = model_->forward(tokens, positions, kv_caches, params);
      maybe_empty_prefill_cache();
      return result;
    }

    // Check if piecewise graph exists for this bucket
    auto it = prefill_graphs_.find(bucket_num_tokens);
    if (it != prefill_graphs_.end()) {
      // Replay existing piecewise graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "MusaGraphExecutorImpl::run() in prefill piecewise replay mode";
      auto result = timed_prefill_graph_replay(
          device_.index(),
          n_tokens,
          params.meta.num_sequences,
          enable_packed_prefill_,
          "piecewise_replay",
          [&]() {
            return it->second->replay(tokens, positions, kv_caches, params);
          });
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Graph doesn't exist, try to create it lazily with piecewise capture
    auto graph =
        std::make_unique<MusaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()),
                                    /*is_piecewise=*/true);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphExecutorImpl::run() in prefill piecewise capture mode";

    MusaMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdPrefill);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdPrefill, shape_id);
    }
    const c10::musa::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdPrefill, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing piecewise MUSA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      log_graph_memory_after_capture();

      // Save the graph for future reuse
      prefill_graphs_[bucket_num_tokens] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests.
      auto result = timed_prefill_graph_replay(
          device_.index(),
          n_tokens,
          params.meta.num_sequences,
          enable_packed_prefill_,
          "piecewise_capture_replay",
          [&]() {
            return prefill_graphs_[bucket_num_tokens]->replay(
                tokens, positions, kv_caches, params);
          });
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Fail fast intentionally: after entering graph mode, silently falling back
    // to eager can hide allocator/capture regressions and make latency behavior
    // non-deterministic in production. Operators can disable graph mode via
    // ::xllm::ExecutionConfig::get_instance().enable_graph() or
    // ::xllm::ExecutionConfig::get_instance().enable_prefill_piecewise_graph()
    // when fallback behavior is preferred over strict graph-mode enforcement.
    LOG(FATAL)
        << "Failed to capture piecewise MUSA graph for bucket num_tokens: "
        << bucket_num_tokens << " (actual num_tokens: " << n_tokens << ")";
  }

  // Chunked-prefill batches use paged attention rather than ragged prefill
  // attention. They still use piecewise capture, but require an exact packed
  // query shape because Qwen3.5 GDN host control flow is specialized to each
  // sequence's query length and initial-state presence. MIXED is intentionally
  // excluded: the graph-compatible chunked scheduler emits homogeneous
  // prefill/chunked or decode batches to avoid unbounded mixed-shape captures.
  // Also serves packed multi-seq pure-prefill when
  // XLLM_PACKED_PREFILL_PIECEWISE=1. Packed graphs are keyed only by bucket
  // capacity and effective sequence count; live CU/host lengths are replay
  // metadata and must not trigger a fresh capture.
  if ((is_chunked_prefill || packed_multi_piecewise) &&
      prefill_bucket_shape_supported && !force_eager_prefill &&
      enable_prefill_piecewise_graph_ && !params.is_spec_verify) {
    CHECK_LE(bucket_num_tokens, max_tokens_for_graph_mode_)
        << "Chunked-prefill graph batch exceeds max_tokens_for_graph_mode: "
        << bucket_num_tokens << " > " << max_tokens_for_graph_mode_;

    const uint64_t graph_key =
        packed_multi_piecewise
            ? get_packed_prefill_graph_key(bucket_num_tokens, params)
            : get_chunked_prefill_graph_key(bucket_num_tokens, params);
    auto& graph_cache = packed_multi_piecewise ? packed_prefill_graphs_
                                               : chunked_prefill_graphs_;
    auto it = graph_cache.find(graph_key);
    if (it != graph_cache.end()) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "MusaGraphExecutorImpl::run() in chunked-prefill piecewise "
             "replay mode";
      auto result = timed_prefill_graph_replay(
          device_.index(),
          n_tokens,
          params.meta.num_sequences,
          enable_packed_prefill_,
          packed_multi_piecewise ? "packed_piecewise_replay"
                                 : "chunked_piecewise_replay",
          [&]() {
            return it->second->replay(tokens, positions, kv_caches, params);
          });
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    auto graph =
        std::make_unique<MusaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()),
                                    /*is_piecewise=*/true);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphExecutorImpl::run() in chunked-prefill piecewise "
           "capture mode";

    MusaMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdPrefill);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdPrefill, shape_id);
    }
    const c10::musa::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdPrefill, bucket_num_tokens);

    const bool capture_success = graph->capture(model_,
                                                args_,
                                                options_,
                                                tokens,
                                                positions,
                                                params,
                                                kv_caches,
                                                bucket_num_tokens,
                                                mem_pool,
                                                pool_ptr);
    if (capture_success) {
      LOG(INFO) << "Lazy capturing piecewise MUSA graph for "
                << params.meta.batch_forward_type.to_string()
                << " bucket num_tokens: " << bucket_num_tokens
                << " (actual num_tokens: " << n_tokens << ") done";
      log_graph_memory_after_capture();
      graph_cache[graph_key] = std::move(graph);
      auto result = timed_prefill_graph_replay(
          device_.index(),
          n_tokens,
          params.meta.num_sequences,
          enable_packed_prefill_,
          packed_multi_piecewise ? "packed_piecewise_capture_replay"
                                 : "chunked_piecewise_capture_replay",
          [&]() {
            return graph_cache[graph_key]->replay(
                tokens, positions, kv_caches, params);
          });
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    LOG(FATAL) << "Failed to capture piecewise MUSA graph for "
               << params.meta.batch_forward_type.to_string()
               << " bucket num_tokens: " << bucket_num_tokens
               << " (actual num_tokens: " << n_tokens << ")";
  }

  // Prefill without piecewise graph: use eager mode
  if (is_prefill) {
    COUNTER_INC(num_model_execution_total_eager);
    const bool time_fwd =
        s_enable_prefill_fwd_timing() || PrefillBreakdown::enabled();
    if (time_fwd) {
      PrefillBreakdown::begin();
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
                  << " packed_prefill=" << enable_packed_prefill_
                  << " mode=eager"
                  << " fwd_ms=" << ms;
      }
      PrefillBreakdown::end_and_log(
          static_cast<int64_t>(n_tokens), params.meta.num_sequences, ms);
      maybe_empty_prefill_cache();
      return result;
    }
    auto result = model_->forward(tokens, positions, kv_caches, params);
    maybe_empty_prefill_cache();
    return result;
  }

  // Decode phase with full graph
  if (is_decode) {
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

    // Check if captured graph exists for this bucket num_tokens
    auto it = graphs_.find(bucket_num_tokens);
    if (it != graphs_.end()) {
      // Replay the existing graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "MusaGraphExecutorImpl::run() in decode replay mode";
      auto result =
          it->second->replay(tokens, positions, kv_caches, graph_params);
      auto output =
          attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
      if (result.logits.defined()) {
        output.logits = result.logits;
      }
      return output;
    }

    // Graph doesn't exist for this bucket num_tokens, try to create it lazily
    auto graph =
        std::make_unique<MusaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphExecutorImpl::run() in decode capture mode";

    MusaMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdDecode);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdDecode, shape_id);
    }
    const c10::musa::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdDecode, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          graph_params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing MUSA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      log_graph_memory_after_capture();

      // Save the graph for future reuse
      graphs_[bucket_num_tokens] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests. Recompute the embedding so the persistent buffer
      // reflects the current token batch (the capture-time embedding above
      // would otherwise be reused unchanged, which is only correct in the
      // unlikely case the post-capture request happens to match exactly).
      const ModelInputParams replay_params =
          maybe_precompute_embedding_for_graph(tokens, params);
      auto result = graphs_[bucket_num_tokens]->replay(
          tokens, positions, kv_caches, replay_params);
      auto output =
          attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
      if (result.logits.defined()) {
        output.logits = result.logits;
      }
      return output;
    }

    // Same fail-fast policy as prefill capture above: keep graph-mode behavior
    // explicit and avoid silently switching execution semantics after a capture
    // failure. Use ::xllm::ExecutionConfig::get_instance().enable_graph() to
    // turn off graph mode if eager fallback is desired for resiliency.
    LOG(FATAL) << "Failed to capture MUSA graph for bucket num_tokens: "
               << bucket_num_tokens << " (actual num_tokens: " << n_tokens
               << ")";
  }

  // MTP spec-verify validate phase (Qwen3.5 hybrid linear attention).
  const bool in_spec_verify_phase =
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  if (in_spec_verify_phase) {
    static const bool force_spec_verify_eager =
        std::getenv("XLLM_SPEC_VERIFY_EAGER") != nullptr;
    // Qwen3.5 FP8 MoE AOT artifacts are fixed-batch kernels (B<=8).  The
    // larger MTP verify buckets (C=4 produces 9/12 rows) must be executed
    // eagerly: capturing the chunked routed-MoE path and replaying it across
    // waves can leave the graph/state stream stalled.  Keep the <=8 path on
    // graph so C=1 (three verify rows for K=2) retains its decode graph speed.
    static const bool eager_large_spec_verify = util::get_bool_env(
        "XLLM_SPEC_VERIFY_LARGE_BATCH_EAGER", true);
    constexpr uint32_t kMaxQwen35SpecVerifyGraphTokens = 8;
    const bool is_qwen35_model =
        args_.model_type().find("qwen3_5") != std::string::npos;
    const bool large_qwen35_spec_verify =
        is_qwen35_model && n_tokens > kMaxQwen35SpecVerifyGraphTokens;
    if (force_spec_verify_eager ||
        (eager_large_spec_verify && large_qwen35_spec_verify)) {
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
    if (!model_->is_hybrid_linear_attention()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode for spec verify because the "
             "chunked-prefill validate graph path is currently only adapted "
             "for hybrid linear attention models.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
    if (!params.graph.use_expanded_decode_for_spec_verify_attention) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode for spec verify because expanded "
             "decode attention graph input was not prepared.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    const auto max_seq_len = args_.max_position_embeddings();
    if (params.meta.kv_max_seq_len > max_seq_len) {
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    const ModelInputParams graph_params =
        maybe_precompute_embedding_for_graph(tokens, params);
    const uint64_t graph_key = get_graph_key(bucket_num_tokens, graph_params);

    auto it = spec_verify_graphs_.find(graph_key);
    if (it != spec_verify_graphs_.end()) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "MusaGraphExecutorImpl::run() in spec-verify replay mode";
      auto result =
          it->second->replay(tokens, positions, kv_caches, graph_params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    auto graph =
        std::make_unique<MusaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "MusaGraphExecutorImpl::run() in spec-verify capture mode";

    MusaMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdDecode);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdDecode, shape_id);
    }
    const c10::musa::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdDecode, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          graph_params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing MUSA spec-verify graph for bucket "
                   "num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ", q_max_seq_len: " << graph_params.meta.q_max_seq_len
                << ") done";
      log_graph_memory_after_capture();
      spec_verify_graphs_[graph_key] = std::move(graph);
      const ModelInputParams replay_params =
          maybe_precompute_embedding_for_graph(tokens, params);
      auto result = spec_verify_graphs_[graph_key]->replay(
          tokens, positions, kv_caches, replay_params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    LOG_FIRST_N(WARNING, 1)
        << "Failed to capture MUSA spec-verify graph for bucket num_tokens: "
        << bucket_num_tokens << ", falling back to eager mode.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Chunked prefill without a spec-verify graph path (e.g. draft extend) runs
  // eager: only MTP validate expanded-decode has a captured graph today.
  if (params.meta.batch_forward_type.is_chunked_prefill() ||
      params.meta.batch_forward_type.is_mixed()) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for chunked prefill/mixed batch "
           "without a MUSA graph path (bucket num_tokens="
        << bucket_num_tokens
        << ", type=" << params.meta.batch_forward_type.to_string() << ").";
    COUNTER_INC(num_model_execution_total_eager);
    auto result = model_->forward(tokens, positions, kv_caches, params);
    Device::empty_cache(/*device_index=*/-1);
    return result;
  }

  // Defensive fallback for unsupported forward types (should be unreachable for
  // normal prefill/decode paths).
  LOG(ERROR) << "Failed to capture MUSA graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

uint64_t MusaGraphExecutorImpl::get_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params) const {
  if (params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill()) {
    const uint64_t q_max_seq_len =
        static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1));
    return static_cast<uint64_t>(bucket_num_tokens) | kSpecVerifyGraphKeyMask |
           (q_max_seq_len << kSpecVerifyQMaxSeqLenShift);
  }
  return static_cast<uint64_t>(bucket_num_tokens);
}

uint64_t MusaGraphExecutorImpl::get_chunked_prefill_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params) const {
  constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffsetBasis;
  const auto mix = [&hash](uint64_t value) {
    hash ^= value;
    hash *= kFnvPrime;
  };

  mix(static_cast<uint64_t>(bucket_num_tokens));
  mix(static_cast<uint64_t>(params.meta.batch_forward_type.value()));
  mix(static_cast<uint64_t>(params.meta.num_sequences));

  const std::vector<int32_t>& q_seq_lens = params.attention.host.q_seq_lens;
  mix(static_cast<uint64_t>(q_seq_lens.size()));
  for (int32_t q_seq_len : q_seq_lens) {
    mix(static_cast<uint64_t>(q_seq_len));
  }

  const std::vector<int64_t>& query_start_loc = params.parallel.query_start_loc;
  mix(static_cast<uint64_t>(query_start_loc.size()));
  for (int64_t query_start : query_start_loc) {
    mix(static_cast<uint64_t>(query_start));
  }

  const std::vector<int64_t>& has_initial_state =
      params.parallel.has_initial_state;
  mix(static_cast<uint64_t>(has_initial_state.size()));
  for (int64_t has_state : has_initial_state) {
    mix(static_cast<uint64_t>(has_state));
  }
  return hash;
}

uint64_t MusaGraphExecutorImpl::get_packed_prefill_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params) const {
  const uint64_t effective_num_sequences =
      params.meta.actual_num_sequences > 0
          ? static_cast<uint64_t>(params.meta.actual_num_sequences)
          : static_cast<uint64_t>(params.meta.num_sequences);
  CHECK_GT(effective_num_sequences, 0u)
      << "Packed prefill graph requires at least one sequence";
  CHECK_LE(effective_num_sequences,
           static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      << "Packed prefill sequence count does not fit graph key";
  // The low 32 bits are the token bucket; the high 32 bits are the effective
  // batch size.  No request-specific lengths belong in this key: they are
  // copied into the persistent live CU buffers before each replay.
  return static_cast<uint64_t>(bucket_num_tokens) |
         (effective_num_sequences << 32);
}

uint32_t MusaGraphExecutorImpl::get_bucket_num_tokens(uint32_t num_tokens,
                                                      bool is_prefill) const {
  // Prefill pads to the SGLang-aligned fixed ladder so nearby prompt lengths
  // share one captured graph (e.g. 2558..2560 -> 2560, 2561..2816 -> 2816).
  if (is_prefill) {
    if (prefill_graph_token_buckets_.empty()) {
      return num_tokens;
    }
    auto it = std::lower_bound(prefill_graph_token_buckets_.begin(),
                               prefill_graph_token_buckets_.end(),
                               num_tokens);
    if (it != prefill_graph_token_buckets_.end()) {
      return *it;
    }
    // Above the ladder: keep exact size; caller already gates on max tokens.
    return num_tokens;
  }
  return static_cast<uint32_t>(get_decode_graph_bucket_num_tokens(num_tokens));
}

// NOTE: REGISTER_EXECUTOR for MusaGraphExecutorImpl lives in
// musa_graph_executor_impl.h. Keeping it in this .cpp meant the static
// initializer's TU was referenced only via runtime factory lookup, and the
// linker dropped libmusa_graph_executor.a's only .o as unused. Putting the
// macro in the header matches base/vlm/acl/mlu/dcu graph executors.

}  // namespace xllm::runtime::musa
