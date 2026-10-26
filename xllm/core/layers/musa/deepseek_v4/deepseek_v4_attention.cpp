/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/layers/musa/deepseek_v4/deepseek_v4_attention.h"

#include <c10/musa/MUSAGuard.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <torch/serialize.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/musa/musa_dsv4_compat.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/ops_api.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_compressor_projection.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_graph_policy.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_graph_sequence.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_indexer_gemm.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_indexer_reference.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_paged_cache_view.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_pending_cache_rows.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_prefill_continuation.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_rope_assembly.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_rope_main_assembly.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_sequence_output.h"

namespace xllm {
namespace layer {
namespace {

constexpr char kDsv4StateBlockIndicesMetadataKey[] =
    "__musa_dsv4_state_block_indices";
constexpr char kDsv4PrefillResetBlocksMetadataKey[] =
    "__musa_dsv4_prefill_reset_blocks";
constexpr char kDsv4PrefillStateReadIndicesMetadataKey[] =
    "__musa_dsv4_prefill_state_read_indices";
constexpr char kDsv4HostSequenceLengthsMetadataKey[] =
    "__musa_dsv4_host_sequence_lengths";
constexpr char kDsv4BatchedQkvProjectionMetadataKey[] =
    "__musa_dsv4_batched_qkv_projection";
constexpr char kDsv4BatchedQrMetadataKey[] = "__musa_dsv4_batched_qr";
constexpr char kDsv4BatchedQbMetadataKey[] = "__musa_dsv4_batched_qb";
constexpr char kDsv4BatchedHeadWeightsMetadataKey[] =
    "__musa_dsv4_batched_head_weights";
constexpr char kDsv4BatchedCompressorMetadataKey[] =
    "__musa_dsv4_batched_compressor";
constexpr char kDsv4GraphSequenceIndexMetadataKey[] =
    "__musa_dsv4_graph_sequence_index";

thread_local bool dsv4_sequence_view_active = false;
thread_local bool dsv4_defer_output_projection = false;
thread_local bool dsv4_defer_o_a_projection = false;

std::optional<size_t> dsv4_graph_sequence_index(
    const DSAMetadata& attn_metadata) {
  const auto it =
      attn_metadata.cp_input_dict.find(kDsv4GraphSequenceIndexMetadataKey);
  if (it == attn_metadata.cp_input_dict.end()) {
    return std::nullopt;
  }
  const torch::Tensor& index = it->second;
  if (!index.defined() || !index.device().is_cpu() || index.numel() != 1 ||
      !index.is_contiguous()) {
    return std::nullopt;
  }
  int64_t value = -1;
  if (index.scalar_type() == torch::kInt32) {
    value = static_cast<int64_t>(index.data_ptr<int32_t>()[0]);
  } else if (index.scalar_type() == torch::kInt64) {
    value = index.data_ptr<int64_t>()[0];
  } else {
    return std::nullopt;
  }
  if (value < 0) {
    return std::nullopt;
  }
  return static_cast<size_t>(value);
}

class SequenceViewScope final {
 public:
  explicit SequenceViewScope(bool defer_output_projection,
                             bool defer_o_a_projection)
      : previous_view_(dsv4_sequence_view_active),
        previous_defer_(dsv4_defer_output_projection),
        previous_o_a_(dsv4_defer_o_a_projection) {
    dsv4_sequence_view_active = true;
    dsv4_defer_output_projection = defer_output_projection;
    dsv4_defer_o_a_projection = defer_o_a_projection;
  }
  ~SequenceViewScope() {
    dsv4_sequence_view_active = previous_view_;
    dsv4_defer_output_projection = previous_defer_;
    dsv4_defer_o_a_projection = previous_o_a_;
  }
  SequenceViewScope(const SequenceViewScope&) = delete;
  SequenceViewScope& operator=(const SequenceViewScope&) = delete;

 private:
  const bool previous_view_;
  const bool previous_defer_;
  const bool previous_o_a_;
};

bool packed_fp8_activations_enabled() {
  const char* value = std::getenv("XLLM_DSV4_FP8_PACKED_ACTIVATIONS");
  return value != nullptr && std::string(value) == "1";
}

torch::Tensor packed_projection_input(const torch::Tensor& input) {
  if (!packed_fp8_activations_enabled() || input.numel() == 0 ||
      input.scalar_type() != torch::kBFloat16 || input.size(-1) % 128 != 0) {
    return input;
  }
  // Diagnostic only: restore FP8 activation rounding before materialized
  // BF16 projector weights. Never alter the caller's HC/compressor input.
  auto contiguous = input.contiguous();
  auto quantized = kernel::musa::per_token_group_quant_fp8(contiguous, 128);
  const int64_t groups = input.size(-1) / 128;
  auto values =
      std::get<0>(quantized).to(torch::kFloat32).view({-1, groups, 128});
  auto scales = std::get<1>(quantized).view({-1, groups, 1});
  return (values * scales).to(input.scalar_type()).view(input.sizes());
}

void dump_local_prefill_stage(int32_t layer_id,
                              const char* stage,
                              const torch::Tensor& tensor) {
  const char* directory = std::getenv("XLLM_DSV4_PREFILL_CORE_DUMP");
  if (directory == nullptr || *directory == '\0' ||
      tensor.device().index() != 7) {
    return;
  }
  const std::filesystem::path path =
      std::filesystem::path(directory) /
      ("layer_" + std::to_string(layer_id) + "_" + stage + "_rank7.pt");
  if (!std::filesystem::exists(path)) {
    torch::serialize::OutputArchive archive;
    archive.write("value", tensor.to(torch::kCPU));
    archive.save_to(path.string());
  }
}

void dump_local_prefill_attention(int32_t layer_id,
                                  const torch::Tensor& query,
                                  const torch::Tensor& kv,
                                  const torch::Tensor& compressed,
                                  const torch::Tensor& indices,
                                  const torch::Tensor& sink,
                                  const torch::Tensor& output,
                                  int64_t query_start,
                                  int64_t ratio,
                                  double scale) {
  const char* directory = std::getenv("XLLM_DSV4_PREFILL_CORE_DUMP");
  if (directory == nullptr || *directory == '\0' ||
      query.device().index() != 7) {
    return;
  }
  // Caller is the uncaptured local-SP prefill path, never Graph replay.
  // Save the first invocation per layer; don't replace diagnostic evidence.
  const std::filesystem::path path =
      std::filesystem::path(directory) /
      ("layer_" + std::to_string(layer_id) + "_rank7.pt");
  if (std::filesystem::exists(path)) {
    return;
  }
  torch::serialize::OutputArchive archive;
  archive.write("q", query.to(torch::kCPU));
  archive.write("kv", kv.to(torch::kCPU));
  archive.write("output", output.to(torch::kCPU));
  archive.write("sink", sink.to(torch::kCPU));
  if (compressed.defined()) {
    archive.write("compressed", compressed.to(torch::kCPU));
  }
  if (indices.defined()) {
    archive.write("indices", indices.to(torch::kCPU));
  }
  archive.write("metadata", torch::tensor({query_start, ratio}, torch::kInt64));
  archive.write("softmax_scale", torch::tensor(scale, torch::kFloat64));
  archive.save_to(path.string());
  LOG(INFO) << "[DSV4_PREFILL_CORE_DUMP] layer=" << layer_id
            << " query_start=" << query_start << " ratio=" << ratio
            << " path=" << path;
}

const torch::Tensor* dsv4_musa_host_sequence_lengths(
    const DSAMetadata& attn_metadata) {
  const auto it =
      attn_metadata.cp_input_dict.find(kDsv4HostSequenceLengthsMetadataKey);
  if (it == attn_metadata.cp_input_dict.end()) {
    return nullptr;
  }
  const torch::Tensor& lengths = it->second;
  if (!lengths.defined() || !lengths.device().is_cpu() ||
      lengths.scalar_type() != torch::kInt32 || lengths.dim() != 2 ||
      lengths.size(0) != 2 || !lengths.is_contiguous()) {
    return nullptr;
  }
  return &lengths;
}

bool dsv4_musa_starts_new_request(const DSAMetadata& attn_metadata) {
  const torch::Tensor* lengths = dsv4_musa_host_sequence_lengths(attn_metadata);
  if (lengths == nullptr) {
    return false;
  }
  const int32_t* q_lengths = lengths->data_ptr<int32_t>();
  const int32_t* kv_lengths = q_lengths + lengths->size(1);
  for (int64_t index = 0; index < lengths->size(1); ++index) {
    if (q_lengths[index] != kv_lengths[index]) {
      return false;
    }
  }
  return lengths->size(1) > 0;
}

bool dsv4_musa_clear_state_cache_on_request_enabled() {
  const char* value = std::getenv("XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST");
  return value != nullptr && std::string(value) == "1";
}

bool dsv4_musa_clear_decode_workspaces_on_request_enabled() {
  const char* value =
      std::getenv("XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST");
  return value != nullptr && std::string(value) == "1";
}

bool dsv4_musa_graph_batched_mate_verify_enabled() {
  const char* value = std::getenv("XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY");
  return value != nullptr && std::string(value) == "1";
}

QuantArgs dsv4_musa_quant_args(const QuantArgs& quant_args) {
  QuantArgs result = quant_args;
  // DeepSeek-V4 stores block scales under `scale`; keep its projections on
  // the validated private FP8 path instead of the Qwen `weight_scale_inv`
  // path selected by the shared block-size marker.
  result.weight_block_size().clear();
  return result;
}

torch::Tensor dsv4_musa_block_tensor(const KVCache& kv_cache,
                                     BlockType block_type,
                                     KVCacheTensorRole::Value role) {
  const auto tensors = kv_cache.get_block_type_tensors(block_type);
  const auto it = tensors.find(role);
  return it == tensors.end() ? torch::Tensor() : it->second;
}

torch::Tensor dsv4_musa_indexer_scale_cache(const KVCache& kv_cache) {
  const auto scale = kv_cache.get_indexer_cache_scale();
  return scale.has_value() ? scale.value() : torch::Tensor();
}

std::vector<int64_t> tensor_to_i64_vector(const torch::Tensor& tensor) {
  std::vector<int64_t> values;
  if (!tensor.defined() || tensor.numel() == 0) {
    return values;
  }
  CHECK(tensor.device().is_cpu() || !kernel::musa::is_stream_capturing())
      << "Graph execution requires host lengths or persistent device metadata";
  auto cpu = tensor.to(torch::kCPU).to(torch::kLong).contiguous().view({-1});
  values.reserve(static_cast<size_t>(cpu.numel()));
  for (int64_t i = 0; i < cpu.numel(); ++i) {
    values.push_back(cpu[i].item<int64_t>());
  }
  return values;
}

int64_t effective_compress_ratio(int64_t ratio) {
  return ratio <= 1 ? 1 : ratio;
}

bool is_fp8_weight(const torch::Tensor& tensor) {
  return tensor.defined() && (tensor.scalar_type() == torch::kFloat8_e4m3fn ||
                              tensor.scalar_type() == torch::kFloat8_e5m2);
}

torch::Tensor dequantize_fp8_projection_weight(const torch::Tensor& weight,
                                               const torch::Tensor& scale,
                                               at::ScalarType output_dtype) {
  if (!is_fp8_weight(weight)) {
    return weight;
  }
  CHECK(scale.defined())
      << "MUSA FP8 projection requires a loaded weight scale.";
  // MCC cannot copy an E8M0 scale tensor to MUSA directly. Convert the
  // safetensors FP8 block scale to float32 on the host first, then move it to
  // the projection device once during model loading.
  auto scale_fp32 =
      scale.to(torch::kCPU).to(torch::kFloat32).to(weight.device());
  torch::Tensor expanded_scale;
  if (scale_fp32.numel() == 1) {
    expanded_scale = scale_fp32.reshape({1, 1});
  } else if (scale_fp32.dim() == 2) {
    const int64_t block_m =
        (weight.size(0) + scale_fp32.size(0) - 1) / scale_fp32.size(0);
    const int64_t block_k =
        (weight.size(1) + scale_fp32.size(1) - 1) / scale_fp32.size(1);
    expanded_scale = scale_fp32.unsqueeze(1)
                         .unsqueeze(3)
                         .repeat({1, block_m, 1, block_k})
                         .reshape({scale_fp32.size(0) * block_m,
                                   scale_fp32.size(1) * block_k})
                         .slice(0, 0, weight.size(0))
                         .slice(1, 0, weight.size(1));
  } else if (scale_fp32.dim() == 1 && scale_fp32.size(0) == weight.size(0)) {
    expanded_scale = scale_fp32.reshape({-1, 1});
  } else if (scale_fp32.dim() == 1 && scale_fp32.size(0) == weight.size(1)) {
    expanded_scale = scale_fp32.reshape({1, -1});
  } else {
    LOG(FATAL) << "Unsupported MUSA FP8 projection scale shape "
               << scale_fp32.sizes() << " for weight " << weight.sizes();
  }
  return (weight.to(torch::kFloat32) * expanded_scale)
      .to(output_dtype)
      .contiguous();
}

bool dsv4_musa_decode_q_kv_overlap_enabled() {
  const char* env = std::getenv("XLLM_DSV4_DISABLE_QKV_OVERLAP");
  return !(env != nullptr && std::string(env) == "1");
}

struct DecodeQKvOverlapState {
  explicit DecodeQKvOverlapState(int32_t device_id)
      : q_stream(
            c10::musa::getStreamFromPool(/*isHighPriority=*/false, device_id)) {
    CHECK_EQ(musaEventCreateWithFlags(&input_ready, musaEventDisableTiming),
             musaSuccess);
    CHECK_EQ(musaEventCreateWithFlags(&q_ready, musaEventDisableTiming),
             musaSuccess);
  }

  c10::musa::MUSAStream q_stream;
  musaEvent_t input_ready = nullptr;
  musaEvent_t q_ready = nullptr;
};

DecodeQKvOverlapState* get_decode_q_kv_overlap_state(int32_t device_id) {
  // One worker thread owns each device. Keep process-lifetime events alive so
  // shutdown cannot destroy them after the MUSA context has already exited.
  using DecodeQKvOverlapStates =
      std::unordered_map<int32_t, std::unique_ptr<DecodeQKvOverlapState>>;
  thread_local DecodeQKvOverlapStates* states = new DecodeQKvOverlapStates();
  auto& state = (*states)[device_id];
  if (!state) {
    state = std::make_unique<DecodeQKvOverlapState>(device_id);
  }
  return state.get();
}

bool dsv4_musa_trace_enabled() {
  const char* env = std::getenv("XLLM_DSV4_MUSA_TRACE");
  return env != nullptr && std::string(env) == "1";
}
bool dsv4_musa_graph_capture_warmup() {
  return ::xllm::kernel::musa::deepseek_v4_graph_capture_warmup();
}
bool dsv4_musa_cache_trace_enabled() { return false; }
bool dsv4_musa_mate_fused_dsa_trace_enabled() {
  const char* value = std::getenv("XLLM_DSV4_MATE_ROUTE_TRACE");
  return value != nullptr && std::string(value) == "1";
}
bool dsv4_musa_mate_fused_dsa_verify_enabled() {
  const char* env = std::getenv("XLLM_DSV4_MATE_VERIFY");
  return env != nullptr && std::string(env) == "1";
}
bool dsv4_musa_mate_prefill_trace_enabled() { return false; }
bool dsv4_musa_sync_trace_enabled() { return false; }

struct PrefillProfileState {};

template <typename... Args>
PrefillProfileState* begin_prefill_profile(Args&&...) {
  return nullptr;
}

template <typename... Args>
void record_prefill_profile_boundary(Args&&...) {}

template <typename... Args>
void finish_prefill_profile(Args&&...) {}

bool dsv4_musa_compress_fallback_enabled() { return true; }
bool dsv4_musa_fused_compress_pool_enabled() { return true; }
bool dsv4_musa_compress_pool_sync_enabled() { return false; }
bool dsv4_musa_indexer_enabled() { return true; }
bool dsv4_musa_paged_indexer_cache_enabled() { return true; }
bool dsv4_musa_persistent_compressor_state_enabled() { return true; }
bool dsv4_musa_persistent_indexer_state_enabled() { return true; }
bool dsv4_musa_indexer_query_int8_enabled() { return false; }
bool dsv4_musa_multi_sequence_indexer_prefill_enabled() { return true; }
bool dsv4_musa_fused_indexer_topk_enabled() { return true; }
bool dsv4_musa_c128_mate_enabled() {
  const char* value = std::getenv("XLLM_DSV4_C128_MATE_VERIFY");
  return value != nullptr && std::string(value) == "1";
}
bool dsv4_musa_batched_graph_indexer_enabled() {
  const char* value = std::getenv("XLLM_DSV4_GRAPH_BATCHED_INDEXER");
  return value != nullptr && std::string(value) == "1";
}
bool dsv4_musa_mate_fused_dsa_multi_sequence_enabled() {
  const char* projection_value =
      std::getenv("XLLM_DSV4_BATCH_ATTENTION_PROJECTIONS");
  const char* mate_value = std::getenv("XLLM_DSV4_BATCH_MATE_ACROSS_SEQUENCES");
  return (projection_value != nullptr &&
          std::string(projection_value) == "1") ||
         (mate_value != nullptr && std::string(mate_value) == "1");
}

size_t dsv4_musa_mate_max_sequences_per_launch() {
  const char* value = std::getenv("XLLM_DSV4_MATE_MAX_SEQUENCES_PER_LAUNCH");
  if (value == nullptr || *value == '\0') {
    return 1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 1 || parsed > 16) {
    return 1;
  }
  return static_cast<size_t>(parsed);
}
bool dsv4_musa_mate_sparse_decode_enabled() {
  // Diagnostic escape hatch for the q_len=1 DSV4 path. Keeping this enabled
  // preserves the performance route; disabling it falls through to the
  // reference composite attention while retaining compressed-history rows.
  const char* env = std::getenv("XLLM_DSV4_DISABLE_MATE_SPARSE_DECODE");
  return !(env != nullptr && std::string(env) == "1");
}
bool dsv4_musa_fused_dense_decode_enabled() {
  const char* env = std::getenv("XLLM_DSV4_DISABLE_FUSED_DENSE_DECODE");
  return !(env != nullptr && std::string(env) == "1");
}
bool dsv4_musa_fused_cache_gather_enabled() { return true; }
bool dsv4_musa_fused_cache_scatter_enabled() { return false; }
bool dsv4_musa_fused_partial_rope_enabled() { return true; }
int64_t dsv4_musa_fused_partial_rope_max_tokens() { return 4; }
bool dsv4_musa_paged_dense_decode_enabled() {
  // Keep the fused paged path as the default performance route, but allow a
  // controlled dense-fallback A/B for accuracy debugging. This is useful for
  // separating a q_len=1/paged-kernel issue from target-state or sampler
  // contamination without rebuilding the worker again.
  const char* env = std::getenv("XLLM_DSV4_DISABLE_PAGED_DENSE_DECODE");
  return !(env != nullptr && std::string(env) == "1");
}
bool dsv4_musa_paged_compressed_decode_enabled() { return true; }
bool dsv4_musa_split_ob_proj_enabled() { return true; }
bool dsv4_musa_output_proj_prefill_sync_enabled() { return false; }
bool dsv4_musa_output_proj_host_barrier_enabled() { return false; }

int64_t dsv4_musa_paged_dense_decode_max_total_len() { return 4096; }
int64_t dsv4_musa_attention_reduce_delay_us() { return 0; }
int64_t dsv4_musa_attention_sync_delay_us() { return 0; }
int64_t dsv4_musa_output_proj_chunk_tokens() { return 4096; }
int64_t dsv4_musa_prefill_full_kv_threshold() {
  return dsv4_full_kv_threshold();
}
int64_t dsv4_musa_prefill_attention_chunk_size() { return 32; }

template <typename... Args>
void maybe_sync_output_proj_prefill(Args&&...) {}

struct StageSyncConfig {
  bool enabled = false;
  bool all = false;
  std::vector<std::string> stages;
};
StageSyncConfig dsv4_musa_attention_stage_sync_config() { return {}; }
StageSyncConfig dsv4_musa_attention_stage_trace_config() { return {}; }
bool dsv4_musa_sync_attention_stage_selected(const char*) { return false; }
bool dsv4_musa_trace_attention_stage_selected(const char*) { return false; }
template <typename... Args>
void maybe_sync_attention_stage(int32_t layer_id,
                                const char* stage,
                                Args&&...) {
  static const bool enabled = []() {
    const char* setting = std::getenv("XLLM_DSV4_PCP_SYNC_TRACE");
    return setting != nullptr && std::string(setting) == "1";
  }();
  if (!enabled || kernel::musa::is_stream_capturing()) {
    return;
  }
  LOG(INFO) << "[DSV4_PCP_STAGE] enter layer=" << layer_id
            << " stage=" << stage;
  CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  LOG(INFO) << "[DSV4_PCP_STAGE] exit layer=" << layer_id << " stage=" << stage;
}

std::string i64_prefix_to_string(const std::vector<int64_t>& values,
                                 size_t limit = 16) {
  std::string out = "[";
  const size_t n = std::min(values.size(), limit);
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  if (values.size() > limit) {
    out += ",...";
  }
  out += "]";
  return out;
}

std::string tensor_i64_prefix(const torch::Tensor& tensor, size_t limit = 16) {
  if (kernel::musa::is_stream_capturing() && tensor.defined() &&
      !tensor.device().is_cpu()) {
    return "[capture-suppressed]";
  }
  return i64_prefix_to_string(tensor_to_i64_vector(tensor), limit);
}

std::string tensor_stat_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  if (tensor.numel() == 0) {
    return "empty";
  }
  // Statistics require reductions and scalar host reads.  They are diagnostic
  // only and must not execute while a MUSA graph is being captured.
  if (kernel::musa::is_stream_capturing()) {
    return "capture-suppressed";
  }
  auto fp32 = tensor.to(torch::kFloat32);
  const bool finite = torch::isfinite(fp32).all().item<bool>();
  const float absmax = fp32.abs().max().item<float>();
  const float mean = fp32.mean().item<float>();
  return "finite=" + std::string(finite ? "true" : "false") +
         ",absmax=" + std::to_string(absmax) + ",mean=" + std::to_string(mean);
}

std::string tensor_sizes_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << tensor.sizes();
  return oss.str();
}

std::pair<torch::Tensor, torch::Tensor> select_rope_tables(
    const DSAMetadata& attn_metadata,
    int64_t compress_ratio,
    int64_t token_count) {
  if (compress_ratio == 4 && attn_metadata.c4_input_cos.defined() &&
      attn_metadata.c4_input_sin.defined() &&
      attn_metadata.c4_input_cos.size(0) >= token_count &&
      attn_metadata.c4_input_sin.size(0) >= token_count) {
    return {attn_metadata.c4_input_cos, attn_metadata.c4_input_sin};
  }
  if (compress_ratio == 128 && attn_metadata.c128_input_cos.defined() &&
      attn_metadata.c128_input_sin.defined() &&
      attn_metadata.c128_input_cos.size(0) >= token_count &&
      attn_metadata.c128_input_sin.size(0) >= token_count) {
    return {attn_metadata.c128_input_cos, attn_metadata.c128_input_sin};
  }
  return {attn_metadata.cos, attn_metadata.sin};
}

torch::Tensor apply_interleaved_partial_rope(
    torch::Tensor input,
    int64_t rope_start_dim,
    int64_t rope_head_dim,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    bool inverse = false,
    torch::Tensor output_workspace = torch::Tensor(),
    bool allow_prefill_assembly = false,
    bool allow_main_assembly = false) {
  if (!input.defined() || !cos.defined() || !sin.defined() ||
      rope_head_dim <= 0 || rope_start_dim < 0 ||
      input.size(-1) < rope_start_dim + rope_head_dim) {
    return input;
  }

  CHECK(input.dim() == 2 || input.dim() == 3)
      << "apply_interleaved_partial_rope supports dim 2/3, got " << input.dim();
  CHECK(cos.dim() == 2 && sin.dim() == 2)
      << "DeepSeek-V4 MUSA RoPE expects cos/sin dim 2.";
  CHECK_EQ(cos.size(0), input.size(0));
  CHECK_EQ(sin.size(0), input.size(0));

  // Only dispatch the fused decode kernel when the caller supplies persistent
  // storage. This keeps whole-graph replay allocation-free and leaves every
  // existing prefill/Q/KV call on its original implementation.
  if (output_workspace.defined() && dsv4_musa_fused_partial_rope_enabled() &&
      input.size(0) <= dsv4_musa_fused_partial_rope_max_tokens() &&
      input.is_contiguous() && cos.is_contiguous() && sin.is_contiguous() &&
      input.device() == cos.device() && input.device() == sin.device() &&
      input.scalar_type() == cos.scalar_type() &&
      input.scalar_type() == sin.scalar_type()) {
    if (kernel::musa::deepseek_v4_partial_rope(input,
                                               cos,
                                               sin,
                                               rope_start_dim,
                                               rope_head_dim,
                                               inverse,
                                               output_workspace)) {
      return output_workspace;
    }
  }

  auto before = input.slice(/*dim=*/-1, /*start=*/0, /*end=*/rope_start_dim);
  auto rope = input.slice(
      /*dim=*/-1,
      /*start=*/rope_start_dim,
      /*end=*/rope_start_dim + rope_head_dim);
  auto after =
      input.slice(/*dim=*/-1, /*start=*/rope_start_dim + rope_head_dim);

  auto cos_view = cos.slice(/*dim=*/-1, /*start=*/0, /*end=*/rope_head_dim);
  auto sin_view = sin.slice(/*dim=*/-1, /*start=*/0, /*end=*/rope_head_dim);
  if (inverse) {
    sin_view = -sin_view;
  }
  if (input.dim() == 3) {
    cos_view = cos_view.unsqueeze(1);
    sin_view = sin_view.unsqueeze(1);
  }

  auto even = rope.slice(/*dim=*/-1,
                         /*start=*/0,
                         /*end=*/rope_head_dim,
                         /*step=*/2);
  auto odd = rope.slice(/*dim=*/-1,
                        /*start=*/1,
                        /*end=*/rope_head_dim,
                        /*step=*/2);
  auto cos_even = cos_view.slice(
      /*dim=*/-1, /*start=*/0, /*end=*/rope_head_dim, /*step=*/2);
  auto sin_even = sin_view.slice(
      /*dim=*/-1, /*start=*/0, /*end=*/rope_head_dim, /*step=*/2);

  auto rot_even = even * cos_even - odd * sin_even;
  auto rot_odd = odd * cos_even + even * sin_even;
  const char* assembly_setting =
      std::getenv("XLLM_DSV4_PREFILL_INDEXER_ROPE_COPY");
  if (allow_prefill_assembly && assembly_setting != nullptr &&
      std::string(assembly_setting) == "1" &&
      !kernel::musa::is_stream_capturing() &&
      dsv4_prefill_rope_assembly_eligible(
          input, rope_start_dim, rope_head_dim)) {
    LOG_FIRST_N(INFO, 2) << "[DSV4_PREFILL_INDEXER_ROPE_COPY] rows="
                         << input.size(0) << " copy_only=1";
    return dsv4_assemble_prefill_rope(input, rot_even, rot_odd);
  }
  const char* main_assembly_setting =
      std::getenv("XLLM_DSV4_PREFILL_MAIN_ROPE_COPY");
  if (allow_main_assembly && main_assembly_setting != nullptr &&
      std::string(main_assembly_setting) == "1") {
    LOG_FIRST_N(INFO, 2) << "[DSV4_MAIN_ROPE_ASSEMBLY_INPUT] shape="
                         << input.sizes() << " dtype=" << input.scalar_type()
                         << " contiguous=" << input.is_contiguous()
                         << " rope_start=" << rope_start_dim
                         << " rope_width=" << rope_head_dim;
  }
  if (allow_main_assembly && main_assembly_setting != nullptr &&
      std::string(main_assembly_setting) == "1" &&
      !kernel::musa::is_stream_capturing() &&
      dsv4_prefill_main_rope_assembly_eligible(
          input, rope_start_dim, rope_head_dim)) {
    LOG_FIRST_N(INFO, 2) << "[DSV4_PREFILL_MAIN_ROPE_COPY] rows="
                         << input.size(0) << " heads=" << input.size(1)
                         << " inverse=" << inverse << " copy_only=1";
    return dsv4_assemble_prefill_main_rope(input, rot_even, rot_odd);
  }
  auto rotated = torch::stack({rot_even, rot_odd}, /*dim=*/-1).flatten(-2);
  return torch::cat({before, rotated, after}, /*dim=*/-1).contiguous();
}

torch::Tensor rms_norm_last_dim(const torch::Tensor& input,
                                const torch::Tensor& weight,
                                double eps) {
  auto fp32 = input.to(torch::kFloat32);
  auto rsqrt =
      torch::rsqrt(fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true) + eps);
  return (fp32 * rsqrt * weight.to(torch::kFloat32)).to(input.dtype());
}

torch::Tensor create_normalized_hadamard(int64_t dim,
                                         const torch::TensorOptions& options) {
  CHECK_GT(dim, 0);
  CHECK_EQ(dim & (dim - 1), 0)
      << "DeepSeek-V4 indexer head dim must be a power of two.";
  auto cpu_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto matrix = torch::ones({1, 1}, cpu_options);
  for (int64_t size = 1; size < dim; size <<= 1) {
    auto top = torch::cat({matrix, matrix}, /*dim=*/1);
    auto bottom = torch::cat({matrix, -matrix}, /*dim=*/1);
    matrix = torch::cat({top, bottom}, /*dim=*/0);
  }
  matrix.mul_(1.0 / std::sqrt(static_cast<double>(dim)));
  return matrix.to(options.device()).contiguous();
}

torch::Tensor apply_indexer_hadamard(const torch::Tensor& input,
                                     const torch::Tensor& hadamard) {
  if (!input.defined() || input.numel() == 0) {
    return input;
  }
  CHECK(hadamard.defined());
  CHECK_EQ(input.size(-1), hadamard.size(0));
  auto output = torch::matmul(input.to(torch::kFloat32),
                              hadamard.to(torch::kFloat32).transpose(0, 1));
  return output.to(input.dtype()).contiguous();
}

torch::Tensor dynamic_int8_roundtrip_last_dim(const torch::Tensor& input) {
  if (!input.defined() || input.numel() == 0) {
    return input;
  }
  auto fp32 = input.to(torch::kFloat32);
  auto max_abs = fp32.abs().amax(/*dim=*/-1, /*keepdim=*/true);
  auto safe_max =
      torch::where(max_abs.gt(0), max_abs, torch::ones_like(max_abs));
  auto scale = safe_max * (1.0 / 127.0);
  auto quantized =
      torch::round(fp32 / scale).clamp(-128.0, 127.0).to(torch::kInt8);
  return (quantized.to(torch::kFloat32) * scale).to(input.dtype()).contiguous();
}

void fp8_act_quant_roundtrip_inplace(torch::Tensor& tensor,
                                     int64_t quant_dim,
                                     int64_t block_size) {
  // The FP8 checkpoint quantizes weights, not the BF16 KV activations.
  // Preserve the reference MUSA path's BF16 activations directly; an ATen
  // dtype roundtrip here also allocates tensors while a decode graph captures.
  (void)tensor;
  (void)quant_dim;
  (void)block_size;
}

void scatter_by_slot(torch::Tensor& cache,
                     const torch::Tensor& slot_mapping,
                     const torch::Tensor& value) {
  if (!cache.defined() || !slot_mapping.defined() || !value.defined() ||
      slot_mapping.numel() == 0 || value.numel() == 0) {
    return;
  }
  constexpr int64_t kFlashMlaRopeHeadDim = 64;
  const bool has_row_dims = value.dim() > 0 && cache.dim() > 0;
  const int64_t value_width = has_row_dims ? value.size(value.dim() - 1) : 0;
  const int64_t cache_width = has_row_dims ? cache.size(cache.dim() - 1) : 0;
  const bool flashmla_layout =
      has_row_dims && cache_width == value_width + kFlashMlaRopeHeadDim;
  if ((dsv4_musa_fused_cache_scatter_enabled() || flashmla_layout) &&
      has_row_dims && cache.is_contiguous() &&
      cache.scalar_type() == value.scalar_type() &&
      cache.device() == value.device() &&
      (slot_mapping.scalar_type() == torch::kInt32 ||
       slot_mapping.scalar_type() == torch::kInt64) &&
      cache.numel() % cache_width == 0) {
    ::xllm::kernel::DeepseekV4ScatterCacheParams params;
    params.cache = cache;
    params.slot_mapping = slot_mapping;
    params.value = value;
    ::xllm::kernel::deepseek_v4_scatter_cache(params);
    return;
  }
  ::xllm::kernel::ScatterBySlotParams params;
  params.cache = cache;
  params.slot_mapping = slot_mapping;
  params.value = value;
  ::xllm::kernel::scatter_by_slot(params);
}

void pack_mate_model1_cache(torch::Tensor& cache,
                            const torch::Tensor& slot_mapping,
                            const torch::Tensor& value) {
  if (!cache.defined() || !slot_mapping.defined() || !value.defined() ||
      cache.numel() == 0 || slot_mapping.numel() == 0 || value.numel() == 0) {
    return;
  }
  ::xllm::kernel::DeepseekV4PackFp8CacheParams params;
  params.cache = cache;
  params.slot_mapping = slot_mapping;
  params.value = value;
  ::xllm::kernel::deepseek_v4_pack_fp8_cache(params);
}

torch::Tensor dequantize_mate_model1_rows(
    const torch::Tensor& cache,
    const torch::Tensor& physical_indices) {
  constexpr int64_t kPayloadBytes = 576;
  constexpr int64_t kBytesPerToken = 584;
  constexpr int64_t kNopeDim = 448;
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kQuantGroup = 64;
  if (!cache.defined() || !physical_indices.defined() || cache.dim() != 4 ||
      cache.scalar_type() != torch::kUInt8 || cache.size(2) != 1 ||
      cache.size(3) != kBytesPerToken) {
    return torch::Tensor();
  }

  auto physical = physical_indices.reshape({-1}).to(torch::kLong);
  auto valid =
      physical.ge(0).logical_and(physical.lt(cache.size(0) * cache.size(1)));
  physical = torch::masked_select(physical, valid).contiguous();
  if (physical.numel() == 0) {
    return torch::empty({0, kHeadDim}, cache.options().dtype(torch::kFloat32));
  }

  const int64_t block_size = cache.size(1);
  const int64_t page_stride = cache.stride(0);
  auto pages = torch::floor_divide(physical, block_size);
  auto page_offsets = torch::remainder(physical, block_size);
  auto payload_base = pages * page_stride + page_offsets * kPayloadBytes;
  auto payload_offsets = torch::arange(
      kPayloadBytes,
      torch::TensorOptions().dtype(torch::kLong).device(cache.device()));
  auto payload_indices =
      (payload_base.unsqueeze(1) + payload_offsets.unsqueeze(0)).reshape({-1});
  auto storage = cache.as_strided({cache.size(0) * page_stride}, {1});
  auto payload = storage.index_select(/*dim=*/0, payload_indices)
                     .view({physical.numel(), kPayloadBytes});

  auto scale_base =
      pages * page_stride + block_size * kPayloadBytes + page_offsets * 8;
  auto scale_offsets = torch::arange(
      7, torch::TensorOptions().dtype(torch::kLong).device(cache.device()));
  auto scale_indices =
      (scale_base.unsqueeze(1) + scale_offsets.unsqueeze(0)).reshape({-1});
  auto scale_exponents = storage.index_select(/*dim=*/0, scale_indices)
                             .view({physical.numel(), 7})
                             .to(torch::kFloat32) -
                         127.0;
  auto scales =
      torch::pow(torch::full_like(scale_exponents, 2.0), scale_exponents)
          .repeat_interleave(kQuantGroup, /*dim=*/1);
  auto nope = payload.slice(/*dim=*/1, /*start=*/0, /*end=*/kNopeDim)
                  .view(torch::kFloat8_e4m3fn)
                  .to(torch::kFloat32) *
              scales;
  auto rope = payload
                  .slice(/*dim=*/1,
                         /*start=*/kNopeDim,
                         /*end=*/kPayloadBytes)
                  .view(torch::kBFloat16)
                  .to(torch::kFloat32);
  return torch::cat({nope, rope}, /*dim=*/1).contiguous();
}

torch::Tensor dequantize_mate_model1_logical_rows(
    const torch::Tensor& cache,
    const torch::Tensor& logical_indices,
    const torch::Tensor& block_table,
    int64_t sequence_index) {
  if (!cache.defined() || !logical_indices.defined() ||
      !block_table.defined()) {
    return torch::Tensor();
  }
  ::xllm::kernel::DeepseekV4LogicalToPhysicalTopkParams map;
  map.logical_indices = logical_indices.to(torch::kInt32).contiguous();
  map.block_table = block_table.to(torch::kInt32).contiguous();
  map.sequence_index = sequence_index;
  map.block_size = cache.size(1);
  auto physical = ::xllm::kernel::deepseek_v4_logical_to_physical_topk(map);
  return dequantize_mate_model1_rows(cache, physical);
}

torch::Tensor selected_kv_attention_reference(const torch::Tensor& q,
                                              const torch::Tensor& selected_kv,
                                              const torch::Tensor& attn_sink,
                                              double softmax_scale) {
  if (!q.defined() || !selected_kv.defined() || selected_kv.numel() == 0) {
    return torch::Tensor();
  }
  auto q_fp32 = q.permute({1, 0, 2}).contiguous().to(torch::kFloat32);
  auto kv_fp32 = selected_kv.to(torch::kFloat32);
  auto scores = torch::matmul(q_fp32, kv_fp32.transpose(0, 1).unsqueeze(0)) *
                softmax_scale;
  auto max_scores = std::get<0>(scores.max(/*dim=*/-1, /*keepdim=*/true));
  auto exp_scores = torch::exp(scores - max_scores);
  auto denom = exp_scores.sum(/*dim=*/-1, /*keepdim=*/true);
  if (attn_sink.defined()) {
    auto sink =
        attn_sink.to(q.device()).to(torch::kFloat32).view({q.size(1), 1, 1});
    denom = denom + torch::exp(sink - max_scores);
  }
  auto output = torch::matmul(exp_scores / denom, kv_fp32.unsqueeze(0));
  return output.to(q.dtype()).permute({1, 0, 2}).contiguous();
}

struct Dsv4MateDsaIndexWorkspace {
  c10::DeviceIndex device = -1;
  torch::Tensor physical_indices;
  torch::Tensor extra_physical_indices;
};

Dsv4MateDsaIndexWorkspace& dsv4_mate_dsa_index_workspace(
    const torch::Tensor& logical_indices) {
  constexpr int64_t kExtraTopk = 128;
  thread_local Dsv4MateDsaIndexWorkspace workspace;
  auto extra_sizes = logical_indices.sizes().vec();
  extra_sizes.back() = kExtraTopk;
  if (workspace.device != logical_indices.device().index() ||
      !workspace.physical_indices.defined() ||
      workspace.physical_indices.sizes() != logical_indices.sizes() ||
      !workspace.extra_physical_indices.defined() ||
      workspace.extra_physical_indices.sizes() !=
          torch::IntArrayRef(extra_sizes)) {
    workspace.device = logical_indices.device().index();
    workspace.physical_indices = torch::empty_like(logical_indices);
    workspace.extra_physical_indices =
        torch::empty(extra_sizes, logical_indices.options());
  }
  return workspace;
}

void quant_scatter_indexer_cache(torch::Tensor& cache,
                                 torch::Tensor& scale_cache,
                                 const torch::Tensor& slot_mapping,
                                 const torch::Tensor& value) {
  if (!cache.defined() || !scale_cache.defined() || !slot_mapping.defined() ||
      !value.defined() || cache.numel() == 0 || scale_cache.numel() == 0 ||
      slot_mapping.numel() == 0 || value.numel() == 0) {
    return;
  }
  ::xllm::kernel::DeepseekV4IndexerCacheQuantScatterParams params;
  params.cache = cache;
  params.scale_cache = scale_cache;
  params.slot_mapping = slot_mapping;
  params.value = value;
  ::xllm::kernel::deepseek_v4_indexer_cache_quant_scatter(params);
}

torch::Tensor gather_indexer_cache(const torch::Tensor& cache,
                                   const torch::Tensor& scale_cache,
                                   const torch::Tensor& block_table,
                                   int64_t sequence_index,
                                   int64_t index_len,
                                   torch::ScalarType output_dtype) {
  if (!cache.defined() || !scale_cache.defined() || !block_table.defined() ||
      index_len <= 0 || block_table.dim() != 2 || sequence_index < 0 ||
      sequence_index >= block_table.size(0)) {
    return torch::Tensor();
  }
  ::xllm::kernel::DeepseekV4IndexerCacheGatherParams params;
  params.cache = cache;
  params.scale_cache = scale_cache;
  params.block_table = block_table.to(cache.device()).contiguous();
  params.sequence_index = sequence_index;
  params.kv_len = index_len;
  params.output_dtype = output_dtype;
  return ::xllm::kernel::deepseek_v4_indexer_cache_gather(params);
}

int64_t find_swa_cache_index(const DSAMetadata& attn_metadata,
                             int32_t layer_id,
                             size_t ordinal = 0) {
  if (attn_metadata.caches_info == nullptr || layer_id < 0 ||
      layer_id >= static_cast<int32_t>(attn_metadata.caches_info->size())) {
    return -1;
  }
  const auto& layer_caches = (*attn_metadata.caches_info)[layer_id];
  size_t seen = 0;
  for (size_t cache_idx = 0; cache_idx < layer_caches.size(); ++cache_idx) {
    if (layer_caches[cache_idx].type == DSACacheType::SLIDING_WINDOW) {
      if (seen == ordinal) {
        return static_cast<int64_t>(cache_idx);
      }
      ++seen;
    }
  }
  return -1;
}

torch::Tensor compressor_state_slots(const torch::Tensor& block_table,
                                     int64_t sequence_index,
                                     int64_t state_rows,
                                     int64_t block_size,
                                     const torch::Device& device) {
  if (!block_table.defined() || block_table.dim() != 2 || sequence_index < 0 ||
      sequence_index >= block_table.size(0) || block_table.size(1) <= 0 ||
      state_rows <= 0 || block_size < state_rows) {
    return torch::Tensor();
  }
  auto block_row = block_table.select(/*dim=*/0, /*index=*/sequence_index)
                       .to(device)
                       .to(torch::kLong);
  auto columns =
      torch::arange(block_row.size(0),
                    torch::TensorOptions().device(device).dtype(torch::kLong));
  auto valid_columns =
      torch::where(block_row.ge(0), columns, torch::full_like(columns, -1));
  auto last_valid_column = std::get<0>(valid_columns.max(/*dim=*/0, false))
                               .clamp_min(0)
                               .reshape({1});
  auto block_id = block_row.index_select(/*dim=*/0, last_valid_column);
  auto offsets = torch::arange(
      state_rows, torch::TensorOptions().device(device).dtype(torch::kLong));
  return block_id * block_size + offsets;
}

torch::Tensor cached_state_slot_offsets(const torch::Device& device,
                                        int64_t state_rows) {
  static thread_local std::unordered_map<int64_t, torch::Tensor> offsets;
  const int64_t key = static_cast<int64_t>(device.index()) * 1024 + state_rows;
  auto& value = offsets[key];
  if (!value.defined() || value.device() != device ||
      value.numel() != state_rows) {
    value = torch::arange(
        state_rows, torch::TensorOptions().device(device).dtype(torch::kLong));
  }
  return value;
}

torch::Tensor compressor_state_slots_from_indices(
    const torch::Tensor& state_block_indices,
    int64_t sequence_index,
    int64_t state_blocks,
    int64_t state_rows,
    int64_t block_size,
    const torch::Device& device,
    const torch::Tensor& slot_workspace) {
  if (!state_block_indices.defined() || state_block_indices.dim() != 1 ||
      sequence_index < 0 || sequence_index >= state_block_indices.size(0) ||
      state_blocks <= 0 || state_rows <= 0 || block_size < state_rows) {
    return torch::Tensor();
  }
  auto block_id = state_block_indices
                      .slice(/*dim=*/0,
                             /*start=*/sequence_index,
                             /*end=*/sequence_index + 1)
                      .to(device, torch::kLong, /*non_blocking=*/true);
  auto state_offsets = cached_state_slot_offsets(device, state_rows);
  if (slot_workspace.defined() && slot_workspace.device() == device &&
      slot_workspace.scalar_type() == torch::kInt64 &&
      slot_workspace.dim() == 1 && slot_workspace.numel() >= state_rows) {
    auto slots = slot_workspace.narrow(/*dim=*/0, /*start=*/0, state_rows);
    slots.copy_(block_id, /*non_blocking=*/true);
    slots.clamp_(/*min=*/0, /*max=*/state_blocks - 1);
    slots.mul_(block_size);
    slots.add_(state_offsets);
    return slots;
  }
  return block_id.clamp(/*min=*/0, /*max=*/state_blocks - 1) * block_size +
         state_offsets;
}

int64_t find_token_cache_index(const DSAMetadata& attn_metadata,
                               int32_t layer_id,
                               int64_t ratio,
                               size_t ordinal = 0) {
  if (attn_metadata.caches_info == nullptr || layer_id < 0 ||
      layer_id >= static_cast<int32_t>(attn_metadata.caches_info->size())) {
    return -1;
  }
  const auto& layer_caches = (*attn_metadata.caches_info)[layer_id];
  size_t seen = 0;
  for (size_t cache_idx = 0; cache_idx < layer_caches.size(); ++cache_idx) {
    if (layer_caches[cache_idx].type == DSACacheType::TOKEN &&
        effective_compress_ratio(layer_caches[cache_idx].ratio) == ratio) {
      if (seen == ordinal) {
        return static_cast<int64_t>(cache_idx);
      }
      ++seen;
    }
  }
  return -1;
}

torch::Tensor layer_cache_tensor(
    const std::vector<std::vector<torch::Tensor>>& tensors,
    int32_t layer_id,
    int64_t preferred_cache_idx) {
  if (layer_id < 0 || layer_id >= static_cast<int32_t>(tensors.size())) {
    return torch::Tensor();
  }
  const auto& layer_tensors = tensors[layer_id];
  if (preferred_cache_idx >= 0 &&
      preferred_cache_idx < static_cast<int64_t>(layer_tensors.size()) &&
      layer_tensors[preferred_cache_idx].defined()) {
    return layer_tensors[preferred_cache_idx];
  }
  for (const auto& tensor : layer_tensors) {
    if (tensor.defined()) {
      return tensor;
    }
  }
  return torch::Tensor();
}

}  // namespace

DeepseekV4AttentionImpl::DeepseekV4AttentionImpl(const ModelContext& context,
                                                 int32_t layer_id)
    : DeepseekV4AttentionImpl(context.get_model_args(),
                              dsv4_musa_quant_args(context.get_quant_args()),
                              context.get_parallel_args(),
                              context.get_tensor_options(),
                              layer_id) {}

DeepseekV4AttentionImpl::DeepseekV4AttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    int32_t layer_id)
    : num_heads_(args.n_heads()),
      hidden_dim_(args.hidden_size()),
      head_dim_(args.head_dim()),
      q_lora_rank_(args.q_lora_rank()),
      o_lora_rank_(args.o_lora_rank()),
      o_groups_(args.o_groups()),
      rope_head_dim_(args.rope_head_dim()),
      window_size_(args.window_size()),
      dspark_block_size_(args.dspark_block_size()),
      softmax_scale_(std::pow(static_cast<double>(args.head_dim()), -0.5)),
      eps_(args.rms_norm_eps()),
      layer_id_(std::max<int32_t>(layer_id, 0)) {
  CHECK_GT(num_heads_, 0);
  CHECK_GT(head_dim_, 0);
  CHECK_GT(q_lora_rank_, 0);
  CHECK_GT(o_lora_rank_, 0);
  CHECK_GT(o_groups_, 0);

  const int64_t tp_size = parallel_args.tp_group_->world_size();
  tp_rank_ = parallel_args.tp_group_->rank();
  tp_size_ = std::max<int64_t>(tp_size, 1);
  decode_workspace_tokens_ =
      std::max<int64_t>(16, args.num_speculative_tokens() + 1);
  const char* multi_sequence_workspace_env =
      std::getenv("XLLM_DSV4_MULTI_SEQUENCE_WORKSPACE_TOKENS");
  if (multi_sequence_workspace_env != nullptr) {
    const int64_t requested_workspace =
        std::strtoll(multi_sequence_workspace_env, nullptr, 10);
    CHECK(requested_workspace == 64 || requested_workspace == 128 ||
          requested_workspace == 256)
        << "XLLM_DSV4_MULTI_SEQUENCE_WORKSPACE_TOKENS must be 64, 128 or 256";
    decode_workspace_tokens_ =
        std::max<int64_t>(decode_workspace_tokens_, requested_workspace);
  }
  CHECK_EQ(num_heads_ % tp_size_, 0)
      << "DeepSeek-V4 MUSA attention requires num_heads divisible by TP size.";
  CHECK_EQ(o_groups_ % tp_size_, 0)
      << "DeepSeek-V4 MUSA attention requires o_groups divisible by TP size.";
  n_local_heads_ = num_heads_ / tp_size_;
  n_local_groups_ = o_groups_ / tp_size_;
  nope_head_dim_ = head_dim_ - rope_head_dim_;
  CHECK_GE(nope_head_dim_, 0);
  if (layer_id_ < static_cast<int32_t>(args.compress_ratios().size())) {
    compress_ratio_ =
        effective_compress_ratio(args.compress_ratios()[layer_id_]);
  } else {
    compress_ratio_ = 1;
  }
  compressor_coff_ = compress_ratio_ == 4 ? 2 : 1;
  index_n_heads_ = args.index_n_heads();
  index_head_dim_ = args.index_head_dim();
  index_topk_ = args.index_topk();
  if (compress_ratio_ == 4) {
    CHECK_GT(index_n_heads_, 0);
    CHECK_GT(index_head_dim_, 0);
    CHECK_GT(index_topk_, 0);
    CHECK_GE(index_head_dim_, rope_head_dim_);
    indexer_score_scale_ =
        std::pow(static_cast<double>(index_head_dim_), -0.5) *
        std::pow(static_cast<double>(index_n_heads_), -0.5);
  }

  const int64_t hidden_size = args.hidden_size();
  q_a_proj_ = register_module(
      "q_a_proj",
      ReplicatedLinear(
          hidden_size, q_lora_rank_, /*bias=*/false, quant_args, options));
  q_layernorm_ =
      register_module("q_a_layernorm", RMSNorm(q_lora_rank_, eps_, options));
  decode_q_a_norm_output_ =
      torch::empty({decode_workspace_tokens_, q_lora_rank_}, options);
  q_b_proj_ = register_module("q_b_proj",
                              ColumnParallelLinear(q_lora_rank_,
                                                   num_heads_ * head_dim_,
                                                   false,
                                                   false,
                                                   quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  kv_proj_ = register_module(
      "kv_proj",
      ReplicatedLinear(
          hidden_size, head_dim_, /*bias=*/false, quant_args, options));
  kv_layernorm_ =
      register_module("kv_layernorm", RMSNorm(head_dim_, eps_, options));
  o_a_proj_ =
      register_module("o_a_proj",
                      ColumnParallelLinear(num_heads_ * head_dim_ / o_groups_,
                                           o_groups_ * o_lora_rank_,
                                           false,
                                           true,
                                           quant_args,
                                           parallel_args.tp_group_,
                                           options));
  o_b_proj_ = register_module("o_b_proj",
                              RowParallelLinear(o_groups_ * o_lora_rank_,
                                                hidden_size,
                                                false,
                                                true,
                                                /*reduce=*/false,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));
  decode_o_b_reduce_output_ =
      torch::empty({decode_workspace_tokens_, hidden_size}, options);
  decode_paged_attention_output_ =
      torch::empty({1, n_local_heads_, head_dim_}, options);
  decode_spec_attention_output_ = torch::empty(
      {decode_workspace_tokens_, n_local_heads_, head_dim_}, options);
  decode_inverse_rope_output_ = torch::empty(
      {decode_workspace_tokens_, n_local_heads_, head_dim_}, options);
  decode_kv_seq_lens_ =
      torch::empty({decode_workspace_tokens_}, options.dtype(torch::kInt32));
  if (index_topk_ > 0) {
    decode_spec_indices_output_ = torch::empty(
        {decode_workspace_tokens_, index_topk_}, options.dtype(torch::kInt32));
    decode_spec_mate_physical_indices_ = torch::empty(
        {decode_workspace_tokens_, index_topk_}, options.dtype(torch::kInt32));
    decode_spec_mate_extra_physical_indices_ = torch::empty(
        {decode_workspace_tokens_, 128}, options.dtype(torch::kInt32));
  }
  decode_compressor_kv_slots_ =
      torch::empty({/*max state rows=*/128}, options.dtype(torch::kInt64));
  decode_compressor_score_slots_ =
      torch::empty({/*max state rows=*/128}, options.dtype(torch::kInt64));
  decode_indexer_kv_slots_ =
      torch::empty({/*max state rows=*/128}, options.dtype(torch::kInt64));
  decode_indexer_score_slots_ =
      torch::empty({/*max state rows=*/128}, options.dtype(torch::kInt64));
  if (compress_ratio_ == 4 || compress_ratio_ == 128) {
    const int64_t state_rows = compressor_coff_ * compress_ratio_;
    const int64_t state_dim = compressor_coff_ * head_dim_;
    decode_compressor_kv_state_ =
        torch::empty({state_rows, state_dim}, options.dtype(torch::kFloat32));
    decode_compressor_score_state_ =
        torch::empty({state_rows, state_dim}, options.dtype(torch::kFloat32));
    int64_t projection_rows = decode_workspace_tokens_;
    const char* joint_decode_env =
        std::getenv("XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION");
    if (compress_ratio_ == 4 && joint_decode_env != nullptr &&
        std::string(joint_decode_env) == "1") {
      const int64_t packed_width = 2 * state_dim + 4 * index_head_dim_;
      projection_rows =
          (decode_workspace_tokens_ * packed_width + state_dim - 1) / state_dim;
    }
    decode_compressor_kv_projection_ =
        torch::empty({projection_rows, state_dim}, options);
    decode_compressor_score_projection_ =
        torch::empty({decode_workspace_tokens_, state_dim}, options);
    decode_compressor_output_ =
        torch::empty({decode_workspace_tokens_, head_dim_}, options);
    decode_compressor_norm_output_ =
        torch::empty({decode_workspace_tokens_, head_dim_}, options);
    decode_compressor_rope_output_ =
        torch::empty({decode_workspace_tokens_, head_dim_}, options);
  }
  if (compress_ratio_ == 4) {
    constexpr int64_t kIndexerStateRows = 2 * 4;
    const int64_t state_dim = 2 * index_head_dim_;
    decode_indexer_kv_state_ = torch::empty({kIndexerStateRows, state_dim},
                                            options.dtype(torch::kFloat32));
    decode_indexer_score_state_ = torch::empty({kIndexerStateRows, state_dim},
                                               options.dtype(torch::kFloat32));
    decode_indexer_kv_projection_ =
        torch::empty({decode_workspace_tokens_, state_dim}, options);
    decode_indexer_score_projection_ =
        torch::empty({decode_workspace_tokens_, state_dim}, options);
    decode_indexer_output_ =
        torch::empty({decode_workspace_tokens_, index_head_dim_}, options);
    decode_indexer_norm_output_ =
        torch::empty({decode_workspace_tokens_, index_head_dim_}, options);
    decode_indexer_rope_output_ =
        torch::empty({decode_workspace_tokens_, index_head_dim_}, options);
    decode_indexer_head_weight_projection_ =
        torch::empty({1, index_n_heads_}, options);
    decode_indexer_head_weights_ =
        torch::empty({1, index_n_heads_}, options.dtype(torch::kFloat32));
    decode_indexer_topk_output_ =
        torch::empty({1, index_topk_}, options.dtype(torch::kInt32));
  }
  if (n_local_groups_ == 1) {
    decode_wo_a_output_ =
        torch::empty({decode_workspace_tokens_, 1, o_lora_rank_}, options);
  }
  attn_sink_ = register_parameter(
      "attn_sink",
      torch::zeros({n_local_heads_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  if (compress_ratio_ == 4 || compress_ratio_ == 128) {
    const QuantArgs empty_quant_args;
    const int64_t compressor_out_dim = compressor_coff_ * head_dim_;
    compressor_wkv_ = register_module("compressor_wkv",
                                      ReplicatedLinear(hidden_size,
                                                       compressor_out_dim,
                                                       /*bias=*/false,
                                                       empty_quant_args,
                                                       options));
    compressor_wgate_ = register_module("compressor_wgate",
                                        ReplicatedLinear(hidden_size,
                                                         compressor_out_dim,
                                                         /*bias=*/false,
                                                         empty_quant_args,
                                                         options));
    compressor_norm_ =
        register_module("compressor_norm", RMSNorm(head_dim_, eps_, options));
    compressor_ape_ =
        register_parameter("compressor_ape",
                           torch::empty({compress_ratio_, compressor_out_dim},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
  }
  if (compress_ratio_ == 4) {
    const QuantArgs empty_quant_args;
    const int64_t indexer_compressor_out_dim = 2 * index_head_dim_;
    indexer_wq_b_ =
        register_module("indexer_wq_b",
                        ReplicatedLinear(q_lora_rank_,
                                         index_n_heads_ * index_head_dim_,
                                         /*bias=*/false,
                                         quant_args,
                                         options));
    indexer_weights_proj_ = register_module("indexer_weights_proj",
                                            ReplicatedLinear(hidden_size,
                                                             index_n_heads_,
                                                             /*bias=*/false,
                                                             empty_quant_args,
                                                             options));
    indexer_compressor_wkv_ =
        register_module("indexer_compressor_wkv",
                        ReplicatedLinear(hidden_size,
                                         indexer_compressor_out_dim,
                                         /*bias=*/false,
                                         empty_quant_args,
                                         options));
    indexer_compressor_wgate_ =
        register_module("indexer_compressor_wgate",
                        ReplicatedLinear(hidden_size,
                                         indexer_compressor_out_dim,
                                         /*bias=*/false,
                                         empty_quant_args,
                                         options));
    indexer_compressor_norm_ = register_module(
        "indexer_compressor_norm", RMSNorm(index_head_dim_, eps_, options));
    indexer_compressor_ape_ =
        register_parameter("indexer_compressor_ape",
                           torch::empty({4, indexer_compressor_out_dim},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    indexer_hadamard_ =
        register_buffer("indexer_hadamard",
                        create_normalized_hadamard(
                            index_head_dim_, options.dtype(torch::kFloat32)));
  }
  q_rms_gamma_ = register_buffer("q_rms_gamma",
                                 torch::ones({head_dim_},
                                             torch::TensorOptions()
                                                 .dtype(options.dtype())
                                                 .device(options.device())));
}

void DeepseekV4AttentionImpl::write_context_kv(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& cache_slots,
    KVCache& kv_cache) {
  CHECK_GT(dspark_block_size_, 0);
  CHECK(hidden_states.defined() && hidden_states.dim() == 2);
  CHECK_EQ(hidden_states.size(0), cache_slots.numel());
  if (dsv4_musa_trace_enabled()) {
    LOG_FIRST_N(INFO, 1) << "[DSV4_CONTEXT_KV] hidden="
                         << tensor_sizes_string(hidden_states)
                         << " slots=" << tensor_sizes_string(cache_slots)
                         << " slots_device=" << cache_slots.device()
                         << " layer_device=" << hidden_states.device();
  }
  torch::Tensor kv;
  if (qkv_a_weight_loaded_) {
    // Loading packs wq_a/wkv and releases the original linear modules.
    // Context injection uses the same materialized wkv rows as prefill.
    kernel::MatmulParams params;
    params.a = hidden_states;
    params.b = qkv_a_weight_.slice(
        /*dim=*/0, /*start=*/q_lora_rank_, /*end=*/q_lora_rank_ + head_dim_);
    params.bias = std::nullopt;
    kv = kernel::matmul(params);
  } else {
    CHECK(kv_proj_) << "DSpark context KV projection is not initialized.";
    kv = kv_proj_->forward(hidden_states);
  }
  kv = std::get<0>(kv_layernorm_->forward(kv)).view({-1, 1, head_dim_});
  kv = apply_interleaved_partial_rope(
      kv, nope_head_dim_, rope_head_dim_, cos, sin);
  fp8_act_quant_roundtrip_inplace(kv, nope_head_dim_, /*block_size=*/64);
  torch::Tensor swa_cache = kv_cache.get_swa_cache();
  CHECK(swa_cache.defined());
  scatter_by_slot(swa_cache, cache_slots, kv);
  torch::Tensor packed_swa_cache = dsv4_musa_block_tensor(
      kv_cache, BlockType::SWA, KVCacheTensorRole::VALUE);
  pack_mate_model1_cache(packed_swa_cache, cache_slots, kv);
}

std::vector<std::pair<int64_t, int64_t>>
DeepseekV4AttentionImpl::build_query_ranges(const DSAMetadata& attn_metadata,
                                            int64_t total_tokens) const {
  std::vector<std::pair<int64_t, int64_t>> ranges;
  // Compact MTP validation is one sequence containing current + proposal.
  // Preserve that shape even when the generic Graph carrier exposes one row
  // per token; otherwise the proposal is treated as another request.
  const bool graph_has_one_sequence_kv =
      attn_metadata.is_acl_graph &&
      ((attn_metadata.actual_seq_lengths_kv.defined() &&
        attn_metadata.actual_seq_lengths_kv.numel() == 1) ||
       (attn_metadata.seq_lens.defined() &&
        attn_metadata.seq_lens.numel() == 1));
  const bool eager_mtp_q2 = !attn_metadata.is_acl_graph && total_tokens == 2 &&
                            attn_metadata.num_speculative_tokens == 1 &&
                            attn_metadata.actual_seq_lengths_kv.defined() &&
                            attn_metadata.actual_seq_lengths_kv.numel() == 2;
  const bool compact_spec_verify =
      total_tokens > 1 &&
      ((attn_metadata.is_acl_graph && graph_has_one_sequence_kv &&
        ((attn_metadata.num_speculative_tokens > 0 &&
          total_tokens == attn_metadata.num_speculative_tokens + 1) ||
         attn_metadata.max_query_len == total_tokens || total_tokens == 2)) ||
       eager_mtp_q2);
  if (compact_spec_verify) {
    ranges.emplace_back(0, total_tokens);
    return ranges;
  }
  // Physical multi-sequence Graph verify is laid out as one contiguous
  // q-width segment per request.  Do not fall through to one range per token:
  // the cache/block tables still have one row per request, not one row per
  // draft/target token.  The old fallback produced seq_idx=16 for B16/q7.
  if (attn_metadata.is_acl_graph && attn_metadata.num_speculative_tokens > 0) {
    const int64_t query_width =
        static_cast<int64_t>(attn_metadata.num_speculative_tokens) + 1;
    if (query_width > 1 && total_tokens % query_width == 0) {
      const int64_t sequence_count = total_tokens / query_width;
      ranges.reserve(static_cast<size_t>(sequence_count));
      for (int64_t sequence = 0; sequence < sequence_count; ++sequence) {
        const int64_t start = sequence * query_width;
        ranges.emplace_back(start, start + query_width);
      }
      return ranges;
    }
  }
  // Graph decode metadata is deliberately kept on device. Reading the
  // cumulative lengths back to host here would issue tensor.to(kCPU) while
  // the MUSA stream is being captured. The graph executor pads decode to one
  // token per logical sequence, so use one stable range per token.
  if (attn_metadata.is_acl_graph) {
    ranges.reserve(static_cast<size_t>(std::max<int64_t>(total_tokens, 1)));
    for (int64_t token = 0; token < total_tokens; ++token) {
      ranges.emplace_back(token, token + 1);
    }
    if (ranges.empty()) {
      ranges.emplace_back(0, 0);
    }
    return ranges;
  }
  if (const torch::Tensor* host_lengths =
          dsv4_musa_host_sequence_lengths(attn_metadata)) {
    const int32_t* q_lens = host_lengths->data_ptr<int32_t>();
    const int64_t num_sequences = host_lengths->size(1);
    ranges.reserve(static_cast<size_t>(num_sequences));
    int64_t start = 0;
    for (int64_t i = 0; i < num_sequences; ++i) {
      const int64_t len = std::clamp<int64_t>(
          q_lens[i], 0, std::max<int64_t>(total_tokens - start, 0));
      ranges.emplace_back(start, start + len);
      start += len;
    }
    if (!ranges.empty()) {
      return ranges;
    }
  }
  auto cu = tensor_to_i64_vector(attn_metadata.actual_seq_lengths_query);
  if (cu.size() > 1 && cu.front() == 0) {
    ranges.reserve(cu.size() - 1);
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
      ranges.emplace_back(cu[i], cu[i + 1]);
    }
    return ranges;
  }

  auto lens = tensor_to_i64_vector(attn_metadata.seq_lens_q);
  if (lens.empty()) {
    lens = tensor_to_i64_vector(attn_metadata.q_seq_lens);
  }
  if (!lens.empty()) {
    int64_t start = 0;
    for (int64_t len : lens) {
      ranges.emplace_back(start, std::min<int64_t>(start + len, total_tokens));
      start += len;
    }
    return ranges;
  }

  ranges.emplace_back(0, total_tokens);
  return ranges;
}

std::vector<int64_t> DeepseekV4AttentionImpl::build_kv_lens(
    const DSAMetadata& attn_metadata,
    size_t num_sequences,
    const std::vector<std::pair<int64_t, int64_t>>& query_ranges) const {
  if (attn_metadata.is_acl_graph) {
    const char* legacy_graph_kv_env =
        std::getenv("XLLM_DSV4_LEGACY_GRAPH_KV_LEN");
    if (legacy_graph_kv_env != nullptr &&
        std::string(legacy_graph_kv_env) == "1") {
      const int64_t graph_kv_len =
          std::max<int64_t>(attn_metadata.max_seq_len, 1);
      return std::vector<int64_t>(num_sequences, graph_kv_len);
    }
    // Graph decode cannot perform device-to-host scalar copies during capture,
    // but the DSA builder intentionally keeps these sequence lengths on CPU.
    // Preserve per-sequence values instead of collapsing every row to the
    // batch maximum; the latter makes row reuse and multi-sequence replay read
    // another request's effective KV range.
    if (const torch::Tensor* host_lengths =
            dsv4_musa_host_sequence_lengths(attn_metadata)) {
      const int64_t available_sequences = host_lengths->size(1);
      const int32_t* host_data = host_lengths->data_ptr<int32_t>();
      const int32_t* host_kv_lens = host_data + available_sequences;
      const int64_t copy_count = std::min<int64_t>(
          static_cast<int64_t>(num_sequences), available_sequences);
      std::vector<int64_t> graph_kv_lens;
      graph_kv_lens.reserve(static_cast<size_t>(copy_count));
      for (int64_t i = 0; i < copy_count; ++i) {
        graph_kv_lens.push_back(std::max<int64_t>(host_kv_lens[i], 0));
      }
      if (!graph_kv_lens.empty()) {
        return graph_kv_lens;
      }
    }
    const auto cpu_lengths = [](const torch::Tensor& tensor) {
      return tensor.defined() && tensor.device().is_cpu() &&
                     tensor.scalar_type() == torch::kInt32 && tensor.dim() == 1
                 ? tensor_to_i64_vector(tensor)
                 : std::vector<int64_t>();
    };
    auto graph_kv_lens = cpu_lengths(attn_metadata.actual_seq_lengths_kv);
    if (graph_kv_lens.empty()) {
      graph_kv_lens = cpu_lengths(attn_metadata.seq_lens);
    }
    if (graph_kv_lens.size() > num_sequences) {
      graph_kv_lens.resize(num_sequences);
    }
    if (!graph_kv_lens.empty()) {
      while (graph_kv_lens.size() < num_sequences) {
        const size_t idx = graph_kv_lens.size();
        graph_kv_lens.push_back(query_ranges[idx].second -
                                query_ranges[idx].first);
      }
      return graph_kv_lens;
    }
    // Last-resort fallback for metadata built without CPU lengths.
    const int64_t graph_kv_len =
        std::max<int64_t>(attn_metadata.max_seq_len, 1);
    return std::vector<int64_t>(num_sequences, graph_kv_len);
  }
  std::vector<int64_t> kv_lens;
  if (const torch::Tensor* host_lengths =
          dsv4_musa_host_sequence_lengths(attn_metadata)) {
    const int64_t available_sequences = host_lengths->size(1);
    const int32_t* host_data = host_lengths->data_ptr<int32_t>();
    const int32_t* host_kv_lens = host_data + available_sequences;
    const int64_t copy_count = std::min<int64_t>(
        static_cast<int64_t>(num_sequences), available_sequences);
    kv_lens.reserve(static_cast<size_t>(copy_count));
    for (int64_t i = 0; i < copy_count; ++i) {
      kv_lens.push_back(std::max<int64_t>(host_kv_lens[i], 0));
    }
  }
  if (kv_lens.empty()) {
    kv_lens = tensor_to_i64_vector(attn_metadata.actual_seq_lengths_kv);
  }
  if (kv_lens.empty()) {
    kv_lens = tensor_to_i64_vector(attn_metadata.seq_lens);
  }
  if (kv_lens.size() > num_sequences) {
    kv_lens.resize(num_sequences);
  }
  while (kv_lens.size() < num_sequences) {
    const size_t idx = kv_lens.size();
    kv_lens.push_back(query_ranges[idx].second - query_ranges[idx].first);
  }
  return kv_lens;
}

torch::Tensor DeepseekV4AttentionImpl::gather_cache(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len) const {
  if (!cache.defined() || !block_table.defined() || kv_len <= 0) {
    return torch::Tensor();
  }
  if (block_table.dim() == 0 || sequence_index < 0 ||
      sequence_index >= block_table.size(0)) {
    return torch::Tensor();
  }
  CHECK_GE(cache.dim(), 3) << "DeepSeek-V4 MUSA cache expects block layout.";
  const int64_t block_size = cache.size(1);
  const int64_t num_blocks = cache.size(0);
  if (dsv4_musa_fused_cache_gather_enabled() && block_table.dim() == 2 &&
      (block_table.scalar_type() == torch::kInt32 ||
       block_table.scalar_type() == torch::kInt64) &&
      cache.is_contiguous()) {
    const int64_t gather_len =
        std::min<int64_t>(kv_len, block_table.size(1) * block_size);
    if (gather_len > 0) {
      ::xllm::kernel::DeepseekV4GatherCacheParams params;
      params.cache = cache;
      params.block_table = block_table.to(cache.device()).contiguous();
      params.sequence_index = sequence_index;
      params.kv_len = kv_len;
      auto gathered = ::xllm::kernel::deepseek_v4_gather_cache(params);
      if (gathered.defined() && gathered.size(-1) > head_dim_) {
        gathered = gathered
                       .slice(/*dim=*/-1,
                              /*start=*/0,
                              /*end=*/head_dim_)
                       .contiguous();
      }
      return gathered;
    }
  }
  auto row = block_table.index({sequence_index})
                 .to(torch::kCPU)
                 .to(torch::kLong)
                 .contiguous();
  std::vector<int64_t> flat_indices;
  flat_indices.reserve(static_cast<size_t>(kv_len));
  for (int64_t pos = 0; pos < kv_len; ++pos) {
    const int64_t table_idx = pos / block_size;
    if (table_idx >= row.numel()) {
      break;
    }
    int64_t block_id = row[table_idx].item<int64_t>();
    int64_t flat = block_id * block_size + (pos % block_size);
    if ((flat < 0 || flat >= num_blocks * block_size) && block_id > 0) {
      flat = (block_id - 1) * block_size + (pos % block_size);
    }
    if (flat >= 0 && flat < num_blocks * block_size) {
      flat_indices.push_back(flat);
    }
  }
  if (flat_indices.empty()) {
    return torch::Tensor();
  }
  auto idx = torch::tensor(
      flat_indices,
      torch::TensorOptions().dtype(torch::kLong).device(cache.device()));
  auto flat_cache = cache.view({-1, cache.size(cache.dim() - 1)});
  auto gathered = flat_cache.index_select(/*dim=*/0, idx);
  if (gathered.size(-1) > head_dim_) {
    gathered = gathered
                   .slice(/*dim=*/-1,
                          /*start=*/0,
                          /*end=*/head_dim_)
                   .contiguous();
  }
  return gathered;
}

torch::Tensor DeepseekV4AttentionImpl::paged_dense_attention(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_kv_len) const {
  if (!dsv4_musa_paged_dense_decode_enabled() ||
      !dsv4_musa_fused_dense_decode_enabled()) {
    return torch::Tensor();
  }
  if (!q.defined() || !cache.defined() || !block_table.defined() ||
      kv_len <= 0 || q.dim() != 3 || q.size(0) != 1 || block_table.dim() != 2 ||
      sequence_index < 0 || sequence_index >= block_table.size(0) ||
      cache.dim() < 3 || !cache.is_contiguous()) {
    return torch::Tensor();
  }
  constexpr int64_t kFlashMlaRopeHeadDim = 64;
  const int64_t cache_width = cache.size(cache.dim() - 1);
  const bool cache_width_compatible =
      cache_width == q.size(2) ||
      cache_width == q.size(2) + kFlashMlaRopeHeadDim;
  if (!cache_width_compatible || q.scalar_type() != cache.scalar_type()) {
    return torch::Tensor();
  }
  if (block_table.scalar_type() != torch::kInt32 &&
      block_table.scalar_type() != torch::kInt64) {
    return torch::Tensor();
  }

  const bool use_paged_compressed = compressed_kv_len > 0 &&
                                    compressed_cache.defined() &&
                                    compressed_block_table.defined();
  if (use_paged_compressed &&
      (compressed_cache.dim() < 3 || !compressed_cache.is_contiguous() ||
       (compressed_cache.size(compressed_cache.dim() - 1) != q.size(2) &&
        compressed_cache.size(compressed_cache.dim() - 1) !=
            q.size(2) + kFlashMlaRopeHeadDim) ||
       compressed_cache.scalar_type() != q.scalar_type() ||
       compressed_block_table.dim() != 2 || sequence_index < 0 ||
       sequence_index >= compressed_block_table.size(0) ||
       (compressed_block_table.scalar_type() != torch::kInt32 &&
        compressed_block_table.scalar_type() != torch::kInt64))) {
    return torch::Tensor();
  }

  torch::Tensor cmp_2d;
  if (compressed_kv.defined() && compressed_kv.numel() > 0) {
    if (compressed_kv.size(compressed_kv.dim() - 1) != q.size(2)) {
      return torch::Tensor();
    }
    cmp_2d = compressed_kv.reshape(
        {-1, compressed_kv.size(compressed_kv.dim() - 1)});
  }

  const int64_t max_total_len = dsv4_musa_paged_dense_decode_max_total_len();
  if (max_total_len > 0) {
    const int64_t effective_kv_len =
        std::min<int64_t>(kv_len, block_table.size(1) * cache.size(1));
    const int64_t kv_window_len =
        window_size_ > 0 ? std::min<int64_t>(effective_kv_len, window_size_)
                         : effective_kv_len;
    const int64_t compressed_len =
        (cmp_2d.defined() ? cmp_2d.size(0) : 0) +
        (use_paged_compressed ? compressed_kv_len : 0);
    const int64_t total_len = kv_window_len + compressed_len;
    if (total_len > max_total_len) {
      if (dsv4_musa_trace_enabled()) {
        LOG(INFO) << "[DSV4_MUSA_TRACE] paged_dense_attention_fallback "
                  << "total_len=" << total_len
                  << " max_total_len=" << max_total_len
                  << " kv_window_len=" << kv_window_len
                  << " compressed_len=" << compressed_len;
      }
      return torch::Tensor();
    }
  }

  ::xllm::kernel::DeepseekV4PagedAttentionDecodeParams params;
  params.q = q.contiguous();
  params.cache = cache;
  params.block_table = block_table;
  params.sequence_index = sequence_index;
  params.kv_len = kv_len;
  params.kv_seq_lens = kv_seq_lens;
  params.window_size = window_size_;
  params.compressed_kv =
      cmp_2d.defined() ? cmp_2d.to(q.dtype()).contiguous() : torch::Tensor();
  params.compressed_cache =
      use_paged_compressed ? compressed_cache : torch::Tensor();
  params.compressed_block_table =
      use_paged_compressed ? compressed_block_table : torch::Tensor();
  params.compressed_sequence_index = sequence_index;
  params.compressed_kv_len =
      use_paged_compressed ? compressed_kv_len : int64_t{0};
  params.attn_sink =
      attn_sink_.defined()
          ? attn_sink_.to(q.device()).to(torch::kFloat32).contiguous()
          : torch::Tensor();
  params.softmax_scale = softmax_scale_;
  if (decode_paged_attention_output_.defined() &&
      decode_paged_attention_output_.sizes() == q.sizes() &&
      decode_paged_attention_output_.scalar_type() == q.scalar_type() &&
      decode_paged_attention_output_.device() == q.device()) {
    params.output = decode_paged_attention_output_;
  }
  return ::xllm::kernel::deepseek_v4_paged_attention_decode(params);
}

torch::Tensor DeepseekV4AttentionImpl::mate_sparse_attention(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& logical_indices,
    const torch::Tensor& compressed_block_table,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& swa_block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const torch::Tensor& output,
    const torch::Tensor& precomputed_physical_indices,
    const torch::Tensor& precomputed_extra_physical_indices) const {
  if (!q.defined() || q.size(0) <= 0 || !packed_cache.defined() ||
      !extra_packed_cache.defined() || !logical_indices.defined() ||
      !compressed_block_table.defined() || !swa_block_table.defined()) {
    return torch::Tensor();
  }
  if (logical_indices.dim() != 2 || logical_indices.size(0) != q.size(0)) {
    return torch::Tensor();
  }
  const int64_t effective_kv_len = std::max<int64_t>(kv_len, 0);
  const int64_t extra_len =
      std::min<int64_t>({window_size_, int64_t{128}, effective_kv_len});
  torch::Tensor physical_indices = precomputed_physical_indices;
  torch::Tensor extra_physical_indices = precomputed_extra_physical_indices;
  const bool use_precomputed_indices =
      physical_indices.defined() && extra_physical_indices.defined() &&
      physical_indices.scalar_type() == torch::kInt32 &&
      extra_physical_indices.scalar_type() == torch::kInt32 &&
      physical_indices.sizes() == logical_indices.sizes() &&
      extra_physical_indices.sizes() ==
          torch::IntArrayRef({logical_indices.size(0), int64_t{128}});
  if (!use_precomputed_indices) {
    auto logical_i32 = logical_indices.to(torch::kInt32).contiguous();
    auto& index_workspace = dsv4_mate_dsa_index_workspace(logical_i32);
    ::xllm::kernel::DeepseekV4PrepareMateDsaIndicesParams index_params;
    index_params.logical_indices = logical_i32;
    index_params.compressed_block_table = compressed_block_table.contiguous();
    index_params.swa_block_table = swa_block_table.contiguous();
    index_params.sequence_index = sequence_index;
    index_params.kv_len = effective_kv_len;
    index_params.window_size = window_size_;
    index_params.block_size = packed_cache.size(1);
    index_params.physical_indices = index_workspace.physical_indices;
    index_params.extra_physical_indices =
        index_workspace.extra_physical_indices;
    std::tie(physical_indices, extra_physical_indices) =
        ::xllm::kernel::deepseek_v4_prepare_mate_dsa_indices(index_params);
  }

  ::xllm::kernel::DeepseekV4MateSparseAttentionParams params;
  params.q = q;
  params.packed_cache = packed_cache;
  params.physical_indices = physical_indices;
  params.topk_length = std::min<int64_t>(
      {logical_indices.size(-1), index_topk_, effective_kv_len / 4});
  params.extra_packed_cache = extra_packed_cache;
  params.extra_physical_indices = extra_physical_indices;
  params.extra_topk_length = extra_len;
  params.attn_sink =
      attn_sink_.defined()
          ? attn_sink_.to(q.device()).to(torch::kFloat32).contiguous()
          : torch::Tensor();
  params.softmax_scale = softmax_scale_;
  if (output.defined()) {
    CHECK_EQ(output.sizes(), q.sizes());
    params.output = output;
  }
  auto mate_output =
      ::xllm::kernel::deepseek_v4_mate_sparse_attention_decode(params);
  if (mate_output.defined() && dsv4_musa_mate_fused_dsa_verify_enabled()) {
    auto main_kv = dequantize_mate_model1_rows(packed_cache, physical_indices);
    auto extra_kv =
        dequantize_mate_model1_rows(extra_packed_cache, extra_physical_indices);
    std::vector<torch::Tensor> selected_parts;
    if (main_kv.defined() && main_kv.numel() > 0) {
      selected_parts.push_back(main_kv);
    }
    if (extra_kv.defined() && extra_kv.numel() > 0) {
      selected_parts.push_back(extra_kv);
    }
    if (!selected_parts.empty()) {
      auto selected_kv = selected_parts.size() == 1
                             ? selected_parts.front()
                             : torch::cat(selected_parts, /*dim=*/0);
      auto packed_reference = selected_kv_attention_reference(
          q, selected_kv, params.attn_sink, softmax_scale_);
      if (packed_reference.defined()) {
        auto diff = (mate_output.to(torch::kFloat32) -
                     packed_reference.to(torch::kFloat32))
                        .abs();
        LOG(INFO) << "[DSV4_MATE_DSA_PACKED_VERIFY] layer=" << layer_id_
                  << " main_rows=" << main_kv.size(0)
                  << " extra_rows=" << extra_kv.size(0)
                  << " max_abs=" << diff.max().item<float>()
                  << " mean_abs=" << diff.mean().item<float>();
      }
    }
  }
  return mate_output.defined() ? mate_output.contiguous() : torch::Tensor();
}

torch::Tensor DeepseekV4AttentionImpl::mate_sparse_attention_batched(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& logical_indices,
    const torch::Tensor& compressed_block_table,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& swa_block_table,
    int64_t sequence_index,
    int64_t scalar_kv_len,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& output,
    const torch::Tensor& physical_indices,
    const torch::Tensor& extra_physical_indices) {
  if (!q.defined() || q.size(0) <= 1 || !packed_cache.defined() ||
      !extra_packed_cache.defined() || !logical_indices.defined() ||
      !compressed_block_table.defined() || !swa_block_table.defined() ||
      !kv_seq_lens.defined()) {
    return torch::Tensor();
  }
  if (logical_indices.dim() != 2 || logical_indices.size(0) != q.size(0) ||
      kv_seq_lens.dim() != 1 || kv_seq_lens.size(0) != q.size(0) ||
      kv_seq_lens.scalar_type() != torch::kInt32 ||
      kv_seq_lens.device() != q.device() || !kv_seq_lens.is_contiguous()) {
    return torch::Tensor();
  }
  if (!physical_indices.defined() || !extra_physical_indices.defined() ||
      physical_indices.sizes() != logical_indices.sizes() ||
      extra_physical_indices.sizes() !=
          torch::IntArrayRef({q.size(0), int64_t{128}})) {
    return torch::Tensor();
  }
  const char* static_lengths_env =
      std::getenv("XLLM_DSV4_MATE_BATCHED_STATIC_LENGTHS");
  if (static_lengths_env != nullptr && std::string(static_lengths_env) == "1" &&
      scalar_kv_len > 0) {
    // The physical-index preparation already pads each row with -1 beyond
    // its causal prefix. Reuse the legacy scalar MATE metadata with the
    // maximum row length; this avoids rebuilding TileLang metadata for every
    // q6 replay while retaining per-row masking through those sentinels.
    return mate_sparse_attention(q,
                                 packed_cache,
                                 logical_indices,
                                 compressed_block_table,
                                 extra_packed_cache,
                                 swa_block_table,
                                 sequence_index,
                                 scalar_kv_len,
                                 output,
                                 physical_indices,
                                 extra_physical_indices);
  }
  // Reuse the first two columns of the persistent logical-index buffer after
  // physical indices have been materialized. This keeps the Attention object
  // ABI unchanged (the candidate build intentionally relinks only selected
  // translation units) and avoids a graph-time allocation for length vectors.
  if (!decode_spec_indices_output_.defined() ||
      decode_spec_indices_output_.dim() != 2 ||
      decode_spec_indices_output_.size(0) < q.size(0) ||
      decode_spec_indices_output_.numel() < 2 * q.size(0)) {
    return torch::Tensor();
  }

  const int64_t q_tokens = q.size(0);
  const char* partitioned_c128_env =
      std::getenv("XLLM_DSV4_PARTITIONED_C128_MATE");
  const bool partitioned_c128 = compress_ratio_ == 128 && q_tokens > 7 &&
                                q_tokens % 7 == 0 &&
                                partitioned_c128_env != nullptr &&
                                std::string(partitioned_c128_env) == "1";
  const char* share_metadata_env = std::getenv("XLLM_DSV4_MATE_SHARE_METADATA");
  const char* grouped_metadata_env =
      std::getenv("XLLM_DSV4_MATE_GROUPED_METADATA");
  const char* batch_grouped_metadata_env =
      std::getenv("XLLM_DSV4_BATCH_MATE_GROUPED_METADATA");
  const bool batch_grouped_metadata =
      batch_grouped_metadata_env != nullptr &&
      std::string(batch_grouped_metadata_env) == "1" &&
      dsv4_musa_mate_fused_dsa_multi_sequence_enabled();
  const bool grouped_metadata =
      grouped_metadata_env != nullptr &&
      std::string(grouped_metadata_env) == "1" &&
      (!dsv4_sequence_view_active || batch_grouped_metadata);
  const bool share_metadata =
      (!dsv4_sequence_view_active || batch_grouped_metadata) &&
      (grouped_metadata || (share_metadata_env != nullptr &&
                            std::string(share_metadata_env) == "1"));
  const int64_t metadata_key = share_metadata
                                   ? ((scalar_kv_len << 12) ^ q_tokens)
                                   : std::numeric_limits<int64_t>::min();
  ::xllm::kernel::musa::deepseek_v4_mate_sparse_attention_set_metadata_key(
      metadata_key);
  kernel::musa::deepseek_v4_mate_sparse_attention_set_metadata_group(
      grouped_metadata ? compress_ratio_ : 0);
  auto length_workspace = decode_spec_indices_output_.view({-1});
  auto topk_lengths = length_workspace.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/q_tokens);
  auto extra_topk_lengths = length_workspace.narrow(
      /*dim=*/0, /*start=*/q_tokens, /*length=*/q_tokens);
  // `Tensor::div_(int)` promotes the integer destination through a floating
  // TensorIterator on TorchMUSA and is illegal during Graph capture. Keep the
  // length carrier int32 and use floor_divide, which has an integer kernel.
  kernel::musa::deepseek_v4_mate_causal_lengths(
      kv_seq_lens,
      compress_ratio_,
      index_topk_,
      std::min<int64_t>(window_size_, int64_t{128}),
      topk_lengths,
      extra_topk_lengths);

  const auto sink =
      attn_sink_.defined()
          ? attn_sink_.to(q.device()).to(torch::kFloat32).contiguous()
          : torch::Tensor();
  const auto destination =
      output.defined() ? std::optional<torch::Tensor>(output) : std::nullopt;
  auto mate_output =
      partitioned_c128
          ? kernel::musa::deepseek_v4_mate_sparse_attention_decode_partitioned(
                q,
                packed_cache,
                physical_indices,
                topk_lengths,
                extra_packed_cache,
                extra_physical_indices,
                extra_topk_lengths,
                sink,
                softmax_scale_,
                /*query_width=*/7,
                destination)
          : kernel::musa::deepseek_v4_mate_sparse_attention_decode_batched(
                q,
                packed_cache,
                physical_indices,
                topk_lengths,
                extra_packed_cache,
                extra_physical_indices,
                extra_topk_lengths,
                sink,
                softmax_scale_,
                destination);
  return mate_output.defined() ? mate_output.contiguous() : torch::Tensor();
}

torch::Tensor DeepseekV4AttentionImpl::dense_attention(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_indices,
    bool causal,
    int64_t q_abs_start,
    bool use_full_kv) const {
  if (!kv.defined() || kv.numel() == 0) {
    return torch::zeros_like(q);
  }
  const char* native_prefill_env =
      std::getenv("XLLM_DSV4_MULTI_PREFILL_LOCAL_MATE");
  const bool native_compressed_prefill =
      native_prefill_env != nullptr && std::string(native_prefill_env) == "1" &&
      sequence_parallel_context_ == nullptr && causal && use_full_kv &&
      (compress_ratio_ == 4 || compress_ratio_ == 128) &&
      compressed_kv.defined() && compressed_kv.numel() > 0 && q.dim() == 3 &&
      q.size(0) >= 128 && !kernel::musa::is_stream_capturing();
  if (native_compressed_prefill && q.size(0) > 1024) {
    std::vector<torch::Tensor> chunks;
    for (int64_t start = 0; start < q.size(0); start += 1024) {
      const int64_t count = std::min<int64_t>(1024, q.size(0) - start);
      auto indices = compressed_indices.defined()
                         ? compressed_indices.narrow(0, start, count)
                         : torch::Tensor();
      chunks.push_back(dense_attention(q.narrow(0, start, count),
                                       kv,
                                       compressed_kv,
                                       indices,
                                       true,
                                       q_abs_start + start,
                                       true));
    }
    return torch::cat(chunks, 0);
  }
  if (dsv4_musa_mate_prefill_trace_enabled() && causal && use_full_kv) {
    LOG(INFO) << "[DSV4_MATE_PREFILL_ROUTE] layer=" << layer_id_
              << " q=" << tensor_sizes_string(q)
              << " kv=" << tensor_sizes_string(kv)
              << " compressed=" << tensor_sizes_string(compressed_kv)
              << " indices_defined=" << compressed_indices.defined();
  }
  // Sequence-parallel prefill owns only this rank's query rows while KV is
  // global. Mate's local sparse-prefill ABI is fixed at 64 heads, so pad a
  // narrower query/sink and crop the output back. Unsupported shapes or a
  // missing external artifact return an undefined tensor and retain the
  // established composite fallback below.
  if ((sequence_parallel_context_ != nullptr || native_compressed_prefill) &&
      causal && use_full_kv && q.dim() == 3 && q.size(1) > 0 &&
      q.size(1) <= 64 && q.size(2) == 512 && kv.dim() == 2 &&
      kv.size(1) == 512 && q_abs_start >= 0 &&
      q_abs_start + q.size(0) <= kv.size(0) &&
      (native_compressed_prefill || q.size(0) != kv.size(0) ||
       q_abs_start != 0) &&
      (compress_ratio_ == 1 || compress_ratio_ == 4 ||
       compress_ratio_ == 128)) {
    constexpr int64_t kMateLocalHeads = 64;
    auto q_local = q.contiguous();
    auto q_padded = q_local;
    if (q_local.size(1) != kMateLocalHeads) {
      q_padded =
          torch::zeros({q_local.size(0), kMateLocalHeads, q_local.size(2)},
                       q_local.options());
      q_padded.slice(/*dim=*/1, 0, q_local.size(1)).copy_(q_local);
    }

    const bool has_compressed_kv =
        compressed_kv.defined() && compressed_kv.numel() > 0;
    const int64_t local_compress_ratio =
        has_compressed_kv ? compress_ratio_ : 1;
    auto local_compressed_kv =
        has_compressed_kv ? compressed_kv.contiguous() : torch::Tensor();
    auto local_compressed_indices =
        local_compress_ratio == 4 && compressed_indices.defined()
            ? (compressed_indices.scalar_type() == torch::kInt32
                   ? compressed_indices.contiguous()
                   : compressed_indices.to(torch::kInt32).contiguous())
            : torch::Tensor();
    auto local_attn_sink = torch::full({kMateLocalHeads},
                                       -std::numeric_limits<float>::infinity(),
                                       q.options().dtype(torch::kFloat32));
    if (attn_sink_.defined()) {
      local_attn_sink.slice(/*dim=*/0, 0, q_local.size(1))
          .copy_(attn_sink_.to(q.device()).to(torch::kFloat32).contiguous());
    }

    auto local_sparse_prefill =
        kernel::musa::deepseek_v4_mate_local_sparse_attention_prefill(
            q_padded,
            kv.contiguous(),
            local_compressed_kv,
            local_compressed_indices,
            local_attn_sink,
            q_abs_start,
            local_compress_ratio,
            softmax_scale_,
            native_compressed_prefill
                ? std::optional<torch::Tensor>(torch::empty_like(q_padded))
                : std::nullopt);
    if (local_sparse_prefill.defined()) {
      if (native_compressed_prefill) {
        LOG_FIRST_N(INFO, 2)
            << "[DSV4_NATIVE_COMPRESSED_PREFILL] layer=" << layer_id_
            << " q_tokens=" << q.size(0) << " q_abs_start=" << q_abs_start
            << " ratio=" << compress_ratio_;
      }
      dump_local_prefill_attention(layer_id_,
                                   q_padded,
                                   kv,
                                   local_compressed_kv,
                                   local_compressed_indices,
                                   local_attn_sink,
                                   local_sparse_prefill,
                                   q_abs_start,
                                   local_compress_ratio,
                                   softmax_scale_);
      if (q_local.size(1) != kMateLocalHeads) {
        local_sparse_prefill =
            local_sparse_prefill.slice(/*dim=*/1, 0, q_local.size(1))
                .contiguous();
      }
      maybe_sync_attention_stage(
          layer_id_, "mate_local_sparse_prefill", local_sparse_prefill);
      LOG_FIRST_N(INFO, 1) << "[DSV4_MATE_LOCAL_PREFILL_PADDED] layer="
                           << layer_id_ << " q_tokens=" << q_local.size(0)
                           << " local_heads=" << q_local.size(1)
                           << " q_abs_start=" << q_abs_start
                           << " ratio=" << compress_ratio_;
      return local_sparse_prefill;
    }
  }
  // C4 uses the per-token compressed TopK plus a contiguous SWA tail. The
  // MUSA bridge merges both index streams and runs one Mate softmax.
  if (causal && use_full_kv && compress_ratio_ == 4 &&
      compressed_indices.defined() && compressed_kv.defined() &&
      compressed_kv.numel() > 0 && q.dim() == 3 && kv.dim() == 2 &&
      compressed_kv.dim() == 2 && compressed_indices.dim() == 2) {
    ::xllm::kernel::DeepseekV4MateC4SparsePrefillParams params;
    params.q = q.contiguous();
    params.kv = kv.contiguous();
    params.compressed_kv = compressed_kv.contiguous();
    params.compressed_indices =
        compressed_indices.to(torch::kInt32).contiguous();
    params.attn_sink =
        attn_sink_.defined()
            ? attn_sink_.to(q.device()).to(torch::kFloat32).contiguous()
            : torch::Tensor();
    params.softmax_scale = softmax_scale_;
    // Mate's prefill workspace is reused by consecutive sequences. Return
    // into sequence-owned storage so outputs retained by the caller cannot be
    // overwritten by the next FFI invocation.
    params.output = torch::empty_like(q);
    auto fused_prefill =
        ::xllm::kernel::deepseek_v4_mate_c4_sparse_attention_prefill(params);
    if (fused_prefill.defined()) {
      maybe_sync_attention_stage(layer_id_, "mate_c4_prefill", fused_prefill);
      return fused_prefill;
    }
  }
  // C128 prefill can use Mate's causal sequence-pack kernel directly.
  if (causal && use_full_kv && !compressed_indices.defined() &&
      compressed_kv.defined() && compressed_kv.numel() > 0 && q.dim() == 3 &&
      kv.dim() == 2 && compressed_kv.dim() == 2) {
    ::xllm::kernel::DeepseekV4MateSparsePrefillParams params;
    params.q = q.contiguous();
    params.kv = kv.contiguous();
    params.compressed_kv = compressed_kv.contiguous();
    params.attn_sink =
        attn_sink_.defined()
            ? attn_sink_.to(q.device()).to(torch::kFloat32).contiguous()
            : torch::Tensor();
    params.softmax_scale = softmax_scale_;
    params.output = torch::empty_like(q);
    auto fused_prefill =
        ::xllm::kernel::deepseek_v4_mate_sparse_attention_prefill(params);
    if (fused_prefill.defined()) {
      maybe_sync_attention_stage(layer_id_, "mate_c128_prefill", fused_prefill);
      return fused_prefill;
    }
  }
  const int64_t chunk_size = dsv4_musa_prefill_attention_chunk_size();
  if (causal && q.size(0) > chunk_size && chunk_size > 0) {
    std::vector<torch::Tensor> chunks;
    chunks.reserve(
        static_cast<size_t>((q.size(0) + chunk_size - 1) / chunk_size));
    for (int64_t start = 0; start < q.size(0); start += chunk_size) {
      const int64_t end = std::min<int64_t>(start + chunk_size, q.size(0));
      auto q_chunk = q.slice(/*dim=*/0, /*start=*/start, /*end=*/end);
      torch::Tensor index_chunk;
      if (compressed_indices.defined() && compressed_indices.dim() >= 2 &&
          compressed_indices.size(0) >= end) {
        index_chunk =
            compressed_indices.slice(/*dim=*/0, /*start=*/start, /*end=*/end);
      }
      chunks.push_back(dense_attention(q_chunk,
                                       kv,
                                       compressed_kv,
                                       index_chunk,
                                       causal,
                                       q_abs_start + start,
                                       use_full_kv));
    }
    return torch::cat(chunks, /*dim=*/0);
  }

  auto kv_2d = kv.reshape({-1, kv.size(kv.dim() - 1)}).to(q.dtype());
  const int64_t full_kv_len = kv_2d.size(0);
  int64_t kv_start_abs = 0;
  if (!use_full_kv && window_size_ > 0 && kv_2d.size(0) > window_size_) {
    kv_start_abs = kv_2d.size(0) - window_size_;
    kv_2d = kv_2d.slice(/*dim=*/0,
                        /*start=*/kv_2d.size(0) - window_size_,
                        /*end=*/kv_2d.size(0));
  }
  maybe_sync_attention_stage(layer_id_, "dense_kv_prepare", kv_2d);
  torch::Tensor cmp_2d;
  const bool has_compressed = compressed_kv.defined() &&
                              compressed_kv.numel() > 0 &&
                              (compress_ratio_ == 4 || compress_ratio_ == 128);
  if (has_compressed) {
    cmp_2d = compressed_kv.reshape(
        {-1, compressed_kv.size(compressed_kv.dim() - 1)});
    maybe_sync_attention_stage(layer_id_, "dense_compressed_prepare", cmp_2d);
  }

  const bool has_compressed_indices =
      compressed_indices.defined() && compressed_indices.numel() > 0;
  if (dsv4_musa_fused_dense_decode_enabled() && q.size(0) == 1 && !causal &&
      !has_compressed_indices && q.dim() == 3 && kv_2d.dim() == 2 &&
      kv_2d.size(1) == q.size(2) &&
      (!has_compressed || (cmp_2d.dim() == 2 && cmp_2d.size(1) == q.size(2)))) {
    ::xllm::kernel::DeepseekV4DenseAttentionDecodeParams params;
    params.q = q.contiguous();
    params.kv = kv_2d.contiguous();
    params.compressed_kv =
        has_compressed ? cmp_2d.to(q.dtype()).contiguous() : torch::Tensor();
    params.attn_sink =
        attn_sink_.defined()
            ? attn_sink_.to(q.device()).to(torch::kFloat32).contiguous()
            : torch::Tensor();
    params.softmax_scale = softmax_scale_;
    return ::xllm::kernel::deepseek_v4_dense_attention_decode(params);
  }

  auto q_h = q.permute({1, 0, 2}).contiguous().to(torch::kFloat32);
  auto kv_fp32 = kv_2d.to(torch::kFloat32);
  auto ori_scores =
      torch::matmul(q_h, kv_fp32.transpose(0, 1).unsqueeze(0)) * softmax_scale_;
  maybe_sync_attention_stage(layer_id_, "dense_ori_scores", ori_scores);
  if (causal) {
    const int64_t q_len = q.size(0);
    const int64_t kv_len = kv_2d.size(0);
    auto q_pos =
        torch::arange(
            q_len,
            torch::TensorOptions().dtype(torch::kLong).device(q.device()))
            .unsqueeze(1);
    auto k_pos =
        torch::arange(
            kv_len,
            torch::TensorOptions().dtype(torch::kLong).device(q.device()))
            .unsqueeze(0);
    auto q_abs = q_pos + q_abs_start;
    auto k_abs = k_pos + kv_start_abs;
    auto mask = k_abs.le(q_abs);
    if (window_size_ > 0) {
      mask = mask.logical_and(k_abs.ge(q_abs - (window_size_ - 1)));
    }
    ori_scores =
        ori_scores.masked_fill(mask.logical_not().unsqueeze(0),
                               -std::numeric_limits<float>::infinity());
    maybe_sync_attention_stage(layer_id_, "dense_causal_mask", ori_scores);
  }
  auto value_fp32 = kv_fp32;
  torch::Tensor cmp_scores;
  torch::Tensor cmp_fp32;
  if (has_compressed) {
    cmp_fp32 = cmp_2d.to(torch::kFloat32);
    cmp_scores = torch::matmul(q_h, cmp_fp32.transpose(0, 1).unsqueeze(0)) *
                 softmax_scale_;
    maybe_sync_attention_stage(layer_id_, "dense_cmp_scores", cmp_scores);
    if (causal) {
      const int64_t q_len = q.size(0);
      auto q_pos =
          torch::arange(
              q_len,
              torch::TensorOptions().dtype(torch::kLong).device(q.device()))
              .unsqueeze(1);
      auto q_abs = q_pos + q_abs_start;
      auto cmp_pos =
          torch::arange(
              cmp_2d.size(0),
              torch::TensorOptions().dtype(torch::kLong).device(q.device()))
              .unsqueeze(0);
      auto valid_cmp = cmp_pos.lt((q_abs + 1) / compress_ratio_);
      cmp_scores =
          cmp_scores.masked_fill(valid_cmp.logical_not().unsqueeze(0),
                                 -std::numeric_limits<float>::infinity());
      maybe_sync_attention_stage(
          layer_id_, "dense_cmp_causal_mask", cmp_scores);
    }
    if (has_compressed_indices && compressed_indices.dim() == 2 &&
        compressed_indices.size(0) == q.size(0)) {
      auto indices =
          compressed_indices.to(torch::kLong).to(q.device()).contiguous();
      auto valid_indices =
          indices.ge(0).logical_and(indices.lt(cmp_2d.size(0)));
      // Map padding to a dedicated sentinel column. Mapping it to row 0 and
      // scattering valid_indices lets trailing false entries overwrite a
      // legitimate TopK hit at compressed row 0.
      auto sentinel_indices = torch::full_like(indices, cmp_2d.size(0));
      auto safe_indices =
          torch::where(valid_indices, indices, sentinel_indices);
      auto topk_mask = torch::zeros(
          {q.size(0), cmp_2d.size(0) + 1},
          torch::TensorOptions().dtype(torch::kBool).device(q.device()));
      topk_mask = topk_mask.scatter(
          /*dim=*/1, safe_indices, torch::ones_like(valid_indices));
      topk_mask = topk_mask.slice(/*dim=*/1, /*start=*/0, cmp_2d.size(0));
      cmp_scores =
          cmp_scores.masked_fill(topk_mask.logical_not().unsqueeze(0),
                                 -std::numeric_limits<float>::infinity());
      maybe_sync_attention_stage(layer_id_, "dense_cmp_topk_mask", cmp_scores);
    }
  }
  auto max_scores = std::get<0>(ori_scores.max(/*dim=*/-1, /*keepdim=*/true));
  if (cmp_scores.defined()) {
    auto cmp_max = std::get<0>(cmp_scores.max(/*dim=*/-1, /*keepdim=*/true));
    max_scores = torch::max(max_scores, cmp_max);
  }
  auto exp_scores = torch::exp(ori_scores - max_scores);
  auto denom = exp_scores.sum(/*dim=*/-1, /*keepdim=*/true);
  torch::Tensor exp_cmp_scores;
  if (cmp_scores.defined()) {
    exp_cmp_scores = torch::exp(cmp_scores - max_scores);
    denom = denom + exp_cmp_scores.sum(/*dim=*/-1, /*keepdim=*/true);
  }
  if (attn_sink_.defined()) {
    auto sink = attn_sink_.to(ori_scores.device())
                    .to(torch::kFloat32)
                    .view({n_local_heads_, 1, 1});
    denom = denom + torch::exp(sink - max_scores);
  }
  auto probs = exp_scores / denom;
  maybe_sync_attention_stage(layer_id_, "dense_probs", probs);
  auto ctx = torch::matmul(probs, value_fp32.unsqueeze(0));
  maybe_sync_attention_stage(layer_id_, "dense_ctx_ori", ctx);
  if (exp_cmp_scores.defined()) {
    auto cmp_probs = exp_cmp_scores / denom;
    std::vector<torch::Tensor> cmp_ctx_heads;
    cmp_ctx_heads.reserve(static_cast<size_t>(cmp_probs.size(0)));
    auto cmp_value = cmp_fp32.contiguous();
    for (int64_t head = 0; head < cmp_probs.size(0); ++head) {
      cmp_ctx_heads.push_back(torch::matmul(
          cmp_probs.select(/*dim=*/0, /*index=*/head).contiguous(), cmp_value));
    }
    auto cmp_ctx = torch::stack(cmp_ctx_heads, /*dim=*/0);
    maybe_sync_attention_stage(layer_id_, "dense_ctx_cmp_matmul", cmp_ctx);
    std::vector<torch::Tensor> merged_ctx_heads;
    merged_ctx_heads.reserve(static_cast<size_t>(ctx.size(0)));
    auto ctx_value = ctx.contiguous();
    auto cmp_ctx_value = cmp_ctx.contiguous();
    for (int64_t head = 0; head < ctx_value.size(0); ++head) {
      merged_ctx_heads.push_back(ctx_value.select(/*dim=*/0, /*index=*/head) +
                                 cmp_ctx_value.select(/*dim=*/0,
                                                      /*index=*/head));
    }
    ctx = torch::stack(merged_ctx_heads, /*dim=*/0);
    maybe_sync_attention_stage(layer_id_, "dense_ctx_cmp_add", ctx);
    maybe_sync_attention_stage(layer_id_, "dense_ctx_cmp", ctx);
  }
  auto ctx_cast = ctx.to(q.dtype());
  maybe_sync_attention_stage(layer_id_, "dense_ctx_cast", ctx_cast);
  auto out = ctx_cast.permute({1, 0, 2}).contiguous();
  maybe_sync_attention_stage(layer_id_, "dense_out", out);
  return out;
}

torch::Tensor DeepseekV4AttentionImpl::compress_prefill_sequence(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    size_t sequence_index,
    int64_t prefix_snapshot_tokens,
    int64_t cached_prefix_tokens,
    const torch::Tensor& projected_kv,
    const torch::Tensor& projected_score) {
  const bool zero_local_sp_shard =
      sequence_parallel_context_ != nullptr && hidden_states.defined() &&
      hidden_states.dim() > 0 && hidden_states.size(0) == 0 &&
      sequence_parallel_context_->local_valid_tokens == 0;
  if (!(compress_ratio_ == 4 || compress_ratio_ == 128) || !compressor_wkv_ ||
      !compressor_wgate_ || !compressor_norm_ || !compressor_ape_loaded_ ||
      !hidden_states.defined() ||
      (hidden_states.numel() == 0 && !zero_local_sp_shard)) {
    return torch::Tensor();
  }
  CHECK_EQ(projected_kv.defined(), projected_score.defined());
  torch::Tensor kv_projection;
  torch::Tensor score_projection;
  if (projected_kv.defined()) {
    kv_projection = projected_kv;
    score_projection = projected_score;
  } else if (zero_local_sp_shard) {
    kv_projection = torch::empty(
        {0, compressor_wkv_->weight().size(0)},
        hidden_states.options().dtype(compressor_wkv_->weight().scalar_type()));
    score_projection =
        torch::empty({0, compressor_wgate_->weight().size(0)},
                     hidden_states.options().dtype(
                         compressor_wgate_->weight().scalar_type()));
  } else {
    kv_projection = compressor_wkv_->forward(hidden_states);
    score_projection = compressor_wgate_->forward(hidden_states);
  }
  if (!projected_kv.defined() && sequence_parallel_context_ != nullptr &&
      hidden_states.size(0) == sequence_parallel_context_->local_valid_tokens) {
    CHECK_EQ(kv_projection.scalar_type(), score_projection.scalar_type());
    const int64_t kv_width = kv_projection.size(1);
    torch::Tensor packed_projection =
        torch::cat({kv_projection.contiguous(), score_projection.contiguous()},
                   /*dim=*/1);
    packed_projection = dsv4_musa_sp::gather_global(
        dsv4_musa_sp::pad_local(packed_projection, *sequence_parallel_context_),
        *sequence_parallel_context_);
    kv_projection = packed_projection.slice(
        /*dim=*/1, /*start=*/0, /*end=*/kv_width);
    score_projection = packed_projection.slice(
        /*dim=*/1, /*start=*/kv_width, /*end=*/packed_projection.size(1));
  }
  torch::Tensor kv_full = kv_projection.to(torch::kFloat32).contiguous();
  maybe_sync_attention_stage(layer_id_, "compress_prefill_wkv", kv_full);
  torch::Tensor score_full = score_projection.to(torch::kFloat32).contiguous();
  maybe_sync_attention_stage(layer_id_, "compress_prefill_wgate", score_full);
  auto ape =
      compressor_ape_.to(score_full.device()).to(torch::kFloat32).contiguous();

  const int64_t state_rows = compressor_coff_ * compress_ratio_;
  const int64_t state_dim = compressor_coff_ * head_dim_;
  if (sequence_index >= compressor_kv_state_.size()) {
    compressor_kv_state_.resize(sequence_index + 1);
    compressor_score_state_.resize(sequence_index + 1);
  }
  if (sequence_index >= prefix_compressor_kv_state_.size()) {
    prefix_compressor_kv_state_.resize(sequence_index + 1);
    prefix_compressor_score_state_.resize(sequence_index + 1);
  }
  auto state_options = hidden_states.options().dtype(torch::kFloat32);
  if (pcp_context_ != nullptr && pcp_context_->cp_rank == 1) {
    if (!compressor_kv_state_[sequence_index].defined()) {
      compressor_kv_state_[sequence_index] =
          torch::zeros({state_rows, state_dim}, state_options);
      compressor_score_state_[sequence_index] =
          torch::full({state_rows, state_dim},
                      -std::numeric_limits<float>::infinity(),
                      state_options);
    }
    compressor_kv_state_[sequence_index] = dsv4_musa_pcp::relay_state(
        compressor_kv_state_[sequence_index], *pcp_context_);
    compressor_score_state_[sequence_index] = dsv4_musa_pcp::relay_state(
        compressor_score_state_[sequence_index], *pcp_context_);
  }
  prefix_compressor_kv_state_[sequence_index] = torch::Tensor();
  prefix_compressor_score_state_[sequence_index] = torch::Tensor();

  int64_t output_block_offset = 0;
  torch::Tensor effective_cos = cos;
  torch::Tensor effective_sin = sin;
  if (cached_prefix_tokens > 0) {
    auto continuation =
        dsv4_prefill_continuation(kv_full,
                                  score_full,
                                  ape,
                                  compressor_kv_state_[sequence_index],
                                  compressor_score_state_[sequence_index],
                                  cached_prefix_tokens,
                                  compress_ratio_);
    kv_full = continuation.kv;
    score_full = continuation.score;
    output_block_offset = continuation.output_block_offset;
    if (prefix_snapshot_tokens > 0) {
      prefix_snapshot_tokens +=
          continuation.prepended_rows - cached_prefix_tokens;
    }
    if (cached_prefix_tokens % compress_ratio_ != 0) {
      LOG_FIRST_N(INFO, 8) << "[DSV4_PREFILL_CONTINUATION] ratio="
                           << compress_ratio_
                           << " cached=" << cached_prefix_tokens
                           << " restored_partial="
                           << cached_prefix_tokens % compress_ratio_;
    }
    if (output_block_offset) {
      if (cos.defined() && sin.defined() && cos.size(0) > 0 &&
          sin.size(0) > 0) {
        effective_cos = torch::cat(
            {cos.slice(/*dim=*/0, /*start=*/0, /*end=*/1), cos}, /*dim=*/0);
        effective_sin = torch::cat(
            {sin.slice(/*dim=*/0, /*start=*/0, /*end=*/1), sin}, /*dim=*/0);
      }
    }
  }

  const int64_t seq_len = kv_full.size(0);
  const int64_t cutoff = seq_len - (seq_len % compress_ratio_);
  if (prefix_snapshot_tokens > 0 && prefix_snapshot_tokens <= seq_len &&
      prefix_snapshot_tokens % compress_ratio_ == 0) {
    auto snapshot_kv = torch::zeros({state_rows, state_dim}, state_options);
    auto snapshot_score = torch::full({state_rows, state_dim},
                                      -std::numeric_limits<float>::infinity(),
                                      state_options);
    if (compress_ratio_ == 4 && prefix_snapshot_tokens >= compress_ratio_) {
      snapshot_kv.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
          .copy_(
              kv_full.slice(/*dim=*/0,
                            /*start=*/prefix_snapshot_tokens - compress_ratio_,
                            /*end=*/prefix_snapshot_tokens));
      snapshot_score.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
          .copy_(score_full.slice(
                     /*dim=*/0,
                     /*start=*/prefix_snapshot_tokens - compress_ratio_,
                     /*end=*/prefix_snapshot_tokens) +
                 ape);
    }
    prefix_compressor_kv_state_[sequence_index] = snapshot_kv.contiguous();
    prefix_compressor_score_state_[sequence_index] =
        snapshot_score.contiguous();
  }
  auto kv_state = torch::zeros({state_rows, state_dim}, state_options);
  auto score_state = torch::full({state_rows, state_dim},
                                 -std::numeric_limits<float>::infinity(),
                                 state_options);
  const int64_t remainder = seq_len - cutoff;
  if (compress_ratio_ == 4) {
    if (cutoff >= compress_ratio_) {
      kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
          .copy_(kv_full.slice(/*dim=*/0,
                               /*start=*/cutoff - compress_ratio_,
                               /*end=*/cutoff));
      score_state.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
          .copy_(score_full.slice(/*dim=*/0,
                                  /*start=*/cutoff - compress_ratio_,
                                  /*end=*/cutoff) +
                 ape);
    }
    if (remainder > 0) {
      kv_state
          .slice(/*dim=*/0,
                 /*start=*/compress_ratio_,
                 /*end=*/compress_ratio_ + remainder)
          .copy_(kv_full.slice(/*dim=*/0, /*start=*/cutoff, /*end=*/seq_len));
      score_state
          .slice(/*dim=*/0,
                 /*start=*/compress_ratio_,
                 /*end=*/compress_ratio_ + remainder)
          .copy_(score_full.slice(/*dim=*/0,
                                  /*start=*/cutoff,
                                  /*end=*/seq_len) +
                 ape.slice(/*dim=*/0, /*start=*/0, /*end=*/remainder));
    }
  } else if (remainder > 0) {
    kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/remainder)
        .copy_(kv_full.slice(/*dim=*/0, /*start=*/cutoff, /*end=*/seq_len));
    score_state.slice(/*dim=*/0, /*start=*/0, /*end=*/remainder)
        .copy_(score_full.slice(/*dim=*/0, /*start=*/cutoff, /*end=*/seq_len) +
               ape.slice(/*dim=*/0, /*start=*/0, /*end=*/remainder));
  }
  compressor_kv_state_[sequence_index] = kv_state.contiguous();
  compressor_score_state_[sequence_index] = score_state.contiguous();
  if (pcp_context_ != nullptr) {
    if (pcp_context_->cp_rank == 0) {
      dsv4_musa_pcp::relay_state(compressor_kv_state_[sequence_index],
                                 *pcp_context_);
      dsv4_musa_pcp::relay_state(compressor_score_state_[sequence_index],
                                 *pcp_context_);
    }
    dsv4_musa_pcp::relay_state_back(compressor_kv_state_[sequence_index],
                                    *pcp_context_);
    dsv4_musa_pcp::relay_state_back(compressor_score_state_[sequence_index],
                                    *pcp_context_);
  }

  if (cutoff <= 0) {
    return torch::Tensor();
  }

  auto kv = kv_full.slice(/*dim=*/0, /*start=*/0, /*end=*/cutoff);
  auto score = score_full.slice(/*dim=*/0, /*start=*/0, /*end=*/cutoff);
  const int64_t blocks = cutoff / compress_ratio_;
  kv = kv.view({blocks, compress_ratio_, compressor_coff_ * head_dim_});
  score = score.view({blocks, compress_ratio_, compressor_coff_ * head_dim_}) +
          ape.unsqueeze(0);

  torch::Tensor compressed;
  if (dsv4_musa_fused_compress_pool_enabled()) {
    ::xllm::kernel::DeepseekV4CompressPoolParams params;
    params.kv = kv_full;
    params.score = score_full;
    params.ape = ape;
    params.compress_ratio = compress_ratio_;
    params.head_dim = head_dim_;
    params.compressor_coeff = compressor_coff_;
    compressed = ::xllm::kernel::deepseek_v4_compress_pool(params);
    maybe_sync_attention_stage(
        layer_id_, "compress_prefill_fused_pool", compressed);
  }
  if (!compressed.defined()) {
    torch::Tensor pooled_kv;
    torch::Tensor pooled_score;
    if (compress_ratio_ == 4) {
      auto kv_options = kv.options();
      auto score_options = score.options();
      pooled_kv =
          torch::zeros({blocks, compress_ratio_ * 2, head_dim_}, kv_options);
      pooled_score = torch::full({blocks, compress_ratio_ * 2, head_dim_},
                                 -std::numeric_limits<float>::infinity(),
                                 score_options);
      pooled_kv.slice(/*dim=*/1, /*start=*/compress_ratio_)
          .copy_(kv.slice(/*dim=*/2, /*start=*/head_dim_));
      pooled_score.slice(/*dim=*/1, /*start=*/compress_ratio_)
          .copy_(score.slice(/*dim=*/2, /*start=*/head_dim_));
      if (blocks > 1) {
        pooled_kv.slice(/*dim=*/0, /*start=*/1)
            .slice(/*dim=*/1, /*start=*/0, /*end=*/compress_ratio_)
            .copy_(kv.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks - 1)
                       .slice(/*dim=*/2, /*start=*/0, /*end=*/head_dim_));
        pooled_score.slice(/*dim=*/0, /*start=*/1)
            .slice(/*dim=*/1, /*start=*/0, /*end=*/compress_ratio_)
            .copy_(score.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks - 1)
                       .slice(/*dim=*/2, /*start=*/0, /*end=*/head_dim_));
      }
    } else {
      pooled_kv = kv;
      pooled_score = score;
    }
    auto weights = torch::softmax(pooled_score, /*dim=*/1);
    maybe_sync_attention_stage(layer_id_, "compress_prefill_softmax", weights);
    compressed = (pooled_kv * weights).sum(/*dim=*/1);
  }
  compressed = compressed.to(hidden_states.dtype());
  compressed = std::get<0>(compressor_norm_->forward(compressed));
  maybe_sync_attention_stage(layer_id_, "compress_prefill_norm", compressed);
  if (effective_cos.defined() && effective_sin.defined() &&
      effective_cos.numel() >= blocks * rope_head_dim_ &&
      effective_sin.numel() >= blocks * rope_head_dim_) {
    auto cos_slice =
        effective_cos.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks);
    auto sin_slice =
        effective_sin.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks);
    compressed = apply_interleaved_partial_rope(
        compressed, nope_head_dim_, rope_head_dim_, cos_slice, sin_slice);
    maybe_sync_attention_stage(layer_id_, "compress_prefill_rope", compressed);
  }
  fp8_act_quant_roundtrip_inplace(
      compressed, nope_head_dim_, /*block_size=*/64);
  maybe_sync_attention_stage(
      layer_id_, "compress_prefill_fp8_roundtrip", compressed);
  if (output_block_offset > 0) {
    CHECK_GE(compressed.size(0), output_block_offset);
    compressed = compressed.slice(/*dim=*/0,
                                  /*start=*/output_block_offset,
                                  /*end=*/compressed.size(0));
  }
  return compressed;
}

torch::Tensor DeepseekV4AttentionImpl::compress_decode_step(
    const torch::Tensor& hidden_step,
    int64_t start_pos,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    size_t sequence_index,
    const torch::Tensor& projected_kv,
    const torch::Tensor& projected_score) {
  if (!(compress_ratio_ == 4 || compress_ratio_ == 128) || !compressor_wkv_ ||
      !compressor_wgate_ || !compressor_norm_ || !compressor_ape_loaded_ ||
      !hidden_step.defined() || hidden_step.numel() == 0 ||
      hidden_step.size(0) != 1 || start_pos < 0) {
    return torch::Tensor();
  }

  CHECK_EQ(projected_kv.defined(), projected_score.defined());
  const bool use_preprojected = projected_kv.defined();
  if (use_preprojected) {
    CHECK_EQ(projected_kv.dim(), 2);
    CHECK_EQ(projected_score.dim(), 2);
    CHECK_EQ(projected_kv.size(0), 1);
    CHECK_EQ(projected_score.size(0), 1);
    CHECK_EQ(projected_kv.size(1), compressor_coff_ * head_dim_);
    CHECK_EQ(projected_score.size(1), compressor_coff_ * head_dim_);
  }

  const int64_t state_rows = compressor_coff_ * compress_ratio_;
  const int64_t state_dim = compressor_coff_ * head_dim_;
  if (sequence_index >= compressor_kv_state_.size()) {
    compressor_kv_state_.resize(sequence_index + 1);
    compressor_score_state_.resize(sequence_index + 1);
  }
  auto state_options = hidden_step.options().dtype(torch::kFloat32);
  if (!compressor_kv_state_[sequence_index].defined() ||
      compressor_kv_state_[sequence_index].dim() != 2 ||
      compressor_kv_state_[sequence_index].size(0) != state_rows ||
      compressor_kv_state_[sequence_index].size(1) != state_dim) {
    compressor_kv_state_[sequence_index] =
        torch::zeros({state_rows, state_dim}, state_options);
  }
  if (!compressor_score_state_[sequence_index].defined() ||
      compressor_score_state_[sequence_index].dim() != 2 ||
      compressor_score_state_[sequence_index].size(0) != state_rows ||
      compressor_score_state_[sequence_index].size(1) != state_dim) {
    compressor_score_state_[sequence_index] =
        torch::full({state_rows, state_dim},
                    -std::numeric_limits<float>::infinity(),
                    state_options);
  }

  auto& kv_state = compressor_kv_state_[sequence_index];
  auto& score_state = compressor_score_state_[sequence_index];
  const bool use_workspace_decode =
      (use_preprojected || (decode_compressor_kv_projection_.defined() &&
                            decode_compressor_score_projection_.defined())) &&
      decode_compressor_kv_projection_.defined() &&
      decode_compressor_score_projection_.defined() &&
      decode_compressor_output_.defined() &&
      decode_compressor_norm_output_.defined() &&
      decode_compressor_rope_output_.defined() && compressor_ape_.defined() &&
      compressor_ape_.scalar_type() == torch::kFloat32 &&
      compressor_ape_.is_contiguous();
  if (use_workspace_decode) {
    auto kv_projection =
        use_preprojected
            ? projected_kv
            : kernel::musa::matmul(hidden_step,
                                   compressor_wkv_->weight(),
                                   std::nullopt,
                                   std::optional<torch::Tensor>(
                                       decode_compressor_kv_projection_.narrow(
                                           /*dim=*/0,
                                           /*start=*/0,
                                           /*length=*/1)));
    auto score_projection =
        use_preprojected ? projected_score
                         : kernel::musa::matmul(
                               hidden_step,
                               compressor_wgate_->weight(),
                               std::nullopt,
                               std::optional<torch::Tensor>(
                                   decode_compressor_score_projection_.narrow(
                                       /*dim=*/0,
                                       /*start=*/0,
                                       /*length=*/1)));
    auto compressed = kernel::musa::deepseek_v4_compress_decode(
        kv_state,
        score_state,
        kv_projection,
        score_projection,
        compressor_ape_,
        compress_ratio_,
        head_dim_,
        compressor_coff_,
        start_pos,
        /*compute_boundary_output=*/true,
        std::optional<torch::Tensor>(decode_compressor_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/1)));
    if (!compressed.defined()) {
      return torch::Tensor();
    }
    compressed = std::get<0>(compressor_norm_->forward(
        compressed,
        std::nullopt,
        std::optional<torch::Tensor>(decode_compressor_norm_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/1))));
    if (cos.defined() && sin.defined() && cos.size(0) > 0 && sin.size(0) > 0) {
      compressed = apply_interleaved_partial_rope(
          compressed,
          nope_head_dim_,
          rope_head_dim_,
          cos.slice(/*dim=*/0, /*start=*/0, /*end=*/1),
          sin.slice(/*dim=*/0, /*start=*/0, /*end=*/1),
          /*inverse=*/false,
          decode_compressor_rope_output_.narrow(
              /*dim=*/0, /*start=*/0, /*length=*/1));
    }
    fp8_act_quant_roundtrip_inplace(
        compressed, nope_head_dim_, /*block_size=*/64);
    return compressed;
  }
  auto kv_step =
      (use_preprojected ? projected_kv : compressor_wkv_->forward(hidden_step))
          .to(torch::kFloat32)
          .contiguous();
  maybe_sync_attention_stage(layer_id_, "compress_decode_wkv", kv_step);
  auto score_step = (use_preprojected ? projected_score
                                      : compressor_wgate_->forward(hidden_step))
                        .to(torch::kFloat32)
                        .contiguous();
  maybe_sync_attention_stage(layer_id_, "compress_decode_wgate", score_step);
  auto ape = compressor_ape_.to(score_step.device()).to(torch::kFloat32);
  const int64_t ratio_pos = start_pos % compress_ratio_;
  score_step = score_step + ape.index({ratio_pos}).unsqueeze(0);
  const bool should_compress = (start_pos + 1) % compress_ratio_ == 0;

  if (compress_ratio_ == 4) {
    kv_state.index_put_({compress_ratio_ + ratio_pos}, kv_step.squeeze(0));
    score_state.index_put_({compress_ratio_ + ratio_pos},
                           score_step.squeeze(0));
  } else {
    kv_state.index_put_({ratio_pos}, kv_step.squeeze(0));
    score_state.index_put_({ratio_pos}, score_step.squeeze(0));
  }
  if (!should_compress) {
    return torch::Tensor();
  }

  torch::Tensor pooled_kv;
  torch::Tensor pooled_score;
  if (compress_ratio_ == 4) {
    pooled_kv =
        torch::cat({kv_state
                        .slice(/*dim=*/0,
                               /*start=*/0,
                               /*end=*/compress_ratio_)
                        .slice(/*dim=*/1, /*start=*/0, /*end=*/head_dim_),
                    kv_state.slice(/*dim=*/0, /*start=*/compress_ratio_)
                        .slice(/*dim=*/1, /*start=*/head_dim_)},
                   /*dim=*/0);
    pooled_score =
        torch::cat({score_state
                        .slice(/*dim=*/0,
                               /*start=*/0,
                               /*end=*/compress_ratio_)
                        .slice(/*dim=*/1, /*start=*/0, /*end=*/head_dim_),
                    score_state.slice(/*dim=*/0, /*start=*/compress_ratio_)
                        .slice(/*dim=*/1, /*start=*/head_dim_)},
                   /*dim=*/0);
  } else {
    pooled_kv = kv_state;
    pooled_score = score_state;
  }

  auto weights = torch::softmax(pooled_score, /*dim=*/0);
  maybe_sync_attention_stage(layer_id_, "compress_decode_softmax", weights);
  auto compressed =
      (pooled_kv * weights).sum(/*dim=*/0).unsqueeze(0).to(hidden_step.dtype());
  compressed = std::get<0>(compressor_norm_->forward(compressed));
  maybe_sync_attention_stage(layer_id_, "compress_decode_norm", compressed);
  if (cos.defined() && sin.defined() && cos.size(0) > 0 && sin.size(0) > 0) {
    compressed = apply_interleaved_partial_rope(compressed,
                                                nope_head_dim_,
                                                rope_head_dim_,
                                                cos.slice(/*dim=*/0,
                                                          /*start=*/0,
                                                          /*end=*/1),
                                                sin.slice(/*dim=*/0,
                                                          /*start=*/0,
                                                          /*end=*/1));
    maybe_sync_attention_stage(layer_id_, "compress_decode_rope", compressed);
  }
  fp8_act_quant_roundtrip_inplace(
      compressed, nope_head_dim_, /*block_size=*/64);
  maybe_sync_attention_stage(
      layer_id_, "compress_decode_fp8_roundtrip", compressed);
  if (compress_ratio_ == 4) {
    kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
        .copy_(kv_state.slice(/*dim=*/0, /*start=*/compress_ratio_));
    score_state.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
        .copy_(score_state.slice(/*dim=*/0, /*start=*/compress_ratio_));
  }
  return compressed;
}

torch::Tensor DeepseekV4AttentionImpl::compress_indexer_prefill_sequence(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    size_t sequence_index,
    int64_t prefix_snapshot_tokens,
    int64_t cached_prefix_tokens,
    const torch::Tensor& projected_kv,
    const torch::Tensor& projected_score) {
  constexpr int64_t kCompressRatio = 4;
  constexpr int64_t kCompressorCoff = 2;
  const bool zero_local_sp_shard =
      sequence_parallel_context_ != nullptr && hidden_states.defined() &&
      hidden_states.dim() > 0 && hidden_states.size(0) == 0 &&
      sequence_parallel_context_->local_valid_tokens == 0;
  if (compress_ratio_ != kCompressRatio || !indexer_compressor_wkv_ ||
      !indexer_compressor_wgate_ || !indexer_compressor_norm_ ||
      !indexer_compressor_ape_loaded_ || !hidden_states.defined() ||
      (hidden_states.numel() == 0 && !zero_local_sp_shard)) {
    return torch::Tensor();
  }

  CHECK_EQ(projected_kv.defined(), projected_score.defined());
  torch::Tensor kv_projection;
  torch::Tensor score_projection;
  if (projected_kv.defined()) {
    kv_projection = projected_kv;
    score_projection = projected_score;
  } else if (zero_local_sp_shard) {
    kv_projection =
        torch::empty({0, indexer_compressor_wkv_->weight().size(0)},
                     hidden_states.options().dtype(
                         indexer_compressor_wkv_->weight().scalar_type()));
    score_projection =
        torch::empty({0, indexer_compressor_wgate_->weight().size(0)},
                     hidden_states.options().dtype(
                         indexer_compressor_wgate_->weight().scalar_type()));
  } else {
    kv_projection = indexer_compressor_wkv_->forward(hidden_states);
    score_projection = indexer_compressor_wgate_->forward(hidden_states);
  }
  if (!projected_kv.defined() && sequence_parallel_context_ != nullptr &&
      hidden_states.size(0) == sequence_parallel_context_->local_valid_tokens) {
    CHECK_EQ(kv_projection.scalar_type(), score_projection.scalar_type());
    const int64_t kv_width = kv_projection.size(1);
    torch::Tensor packed_projection =
        torch::cat({kv_projection.contiguous(), score_projection.contiguous()},
                   /*dim=*/1);
    packed_projection = dsv4_musa_sp::gather_global(
        dsv4_musa_sp::pad_local(packed_projection, *sequence_parallel_context_),
        *sequence_parallel_context_);
    kv_projection = packed_projection.slice(
        /*dim=*/1, /*start=*/0, /*end=*/kv_width);
    score_projection = packed_projection.slice(
        /*dim=*/1, /*start=*/kv_width, /*end=*/packed_projection.size(1));
  }
  torch::Tensor kv_full = kv_projection.to(torch::kFloat32).contiguous();
  torch::Tensor score_full = score_projection.to(torch::kFloat32).contiguous();
  auto ape = indexer_compressor_ape_.to(score_full.device())
                 .to(torch::kFloat32)
                 .contiguous();

  const int64_t state_rows = kCompressorCoff * kCompressRatio;
  const int64_t state_dim = kCompressorCoff * index_head_dim_;
  if (sequence_index >= indexer_kv_state_.size()) {
    indexer_kv_state_.resize(sequence_index + 1);
    indexer_score_state_.resize(sequence_index + 1);
  }
  if (sequence_index >= prefix_indexer_kv_state_.size()) {
    prefix_indexer_kv_state_.resize(sequence_index + 1);
    prefix_indexer_score_state_.resize(sequence_index + 1);
  }
  auto state_options = hidden_states.options().dtype(torch::kFloat32);
  if (pcp_context_ != nullptr && pcp_context_->cp_rank == 1) {
    if (!indexer_kv_state_[sequence_index].defined()) {
      indexer_kv_state_[sequence_index] =
          torch::zeros({state_rows, state_dim}, state_options);
      indexer_score_state_[sequence_index] =
          torch::full({state_rows, state_dim},
                      -std::numeric_limits<float>::infinity(),
                      state_options);
    }
    indexer_kv_state_[sequence_index] = dsv4_musa_pcp::relay_state(
        indexer_kv_state_[sequence_index], *pcp_context_);
    indexer_score_state_[sequence_index] = dsv4_musa_pcp::relay_state(
        indexer_score_state_[sequence_index], *pcp_context_);
  }
  prefix_indexer_kv_state_[sequence_index] = torch::Tensor();
  prefix_indexer_score_state_[sequence_index] = torch::Tensor();
  int64_t output_block_offset = 0;
  torch::Tensor effective_cos = cos;
  torch::Tensor effective_sin = sin;
  if (cached_prefix_tokens > 0) {
    auto continuation =
        dsv4_prefill_continuation(kv_full,
                                  score_full,
                                  ape,
                                  indexer_kv_state_[sequence_index],
                                  indexer_score_state_[sequence_index],
                                  cached_prefix_tokens,
                                  kCompressRatio);
    kv_full = continuation.kv;
    score_full = continuation.score;
    output_block_offset = continuation.output_block_offset;
    if (prefix_snapshot_tokens > 0) {
      prefix_snapshot_tokens +=
          continuation.prepended_rows - cached_prefix_tokens;
    }
    if (cos.defined() && sin.defined() && cos.size(0) > 0 && sin.size(0) > 0) {
      effective_cos = torch::cat(
          {cos.slice(/*dim=*/0, /*start=*/0, /*end=*/1), cos}, /*dim=*/0);
      effective_sin = torch::cat(
          {sin.slice(/*dim=*/0, /*start=*/0, /*end=*/1), sin}, /*dim=*/0);
    }
  }
  const int64_t seq_len = kv_full.size(0);
  const int64_t cutoff = seq_len - (seq_len % kCompressRatio);
  if (prefix_snapshot_tokens > 0 && prefix_snapshot_tokens <= seq_len &&
      prefix_snapshot_tokens % kCompressRatio == 0) {
    auto snapshot_kv = torch::zeros({state_rows, state_dim}, state_options);
    auto snapshot_score = torch::full({state_rows, state_dim},
                                      -std::numeric_limits<float>::infinity(),
                                      state_options);
    snapshot_kv.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
        .copy_(kv_full.slice(/*dim=*/0,
                             /*start=*/prefix_snapshot_tokens - kCompressRatio,
                             /*end=*/prefix_snapshot_tokens));
    snapshot_score.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
        .copy_(score_full.slice(
                   /*dim=*/0,
                   /*start=*/prefix_snapshot_tokens - kCompressRatio,
                   /*end=*/prefix_snapshot_tokens) +
               ape);
    prefix_indexer_kv_state_[sequence_index] = snapshot_kv.contiguous();
    prefix_indexer_score_state_[sequence_index] = snapshot_score.contiguous();
  }
  auto kv_state = torch::zeros({state_rows, state_dim}, state_options);
  auto score_state = torch::full({state_rows, state_dim},
                                 -std::numeric_limits<float>::infinity(),
                                 state_options);
  const int64_t remainder = seq_len - cutoff;
  if (cutoff >= kCompressRatio) {
    kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
        .copy_(kv_full.slice(/*dim=*/0,
                             /*start=*/cutoff - kCompressRatio,
                             /*end=*/cutoff));
    score_state.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
        .copy_(score_full.slice(/*dim=*/0,
                                /*start=*/cutoff - kCompressRatio,
                                /*end=*/cutoff) +
               ape);
  }
  if (remainder > 0) {
    kv_state
        .slice(/*dim=*/0,
               /*start=*/kCompressRatio,
               /*end=*/kCompressRatio + remainder)
        .copy_(kv_full.slice(/*dim=*/0, /*start=*/cutoff, /*end=*/seq_len));
    score_state
        .slice(/*dim=*/0,
               /*start=*/kCompressRatio,
               /*end=*/kCompressRatio + remainder)
        .copy_(score_full.slice(/*dim=*/0, /*start=*/cutoff, /*end=*/seq_len) +
               ape.slice(/*dim=*/0, /*start=*/0, /*end=*/remainder));
  }
  indexer_kv_state_[sequence_index] = kv_state.contiguous();
  indexer_score_state_[sequence_index] = score_state.contiguous();
  if (pcp_context_ != nullptr) {
    if (pcp_context_->cp_rank == 0) {
      dsv4_musa_pcp::relay_state(indexer_kv_state_[sequence_index],
                                 *pcp_context_);
      dsv4_musa_pcp::relay_state(indexer_score_state_[sequence_index],
                                 *pcp_context_);
    }
    dsv4_musa_pcp::relay_state_back(indexer_kv_state_[sequence_index],
                                    *pcp_context_);
    dsv4_musa_pcp::relay_state_back(indexer_score_state_[sequence_index],
                                    *pcp_context_);
  }

  if (cutoff <= 0) {
    return torch::Tensor();
  }

  const int64_t blocks = cutoff / kCompressRatio;
  auto kv = kv_full.slice(/*dim=*/0, /*start=*/0, /*end=*/cutoff)
                .view({blocks, kCompressRatio, state_dim});
  auto score = score_full.slice(/*dim=*/0, /*start=*/0, /*end=*/cutoff)
                   .view({blocks, kCompressRatio, state_dim}) +
               ape.unsqueeze(0);
  torch::Tensor compressed;
  if (dsv4_musa_fused_compress_pool_enabled()) {
    ::xllm::kernel::DeepseekV4CompressPoolParams params;
    params.kv = kv_full;
    params.score = score_full;
    params.ape = ape;
    params.compress_ratio = kCompressRatio;
    params.head_dim = index_head_dim_;
    params.compressor_coeff = kCompressorCoff;
    compressed = ::xllm::kernel::deepseek_v4_compress_pool(params);
  }
  if (!compressed.defined()) {
    auto pooled_kv = torch::zeros(
        {blocks, kCompressRatio * kCompressorCoff, index_head_dim_},
        kv.options());
    auto pooled_score =
        torch::full({blocks, kCompressRatio * kCompressorCoff, index_head_dim_},
                    -std::numeric_limits<float>::infinity(),
                    score.options());
    pooled_kv.slice(/*dim=*/1, /*start=*/kCompressRatio)
        .copy_(kv.slice(/*dim=*/2, /*start=*/index_head_dim_));
    pooled_score.slice(/*dim=*/1, /*start=*/kCompressRatio)
        .copy_(score.slice(/*dim=*/2, /*start=*/index_head_dim_));
    if (blocks > 1) {
      pooled_kv.slice(/*dim=*/0, /*start=*/1)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/kCompressRatio)
          .copy_(kv.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks - 1)
                     .slice(/*dim=*/2, /*start=*/0, /*end=*/index_head_dim_));
      pooled_score.slice(/*dim=*/0, /*start=*/1)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/kCompressRatio)
          .copy_(score.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks - 1)
                     .slice(/*dim=*/2, /*start=*/0, /*end=*/index_head_dim_));
    }
    auto weights = torch::softmax(pooled_score, /*dim=*/1);
    compressed = (pooled_kv * weights).sum(/*dim=*/1);
  }
  compressed = compressed.to(hidden_states.dtype());
  compressed = std::get<0>(indexer_compressor_norm_->forward(compressed));
  if (effective_cos.defined() && effective_sin.defined() &&
      effective_cos.numel() >= blocks * rope_head_dim_ &&
      effective_sin.numel() >= blocks * rope_head_dim_) {
    compressed = apply_interleaved_partial_rope(
        compressed,
        index_head_dim_ - rope_head_dim_,
        rope_head_dim_,
        effective_cos.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks),
        effective_sin.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks));
  }
  if (output_block_offset > 0) {
    CHECK_GE(compressed.size(0), output_block_offset);
    compressed = compressed.slice(/*dim=*/0,
                                  /*start=*/output_block_offset,
                                  /*end=*/compressed.size(0));
  }
  return apply_indexer_hadamard(compressed, indexer_hadamard_);
}

torch::Tensor DeepseekV4AttentionImpl::compress_indexer_decode_step(
    const torch::Tensor& hidden_step,
    int64_t start_pos,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    size_t sequence_index,
    const torch::Tensor& projected_kv,
    const torch::Tensor& projected_score) {
  constexpr int64_t kCompressRatio = 4;
  constexpr int64_t kCompressorCoff = 2;
  if (compress_ratio_ != kCompressRatio || !indexer_compressor_wkv_ ||
      !indexer_compressor_wgate_ || !indexer_compressor_norm_ ||
      !indexer_compressor_ape_loaded_ || !hidden_step.defined() ||
      hidden_step.numel() == 0 || hidden_step.size(0) != 1 || start_pos < 0) {
    return torch::Tensor();
  }

  CHECK_EQ(projected_kv.defined(), projected_score.defined());
  const bool use_preprojected = projected_kv.defined();
  if (use_preprojected) {
    CHECK_EQ(projected_kv.dim(), 2);
    CHECK_EQ(projected_score.dim(), 2);
    CHECK_EQ(projected_kv.size(0), 1);
    CHECK_EQ(projected_score.size(0), 1);
    CHECK_EQ(projected_kv.size(1), kCompressorCoff * index_head_dim_);
    CHECK_EQ(projected_score.size(1), kCompressorCoff * index_head_dim_);
  }

  const int64_t state_rows = kCompressorCoff * kCompressRatio;
  const int64_t state_dim = kCompressorCoff * index_head_dim_;
  if (sequence_index >= indexer_kv_state_.size()) {
    indexer_kv_state_.resize(sequence_index + 1);
    indexer_score_state_.resize(sequence_index + 1);
  }
  auto state_options = hidden_step.options().dtype(torch::kFloat32);
  if (!indexer_kv_state_[sequence_index].defined() ||
      indexer_kv_state_[sequence_index].dim() != 2 ||
      indexer_kv_state_[sequence_index].size(0) != state_rows ||
      indexer_kv_state_[sequence_index].size(1) != state_dim) {
    indexer_kv_state_[sequence_index] =
        torch::zeros({state_rows, state_dim}, state_options);
  }
  if (!indexer_score_state_[sequence_index].defined() ||
      indexer_score_state_[sequence_index].dim() != 2 ||
      indexer_score_state_[sequence_index].size(0) != state_rows ||
      indexer_score_state_[sequence_index].size(1) != state_dim) {
    indexer_score_state_[sequence_index] =
        torch::full({state_rows, state_dim},
                    -std::numeric_limits<float>::infinity(),
                    state_options);
  }

  auto& kv_state = indexer_kv_state_[sequence_index];
  auto& score_state = indexer_score_state_[sequence_index];
  const bool use_workspace_decode =
      (use_preprojected || (decode_indexer_kv_projection_.defined() &&
                            decode_indexer_score_projection_.defined())) &&
      decode_indexer_kv_projection_.defined() &&
      decode_indexer_score_projection_.defined() &&
      decode_indexer_output_.defined() &&
      decode_indexer_norm_output_.defined() &&
      indexer_compressor_ape_.defined() &&
      indexer_compressor_ape_.scalar_type() == torch::kFloat32 &&
      indexer_compressor_ape_.is_contiguous();
  if (use_workspace_decode) {
    auto kv_projection =
        use_preprojected ? projected_kv
                         : kernel::musa::matmul(
                               hidden_step,
                               indexer_compressor_wkv_->weight(),
                               std::nullopt,
                               std::optional<torch::Tensor>(
                                   decode_indexer_kv_projection_.narrow(
                                       /*dim=*/0, /*start=*/0, /*length=*/1)));
    auto score_projection =
        use_preprojected
            ? projected_score
            : kernel::musa::matmul(hidden_step,
                                   indexer_compressor_wgate_->weight(),
                                   std::nullopt,
                                   std::optional<torch::Tensor>(
                                       decode_indexer_score_projection_.narrow(
                                           /*dim=*/0,
                                           /*start=*/0,
                                           /*length=*/1)));
    auto compressed = kernel::musa::deepseek_v4_compress_decode(
        kv_state,
        score_state,
        kv_projection,
        score_projection,
        indexer_compressor_ape_,
        kCompressRatio,
        index_head_dim_,
        kCompressorCoff,
        start_pos,
        /*compute_boundary_output=*/true,
        std::optional<torch::Tensor>(decode_indexer_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/1)));
    if (!compressed.defined()) {
      return torch::Tensor();
    }
    compressed = std::get<0>(indexer_compressor_norm_->forward(
        compressed,
        std::nullopt,
        std::optional<torch::Tensor>(decode_indexer_norm_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/1))));
    if (cos.defined() && sin.defined() && cos.size(0) > 0 && sin.size(0) > 0 &&
        kernel::musa::deepseek_v4_rope_hadamard_inplace(
            compressed,
            cos.slice(/*dim=*/0, /*start=*/0, /*end=*/1),
            sin.slice(/*dim=*/0, /*start=*/0, /*end=*/1))) {
      return compressed;
    }
    if (cos.defined() && sin.defined() && cos.size(0) > 0 && sin.size(0) > 0) {
      compressed = apply_interleaved_partial_rope(
          compressed,
          index_head_dim_ - rope_head_dim_,
          rope_head_dim_,
          cos.slice(/*dim=*/0, /*start=*/0, /*end=*/1),
          sin.slice(/*dim=*/0, /*start=*/0, /*end=*/1),
          /*inverse=*/false,
          decode_indexer_rope_output_.narrow(
              /*dim=*/0, /*start=*/0, /*length=*/1));
    }
    return apply_indexer_hadamard(compressed, indexer_hadamard_);
  }
  auto kv_step =
      (use_preprojected ? projected_kv
                        : indexer_compressor_wkv_->forward(hidden_step))
          .to(torch::kFloat32)
          .contiguous();
  auto score_step =
      (use_preprojected ? projected_score
                        : indexer_compressor_wgate_->forward(hidden_step))
          .to(torch::kFloat32)
          .contiguous();
  auto ape =
      indexer_compressor_ape_.to(score_step.device()).to(torch::kFloat32);
  const int64_t ratio_pos = start_pos % kCompressRatio;
  score_step = score_step + ape.index({ratio_pos}).unsqueeze(0);
  kv_state.index_put_({kCompressRatio + ratio_pos}, kv_step.squeeze(0));
  score_state.index_put_({kCompressRatio + ratio_pos}, score_step.squeeze(0));
  if ((start_pos + 1) % kCompressRatio != 0) {
    return torch::Tensor();
  }

  auto pooled_kv =
      torch::cat({kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
                      .slice(/*dim=*/1, /*start=*/0, /*end=*/index_head_dim_),
                  kv_state.slice(/*dim=*/0, /*start=*/kCompressRatio)
                      .slice(/*dim=*/1, /*start=*/index_head_dim_)},
                 /*dim=*/0);
  auto pooled_score = torch::cat(
      {score_state.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
           .slice(/*dim=*/1, /*start=*/0, /*end=*/index_head_dim_),
       score_state.slice(/*dim=*/0, /*start=*/kCompressRatio)
           .slice(/*dim=*/1, /*start=*/index_head_dim_)},
      /*dim=*/0);
  auto weights = torch::softmax(pooled_score, /*dim=*/0);
  auto compressed =
      (pooled_kv * weights).sum(/*dim=*/0).unsqueeze(0).to(hidden_step.dtype());
  compressed = std::get<0>(indexer_compressor_norm_->forward(compressed));
  if (cos.defined() && sin.defined() && cos.size(0) > 0 && sin.size(0) > 0) {
    compressed = apply_interleaved_partial_rope(
        compressed,
        index_head_dim_ - rope_head_dim_,
        rope_head_dim_,
        cos.slice(/*dim=*/0, /*start=*/0, /*end=*/1),
        sin.slice(/*dim=*/0, /*start=*/0, /*end=*/1));
  }
  compressed = apply_indexer_hadamard(compressed, indexer_hadamard_);
  kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
      .copy_(kv_state.slice(/*dim=*/0, /*start=*/kCompressRatio));
  score_state.slice(/*dim=*/0, /*start=*/0, /*end=*/kCompressRatio)
      .copy_(score_state.slice(/*dim=*/0, /*start=*/kCompressRatio));
  return compressed;
}

torch::Tensor DeepseekV4AttentionImpl::build_indexer_topk(
    const torch::Tensor& qr,
    const torch::Tensor& hidden_states,
    const torch::Tensor& precomputed_head_weights,
    const torch::Tensor& precomputed_query,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& index_kv,
    const torch::Tensor& index_cache,
    const torch::Tensor& index_scale_cache,
    const torch::Tensor& index_block_table,
    int64_t sequence_index,
    int64_t index_len,
    int64_t q_abs_start,
    const torch::Tensor& output,
    const torch::Tensor& mate_compressed_block_table,
    const torch::Tensor& mate_swa_block_table,
    int64_t mate_kv_len,
    int64_t mate_block_size,
    torch::Tensor* mate_physical_indices,
    torch::Tensor* mate_extra_physical_indices,
    const torch::Tensor& dynamic_kv_len,
    int64_t batched_query_rows_per_sequence,
    int64_t batched_sequence_count) {
  if (!dsv4_musa_indexer_enabled() ||
      (!indexer_wq_b_ && !q_indexer_b_weight_loaded_ &&
       !precomputed_query.defined()) ||
      !indexer_weights_proj_ || qr.numel() == 0 || index_len <= 0) {
    return torch::Tensor();
  }
  CHECK_EQ(qr.size(0), hidden_states.size(0));

  const int64_t q_tokens = qr.size(0);
  const bool batched_graph_indexer = dynamic_kv_len.defined() && q_tokens > 2 &&
                                     dsv4_musa_batched_graph_indexer_enabled();
  const bool use_fused_topk =
      dsv4_musa_fused_indexer_topk_enabled() &&
      (q_tokens == 1 || batched_graph_indexer) && index_cache.defined() &&
      index_scale_cache.defined() && index_block_table.defined();
  if (dsv4_musa_trace_enabled() && !use_fused_topk) {
    LOG_FIRST_N(WARNING, 1)
        << "[DSV4_INDEXER_FALLBACK] q_tokens=" << q_tokens
        << " index_cache=" << index_cache.defined()
        << " scale_cache=" << index_scale_cache.defined()
        << " block_table=" << index_block_table.defined()
        << " dynamic_kv=" << dynamic_kv_len.defined()
        << " fused=" << dsv4_musa_fused_indexer_topk_enabled();
  }
  torch::Tensor active_index_kv = index_kv;
  if (!use_fused_topk) {
    if (!active_index_kv.defined() || active_index_kv.numel() == 0) {
      active_index_kv = gather_indexer_cache(index_cache,
                                             index_scale_cache,
                                             index_block_table,
                                             sequence_index,
                                             index_len,
                                             hidden_states.scalar_type());
    }
    if (!active_index_kv.defined() || active_index_kv.numel() == 0) {
      return torch::Tensor();
    }
    CHECK_EQ(active_index_kv.size(-1), index_head_dim_);
  }
  const int64_t kv_tokens = index_len;
  torch::Tensor query;
  if (precomputed_query.defined()) {
    query = precomputed_query;
  } else if (q_indexer_b_weight_loaded_) {
    const int64_t main_q_width = n_local_heads_ * head_dim_;
    kernel::MatmulParams params;
    params.a = qr;
    params.b = q_indexer_b_weight_.slice(
        /*dim=*/0,
        /*start=*/main_q_width,
        /*end=*/main_q_width + index_n_heads_ * index_head_dim_);
    params.bias = std::nullopt;
    query = kernel::matmul(params).view(
        {q_tokens, index_n_heads_, index_head_dim_});
  } else {
    query = indexer_wq_b_->forward(qr).view(
        {q_tokens, index_n_heads_, index_head_dim_});
  }
  CHECK_EQ(query.sizes(),
           torch::IntArrayRef({q_tokens, index_n_heads_, index_head_dim_}));
  bool fused_rope_hadamard = false;
  if (q_tokens == 1 && cos.defined() && sin.defined() &&
      cos.size(0) >= q_tokens && sin.size(0) >= q_tokens) {
    fused_rope_hadamard = kernel::musa::deepseek_v4_rope_hadamard_inplace(
        query,
        cos.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens),
        sin.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens));
  } else if (q_tokens > 2 && q_tokens < 128 && cos.defined() && sin.defined() &&
             cos.size(0) >= q_tokens && sin.size(0) >= q_tokens) {
    torch::Tensor transform_output =
        torch::empty(query.sizes(), query.options());
    fused_rope_hadamard = kernel::musa::deepseek_v4_rope_hadamard_out(
        query,
        cos.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens),
        sin.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens),
        transform_output);
    if (fused_rope_hadamard) {
      query = transform_output;
    }
  }
  if (!fused_rope_hadamard && cos.defined() && sin.defined() &&
      cos.size(0) >= q_tokens && sin.size(0) >= q_tokens) {
    query = apply_interleaved_partial_rope(
        query,
        index_head_dim_ - rope_head_dim_,
        rope_head_dim_,
        cos.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens),
        sin.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens),
        /*inverse=*/false,
        /*output_workspace=*/torch::Tensor(),
        /*allow_prefill_assembly=*/q_tokens > decode_workspace_tokens_);
  }
  if (!fused_rope_hadamard) {
    query = apply_indexer_hadamard(query, indexer_hadamard_);
  }
  if (dsv4_musa_indexer_query_int8_enabled()) {
    query = dynamic_int8_roundtrip_last_dim(query);
  }
  torch::Tensor head_weights;
  if (precomputed_head_weights.defined()) {
    CHECK_EQ(precomputed_head_weights.sizes(),
             torch::IntArrayRef({q_tokens, index_n_heads_}));
    if (q_tokens == 1 && decode_indexer_head_weights_.defined()) {
      head_weights = kernel::musa::deepseek_v4_scale_to_fp32(
          precomputed_head_weights,
          indexer_score_scale_,
          std::optional<torch::Tensor>(decode_indexer_head_weights_));
    } else {
      head_weights =
          precomputed_head_weights.to(torch::kFloat32) * indexer_score_scale_;
    }
  } else if (q_tokens == 1 &&
             decode_indexer_head_weight_projection_.defined() &&
             decode_indexer_head_weights_.defined() &&
             hidden_states.scalar_type() ==
                 indexer_weights_proj_->weight().scalar_type()) {
    auto projection = kernel::musa::matmul(
        hidden_states,
        indexer_weights_proj_->weight(),
        std::nullopt,
        std::optional<torch::Tensor>(decode_indexer_head_weight_projection_));
    head_weights = kernel::musa::deepseek_v4_scale_to_fp32(
        projection,
        indexer_score_scale_,
        std::optional<torch::Tensor>(decode_indexer_head_weights_));
  } else {
    head_weights =
        indexer_weights_proj_->forward(hidden_states).to(torch::kFloat32) *
        indexer_score_scale_;
  }
  if (use_fused_topk) {
    ::xllm::kernel::DeepseekV4IndexerTopkParams params;
    params.query = query.contiguous();
    params.weights = head_weights.contiguous();
    params.cache = index_cache;
    params.scale_cache = index_scale_cache;
    // Graph capture must not perform a dtype/device conversion. The graph
    // metadata builder already stages this table on the cache device; retain
    // the alias when it is there and only migrate in eager execution.
    if (index_block_table.device() != index_cache.device()) {
      if (kernel::musa::is_stream_capturing()) {
        LOG_FIRST_N(ERROR, 1)
            << "DSV4 indexer block table is on a different device during "
               "Graph capture; refusing a dynamic copy";
        return torch::Tensor();
      }
      params.block_table =
          index_block_table.to(index_cache.device()).contiguous();
    } else {
      params.block_table = index_block_table.contiguous();
    }
    params.sequence_index = sequence_index;
    params.kv_len = index_len;
    params.topk = index_topk_;
    params.query_absolute_start = q_abs_start;
    params.compress_ratio = 4;
    if (output.defined()) {
      CHECK_EQ(output.sizes(), torch::IntArrayRef({q_tokens, index_topk_}));
      params.output = output;
    }
    const bool prepare_mate_indices = mate_physical_indices != nullptr &&
                                      mate_extra_physical_indices != nullptr &&
                                      mate_compressed_block_table.defined() &&
                                      mate_swa_block_table.defined() &&
                                      output.defined() && mate_kv_len > 0 &&
                                      mate_block_size > 0;
    if (prepare_mate_indices) {
      params.output = output;
      auto physical_output = *mate_physical_indices;
      auto extra_physical_output = *mate_extra_physical_indices;
      if (!physical_output.defined() || !extra_physical_output.defined()) {
        auto& workspace = dsv4_mate_dsa_index_workspace(output);
        if (!physical_output.defined()) {
          physical_output = workspace.physical_indices;
        }
        if (!extra_physical_output.defined()) {
          extra_physical_output = workspace.extra_physical_indices;
        }
      }
      params.mate_compressed_block_table =
          mate_compressed_block_table.contiguous();
      params.mate_swa_block_table = mate_swa_block_table.contiguous();
      params.mate_kv_len = mate_kv_len;
      params.mate_window_size = window_size_;
      params.mate_block_size = mate_block_size;
      params.mate_physical_indices = physical_output;
      params.mate_extra_physical_indices = extra_physical_output;
      *mate_physical_indices = physical_output;
      *mate_extra_physical_indices = extra_physical_output;
    }
    torch::Tensor selected;
    const char* single_gemm_env = std::getenv("XLLM_DSV4_SINGLE_INDEXER_GEMM");
    const bool single_gemm =
        single_gemm_env != nullptr && std::string(single_gemm_env) == "1" &&
        batched_sequence_count <= 1 && dynamic_kv_len.defined() &&
        dynamic_kv_len.numel() == q_tokens && prepare_mate_indices &&
        params.query.scalar_type() == torch::kBFloat16 &&
        dsv4_single_indexer_gemm_shape(q_tokens,
                                       params.query.size(1),
                                       params.query.size(2),
                                       index_len,
                                       index_topk_);
    if (single_gemm) {
      // The helper expects request-local tables. Preserve the real owner
      // instead of implicitly using row zero of a shared block table.
      CHECK_GE(sequence_index, 0);
      CHECK_LT(sequence_index, params.block_table.size(0));
      CHECK_LT(sequence_index, params.mate_compressed_block_table.size(0));
      CHECK_LT(sequence_index, params.mate_swa_block_table.size(0));
      selected = dsv4_batched_indexer_gemm(
          params.query,
          params.weights,
          params.cache,
          params.scale_cache,
          params.block_table.narrow(0, sequence_index, 1),
          q_tokens,
          index_len,
          index_topk_,
          params.output.value(),
          params.mate_compressed_block_table.narrow(0, sequence_index, 1),
          params.mate_swa_block_table.narrow(0, sequence_index, 1),
          window_size_,
          mate_block_size,
          params.mate_physical_indices.value(),
          params.mate_extra_physical_indices.value(),
          dynamic_kv_len);
      LOG_FIRST_N(INFO, 1) << "[DSV4_SINGLE_INDEXER_GEMM] rows=" << q_tokens
                           << " capacity=" << index_len
                           << " sequence_index=" << sequence_index;
    } else if (dynamic_kv_len.defined() && batched_sequence_count > 1) {
      CHECK_EQ(batched_query_rows_per_sequence * batched_sequence_count,
               q_tokens);
      const char* gemm_env = std::getenv("XLLM_DSV4_BATCH_INDEXER_GEMM");
      const bool use_gemm = gemm_env != nullptr && std::string(gemm_env) == "1";
      if (use_gemm) {
        selected = dsv4_batched_indexer_gemm(
            params.query,
            params.weights,
            params.cache,
            params.scale_cache,
            params.block_table,
            batched_query_rows_per_sequence,
            index_len,
            index_topk_,
            params.output.value(),
            params.mate_compressed_block_table,
            params.mate_swa_block_table,
            window_size_,
            mate_block_size,
            params.mate_physical_indices.value(),
            params.mate_extra_physical_indices.value(),
            dynamic_kv_len);
        LOG_FIRST_N(INFO, 1)
            << "[DSV4_BATCH_INDEXER_GEMM] sequences=" << batched_sequence_count
            << " rows=" << q_tokens << " capacity=" << index_len;
      } else {
        selected = kernel::musa::deepseek_v4_indexer_topk_graph_batched(
            params.query,
            params.weights,
            params.cache,
            params.scale_cache,
            params.block_table,
            batched_query_rows_per_sequence,
            index_len,
            index_topk_,
            params.output.value(),
            params.mate_compressed_block_table,
            params.mate_swa_block_table,
            window_size_,
            mate_block_size,
            params.mate_physical_indices.value(),
            params.mate_extra_physical_indices.value(),
            dynamic_kv_len);
      }
    } else if (dynamic_kv_len.defined()) {
      selected = kernel::musa::deepseek_v4_indexer_topk_graph(
          params.query,
          params.weights,
          params.cache,
          params.scale_cache,
          params.block_table,
          sequence_index,
          index_len,
          index_topk_,
          params.output.value(),
          params.mate_compressed_block_table,
          params.mate_swa_block_table,
          window_size_,
          mate_block_size,
          params.mate_physical_indices.value(),
          params.mate_extra_physical_indices.value(),
          dynamic_kv_len);
    } else {
      selected = ::xllm::kernel::deepseek_v4_indexer_topk(params);
    }
    if (dsv4_musa_mate_fused_dsa_trace_enabled()) {
      LOG(INFO) << "[DSV4_MUSA_FUSED_INDEXER] layer=" << layer_id_
                << " q_tokens=" << q_tokens << " kv_tokens=" << kv_tokens
                << " topk=" << index_topk_;
    }
    return selected;
  }
  constexpr int64_t max_reference_score_bytes = 64 * 1024 * 1024;
  if (q_tokens > decode_workspace_tokens_ &&
      q_tokens * index_n_heads_ * kv_tokens * int64_t{sizeof(float)} >
          max_reference_score_bytes) {
    LOG_FIRST_N(INFO, 2) << "[DSV4_TILED_PREFILL_INDEXER] query=" << q_tokens
                         << " heads=" << index_n_heads_ << " keys=" << kv_tokens
                         << " scratch_limit_bytes="
                         << max_reference_score_bytes;
    const char* prefill_gemm_env =
        std::getenv("XLLM_DSV4_PREFILL_INDEXER_BF16_FP32");
    const bool prefill_bf16_fp32 =
        prefill_gemm_env != nullptr && std::string(prefill_gemm_env) == "1" &&
        query.scalar_type() == torch::kBFloat16 &&
        active_index_kv.scalar_type() == torch::kBFloat16 &&
        dsv4_prefill_indexer_bf16_fp32_shape(
            q_tokens, index_n_heads_, kv_tokens, index_head_dim_);
    if (prefill_bf16_fp32) {
      LOG_FIRST_N(INFO, 2) << "[DSV4_PREFILL_INDEXER_BF16_FP32] query="
                           << q_tokens << " heads=" << index_n_heads_
                           << " keys=" << kv_tokens
                           << " checked_bmm_out=1 fp32_scores=1";
    }
    const char* mask_once_env =
        std::getenv("XLLM_DSV4_PREFILL_INDEXER_MASK_ONCE");
    const bool prepare_masks_once =
        prefill_bf16_fp32 && q_tokens <= 2048 && mask_once_env != nullptr &&
        std::string(mask_once_env) == "1" && q_abs_start >= 0 &&
        !kernel::musa::is_stream_capturing();
    if (prepare_masks_once) {
      LOG_FIRST_N(INFO, 2) << "[DSV4_PREFILL_INDEXER_MASK_ONCE] q=" << q_tokens
                           << " keys=" << kv_tokens << " mask_bytes="
                           << q_tokens *
                                  (kv_tokens + std::min(index_topk_, kv_tokens))
                           << " per_call=1 unchanged_score_topk=1";
    }
    return dsv4_tiled_indexer_topk(query,
                                   active_index_kv,
                                   head_weights,
                                   q_abs_start,
                                   index_topk_,
                                   max_reference_score_bytes,
                                   prefill_bf16_fp32,
                                   prepare_masks_once);
  }
  auto scores =
      torch::matmul(query.to(torch::kFloat32),
                    active_index_kv.to(torch::kFloat32).transpose(0, 1));
  scores = torch::relu(scores);
  scores = (scores * head_weights.unsqueeze(-1)).sum(/*dim=*/1);

  // Keep the validity vector on-device. Constructing a host vector and
  // torch::tensor(..., device) here performs a host-to-device copy during
  // Graph capture and aborts on MUSA for multi-sequence q7.
  auto valid =
      torch::arange(
          q_tokens,
          torch::TensorOptions().dtype(torch::kLong).device(scores.device()))
          .add(q_abs_start + 1)
          .div(4, /*rounding_mode=*/"floor")
          .clamp(0, kv_tokens);
  auto key_positions = torch::arange(
      kv_tokens,
      torch::TensorOptions().dtype(torch::kLong).device(scores.device()));
  scores.masked_fill_(key_positions.unsqueeze(0).ge(valid.unsqueeze(1)),
                      -std::numeric_limits<float>::infinity());

  const int64_t selected_count = std::min<int64_t>(index_topk_, kv_tokens);
  auto selected = std::get<1>(scores.topk(selected_count,
                                          /*dim=*/-1,
                                          /*largest=*/true,
                                          /*sorted=*/true))
                      .to(torch::kInt32);
  auto valid_ranks =
      torch::arange(
          selected_count,
          torch::TensorOptions().dtype(torch::kLong).device(scores.device()))
          .unsqueeze(0)
          .lt(valid.clamp_max(selected_count).unsqueeze(1));
  selected =
      torch::where(valid_ranks, selected, torch::full_like(selected, -1));
  if (selected_count < index_topk_) {
    selected = torch::cat(
        {selected,
         torch::full(
             {q_tokens, index_topk_ - selected_count}, -1, selected.options())},
        /*dim=*/1);
  }
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_INDEXER] layer=" << layer_id_
              << " q_tokens=" << q_tokens << " kv_tokens=" << kv_tokens
              << " topk=" << index_topk_;
  }
  return selected.contiguous();
}

using Dsv4LegacyForwardResult =
    std::tuple<torch::Tensor, std::optional<torch::Tensor>>;
using Dsv4LegacyForwardFn = Dsv4LegacyForwardResult (*)(
    DeepseekV4AttentionImpl*,
    const DSAMetadata&,
    torch::Tensor&,
    KVCache&,
    KVState&,
    bool,
    bool,
    const std::
        tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&);

// The symbol is intentionally weak: normal builds have no legacy object and
// therefore always use forward_current_impl.  A controlled A/B build can
// provide the strong symbol from the historical Attention object and route
// only Graph target-q6 through it.
extern "C" Dsv4LegacyForwardResult
dsv4_legacy_graph_forward(DeepseekV4AttentionImpl*, const DSAMetadata&, torch::Tensor&, KVCache&, KVState&, bool, bool, const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&) __asm__(
    "_ZN4xllm5layer23DeepseekV4AttentionImpl20forward_legacy_graphERKNS0_"
    "11DSAMetadataERN2at6TensorERNS_7KVCacheERSt5tupleIJS6_S6_S6_S6_S6_"
    "EEbbRKSA_IJS6_S6_S6_S6_EE") __attribute__((weak));

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
DeepseekV4AttentionImpl::forward(
    const DSAMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    KVCache& kv_cache,
    KVState& kv_state,
    bool is_prefill,
    bool is_chunked_prefill,
    const std::
        tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&
            compress_metadata) {
  const bool graph_target_q6 = attn_metadata.is_acl_graph && !is_prefill &&
                               !is_chunked_prefill && hidden_states.defined() &&
                               hidden_states.dim() > 0 &&
                               hidden_states.size(0) > 2;
  const char* legacy_graph_env =
      std::getenv("XLLM_DSV4_ENABLE_LEGACY_GRAPH_Q6");
  const bool enable_legacy_graph_q6 =
      legacy_graph_env != nullptr && std::string(legacy_graph_env) == "1";
  const auto legacy_fn =
      reinterpret_cast<Dsv4LegacyForwardFn>(dsv4_legacy_graph_forward);
  if (graph_target_q6 && enable_legacy_graph_q6 && legacy_fn != nullptr) {
    return legacy_fn(this,
                     attn_metadata,
                     hidden_states,
                     kv_cache,
                     kv_state,
                     is_prefill,
                     is_chunked_prefill,
                     compress_metadata);
  }
  const int64_t width = attn_metadata.num_speculative_tokens + 1;
  if (dsv4_graph_sequence_views_enabled() && graph_target_q6 &&
      dspark_block_size_ == 0 && width > 1 && hidden_states.size(0) > width) {
    CHECK_EQ(hidden_states.size(0) % width, 0);
    const int64_t sequences = hidden_states.size(0) / width;
    const auto* host_lengths = dsv4_musa_host_sequence_lengths(attn_metadata);
    CHECK(host_lengths != nullptr);
    CHECK_EQ(host_lengths->size(1), sequences);
    const auto* host_q = host_lengths->data_ptr<int32_t>();
    const auto* host_kv = host_q + sequences;
    const char* batch_output_env =
        std::getenv("XLLM_DSV4_BATCH_OUTPUT_PROJECTION");
    const bool batch_output_projection =
        batch_output_env != nullptr && std::string(batch_output_env) == "1";
    const char* batch_qkv_env = std::getenv("XLLM_DSV4_BATCH_QKV_PROJECTION");
    const bool batch_qkv_projection =
        batch_qkv_env != nullptr && std::string(batch_qkv_env) == "1" &&
        qkv_a_weight_loaded_ && hidden_states.dim() == 2 &&
        decode_qkv_a_output_.defined() &&
        hidden_states.size(0) <= decode_qkv_a_output_.size(0);
    const char* batch_attention_env =
        std::getenv("XLLM_DSV4_BATCH_ATTENTION_PROJECTIONS");
    // The direct multi-sequence path is useful even when the optional packed
    // qkv scratch is unavailable (for example, a candidate that reuses an
    // older model-loader archive).  forward_current_impl already owns the
    // regular flattened q/kv projections; requiring batch_qkv_projection here
    // silently fell back to one sequence view and prevented cross-request
    // MATE fusion.  Keep the packed projection as an optimization, not as a
    // prerequisite for the multi-sequence state/attention path.
    const bool batch_attention_projections =
        batch_output_projection && batch_attention_env != nullptr &&
        std::string(batch_attention_env) == "1";
    if (batch_attention_projections) {
      // Keep one forward_current_impl invocation for the physical batch. It
      // still iterates request ranges for cache/state identity, but q/kv,
      // indexer metadata, MATE dispatch and output projection can share the
      // batched token carrier. The old per-sequence view remains the default
      // and is used whenever this opt-in path is not explicitly requested.
      SequenceViewScope sequence_view_scope(/*defer_output_projection=*/false,
                                            /*defer_o_a_projection=*/false);
      DSAMetadata batched_metadata = attn_metadata;
      // The direct multi-sequence path still processes each request range
      // independently for C4/C128 boundary/state semantics. Carry the same
      // private graph-view marker used by the proven per-sequence path so
      // compressed strides and persistent state rows are not interpreted as a
      // single flattened 112-token request.
      batched_metadata.cp_input_dict[kDsv4GraphSequenceView] = torch::Tensor();
      // Precompute the row-independent projections before entering the direct
      // multi-sequence implementation. The old early return skipped this
      // block entirely, so the path fused MATE but still repeated QKV/QB and
      // compressor projections inside every layer/request view.
      if (batch_qkv_projection) {
        kernel::MatmulParams params;
        params.a = packed_projection_input(hidden_states);
        params.b = qkv_a_weight_;
        params.bias = std::nullopt;
        params.c = torch::empty({hidden_states.size(0), qkv_a_weight_.size(0)},
                                hidden_states.options());
        auto batched_qkv = kernel::matmul(params);
        batched_metadata.cp_input_dict[kDsv4BatchedQkvProjectionMetadataKey] =
            batched_qkv;
        auto q_down = batched_qkv.narrow(1, 0, q_lora_rank_).contiguous();
        auto qr = std::get<0>(q_layernorm_->forward(q_down));
        batched_metadata.cp_input_dict[kDsv4BatchedQrMetadataKey] = qr;
        torch::Tensor qb;
        if (q_indexer_b_weight_loaded_) {
          kernel::MatmulParams qb_params;
          qb_params.a = packed_projection_input(qr);
          qb_params.b = q_indexer_b_weight_;
          qb_params.bias = std::nullopt;
          qb = kernel::matmul(qb_params);
        } else {
          qb = q_b_proj_->forward(qr);
        }
        batched_metadata.cp_input_dict[kDsv4BatchedQbMetadataKey] = qb;
        if (qkv_a_indexer_weight_loaded_ && packed_fp8_activations_enabled()) {
          batched_metadata.cp_input_dict[kDsv4BatchedHeadWeightsMetadataKey] =
              indexer_weights_proj_->forward(hidden_states);
        }
        const char* joint_env =
            std::getenv("XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION");
        if (compress_ratio_ == 4 && joint_env != nullptr &&
            std::string(joint_env) == "1" &&
            sp_joint_compressor_projection_weight_.defined()) {
          batched_metadata.cp_input_dict[kDsv4BatchedCompressorMetadataKey] =
              dsv4_pack_compressor_projections(
                  hidden_states,
                  width,
                  compressor_coff_ * head_dim_,
                  2 * index_head_dim_,
                  [this](const torch::Tensor& input) {
                    return compressor_wkv_->forward(input);
                  },
                  [this](const torch::Tensor& input) {
                    return compressor_wgate_->forward(input);
                  },
                  [this](const torch::Tensor& input) {
                    return indexer_compressor_wkv_->forward(input);
                  },
                  [this](const torch::Tensor& input) {
                    return indexer_compressor_wgate_->forward(input);
                  });
        } else if (compress_ratio_ == 128 && compressor_wkv_ &&
                   compressor_wgate_) {
          auto kv_steps = kernel::musa::matmul(hidden_states,
                                               compressor_wkv_->weight(),
                                               std::nullopt,
                                               std::nullopt);
          auto score_steps = kernel::musa::matmul(hidden_states,
                                                  compressor_wgate_->weight(),
                                                  std::nullopt,
                                                  std::nullopt);
          batched_metadata.cp_input_dict[kDsv4BatchedCompressorMetadataKey] =
              torch::cat({kv_steps, score_steps}, 1);
        }
      }
      return forward_current_impl(batched_metadata,
                                  hidden_states,
                                  kv_cache,
                                  kv_state,
                                  is_prefill,
                                  is_chunked_prefill,
                                  compress_metadata);
    }
    SequenceViewScope sequence_view_scope(batch_output_projection,
                                          batch_attention_projections);
    torch::Tensor batched_qkv;
    if (batch_qkv_projection) {
      kernel::MatmulParams params;
      params.a = packed_projection_input(hidden_states);
      params.b = qkv_a_weight_;
      params.bias = std::nullopt;
      // Do not use the layer's B1 scratch: every row view must remain valid
      // while subsequent sequences reuse all their ordinary decode buffers.
      params.c = torch::empty({hidden_states.size(0), qkv_a_weight_.size(0)},
                              hidden_states.options());
      batched_qkv = kernel::matmul(params);
    }
    // These operations are row-independent. Keep their outputs separate from
    // B1 scratch because every request-local state path reuses that scratch.
    // All state gather/update/scatter and causal masks still run per request.
    std::vector<std::pair<const char*, torch::Tensor>> batched_projections;
    batched_projections.reserve(4);
    if (batch_attention_projections) {
      auto q_down = batched_qkv.narrow(1, 0, q_lora_rank_).contiguous();
      auto qr = std::get<0>(q_layernorm_->forward(q_down));
      batched_projections.emplace_back(kDsv4BatchedQrMetadataKey, qr);
      torch::Tensor qb;
      if (q_indexer_b_weight_loaded_) {
        kernel::MatmulParams params;
        params.a = packed_projection_input(qr);
        params.b = q_indexer_b_weight_;
        params.bias = std::nullopt;
        qb = kernel::matmul(params);
      } else {
        qb = q_b_proj_->forward(qr);
      }
      batched_projections.emplace_back(kDsv4BatchedQbMetadataKey, qb);
      if (qkv_a_indexer_weight_loaded_ && packed_fp8_activations_enabled()) {
        batched_projections.emplace_back(
            kDsv4BatchedHeadWeightsMetadataKey,
            indexer_weights_proj_->forward(hidden_states));
      }
      const char* joint_env =
          std::getenv("XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION");
      if (compress_ratio_ == 4 && joint_env != nullptr &&
          std::string(joint_env) == "1" &&
          sp_joint_compressor_projection_weight_.defined()) {
        batched_projections.emplace_back(
            kDsv4BatchedCompressorMetadataKey,
            dsv4_pack_compressor_projections(
                hidden_states,
                width,
                compressor_coff_ * head_dim_,
                2 * index_head_dim_,
                [this](const torch::Tensor& input) {
                  return compressor_wkv_->forward(input);
                },
                [this](const torch::Tensor& input) {
                  return compressor_wgate_->forward(input);
                },
                [this](const torch::Tensor& input) {
                  return indexer_compressor_wkv_->forward(input);
                },
                [this](const torch::Tensor& input) {
                  return indexer_compressor_wgate_->forward(input);
                }));
      } else if (compress_ratio_ == 128 && compressor_wkv_ &&
                 compressor_wgate_) {
        auto kv_steps = kernel::musa::matmul(hidden_states,
                                             compressor_wkv_->weight(),
                                             std::nullopt,
                                             std::nullopt);
        auto score_steps = kernel::musa::matmul(hidden_states,
                                                compressor_wgate_->weight(),
                                                std::nullopt,
                                                std::nullopt);
        batched_projections.emplace_back(
            kDsv4BatchedCompressorMetadataKey,
            torch::cat({kv_steps, score_steps}, 1));
      }
    }
    std::vector<torch::Tensor> sequence_outputs;
    sequence_outputs.reserve(sequences);
    for (int64_t seq = 0; seq < sequences; ++seq) {
      CHECK_EQ(host_q[seq], width)
          << "No padded or ragged row may become a request";
      DSAMetadata row = attn_metadata;
      row.cp_input_dict[kDsv4HostSequenceLengthsMetadataKey] =
          host_lengths->narrow(1, seq, 1).contiguous();
      // The row-local block table is intentionally narrowed to row zero, but
      // recurrent compressor/indexer state is owned by the original request
      // slot. Carry that identity separately so a sequence-view replay does
      // not load sequence zero's state for every physical request.
      row.cp_input_dict[kDsv4GraphSequenceIndexMetadataKey] = torch::tensor(
          {static_cast<int32_t>(seq)},
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
      row.cp_input_dict[kDsv4GraphSequenceView] = torch::Tensor();
      if (batched_qkv.defined()) {
        row.cp_input_dict[kDsv4BatchedQkvProjectionMetadataKey] =
            batched_qkv.narrow(
                /*dim=*/0, /*start=*/seq * width, /*length=*/width);
      }
      for (const auto& [key, projection] : batched_projections) {
        row.cp_input_dict[key] = projection.narrow(0, seq * width, width);
      }
      row.max_query_len = width;
      row.max_seq_len = host_kv[seq];
      const auto slice =
          [](torch::Tensor& tensor, int64_t start, int64_t count) {
            if (tensor.defined()) {
              CHECK_GT(tensor.dim(), 0);
              CHECK_GE(tensor.size(0), start + count);
              tensor = tensor.narrow(0, start, count);
            }
          };
      for (auto* tensor : {&row.seq_lens,
                           &row.seq_lens_q,
                           &row.actual_seq_lengths_kv,
                           &row.start_pos}) {
        slice(*tensor, seq, 1);
      }
      // Compact verify uses the single-sequence KV carrier above. The
      // cumulative query tensor is retained only as a request-local view.
      slice(row.actual_seq_lengths_query, seq, 2);
      for (auto* tensor : {&row.cos,
                           &row.sin,
                           &row.kv_cos,
                           &row.kv_sin,
                           &row.c4_input_cos,
                           &row.c4_input_sin,
                           &row.c128_input_cos,
                           &row.c128_input_sin,
                           &row.input_positions}) {
        slice(*tensor, seq * width, width);
      }
      const int64_t c4_stride = dsv4_graph_compressed_stride(width, 4);
      const int64_t c128_stride = dsv4_graph_compressed_stride(width, 128);
      for (auto* tensor : {&row.c4_cos, &row.c4_sin, &row.c4_pad_positions}) {
        slice(*tensor, seq * c4_stride, c4_stride);
      }
      for (auto* tensor :
           {&row.c128_cos, &row.c128_sin, &row.c128_pad_positions}) {
        slice(*tensor, seq * c128_stride, c128_stride);
      }
      CHECK(row.caches_info != nullptr);
      for (size_t ci = 0; ci < row.block_tables[layer_id_].size(); ++ci) {
        slice(row.block_tables[layer_id_][ci], seq, 1);
        const auto& info = (*row.caches_info)[layer_id_][ci];
        const int64_t stride =
            info.type == DSACacheType::TOKEN
                ? dsv4_graph_compressed_stride(width, info.ratio)
                : width;
        slice(row.slot_mappings[layer_id_][ci], seq * stride, stride);
      }
      auto hidden_row = hidden_states.narrow(0, seq * width, width);
      auto result = forward_current_impl(
          row, hidden_row, kv_cache, kv_state, false, false, compress_metadata);
      // Each invocation reuses the layer's B1 scratch/output buffers. Snapshot
      // the result on-device before the next sequence can overwrite it.
      sequence_outputs.push_back(std::get<0>(result).clone());
    }
    LOG_FIRST_N(INFO, 1) << "[DSV4_GRAPH_SEQUENCE_VIEWS] physical_sequences="
                         << sequences << " query_width=" << width;
    if (batch_qkv_projection) {
      LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_QKV_PROJECTION] sequences="
                           << sequences << " tokens=" << hidden_states.size(0);
    }
    auto output = torch::cat(sequence_outputs, /*dim=*/0);
    if (batch_attention_projections) {
      auto grouped = output.view({output.size(0), n_local_groups_, -1});
      if (n_local_groups_ == 1) {
        output = torch::mm(grouped.select(1, 0),
                           o_a_projection_weight_.select(0, 0).transpose(0, 1));
      } else {
        output =
            torch::einsum("tgd,grd->tgr", {grouped, o_a_projection_weight_})
                .reshape({output.size(0), -1});
      }
      LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_ATTENTION_PROJECTIONS] sequences="
                           << sequences << " tokens=" << hidden_states.size(0);
    }
    if (batch_output_projection) {
      // Each sequence keeps its own state/cache/attention mapping and copied
      // low-rank result. wo_b is row-independent: execute it once, then perform
      // one TP collective for all live rows instead of one per sequence.
      CHECK_EQ(output.dim(), 2);
      CHECK_EQ(output.size(0), hidden_states.size(0));
      output = o_b_proj_->forward(output);
      ProcessGroup* process_group = o_b_proj_->process_group();
      if (process_group != nullptr && process_group->world_size() > 1) {
        std::optional<torch::Tensor> output_workspace;
        if (decode_o_b_reduce_output_.defined() &&
            output.size(0) <= decode_o_b_reduce_output_.size(0)) {
          output_workspace = decode_o_b_reduce_output_.narrow(
              /*dim=*/0, /*start=*/0, /*length=*/output.size(0));
        }
        if (auto reduced = musa_allreduce_custom(
                process_group, output, output_workspace)) {
          output = *reduced;
        } else {
          process_group->allreduce(output);
        }
      }
      LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_OUTPUT_PROJECTION] sequences="
                           << sequences << " tokens=" << output.size(0);
    }
    return {output, std::nullopt};
  }
  return forward_current_impl(attn_metadata,
                              hidden_states,
                              kv_cache,
                              kv_state,
                              is_prefill,
                              is_chunked_prefill,
                              compress_metadata);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
DeepseekV4AttentionImpl::forward_current_impl(
    const DSAMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    KVCache& kv_cache,
    KVState& kv_state,
    bool is_prefill,
    bool is_chunked_prefill,
    const std::
        tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&
            compress_metadata) {
  (void)compress_metadata;

  // Ordinary prefill starts a fresh request.  Clear all live recurrent
  // compressor/indexer state before route selection, including the full-KV
  // (short-context) path where compressed attention is intentionally bypassed.
  // Otherwise a later decode could reuse tensors left by a previous request
  // after its prompt stayed below the full-KV threshold. This is a defensive
  // request-isolation reset, not a complete GSM8K accuracy fix. Prefix
  // snapshots are also cleared: a short full-KV request can legitimately
  // bypass compress_prefill_sequence, so retaining a snapshot by sequence
  // slot would let a later cached-prefix request restore another request's
  // state.
  const bool starts_new_request =
      !is_chunked_prefill || dsv4_musa_starts_new_request(attn_metadata);
  if (is_prefill && starts_new_request) {
    compressor_kv_history_.clear();
    compressor_kv_state_.clear();
    compressor_score_state_.clear();
    indexer_kv_history_.clear();
    indexer_kv_state_.clear();
    indexer_score_state_.clear();
    prefix_compressor_kv_state_.clear();
    prefix_compressor_score_state_.clear();
    prefix_indexer_kv_state_.clear();
    prefix_indexer_score_state_.clear();
    if (dsv4_musa_clear_state_cache_on_request_enabled()) {
      const auto state_ids_it =
          attn_metadata.cp_input_dict.find(kDsv4PrefillResetBlocksMetadataKey);
      torch::Tensor active_state_blocks;
      for (const auto& cache : {std::get<1>(kv_state),
                                std::get<2>(kv_state),
                                std::get<3>(kv_state),
                                std::get<4>(kv_state)}) {
        if (cache.defined() && cache.numel() > 0) {
          if (!active_state_blocks.defined()) {
            CHECK(state_ids_it != attn_metadata.cp_input_dict.end())
                << "Eager new-request prefill needs request-owned state block "
                   "IDs";
            // Filtering is done once on the existing host metadata. MUSA
            // masked_select performs a blocking D2H size query and can wait
            // behind an outstanding collective from a faster rank.
            active_state_blocks = state_ids_it->second;
            CHECK_EQ(active_state_blocks.device(), cache.device());
            CHECK_EQ(active_state_blocks.scalar_type(), torch::kInt64);
          }
          if (active_state_blocks.numel() > 0) {
            dsv4_clear_prefill_state_rows(cache, active_state_blocks);
          }
        }
      }
      LOG(INFO)
          << "[DSV4_STATE_RESET] cleared request-owned compressor/indexer "
             "state rows for new request, layer="
          << layer_id_;
    }
    if (dsv4_musa_clear_decode_workspaces_on_request_enabled()) {
      const std::array<torch::Tensor*, 22> workspaces = {
          &decode_spec_attention_output_,
          &decode_kv_seq_lens_,
          &decode_spec_indices_output_,
          &decode_spec_mate_physical_indices_,
          &decode_spec_mate_extra_physical_indices_,
          &decode_compressor_kv_slots_,
          &decode_compressor_score_slots_,
          &decode_indexer_kv_slots_,
          &decode_indexer_score_slots_,
          &decode_compressor_kv_state_,
          &decode_compressor_score_state_,
          &decode_indexer_kv_state_,
          &decode_indexer_score_state_,
          &decode_compressor_kv_projection_,
          &decode_compressor_score_projection_,
          &decode_compressor_output_,
          &decode_compressor_norm_output_,
          &decode_compressor_rope_output_,
          &decode_indexer_kv_projection_,
          &decode_indexer_score_projection_,
          &decode_indexer_output_,
          &decode_indexer_norm_output_};
      for (torch::Tensor* workspace : workspaces) {
        if (workspace != nullptr && workspace->defined() &&
            workspace->numel() > 0) {
          workspace->zero_();
        }
      }
      if (decode_indexer_rope_output_.defined() &&
          decode_indexer_rope_output_.numel() > 0) {
        decode_indexer_rope_output_.zero_();
      }
      if (decode_indexer_head_weight_projection_.defined() &&
          decode_indexer_head_weight_projection_.numel() > 0) {
        decode_indexer_head_weight_projection_.zero_();
      }
      if (decode_indexer_head_weights_.defined() &&
          decode_indexer_head_weights_.numel() > 0) {
        decode_indexer_head_weights_.zero_();
      }
      if (decode_indexer_topk_output_.defined() &&
          decode_indexer_topk_output_.numel() > 0) {
        decode_indexer_topk_output_.zero_();
      }
      LOG(INFO) << "[DSV4_STATE_RESET] cleared decode workspaces for new "
                   "request, layer="
                << layer_id_;
    }
  }

  const dsv4_musa_sp::DeepseekV4SPContext* sp_context =
      (is_prefill || is_chunked_prefill) ? sequence_parallel_context_ : nullptr;
  if (sp_context != nullptr) {
    CHECK_EQ(hidden_states.size(0), sp_context->local_tokens)
        << "DeepSeek-V4 replicated prefill attention expects padded local "
           "hidden states";
  }
  const bool zero_local_sp_shard =
      sp_context != nullptr && sp_context->local_valid_tokens == 0;
  torch::Tensor projection_hidden_states =
      sp_context == nullptr ? hidden_states
      : zero_local_sp_shard
          ? hidden_states.narrow(/*dim=*/0, /*start=*/0, /*length=*/1)
          : hidden_states.narrow(/*dim=*/0,
                                 /*start=*/0,
                                 sp_context->local_valid_tokens);
  // The dummy projection row above keeps q/kv MUSA kernels out of the
  // zero-work launch corner. Compression must still participate in the SP
  // gather with an actually empty local shard, otherwise this rank would
  // contribute a padded dummy token to the global recurrent state.
  torch::Tensor compression_hidden_states =
      zero_local_sp_shard
          ? hidden_states.narrow(/*dim=*/0, /*start=*/0, /*length=*/0)
          : projection_hidden_states;
  if (sp_context != nullptr) {
    dump_local_prefill_stage(layer_id_, "input", projection_hidden_states);
  }

  const int32_t profile_device_id = hidden_states.device().index();
  auto* prefill_profile = begin_prefill_profile(
      layer_id_, hidden_states, is_prefill || is_chunked_prefill);

  const bool use_packed_qkv_a =
      qkv_a_weight_loaded_ && projection_hidden_states.dim() == 2 &&
      projection_hidden_states.size(0) > 0 &&
      projection_hidden_states.size(0) <= decode_qkv_a_output_.size(0);
  const torch::Tensor packed_qkv_input =
      packed_projection_input(projection_hidden_states);
  auto packed_qkv_a_matmul = [&](const torch::Tensor& weight) {
    kernel::MatmulParams params;
    params.a = packed_qkv_input;
    params.b = weight;
    params.bias = std::nullopt;
    if (decode_qkv_a_output_.defined() &&
        projection_hidden_states.size(0) <= decode_qkv_a_output_.size(0) &&
        weight.size(0) == decode_qkv_a_output_.size(1)) {
      params.c = decode_qkv_a_output_.narrow(
          /*dim=*/0,
          /*start=*/0,
          /*length=*/projection_hidden_states.size(0));
    }
    return kernel::matmul(params);
  };

  torch::Tensor q_down;
  torch::Tensor kv_down;
  torch::Tensor deferred_kv_norm_output;
  const char* projection_norm_copy_env =
      dsv4_projection_norm_copy_shape(
          projection_hidden_states.size(0), is_prefill, is_chunked_prefill)
          ? std::getenv("XLLM_DSV4_FUSED_PROJECTION_NORM_COPY")
          : nullptr;
  const bool fuse_projection_norm_copy = projection_norm_copy_env != nullptr &&
                                         projection_norm_copy_env[0] == '1' &&
                                         projection_norm_copy_env[1] == '\0';
  torch::Tensor packed_indexer_head_weights;
  if (use_packed_qkv_a) {
    const auto batched_projection =
        attn_metadata.cp_input_dict.find(kDsv4BatchedQkvProjectionMetadataKey);
    torch::Tensor qkv_down;
    if (batched_projection != attn_metadata.cp_input_dict.end()) {
      CHECK(dsv4_sequence_view_active && attn_metadata.is_acl_graph &&
            !is_prefill && !is_chunked_prefill);
      qkv_down = batched_projection->second;
      CHECK(qkv_down.defined() && qkv_down.dim() == 2 &&
            qkv_down.is_contiguous());
      CHECK_EQ(qkv_down.size(0), projection_hidden_states.size(0));
      CHECK_EQ(qkv_down.size(1), qkv_a_weight_.size(0));
    } else {
      qkv_down = packed_qkv_a_matmul(qkv_a_weight_);
    }
    if (projection_hidden_states.size(0) == 1) {
      q_down = qkv_down.slice(
          /*dim=*/1, /*start=*/0, /*end=*/q_lora_rank_);
      kv_down = qkv_down.slice(/*dim=*/1,
                               /*start=*/q_lora_rank_,
                               /*end=*/q_lora_rank_ + head_dim_);
      CHECK(q_down.is_contiguous() && kv_down.is_contiguous())
          << "T=1 packed q/kv slices must remain contiguous.";
    } else {
      auto q_down_workspace = decode_q_down_.narrow(
          /*dim=*/0,
          /*start=*/0,
          /*length=*/projection_hidden_states.size(0));
      q_down_workspace.copy_(qkv_down.slice(
          /*dim=*/1, /*start=*/0, /*end=*/q_lora_rank_));
      q_down = q_down_workspace;
      auto kv_down_workspace = decode_kv_down_.narrow(
          /*dim=*/0,
          /*start=*/0,
          /*length=*/projection_hidden_states.size(0));
      auto kv_view = qkv_down.slice(/*dim=*/1,
                                    /*start=*/q_lora_rank_,
                                    /*end=*/q_lora_rank_ + head_dim_);
      if (fuse_projection_norm_copy) {
        kv_down = kv_view;
        deferred_kv_norm_output = kv_down_workspace;
      } else {
        kv_down_workspace.copy_(kv_view);
        kv_down = kv_down_workspace;
      }
    }
    if (qkv_a_indexer_weight_loaded_) {
      packed_indexer_head_weights = qkv_down.slice(
          /*dim=*/1,
          /*start=*/q_lora_rank_ + head_dim_,
          /*end=*/q_lora_rank_ + head_dim_ + index_n_heads_);
      if (packed_fp8_activations_enabled()) {
        // The packed head-weight projection is BF16, not an FP8 linear.
        const auto batch_weights = attn_metadata.cp_input_dict.find(
            kDsv4BatchedHeadWeightsMetadataKey);
        packed_indexer_head_weights =
            batch_weights != attn_metadata.cp_input_dict.end()
                ? batch_weights->second
                : indexer_weights_proj_->forward(projection_hidden_states);
      }
    }
  } else if (qkv_a_weight_loaded_) {
    // Preserve the smaller projection shape for long prefill while sharing
    // the materialized packed storage.
    q_down = packed_qkv_a_matmul(qkv_a_weight_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/q_lora_rank_));
  } else {
    q_down = q_a_proj_->forward(projection_hidden_states);
  }
  maybe_sync_attention_stage(layer_id_, "q_a_proj", q_down);
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " after_q_a shape=" << q_down.sizes();
  }
  std::optional<torch::Tensor> q_a_norm_output = std::nullopt;
  if (!is_prefill && !is_chunked_prefill && q_down.dim() == 2 &&
      q_down.size(0) <= decode_q_a_norm_output_.size(0)) {
    q_a_norm_output = decode_q_a_norm_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/q_down.size(0));
  }
  const auto batch_qr =
      attn_metadata.cp_input_dict.find(kDsv4BatchedQrMetadataKey);
  auto qr = batch_qr != attn_metadata.cp_input_dict.end()
                ? batch_qr->second
                : std::get<0>(q_layernorm_->forward(
                      q_down, std::nullopt, q_a_norm_output));
  maybe_sync_attention_stage(layer_id_, "q_layernorm", qr);
  const int64_t rope_token_count = sp_context == nullptr
                                       ? projection_hidden_states.size(0)
                                       : sp_context->total_tokens;
  const auto rope_tables =
      select_rope_tables(attn_metadata, compress_ratio_, rope_token_count);
  const torch::Tensor& global_cos = rope_tables.first;
  const torch::Tensor& global_sin = rope_tables.second;
  torch::Tensor active_cos = global_cos;
  torch::Tensor active_sin = global_sin;
  if (sp_context != nullptr) {
    // Absolute prompt row of this shard's rows (RoPE tables cover the full
    // prompt); plain TP+SP has global_row_start == 0 so behavior is
    // unchanged there.
    const int64_t rope_start =
        sp_context->global_row_start + sp_context->local_offset;
    if (global_cos.size(0) < rope_start + sp_context->local_valid_tokens ||
        global_sin.size(0) < rope_start + sp_context->local_valid_tokens) {
      LOG(FATAL) << "[DSV4_PCP_ROPE_DIAG] cos=" << global_cos.size(0)
                 << " sin=" << global_sin.size(0)
                 << " rope_start=" << rope_start
                 << " local_valid=" << sp_context->local_valid_tokens
                 << " sp_total=" << sp_context->total_tokens
                 << " global_row_start=" << sp_context->global_row_start
                 << " layer=" << layer_id_ << " ratio=" << compress_ratio_;
    }
    CHECK_GE(global_cos.size(0), rope_start + sp_context->local_valid_tokens);
    CHECK_GE(global_sin.size(0), rope_start + sp_context->local_valid_tokens);
    if (zero_local_sp_shard) {
      CHECK_GT(global_cos.size(0), 0);
      CHECK_GT(global_sin.size(0), 0);
      active_cos = global_cos.narrow(/*dim=*/0, /*start=*/0, /*length=*/1);
      active_sin = global_sin.narrow(/*dim=*/0, /*start=*/0, /*length=*/1);
    } else {
      active_cos = global_cos.slice(
          /*dim=*/0,
          /*start=*/rope_start,
          /*end=*/rope_start + sp_context->local_valid_tokens);
      active_sin = global_sin.slice(
          /*dim=*/0,
          /*start=*/rope_start,
          /*end=*/rope_start + sp_context->local_valid_tokens);
    }
  }
  torch::Tensor q;
  torch::Tensor packed_indexer_query;
  auto compute_q_branch = [&]() {
    torch::Tensor q_projection;
    torch::Tensor deferred_q_norm_output;
    const int64_t main_q_width = n_local_heads_ * head_dim_;
    const bool use_packed_q_indexer =
        q_indexer_b_weight_loaded_ && qr.size(0) > 0 &&
        decode_q_indexer_b_output_.defined() &&
        qr.size(0) <= decode_q_indexer_b_output_.size(0);
    if (q_indexer_b_weight_loaded_) {
      kernel::MatmulParams params;
      params.a = packed_projection_input(qr);
      params.b = use_packed_q_indexer
                     ? q_indexer_b_weight_
                     : q_indexer_b_weight_.slice(
                           /*dim=*/0, /*start=*/0, /*end=*/main_q_width);
      params.bias = std::nullopt;
      if (use_packed_q_indexer) {
        params.c = decode_q_indexer_b_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/qr.size(0));
      }
      const auto batch_qb =
          attn_metadata.cp_input_dict.find(kDsv4BatchedQbMetadataKey);
      auto packed_q = batch_qb != attn_metadata.cp_input_dict.end()
                          ? batch_qb->second
                          : kernel::matmul(params);
      if (use_packed_q_indexer && qr.size(0) > 1) {
        auto q_workspace = decode_q_main_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/qr.size(0));
        auto q_view =
            packed_q.slice(/*dim=*/1, /*start=*/0, /*end=*/main_q_width);
        if (fuse_projection_norm_copy) {
          q_projection = q_view;
          deferred_q_norm_output = q_workspace;
        } else {
          q_workspace.copy_(q_view);
          q_projection = q_workspace;
        }
      } else if (use_packed_q_indexer) {
        q_projection = packed_q.slice(
            /*dim=*/1, /*start=*/0, /*end=*/main_q_width);
      } else {
        q_projection = packed_q;
      }
      if (use_packed_q_indexer) {
        packed_indexer_query =
            packed_q
                .slice(/*dim=*/1,
                       /*start=*/main_q_width,
                       /*end=*/main_q_width + index_n_heads_ * index_head_dim_)
                .view({qr.size(0), index_n_heads_, index_head_dim_});
      }
    } else {
      const auto batch_qb =
          attn_metadata.cp_input_dict.find(kDsv4BatchedQbMetadataKey);
      q_projection = batch_qb != attn_metadata.cp_input_dict.end()
                         ? batch_qb->second
                         : q_b_proj_->forward(qr);
    }
    q = q_projection.view({-1, n_local_heads_, head_dim_});
    maybe_sync_attention_stage(layer_id_, "q_b_proj", q);
    if (dsv4_musa_trace_enabled()) {
      LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
                << " after_q_b shape=" << q.sizes();
    }
    bool fused_q_norm_rope = false;
    if (deferred_q_norm_output.defined()) {
      auto q_output =
          deferred_q_norm_output.view({-1, n_local_heads_, head_dim_});
      fused_q_norm_rope = head_dim_ == 512 && nope_head_dim_ == 448 &&
                          rope_head_dim_ == 64 &&
                          kernel::musa::deepseek_v4_q_rmsnorm_rope_out(
                              q, active_cos, active_sin, eps_, q_output);
      if (fused_q_norm_rope) {
        LOG_FIRST_N(INFO, 2)
            << "[DSV4_FUSED_Q_COPY_NORM_ROPE] rows=" << q.size(0)
            << " source_row_stride=" << q.stride(0);
      } else {
        deferred_q_norm_output.copy_(q_projection);
      }
      q = q_output;
    }
    if (!fused_q_norm_rope) {
      fused_q_norm_rope = head_dim_ == 512 && nope_head_dim_ == 448 &&
                          rope_head_dim_ == 64 &&
                          kernel::musa::deepseek_v4_q_rmsnorm_rope_inplace(
                              q, active_cos, active_sin, eps_);
    }
    if (fused_q_norm_rope) {
      maybe_sync_attention_stage(layer_id_, "q_rms_norm_rope", q);
    } else {
      q = rms_norm_last_dim(q, q_rms_gamma_, eps_);
      maybe_sync_attention_stage(layer_id_, "q_rms_norm", q);
      q = apply_interleaved_partial_rope(
          q, nope_head_dim_, rope_head_dim_, active_cos, active_sin);
      maybe_sync_attention_stage(layer_id_, "q_rope", q);
    }
  };

  const bool overlap_decode_q_kv = dsv4_musa_decode_q_kv_overlap_enabled() &&
                                   use_packed_qkv_a && !is_prefill &&
                                   !is_chunked_prefill &&
                                   projection_hidden_states.size(0) == 1;
  DecodeQKvOverlapState* q_overlap_state = nullptr;
  c10::musa::MUSAStream current_stream =
      c10::musa::getCurrentMUSAStream(hidden_states.device().index());
  if (overlap_decode_q_kv) {
    q_overlap_state =
        get_decode_q_kv_overlap_state(hidden_states.device().index());
    CHECK_EQ(
        musaEventRecord(q_overlap_state->input_ready, current_stream.stream()),
        musaSuccess);
    CHECK_EQ(musaStreamWaitEvent(q_overlap_state->q_stream.stream(),
                                 q_overlap_state->input_ready,
                                 /*flags=*/0),
             musaSuccess);
    {
      c10::StreamGuard stream_guard(q_overlap_state->q_stream.unwrap());
      compute_q_branch();
      CHECK_EQ(musaEventRecord(q_overlap_state->q_ready,
                               q_overlap_state->q_stream.stream()),
               musaSuccess);
    }
  } else {
    compute_q_branch();
  }

  if (!kv_down.defined()) {
    if (qkv_a_weight_loaded_) {
      kv_down = packed_qkv_a_matmul(qkv_a_weight_.slice(
          /*dim=*/0,
          /*start=*/q_lora_rank_,
          /*end=*/q_lora_rank_ + head_dim_));
    } else {
      kv_down = kv_proj_->forward(projection_hidden_states);
    }
  }
  maybe_sync_attention_stage(layer_id_, "kv_proj", kv_down);
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " after_kv_proj shape=" << kv_down.sizes();
  }
  auto kv = kv_down.view({-1, 1, head_dim_});
  bool fused_kv_norm_rope = false;
  if (deferred_kv_norm_output.defined()) {
    auto kv_output = deferred_kv_norm_output.view({-1, 1, head_dim_});
    fused_kv_norm_rope = head_dim_ == 512 && nope_head_dim_ == 448 &&
                         rope_head_dim_ == 64 &&
                         kernel::musa::deepseek_v4_weighted_rmsnorm_rope_out(
                             kv,
                             kv_layernorm_->weight(),
                             active_cos,
                             active_sin,
                             eps_,
                             kv_output);
    if (fused_kv_norm_rope) {
      LOG_FIRST_N(INFO, 2) << "[DSV4_FUSED_KV_COPY_NORM_ROPE] rows="
                           << kv.size(0)
                           << " source_row_stride=" << kv.stride(0);
    } else {
      deferred_kv_norm_output.copy_(kv_down);
    }
    kv_down = deferred_kv_norm_output;
    kv = kv_output;
  }
  if (!fused_kv_norm_rope) {
    fused_kv_norm_rope =
        head_dim_ == 512 && nope_head_dim_ == 448 && rope_head_dim_ == 64 &&
        kernel::musa::deepseek_v4_weighted_rmsnorm_rope_inplace(
            kv, kv_layernorm_->weight(), active_cos, active_sin, eps_);
  }
  if (fused_kv_norm_rope) {
    maybe_sync_attention_stage(layer_id_, "kv_rms_norm_rope", kv);
  } else {
    kv = std::get<0>(kv_layernorm_->forward(kv_down)).view({-1, 1, head_dim_});
    maybe_sync_attention_stage(layer_id_, "kv_layernorm", kv);
    kv = apply_interleaved_partial_rope(
        kv, nope_head_dim_, rope_head_dim_, active_cos, active_sin);
    maybe_sync_attention_stage(layer_id_, "kv_rope", kv);
  }
  fp8_act_quant_roundtrip_inplace(kv, nope_head_dim_, /*block_size=*/64);
  maybe_sync_attention_stage(layer_id_, "kv_fp8_roundtrip", kv);
  if (sp_context != nullptr) {
    if (zero_local_sp_shard) {
      // q/kv projection used one dummy row to avoid zero-work kernels; only
      // the valid KV rows may enter the replicated prefill gather.
      kv = kv.narrow(/*dim=*/0, /*start=*/0, /*length=*/0);
    }
    const auto& gather_context =
        pcp_context_ != nullptr && pcp_context_->global_sp.has_value()
            ? pcp_context_->global_sp.value()
            : *sp_context;
    kv = dsv4_musa_sp::gather_global(
        dsv4_musa_sp::pad_local(kv, gather_context), gather_context);
    maybe_sync_attention_stage(layer_id_, "sp_kv_all_gather", kv);
  }

  auto ori_kv = kv_cache.get_swa_cache();
  auto mate_swa_cache = dsv4_musa_block_tensor(
      kv_cache, BlockType::SWA, KVCacheTensorRole::VALUE);
  const int64_t swa_cache_idx = find_swa_cache_index(attn_metadata, layer_id_);
  auto swa_slot_mapping =
      layer_cache_tensor(attn_metadata.slot_mappings, layer_id_, swa_cache_idx);
  auto swa_block_table =
      layer_cache_tensor(attn_metadata.block_tables, layer_id_, swa_cache_idx);
  const bool trace_layer0 =
      (dsv4_musa_trace_enabled() || dsv4_musa_cache_trace_enabled()) &&
      layer_id_ == 0;
  if (trace_layer0) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] layer0 is_prefill=" << is_prefill
              << " is_chunked_prefill=" << is_chunked_prefill
              << " q_tokens=" << q.size(0) << " kv_tokens=" << kv.size(0)
              << " swa_cache_idx=" << swa_cache_idx
              << " swa_cache_shape=" << tensor_sizes_string(ori_kv)
              << " slot_shape=" << tensor_sizes_string(swa_slot_mapping)
              << " slot_prefix=" << tensor_i64_prefix(swa_slot_mapping)
              << " bt_shape=" << tensor_sizes_string(swa_block_table)
              << " bt_prefix=" << tensor_i64_prefix(swa_block_table);
  }
  if (pcp_context_ != nullptr && !pcp_context_->global_sp.has_value() &&
      ori_kv.defined() && (is_prefill || is_chunked_prefill)) {
    kv = dsv4_musa_pcp::relay_kv(kv, *pcp_context_);
  }
  if (ori_kv.defined()) {
    scatter_by_slot(ori_kv, swa_slot_mapping, kv);
    maybe_sync_attention_stage(layer_id_, "swa_scatter_by_slot", kv);
  }
  const bool skip_mate_swa_graph_decode_pack =
      !is_prefill && attn_metadata.is_acl_graph &&
      !((compress_ratio_ == 4 ||
         (compress_ratio_ == 128 && dsv4_musa_c128_mate_enabled())) &&
        q.size(0) > 2 && dsv4_musa_graph_batched_mate_verify_enabled() &&
        dsv4_musa_mate_sparse_decode_enabled());
  // The old Graph decode path consumed the BF16 paged SWA cache and could
  // omit MODEL1 packing. Batched C4 verify consumes mate_swa_cache instead:
  // scatter_by_slot above updates a DIFFERENT tensor. Pack the current target
  // rows on this path, including proposals; causal per-row lengths hide later
  // rows. DSpark's accepted-context writeback updates the draft cache, not this
  // target MODEL1 cache. Leaving it frozen reuses prefill/previous-request KV.
  if (mate_swa_cache.defined() && !skip_mate_swa_graph_decode_pack) {
    pack_mate_model1_cache(mate_swa_cache, swa_slot_mapping, kv);
    maybe_sync_attention_stage(layer_id_, "mate_swa_pack", kv);
  }

  if (overlap_decode_q_kv) {
    CHECK_EQ(musaStreamWaitEvent(current_stream.stream(),
                                 q_overlap_state->q_ready,
                                 /*flags=*/0),
             musaSuccess);
    LOG_FIRST_N(INFO, 1)
        << "DeepSeek-V4 MUSA decode Q projection/rope overlaps KV "
           "projection/rope/cache.";
  }

  const int64_t total_tokens = q.size(0);
  torch::Tensor graph_decode_kv_seq_lens;
  if (attn_metadata.is_acl_graph && attn_metadata.input_positions.defined() &&
      attn_metadata.input_positions.dim() == 1 &&
      attn_metadata.input_positions.size(0) >= total_tokens &&
      decode_kv_seq_lens_.defined() &&
      decode_kv_seq_lens_.size(0) >= total_tokens) {
    graph_decode_kv_seq_lens = decode_kv_seq_lens_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/total_tokens);
    graph_decode_kv_seq_lens.copy_(attn_metadata.input_positions.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/total_tokens));
    graph_decode_kv_seq_lens.add_(1);
  }
  auto query_ranges = build_query_ranges(attn_metadata, total_tokens);
  if (sp_context != nullptr) {
    // Scheduler metadata still describes the global prefill.  Attention on
    // each TP rank only owns the valid (unpadded) rows of its local shard.
    query_ranges = {{0, sp_context->local_valid_tokens}};
  }
  std::vector<std::pair<int64_t, int64_t>> compression_query_ranges =
      query_ranges;
  if (sp_context != nullptr) {
    compression_query_ranges = {{0, sp_context->total_tokens}};
  }
  const bool graph_has_one_sequence_kv =
      (attn_metadata.actual_seq_lengths_kv.defined() &&
       attn_metadata.actual_seq_lengths_kv.numel() == 1) ||
      (attn_metadata.seq_lens.defined() && attn_metadata.seq_lens.numel() == 1);
  const bool compact_spec_verify_single =
      query_ranges.size() == 1 && query_ranges.front().first == 0 &&
      query_ranges.front().second == total_tokens && total_tokens > 1 &&
      graph_has_one_sequence_kv &&
      ((attn_metadata.num_speculative_tokens > 0 &&
        total_tokens == attn_metadata.num_speculative_tokens + 1) ||
       attn_metadata.max_query_len == total_tokens || total_tokens == 2);
  const bool compact_spec_verify_multi =
      attn_metadata.is_acl_graph && query_ranges.size() > 1 &&
      attn_metadata.num_speculative_tokens > 0 &&
      total_tokens % (attn_metadata.num_speculative_tokens + 1) == 0 &&
      std::all_of(query_ranges.begin(),
                  query_ranges.end(),
                  [&](const std::pair<int64_t, int64_t>& range) {
                    return range.second - range.first ==
                           attn_metadata.num_speculative_tokens + 1;
                  });
  const std::optional<size_t> graph_state_sequence_index =
      dsv4_graph_sequence_index(attn_metadata);
  const auto state_sequence_index_for = [&](size_t local_sequence) {
    return graph_state_sequence_index.value_or(local_sequence);
  };
  const size_t state_sequence_count = graph_state_sequence_index.has_value()
                                          ? *graph_state_sequence_index + 1
                                          : query_ranges.size();
  const bool compact_spec_verify =
      compact_spec_verify_single || compact_spec_verify_multi;
  auto kv_lens =
      build_kv_lens(attn_metadata, query_ranges.size(), query_ranges);
  const char* dynamic_short_keys =
      std::getenv("XLLM_DSV4_DYNAMIC_SHORT_GRAPH_KEYS");
  if (compact_spec_verify_multi && dynamic_short_keys != nullptr &&
      std::string(dynamic_short_keys) == "1") {
    CHECK(graph_decode_kv_seq_lens.defined())
        << "Dynamic short Graph key requires live positions, not a captured "
           "host KV fill";
  }
  // Direct multi-sequence Graph capture may not carry the optional
  // input_positions view. Batched MATE still needs one causal KV length per
  // proposal row, not one maximum length for the whole request. Materialize
  // the rows into persistent storage from the already validated per-sequence
  // KV lengths; scalar fill is capture-safe and introduces no allocation. The
  // MATE physical-index rows still carry the causal -1 tail masking.
  if (!graph_decode_kv_seq_lens.defined() && !is_prefill &&
      !is_chunked_prefill && attn_metadata.is_acl_graph &&
      query_ranges.size() > 1 && decode_kv_seq_lens_.defined() &&
      decode_kv_seq_lens_.size(0) >= total_tokens &&
      kv_lens.size() >= query_ranges.size()) {
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      const int64_t q_start = query_ranges[seq_idx].first;
      const int64_t q_end = query_ranges[seq_idx].second;
      const int64_t q_len = q_end - q_start;
      if (q_len <= 0) {
        continue;
      }
      auto rows = decode_kv_seq_lens_.narrow(0, q_start, q_len);
      rows.fill_(static_cast<int32_t>(std::max<int64_t>(kv_lens[seq_idx], 1)));
    }
    graph_decode_kv_seq_lens = decode_kv_seq_lens_.narrow(0, 0, total_tokens);
  }
  const int64_t full_kv_threshold = dsv4_musa_prefill_full_kv_threshold();
  const int64_t max_kv_len =
      kv_lens.empty() ? total_tokens
                      : *std::max_element(kv_lens.begin(), kv_lens.end());
  // Full-KV fallback is safe only while the complete history fits in the
  // model's SWA window. For a decode request longer than that window,
  // disabling C4/indexer attention would silently discard the older context
  // and leave only the trailing SWA rows. Keep compressed history active in
  // that case; it is the intended DSV4 representation for q_len=1 decode.
  const bool force_full_kv_attention =
      dsv4_uses_full_kv(max_kv_len, full_kv_threshold, window_size_);
  if (trace_layer0) {
    std::vector<int64_t> q_range_flat;
    q_range_flat.reserve(query_ranges.size() * 2);
    for (const auto& range : query_ranges) {
      q_range_flat.push_back(range.first);
      q_range_flat.push_back(range.second);
    }
    LOG(INFO) << "[DSV4_MUSA_TRACE] layer0 query_ranges="
              << i64_prefix_to_string(q_range_flat)
              << " kv_lens=" << i64_prefix_to_string(kv_lens) << " actual_q="
              << tensor_i64_prefix(attn_metadata.actual_seq_lengths_query)
              << " actual_kv="
              << tensor_i64_prefix(attn_metadata.actual_seq_lengths_kv)
              << " seq_lens=" << tensor_i64_prefix(attn_metadata.seq_lens)
              << " seq_lens_q=" << tensor_i64_prefix(attn_metadata.seq_lens_q);
  }
  std::vector<torch::Tensor> outputs;
  outputs.reserve(query_ranges.size());
  struct PendingBatchedMate {
    torch::Tensor q;
    torch::Tensor logical_indices;
    torch::Tensor physical_indices;
    torch::Tensor extra_physical_indices;
    int64_t kv_len = 0;
    int64_t q_start = 0;
  };
  const bool batch_mate_across_sequences =
      dsv4_sequence_view_active && query_ranges.size() > 1 &&
      dsv4_musa_mate_fused_dsa_multi_sequence_enabled() && [] {
        const char* value =
            std::getenv("XLLM_DSV4_DISABLE_BATCH_MATE_ACROSS_SEQUENCES");
        return value == nullptr || std::string(value) != "1";
      }();
  std::vector<PendingBatchedMate> pending_batched_mate;
  if (batch_mate_across_sequences) {
    pending_batched_mate.reserve(query_ranges.size());
  }
  // Compression/cache construction is independent from the attention route.
  // A short normal prefill may use dense full-KV attention, but it must still
  // populate C4/C128 and indexer caches for the first decode step that crosses
  // the SWA window. Previously force_full_kv_attention disabled construction
  // altogether, leaving empty C4 pages that MATE consumed at q_len=1.
  const bool use_compressed_attention =
      dsv4_musa_compress_fallback_enabled() &&
      (compress_ratio_ == 4 || compress_ratio_ == 128) && compressor_wkv_ &&
      compressor_wgate_ && compressor_norm_ && compressor_ape_loaded_;
  const bool use_indexer_attention =
      use_compressed_attention && compress_ratio_ == 4 &&
      (is_prefill || is_chunked_prefill || !force_full_kv_attention) &&
      dsv4_musa_indexer_enabled() &&
      (indexer_wq_b_ || q_indexer_b_weight_loaded_) && indexer_weights_proj_ &&
      indexer_compressor_wkv_ && indexer_compressor_wgate_ &&
      indexer_compressor_norm_ && indexer_compressor_ape_loaded_;
  const bool disable_graph_compact_indexer = [] {
    const char* env = std::getenv("XLLM_DSV4_DISABLE_GRAPH_COMPACT_INDEXER");
    return env != nullptr && std::string(env) == "1";
  }();
  const bool effective_use_indexer_attention =
      use_indexer_attention &&
      !(disable_graph_compact_indexer && attn_metadata.is_acl_graph &&
        compact_spec_verify && total_tokens > 1);
  const bool apply_sparse_indexer_attention =
      effective_use_indexer_attention &&
      (!is_prefill || query_ranges.size() <= 1 ||
       dsv4_musa_multi_sequence_indexer_prefill_enabled());
  if (compress_ratio_ == 4 && dsv4_musa_mate_fused_dsa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MATE_ROUTE] layer=" << layer_id_
              << " is_prefill=" << is_prefill
              << " is_chunked_prefill=" << is_chunked_prefill
              << " max_kv_len=" << max_kv_len
              << " force_full_kv=" << force_full_kv_attention
              << " compress_enabled=" << dsv4_musa_compress_fallback_enabled()
              << " indexer_enabled=" << dsv4_musa_indexer_enabled()
              << " compressor_ape_loaded=" << compressor_ape_loaded_
              << " indexer_ape_loaded=" << indexer_compressor_ape_loaded_
              << " use_compressed=" << use_compressed_attention
              << " use_indexer=" << effective_use_indexer_attention;
  }
  const int64_t cmp_cache_idx =
      use_compressed_attention
          ? find_token_cache_index(attn_metadata, layer_id_, compress_ratio_)
          : -1;
  auto cmp_cache =
      use_compressed_attention ? kv_cache.get_k_cache() : torch::Tensor();
  auto mate_cmp_cache =
      use_compressed_attention
          ? dsv4_musa_block_tensor(
                kv_cache,
                compress_ratio_ == 4 ? BlockType::C4 : BlockType::C128,
                KVCacheTensorRole::VALUE)
          : torch::Tensor();
  auto cmp_slot_mapping =
      use_compressed_attention
          ? layer_cache_tensor(
                attn_metadata.slot_mappings, layer_id_, cmp_cache_idx)
          : torch::Tensor();
  auto cmp_block_table =
      use_compressed_attention
          ? layer_cache_tensor(
                attn_metadata.block_tables, layer_id_, cmp_cache_idx)
          : torch::Tensor();
  if (dsv4_musa_trace_enabled() && use_compressed_attention) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] layer=" << layer_id_
              << " cmp_cache_idx=" << cmp_cache_idx
              << " cmp_cache_shape=" << tensor_sizes_string(cmp_cache)
              << " mate_cmp_shape=" << tensor_sizes_string(mate_cmp_cache)
              << " cmp_slot_shape=" << tensor_sizes_string(cmp_slot_mapping)
              << " cmp_slot_prefix=" << tensor_i64_prefix(cmp_slot_mapping)
              << " cmp_bt_shape=" << tensor_sizes_string(cmp_block_table)
              << " cmp_bt_prefix=" << tensor_i64_prefix(cmp_block_table);
  }
  auto index_cache = effective_use_indexer_attention
                         ? kv_cache.get_index_cache()
                         : torch::Tensor();
  auto index_scale_cache = effective_use_indexer_attention
                               ? dsv4_musa_indexer_scale_cache(kv_cache)
                               : torch::Tensor();
  const int64_t index_cache_idx =
      effective_use_indexer_attention
          ? find_token_cache_index(
                attn_metadata, layer_id_, compress_ratio_, /*ordinal=*/1)
          : -1;
  auto index_slot_mapping =
      effective_use_indexer_attention
          ? layer_cache_tensor(
                attn_metadata.slot_mappings, layer_id_, index_cache_idx)
          : torch::Tensor();
  auto index_block_table =
      effective_use_indexer_attention
          ? layer_cache_tensor(
                attn_metadata.block_tables, layer_id_, index_cache_idx)
          : torch::Tensor();
  const bool use_paged_indexer_cache =
      dsv4_musa_paged_indexer_cache_enabled() && index_cache.defined() &&
      index_scale_cache.defined() && index_slot_mapping.defined() &&
      index_block_table.defined() &&
      index_cache.scalar_type() == torch::kInt8 &&
      index_scale_cache.scalar_type() == torch::kFloat16;
  const bool use_paged_compressed_cache = cmp_cache.defined() &&
                                          cmp_slot_mapping.defined() &&
                                          cmp_block_table.defined();
  const bool retain_compressed_history = !use_paged_compressed_cache;

  auto compressor_kv_cache = std::get<1>(kv_state);
  auto compressor_score_cache = std::get<2>(kv_state);
  auto indexer_kv_state_cache = std::get<3>(kv_state);
  auto indexer_score_state_cache = std::get<4>(kv_state);
  const auto state_block_indices_it =
      attn_metadata.cp_input_dict.find(kDsv4StateBlockIndicesMetadataKey);
  const torch::Tensor state_block_indices =
      !attn_metadata.is_acl_graph &&
              state_block_indices_it != attn_metadata.cp_input_dict.end()
          ? state_block_indices_it->second
          : torch::Tensor();
  const auto read_indices_it =
      attn_metadata.cp_input_dict.find(kDsv4PrefillStateReadIndicesMetadataKey);
  const auto prefill_read_slots = [&](const torch::Tensor& write_slots,
                                      size_t seq,
                                      const torch::Tensor& cache,
                                      int64_t rows,
                                      int64_t block_size) {
    if (!is_chunked_prefill ||
        read_indices_it == attn_metadata.cp_input_dict.end()) {
      return write_slots;
    }
    // Do not reuse the write-slot workspace: its original addresses are
    // retained for scattering the resulting state into this chunk's page.
    return compressor_state_slots_from_indices(read_indices_it->second,
                                               seq,
                                               cache.size(0),
                                               rows,
                                               block_size,
                                               cache.device(),
                                               torch::Tensor());
  };
  const int64_t compressor_state_cache_idx =
      use_compressed_attention
          ? find_swa_cache_index(attn_metadata, layer_id_, /*ordinal=*/1)
          : -1;
  const int64_t compressor_score_state_cache_idx =
      use_compressed_attention
          ? find_swa_cache_index(attn_metadata, layer_id_, /*ordinal=*/2)
          : -1;
  auto compressor_state_block_table =
      use_compressed_attention ? layer_cache_tensor(attn_metadata.block_tables,
                                                    layer_id_,
                                                    compressor_state_cache_idx)
                               : torch::Tensor();
  auto compressor_score_state_block_table =
      use_compressed_attention
          ? layer_cache_tensor(attn_metadata.block_tables,
                               layer_id_,
                               compressor_score_state_cache_idx)
          : torch::Tensor();
  if (!compressor_state_block_table.defined()) {
    compressor_state_block_table = swa_block_table;
  }
  if (!compressor_score_state_block_table.defined()) {
    compressor_score_state_block_table = compressor_state_block_table;
  }
  const int64_t compressor_state_rows = compressor_coff_ * compress_ratio_;
  const int64_t compressor_state_dim = compressor_coff_ * head_dim_;
  // Preserve the established q6 Graph performance path by freezing its
  // host-side compressor/indexer update.  Accuracy A/B can opt into the q2
  // Eager state-update semantics while retaining Graph execution; this is
  // deliberately opt-in because the extra state gather/scatter may reduce
  // the q6 replay benefit or expose an unsupported capture allocation.
  const char* disable_graph_q6_freeze_env =
      std::getenv("XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE");
  const bool disable_graph_q6_freeze =
      disable_graph_q6_freeze_env != nullptr &&
      std::string(disable_graph_q6_freeze_env) == "1";
  const char* disable_graph_q6_compressor_state_env =
      std::getenv("XLLM_DSV4_DISABLE_GRAPH_Q6_COMPRESSOR_STATE");
  const bool disable_graph_q6_compressor_state =
      disable_graph_q6_compressor_state_env != nullptr &&
      std::string(disable_graph_q6_compressor_state_env) == "1";
  const char* disable_graph_q6_indexer_state_env =
      std::getenv("XLLM_DSV4_DISABLE_GRAPH_Q6_INDEXER_STATE");
  const bool disable_graph_q6_indexer_state =
      disable_graph_q6_indexer_state_env != nullptr &&
      std::string(disable_graph_q6_indexer_state_env) == "1";
  const bool freeze_compact_graph_compressor =
      compact_spec_verify && attn_metadata.is_acl_graph && total_tokens > 2 &&
      (!disable_graph_q6_freeze || dsv4_musa_graph_capture_warmup());
  const bool use_persistent_compressor_state =
      use_compressed_attention && !freeze_compact_graph_compressor &&
      !disable_graph_q6_compressor_state &&
      dsv4_musa_persistent_compressor_state_enabled() &&
      compressor_kv_cache.defined() && compressor_score_cache.defined() &&
      compressor_kv_cache.dim() == 3 && compressor_score_cache.dim() == 3 &&
      compressor_kv_cache.size(1) >= compressor_state_rows &&
      compressor_score_cache.size(1) >= compressor_state_rows &&
      compressor_kv_cache.size(2) == compressor_state_dim &&
      compressor_score_cache.size(2) == compressor_state_dim &&
      compressor_state_block_table.defined() &&
      compressor_score_state_block_table.defined();

  const int64_t indexer_state_rows = 2 * 4;
  const int64_t indexer_state_dim = 2 * index_head_dim_;
  const int64_t indexer_state_cache_idx =
      effective_use_indexer_attention
          ? find_swa_cache_index(attn_metadata, layer_id_, /*ordinal=*/3)
          : -1;
  const int64_t indexer_score_state_cache_idx =
      effective_use_indexer_attention
          ? find_swa_cache_index(attn_metadata, layer_id_, /*ordinal=*/4)
          : -1;
  auto indexer_state_block_table =
      effective_use_indexer_attention
          ? layer_cache_tensor(
                attn_metadata.block_tables, layer_id_, indexer_state_cache_idx)
          : torch::Tensor();
  auto indexer_score_state_block_table =
      effective_use_indexer_attention
          ? layer_cache_tensor(attn_metadata.block_tables,
                               layer_id_,
                               indexer_score_state_cache_idx)
          : torch::Tensor();
  if (!indexer_state_block_table.defined()) {
    indexer_state_block_table = compressor_state_block_table;
  }
  if (!indexer_score_state_block_table.defined()) {
    indexer_score_state_block_table = indexer_state_block_table;
  }
  const bool use_persistent_indexer_state =
      effective_use_indexer_attention && !freeze_compact_graph_compressor &&
      !disable_graph_q6_indexer_state &&
      dsv4_musa_persistent_indexer_state_enabled() &&
      indexer_kv_state_cache.defined() && indexer_score_state_cache.defined() &&
      indexer_kv_state_cache.dim() == 3 &&
      indexer_score_state_cache.dim() == 3 &&
      indexer_kv_state_cache.size(1) >= indexer_state_rows &&
      indexer_score_state_cache.size(1) >= indexer_state_rows &&
      indexer_kv_state_cache.size(2) == indexer_state_dim &&
      indexer_score_state_cache.size(2) == indexer_state_dim &&
      indexer_state_block_table.defined() &&
      indexer_score_state_block_table.defined();

  std::vector<torch::Tensor> compressor_kv_state_slots_by_seq(
      query_ranges.size());
  std::vector<torch::Tensor> compressor_score_state_slots_by_seq(
      query_ranges.size());
  std::vector<torch::Tensor> indexer_kv_state_slots_by_seq(query_ranges.size());
  std::vector<torch::Tensor> indexer_score_state_slots_by_seq(
      query_ranges.size());
  std::vector<uint8_t> compact_indexer_state_gathered(query_ranges.size(), 0);
  const bool use_compact_decode_workspace =
      !is_prefill && query_ranges.size() == 1 &&
      (total_tokens == 1 ||
       (compact_spec_verify && attn_metadata.is_acl_graph &&
        total_tokens <= decode_workspace_tokens_));
  torch::Tensor compact_decode_state_position = attn_metadata.input_positions;
  if (compact_spec_verify && attn_metadata.is_acl_graph && total_tokens > 1 &&
      attn_metadata.input_positions.defined()) {
    compact_decode_state_position = attn_metadata.input_positions.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/1);
  }
  const char* grouped_state_env = std::getenv("XLLM_DSV4_GROUPED_STATE_GATHER");
  const bool grouped_state_gather =
      grouped_state_env != nullptr && std::string(grouped_state_env) == "1" &&
      !graph_state_sequence_index.has_value() && use_compact_decode_workspace &&
      use_persistent_compressor_state &&
      decode_compressor_kv_state_.defined() &&
      decode_compressor_score_state_.defined() &&
      (!use_persistent_indexer_state ||
       (decode_indexer_kv_state_.defined() &&
        decode_indexer_score_state_.defined()));
  if (grouped_state_gather) {
    std::vector<torch::Tensor> caches{compressor_kv_cache,
                                      compressor_score_cache};
    std::vector<torch::Tensor> tables{compressor_state_block_table,
                                      compressor_score_state_block_table};
    std::vector<int64_t> rows{compressor_state_rows, compressor_state_rows};
    std::vector<torch::Tensor> slots{decode_compressor_kv_slots_,
                                     decode_compressor_score_slots_};
    std::vector<torch::Tensor> outputs{decode_compressor_kv_state_,
                                       decode_compressor_score_state_};
    if (use_persistent_indexer_state) {
      caches.insert(caches.end(),
                    {indexer_kv_state_cache, indexer_score_state_cache});
      tables.insert(
          tables.end(),
          {indexer_state_block_table, indexer_score_state_block_table});
      rows.insert(rows.end(), {indexer_state_rows, indexer_state_rows});
      slots.insert(slots.end(),
                   {decode_indexer_kv_slots_, decode_indexer_score_slots_});
      outputs.insert(outputs.end(),
                     {decode_indexer_kv_state_, decode_indexer_score_state_});
    }
    kernel::musa::deepseek_v4_state_gather_many(
        caches, tables, rows, slots, outputs, 0, compact_decode_state_position);
    LOG_FIRST_N(INFO, 1) << "[DSV4_GROUPED_STATE_GATHER] roles="
                         << caches.size();
  }
  const char* batched_state_env = std::getenv("XLLM_DSV4_BATCH_STATE_GATHER");
  const bool batched_state_gather =
      compact_spec_verify_multi && dsv4_sequence_view_active && !is_prefill &&
      !is_chunked_prefill && use_persistent_compressor_state &&
      batched_state_env != nullptr && std::string(batched_state_env) == "1";
  std::vector<torch::Tensor> batched_decode_states;
  if (batched_state_gather) {
    const int64_t sequences = static_cast<int64_t>(query_ranges.size());
    const int64_t query_width = attn_metadata.num_speculative_tokens + 1;
    CHECK_EQ(query_width, 7);
    CHECK_LE(sequences, 16);
    CHECK(!graph_state_sequence_index.has_value());
    compressor_kv_state_.resize(state_sequence_count);
    compressor_score_state_.resize(state_sequence_count);
    std::vector<torch::Tensor> caches{compressor_kv_cache,
                                      compressor_score_cache};
    std::vector<torch::Tensor> tables{compressor_state_block_table,
                                      compressor_score_state_block_table};
    std::vector<int64_t> rows{compressor_state_rows, compressor_state_rows};
    if (use_persistent_indexer_state) {
      indexer_kv_state_.resize(state_sequence_count);
      indexer_score_state_.resize(state_sequence_count);
      caches.insert(caches.end(),
                    {indexer_kv_state_cache, indexer_score_state_cache});
      tables.insert(
          tables.end(),
          {indexer_state_block_table, indexer_score_state_block_table});
      rows.insert(rows.end(), {indexer_state_rows, indexer_state_rows});
    }
    std::vector<torch::Tensor> slots;
    slots.reserve(caches.size());
    batched_decode_states.reserve(caches.size());
    for (size_t role = 0; role < caches.size(); ++role) {
      slots.emplace_back(
          torch::empty({sequences, rows[role]},
                       caches[role].options().dtype(torch::kInt64)));
      batched_decode_states.emplace_back(
          torch::empty({sequences, rows[role], caches[role].size(2)},
                       caches[role].options()));
    }
    kernel::musa::deepseek_v4_state_gather_batched(
        caches,
        tables,
        rows,
        slots,
        batched_decode_states,
        sequences,
        query_width,
        attn_metadata.input_positions);
    for (size_t seq = 0; seq < query_ranges.size(); ++seq) {
      compressor_kv_state_[seq] = batched_decode_states[0].select(0, seq);
      compressor_score_state_[seq] = batched_decode_states[1].select(0, seq);
      compressor_kv_state_slots_by_seq[seq] = slots[0].select(0, seq);
      compressor_score_state_slots_by_seq[seq] = slots[1].select(0, seq);
      if (use_persistent_indexer_state) {
        indexer_kv_state_[seq] = batched_decode_states[2].select(0, seq);
        indexer_score_state_[seq] = batched_decode_states[3].select(0, seq);
        indexer_kv_state_slots_by_seq[seq] = slots[2].select(0, seq);
        indexer_score_state_slots_by_seq[seq] = slots[3].select(0, seq);
        compact_indexer_state_gathered[seq] = 1;
      }
    }
    LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_STATE_GATHER] sequences=" << sequences
                         << " roles=" << caches.size()
                         << " query_width=" << query_width;
  }
  if (use_persistent_compressor_state) {
    compressor_kv_state_.resize(state_sequence_count);
    compressor_score_state_.resize(state_sequence_count);
    auto kv_state_2d = compressor_kv_cache.view({-1, compressor_state_dim});
    auto score_state_2d =
        compressor_score_cache.view({-1, compressor_state_dim});
    const int64_t state_block_size = compressor_kv_cache.size(1);
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      if (batched_state_gather) {
        continue;
      }
      const size_t state_seq_idx = state_sequence_index_for(seq_idx);
      if (compact_spec_verify_multi) {
        // A verify segment may straddle an SWA block boundary. Read the state
        // belonging to this request's first input position, not the last
        // valid block in the capacity-padded table (which can be a proposal).
        auto position = attn_metadata.input_positions.narrow(
            0, query_ranges[seq_idx].first, 1);
        auto kv_slots = torch::empty_like(decode_compressor_kv_slots_);
        auto score_slots = torch::empty_like(decode_compressor_score_slots_);
        auto kv_output = torch::empty_like(decode_compressor_kv_state_);
        auto score_output = torch::empty_like(decode_compressor_score_state_);
        if (use_persistent_indexer_state) {
          auto indexer_kv_slots = torch::empty_like(decode_indexer_kv_slots_);
          auto indexer_score_slots =
              torch::empty_like(decode_indexer_score_slots_);
          auto indexer_kv_output = torch::empty_like(decode_indexer_kv_state_);
          auto indexer_score_output =
              torch::empty_like(decode_indexer_score_state_);
          kernel::musa::deepseek_v4_state_gather_many(
              {compressor_kv_cache,
               compressor_score_cache,
               indexer_kv_state_cache,
               indexer_score_state_cache},
              {compressor_state_block_table,
               compressor_score_state_block_table,
               indexer_state_block_table,
               indexer_score_state_block_table},
              {compressor_state_rows,
               compressor_state_rows,
               indexer_state_rows,
               indexer_state_rows},
              {kv_slots, score_slots, indexer_kv_slots, indexer_score_slots},
              {kv_output,
               score_output,
               indexer_kv_output,
               indexer_score_output},
              static_cast<int64_t>(seq_idx),
              position);
          indexer_kv_state_[state_seq_idx] = indexer_kv_output;
          indexer_score_state_[state_seq_idx] = indexer_score_output;
          indexer_kv_state_slots_by_seq[seq_idx] =
              indexer_kv_slots.narrow(0, 0, indexer_state_rows);
          indexer_score_state_slots_by_seq[seq_idx] =
              indexer_score_slots.narrow(0, 0, indexer_state_rows);
          compact_indexer_state_gathered[seq_idx] = 1;
        } else {
          kernel::musa::deepseek_v4_state_gather_many(
              {compressor_kv_cache, compressor_score_cache},
              {compressor_state_block_table,
               compressor_score_state_block_table},
              {compressor_state_rows, compressor_state_rows},
              {kv_slots, score_slots},
              {kv_output, score_output},
              static_cast<int64_t>(seq_idx),
              position);
        }
        compressor_kv_state_[state_seq_idx] = kv_output;
        compressor_score_state_[state_seq_idx] = score_output;
        compressor_kv_state_slots_by_seq[seq_idx] =
            kv_slots.narrow(0, 0, compressor_state_rows);
        compressor_score_state_slots_by_seq[seq_idx] =
            score_slots.narrow(0, 0, compressor_state_rows);
        continue;
      }
      const bool use_decode_workspace =
          use_compact_decode_workspace &&
          decode_compressor_kv_state_.defined() &&
          decode_compressor_score_state_.defined();
      if (use_decode_workspace) {
        compressor_kv_state_[state_seq_idx] =
            grouped_state_gather ? decode_compressor_kv_state_.narrow(
                                       0, 0, compressor_state_rows)
                                 : kernel::musa::deepseek_v4_state_gather(
                                       compressor_kv_cache,
                                       compressor_state_block_table,
                                       static_cast<int64_t>(seq_idx),
                                       compressor_state_rows,
                                       state_block_size,
                                       decode_compressor_kv_slots_,
                                       decode_compressor_kv_state_,
                                       compact_decode_state_position);
        compressor_score_state_[state_seq_idx] =
            grouped_state_gather ? decode_compressor_score_state_.narrow(
                                       0, 0, compressor_state_rows)
                                 : kernel::musa::deepseek_v4_state_gather(
                                       compressor_score_cache,
                                       compressor_score_state_block_table,
                                       static_cast<int64_t>(seq_idx),
                                       compressor_state_rows,
                                       state_block_size,
                                       decode_compressor_score_slots_,
                                       decode_compressor_score_state_,
                                       compact_decode_state_position);
        compressor_kv_state_slots_by_seq[seq_idx] =
            decode_compressor_kv_slots_.narrow(
                /*dim=*/0, /*start=*/0, compressor_state_rows);
        compressor_score_state_slots_by_seq[seq_idx] =
            decode_compressor_score_slots_.narrow(
                /*dim=*/0, /*start=*/0, compressor_state_rows);
        continue;
      }
      auto kv_slots = compressor_state_slots_from_indices(
          state_block_indices,
          static_cast<int64_t>(state_seq_idx),
          compressor_kv_cache.size(0),
          compressor_state_rows,
          state_block_size,
          compressor_kv_cache.device(),
          query_ranges.size() == 1 ? decode_compressor_kv_slots_
                                   : torch::Tensor());
      if (!kv_slots.defined()) {
        kv_slots = compressor_state_slots(compressor_state_block_table,
                                          static_cast<int64_t>(seq_idx),
                                          compressor_state_rows,
                                          state_block_size,
                                          compressor_kv_cache.device());
      }
      auto score_slots = compressor_state_slots_from_indices(
          state_block_indices,
          static_cast<int64_t>(state_seq_idx),
          compressor_score_cache.size(0),
          compressor_state_rows,
          state_block_size,
          compressor_score_cache.device(),
          query_ranges.size() == 1 ? decode_compressor_score_slots_
                                   : torch::Tensor());
      if (!score_slots.defined()) {
        score_slots = compressor_state_slots(compressor_score_state_block_table,
                                             static_cast<int64_t>(seq_idx),
                                             compressor_state_rows,
                                             state_block_size,
                                             compressor_score_cache.device());
      }
      compressor_kv_state_slots_by_seq[seq_idx] = kv_slots;
      compressor_score_state_slots_by_seq[seq_idx] = score_slots;
      if ((!is_prefill || is_chunked_prefill) && kv_slots.defined() &&
          score_slots.defined()) {
        compressor_kv_state_[state_seq_idx] =
            kv_state_2d.index_select(0,
                                     prefill_read_slots(kv_slots,
                                                        seq_idx,
                                                        compressor_kv_cache,
                                                        compressor_state_rows,
                                                        state_block_size));
        compressor_score_state_[state_seq_idx] = score_state_2d.index_select(
            0,
            prefill_read_slots(score_slots,
                               seq_idx,
                               compressor_score_cache,
                               compressor_state_rows,
                               state_block_size));
        if (trace_layer0) {
          LOG(INFO) << "[DSV4_MUSA_STATE] phase=load kind=compressor seq="
                    << seq_idx << " kv_slots=" << tensor_i64_prefix(kv_slots)
                    << " score_slots=" << tensor_i64_prefix(score_slots)
                    << " kv="
                    << tensor_stat_string(compressor_kv_state_[state_seq_idx])
                    << " score="
                    << tensor_stat_string(
                           compressor_score_state_[state_seq_idx]);
        }
      }
    }
  }
  if (use_persistent_indexer_state) {
    indexer_kv_state_.resize(state_sequence_count);
    indexer_score_state_.resize(state_sequence_count);
    auto kv_state_2d = indexer_kv_state_cache.view({-1, indexer_state_dim});
    auto score_state_2d =
        indexer_score_state_cache.view({-1, indexer_state_dim});
    const int64_t state_block_size = indexer_kv_state_cache.size(1);
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      const size_t state_seq_idx = state_sequence_index_for(seq_idx);
      if (compact_spec_verify_multi) {
        if (compact_indexer_state_gathered[seq_idx] != 0) {
          continue;
        }
        auto position = attn_metadata.input_positions.narrow(
            0, query_ranges[seq_idx].first, 1);
        auto kv_slots = torch::empty_like(decode_indexer_kv_slots_);
        auto score_slots = torch::empty_like(decode_indexer_score_slots_);
        auto kv_output = torch::empty_like(decode_indexer_kv_state_);
        auto score_output = torch::empty_like(decode_indexer_score_state_);
        kernel::musa::deepseek_v4_state_gather_many(
            {indexer_kv_state_cache, indexer_score_state_cache},
            {indexer_state_block_table, indexer_score_state_block_table},
            {indexer_state_rows, indexer_state_rows},
            {kv_slots, score_slots},
            {kv_output, score_output},
            static_cast<int64_t>(seq_idx),
            position);
        indexer_kv_state_[state_seq_idx] = kv_output;
        indexer_score_state_[state_seq_idx] = score_output;
        indexer_kv_state_slots_by_seq[seq_idx] =
            kv_slots.narrow(0, 0, indexer_state_rows);
        indexer_score_state_slots_by_seq[seq_idx] =
            score_slots.narrow(0, 0, indexer_state_rows);
        continue;
      }
      const bool use_decode_workspace = use_compact_decode_workspace &&
                                        decode_indexer_kv_state_.defined() &&
                                        decode_indexer_score_state_.defined();
      if (use_decode_workspace) {
        indexer_kv_state_[state_seq_idx] =
            grouped_state_gather
                ? decode_indexer_kv_state_.narrow(0, 0, indexer_state_rows)
                : kernel::musa::deepseek_v4_state_gather(
                      indexer_kv_state_cache,
                      indexer_state_block_table,
                      static_cast<int64_t>(seq_idx),
                      indexer_state_rows,
                      state_block_size,
                      decode_indexer_kv_slots_,
                      decode_indexer_kv_state_,
                      compact_decode_state_position);
        indexer_score_state_[state_seq_idx] =
            grouped_state_gather
                ? decode_indexer_score_state_.narrow(0, 0, indexer_state_rows)
                : kernel::musa::deepseek_v4_state_gather(
                      indexer_score_state_cache,
                      indexer_score_state_block_table,
                      static_cast<int64_t>(seq_idx),
                      indexer_state_rows,
                      state_block_size,
                      decode_indexer_score_slots_,
                      decode_indexer_score_state_,
                      compact_decode_state_position);
        indexer_kv_state_slots_by_seq[seq_idx] =
            decode_indexer_kv_slots_.narrow(
                /*dim=*/0, /*start=*/0, indexer_state_rows);
        indexer_score_state_slots_by_seq[seq_idx] =
            decode_indexer_score_slots_.narrow(
                /*dim=*/0, /*start=*/0, indexer_state_rows);
        continue;
      }
      auto kv_slots = compressor_state_slots_from_indices(
          state_block_indices,
          static_cast<int64_t>(state_seq_idx),
          indexer_kv_state_cache.size(0),
          indexer_state_rows,
          state_block_size,
          indexer_kv_state_cache.device(),
          query_ranges.size() == 1 ? decode_indexer_kv_slots_
                                   : torch::Tensor());
      if (!kv_slots.defined()) {
        kv_slots = compressor_state_slots(indexer_state_block_table,
                                          static_cast<int64_t>(seq_idx),
                                          indexer_state_rows,
                                          state_block_size,
                                          indexer_kv_state_cache.device());
      }
      auto score_slots = compressor_state_slots_from_indices(
          state_block_indices,
          static_cast<int64_t>(state_seq_idx),
          indexer_score_state_cache.size(0),
          indexer_state_rows,
          state_block_size,
          indexer_score_state_cache.device(),
          query_ranges.size() == 1 ? decode_indexer_score_slots_
                                   : torch::Tensor());
      if (!score_slots.defined()) {
        score_slots =
            compressor_state_slots(indexer_score_state_block_table,
                                   static_cast<int64_t>(seq_idx),
                                   indexer_state_rows,
                                   state_block_size,
                                   indexer_score_state_cache.device());
      }
      indexer_kv_state_slots_by_seq[seq_idx] = kv_slots;
      indexer_score_state_slots_by_seq[seq_idx] = score_slots;
      if ((!is_prefill || is_chunked_prefill) && kv_slots.defined() &&
          score_slots.defined()) {
        indexer_kv_state_[state_seq_idx] =
            kv_state_2d.index_select(0,
                                     prefill_read_slots(kv_slots,
                                                        seq_idx,
                                                        indexer_kv_state_cache,
                                                        indexer_state_rows,
                                                        state_block_size));
        indexer_score_state_[state_seq_idx] = score_state_2d.index_select(
            0,
            prefill_read_slots(score_slots,
                               seq_idx,
                               indexer_score_state_cache,
                               indexer_state_rows,
                               state_block_size));
        if (trace_layer0) {
          LOG(INFO) << "[DSV4_MUSA_STATE] phase=load kind=indexer seq="
                    << seq_idx << " kv_slots=" << tensor_i64_prefix(kv_slots)
                    << " score_slots=" << tensor_i64_prefix(score_slots)
                    << " kv="
                    << tensor_stat_string(indexer_kv_state_[state_seq_idx])
                    << " score="
                    << tensor_stat_string(indexer_score_state_[state_seq_idx]);
        }
      }
    }
  }
  if (compress_ratio_ == 4 && dsv4_musa_mate_fused_dsa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_INDEX_CACHE] layer=" << layer_id_
              << " paged_int8=" << use_paged_indexer_cache
              << " cache=" << tensor_sizes_string(index_cache)
              << " scale=" << tensor_sizes_string(index_scale_cache);
  }
  const torch::Tensor& compressed_cos =
      compress_ratio_ == 4 ? attn_metadata.c4_cos : attn_metadata.c128_cos;
  const torch::Tensor& compressed_sin =
      compress_ratio_ == 4 ? attn_metadata.c4_sin : attn_metadata.c128_sin;
  std::vector<torch::Tensor> prefill_compressed_by_seq(query_ranges.size());
  record_prefill_profile_boundary(prefill_profile, 1, profile_device_id);
  torch::Tensor joint_main_kv_projection;
  torch::Tensor joint_main_score_projection;
  torch::Tensor joint_indexer_kv_projection;
  torch::Tensor joint_indexer_score_projection;
  const bool use_joint_sp_compressor_projection =
      use_compressed_attention && effective_use_indexer_attention &&
      (is_prefill || is_chunked_prefill) && sp_context != nullptr &&
      query_ranges.size() == 1 &&
      [] {
        const char* env =
            std::getenv("XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION");
        return env != nullptr && std::string(env) == "1";
      }() &&
      sp_joint_compressor_projection_weight_.defined();
  if (use_joint_sp_compressor_projection) {
    auto project =
        [this, &compression_hidden_states, zero_local_sp_shard](auto& linear) {
          if (zero_local_sp_shard) {
            return torch::empty({0, linear->weight().size(0)},
                                compression_hidden_states.options().dtype(
                                    linear->weight().scalar_type()));
          }
          return linear->forward(compression_hidden_states);
        };
    auto gather = [sp_context](const torch::Tensor& local) {
      return dsv4_musa_sp::gather_global(
          dsv4_musa_sp::pad_local(local, *sp_context), *sp_context);
    };
    joint_main_kv_projection = gather(project(compressor_wkv_));
    joint_main_score_projection = gather(project(compressor_wgate_));
    joint_indexer_kv_projection = gather(project(indexer_compressor_wkv_));
    joint_indexer_score_projection = gather(project(indexer_compressor_wgate_));
    LOG_FIRST_N(INFO, 1)
        << "DeepSeek-V4 MUSA SP compressor projections preserve independent "
           "Linear forward paths before packing.";
  }
  if (use_compressed_attention && (is_prefill || is_chunked_prefill)) {
    // A new request can enter through chunked prefill (DSpark requires the
    // chunked flag). Detect its first chunk from the global compression span:
    // its KV length equals the number of tokens being compressed; later chunks
    // include the already-computed prefix and therefore have a larger length.
    bool starts_new_chunked_prefill = false;
    if (is_chunked_prefill && !kv_lens.empty() &&
        compression_query_ranges.size() == kv_lens.size()) {
      starts_new_chunked_prefill = true;
      for (size_t seq_idx = 0; seq_idx < kv_lens.size(); ++seq_idx) {
        const auto& range = compression_query_ranges[seq_idx];
        const int64_t compression_tokens = range.second - range.first;
        if (compression_tokens <= 0 || kv_lens[seq_idx] != compression_tokens) {
          starts_new_chunked_prefill = false;
          break;
        }
      }
    }
    if ((is_prefill && !is_chunked_prefill) || starts_new_chunked_prefill) {
      compressor_kv_history_.clear();
      compressor_kv_state_.clear();
      compressor_score_state_.clear();
      indexer_kv_history_.clear();
      indexer_kv_state_.clear();
      indexer_score_state_.clear();
    }
    if (retain_compressed_history) {
      compressor_kv_history_.resize(state_sequence_count);
    }
    compressor_kv_state_.resize(state_sequence_count);
    compressor_score_state_.resize(state_sequence_count);
    if (effective_use_indexer_attention) {
      if (!use_paged_indexer_cache) {
        indexer_kv_history_.resize(state_sequence_count);
      }
      indexer_kv_state_.resize(state_sequence_count);
      indexer_score_state_.resize(state_sequence_count);
    }
    std::vector<torch::Tensor> compressed_to_cache;
    compressed_to_cache.reserve(query_ranges.size());
    std::vector<torch::Tensor> indexer_to_cache;
    indexer_to_cache.reserve(query_ranges.size());
    const auto prefix_snapshot_it =
        attn_metadata.cp_input_dict.find("__musa_dsv4_prefix_state_snapshot");
    const torch::Tensor prefix_snapshot_metadata =
        prefix_snapshot_it == attn_metadata.cp_input_dict.end()
            ? torch::Tensor()
            : prefix_snapshot_it->second;
    const int32_t* prefix_snapshot_data =
        prefix_snapshot_metadata.defined() &&
                prefix_snapshot_metadata.device().is_cpu() &&
                prefix_snapshot_metadata.scalar_type() == torch::kInt32 &&
                prefix_snapshot_metadata.dim() == 2 &&
                prefix_snapshot_metadata.size(1) == 2 &&
                prefix_snapshot_metadata.is_contiguous()
            ? prefix_snapshot_metadata.data_ptr<int32_t>()
            : nullptr;
    const size_t prefix_snapshot_rows =
        prefix_snapshot_data == nullptr
            ? 0
            : static_cast<size_t>(prefix_snapshot_metadata.size(0));
    int64_t rope_offset = pcp_context_ != nullptr && pcp_context_->cp_rank == 1
                              ? pcp_context_->front_rows / compress_ratio_
                              : 0;
    torch::Tensor precomputed_projection;
    const auto precomputed_projection_it =
        attn_metadata.cp_input_dict.find(kDsv4BatchedCompressorMetadataKey);
    const char* joint_projection_env =
        std::getenv("XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION");
    if (precomputed_projection_it != attn_metadata.cp_input_dict.end() &&
        joint_projection_env != nullptr &&
        std::string(joint_projection_env) == "1" && compress_ratio_ == 4 &&
        query_ranges.size() > 1) {
      precomputed_projection =
          dsv4_compressor_projection_view(precomputed_projection_it->second,
                                          compressor_coff_ * head_dim_,
                                          2 * index_head_dim_,
                                          effective_use_indexer_attention);
    }
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      const int64_t q_start = compression_query_ranges[seq_idx].first;
      const int64_t q_end = compression_query_ranges[seq_idx].second;
      const int64_t q_len = std::max<int64_t>(q_end - q_start, 0);
      const int64_t global_chunk_rows =
          pcp_context_ != nullptr ? pcp_context_->total_rows : q_len;
      const int64_t group_row_start =
          pcp_context_ != nullptr && pcp_context_->cp_rank == 1
              ? pcp_context_->front_rows
              : 0;
      const int64_t cached_prefix_tokens =
          (is_chunked_prefill && seq_idx < kv_lens.size()
               ? std::max<int64_t>(kv_lens[seq_idx] - global_chunk_rows, 0)
               : 0) +
          group_row_start;
      const int64_t compressed_blocks = dsv4_prefill_compressed_rows(
          cached_prefix_tokens, q_len, compress_ratio_);
      torch::Tensor cos_slice;
      torch::Tensor sin_slice;
      if (compressed_blocks > 0 && compressed_cos.defined() &&
          compressed_sin.defined() &&
          compressed_cos.size(0) >= rope_offset + compressed_blocks &&
          compressed_sin.size(0) >= rope_offset + compressed_blocks) {
        cos_slice = compressed_cos.slice(
            /*dim=*/0,
            /*start=*/rope_offset,
            /*end=*/rope_offset + compressed_blocks);
        sin_slice = compressed_sin.slice(
            /*dim=*/0,
            /*start=*/rope_offset,
            /*end=*/rope_offset + compressed_blocks);
      }
      rope_offset += compressed_blocks;
      if (q_end <= q_start) {
        continue;
      }
      torch::Tensor hidden_seq =
          sp_context == nullptr
              ? compression_hidden_states.slice(
                    /*dim=*/0, /*start=*/q_start, /*end=*/q_end)
              : compression_hidden_states;
      int32_t prefix_snapshot_state_id = -1;
      int64_t prefix_snapshot_tokens = 0;
      if (seq_idx < prefix_snapshot_rows) {
        prefix_snapshot_state_id = prefix_snapshot_data[seq_idx * 2];
        prefix_snapshot_tokens = prefix_snapshot_data[seq_idx * 2 + 1];
      }
      torch::Tensor projected_main_kv;
      torch::Tensor projected_main_score;
      torch::Tensor projected_indexer_kv;
      torch::Tensor projected_indexer_score;
      if (precomputed_projection.defined()) {
        auto sequence_projection = precomputed_projection.narrow(
            /*dim=*/0, /*start=*/q_start, /*length=*/q_len);
        const int64_t main_width = compressor_coff_ * head_dim_;
        const int64_t index_width = 2 * index_head_dim_;
        projected_main_kv = sequence_projection.narrow(1, 0, main_width);
        projected_main_score =
            sequence_projection.narrow(1, main_width, main_width);
        projected_indexer_kv =
            sequence_projection.narrow(1, 2 * main_width, index_width);
        projected_indexer_score = sequence_projection.narrow(
            1, 2 * main_width + index_width, index_width);
      }
      torch::Tensor cmp_seq = compress_prefill_sequence(
          hidden_seq,
          cos_slice,
          sin_slice,
          seq_idx,
          prefix_snapshot_tokens,
          cached_prefix_tokens,
          precomputed_projection.defined() ? projected_main_kv
                                           : joint_main_kv_projection,
          precomputed_projection.defined() ? projected_main_score
                                           : joint_main_score_projection);
      if (cmp_seq.defined() && cmp_seq.numel() > 0) {
        maybe_sync_attention_stage(layer_id_, "prefill_compress_seq", cmp_seq);
        prefill_compressed_by_seq[seq_idx] = cmp_seq;
        if (retain_compressed_history) {
          compressor_kv_history_[seq_idx] = cmp_seq.contiguous();
        }
        compressed_to_cache.push_back(cmp_seq);
      } else if (retain_compressed_history) {
        compressor_kv_history_[seq_idx] =
            torch::empty({0, head_dim_}, hidden_states.options());
      }
      if (effective_use_indexer_attention) {
        torch::Tensor index_seq = compress_indexer_prefill_sequence(
            hidden_seq,
            cos_slice,
            sin_slice,
            seq_idx,
            prefix_snapshot_tokens,
            cached_prefix_tokens,
            precomputed_projection.defined() ? projected_indexer_kv
                                             : joint_indexer_kv_projection,
            precomputed_projection.defined() ? projected_indexer_score
                                             : joint_indexer_score_projection);
        if (use_paged_indexer_cache) {
          if (index_seq.defined() && index_seq.numel() > 0) {
            indexer_to_cache.push_back(index_seq);
          }
        } else {
          indexer_kv_history_[seq_idx] =
              index_seq.defined()
                  ? index_seq.contiguous()
                  : torch::empty({0, index_head_dim_}, hidden_states.options());
        }
        maybe_sync_attention_stage(
            layer_id_, "prefill_indexer_compress_seq", index_seq);
      }
      if (prefix_snapshot_state_id >= 0) {
        if (pcp_context_ != nullptr) {
          const int64_t chunk_start =
              kv_lens[seq_idx] - pcp_context_->total_rows;
          CHECK_GT(prefix_snapshot_tokens, chunk_start);
          CHECK_LE(prefix_snapshot_tokens, kv_lens[seq_idx]);
          const int32_t snapshot_owner =
              prefix_snapshot_tokens <= chunk_start + pcp_context_->front_rows
                  ? 0
                  : 1;
          // Only the group containing the retained boundary produces this
          // snapshot. Both groups must restore that same state on a cache hit.
          auto share_snapshot = [&](std::vector<torch::Tensor>& snapshots,
                                    const std::vector<torch::Tensor>& states) {
            CHECK_LT(seq_idx, states.size());
            CHECK(states[seq_idx].defined());
            if (snapshots.size() <= seq_idx) {
              snapshots.resize(seq_idx + 1);
            }
            if (pcp_context_->cp_rank != snapshot_owner) {
              snapshots[seq_idx] = torch::empty_like(states[seq_idx]);
            }
            CHECK(snapshots[seq_idx].defined())
                << "PCP snapshot owner did not capture the retained boundary";
            pcp_context_->cross_group->broadcast(snapshots[seq_idx],
                                                 snapshot_owner);
          };
          share_snapshot(prefix_compressor_kv_state_, compressor_kv_state_);
          share_snapshot(prefix_compressor_score_state_,
                         compressor_score_state_);
          if (effective_use_indexer_attention) {
            share_snapshot(prefix_indexer_kv_state_, indexer_kv_state_);
            share_snapshot(prefix_indexer_score_state_, indexer_score_state_);
          }
          LOG_FIRST_N(INFO, 8)
              << "[DSV4_PCP_PREFIX_STATE] cp_rank=" << pcp_context_->cp_rank
              << " owner=" << snapshot_owner
              << " prefix=" << prefix_snapshot_tokens;
        }
        auto store_snapshot = [prefix_snapshot_state_id](
                                  const torch::Tensor& cache,
                                  const std::vector<torch::Tensor>& states,
                                  size_t sequence_index) {
          if (!cache.defined() || sequence_index >= states.size() ||
              !states[sequence_index].defined() ||
              prefix_snapshot_state_id >= cache.size(0)) {
            return;
          }
          const torch::Tensor& state = states[sequence_index];
          CHECK_LE(state.size(0), cache.size(1));
          cache.index({prefix_snapshot_state_id})
              .slice(/*dim=*/0, /*start=*/0, /*end=*/state.size(0))
              .copy_(state);
        };
        store_snapshot(
            compressor_kv_cache, prefix_compressor_kv_state_, seq_idx);
        store_snapshot(
            compressor_score_cache, prefix_compressor_score_state_, seq_idx);
        if (effective_use_indexer_attention) {
          store_snapshot(
              indexer_kv_state_cache, prefix_indexer_kv_state_, seq_idx);
          store_snapshot(
              indexer_score_state_cache, prefix_indexer_score_state_, seq_idx);
        }
      }
    }
    if (!compressed_to_cache.empty() && cmp_cache.defined() &&
        cmp_slot_mapping.defined()) {
      auto cmp_all = compressed_to_cache.size() == 1
                         ? compressed_to_cache.front()
                         : torch::cat(compressed_to_cache, /*dim=*/0);
      auto valid_slots = attn_metadata.is_acl_graph
                             ? cmp_slot_mapping.reshape({-1}).slice(
                                   /*dim=*/0,
                                   /*start=*/0,
                                   /*end=*/cmp_all.size(0))
                             : cmp_slot_mapping;
      if (pcp_context_ != nullptr) {
        cmp_all = dsv4_musa_pcp::exchange_rows_global_slots(
            cmp_all, valid_slots, *pcp_context_);
        CHECK_EQ(prefill_compressed_by_seq.size(), 1);
        prefill_compressed_by_seq[0] = cmp_all;
        if (retain_compressed_history) {
          compressor_kv_history_[0] = cmp_all;
        }
      }
      scatter_by_slot(cmp_cache, valid_slots, cmp_all);
      maybe_sync_attention_stage(
          layer_id_, "prefill_cmp_scatter_by_slot", cmp_all);
      if (mate_cmp_cache.defined()) {
        pack_mate_model1_cache(mate_cmp_cache, valid_slots, cmp_all);
        maybe_sync_attention_stage(layer_id_, "prefill_mate_cmp_pack", cmp_all);
      }
    }
    if (use_paged_indexer_cache && !indexer_to_cache.empty()) {
      auto index_all = indexer_to_cache.size() == 1
                           ? indexer_to_cache.front()
                           : torch::cat(indexer_to_cache, /*dim=*/0);
      auto valid_index_slots = attn_metadata.is_acl_graph
                                   ? index_slot_mapping.reshape({-1}).slice(
                                         /*dim=*/0,
                                         /*start=*/0,
                                         /*end=*/index_all.size(0))
                                   : index_slot_mapping;
      if (pcp_context_ != nullptr) {
        index_all = dsv4_musa_pcp::exchange_rows_global_slots(
            index_all, valid_index_slots, *pcp_context_);
      }
      quant_scatter_indexer_cache(
          index_cache, index_scale_cache, valid_index_slots, index_all);
      maybe_sync_attention_stage(
          layer_id_, "prefill_indexer_cache_scatter", index_all);
    }
  }
  record_prefill_profile_boundary(prefill_profile, 2, profile_device_id);
  const char* partitioned_c128_env =
      std::getenv("XLLM_DSV4_PARTITIONED_C128_MATE");
  const bool partitioned_c128_requests =
      partitioned_c128_env != nullptr &&
      std::string(partitioned_c128_env) == "1" && batch_mate_across_sequences &&
      compact_spec_verify_multi && attn_metadata.is_acl_graph && !is_prefill &&
      !is_chunked_prefill && dspark_block_size_ == 0 &&
      attn_metadata.num_speculative_tokens == 6 && compress_ratio_ == 128 &&
      window_size_ == 128 && index_topk_ == 512 && n_local_heads_ == 8 &&
      dsv4_musa_c128_mate_enabled() && mate_cmp_cache.defined() &&
      mate_swa_cache.defined() && cmp_cache.defined() &&
      cmp_block_table.defined() && graph_decode_kv_seq_lens.defined() &&
      decode_spec_indices_output_.defined() &&
      decode_spec_mate_physical_indices_.defined() &&
      decode_spec_mate_extra_physical_indices_.defined() &&
      [] {
        const char* value = std::getenv("XLLM_DSV4_DYNAMIC_GRAPH_STATE");
        return value != nullptr && std::string(value) == "1";
      }() &&
      // Bound the entire Graph capacity bucket,not just the capture's live
      // length. At most floor((4*16384+3)/128)=512 compressed keys are valid.
      dsv4_graph_indexer_capacity(max_kv_len / 4, /*multi_sequence=*/true) <=
          16384;
  const bool batch_mate_metadata = batch_mate_across_sequences && [] {
    const char* value = std::getenv("XLLM_DSV4_BATCH_MATE_METADATA");
    return value != nullptr && std::string(value) == "1";
  }();
  const bool batch_c128_indices =
      batch_mate_metadata && partitioned_c128_requests;
  std::vector<torch::Tensor> decode_compressed_to_cache;
  std::vector<torch::Tensor> decode_indexer_to_cache;
  int64_t decode_compressed_rows_scattered = 0;
  int64_t decode_compressed_tensors_scattered = 0;
  int64_t decode_indexer_tensors_scattered = 0;
  bool batch_compression = false;
  auto flush_decode_compressed_cache = [&]() {
    if (decode_compressed_to_cache.empty() || !cmp_cache.defined() ||
        !cmp_slot_mapping.defined()) {
      return;
    }
    torch::Tensor pending;
    int64_t row_end = 0;
    if (batch_compression) {
      auto rows = dsv4_pending_cache_rows(decode_compressed_to_cache,
                                          decode_compressed_tensors_scattered,
                                          decode_compressed_rows_scattered);
      if (!rows.values.defined()) {
        return;
      }
      pending = rows.values;
      row_end = rows.row_end;
    } else {
      auto cmp_all = decode_compressed_to_cache.size() == 1
                         ? decode_compressed_to_cache.front()
                         : torch::cat(decode_compressed_to_cache, /*dim=*/0);
      if (decode_compressed_rows_scattered >= cmp_all.size(0)) {
        return;
      }
      row_end = cmp_all.size(0);
      pending = cmp_all.slice(/*dim=*/0,
                              /*start=*/decode_compressed_rows_scattered,
                              /*end=*/row_end);
    }
    auto flat_slots = cmp_slot_mapping.reshape({-1});
    CHECK_GE(flat_slots.numel(), row_end);
    auto pending_slots =
        attn_metadata.is_acl_graph
            ? flat_slots
                  .slice(
                      /*dim=*/0,
                      /*start=*/decode_compressed_rows_scattered,
                      /*end=*/row_end)
                  .contiguous()
            : cmp_slot_mapping;
    scatter_by_slot(cmp_cache, pending_slots, pending);
    maybe_sync_attention_stage(
        layer_id_, "decode_cmp_scatter_by_slot", pending);
    if (mate_cmp_cache.defined()) {
      pack_mate_model1_cache(mate_cmp_cache, pending_slots, pending);
      maybe_sync_attention_stage(layer_id_, "decode_mate_cmp_pack", pending);
    }
    if (use_paged_indexer_cache && !decode_indexer_to_cache.empty()) {
      torch::Tensor pending_index;
      if (batch_compression) {
        auto rows = dsv4_pending_cache_rows(decode_indexer_to_cache,
                                            decode_indexer_tensors_scattered,
                                            decode_compressed_rows_scattered);
        CHECK(rows.values.defined());
        CHECK_EQ(rows.row_end, row_end);
        pending_index = rows.values;
      } else {
        auto index_all = decode_indexer_to_cache.size() == 1
                             ? decode_indexer_to_cache.front()
                             : torch::cat(decode_indexer_to_cache, /*dim=*/0);
        CHECK_EQ(index_all.size(0), row_end);
        pending_index = index_all.slice(
            /*dim=*/0,
            /*start=*/decode_compressed_rows_scattered,
            /*end=*/row_end);
      }
      auto flat_index_slots = index_slot_mapping.reshape({-1});
      CHECK_GE(flat_index_slots.numel(), row_end);
      auto pending_index_slots =
          attn_metadata.is_acl_graph
              ? flat_index_slots
                    .slice(
                        /*dim=*/0,
                        /*start=*/decode_compressed_rows_scattered,
                        /*end=*/row_end)
                    .contiguous()
              : index_slot_mapping;
      quant_scatter_indexer_cache(
          index_cache, index_scale_cache, pending_index_slots, pending_index);
      maybe_sync_attention_stage(
          layer_id_, "decode_indexer_cache_scatter", pending_index);
    }
    decode_compressed_rows_scattered = row_end;
    decode_compressed_tensors_scattered =
        static_cast<int64_t>(decode_compressed_to_cache.size());
    decode_indexer_tensors_scattered =
        static_cast<int64_t>(decode_indexer_to_cache.size());
    if (batch_compression) {
      LOG_FIRST_N(INFO, 1) << "[DSV4_INCREMENTAL_CACHE_FLUSH] layer="
                           << layer_id_ << " rows=" << pending.size(0)
                           << " end=" << row_end;
    }
  };
  int64_t decode_compressed_rope_offset = 0;
  const char* graph_q6_first_state_only_env =
      std::getenv("XLLM_DSV4_GRAPH_Q6_FIRST_STATE_ONLY");
  const bool graph_q6_first_state_only =
      graph_q6_first_state_only_env != nullptr &&
      std::string(graph_q6_first_state_only_env) == "1" &&
      compact_spec_verify && attn_metadata.is_acl_graph && total_tokens > 2 &&
      use_persistent_compressor_state;
  // Optional B16 fast path: q7 Indexer rows from all fixed-width sequence
  // views share one fused score/radix launch.  The kernel still selects the
  // block-table row and dynamic KV length from the sequence owning each row;
  // this only removes the per-sequence launch/queue boundary.
  bool batched_indexer_across_sequences = false;
  int64_t batched_indexer_query_rows = 0;
  int64_t batched_indexer_sequence_count = 0;
  torch::Tensor batched_indexer_indices;
  torch::Tensor batched_indexer_physical_indices;
  torch::Tensor batched_indexer_extra_physical_indices;
  const char* batched_indexer_env =
      std::getenv("XLLM_DSV4_BATCH_INDEXER_ACROSS_SEQUENCES");
  if (batched_indexer_env != nullptr &&
      std::string(batched_indexer_env) == "1" && compact_spec_verify_multi &&
      attn_metadata.is_acl_graph && dsv4_sequence_view_active &&
      batch_mate_across_sequences && mate_cmp_cache.defined() &&
      mate_swa_cache.defined() && apply_sparse_indexer_attention &&
      use_paged_indexer_cache && dsv4_musa_batched_graph_indexer_enabled() &&
      query_ranges.size() > 1 &&
      query_ranges.front().second > query_ranges.front().first) {
    const int64_t candidate_rows =
        query_ranges.front().second - query_ranges.front().first;
    bool fixed_width = candidate_rows > 2;
    for (const auto& range : query_ranges) {
      fixed_width = fixed_width && range.second - range.first == candidate_rows;
    }
    if (fixed_width &&
        candidate_rows * static_cast<int64_t>(query_ranges.size()) ==
            total_tokens &&
        graph_decode_kv_seq_lens.defined() &&
        graph_decode_kv_seq_lens.size(0) >= total_tokens &&
        decode_spec_indices_output_.defined() &&
        decode_spec_indices_output_.size(0) >= total_tokens &&
        decode_spec_mate_physical_indices_.defined() &&
        decode_spec_mate_physical_indices_.size(0) >= total_tokens &&
        decode_spec_mate_extra_physical_indices_.defined() &&
        decode_spec_mate_extra_physical_indices_.size(0) >= total_tokens) {
      int64_t max_index_len = 0;
      for (const int64_t kv_len : kv_lens) {
        max_index_len = std::max(max_index_len, kv_len / 4);
      }
      const int64_t index_capacity =
          dsv4_graph_indexer_capacity(max_index_len,
                                      /*multi_sequence=*/true);
      const char* exact_capacity_env =
          std::getenv("XLLM_DSV4_INDEXER_EXACT_RUNTIME_CAPACITY");
      const bool exact_runtime_capacity =
          exact_capacity_env != nullptr &&
          std::string(exact_capacity_env) == "1";
      const int64_t cache_capacity =
          index_block_table.size(1) * index_cache.size(1);
      CHECK(!exact_runtime_capacity)
          << "Exact capture-time capacity truncates growing KV history on "
             "Graph replay; use the graph-key capacity bucket";
      const int64_t fused_capacity = std::min(index_capacity, cache_capacity);
      if (fused_capacity >= index_topk_) {
        batched_indexer_query_rows = candidate_rows;
        batched_indexer_sequence_count = query_ranges.size();
        batched_indexer_indices = decode_spec_indices_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/total_tokens);
        batched_indexer_physical_indices =
            decode_spec_mate_physical_indices_.narrow(
                /*dim=*/0, /*start=*/0, /*length=*/total_tokens);
        batched_indexer_extra_physical_indices =
            decode_spec_mate_extra_physical_indices_.narrow(
                /*dim=*/0, /*start=*/0, /*length=*/total_tokens);
        batched_indexer_across_sequences = true;
      }
    }
  }
  // Define the launch here, but enqueue it only after every sequence has
  // produced and published this step's C4/Indexer rows. An earlier flush is
  // a no-op because both pending-cache vectors are still empty at this point.
  const char* global_short_family =
      std::getenv("XLLM_DSV4_GLOBAL_SHORT_GRAPH_FAMILY");
  if (compact_spec_verify_multi && max_kv_len <= 2048 &&
      global_short_family != nullptr &&
      std::string(global_short_family) == "1" && compress_ratio_ == 4 &&
      !force_full_kv_attention) {
    CHECK(batched_indexer_across_sequences)
        << "Global short Graph family must use the merged live-length Indexer";
  }
  auto launch_batched_indexer = [&]() {
    if (!batched_indexer_across_sequences) {
      return;
    }
    const int64_t candidate_rows = batched_indexer_query_rows;
    const int64_t fused_capacity = std::min(
        dsv4_graph_indexer_capacity(max_kv_len / 4, /*multi_sequence=*/true),
        index_block_table.size(1) * index_cache.size(1));
    // Bound each launch to the previously tested eight q7 sequences.
    // Larger flattened captures hung in v86; their root cause is not
    // established by the successful chunked workaround alone.
    constexpr int64_t kMaxBatchedIndexerSequences = 8;
    for (int64_t sequence_start = 0;
         sequence_start < batched_indexer_sequence_count;
         sequence_start += kMaxBatchedIndexerSequences) {
      const int64_t sequence_count =
          std::min<int64_t>(kMaxBatchedIndexerSequences,
                            batched_indexer_sequence_count - sequence_start);
      const int64_t row_start = query_ranges[sequence_start].first;
      const int64_t row_count = candidate_rows * sequence_count;
      auto chunk_indices = batched_indexer_indices.narrow(
          /*dim=*/0, row_start, row_count);
      auto chunk_physical = batched_indexer_physical_indices.narrow(
          /*dim=*/0, row_start, row_count);
      auto chunk_extra = batched_indexer_extra_physical_indices.narrow(
          /*dim=*/0, row_start, row_count);
      auto selected = build_indexer_topk(
          qr.narrow(/*dim=*/0, row_start, row_count),
          hidden_states.narrow(/*dim=*/0, row_start, row_count),
          packed_indexer_head_weights.defined()
              ? packed_indexer_head_weights.narrow(
                    /*dim=*/0, row_start, row_count)
              : torch::Tensor(),
          packed_indexer_query.defined()
              ? packed_indexer_query.narrow(/*dim=*/0, row_start, row_count)
              : torch::Tensor(),
          active_cos.narrow(/*dim=*/0, row_start, row_count),
          active_sin.narrow(/*dim=*/0, row_start, row_count),
          torch::Tensor(),
          index_cache,
          index_scale_cache,
          index_block_table.narrow(/*dim=*/0, sequence_start, sequence_count),
          /*sequence_index=*/0,
          fused_capacity,
          /*q_abs_start=*/0,
          chunk_indices,
          cmp_block_table.narrow(/*dim=*/0, sequence_start, sequence_count),
          swa_block_table.narrow(/*dim=*/0, sequence_start, sequence_count),
          max_kv_len,
          mate_cmp_cache.size(1),
          &chunk_physical,
          &chunk_extra,
          graph_decode_kv_seq_lens.narrow(/*dim=*/0, row_start, row_count),
          batched_indexer_query_rows,
          sequence_count);
      CHECK(selected.defined())
          << "Cross-sequence Graph indexer chunk did not execute";
    }
    LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_INDEXER_ACROSS_SEQUENCES] sequences="
                         << batched_indexer_sequence_count
                         << " query_rows=" << batched_indexer_query_rows
                         << " capacity=" << fused_capacity
                         << " after_cache_publish=1";
  };
  torch::Tensor batched_main_compressed;
  torch::Tensor batched_indexer_compressed;
  bool batch_indexer_hadamard = false;
  int64_t batched_compressed_stride = 0;
  const char* batch_compression_env =
      std::getenv("XLLM_DSV4_BATCH_COMPRESSION");
  const char* dynamic_state_env = std::getenv("XLLM_DSV4_DYNAMIC_GRAPH_STATE");
  const auto batch_projection =
      attn_metadata.cp_input_dict.find(kDsv4BatchedCompressorMetadataKey);
  batch_compression =
      batch_compression_env != nullptr &&
      std::string(batch_compression_env) == "1" &&
      dynamic_state_env != nullptr && std::string(dynamic_state_env) == "1" &&
      batched_state_gather && attn_metadata.is_acl_graph &&
      !freeze_compact_graph_compressor && !graph_q6_first_state_only &&
      use_compressed_attention &&
      (compress_ratio_ == 4 || compress_ratio_ == 128) &&
      (!effective_use_indexer_attention || use_persistent_indexer_state) &&
      batch_projection != attn_metadata.cp_input_dict.end();
  if (batch_compression) {
    // The state gather already owns contiguous [request, state-row, width]
    // tensors. Reuse them directly: no stack/copy and no collapsed identity.
    const int64_t sequences = static_cast<int64_t>(query_ranges.size());
    const int64_t query_width = attn_metadata.num_speculative_tokens + 1;
    CHECK_EQ(query_width, 7);
    CHECK_GT(sequences, 1);
    CHECK_LE(sequences, 16);
    CHECK_EQ(total_tokens, sequences * query_width);
    for (int64_t seq = 0; seq < sequences; ++seq) {
      CHECK_EQ(query_ranges[seq].first, seq * query_width);
      CHECK_EQ(query_ranges[seq].second, (seq + 1) * query_width);
    }
    batched_compressed_stride =
        dsv4_graph_compressed_stride(query_width, compress_ratio_);
    const int64_t output_rows = sequences * batched_compressed_stride;
    const int64_t main_width = compressor_coff_ * head_dim_;
    const int64_t index_width = 2 * index_head_dim_;
    auto projections =
        dsv4_compressor_projection_view(batch_projection->second,
                                        main_width,
                                        compress_ratio_ == 4 ? index_width : 0,
                                        effective_use_indexer_attention);
    CHECK_EQ(projections.size(0), total_tokens);
    CHECK_GE(decode_compressor_output_.size(0), output_rows);
    CHECK_GE(decode_compressor_norm_output_.size(0), output_rows);
    CHECK_GE(decode_compressor_rope_output_.size(0), output_rows);
    CHECK(compressed_cos.defined() && compressed_sin.defined());
    CHECK_GE(compressed_cos.size(0), output_rows);
    CHECK_GE(compressed_sin.size(0), output_rows);
    const auto batch_cos = compressed_cos.narrow(0, 0, output_rows);
    const auto batch_sin = compressed_sin.narrow(0, 0, output_rows);
    CHECK_GE(attn_metadata.input_positions.numel(), total_tokens);
    const auto batch_positions =
        attn_metadata.input_positions.narrow(0, 0, total_tokens);
    batched_main_compressed =
        kernel::musa::deepseek_v4_compress_decode_batch_graph(
            batched_decode_states[0],
            batched_decode_states[1],
            projections.narrow(1, 0, main_width),
            projections.narrow(1, main_width, main_width),
            compressor_ape_,
            compress_ratio_,
            head_dim_,
            query_width,
            batch_positions,
            decode_compressor_output_.narrow(0, 0, output_rows));
    batched_main_compressed = std::get<0>(compressor_norm_->forward(
        batched_main_compressed,
        std::nullopt,
        std::optional<torch::Tensor>(
            decode_compressor_norm_output_.narrow(0, 0, output_rows))));
    // The scalar helper deliberately caps fused RoPE at four tokens. Call
    // the same row-independent kernel directly for the tested batch shape,
    // without changing the global cap or falling back to allocating ATen.
    auto main_rope_output =
        decode_compressor_rope_output_.narrow(0, 0, output_rows);
    CHECK(kernel::musa::deepseek_v4_partial_rope(batched_main_compressed,
                                                 batch_cos,
                                                 batch_sin,
                                                 nope_head_dim_,
                                                 rope_head_dim_,
                                                 /*inverse=*/false,
                                                 main_rope_output));
    batched_main_compressed = main_rope_output;
    fp8_act_quant_roundtrip_inplace(batched_main_compressed,
                                    nope_head_dim_,
                                    /*block_size=*/64);
    if (effective_use_indexer_attention) {
      CHECK_EQ(compress_ratio_, 4);
      CHECK_EQ(batched_decode_states.size(), 4);
      CHECK_GE(decode_indexer_output_.size(0), output_rows);
      CHECK_GE(decode_indexer_norm_output_.size(0), output_rows);
      CHECK_GE(decode_indexer_rope_output_.size(0), output_rows);
      batched_indexer_compressed =
          kernel::musa::deepseek_v4_compress_decode_batch_graph(
              batched_decode_states[2],
              batched_decode_states[3],
              projections.narrow(1, 2 * main_width, index_width),
              projections.narrow(1, 2 * main_width + index_width, index_width),
              indexer_compressor_ape_,
              compress_ratio_,
              index_head_dim_,
              query_width,
              batch_positions,
              decode_indexer_output_.narrow(0, 0, output_rows));
      batched_indexer_compressed =
          std::get<0>(indexer_compressor_norm_->forward(
              batched_indexer_compressed,
              std::nullopt,
              std::optional<torch::Tensor>(
                  decode_indexer_norm_output_.narrow(0, 0, output_rows))));
      auto indexer_rope_output =
          decode_indexer_rope_output_.narrow(0, 0, output_rows);
      CHECK(kernel::musa::deepseek_v4_partial_rope(
          batched_indexer_compressed,
          batch_cos,
          batch_sin,
          index_head_dim_ - rope_head_dim_,
          rope_head_dim_,
          /*inverse=*/false,
          indexer_rope_output));
      batched_indexer_compressed = indexer_rope_output;
      const char* hadamard_env =
          std::getenv("XLLM_DSV4_BATCH_INDEXER_HADAMARD");
      batch_indexer_hadamard =
          hadamard_env != nullptr && std::string(hadamard_env) == "1" &&
          batched_compressed_stride == 2 && index_head_dim_ == 128 &&
          batched_indexer_compressed.scalar_type() == torch::kBFloat16;
      if (batch_indexer_hadamard) {
        // The q7/C4 transform is row-independent. Validate the merged BF16
        // result against each original two-row GEMM before enabling this path.
        batched_indexer_compressed = apply_indexer_hadamard(
            batched_indexer_compressed, indexer_hadamard_);
        LOG_FIRST_N(INFO, 1)
            << "[DSV4_BATCH_INDEXER_HADAMARD] sequences=" << sequences
            << " rows=" << output_rows << " dim=" << index_head_dim_;
      }
    }
    LOG_FIRST_N(INFO, 2) << "[DSV4_BATCH_COMPRESSION] sequences=" << sequences
                         << " query_width=" << query_width
                         << " ratio=" << compress_ratio_
                         << " output_rows=" << output_rows
                         << " layer=" << layer_id_;
  }
  torch::Tensor batched_sas_output;
  const auto sas_lengths_it =
      attn_metadata.cp_input_dict.find(kDsv4BatchedSasKvLengths);
  if (dspark_block_size_ > 0 && query_ranges.size() > 1 &&
      query_ranges.size() <= 16 && !(is_prefill && !is_chunked_prefill) &&
      !attn_metadata.is_acl_graph && decode_spec_attention_output_.defined() &&
      ori_kv.defined() && swa_block_table.defined() &&
      sas_lengths_it != attn_metadata.cp_input_dict.end() && [] {
        const char* batched = std::getenv("XLLM_DSV4_BATCH_PAGED_SAS");
        const char* scalar = std::getenv("XLLM_DSV4_FUSED_PAGED_SAS");
        return batched != nullptr && std::string(batched) == "1" &&
               scalar != nullptr && std::string(scalar) == "1";
      }()) {
    const int64_t sequences = static_cast<int64_t>(query_ranges.size());
    const int64_t query_width =
        query_ranges.front().second - query_ranges.front().first;
    const torch::Tensor& live_lengths = sas_lengths_it->second;
    bool fixed_ranges = query_width >= 5 && query_width <= 7 &&
                        query_width <= decode_workspace_tokens_ &&
                        sequences * query_width == q.size(0) &&
                        kv_lens.size() == query_ranges.size();
    for (int64_t sequence = 0; sequence < sequences; ++sequence) {
      fixed_ranges =
          fixed_ranges &&
          query_ranges[sequence].first == sequence * query_width &&
          query_ranges[sequence].second == (sequence + 1) * query_width;
    }
    if (fixed_ranges && live_lengths.dim() == 1 &&
        live_lengths.numel() == sequences &&
        live_lengths.device() == q.device() &&
        live_lengths.scalar_type() == torch::kInt32 &&
        live_lengths.is_contiguous() && swa_block_table.dim() == 2 &&
        swa_block_table.size(0) >= sequences &&
        window_size_ + query_width <= 256) {
      CHECK_EQ(compress_ratio_, 1);
      CHECK_LE(max_kv_len, swa_block_table.size(1) * ori_kv.size(1));
      // The metadata builder creates this device tensor and the scalar path's
      // host sidecar from the same ctx_lens vector. Reuse it without new per-
      // layer H2D copies. All cache rows were published before this loop.
      auto sas_query = q.is_contiguous() ? q : q.contiguous();
      batched_sas_output = torch::empty(q.sizes(), q.options());
      kernel::musa::deepseek_v4_paged_attention_sas_batched(
          sas_query,
          ori_kv,
          swa_block_table.narrow(0, 0, sequences),
          live_lengths,
          query_width,
          window_size_ + query_width,
          attn_sink_.to(q.device()).to(torch::kFloat32).contiguous(),
          softmax_scale_,
          batched_sas_output);
      LOG_FIRST_N(INFO, 3) << "[DSV4_BATCH_PAGED_SAS] layer=" << layer_id_
                           << " sequences=" << sequences
                           << " query_width=" << query_width
                           << " draft_noncausal=1 device_lengths_reused=1";
      if (sequences == 16) {
        LOG_FIRST_N(INFO, 1)
            << "[DSV4_BATCH_PAGED_SAS_B16] query_width=" << query_width
            << " actual_rows=" << q.size(0)
            << " device_length_rows=" << live_lengths.numel();
      }
    }
  }
  for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
    const int64_t q_start = query_ranges[seq_idx].first;
    const int64_t q_end = query_ranges[seq_idx].second;
    if (q_end <= q_start) {
      continue;
    }
    const size_t state_seq_idx = state_sequence_index_for(seq_idx);
    auto q_seq = q.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
    const auto sequence_state_position =
        compact_spec_verify_multi
            ? attn_metadata.input_positions.narrow(0, q_start, 1)
            : compact_decode_state_position;
    torch::Tensor kv_seq;
    const bool use_current_prefill_kv = is_prefill && !is_chunked_prefill;
    const bool use_fused_paged_sas =
        dspark_block_size_ > 0 && !use_current_prefill_kv &&
        !attn_metadata.is_acl_graph &&
        q_seq.size(0) <= decode_workspace_tokens_ &&
        decode_spec_attention_output_.defined() && ori_kv.defined() &&
        swa_block_table.defined() && [] {
          const char* env = std::getenv("XLLM_DSV4_FUSED_PAGED_SAS");
          return env != nullptr && std::string(env) == "1";
        }();
    const bool use_paged_decode_attention =
        dspark_block_size_ == 0 && !use_current_prefill_kv &&
        !effective_use_indexer_attention &&
        dsv4_musa_paged_dense_decode_enabled() &&
        dsv4_musa_fused_dense_decode_enabled();
    // Mate consumes the packed compressed/SWA caches directly. Avoid
    // materializing the complete dense SWA history unless a fallback path
    // actually needs it.
    const bool defer_swa_gather_for_mate =
        !use_current_prefill_kv && effective_use_indexer_attention &&
        (query_ranges.size() <= 1 ||
         dsv4_musa_mate_fused_dsa_multi_sequence_enabled()) &&
        mate_cmp_cache.defined() && mate_swa_cache.defined() &&
        cmp_block_table.defined() && swa_block_table.defined();
    if (use_current_prefill_kv) {
      kv_seq =
          sp_context == nullptr
              ? kv.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end).squeeze(1)
              : kv.squeeze(1);
    } else if (!use_paged_decode_attention && !defer_swa_gather_for_mate &&
               !use_fused_paged_sas) {
      kv_seq = gather_cache(ori_kv,
                            swa_block_table,
                            static_cast<int64_t>(seq_idx),
                            kv_lens[seq_idx]);
      maybe_sync_attention_stage(layer_id_, "gather_swa_cache", kv_seq);
      if (!kv_seq.defined()) {
        kv_seq =
            kv.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end).squeeze(1);
      }
    }
    torch::Tensor cmp_seq;
    torch::Tensor paged_cmp_cache;
    torch::Tensor paged_cmp_block_table;
    int64_t paged_cmp_len = 0;
    std::vector<torch::Tensor> current_compressed_rows;
    if (use_compressed_attention) {
      if (use_current_prefill_kv) {
        // Full-KV normal prefill still builds the compressed cache above, but
        // must not merge the compressed rows into the dense attention result.
        if (!force_full_kv_attention) {
          cmp_seq = prefill_compressed_by_seq[seq_idx];
        }
      } else {
        if (retain_compressed_history &&
            state_seq_idx >= compressor_kv_history_.size()) {
          compressor_kv_history_.resize(state_seq_idx + 1);
        }
        if (effective_use_indexer_attention && !use_paged_indexer_cache &&
            state_seq_idx >= indexer_kv_history_.size()) {
          indexer_kv_history_.resize(state_seq_idx + 1);
        }
        if (effective_use_indexer_attention &&
            state_seq_idx >= indexer_kv_state_.size()) {
          indexer_kv_state_.resize(state_seq_idx + 1);
          indexer_score_state_.resize(state_seq_idx + 1);
        }
        auto hidden_seq =
            hidden_states.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
        if (!freeze_compact_graph_compressor && !is_chunked_prefill &&
            hidden_seq.defined() && hidden_seq.numel() > 0) {
          const int64_t seq_q_len = q_end - q_start;
          const int64_t projection_start =
              sp_context == nullptr ? q_start
                                    : sp_context->global_row_start +
                                          sp_context->local_offset + q_start;
          const bool reuse_joint_chunked_projections =
              is_chunked_prefill && joint_main_kv_projection.defined() &&
              joint_main_score_projection.defined() &&
              joint_indexer_kv_projection.defined() &&
              joint_indexer_score_projection.defined() &&
              joint_main_kv_projection.size(0) >=
                  projection_start + seq_q_len &&
              joint_main_score_projection.size(0) >=
                  projection_start + seq_q_len &&
              joint_indexer_kv_projection.size(0) >=
                  projection_start + seq_q_len &&
              joint_indexer_score_projection.size(0) >=
                  projection_start + seq_q_len;
          const int64_t decode_start_pos =
              std::max<int64_t>(kv_lens[seq_idx] - seq_q_len, 0);
          torch::Tensor fused_chunked_main_compressed;
          torch::Tensor fused_decode_main_compressed;
          torch::Tensor fused_decode_indexer_compressed;
          bool sequence_decode_state = false;
          const int64_t main_state_rows = compressor_coff_ * compress_ratio_;
          const int64_t main_state_dim = compressor_coff_ * head_dim_;
          const bool fuse_chunked_main_state_sequence =
              reuse_joint_chunked_projections && compress_ratio_ == 4 &&
              compressor_coff_ == 2 && compressor_ape_.defined() &&
              compressor_ape_.scalar_type() == torch::kFloat32 &&
              compressor_ape_.is_contiguous() &&
              state_seq_idx < compressor_kv_state_.size() &&
              state_seq_idx < compressor_score_state_.size() &&
              compressor_kv_state_[state_seq_idx].defined() &&
              compressor_score_state_[state_seq_idx].defined() &&
              compressor_kv_state_[state_seq_idx].sizes() ==
                  torch::IntArrayRef({main_state_rows, main_state_dim}) &&
              compressor_score_state_[state_seq_idx].sizes() ==
                  compressor_kv_state_[state_seq_idx].sizes() &&
              joint_main_kv_projection.stride(1) == 1 &&
              joint_main_score_projection.stride(1) == 1;
          if (fuse_chunked_main_state_sequence) {
            auto main_kv_sequence = joint_main_kv_projection.slice(
                /*dim=*/0,
                /*start=*/projection_start,
                /*end=*/projection_start + seq_q_len);
            auto main_score_sequence = joint_main_score_projection.slice(
                /*dim=*/0,
                /*start=*/projection_start,
                /*end=*/projection_start + seq_q_len);
            fused_chunked_main_compressed =
                kernel::musa::deepseek_v4_compress_decode_sequence_c4(
                    compressor_kv_state_[state_seq_idx],
                    compressor_score_state_[state_seq_idx],
                    main_kv_sequence,
                    main_score_sequence,
                    compressor_ape_,
                    head_dim_,
                    decode_start_pos);
            if (fused_chunked_main_compressed.numel() > 0) {
              fused_chunked_main_compressed = std::get<0>(
                  compressor_norm_->forward(fused_chunked_main_compressed));
              const int64_t fused_rows = fused_chunked_main_compressed.size(0);
              if (compressed_cos.defined() && compressed_sin.defined() &&
                  compressed_cos.size(0) >=
                      decode_compressed_rope_offset + fused_rows &&
                  compressed_sin.size(0) >=
                      decode_compressed_rope_offset + fused_rows) {
                fused_chunked_main_compressed = apply_interleaved_partial_rope(
                    fused_chunked_main_compressed,
                    nope_head_dim_,
                    rope_head_dim_,
                    compressed_cos.slice(
                        /*dim=*/0,
                        /*start=*/decode_compressed_rope_offset,
                        /*end=*/decode_compressed_rope_offset + fused_rows),
                    compressed_sin.slice(
                        /*dim=*/0,
                        /*start=*/decode_compressed_rope_offset,
                        /*end=*/decode_compressed_rope_offset + fused_rows));
              }
              fp8_act_quant_roundtrip_inplace(fused_chunked_main_compressed,
                                              nope_head_dim_,
                                              /*block_size=*/64);
            }
          }
          constexpr int64_t kIndexerCompressRatio = 4;
          constexpr int64_t kIndexerCompressorCoeff = 2;
          const int64_t indexer_state_rows =
              kIndexerCompressRatio * kIndexerCompressorCoeff;
          const int64_t indexer_state_dim =
              kIndexerCompressorCoeff * index_head_dim_;
          const bool fuse_chunked_indexer_state_only =
              reuse_joint_chunked_projections &&
              effective_use_indexer_attention &&
              compress_ratio_ == kIndexerCompressRatio &&
              indexer_compressor_ape_.defined() &&
              indexer_compressor_ape_.scalar_type() == torch::kFloat32 &&
              indexer_compressor_ape_.is_contiguous() &&
              decode_indexer_norm_output_.defined() &&
              decode_indexer_rope_output_.defined() &&
              state_seq_idx < indexer_kv_state_.size() &&
              state_seq_idx < indexer_score_state_.size() &&
              indexer_kv_state_[state_seq_idx].defined() &&
              indexer_score_state_[state_seq_idx].defined() &&
              indexer_kv_state_[state_seq_idx].sizes() ==
                  torch::IntArrayRef({indexer_state_rows, indexer_state_dim}) &&
              indexer_score_state_[state_seq_idx].sizes() ==
                  indexer_kv_state_[state_seq_idx].sizes() &&
              joint_indexer_kv_projection.stride(1) == 1 &&
              joint_indexer_score_projection.stride(1) == 1;
          torch::Tensor fused_chunked_indexer_raw;
          const char* dynamic_graph_state_env =
              std::getenv("XLLM_DSV4_DYNAMIC_GRAPH_STATE");
          const bool use_dynamic_graph_state =
              dynamic_graph_state_env != nullptr &&
              std::string(dynamic_graph_state_env) == "1";
          const bool use_batched_decode_state =
              !is_chunked_prefill && compact_spec_verify &&
              attn_metadata.is_acl_graph && total_tokens > 2 &&
              (compress_ratio_ == 4 ||
               (compress_ratio_ == 128 && use_dynamic_graph_state)) &&
              use_persistent_compressor_state &&
              state_seq_idx < compressor_kv_state_.size() &&
              state_seq_idx < compressor_score_state_.size() &&
              compressor_kv_state_[state_seq_idx].defined() &&
              compressor_score_state_[state_seq_idx].defined() &&
              compressor_ape_.defined() &&
              compressor_ape_.scalar_type() == torch::kFloat32 &&
              compressor_ape_.is_contiguous() &&
              decode_compressor_kv_projection_.size(0) >= seq_q_len &&
              decode_compressor_score_projection_.size(0) >= seq_q_len &&
              (!effective_use_indexer_attention ||
               (use_persistent_indexer_state &&
                state_seq_idx < indexer_kv_state_.size() &&
                state_seq_idx < indexer_score_state_.size() &&
                indexer_kv_state_[state_seq_idx].defined() &&
                indexer_score_state_[state_seq_idx].defined() &&
                indexer_compressor_ape_.defined() &&
                indexer_compressor_ape_.scalar_type() == torch::kFloat32 &&
                indexer_compressor_ape_.is_contiguous() &&
                decode_indexer_kv_projection_.size(0) >= seq_q_len &&
                decode_indexer_score_projection_.size(0) >= seq_q_len));
          if (batch_compression) {
            CHECK(use_batched_decode_state);
            CHECK_EQ(decode_compressed_rope_offset,
                     static_cast<int64_t>(seq_idx) * batched_compressed_stride);
            fused_decode_main_compressed = batched_main_compressed.narrow(
                0, decode_compressed_rope_offset, batched_compressed_stride);
            if (effective_use_indexer_attention) {
              auto indexer_rows = batched_indexer_compressed.narrow(
                  0, decode_compressed_rope_offset, batched_compressed_stride);
              fused_decode_indexer_compressed =
                  batch_indexer_hadamard
                      ? indexer_rows
                      : apply_indexer_hadamard(indexer_rows, indexer_hadamard_);
            }
            decode_compressed_rope_offset += batched_compressed_stride;
            sequence_decode_state = true;
          } else if (use_batched_decode_state) {
            // In the q6 Graph fast-state A/B we only need to commit the
            // first target row.  The previous implementation applied this
            // policy only to the scalar fallback below, while the batched
            // compressor/indexer path still projected and compressed all
            // speculative rows.  That made GRAPH_Q6_FIRST_STATE_ONLY a
            // no-op for the expensive path.  Keep the attention verification
            // rows unchanged, but bound state projection/update work to the
            // first row when explicitly requested.
            const int64_t state_sequence_tokens =
                graph_q6_first_state_only ? std::min<int64_t>(seq_q_len, 1)
                                          : seq_q_len;
            torch::Tensor packed_decode_projections;
            const int64_t main_projection_width = compressor_coff_ * head_dim_;
            const int64_t index_projection_width = 2 * index_head_dim_;
            const int64_t packed_projection_width =
                2 * main_projection_width + (effective_use_indexer_attention
                                                 ? 2 * index_projection_width
                                                 : 0);
            const char* joint_decode_env =
                std::getenv("XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION");
            const auto batch_compressor = attn_metadata.cp_input_dict.find(
                kDsv4BatchedCompressorMetadataKey);
            if (batch_compressor != attn_metadata.cp_input_dict.end()) {
              packed_decode_projections =
                  dsv4_compressor_projection_view(
                      batch_compressor->second,
                      main_projection_width,
                      compress_ratio_ == 4 ? index_projection_width : 0,
                      effective_use_indexer_attention)
                      .narrow(
                          /*dim=*/0,
                          /*start=*/q_start,
                          /*length=*/state_sequence_tokens);
            } else if (compress_ratio_ == 4 && joint_decode_env != nullptr &&
                       std::string(joint_decode_env) == "1" &&
                       sp_joint_compressor_projection_weight_.defined() &&
                       sp_joint_compressor_projection_weight_.size(0) ==
                           packed_projection_width &&
                       decode_compressor_kv_projection_.numel() >=
                           state_sequence_tokens * packed_projection_width) {
              // Reuse the existing persistent allocation as a larger-row
              // carrier. The compressor kernels accept explicit row strides,
              // so its column views need no scatter/copy or capture allocation.
              packed_decode_projections =
                  decode_compressor_kv_projection_.view({-1})
                      .narrow(
                          0, 0, state_sequence_tokens * packed_projection_width)
                      .view({state_sequence_tokens, packed_projection_width});
              packed_decode_projections = dsv4_pack_compressor_projections(
                  hidden_seq.narrow(0, 0, state_sequence_tokens),
                  state_sequence_tokens,
                  main_projection_width,
                  index_projection_width,
                  [this](const torch::Tensor& input) {
                    return compressor_wkv_->forward(input);
                  },
                  [this](const torch::Tensor& input) {
                    return compressor_wgate_->forward(input);
                  },
                  [this](const torch::Tensor& input) {
                    return indexer_compressor_wkv_->forward(input);
                  },
                  [this](const torch::Tensor& input) {
                    return indexer_compressor_wgate_->forward(input);
                  },
                  packed_decode_projections);
              LOG_FIRST_N(INFO, 1)
                  << "[DSV4_PACKED_DECODE_PROJECTION] ratio=" << compress_ratio_
                  << " columns=" << packed_projection_width;
            }
            const int64_t sequence_rope_offset = decode_compressed_rope_offset;
            const int64_t first_boundary =
                (compress_ratio_ - ((decode_start_pos + 1) % compress_ratio_)) %
                compress_ratio_;
            const int64_t sequence_boundary_count =
                attn_metadata.cp_input_dict.count(kDsv4GraphSequenceView) != 0
                    ? dsv4_graph_compressed_stride(state_sequence_tokens,
                                                   compress_ratio_)
                : first_boundary < state_sequence_tokens
                    ? 1 + (state_sequence_tokens - 1 - first_boundary) /
                              compress_ratio_
                    : 0;
            auto main_kv_steps =
                packed_decode_projections.defined()
                    ? packed_decode_projections.narrow(
                          1, 0, main_projection_width)
                    : kernel::musa::matmul(
                          hidden_seq.narrow(/*dim=*/0,
                                            /*start=*/0,
                                            /*length=*/state_sequence_tokens),
                          compressor_wkv_->weight(),
                          std::nullopt,
                          std::optional<torch::Tensor>(
                              decode_compressor_kv_projection_.narrow(
                                  /*dim=*/0,
                                  /*start=*/0,
                                  /*length=*/state_sequence_tokens)));
            auto main_score_steps =
                packed_decode_projections.defined()
                    ? packed_decode_projections.narrow(
                          1, main_projection_width, main_projection_width)
                    : kernel::musa::matmul(
                          hidden_seq.narrow(/*dim=*/0,
                                            /*start=*/0,
                                            /*length=*/state_sequence_tokens),
                          compressor_wgate_->weight(),
                          std::nullopt,
                          std::optional<torch::Tensor>(
                              decode_compressor_score_projection_.narrow(
                                  /*dim=*/0,
                                  /*start=*/0,
                                  /*length=*/state_sequence_tokens)));
            if (use_dynamic_graph_state) {
              fused_decode_main_compressed =
                  kernel::musa::deepseek_v4_compress_decode_sequence_graph(
                      compressor_kv_state_[state_seq_idx],
                      compressor_score_state_[state_seq_idx],
                      main_kv_steps,
                      main_score_steps,
                      compressor_ape_,
                      compress_ratio_,
                      head_dim_,
                      sequence_state_position,
                      decode_compressor_output_.narrow(
                          /*dim=*/0,
                          /*start=*/0,
                          /*length=*/sequence_boundary_count));
            } else {
              fused_decode_main_compressed =
                  kernel::musa::deepseek_v4_compress_decode_sequence_c4(
                      compressor_kv_state_[state_seq_idx],
                      compressor_score_state_[state_seq_idx],
                      main_kv_steps,
                      main_score_steps,
                      compressor_ape_,
                      head_dim_,
                      decode_start_pos,
                      std::optional<torch::Tensor>(
                          decode_compressor_output_.narrow(
                              /*dim=*/0,
                              /*start=*/0,
                              /*length=*/sequence_boundary_count)));
            }
            if (fused_decode_main_compressed.defined() &&
                fused_decode_main_compressed.numel() > 0) {
              const int64_t boundary_rows =
                  fused_decode_main_compressed.size(0);
              fused_decode_main_compressed =
                  std::get<0>(compressor_norm_->forward(
                      fused_decode_main_compressed,
                      std::nullopt,
                      std::optional<torch::Tensor>(
                          decode_compressor_norm_output_.narrow(
                              /*dim=*/0,
                              /*start=*/0,
                              /*length=*/boundary_rows))));
              if (compressed_cos.defined() && compressed_sin.defined() &&
                  compressed_cos.size(0) >=
                      sequence_rope_offset + boundary_rows &&
                  compressed_sin.size(0) >=
                      sequence_rope_offset + boundary_rows) {
                fused_decode_main_compressed = apply_interleaved_partial_rope(
                    fused_decode_main_compressed,
                    nope_head_dim_,
                    rope_head_dim_,
                    compressed_cos.slice(
                        /*dim=*/0,
                        /*start=*/sequence_rope_offset,
                        /*end=*/sequence_rope_offset + boundary_rows),
                    compressed_sin.slice(
                        /*dim=*/0,
                        /*start=*/sequence_rope_offset,
                        /*end=*/sequence_rope_offset + boundary_rows),
                    /*inverse=*/false,
                    decode_compressor_rope_output_.narrow(
                        /*dim=*/0, /*start=*/0, /*length=*/boundary_rows));
              }
              fp8_act_quant_roundtrip_inplace(fused_decode_main_compressed,
                                              nope_head_dim_,
                                              /*block_size=*/64);
            }
            if (effective_use_indexer_attention) {
              auto indexer_kv_steps =
                  packed_decode_projections.defined()
                      ? packed_decode_projections.narrow(
                            1,
                            2 * main_projection_width,
                            index_projection_width)
                      : kernel::musa::matmul(
                            hidden_seq.narrow(/*dim=*/0,
                                              /*start=*/0,
                                              /*length=*/state_sequence_tokens),
                            indexer_compressor_wkv_->weight(),
                            std::nullopt,
                            std::optional<torch::Tensor>(
                                decode_indexer_kv_projection_.narrow(
                                    /*dim=*/0,
                                    /*start=*/0,
                                    /*length=*/state_sequence_tokens)));
              auto indexer_score_steps =
                  packed_decode_projections.defined()
                      ? packed_decode_projections.narrow(
                            1,
                            2 * main_projection_width + index_projection_width,
                            index_projection_width)
                      : kernel::musa::matmul(
                            hidden_seq.narrow(/*dim=*/0,
                                              /*start=*/0,
                                              /*length=*/state_sequence_tokens),
                            indexer_compressor_wgate_->weight(),
                            std::nullopt,
                            std::optional<torch::Tensor>(
                                decode_indexer_score_projection_.narrow(
                                    /*dim=*/0,
                                    /*start=*/0,
                                    /*length=*/state_sequence_tokens)));
              if (use_dynamic_graph_state) {
                fused_decode_indexer_compressed =
                    kernel::musa::deepseek_v4_compress_decode_sequence_graph(
                        indexer_kv_state_[state_seq_idx],
                        indexer_score_state_[state_seq_idx],
                        indexer_kv_steps,
                        indexer_score_steps,
                        indexer_compressor_ape_,
                        compress_ratio_,
                        index_head_dim_,
                        sequence_state_position,
                        decode_indexer_output_.narrow(
                            /*dim=*/0,
                            /*start=*/0,
                            /*length=*/sequence_boundary_count));
              } else {
                fused_decode_indexer_compressed =
                    kernel::musa::deepseek_v4_compress_decode_sequence_c4(
                        indexer_kv_state_[state_seq_idx],
                        indexer_score_state_[state_seq_idx],
                        indexer_kv_steps,
                        indexer_score_steps,
                        indexer_compressor_ape_,
                        index_head_dim_,
                        decode_start_pos,
                        std::optional<torch::Tensor>(
                            decode_indexer_output_.narrow(
                                /*dim=*/0,
                                /*start=*/0,
                                /*length=*/sequence_boundary_count)));
              }
              if (fused_decode_indexer_compressed.defined() &&
                  fused_decode_indexer_compressed.numel() > 0) {
                const int64_t boundary_rows =
                    fused_decode_indexer_compressed.size(0);
                fused_decode_indexer_compressed =
                    std::get<0>(indexer_compressor_norm_->forward(
                        fused_decode_indexer_compressed,
                        std::nullopt,
                        std::optional<torch::Tensor>(
                            decode_indexer_norm_output_.narrow(
                                /*dim=*/0,
                                /*start=*/0,
                                /*length=*/boundary_rows))));
                if (compressed_cos.defined() && compressed_sin.defined() &&
                    compressed_cos.size(0) >=
                        sequence_rope_offset + boundary_rows &&
                    compressed_sin.size(0) >=
                        sequence_rope_offset + boundary_rows) {
                  fused_decode_indexer_compressed =
                      apply_interleaved_partial_rope(
                          fused_decode_indexer_compressed,
                          index_head_dim_ - rope_head_dim_,
                          rope_head_dim_,
                          compressed_cos.slice(
                              /*dim=*/0,
                              /*start=*/sequence_rope_offset,
                              /*end=*/sequence_rope_offset + boundary_rows),
                          compressed_sin.slice(
                              /*dim=*/0,
                              /*start=*/sequence_rope_offset,
                              /*end=*/sequence_rope_offset + boundary_rows),
                          /*inverse=*/false,
                          decode_indexer_rope_output_.narrow(
                              /*dim=*/0,
                              /*start=*/0,
                              /*length=*/boundary_rows));
                }
                fused_decode_indexer_compressed = apply_indexer_hadamard(
                    fused_decode_indexer_compressed, indexer_hadamard_);
              }
            }
            if (fused_decode_main_compressed.defined()) {
              decode_compressed_rope_offset +=
                  fused_decode_main_compressed.size(0);
            }
            sequence_decode_state = true;
          }
          if (fuse_chunked_indexer_state_only) {
            fused_chunked_indexer_raw =
                kernel::musa::deepseek_v4_compress_decode_sequence_c4(
                    indexer_kv_state_[state_seq_idx],
                    indexer_score_state_[state_seq_idx],
                    joint_indexer_kv_projection.slice(
                        /*dim=*/0,
                        /*start=*/projection_start,
                        /*end=*/projection_start + seq_q_len),
                    joint_indexer_score_projection.slice(
                        /*dim=*/0,
                        /*start=*/projection_start,
                        /*end=*/projection_start + seq_q_len),
                    indexer_compressor_ape_,
                    index_head_dim_,
                    decode_start_pos);
          }
          int64_t fused_indexer_output_row = 0;
          const int64_t state_update_q_len =
              graph_q6_first_state_only ? std::min<int64_t>(seq_q_len, 1)
                                        : seq_q_len;
          if (!sequence_decode_state) {
            for (int64_t token_idx = 0; token_idx < state_update_q_len;
                 ++token_idx) {
              const int64_t abs_pos = decode_start_pos + token_idx;
              torch::Tensor cos_row;
              torch::Tensor sin_row;
              if ((abs_pos + 1) % compress_ratio_ == 0) {
                if (compressed_cos.defined() && compressed_sin.defined() &&
                    compressed_cos.size(0) > decode_compressed_rope_offset &&
                    compressed_sin.size(0) > decode_compressed_rope_offset) {
                  cos_row = compressed_cos.slice(
                      /*dim=*/0,
                      /*start=*/decode_compressed_rope_offset,
                      /*end=*/decode_compressed_rope_offset + 1);
                  sin_row = compressed_sin.slice(
                      /*dim=*/0,
                      /*start=*/decode_compressed_rope_offset,
                      /*end=*/decode_compressed_rope_offset + 1);
                }
                ++decode_compressed_rope_offset;
              }
              auto token_hidden = hidden_seq.slice(
                  /*dim=*/0, /*start=*/token_idx, /*end=*/token_idx + 1);
              const int64_t projection_row = projection_start + token_idx;
              auto main_kv_projection =
                  reuse_joint_chunked_projections &&
                          !fuse_chunked_main_state_sequence
                      ? joint_main_kv_projection.slice(
                            /*dim=*/0,
                            /*start=*/projection_row,
                            /*end=*/projection_row + 1)
                      : torch::Tensor();
              auto main_score_projection =
                  reuse_joint_chunked_projections &&
                          !fuse_chunked_main_state_sequence
                      ? joint_main_score_projection.slice(
                            /*dim=*/0,
                            /*start=*/projection_row,
                            /*end=*/projection_row + 1)
                      : torch::Tensor();
              auto new_cmp = fuse_chunked_main_state_sequence
                                 ? torch::Tensor()
                                 : compress_decode_step(token_hidden,
                                                        abs_pos,
                                                        cos_row,
                                                        sin_row,
                                                        state_seq_idx,
                                                        main_kv_projection,
                                                        main_score_projection);
              torch::Tensor new_indexer_cmp;
              if (effective_use_indexer_attention) {
                if (fuse_chunked_indexer_state_only) {
                  if ((abs_pos + 1) % kIndexerCompressRatio == 0) {
                    CHECK_LT(fused_indexer_output_row,
                             fused_chunked_indexer_raw.size(0));
                    new_indexer_cmp = fused_chunked_indexer_raw.slice(
                        /*dim=*/0,
                        /*start=*/fused_indexer_output_row,
                        /*end=*/fused_indexer_output_row + 1);
                    ++fused_indexer_output_row;
                    new_indexer_cmp =
                        std::get<0>(indexer_compressor_norm_->forward(
                            new_indexer_cmp,
                            std::nullopt,
                            std::optional<torch::Tensor>(
                                decode_indexer_norm_output_)));
                    bool fused_rope_hadamard = false;
                    if (cos_row.defined() && sin_row.defined() &&
                        cos_row.size(0) > 0 && sin_row.size(0) > 0) {
                      fused_rope_hadamard =
                          kernel::musa::deepseek_v4_rope_hadamard_inplace(
                              new_indexer_cmp, cos_row, sin_row);
                    }
                    if (!fused_rope_hadamard && cos_row.defined() &&
                        sin_row.defined() && cos_row.size(0) > 0 &&
                        sin_row.size(0) > 0) {
                      new_indexer_cmp = apply_interleaved_partial_rope(
                          new_indexer_cmp,
                          index_head_dim_ - rope_head_dim_,
                          rope_head_dim_,
                          cos_row,
                          sin_row,
                          /*inverse=*/false,
                          decode_indexer_rope_output_);
                    }
                    if (!fused_rope_hadamard) {
                      new_indexer_cmp = apply_indexer_hadamard(
                          new_indexer_cmp, indexer_hadamard_);
                    }
                  }
                } else {
                  auto indexer_kv_projection =
                      reuse_joint_chunked_projections
                          ? joint_indexer_kv_projection.slice(
                                /*dim=*/0,
                                /*start=*/projection_row,
                                /*end=*/projection_row + 1)
                          : torch::Tensor();
                  auto indexer_score_projection =
                      reuse_joint_chunked_projections
                          ? joint_indexer_score_projection.slice(
                                /*dim=*/0,
                                /*start=*/projection_row,
                                /*end=*/projection_row + 1)
                          : torch::Tensor();
                  new_indexer_cmp =
                      compress_indexer_decode_step(token_hidden,
                                                   abs_pos,
                                                   cos_row,
                                                   sin_row,
                                                   state_seq_idx,
                                                   indexer_kv_projection,
                                                   indexer_score_projection);
                }
              }
              if (new_cmp.defined() && new_cmp.numel() > 0) {
                if (retain_compressed_history) {
                  if (compressor_kv_history_[state_seq_idx].defined() &&
                      compressor_kv_history_[state_seq_idx].numel() > 0) {
                    compressor_kv_history_[state_seq_idx] =
                        torch::cat(
                            {compressor_kv_history_[state_seq_idx], new_cmp},
                            /*dim=*/0)
                            .contiguous();
                  } else {
                    compressor_kv_history_[state_seq_idx] =
                        new_cmp.contiguous();
                  }
                }
                current_compressed_rows.push_back(new_cmp);
                decode_compressed_to_cache.push_back(
                    batched_indexer_across_sequences ? new_cmp.clone()
                                                     : new_cmp);
              }
              if (new_indexer_cmp.defined() && new_indexer_cmp.numel() > 0) {
                if (use_paged_indexer_cache) {
                  decode_indexer_to_cache.push_back(
                      batched_indexer_across_sequences ? new_indexer_cmp.clone()
                                                       : new_indexer_cmp);
                } else {
                  if (indexer_kv_history_[state_seq_idx].defined() &&
                      indexer_kv_history_[state_seq_idx].numel() > 0) {
                    indexer_kv_history_[state_seq_idx] =
                        torch::cat({indexer_kv_history_[state_seq_idx],
                                    new_indexer_cmp},
                                   /*dim=*/0)
                            .contiguous();
                  } else {
                    indexer_kv_history_[state_seq_idx] =
                        new_indexer_cmp.contiguous();
                  }
                }
              }
            }
          }
          if (fuse_chunked_indexer_state_only) {
            CHECK_EQ(fused_indexer_output_row,
                     fused_chunked_indexer_raw.size(0));
          }
          if (sequence_decode_state && fused_decode_main_compressed.defined() &&
              fused_decode_main_compressed.numel() > 0) {
            current_compressed_rows.push_back(fused_decode_main_compressed);
            decode_compressed_to_cache.push_back(
                batched_indexer_across_sequences && !batch_compression
                    ? fused_decode_main_compressed.clone()
                    : fused_decode_main_compressed);
          }
          if (sequence_decode_state &&
              fused_decode_indexer_compressed.defined() &&
              fused_decode_indexer_compressed.numel() > 0 &&
              use_paged_indexer_cache) {
            decode_indexer_to_cache.push_back(
                batched_indexer_across_sequences && !batch_compression
                    ? fused_decode_indexer_compressed.clone()
                    : fused_decode_indexer_compressed);
          }
          if (fused_chunked_main_compressed.defined() &&
              fused_chunked_main_compressed.numel() > 0) {
            if (retain_compressed_history) {
              if (compressor_kv_history_[state_seq_idx].defined() &&
                  compressor_kv_history_[state_seq_idx].numel() > 0) {
                compressor_kv_history_[state_seq_idx] =
                    torch::cat({compressor_kv_history_[state_seq_idx],
                                fused_chunked_main_compressed},
                               /*dim=*/0)
                        .contiguous();
              } else {
                compressor_kv_history_[state_seq_idx] =
                    fused_chunked_main_compressed.contiguous();
              }
            }
            current_compressed_rows.push_back(fused_chunked_main_compressed);
            decode_compressed_to_cache.push_back(fused_chunked_main_compressed);
          }
        }
        const int64_t cmp_len = kv_lens[seq_idx] / compress_ratio_;
        int64_t dense_tail_len = 0;
        for (const auto& rows : current_compressed_rows) {
          dense_tail_len += rows.size(0);
        }
        if (use_paged_decode_attention &&
            dsv4_musa_paged_compressed_decode_enabled() &&
            cmp_len >= dense_tail_len && cmp_cache.defined() &&
            cmp_block_table.defined()) {
          paged_cmp_cache = cmp_cache;
          paged_cmp_block_table = cmp_block_table;
          paged_cmp_len = cmp_len - dense_tail_len;
          if (!current_compressed_rows.empty()) {
            cmp_seq = torch::cat(current_compressed_rows, /*dim=*/0);
          }
        } else {
          if (retain_compressed_history) {
            cmp_seq = compressor_kv_history_[state_seq_idx];
          }
          if (!cmp_seq.defined() && !effective_use_indexer_attention) {
            cmp_seq = gather_cache(cmp_cache,
                                   cmp_block_table,
                                   static_cast<int64_t>(seq_idx),
                                   cmp_len);
            maybe_sync_attention_stage(layer_id_, "gather_cmp_cache", cmp_seq);
          }
        }
      }
    }
    torch::Tensor cmp_indices;
    torch::Tensor cmp_physical_indices;
    torch::Tensor cmp_extra_physical_indices;
    const int64_t seq_q_len = q_end - q_start;
    const int64_t q_abs_start =
        use_current_prefill_kv
            ? (sp_context == nullptr
                   ? 0
                   : sp_context->global_row_start + sp_context->local_offset)
        : is_chunked_prefill && sp_context != nullptr
            ? std::max<int64_t>(
                  kv_lens[seq_idx] -
                      (pcp_context_ != nullptr ? pcp_context_->total_rows
                                               : sp_context->total_tokens) +
                      sp_context->global_row_start + sp_context->local_offset,
                  0)
            : std::max<int64_t>(kv_lens[seq_idx] - seq_q_len, 0);
    if (pcp_context_ != nullptr) {
      // CP only changes this rank's row ownership. The cached history still
      // excludes the entire global chunk, not merely this group's half.
      CHECK_GE(q_abs_start, 0);
      CHECK_LE(q_abs_start + seq_q_len, kv_lens[seq_idx]);
      LOG_FIRST_N(INFO, 16)
          << "[DSV4_PCP_QUERY_ROWS] cp_rank=" << pcp_context_->cp_rank
          << " kv_rows=" << kv_lens[seq_idx]
          << " global_chunk_rows=" << pcp_context_->total_rows
          << " q_abs_start=" << q_abs_start << " q_rows=" << seq_q_len;
    }
    torch::Tensor index_kv;
    const int64_t index_len = kv_lens[seq_idx] / 4;
    const bool use_fused_indexer_topk =
        use_paged_indexer_cache && dsv4_musa_fused_indexer_topk_enabled() &&
        (seq_q_len == 1 || compact_spec_verify);
    if (use_paged_indexer_cache && index_len > 0) {
      // A decode step on a C4 boundary produces a row consumed immediately.
      // The merged path publishes all sequences once after this loop. Its
      // Indexer and MATE consumers are both deferred until that publication.
      if (!batched_indexer_across_sequences) {
        flush_decode_compressed_cache();
      }
      if (!use_fused_indexer_topk) {
        index_kv = gather_indexer_cache(index_cache,
                                        index_scale_cache,
                                        index_block_table,
                                        static_cast<int64_t>(seq_idx),
                                        index_len,
                                        hidden_states.scalar_type());
        maybe_sync_attention_stage(layer_id_, "indexer_cache_gather", index_kv);
      }
    } else if (state_seq_idx < indexer_kv_history_.size()) {
      index_kv = indexer_kv_history_[state_seq_idx];
    }
    if (apply_sparse_indexer_attention && index_len > 0 &&
        (use_fused_indexer_topk ||
         (index_kv.defined() && index_kv.size(0) == index_len))) {
      auto qr_seq = qr.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
      auto hidden_seq =
          hidden_states.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
      auto index_head_weights =
          packed_indexer_head_weights.defined()
              ? packed_indexer_head_weights.slice(
                    /*dim=*/0, /*start=*/q_start, /*end=*/q_end)
              : torch::Tensor();
      auto index_query = packed_indexer_query.defined()
                             ? packed_indexer_query.slice(
                                   /*dim=*/0, /*start=*/q_start, /*end=*/q_end)
                             : torch::Tensor();
      auto index_cos =
          active_cos.defined()
              ? active_cos.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end)
              : torch::Tensor();
      auto index_sin =
          active_sin.defined()
              ? active_sin.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end)
              : torch::Tensor();
      if (batched_indexer_across_sequences) {
        cmp_indices = batched_indexer_indices.narrow(
            /*dim=*/0, /*start=*/q_start, /*length=*/seq_q_len);
        cmp_physical_indices = batched_indexer_physical_indices.narrow(
            /*dim=*/0, /*start=*/q_start, /*length=*/seq_q_len);
        cmp_extra_physical_indices =
            batched_indexer_extra_physical_indices.narrow(
                /*dim=*/0, /*start=*/q_start, /*length=*/seq_q_len);
      } else if (compact_spec_verify && seq_q_len > 1 &&
                 use_paged_indexer_cache &&
                 decode_spec_indices_output_.defined() &&
                 decode_spec_indices_output_.size(0) >= seq_q_len &&
                 decode_spec_mate_physical_indices_.defined() &&
                 decode_spec_mate_extra_physical_indices_.defined() &&
                 decode_spec_mate_physical_indices_.size(0) >= seq_q_len &&
                 decode_spec_mate_extra_physical_indices_.size(0) >=
                     seq_q_len) {
        auto spec_indices = decode_spec_indices_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/seq_q_len);
        auto spec_physical = decode_spec_mate_physical_indices_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/seq_q_len);
        auto spec_extra_physical =
            decode_spec_mate_extra_physical_indices_.narrow(
                /*dim=*/0, /*start=*/0, /*length=*/seq_q_len);
        const bool graph_dynamic_indexer =
            attn_metadata.is_acl_graph && compact_spec_verify &&
            graph_decode_kv_seq_lens.defined() && [] {
              const char* env = std::getenv("XLLM_DSV4_GRAPH_DYNAMIC_INDEXER");
              return env != nullptr && std::string(env) == "1";
            }();
        const int64_t index_cache_capacity =
            index_block_table.defined() && index_cache.defined()
                ? index_block_table.size(1) * index_cache.size(1)
                : index_len;
        const int64_t rounded_index_capacity =
            dsv4_graph_indexer_capacity(index_len, dsv4_sequence_view_active);
        int64_t completed_tokens = 0;
        if (graph_dynamic_indexer && seq_q_len > 2 &&
            dsv4_musa_batched_graph_indexer_enabled()) {
          auto selected = build_indexer_topk(
              qr_seq,
              hidden_seq,
              index_head_weights,
              index_query,
              index_cos,
              index_sin,
              index_kv,
              index_cache,
              index_scale_cache,
              index_block_table,
              static_cast<int64_t>(seq_idx),
              std::min(index_cache_capacity, rounded_index_capacity),
              q_abs_start,
              spec_indices,
              cmp_block_table,
              swa_block_table,
              kv_lens[seq_idx],
              mate_cmp_cache.size(1),
              &spec_physical,
              &spec_extra_physical,
              graph_decode_kv_seq_lens.narrow(
                  /*dim=*/0, /*start=*/q_start, /*length=*/seq_q_len));
          CHECK(selected.defined()) << "Batched Graph indexer did not execute";
          completed_tokens = seq_q_len;
          LOG_FIRST_N(INFO, 1)
              << "[DSV4_BATCHED_INDEXER] fused causal query rows=" << seq_q_len;
        }
        for (int64_t token_idx = completed_tokens; token_idx < seq_q_len;
             ++token_idx) {
          const int64_t token_index_len =
              graph_dynamic_indexer
                  ? std::min<int64_t>(
                        index_cache_capacity,
                        std::max<int64_t>(index_len, rounded_index_capacity))
                  : std::clamp<int64_t>(
                        (q_abs_start + token_idx + 1) / 4, 0, index_len);
          if (token_index_len <= 0) {
            break;
          }
          const torch::Tensor dynamic_token_kv_len =
              graph_dynamic_indexer
                  ? graph_decode_kv_seq_lens.narrow(
                        /*dim=*/0, /*start=*/q_start + token_idx, /*length=*/1)
                  : torch::Tensor();
          auto token_physical = spec_physical.narrow(
              /*dim=*/0, /*start=*/token_idx, /*length=*/1);
          auto token_extra_physical = spec_extra_physical.narrow(
              /*dim=*/0, /*start=*/token_idx, /*length=*/1);
          auto token_indices = build_indexer_topk(
              qr_seq.narrow(/*dim=*/0, /*start=*/token_idx, /*length=*/1),
              hidden_seq.narrow(
                  /*dim=*/0, /*start=*/token_idx, /*length=*/1),
              index_head_weights.defined()
                  ? index_head_weights.narrow(
                        /*dim=*/0, /*start=*/token_idx, /*length=*/1)
                  : torch::Tensor(),
              index_query.defined()
                  ? index_query.narrow(
                        /*dim=*/0, /*start=*/token_idx, /*length=*/1)
                  : torch::Tensor(),
              index_cos.defined()
                  ? index_cos.narrow(
                        /*dim=*/0, /*start=*/token_idx, /*length=*/1)
                  : torch::Tensor(),
              index_sin.defined()
                  ? index_sin.narrow(
                        /*dim=*/0, /*start=*/token_idx, /*length=*/1)
                  : torch::Tensor(),
              index_kv,
              index_cache,
              index_scale_cache,
              index_block_table,
              static_cast<int64_t>(seq_idx),
              token_index_len,
              q_abs_start + token_idx,
              spec_indices.narrow(
                  /*dim=*/0, /*start=*/token_idx, /*length=*/1),
              cmp_block_table,
              swa_block_table,
              q_abs_start + token_idx + 1,
              mate_cmp_cache.size(1),
              &token_physical,
              &token_extra_physical,
              dynamic_token_kv_len);
          if (!token_indices.defined()) {
            break;
          }
          ++completed_tokens;
        }
        if (completed_tokens == seq_q_len) {
          cmp_indices = spec_indices;
          cmp_physical_indices = spec_physical;
          cmp_extra_physical_indices = spec_extra_physical;
        }
      }
      if (!cmp_indices.defined()) {
        cmp_indices = build_indexer_topk(qr_seq,
                                         hidden_seq,
                                         index_head_weights,
                                         index_query,
                                         index_cos,
                                         index_sin,
                                         index_kv,
                                         index_cache,
                                         index_scale_cache,
                                         index_block_table,
                                         static_cast<int64_t>(seq_idx),
                                         index_len,
                                         q_abs_start);
      }
      maybe_sync_attention_stage(layer_id_, "indexer_topk", cmp_indices);
    }
    torch::Tensor seq_output;
    if (dspark_block_size_ > 0) {
      // All rows in a DSpark block see the same trailing context plus the
      // entire diffusion block, without a causal mask within that block.
      CHECK_EQ(compress_ratio_, 1);
      const int64_t visible_rows = window_size_ + q_seq.size(0);
      if (batched_sas_output.defined()) {
        CHECK(use_fused_paged_sas);
        seq_output = batched_sas_output.narrow(0, q_start, q_end - q_start);
      } else if (use_fused_paged_sas) {
        // Every diffusion-block row sees the SAME context+block, not a
        // per-row causal prefix. Avoid gathering the entire logical history
        // and building FP32 score/probability/intermediate tensors in eager.
        seq_output = kernel::musa::deepseek_v4_paged_attention_sas(
            q_seq.contiguous(),
            ori_kv,
            swa_block_table,
            static_cast<int64_t>(seq_idx),
            kv_lens[seq_idx],
            visible_rows,
            attn_sink_.to(q.device()).to(torch::kFloat32).contiguous(),
            softmax_scale_,
            dsv4_sequence_output_buffer(
                q_seq, decode_spec_attention_output_, query_ranges.size()));
      } else {
        CHECK(kv_seq.defined());
        const int64_t start =
            std::max<int64_t>(kv_seq.size(0) - visible_rows, 0);
        seq_output = selected_kv_attention_reference(
            q_seq, kv_seq.slice(/*dim=*/0, start), attn_sink_, softmax_scale_);
      }
    }
    if (effective_use_indexer_attention && !use_current_prefill_kv &&
        dsv4_musa_mate_fused_dsa_trace_enabled()) {
      const int64_t index_history_rows =
          index_kv.defined() ? index_kv.size(0) : -1;
      LOG(INFO) << "[DSV4_MATE_ROUTE] layer=" << layer_id_ << " seq=" << seq_idx
                << " decode_ready"
                << " cmp_rows="
                << (cmp_seq.defined() ? cmp_seq.size(0) : int64_t{-1})
                << " index_history_rows=" << index_history_rows
                << " indices_defined=" << cmp_indices.defined()
                << " mate_cmp_cache=" << mate_cmp_cache.defined()
                << " mate_swa_cache=" << mate_swa_cache.defined()
                << " cmp_block_table=" << cmp_block_table.defined()
                << " swa_block_table=" << swa_block_table.defined();
    }
    const bool allow_mate_sparse_attention =
        query_ranges.size() <= 1 ||
        dsv4_musa_mate_fused_dsa_multi_sequence_enabled();
    // MATE has no causal mask between rows of a compact speculative verify
    // block. Recompute each row independently with its effective KV length;
    // otherwise a later proposal can affect an earlier target token.
    const bool compact_verify_needs_causal_fallback =
        !is_prefill && q_seq.size(0) > 1 && query_ranges.size() == 1;
    const bool enable_graph_batched_mate_verify =
        dsv4_musa_graph_batched_mate_verify_enabled();
    const bool use_graph_batched_mate_verify =
        enable_graph_batched_mate_verify && attn_metadata.is_acl_graph &&
        compact_spec_verify && q_seq.size(0) > 2;
    if (batch_mate_across_sequences && compact_spec_verify &&
        q_seq.size(0) > 1 && cmp_indices.defined() &&
        cmp_physical_indices.defined() &&
        cmp_extra_physical_indices.defined() && mate_cmp_cache.defined() &&
        mate_swa_cache.defined()) {
      // Defer the fused MATE launch until every request has published its
      // causal physical rows. The per-request compressor/indexer/state work
      // above remains isolated; only the row-independent FlashMLA launch is
      // merged. Per-sequence Indexer scratch needs a snapshot; merged Indexer
      // rows are disjoint views written after this loop, so retain their
      // addresses instead of copying the as-yet-uninitialized output.
      pending_batched_mate.push_back(PendingBatchedMate{
          q_seq,
          batched_indexer_across_sequences ? cmp_indices
                                           : cmp_indices.contiguous().clone(),
          batched_indexer_across_sequences
              ? cmp_physical_indices
              : cmp_physical_indices.contiguous().clone(),
          batched_indexer_across_sequences
              ? cmp_extra_physical_indices
              : cmp_extra_physical_indices.contiguous().clone(),
          kv_lens[seq_idx],
          q_start});
      continue;
    }
    if (dsv4_musa_mate_sparse_decode_enabled() &&
        effective_use_indexer_attention &&
        (!compact_verify_needs_causal_fallback ||
         use_graph_batched_mate_verify) &&
        !use_current_prefill_kv && allow_mate_sparse_attention &&
        cmp_indices.defined() && mate_cmp_cache.defined() &&
        mate_swa_cache.defined() && cmp_block_table.defined() &&
        swa_block_table.defined()) {
      // TopK may select the compressed row produced at this decode step.
      // Commit pending rows before the packed cache is consumed by Mate.
      if (!batched_indexer_across_sequences) {
        flush_decode_compressed_cache();
      }
      if (compact_spec_verify && q_seq.size(0) > 1 &&
          !use_graph_batched_mate_verify &&
          decode_spec_attention_output_.defined() &&
          decode_spec_attention_output_.size(0) >= q_seq.size(0) &&
          cmp_physical_indices.defined() &&
          cmp_extra_physical_indices.defined()) {
        auto compact_output = decode_spec_attention_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/q_seq.size(0));
        bool all_tokens_completed = true;
        for (int64_t token_idx = 0; token_idx < q_seq.size(0); ++token_idx) {
          const int64_t token_kv_len = std::max<int64_t>(
              kv_lens[seq_idx] - (q_seq.size(0) - 1 - token_idx), 1);
          auto token_output = mate_sparse_attention(
              q_seq.narrow(/*dim=*/0, /*start=*/token_idx, /*length=*/1),
              mate_cmp_cache,
              cmp_indices.narrow(
                  /*dim=*/0, /*start=*/token_idx, /*length=*/1),
              cmp_block_table,
              mate_swa_cache,
              swa_block_table,
              static_cast<int64_t>(seq_idx),
              token_kv_len,
              compact_output.narrow(
                  /*dim=*/0, /*start=*/token_idx, /*length=*/1),
              cmp_physical_indices.narrow(
                  /*dim=*/0, /*start=*/token_idx, /*length=*/1),
              cmp_extra_physical_indices.narrow(
                  /*dim=*/0, /*start=*/token_idx, /*length=*/1));
          if (!token_output.defined()) {
            all_tokens_completed = false;
            break;
          }
        }
        if (all_tokens_completed) {
          seq_output = compact_output;
        }
      } else if (compact_spec_verify && q_seq.size(0) > 1 &&
                 use_graph_batched_mate_verify &&
                 graph_decode_kv_seq_lens.defined() &&
                 graph_decode_kv_seq_lens.size(0) >= q_end &&
                 cmp_physical_indices.defined() &&
                 cmp_extra_physical_indices.defined()) {
        // MATE's batched ABI accepts a per-row top-k and SWA length. This is
        // the causal equivalent of the legacy single-launch q6 path: every
        // row sees only its own prefix, while metadata preparation and the
        // fused FlashMLA/MATE execution happen once for the whole verify.
        auto token_kv_lens = graph_decode_kv_seq_lens.narrow(
            /*dim=*/0, /*start=*/q_start, /*length=*/q_seq.size(0));
        auto compact_output =
            decode_spec_attention_output_.defined() &&
                    decode_spec_attention_output_.size(0) >= q_seq.size(0)
                ? decode_spec_attention_output_.narrow(
                      /*dim=*/0, /*start=*/0, /*length=*/q_seq.size(0))
                : torch::Tensor();
        seq_output =
            mate_sparse_attention_batched(q_seq,
                                          mate_cmp_cache,
                                          cmp_indices,
                                          cmp_block_table,
                                          mate_swa_cache,
                                          swa_block_table,
                                          static_cast<int64_t>(seq_idx),
                                          kv_lens[seq_idx],
                                          token_kv_lens,
                                          compact_output,
                                          cmp_physical_indices,
                                          cmp_extra_physical_indices);
      } else {
        seq_output = mate_sparse_attention(q_seq,
                                           mate_cmp_cache,
                                           cmp_indices,
                                           cmp_block_table,
                                           mate_swa_cache,
                                           swa_block_table,
                                           static_cast<int64_t>(seq_idx),
                                           kv_lens[seq_idx]);
      }
      maybe_sync_attention_stage(
          layer_id_, "mate_sparse_attention", seq_output);
    }
    const bool verify_mate_dsa =
        dsv4_musa_mate_fused_dsa_verify_enabled() && seq_output.defined();
    if (verify_mate_dsa) {
      auto reference_kv = kv_seq;
      if (!reference_kv.defined()) {
        reference_kv = gather_cache(ori_kv,
                                    swa_block_table,
                                    static_cast<int64_t>(seq_idx),
                                    kv_lens[seq_idx]);
      }
      auto reference_cmp = cmp_seq;
      if ((!reference_cmp.defined() || reference_cmp.numel() == 0) &&
          cmp_cache.defined() && cmp_block_table.defined()) {
        const int64_t cmp_len = kv_lens[seq_idx] / compress_ratio_;
        reference_cmp = gather_cache(
            cmp_cache, cmp_block_table, static_cast<int64_t>(seq_idx), cmp_len);
      }
      torch::Tensor selected_reference_output;
      if (layer_id_ == 2 && reference_kv.defined() && reference_cmp.defined()) {
        auto flat_logical = cmp_indices.reshape({-1}).to(torch::kLong);
        auto valid_logical = flat_logical.ge(0).logical_and(
            flat_logical.lt(reference_cmp.size(0)));
        auto selected_logical =
            torch::masked_select(flat_logical, valid_logical).contiguous();
        auto packed_cmp =
            dequantize_mate_model1_logical_rows(mate_cmp_cache,
                                                cmp_indices,
                                                cmp_block_table,
                                                static_cast<int64_t>(seq_idx));
        auto reference_cmp_selected = reference_cmp.index_select(
            /*dim=*/0, selected_logical);

        const int64_t extra_len =
            std::min<int64_t>({window_size_, int64_t{128}, kv_lens[seq_idx]});
        auto extra_logical = torch::arange(kv_lens[seq_idx] - extra_len,
                                           kv_lens[seq_idx],
                                           torch::TensorOptions()
                                               .dtype(torch::kInt32)
                                               .device(q_seq.device()))
                                 .view({1, -1});
        auto packed_extra =
            dequantize_mate_model1_logical_rows(mate_swa_cache,
                                                extra_logical,
                                                swa_block_table,
                                                static_cast<int64_t>(seq_idx));
        auto reference_extra = reference_kv.slice(
            /*dim=*/0,
            /*start=*/reference_kv.size(0) - extra_len,
            /*end=*/reference_kv.size(0));

        if (packed_cmp.defined() &&
            packed_cmp.sizes() == reference_cmp_selected.sizes() &&
            packed_extra.defined() &&
            packed_extra.sizes() == reference_extra.sizes()) {
          auto cmp_diff = (packed_cmp.to(torch::kFloat32) -
                           reference_cmp_selected.to(torch::kFloat32))
                              .abs();
          auto extra_diff = (packed_extra.to(torch::kFloat32) -
                             reference_extra.to(torch::kFloat32))
                                .abs();
          LOG(INFO) << "[DSV4_MATE_DSA_CACHE_VERIFY] layer=" << layer_id_
                    << " cmp_rows=" << packed_cmp.size(0)
                    << " cmp_max_abs=" << cmp_diff.max().item<float>()
                    << " cmp_mean_abs=" << cmp_diff.mean().item<float>()
                    << " extra_rows=" << packed_extra.size(0)
                    << " extra_max_abs=" << extra_diff.max().item<float>()
                    << " extra_mean_abs=" << extra_diff.mean().item<float>()
                    << " packed_cmp_stats=" << tensor_stat_string(packed_cmp)
                    << " reference_cmp_stats="
                    << tensor_stat_string(reference_cmp_selected);
        }
        auto selected_reference_kv =
            torch::cat({reference_cmp_selected, reference_extra}, /*dim=*/0);
        selected_reference_output = selected_kv_attention_reference(
            q_seq, selected_reference_kv, attn_sink_, softmax_scale_);
        auto all_reference_kv =
            torch::cat({reference_cmp, reference_extra}, /*dim=*/0);
        auto all_reference_output = selected_kv_attention_reference(
            q_seq, all_reference_kv, attn_sink_, softmax_scale_);
        if (selected_reference_output.defined() &&
            all_reference_output.defined()) {
          auto selected_all_diff =
              (selected_reference_output.to(torch::kFloat32) -
               all_reference_output.to(torch::kFloat32))
                  .abs();
          LOG(INFO) << "[DSV4_MATE_DSA_INDEX_VERIFY] layer=" << layer_id_
                    << " indices=" << tensor_i64_prefix(cmp_indices)
                    << " selected_vs_all_max="
                    << selected_all_diff.max().item<float>()
                    << " selected_vs_all_mean="
                    << selected_all_diff.mean().item<float>();
        }
      }
      auto reference_output = dense_attention(q_seq,
                                              reference_kv,
                                              reference_cmp,
                                              cmp_indices,
                                              /*causal=*/false,
                                              q_abs_start,
                                              /*use_full_kv=*/false);
      if (reference_output.defined()) {
        if (selected_reference_output.defined() && layer_id_ == 2) {
          auto ref_diff = (reference_output.to(torch::kFloat32) -
                           selected_reference_output.to(torch::kFloat32))
                              .abs();
          LOG(INFO) << "[DSV4_MATE_DSA_REFERENCE_VERIFY] layer=" << layer_id_
                    << " max_abs=" << ref_diff.max().item<float>()
                    << " mean_abs=" << ref_diff.mean().item<float>();
        }
        auto diff = (seq_output.to(torch::kFloat32) -
                     reference_output.to(torch::kFloat32))
                        .abs();
        LOG(INFO) << "[DSV4_MATE_DSA_VERIFY] layer=" << layer_id_
                  << " seq=" << seq_idx << " kv_len=" << kv_lens[seq_idx]
                  << " max_abs=" << diff.max().item<float>()
                  << " mean_abs=" << diff.mean().item<float>();
        // Verification is non-invasive: return the established DSA fallback.
        seq_output = reference_output;
      }
    }
    if (use_paged_decode_attention &&
        (!seq_output.defined() || verify_mate_dsa)) {
      const bool direct_multi_sequence =
          dsv4_sequence_view_active && query_ranges.size() > 1 &&
          dsv4_musa_mate_fused_dsa_multi_sequence_enabled();
      const torch::Tensor paged_swa_table =
          direct_multi_sequence
              ? swa_block_table.narrow(/*dim=*/0,
                                       /*start=*/static_cast<int64_t>(seq_idx),
                                       /*length=*/1)
              : swa_block_table;
      const int64_t paged_sequence_index =
          direct_multi_sequence ? 0 : static_cast<int64_t>(seq_idx);
      torch::Tensor paged_output;
      const bool dynamic_compressed_verify =
          attn_metadata.is_acl_graph && compact_spec_verify &&
          q_seq.size(0) > 2 && window_size_ == 128 &&
          (query_ranges.size() == 1 || direct_multi_sequence) &&
          graph_decode_kv_seq_lens.defined() && cmp_cache.defined() &&
          cmp_block_table.defined() &&
          decode_spec_attention_output_.defined() && [] {
            const char* env = std::getenv("XLLM_DSV4_DYNAMIC_GRAPH_STATE");
            return env != nullptr && std::string(env) == "1";
          }();
      const bool dynamic_c128_verify =
          dynamic_compressed_verify && compress_ratio_ == 128;
      const torch::Tensor paged_compressed_table =
          dsv4_paged_compressed_table_view(cmp_block_table,
                                           paged_cmp_block_table,
                                           dynamic_compressed_verify,
                                           direct_multi_sequence,
                                           static_cast<int64_t>(seq_idx));
      const bool dynamic_short_c4_verify =
          dynamic_compressed_verify && compress_ratio_ == 4 &&
          force_full_kv_attention && [] {
            const char* env = std::getenv("XLLM_DSV4_GRAPH_CAUSAL_SHORT_C4");
            return env != nullptr && std::string(env) == "1";
          }();
      if (dynamic_c128_verify || dynamic_short_c4_verify) {
        // Publishing the whole proposal is safe only when each target row
        // masks compressed history by its own runtime KV length. A dense
        // tail shared by all q6 rows leaks the later compression boundary.
        if (!partitioned_c128_requests) {
          flush_decode_compressed_cache();
        }
        if (dynamic_c128_verify && dsv4_musa_c128_mate_enabled() &&
            kv_lens[seq_idx] / 128 <= 512 && mate_cmp_cache.defined() &&
            mate_swa_cache.defined() && decode_spec_indices_output_.defined() &&
            index_topk_ == 512) {
          const int64_t row_start = partitioned_c128_requests ? q_start : 0;
          auto logical =
              decode_spec_indices_output_.narrow(0, row_start, q_seq.size(0));
          auto physical = decode_spec_mate_physical_indices_.narrow(
              0, row_start, q_seq.size(0));
          auto extra = decode_spec_mate_extra_physical_indices_.narrow(
              0, row_start, q_seq.size(0));
          auto lengths =
              graph_decode_kv_seq_lens.narrow(0, q_start, q_seq.size(0));
          if (!batch_c128_indices) {
            kernel::musa::deepseek_v4_prepare_c128_mate_indices(
                lengths,
                paged_compressed_table,
                paged_swa_table,
                mate_cmp_cache.size(1),
                logical,
                physical,
                extra);
          }
          if (partitioned_c128_requests) {
            CHECK_EQ(q_seq.size(0), 7);
            // Views are disjoint for every request. Cache publication occurs
            // once after the loop,before the merged MATE reader is enqueued.
            pending_batched_mate.emplace_back(PendingBatchedMate{
                q_seq, logical, physical, extra, kv_lens[seq_idx], q_start});
            LOG_FIRST_N(INFO, 1) << "[DSV4_PARTITIONED_C128_MATE] sequences="
                                 << query_ranges.size() << " query_width=7";
            continue;
          }
          paged_output = mate_sparse_attention_batched(
              q_seq,
              mate_cmp_cache,
              logical,
              paged_compressed_table,
              mate_swa_cache,
              paged_swa_table,
              paged_sequence_index,
              kv_lens[seq_idx],
              lengths,
              decode_spec_attention_output_.narrow(0, 0, q_seq.size(0)),
              physical,
              extra);
          CHECK(paged_output.defined()) << "C128 MODEL1 path unavailable";
          LOG_FIRST_N(INFO, 1)
              << "[DSV4_C128_MATE] causal batched verify enabled";
        }
        if (!paged_output.defined()) {
          const int64_t capacity_bucket =
              (std::max<int64_t>(kv_lens[seq_idx] / 4, 1) + 511) / 512;
          // The old short-C4 scalar fallback captured a host compressed
          // length and an unmasked dense proposal tail. A shorter request
          // could then consume stale rows. Publish proposals above, but mask
          // them by each query's device KV length. The full-KV graph key
          // bounds this route to 128 tokens: 32 is capacity, not validity.
          const int64_t compressed_capacity = std::min<int64_t>(
              dynamic_short_c4_verify ? 32 : capacity_bucket * 16,
              cmp_block_table.size(1) * cmp_cache.size(1));
          paged_output =
              kernel::musa::deepseek_v4_paged_attention_verify_compressed(
                  q_seq,
                  ori_kv,
                  paged_swa_table,
                  graph_decode_kv_seq_lens.narrow(
                      /*dim=*/0, q_start, /*length=*/q_seq.size(0)),
                  cmp_cache,
                  paged_compressed_table,
                  compressed_capacity,
                  compress_ratio_,
                  attn_sink_.to(q.device()).to(torch::kFloat32).contiguous(),
                  softmax_scale_,
                  decode_spec_attention_output_.narrow(
                      /*dim=*/0, /*start=*/0, /*length=*/q_seq.size(0)));
          if (dynamic_short_c4_verify) {
            LOG_FIRST_N(INFO, 1) << "[DSV4_SHORT_C4_CAUSAL] per-query "
                                    "compressed lengths enabled";
          }
        }
      } else if (compact_spec_verify && q_seq.size(0) > 1 &&
                 decode_spec_attention_output_.defined() &&
                 decode_spec_attention_output_.size(0) >= q_seq.size(0) &&
                 decode_spec_attention_output_.size(1) == q_seq.size(1) &&
                 decode_spec_attention_output_.size(2) == q_seq.size(2)) {
        auto spec_output = decode_spec_attention_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/q_seq.size(0));
        bool all_tokens_completed = true;
        for (int64_t token_idx = 0; token_idx < q_seq.size(0); ++token_idx) {
          const int64_t token_kv_len = std::max<int64_t>(
              kv_lens[seq_idx] - (q_seq.size(0) - 1 - token_idx), 1);
          const int64_t dense_compressed_rows =
              cmp_seq.defined() ? cmp_seq.size(0) : 0;
          const int64_t token_compressed_len =
              compress_ratio_ > 1 ? token_kv_len / compress_ratio_ : 0;
          const int64_t token_paged_compressed_len = std::min<int64_t>(
              paged_cmp_len,
              std::max<int64_t>(token_compressed_len - dense_compressed_rows,
                                0));
          auto token_output = paged_dense_attention(
              q_seq.narrow(/*dim=*/0, /*start=*/token_idx, /*length=*/1),
              ori_kv,
              paged_swa_table,
              paged_sequence_index,
              token_kv_len,
              graph_decode_kv_seq_lens.defined()
                  ? graph_decode_kv_seq_lens.narrow(
                        /*dim=*/0,
                        /*start=*/q_start + token_idx,
                        /*length=*/1)
                  : torch::Tensor(),
              cmp_seq,
              paged_cmp_cache,
              paged_compressed_table,
              token_paged_compressed_len);
          if (!token_output.defined()) {
            all_tokens_completed = false;
            break;
          }
          spec_output.narrow(/*dim=*/0, /*start=*/token_idx, /*length=*/1)
              .copy_(token_output);
        }
        if (all_tokens_completed) {
          paged_output = spec_output;
        }
      } else {
        paged_output = paged_dense_attention(
            q_seq,
            ori_kv,
            paged_swa_table,
            paged_sequence_index,
            kv_lens[seq_idx],
            graph_decode_kv_seq_lens.defined()
                ? graph_decode_kv_seq_lens.narrow(
                      /*dim=*/0, /*start=*/q_start, /*length=*/1)
                : torch::Tensor(),
            cmp_seq,
            paged_cmp_cache,
            paged_compressed_table,
            paged_cmp_len);
      }
      maybe_sync_attention_stage(
          layer_id_, "paged_dense_attention", paged_output);
      if (!seq_output.defined()) {
        seq_output = paged_output;
      }
    }
    if (!seq_output.defined()) {
      if (use_paged_indexer_cache &&
          (!cmp_seq.defined() || cmp_seq.numel() == 0)) {
        const int64_t cmp_len = kv_lens[seq_idx] / compress_ratio_;
        cmp_seq = gather_cache(
            cmp_cache, cmp_block_table, static_cast<int64_t>(seq_idx), cmp_len);
        maybe_sync_attention_stage(
            layer_id_, "gather_cmp_cache_mate_fallback", cmp_seq);
      }
      if (paged_cmp_len > 0) {
        auto gathered_cmp = gather_cache(paged_cmp_cache,
                                         paged_cmp_block_table,
                                         static_cast<int64_t>(seq_idx),
                                         paged_cmp_len);
        cmp_seq = cmp_seq.defined() && cmp_seq.numel() > 0
                      ? torch::cat({gathered_cmp, cmp_seq}, /*dim=*/0)
                      : gathered_cmp;
        maybe_sync_attention_stage(
            layer_id_, "gather_cmp_cache_fallback", cmp_seq);
      }
      if (!kv_seq.defined()) {
        kv_seq = gather_cache(ori_kv,
                              swa_block_table,
                              static_cast<int64_t>(seq_idx),
                              kv_lens[seq_idx]);
        maybe_sync_attention_stage(layer_id_, "gather_swa_cache", kv_seq);
        if (!kv_seq.defined()) {
          kv_seq =
              kv.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end).squeeze(1);
        }
      }
      // Prefix-cache hits are represented as chunked prefill. Their query
      // rows are already present in the gathered KV cache, so retain causal
      // absolute-position masking and allow the validated local-prefill AOT
      // instead of falling through to the decode-style composite path.
      const bool causal_full_query = use_current_prefill_kv ||
                                     is_chunked_prefill ||
                                     compact_verify_needs_causal_fallback;
      seq_output = dense_attention(q_seq,
                                   kv_seq,
                                   cmp_seq,
                                   cmp_indices,
                                   causal_full_query,
                                   q_abs_start,
                                   causal_full_query);
    }
    maybe_sync_attention_stage(layer_id_, "seq_dense_attention", seq_output);
    if (trace_layer0) {
      LOG(INFO) << "[DSV4_MUSA_TRACE] layer0 seq=" << seq_idx << " q_range=["
                << q_start << "," << q_end << ")"
                << " kv_seq_shape=" << tensor_sizes_string(kv_seq)
                << " kv_seq_stats=" << tensor_stat_string(kv_seq)
                << " out_stats=" << tensor_stat_string(seq_output);
    }
    outputs.emplace_back(
        batched_sas_output.defined()
            ? seq_output
            : dsv4_retain_sequence_output(seq_output, query_ranges.size()));
  }
  flush_decode_compressed_cache();
  launch_batched_indexer();
  const bool batch_state_publish_views =
      batched_state_gather && batch_compression && [] {
        const char* value = std::getenv("XLLM_DSV4_BATCH_STATE_PUBLISH_VIEWS");
        return value != nullptr && std::string(value) == "1";
      }();
  auto persist_sequence_state = [&](torch::Tensor cache,
                                    const std::vector<torch::Tensor>& slots,
                                    const std::vector<torch::Tensor>& states,
                                    const char* stage) {
    if (!cache.defined()) {
      return;
    }
    std::vector<torch::Tensor> active_slots;
    std::vector<torch::Tensor> active_states;
    active_slots.reserve(query_ranges.size());
    active_states.reserve(query_ranges.size());
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      const size_t state_seq_idx = state_sequence_index_for(seq_idx);
      if (seq_idx < slots.size() && slots[seq_idx].defined() &&
          state_seq_idx < states.size() && states[state_seq_idx].defined()) {
        if (trace_layer0) {
          LOG(INFO) << "[DSV4_MUSA_STATE] phase=persist kind=" << stage
                    << " seq=" << seq_idx
                    << " slots=" << tensor_i64_prefix(slots[seq_idx])
                    << " state_seq=" << state_seq_idx
                    << " state=" << tensor_stat_string(states[state_seq_idx]);
        }
        active_slots.push_back(slots[seq_idx]);
        active_states.push_back(states[state_seq_idx]);
      }
    }
    if (active_slots.empty()) {
      return;
    }
    // Batched gather owns these buffers independently of the persistent cache;
    // compression updates them in place. Adjacent slices need no new storage.
    // Reordered or independent scratch rows retain cat/copy semantics.
    auto all_slots = batch_state_publish_views
                         ? dsv4_adjacent_row_view_or_cat(active_slots)
                         : torch::cat(active_slots, /*dim=*/0);
    auto all_states = batch_state_publish_views
                          ? dsv4_adjacent_row_view_or_cat(active_states)
                          : torch::cat(active_states, /*dim=*/0);
    if (batch_state_publish_views && query_ranges.size() == 16) {
      LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_STATE_PUBLISH_B16] stage=" << stage
                           << " slots_view="
                           << all_slots.is_alias_of(active_slots.front())
                           << " states_view="
                           << all_states.is_alias_of(active_states.front());
    }
    scatter_by_slot(cache, all_slots, all_states);
    maybe_sync_attention_stage(layer_id_, stage, all_states);
  };
  if (use_persistent_compressor_state) {
    persist_sequence_state(compressor_kv_cache,
                           compressor_kv_state_slots_by_seq,
                           compressor_kv_state_,
                           "compressor_kv_state_scatter");
    persist_sequence_state(compressor_score_cache,
                           compressor_score_state_slots_by_seq,
                           compressor_score_state_,
                           "compressor_score_state_scatter");
  }
  if (use_persistent_indexer_state) {
    persist_sequence_state(indexer_kv_state_cache,
                           indexer_kv_state_slots_by_seq,
                           indexer_kv_state_,
                           "indexer_kv_state_scatter");
    persist_sequence_state(indexer_score_state_cache,
                           indexer_score_state_slots_by_seq,
                           indexer_score_state_,
                           "indexer_score_state_scatter");
  }
  if (!pending_batched_mate.empty()) {
    // Diagnostic baseline: keep one request per MATE launch. If this restores
    // row identity, the remaining error is in batched MATE/Graph metadata
    // rather than compressor/indexer state publication.
    const size_t max_mate_sequences_per_launch =
        dsv4_musa_mate_max_sequences_per_launch();
    if (partitioned_c128_requests) {
      CHECK_EQ(pending_batched_mate.size(), query_ranges.size());
      CHECK_GE(max_mate_sequences_per_launch, pending_batched_mate.size())
          << "Partitioned C128 metadata sharing requires the whole physical "
             "batch";
      if (batch_c128_indices) {
        const int64_t sequences = static_cast<int64_t>(query_ranges.size());
        const int64_t rows = sequences * 7;
        CHECK_EQ(q.size(0), rows);
        for (int64_t seq = 0; seq < sequences; ++seq) {
          CHECK_EQ(pending_batched_mate[seq].q_start, seq * 7);
          CHECK_EQ(pending_batched_mate[seq].q.size(0), 7);
        }
        kernel::musa::deepseek_v4_prepare_c128_mate_indices_batched(
            graph_decode_kv_seq_lens.narrow(0, 0, rows),
            cmp_block_table.narrow(0, 0, sequences),
            swa_block_table.narrow(0, 0, sequences),
            mate_cmp_cache.size(1),
            7,
            decode_spec_indices_output_.narrow(0, 0, rows),
            decode_spec_mate_physical_indices_.narrow(0, 0, rows),
            decode_spec_mate_extra_physical_indices_.narrow(0, 0, rows));
        LOG_FIRST_N(INFO, 1)
            << "[DSV4_BATCH_C128_INDICES] sequences=" << sequences
            << " query_width=7 partition_order_unchanged=1";
      }
    }
    for (size_t group_start = 0; group_start < pending_batched_mate.size();
         group_start += max_mate_sequences_per_launch) {
      const size_t group_end =
          std::min(group_start + max_mate_sequences_per_launch,
                   pending_batched_mate.size());
      std::vector<torch::Tensor> pending_q;
      std::vector<torch::Tensor> pending_logical;
      std::vector<torch::Tensor> pending_physical;
      std::vector<torch::Tensor> pending_extra;
      std::vector<torch::Tensor> pending_lengths;
      pending_q.reserve(group_end - group_start);
      pending_logical.reserve(group_end - group_start);
      pending_physical.reserve(group_end - group_start);
      pending_extra.reserve(group_end - group_start);
      pending_lengths.reserve(group_end - group_start);
      int64_t max_pending_kv_len = 0;
      for (size_t pending_index = group_start; pending_index < group_end;
           ++pending_index) {
        const auto& pending = pending_batched_mate[pending_index];
        pending_q.push_back(pending.q);
        pending_logical.push_back(pending.logical_indices);
        pending_physical.push_back(pending.physical_indices);
        pending_extra.push_back(pending.extra_physical_indices);
        max_pending_kv_len = std::max(max_pending_kv_len, pending.kv_len);
        CHECK(graph_decode_kv_seq_lens.defined());
        pending_lengths.push_back(graph_decode_kv_seq_lens.narrow(
            /*dim=*/0,
            /*start=*/pending.q_start,
            /*length=*/pending.q.size(0)));
      }
      auto merge_rows = [&](const std::vector<torch::Tensor>& rows) {
        return batch_mate_metadata ? dsv4_adjacent_row_view_or_cat(rows)
                                   : torch::cat(rows, 0);
      };
      auto batched_q = merge_rows(pending_q);
      auto batched_logical = merge_rows(pending_logical);
      auto batched_physical = merge_rows(pending_physical);
      auto batched_extra = merge_rows(pending_extra);
      auto batched_lengths = merge_rows(pending_lengths).contiguous();
      auto batched_output = decode_spec_attention_output_.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/batched_q.size(0));
      auto batched_mate_output =
          mate_sparse_attention_batched(batched_q,
                                        mate_cmp_cache,
                                        batched_logical,
                                        cmp_block_table,
                                        mate_swa_cache,
                                        swa_block_table,
                                        /*sequence_index=*/0,
                                        max_pending_kv_len,
                                        batched_lengths,
                                        batched_output,
                                        batched_physical,
                                        batched_extra);
      CHECK(batched_mate_output.defined())
          << "Batched MATE launch failed after per-sequence state publication";
      // MATE writes every group into the same persistent decode scratch. A
      // view would be overwritten by the next group and reorder/contaminate
      // requests at the final cat; snapshot each group before continuing.
      const bool only_launch = batch_mate_metadata && group_start == 0 &&
                               group_end == pending_batched_mate.size() &&
                               outputs.empty();
      outputs.push_back(only_launch ? batched_mate_output
                                    : batched_mate_output.clone());
    }
    LOG_FIRST_N(INFO, 1) << "[DSV4_BATCH_MATE_ACROSS_SEQUENCES] sequences="
                         << pending_batched_mate.size()
                         << " group_size=" << max_mate_sequences_per_launch;
  }
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " dense_done seqs=" << query_ranges.size();
  }

  torch::Tensor attn_output;
  if (batched_sas_output.defined()) {
    CHECK_EQ(outputs.size(), query_ranges.size());
    CHECK(pending_batched_mate.empty());
    // The full-batch output has private storage: no reused scalar scratch,
    // per-sequence clone, or concatenation is required.
    attn_output = batched_sas_output;
  } else if (outputs.empty()) {
    if (decode_spec_attention_output_.defined() &&
        decode_spec_attention_output_.size(0) >= q.size(0) &&
        decode_spec_attention_output_.size(1) == q.size(1) &&
        decode_spec_attention_output_.size(2) == q.size(2) &&
        decode_spec_attention_output_.scalar_type() == q.scalar_type() &&
        decode_spec_attention_output_.device() == q.device()) {
      attn_output = decode_spec_attention_output_.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/q.size(0));
      attn_output.zero_();
    } else {
      attn_output = torch::zeros_like(q);
    }
  } else if (outputs.size() == 1) {
    attn_output = outputs.front().is_contiguous()
                      ? outputs.front()
                      : outputs.front().contiguous();
  } else {
    attn_output = torch::cat(outputs, /*dim=*/0);
  }
  maybe_sync_attention_stage(layer_id_, "outputs_cat", attn_output);
  record_prefill_profile_boundary(prefill_profile, 3, profile_device_id);

  torch::Tensor inverse_rope_workspace;
  if (decode_inverse_rope_output_.defined() &&
      decode_inverse_rope_output_.dim() == attn_output.dim() &&
      decode_inverse_rope_output_.size(0) >= attn_output.size(0) &&
      decode_inverse_rope_output_.size(1) == attn_output.size(1) &&
      decode_inverse_rope_output_.size(2) == attn_output.size(2) &&
      decode_inverse_rope_output_.scalar_type() == attn_output.scalar_type() &&
      decode_inverse_rope_output_.device() == attn_output.device()) {
    inverse_rope_workspace = decode_inverse_rope_output_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/attn_output.size(0));
  }
  auto o = apply_interleaved_partial_rope(
      attn_output,
      nope_head_dim_,
      rope_head_dim_,
      active_cos,
      active_sin,
      /*inverse=*/true,
      inverse_rope_workspace,
      /*allow_prefill_assembly=*/false,
      /*allow_main_assembly=*/is_prefill || is_chunked_prefill);
  maybe_sync_attention_stage(layer_id_, "output_inverse_rope", o);
  const int64_t num_tokens = o.size(0);
  if (dsv4_defer_o_a_projection) {
    CHECK(dsv4_defer_output_projection && dsv4_sequence_view_active &&
          attn_metadata.is_acl_graph && !is_prefill && !is_chunked_prefill);
    // State persistence above has completed. The caller snapshots this
    // inverse-RoPE output before the next sequence reuses the same buffer.
    return {o.reshape({num_tokens, -1}), std::nullopt};
  }
  auto o_group = o.view({num_tokens, n_local_groups_, -1});
  CHECK(o_a_projection_weight_.defined())
      << "DeepSeek-V4 MUSA wo_a weight was not materialized during loading.";
  torch::Tensor o_low_rank;
  bool used_wo_a_m1 = false;
  if (n_local_groups_ == 1) {
    // TP8 owns one output group per rank. The latest MUSA tree already builds
    // the worker2 M=1 split-K kernel; use it for q1 instead of paying the
    // generic GEMM setup cost in every attention layer. Unsupported shapes
    // retain the existing graph-safe mm_out/einsum path.
    if (num_tokens == 1) {
      std::optional<torch::Tensor> wo_a_output = std::nullopt;
      if (decode_wo_a_output_.defined() &&
          decode_wo_a_output_.size(0) >= num_tokens) {
        wo_a_output = decode_wo_a_output_.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/num_tokens);
      }
      o_low_rank =
          kernel::musa::deepseek_v4_wo_a_m1(o_group,
                                            o_a_projection_weight_,
                                            /*head_scales=*/torch::Tensor(),
                                            wo_a_output);
      used_wo_a_m1 = o_low_rank.defined();
    }
    if (!o_low_rank.defined() && decode_wo_a_output_.defined() &&
        num_tokens <= decode_wo_a_output_.size(0)) {
      auto wo_a_output = decode_wo_a_output_
                             .narrow(/*dim=*/0,
                                     /*start=*/0,
                                     /*length=*/num_tokens)
                             .view({num_tokens, o_lora_rank_});
      at::mm_out(wo_a_output,
                 o_group.select(/*dim=*/1, /*index=*/0),
                 o_a_projection_weight_.select(/*dim=*/0, /*index=*/0)
                     .transpose(/*dim0=*/0, /*dim1=*/1));
      o_low_rank = wo_a_output.view({num_tokens, 1, o_lora_rank_});
    }
    if (used_wo_a_m1) {
      LOG_FIRST_N(INFO, 1)
          << "DeepSeek-V4 MUSA TP8 M=1 wo_a split-K kernel enabled.";
    }
  }
  if (!o_low_rank.defined()) {
    o_low_rank =
        torch::einsum("tgd,grd->tgr", {o_group, o_a_projection_weight_});
  } else {
    CHECK_EQ(o_low_rank.dim(), 3);
  }
  maybe_sync_attention_stage(layer_id_, "output_o_a_einsum", o_low_rank);
  record_prefill_profile_boundary(prefill_profile, 4, profile_device_id);
  if (dsv4_defer_output_projection) {
    CHECK(dsv4_sequence_view_active && attn_metadata.is_acl_graph &&
          !is_prefill && !is_chunked_prefill);
    // The outer view loop must clone this scratch-backed result before the
    // next sequence uses it. No collective is issued on the per-row path.
    return {o_low_rank.reshape({num_tokens, -1}), std::nullopt};
  }
  const char* prefill_fence_env =
      std::getenv("XLLM_DSV4_MULTI_PREFILL_COLLECTIVE_FENCE");
  const bool fence_prefill_collective =
      prefill_fence_env != nullptr && std::string(prefill_fence_env) == "1" &&
      (is_prefill || is_chunked_prefill) && sp_context == nullptr &&
      num_tokens >= 128 && !kernel::musa::is_stream_capturing();
  auto reduce_o_b_output =
      [this, fence_prefill_collective](torch::Tensor local_output) {
        ProcessGroup* process_group = o_b_proj_->process_group();
        if (process_group == nullptr || process_group->world_size() <= 1) {
          return local_output;
        }

        std::optional<torch::Tensor> output_workspace;
        if (decode_o_b_reduce_output_.defined() && local_output.dim() == 2 &&
            local_output.size(0) <= decode_o_b_reduce_output_.size(0) &&
            local_output.size(1) == decode_o_b_reduce_output_.size(1) &&
            local_output.scalar_type() ==
                decode_o_b_reduce_output_.scalar_type() &&
            local_output.device() == decode_o_b_reduce_output_.device()) {
          output_workspace = decode_o_b_reduce_output_.narrow(
              /*dim=*/0, /*start=*/0, /*length=*/local_output.size(0));
        }
        if (std::optional<torch::Tensor> custom_output = musa_allreduce_custom(
                process_group, local_output, output_workspace)) {
          LOG_FIRST_N(INFO, 1)
              << "DeepSeek-V4 MUSA attention out-of-place custom all-reduce "
                 "enabled.";
          return *custom_output;
        }

        if (fence_prefill_collective) {
          c10::musa::getCurrentMUSAStream(local_output.device().index())
              .synchronize();
        }
        process_group->allreduce(local_output);
        if (fence_prefill_collective) {
          c10::musa::getCurrentMUSAStream(local_output.device().index())
              .synchronize();
        }
        return local_output;
      };
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " before_o_b shape=" << o_low_rank.sizes();
  }
  torch::Tensor output;
  if (dsv4_musa_split_ob_proj_enabled()) {
    auto o_b_input = o_low_rank.reshape({num_tokens, -1});
    const int64_t reduce_delay_us = dsv4_musa_attention_reduce_delay_us();
    const int64_t output_chunk_tokens = dsv4_musa_output_proj_chunk_tokens();
    // MCCL uses a communication stream. On long C128 prefills a faster rank
    // can otherwise enqueue output-projection collectives while another rank
    // still has dense attention work pending. Fence local prefill work once
    // before the first chunked collective; decode remains asynchronous.
    maybe_sync_output_proj_prefill(
        o_low_rank, layer_id_, output_chunk_tokens, o_b_proj_->process_group());
    if (output_chunk_tokens > 0 && num_tokens > output_chunk_tokens) {
      std::vector<torch::Tensor> output_chunks;
      output_chunks.reserve(static_cast<size_t>(
          (num_tokens + output_chunk_tokens - 1) / output_chunk_tokens));
      for (int64_t start = 0; start < num_tokens;
           start += output_chunk_tokens) {
        const int64_t end =
            std::min<int64_t>(start + output_chunk_tokens, num_tokens);
        auto input_chunk =
            o_b_input.slice(/*dim=*/0, /*start=*/start, /*end=*/end);
        torch::Tensor output_chunk = o_b_proj_->forward(input_chunk);
        output_chunk = reduce_o_b_output(std::move(output_chunk));
        maybe_sync_attention_stage(
            layer_id_, "output_o_b_local_matmul", output_chunk);
        if (layer_id_ == 0 && output_chunk.size(0) > 1 && reduce_delay_us > 0) {
          std::this_thread::sleep_for(
              std::chrono::microseconds(reduce_delay_us));
        }
        output_chunks.push_back(output_chunk);
      }
      output = torch::cat(output_chunks, /*dim=*/0).contiguous();
      maybe_sync_attention_stage(layer_id_, "output_o_b_chunks_cat", output);
    } else {
      output = o_b_proj_->forward(o_b_input);
      output = reduce_o_b_output(std::move(output));
      maybe_sync_attention_stage(layer_id_, "output_o_b_local_matmul", output);
      if (layer_id_ == 0 && num_tokens > 1 && reduce_delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(reduce_delay_us));
      }
    }
  } else {
    output = o_b_proj_->forward(o_low_rank.reshape({num_tokens, -1}));
    output = reduce_o_b_output(std::move(output));
  }
  maybe_sync_attention_stage(layer_id_, "output_o_b_proj", output);
  if (zero_local_sp_shard) {
    // The dummy query row is only to keep all ranks on the same MUSA/SP
    // collective schedule. It must not escape the empty local shard.
    output = output.narrow(/*dim=*/0, /*start=*/0, /*length=*/0);
  }
  if (sp_context != nullptr) {
    dump_local_prefill_stage(layer_id_, "attention", output);
  }
  finish_prefill_profile(prefill_profile,
                         layer_id_,
                         profile_device_id,
                         hidden_states.size(0),
                         compress_ratio_);
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " end shape=" << output.sizes();
  }
  return std::make_tuple(output, std::nullopt);
}

void DeepseekV4AttentionImpl::load_state_dict(const StateDict& state_dict) {
  q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_a."));
  q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  q_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
  kv_proj_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  kv_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("kv_norm."));

  // Load C4 indexer projections before packing the shared decode GEMMs below.
  // The original modules remain available as the large-prefill fallback.
  if (indexer_wq_b_) {
    auto indexer_state = state_dict.get_dict_with_prefix("indexer.");
    indexer_wq_b_->load_state_dict(indexer_state.get_dict_with_prefix("wq_b."));
    indexer_weights_proj_->load_state_dict(
        indexer_state.get_dict_with_prefix("weights_proj."));
    indexer_compressor_wkv_->load_state_dict(
        indexer_state.get_dict_with_prefix("compressor.wkv."));
    indexer_compressor_wgate_->load_state_dict(
        indexer_state.get_dict_with_prefix("compressor.wgate."));
    indexer_compressor_norm_->load_state_dict(
        indexer_state.get_dict_with_prefix("compressor.norm."));
    auto indexer_ape = indexer_state.get_tensor("compressor.ape");
    if (indexer_ape.defined()) {
      CHECK(indexer_ape.dim() == 2 && indexer_ape.size(0) == 4 &&
            indexer_ape.size(1) == 2 * index_head_dim_)
          << "DeepSeek-V4 MUSA indexer compressor.ape shape mismatch, expected "
             "[4,"
          << 2 * index_head_dim_ << "], got " << indexer_ape.sizes();
      torch::NoGradGuard no_grad;
      indexer_compressor_ape_.copy_(
          indexer_ape.to(indexer_compressor_ape_.device())
              .to(indexer_compressor_ape_.dtype()));
      indexer_compressor_ape_loaded_ = true;
    }
  }

  // Materialize and pack the two replicated FP8 projections once. Decode
  // issues one wide GEMM, while prefill slices the same storage back to the
  // original projection shapes.
  const auto q_a_weight = q_a_proj_->weight();
  const auto kv_weight = kv_proj_->weight();
  const bool compatible_qkv_a_weights =
      q_a_weight.defined() && kv_weight.defined() && q_a_weight.dim() == 2 &&
      kv_weight.dim() == 2 && q_a_weight.size(0) == q_lora_rank_ &&
      kv_weight.size(0) == head_dim_ && q_a_weight.size(1) == hidden_dim_ &&
      kv_weight.size(1) == hidden_dim_ &&
      (is_fp8_weight(q_a_weight) ||
       q_a_weight.scalar_type() == torch::kBFloat16 ||
       q_a_weight.scalar_type() == torch::kFloat16) &&
      q_a_weight.scalar_type() == kv_weight.scalar_type() &&
      q_a_weight.device() == kv_weight.device();
  if (compatible_qkv_a_weights) {
    const auto output_dtype = decode_q_a_norm_output_.scalar_type();
    auto q_a_state = state_dict.get_dict_with_prefix("wq_a.");
    auto kv_state = state_dict.get_dict_with_prefix("wkv.");
    auto q_a_scale = q_a_state.get_tensor("weight_scale");
    if (!q_a_scale.defined()) {
      q_a_scale = q_a_state.get_tensor("scale");
    }
    auto kv_scale = kv_state.get_tensor("weight_scale");
    if (!kv_scale.defined()) {
      kv_scale = kv_state.get_tensor("scale");
    }
    auto q_a_materialized =
        dequantize_fp8_projection_weight(q_a_weight, q_a_scale, output_dtype);
    auto kv_materialized =
        dequantize_fp8_projection_weight(kv_weight, kv_scale, output_dtype);
    if (q_a_materialized.defined() && kv_materialized.defined() &&
        q_a_materialized.scalar_type() == kv_materialized.scalar_type() &&
        q_a_materialized.device() == kv_materialized.device()) {
      std::vector<torch::Tensor> packed_weights{q_a_materialized,
                                                kv_materialized};
      if (indexer_weights_proj_) {
        const auto indexer_weight = indexer_weights_proj_->weight();
        qkv_a_indexer_weight_loaded_ =
            indexer_weight.defined() && indexer_weight.dim() == 2 &&
            indexer_weight.size(0) == index_n_heads_ &&
            indexer_weight.size(1) == hidden_dim_ &&
            indexer_weight.scalar_type() == q_a_materialized.scalar_type() &&
            indexer_weight.device() == q_a_materialized.device();
        if (qkv_a_indexer_weight_loaded_) {
          packed_weights.push_back(indexer_weight);
        }
      }
      torch::NoGradGuard no_grad;
      qkv_a_weight_ =
          register_parameter("qkv_a_weight",
                             torch::cat(packed_weights, /*dim=*/0).contiguous(),
                             /*requires_grad=*/false);
      qkv_a_weight_loaded_ = true;
      decode_qkv_a_output_ =
          torch::empty({decode_workspace_tokens_, qkv_a_weight_.size(0)},
                       qkv_a_weight_.options());
      decode_q_down_ = torch::empty({decode_workspace_tokens_, q_lora_rank_},
                                    qkv_a_weight_.options());
      decode_kv_down_ = torch::empty({decode_workspace_tokens_, head_dim_},
                                     qkv_a_weight_.options());
      unregister_module("q_a_proj");
      unregister_module("kv_proj");
      q_a_proj_ = nullptr;
      kv_proj_ = nullptr;
      if (layer_id_ == 0) {
        LOG(INFO) << "DeepSeek-V4 MUSA packed wq_a+wkv decode projection "
                     "enabled; prefill uses row slices.";
      }
    }
  }

  if (indexer_wq_b_) {
    const auto output_dtype = decode_q_a_norm_output_.scalar_type();
    const auto main_q_b_weight = q_b_proj_->weight();
    const auto indexer_q_b_weight = indexer_wq_b_->weight();
    auto indexer_wq_b_state = state_dict.get_dict_with_prefix("indexer.wq_b.");
    auto indexer_q_b_scale = indexer_wq_b_state.get_tensor("weight_scale");
    if (!indexer_q_b_scale.defined()) {
      indexer_q_b_scale = indexer_wq_b_state.get_tensor("scale");
    }
    auto main_q_b_materialized = dequantize_fp8_projection_weight(
        main_q_b_weight, q_b_proj_->weight_scale(), output_dtype);
    auto indexer_q_b_materialized = dequantize_fp8_projection_weight(
        indexer_q_b_weight, indexer_q_b_scale, output_dtype);
    const int64_t main_q_width = n_local_heads_ * head_dim_;
    const int64_t indexer_q_width = index_n_heads_ * index_head_dim_;
    const bool compatible_q_indexer_b_weights =
        main_q_b_materialized.defined() && indexer_q_b_materialized.defined() &&
        main_q_b_materialized.dim() == 2 &&
        indexer_q_b_materialized.dim() == 2 &&
        main_q_b_materialized.size(0) == main_q_width &&
        indexer_q_b_materialized.size(0) == indexer_q_width &&
        main_q_b_materialized.size(1) == q_lora_rank_ &&
        indexer_q_b_materialized.size(1) == q_lora_rank_ &&
        main_q_b_materialized.scalar_type() ==
            indexer_q_b_materialized.scalar_type() &&
        main_q_b_materialized.device() == indexer_q_b_materialized.device();
    if (compatible_q_indexer_b_weights) {
      torch::NoGradGuard no_grad;
      q_indexer_b_weight_ = register_parameter(
          "q_indexer_b_weight",
          torch::cat({main_q_b_materialized, indexer_q_b_materialized},
                     /*dim=*/0)
              .contiguous(),
          /*requires_grad=*/false);
      q_indexer_b_weight_loaded_ = true;
      decode_q_indexer_b_output_ =
          torch::empty({decode_workspace_tokens_, q_indexer_b_weight_.size(0)},
                       q_indexer_b_weight_.options());
      decode_q_main_output_ =
          torch::empty({decode_workspace_tokens_, main_q_width},
                       q_indexer_b_weight_.options());
      unregister_module("q_b_proj");
      unregister_module("indexer_wq_b");
      q_b_proj_ = nullptr;
      indexer_wq_b_ = nullptr;
    }
  }
  o_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
  o_a_projection_weight_ =
      dequantize_fp8_projection_weight(o_a_proj_->weight(),
                                       o_a_proj_->weight_scale(),
                                       decode_o_b_reduce_output_.scalar_type())
          .view({n_local_groups_, o_lora_rank_, -1})
          .contiguous();
  o_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));

  auto attn_sink = state_dict.get_tensor("attn_sink");
  if (!attn_sink.defined()) {
    attn_sink = state_dict.get_tensor("attn_sink.weight");
  }
  if (attn_sink.defined()) {
    if (attn_sink.dim() == 1 && attn_sink.size(0) == num_heads_ &&
        tp_size_ > 1) {
      CHECK_EQ(num_heads_ % tp_size_, 0)
          << "attn_sink full-head tensor size is not divisible by tp_size.";
      const int64_t shard_size = num_heads_ / tp_size_;
      const int64_t shard_start = tp_rank_ * shard_size;
      attn_sink = attn_sink.slice(/*dim=*/0,
                                  /*start=*/shard_start,
                                  /*end=*/shard_start + shard_size);
    }
    CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == n_local_heads_)
        << "attn_sink shape mismatch, expected [" << n_local_heads_ << "], got "
        << attn_sink.sizes();

    torch::NoGradGuard no_grad;
    attn_sink_.copy_(attn_sink.to(attn_sink_.device()).to(attn_sink_.dtype()));
    attn_sink_loaded_ = true;
  }

  if (compressor_wkv_) {
    compressor_wkv_->load_state_dict(
        state_dict.get_dict_with_prefix("compressor.wkv."));
  }
  if (compressor_wgate_) {
    compressor_wgate_->load_state_dict(
        state_dict.get_dict_with_prefix("compressor.wgate."));
  }
  if (compressor_norm_) {
    compressor_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("compressor.norm."));
  }
  auto compressor_ape = state_dict.get_tensor("compressor.ape");
  if (compressor_ape.defined() && compressor_ape_.defined()) {
    CHECK(compressor_ape.dim() == 2 &&
          compressor_ape.size(0) == compress_ratio_ &&
          compressor_ape.size(1) == compressor_coff_ * head_dim_)
        << "DeepSeek-V4 MUSA compressor.ape shape mismatch, expected ["
        << compress_ratio_ << "," << compressor_coff_ * head_dim_ << "], got "
        << compressor_ape.sizes();
    torch::NoGradGuard no_grad;
    compressor_ape_.copy_(compressor_ape.to(compressor_ape_.device())
                              .to(compressor_ape_.dtype()));
    compressor_ape_loaded_ = true;
  }

  if (compressor_wkv_ && compressor_wgate_ && indexer_compressor_wkv_ &&
      indexer_compressor_wgate_) {
    const auto main_kv_weight = compressor_wkv_->weight();
    const auto main_score_weight = compressor_wgate_->weight();
    const auto indexer_kv_weight = indexer_compressor_wkv_->weight();
    const auto indexer_score_weight = indexer_compressor_wgate_->weight();
    const bool compatible =
        compressor_wkv_->is_weight_loaded() &&
        compressor_wgate_->is_weight_loaded() &&
        indexer_compressor_wkv_->is_weight_loaded() &&
        indexer_compressor_wgate_->is_weight_loaded() &&
        main_kv_weight.dim() == 2 && main_score_weight.dim() == 2 &&
        indexer_kv_weight.dim() == 2 && indexer_score_weight.dim() == 2 &&
        main_kv_weight.size(1) == hidden_dim_ &&
        main_score_weight.size(1) == hidden_dim_ &&
        indexer_kv_weight.size(1) == hidden_dim_ &&
        indexer_score_weight.size(1) == hidden_dim_ &&
        main_kv_weight.scalar_type() == main_score_weight.scalar_type() &&
        main_kv_weight.scalar_type() == indexer_kv_weight.scalar_type() &&
        main_kv_weight.scalar_type() == indexer_score_weight.scalar_type() &&
        main_kv_weight.device() == main_score_weight.device() &&
        main_kv_weight.device() == indexer_kv_weight.device() &&
        main_kv_weight.device() == indexer_score_weight.device();
    if (compatible) {
      torch::NoGradGuard no_grad;
      sp_joint_compressor_projection_weight_ =
          register_parameter("sp_joint_compressor_projection_weight",
                             torch::cat({main_kv_weight,
                                         main_score_weight,
                                         indexer_kv_weight,
                                         indexer_score_weight},
                                        /*dim=*/0)
                                 .contiguous(),
                             /*requires_grad=*/false);
    }
  }
}

}  // namespace layer
}  // namespace xllm
