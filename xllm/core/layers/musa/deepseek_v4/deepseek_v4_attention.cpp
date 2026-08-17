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

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/musa/musa_dsv4_compat.h"

namespace xllm {
namespace layer {
namespace {

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
  auto scale_fp32 = scale.to(torch::kFloat32);
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

bool dsv4_musa_trace_enabled() { return false; }
bool dsv4_musa_cache_trace_enabled() { return false; }
bool dsv4_musa_mate_fused_dsa_trace_enabled() { return false; }
bool dsv4_musa_mate_fused_dsa_verify_enabled() { return false; }
bool dsv4_musa_mate_prefill_trace_enabled() { return false; }
bool dsv4_musa_sync_trace_enabled() { return false; }

struct PrefillProfileState {};
bool dsv4_musa_prefill_profile_enabled() { return false; }
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
bool dsv4_musa_mate_fused_dsa_multi_sequence_enabled() { return false; }
bool dsv4_musa_fused_dense_decode_enabled() { return true; }
bool dsv4_musa_fused_cache_gather_enabled() { return true; }
bool dsv4_musa_fused_cache_scatter_enabled() { return false; }
bool dsv4_musa_paged_dense_decode_enabled() { return false; }
bool dsv4_musa_paged_compressed_decode_enabled() { return true; }
bool dsv4_musa_split_ob_proj_enabled() { return true; }
bool dsv4_musa_output_proj_prefill_sync_enabled() { return false; }
bool dsv4_musa_output_proj_host_barrier_enabled() { return false; }

int64_t dsv4_musa_paged_dense_decode_max_total_len() { return 0; }
int64_t dsv4_musa_attention_reduce_delay_us() { return 0; }
int64_t dsv4_musa_attention_sync_delay_us() { return 0; }
int64_t dsv4_musa_output_proj_chunk_tokens() { return 4096; }
int64_t dsv4_musa_prefill_full_kv_threshold() { return 2048; }
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
void maybe_sync_attention_stage(Args&&...) {}

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
  return i64_prefix_to_string(tensor_to_i64_vector(tensor), limit);
}

std::string tensor_stat_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  if (tensor.numel() == 0) {
    return "empty";
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

torch::Tensor apply_interleaved_partial_rope(torch::Tensor input,
                                             int64_t rope_start_dim,
                                             int64_t rope_head_dim,
                                             const torch::Tensor& cos,
                                             const torch::Tensor& sin,
                                             bool inverse = false) {
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
  if (!tensor.defined() || quant_dim <= 0 || block_size <= 0) {
    return;
  }
  CHECK_EQ(quant_dim % block_size, 0)
      << "DeepSeek-V4 MUSA KV act_quant requires quant_dim divisible by "
      << block_size << ", got " << quant_dim;

  auto target = tensor.slice(/*dim=*/-1, /*start=*/0, /*end=*/quant_dim);
  auto fp32 = target.to(torch::kFloat32)
                  .contiguous()
                  .view({-1, quant_dim / block_size, block_size});
  auto amax = std::get<0>(fp32.abs().max(/*dim=*/-1, /*keepdim=*/true));
  amax = torch::clamp_min(amax, 1e-4);
  auto scale = torch::pow(torch::full_like(amax, 2.0),
                          torch::ceil(torch::log2(amax * (1.0 / 448.0))));
  auto quantized = torch::clamp(fp32 / scale, -448.0, 448.0)
                       .to(torch::kFloat8_e4m3fn)
                       .to(torch::kFloat32);
  auto dequantized = (quantized * scale).view_as(target).to(target.dtype());
  target.copy_(dequantized);
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
                                                /*reduce=*/true,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));
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

std::vector<std::pair<int64_t, int64_t>>
DeepseekV4AttentionImpl::build_query_ranges(const DSAMetadata& attn_metadata,
                                            int64_t total_tokens) const {
  std::vector<std::pair<int64_t, int64_t>> ranges;
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
  std::vector<int64_t> kv_lens =
      tensor_to_i64_vector(attn_metadata.actual_seq_lengths_kv);
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
    int64_t kv_len) const {
  if (!q.defined() || q.size(0) != 1 || !packed_cache.defined() ||
      !extra_packed_cache.defined() || !logical_indices.defined() ||
      !compressed_block_table.defined() || !swa_block_table.defined()) {
    return torch::Tensor();
  }
  const int64_t effective_kv_len = std::max<int64_t>(kv_len, 0);
  const int64_t extra_len =
      std::min<int64_t>({window_size_, int64_t{128}, effective_kv_len});
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
  index_params.extra_physical_indices = index_workspace.extra_physical_indices;
  auto [physical_indices, extra_physical_indices] =
      ::xllm::kernel::deepseek_v4_prepare_mate_dsa_indices(index_params);

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
  auto output =
      ::xllm::kernel::deepseek_v4_mate_sparse_attention_decode(params);
  if (output.defined() && dsv4_musa_mate_fused_dsa_verify_enabled()) {
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
        auto diff =
            (output.to(torch::kFloat32) - packed_reference.to(torch::kFloat32))
                .abs();
        LOG(INFO) << "[DSV4_MATE_DSA_PACKED_VERIFY] layer=" << layer_id_
                  << " main_rows=" << main_kv.size(0)
                  << " extra_rows=" << extra_kv.size(0)
                  << " max_abs=" << diff.max().item<float>()
                  << " mean_abs=" << diff.mean().item<float>();
      }
    }
  }
  return output.defined() ? output.contiguous() : torch::Tensor();
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
  if (dsv4_musa_mate_prefill_trace_enabled() && causal && use_full_kv) {
    LOG(INFO) << "[DSV4_MATE_PREFILL_ROUTE] layer=" << layer_id_
              << " q=" << tensor_sizes_string(q)
              << " kv=" << tensor_sizes_string(kv)
              << " compressed=" << tensor_sizes_string(compressed_kv)
              << " indices_defined=" << compressed_indices.defined();
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
    size_t sequence_index) {
  if (!(compress_ratio_ == 4 || compress_ratio_ == 128) || !compressor_wkv_ ||
      !compressor_wgate_ || !compressor_norm_ || !compressor_ape_loaded_ ||
      !hidden_states.defined() || hidden_states.numel() == 0) {
    return torch::Tensor();
  }
  const int64_t seq_len = hidden_states.size(0);
  const int64_t cutoff = seq_len - (seq_len % compress_ratio_);

  auto kv_full =
      compressor_wkv_->forward(hidden_states).to(torch::kFloat32).contiguous();
  maybe_sync_attention_stage(layer_id_, "compress_prefill_wkv", kv_full);
  auto score_full = compressor_wgate_->forward(hidden_states)
                        .to(torch::kFloat32)
                        .contiguous();
  maybe_sync_attention_stage(layer_id_, "compress_prefill_wgate", score_full);
  auto ape =
      compressor_ape_.to(score_full.device()).to(torch::kFloat32).contiguous();

  const int64_t state_rows = compressor_coff_ * compress_ratio_;
  const int64_t state_dim = compressor_coff_ * head_dim_;
  if (sequence_index >= compressor_kv_state_.size()) {
    compressor_kv_state_.resize(sequence_index + 1);
    compressor_score_state_.resize(sequence_index + 1);
  }
  auto state_options = hidden_states.options().dtype(torch::kFloat32);
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
  if (cos.defined() && sin.defined() &&
      cos.numel() >= blocks * rope_head_dim_ &&
      sin.numel() >= blocks * rope_head_dim_) {
    auto cos_slice = cos.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks);
    auto sin_slice = sin.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks);
    compressed = apply_interleaved_partial_rope(
        compressed, nope_head_dim_, rope_head_dim_, cos_slice, sin_slice);
    maybe_sync_attention_stage(layer_id_, "compress_prefill_rope", compressed);
  }
  fp8_act_quant_roundtrip_inplace(
      compressed, nope_head_dim_, /*block_size=*/64);
  maybe_sync_attention_stage(
      layer_id_, "compress_prefill_fp8_roundtrip", compressed);
  return compressed;
}

torch::Tensor DeepseekV4AttentionImpl::compress_decode_step(
    const torch::Tensor& hidden_step,
    int64_t start_pos,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    size_t sequence_index) {
  if (!(compress_ratio_ == 4 || compress_ratio_ == 128) || !compressor_wkv_ ||
      !compressor_wgate_ || !compressor_norm_ || !compressor_ape_loaded_ ||
      !hidden_step.defined() || hidden_step.numel() == 0 ||
      hidden_step.size(0) != 1 || start_pos < 0) {
    return torch::Tensor();
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
  auto kv_step =
      compressor_wkv_->forward(hidden_step).to(torch::kFloat32).contiguous();
  maybe_sync_attention_stage(layer_id_, "compress_decode_wkv", kv_step);
  auto score_step =
      compressor_wgate_->forward(hidden_step).to(torch::kFloat32).contiguous();
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
    size_t sequence_index) {
  constexpr int64_t kCompressRatio = 4;
  constexpr int64_t kCompressorCoff = 2;
  if (compress_ratio_ != kCompressRatio || !indexer_compressor_wkv_ ||
      !indexer_compressor_wgate_ || !indexer_compressor_norm_ ||
      !indexer_compressor_ape_loaded_ || !hidden_states.defined() ||
      hidden_states.numel() == 0) {
    return torch::Tensor();
  }

  const int64_t seq_len = hidden_states.size(0);
  const int64_t cutoff = seq_len - (seq_len % kCompressRatio);
  auto kv_full = indexer_compressor_wkv_->forward(hidden_states)
                     .to(torch::kFloat32)
                     .contiguous();
  auto score_full = indexer_compressor_wgate_->forward(hidden_states)
                        .to(torch::kFloat32)
                        .contiguous();
  auto ape = indexer_compressor_ape_.to(score_full.device())
                 .to(torch::kFloat32)
                 .contiguous();

  const int64_t state_rows = kCompressorCoff * kCompressRatio;
  const int64_t state_dim = kCompressorCoff * index_head_dim_;
  if (sequence_index >= indexer_kv_state_.size()) {
    indexer_kv_state_.resize(sequence_index + 1);
    indexer_score_state_.resize(sequence_index + 1);
  }
  auto state_options = hidden_states.options().dtype(torch::kFloat32);
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
  if (cos.defined() && sin.defined() &&
      cos.numel() >= blocks * rope_head_dim_ &&
      sin.numel() >= blocks * rope_head_dim_) {
    compressed = apply_interleaved_partial_rope(
        compressed,
        index_head_dim_ - rope_head_dim_,
        rope_head_dim_,
        cos.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks),
        sin.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks));
  }
  return apply_indexer_hadamard(compressed, indexer_hadamard_);
}

torch::Tensor DeepseekV4AttentionImpl::compress_indexer_decode_step(
    const torch::Tensor& hidden_step,
    int64_t start_pos,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    size_t sequence_index) {
  constexpr int64_t kCompressRatio = 4;
  constexpr int64_t kCompressorCoff = 2;
  if (compress_ratio_ != kCompressRatio || !indexer_compressor_wkv_ ||
      !indexer_compressor_wgate_ || !indexer_compressor_norm_ ||
      !indexer_compressor_ape_loaded_ || !hidden_step.defined() ||
      hidden_step.numel() == 0 || hidden_step.size(0) != 1 || start_pos < 0) {
    return torch::Tensor();
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
  auto kv_step = indexer_compressor_wkv_->forward(hidden_step)
                     .to(torch::kFloat32)
                     .contiguous();
  auto score_step = indexer_compressor_wgate_->forward(hidden_step)
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
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& index_kv,
    const torch::Tensor& index_cache,
    const torch::Tensor& index_scale_cache,
    const torch::Tensor& index_block_table,
    int64_t sequence_index,
    int64_t index_len,
    int64_t q_abs_start) {
  if (!dsv4_musa_indexer_enabled() || !indexer_wq_b_ ||
      !indexer_weights_proj_ || qr.numel() == 0 || index_len <= 0) {
    return torch::Tensor();
  }
  CHECK_EQ(qr.size(0), hidden_states.size(0));

  const int64_t q_tokens = qr.size(0);
  const bool use_fused_topk = dsv4_musa_fused_indexer_topk_enabled() &&
                              q_tokens == 1 && index_cache.defined() &&
                              index_scale_cache.defined() &&
                              index_block_table.defined();
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
  auto query = indexer_wq_b_->forward(qr).view(
      {q_tokens, index_n_heads_, index_head_dim_});
  if (cos.defined() && sin.defined() && cos.size(0) >= q_tokens &&
      sin.size(0) >= q_tokens) {
    query = apply_interleaved_partial_rope(
        query,
        index_head_dim_ - rope_head_dim_,
        rope_head_dim_,
        cos.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens),
        sin.slice(/*dim=*/0, /*start=*/0, /*end=*/q_tokens));
  }
  query = apply_indexer_hadamard(query, indexer_hadamard_);
  if (dsv4_musa_indexer_query_int8_enabled()) {
    query = dynamic_int8_roundtrip_last_dim(query);
  }
  auto head_weights =
      indexer_weights_proj_->forward(hidden_states).to(torch::kFloat32) *
      indexer_score_scale_;
  if (use_fused_topk) {
    ::xllm::kernel::DeepseekV4IndexerTopkParams params;
    params.query = query.contiguous();
    params.weights = head_weights.contiguous();
    params.cache = index_cache;
    params.scale_cache = index_scale_cache;
    params.block_table =
        index_block_table.to(index_cache.device()).contiguous();
    params.sequence_index = sequence_index;
    params.kv_len = index_len;
    params.topk = index_topk_;
    params.query_absolute_start = q_abs_start;
    params.compress_ratio = 4;
    auto selected = ::xllm::kernel::deepseek_v4_indexer_topk(params);
    if (dsv4_musa_mate_fused_dsa_trace_enabled()) {
      LOG(INFO) << "[DSV4_MUSA_FUSED_INDEXER] layer=" << layer_id_
                << " q_tokens=" << q_tokens << " kv_tokens=" << kv_tokens
                << " topk=" << index_topk_;
    }
    return selected;
  }
  auto scores =
      torch::matmul(query.to(torch::kFloat32),
                    active_index_kv.to(torch::kFloat32).transpose(0, 1));
  scores = torch::relu(scores);
  scores = (scores * head_weights.unsqueeze(-1)).sum(/*dim=*/1);

  std::vector<int64_t> valid_counts;
  valid_counts.reserve(static_cast<size_t>(q_tokens));
  for (int64_t token = 0; token < q_tokens; ++token) {
    valid_counts.push_back(
        std::clamp<int64_t>((q_abs_start + token + 1) / 4, 0, kv_tokens));
  }
  auto valid = torch::tensor(
      valid_counts,
      torch::TensorOptions().dtype(torch::kLong).device(scores.device()));
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
  (void)compress_metadata;

  const int32_t profile_device_id = hidden_states.device().index();
  auto* prefill_profile =
      begin_prefill_profile(layer_id_, hidden_states, is_prefill);

  auto q_down = q_a_proj_->forward(hidden_states);
  maybe_sync_attention_stage(layer_id_, "q_a_proj", q_down);
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " after_q_a shape=" << q_down.sizes();
  }
  auto qr = std::get<0>(q_layernorm_->forward(q_down));
  maybe_sync_attention_stage(layer_id_, "q_layernorm", qr);
  auto q = q_b_proj_->forward(qr).view({-1, n_local_heads_, head_dim_});
  maybe_sync_attention_stage(layer_id_, "q_b_proj", q);
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " after_q_b shape=" << q.sizes();
  }
  q = rms_norm_last_dim(q, q_rms_gamma_, eps_);
  maybe_sync_attention_stage(layer_id_, "q_rms_norm", q);

  auto kv_down = kv_proj_->forward(hidden_states);
  maybe_sync_attention_stage(layer_id_, "kv_proj", kv_down);
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " after_kv_proj shape=" << kv_down.sizes();
  }
  auto kv =
      std::get<0>(kv_layernorm_->forward(kv_down)).view({-1, 1, head_dim_});
  maybe_sync_attention_stage(layer_id_, "kv_layernorm", kv);

  const auto [active_cos, active_sin] =
      select_rope_tables(attn_metadata, compress_ratio_, q.size(0));
  q = apply_interleaved_partial_rope(
      q, nope_head_dim_, rope_head_dim_, active_cos, active_sin);
  maybe_sync_attention_stage(layer_id_, "q_rope", q);
  kv = apply_interleaved_partial_rope(
      kv, nope_head_dim_, rope_head_dim_, active_cos, active_sin);
  maybe_sync_attention_stage(layer_id_, "kv_rope", kv);
  fp8_act_quant_roundtrip_inplace(kv, nope_head_dim_, /*block_size=*/64);
  maybe_sync_attention_stage(layer_id_, "kv_fp8_roundtrip", kv);

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
  if (ori_kv.defined()) {
    scatter_by_slot(ori_kv, swa_slot_mapping, kv);
    maybe_sync_attention_stage(layer_id_, "swa_scatter_by_slot", kv);
  }
  if (mate_swa_cache.defined()) {
    pack_mate_model1_cache(mate_swa_cache, swa_slot_mapping, kv);
    maybe_sync_attention_stage(layer_id_, "mate_swa_pack", kv);
  }

  const int64_t total_tokens = q.size(0);
  auto query_ranges = build_query_ranges(attn_metadata, total_tokens);
  auto kv_lens =
      build_kv_lens(attn_metadata, query_ranges.size(), query_ranges);
  const int64_t full_kv_threshold = dsv4_musa_prefill_full_kv_threshold();
  const int64_t max_kv_len =
      kv_lens.empty() ? total_tokens
                      : *std::max_element(kv_lens.begin(), kv_lens.end());
  const bool force_full_kv_attention =
      full_kv_threshold > 0 && max_kv_len <= full_kv_threshold;
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

  const bool use_compressed_attention =
      dsv4_musa_compress_fallback_enabled() && !force_full_kv_attention &&
      (compress_ratio_ == 4 || compress_ratio_ == 128) && compressor_wkv_ &&
      compressor_wgate_ && compressor_norm_ && compressor_ape_loaded_;
  const bool use_indexer_attention =
      use_compressed_attention && compress_ratio_ == 4 && !is_chunked_prefill &&
      dsv4_musa_indexer_enabled() && indexer_wq_b_ && indexer_weights_proj_ &&
      indexer_compressor_wkv_ && indexer_compressor_wgate_ &&
      indexer_compressor_norm_ && indexer_compressor_ape_loaded_;
  const bool apply_sparse_indexer_attention =
      use_indexer_attention &&
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
              << " use_indexer=" << use_indexer_attention;
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
  auto index_cache =
      use_indexer_attention ? kv_cache.get_index_cache() : torch::Tensor();
  auto index_scale_cache = use_indexer_attention
                               ? dsv4_musa_indexer_scale_cache(kv_cache)
                               : torch::Tensor();
  const int64_t index_cache_idx =
      use_indexer_attention
          ? find_token_cache_index(
                attn_metadata, layer_id_, compress_ratio_, /*ordinal=*/1)
          : -1;
  auto index_slot_mapping =
      use_indexer_attention
          ? layer_cache_tensor(
                attn_metadata.slot_mappings, layer_id_, index_cache_idx)
          : torch::Tensor();
  auto index_block_table =
      use_indexer_attention
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
  const bool use_persistent_compressor_state =
      use_compressed_attention &&
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
      use_indexer_attention
          ? find_swa_cache_index(attn_metadata, layer_id_, /*ordinal=*/3)
          : -1;
  const int64_t indexer_score_state_cache_idx =
      use_indexer_attention
          ? find_swa_cache_index(attn_metadata, layer_id_, /*ordinal=*/4)
          : -1;
  auto indexer_state_block_table =
      use_indexer_attention
          ? layer_cache_tensor(
                attn_metadata.block_tables, layer_id_, indexer_state_cache_idx)
          : torch::Tensor();
  auto indexer_score_state_block_table =
      use_indexer_attention ? layer_cache_tensor(attn_metadata.block_tables,
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
      use_indexer_attention && dsv4_musa_persistent_indexer_state_enabled() &&
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
  if (use_persistent_compressor_state) {
    compressor_kv_state_.resize(query_ranges.size());
    compressor_score_state_.resize(query_ranges.size());
    auto kv_state_2d = compressor_kv_cache.view({-1, compressor_state_dim});
    auto score_state_2d =
        compressor_score_cache.view({-1, compressor_state_dim});
    const int64_t state_block_size = compressor_kv_cache.size(1);
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      auto kv_slots = compressor_state_slots(compressor_state_block_table,
                                             static_cast<int64_t>(seq_idx),
                                             compressor_state_rows,
                                             state_block_size,
                                             compressor_kv_cache.device());
      auto score_slots =
          compressor_state_slots(compressor_score_state_block_table,
                                 static_cast<int64_t>(seq_idx),
                                 compressor_state_rows,
                                 state_block_size,
                                 compressor_score_cache.device());
      compressor_kv_state_slots_by_seq[seq_idx] = kv_slots;
      compressor_score_state_slots_by_seq[seq_idx] = score_slots;
      if ((!is_prefill || is_chunked_prefill) && kv_slots.defined() &&
          score_slots.defined()) {
        compressor_kv_state_[seq_idx] = kv_state_2d.index_select(0, kv_slots);
        compressor_score_state_[seq_idx] =
            score_state_2d.index_select(0, score_slots);
        if (trace_layer0) {
          LOG(INFO) << "[DSV4_MUSA_STATE] phase=load kind=compressor seq="
                    << seq_idx << " kv_slots=" << tensor_i64_prefix(kv_slots)
                    << " score_slots=" << tensor_i64_prefix(score_slots)
                    << " kv="
                    << tensor_stat_string(compressor_kv_state_[seq_idx])
                    << " score="
                    << tensor_stat_string(compressor_score_state_[seq_idx]);
        }
      }
    }
  }
  if (use_persistent_indexer_state) {
    indexer_kv_state_.resize(query_ranges.size());
    indexer_score_state_.resize(query_ranges.size());
    auto kv_state_2d = indexer_kv_state_cache.view({-1, indexer_state_dim});
    auto score_state_2d =
        indexer_score_state_cache.view({-1, indexer_state_dim});
    const int64_t state_block_size = indexer_kv_state_cache.size(1);
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      auto kv_slots = compressor_state_slots(indexer_state_block_table,
                                             static_cast<int64_t>(seq_idx),
                                             indexer_state_rows,
                                             state_block_size,
                                             indexer_kv_state_cache.device());
      auto score_slots =
          compressor_state_slots(indexer_score_state_block_table,
                                 static_cast<int64_t>(seq_idx),
                                 indexer_state_rows,
                                 state_block_size,
                                 indexer_score_state_cache.device());
      indexer_kv_state_slots_by_seq[seq_idx] = kv_slots;
      indexer_score_state_slots_by_seq[seq_idx] = score_slots;
      if ((!is_prefill || is_chunked_prefill) && kv_slots.defined() &&
          score_slots.defined()) {
        indexer_kv_state_[seq_idx] = kv_state_2d.index_select(0, kv_slots);
        indexer_score_state_[seq_idx] =
            score_state_2d.index_select(0, score_slots);
        if (trace_layer0) {
          LOG(INFO) << "[DSV4_MUSA_STATE] phase=load kind=indexer seq="
                    << seq_idx << " kv_slots=" << tensor_i64_prefix(kv_slots)
                    << " score_slots=" << tensor_i64_prefix(score_slots)
                    << " kv=" << tensor_stat_string(indexer_kv_state_[seq_idx])
                    << " score="
                    << tensor_stat_string(indexer_score_state_[seq_idx]);
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
  if (use_compressed_attention && is_prefill && !is_chunked_prefill) {
    compressor_kv_history_.clear();
    compressor_kv_state_.clear();
    compressor_score_state_.clear();
    indexer_kv_history_.clear();
    indexer_kv_state_.clear();
    indexer_score_state_.clear();
    if (retain_compressed_history) {
      compressor_kv_history_.resize(query_ranges.size());
    }
    compressor_kv_state_.resize(query_ranges.size());
    compressor_score_state_.resize(query_ranges.size());
    if (use_indexer_attention) {
      if (!use_paged_indexer_cache) {
        indexer_kv_history_.resize(query_ranges.size());
      }
      indexer_kv_state_.resize(query_ranges.size());
      indexer_score_state_.resize(query_ranges.size());
    }
    std::vector<torch::Tensor> compressed_to_cache;
    compressed_to_cache.reserve(query_ranges.size());
    std::vector<torch::Tensor> indexer_to_cache;
    indexer_to_cache.reserve(query_ranges.size());
    int64_t rope_offset = 0;
    for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
      const int64_t q_start = query_ranges[seq_idx].first;
      const int64_t q_end = query_ranges[seq_idx].second;
      const int64_t q_len = std::max<int64_t>(q_end - q_start, 0);
      const int64_t compressed_blocks = q_len / compress_ratio_;
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
      auto hidden_seq =
          hidden_states.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
      auto cmp_seq =
          compress_prefill_sequence(hidden_seq, cos_slice, sin_slice, seq_idx);
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
      if (use_indexer_attention) {
        auto index_seq = compress_indexer_prefill_sequence(
            hidden_seq, cos_slice, sin_slice, seq_idx);
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
    }
    if (!compressed_to_cache.empty() && cmp_cache.defined() &&
        cmp_slot_mapping.defined()) {
      auto cmp_all = torch::cat(compressed_to_cache, /*dim=*/0);
      scatter_by_slot(cmp_cache, cmp_slot_mapping, cmp_all);
      maybe_sync_attention_stage(
          layer_id_, "prefill_cmp_scatter_by_slot", cmp_all);
      if (mate_cmp_cache.defined()) {
        pack_mate_model1_cache(mate_cmp_cache, cmp_slot_mapping, cmp_all);
        maybe_sync_attention_stage(layer_id_, "prefill_mate_cmp_pack", cmp_all);
      }
    }
    if (use_paged_indexer_cache && !indexer_to_cache.empty()) {
      auto index_all = torch::cat(indexer_to_cache, /*dim=*/0);
      quant_scatter_indexer_cache(
          index_cache, index_scale_cache, index_slot_mapping, index_all);
      maybe_sync_attention_stage(
          layer_id_, "prefill_indexer_cache_scatter", index_all);
    }
  }
  record_prefill_profile_boundary(prefill_profile, 2, profile_device_id);

  std::vector<torch::Tensor> decode_compressed_to_cache;
  std::vector<torch::Tensor> decode_indexer_to_cache;
  int64_t decode_compressed_rows_scattered = 0;
  auto flush_decode_compressed_cache = [&]() {
    if (decode_compressed_to_cache.empty() || !cmp_cache.defined() ||
        !cmp_slot_mapping.defined()) {
      return;
    }
    auto cmp_all = torch::cat(decode_compressed_to_cache, /*dim=*/0);
    if (decode_compressed_rows_scattered >= cmp_all.size(0)) {
      return;
    }
    auto pending = cmp_all.slice(/*dim=*/0,
                                 /*start=*/decode_compressed_rows_scattered,
                                 /*end=*/cmp_all.size(0));
    auto flat_slots = cmp_slot_mapping.reshape({-1});
    CHECK_GE(flat_slots.numel(), cmp_all.size(0));
    auto pending_slots = flat_slots
                             .slice(/*dim=*/0,
                                    /*start=*/decode_compressed_rows_scattered,
                                    /*end=*/cmp_all.size(0))
                             .contiguous();
    scatter_by_slot(cmp_cache, pending_slots, pending);
    maybe_sync_attention_stage(
        layer_id_, "decode_cmp_scatter_by_slot", pending);
    if (mate_cmp_cache.defined()) {
      pack_mate_model1_cache(mate_cmp_cache, pending_slots, pending);
      maybe_sync_attention_stage(layer_id_, "decode_mate_cmp_pack", pending);
    }
    if (use_paged_indexer_cache && !decode_indexer_to_cache.empty()) {
      auto index_all = torch::cat(decode_indexer_to_cache, /*dim=*/0);
      CHECK_EQ(index_all.size(0), cmp_all.size(0));
      auto pending_index = index_all.slice(
          /*dim=*/0,
          /*start=*/decode_compressed_rows_scattered,
          /*end=*/index_all.size(0));
      auto flat_index_slots = index_slot_mapping.reshape({-1});
      CHECK_GE(flat_index_slots.numel(), index_all.size(0));
      auto pending_index_slots =
          flat_index_slots
              .slice(/*dim=*/0,
                     /*start=*/decode_compressed_rows_scattered,
                     /*end=*/index_all.size(0))
              .contiguous();
      quant_scatter_indexer_cache(
          index_cache, index_scale_cache, pending_index_slots, pending_index);
      maybe_sync_attention_stage(
          layer_id_, "decode_indexer_cache_scatter", pending_index);
    }
    decode_compressed_rows_scattered = cmp_all.size(0);
  };
  int64_t decode_compressed_rope_offset = 0;
  for (size_t seq_idx = 0; seq_idx < query_ranges.size(); ++seq_idx) {
    const int64_t q_start = query_ranges[seq_idx].first;
    const int64_t q_end = query_ranges[seq_idx].second;
    if (q_end <= q_start) {
      continue;
    }
    auto q_seq = q.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
    torch::Tensor kv_seq;
    const bool use_current_prefill_kv = is_prefill && !is_chunked_prefill;
    const bool use_paged_decode_attention =
        !use_current_prefill_kv && !use_indexer_attention &&
        dsv4_musa_paged_dense_decode_enabled() &&
        dsv4_musa_fused_dense_decode_enabled();
    if (use_current_prefill_kv) {
      kv_seq = kv.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end).squeeze(1);
    } else if (!use_paged_decode_attention) {
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
        cmp_seq = prefill_compressed_by_seq[seq_idx];
      } else {
        if (retain_compressed_history &&
            seq_idx >= compressor_kv_history_.size()) {
          compressor_kv_history_.resize(seq_idx + 1);
        }
        if (use_indexer_attention && !use_paged_indexer_cache &&
            seq_idx >= indexer_kv_history_.size()) {
          indexer_kv_history_.resize(seq_idx + 1);
        }
        if (use_indexer_attention && seq_idx >= indexer_kv_state_.size()) {
          indexer_kv_state_.resize(seq_idx + 1);
          indexer_score_state_.resize(seq_idx + 1);
        }
        auto hidden_seq =
            hidden_states.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
        if (hidden_seq.defined() && hidden_seq.numel() > 0) {
          const int64_t seq_q_len = q_end - q_start;
          const int64_t decode_start_pos =
              std::max<int64_t>(kv_lens[seq_idx] - seq_q_len, 0);
          for (int64_t token_idx = 0; token_idx < seq_q_len; ++token_idx) {
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
            auto new_cmp = compress_decode_step(
                token_hidden, abs_pos, cos_row, sin_row, seq_idx);
            torch::Tensor new_indexer_cmp;
            if (use_indexer_attention) {
              new_indexer_cmp = compress_indexer_decode_step(
                  token_hidden, abs_pos, cos_row, sin_row, seq_idx);
            }
            if (new_cmp.defined() && new_cmp.numel() > 0) {
              if (retain_compressed_history) {
                if (compressor_kv_history_[seq_idx].defined() &&
                    compressor_kv_history_[seq_idx].numel() > 0) {
                  compressor_kv_history_[seq_idx] =
                      torch::cat({compressor_kv_history_[seq_idx], new_cmp},
                                 /*dim=*/0)
                          .contiguous();
                } else {
                  compressor_kv_history_[seq_idx] = new_cmp.contiguous();
                }
              }
              current_compressed_rows.push_back(new_cmp);
              decode_compressed_to_cache.push_back(new_cmp);
            }
            if (new_indexer_cmp.defined() && new_indexer_cmp.numel() > 0) {
              if (use_paged_indexer_cache) {
                decode_indexer_to_cache.push_back(new_indexer_cmp);
              } else {
                if (indexer_kv_history_[seq_idx].defined() &&
                    indexer_kv_history_[seq_idx].numel() > 0) {
                  indexer_kv_history_[seq_idx] =
                      torch::cat(
                          {indexer_kv_history_[seq_idx], new_indexer_cmp},
                          /*dim=*/0)
                          .contiguous();
                } else {
                  indexer_kv_history_[seq_idx] = new_indexer_cmp.contiguous();
                }
              }
            }
          }
        }
        const int64_t cmp_len = kv_lens[seq_idx] / compress_ratio_;
        const int64_t dense_tail_len =
            static_cast<int64_t>(current_compressed_rows.size());
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
            cmp_seq = compressor_kv_history_[seq_idx];
          }
          if (!cmp_seq.defined() && !use_indexer_attention) {
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
    const int64_t seq_q_len = q_end - q_start;
    const int64_t q_abs_start =
        use_current_prefill_kv
            ? 0
            : std::max<int64_t>(kv_lens[seq_idx] - seq_q_len, 0);
    torch::Tensor index_kv;
    const int64_t index_len = kv_lens[seq_idx] / 4;
    const bool use_fused_indexer_topk =
        use_paged_indexer_cache && dsv4_musa_fused_indexer_topk_enabled() &&
        seq_q_len == 1;
    if (use_paged_indexer_cache && index_len > 0) {
      // A decode step on a C4 boundary produces a row consumed immediately.
      flush_decode_compressed_cache();
      if (!use_fused_indexer_topk) {
        index_kv = gather_indexer_cache(index_cache,
                                        index_scale_cache,
                                        index_block_table,
                                        static_cast<int64_t>(seq_idx),
                                        index_len,
                                        hidden_states.scalar_type());
        maybe_sync_attention_stage(layer_id_, "indexer_cache_gather", index_kv);
      }
    } else if (seq_idx < indexer_kv_history_.size()) {
      index_kv = indexer_kv_history_[seq_idx];
    }
    if (apply_sparse_indexer_attention && index_len > 0 &&
        (use_fused_indexer_topk ||
         (index_kv.defined() && index_kv.size(0) == index_len))) {
      auto qr_seq = qr.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
      auto hidden_seq =
          hidden_states.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
      auto index_cos =
          active_cos.defined()
              ? active_cos.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end)
              : torch::Tensor();
      auto index_sin =
          active_sin.defined()
              ? active_sin.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end)
              : torch::Tensor();
      cmp_indices = build_indexer_topk(qr_seq,
                                       hidden_seq,
                                       index_cos,
                                       index_sin,
                                       index_kv,
                                       index_cache,
                                       index_scale_cache,
                                       index_block_table,
                                       static_cast<int64_t>(seq_idx),
                                       index_len,
                                       q_abs_start);
      maybe_sync_attention_stage(layer_id_, "indexer_topk", cmp_indices);
    }
    torch::Tensor seq_output;
    if (use_indexer_attention && !use_current_prefill_kv &&
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
    if (use_indexer_attention && !use_current_prefill_kv &&
        allow_mate_sparse_attention && cmp_indices.defined() &&
        mate_cmp_cache.defined() && mate_swa_cache.defined() &&
        cmp_block_table.defined() && swa_block_table.defined()) {
      // TopK may select the compressed row produced at this decode step.
      // Commit pending rows before the packed cache is consumed by Mate.
      flush_decode_compressed_cache();
      seq_output = mate_sparse_attention(q_seq,
                                         mate_cmp_cache,
                                         cmp_indices,
                                         cmp_block_table,
                                         mate_swa_cache,
                                         swa_block_table,
                                         static_cast<int64_t>(seq_idx),
                                         kv_lens[seq_idx]);
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
                    << " extra_mean_abs=" << extra_diff.mean().item<float>();
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
      auto paged_output = paged_dense_attention(q_seq,
                                                ori_kv,
                                                swa_block_table,
                                                static_cast<int64_t>(seq_idx),
                                                kv_lens[seq_idx],
                                                cmp_seq,
                                                paged_cmp_cache,
                                                paged_cmp_block_table,
                                                paged_cmp_len);
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
      seq_output = dense_attention(q_seq,
                                   kv_seq,
                                   cmp_seq,
                                   cmp_indices,
                                   use_current_prefill_kv,
                                   q_abs_start,
                                   use_current_prefill_kv);
    }
    maybe_sync_attention_stage(layer_id_, "seq_dense_attention", seq_output);
    if (trace_layer0) {
      LOG(INFO) << "[DSV4_MUSA_TRACE] layer0 seq=" << seq_idx << " q_range=["
                << q_start << "," << q_end << ")"
                << " kv_seq_shape=" << tensor_sizes_string(kv_seq)
                << " kv_seq_stats=" << tensor_stat_string(kv_seq)
                << " out_stats=" << tensor_stat_string(seq_output);
    }
    outputs.push_back(seq_output);
  }
  flush_decode_compressed_cache();
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
      if (seq_idx < slots.size() && slots[seq_idx].defined() &&
          seq_idx < states.size() && states[seq_idx].defined()) {
        if (trace_layer0) {
          LOG(INFO) << "[DSV4_MUSA_STATE] phase=persist kind=" << stage
                    << " seq=" << seq_idx
                    << " slots=" << tensor_i64_prefix(slots[seq_idx])
                    << " state=" << tensor_stat_string(states[seq_idx]);
        }
        active_slots.push_back(slots[seq_idx]);
        active_states.push_back(states[seq_idx]);
      }
    }
    if (active_slots.empty()) {
      return;
    }
    auto all_slots = torch::cat(active_slots, /*dim=*/0);
    auto all_states = torch::cat(active_states, /*dim=*/0);
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
  if (dsv4_musa_trace_enabled()) {
    LOG(INFO) << "[DSV4_MUSA_TRACE] attention layer=" << layer_id_
              << " dense_done seqs=" << query_ranges.size();
  }

  torch::Tensor attn_output =
      outputs.empty() ? torch::zeros_like(q) : torch::cat(outputs, /*dim=*/0);
  maybe_sync_attention_stage(layer_id_, "outputs_cat", attn_output);
  record_prefill_profile_boundary(prefill_profile, 3, profile_device_id);

  auto o = apply_interleaved_partial_rope(attn_output,
                                          nope_head_dim_,
                                          rope_head_dim_,
                                          active_cos,
                                          active_sin,
                                          /*inverse=*/true);
  maybe_sync_attention_stage(layer_id_, "output_inverse_rope", o);
  const int64_t num_tokens = o.size(0);
  auto o_group = o.view({num_tokens, n_local_groups_, -1});
  auto wo_a =
      dequantize_fp8_projection_weight(
          o_a_proj_->weight(), o_a_proj_->weight_scale(), o_group.scalar_type())
          .view({n_local_groups_, o_lora_rank_, -1});
  auto o_low_rank = torch::einsum("tgd,grd->tgr", {o_group, wo_a});
  maybe_sync_attention_stage(layer_id_, "output_o_a_einsum", o_low_rank);
  record_prefill_profile_boundary(prefill_profile, 4, profile_device_id);
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
        auto output_chunk = o_b_proj_->forward(input_chunk);
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
      maybe_sync_attention_stage(layer_id_, "output_o_b_local_matmul", output);
      if (layer_id_ == 0 && num_tokens > 1 && reduce_delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(reduce_delay_us));
      }
    }
  } else {
    output = o_b_proj_->forward(o_low_rank.reshape({num_tokens, -1}));
  }
  maybe_sync_attention_stage(layer_id_, "output_o_b_proj", output);
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
  o_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
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
}

}  // namespace layer
}  // namespace xllm
