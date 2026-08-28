/* Copyright 2026 The xLLM Authors.

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

// xllm_ops: MUSA dispatch registration for the Python model executor.
// Mirrors the schema in cuda_ops_library.cpp (compiled only for USE_CUDA).

#include "core/kernels/musa/musa_ops_library.h"

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/kernels/musa/attention_runner.h"
#include "core/kernels/musa/gdn_ops.h"
#include "core/kernels/musa/global_capture_instance.h"
#include "core/kernels/musa/llm_decode_metadata_update.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/kernels/musa/ops_api.h"
#include "core/kernels/musa/piecewise_graphs.h"
#include "core/kernels/ops_api.h"
#include "core/kernels/param.h"
#include "core/util/env_var.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm {
namespace {

thread_local torch::Tensor g_causal_conv1d_decode_out;
thread_local torch::Tensor g_causal_conv1d_prefill_in;
thread_local torch::Tensor g_causal_conv1d_prefill_out;
thread_local torch::Tensor g_fused_swiglu_q;
thread_local torch::Tensor g_fused_swiglu_s;
thread_local std::unordered_map<int64_t, torch::Tensor> g_ptq_q_by_cols;
thread_local std::unordered_map<int64_t, torch::Tensor> g_ptq_s_by_cols;
thread_local std::unordered_map<int64_t, torch::Tensor> g_ptq_in_by_cols;
thread_local torch::Tensor g_fused_gdn_decode_out;
thread_local torch::Tensor g_mate_gdn_decode_out;
thread_local torch::Tensor g_gdn_prefill_query;
thread_local torch::Tensor g_gdn_prefill_key;
thread_local torch::Tensor g_gdn_prefill_value;
thread_local torch::Tensor g_gdn_prefill_gate;
thread_local torch::Tensor g_gdn_prefill_beta;
thread_local torch::Tensor g_gdn_mate_output;
thread_local torch::Tensor g_gdn_mate_final_state;
thread_local torch::Tensor g_gdn_mate_kkt;
thread_local int64_t g_piecewise_next_handle = 1;
thread_local std::unordered_map<
    int64_t,
    std::unique_ptr<xllm::runtime::musa::PiecewiseGraphs>>
    g_piecewise_graphs;
thread_local std::optional<xllm::kernel::musa::MusaTvmffiStreamOverrideGuard>
    g_piecewise_ffi_guard;

constexpr int64_t kGdnChunkSize = 64;

int64_t align_gdn_prefill_tokens(int64_t num_tokens) {
  if (num_tokens < kGdnChunkSize) {
    return num_tokens;
  }
  return ((num_tokens + kGdnChunkSize - 1) / kGdnChunkSize) * kGdnChunkSize;
}

torch::Tensor grow_only_2d(torch::Tensor& buffer,
                           int64_t rows,
                           int64_t cols,
                           const torch::TensorOptions& options) {
  if (!buffer.defined() || buffer.size(0) < rows || buffer.size(1) != cols ||
      buffer.scalar_type() != options.dtype().toScalarType() ||
      buffer.device() != options.device()) {
    const int64_t target_rows =
        buffer.defined() ? std::max(rows, buffer.size(0)) : rows;
    buffer = torch::empty({target_rows, cols}, options);
  }
  return buffer.narrow(/*dim=*/0, /*start=*/0, /*length=*/rows);
}

// GDN o_proj and MLP gate_up alternate K. A single grow-only buffer
// reallocates on every col change; key by K so each shape grows once.
torch::Tensor grow_only_2d_keyed(
    std::unordered_map<int64_t, torch::Tensor>& buffers,
    int64_t rows,
    int64_t cols,
    const torch::TensorOptions& options) {
  torch::Tensor& buffer = buffers[cols];
  return grow_only_2d(buffer, rows, cols, options);
}

torch::Tensor grow_only_3d(torch::Tensor& buffer,
                           int64_t dim0,
                           int64_t dim1,
                           int64_t dim2,
                           const torch::TensorOptions& options) {
  if (!buffer.defined() || buffer.size(0) < dim0 || buffer.size(1) != dim1 ||
      buffer.size(2) != dim2 ||
      buffer.scalar_type() != options.dtype().toScalarType() ||
      buffer.device() != options.device()) {
    const int64_t target_dim0 =
        buffer.defined() ? std::max(dim0, buffer.size(0)) : dim0;
    buffer = torch::empty({target_dim0, dim1, dim2}, options);
  }
  return buffer.narrow(/*dim=*/0, /*start=*/0, /*length=*/dim0);
}

torch::Tensor grow_only_4d(torch::Tensor& buffer,
                           int64_t dim0,
                           int64_t dim1,
                           int64_t dim2,
                           int64_t dim3,
                           const torch::TensorOptions& options) {
  const bool same_layout =
      buffer.defined() && buffer.size(0) == dim0 && buffer.size(2) == dim2 &&
      buffer.size(3) == dim3 &&
      buffer.scalar_type() == options.dtype().toScalarType() &&
      buffer.device() == options.device();
  if (!same_layout || buffer.size(1) < dim1) {
    const int64_t target_dim1 =
        same_layout ? std::max(dim1, buffer.size(1)) : dim1;
    buffer = torch::empty({dim0, target_dim1, dim2, dim3}, options);
  }
  torch::Tensor live = buffer.narrow(/*dim=*/1, /*start=*/0, /*length=*/dim1);
  if (dim0 != 1 || buffer.size(1) == dim1) {
    return live;
  }
  // A narrow view of [1, capacity, H, D] is logically contiguous because the
  // singleton batch stride is ignored by PyTorch, but TileLang validates every
  // explicit stride. Normalize stride(0) to the live T without moving data.
  return live.as_strided({dim0, dim1, dim2, dim3},
                         {dim1 * dim2 * dim3, dim2 * dim3, dim3, 1});
}

torch::Tensor grow_only_like(torch::Tensor& buffer,
                             const torch::Tensor& reference) {
  if (!buffer.defined() || buffer.sizes() != reference.sizes() ||
      buffer.scalar_type() != reference.scalar_type() ||
      buffer.device() != reference.device()) {
    buffer = torch::empty_like(reference);
  }
  return buffer;
}

void* current_musa_stream_handle(const torch::Device& device) {
  const c10::musa::MUSAStream stream =
      c10::musa::getCurrentMUSAStream(device.index());
  return reinterpret_cast<void*>(stream.stream());
}

void fill_gdn_decode_params(
    xllm::kernel::musa::MateGatedDeltaRuleDecodeParams& params,
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& initial_state,
    const torch::Tensor& state_indices,
    double scale,
    int64_t num_k_heads,
    int64_t key_head_dim,
    int64_t value_head_dim,
    torch::Tensor decode_output) {
  params.mixed_qkv = mixed_qkv;
  params.state = initial_state;
  params.A_log = a_log;
  params.a = a.dim() == 3 ? a.select(/*dim=*/1, /*index=*/0) : a;
  params.dt_bias = dt_bias;
  params.b = b.dim() == 3 ? b.select(/*dim=*/1, /*index=*/0) : b;
  params.state_indices = state_indices;
  params.num_k_heads = num_k_heads;
  params.num_v_heads = a_log.numel();
  params.head_k_dim = key_head_dim;
  params.head_v_dim = value_head_dim;
  params.scale = scale;
  params.use_qk_l2norm = true;
  params.decode_output = decode_output;
}

torch::Tensor rms_norm_musa(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            double eps,
                            torch::Tensor& output) {
  CHECK(output.defined()) << "rms_norm requires an output buffer";
  CHECK_EQ(output.sizes(), input.sizes());
  CHECK_EQ(output.scalar_type(), input.scalar_type());
  CHECK_EQ(output.device(), input.device());
  CHECK(output.is_contiguous()) << "rms_norm output must be contiguous";
  torch::Tensor weight_arg = weight;
  torch::Tensor input_arg = input;
  xllm::kernel::musa::rms_norm(output, input_arg, weight_arg, eps);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm_musa(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  torch::Tensor weight_arg = weight;
  xllm::kernel::musa::fused_add_rms_norm(input, residual, weight_arg, eps);
  return std::make_tuple(input, residual);
}

torch::Tensor gemma_rms_norm_musa(const torch::Tensor& input,
                                  const torch::Tensor& weight,
                                  double eps,
                                  torch::Tensor& output) {
  CHECK(output.defined()) << "gemma_rms_norm requires an output buffer";
  CHECK_EQ(output.sizes(), input.sizes());
  CHECK_EQ(output.scalar_type(), input.scalar_type());
  CHECK_EQ(output.device(), input.device());
  CHECK(output.is_contiguous()) << "gemma_rms_norm output must be contiguous";
  CHECK_EQ(weight.scalar_type(), input.scalar_type())
      << "gemma_rms_norm weight dtype must match input";
  torch::Tensor weight_arg = weight;
  torch::Tensor input_arg = input;
  xllm::kernel::musa::gemma_rms_norm(output, input_arg, weight_arg, eps);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_gemma_rms_norm_musa(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  CHECK(input.data_ptr() != residual.data_ptr())
      << "fused_add_gemma_rms_norm requires distinct residual storage";
  CHECK_EQ(weight.scalar_type(), input.scalar_type())
      << "fused_add_gemma_rms_norm weight dtype must match input";
  torch::Tensor weight_arg = weight;
  xllm::kernel::musa::fused_add_gemma_rms_norm(
      input, residual, weight_arg, eps);
  return std::make_tuple(input, residual);
}

torch::Tensor mul_sigmoid_gate_inplace_musa(torch::Tensor& out,
                                            const torch::Tensor& gate) {
  xllm::kernel::musa::mul_sigmoid_gate_inplace(out, gate);
  return out;
}

torch::Tensor silu_and_mul_musa(const torch::Tensor& input,
                                torch::Tensor& output) {
  CHECK_GT(input.dim(), 0);
  const int64_t last_dim = input.size(-1);
  CHECK_EQ(last_dim % 2, 0)
      << "silu_and_mul: last dim must be even, got " << input.sizes();
  CHECK(output.defined()) << "silu_and_mul requires an output buffer";
  CHECK_EQ(output.dim(), input.dim());
  CHECK_EQ(output.size(-1), last_dim / 2);
  CHECK_EQ(output.numel(), input.numel() / 2);
  CHECK_EQ(output.scalar_type(), input.scalar_type());
  CHECK_EQ(output.device(), input.device());
  CHECK(output.is_contiguous()) << "silu_and_mul output must be contiguous";
  xllm::kernel::musa::act_and_mul(output, input, "silu");
  return output;
}

torch::Tensor fused_qk_norm_rope_musa(torch::Tensor& qkv,
                                      int64_t num_heads_q,
                                      int64_t num_heads_k,
                                      int64_t num_heads_v,
                                      int64_t head_dim,
                                      double eps,
                                      const torch::Tensor& q_weight,
                                      const torch::Tensor& k_weight,
                                      const torch::Tensor& cos_sin_cache,
                                      bool interleaved,
                                      const torch::Tensor& position_ids,
                                      int64_t k_head_offset) {
  torch::Tensor position_ids_i32 = position_ids;
  if (position_ids.scalar_type() != torch::kInt32) {
    CHECK_EQ(position_ids.scalar_type(), torch::kInt64)
        << "position_ids must be int32 or int64";
    position_ids_i32 = position_ids.to(torch::kInt32);
  }
  xllm::kernel::musa::fused_qk_norm_rope(qkv,
                                         num_heads_q,
                                         num_heads_k,
                                         num_heads_v,
                                         head_dim,
                                         eps,
                                         q_weight,
                                         k_weight,
                                         cos_sin_cache,
                                         interleaved,
                                         position_ids_i32,
                                         k_head_offset);
  return qkv;
}

torch::Tensor reshape_paged_cache_musa(const torch::Tensor& slot_mapping,
                                       const torch::Tensor& keys,
                                       const torch::Tensor& values,
                                       torch::Tensor& key_cache,
                                       torch::Tensor& value_cache) {
  xllm::kernel::musa::reshape_paged_cache(
      slot_mapping, keys, values, key_cache, value_cache);
  return key_cache;
}

void check_int32_musa(const torch::Tensor& tensor,
                      const torch::Device& device,
                      const char* name) {
  CHECK(tensor.defined()) << name << " must be defined";
  CHECK(!tensor.device().is_cpu()) << name << " must be a MUSA device tensor";
  CHECK_EQ(tensor.device(), device) << name << " must be on " << device;
  CHECK_EQ(tensor.scalar_type(), torch::kInt32)
      << name << " must have dtype int32";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

torch::Tensor update_decode_graph_metadata_musa(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices,
    const torch::Tensor& paged_kv_last_page_len,
    torch::Tensor& dst_tokens,
    torch::Tensor& dst_positions,
    torch::Tensor& dst_slot_mapping,
    torch::Tensor& dst_kv_seq_lens,
    torch::Tensor& dst_kv_seq_lens_delta,
    torch::Tensor& dst_paged_kv_indptr,
    torch::Tensor& dst_paged_kv_indices,
    torch::Tensor& dst_paged_kv_last_page_len,
    int64_t padded_num_tokens) {
  CHECK(tokens.defined()) << "tokens must be defined";
  const torch::Device device = tokens.device();
  check_int32_musa(tokens, device, "tokens");
  check_int32_musa(positions, device, "positions");
  check_int32_musa(slot_mapping, device, "slot_mapping");
  check_int32_musa(kv_seq_lens, device, "kv_seq_lens");
  check_int32_musa(paged_kv_indptr, device, "paged_kv_indptr");
  check_int32_musa(paged_kv_indices, device, "paged_kv_indices");
  check_int32_musa(paged_kv_last_page_len, device, "paged_kv_last_page_len");
  check_int32_musa(dst_tokens, device, "dst_tokens");
  check_int32_musa(dst_positions, device, "dst_positions");
  check_int32_musa(dst_slot_mapping, device, "dst_slot_mapping");
  check_int32_musa(dst_kv_seq_lens, device, "dst_kv_seq_lens");
  check_int32_musa(dst_kv_seq_lens_delta, device, "dst_kv_seq_lens_delta");
  check_int32_musa(dst_paged_kv_indptr, device, "dst_paged_kv_indptr");
  check_int32_musa(dst_paged_kv_indices, device, "dst_paged_kv_indices");
  check_int32_musa(
      dst_paged_kv_last_page_len, device, "dst_paged_kv_last_page_len");

  const int64_t actual_num_tokens = tokens.numel();
  const int64_t actual_batch_size = paged_kv_last_page_len.numel();
  const int64_t actual_indices_size = paged_kv_indices.numel();
  CHECK_EQ(actual_num_tokens, actual_batch_size)
      << "decode graph requires one token per sequence";
  CHECK_GE(padded_num_tokens, actual_num_tokens);
  CHECK_GE(positions.numel(), actual_num_tokens);
  CHECK_GE(slot_mapping.numel(), actual_num_tokens);
  CHECK_GE(kv_seq_lens.numel(), actual_batch_size + 1);
  CHECK_GE(paged_kv_indptr.numel(), actual_batch_size + 1);
  CHECK_GE(dst_tokens.numel(), padded_num_tokens);
  CHECK_GE(dst_positions.numel(), padded_num_tokens);
  CHECK_GE(dst_slot_mapping.numel(), padded_num_tokens);
  CHECK_GE(dst_kv_seq_lens.numel(), padded_num_tokens + 1);
  CHECK_GE(dst_kv_seq_lens_delta.numel(), padded_num_tokens);
  CHECK_GE(dst_paged_kv_indptr.numel(), padded_num_tokens + 1);
  CHECK_GE(dst_paged_kv_indices.numel(), actual_indices_size);
  CHECK_GE(dst_paged_kv_last_page_len.numel(), padded_num_tokens);

  xllm::kernel::musa::LlmDecodeMetadataUpdateParams params{
      .src_tokens = tokens.data_ptr<int32_t>(),
      .src_positions = positions.data_ptr<int32_t>(),
      .src_new_cache_slots = slot_mapping.data_ptr<int32_t>(),
      .src_kv_seq_lens = kv_seq_lens.data_ptr<int32_t>(),
      .src_paged_kv_indptr = paged_kv_indptr.data_ptr<int32_t>(),
      .src_paged_kv_indices = paged_kv_indices.data_ptr<int32_t>(),
      .src_paged_kv_last_page_len = paged_kv_last_page_len.data_ptr<int32_t>(),
      .dst_tokens = dst_tokens.data_ptr<int32_t>(),
      .dst_positions = dst_positions.data_ptr<int32_t>(),
      .dst_new_cache_slots = dst_slot_mapping.data_ptr<int32_t>(),
      .dst_kv_seq_lens = dst_kv_seq_lens.data_ptr<int32_t>(),
      .dst_kv_seq_lens_delta = dst_kv_seq_lens_delta.data_ptr<int32_t>(),
      .dst_paged_kv_indptr = dst_paged_kv_indptr.data_ptr<int32_t>(),
      .dst_paged_kv_indices = dst_paged_kv_indices.data_ptr<int32_t>(),
      .dst_paged_kv_last_page_len =
          dst_paged_kv_last_page_len.data_ptr<int32_t>(),
      .actual_num_tokens = actual_num_tokens,
      .padded_num_tokens = padded_num_tokens,
      .actual_batch_size = actual_batch_size,
      .actual_indices_size = actual_indices_size,
      .max_indices_size_for_graph_capacity = dst_paged_kv_indices.numel(),
  };
  const musaStream_t stream =
      c10::musa::getCurrentMUSAStream(tokens.device().index());
  xllm::kernel::musa::update_llm_decode_metadata(params, stream);
  return dst_tokens;
}

torch::Tensor update_fa3_graph_metadata_musa(const torch::Tensor& kv_seq_lens,
                                             const torch::Tensor& block_table,
                                             torch::Tensor& dst_kv_seq_lens,
                                             torch::Tensor& dst_block_table,
                                             int64_t actual_batch_size) {
  CHECK(kv_seq_lens.defined()) << "kv_seq_lens must be defined";
  const torch::Device device = kv_seq_lens.device();
  check_int32_musa(kv_seq_lens, device, "kv_seq_lens");
  check_int32_musa(block_table, device, "block_table");
  check_int32_musa(dst_kv_seq_lens, device, "dst_kv_seq_lens");
  check_int32_musa(dst_block_table, device, "dst_block_table");
  CHECK_EQ(block_table.dim(), 2) << "block_table must be [batch, pages]";
  CHECK_EQ(dst_block_table.dim(), 2)
      << "dst_block_table must be [padded_batch, pages]";
  CHECK_GT(actual_batch_size, 0);
  CHECK_GE(kv_seq_lens.numel(), actual_batch_size);
  CHECK_GE(block_table.size(0), actual_batch_size);
  CHECK_GE(dst_kv_seq_lens.numel(), dst_block_table.size(0));
  CHECK_GE(dst_block_table.size(1), block_table.size(1));

  xllm::kernel::musa::Fa3GraphMetadataUpdateParams params{
      .src_kv_seq_lens = kv_seq_lens.data_ptr<int32_t>(),
      .src_block_tables = block_table.data_ptr<int32_t>(),
      .dst_kv_seq_lens = dst_kv_seq_lens.data_ptr<int32_t>(),
      .dst_block_tables = dst_block_table.data_ptr<int32_t>(),
      .actual_batch_size = actual_batch_size,
      .padded_batch_size = dst_block_table.size(0),
      .src_block_table_width = block_table.size(1),
      .dst_block_table_width = dst_block_table.size(1),
  };
  const musaStream_t stream =
      c10::musa::getCurrentMUSAStream(kv_seq_lens.device().index());
  xllm::kernel::musa::update_fa3_graph_metadata(params, stream);
  return dst_kv_seq_lens;
}

torch::Tensor fa3_decode_scheduler_metadata_musa(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& seqused_k,
    int64_t batch_size,
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t num_splits,
    torch::Tensor& metadata) {
  CHECK_GT(batch_size, 0);
  CHECK_EQ(cu_seqlens_q.numel(), batch_size + 1);
  CHECK_EQ(seqused_k.numel(), batch_size);
  xllm::kernel::musa::fa3_decode_scheduler_metadata(
      cu_seqlens_q.device(),
      static_cast<int32_t>(batch_size),
      static_cast<int32_t>(num_heads_q),
      static_cast<int32_t>(num_heads_kv),
      static_cast<int32_t>(head_dim_qk),
      static_cast<int32_t>(head_dim_vo),
      static_cast<int32_t>(max_seqlen_q),
      static_cast<int32_t>(max_seqlen_k),
      static_cast<int32_t>(window_size_left),
      static_cast<int32_t>(window_size_right),
      cu_seqlens_q,
      seqused_k,
      num_splits,
      metadata);
  return metadata;
}

torch::Tensor fa3_decode_musa(const torch::Tensor& query,
                              const torch::Tensor& k_cache,
                              const torch::Tensor& v_cache,
                              const torch::Tensor& cu_seqlens_q,
                              const torch::Tensor& seqused_k,
                              const torch::Tensor& page_table,
                              const torch::Tensor& scheduler_metadata,
                              int64_t max_seqlen_q,
                              int64_t window_left,
                              int64_t window_right,
                              double sm_scale,
                              int64_t num_splits,
                              torch::Tensor& output,
                              torch::Tensor& output_lse) {
  CHECK_EQ(query.dim(), 3) << "fa3_decode expects [tokens, heads, dim]";
  CHECK(output.defined()) << "fa3_decode requires a persistent output buffer";
  CHECK(output_lse.defined()) << "fa3_decode requires a persistent LSE buffer";
  CHECK_EQ(output.sizes(), query.sizes())
      << "fa3_decode output shape must match query";
  CHECK_EQ(output.scalar_type(), query.scalar_type());
  CHECK_EQ(output.device(), query.device());
  CHECK(output.is_contiguous()) << "fa3_decode output must be contiguous";
  CHECK_EQ(output_lse.dim(), 2);
  CHECK_EQ(output_lse.size(0), query.size(1));
  CHECK_EQ(output_lse.size(1), query.size(0));
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32);
  CHECK_EQ(output_lse.device(), query.device());
  CHECK(output_lse.is_contiguous()) << "fa3_decode LSE must be contiguous";
  torch::Tensor query_arg = query;
  xllm::kernel::musa::fa3_decode(query_arg,
                                 k_cache,
                                 v_cache,
                                 cu_seqlens_q,
                                 seqused_k,
                                 page_table,
                                 scheduler_metadata,
                                 max_seqlen_q,
                                 window_left,
                                 window_right,
                                 sm_scale,
                                 num_splits,
                                 output,
                                 output_lse);
  return output;
}

torch::Tensor fa3_prefill_musa(const torch::Tensor& query,
                               const torch::Tensor& key,
                               const torch::Tensor& value,
                               const torch::Tensor& cu_seqlens_q,
                               const torch::Tensor& cu_seqlens_k,
                               int64_t max_seqlen_q,
                               int64_t max_seqlen_k,
                               int64_t window_left,
                               int64_t window_right,
                               double sm_scale,
                               torch::Tensor& output,
                               torch::Tensor& output_lse) {
  CHECK_EQ(query.dim(), 3) << "fa3_prefill expects [tokens, heads, dim]";
  CHECK(output.defined()) << "fa3_prefill requires a persistent output buffer";
  CHECK(output_lse.defined()) << "fa3_prefill requires a persistent LSE buffer";
  CHECK_EQ(output.sizes(), query.sizes())
      << "fa3_prefill output shape must match query";
  CHECK_EQ(output.scalar_type(), query.scalar_type());
  CHECK_EQ(output.device(), query.device());
  CHECK(output.is_contiguous()) << "fa3_prefill output must be contiguous";
  CHECK_EQ(output_lse.dim(), 2);
  CHECK_EQ(output_lse.size(0), query.size(1));
  CHECK_EQ(output_lse.size(1), query.size(0));
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32);
  CHECK_EQ(output_lse.device(), query.device());
  CHECK(output_lse.is_contiguous()) << "fa3_prefill LSE must be contiguous";
  torch::Tensor output_arg = output;
  torch::Tensor output_lse_arg = output_lse;
  xllm::kernel::musa::fa3_prefill(query,
                                  key,
                                  value,
                                  cu_seqlens_q,
                                  cu_seqlens_k,
                                  max_seqlen_q,
                                  max_seqlen_k,
                                  window_left,
                                  window_right,
                                  sm_scale,
                                  output_arg,
                                  output_lse_arg);
  return output;
}

torch::Tensor block_fp8_linear_musa(const torch::Tensor& input,
                                    const torch::Tensor& weight,
                                    const torch::Tensor& weight_scale_inv,
                                    int64_t block_n,
                                    int64_t block_k,
                                    torch::Tensor& output) {
  CHECK_EQ(input.dim(), 2) << "block_fp8_linear expects a 2D activation";
  CHECK_EQ(weight.dim(), 2) << "block_fp8_linear expects a 2D weight";
  const int64_t k = input.size(1);
  CHECK_EQ(k % block_k, 0) << "block_fp8_linear requires K % block_k == 0";
  CHECK_GT(block_n, 0);
  CHECK_EQ(weight.size(1), k);
  torch::Tensor input_2d = input;
  if (!input.is_contiguous()) {
    input_2d =
        grow_only_2d_keyed(g_ptq_in_by_cols, input.size(0), k, input.options());
    input_2d.copy_(input);
  }
  torch::Tensor quantized =
      grow_only_2d_keyed(g_ptq_q_by_cols,
                         input_2d.size(0),
                         k,
                         input_2d.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor input_scale =
      grow_only_2d_keyed(g_ptq_s_by_cols,
                         input_2d.size(0),
                         k / block_k,
                         input_2d.options().dtype(torch::kFloat32));
  xllm::kernel::musa::per_token_group_quant_fp8_out(
      input_2d, /*group_size=*/block_k, quantized, input_scale);
  xllm::kernel::musa::Fp8BlockMatmulParams params;
  params.a = quantized;
  params.b = weight;
  params.a_scale = input_scale;
  params.b_scale = weight_scale_inv;
  params.output_dtype = (input.scalar_type() == torch::kFloat16)
                            ? torch::kFloat16
                            : torch::kBFloat16;
  if (output.defined()) {
    CHECK_EQ(output.size(0), input.size(0));
    CHECK_EQ(output.size(1), weight.size(0));
    params.output = output;
  }
  return xllm::kernel::musa::fp8_block_matmul(params);
}

torch::Tensor block_fp8_linear_quantized_musa(
    const torch::Tensor& input,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale_inv,
    int64_t block_n,
    int64_t block_k,
    torch::Tensor& output) {
  CHECK_EQ(input.dim(), 2) << "block_fp8_linear_quantized expects 2D FP8 input";
  CHECK_EQ(weight.dim(), 2) << "block_fp8_linear_quantized expects a 2D weight";
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK(input.is_contiguous());
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK(input_scale.is_contiguous());
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_GT(block_k, 0);
  CHECK_EQ(input.size(1) % block_k, 0)
      << "block_fp8_linear_quantized requires K % block_k == 0";
  CHECK_EQ(input_scale.size(1), input.size(1) / block_k);
  CHECK_GT(block_n, 0);
  CHECK_EQ(weight.size(1), input.size(1));
  CHECK(output.defined()) << "block_fp8_linear_quantized requires an output";
  CHECK_EQ(output.size(0), input.size(0));
  CHECK_EQ(output.size(1), weight.size(0));
  xllm::kernel::musa::Fp8BlockMatmulParams params;
  params.a = input;
  params.b = weight;
  params.a_scale = input_scale;
  params.b_scale = weight_scale_inv;
  params.output_dtype = output.scalar_type();
  params.output = output;
  return xllm::kernel::musa::fp8_block_matmul(params);
}

std::tuple<torch::Tensor, torch::Tensor> fused_swiglu_quant_fp8_musa(
    const torch::Tensor& input,
    int64_t group_size) {
  CHECK_EQ(input.dim(), 2) << "fused_swiglu_quant_fp8 expects [tokens, 2N]";
  CHECK_GT(group_size, 0);
  CHECK_EQ(input.size(1) % (2 * group_size), 0);
  const int64_t num_rows = input.size(0);
  const int64_t intermediate_size = input.size(1) / 2;
  torch::Tensor output_q =
      grow_only_2d(g_fused_swiglu_q,
                   num_rows,
                   intermediate_size,
                   input.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor output_s = grow_only_2d(g_fused_swiglu_s,
                                        num_rows,
                                        intermediate_size / group_size,
                                        input.options().dtype(torch::kFloat32));
  xllm::kernel::musa::fused_swiglu_quant_fp8_out(
      input, group_size, output_q, output_s);
  return std::make_tuple(output_q, output_s);
}

std::tuple<torch::Tensor, torch::Tensor> per_token_group_quant_fp8_musa(
    const torch::Tensor& input,
    int64_t group_size) {
  CHECK_EQ(input.dim(), 2) << "per_token_group_quant_fp8 expects [tokens, K]";
  CHECK_GT(group_size, 0);
  CHECK_EQ(input.size(1) % group_size, 0);
  torch::Tensor packed = input;
  if (!input.is_contiguous()) {
    packed = grow_only_2d_keyed(
        g_ptq_in_by_cols, input.size(0), input.size(1), input.options());
    packed.copy_(input);
  }
  torch::Tensor output_q =
      grow_only_2d_keyed(g_ptq_q_by_cols,
                         packed.size(0),
                         packed.size(1),
                         packed.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor output_s =
      grow_only_2d_keyed(g_ptq_s_by_cols,
                         packed.size(0),
                         packed.size(1) / group_size,
                         packed.options().dtype(torch::kFloat32));
  xllm::kernel::musa::per_token_group_quant_fp8_out(
      packed, group_size, output_q, output_s);
  return std::make_tuple(output_q, output_s);
}

torch::Tensor causal_conv1d_prefill_musa(const torch::Tensor& value,
                                         const torch::Tensor& weight,
                                         torch::Tensor& conv_state,
                                         const torch::Tensor& state_indices,
                                         const torch::Tensor& has_initial_state,
                                         const torch::Tensor& query_start_loc) {
  // Grow-only workspace: empty_like during MUSA graph capture aborts with
  // "operation not permitted when stream is capturing".
  CHECK_EQ(value.dim(), 2)
      << "causal_conv1d_prefill expects value [tokens, dim]";
  // in_proj_qkvz.split() yields a non-contiguous [T, conv_dim] view. The
  // token-major kernel matches C++ reshape_qkvz_unpad and requires packed
  // [T, D]. Pack into a grow-only buffer so capture does not record
  // contiguous() allocations.
  torch::Tensor packed = value;
  if (!value.is_contiguous()) {
    packed = grow_only_2d(g_causal_conv1d_prefill_in,
                          value.size(0),
                          value.size(1),
                          value.options());
    packed.copy_(value);
  }
  torch::Tensor output = grow_only_2d(g_causal_conv1d_prefill_out,
                                      packed.size(0),
                                      packed.size(1),
                                      packed.options());
  static const bool use_token_major = [] {
    const char* env = std::getenv("XLLM_TOKEN_MAJOR_PREFILL_CONV");
    return env == nullptr || std::string(env) != "0";
  }();
  if (use_token_major && packed.is_contiguous() && weight.dim() == 2 &&
      weight.size(1) == 4) {
    xllm::kernel::musa::causal_conv1d_fwd_token_major(packed,
                                                      weight,
                                                      output,
                                                      std::nullopt,
                                                      conv_state,
                                                      query_start_loc,
                                                      state_indices,
                                                      has_initial_state,
                                                      /*silu_activation=*/true,
                                                      /*pad_slot_id=*/-1);
    return output;
  }
  return xllm::kernel::musa::causal_conv1d_prefill(packed,
                                                   weight,
                                                   conv_state,
                                                   std::nullopt,
                                                   query_start_loc,
                                                   state_indices,
                                                   has_initial_state,
                                                   /*silu_activation=*/true);
}

torch::Tensor causal_conv1d_decode_musa(const torch::Tensor& value,
                                        const torch::Tensor& weight,
                                        torch::Tensor& conv_state,
                                        const torch::Tensor& state_indices) {
  // Grow-only workspace: empty_like during MUSA graph capture aborts with
  // "operation not permitted when stream is capturing". Capture warmup
  // allocates; replay reuses the same address.
  CHECK_EQ(value.dim(), 2)
      << "causal_conv1d_decode expects value [tokens, dim]";
  torch::Tensor output = grow_only_2d(g_causal_conv1d_decode_out,
                                      value.size(0),
                                      value.size(1),
                                      value.options());
  xllm::kernel::musa::causal_conv1d_decode_fused(value,
                                                 weight,
                                                 std::nullopt,
                                                 conv_state,
                                                 state_indices,
                                                 output,
                                                 /*pad_slot_id=*/-1,
                                                 /*silu_activation=*/true);
  return output;
}

torch::Tensor causal_conv1d_mtp_verify_musa(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& weight,
    torch::Tensor& conv_state,
    const torch::Tensor& cache_indices,
    const torch::Tensor& num_accepted_tokens,
    int64_t num_key_heads,
    int64_t key_head_dim,
    int64_t value_head_dim) {
  CHECK_EQ(mixed_qkv.dim(), 2) << "MTP conv verify expects mixed_qkv [B*T, D]";
  CHECK_EQ(cache_indices.dim(), 1);
  CHECK_EQ(num_accepted_tokens.dim(), 1);
  const int64_t batch_size = cache_indices.numel();
  CHECK_GT(batch_size, 0);
  CHECK_EQ(num_accepted_tokens.numel(), batch_size);
  const int64_t num_tokens = mixed_qkv.size(0);
  CHECK_EQ(num_tokens % batch_size, 0)
      << "MTP conv verify tokens must be dense per sequence";
  const int64_t sequence_length = num_tokens / batch_size;
  CHECK_GT(sequence_length, 0);
  CHECK_GT(num_key_heads, 0);
  CHECK_GT(key_head_dim, 0);
  CHECK_GT(value_head_dim, 0);
  const int64_t key_dim = num_key_heads * key_head_dim;
  const int64_t mixed_dim = mixed_qkv.size(1);
  CHECK_GE(mixed_dim, 2 * key_dim);
  CHECK_EQ((mixed_dim - 2 * key_dim) % value_head_dim, 0);
  const int64_t value_dim = mixed_dim - 2 * key_dim;
  const int64_t num_value_heads = value_dim / value_head_dim;
  CHECK_GT(num_value_heads, 0);

  torch::Tensor x = mixed_qkv.view({batch_size, sequence_length, mixed_dim})
                        .transpose(/*dim0=*/1, /*dim1=*/2)
                        .contiguous();
  torch::Tensor query =
      torch::empty({batch_size, sequence_length, num_key_heads, key_head_dim},
                   mixed_qkv.options());
  torch::Tensor key = torch::empty_like(query);
  torch::Tensor value = torch::empty(
      {batch_size, sequence_length, num_value_heads, value_head_dim},
      mixed_qkv.options());
  torch::Tensor intermediate;
  torch::Tensor cache_indices_i32 =
      cache_indices.to(torch::kInt32).contiguous();
  xllm::kernel::musa::causal_conv1d_mtp_verify(x,
                                               weight.contiguous(),
                                               conv_state,
                                               cache_indices_i32,
                                               num_accepted_tokens.contiguous(),
                                               query,
                                               key,
                                               value,
                                               intermediate,
                                               /*silu_activation=*/true,
                                               /*write_superstate=*/true);
  return torch::cat({query.reshape({num_tokens, key_dim}),
                     key.reshape({num_tokens, key_dim}),
                     value.reshape({num_tokens, value_dim})},
                    /*dim=*/1);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
fused_gdn_prefill_post_conv_musa(const torch::Tensor& mixed_qkv,
                                 const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& a_log,
                                 const torch::Tensor& dt_bias,
                                 int64_t num_key_heads,
                                 int64_t key_head_dim,
                                 int64_t value_head_dim) {
  CHECK_EQ(mixed_qkv.dim(), 2);
  const int64_t num_tokens = mixed_qkv.size(0);
  const int64_t num_value_heads = a_log.numel();
  const int64_t key_dim = num_key_heads * key_head_dim;
  const int64_t value_dim = num_value_heads * value_head_dim;
  CHECK_EQ(mixed_qkv.size(1), 2 * key_dim + value_dim);
  // C++ process_mixed_qkv is a strided view-split. Copying q/k/v into
  // 64-aligned grow-only buffers every layer was extra D2D in the Python
  // graph. Keep the copy path as an opt-in for non-contiguous conv output
  // or XLLM_PYTHON_GDN_ALIGN_KKT=1.
  static const bool align_kkt =
      util::get_bool_env("XLLM_PYTHON_GDN_ALIGN_KKT", /*defaultValue=*/false);
  const int64_t aligned_tokens =
      align_kkt ? align_gdn_prefill_tokens(num_tokens) : num_tokens;
  const int64_t pad_tokens = aligned_tokens - num_tokens;
  const bool use_strided_views = pad_tokens == 0 && mixed_qkv.is_contiguous();
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  if (use_strided_views) {
    query = mixed_qkv.narrow(/*dim=*/1, /*start=*/0, /*length=*/key_dim)
                .view({num_tokens, num_key_heads, key_head_dim});
    key = mixed_qkv.narrow(/*dim=*/1, /*start=*/key_dim, /*length=*/key_dim)
              .view({num_tokens, num_key_heads, key_head_dim});
    value =
        mixed_qkv.narrow(/*dim=*/1, /*start=*/2 * key_dim, /*length=*/value_dim)
            .view({num_tokens, num_value_heads, value_head_dim});
  } else {
    std::vector<torch::Tensor> split =
        mixed_qkv.split({key_dim, key_dim, value_dim}, /*dim=*/1);
    query = grow_only_3d(g_gdn_prefill_query,
                         aligned_tokens,
                         num_key_heads,
                         key_head_dim,
                         mixed_qkv.options());
    key = grow_only_3d(g_gdn_prefill_key,
                       aligned_tokens,
                       num_key_heads,
                       key_head_dim,
                       mixed_qkv.options());
    value = grow_only_3d(g_gdn_prefill_value,
                         aligned_tokens,
                         num_value_heads,
                         value_head_dim,
                         mixed_qkv.options());
    query.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_tokens)
        .copy_(split[0].view({num_tokens, num_key_heads, key_head_dim}));
    key.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_tokens)
        .copy_(split[1].view({num_tokens, num_key_heads, key_head_dim}));
    value.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_tokens)
        .copy_(split[2].view({num_tokens, num_value_heads, value_head_dim}));
    if (pad_tokens > 0) {
      query.narrow(/*dim=*/0, /*start=*/num_tokens, /*length=*/pad_tokens)
          .zero_();
      key.narrow(/*dim=*/0, /*start=*/num_tokens, /*length=*/pad_tokens)
          .zero_();
      value.narrow(/*dim=*/0, /*start=*/num_tokens, /*length=*/pad_tokens)
          .zero_();
      LOG_FIRST_N(INFO, 1) << "[FusedGdnPrefill] aligned KKT tokens "
                           << num_tokens << " -> " << aligned_tokens;
    }
  }
  torch::Tensor gate_a = a.reshape({num_tokens, num_value_heads}).contiguous();
  torch::Tensor gate_b = b.reshape({num_tokens, num_value_heads}).contiguous();
  torch::Tensor gate = grow_only_2d(g_gdn_prefill_gate,
                                    aligned_tokens,
                                    num_value_heads,
                                    gate_a.options().dtype(torch::kFloat32));
  torch::Tensor beta = grow_only_2d(g_gdn_prefill_beta,
                                    aligned_tokens,
                                    num_value_heads,
                                    gate_a.options().dtype(torch::kFloat32));
  torch::Tensor gate_live =
      pad_tokens > 0
          ? gate.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_tokens)
          : gate;
  torch::Tensor beta_live =
      pad_tokens > 0
          ? beta.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_tokens)
          : beta;
  xllm::kernel::musa::gdn_gating(gate_a,
                                 gate_b,
                                 a_log,
                                 dt_bias,
                                 /*sp_beta=*/1.0,
                                 /*threshold=*/20.0,
                                 gate_live,
                                 beta_live);
  if (pad_tokens > 0) {
    gate.narrow(/*dim=*/0, /*start=*/num_tokens, /*length=*/pad_tokens).zero_();
    beta.narrow(/*dim=*/0, /*start=*/num_tokens, /*length=*/pad_tokens).zero_();
  }
  return std::make_tuple(query, key, value, gate, beta);
}

std::tuple<torch::Tensor, torch::Tensor> chunk_gated_delta_rule_musa(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& gate,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens,
    const std::string& backend) {
  CHECK(backend == "mate" || backend == "chunk" || backend.empty())
      << "unsupported GDN prefill backend: " << backend;
  torch::Tensor query_4d = query.dim() == 3 ? query.unsqueeze(0) : query;
  torch::Tensor key_4d = key.dim() == 3 ? key.unsqueeze(0) : key;
  torch::Tensor value_4d = value.dim() == 3 ? value.unsqueeze(0) : value;
  torch::Tensor gate_3d = gate.dim() == 2 ? gate.unsqueeze(0) : gate;
  torch::Tensor beta_3d = beta.dim() == 2 ? beta.unsqueeze(0) : beta;
  if (backend != "chunk") {
    CHECK_EQ(query_4d.dim(), 4);
    CHECK_EQ(value_4d.dim(), 4);
    CHECK_EQ(cu_seqlens.dim(), 1);
    CHECK_GE(cu_seqlens.numel(), 2);
    torch::Tensor init_state = initial_state;
    if (init_state.dim() == 3) {
      init_state = init_state.unsqueeze(0);
    }
    CHECK_EQ(init_state.dim(), 4)
        << "mate GDN prefill initial_state must be [B, H, V, K]";
    torch::Tensor cu_int32 = cu_seqlens;
    if (cu_int32.scalar_type() != torch::kInt32 || !cu_int32.is_contiguous()) {
      cu_int32 = cu_int32.to(torch::kInt32).contiguous();
    }
    // Allocate grow-only persistent buffers on eager warmup so capture
    // reuses the same addresses. Native C++ piecewise uses the matmul
    // pool instead and never enters this op.
    torch::Tensor persistent_output = grow_only_4d(g_gdn_mate_output,
                                                   /*dim0=*/1,
                                                   query_4d.size(1),
                                                   value_4d.size(2),
                                                   value_4d.size(3),
                                                   value_4d.options());
    torch::Tensor persistent_final_state =
        grow_only_like(g_gdn_mate_final_state, init_state);
    torch::Tensor persistent_kkt = grow_only_4d(g_gdn_mate_kkt,
                                                /*dim0=*/1,
                                                query_4d.size(1),
                                                value_4d.size(2),
                                                kGdnChunkSize,
                                                key_4d.options());
    if (xllm::runtime::musa::GlobalCaptureInstance::get_instance()
            .is_capturing()) {
      // Match C++ piecewise: Mate stays off the graph. Do not enable
      // MateGdnPythonGraphVarlenScope here 鈥?that path is only for the
      // legacy full-model torch.musa.graph capture.
      const float scale =
          1.0f / std::sqrt(static_cast<float>(query_4d.size(-1)));
      xllm::kernel::musa::AttentionRunner runner;
      runner.run_gdn_prefill_capture(query_4d,
                                     key_4d,
                                     value_4d,
                                     gate_3d,
                                     beta_3d,
                                     init_state,
                                     cu_int32,
                                     persistent_output,
                                     persistent_final_state,
                                     persistent_kkt,
                                     /*ssm_cache=*/torch::Tensor(),
                                     /*state_indices=*/torch::Tensor(),
                                     /*scale=*/scale);
      xllm::runtime::musa::GlobalCaptureInstance::get_instance()
          .register_attention_runner(std::move(runner));
      torch::Tensor output = persistent_output;
      if (query.dim() == 3 && output.dim() == 4 && output.size(0) == 1) {
        output = output.squeeze(0);
      }
      return std::make_tuple(output, persistent_final_state);
    }
    xllm::kernel::musa::MateGatedDeltaRulePrefillParams mate_params;
    mate_params.q = query_4d;
    mate_params.k = key_4d;
    mate_params.v = value_4d;
    mate_params.g = gate_3d;
    mate_params.beta = beta_3d;
    mate_params.initial_state = init_state;
    mate_params.cu_seqlens = cu_int32;
    mate_params.output = persistent_output;
    mate_params.final_state = persistent_final_state;
    mate_params.kkt_output = persistent_kkt;
    mate_params.output_final_state = true;
    mate_params.use_qk_l2norm_in_kernel = true;
    mate_params.allow_inplace_qk_l2norm = true;
    // torch.musa.graph capture-safe C=1 scratch-reuse varlen. Padded Mate
    // on this op is slower for C=1/384 (replay_gpu 81ms -> 124ms).
    xllm::kernel::musa::MateGdnPythonGraphVarlenScope python_varlen_scope(
        /*enabled=*/true);
    std::pair<torch::Tensor, torch::Tensor> mate_result =
        xllm::kernel::musa::mate_gated_delta_rule_prefill(mate_params);
    torch::Tensor output = mate_result.first;
    torch::Tensor final_state = mate_result.second;
    if (query.dim() == 3 && output.dim() == 4 && output.size(0) == 1) {
      output = output.squeeze(0);
    }
    return std::make_tuple(output, final_state);
  }
  xllm::kernel::ChunkGatedDeltaRuleParams params;
  params.q = query_4d;
  params.k = key_4d;
  params.v = value_4d;
  params.g = gate_3d;
  params.beta = beta_3d;
  params.initial_state = initial_state;
  params.output_final_state = true;
  params.cu_seqlens = cu_seqlens;
  params.head_first = false;
  params.use_qk_l2norm_in_kernel = true;
  std::pair<torch::Tensor, torch::Tensor> result =
      xllm::kernel::chunk_gated_delta_rule(params);
  torch::Tensor output = result.first;
  torch::Tensor final_state = result.second;
  if (query.dim() == 3 && output.dim() == 4 && output.size(0) == 1) {
    output = output.squeeze(0);
  }
  return std::make_tuple(output, final_state);
}

torch::Tensor fused_recurrent_gated_delta_rule_packed_decode_musa(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& initial_state,
    const torch::Tensor& state_indices,
    double scale,
    int64_t num_k_heads,
    int64_t key_head_dim,
    int64_t value_head_dim) {
  CHECK_EQ(mixed_qkv.dim(), 2)
      << "fused GDN decode expects mixed_qkv [tokens, dim]";
  torch::Tensor decode_output = grow_only_3d(g_fused_gdn_decode_out,
                                             mixed_qkv.size(0),
                                             a_log.numel(),
                                             value_head_dim,
                                             mixed_qkv.options());
  xllm::kernel::musa::MateGatedDeltaRuleDecodeParams params;
  fill_gdn_decode_params(params,
                         mixed_qkv,
                         a,
                         b,
                         a_log,
                         dt_bias,
                         initial_state,
                         state_indices,
                         scale,
                         num_k_heads,
                         key_head_dim,
                         value_head_dim,
                         decode_output);
  return xllm::kernel::musa::fused_gated_delta_rule_decode(params);
}

torch::Tensor mate_gated_delta_rule_decode_musa(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& initial_state,
    const torch::Tensor& state_indices,
    double scale,
    int64_t num_k_heads,
    int64_t key_head_dim,
    int64_t value_head_dim) {
  CHECK_EQ(mixed_qkv.dim(), 2)
      << "mate GDN decode expects mixed_qkv [tokens, dim]";
  torch::Tensor decode_output = grow_only_3d(g_mate_gdn_decode_out,
                                             mixed_qkv.size(0),
                                             a_log.numel(),
                                             value_head_dim,
                                             mixed_qkv.options());
  xllm::kernel::musa::MateGatedDeltaRuleDecodeParams params;
  fill_gdn_decode_params(params,
                         mixed_qkv,
                         a,
                         b,
                         a_log,
                         dt_bias,
                         initial_state,
                         state_indices,
                         scale,
                         num_k_heads,
                         key_head_dim,
                         value_head_dim,
                         decode_output);
  return xllm::kernel::musa::mate_gated_delta_rule_decode(params);
}

torch::Tensor fused_gdn_mtp_checkpoint_musa(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const torch::Tensor& logical_state_indices,
    const torch::Tensor& num_accepted_tokens,
    int64_t checkpoint_stride,
    double scale,
    int64_t num_key_heads,
    int64_t key_head_dim,
    int64_t value_head_dim) {
  CHECK_EQ(mixed_qkv.dim(), 2)
      << "fused GDN MTP checkpoint expects mixed_qkv [B*T, D]";
  CHECK_EQ(logical_state_indices.dim(), 1);
  CHECK_EQ(num_accepted_tokens.dim(), 1);
  const int64_t batch_size = logical_state_indices.numel();
  CHECK_GT(batch_size, 0);
  CHECK_EQ(num_accepted_tokens.numel(), batch_size);
  const int64_t num_tokens = mixed_qkv.size(0);
  CHECK_EQ(num_tokens % batch_size, 0);
  const int64_t sequence_length = num_tokens / batch_size;
  CHECK_EQ(checkpoint_stride, sequence_length)
      << "fused GDN MTP checkpoint requires one cache slot per verify token";

  const int64_t num_value_heads = a_log.numel();
  const int64_t key_dim = num_key_heads * key_head_dim;
  const int64_t value_dim = num_value_heads * value_head_dim;
  CHECK_EQ(mixed_qkv.size(1), 2 * key_dim + value_dim);
  CHECK_EQ(a.numel(), num_tokens * num_value_heads);
  CHECK_EQ(b.numel(), num_tokens * num_value_heads);
  CHECK(xllm::kernel::musa::mate_gdn_mtp_checkpoint_module_available(
      num_key_heads, num_value_heads, mixed_qkv.scalar_type(), sequence_length))
      << "MATE GDN MTP checkpoint artifact is unavailable for Hq="
      << num_key_heads << ", Hv=" << num_value_heads
      << ", T=" << sequence_length;

  std::vector<torch::Tensor> split =
      mixed_qkv.split({key_dim, key_dim, value_dim}, /*dim=*/1);
  torch::Tensor query =
      split[0]
          .view({batch_size, sequence_length, num_key_heads, key_head_dim})
          .contiguous();
  torch::Tensor key =
      split[1]
          .view({batch_size, sequence_length, num_key_heads, key_head_dim})
          .contiguous();
  torch::Tensor value =
      split[2]
          .view({batch_size, sequence_length, num_value_heads, value_head_dim})
          .contiguous();
  torch::Tensor gate_a =
      a.view({batch_size, sequence_length, num_value_heads}).contiguous();
  torch::Tensor gate_b =
      b.view({batch_size, sequence_length, num_value_heads}).contiguous();
  torch::Tensor a_log_f32 = a_log.to(torch::kFloat32).contiguous();
  torch::Tensor dt_bias_f32 = dt_bias.to(torch::kFloat32).contiguous();
  torch::Tensor logical_indices_i32 =
      logical_state_indices.to(mixed_qkv.device(), torch::kInt32).contiguous();
  torch::Tensor accepted_i32 =
      num_accepted_tokens.to(mixed_qkv.device(), torch::kInt32)
          .clamp(/*min=*/1, /*max=*/sequence_length)
          .contiguous();
  torch::Tensor base_indices = logical_indices_i32 * checkpoint_stride;
  torch::Tensor initial_state_indices =
      (base_indices + accepted_i32 - 1).contiguous();
  torch::Tensor step_offsets =
      torch::arange(sequence_length, base_indices.options());
  torch::Tensor checkpoint_state_indices =
      (base_indices.unsqueeze(/*dim=*/1) + step_offsets).contiguous();
  torch::Tensor output = torch::empty_like(value);

  xllm::kernel::musa::MateGatedDeltaRuleMtpParams params;
  params.q = query;
  params.k = key;
  params.v = value;
  params.A_log = a_log_f32;
  params.a = gate_a;
  params.dt_bias = dt_bias_f32;
  params.b = gate_b;
  params.state = ssm_state;
  params.state_indices = initial_state_indices;
  params.checkpoint_state_indices = checkpoint_state_indices;
  params.output = output;
  params.num_k_heads = num_key_heads;
  params.num_v_heads = num_value_heads;
  params.head_k_dim = key_head_dim;
  params.head_v_dim = value_head_dim;
  params.scale = scale;
  xllm::kernel::musa::mate_gated_delta_rule_mtp(params);
  return output.view({num_tokens, num_value_heads, value_head_dim});
}

torch::Tensor rms_norm_gated_musa(const torch::Tensor& value,
                                  const torch::Tensor& gate,
                                  const torch::Tensor& weight,
                                  double eps,
                                  torch::Tensor& output) {
  CHECK(output.defined()) << "rms_norm_gated requires an output buffer";
  CHECK_EQ(output.sizes(), value.sizes());
  xllm::kernel::musa::gated_rms_norm_fused(
      value, weight, gate, output, /*eps=*/eps);
  return output;
}

void python_prefill_piecewise_begin_musa(const torch::Tensor& tokens) {
  CHECK(tokens.defined()) << "piecewise begin requires a device tensor";
  CHECK(!g_piecewise_ffi_guard.has_value())
      << "Python piecewise capture already has an FFI stream guard";
  const torch::Device device = tokens.device();
  xllm::kernel::musa::sync_musa_ffi_stream(device);
  g_piecewise_ffi_guard.emplace(device, current_musa_stream_handle(device));
  const c10::musa::MempoolId_t pool = at::musa::graph_pool_handle();
  xllm::runtime::musa::GlobalCaptureInstance::get_instance().begin_capture(
      /*pool=*/pool);
}

std::tuple<int64_t, int64_t, int64_t> python_prefill_piecewise_end_musa(
    const torch::Tensor& tokens) {
  CHECK(tokens.defined()) << "piecewise end requires a device tensor";
  std::unique_ptr<xllm::runtime::musa::PiecewiseGraphs> graphs =
      xllm::runtime::musa::GlobalCaptureInstance::get_instance().end_capture();
  g_piecewise_ffi_guard.reset();
  CHECK(graphs != nullptr && !graphs->empty())
      << "Python piecewise capture produced no graphs";
  const int64_t num_graphs = static_cast<int64_t>(graphs->num_graphs());
  const int64_t num_runners = static_cast<int64_t>(graphs->num_runners());
  const int64_t handle = g_piecewise_next_handle;
  g_piecewise_next_handle += 1;
  g_piecewise_graphs.emplace(handle, std::move(graphs));
  return {handle, num_graphs, num_runners};
}

void python_prefill_piecewise_replay_musa(
    const torch::Tensor& tokens,
    int64_t handle,
    const torch::Tensor& gdn_cu_seq_lens,
    const torch::Tensor& gdn_kkt_cu_seq_lens,
    int64_t actual_num_tokens,
    int64_t gdn_kkt_num_tokens) {
  CHECK(tokens.defined()) << "piecewise replay requires a device tensor";
  auto graph_it = g_piecewise_graphs.find(handle);
  CHECK(graph_it != g_piecewise_graphs.end())
      << "invalid Python piecewise handle: " << handle;
  CHECK_GT(actual_num_tokens, 0);

  xllm::kernel::musa::AttentionReplayParams replay_params;
  replay_params.actual_num_tokens = static_cast<uint32_t>(actual_num_tokens);
  if (graph_it->second->has_gdn_prefill_runner()) {
    CHECK(gdn_cu_seq_lens.defined() && gdn_cu_seq_lens.is_contiguous())
        << "GDN cu_seqlens must be a contiguous device tensor";
    CHECK_EQ(gdn_cu_seq_lens.scalar_type(), torch::kInt32);
    CHECK(gdn_kkt_cu_seq_lens.defined() && gdn_kkt_cu_seq_lens.is_contiguous())
        << "GDN KKT cu_seqlens must be a contiguous device tensor";
    CHECK_EQ(gdn_kkt_cu_seq_lens.scalar_type(), torch::kInt32);
    CHECK_EQ(gdn_kkt_cu_seq_lens.size(0), gdn_cu_seq_lens.size(0));
    CHECK_GE(gdn_kkt_num_tokens, actual_num_tokens);
    CHECK_EQ(gdn_kkt_num_tokens % kGdnChunkSize, 0);
    replay_params.gdn_kkt_num_tokens =
        static_cast<uint32_t>(gdn_kkt_num_tokens);
    replay_params.q_cu_seq_lens = gdn_cu_seq_lens;
    replay_params.gdn_cu_seq_lens = gdn_cu_seq_lens;
    replay_params.gdn_kkt_cu_seq_lens = gdn_kkt_cu_seq_lens;
    replay_params.q_cu_seq_lens_host.clear();
    replay_params.q_cu_seq_lens_host.reserve(2);
    replay_params.q_cu_seq_lens_host.emplace_back(0);
    replay_params.q_cu_seq_lens_host.emplace_back(
        static_cast<int32_t>(actual_num_tokens));
  }

  const torch::Device device = tokens.device();
  xllm::kernel::musa::MusaTvmffiStreamOverrideGuard ffi_guard(
      device, current_musa_stream_handle(device));
  graph_it->second->replay(replay_params);
}

}  // namespace

void ensure_xllm_ops_registered() {
  // Intentionally empty. Referencing this symbol keeps the object file (and its
  // TORCH_LIBRARY static initializers below) from being stripped by the linker.
}

void register_xllm_ops_musa(torch::Library& library) {
  library.impl("rms_norm", TORCH_FN(rms_norm_musa));
  library.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm_musa));
  library.impl("gemma_rms_norm", TORCH_FN(gemma_rms_norm_musa));
  library.impl("fused_add_gemma_rms_norm",
               TORCH_FN(fused_add_gemma_rms_norm_musa));
  library.impl("mul_sigmoid_gate_inplace",
               TORCH_FN(mul_sigmoid_gate_inplace_musa));
  library.impl("silu_and_mul", TORCH_FN(silu_and_mul_musa));
  library.impl("fused_qk_norm_rope", TORCH_FN(fused_qk_norm_rope_musa));
  library.impl("reshape_paged_cache", TORCH_FN(reshape_paged_cache_musa));
  library.impl("update_decode_graph_metadata",
               TORCH_FN(update_decode_graph_metadata_musa));
  library.impl("update_fa3_graph_metadata",
               TORCH_FN(update_fa3_graph_metadata_musa));
  library.impl("fa3_decode_scheduler_metadata",
               TORCH_FN(fa3_decode_scheduler_metadata_musa));
  library.impl("fa3_decode", TORCH_FN(fa3_decode_musa));
  library.impl("fa3_prefill", TORCH_FN(fa3_prefill_musa));
  library.impl("block_fp8_linear", TORCH_FN(block_fp8_linear_musa));
  library.impl("block_fp8_linear_quantized",
               TORCH_FN(block_fp8_linear_quantized_musa));
  library.impl("fused_swiglu_quant_fp8", TORCH_FN(fused_swiglu_quant_fp8_musa));
  library.impl("per_token_group_quant_fp8",
               TORCH_FN(per_token_group_quant_fp8_musa));
  library.impl("causal_conv1d_prefill", TORCH_FN(causal_conv1d_prefill_musa));
  library.impl("causal_conv1d_decode", TORCH_FN(causal_conv1d_decode_musa));
  library.impl("causal_conv1d_mtp_verify",
               TORCH_FN(causal_conv1d_mtp_verify_musa));
  library.impl("fused_gdn_prefill_post_conv",
               TORCH_FN(fused_gdn_prefill_post_conv_musa));
  library.impl("chunk_gated_delta_rule", TORCH_FN(chunk_gated_delta_rule_musa));
  library.impl("fused_recurrent_gated_delta_rule_packed_decode",
               TORCH_FN(fused_recurrent_gated_delta_rule_packed_decode_musa));
  library.impl("mate_gated_delta_rule_decode",
               TORCH_FN(mate_gated_delta_rule_decode_musa));
  library.impl("fused_gdn_mtp_checkpoint",
               TORCH_FN(fused_gdn_mtp_checkpoint_musa));
  library.impl("rms_norm_gated", TORCH_FN(rms_norm_gated_musa));
  library.impl("python_prefill_piecewise_begin",
               TORCH_FN(python_prefill_piecewise_begin_musa));
  library.impl("python_prefill_piecewise_end",
               TORCH_FN(python_prefill_piecewise_end_musa));
  library.impl("python_prefill_piecewise_replay",
               TORCH_FN(python_prefill_piecewise_replay_musa));
}

}  // namespace xllm

TORCH_LIBRARY(xllm_ops, m) {
  m.def(
      "rms_norm(Tensor input, Tensor weight, float eps, Tensor(a!) output) -> "
      "Tensor(a!)");
  m.def(
      "gemma_rms_norm(Tensor input, Tensor weight, float eps, Tensor(a!) "
      "output) -> Tensor(a!)");
  m.def(
      "fused_add_gemma_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor "
      "weight, float eps) -> (Tensor(a!), Tensor(b!))");
  m.def("mul_sigmoid_gate_inplace(Tensor(a!) out, Tensor gate) -> Tensor(a!)");
  m.def(
      "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor "
      "weight, "
      "float eps) -> (Tensor, Tensor)");
  m.def("silu_and_mul(Tensor input, Tensor(a!) output) -> Tensor(a!)");
  m.def(
      "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
      "int "
      "num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, bool interleaved, Tensor position_ids, "
      "int k_head_offset=0) -> Tensor");
  m.def(
      "reshape_paged_cache(Tensor slot_mapping, Tensor keys, Tensor values, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor");
  m.def(
      "update_decode_graph_metadata(Tensor tokens, Tensor positions, Tensor "
      "slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor "
      "paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, "
      "Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) "
      "dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) "
      "dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) "
      "dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor(a!)");
  m.def(
      "update_fa3_graph_metadata(Tensor kv_seq_lens, Tensor block_table, "
      "Tensor(a!) dst_kv_seq_lens, Tensor(b!) dst_block_table, "
      "int actual_batch_size) -> Tensor(a!)");
  m.def(
      "fa3_decode_scheduler_metadata(Tensor cu_seqlens_q, Tensor seqused_k, "
      "int batch_size, int num_heads_q, int num_heads_kv, int head_dim_qk, "
      "int head_dim_vo, int max_seqlen_q, int max_seqlen_k, int "
      "window_size_left, int window_size_right, int num_splits, "
      "Tensor(a!) scheduler_metadata) -> Tensor(a!)");
  m.def(
      "fa3_decode(Tensor query, Tensor k_cache, Tensor v_cache, Tensor "
      "cu_seqlens_q, Tensor seqused_k, Tensor page_table, Tensor "
      "scheduler_metadata, int max_seqlen_q, int window_left, int "
      "window_right, float sm_scale, int num_splits, Tensor(a!) output, "
      "Tensor(b!) output_lse) -> Tensor(a!)");
  m.def(
      "fa3_prefill(Tensor query, Tensor key, Tensor value, Tensor "
      "cu_seqlens_q, Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, "
      "int window_left, int window_right, float sm_scale, Tensor(a!) output, "
      "Tensor(b!) output_lse) -> Tensor(a!)");
  m.def(
      "block_fp8_linear(Tensor input, Tensor weight, Tensor "
      "weight_scale_inv, int block_n, int block_k, Tensor(a!) output) -> "
      "Tensor");
  m.def(
      "block_fp8_linear_quantized(Tensor input, Tensor input_scale, Tensor "
      "weight, Tensor weight_scale_inv, int block_n, int block_k, "
      "Tensor(a!) output) -> Tensor");
  m.def(
      "fused_swiglu_quant_fp8(Tensor input, int group_size) -> (Tensor, "
      "Tensor)");
  m.def(
      "per_token_group_quant_fp8(Tensor input, int group_size) -> (Tensor, "
      "Tensor)");
  m.def(
      "causal_conv1d_prefill(Tensor value, Tensor weight, Tensor(a!) "
      "conv_state, Tensor state_indices, Tensor has_initial_state, Tensor "
      "query_start_loc) -> Tensor");
  m.def(
      "causal_conv1d_decode(Tensor value, Tensor weight, Tensor(a!) "
      "conv_state, Tensor state_indices) -> Tensor");
  m.def(
      "causal_conv1d_mtp_verify(Tensor mixed_qkv, Tensor weight, Tensor(a!) "
      "conv_state, Tensor cache_indices, Tensor num_accepted_tokens, int "
      "num_key_heads, int key_head_dim, int value_head_dim) -> Tensor");
  m.def(
      "fused_gdn_prefill_post_conv(Tensor mixed_qkv, Tensor a, Tensor b, "
      "Tensor a_log, Tensor dt_bias, int num_key_heads, int key_head_dim, "
      "int value_head_dim) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "chunk_gated_delta_rule(Tensor q, Tensor k, Tensor v, Tensor g, "
      "Tensor beta, Tensor initial_state, Tensor cu_seqlens, str backend) -> "
      "(Tensor, Tensor)");
  m.def(
      "fused_recurrent_gated_delta_rule_packed_decode(Tensor mixed_qkv, "
      "Tensor a, Tensor b, Tensor a_log, Tensor dt_bias, Tensor(a!) "
      "initial_state, Tensor state_indices, float scale, int num_k_heads, "
      "int key_head_dim, int value_head_dim) -> Tensor");
  m.def(
      "mate_gated_delta_rule_decode(Tensor mixed_qkv, Tensor a, Tensor b, "
      "Tensor a_log, Tensor dt_bias, Tensor(a!) initial_state, Tensor "
      "state_indices, float scale, int num_k_heads, int key_head_dim, int "
      "value_head_dim) -> Tensor");
  m.def(
      "fused_gdn_mtp_checkpoint(Tensor mixed_qkv, Tensor a, Tensor b, Tensor "
      "a_log, Tensor dt_bias, Tensor(a!) ssm_state, Tensor "
      "logical_state_indices, Tensor num_accepted_tokens, int "
      "checkpoint_stride, float scale, int num_key_heads, int key_head_dim, "
      "int value_head_dim) -> Tensor");
  m.def(
      "rms_norm_gated(Tensor value, Tensor gate, Tensor weight, float eps, "
      "Tensor(a!) output) -> Tensor(a!)");
  m.def("python_prefill_piecewise_begin(Tensor tokens) -> ()");
  m.def("python_prefill_piecewise_end(Tensor tokens) -> (int, int, int)");
  m.def(
      "python_prefill_piecewise_replay(Tensor tokens, int handle, Tensor "
      "gdn_cu_seq_lens, Tensor gdn_kkt_cu_seq_lens, int actual_num_tokens, "
      "int gdn_kkt_num_tokens) -> ()");
}

// CUDA-compat headers have no DispatchKey::MUSA. Host tensors may still
// dispatch as CUDA, while libmusa_backend_init names PrivateUse1 "musa".
TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m) { xllm::register_xllm_ops_musa(m); }
TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  xllm::register_xllm_ops_musa(m);
}
