/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <string>

#include "core/kernels/musa/musa_aot_config.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kFa3TileM = 32;
constexpr int32_t kFa3TileN = 64;

const char* fa3_metadata_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "MUSA FA3 requires GQA ratio 6 or 8";
  return gqa_ratio == 6 ? aot::kFa3MetadataUriGqa6 : aot::kFa3MetadataUriGqa8;
}

std::string fa3_fwd_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "MUSA FA3 requires GQA ratio 6 or 8";
  return gqa_ratio == 6 ? aot::kFa3FwdUriGqa6 : aot::kFa3FwdUriGqa8;
}

std::string fa3_prefill_fwd_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "MUSA FA3 requires GQA ratio 6 or 8";
  return gqa_ratio == 6 ? aot::kFa3PrefillFwdUriGqa6
                        : aot::kFa3PrefillFwdUriGqa8;
}

ffi::Optional<ffi::Tensor> none_tensor() {
  return ffi::Optional<ffi::Tensor>();
}

ffi::Optional<int64_t> none_int() { return ffi::Optional<int64_t>(); }

void combine_fa3_split_k(const ffi::Any& fwd_result,
                         const torch::Tensor& cu_seqlens_q,
                         int64_t max_seqlen_q,
                         const torch::Tensor& scheduler_metadata,
                         torch::Tensor& output,
                         torch::Tensor& output_lse) {
  auto result_values = fwd_result.cast<ffi::Array<ffi::Any>>();
  CHECK_EQ(result_values.size(), 2)
      << "FA3 forward must return [accumulators, num_splits]";
  auto accumulators = result_values[0].cast<ffi::Array<ffi::Tensor>>();
  CHECK_EQ(accumulators.size(), 2)
      << "FA3 forward must return output and LSE accumulators";
  const int64_t num_splits = result_values[1].cast<int64_t>();
  CHECK_GT(num_splits, 0) << "FA3 forward returned invalid split count";
  // The pinned MATE heuristic caps the split count at 128.
  CHECK_LE(num_splits, 128)
      << "FA3 split count exceeds the supported MATE heuristic limit";
  const char* combine_uri =
      num_splits <= 16
          ? aot::kFa3FwdCombine16Uri
          : (num_splits <= 32 ? aot::kFa3FwdCombine32Uri
                              : (num_splits <= 64 ? aot::kFa3FwdCombine64Uri
                                                  : aot::kFa3FwdCombine128Uri));

  get_function(combine_uri, combine_uri)(to_ffi_tensor(cu_seqlens_q),
                                         none_tensor(),
                                         ffi::Optional<int64_t>(max_seqlen_q),
                                         to_ffi_tensor(output),
                                         to_ffi_tensor(output_lse),
                                         accumulators[0],
                                         accumulators[1],
                                         to_ffi_tensor(scheduler_metadata),
                                         num_splits);
}

}  // namespace

torch::Tensor fa3_decode_scheduler_metadata(const torch::Device& device,
                                            int32_t batch_size,
                                            int32_t num_heads_q,
                                            int32_t num_heads_kv,
                                            int32_t head_dim_qk,
                                            int32_t head_dim_vo,
                                            int32_t max_seqlen_q,
                                            int32_t max_seqlen_k,
                                            int32_t window_size_left,
                                            int32_t window_size_right,
                                            const torch::Tensor& cu_seqlens_q,
                                            const torch::Tensor& seqused_k) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_heads_kv, 0);
  CHECK_EQ(num_heads_q % num_heads_kv, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor metadata =
      torch::zeros({fa3_scheduler_metadata_size(batch_size)}, options);

  TvmffiStreamGuard stream_guard(device);

  const int64_t gqa_ratio = num_heads_q / num_heads_kv;
  const std::string uri = fa3_metadata_uri(gqa_ratio);
  // Current MATE ABI: cu_q, cu_k, seqused_q, seqused_k, cu_k_new;
  // four metadata arrays, each padded to a multiple of four.
  get_function(uri, uri)(static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(num_heads_q),
                         static_cast<int64_t>(num_heads_kv),
                         static_cast<int64_t>(head_dim_qk),
                         static_cast<int64_t>(head_dim_vo),
                         static_cast<int64_t>(max_seqlen_q),
                         static_cast<int64_t>(max_seqlen_k),
                         /*max_seqlen_k_new=*/static_cast<int64_t>(0),
                         to_ffi_tensor(cu_seqlens_q),
                         none_tensor(),
                         none_tensor(),
                         to_ffi_tensor(seqused_k),
                         none_tensor(),
                         static_cast<int64_t>(window_size_left),
                         static_cast<int64_t>(window_size_right),
                         none_tensor(),
                         to_ffi_tensor(metadata),
                         /*num_splits=*/static_cast<int64_t>(0),
                         static_cast<int64_t>(kFa3TileM),
                         static_cast<int64_t>(kFa3TileN),
                         /*mp_margin=*/static_cast<int64_t>(0));
  return metadata;
}

void fa3_decode(const torch::Tensor& query,
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
                torch::Tensor& output,
                torch::Tensor& output_lse) {
  CHECK(scheduler_metadata.defined())
      << "fa3_decode: scheduler_metadata must be precomputed";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK(page_table.defined() && page_table.scalar_type() == torch::kInt32);
  const int64_t batch_size = seqused_k.numel();
  CHECK_GT(batch_size, 0);
  CHECK_EQ(cu_seqlens_q.numel(), batch_size + 1);
  CHECK_EQ(page_table.size(0), batch_size);
  const int64_t expected_metadata_size =
      fa3_scheduler_metadata_size(batch_size);
  CHECK_EQ(scheduler_metadata.numel(), expected_metadata_size)
      << "fa3_decode: scheduler_metadata size must be "
      << expected_metadata_size;

  CHECK_GT(k_cache.size(-2), 0);
  CHECK_EQ(query.size(-2) % k_cache.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / k_cache.size(-2);
  const std::string uri = fa3_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  // Both supported ratios use the freshly generated MATE 0.2.7 ABI.
  auto fwd_result =
      get_function(uri, uri)(to_ffi_tensor(query),
                             to_ffi_tensor(k_cache),
                             to_ffi_tensor(v_cache),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             to_ffi_tensor(cu_seqlens_q),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             to_ffi_tensor(seqused_k),
                             ffi::Optional<int64_t>(max_seqlen_q),
                             none_int(),
                             to_ffi_tensor(page_table),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             sm_scale,
                             /*is_causal=*/true,
                             window_left,
                             window_right,
                             /*attention_chunk=*/static_cast<int64_t>(0),
                             /*softcap=*/0.0,
                             /*mp_margin=*/static_cast<int64_t>(0),
                             /*num_splits=*/static_cast<int64_t>(0),
                             to_ffi_tensor(scheduler_metadata),
                             none_tensor(),
                             to_ffi_tensor(output),
                             to_ffi_tensor(output_lse),
                             /*cp_world_size=*/static_cast<int64_t>(1),
                             /*cp_rank=*/static_cast<int64_t>(0),
                             none_tensor(),
                             /*dry_run=*/false);

  // Metadata-enabled FA3 writes split-K partials. Keep the returned tensors
  // alive through combine; graph capture's FFI allocation replay owns their
  // backing storage across subsequent graph replays.
  combine_fa3_split_k(fwd_result,
                      cu_seqlens_q,
                      max_seqlen_q,
                      scheduler_metadata,
                      output,
                      output_lse);
}

void fa3_prefill(const torch::Tensor& query,
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
  CHECK(query.defined() && key.defined() && value.defined());
  CHECK(output.defined() && output_lse.defined());
  CHECK_EQ(query.dim(), 3) << "fa3_prefill: query must be [tokens, heads, dim]";
  CHECK_EQ(key.dim(), 3) << "fa3_prefill: key must be [tokens, heads, dim]";
  CHECK_EQ(value.dim(), 3) << "fa3_prefill: value must be [tokens, heads, dim]";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(cu_seqlens_k.defined() && cu_seqlens_k.scalar_type() == torch::kInt32);
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: only bf16 is supported for the cached dense FA3 URI";
  CHECK_EQ(key.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: key must be bf16";
  CHECK_EQ(value.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: value must be bf16";
  CHECK_EQ(output.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: output must be bf16";
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32)
      << "fa3_prefill: output_lse must be fp32";
  CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous())
      << "fa3_prefill: query, key, and value must be contiguous";
  CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous())
      << "fa3_prefill: cumulative sequence lengths must be contiguous";
  CHECK(output.is_contiguous() && output_lse.is_contiguous())
      << "fa3_prefill: output tensors must be contiguous";
  CHECK_EQ(query.size(-1), 256) << "fa3_prefill: head_dim must be 256";
  CHECK_EQ(key.size(-1), 256) << "fa3_prefill: key head_dim must be 256";
  CHECK_EQ(value.sizes(), key.sizes())
      << "fa3_prefill: key and value shapes must match";
  CHECK_GT(key.size(-2), 0);
  CHECK_EQ(query.size(-2) % key.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / key.size(-2);
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "fa3_prefill: requires GQA ratio 6 or 8 (nq=" << query.size(-2)
      << ", nkv=" << key.size(-2) << ")";
  CHECK_EQ(output.sizes(), query.sizes())
      << "fa3_prefill: output shape must match query";
  CHECK_EQ(output_lse.dim(), 2)
      << "fa3_prefill: output_lse must be [heads, tokens]";
  CHECK_EQ(output_lse.size(0), query.size(1))
      << "fa3_prefill: output_lse head count must match query";
  CHECK_GE(output_lse.numel(), query.size(0) * query.size(1))
      << "fa3_prefill: output_lse is too small";
  CHECK_GT(max_seqlen_q, 0);
  CHECK_GT(max_seqlen_k, 0);

  const std::string uri = fa3_prefill_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  // Arg order matches mate jit `_fmha_fwd` / flash_attn_varlen_func mutlass
  // path for dense ragged Q/K (has_metadata=False).
  get_function(uri, uri)(to_ffi_tensor(query),
                         to_ffi_tensor(key),
                         to_ffi_tensor(value),
                         none_tensor(),  // k_new
                         none_tensor(),  // v_new
                         none_tensor(),  // q_v
                         to_ffi_tensor(cu_seqlens_q),
                         to_ffi_tensor(cu_seqlens_k),
                         none_tensor(),  // cu_seqlens_k_new
                         none_tensor(),  // seqused_q
                         none_tensor(),  // seqused_k
                         max_seqlen_q,
                         max_seqlen_k,
                         none_tensor(),  // page_table
                         none_tensor(),  // kv_batch_idx
                         none_tensor(),  // leftpad_k
                         none_tensor(),  // rotary_cos
                         none_tensor(),  // rotary_sin
                         none_tensor(),  // seqlens_rotary
                         none_tensor(),  // q_descale
                         none_tensor(),  // k_descale
                         none_tensor(),  // v_descale
                         sm_scale,
                         /*is_causal=*/true,
                         window_left,
                         window_right,
                         /*attention_chunk=*/static_cast<int64_t>(0),
                         /*softcap=*/0.0,
                         /*mp_margin=*/static_cast<int64_t>(0),
                         /*num_splits=*/static_cast<int64_t>(0),
                         none_tensor(),  // scheduler_metadata
                         none_tensor(),  // learnable_sink
                         to_ffi_tensor(output),
                         to_ffi_tensor(output_lse),
                         /*cp_world_size=*/static_cast<int64_t>(1),
                         /*cp_rank=*/static_cast<int64_t>(0),
                         none_tensor(),
                         /*dry_run=*/false);
}

torch::Tensor fa3_prefill_scheduler_metadata(const torch::Device& device,
                                             int32_t batch_size,
                                             int32_t num_heads_q,
                                             int32_t num_heads_kv,
                                             int32_t head_dim_qk,
                                             int32_t head_dim_vo,
                                             int32_t max_seqlen_q,
                                             int32_t max_seqlen_k,
                                             int32_t window_size_left,
                                             int32_t window_size_right,
                                             const torch::Tensor& cu_seqlens_q,
                                             const torch::Tensor& seqused_k) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_heads_kv, 0);
  CHECK_EQ(num_heads_q % num_heads_kv, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK_EQ(cu_seqlens_q.dim(), 1);
  CHECK_EQ(seqused_k.dim(), 1);
  CHECK_EQ(cu_seqlens_q.numel(), static_cast<int64_t>(batch_size) + 1);
  CHECK_EQ(seqused_k.numel(), static_cast<int64_t>(batch_size));
  CHECK_GT(max_seqlen_q, 0);
  CHECK_GT(max_seqlen_k, 0);

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor metadata =
      torch::zeros({fa3_scheduler_metadata_size(batch_size)}, options);
  TvmffiStreamGuard stream_guard(device);
  const std::string uri = fa3_metadata_uri(num_heads_q / num_heads_kv);
  // K/V is already stored; no new-KV sequence is appended by this module.
  get_function(uri, uri)(static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(num_heads_q),
                         static_cast<int64_t>(num_heads_kv),
                         static_cast<int64_t>(head_dim_qk),
                         static_cast<int64_t>(head_dim_vo),
                         static_cast<int64_t>(max_seqlen_q),
                         static_cast<int64_t>(max_seqlen_k),
                         /*max_seqlen_k_new=*/static_cast<int64_t>(0),
                         to_ffi_tensor(cu_seqlens_q),
                         none_tensor(),
                         none_tensor(),
                         to_ffi_tensor(seqused_k),
                         none_tensor(),
                         static_cast<int64_t>(window_size_left),
                         static_cast<int64_t>(window_size_right),
                         none_tensor(),
                         to_ffi_tensor(metadata),
                         /*num_splits=*/static_cast<int64_t>(1),
                         static_cast<int64_t>(kFa3TileM),
                         static_cast<int64_t>(kFa3TileN),
                         /*mp_margin=*/static_cast<int64_t>(0));
  return metadata;
}

void fa3_prefill_paged(const torch::Tensor& query,
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
                       torch::Tensor& output,
                       torch::Tensor& output_lse) {
  CHECK(query.defined() && k_cache.defined() && v_cache.defined());
  CHECK(cu_seqlens_q.defined());
  CHECK(seqused_k.defined() && page_table.defined());
  CHECK(scheduler_metadata.defined());
  CHECK(output.defined() && output_lse.defined());
  CHECK_EQ(query.dim(), 3)
      << "fa3_prefill_paged: query must be [tokens, heads, dim]";
  CHECK_EQ(k_cache.dim(), 4)
      << "fa3_prefill_paged: k_cache must be [blocks, page, heads, dim]";
  CHECK_EQ(v_cache.dim(), 4)
      << "fa3_prefill_paged: v_cache must be [blocks, page, heads, dim]";
  CHECK_EQ(k_cache.sizes(), v_cache.sizes())
      << "fa3_prefill_paged: key/value cache shapes must match";
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "fa3_prefill_paged: only bf16 is supported";
  CHECK_EQ(k_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(v_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(output.scalar_type(), torch::kBFloat16);
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32);
  CHECK(query.is_contiguous() && k_cache.is_contiguous() &&
        v_cache.is_contiguous() && output.is_contiguous() &&
        output_lse.is_contiguous())
      << "fa3_prefill_paged: tensors must be contiguous";
  CHECK(cu_seqlens_q.is_contiguous() && seqused_k.is_contiguous() &&
        page_table.is_contiguous() && scheduler_metadata.is_contiguous())
      << "fa3_prefill_paged: metadata tensors must be contiguous";
  CHECK_EQ(cu_seqlens_q.scalar_type(), torch::kInt32);
  CHECK_EQ(seqused_k.scalar_type(), torch::kInt32);
  CHECK_EQ(page_table.scalar_type(), torch::kInt32);
  CHECK_EQ(scheduler_metadata.scalar_type(), torch::kInt32);
  CHECK_EQ(query.size(-1), 256);
  CHECK_EQ(k_cache.size(-1), 256);
  CHECK_EQ(v_cache.size(-1), 256);
  CHECK_GT(k_cache.size(-2), 0);
  CHECK_EQ(query.size(-2) % k_cache.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / k_cache.size(-2);
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "fa3_prefill_paged: requires GQA ratio 6 or 8, got " << gqa_ratio;
  CHECK_EQ(page_table.size(0), seqused_k.numel());
  CHECK_GT(seqused_k.numel(), 0);
  CHECK_EQ(cu_seqlens_q.dim(), 1);
  CHECK_EQ(seqused_k.dim(), 1);
  CHECK_EQ(cu_seqlens_q.numel(), seqused_k.numel() + 1);
  CHECK_EQ(scheduler_metadata.numel(),
           fa3_scheduler_metadata_size(seqused_k.numel()));
  CHECK_GT(max_seqlen_q, 0);

  const std::string uri = fa3_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  // The cache must contain the prefix and all current K/V before this call.
  // Each query is a suffix of its full cached sequence; the layer validates
  // kv_length >= query_length before writing the cache. This is not append-KV.
  auto fwd_result = get_function(uri, uri)(
      to_ffi_tensor(query),
      to_ffi_tensor(k_cache),
      to_ffi_tensor(v_cache),
      none_tensor(),  // k_new (cache was populated by reshape_paged_cache)
      none_tensor(),  // v_new
      none_tensor(),  // q_v
      to_ffi_tensor(cu_seqlens_q),
      none_tensor(),  // cu_seqlens_k
      none_tensor(),  // cache already contains all K/V
      none_tensor(),  // seqused_q
      to_ffi_tensor(seqused_k),
      static_cast<int64_t>(max_seqlen_q),
      none_int(),  // max_seqlen_k
      to_ffi_tensor(page_table),
      none_tensor(),  // kv_batch_idx
      none_tensor(),  // leftpad_k
      none_tensor(),  // rotary_cos
      none_tensor(),  // rotary_sin
      none_tensor(),  // seqlens_rotary
      none_tensor(),  // q_descale
      none_tensor(),  // k_descale
      none_tensor(),  // v_descale
      sm_scale,
      /*is_causal=*/true,
      window_left,
      window_right,
      /*attention_chunk=*/static_cast<int64_t>(0),
      /*softcap=*/0.0,
      /*mp_margin=*/static_cast<int64_t>(0),
      /*num_splits=*/static_cast<int64_t>(1),
      to_ffi_tensor(scheduler_metadata),
      none_tensor(),  // learnable_sink
      to_ffi_tensor(output),
      to_ffi_tensor(output_lse),
      /*cp_world_size=*/static_cast<int64_t>(1),
      /*cp_rank=*/static_cast<int64_t>(0),
      none_tensor(),
      /*dry_run=*/false);
  combine_fa3_split_k(fwd_result,
                      cu_seqlens_q,
                      max_seqlen_q,
                      scheduler_metadata,
                      output,
                      output_lse);
}

}  // namespace xllm::kernel::musa
