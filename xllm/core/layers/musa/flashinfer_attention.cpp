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

#include "layers/musa/flashinfer_attention.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "kernels/musa/musa_ops_api.h"
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

namespace {

bool is_fa3_shape_supported(int64_t head_size,
                            int64_t num_heads,
                            int64_t num_kv_heads) {
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return false;
  }
  const int64_t gqa_ratio = num_heads / num_kv_heads;
  return head_size == 256 && (gqa_ratio == 6 || gqa_ratio == 8);
}

bool check_prefill_lengths(const AttentionMetadata& metadata,
                           int64_t query_tokens) {
  CHECK(!metadata.q_seq_lens_vec.empty())
      << "FA3 prefill requires host per-sequence lengths";
  CHECK_EQ(metadata.q_seq_lens_vec.size(), metadata.kv_seq_lens_vec.size())
      << "FA3 prefill query/KV batch sizes must match";
  CHECK(metadata.q_cu_seq_lens.defined());
  CHECK_EQ(metadata.q_cu_seq_lens.numel(), metadata.q_seq_lens_vec.size() + 1);
  int64_t valid_query_tokens = 0;
  bool has_cached_prefix = false;
  for (size_t i = 0; i < metadata.q_seq_lens_vec.size(); ++i) {
    const int32_t query_length = metadata.q_seq_lens_vec[i];
    const int32_t kv_length = metadata.kv_seq_lens_vec[i];
    CHECK_GE(query_length, 0);
    CHECK_GE(kv_length, query_length)
        << "FA3 prefill KV length must include every current query token";
    valid_query_tokens += query_length;
    has_cached_prefix = has_cached_prefix || kv_length > query_length;
  }
  // Execution buffers may also contain padding after the valid query rows.
  CHECK_LE(valid_query_tokens, query_tokens);
  return has_cached_prefix;
}

bool use_expanded_spec_decode_attention(
    const AttentionMetadata& attn_metadata) {
  const ExpandedDecodeMetadata& expanded = attn_metadata.expanded_decode;
  return expanded.enabled && expanded.paged_kv_indptr.defined() &&
         expanded.paged_kv_indptr.numel() >= 2;
}

AttentionMetadata build_expanded_decode_metadata(
    const AttentionMetadata& attn_metadata) {
  AttentionMetadata decode_meta = attn_metadata;
  const ExpandedDecodeMetadata& expanded = attn_metadata.expanded_decode;
  const int64_t expanded_batch = expanded.paged_kv_indptr.size(0) - 1;
  const torch::Device expanded_device = expanded.paged_kv_indptr.device();
  const torch::TensorOptions expanded_int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(expanded_device);
  decode_meta.paged_kv_indptr = expanded.paged_kv_indptr;
  decode_meta.paged_kv_indices = expanded.paged_kv_indices;
  decode_meta.paged_kv_last_page_len = expanded.paged_kv_last_page_len;
  decode_meta.block_table = expanded.block_table;
  decode_meta.kv_seq_lens = expanded.kv_seq_lens;
  decode_meta.qo_indptr =
      torch::arange(0, expanded_batch + 1, expanded_int_options);
  decode_meta.max_query_len = 1;
  // Host mirrors live under fa3_metadata after the common metadata move.
  decode_meta.fa3_metadata.paged_kv_indptr_host = torch::Tensor();
  decode_meta.fa3_metadata.paged_kv_indices_host = torch::Tensor();
  decode_meta.fa3_metadata.paged_kv_last_page_len_host = torch::Tensor();
  return decode_meta;
}

}  // namespace

bool use_fa3_attention(int64_t head_size,
                       int64_t num_heads,
                       int64_t num_kv_heads,
                       torch::ScalarType dtype,
                       bool is_decode,
                       int64_t window_left) {
  static const int32_t setting = [] {
    const char* env = std::getenv("XLLM_USE_FA3");
    return env == nullptr ? int32_t{-1}
                          : (std::string(env) == "1" ? int32_t{1} : int32_t{0});
  }();
  static const int32_t decode_setting = [] {
    const char* env = std::getenv("XLLM_USE_FA3_DECODE");
    return env == nullptr ? setting
                          : (std::string(env) == "1" ? int32_t{1} : int32_t{0});
  }();
  const int32_t selected = is_decode ? decode_setting : setting;
  if (selected == 0) {
    return false;
  }
  // Both compiled GQA ratios default to FA3 for prefill and decode.
  // The compiled profile is causal but not local, including window_left=0.
  // Explicit enablement cannot bypass the compiled attention contract.
  return window_left < 0 && dtype == torch::kBFloat16 &&
         is_fa3_shape_supported(head_size, num_heads, num_kv_heads);
}

void check_fa3_attention(int64_t head_size,
                         int64_t num_heads,
                         int64_t num_kv_heads,
                         torch::ScalarType dtype,
                         bool is_decode,
                         int64_t window_left) {
  CHECK(use_fa3_attention(
      head_size, num_heads, num_kv_heads, dtype, is_decode, window_left))
      << "Unsupported MUSA attention configuration: this build requires FA3 "
         "with bf16, head_dim=256, GQA=6/8 and no sliding window. "
         "XLLM_USE_FA3 and XLLM_USE_FA3_DECODE must not disable FA3; "
         "legacy FlashInfer FA2 operators are not built.";
}

FlashInferAttentionImpl::FlashInferAttentionImpl(int64_t num_heads,
                                                 int64_t head_size,
                                                 float scale,
                                                 int64_t num_kv_heads,
                                                 int64_t sliding_window)
    : BaseAttentionImpl(num_heads,
                        head_size,
                        scale,
                        num_kv_heads,
                        sliding_window > 0 ? sliding_window - 1 : -1) {
  check_fa3_attention(head_size_,
                      num_heads_,
                      num_kv_heads_,
                      torch::kBFloat16,
                      /*is_decode=*/false,
                      sliding_window_);
  check_fa3_attention(head_size_,
                      num_heads_,
                      num_kv_heads_,
                      torch::kBFloat16,
                      /*is_decode=*/true,
                      sliding_window_);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
FlashInferAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                                 torch::Tensor& query,
                                 torch::Tensor& key,
                                 torch::Tensor& value,
                                 torch::Tensor& output,
                                 KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  const bool is_decode = !attn_metadata.is_prefill &&
                         (!attn_metadata.is_chunked_prefill ||
                          attn_metadata.expanded_decode.enabled ||
                          use_expanded_spec_decode_attention(attn_metadata) ||
                          attn_metadata.is_spec_verify);
  check_fa3_attention(head_size_,
                      num_heads_,
                      num_kv_heads_,
                      query.scalar_type(),
                      is_decode,
                      sliding_window_);
  CHECK(!attn_metadata.attn_mask.defined())
      << "MUSA FA3 does not support custom attention masks";
  const bool has_cached_prefix =
      !is_decode && check_prefill_lengths(attn_metadata, query.size(0));
  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  // Only reshape and store to cache if k_cache is properly initialized
  // For prefill without KV cache (e.g., LongCat text encoding), skip this step
  if (k_cache.defined() && k_cache.dim() >= 2) {
    CHECK(attn_metadata.slot_mapping.defined())
        << "FlashInferAttention requires slot_mapping when KV cache is defined";
    CHECK_EQ(attn_metadata.slot_mapping.numel(), key.size(0))
        << "slot_mapping token count mismatch: slot_mapping="
        << attn_metadata.slot_mapping.sizes() << ", key=" << key.sizes()
        << ", value=" << value.sizes() << ", k_cache=" << k_cache.sizes()
        << ", v_cache=" << v_cache.sizes()
        << ", is_prefill=" << attn_metadata.is_prefill
        << ", is_chunked_prefill=" << attn_metadata.is_chunked_prefill;
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  const bool spec_verify_expanded_decode =
      attn_metadata.is_chunked_prefill &&
      (attn_metadata.expanded_decode.enabled ||
       use_expanded_spec_decode_attention(attn_metadata) ||
       attn_metadata.is_spec_verify);
  if (attn_metadata.is_prefill) {
    prefill_forward(attn_metadata,
                    query,
                    key,
                    value,
                    output,
                    output_lse,
                    k_cache,
                    v_cache,
                    has_cached_prefix);
  } else if (spec_verify_expanded_decode) {
    decoder_forward(attn_metadata, query, output, output_lse, k_cache, v_cache);
  } else if (attn_metadata.is_chunked_prefill) {
    prefill_forward(attn_metadata,
                    query,
                    key,
                    value,
                    output,
                    output_lse,
                    k_cache,
                    v_cache,
                    has_cached_prefix);
  } else {
    decoder_forward(attn_metadata, query, output, output_lse, k_cache, v_cache);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void FlashInferAttentionImpl::prefill_forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& output,
    std::optional<torch::Tensor>& output_lse,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    bool has_cached_prefix) {
  const Fa3AttentionMetadata& fa3_metadata = attn_metadata.fa3_metadata;
  CHECK(attn_metadata.q_cu_seq_lens.defined())
      << "XLLM_USE_FA3=1 requires q_cu_seq_lens";
  CHECK_GT(attn_metadata.max_query_len, 0)
      << "XLLM_USE_FA3=1 requires max_query_len > 0";
  CHECK_GT(attn_metadata.max_seq_len, 0)
      << "XLLM_USE_FA3=1 requires max_seq_len > 0";

  // Ordinary-prefill dispatch: a fresh request has no
  // existing prefix and uses the dense ragged FA3 kernel; paged
  // flash_attn_with_kvcache is reserved for an actual cached prefix (or a
  // later chunk). Merely having a block table is not evidence of a prefix --
  // xLLM allocates one for every request before the first forward.
  const bool paged_prefill_supported = k_cache.defined() && v_cache.defined() &&
                                       attn_metadata.block_table.defined() &&
                                       attn_metadata.kv_seq_lens.defined() &&
                                       has_cached_prefix;
  CHECK(!has_cached_prefix || paged_prefill_supported)
      << "FA3 cached-prefix prefill requires paged KV caches, block_table "
         "and kv_seq_lens";
  if (paged_prefill_supported) {
    CHECK_EQ(k_cache.dim(), 4) << "FA3 paged prefill requires a 4D key cache";
    CHECK_EQ(v_cache.dim(), 4) << "FA3 paged prefill requires a 4D value cache";
    CHECK_EQ(attn_metadata.block_table.scalar_type(), torch::kInt32)
        << "FA3 paged prefill requires int32 block_table";
    CHECK(attn_metadata.block_table.is_contiguous())
        << "FA3 paged prefill requires contiguous block_table";
    CHECK_EQ(attn_metadata.block_table.size(0),
             attn_metadata.kv_seq_lens.numel())
        << "FA3 paged prefill batch metadata mismatch";

    torch::Tensor seqused_k = attn_metadata.kv_seq_lens;
    if (seqused_k.scalar_type() != torch::kInt32) {
      seqused_k = seqused_k.to(torch::kInt32);
    }
    seqused_k = seqused_k.contiguous();
    const torch::Tensor cu_seqlens_q = attn_metadata.q_cu_seq_lens.contiguous();
    const int32_t batch_size =
        static_cast<int32_t>(attn_metadata.block_table.size(0));
    torch::Tensor scheduler_metadata;
    if (fa3_metadata.share_fa3_scheduler_metadata &&
        fa3_metadata.fa3_scheduler_metadata.defined()) {
      CHECK_EQ(fa3_metadata.fa3_scheduler_metadata.numel(),
               xllm::kernel::musa::fa3_scheduler_metadata_size(batch_size))
          << "FA3 prefill scheduler metadata shape changed within one "
             "forward";
      scheduler_metadata = fa3_metadata.fa3_scheduler_metadata;
    }
    if (!scheduler_metadata.defined()) {
      scheduler_metadata = xllm::kernel::musa::fa3_prefill_scheduler_metadata(
          query.device(),
          batch_size,
          static_cast<int32_t>(num_heads_),
          static_cast<int32_t>(num_kv_heads_),
          static_cast<int32_t>(head_size_),
          static_cast<int32_t>(head_size_),
          static_cast<int32_t>(attn_metadata.max_query_len),
          static_cast<int32_t>(attn_metadata.max_seq_len),
          static_cast<int32_t>(sliding_window_),
          /*window_size_right=*/0,
          cu_seqlens_q,
          seqused_k);
      if (fa3_metadata.share_fa3_scheduler_metadata) {
        fa3_metadata.fa3_scheduler_metadata = scheduler_metadata;
      }
    }

    torch::Tensor lse_tensor;
    if (output_lse.has_value() && output_lse->defined()) {
      lse_tensor = *output_lse;
    } else {
      const int64_t required = num_heads_ * query.size(0);
      const auto lse_options =
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
      const bool need_realloc =
          !prefill_lse_buf_.defined() ||
          prefill_lse_buf_.dtype() != lse_options.dtype() ||
          prefill_lse_buf_.device() != lse_options.device() ||
          prefill_lse_buf_.numel() < required;
      if (need_realloc) {
        prefill_lse_buf_ = torch::empty({required}, lse_options);
      }
      lse_tensor = prefill_lse_buf_.narrow(0, 0, required)
                       .view({num_heads_, query.size(0)});
    }

    torch::Tensor query_contiguous = query.contiguous();
    xllm::kernel::musa::fa3_prefill_paged(
        query_contiguous,
        k_cache,
        v_cache,
        cu_seqlens_q,
        seqused_k,
        attn_metadata.block_table,
        scheduler_metadata,
        /*max_seqlen_q=*/attn_metadata.max_query_len,
        /*window_left=*/sliding_window_,
        /*window_right=*/0,
        scale_,
        output,
        lse_tensor);
    if (output_lse.has_value()) {
      *output_lse = lse_tensor;
    }
    return;
  }

  CHECK(attn_metadata.kv_cu_seq_lens.defined())
      << "FA3 dense prefill requires kv_cu_seq_lens";
  const int64_t total_q = query.size(0);
  torch::Tensor lse_tensor;
  if (output_lse.has_value() && output_lse->defined()) {
    lse_tensor = *output_lse;
  } else {
    const int64_t required = num_heads_ * total_q;
    const auto lse_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    const bool need_realloc =
        !prefill_lse_buf_.defined() ||
        prefill_lse_buf_.dtype() != lse_options.dtype() ||
        prefill_lse_buf_.device() != lse_options.device() ||
        prefill_lse_buf_.numel() < required;
    if (need_realloc) {
      prefill_lse_buf_ = torch::empty({required}, lse_options);
    }
    lse_tensor =
        prefill_lse_buf_.narrow(0, 0, required).view({num_heads_, total_q});
  }

  torch::Tensor query_contiguous = query.contiguous();
  torch::Tensor key_contiguous = key.contiguous();
  torch::Tensor value_contiguous = value.contiguous();
  torch::Tensor q_cu_seq_lens = attn_metadata.q_cu_seq_lens.contiguous();
  torch::Tensor kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens.contiguous();
  xllm::kernel::musa::fa3_prefill(query_contiguous,
                                  key_contiguous,
                                  value_contiguous,
                                  q_cu_seq_lens,
                                  kv_cu_seq_lens,
                                  /*max_seqlen_q=*/attn_metadata.max_query_len,
                                  /*max_seqlen_k=*/attn_metadata.max_seq_len,
                                  /*window_left=*/-1,
                                  /*window_right=*/-1,
                                  scale_,
                                  output,
                                  lse_tensor);
  if (output_lse.has_value()) {
    *output_lse = lse_tensor;
  }
}

void FlashInferAttentionImpl::decoder_forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& output,
    std::optional<torch::Tensor>& output_lse,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache) {
  std::optional<AttentionMetadata> expanded_decode_meta;
  if (use_expanded_spec_decode_attention(attn_metadata)) {
    expanded_decode_meta = build_expanded_decode_metadata(attn_metadata);
  }
  const AttentionMetadata& decode_attn =
      expanded_decode_meta.has_value() ? *expanded_decode_meta : attn_metadata;
  // Keep FA3 scheduler sharing on the original metadata so layer-to-layer
  // reuse still works when decode_attn is an expanded copy.
  const Fa3AttentionMetadata& fa3_metadata = attn_metadata.fa3_metadata;
  CHECK(decode_attn.block_table.defined())
      << "FA3 decode requires block_table (rectangular page_table)";
  const int64_t batch_size = decode_attn.block_table.size(0);

  // seqused_k = per-seq kv length (NOT cumulative). attn_metadata
  // already keeps it under `kv_seq_lens`; if undefined fall back to
  // torch::diff of the cumulative form.
  torch::Tensor seqused_k = decode_attn.kv_seq_lens;
  if (!seqused_k.defined() || seqused_k.numel() == 0) {
    CHECK(decode_attn.kv_cu_seq_lens.defined())
        << "FA3 decode requires kv_seq_lens or kv_cu_seq_lens";
    seqused_k = torch::diff(decode_attn.kv_cu_seq_lens).to(torch::kInt32);
  } else if (seqused_k.scalar_type() != torch::kInt32) {
    seqused_k = seqused_k.to(torch::kInt32);
  }
  seqused_k = seqused_k.contiguous();

  // page_table: native rectangular block_table built by the input builder
  // from allocated KV blocks. Unused slots are
  // -1; graph mode reuses persistent_block_tables_ updated each step.
  const torch::Tensor page_table = decode_attn.block_table;

  // Use the exact host-side KV length for scheduler partitioning.
  const int32_t max_seqlen_k = static_cast<int32_t>(decode_attn.max_seq_len);
  const int32_t max_seqlen_q =
      static_cast<int32_t>(std::max<int64_t>(decode_attn.max_query_len, 1));
  CHECK_GT(max_seqlen_k, 0)
      << "FA3 decode requires attn_metadata.max_seq_len > 0";

  // Qwen hybrid explicitly scopes this cache to one model forward. Other
  // models retain per-layer generation so shared graph-capture metadata
  // cannot accidentally keep scheduler values from a previous step.
  torch::Tensor scheduler_metadata;
  if (fa3_metadata.share_fa3_scheduler_metadata &&
      fa3_metadata.fa3_scheduler_metadata.defined()) {
    CHECK_EQ(fa3_metadata.fa3_scheduler_metadata.numel(),
             xllm::kernel::musa::fa3_scheduler_metadata_size(batch_size))
        << "FA3 scheduler metadata shape changed within one forward";
    scheduler_metadata = fa3_metadata.fa3_scheduler_metadata;
  }
  const torch::Tensor cu_seqlens_q =
      decode_attn.qo_indptr.has_value() && decode_attn.qo_indptr->defined()
          ? *decode_attn.qo_indptr
          : decode_attn.q_cu_seq_lens;
  if (!scheduler_metadata.defined()) {
    scheduler_metadata = xllm::kernel::musa::fa3_decode_scheduler_metadata(
        query.device(),
        /*batch_size=*/static_cast<int32_t>(batch_size),
        /*num_heads_q=*/static_cast<int32_t>(num_heads_),
        /*num_heads_kv=*/static_cast<int32_t>(num_kv_heads_),
        /*head_dim_qk=*/static_cast<int32_t>(head_size_),
        /*head_dim_vo=*/static_cast<int32_t>(head_size_),
        /*max_seqlen_q=*/max_seqlen_q,
        /*max_seqlen_k=*/max_seqlen_k,
        /*window_size_left=*/static_cast<int32_t>(sliding_window_),
        /*window_size_right=*/0,
        /*cu_seqlens_q=*/cu_seqlens_q,
        /*seqused_k=*/seqused_k);
    if (fa3_metadata.share_fa3_scheduler_metadata) {
      fa3_metadata.fa3_scheduler_metadata = scheduler_metadata;
    }
  }

  // Keep the FA3 LSE address stable across graph capture and replay.
  const int64_t total_q = query.size(0);
  torch::Tensor lse_tensor;
  if (output_lse.has_value() && output_lse->defined()) {
    lse_tensor = *output_lse;
  } else {
    const int64_t required = num_heads_ * total_q;
    const auto lse_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    const bool need_realloc =
        !decode_lse_buf_.defined() ||
        decode_lse_buf_.dtype() != lse_options.dtype() ||
        decode_lse_buf_.device() != lse_options.device() ||
        decode_lse_buf_.numel() < required;
    if (need_realloc) {
      constexpr int64_t kDefaultMaxGraphBatchSize = 256;
      decode_lse_buf_ = torch::empty(
          {std::max(required, kDefaultMaxGraphBatchSize * num_heads_)},
          lse_options);
    }
    lse_tensor =
        decode_lse_buf_.narrow(0, 0, required).view({num_heads_, total_q});
  }

  xllm::kernel::musa::fa3_decode(
      query,
      k_cache,
      v_cache,
      cu_seqlens_q,
      seqused_k,
      page_table,
      scheduler_metadata,
      /*max_seqlen_q=*/max_seqlen_q,
      /*window_left=*/static_cast<int64_t>(sliding_window_),
      /*window_right=*/0,
      scale_,
      output,
      lse_tensor);
  if (output_lse.has_value()) {
    *output_lse = lse_tensor;
  }
}

}  // namespace layer
}  // namespace xllm
