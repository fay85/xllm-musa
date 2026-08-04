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

#include "attention_metadata_builder.h"

#include "framework/model/model_input_params.h"
#include "layers/common/attention_metadata.h"

namespace xllm::layer::musa {

namespace {

void normalize_host_sequence_lengths(AttentionMetadata& attn_metadata) {
  if (attn_metadata.q_seq_lens_vec.size() >= 2 &&
      attn_metadata.q_seq_lens_vec.front() == 0) {
    attn_metadata.musa.q_cu_seq_lens_host_vec = attn_metadata.q_seq_lens_vec;
    std::vector<int32_t> per_seq;
    per_seq.reserve(attn_metadata.q_seq_lens_vec.size() - 1);
    for (size_t i = 1; i < attn_metadata.q_seq_lens_vec.size(); ++i) {
      per_seq.emplace_back(attn_metadata.q_seq_lens_vec[i] -
                           attn_metadata.q_seq_lens_vec[i - 1]);
    }
    attn_metadata.q_seq_lens_vec = std::move(per_seq);
  } else if (!attn_metadata.q_seq_lens_vec.empty()) {
    attn_metadata.musa.q_cu_seq_lens_host_vec.reserve(
        attn_metadata.q_seq_lens_vec.size() + 1);
    attn_metadata.musa.q_cu_seq_lens_host_vec.emplace_back(0);
    int32_t total = 0;
    for (int32_t len : attn_metadata.q_seq_lens_vec) {
      total += len;
      attn_metadata.musa.q_cu_seq_lens_host_vec.emplace_back(total);
    }
  }

  if (attn_metadata.kv_seq_lens_vec.size() >= 2 &&
      attn_metadata.kv_seq_lens_vec.front() == 0) {
    std::vector<int32_t> per_seq;
    per_seq.reserve(attn_metadata.kv_seq_lens_vec.size() - 1);
    for (size_t i = 1; i < attn_metadata.kv_seq_lens_vec.size(); ++i) {
      per_seq.emplace_back(attn_metadata.kv_seq_lens_vec[i] -
                           attn_metadata.kv_seq_lens_vec[i - 1]);
    }
    attn_metadata.kv_seq_lens_vec = std::move(per_seq);
  }
}

}  // namespace

void populate_attention_metadata(
    AttentionMetadata& attn_metadata,
    const ModelInputParams& params,
    const std::optional<torch::Tensor>& attn_mask) {
  normalize_host_sequence_lengths(attn_metadata);

  attn_metadata.musa.paged_kv_indptr_host =
      params.attention.host.paged_kv_indptr;
  attn_metadata.musa.paged_kv_indices_host =
      params.attention.host.paged_kv_indices;
  attn_metadata.musa.paged_kv_last_page_len_host =
      params.attention.host.paged_kv_last_page_len;

  // MUSA FlashInfer accepts the explicit one-dimensional padding mask. Dense
  // graph-buffer masks remain on the custom-mask path.
  if (attn_mask.has_value() && attn_mask->dim() == 1) {
    attn_metadata.attn_mask = attn_mask.value();
  }

  if (params.attention.device.block_tables.defined()) {
    attn_metadata.block_table = params.attention.device.block_tables;
  }
  if (params.attention.device.q_seq_lens.defined() &&
      params.attention.device.q_seq_lens.numel() >= 2) {
    attn_metadata.q_seq_lens = torch::diff(params.attention.device.q_seq_lens);
  }
  if (params.attention.device.kv_seq_lens.defined() &&
      params.attention.device.kv_seq_lens.numel() >= 2) {
    attn_metadata.kv_seq_lens =
        torch::diff(params.attention.device.kv_seq_lens);
  }
}

void finalize_attention_metadata(AttentionMetadata& attn_metadata) {
  if (attn_metadata.is_causal && !attn_metadata.enable_cuda_graph &&
      attn_metadata.q_cu_seq_lens.defined()) {
    attn_metadata.qo_indptr =
        attn_metadata.q_cu_seq_lens.to(torch::kPrivateUse1);
  }
}

}  // namespace xllm::layer::musa
