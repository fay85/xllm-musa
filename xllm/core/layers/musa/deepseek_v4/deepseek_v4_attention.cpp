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

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace xllm {
namespace layer {
namespace {

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

bool dsv4_musa_compress_fallback_enabled() {
  // DeepSeek-V4 DSA correctness depends on the compressed-KV contribution.
  // Keep the torch fallback enabled unconditionally for the minimal MUSA
  // port; this avoids introducing another runtime environment switch.
  return true;
}

int64_t dsv4_musa_prefill_full_kv_threshold() { return 2048; }

int64_t dsv4_musa_prefill_attention_chunk_size() {
  // Keep the fallback's temporary score tensor below the workspace budget
  // while avoiding thousands of tiny MUSA matmul launches on long prefill.
  return 256;
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

  auto value_2d = value.reshape({-1, value.size(value.dim() - 1)});
  auto cache_2d = cache.view({-1, value_2d.size(1)});
  auto slots = slot_mapping.reshape({-1}).to(torch::kLong).to(cache.device());
  const int64_t update_rows =
      std::min<int64_t>(slots.size(0), value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }

  auto slots_slice = slots.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  auto value_slice =
      value_2d.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  auto valid_mask = slots_slice.ge(0);
  if (!valid_mask.any().item<bool>()) {
    return;
  }
  auto valid_slots = slots_slice.index({valid_mask});
  auto valid_values = value_slice.index({valid_mask});
  cache_2d.index_copy_(/*dim=*/0, valid_slots, valid_values);
}

int64_t find_swa_cache_index(const DSAMetadata& attn_metadata,
                             int32_t layer_id) {
  if (attn_metadata.caches_info == nullptr || layer_id < 0 ||
      layer_id >= static_cast<int32_t>(attn_metadata.caches_info->size())) {
    return -1;
  }
  const auto& layer_caches = (*attn_metadata.caches_info)[layer_id];
  for (size_t cache_idx = 0; cache_idx < layer_caches.size(); ++cache_idx) {
    if (layer_caches[cache_idx].type == DSACacheType::SLIDING_WINDOW) {
      return static_cast<int64_t>(cache_idx);
    }
  }
  return -1;
}

int64_t find_token_cache_index(const DSAMetadata& attn_metadata,
                               int32_t layer_id,
                               int64_t ratio) {
  if (attn_metadata.caches_info == nullptr || layer_id < 0 ||
      layer_id >= static_cast<int32_t>(attn_metadata.caches_info->size())) {
    return -1;
  }
  const auto& layer_caches = (*attn_metadata.caches_info)[layer_id];
  for (size_t cache_idx = 0; cache_idx < layer_caches.size(); ++cache_idx) {
    if (layer_caches[cache_idx].type == DSACacheType::TOKEN &&
        effective_compress_ratio(layer_caches[cache_idx].ratio) == ratio) {
      return static_cast<int64_t>(cache_idx);
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
                              context.get_quant_args(),
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

  if (quant_args.quant_method() != "" || !quant_args.quant_descs().empty()) {
    LOG(WARNING)
        << "DeepseekV4 MUSA attention fallback only guarantees unquantized "
           "weights. Quantized DeepSeek-V4 checkpoints need dedicated MUSA "
           "attention/projection kernels for production use.";
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
  return flat_cache.index_select(/*dim=*/0, idx);
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
  torch::Tensor cmp_2d;
  const bool has_compressed = compressed_kv.defined() &&
                              compressed_kv.numel() > 0 &&
                              (compress_ratio_ == 4 || compress_ratio_ == 128);
  if (has_compressed) {
    cmp_2d = compressed_kv.reshape(
        {-1, compressed_kv.size(compressed_kv.dim() - 1)});
  }

  auto q_h = q.permute({1, 0, 2}).contiguous().to(torch::kFloat32);
  auto kv_fp32 = kv_2d.to(torch::kFloat32);
  auto ori_scores =
      torch::matmul(q_h, kv_fp32.transpose(0, 1).unsqueeze(0)) * softmax_scale_;
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
  }
  auto value_fp32 = kv_fp32;
  torch::Tensor cmp_scores;
  torch::Tensor cmp_fp32;
  if (has_compressed) {
    cmp_fp32 = cmp_2d.to(torch::kFloat32);
    cmp_scores = torch::matmul(q_h, cmp_fp32.transpose(0, 1).unsqueeze(0)) *
                 softmax_scale_;
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
    }
    if (compressed_indices.defined() && compressed_indices.numel() > 0 &&
        compressed_indices.dim() == 2 &&
        compressed_indices.size(0) == q.size(0)) {
      auto indices =
          compressed_indices.to(torch::kLong).to(q.device()).contiguous();
      auto valid_indices =
          indices.ge(0).logical_and(indices.lt(cmp_2d.size(0)));
      auto safe_indices =
          torch::where(valid_indices, indices, torch::zeros_like(indices));
      auto topk_mask = torch::zeros(
          {q.size(0), cmp_2d.size(0)},
          torch::TensorOptions().dtype(torch::kBool).device(q.device()));
      topk_mask = topk_mask.scatter(/*dim=*/1, safe_indices, valid_indices);
      cmp_scores =
          cmp_scores.masked_fill(topk_mask.logical_not().unsqueeze(0),
                                 -std::numeric_limits<float>::infinity());
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
  auto inv_denom = denom.reciprocal();
  auto probs = exp_scores * inv_denom;
  auto ctx = torch::matmul(probs, value_fp32.unsqueeze(0));
  if (exp_cmp_scores.defined()) {
    auto cmp_probs = exp_cmp_scores * inv_denom;
    ctx = ctx + torch::matmul(cmp_probs, cmp_fp32.unsqueeze(0));
  }
  return ctx.permute({1, 0, 2}).contiguous().to(q.dtype());
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
  auto score_full = compressor_wgate_->forward(hidden_states)
                        .to(torch::kFloat32)
                        .contiguous();
  auto ape = compressor_ape_.to(score_full.device()).to(torch::kFloat32);

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
  auto compressed =
      (pooled_kv * weights).sum(/*dim=*/1).to(hidden_states.dtype());
  compressed = std::get<0>(compressor_norm_->forward(compressed));
  if (cos.defined() && sin.defined() &&
      cos.numel() >= blocks * rope_head_dim_ &&
      sin.numel() >= blocks * rope_head_dim_) {
    auto cos_slice = cos.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks);
    auto sin_slice = sin.slice(/*dim=*/0, /*start=*/0, /*end=*/blocks);
    compressed = apply_interleaved_partial_rope(
        compressed, nope_head_dim_, rope_head_dim_, cos_slice, sin_slice);
  }
  fp8_act_quant_roundtrip_inplace(
      compressed, nope_head_dim_, /*block_size=*/64);
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
  auto score_step =
      compressor_wgate_->forward(hidden_step).to(torch::kFloat32).contiguous();
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
  auto compressed =
      (pooled_kv * weights).sum(/*dim=*/0).unsqueeze(0).to(hidden_step.dtype());
  compressed = std::get<0>(compressor_norm_->forward(compressed));
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
  }
  fp8_act_quant_roundtrip_inplace(
      compressed, nope_head_dim_, /*block_size=*/64);
  if (compress_ratio_ == 4) {
    kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
        .copy_(kv_state.slice(/*dim=*/0, /*start=*/compress_ratio_));
    score_state.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio_)
        .copy_(score_state.slice(/*dim=*/0, /*start=*/compress_ratio_));
  }
  return compressed;
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
  (void)kv_state;
  (void)compress_metadata;

  auto q_down = q_a_proj_->forward(hidden_states);
  auto qr = std::get<0>(q_layernorm_->forward(q_down));
  auto q = q_b_proj_->forward(qr).view({-1, n_local_heads_, head_dim_});
  q = rms_norm_last_dim(q, q_rms_gamma_, eps_);

  auto kv_down = kv_proj_->forward(hidden_states);
  auto kv =
      std::get<0>(kv_layernorm_->forward(kv_down)).view({-1, 1, head_dim_});

  const auto [active_cos, active_sin] =
      select_rope_tables(attn_metadata, compress_ratio_, q.size(0));
  q = apply_interleaved_partial_rope(
      q, nope_head_dim_, rope_head_dim_, active_cos, active_sin);
  kv = apply_interleaved_partial_rope(
      kv, nope_head_dim_, rope_head_dim_, active_cos, active_sin);
  fp8_act_quant_roundtrip_inplace(kv, nope_head_dim_, /*block_size=*/64);

  auto ori_kv = kv_cache.get_swa_cache();
  const int64_t swa_cache_idx = find_swa_cache_index(attn_metadata, layer_id_);
  auto swa_slot_mapping =
      layer_cache_tensor(attn_metadata.slot_mappings, layer_id_, swa_cache_idx);
  auto swa_block_table =
      layer_cache_tensor(attn_metadata.block_tables, layer_id_, swa_cache_idx);
  if (ori_kv.defined()) {
    scatter_by_slot(ori_kv, swa_slot_mapping, kv);
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
  std::vector<torch::Tensor> outputs;
  outputs.reserve(query_ranges.size());

  const bool use_compressed_attention =
      dsv4_musa_compress_fallback_enabled() && !force_full_kv_attention &&
      (compress_ratio_ == 4 || compress_ratio_ == 128) && compressor_wkv_ &&
      compressor_wgate_ && compressor_norm_ && compressor_ape_loaded_;
  const int64_t cmp_cache_idx =
      use_compressed_attention
          ? find_token_cache_index(attn_metadata, layer_id_, compress_ratio_)
          : -1;
  auto cmp_cache =
      use_compressed_attention ? kv_cache.get_k_cache() : torch::Tensor();
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
  const torch::Tensor& compressed_cos =
      compress_ratio_ == 4 ? attn_metadata.c4_cos : attn_metadata.c128_cos;
  const torch::Tensor& compressed_sin =
      compress_ratio_ == 4 ? attn_metadata.c4_sin : attn_metadata.c128_sin;
  std::vector<torch::Tensor> prefill_compressed_by_seq(query_ranges.size());
  if (use_compressed_attention && is_prefill && !is_chunked_prefill) {
    compressor_hidden_history_.clear();
    compressor_kv_history_.clear();
    compressor_kv_state_.clear();
    compressor_score_state_.clear();
    compressor_hidden_history_.resize(query_ranges.size());
    compressor_kv_history_.resize(query_ranges.size());
    compressor_kv_state_.resize(query_ranges.size());
    compressor_score_state_.resize(query_ranges.size());
    std::vector<torch::Tensor> compressed_to_cache;
    compressed_to_cache.reserve(query_ranges.size());
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
      compressor_hidden_history_[seq_idx] = hidden_seq.contiguous();
      auto cmp_seq =
          compress_prefill_sequence(hidden_seq, cos_slice, sin_slice, seq_idx);
      if (cmp_seq.defined() && cmp_seq.numel() > 0) {
        prefill_compressed_by_seq[seq_idx] = cmp_seq;
        compressor_kv_history_[seq_idx] = cmp_seq.contiguous();
        compressed_to_cache.push_back(cmp_seq);
      } else {
        compressor_kv_history_[seq_idx] =
            torch::empty({0, head_dim_}, hidden_states.options());
      }
    }
    if (!compressed_to_cache.empty() && cmp_cache.defined() &&
        cmp_slot_mapping.defined()) {
      auto cmp_all = torch::cat(compressed_to_cache, /*dim=*/0);
      scatter_by_slot(cmp_cache, cmp_slot_mapping, cmp_all);
    }
  }

  std::vector<torch::Tensor> decode_compressed_to_cache;
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
    if (use_current_prefill_kv) {
      kv_seq = kv.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end).squeeze(1);
    } else {
      kv_seq = gather_cache(ori_kv,
                            swa_block_table,
                            static_cast<int64_t>(seq_idx),
                            kv_lens[seq_idx]);
      if (!kv_seq.defined()) {
        kv_seq =
            kv.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end).squeeze(1);
      }
    }
    torch::Tensor cmp_seq;
    if (use_compressed_attention) {
      if (use_current_prefill_kv) {
        cmp_seq = prefill_compressed_by_seq[seq_idx];
      } else {
        if (seq_idx >= compressor_hidden_history_.size()) {
          compressor_hidden_history_.resize(seq_idx + 1);
        }
        if (seq_idx >= compressor_kv_history_.size()) {
          compressor_kv_history_.resize(seq_idx + 1);
        }
        auto hidden_seq =
            hidden_states.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
        if (hidden_seq.defined() && hidden_seq.numel() > 0) {
          if (compressor_hidden_history_[seq_idx].defined() &&
              compressor_hidden_history_[seq_idx].numel() > 0) {
            compressor_hidden_history_[seq_idx] =
                torch::cat({compressor_hidden_history_[seq_idx], hidden_seq},
                           /*dim=*/0)
                    .contiguous();
          } else {
            compressor_hidden_history_[seq_idx] = hidden_seq.contiguous();
          }
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
            if (new_cmp.defined() && new_cmp.numel() > 0) {
              if (compressor_kv_history_[seq_idx].defined() &&
                  compressor_kv_history_[seq_idx].numel() > 0) {
                compressor_kv_history_[seq_idx] =
                    torch::cat({compressor_kv_history_[seq_idx], new_cmp},
                               /*dim=*/0)
                        .contiguous();
              } else {
                compressor_kv_history_[seq_idx] = new_cmp.contiguous();
              }
              decode_compressed_to_cache.push_back(new_cmp);
            }
          }
        }
        cmp_seq = compressor_kv_history_[seq_idx];
        if (!cmp_seq.defined()) {
          const int64_t cmp_len = kv_lens[seq_idx] / compress_ratio_;
          cmp_seq = gather_cache(cmp_cache,
                                 cmp_block_table,
                                 static_cast<int64_t>(seq_idx),
                                 cmp_len);
        }
      }
    }
    torch::Tensor cmp_indices;
    const int64_t seq_q_len = q_end - q_start;
    const int64_t q_abs_start =
        use_current_prefill_kv
            ? 0
            : std::max<int64_t>(kv_lens[seq_idx] - seq_q_len, 0);
    auto seq_output = dense_attention(q_seq,
                                      kv_seq,
                                      cmp_seq,
                                      cmp_indices,
                                      use_current_prefill_kv,
                                      q_abs_start,
                                      use_current_prefill_kv);
    outputs.push_back(seq_output);
  }
  if (!decode_compressed_to_cache.empty() && cmp_cache.defined() &&
      cmp_slot_mapping.defined()) {
    auto cmp_all = torch::cat(decode_compressed_to_cache, /*dim=*/0);
    scatter_by_slot(cmp_cache, cmp_slot_mapping, cmp_all);
  }
  torch::Tensor attn_output =
      outputs.empty() ? torch::zeros_like(q) : torch::cat(outputs, /*dim=*/0);

  auto o = apply_interleaved_partial_rope(attn_output,
                                          nope_head_dim_,
                                          rope_head_dim_,
                                          active_cos,
                                          active_sin,
                                          /*inverse=*/true);
  const int64_t num_tokens = o.size(0);
  auto o_group = o.view({num_tokens, n_local_groups_, -1});
  auto wo_a_weight = dequantize_fp8_projection_weight(
      o_a_proj_->weight(), o_a_proj_->weight_scale(), o.scalar_type());
  auto wo_a = wo_a_weight.view({n_local_groups_, o_lora_rank_, -1});
  auto o_low_rank = torch::einsum("tgd,grd->tgr", {o_group, wo_a});
  auto output = o_b_proj_->forward(o_low_rank.reshape({num_tokens, -1}));
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
}

}  // namespace layer
}  // namespace xllm
