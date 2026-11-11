/* Copyright 2026 The xLLM Authors.

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

#include "layers/mlu/deepseek_v41/deepseek_v41_attention.h"

#include <ATen/ops/mm.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "framework/model/graph_eager_context.h"
#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "kernels/mlu/bangc/include/deepseek_v41_constants.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_attention_utils.h"

namespace xllm::layer {
namespace {
using deepseek_v41::dense_projection;
using deepseek_v41::norm;
using deepseek_v41::rotate;
using kernel::mlu::bangc::kDeepseekV41MluKVBlockSize;

torch::Tensor slots(const torch::Tensor& table,
                    int64_t seq,
                    int64_t begin,
                    int64_t end,
                    int64_t block_size,
                    bool ring,
                    const torch::TensorOptions& options) {
  torch::Tensor pos = torch::arange(begin, end, options.dtype(torch::kInt64));
  torch::Tensor columns = torch::floor_divide(pos, block_size);
  if (ring) {
    columns = columns.remainder(table.size(1));
  }
  return table[seq]
                 .to(options.device(), torch::kInt64)
                 .index_select(0, columns) *
             block_size +
         pos.remainder(block_size);
}

torch::Tensor flat_cache(const torch::Tensor& cache) {
  return cache.view({-1, cache.size(-1)});
}

bool contains(const std::vector<int32_t>& ids, int32_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}
}  // namespace

DeepseekV41AttentionImpl::DeepseekV41AttentionImpl(const ModelContext& context,
                                                   int32_t layer_id)
    : args_(context.get_model_args()),
      layer_id_(layer_id),
      tp_group_(context.get_parallel_args().tp_group_),
      rank_(tp_group_->rank()),
      tp_size_(tp_group_->world_size()),
      options_(context.get_tensor_options()) {
  ratio_ = args_.compress_ratios().at(layer_id);
  CHECK(ratio_ == 0 || ratio_ == 1 || ratio_ == 2);
  CHECK_EQ(args_.o_groups() % tp_size_, 0);
  for (int32_t source : args_.kv_source_layer_ids()) {
    if (source <= layer_id) {
      source_ = std::max(source_, source);
    }
  }
  index_source_ = contains(args_.index_source_layer_ids(), layer_id);
  indexer_ = register_module("indexer", DeepseekV41Indexer(context, layer_id));
  const auto& quant = context.get_quant_args();
  const std::string prefix = "layers." + std::to_string(layer_id);
  wq_b_ = register_module(
      "wq_b",
      ColumnParallelLinear(args_.q_lora_rank(),
                           args_.n_heads() * args_.head_dim(),
                           false,
                           false,
                           quant.for_module(prefix + ".attn.wq_b"),
                           tp_group_,
                           options_));
  wo_b_ =
      register_module("wo_b",
                      RowParallelLinear(args_.o_groups() * args_.o_lora_rank(),
                                        args_.hidden_size(),
                                        false,
                                        true,
                                        true,
                                        quant.for_module(prefix + ".attn.wo_b"),
                                        tp_group_,
                                        options_));
}

void DeepseekV41AttentionImpl::load_state_dict(const StateDict& state) {
  indexer_->load_state_dict(state.get_dict_with_prefix("indexer."));
  wq_b_->load_state_dict(state.get_dict_with_prefix("wq_b."));
  wo_b_->load_state_dict(state.get_dict_with_prefix("wo_b."));
  for (const std::string name : {"wq_a.weight",
                                 "wkv.weight",
                                 "q_norm.weight",
                                 "kv_norm.weight",
                                 "wo_a.weight",
                                 "attn_sink",
                                 "compressor.wkv.weight",
                                 "compressor.wgate.weight",
                                 "compressor.norm.weight"}) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    if (name == "wo_a.weight" || name == "attn_sink") {
      value = value.chunk(tp_size_, 0).at(rank_);
    }
    if (name == "attn_sink") {
      value = value.to(torch::kFloat32);
    }
    weights_[name] = value.to(options_.device()).contiguous();
  }
}

void DeepseekV41AttentionImpl::verify_loaded_weights() const {
  const std::string prefix = "layers." + std::to_string(layer_id_) + ".attn.";
  for (const std::string name : {"wq_a.weight",
                                 "wkv.weight",
                                 "q_norm.weight",
                                 "kv_norm.weight",
                                 "wo_a.weight",
                                 "attn_sink"}) {
    CHECK(weights_.count(name)) << "Missing weight " << prefix + name;
  }
  CHECK(wq_b_->is_weight_loaded())
      << "Missing weight or quantization state for " << prefix + "wq_b.weight";
  CHECK(wo_b_->is_weight_loaded())
      << "Missing weight or quantization state for " << prefix + "wo_b.weight";
  if (source_ == layer_id_ && ratio_ > 0) {
    for (const std::string name :
         {"compressor.wkv.weight", "compressor.norm.weight"}) {
      CHECK(weights_.count(name)) << "Missing weight " << prefix + name;
    }
    CHECK(weights_.at("compressor.wkv.weight").scalar_type() ==
          torch::kBFloat16);
    if (ratio_ == 2) {
      CHECK(weights_.count("compressor.wgate.weight"))
          << "Missing weight " << prefix + "compressor.wgate.weight";
      CHECK(weights_.at("compressor.wgate.weight").scalar_type() ==
            torch::kBFloat16);
    }
  }
  indexer_->verify_loaded_weights();
}

void DeepseekV41AttentionImpl::write_context_kv(
    const torch::Tensor& hidden,
    const torch::Tensor& positions,
    const torch::Tensor& rope,
    const torch::Tensor& cache_slots,
    KVCache& cache) {
  auto valid = torch::nonzero(cache_slots.ge(0)).flatten();
  auto rows = hidden.index_select(0, valid);
  auto kv = rotate(norm(dense_projection(rows, weights_.at("wkv.weight")),
                        weights_.at("kv_norm.weight"),
                        args_.rms_norm_eps()),
                   positions.index_select(0, valid),
                   rope,
                   args_.qk_rope_head_dim());
  flat_cache(cache.get_swa_cache())
      .index_copy_(0, cache_slots.index_select(0, valid).to(torch::kInt64), kv);
}

torch::Tensor DeepseekV41AttentionImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& positions,
    const torch::Tensor& rope,
    std::vector<KVCache>& caches,
    const ModelInputParams& input,
    Csa2Context& shared,
    const mlu_v4_cp::DeepseekV4CpContext* cp_context) {
  const int64_t dim = args_.head_dim();
  const int64_t rope_dim = args_.qk_rope_head_dim();
  const int64_t heads = args_.n_heads() / tp_size_;
  const double eps = args_.rms_norm_eps();
  torch::Tensor kv = rotate(norm(dense_projection(x, weights_.at("wkv.weight")),
                                 weights_.at("kv_norm.weight"),
                                 eps),
                            positions,
                            rope,
                            rope_dim);
  torch::Tensor raw, scores, iq;
  if (source_ == layer_id_ && ratio_ > 0) {
    CHECK(x.scalar_type() == torch::kBFloat16);
    if (ratio_ == 1) {
      raw = dense_projection(x, weights_.at("compressor.wkv.weight"));
    } else {
      // Keep the BF16 operands and request FP32 output before any rounding.
      raw = torch::mm(
          x, weights_.at("compressor.wkv.weight").t(), torch::kFloat32);
      scores = torch::mm(
          x, weights_.at("compressor.wgate.weight").t(), torch::kFloat32);
    }
  }
  std::optional<parallel_state::GatherAsyncCtx> kv_gather, compressor_gather;
  if (cp_context != nullptr) {
    kv_gather = mlu_v4_cp::launch_gather_rows(kv, *cp_context);
    if (raw.defined()) {
      torch::Tensor projection =
          scores.defined() ? torch::cat({raw, scores}, -1) : raw.contiguous();
      compressor_gather =
          mlu_v4_cp::launch_gather_rows(projection, *cp_context);
    }
  }
  torch::Tensor qr = norm(dense_projection(x, weights_.at("wq_a.weight")),
                          weights_.at("q_norm.weight"),
                          eps);
  torch::Tensor q = rotate(
      wq_b_->forward(qr).view({-1, heads, dim}), positions, rope, rope_dim);
  torch::Tensor output = torch::empty_like(q);
  if (index_source_) {
    iq = indexer_->project_query(qr, positions, rope);
  }
  if (kv_gather) {
    kv = mlu_v4_cp::finish_gather_restore(std::move(*kv_gather), *cp_context);
  }
  std::optional<torch::Tensor> row_map;
  if (compressor_gather) {
    torch::Tensor projection =
        parallel_state::finish_gather(std::move(*compressor_gather));
    row_map = cp_context->restore_indices.to(torch::kInt32);
    raw = projection.slice(-1, 0, dim);
    if (scores.defined()) {
      scores = projection.slice(-1, dim);
    }
  }
  torch::Tensor iw;
  if (ratio_ > 0 && (source_ == layer_id_ || index_source_)) {
    CHECK_GE(source_, 0);
    const auto& tables = shared.block_tables;
    torch::Tensor latent, starts, target;
    std::optional<torch::Tensor> num_valid_compressed_rows;
    if (source_ == layer_id_) {
      auto& out = shared.compressed[ratio_ - 1];
      auto& cache = caches[source_];
      const bool pooled = ratio_ == 2;
      kernel::mlu::bangc::CompressKvMetadata metadata;
      metadata.query_start_loc = shared.query_prefix;
      metadata.chunk_begins = shared.chunk_begins;
      metadata.compressed_start_loc = out.prefix;
      metadata.compressed_block_table = tables[pooled ? 1 : 2];
      metadata.input_row_map = std::move(row_map);
      std::optional<kernel::mlu::bangc::CompressKvPoolingInputs> pooling;
      if (pooled) {
        pooling.emplace();
        pooling->scores = scores;
        pooling->state_block_table = tables[0];
        pooling->state_raw = cache.get_compress_kv_state();
        pooling->state_score = cache.get_compress_score_state();
        pooling->save_full_tail = args_.num_speculative_tokens() > 0;
      }
      kernel::mlu::bangc::CompressKvOptions options;
      options.ratio = ratio_;
      options.rms_norm_eps = eps;
      kernel::mlu::bangc::fused_compress_kv_cache(
          raw,
          weights_.at("compressor.norm.weight"),
          rope,
          metadata,
          pooling,
          cache.get_k_cache().view({-1, 128, 512}),
          out.outputs,
          options);
      latent = out.outputs.latent;
      starts = out.outputs.positions;
      target = out.outputs.slots;
      num_valid_compressed_rows = out.outputs.num_valid_rows;
    }
    iw = indexer_->project_indexer(index_source_ ? x : torch::Tensor(),
                                   latent,
                                   starts,
                                   target,
                                   rope,
                                   caches[source_].get_index_cache(),
                                   shared.num_valid_input_rows,
                                   num_valid_compressed_rows);
  }
  torch::Tensor selected, lengths;
  const int64_t capacity =
      args_.window_size() + (input.use_dflash_proposal_attention
                                 ? args_.dspark_block_size()
                                 : (ratio_ > 0 ? args_.index_topk() : 0));
  const bool decode = deepseek_v41::uses_decode_attention(input);
  if (decode) {
    CHECK(cp_context == nullptr);
    CHECK(shared.selected_kv.defined());
    CHECK(shared.selected_lengths.defined());
    CHECK_EQ(shared.selected_lengths.numel(), q.size(0));
    selected = shared.selected_kv.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/q.size(0) * capacity);
    lengths = shared.selected_lengths;
  }
  auto cached_attention = [this,
                           iw,
                           q,
                           kv,
                           iq,
                           output,
                           selected,
                           lengths,
                           &caches,
                           &input,
                           &shared,
                           cp_context](int64_t /*unused*/) {
    forward_cached_attention(iw,
                             q,
                             kv,
                             iq,
                             caches,
                             input,
                             shared,
                             output,
                             selected,
                             lengths,
                             cp_context);
  };
  if (input.graph.eager_context != nullptr) {
    input.graph.eager_context->run_eager(std::move(cached_attention));
  } else {
    cached_attention(0);
  }
  if (decode) {
    decode_attention(q, selected, lengths, output);
  }
  return project_output(output, positions, rope);
}

void DeepseekV41AttentionImpl::decode_attention(const torch::Tensor& q,
                                                const torch::Tensor& selected,
                                                const torch::Tensor& lengths,
                                                torch::Tensor output) {
  const int64_t dim = args_.head_dim();
  const int64_t capacity = selected.size(0) / q.size(0);
  auto table =
      torch::arange(q.size(0) * capacity, options_.dtype(torch::kInt32))
          .view({q.size(0), 1, capacity});
  auto query = q.unsqueeze(1);
  auto out = output.unsqueeze(1);
  std::optional<torch::Tensor> lse;
  kernel::mlu::batch_decode(query,
                            selected,
                            out,
                            table,
                            lengths,
                            std::nullopt,
                            lse,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            "half",
                            capacity,
                            -1,
                            -1,
                            1.0 / std::sqrt(dim),
                            false,
                            -1,
                            std::nullopt,
                            -1,
                            weights_.at("attn_sink"));
}

void DeepseekV41AttentionImpl::forward_cached_attention(
    const torch::Tensor& iw,
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& iq,
    std::vector<KVCache>& caches,
    const ModelInputParams& input,
    Csa2Context& shared,
    torch::Tensor output,
    torch::Tensor selected,
    torch::Tensor lengths,
    const mlu_v4_cp::DeepseekV4CpContext* cp_context) {
  const int64_t block = kDeepseekV41MluKVBlockSize;
  const bool decode = deepseek_v41::uses_decode_attention(input);
  if (decode) {
    lengths.zero_();
  } else {
    output.zero_();
  }
  const auto& tables = shared.block_tables;
  CHECK_GE(tables.size(), 1);
  torch::Tensor swa = flat_cache(caches[layer_id_].get_swa_cache());
  const auto& swa_table = shared.swa_block_table;
  int64_t front_offset = 0;
  int64_t back_offset = 0;
  if (cp_context != nullptr) {
    const auto& front =
        cp_context->geometry.front_seq_lens_by_rank[cp_context->cp_rank];
    back_offset = std::accumulate(front.begin(), front.end(), int64_t{0});
  }
  for (int32_t seq = 0; seq < shared.num_sequences; ++seq) {
    const int64_t offset = shared.host_query_prefix[seq];
    const int64_t count = shared.host_query_prefix[seq + 1] - offset;
    if (count == 0) {
      continue;
    }
    const bool single_decode = decode && count == 1;
    const bool candidate_decode = single_decode &&
                                  args_.candidate_source_layer_id() >= 0 &&
                                  layer_id_ > args_.candidate_source_layer_id();
    const int64_t begin = shared.host_chunk_begins[seq];
    const int64_t end = begin + count;
    const int64_t old_begin =
        std::max<int64_t>(0,
                          begin - args_.window_size() +
                              (input.use_dflash_proposal_attention ? 0 : 1));
    // Read the old ring before publishing a chunk that can wrap it.
    torch::Tensor window;
    if (!decode || input.use_dflash_proposal_attention) {
      window = torch::cat(
          {swa.index_select(
               0,
               slots(swa_table, seq, old_begin, begin, block, true, options_)),
           kv.slice(0, offset, offset + count)},
          0);
    }
    torch::Tensor main, keys;
    if (ratio_ > 0) {
      CHECK_GE(source_, 0);
      const int64_t group = ratio_ == 2 ? 1 : 2;
      CHECK_LT(group, tables.size());
      torch::Tensor main_cache = flat_cache(caches[source_].get_k_cache());
      torch::Tensor index_cache = flat_cache(caches[source_].get_index_cache());
      torch::Tensor all_slots;
      const bool paged_decode = single_decode && index_source_ && ratio_ == 2 &&
                                !candidate_decode &&
                                layer_id_ != args_.candidate_source_layer_id();
      if (!single_decode ||
          (index_source_ && !paged_decode && !candidate_decode)) {
        all_slots =
            slots(tables[group], seq, 0, end / ratio_, block, false, options_);
      }
      if (!decode) {
        main = main_cache.index_select(0, all_slots);
      }
      if (paged_decode) {
        indexer_->select_decode_indices(iq[offset],
                                        iw[offset],
                                        caches[source_].get_index_cache(),
                                        tables[group][seq],
                                        end / ratio_,
                                        seq,
                                        shared);
      } else if (index_source_ && candidate_decode) {
        indexer_->select_candidate_indices(iq[offset],
                                           iw[offset],
                                           index_cache,
                                           tables[group][seq],
                                           end / ratio_,
                                           seq,
                                           shared);
      } else if (index_source_) {
        keys = index_cache.index_select(0, all_slots).to(torch::kFloat32);
        if (cp_context == nullptr) {
          indexer_->select_indices(iq.slice(0, offset, offset + count),
                                   iw.slice(0, offset, offset + count),
                                   keys,
                                   begin,
                                   seq,
                                   shared);
        }
      }
    }
    if (input.use_dflash_proposal_attention) {
      const int64_t capacity = selected.size(0) / q.size(0);
      const int64_t visible = window.size(0);
      selected.view({q.size(0), capacity, args_.head_dim()})
          .slice(0, offset, offset + count)
          .slice(1, 0, visible)
          .copy_(
              window.unsqueeze(0).expand({count, visible, args_.head_dim()}));
      lengths.slice(0, offset, offset + count).fill_(visible);
    } else if (cp_context != nullptr) {
      const auto& geometry = cp_context->geometry;
      const int32_t rank = cp_context->cp_rank;
      const int64_t counts[2] = {geometry.front_seq_lens_by_rank[rank][seq],
                                 geometry.back_seq_lens_by_rank[rank][seq]};
      int64_t* local_offsets[2] = {&front_offset, &back_offset};
      for (int32_t part = 0; part < 2; ++part) {
        const int64_t rows = counts[part];
        const int64_t local_offset = *local_offsets[part];
        *local_offsets[part] += rows;
        if (rows == 0) {
          continue;
        }
        const int64_t segment_begin = shared.host_positions.const_data_ptr<
            int64_t>()[geometry.rows_by_rank[rank][local_offset]];
        const int32_t shared_seq = seq * 2 + part;
        if (ratio_ > 0 && index_source_) {
          indexer_->select_indices(
              iq.slice(0, local_offset, local_offset + rows),
              iw.slice(0, local_offset, local_offset + rows),
              keys,
              segment_begin,
              shared_seq,
              shared);
        }
        const int64_t segment_old_begin =
            std::max<int64_t>(0, segment_begin - args_.window_size() + 1);
        compute_attention(q.slice(0, local_offset, local_offset + rows),
                          window.slice(0, segment_old_begin - old_begin),
                          main,
                          shared.topk.at(shared_seq),
                          segment_begin,
                          output.slice(0, local_offset, local_offset + rows));
      }
    } else if (decode) {
      const int64_t capacity = selected.size(0) / q.size(0);
      auto main_cache =
          ratio_ > 0 ? flat_cache(caches[source_].get_k_cache()) : swa;
      auto comp_table = tables[ratio_ == 2 ? 1 : (ratio_ == 1 ? 2 : 0)][seq];
      for (int64_t row = 0; row < count; ++row) {
        auto topk = ratio_ > 0
                        ? shared.topk.at(seq)[row].flatten()
                        : torch::empty({0}, options_.dtype(torch::kInt64));
        const int64_t query = offset + row;
        kernel::mlu::bangc::deepseek_v41_decode_gather(
            swa,
            main_cache,
            kv[query],
            swa_table[seq],
            comp_table,
            topk,
            begin + row,
            args_.window_size(),
            selected.slice(0, query * capacity, (query + 1) * capacity),
            lengths.slice(0, query, query + 1));
      }
    } else {
      compute_attention(q.slice(0, offset, offset + count),
                        window,
                        main,
                        shared.topk.at(seq),
                        begin,
                        output.slice(0, offset, offset + count));
    }

    if (!decode) {
      const int64_t tail = std::min<int64_t>(count, swa_table.size(1) * block);
      swa.index_copy_(
          0,
          slots(swa_table, seq, end - tail, end, block, true, options_),
          kv.slice(0, offset + count - tail, offset + count));
    }
  }
}

void DeepseekV41AttentionImpl::compute_attention(const torch::Tensor& q,
                                                 const torch::Tensor& window,
                                                 const torch::Tensor& main,
                                                 const torch::Tensor& indices,
                                                 int64_t begin,
                                                 torch::Tensor output) {
  const int64_t count = q.size(0);
  const int64_t dim = args_.head_dim();
  const int64_t heads = args_.n_heads() / tp_size_;
  const int64_t old_begin =
      std::max<int64_t>(0, begin - args_.window_size() + 1);
  const double neg_inf = -std::numeric_limits<double>::infinity();
  // Bounded query tiles keep sparse gather memory independent of chunk length.
  for (int64_t row = 0; row < count; row += 16) {
    const int64_t rows = std::min<int64_t>(16, count - row);
    torch::Tensor pos = torch::arange(
        begin + row, begin + row + rows, options_.dtype(torch::kInt64));
    torch::Tensor win_pos =
        pos.unsqueeze(1) - args_.window_size() + 1 +
        torch::arange(args_.window_size(), options_.dtype(torch::kInt64))
            .unsqueeze(0);
    torch::Tensor valid = win_pos.ge(0);
    torch::Tensor selected =
        window.index_select(0, (win_pos - old_begin).clamp_min(0).flatten())
            .view({rows, args_.window_size(), dim});
    if (ratio_ > 0 && main.size(0) > 0) {
      torch::Tensor topk = indices.slice(0, row, row + rows);
      torch::Tensor compressed =
          main.index_select(0, topk.clamp_min(0).flatten())
              .view({rows, topk.size(1), dim});
      selected = torch::cat({selected, compressed}, 1);
      valid = torch::cat({valid, topk.ge(0)}, 1);
    }
    torch::Tensor logits =
        torch::bmm(q.slice(0, row, row + rows).to(torch::kFloat32),
                   selected.to(torch::kFloat32).transpose(1, 2)) /
        std::sqrt(dim);
    logits.masked_fill_(~valid.unsqueeze(1), neg_inf);
    torch::Tensor sink = weights_.at("attn_sink")
                             .to(torch::kFloat32)
                             .view({1, heads, 1})
                             .expand({rows, heads, 1});
    torch::Tensor probability = torch::cat({logits, sink}, -1)
                                    .softmax(-1)
                                    .slice(-1, 0, logits.size(-1));
    output.slice(0, row, row + rows)
        .copy_(torch::bmm(probability, selected.to(torch::kFloat32))
                   .to(q.scalar_type()));
  }
}

torch::Tensor DeepseekV41AttentionImpl::project_output(
    torch::Tensor output,
    const torch::Tensor& positions,
    const torch::Tensor& rope) {
  const int64_t dim = args_.head_dim();
  const int64_t heads = args_.n_heads() / tp_size_;
  const int64_t rope_dim = args_.qk_rope_head_dim();
  output = rotate(output, positions, rope, rope_dim, true);
  const int64_t groups = args_.o_groups() / tp_size_;
  torch::Tensor projected = torch::bmm(
      output.view({-1, groups, heads * dim / groups}).transpose(0, 1),
      weights_.at("wo_a.weight")
          .view({groups, args_.o_lora_rank(), heads * dim / groups})
          .transpose(1, 2));
  return wo_b_->forward(projected.transpose(0, 1).contiguous().flatten(1));
}
}  // namespace xllm::layer
