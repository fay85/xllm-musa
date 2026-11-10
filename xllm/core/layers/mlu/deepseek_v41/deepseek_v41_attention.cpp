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
#include <tuple>

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
  torch::Tensor raw, scores, iq, iw;
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
    std::tie(iq, iw) = indexer_->project_query(x, qr, positions, rope);
  }
  if (kv_gather) {
    kv = mlu_v4_cp::finish_gather_restore(std::move(*kv_gather), *cp_context);
  }
  if (compressor_gather) {
    torch::Tensor projection = mlu_v4_cp::finish_gather_restore(
        std::move(*compressor_gather), *cp_context);
    raw = projection.slice(-1, 0, dim);
    if (scores.defined()) {
      scores = projection.slice(-1, dim);
    }
  }
  torch::Tensor selected, lengths;
  const int64_t capacity =
      args_.window_size() + (input.use_dflash_proposal_attention
                                 ? args_.dspark_block_size()
                                 : (ratio_ > 0 ? args_.index_topk() : 0));
  if (input.enable_graph) {
    CHECK(shared.selected_kv.defined());
    CHECK(shared.selected_lengths.defined());
    CHECK_EQ(shared.selected_lengths.numel(), q.size(0));
    selected = shared.selected_kv.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/q.size(0) * capacity);
    lengths = shared.selected_lengths;
  } else if (input.use_dflash_proposal_attention) {
    selected = torch::empty({q.size(0) * capacity, 1, 1, dim}, options_);
    lengths = torch::empty({q.size(0)}, options_.dtype(torch::kInt32));
  }
  auto cached_attention = [this,
                           q,
                           kv,
                           raw,
                           scores,
                           iq,
                           iw,
                           positions,
                           rope,
                           output,
                           selected,
                           lengths,
                           &caches,
                           &input,
                           &shared,
                           cp_context](int64_t /*unused*/) {
    forward_cached_attention(q,
                             kv,
                             raw,
                             scores,
                             iq,
                             iw,
                             positions,
                             rope,
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
  if (selected.defined()) {
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
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& raw,
    const torch::Tensor& scores,
    const torch::Tensor& iq,
    const torch::Tensor& iw,
    const torch::Tensor& positions,
    const torch::Tensor& rope,
    std::vector<KVCache>& caches,
    const ModelInputParams& input,
    Csa2Context& shared,
    torch::Tensor output,
    torch::Tensor selected,
    torch::Tensor lengths,
    const mlu_v4_cp::DeepseekV4CpContext* cp_context) {
  const int64_t block = kDeepseekV41MluKVBlockSize;
  torch::Tensor position_cpu;
  const int32_t num_sequences = input.enable_graph
                                    ? input.meta.actual_num_sequences
                                    : input.meta.num_sequences;
  if (!input.enable_graph) {
    position_cpu = (cp_context ? cp_context->global_positions : positions)
                       .to(torch::kCPU, torch::kInt64)
                       .contiguous();
  }
  if (selected.defined()) {
    lengths.zero_();
  } else {
    output.zero_();
  }
  const auto& tables =
      input.enable_graph ? shared.block_tables : input.multi_block_tables;
  CHECK_GE(tables.size(), 1);
  torch::Tensor swa = flat_cache(caches[layer_id_].get_swa_cache());
  int64_t offset = 0;
  int64_t front_offset = 0;
  int64_t back_offset = 0;
  if (cp_context != nullptr) {
    const auto& front =
        cp_context->geometry.front_seq_lens_by_rank[cp_context->cp_rank];
    back_offset = std::accumulate(front.begin(), front.end(), int64_t{0});
  }
  for (int32_t seq = 0; seq < num_sequences; ++seq) {
    const int64_t count = input.get_q_seq_len(seq);
    if (count == 0) {
      continue;
    }
    const bool single_decode = input.enable_graph && count == 1;
    const bool candidate_decode = single_decode &&
                                  args_.candidate_source_layer_id() >= 0 &&
                                  layer_id_ > args_.candidate_source_layer_id();
    const int64_t begin =
        input.enable_graph ? input.attention.host.kv_seq_lens[seq + 1] -
                                 input.attention.host.kv_seq_lens[seq] - count
                           : position_cpu.data_ptr<int64_t>()[offset];
    const int64_t end = begin + count;
    const int64_t old_begin =
        std::max<int64_t>(0,
                          begin - args_.window_size() +
                              (input.use_dflash_proposal_attention ? 0 : 1));
    // Read the old ring before publishing a chunk that can wrap it.
    torch::Tensor window;
    if (!input.enable_graph || input.use_dflash_proposal_attention) {
      window = torch::cat(
          {swa.index_select(
               0,
               slots(tables[0], seq, old_begin, begin, block, true, options_)),
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
      if (source_ == layer_id_) {
        update_compressed_cache(raw.slice(0, offset, offset + count),
                                ratio_ == 2
                                    ? scores.slice(0, offset, offset + count)
                                    : torch::Tensor(),
                                begin,
                                seq,
                                rope,
                                caches[source_],
                                tables);
      }
      torch::Tensor all_slots;
      if (!single_decode ||
          (index_source_ && ratio_ != 2 && !candidate_decode)) {
        all_slots =
            slots(tables[group], seq, 0, end / ratio_, block, false, options_);
      }
      if (!input.enable_graph) {
        main = main_cache.index_select(0, all_slots);
      }
      if (single_decode && index_source_ && ratio_ == 2) {
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
        const int64_t segment_begin =
            position_cpu
                .data_ptr<int64_t>()[geometry.rows_by_rank[rank][local_offset]];
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
    } else if (input.enable_graph) {
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
            tables[0][seq],
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

    if (!input.enable_graph && !input.use_dflash_proposal_attention) {
      const int64_t tail = std::min<int64_t>(count, tables[0].size(1) * block);
      swa.index_copy_(
          0,
          slots(tables[0], seq, end - tail, end, block, true, options_),
          kv.slice(0, offset + count - tail, offset + count));
    }
    offset += count;
  }
}

void DeepseekV41AttentionImpl::update_compressed_cache(
    const torch::Tensor& raw,
    const torch::Tensor& scores,
    int64_t begin,
    int32_t seq,
    const torch::Tensor& rope,
    KVCache& cache,
    const std::vector<torch::Tensor>& tables) {
  const int64_t count = raw.size(0);
  const int64_t end = begin + count;
  const int64_t dim = args_.head_dim();
  const int64_t rope_dim = args_.qk_rope_head_dim();
  const int64_t block = kDeepseekV41MluKVBlockSize;
  const double eps = args_.rms_norm_eps();
  const int64_t group = ratio_ == 2 ? 1 : 2;
  torch::Tensor main_cache = flat_cache(cache.get_k_cache());
  torch::Tensor index_cache = flat_cache(cache.get_index_cache());
  torch::Tensor latent;
  const int64_t first = begin / ratio_;
  const int64_t committed = end / ratio_;
  if (ratio_ == 1) {
    CHECK(raw.scalar_type() == torch::kBFloat16);
    latent = raw;
  } else {
    CHECK(raw.scalar_type() == torch::kFloat32);
    CHECK(scores.scalar_type() == torch::kFloat32);
    torch::Tensor state = flat_cache(cache.get_compress_kv_state());
    torch::Tensor score_state = flat_cache(cache.get_compress_score_state());
    const int64_t state_rows =
        state.size(0) / cache.get_compress_kv_state().size(0);
    torch::Tensor members =
        torch::arange(first * 2, committed * 2, options_.dtype(torch::kInt64));
    torch::Tensor relative = (members - begin).clamp(0, count - 1);
    torch::Tensor physical =
        slots(tables[0], seq, first * 2, committed * 2, block, true, options_);
    torch::Tensor parked = torch::floor_divide(physical, block) * state_rows +
                           members.remainder(state_rows);
    torch::Tensor in_chunk = members.ge(begin).unsqueeze(-1);
    torch::Tensor values = torch::where(
        in_chunk, raw.index_select(0, relative), state.index_select(0, parked));
    torch::Tensor gates = torch::where(in_chunk,
                                       scores.index_select(0, relative),
                                       score_state.index_select(0, parked));
    latent = (values.view({committed - first, 2, dim}) *
              gates.view({committed - first, 2, dim}).softmax(1))
                 .sum(1);
    // Verify may reject an interior row; the next odd boundary needs the
    // preceding accepted raw row, not the final proposed row.
    const int64_t parked_count = args_.num_speculative_tokens() > 0
                                     ? std::min(count, state_rows)
                                     : end % 2;
    if (parked_count > 0) {
      auto positions =
          torch::arange(end - parked_count, end, options_.dtype(torch::kInt64));
      auto target =
          slots(tables[0], seq, end - parked_count, end, block, true, options_);
      target = torch::floor_divide(target, block) * state_rows +
               positions.remainder(state_rows);
      state.index_copy_(0, target, raw.slice(0, count - parked_count, count));
      score_state.index_copy_(
          0, target, scores.slice(0, count - parked_count, count));
    }
  }
  if (committed > first) {
    // Pooling and Norm each have a BF16 rounding point; state stays FP32.
    latent = norm(latent.to(torch::kBFloat16),
                  weights_.at("compressor.norm.weight"),
                  eps);
    torch::Tensor starts =
        torch::arange(first, committed, options_.dtype(torch::kInt64)) * ratio_;
    torch::Tensor target =
        slots(tables[group], seq, first, committed, block, false, options_);
    index_cache.index_copy_(
        0, target, indexer_->project_key(latent, starts, rope));
    main_cache.index_copy_(0, target, rotate(latent, starts, rope, rope_dim));
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
