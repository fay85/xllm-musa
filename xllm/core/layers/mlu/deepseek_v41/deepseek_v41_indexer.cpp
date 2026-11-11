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

#include "layers/mlu/deepseek_v41/deepseek_v41_indexer.h"

#include <algorithm>
#include <limits>

#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "kernels/mlu/bangc/include/deepseek_v41_constants.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::layer {
namespace {
using kernel::mlu::bangc::kDeepseekV41MluKVBlockSize;
}  // namespace

DeepseekV41IndexerImpl::DeepseekV41IndexerImpl(const ModelContext& context,
                                               int32_t layer_id)
    : args_(context.get_model_args()),
      layer_id_(layer_id),
      ratio_(args_.compress_ratios().at(layer_id)),
      options_(context.get_tensor_options()) {
  const auto& kv_sources = args_.kv_source_layer_ids();
  const auto& index_sources = args_.index_source_layer_ids();
  const bool writes_k = ratio_ > 0 && std::find(kv_sources.begin(),
                                                kv_sources.end(),
                                                layer_id_) != kv_sources.end();
  const bool writes_head =
      ratio_ > 0 &&
      std::find(index_sources.begin(), index_sources.end(), layer_id_) !=
          index_sources.end();
  if (writes_k || writes_head) {
    CHECK_EQ(options_.dtype().toScalarType(), torch::kBFloat16);
    CHECK_EQ(args_.index_head_dim(), 128);
    CHECK_EQ(args_.qk_rope_head_dim(), 64);
  }
  if (writes_head) {
    CHECK_EQ(args_.hidden_size(), 5120);
    CHECK_EQ(args_.index_n_heads(), 32);
  }
  if (writes_k) {
    CHECK_EQ(args_.head_dim(), 512);
    CHECK_EQ(args_.rms_norm_eps(), 1e-20f);
  }
}

void DeepseekV41IndexerImpl::load_state_dict(const StateDict& state) {
  for (const std::string name :
       {"wq_b.weight", "wk.weight", "k_norm.weight", "weights_proj.weight"}) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    weights_[name] = value.to(options_.device()).contiguous();
  }
}

void DeepseekV41IndexerImpl::verify_loaded_weights() const {
  const std::string prefix =
      "layers." + std::to_string(layer_id_) + ".attn.indexer.";
  const auto& index_sources = args_.index_source_layer_ids();
  if (std::find(index_sources.begin(), index_sources.end(), layer_id_) !=
      index_sources.end()) {
    for (const std::string name : {"wq_b.weight", "weights_proj.weight"}) {
      CHECK(weights_.count(name)) << "Missing weight " << prefix + name;
    }
  }
  const auto& kv_sources = args_.kv_source_layer_ids();
  if (ratio_ > 0 &&
      std::find(kv_sources.begin(), kv_sources.end(), layer_id_) !=
          kv_sources.end()) {
    for (const std::string name : {"wk.weight", "k_norm.weight"}) {
      CHECK(weights_.count(name)) << "Missing weight " << prefix + name;
    }
  }
}

torch::Tensor DeepseekV41IndexerImpl::project_query(
    const torch::Tensor& qr,
    const torch::Tensor& positions,
    const torch::Tensor& rope) {
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  return kernel::mlu::bangc::fused_indexer_q(
             qr,
             weights_.at("wq_b.weight"),
             positions.to(torch::kInt32).contiguous(),
             rope)
      .to(torch::kFloat32);
}

torch::Tensor DeepseekV41IndexerImpl::project_indexer(
    const torch::Tensor& x,
    const torch::Tensor& latent,
    const torch::Tensor& positions,
    const torch::Tensor& slots,
    const torch::Tensor& rope,
    const torch::Tensor& cache,
    const std::optional<torch::Tensor>& num_valid_input_rows,
    const std::optional<torch::Tensor>& num_valid_compressed_rows) {
  const bool has_head = x.defined() && x.size(0) > 0;
  const bool has_k = latent.defined() && latent.size(0) > 0;
  torch::Tensor head = torch::empty({x.defined() ? x.size(0) : 0, 32},
                                    options_.dtype(torch::kFloat32));
  return kernel::mlu::bangc::fused_indexer_k(
      x,
      latent,
      has_head ? weights_.at("weights_proj.weight") : torch::Tensor(),
      has_k ? weights_.at("wk.weight") : torch::Tensor(),
      has_k ? weights_.at("k_norm.weight") : torch::Tensor(),
      rope,
      positions,
      slots,
      cache,
      head,
      num_valid_input_rows,
      num_valid_compressed_rows);
}

void DeepseekV41IndexerImpl::select_candidate_indices(
    const torch::Tensor& iq,
    const torch::Tensor& iw,
    const torch::Tensor& cache,
    const torch::Tensor& table,
    int64_t length,
    int32_t seq,
    Csa2Context& shared) {
  const int64_t k = std::min<int64_t>(length, args_.index_topk());
  if (k == 0) {
    shared.topk.at(seq) = torch::empty({1, 0}, options_.dtype(torch::kInt64));
    return;
  }
  const int64_t candidate_block = args_.candidate_block_size();
  auto logical =
      (shared.candidates.at(seq).flatten().unsqueeze(1) * candidate_block +
       torch::arange(candidate_block, options_.dtype(torch::kInt64)))
          .flatten();
  auto safe = logical.clamp(0, length - 1);
  auto physical =
      table.to(torch::kInt64)
              .index_select(
                  0, torch::floor_divide(safe, kDeepseekV41MluKVBlockSize)) *
          kDeepseekV41MluKVBlockSize +
      safe.remainder(kDeepseekV41MluKVBlockSize);
  auto keys = cache.index_select(0, physical).to(torch::kFloat32);
  auto scores = (torch::matmul(iq, keys.t()).relu() * iw.unsqueeze(1)).sum(0);
  scores.masked_fill_((logical < 0) | (logical >= length),
                      -std::numeric_limits<double>::infinity());
  auto [values, selected] = scores.topk(std::min<int64_t>(k, logical.numel()));
  auto indices = logical.index_select(0, selected);
  indices.masked_fill_(values == -std::numeric_limits<double>::infinity(), -1);
  shared.topk.at(seq) = indices.view({1, -1});
}

void DeepseekV41IndexerImpl::select_decode_indices(const torch::Tensor& iq,
                                                   const torch::Tensor& iw,
                                                   const torch::Tensor& cache,
                                                   const torch::Tensor& table,
                                                   int64_t length,
                                                   int32_t seq,
                                                   Csa2Context& shared) {
  const int64_t k = std::min<int64_t>(length, args_.index_topk());
  if (k == 0) {
    shared.topk.at(seq) = torch::empty({1, 0}, options_.dtype(torch::kInt64));
    return;
  }
  auto indices =
      torch::empty({1, 1, args_.index_topk()}, options_.dtype(torch::kInt32));
  auto lengths = torch::tensor({length}, options_.dtype(torch::kInt32));
  auto selected_lengths = torch::empty_like(lengths);
  // Identity output pages preserve logical indices for cross-layer sharing;
  // the separate input table addresses the real paged index cache.
  auto logical_pages =
      torch::arange(table.numel(), options_.dtype(torch::kInt32)).view({1, -1});
  kernel::mlu::masked_indexer_select_paged_kv(
      iq.to(options_.dtype())
          .view({1, 1, args_.index_n_heads(), args_.index_head_dim()}),
      cache.view({-1, 1, kDeepseekV41MluKVBlockSize, args_.index_head_dim()}),
      iw.view({1, 1, -1}),
      logical_pages,
      std::nullopt,
      std::nullopt,
      lengths,
      table.view({1, -1}),
      false,
      args_.index_topk(),
      kDeepseekV41MluKVBlockSize,
      1.0,
      std::nullopt,
      std::nullopt,
      indices,
      selected_lengths,
      true,
      1,
      std::nullopt);
  // The paged-indexer contract returns min(causal_mask, index_topk) live rows.
  // Decode uses seq_q=1 and compress_ratio=1, so this is exactly k; the unused
  // capacity after k is never passed to gather or cross-layer consumers.
  shared.topk.at(seq) = indices.view({1, -1}).slice(1, 0, k).to(torch::kInt64);
}

void DeepseekV41IndexerImpl::select_indices(const torch::Tensor& iq,
                                            const torch::Tensor& iw,
                                            const torch::Tensor& keys,
                                            int64_t begin,
                                            int32_t seq,
                                            Csa2Context& shared) {
  const int64_t count = iq.size(0);
  if (keys.size(0) == 0) {
    shared.topk.at(seq) =
        torch::empty({count, 0}, options_.dtype(torch::kInt64));
    if (layer_id_ == args_.candidate_source_layer_id()) {
      shared.candidates.at(seq) =
          torch::empty({count, 0}, options_.dtype(torch::kInt64));
    }
    return;
  }
  const double neg_inf = -std::numeric_limits<double>::infinity();
  std::vector<torch::Tensor> selections, candidates;
  selections.reserve((count + 31) / 32);
  candidates.reserve((count + 31) / 32);
  for (int64_t row = 0; row < count; row += 32) {
    const int64_t rows = std::min<int64_t>(32, count - row);
    torch::Tensor lengths =
        torch::floor_divide(torch::arange(begin + row + 1,
                                          begin + row + rows + 1,
                                          options_.dtype(torch::kInt64)),
                            ratio_);
    torch::Tensor logits =
        (torch::matmul(iq.slice(0, row, row + rows), keys.t()).relu() *
         iw.slice(0, row, row + rows).unsqueeze(-1))
            .sum(1);
    torch::Tensor columns =
        torch::arange(keys.size(0), options_.dtype(torch::kInt64));
    logits.masked_fill_(columns.unsqueeze(0) >= lengths.unsqueeze(1), neg_inf);
    const int64_t candidate_block = args_.candidate_block_size();
    if (layer_id_ == args_.candidate_source_layer_id()) {
      const int64_t blocks =
          (keys.size(0) + candidate_block - 1) / candidate_block;
      torch::Tensor padded = torch::full(
          {rows, blocks * candidate_block}, neg_inf, logits.options());
      padded.slice(1, 0, keys.size(0)).copy_(logits);
      torch::Tensor block_scores =
          std::get<0>(padded.view({rows, blocks, candidate_block}).max(-1));
      torch::Tensor last =
          torch::floor_divide((lengths - 1).clamp_min(0), candidate_block)
              .unsqueeze(1);
      block_scores.scatter_(1, last, std::numeric_limits<double>::infinity());
      const int64_t selected =
          std::min<int64_t>(blocks, args_.candidate_topk_blocks());
      candidates.emplace_back(std::get<1>(block_scores.topk(selected, -1)));
    } else if (layer_id_ > args_.candidate_source_layer_id() &&
               args_.candidate_source_layer_id() >= 0) {
      torch::Tensor pool = shared.candidates.at(seq).slice(0, row, row + rows);
      torch::Tensor allowed = torch::zeros(
          {rows, (keys.size(0) + candidate_block - 1) / candidate_block},
          options_.dtype(torch::kBool));
      allowed.scatter_(1, pool, true);
      logits.masked_fill_(~allowed.index_select(
                              1, torch::floor_divide(columns, candidate_block)),
                          neg_inf);
    }
    const int64_t k = std::min<int64_t>(args_.index_topk(), keys.size(0));
    auto [values, selected] = logits.topk(k, -1);
    selected.masked_fill_(values == neg_inf, -1);
    selections.emplace_back(selected);
  }
  shared.topk.at(seq) = torch::cat(selections, 0);
  if (!candidates.empty()) {
    shared.candidates.at(seq) = torch::cat(candidates, 0);
  }
}

}  // namespace xllm::layer
