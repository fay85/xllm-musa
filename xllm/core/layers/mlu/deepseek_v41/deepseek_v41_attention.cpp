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

#include <algorithm>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "kernels/mlu/bangc/include/deepseek_v41_constants.h"
#include "kernels/ops_api.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_attention_utils.h"

namespace xllm::layer {
namespace {
using deepseek_v41::dense_projection;
using deepseek_v41::norm;
using deepseek_v41::rotate;
torch::Tensor flat_cache(const torch::Tensor& cache) {
  return cache.view({-1, cache.size(-1)});
}

bool contains(const std::vector<int32_t>& ids, int32_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}
}  // namespace

DeepseekV41AttentionImpl::DeepseekV41AttentionImpl(
    const ModelContext& context,
    int32_t layer_id,
    const std::string& layer_stack)
    : args_(context.get_model_args()),
      layer_id_(layer_id),
      tp_group_(context.get_parallel_args().tp_group_),
      rank_(tp_group_->rank()),
      tp_size_(tp_group_->world_size()),
      options_(context.get_tensor_options()) {
  CHECK_EQ(args_.hidden_size(), 5120);
  CHECK_EQ(args_.q_lora_rank(), 1280);
  CHECK_EQ(args_.head_dim(), 512);
  CHECK_EQ(args_.qk_rope_head_dim(), 64);
  ratio_ = args_.compress_ratios().at(layer_id);
  CHECK(ratio_ == 0 || ratio_ == 1 || ratio_ == 2);
  CHECK_GT(args_.o_groups(), 0);
  CHECK_EQ(args_.o_groups() % tp_size_, 0);
  CHECK_EQ(args_.n_heads(), 8 * args_.o_groups());
  CHECK_EQ(args_.o_lora_rank(), 1024);
  const int64_t groups = args_.o_groups() / tp_size_;
  CHECK(groups == 1 || groups == 2 || groups == 4 || groups == 8);
  for (int32_t source : args_.kv_source_layer_ids()) {
    if (source <= layer_id) {
      source_ = std::max(source_, source);
    }
  }
  index_source_ = contains(args_.index_source_layer_ids(), layer_id);
  indexer_ = register_module("indexer", DeepseekV41Indexer(context, layer_id));
  const auto& quant = context.get_quant_args();
  const std::string prefix = layer_stack + "." + std::to_string(layer_id);
  attn_prefix_ = prefix + ".attn.";
  wq_b_ = register_module(
      "wq_b",
      ColumnParallelLinear(args_.q_lora_rank(),
                           args_.n_heads() * args_.head_dim(),
                           false,
                           false,
                           quant.for_module(prefix + ".attn.wq_b"),
                           tp_group_,
                           options_));
}

void DeepseekV41AttentionImpl::load_output_weights(const StateDict& state) {
  const int64_t groups = args_.o_groups();
  const int64_t rank = args_.o_lora_rank();
  const int64_t hidden = args_.hidden_size();
  const int64_t width = args_.head_dim() * args_.n_heads() / groups;
  const int64_t channels = groups * rank;
  const auto load = [&](const std::string& key,
                        torch::IntArrayRef shape,
                        int32_t shard_dim,
                        bool weight) {
    auto tensor = state.get_tensor(key);
    if (!tensor.defined()) {
      return;
    }
    CHECK(tensor.sizes() == shape) << attn_prefix_ + key << " expected "
                                   << shape << ", got " << tensor.sizes();
    if (weight) {
      CHECK_EQ(tensor.scalar_type(), torch::kInt8) << attn_prefix_ + key;
    } else {
      CHECK(tensor.scalar_type() == torch::kBFloat16 ||
            tensor.scalar_type() == torch::kFloat32)
          << attn_prefix_ + key;
    }
    if (shard_dim >= 0) {
      tensor = tensor.chunk(tp_size_, shard_dim).at(rank_);
    }
    if (!weight) {
      tensor = tensor.flatten().to(torch::kFloat32);
    } else if (key == "wo_a.weight") {
      tensor = tensor.view({groups / tp_size_, rank, width});
    }
    weights_[key] = tensor.to(options_.device()).contiguous();
  };
  load("wo_a.weight", {channels, width}, 0, true);
  load("wo_a.weight_scale", {channels, 1}, 0, false);
  load("wo_a.smooth", {width}, -1, false);
  load("wo_b.weight", {hidden, channels}, 1, true);
  load("wo_b.weight_scale", {hidden, 1}, -1, false);
  load("wo_b.smooth", {channels}, 0, false);
}

void DeepseekV41AttentionImpl::load_state_dict(const StateDict& state) {
  indexer_->load_state_dict(state.get_dict_with_prefix("indexer."));
  wq_b_->load_state_dict(state.get_dict_with_prefix("wq_b."));
  load_output_weights(state);
  for (const std::string name : {"wq_a.weight",
                                 "wkv.weight",
                                 "q_norm.weight",
                                 "kv_norm.weight",
                                 "attn_sink",
                                 "compressor.wkv.weight",
                                 "compressor.wgate.weight",
                                 "compressor.norm.weight"}) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    if (name == "attn_sink") {
      value = value.chunk(tp_size_, 0).at(rank_);
      value = value.to(torch::kFloat32);
    }
    weights_[name] = value.to(options_.device()).contiguous();
  }
}

void DeepseekV41AttentionImpl::verify_loaded_weights() const {
  const auto& prefix = attn_prefix_;
  for (const std::string name : {"wq_a.weight",
                                 "wkv.weight",
                                 "q_norm.weight",
                                 "kv_norm.weight",
                                 "wo_a.weight",
                                 "wo_a.weight_scale",
                                 "wo_a.smooth",
                                 "wo_b.weight",
                                 "wo_b.weight_scale",
                                 "wo_b.smooth",
                                 "attn_sink"}) {
    CHECK(weights_.count(name)) << "Missing weight " << prefix + name;
  }
  CHECK(wq_b_->is_weight_loaded())
      << "Missing weight or quantization state for " << prefix + "wq_b.weight";
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

torch::Tensor DeepseekV41AttentionImpl::project_query(
    const torch::Tensor& qr,
    const torch::Tensor& positions,
    const torch::Tensor& rope) {
  const auto weight = wq_b_->weight();
  if (kernel::mlu::bangc::supports_dsa_qb_width(weight.size(0))) {
    if (weight.scalar_type() == torch::kBFloat16) {
      return kernel::mlu::bangc::fused_dsa_qb(qr, weight, positions, rope);
    }
    const auto smooth = wq_b_->smooth();
    if (wq_b_->uses_w8a8_dynamic_quant() && smooth.has_value()) {
      return kernel::mlu::bangc::fused_dsa_qb(
          qr,
          weight,
          positions,
          rope,
          wq_b_->w8a8_dynamic_weight_scale(),
          *smooth);
    }
  }
  const int64_t heads = args_.n_heads() / tp_size_;
  return rotate(wq_b_->forward(qr).view({qr.size(0), heads, args_.head_dim()}),
                positions,
                rope,
                args_.qk_rope_head_dim());
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
  const float eps = args_.rms_norm_eps();
  const int32_t c = source_ == layer_id_ ? ratio_ : 0;
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  const auto pos = positions.to(torch::kInt32).contiguous();
  auto [kv, qr, packed] = kernel::mlu::bangc::fused_dsa_projection(
      x,
      weights_.at("wkv.weight"),
      weights_.at("wq_a.weight"),
      weights_.at("kv_norm.weight"),
      weights_.at("q_norm.weight"),
      pos,
      rope,
      eps,
      c > 0 ? weights_.at("compressor.wkv.weight") : torch::Tensor(),
      c == 2 ? weights_.at("compressor.wgate.weight") : torch::Tensor());
  std::optional<parallel_state::GatherAsyncCtx> kv_gather, compressor_gather;
  if (cp_context != nullptr) {
    kv_gather = mlu_v4_cp::launch_gather_rows(kv, *cp_context);
    if (packed.defined()) {
      compressor_gather = mlu_v4_cp::launch_gather_rows(packed, *cp_context);
    }
  }
  torch::Tensor iq;
  torch::Tensor q = project_query(qr, pos, rope);
  if (index_source_) {
    iq = indexer_->project_query(qr, pos, rope);
  }
  if (kv_gather) {
    kv = mlu_v4_cp::finish_gather_restore(std::move(*kv_gather), *cp_context);
  }
  std::optional<torch::Tensor> row_map;
  if (compressor_gather) {
    packed = parallel_state::finish_gather(std::move(*compressor_gather));
    row_map = cp_context->restore_indices.to(torch::kInt32);
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
      CHECK_EQ(tables.size(), 2u);
      metadata.compressed_block_table = tables[1];
      metadata.input_row_map = std::move(row_map);
      std::optional<kernel::mlu::bangc::CompressKvPoolingInputs> pooling;
      if (pooled) {
        pooling.emplace();
        pooling->scores = packed.slice(-1, dim);
        pooling->state_block_table = tables[0];
        pooling->state_raw = cache.get_compress_kv_state();
        pooling->state_score = cache.get_compress_score_state();
        pooling->save_full_tail = args_.num_speculative_tokens() > 0;
      }
      kernel::mlu::bangc::CompressKvOptions options;
      options.ratio = ratio_;
      options.rms_norm_eps = eps;
      kernel::mlu::bangc::fused_compress_kv_cache(
          c == 2 ? packed.slice(-1, 0, dim) : packed,
          weights_.at("compressor.norm.weight"),
          rope,
          metadata,
          pooling,
          cache.get_k_cache().view(
              {-1,
               kernel::mlu::bangc::kDeepseekV41MluKVBlockSize / ratio_,
               512}),
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
  if (ratio_ > 0 && index_source_) {
    indexer_->select_indices(iq,
                             iw,
                             caches[source_].get_index_cache(),
                             pos,
                             shared.query_requests,
                             deepseek_v41::uses_decode_attention(input),
                             shared);
  }
  namespace bangc = kernel::mlu::bangc;
  using Stage = bangc::PagedAttentionStage;
  Stage stage = Stage::PREFILL;
  if (input.use_dflash_proposal_attention) {
    stage = Stage::DRAFT;
  } else if (input.is_spec_verify || input.meta.num_speculative_tokens > 0 ||
             (input.meta.batch_forward_type.is_decode() &&
              input.meta.q_max_seq_len > 1)) {
    stage = Stage::VERIFY;
  } else if (input.meta.batch_forward_type.is_decode()) {
    stage = Stage::DECODE;
  }
  bangc::PagedAttentionMetadata metadata;
  metadata.swa_table = shared.swa_block_table;
  metadata.swa_columns = shared.swa_columns;
  metadata.requests = shared.query_requests;
  metadata.positions = pos;
  metadata.prefix = shared.query_prefix;
  metadata.begins = shared.chunk_begins;
  metadata.valid_rows = shared.num_valid_input_rows;
  torch::Tensor compressed;
  if (ratio_ > 0) {
    compressed = flat_cache(caches[source_].get_k_cache());
    metadata.compressed_table = shared.block_tables.at(1);
    metadata.topk = shared.packed_topk;
  }
  auto output = bangc::fused_paged_sparsed_attention(
      stage,
      q,
      kv,
      flat_cache(caches[layer_id_].get_swa_cache()),
      compressed,
      weights_.at("attn_sink"),
      metadata,
      ratio_,
      args_.window_size());
  return project_output(output, pos, rope);
}

torch::Tensor DeepseekV41AttentionImpl::project_output(
    const torch::Tensor& output,
    const torch::Tensor& positions,
    const torch::Tensor& rope) {
  auto [values, scales] = kernel::mlu::bangc::fused_dsa_output_projection(
      output,
      positions,
      rope,
      weights_.at("wo_a.weight"),
      weights_.at("wo_a.weight_scale"),
      weights_.at("wo_a.smooth"),
      weights_.at("wo_b.smooth"));
  kernel::ScaledMatmulParams matmul;
  matmul.a = values;
  matmul.a_scale = scales;
  matmul.b = weights_.at("wo_b.weight");
  matmul.b_scale = weights_.at("wo_b.weight_scale");
  matmul.output_dtype = output.scalar_type();
  matmul.a_quant_bit_size = 8;
  matmul.beta = 0.0;
  auto result = kernel::scaled_matmul(matmul);
  return tp_size_ > 1 ? parallel_state::reduce(result, tp_group_) : result;
}
}  // namespace xllm::layer
