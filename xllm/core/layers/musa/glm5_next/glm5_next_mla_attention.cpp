/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/layers/musa/glm5_next/glm5_next_mla_attention.h"

#include <algorithm>
#include <vector>

namespace xllm::layer {
namespace {
bool is_fp8(const torch::Tensor& tensor) {
  return tensor.scalar_type() == torch::kFloat8_e4m3fn;
}

std::string scale_name(const std::string& weight) {
  CHECK(weight.ends_with("weight"));
  return weight.substr(0, weight.size() - 6) + "weight_scale_inv";
}

StateDict weight_dict(const torch::Tensor& weight) {
  return StateDict({{"weight", weight}});
}
}  // namespace

Glm5NextMLAAttentionImpl::Glm5NextMLAAttentionImpl(const ModelArgs& args,
                                                   const ParallelArgs& parallel,
                                                   torch::TensorOptions options,
                                                   bool enable_indexer)
    : hidden_size_(args.hidden_size()),
      q_rank_(args.q_lora_rank()),
      heads_(args.n_heads()),
      token_topk_(args.index_topk()),
      enable_indexer_(enable_indexer),
      options_(std::move(options)) {
  CHECK(options_.device().is_privateuseone());
  CHECK(options_.dtype() == torch::kBFloat16);
  CHECK_EQ(args.qk_nope_head_dim(), 256);
  CHECK_EQ(args.v_head_dim(), 256);
  CHECK_EQ(args.kv_lora_rank(), 512);
  CHECK_EQ(args.qk_rope_head_dim(), 0);
  CHECK_EQ(args.index_head_dim(), 128);
  CHECK_EQ(args.index_n_heads(), 32);
  CHECK_EQ(args.index_kpool(), 4);
  CHECK(args.index_kpool_compress());
  CHECK(args.index_kpool_always_select_tail());
  CHECK_GT(q_rank_, 0);
  CHECK_GT(hidden_size_, 0);
  CHECK_GT(token_topk_, 0);
  CHECK_EQ(token_topk_ % 4, 0);
  CHECK_LE(token_topk_, 2048);
  CHECK_EQ(parallel.cp_size(), 1)
      << "GLM sparse attention currently supports ordinary TP";
  CHECK_EQ(parallel.kv_split_size_effective(), 1);
  CHECK(parallel.tp_group_ != nullptr);
  const int64_t world = parallel.tp_group_->world_size();
  CHECK_EQ(heads_ % world, 0);
  const int64_t local_heads = heads_ / world;
  CHECK(local_heads == 8 || local_heads == 16 || local_heads == 32 ||
        local_heads == 64);
  fused_qkv_a_proj_ = register_module(
      "fused_qkv_a_proj",
      ReplicatedLinear(
          hidden_size_, q_rank_ + 512, false, QuantArgs(), options_));
  q_norm_ = register_module("q_a_layernorm",
                            RMSNorm(q_rank_, args.rms_norm_eps(), options_));
  kv_norm_ = register_module("kv_a_layernorm",
                             RMSNorm(512, args.rms_norm_eps(), options_));
  q_b_proj_ = register_module("q_b_proj",
                              ColumnParallelLinear(q_rank_,
                                                   heads_ * 256,
                                                   false,
                                                   false,
                                                   QuantArgs(),
                                                   parallel.tp_group_,
                                                   options_));
  kv_b_proj_ = register_module("kv_b_proj",
                               ColumnParallelLinear(512,
                                                    heads_ * 512,
                                                    false,
                                                    false,
                                                    QuantArgs(),
                                                    parallel.tp_group_,
                                                    options_));
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(heads_ * 256,
                                              hidden_size_,
                                              false,
                                              true,
                                              true,
                                              QuantArgs(),
                                              parallel.tp_group_,
                                              options_));
  if (enable_indexer_) {
    index_q_ = register_module(
        "indexer_wq_b",
        ReplicatedLinear(q_rank_, 32 * 128, false, QuantArgs(), options_));
    index_kw_ = register_module(
        "indexer_wk_weights_proj",
        ReplicatedLinear(hidden_size_, 128 + 32, false, QuantArgs(), options_));
    index_weights_ =
        register_module("indexer_weights_proj",
                        ReplicatedLinear(hidden_size_,
                                         32,
                                         false,
                                         QuantArgs(),
                                         options_.dtype(torch::kFloat32)));
    index_gate_ = register_module(
        "indexer_compress_gate",
        ReplicatedLinear(hidden_size_, 128, false, QuantArgs(), options_));
    index_norm_ = register_module(
        "indexer_k_norm", RMSNorm(128, 1e-6, options_.dtype(torch::kFloat32)));
    index_norm_->set_layernorm_mode();
    ape_ = register_parameter(
        "indexer_compress_ape",
        torch::empty({4, 128}, options_.dtype(torch::kFloat32)),
        false);
  }
}

std::vector<std::string> Glm5NextMLAAttentionImpl::required_weights() const {
  std::vector<std::string> names{"q_a_proj.weight",
                                 "kv_a_proj_with_mqa.weight",
                                 "q_b_proj.weight",
                                 "kv_b_proj.weight",
                                 "o_proj.weight",
                                 "q_a_layernorm.weight",
                                 "kv_a_layernorm.weight"};
  if (enable_indexer_) {
    for (const std::string& name : {"wq_b.weight",
                                    "wk.weight",
                                    "weights_proj.weight",
                                    "k_norm.weight",
                                    "k_norm.bias",
                                    "index_kpool_compress_gate",
                                    "index_kpool_compress_ape"}) {
      names.emplace_back("indexer." + name);
    }
  }
  return names;
}

bool Glm5NextMLAAttentionImpl::weights_ready() const {
  for (const auto& name : required_weights()) {
    auto it = pending_.find(name);
    if (it == pending_.end()) {
      return false;
    }
    if (is_fp8(it->second) && !pending_.contains(scale_name(name))) {
      return false;
    }
  }
  return true;
}

torch::Tensor Glm5NextMLAAttentionImpl::dense_weight(
    const std::string& name,
    torch::ScalarType dtype) const {
  const auto& stored = pending_.at(name);
  CHECK(stored.scalar_type() == torch::kBFloat16 ||
        stored.scalar_type() == torch::kFloat32 || is_fp8(stored))
      << "Unsupported GLM attention checkpoint dtype for " << name;
  auto weight = stored.to(options_.device());
  if (is_fp8(stored)) {
    CHECK_EQ(weight.dim(), 2);
    auto scales =
        pending_.at(scale_name(name)).to(options_.device(), torch::kFloat32);
    if (scales.numel() == 1) {
      scales = scales.reshape({1, 1});
    } else {
      CHECK_EQ(scales.sizes(),
               torch::IntArrayRef({(weight.size(0) + 127) / 128,
                                   (weight.size(1) + 127) / 128}));
      scales = scales.repeat_interleave(128, 0)
                   .repeat_interleave(128, 1)
                   .narrow(0, 0, weight.size(0))
                   .narrow(1, 0, weight.size(1));
    }
    weight = weight.to(torch::kFloat32) * scales;
  }
  return weight.to(dtype).contiguous();
}

void Glm5NextMLAAttentionImpl::load_state_dict(const StateDict& state) {
  const auto required = required_weights();
  for (const auto& [name, tensor] : state) {
    const bool wanted =
        std::find(required.begin(), required.end(), name) != required.end() ||
        (name.ends_with("weight_scale_inv") &&
         std::find(required.begin(),
                   required.end(),
                   name.substr(0, name.size() - 16) + "weight") !=
             required.end());
    if (!wanted) {
      continue;
    }
    CHECK(!loaded_)
        << "Reloading a finalized GLM sparse attention layer is unsupported";
    // Safetensors mmap lifetime is owned by StateDict, not the Tensor view.
    pending_[name] = tensor.clone();
  }
  if (!loaded_ && weights_ready()) {
    materialize_weights();
    loaded_ = true;
    pending_.clear();
  }
}

void Glm5NextMLAAttentionImpl::materialize_weights() {
  auto dense = [&](const std::string& name) {
    return dense_weight(name, torch::kBFloat16);
  };
  auto qa = dense("q_a_proj.weight");
  auto kva = dense("kv_a_proj_with_mqa.weight");
  CHECK_EQ(qa.sizes(), torch::IntArrayRef({q_rank_, hidden_size_}));
  CHECK_EQ(kva.sizes(), torch::IntArrayRef({512, hidden_size_}));
  fused_qkv_a_proj_->load_state_dict(weight_dict(torch::cat({qa, kva}, 0)));
  q_b_proj_->load_state_dict(weight_dict(dense("q_b_proj.weight")));
  kv_b_proj_->load_state_dict(weight_dict(dense("kv_b_proj.weight")));
  o_proj_->load_state_dict(weight_dict(dense("o_proj.weight")));
  q_norm_->load_state_dict(weight_dict(dense("q_a_layernorm.weight")));
  kv_norm_->load_state_dict(weight_dict(dense("kv_a_layernorm.weight")));
  auto kv_weights = kv_b_proj_->weight().view({-1, 512, 512});
  w_k_ = kv_weights.narrow(1, 0, 256).transpose(1, 2).contiguous();
  w_v_ = kv_weights.narrow(1, 256, 256).transpose(1, 2).contiguous();
  if (enable_indexer_) {
    index_q_->load_state_dict(weight_dict(dense("indexer.wq_b.weight")));
    auto wk = dense("indexer.wk.weight");
    auto head_weight =
        dense_weight("indexer.weights_proj.weight", torch::kFloat32);
    index_kw_->load_state_dict(
        weight_dict(torch::cat({wk, head_weight.to(torch::kBFloat16)}, 0)));
    index_weights_->load_state_dict(weight_dict(head_weight));
    index_gate_->load_state_dict(
        weight_dict(dense("indexer.index_kpool_compress_gate")));
    index_norm_->load_state_dict(StateDict(
        {{"weight", dense_weight("indexer.k_norm.weight", torch::kFloat32)},
         {"bias", dense_weight("indexer.k_norm.bias", torch::kFloat32)}}));
    auto ape =
        dense_weight("indexer.index_kpool_compress_ape", torch::kFloat32);
    CHECK_EQ(ape.sizes(), torch::IntArrayRef({4, 128}));
    ape_.copy_(ape);
  }
}

void Glm5NextMLAAttentionImpl::verify_loaded_weights() const {
  if (!loaded_) {
    for (const auto& name : required_weights()) {
      CHECK(pending_.contains(name))
          << "Missing GLM attention weight: " << name;
      if (is_fp8(pending_.at(name))) {
        CHECK(pending_.contains(scale_name(name)))
            << "Missing FP8 scale for " << name;
      }
    }
  }
  CHECK(loaded_);
  CHECK(fused_qkv_a_proj_->is_weight_loaded());
  CHECK(q_b_proj_->is_weight_loaded());
  CHECK(kv_b_proj_->is_weight_loaded());
  CHECK(o_proj_->is_weight_loaded());
  q_norm_->verify_loaded_weights("q_a_layernorm.");
  kv_norm_->verify_loaded_weights("kv_a_layernorm.");
  if (enable_indexer_) {
    CHECK(index_q_->is_weight_loaded());
    CHECK(index_kw_->is_weight_loaded());
    CHECK(index_weights_->is_weight_loaded());
    CHECK(index_gate_->is_weight_loaded());
    index_norm_->verify_loaded_weights("indexer.k_norm.");
  }
}

KVCache Glm5NextMLAAttentionImpl::create_cache(
    int64_t pages,
    int64_t state_slots,
    const torch::TensorOptions& options,
    bool fp8_main_cache) {
  CHECK_GT(pages, 0);
  CHECK_GT(state_slots, 1);
  CHECK(options.device().is_privateuseone());
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors.key_cache = torch::zeros(
      {pages, 256, 1, fp8_main_cache ? 528 : 512},
      options.dtype(fp8_main_cache ? torch::kUInt8 : torch::kBFloat16));
  tensors.index_cache =
      torch::zeros({pages, 64, 1, 132}, options.dtype(torch::kUInt8));
  tensors.kpool_tail =
      torch::zeros({state_slots, 2, 4, 128}, options.dtype(torch::kBFloat16));
  return KVCache(tensors);
}

std::pair<torch::Tensor, kernel::musa::Glm5NextIndexerSelection>
Glm5NextMLAAttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& metadata,
    KVCache& cache,
    const kernel::musa::Glm5NextIndexerSelection* shared_selection) {
  CHECK(loaded_)
      << "Load and verify the complete GLM attention state before forward";
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(hidden_states.size(1), hidden_size_);
  CHECK_EQ(hidden_states.device(), options_.device());
  CHECK_EQ(hidden_states.scalar_type(), torch::kBFloat16);
  CHECK_EQ(positions.sizes(), torch::IntArrayRef({hidden_states.size(0)}));
  CHECK_EQ(positions.device(), hidden_states.device());
  c10::DeviceGuard device_guard(hidden_states.device());
  CHECK(!kernel::musa::is_stream_capturing() ||
        (metadata.enable_cuda_graph && !metadata.is_prefill &&
         !metadata.is_chunked_prefill && !metadata.is_spec_verify));
  CHECK(enable_indexer_ ? shared_selection == nullptr
                        : shared_selection != nullptr);
  auto main_cache = cache.get_k_cache();
  CHECK_EQ(main_cache.dim(), 4);
  CHECK_EQ(main_cache.size(1), 256);
  auto projected = fused_qkv_a_proj_->forward(hidden_states);
  auto q_raw = projected.narrow(1, 0, q_rank_).contiguous();
  auto c_raw = projected.narrow(1, q_rank_, 512).contiguous();
  auto q_norm = std::get<0>(q_norm_->forward(q_raw));
  auto latent = std::get<0>(kv_norm_->forward(c_raw));
  auto q = q_b_proj_->forward(q_norm).view({hidden_states.size(0), -1, 256});
  kernel::musa::Glm5NextIndexerSelection selection;
  auto pos = positions.to(torch::kInt64).contiguous();
  auto table = metadata.block_table.to(torch::kInt32).contiguous();
  CHECK_EQ(table.device(), hidden_states.device());
  if (enable_indexer_) {
    auto iq = index_q_->forward(q_norm).view({hidden_states.size(0), 32, 128});
    auto kw = index_kw_->forward(hidden_states);
    auto key_fp32 = kw.narrow(1, 0, 128).to(torch::kFloat32).contiguous();
    auto key = std::get<0>(index_norm_->forward(key_fp32))
                   .to(torch::kBFloat16)
                   .contiguous();
    auto head_weights =
        index_weights_->forward(hidden_states.to(torch::kFloat32));
    auto gate = index_gate_->forward(hidden_states);
    auto index_cache = cache.get_index_cache();
    auto tail = cache.get_kpool_tail();
    auto cu = metadata.q_cu_seq_lens.defined()
                  ? metadata.q_cu_seq_lens.to(torch::kInt32).contiguous()
                  : torch::arange(table.size(0) + 1,
                                  hidden_states.options().dtype(torch::kInt32));
    CHECK(metadata.q_cu_seq_lens.defined() ||
          hidden_states.size(0) == table.size(0))
        << "Multi-token batches require packed query sequence boundaries";
    auto ids = metadata.linear_state_indices.to(torch::kInt32).contiguous();
    auto read_ids =
        metadata.linear_state_read_indices.defined()
            ? metadata.linear_state_read_indices.to(torch::kInt32).contiguous()
            : ids;
    selection = kernel::musa::glm5_next_indexer(iq.contiguous(),
                                                key,
                                                gate.contiguous(),
                                                head_weights.contiguous(),
                                                ape_,
                                                pos,
                                                cu,
                                                ids,
                                                read_ids,
                                                table,
                                                index_cache,
                                                tail,
                                                metadata.max_seq_len,
                                                token_topk_);
  } else {
    selection = *shared_selection;
    CHECK_EQ(selection.tokens.size(1), ((token_topk_ + 3 + 127) / 128) * 128);
  }
  auto slots = metadata.slot_mapping.to(torch::kInt64).contiguous();
  CHECK_EQ(slots.sizes(), pos.sizes());
  slots =
      torch::where(selection.rows.ge(0), slots, torch::full_like(slots, -1));
  auto attention = glm5_next_mla_attention(q,
                                           latent,
                                           w_k_,
                                           w_v_,
                                           selection.tokens,
                                           selection.lengths,
                                           selection.rows,
                                           pos,
                                           table,
                                           slots,
                                           main_cache);
  // MUSA Linear may reuse its matmul scratch on the next invocation. The
  // attention layer returns owned activations so callers can retain a prefill
  // result while extending the same layer's cache with subsequent forwards.
  auto output = o_proj_->forward(attention.flatten(1, 2)).clone();
  return {output, std::move(selection)};
}
}  // namespace xllm::layer
