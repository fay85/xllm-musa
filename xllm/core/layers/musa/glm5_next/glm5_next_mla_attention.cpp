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
#include <limits>
#include <unordered_set>
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/util/env_var.h"

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
// Plain activation ownership and round-robin query ownership are separate.
// Only small scheduling metadata is inspected on the host in this eager path.
class DsaCpLayout final {
 public:
  DsaCpLayout(const AttentionMetadata& metadata,
              const torch::Tensor& positions,
              int64_t local_rows,
              ProcessGroup& group,
              const KVCache& cache,
              bool enable_indexer)
      : group_(group),
        device_(positions.device()),
        offsets_(metadata.glm5_prefill_row_offsets) {
    CHECK(metadata.is_prefill || metadata.is_chunked_prefill);
    CHECK(!metadata.enable_cuda_graph && !kernel::musa::is_stream_capturing());
    CHECK(!metadata.is_dummy && !metadata.is_mixed && !metadata.is_spec_verify);
    CHECK_EQ(metadata.glm5_prefill_cp_rank, group.rank());
    CHECK_EQ(metadata.glm5_prefill_cp_size, group.world_size());
    total = metadata.glm5_prefill_global_rows;
    CHECK_GT(total, 0);
    CHECK_LE(total, std::numeric_limits<int32_t>::max());
    CHECK_EQ(offsets_.size(), group.world_size() + 1);
    counts.reserve(group.world_size());
    for (int32_t rank = 0; rank < group.world_size(); ++rank) {
      CHECK_EQ(offsets_[rank],
               rank * (total / group.world_size()) +
                   std::min<int64_t>(rank, total % group.world_size()));
      counts.emplace_back(total / group.world_size() +
                          (rank < total % group.world_size()));
    }
    CHECK_EQ(offsets_.back(), total);
    begin = offsets_[group.rank()];
    count = counts[group.rank()];
    CHECK_EQ(metadata.glm5_prefill_local_row_begin, begin);
    CHECK_EQ(metadata.glm5_prefill_local_row_end, begin + count);
    CHECK_EQ(local_rows, count) << "DSA CP requires Plain local rows";
    CHECK_EQ(positions.numel(), count);
    global_positions = parallel_state::gather(
        positions.to(torch::kInt64).contiguous(), &group_, counts);
    auto pos_host = global_positions.cpu();
    auto cu_host =
        metadata.q_cu_seq_lens.to(torch::kCPU, torch::kInt32).contiguous();
    auto table_host =
        metadata.block_table.to(torch::kCPU, torch::kInt32).contiguous();
    auto ids_host = metadata.linear_state_indices.to(torch::kCPU, torch::kInt32)
                        .contiguous();
    auto read_host =
        metadata.linear_state_read_indices.defined()
            ? metadata.linear_state_read_indices.to(torch::kCPU, torch::kInt32)
                  .contiguous()
            : ids_host;
    auto slots_host =
        metadata.slot_mapping.to(torch::kCPU, torch::kInt64).contiguous();
    CHECK_EQ(table_host.dim(), 2);
    const int64_t batch = table_host.size(0);
    CHECK_GT(batch, 0);
    CHECK_EQ(cu_host.numel(), batch + 1);
    CHECK_EQ(ids_host.numel(), batch);
    CHECK_EQ(read_host.numel(), batch);
    CHECK_EQ(slots_host.numel(), total);
    CHECK_GT(metadata.max_seq_len, 0);
    CHECK_LE(metadata.max_seq_len, table_host.size(1) * 256);
    const int32_t* cu = cu_host.data_ptr<int32_t>();
    CHECK_EQ(cu[0], 0);
    CHECK_EQ(cu[batch], total);
    std::unordered_set<int32_t> state_owners;
    std::unordered_set<int64_t> write_slots;
    std::vector<int32_t> requests(total);
    std::vector<int64_t> pool_slots;
    pool_slots.reserve(total / 4 + batch);
    pool_rows.reserve(total / 4 + batch);
    tail_rows.reserve(batch * 4);
    std::vector<int32_t> tail_cu{0};
    tail_cu.reserve(batch + 1);
    for (int64_t request = 0; request < batch; ++request) {
      CHECK_LT(cu[request], cu[request + 1]);
      const int32_t id = ids_host.data_ptr<int32_t>()[request];
      CHECK_GT(id, 0);
      if (enable_indexer) {
        CHECK_LT(id, cache.get_kpool_tail().size(0));
      }
      CHECK(state_owners.insert(id).second);
      CHECK_GE(read_host.data_ptr<int32_t>()[request], 0);
      if (enable_indexer) {
        CHECK_LT(read_host.data_ptr<int32_t>()[request],
                 cache.get_kpool_tail().size(0));
      }
      const int64_t start = pos_host.data_ptr<int64_t>()[cu[request]];
      CHECK_GE(start, 0);
      CHECK_LE(start + cu[request + 1] - cu[request], metadata.max_seq_len);
      for (int64_t row = cu[request]; row < cu[request + 1]; ++row) {
        const int64_t position = pos_host.data_ptr<int64_t>()[row];
        CHECK_EQ(position, start + row - cu[request])
            << "CP requires consecutive unpadded request positions";
        const int32_t page =
            table_host.data_ptr<int32_t>()[request * table_host.size(1) +
                                           position / 256];
        CHECK_GE(page, 0);
        CHECK_LT(page, cache.get_k_cache().size(0));
        const int64_t slot = page * 256 + position % 256;
        CHECK_EQ(slots_host.data_ptr<int64_t>()[row], slot);
        CHECK(write_slots.insert(slot).second);
        requests[row] = request;
        if (position % 4 == 3) {
          pool_rows.emplace_back(row);
          pool_slots.emplace_back(page * 64 + (position % 256) / 4);
        }
      }
      for (int64_t row = std::max<int64_t>(cu[request], cu[request + 1] - 4);
           row < cu[request + 1];
           ++row) {
        tail_rows.emplace_back(row);
      }
      tail_cu.emplace_back(tail_rows.size());
    }
    rows = torch::tensor(requests, torch::kInt32).to(device_);
    pool_physical = indices(pool_slots);
    tail_boundaries = torch::tensor(tail_cu, torch::kInt32).to(device_);
    interleaved =
        torch::arange(count, positions.options().dtype(torch::kInt64)) *
            group.world_size() +
        group.rank();
    std::vector<int64_t> permutation;
    permutation.reserve(count);
    send_counts_.reserve(group.world_size());
    recv_counts_.reserve(group.world_size());
    for (int32_t peer = 0; peer < group.world_size(); ++peer) {
      const int64_t before = permutation.size();
      for (int64_t row = begin; row < begin + count; ++row) {
        if (row % group.world_size() == peer) {
          permutation.emplace_back(row - begin);
        }
      }
      send_counts_.emplace_back(permutation.size() - before);
      int64_t received = 0;
      for (int64_t row = offsets_[peer]; row < offsets_[peer + 1]; ++row) {
        received += row % group.world_size() == group.rank();
      }
      recv_counts_.emplace_back(received);
    }
    permutation_ = indices(permutation);
    inverse_ = torch::argsort(permutation_);
    std::vector<int64_t> restore(total);
    for (int64_t row = 0; row < total; ++row) {
      restore[row] =
          offsets_[row % group.world_size()] + row / group.world_size();
    }
    restore_ = indices(restore);
  }

  torch::Tensor indices(const std::vector<int64_t>& values) const {
    return torch::tensor(values, torch::kInt64).to(device_);
  }
  torch::Tensor scatter(const torch::Tensor& plain) const {
    auto input = plain.index_select(0, permutation_).contiguous();
    auto shape = plain.sizes().vec();
    shape[0] = interleaved.numel();
    auto output = torch::empty(shape, plain.options());
    group_.all_to_all_single(output, input, recv_counts_, send_counts_);
    return output;
  }
  torch::Tensor restore(const torch::Tensor& scattered) const {
    auto shape = scattered.sizes().vec();
    shape[0] = count;
    auto output = torch::empty(shape, scattered.options());
    group_.all_to_all_single(
        output, scattered.contiguous(), send_counts_, recv_counts_);
    return output.index_select(0, inverse_);
  }
  torch::Tensor gather_interleaved(const torch::Tensor& value) const {
    return parallel_state::gather(value.contiguous(), &group_, counts)
        .index_select(0, restore_);
  }
  std::vector<int64_t> local_rows(const std::vector<int64_t>& selected) const {
    const auto first =
        std::lower_bound(selected.begin(), selected.end(), begin);
    const auto last = std::lower_bound(first, selected.end(), begin + count);
    return {first, last};
  }
  torch::Tensor gather_packets(const torch::Tensor& local,
                               const std::vector<int64_t>& selected) const {
    if (selected.empty()) {
      return local;
    }
    std::vector<int32_t> sizes;
    sizes.reserve(group_.world_size());
    for (int32_t rank = 0; rank < group_.world_size(); ++rank) {
      sizes.emplace_back(
          std::lower_bound(
              selected.begin(), selected.end(), offsets_[rank + 1]) -
          std::lower_bound(selected.begin(), selected.end(), offsets_[rank]));
    }
    return parallel_state::gather(local.contiguous(), &group_, sizes);
  }
  torch::Tensor with_halo(const torch::Tensor& local) const {
    std::vector<int64_t> send, receive, selected;
    send.reserve(group_.world_size());
    receive.reserve(group_.world_size());
    selected.reserve(3 * group_.world_size());
    const int64_t halo_begin = std::max<int64_t>(0, begin - 3);
    for (int32_t peer = 0; peer < group_.world_size(); ++peer) {
      const int64_t first = std::max<int64_t>(begin, offsets_[peer] - 3);
      const int64_t last = std::min<int64_t>(begin + count, offsets_[peer]);
      send.emplace_back(std::max<int64_t>(0, last - first));
      for (int64_t row = first; row < last; ++row) {
        selected.emplace_back(row - begin);
      }
      receive.emplace_back(
          std::max<int64_t>(0,
                            std::min<int64_t>(begin, offsets_[peer + 1]) -
                                std::max<int64_t>(halo_begin, offsets_[peer])));
    }
    auto halo =
        torch::empty({begin - halo_begin, local.size(1)}, local.options());
    group_.all_to_all_single(
        halo, local.index_select(0, indices(selected)), receive, send);
    return torch::cat({halo, local}, 0);
  }
  int64_t total = 0;
  int64_t begin = 0;
  int64_t count = 0;
  std::vector<int32_t> counts;
  std::vector<int64_t> pool_rows;
  std::vector<int64_t> tail_rows;
  torch::Tensor global_positions, rows, interleaved, pool_physical,
      tail_boundaries;

 private:
  ProcessGroup& group_;
  torch::Device device_;
  std::vector<int32_t> offsets_;
  std::vector<int64_t> send_counts_, recv_counts_;
  torch::Tensor permutation_, inverse_, restore_;
};

// MATE's FP8 pool page stores payload and scale in separate planes. Transfer
// one compact 132-byte record per newly completed pool, never the full cache.
torch::Tensor pool_byte_indices(const torch::Tensor& physical) {
  auto column = torch::arange(132, physical.options()).unsqueeze(0);
  auto page = physical.div(64, "floor").unsqueeze(1);
  auto row = physical.remainder(64).unsqueeze(1);
  return (page * (64 * 132) + torch::where(column.lt(128),
                                           row * 128 + column,
                                           64 * 128 + row * 4 + column - 128))
      .flatten();
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
  if (parallel.cp_size() > 1) {
    CHECK(util::get_bool_env("XLLM_GLM5_PREFILL_CP", false));
    CHECK_EQ(parallel.dp_size(), 1);
    CHECK_EQ(parallel.world_size(), parallel.cp_size());
    CHECK(parallel.tp_group_ != nullptr);
    CHECK_EQ(parallel.tp_group_->world_size(), 1);
    cp_group_ = parallel.cp_group_;
    CHECK(cp_group_ != nullptr);
    CHECK_EQ(cp_group_->world_size(), parallel.cp_size());
  }
  CHECK_EQ(parallel.kv_split_size_effective(), 1)
      << "GLM DSA requires replicated compressed caches: kv_split_size=1";
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
        (metadata.enable_cuda_graph && !metadata.is_spec_verify));
  CHECK(enable_indexer_ ? shared_selection == nullptr
                        : shared_selection != nullptr);
  CHECK_EQ(metadata.glm5_prefill_cp_enabled, cp_group_ != nullptr);
  if (cp_group_ != nullptr) {
    return forward_cp(
        positions, hidden_states, metadata, cache, shared_selection);
  }
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

std::pair<torch::Tensor, kernel::musa::Glm5NextIndexerSelection>
Glm5NextMLAAttentionImpl::forward_cp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& metadata,
    KVCache& cache,
    const kernel::musa::Glm5NextIndexerSelection* shared_selection) {
  namespace km = kernel::musa;
  CHECK_EQ(cache.get_k_cache().dim(), 4);
  if (enable_indexer_) {
    CHECK_EQ(cache.get_kpool_tail().dim(), 4);
    CHECK_EQ(cache.get_kpool_tail().sizes().slice(1),
             torch::IntArrayRef({2, 4, 128}));
    CHECK_EQ(cache.get_index_cache().dim(), 4);
    CHECK_EQ(cache.get_index_cache().size(0), cache.get_k_cache().size(0));
  }
  DsaCpLayout layout(metadata,
                     positions,
                     hidden_states.size(0),
                     *cp_group_,
                     cache,
                     enable_indexer_);
  auto table = metadata.block_table.to(torch::kInt32).contiguous();
  auto ids = metadata.linear_state_indices.to(torch::kInt32).contiguous();
  auto read_ids =
      metadata.linear_state_read_indices.defined()
          ? metadata.linear_state_read_indices.to(torch::kInt32).contiguous()
          : ids;
  auto cu = metadata.q_cu_seq_lens.to(torch::kInt32).contiguous();
  auto main_cache = cache.get_k_cache();
  CHECK_EQ(main_cache.size(1), 256);
  CHECK_EQ(main_cache.size(2), 1);
  CHECK_EQ(main_cache.size(3),
           main_cache.scalar_type() == torch::kUInt8 ? 528 : 512);
  // Compression uses Plain rows, whose short predecessor halo reconstructs
  // every rate-4 window even when a pool crosses several tiny/empty shards.
  if (enable_indexer_) {
    auto key_gate = torch::empty({layout.count, 256}, hidden_states.options());
    if (layout.count > 0) {
      auto kw = index_kw_->forward(hidden_states);
      auto key_fp32 = kw.narrow(1, 0, 128).to(torch::kFloat32).contiguous();
      key_gate.narrow(1, 0, 128).copy_(
          std::get<0>(index_norm_->forward(key_fp32)));
      key_gate.narrow(1, 128, 128).copy_(index_gate_->forward(hidden_states));
    }
    auto extended = layout.with_halo(key_gate);
    auto index_cache = cache.get_index_cache();
    auto tail = cache.get_kpool_tail();
    km::glm5_next_indexer_update_cp(extended.narrow(1, 0, 128).contiguous(),
                                    extended.narrow(1, 128, 128).contiguous(),
                                    ape_,
                                    layout.global_positions,
                                    cu,
                                    ids,
                                    read_ids,
                                    table,
                                    index_cache,
                                    tail,
                                    layout.begin,
                                    layout.count,
                                    std::max<int64_t>(0, layout.begin - 3));
    const auto owned_pools = layout.local_rows(layout.pool_rows);
    const int64_t pool_begin =
        std::lower_bound(
            layout.pool_rows.begin(), layout.pool_rows.end(), layout.begin) -
        layout.pool_rows.begin();
    auto local_bytes = pool_byte_indices(
        layout.pool_physical.narrow(0, pool_begin, owned_pools.size()));
    auto packets = index_cache.flatten()
                       .index_select(0, local_bytes)
                       .view({static_cast<int64_t>(owned_pools.size()), 132});
    auto gathered = layout.gather_packets(packets, layout.pool_rows);
    index_cache.flatten().index_copy_(
        0, pool_byte_indices(layout.pool_physical), gathered.flatten());
    auto tail_local_rows =
        layout.indices(layout.local_rows(layout.tail_rows)) - layout.begin;
    auto tail_data = layout.gather_packets(
        key_gate.index_select(0, tail_local_rows), layout.tail_rows);
    km::glm5_next_indexer_update_tail(
        tail_data.narrow(1, 0, 128).contiguous(),
        tail_data.narrow(1, 128, 128).contiguous(),
        layout.global_positions.index_select(0,
                                             layout.indices(layout.tail_rows)),
        layout.tail_boundaries,
        ids,
        read_ids,
        tail);
  }
  // Reuse the activation exchange for shared selection metadata when present.
  // Byte packing is lossless for BF16 activations and int32 token IDs.
  const int64_t width = ((token_topk_ + 3 + 127) / 128) * 128;
  auto pack_selection = [&](const km::Glm5NextIndexerSelection& selection) {
    CHECK_EQ(selection.tokens.sizes(),
             torch::IntArrayRef({layout.count, width}));
    CHECK_EQ(selection.lengths.numel(), layout.count);
    CHECK_EQ(selection.rows.numel(), layout.count);
    return torch::cat({selection.tokens,
                       selection.lengths.unsqueeze(1),
                       selection.rows.unsqueeze(1)},
                      1)
        .contiguous()
        .view(torch::kUInt8);
  };
  auto input_bytes = hidden_states.contiguous().view(torch::kUInt8);
  if (shared_selection != nullptr) {
    input_bytes =
        torch::cat({input_bytes, pack_selection(*shared_selection)}, 1);
  }
  auto scattered = layout.scatter(input_bytes);
  auto local_hidden = scattered.narrow(1, 0, hidden_size_ * 2)
                          .contiguous()
                          .view(torch::kBFloat16);
  const int64_t local_count = layout.interleaved.numel();
  auto pos = layout.global_positions.index_select(0, layout.interleaved);
  auto query_rows = layout.rows.index_select(0, layout.interleaved);
  auto latent = torch::empty({local_count, 512}, hidden_states.options());
  torch::Tensor q_norm, q;
  if (local_count > 0) {
    auto projected = fused_qkv_a_proj_->forward(local_hidden);
    auto q_raw = projected.narrow(1, 0, q_rank_).contiguous();
    auto c_raw = projected.narrow(1, q_rank_, 512).contiguous();
    q_norm = std::get<0>(q_norm_->forward(q_raw));
    latent.copy_(std::get<0>(kv_norm_->forward(c_raw)));
    q = q_b_proj_->forward(q_norm).view({local_count, heads_, 256});
  }
  auto global_latent = layout.gather_interleaved(latent);
  auto slots = metadata.slot_mapping.to(torch::kInt64).contiguous();
  if (main_cache.scalar_type() == torch::kUInt8) {
    km::flash_mla_sparse_fp8_cache_write(global_latent, slots, main_cache);
  } else {
    km::flash_mla_sparse_cache_write(global_latent, slots, main_cache);
  }
  km::Glm5NextIndexerSelection selection;
  if (enable_indexer_) {
    auto iq = torch::empty({local_count, 32, 128}, hidden_states.options());
    auto weights = torch::empty({local_count, 32},
                                hidden_states.options().dtype(torch::kFloat32));
    if (local_count > 0) {
      iq.copy_(index_q_->forward(q_norm).view({local_count, 32, 128}));
      weights.copy_(index_weights_->forward(local_hidden.to(torch::kFloat32)));
    }
    selection = km::glm5_next_indexer_select(iq,
                                             weights,
                                             pos,
                                             query_rows,
                                             table,
                                             cache.get_index_cache(),
                                             metadata.max_seq_len,
                                             token_topk_);
  } else {
    auto values = scattered.narrow(1, hidden_size_ * 2, (width + 2) * 4)
                      .contiguous()
                      .view(torch::kInt32);
    selection = {values.narrow(1, 0, width).contiguous(),
                 values.select(1, width).contiguous(),
                 values.select(1, width + 1).contiguous()};
  }
  auto output =
      torch::empty({local_count, hidden_size_}, hidden_states.options());
  if (local_count > 0) {
    auto attention =
        glm5_next_mla_attention(q,
                                latent,
                                w_k_,
                                w_v_,
                                selection.tokens,
                                selection.lengths,
                                selection.rows,
                                pos,
                                table,
                                slots.index_select(0, layout.interleaved),
                                main_cache,
                                /*write_cache=*/false);
    output.copy_(o_proj_->forward(attention.flatten(1, 2)));
  }
  auto selected_bytes = torch::cat({selection.tokens,
                                    selection.lengths.unsqueeze(1),
                                    selection.rows.unsqueeze(1)},
                                   1)
                            .contiguous()
                            .view(torch::kUInt8);
  auto plain = layout.restore(
      torch::cat({output.view(torch::kUInt8), selected_bytes}, 1));
  auto plain_output =
      plain.narrow(1, 0, hidden_size_ * 2).contiguous().view(torch::kBFloat16);
  auto values = plain.narrow(1, hidden_size_ * 2, (width + 2) * 4)
                    .contiguous()
                    .view(torch::kInt32);
  return {plain_output,
          {values.narrow(1, 0, width).contiguous(),
           values.select(1, width).contiguous(),
           values.select(1, width + 1).contiguous()}};
}
}  // namespace xllm::layer
