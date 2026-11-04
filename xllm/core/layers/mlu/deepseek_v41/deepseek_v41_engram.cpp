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

#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>

#include "framework/config/model_config.h"
#include "framework/parallel_state/parallel_state.h"
#include "util/json_reader.h"

namespace xllm::layer {
namespace {
// Match ENGRAM_IMAGE_SENTINEL_BASE_ID / ENGRAM_IMAGE_PAD_ID in the Python
// Engram hash contract. These IDs remain fixed independently of image_token_id.
constexpr int64_t kEngramImageSentinelBaseId = 129264;
constexpr int64_t kEngramImagePadId = 129265;
}  // namespace

DeepseekV41EngramImpl::DeepseekV41EngramImpl(const ModelContext& context,
                                             int32_t engram_id)
    : args_(context.get_model_args()),
      engram_id_(engram_id),
      lookup_group_(context.get_parallel_args().cp_size() > 1
                        ? context.get_parallel_args().moe_ep_group_
                        : context.get_parallel_args().tp_group_),
      options_(context.get_tensor_options()) {
  CHECK_GT(args_.vocab_size(), 0) << "Engram vocab_size must be positive";
  CHECK_GE(args_.engram_pad_token_id(), 0)
      << "engram_pad_token_id must be in [0, vocab_size)";
  CHECK_LT(args_.engram_pad_token_id(), args_.vocab_size())
      << "engram_pad_token_id must be in [0, vocab_size)";
  CHECK_EQ(args_.engram_layer_ids().size(),
           args_.engram_num_embeddings().size())
      << "engram_layer_ids and engram_num_embeddings must have the same length";
  CHECK_GE(engram_id_, 0) << "Invalid Engram layer index";
  CHECK_LT(engram_id_, args_.engram_num_embeddings().size())
      << "Invalid Engram layer index";
  CHECK_GE(args_.engram_max_ngram_size(), 2);
  CHECK_GT(args_.engram_n_heads(), 0);
  CHECK_GT(args_.engram_compressed_vocab_size(), 0);
  CHECK_GT(args_.engram_num_embeddings()[engram_id_], 0);
  JsonReader reader;
  CHECK(reader.parse_text(ModelConfig::get_instance().engram_config()))
      << "Invalid Engram JSON configuration";
  const auto config = reader.data();
  CHECK(config.is_object() || config.is_boolean())
      << "engram_config must be a JSON object or boolean";
  if (config.is_object()) {
    for (const auto& [key, value] : config.items()) {
      CHECK(key == "cpu_offload" || key == "dp_shared_memory")
          << "Unknown Engram configuration key: " << key;
      CHECK(value.is_boolean())
          << "Engram configuration " << key << " must be boolean";
    }
    CHECK(!config.value("dp_shared_memory", false))
        << "MLU Engram does not support dp_shared_memory";
  }
  cpu_offload_ = config.is_boolean() ? config.get<bool>()
                                     : config.value("cpu_offload", true);
  const int64_t rows = args_.engram_num_embeddings().at(engram_id_);
  table_start_ = rows * lookup_group_->rank() / lookup_group_->world_size();
  table_end_ = rows * (lookup_group_->rank() + 1) / lookup_group_->world_size();
  LOG(INFO) << "Engram " << engram_id_ << " rank " << lookup_group_->rank()
            << ": table rows [" << table_start_ << ", " << table_end_
            << ") stored on " << (cpu_offload_ ? "host" : "MLU");
  // Decode raw E4M3 bytes by lookup; no FP8 arithmetic is needed on MLU.
  torch::Tensor bytes = torch::arange(256, torch::kInt16).to(torch::kUInt8);
  fp8_values_ = bytes.view(torch::kFloat8_e4m3fn)
                    .to(torch::kFloat32)
                    .to(options_.device());
}

void DeepseekV41EngramImpl::load_state_dict(const StateDict& state) {
  for (const std::string name :
       {"wkv.weight", "q_weight", "k_weight", "embed.weight", "embed.scale"}) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    if (name == "embed.weight" || name == "embed.scale") {
      CHECK_EQ(value.dim(), 2) << "Invalid Engram " << name << " shape";
      CHECK_EQ(value.size(0), args_.engram_num_embeddings()[engram_id_])
          << "Invalid Engram " << name << " rows";
      value = value.slice(0, table_start_, table_end_)
                  .contiguous()
                  .view(torch::kUInt8);
      CHECK_GT(value.size(1), 0) << "Empty Engram " << name << " row";
      if (name == "embed.weight") {
        CHECK_EQ(value.size(1), args_.engram_head_dim());
      } else {
        CHECK_EQ(args_.engram_head_dim() % value.size(1), 0);
      }
      if (cpu_offload_) {
        // Own the host shard independently of the checkpoint mmap lifetime.
        weights_[name] = value.cpu().contiguous().clone();
        continue;
      }
    }
    weights_[name] = value.to(options_.device()).contiguous();
  }
}

void DeepseekV41EngramImpl::lookup_host_rows(const torch::Tensor& hashes,
                                             torch::Tensor& weight_rows,
                                             torch::Tensor& scale_rows) const {
  const auto& weight = weights_.at("embed.weight");
  const auto& scale = weights_.at("embed.scale");
  CHECK(hashes.device().is_cpu() && hashes.is_contiguous());
  CHECK_EQ(hashes.scalar_type(), torch::kInt64);
  for (const auto& tensor : {weight_rows, scale_rows}) {
    CHECK(tensor.device().is_cpu() && tensor.is_contiguous());
    CHECK_EQ(tensor.scalar_type(), torch::kUInt8);
    CHECK_EQ(tensor.dim(), 2);
  }
  CHECK_EQ(weight_rows.size(0), hashes.numel());
  CHECK_EQ(scale_rows.size(0), hashes.numel());
  CHECK_EQ(weight_rows.size(1), weight.size(1));
  CHECK_EQ(scale_rows.size(1), scale.size(1));
  const int64_t dim = weight.size(1);
  const int64_t scale_dim = scale.size(1);
  const int64_t* ids = hashes.data_ptr<int64_t>();
  const uint8_t* table = weight.data_ptr<uint8_t>();
  const uint8_t* scales = scale.data_ptr<uint8_t>();
  uint8_t* rows = weight_rows.data_ptr<uint8_t>();
  uint8_t* row_scales = scale_rows.data_ptr<uint8_t>();
  for (int64_t index = 0; index < hashes.numel(); ++index) {
    const int64_t row = ids[index] - table_start_;
    if (row < 0 || row >= table_end_ - table_start_) {
      std::memset(rows + index * dim, 0, dim);
      std::memset(row_scales + index * scale_dim, 0, scale_dim);
      continue;
    }
    std::memcpy(rows + index * dim, table + row * dim, dim);
    std::memcpy(
        row_scales + index * scale_dim, scales + row * scale_dim, scale_dim);
  }
}

void DeepseekV41EngramImpl::verify_loaded_weights() const {
  CHECK(token_map_.defined()) << "Missing Engram hash state";
  for (const std::string name :
       {"wkv.weight", "q_weight", "k_weight", "embed.weight", "embed.scale"}) {
    CHECK(weights_.count(name)) << "Missing Engram weight " << name;
  }
}

void DeepseekV41EngramImpl::load_hash_state(const StateDict& state) {
  auto map = state.get_tensor("token_map");
  CHECK(map.defined()) << "Missing Engram token_map";
  CHECK(map.scalar_type() == torch::kInt32 ||
        map.scalar_type() == torch::kInt64);
  CHECK_EQ(map.dim(), 1);
  CHECK_EQ(map.numel(), args_.vocab_size())
      << "Engram token_map must cover vocab_size";
  token_map_ = map.to(torch::kCPU, torch::kInt64).contiguous().clone();
  CHECK_GE(token_map_.min().item<int64_t>(), 0)
      << "Engram token_map contains a negative compressed id";
  CHECK_LT(token_map_.max().item<int64_t>(),
           args_.engram_compressed_vocab_size())
      << "Engram token_map compressed id exceeds compressed_vocab_size";
  const int64_t cols =
      (args_.engram_max_ngram_size() - 1) * args_.engram_n_heads();
  auto load_layer = [&](const std::string& name, int64_t elements) {
    const auto tensor = state.get_tensor(name);
    CHECK(tensor.defined()) << "Missing Engram " << name;
    CHECK_EQ(tensor.scalar_type(), torch::kInt64);
    CHECK_GE(tensor.dim(), 2);
    CHECK_EQ(tensor.size(0), args_.engram_layer_ids().size());
    CHECK_EQ(tensor[engram_id_].numel(), elements)
        << "Invalid Engram " << name << " shape";
    return tensor[engram_id_].to(torch::kCPU).flatten().contiguous().clone();
  };
  multipliers_ = load_layer("multipliers", args_.engram_max_ngram_size());
  primes_ = load_layer("primes", cols);
  offsets_ = load_layer("offsets", cols);
  CHECK_GT(multipliers_.min().item<int64_t>(), 0);
  // Training bounds each positive multiplier so every mapped-id product fits
  // int64. XOR of those nonnegative products also leaves the sign bit clear.
  CHECK_LE(multipliers_.max().item<int64_t>(),
           std::numeric_limits<int64_t>::max() /
               std::max<int64_t>(1, args_.engram_compressed_vocab_size() - 1))
      << "Engram multiplier can overflow a compressed-id product";
  CHECK_GT(primes_.min().item<int64_t>(), 0);
  CHECK_GE(offsets_.min().item<int64_t>(), 0);
  const int64_t rows = args_.engram_num_embeddings()[engram_id_];
  CHECK_LE(primes_.max().item<int64_t>(), rows);
  CHECK((offsets_ <= rows - primes_).all().item<bool>())
      << "Engram hash buckets exceed embedding rows";
}

void DeepseekV41EngramImpl::compute_hashes(const int64_t* tokens,
                                           const int64_t* history,
                                           const std::vector<int32_t>& lengths,
                                           int64_t* output) const {
  const int64_t heads = args_.engram_n_heads();
  const int64_t depth = args_.engram_max_ngram_size() - 1;
  const int64_t cols = heads * depth;
  const int64_t* map = token_map_.data_ptr<int64_t>();
  const int64_t* mul = multipliers_.data_ptr<int64_t>();
  const int64_t* prime = primes_.data_ptr<int64_t>();
  const int64_t* offset = offsets_.data_ptr<int64_t>();
  int64_t start = 0;
  for (size_t seq = 0; seq < lengths.size(); ++seq) {
    for (int64_t row = 0; row < lengths[seq]; ++row) {
      int64_t rolling = 0;
      bool blocked = false;
      for (int64_t shift = 0; shift <= depth; ++shift) {
        const int64_t relative = row - shift;
        const int64_t token = relative >= 0
                                  ? tokens[start + relative]
                                  : history[seq * depth - relative - 1];
        blocked = blocked || token < 0 || token == kEngramImageSentinelBaseId ||
                  token == kEngramImagePadId;
        const int64_t value =
            blocked ? map[args_.engram_pad_token_id()]
                    : map[std::min<int64_t>(token, token_map_.numel() - 1)];
        rolling ^= value * mul[shift];
        if (shift == 0) {
          continue;
        }
        for (int64_t head = 0; head < heads; ++head) {
          const int64_t col = (shift - 1) * heads + head;
          output[(start + row) * cols + col] =
              rolling % prime[col] + offset[col];
        }
      }
    }
    start += lengths[seq];
  }
}

DeepseekV41EngramImpl::HashState::HashState(const DeepseekV41EngramImpl& owner,
                                            int64_t rows,
                                            int64_t query_width)
    : owner_(owner) {
  CHECK_GT(query_width, 0);
  CHECK_EQ(rows % query_width, 0);
  lengths_.assign(rows / query_width, static_cast<int32_t>(query_width));
  const int64_t depth = owner.args_.engram_max_ngram_size() - 1;
  auto host = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
  host_tokens_ = torch::empty({rows}, host);
  host_history_ = torch::empty({rows / query_width, depth}, host);
  host_hashes_ =
      torch::empty({rows, depth * owner.args_.engram_n_heads()}, host);
  history_ =
      torch::empty(host_history_.sizes(), owner.options_.dtype(torch::kInt64));
  hashes_ =
      torch::empty(host_hashes_.sizes(), owner.options_.dtype(torch::kInt64));
  if (owner.cpu_offload_) {
    host_weight_rows_ = torch::empty(
        {host_hashes_.numel(), owner.weights_.at("embed.weight").size(1)},
        host.dtype(torch::kUInt8));
    host_scale_rows_ = torch::empty(
        {host_hashes_.numel(), owner.weights_.at("embed.scale").size(1)},
        host.dtype(torch::kUInt8));
  }
}

void DeepseekV41EngramImpl::HashState::prepare(const ModelInputParams& input) {
  const auto& history = input.embedding.engram_lookback_token_ids;
  CHECK_EQ(history.size(), input.meta.actual_num_sequences * history_.size(1));
  auto padded = torch::full(
      history_.sizes(),
      -1,
      torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
  std::copy(history.begin(), history.end(), padded.data_ptr<int64_t>());
  // Copy through device storage: graph callbacks never race host preparation.
  history_.copy_(padded, /*non_blocking=*/true);
}

void DeepseekV41EngramImpl::HashState::compute(void* state) {
  auto& hash = *static_cast<HashState*>(state);
  hash.owner_.compute_hashes(hash.host_tokens_.data_ptr<int64_t>(),
                             hash.host_history_.data_ptr<int64_t>(),
                             hash.lengths_,
                             hash.host_hashes_.data_ptr<int64_t>());
  if (hash.owner_.cpu_offload_) {
    hash.owner_.lookup_host_rows(
        hash.host_hashes_, hash.host_weight_rows_, hash.host_scale_rows_);
  }
}

std::pair<torch::Tensor, torch::Tensor>
DeepseekV41EngramImpl::HashState::lookup_rows() const {
  return {host_weight_rows_.to(owner_.options_.device(), /*non_blocking=*/true),
          host_scale_rows_.to(owner_.options_.device(), /*non_blocking=*/true)};
}

torch::Tensor DeepseekV41EngramImpl::HashState::forward(
    const torch::Tensor& tokens) {
  host_tokens_.copy_(tokens.to(torch::kInt64), /*non_blocking=*/true);
  host_history_.copy_(history_, /*non_blocking=*/true);
  CHECK_EQ(cnrtInvokeHostFunc(torch_mlu::getCurMLUStream(), compute, this),
           cnrtSuccess);
  hashes_.copy_(host_hashes_, /*non_blocking=*/true);
  return hashes_;
}

std::unique_ptr<DeepseekV41EngramImpl::HashState>
DeepseekV41EngramImpl::create_hash_state(int64_t rows,
                                         int64_t query_width) const {
  return std::make_unique<HashState>(*this, rows, query_width);
}

torch::Tensor DeepseekV41EngramImpl::forward(const torch::Tensor& hidden,
                                             const torch::Tensor& tokens,
                                             const ModelInputParams& input,
                                             HashState* hash_state,
                                             const torch::Tensor& row_indices) {
  const int64_t depth = args_.engram_max_ngram_size() - 1;
  const int64_t cols = args_.engram_n_heads() * depth;
  const int64_t dim = args_.engram_head_dim();
  torch::Tensor hashes;
  torch::Tensor bytes;
  torch::Tensor scales;
  if (hash_state != nullptr) {
    hashes = hash_state->forward(tokens);
    if (cpu_offload_) {
      std::tie(bytes, scales) = hash_state->lookup_rows();
    }
  } else {
    auto ids = tokens.to(torch::kCPU, torch::kInt64).contiguous();
    hashes = torch::empty({tokens.numel(), cols}, torch::kInt64);
    std::vector<int32_t> lengths;
    lengths.reserve(input.meta.num_sequences);
    for (int32_t seq = 0; seq < input.meta.num_sequences; ++seq) {
      lengths.emplace_back(input.get_q_seq_len(seq));
    }
    const auto& history = input.embedding.engram_lookback_token_ids;
    CHECK_EQ(history.size(), lengths.size() * depth);
    compute_hashes(ids.data_ptr<int64_t>(),
                   torch::tensor(history, torch::kInt64).data_ptr<int64_t>(),
                   lengths,
                   hashes.data_ptr<int64_t>());
    if (cpu_offload_) {
      const auto host =
          torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);
      auto weight_rows = torch::empty({hashes.numel(), dim}, host);
      auto scale_rows = torch::empty(
          {hashes.numel(), weights_.at("embed.scale").size(1)}, host);
      lookup_host_rows(hashes, weight_rows, scale_rows);
      bytes = weight_rows.to(hidden.device(), /*non_blocking=*/true);
      scales = scale_rows.to(hidden.device(), /*non_blocking=*/true);
    }
    hashes = hashes.to(hidden.device());
  }
  torch::Tensor owned = hashes.ge(table_start_) & hashes.lt(table_end_);
  if (!cpu_offload_) {
    torch::Tensor local = (hashes - table_start_)
                              .clamp(0, table_end_ - table_start_ - 1)
                              .flatten();
    bytes = weights_.at("embed.weight").index_select(0, local);
    scales = weights_.at("embed.scale").index_select(0, local);
  }
  scales = scales.to(torch::kFloat32);
  scales =
      torch::exp2(scales - 127).repeat_interleave(dim / scales.size(-1), -1);
  torch::Tensor rows =
      fp8_values_.index_select(0, bytes.to(torch::kInt64).flatten())
          .reshape_as(scales) *
      scales;
  rows = rows.view({hashes.size(0), cols, dim}) * owned.unsqueeze(-1);
  rows = rows.to(hidden.scalar_type());
  rows = parallel_state::reduce(rows, lookup_group_);
  if (row_indices.defined()) {
    rows = rows.index_select(0, row_indices);
  }
  torch::Tensor kv =
      torch::matmul(rows.flatten(1), weights_.at("wkv.weight").t());
  const int64_t width = args_.hidden_size();
  torch::Tensor keys = kv.slice(1, 0, args_.hc_mult() * width)
                           .view_as(hidden)
                           .to(torch::kFloat32);
  torch::Tensor value =
      kv.slice(1, args_.hc_mult() * width).to(torch::kFloat32);
  torch::Tensor h = hidden.to(torch::kFloat32);
  torch::Tensor dot = (h * keys * weights_.at("q_weight").to(torch::kFloat32) *
                       weights_.at("k_weight").to(torch::kFloat32))
                          .sum(-1);
  dot *= torch::rsqrt(h.square().mean(-1) + args_.rms_norm_eps()) *
         torch::rsqrt(keys.square().mean(-1) + args_.rms_norm_eps()) /
         std::sqrt(width);
  torch::Tensor gate = torch::sigmoid(torch::where(dot < 0, -1.0, 1.0) *
                                      dot.abs().clamp_min(1e-6).sqrt());
  return (h + gate.unsqueeze(-1) * value.unsqueeze(1)).to(hidden.scalar_type());
}

}  // namespace xllm::layer
