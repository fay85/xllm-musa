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
constexpr int64_t kHaloRows = 3;
constexpr int64_t kPoolPacketBytes = 132;
constexpr int64_t kPoolPageRows = 64;
constexpr int32_t kHaloTag = 0;

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

// MATE's FP8 pool page stores payload and scale in separate planes. Transfer
// one compact 132-byte record per newly completed pool, never the full cache.
torch::Tensor pool_byte_indices(const torch::Tensor& physical) {
  auto column = torch::arange(kPoolPacketBytes, physical.options()).unsqueeze(0);
  auto page = physical.div(kPoolPageRows, "floor").unsqueeze(1);
  auto row = physical.remainder(kPoolPageRows).unsqueeze(1);
  return (page * (kPoolPageRows * kPoolPacketBytes) +
          torch::where(column.lt(128),
                       row * 128 + column,
                       kPoolPageRows * 128 + row * 4 + column - 128))
      .flatten();
}
}  // namespace

// Plain activation ownership and round-robin query ownership are separate.
// The plan is immutable for one forward: building it once before the layer loop
// keeps the CP layout's length/padding/restore metadata and its host-derived
// candidate indices out of every DSA layer's critical path.
class Glm5NextCpLayout final {
 public:
  Glm5NextCpLayout(const AttentionMetadata& metadata,
                   const torch::Tensor& global_positions_in,
                   int64_t local_rows,
                   ProcessGroup& group)
      : group_(group),
        device_(global_positions_in.device()),
        offsets_(metadata.glm5_prefill_row_offsets),
        graph_capture_(metadata.enable_cuda_graph ||
                       kernel::musa::is_stream_capturing()) {
    const bool graph_capture = graph_capture_;
    CHECK(metadata.is_prefill || metadata.is_chunked_prefill);
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
    CHECK_EQ(global_positions_in.numel(), total);
    global_positions = global_positions_in.to(torch::kInt64).contiguous();
    auto table = metadata.block_table.to(torch::kInt32).contiguous();
    auto ids = metadata.linear_state_indices.to(torch::kInt32).contiguous();
    auto read_ids =
        metadata.linear_state_read_indices.defined()
            ? metadata.linear_state_read_indices.to(torch::kInt32).contiguous()
            : ids;
    CHECK_EQ(table.dim(), 2);
    const int64_t batch = table.size(0);
    CHECK_GT(batch, 0);
    CHECK_EQ(metadata.q_cu_seq_lens.numel(), batch + 1);
    CHECK_EQ(ids.numel(), batch);
    CHECK_EQ(read_ids.numel(), batch);
    CHECK_GT(metadata.max_seq_len, 0);
    CHECK_LE(metadata.max_seq_len, table.size(1) * 256);
    if (graph_capture) {
      CHECK_EQ(batch, 1)
          << "GLM5 CP graph currently supports one prefill sequence";
      CHECK_EQ(total % group.world_size(), 0)
          << "GLM5 CP graph requires evenly sharded rows";
    }
    std::unordered_set<int32_t> state_owners;
    std::vector<int32_t> requests(static_cast<size_t>(total));
    std::vector<int64_t> pool_slots;
    pool_slots.reserve(total / 4 + batch);
    pool_rows.reserve(total / 4 + batch);
    tail_rows.reserve(batch * 4);
    std::vector<int32_t> tail_cu{0};
    tail_cu.reserve(batch + 1);
    if (graph_capture) {
      for (int64_t row = 0; row < total; ++row) {
        requests[static_cast<size_t>(row)] = 0;
        if (row % 4 == 3) {
          pool_rows.emplace_back(row);
        }
      }
      for (int64_t row = std::max<int64_t>(0, total - 4); row < total; ++row) {
        tail_rows.emplace_back(row);
      }
      tail_cu.emplace_back(tail_rows.size());
    } else {
      auto cu_host =
          metadata.q_cu_seq_lens.to(torch::kCPU, torch::kInt32).contiguous();
      auto start_indices =
          metadata.q_cu_seq_lens.narrow(0, 0, batch).to(torch::kInt64);
      auto starts_host =
          global_positions.index_select(0, start_indices).cpu().contiguous();
      auto table_host = table.to(torch::kCPU, torch::kInt32).contiguous();
      auto ids_host = ids.to(torch::kCPU, torch::kInt32).contiguous();
      auto read_host = read_ids.to(torch::kCPU, torch::kInt32).contiguous();
      const int32_t* cu = cu_host.data_ptr<int32_t>();
      for (int64_t request = 0; request < batch; ++request) {
        CHECK_LT(cu[request], cu[request + 1]);
        const int32_t id = ids_host.data_ptr<int32_t>()[request];
        CHECK_GT(id, 0);
        CHECK(state_owners.insert(id).second);
        CHECK_GE(read_host.data_ptr<int32_t>()[request], 0);
        const int64_t start = starts_host.data_ptr<int64_t>()[request];
        CHECK_GE(start, 0);
        CHECK_LE(start + cu[request + 1] - cu[request], metadata.max_seq_len);
        for (int64_t row = cu[request]; row < cu[request + 1]; ++row) {
          const int64_t position = start + row - cu[request];
          CHECK_EQ(position, start + row - cu[request])
              << "CP requires consecutive unpadded request positions";
          const int32_t page =
              table_host.data_ptr<int32_t>()[request * table_host.size(1) +
                                             position / 256];
          CHECK_GE(page, 0);
          requests[static_cast<size_t>(row)] = request;
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
    }
    if (graph_capture) {
      rows = torch::zeros({total},
                          global_positions.options().dtype(torch::kInt32));
    } else {
      rows = torch::tensor(requests, torch::kInt32).to(device_);
    }
    if (graph_capture) {
      auto slots = metadata.slot_mapping.to(torch::kInt64).contiguous();
      CHECK_EQ(slots.numel(), total);
      pool_physical = torch::floor_divide(slots.slice(0, 3, total, 4), 4);
    } else {
      pool_physical = indices(pool_slots);
    }
    if (graph_capture) {
      tail_boundaries = torch::arange(
          0, 5, 4, global_positions.options().dtype(torch::kInt32));
    } else {
      tail_boundaries = torch::tensor(tail_cu, torch::kInt32).to(device_);
    }
    const int32_t world_size = group_.world_size();
    pool_counts.assign(world_size, 0);
    tail_counts.assign(world_size, 0);
    pool_rank_offsets.assign(world_size + 1, 0);
    std::vector<int32_t> tail_owners(tail_rows.size(), -1);
    std::vector<int32_t> tail_rank_offsets(tail_rows.size(), -1);
    std::vector<std::vector<int64_t>> rank_tail_orders(world_size);
    for (int32_t rank = 0; rank < world_size; ++rank) {
      const auto first =
          std::lower_bound(pool_rows.begin(), pool_rows.end(), offsets_[rank]);
      const auto last = std::lower_bound(
          pool_rows.begin(), pool_rows.end(), offsets_[rank + 1]);
      pool_counts[rank] = static_cast<int32_t>(last - first);
      pool_rank_offsets[rank + 1] =
          pool_rank_offsets[rank] + pool_counts[rank];
      rank_tail_orders[rank].reserve(static_cast<size_t>(batch * 4));
    }
    int32_t tail_owner = 0;
    for (size_t index = 0; index < tail_rows.size(); ++index) {
      const int64_t row = tail_rows[index];
      while (tail_owner + 1 < world_size &&
             row >= offsets_[tail_owner + 1]) {
        ++tail_owner;
      }
      CHECK_GE(row, offsets_[tail_owner]);
      CHECK_LT(row, offsets_[tail_owner + 1]);
      tail_owners[index] = tail_owner;
      tail_rank_offsets[index] =
          static_cast<int32_t>(rank_tail_orders[tail_owner].size());
      rank_tail_orders[tail_owner].emplace_back(index);
      ++tail_counts[tail_owner];
    }
    pool_begin = pool_rank_offsets[group_.rank()];
    local_pool_count = pool_counts[group_.rank()];
    local_tail_count = tail_counts[group_.rank()];
    all_pool_byte_indices = pool_byte_indices(pool_physical);
    CHECK_EQ(all_pool_byte_indices.numel(),
             static_cast<int64_t>(pool_rows.size()) * kPoolPacketBytes);
    local_pool_byte_indices = all_pool_byte_indices.narrow(
        0,
        static_cast<int64_t>(pool_begin) * kPoolPacketBytes,
        static_cast<int64_t>(local_pool_count) * kPoolPacketBytes);
    const auto tail_row_indices = indices(tail_rows);
    tail_positions = global_positions.index_select(0, tail_row_indices);
    std::vector<int64_t> local_tail_rows;
    local_tail_rows.reserve(static_cast<size_t>(local_tail_count));
    for (size_t index = 0; index < tail_rows.size(); ++index) {
      if (tail_owners[index] == group_.rank()) {
        local_tail_rows.emplace_back(tail_rows[index] - begin);
      }
    }
    local_tail_indices = indices(local_tail_rows);

    const int32_t max_pool_count =
        *std::max_element(pool_counts.begin(), pool_counts.end());
    const int32_t max_tail_count =
        *std::max_element(tail_counts.begin(), tail_counts.end());
    const int64_t pool_payload_bytes =
        static_cast<int64_t>(max_pool_count) * kPoolPacketBytes;
    tail_payload_offset = (pool_payload_bytes + 511) / 512 * 512;
    indexer_payload_bytes =
        tail_payload_offset + static_cast<int64_t>(max_tail_count) * 512;
    CHECK_GT(indexer_payload_bytes, 0);
    const auto byte_options = global_positions.options().dtype(torch::kUInt8);
    indexer_payload = torch::empty({indexer_payload_bytes}, byte_options);
    indexer_gather =
        torch::empty({world_size, indexer_payload_bytes}, byte_options);
    std::vector<int64_t> pool_sources;
    pool_sources.reserve(pool_rows.size());
    for (int32_t rank = 0; rank < world_size; ++rank) {
      for (int32_t pool = 0; pool < pool_counts[rank]; ++pool) {
        pool_sources.emplace_back(
            static_cast<int64_t>(rank) * indexer_payload_bytes +
            static_cast<int64_t>(pool) * kPoolPacketBytes);
      }
    }
    pool_source_offsets = indices(pool_sources);
    std::vector<int64_t> tail_sources;
    tail_sources.reserve(tail_rows.size());
    for (size_t index = 0; index < tail_rows.size(); ++index) {
      const int64_t first = static_cast<int64_t>(tail_owners[index]) *
                                indexer_payload_bytes +
                            tail_payload_offset +
                            static_cast<int64_t>(tail_rank_offsets[index]) * 512;
      tail_sources.emplace_back(first);
    }
    tail_source_offsets = indices(tail_sources);

    std::vector<int64_t> local_halo_rows;
    local_halo_rows.reserve(kHaloRows);
    for (int64_t row = std::max<int64_t>(0, count - kHaloRows); row < count;
         ++row) {
      local_halo_rows.emplace_back(row);
    }
    halo_local_indices = indices(local_halo_rows);
    halo_send_ranks.clear();
    if (count > 0) {
      const int64_t source_end = begin + count;
      for (int32_t dst = group_.rank() + 1; dst < world_size; ++dst) {
        if (counts[dst] == 0) {
          continue;
        }
        const int64_t destination_begin = offsets_[dst];
        if (destination_begin - kHaloRows >= source_end) {
          break;
        }
        halo_send_ranks.emplace_back(dst);
      }
    }
    halo_recv_sources.clear();
    halo_recv_counts.clear();
    halo_recv_offsets.clear();
    halo_recv_rows = 0;
    std::vector<int64_t> halo_sources;
    halo_sources.reserve(kHaloRows);
    for (int64_t row = std::max<int64_t>(0, begin - kHaloRows); row < begin;
         ++row) {
      const int32_t owner = static_cast<int32_t>(
          std::upper_bound(offsets_.begin(), offsets_.end(), row) -
          offsets_.begin() - 1);
      CHECK_GE(owner, 0);
      CHECK_LT(owner, world_size);
      if (halo_recv_sources.empty() || halo_recv_sources.back() != owner) {
        halo_recv_sources.emplace_back(owner);
        halo_recv_counts.emplace_back(
            std::min<int64_t>(kHaloRows, counts[owner]));
        halo_recv_offsets.emplace_back(halo_recv_rows);
        halo_recv_rows += halo_recv_counts.back();
      }
      const int64_t source_first =
          std::max<int64_t>(0, counts[owner] - kHaloRows);
      const int64_t source_row = row - offsets_[owner];
      CHECK_GE(source_row, source_first);
      halo_sources.emplace_back(halo_recv_offsets.back() + source_row -
                                source_first);
    }
    halo_source_indices = indices(halo_sources);
    const auto halo_options =
        global_positions.options().dtype(torch::kBFloat16);
    halo_input = torch::zeros({kHaloRows, 256}, halo_options);
    halo_recv = torch::empty({halo_recv_rows, 256}, halo_options);
    interleaved =
        torch::arange(count, global_positions.options().dtype(torch::kInt64)) *
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
    if (graph_capture) {
      std::vector<torch::Tensor> peer_ranges;
      peer_ranges.reserve(group.world_size());
      const auto index_options =
          global_positions.options().dtype(torch::kInt64);
      for (int32_t peer = 0; peer < group.world_size(); ++peer) {
        if (peer < count) {
          peer_ranges.emplace_back(
              torch::arange(peer, count, group.world_size(), index_options));
        } else {
          peer_ranges.emplace_back(torch::empty({0}, index_options));
        }
      }
      permutation_ = torch::cat(peer_ranges);
      inverse_ = torch::argsort(permutation_);
      auto global_rows = torch::arange(total, index_options);
      restore_ = torch::floor_divide(global_rows, group.world_size()) +
                 global_rows.remainder(group.world_size()) * count;
    } else {
      permutation_ = indices(permutation);
      inverse_ = torch::argsort(permutation_);
      std::vector<int64_t> restore(total);
      for (int64_t row = 0; row < total; ++row) {
        restore[row] =
            offsets_[row % group.world_size()] + row / group.world_size();
      }
      restore_ = indices(restore);
    }
  }

  torch::Tensor indices(const std::vector<int64_t>& values) const {
    const auto options =
        torch::TensorOptions().dtype(torch::kInt64).device(device_);
    if (graph_capture_) {
      if (values.empty()) {
        return torch::empty({0}, options);
      }
      std::vector<torch::Tensor> ranges;
      for (size_t start = 0; start < values.size();) {
        size_t end = start + 1;
        while (end < values.size() && values[end] == values[end - 1] + 1) {
          ++end;
        }
        ranges.emplace_back(
            torch::arange(values[start], values[end - 1] + 1, options));
        start = end;
      }
      return ranges.size() == 1 ? ranges.front() : torch::cat(ranges);
    }
    if (values.empty()) {
      return torch::empty({0}, options.device(device_));
    }
    const auto host_options =
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto host = torch::from_blob(const_cast<int64_t*>(values.data()),
                                 {static_cast<int64_t>(values.size())},
                                 host_options);
    return host.clone().to(device_);
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
  void start_halo_recv() const {
    halo_works.clear();
    for (size_t index = 0; index < halo_recv_sources.size(); ++index) {
      auto target = halo_recv.narrow(
          0, halo_recv_offsets[index], halo_recv_counts[index]);
      auto work =
          group_.recv_async(target, halo_recv_sources[index], kHaloTag);
      if (work.defined()) {
        halo_works.emplace_back(std::move(work));
      }
    }
  }
  torch::Tensor with_halo(const torch::Tensor& local) const {
    if (group_.world_size() <= 1) {
      return local;
    }
    const int64_t valid_rows = std::min<int64_t>(kHaloRows, count);
    if (valid_rows > 0) {
      halo_input.narrow(0, 0, valid_rows)
          .copy_(local.index_select(0, halo_local_indices));
      for (const int32_t dst : halo_send_ranks) {
        auto source = halo_input.narrow(0, 0, valid_rows);
        auto work = group_.send_async(source, dst, kHaloTag);
        if (work.defined()) {
          halo_works.emplace_back(std::move(work));
        }
      }
    }
    for (const auto& work : halo_works) {
      work->wait();
    }
    halo_works.clear();
    if (begin == 0) {
      return local;
    }
    auto halo = halo_recv.index_select(0, halo_source_indices);
    return torch::cat({halo, local}, 0);
  }
  void pack_indexer_payload(const torch::Tensor& key_gate,
                            const torch::Tensor& index_cache) const {
    if (local_pool_count > 0) {
      auto packets = index_cache.flatten().index_select(0, local_pool_byte_indices);
      indexer_payload
          .narrow(0, 0, local_pool_count * kPoolPacketBytes)
          .copy_(packets);
    }
    if (local_tail_count > 0) {
      auto tail_rows = key_gate.index_select(0, local_tail_indices)
                           .contiguous()
                           .view(torch::kUInt8)
                           .flatten();
      CHECK_EQ(tail_rows.numel(), local_tail_count * 512);
      indexer_payload
          .narrow(0, tail_payload_offset, local_tail_count * 512)
          .copy_(tail_rows);
    }
  }
  c10::intrusive_ptr<c10d::Work> launch_indexer_gather() const {
    if (group_.world_size() <= 1) {
      indexer_gather.select(0, 0).copy_(indexer_payload);
      return nullptr;
    }
    return group_.allgather_base_async(indexer_payload, indexer_gather);
  }
  int64_t total = 0;
  int64_t begin = 0;
  int64_t count = 0;
  int64_t pool_begin = 0;
  int64_t local_pool_count = 0;
  int64_t local_tail_count = 0;
  int64_t tail_payload_offset = 0;
  int64_t indexer_payload_bytes = 0;
  int64_t halo_recv_rows = 0;
  std::vector<int32_t> counts;
  std::vector<int64_t> pool_rows;
  std::vector<int64_t> tail_rows;
  std::vector<int32_t> pool_counts;
  std::vector<int32_t> tail_counts;
  std::vector<int32_t> pool_rank_offsets;
  std::vector<int32_t> halo_send_ranks;
  std::vector<int32_t> halo_recv_sources;
  std::vector<int64_t> halo_recv_counts;
  std::vector<int64_t> halo_recv_offsets;
  torch::Tensor global_positions, rows, interleaved, pool_physical,
      tail_boundaries, all_pool_byte_indices, local_pool_byte_indices,
      tail_positions, local_tail_indices, pool_source_offsets,
      tail_source_offsets, halo_source_indices, halo_local_indices;
  // These buffers are reused by every layer. The layout itself stays immutable
  // across the forward, so mutation is confined to its communication workspace.
  mutable std::vector<c10::intrusive_ptr<c10d::Work>> halo_works;
  mutable torch::Tensor halo_input, halo_recv, indexer_payload, indexer_gather;

 private:
  ProcessGroup& group_;
  torch::Device device_;
  std::vector<int32_t> offsets_;
  bool graph_capture_ = false;
  std::vector<int64_t> send_counts_, recv_counts_;
  torch::Tensor permutation_, inverse_, restore_;
};

std::shared_ptr<const Glm5NextCpLayout> build_glm5_next_cp_layout(
    const torch::Tensor& global_positions,
    const AttentionMetadata& metadata,
    ProcessGroup& group) {
  return std::make_shared<const Glm5NextCpLayout>(
      metadata,
      global_positions,
      metadata.glm5_prefill_local_row_end -
          metadata.glm5_prefill_local_row_begin,
      group);
}

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
  const bool use_prefill_cp =
      cp_group_ != nullptr && metadata.glm5_prefill_cp_enabled;
  CHECK_EQ(metadata.glm5_prefill_cp_enabled, use_prefill_cp);
  if (use_prefill_cp) {
    return forward_cp(hidden_states, metadata, cache, shared_selection);
  }
  auto main_cache = cache.get_k_cache();
  CHECK_EQ(main_cache.dim(), 4);
  CHECK_EQ(main_cache.size(1), 256);
  torch::Tensor q_norm;
  torch::Tensor latent;
  torch::Tensor q;
  {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.Projections");
    auto projected = fused_qkv_a_proj_->forward(hidden_states);
    auto q_raw = projected.narrow(1, 0, q_rank_).contiguous();
    auto c_raw = projected.narrow(1, q_rank_, 512).contiguous();
    q_norm = std::get<0>(q_norm_->forward(q_raw));
    latent = std::get<0>(kv_norm_->forward(c_raw));
    q = q_b_proj_->forward(q_norm).view({hidden_states.size(0), -1, 256});
  }
  kernel::musa::Glm5NextIndexerSelection selection;
  auto pos = positions.to(torch::kInt64).contiguous();
  auto table = metadata.block_table.to(torch::kInt32).contiguous();
  CHECK_EQ(table.device(), hidden_states.device());
  if (enable_indexer_) {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.Indexer");
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
  torch::Tensor attention;
  {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.SparseAttention");
    attention = glm5_next_mla_attention(q,
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
  }
  // MUSA Linear may reuse its matmul scratch on the next invocation. The
  // attention layer returns owned activations so callers can retain a prefill
  // result while extending the same layer's cache with subsequent forwards.
  torch::Tensor output;
  {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.OutputProjection");
    output = o_proj_->forward(attention.flatten(1, 2)).clone();
  }
  return {output, std::move(selection)};
}

std::pair<torch::Tensor, kernel::musa::Glm5NextIndexerSelection>
Glm5NextMLAAttentionImpl::forward_cp(
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
  CHECK(metadata.glm5_prefill_cp_layout != nullptr)
      << "GLM5 CP layout must be built once before the decoder-layer loop";
  const Glm5NextCpLayout& layout = *metadata.glm5_prefill_cp_layout;
  CHECK_EQ(hidden_states.size(0), layout.count);
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
  torch::Tensor indexer_cache;
  torch::Tensor indexer_tail;
  c10::intrusive_ptr<c10d::Work> indexer_work;
  if (enable_indexer_) {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CPIndexerUpdate");
    // Post the receive side before the local projections so the halo latency
    // overlaps key/gate work instead of serializing in front of it.
    layout.start_halo_recv();
    auto key_gate = torch::empty({layout.count, 256}, hidden_states.options());
    if (layout.count > 0) {
      auto kw = index_kw_->forward(hidden_states);
      auto key_fp32 = kw.narrow(1, 0, 128).to(torch::kFloat32).contiguous();
      key_gate.narrow(1, 0, 128).copy_(
          std::get<0>(index_norm_->forward(key_fp32)));
      key_gate.narrow(1, 128, 128).copy_(index_gate_->forward(hidden_states));
    }
    auto extended = layout.with_halo(key_gate);
    indexer_cache = cache.get_index_cache();
    indexer_tail = cache.get_kpool_tail();
    km::glm5_next_indexer_update_cp(extended.narrow(1, 0, 128).contiguous(),
                                    extended.narrow(1, 128, 128).contiguous(),
                                    ape_,
                                    layout.global_positions,
                                    cu,
                                    ids,
                                    read_ids,
                                    table,
                                    indexer_cache,
                                    indexer_tail,
                                    layout.begin,
                                    layout.count,
                                    std::max<int64_t>(0,
                                                      layout.begin - kHaloRows));
    layout.pack_indexer_payload(key_gate, indexer_cache);
    extended = torch::Tensor();
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
  torch::Tensor scattered;
  {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CPScatter");
    auto input_bytes = hidden_states.contiguous().view(torch::kUInt8);
    if (shared_selection != nullptr) {
      input_bytes =
          torch::cat({input_bytes, pack_selection(*shared_selection)}, 1);
    }
    scattered = layout.scatter(input_bytes);
  }
  if (enable_indexer_) {
    // The compressed pool/tail payload is consumed only by indexer select.
    // Start its collective before the local projection work and join it at the
    // final use so the communication can overlap that work.
    indexer_work = layout.launch_indexer_gather();
  }
  auto local_hidden = scattered.narrow(1, 0, hidden_size_ * 2)
                          .contiguous()
                          .view(torch::kBFloat16);
  const int64_t local_count = layout.interleaved.numel();
  auto pos = layout.global_positions.index_select(0, layout.interleaved);
  auto query_rows = layout.rows.index_select(0, layout.interleaved);
  auto latent = torch::empty({local_count, 512}, hidden_states.options());
  torch::Tensor q_norm, q;
  if (local_count > 0) {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CPProjections");
    auto projected = fused_qkv_a_proj_->forward(local_hidden);
    auto q_raw = projected.narrow(1, 0, q_rank_).contiguous();
    auto c_raw = projected.narrow(1, q_rank_, 512).contiguous();
    q_norm = std::get<0>(q_norm_->forward(q_raw));
    latent.copy_(std::get<0>(kv_norm_->forward(c_raw)));
    q = q_b_proj_->forward(q_norm).view({local_count, heads_, 256});
  }
  auto global_latent = layout.gather_interleaved(latent);
  auto slots = metadata.slot_mapping.to(torch::kInt64).contiguous();
  {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CacheWrite");
    if (main_cache.scalar_type() == torch::kUInt8) {
      km::flash_mla_sparse_fp8_cache_write(global_latent, slots, main_cache);
    } else {
      km::flash_mla_sparse_cache_write(global_latent, slots, main_cache);
    }
  }
  km::Glm5NextIndexerSelection selection;
  if (enable_indexer_) {
    if (indexer_work.defined()) {
      indexer_work->wait();
    }
    km::glm5_next_indexer_scatter_cp(layout.indexer_gather.flatten(),
                                     layout.pool_source_offsets,
                                     layout.all_pool_byte_indices,
                                     layout.tail_source_offsets,
                                     layout.tail_positions,
                                     layout.tail_boundaries,
                                     ids,
                                     read_ids,
                                     indexer_cache,
                                     indexer_tail);
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CPIndexerSelect");
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
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CPSparseAttention");
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
    {
      RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CPOutputProjection");
      output.copy_(o_proj_->forward(attention.flatten(1, 2)));
    }
  }
  auto selected_bytes = torch::cat({selection.tokens,
                                    selection.lengths.unsqueeze(1),
                                    selection.rows.unsqueeze(1)},
                                   1)
                            .contiguous()
                            .view(torch::kUInt8);
  torch::Tensor plain;
  {
    RECORD_USER_SCOPE("nn.Module: DeepseekV2AttentionMLA.CPRestore");
    plain = layout.restore(
        torch::cat({output.view(torch::kUInt8), selected_bytes}, 1));
  }
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
