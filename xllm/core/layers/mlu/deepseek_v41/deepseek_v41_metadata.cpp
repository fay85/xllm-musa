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

#include "layers/mlu/deepseek_v41/deepseek_v41_metadata.h"

#include <algorithm>
#include <limits>

#include "kernels/mlu/bangc/include/deepseek_v41_constants.h"

namespace xllm::layer::deepseek_v41 {
namespace {
using kernel::mlu::bangc::kDeepseekV41MluKVBlockSize;

void prepare_buffer(torch::Tensor& target,
                    const std::vector<int64_t>& shape,
                    const torch::TensorOptions& options) {
  if (!target.defined()) {
    target = torch::empty(shape, options);
  }
  CHECK(target.sizes().vec() == shape);
  CHECK(target.options().device() == options.device());
  CHECK(target.scalar_type() == options.dtype().toScalarType());
}

void prepare_compress_metadata(Csa2Context& shared,
                               const ModelInputParams& input,
                               const torch::Tensor& positions,
                               int32_t num_sequences) {
  const int32_t batch = input.meta.num_sequences;
  CHECK_GE(num_sequences, 0);
  CHECK_LE(num_sequences, batch);
  shared.num_sequences = num_sequences;
  const auto& cpu_positions = input.attention.host.positions;
  if (cpu_positions.defined()) {
    CHECK(cpu_positions.device().is_cpu());
    CHECK_EQ(cpu_positions.dim(), 1);
    CHECK(cpu_positions.scalar_type() == torch::kInt32 ||
          cpu_positions.scalar_type() == torch::kInt64);
    CHECK_LE(cpu_positions.numel(), positions.numel());
    shared.host_positions = cpu_positions.to(torch::kInt64).contiguous();
  } else {
    shared.host_positions =
        positions.to(torch::kCPU, torch::kInt64).contiguous();
  }
  const auto ints = positions.options().dtype(torch::kInt32);
  auto& prefix = shared.host_query_prefix;
  auto& begins = shared.host_chunk_begins;
  prefix.assign(batch + 1, 0);
  begins.assign(batch, 0);
  std::array<std::vector<int32_t>, 2> compressed;
  for (auto& values : compressed) {
    values.resize(batch + 1, 0);
  }
  const int64_t* host_positions =
      shared.host_positions.const_data_ptr<int64_t>();
  for (int32_t seq = 0; seq < batch; ++seq) {
    const int32_t count = seq < num_sequences ? input.get_q_seq_len(seq) : 0;
    CHECK_GE(count, 0);
    CHECK_LE(count, positions.numel() - prefix[seq]);
    prefix[seq + 1] = prefix[seq] + count;
    CHECK_LE(prefix[seq + 1], positions.numel());
    CHECK_LE(prefix[seq + 1], shared.host_positions.numel());
    if (count > 0) {
      const int64_t begin = host_positions[prefix[seq]];
      CHECK_GE(begin, 0);
      CHECK_LE(begin + count, std::numeric_limits<int32_t>::max());
      begins[seq] = static_cast<int32_t>(begin);
      // Compression consumes contiguous logical chunks. CP passes the global
      // position pack, before its local query rows are selected.
      for (int32_t row = 0; row < count; ++row) {
        CHECK_EQ(host_positions[prefix[seq] + row], begin + row);
      }
    }
    for (int32_t ratio : {1, 2}) {
      compressed[ratio - 1][seq + 1] = compressed[ratio - 1][seq] +
                                       (begins[seq] + count) / ratio -
                                       begins[seq] / ratio;
    }
  }
  const int64_t metadata_size = 4 * (static_cast<int64_t>(batch) + 1);
  prepare_buffer(shared.compression_metadata, {metadata_size}, ints);
  // Do not overwrite host storage from a previous asynchronous upload.
  // The pinned allocator retains each staging allocation until its copy ends.
  auto staging = torch::empty({metadata_size},
                              ints.device(torch::kCPU).pinned_memory(true));
  int32_t* staging_data = staging.data_ptr<int32_t>();
  int64_t offset = 0;
  const auto pack = [&](torch::Tensor& target,
                        const std::vector<int32_t>& values) {
    const int64_t size = static_cast<int64_t>(values.size());
    target = shared.compression_metadata.narrow(/*dim=*/0, offset, size);
    std::copy(values.begin(), values.end(), staging_data + offset);
    offset += size;
  };
  pack(shared.query_prefix, prefix);
  pack(shared.chunk_begins, begins);
  pack(shared.num_valid_input_rows, {prefix.back()});
  for (int32_t ratio : {1, 2}) {
    auto& out = shared.compressed[ratio - 1];
    pack(out.prefix, compressed[ratio - 1]);
    const int64_t capacity =
        ratio == 1 ? positions.numel() : (positions.numel() + batch) / 2;
    prepare_buffer(out.outputs.latent,
                   {capacity, 512},
                   positions.options().dtype(torch::kBFloat16));
    prepare_buffer(out.outputs.positions, {capacity}, ints);
    prepare_buffer(out.outputs.slots, {capacity}, ints);
    prepare_buffer(out.outputs.num_valid_rows, {1}, ints);
  }
  CHECK_EQ(offset, metadata_size);
  shared.compression_metadata.copy_(staging, /*non_blocking=*/true);
}

void prepare_block_tables(Csa2Context& shared,
                          const ModelInputParams& input,
                          const torch::Device& device,
                          int64_t kv_seq_len_capacity) {
  const int64_t batch = input.meta.num_sequences;
  CHECK_GE(kv_seq_len_capacity, 0);
  for (const auto& table : input.multi_block_tables) {
    CHECK_EQ(table.dim(), 2);
    CHECK_GE(table.size(0), shared.num_sequences);
    CHECK_LE(table.size(0), batch);
    CHECK(shared.num_sequences == 0 || table.size(1) > 0);
  }
  auto& tables = shared.block_tables;
  if (kv_seq_len_capacity == 0) {
    tables.clear();
    tables.reserve(input.multi_block_tables.size());
    for (const auto& table : input.multi_block_tables) {
      tables.emplace_back(table.to(device, torch::kInt32).contiguous());
    }
    shared.swa_block_table = tables.empty() ? torch::Tensor() : tables[0];
    return;
  }
  if (tables.empty()) {
    tables.reserve(input.multi_block_tables.size());
    for (size_t group = 0; group < input.multi_block_tables.size(); ++group) {
      const int64_t ratio = group == 1 ? 2 : 1;
      const int64_t columns = std::max<int64_t>(
          input.multi_block_tables[group].size(1),
          (kv_seq_len_capacity + ratio * kDeepseekV41MluKVBlockSize - 1) /
              (ratio * kDeepseekV41MluKVBlockSize));
      tables.emplace_back(torch::zeros(
          {batch, columns},
          torch::TensorOptions().device(device).dtype(torch::kInt32)));
    }
  }
  CHECK_EQ(tables.size(), input.multi_block_tables.size());
  for (size_t group = 0; group < tables.size(); ++group) {
    const auto source = input.multi_block_tables[group].to(torch::kInt32);
    auto& target = tables[group];
    CHECK(target.device() == device);
    CHECK_EQ(target.size(0), batch);
    CHECK_LE(source.size(0), batch);
    target.zero_();
    CHECK_LE(source.size(1), target.size(1));
    if (group == 0 && source.size(1) > 0) {
      auto columns =
          torch::arange(target.size(1), source.options().dtype(torch::kInt64))
              .remainder(source.size(1));
      target.slice(0, 0, source.size(0)).copy_(source.index_select(1, columns));
    } else {
      target.slice(0, 0, source.size(0))
          .slice(1, 0, source.size(1))
          .copy_(source);
    }
  }
  if (!tables.empty()) {
    shared.swa_block_table = tables[0].slice(
        /*dim=*/1, /*start=*/0, /*end=*/input.multi_block_tables[0].size(1));
    // Captured compression uses the expanded table without a logical wrap.
    // All pages it can access must lie in that absolute-page extent.
    for (int32_t seq = 0; seq < shared.num_sequences; ++seq) {
      const int32_t count =
          shared.host_query_prefix[seq + 1] - shared.host_query_prefix[seq];
      CHECK_LE(static_cast<int64_t>(shared.host_chunk_begins[seq]) + count,
               tables[0].size(1) * kDeepseekV41MluKVBlockSize);
    }
  }
}

}  // namespace

bool uses_decode_attention(const ModelInputParams& input) {
  return input.meta.batch_forward_type.is_decode() || input.is_spec_verify ||
         input.use_dflash_proposal_attention;
}

void prepare_workspace(Csa2Context& shared,
                       const ModelInputParams& input,
                       const ModelArgs& args,
                       int64_t query_capacity,
                       const torch::TensorOptions& options) {
  if (!uses_decode_attention(input)) {
    return;
  }
  const bool compressed = std::any_of(args.compress_ratios().begin(),
                                      args.compress_ratios().end(),
                                      [](int32_t ratio) { return ratio > 0; });
  const int64_t capacity =
      args.window_size() + (input.use_dflash_proposal_attention
                                ? args.dspark_block_size()
                                : (compressed ? args.index_topk() : 0));
  prepare_buffer(shared.selected_kv,
                 {query_capacity * capacity, 1, 1, args.head_dim()},
                 options);
  prepare_buffer(
      shared.selected_lengths, {query_capacity}, options.dtype(torch::kInt32));
}

void prepare_metadata(Csa2Context& shared,
                      const ModelInputParams& input,
                      const torch::Tensor& positions,
                      int32_t num_sequences,
                      int64_t kv_seq_len_capacity) {
  prepare_compress_metadata(shared, input, positions, num_sequences);
  prepare_block_tables(shared, input, positions.device(), kv_seq_len_capacity);
}
}  // namespace xllm::layer::deepseek_v41
