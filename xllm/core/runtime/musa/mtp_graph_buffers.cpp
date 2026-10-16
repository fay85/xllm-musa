/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/musa/mtp_graph_buffers.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include "core/framework/model/model_input_params.h"
#include "core/runtime/spec_input_builder.h"

namespace xllm::runtime::musa {

namespace {

constexpr uint64_t kSpecVerifyGraphKeyMask = 1ull << 63;
constexpr uint64_t kSpecVerifyQMaxSeqLenShift = 32;

torch::Tensor get_buffer_view(torch::Tensor& buffer,
                              int64_t required_size,
                              const torch::TensorOptions& options) {
  CHECK_GE(required_size, 0) << "buffer size must be non-negative";
  if (!buffer.defined() || buffer.numel() < required_size) {
    const int64_t old_size = buffer.defined() ? buffer.numel() : 0;
    const int64_t new_size =
        std::max<int64_t>(required_size, std::max<int64_t>(old_size * 2, 1));
    buffer = torch::empty({new_size}, options);
  }
  return buffer.narrow(/*dim=*/0, /*start=*/0, /*length=*/required_size);
}

torch::Tensor fill_host_buffer(const std::vector<int32_t>& values,
                               torch::Tensor& buffer) {
  const torch::TensorOptions options = torch::TensorOptions()
                                           .dtype(torch::kInt)
                                           .device(torch::kCPU)
                                           .pinned_memory(true);
  torch::Tensor view =
      get_buffer_view(buffer, static_cast<int64_t>(values.size()), options);
  if (!values.empty()) {
    std::memcpy(view.data_ptr<int32_t>(),
                values.data(),
                values.size() * sizeof(int32_t));
  }
  return view;
}

torch::Tensor copy_to_device_buffer(const torch::Tensor& host,
                                    torch::Tensor& buffer,
                                    const torch::Device& device) {
  torch::Tensor view =
      get_buffer_view(buffer,
                      host.numel(),
                      torch::TensorOptions().dtype(torch::kInt).device(device));
  view.copy_(host.flatten(), /*non_blocking=*/true);
  return view.view(host.sizes());
}

int32_t get_cumulative_sequence_length(const std::vector<int32_t>& lengths,
                                       int64_t sequence_index) {
  if (lengths.size() == 1) {
    CHECK_EQ(sequence_index, 0);
    return lengths.front();
  }
  CHECK_LT(static_cast<size_t>(sequence_index + 1), lengths.size())
      << "spec verify sequence lengths are too short";
  return lengths[static_cast<size_t>(sequence_index + 1)] -
         lengths[static_cast<size_t>(sequence_index)];
}

void clear_expanded_attention(ModelInputParams& input_params) {
  input_params.graph.use_expanded_decode_for_spec_verify_attention = false;
  input_params.graph.expanded_kv_seq_lens = torch::Tensor();
  input_params.graph.expanded_block_tables = torch::Tensor();
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
  input_params.graph.expanded_paged_kv_indptr = torch::Tensor();
  input_params.graph.expanded_paged_kv_indices = torch::Tensor();
  input_params.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  input_params.graph.expanded_paged_kv_indptr_host = torch::Tensor();
  input_params.graph.expanded_paged_kv_indices_host = torch::Tensor();
  input_params.graph.expanded_paged_kv_last_page_len_host = torch::Tensor();
}

}  // namespace

SpecVerifyBucketShape get_mtp_spec_verify_bucket_shape(
    uint32_t actual_num_tokens,
    int32_t live_batch_size,
    int32_t query_width) {
  CHECK_GT(live_batch_size, 0) << "spec verify requires a positive live batch";
  CHECK_GT(query_width, 0) << "spec verify requires a positive query width";

  const uint64_t expected_num_tokens =
      static_cast<uint64_t>(live_batch_size) * query_width;
  CHECK_EQ(expected_num_tokens, actual_num_tokens)
      << "spec verify requires a dense B x (K+1) token layout";

  int32_t bucket_batch_size = live_batch_size;
  if (query_width == 3 && live_batch_size == 3) {
    bucket_batch_size = 4;
  }
  const uint64_t padded_num_tokens =
      static_cast<uint64_t>(bucket_batch_size) * query_width;
  CHECK_LE(padded_num_tokens,
           static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      << "spec verify bucket token count exceeds uint32 capacity";
  return SpecVerifyBucketShape{
      .live_batch_size = live_batch_size,
      .bucket_batch_size = bucket_batch_size,
      .query_width = query_width,
      .actual_num_tokens = actual_num_tokens,
      .padded_num_tokens = static_cast<uint32_t>(padded_num_tokens)};
}

uint64_t get_mtp_spec_verify_graph_key(const SpecVerifyBucketShape& shape) {
  return get_mtp_spec_verify_graph_key(shape.padded_num_tokens,
                                       shape.query_width);
}

uint64_t get_mtp_spec_verify_graph_key(uint32_t padded_num_tokens,
                                       int32_t query_width) {
  CHECK_GT(query_width, 0) << "spec verify key requires a positive query width";
  return static_cast<uint64_t>(padded_num_tokens) | kSpecVerifyGraphKeyMask |
         (static_cast<uint64_t>(query_width) << kSpecVerifyQMaxSeqLenShift);
}

MtpGraphBuffers::MtpGraphBuffers(torch::Device device) : device_(device) {}

MtpGraphBuffers::BufferSlot& MtpGraphBuffers::acquire_slot() {
  for (int32_t attempt = 0; attempt < static_cast<int32_t>(slots_.size());
       ++attempt) {
    BufferSlot& slot = slots_[static_cast<size_t>(next_slot_)];
    next_slot_ = (next_slot_ + 1) % static_cast<int32_t>(slots_.size());
    if (slot.in_use) {
      continue;
    }
    if (slot.ready_event != nullptr && !slot.ready_event->c10_event().query()) {
      CHECK(slot.ready_event->synchronize())
          << "failed to synchronize MUSA MTP graph buffer slot";
    }
    slot.ready_event.reset();
    slot.in_use = true;
    return slot;
  }
  LOG(FATAL) << "no free MUSA MTP graph buffer slot";
}

void MtpGraphBuffers::prepare(
    ModelInputParams& input_params,
    const std::vector<int32_t>& accepted_prefix_lengths,
    int32_t block_size,
    bool build_expanded_attention) {
  BufferSlot& slot = acquire_slot();

  torch::Tensor accepted_host = fill_host_buffer(
      accepted_prefix_lengths, slot.accepted_prefix_lengths_host);
  input_params.num_accepted_tokens = copy_to_device_buffer(
      accepted_host, slot.accepted_prefix_lengths_device, device_);

  clear_expanded_attention(input_params);
  if (!build_expanded_attention || !input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return;
  }

  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return;
  }

  torch::Tensor block_tables_host = input_params.attention.host.block_tables;
  if (!block_tables_host.defined()) {
    block_tables_host =
        input_params.attention.device.block_tables.to(torch::kCPU);
  }
  CHECK(block_tables_host.device().is_cpu());
  CHECK_EQ(block_tables_host.scalar_type(), torch::kInt);
  CHECK_EQ(block_tables_host.dim(), 2);
  block_tables_host = block_tables_host.contiguous();

  const int64_t num_sequences = input_params.meta.num_sequences;
  CHECK_GT(num_sequences, 0);
  CHECK_GE(block_tables_host.size(0), num_sequences);

  std::vector<int32_t> expanded_kv_seq_lens;
  std::vector<int32_t> expanded_row_sequences;
  expanded_kv_seq_lens.reserve(static_cast<size_t>(q_seq_lens.back()));
  expanded_row_sequences.reserve(static_cast<size_t>(q_seq_lens.back()));
  for (int64_t sequence_index = 0; sequence_index < num_sequences;
       ++sequence_index) {
    const int32_t q_len =
        get_cumulative_sequence_length(q_seq_lens, sequence_index);
    const int32_t kv_len =
        get_cumulative_sequence_length(kv_seq_lens, sequence_index);
    CHECK_GE(q_len, 1);
    CHECK_GE(kv_len, q_len);
    for (int32_t token_index = 0; token_index < q_len; ++token_index) {
      expanded_kv_seq_lens.emplace_back(kv_len - q_len + token_index + 1);
      expanded_row_sequences.emplace_back(static_cast<int32_t>(sequence_index));
    }
  }

  const int64_t expanded_rows =
      static_cast<int64_t>(expanded_kv_seq_lens.size());
  const int64_t block_table_width = block_tables_host.size(1);
  const int64_t block_table_elements = expanded_rows * block_table_width;
  torch::Tensor expanded_block_tables_host_flat =
      get_buffer_view(slot.expanded_block_tables_host,
                      block_table_elements,
                      torch::TensorOptions()
                          .dtype(torch::kInt)
                          .device(torch::kCPU)
                          .pinned_memory(true));
  torch::Tensor expanded_block_tables_host =
      expanded_block_tables_host_flat.view({expanded_rows, block_table_width});
  const int32_t* source_block_tables = block_tables_host.data_ptr<int32_t>();
  int32_t* expanded_block_tables =
      expanded_block_tables_host.data_ptr<int32_t>();
  for (int64_t row = 0; row < expanded_rows; ++row) {
    const int64_t source_row =
        static_cast<int64_t>(expanded_row_sequences[static_cast<size_t>(row)]);
    std::memcpy(expanded_block_tables + row * block_table_width,
                source_block_tables + source_row * block_table_width,
                static_cast<size_t>(block_table_width) * sizeof(int32_t));
  }

  torch::Tensor expanded_kv_seq_lens_host =
      fill_host_buffer(expanded_kv_seq_lens, slot.expanded_kv_seq_lens_host);
  input_params.graph.expanded_kv_seq_lens = copy_to_device_buffer(
      expanded_kv_seq_lens_host, slot.expanded_kv_seq_lens_device, device_);
  input_params.graph.expanded_block_tables = copy_to_device_buffer(
      expanded_block_tables_host, slot.expanded_block_tables_device, device_);
  input_params.graph.expanded_kv_seq_lens_vec = expanded_kv_seq_lens;

  std::vector<int32_t> paged_kv_indptr;
  std::vector<int32_t> paged_kv_indices;
  std::vector<int32_t> paged_kv_last_page_len;
  specBuilder::build_expanded_decode_paged_kv(expanded_block_tables_host,
                                              expanded_kv_seq_lens,
                                              block_size,
                                              paged_kv_indptr,
                                              paged_kv_indices,
                                              paged_kv_last_page_len);

  input_params.graph.expanded_paged_kv_indptr_host =
      fill_host_buffer(paged_kv_indptr, slot.paged_kv_indptr_host);
  input_params.graph.expanded_paged_kv_indices_host =
      fill_host_buffer(paged_kv_indices, slot.paged_kv_indices_host);
  input_params.graph.expanded_paged_kv_last_page_len_host = fill_host_buffer(
      paged_kv_last_page_len, slot.paged_kv_last_page_len_host);
  input_params.graph.expanded_paged_kv_indptr =
      copy_to_device_buffer(input_params.graph.expanded_paged_kv_indptr_host,
                            slot.paged_kv_indptr_device,
                            device_);
  input_params.graph.expanded_paged_kv_indices =
      copy_to_device_buffer(input_params.graph.expanded_paged_kv_indices_host,
                            slot.paged_kv_indices_device,
                            device_);
  input_params.graph.expanded_paged_kv_last_page_len = copy_to_device_buffer(
      input_params.graph.expanded_paged_kv_last_page_len_host,
      slot.paged_kv_last_page_len_device,
      device_);
  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
}

void MtpGraphBuffers::mark_consumed(const ModelInputParams& input_params,
                                    StreamEventPtr ready_event) {
  if (!input_params.num_accepted_tokens.defined()) {
    return;
  }
  const int32_t* accepted_prefix_lengths =
      input_params.num_accepted_tokens.data_ptr<int32_t>();
  for (BufferSlot& slot : slots_) {
    if (!slot.in_use || !slot.accepted_prefix_lengths_device.defined() ||
        slot.accepted_prefix_lengths_device.data_ptr<int32_t>() !=
            accepted_prefix_lengths) {
      continue;
    }
    slot.ready_event = std::move(ready_event);
    slot.in_use = false;
    return;
  }
  LOG(FATAL) << "MUSA MTP graph buffer slot was not found";
}

}  // namespace xllm::runtime::musa
