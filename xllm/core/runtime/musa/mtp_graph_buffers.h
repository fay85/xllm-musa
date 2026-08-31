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

#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <vector>

#include "core/platform/stream_event.h"

namespace xllm {
struct ModelInputParams;
}

namespace xllm::runtime::musa {

struct SpecVerifyBucketShape {
  int32_t live_batch_size;
  int32_t bucket_batch_size;
  int32_t query_width;
  uint32_t actual_num_tokens;
  uint32_t padded_num_tokens;
};

SpecVerifyBucketShape get_mtp_spec_verify_bucket_shape(
    uint32_t actual_num_tokens,
    int32_t live_batch_size,
    int32_t query_width);
uint64_t get_mtp_spec_verify_graph_key(const SpecVerifyBucketShape& shape);
uint64_t get_mtp_spec_verify_graph_key(uint32_t padded_num_tokens,
                                       int32_t query_width);

class MtpGraphBuffers final {
 public:
  explicit MtpGraphBuffers(torch::Device device);

  void prepare(ModelInputParams& input_params,
               const std::vector<int32_t>& accepted_prefix_lengths,
               int32_t block_size,
               bool build_expanded_attention);
  void mark_consumed(const ModelInputParams& input_params,
                     StreamEventPtr ready_event);

 private:
  struct BufferSlot {
    torch::Tensor accepted_prefix_lengths_host;
    torch::Tensor accepted_prefix_lengths_device;
    torch::Tensor expanded_kv_seq_lens_host;
    torch::Tensor expanded_kv_seq_lens_device;
    torch::Tensor expanded_block_tables_host;
    torch::Tensor expanded_block_tables_device;
    torch::Tensor paged_kv_indptr_host;
    torch::Tensor paged_kv_indptr_device;
    torch::Tensor paged_kv_indices_host;
    torch::Tensor paged_kv_indices_device;
    torch::Tensor paged_kv_last_page_len_host;
    torch::Tensor paged_kv_last_page_len_device;
    StreamEventPtr ready_event;
    bool in_use = false;
  };

  BufferSlot& acquire_slot();

  torch::Device device_;
  std::array<BufferSlot, 2> slots_;
  int32_t next_slot_ = 0;
};

}  // namespace xllm::runtime::musa
