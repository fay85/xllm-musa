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

#include <cstdint>
#include <optional>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <tuple>
#include <vector>

namespace xllm {

class ProcessGroup;

class MusaCustomAllReduce {
 public:
  MusaCustomAllReduce(int32_t rank,
                      int32_t world_size,
                      const std::string& group_name,
                      const torch::Device& device,
                      c10::intrusive_ptr<c10d::Store> store);
  ~MusaCustomAllReduce();

  MusaCustomAllReduce(const MusaCustomAllReduce&) = delete;
  MusaCustomAllReduce& operator=(const MusaCustomAllReduce&) = delete;

  bool enabled() const { return enabled_; }

  std::optional<torch::Tensor> allreduce(
      torch::Tensor& input,
      const std::optional<torch::Tensor>& output = std::nullopt);

  void finalize_graph_buffers();
  void long_prefill_host_barrier();
  friend void finalize_musa_custom_allreduce_graphs();

 private:
  bool should_use(const torch::Tensor& input) const;
  void init();
  std::vector<int64_t> create_shared_buffer(int64_t size_in_bytes,
                                            const std::string& name);
  std::vector<uint8_t> tensor_to_bytes(const torch::Tensor& tensor) const;
  torch::Tensor bytes_to_tensor(const std::vector<uint8_t>& bytes) const;
  std::vector<uint8_t> serialize_graph_metadata(
      const std::vector<int64_t>& handles,
      const std::vector<int64_t>& offsets) const;
  std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  deserialize_graph_metadata(const std::vector<uint8_t>& bytes) const;

  int32_t rank_ = 0;
  int32_t world_size_ = 0;
  std::string group_name_;
  torch::Device device_;
  c10::intrusive_ptr<c10d::Store> store_;

  bool enabled_ = false;
  bool fully_connected_ = true;
  int64_t max_size_bytes_ = 8 * 1024 * 1024;
  int64_t custom_ar_handle_ = 0;
  std::vector<int64_t> meta_ptrs_;
  std::vector<int64_t> buffer_ptrs_;
  torch::Tensor rank_data_;
  int64_t graph_capture_generation_ = 0;
};

std::optional<torch::Tensor> musa_allreduce_custom(
    ProcessGroup* process_group,
    torch::Tensor& input,
    const std::optional<torch::Tensor>& output = std::nullopt);

void finalize_musa_custom_allreduce_graphs();
void musa_long_prefill_host_barrier(ProcessGroup* process_group);

}  // namespace xllm
