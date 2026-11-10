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
#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace xllm::kernel::npu {

// One owner per captured graph. Context and scratch are reused only by nodes
// captured on the same stream; other streams and graph slots stay independent.
class PagedAttentionGraphResources final {
 public:
  explicit PagedAttentionGraphResources(int64_t num_layers);
  ~PagedAttentionGraphResources();
  size_t stream_count() const;

 private:
  friend class PagedAttentionGraphOperation;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Retains the ATB operation and scratch storage referenced by an updatable
// graph task. Setup regenerates tiling and launch dimensions for each replay.
class PagedAttentionGraphOperation final {
 public:
  PagedAttentionGraphOperation(
      std::shared_ptr<PagedAttentionGraphResources> resources,
      torch::Tensor query,
      torch::Tensor key,
      torch::Tensor value,
      torch::Tensor block_table,
      torch::Tensor output,
      double scale);
  ~PagedAttentionGraphOperation();

  void prepare(const std::vector<int32_t>& kv_seq_lens);
  void execute();
  uint64_t workspace_size() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm::kernel::npu
