/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <vector>

namespace xllm::layer::musa {

// Metadata consumed only by MUSA attention, graph replay, and hybrid GDN
// implementations. Keeping it as a value member preserves AttentionMetadata
// copy semantics while isolating backend-specific fields from the common type.
struct MusaAttentionMetadata {
  torch::Tensor paged_kv_indptr_host;
  torch::Tensor paged_kv_indices_host;
  torch::Tensor paged_kv_last_page_len_host;
  std::vector<int32_t> q_cu_seq_lens_host_vec;
  bool share_fa3_scheduler_metadata = false;
  mutable torch::Tensor fa3_scheduler_metadata;
};

}  // namespace xllm::layer::musa
