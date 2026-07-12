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

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>

#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/kernels/param.h"

namespace xllm::kernel::cuda {

struct AttentionReplayParams {
  ffi::Array<int64_t> plan_info;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor paged_kv_indptr_host;
  torch::Tensor paged_kv_indices_host;
  torch::Tensor paged_kv_last_page_len_host;
  std::optional<torch::Tensor> qo_indptr;
  uint32_t actual_num_tokens;
};

class AttentionRunner final {
 public:
  AttentionRunner() = default;

  void run_capture(const std::string& uri,
                   ffi::Array<int64_t> plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   uint32_t padded_num_tokens);

  void run_chunked_prefill_capture(
      const std::string& uri,
      ffi::Array<int64_t> plan_info,
      torch::Tensor float_workspace_buffer,
      torch::Tensor int_workspace_buffer,
      torch::Tensor page_locked_int_workspace_buffer,
      torch::Tensor query,
      torch::Tensor k_cache,
      torch::Tensor v_cache,
      torch::Tensor paged_kv_indptr,
      torch::Tensor paged_kv_indices,
      torch::Tensor paged_kv_last_page_len,
      int64_t window_left,
      double sm_scale,
      torch::Tensor output,
      std::optional<torch::Tensor>& output_lse,
      std::optional<torch::Tensor> qo_indptr,
      bool causal,
      const torch::Tensor& paged_kv_indptr_host,
      const torch::Tensor& paged_kv_indices_host,
      const torch::Tensor& paged_kv_last_page_len_host,
      uint32_t padded_num_tokens);

  void run_replay(const AttentionReplayParams& params);

 private:
  enum class RunnerType { PREFILL, CHUNKED_PREFILL };

  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;

  torch::Tensor query_;
  torch::Tensor key_;
  torch::Tensor value_;
  torch::Tensor k_cache_;
  torch::Tensor v_cache_;
  torch::Tensor output_;

  std::string uri_;
  int64_t window_size_left_;
  double scale_;
  uint32_t padded_num_tokens_;
  RunnerType runner_type_ = RunnerType::PREFILL;
  bool causal_ = true;
};

}
