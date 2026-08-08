/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

class PiecewiseGraphMatmulBufferPool final {
 public:
  torch::Tensor get(const torch::Tensor& input, const torch::Tensor& weight);
  torch::Tensor get_gated_rms_norm_output(const torch::Tensor& input);
  torch::Tensor get_gdn_query(const torch::Tensor& input);
  torch::Tensor get_gdn_key(const torch::Tensor& input);
  torch::Tensor get_gdn_value(const torch::Tensor& input);
  torch::Tensor get_gdn_output(const torch::Tensor& input);
  torch::Tensor get_gdn_gate(const torch::Tensor& input);
  torch::Tensor get_gdn_beta(const torch::Tensor& input);
  torch::Tensor get_gdn_initial_state(const torch::Tensor& reference);
  torch::Tensor get_gdn_final_state(const torch::Tensor& reference);
  torch::Tensor get_gdn_kkt(const torch::Tensor& key, int64_t num_v_heads);
  void freeze();
  // Start of a capture/replay forward: reuse slot rings from index 0.
  void reset_forward_slots();

 private:
  // Each entry is a ring of same-shaped tensors. get_tensor() advances
  // the cursor so concurrent live values of the same shape (MoE gate,
  // shared-expert, residual projections, multi-layer GDN) do not alias.
  struct BufferRing {
    std::vector<torch::Tensor> bufs;
    size_t next = 0;
  };
  torch::Tensor get_tensor(std::vector<BufferRing>& rings,
                           c10::IntArrayRef sizes,
                           const torch::TensorOptions& options,
                           const char* name);
  void reset_rings(std::vector<BufferRing>& rings);

  std::vector<BufferRing> output_bufs_;
  std::vector<BufferRing> gated_rms_norm_output_bufs_;
  std::vector<BufferRing> gdn_query_bufs_;
  std::vector<BufferRing> gdn_key_bufs_;
  std::vector<BufferRing> gdn_value_bufs_;
  std::vector<BufferRing> gdn_output_bufs_;
  std::vector<BufferRing> gdn_gate_bufs_;
  std::vector<BufferRing> gdn_beta_bufs_;
  std::vector<BufferRing> gdn_initial_state_bufs_;
  std::vector<BufferRing> gdn_final_state_bufs_;
  std::vector<BufferRing> gdn_kkt_bufs_;
  bool frozen_ = false;
};

class PiecewiseGraphMatmulBufferScope final {
 public:
  explicit PiecewiseGraphMatmulBufferScope(
      PiecewiseGraphMatmulBufferPool* buffer_pool);
  ~PiecewiseGraphMatmulBufferScope();

  PiecewiseGraphMatmulBufferScope(const PiecewiseGraphMatmulBufferScope&) =
      delete;
  PiecewiseGraphMatmulBufferScope& operator=(
      const PiecewiseGraphMatmulBufferScope&) = delete;

  static PiecewiseGraphMatmulBufferPool* current_buffer_pool();

 private:
  PiecewiseGraphMatmulBufferPool* previous_buffer_pool_;
};

}  // namespace xllm::layer::musa
