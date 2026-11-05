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

#include <cstdint>

namespace xllm::kernel::musa {

// Bounded-sigmoid KDA gate: log_decay = gate_lower_bound * sigmoid(
// exp(A_log) * (raw_forget + dt_bias)). Both values must be set by the caller;
// gate_lower_bound is finite and negative, scale is finite and positive.
struct KdaOptions {
  float gate_lower_bound = 0.0f;
  float scale = 0.0f;
};

// Current MUSA specialization: dense, equal-head KDA, head dimension 128.
// Q/K L2 normalization is always applied with epsilon 1e-6.
// Activations and raw gates are BF16;
// A_log, dt_bias and the V-first [B,H,128,128] initial state are FP32.
// No sigmoid, decay or Q/K normalization has been applied by the caller.
struct KdaInputs {
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor raw_forget;
  torch::Tensor raw_beta;
  torch::Tensor a_log;
  torch::Tensor dt_bias;
  torch::Tensor initial_state;
  KdaOptions options;
};

// Worker-owned scratch, shared by serial layers on one execution stream.
// Allocate before execution; do not share concurrently between streams.
class KdaWorkspace final {
 public:
  explicit KdaWorkspace(int64_t max_batch,
                        int64_t max_tokens,
                        int64_t heads,
                        const torch::Device& device);
  KdaWorkspace(const KdaWorkspace&) = delete;
  KdaWorkspace& operator=(const KdaWorkspace&) = delete;
  KdaWorkspace(KdaWorkspace&&) = default;
  KdaWorkspace& operator=(KdaWorkspace&&) = default;

  int64_t max_batch() const { return max_batch_; }
  int64_t max_tokens() const { return max_tokens_; }
  int64_t heads() const { return heads_; }
  const torch::Tensor& storage() const { return storage_; }
  const torch::Tensor& state_indices() const { return state_indices_; }

 private:
  int64_t max_batch_;
  int64_t max_tokens_;
  int64_t heads_;
  torch::Tensor storage_;
  torch::Tensor state_indices_;
};

// All outputs are caller-owned and must not alias any input. Initial state is
// read-only; the caller explicitly commits final_state to its request slot.
// These initial interfaces exclude varlen packing, slot pools and speculation.
void kda_prefill(const KdaInputs& inputs,
                 torch::Tensor& output,
                 torch::Tensor& final_state,
                 KdaWorkspace& workspace);

void kda_decode(const KdaInputs& inputs,
                torch::Tensor& output,
                torch::Tensor& final_state);

}  // namespace xllm::kernel::musa
