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

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace xllm::kernel::musa {

// MUSA matmul parameters. The optional output is caller-owned persistent
// storage used to keep graph-captured addresses stable.
struct MatmulParams {
  torch::Tensor a;
  torch::Tensor b;
  std::optional<torch::Tensor> bias;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct Fp8BlockMatmulParams {
  torch::Tensor a;
  torch::Tensor b;
  torch::Tensor a_scale;
  torch::Tensor b_scale;
  torch::ScalarType output_dtype;
  std::optional<torch::Tensor> output;
};

struct MateGatedDeltaRulePrefillParams {
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor g;
  torch::Tensor beta;
  std::optional<float> scale = std::nullopt;
  std::optional<torch::Tensor> initial_state = std::nullopt;
  std::optional<torch::Tensor> cu_seqlens = std::nullopt;
  std::optional<torch::Tensor> cu_seqlens_kkt = std::nullopt;
  std::optional<std::vector<int32_t>> cu_seqlens_host = std::nullopt;
  std::optional<torch::Tensor> output = std::nullopt;
  std::optional<torch::Tensor> final_state = std::nullopt;
  std::optional<torch::Tensor> kkt_output = std::nullopt;
  bool output_final_state = true;
  bool use_qk_l2norm_in_kernel = true;
  bool allow_inplace_qk_l2norm = false;
};

struct MateGatedDeltaRuleDecodeParams {
  torch::Tensor mixed_qkv;
  torch::Tensor state;
  torch::Tensor A_log;
  torch::Tensor a;
  torch::Tensor dt_bias;
  torch::Tensor b;
  torch::Tensor state_indices;
  int64_t num_k_heads = 0;
  int64_t num_v_heads = 0;
  int64_t head_k_dim = 0;
  int64_t head_v_dim = 0;
  double scale = 0.0;
  bool use_qk_l2norm = true;
  std::optional<torch::Tensor> decode_output = std::nullopt;
  std::optional<torch::Tensor> q_buf = std::nullopt;
  std::optional<torch::Tensor> k_buf = std::nullopt;
  std::optional<torch::Tensor> v_buf = std::nullopt;
};

struct MateGatedDeltaRuleMtpParams {
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor A_log;
  torch::Tensor a;
  torch::Tensor dt_bias;
  torch::Tensor b;
  torch::Tensor state;
  torch::Tensor state_indices;
  torch::Tensor intermediate;
  torch::Tensor output;
  int64_t num_k_heads = 0;
  int64_t num_v_heads = 0;
  int64_t head_k_dim = 0;
  int64_t head_v_dim = 0;
  double scale = 0.0;
};

torch::Tensor matmul(MatmulParams& params);

torch::Tensor fp8_block_matmul(Fp8BlockMatmulParams& params);

std::tuple<torch::Tensor, torch::Tensor> per_token_group_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size);

void mul_sigmoid_gate_inplace(torch::Tensor& out, const torch::Tensor& gate);

}  // namespace xllm::kernel::musa
