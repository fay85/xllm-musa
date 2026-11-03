/* Copyright 2025-2026 The xLLM Authors.

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

#include <string>
#include <tuple>
#include <utility>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention.h"

#include "layers/common/linear.h"
#include "layers/musa/normalized_hidden_states.h"
#include "layers/musa/rms_norm_gated.h"

namespace xllm {
namespace layer {

bool use_mate_gdn_mtp_kernel();
void scatter_gdn_mtp_verify_ssm_states(
    const GdnMtpVerifyCache& cache,
    const std::vector<KVCache>& kv_caches,
    const ModelInputParams& input_params,
    const torch::Tensor& accepted_tokens);

class GdnMtpVerifyForwardScope final {
 public:
  explicit GdnMtpVerifyForwardScope(const GdnMtpVerifyCache* cache);
  ~GdnMtpVerifyForwardScope();

  GdnMtpVerifyForwardScope(const GdnMtpVerifyForwardScope&) = delete;
  GdnMtpVerifyForwardScope& operator=(const GdnMtpVerifyForwardScope&) = delete;

 private:
  bool active_ = false;
};
class Qwen3GatedDeltaNetBaseImpl : public torch::nn::Module {
 public:
  Qwen3GatedDeltaNetBaseImpl() = default;
  Qwen3GatedDeltaNetBaseImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;

  torch::Tensor forward(const NormalizedHiddenStates& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

  // Whether forward() consumes NormalizedHiddenStates::fp8_values.
  virtual bool supports_fp8_input() const = 0;

 protected:
  virtual std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const NormalizedHiddenStates& hidden_states) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const NormalizedHiddenStates& hidden_states) = 0;

  static torch::Tensor project_input(
      ColumnParallelLinear& projection,
      const NormalizedHiddenStates& hidden_states);
  virtual bool use_fla_ssm_state_layout() const { return false; }
  virtual bool uses_contiguous_qkvzba_layout() const { return false; }
  // BF16 weight of the ba projection when spec verify may apply it inside the
  // convolution launch; undefined when the input projections must project ba.
  virtual torch::Tensor fusable_ba_weight() const { return torch::Tensor(); }
  // The qkvz half of project_flat_inputs().
  virtual torch::Tensor project_qkvz(
      const NormalizedHiddenStates& hidden_states) {
    return project_flat_inputs(hidden_states).first;
  }
  // fusable_ba_weight() when this forward computes ba inside the spec-verify
  // convolution launch; undefined otherwise.
  torch::Tensor conv_fused_ba_weight(
      const NormalizedHiddenStates& hidden_states,
      const AttentionMetadata& attn_metadata,
      bool use_spec_verify) const;

  void load_common_state_dict(const StateDict& state_dict);
  void verify_common_loaded_weights(const std::string& prefix) const;

  torch::Tensor get_linear_state_indices(const ModelInputParams& input_params,
                                         const torch::Device& device) const;

  std::pair<torch::Tensor, torch::Tensor> project_padded_inputs(
      const NormalizedHiddenStates& hidden_states,
      const AttentionMetadata& attn_metadata);

  // Splits projected qkvz and ba rows into contiguous {mixed_qkv, z, b, a}.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  split_qkvzba(const torch::Tensor& qkvz_flat, const torch::Tensor& ba_flat);

  torch::Tensor reshape_qkvz_unpad(const AttentionMetadata& attn_metadata,
                                   const torch::Tensor& padded_qkvz) const;

  // Whether the gated output norm can emit out_proj's FP8 input directly:
  // `head_rows` holds one 128-wide row per token head and needs no unpadding.
  bool can_fuse_output_norm_quant(const AttentionMetadata& attn_metadata,
                                  const torch::Tensor& head_rows,
                                  int64_t num_tokens) const;

  torch::Tensor reshape_qkvz_with_pad(const AttentionMetadata& attn_metadata,
                                      const torch::Tensor& qkvz) const;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_mixed_qkv(
      torch::Tensor& mixed_qkv) const;

  int64_t num_k_heads_ = 0;
  int64_t num_v_heads_ = 0;
  int64_t head_k_dim_ = 0;
  int64_t head_v_dim_ = 0;
  int64_t k_size_ = 0;
  int64_t v_size_ = 0;
  int64_t tp_size_ = 1;
  int64_t rank_ = 0;
  int32_t conv_kernel_size_ = 0;

  ColumnParallelLinear conv1d_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  musa::RmsNormGated norm_{nullptr};

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);

  // Persistent buffers used by the fused GDN split and decode kernels.
  mutable torch::Tensor mixed_qkv_out_buf_;
  mutable torch::Tensor z_out_buf_;
  mutable torch::Tensor b_out_buf_;
  mutable torch::Tensor a_out_buf_;

  // Persistent output for causal convolution during decode.
  mutable torch::Tensor conv1d_decode_out_buf_;

  // Persistent output for the fused gated-delta-rule decode kernel.
  mutable torch::Tensor fused_gdn_decode_out_buf_;
};

}  // namespace layer
}  // namespace xllm
