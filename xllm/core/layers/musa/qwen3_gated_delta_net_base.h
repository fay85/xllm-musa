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
#include "layers/musa/linear.h"
#include "layers/musa/rms_norm_gated.h"

namespace xllm {
namespace layer {

bool use_mate_gdn_mtp_kernel();
bool use_mate_gdn_prefill_kernel();

// After MTP rejection sampling, scatter per-layer intermediate SSM/conv states
// into the live linear cache slots (post-verify commit).
void scatter_gdn_mtp_verify_ssm_states(const GdnMtpVerifyCache& cache,
                                       const std::vector<KVCache>& kv_caches,
                                       const ModelInputParams& input_params,
                                       const torch::Tensor& accepted_tokens);

class Qwen3GatedDeltaNetBaseImpl : public torch::nn::Module {
 public:
  Qwen3GatedDeltaNetBaseImpl() = default;
  Qwen3GatedDeltaNetBaseImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 protected:
  // Eager multi-seq pure-prefill path that keeps tokens packed through
  // proj/conv/gating, then either feeds Mate varlen (high waste) or pads
  // once at the Mate boundary for the padded warp (low waste).
  torch::Tensor forward_packed_prefill(const torch::Tensor& hidden_states,
                                       const AttentionMetadata& attn_metadata,
                                       KVCache& kv_cache,
                                       const ModelInputParams& input_params);
  virtual std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) = 0;
  virtual bool use_fla_ssm_state_layout() const { return false; }
  virtual bool uses_contiguous_qkvzba_layout() const { return false; }

  void load_common_state_dict(const StateDict& state_dict);
  void verify_common_loaded_weights(const std::string& prefix) const;

  torch::Tensor get_linear_state_indices(const ModelInputParams& input_params,
                                         const torch::Device& device) const;

  std::pair<torch::Tensor, torch::Tensor> project_padded_inputs(
      const torch::Tensor& hidden_states,
      const AttentionMetadata& attn_metadata);

  torch::Tensor reshape_qkvz_unpad(const AttentionMetadata& attn_metadata,
                                   const torch::Tensor& padded_qkvz) const;

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

  musa::ColumnParallelLinear conv1d_{nullptr};
  musa::RowParallelLinear o_proj_{nullptr};
  musa::RmsNormGated norm_{nullptr};

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);

  // Persistent output buffers consumed by xllm::kernel::
  // fused_qkvzba_split_reshape_cat in lieu of the libtorch
  // `reshape().contiguous() ... torch::cat()` chain. Same lazy / grow-only
  // pattern used by ColumnParallelLinearImpl::output_buf_ and
  // AttentionImpl::output_buf_: sized on the first forward (during graph
  // warmup), reused on every replay via narrow() views. Eager calls also
  // benefit (one allocation per process instead of per-step).
  mutable torch::Tensor mixed_qkv_out_buf_;
  mutable torch::Tensor z_out_buf_;
  mutable torch::Tensor b_out_buf_;
  mutable torch::Tensor a_out_buf_;

  // Persistent output buffer for the decode-path causal_conv1d_update call.
  // Without this, the kernel falls through to its libtorch slow path
  // (`weight.to(fp32)` / `x.to(fp32)` / `torch::empty_like(x_f32)`) which
  // triggers EmptyStridedMUSA -> MUSA stream-capture abort. Providing a
  // pre-allocated buffer unlocks the in-house `causal_conv1d_decode_fused`
  // fast path (see gdn_ops.cpp::causal_conv1d_update fast-path guard).
  mutable torch::Tensor conv1d_decode_out_buf_;

  // Persistent output buffer for the in-house fused_gated_delta_rule_decode
  // kernel. Wired into `MateGatedDeltaRuleDecodeParams::decode_output` so
  // the kernel skips its `torch::empty({B, Hv, V}, ...)` fallback (which
  // hits EmptyMUSA mid-capture) and writes directly into pre-allocated
  // storage. Reused across replays; same lazy / grow-only contract as the
  // other graph-safe buffers above.
  mutable torch::Tensor fused_gdn_decode_out_buf_;

  // Persistent q/k/v split buffers for the mate GDN decode path. The mate
  // kernel expects contiguous q/k/v tensors, but mixed_qkv is a flat [B, D]
  // row. Without these buffers the wrapper calls .contiguous() on each
  // strided slice, which allocates inside MUSA graph capture and aborts.
  // Pre-allocated here (lazy / grow-only) and filled via .copy_() at replay.
  mutable torch::Tensor mate_gdn_decode_q_buf_;
  mutable torch::Tensor mate_gdn_decode_k_buf_;
  mutable torch::Tensor mate_gdn_decode_v_buf_;

  // Persistent buffers for mate GDN MTP spec-verify (seq_len == 2).
  mutable torch::Tensor mate_gdn_mtp_intermediate_buf_;
  mutable torch::Tensor mate_gdn_mtp_output_buf_;
};

}  // namespace layer
}  // namespace xllm
