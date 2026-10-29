/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_attention.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_gate.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_moe.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_sp_context.h"

namespace xllm {
namespace layer {

class DeepseekV4DecoderLayerImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV4DecoderLayerImpl(const ModelContext& context,
                                      int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  void write_context_kv(const torch::Tensor& hidden_states,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin,
                        const torch::Tensor& cache_slots,
                        KVCache& kv_cache) {
    attention_->write_context_kv(
        hidden_states, cos, sin, cache_slots, kv_cache);
  }

  void set_sequence_parallel_context(
      const dsv4_musa_sp::DeepseekV4SPContext* context) {
    sequence_parallel_context_ = context;
  }

  void set_pcp_context(dsv4_musa_pcp::Context* context) {
    pcp_context_ = context;
    attention_->set_pcp_context(context);
    if (prefill_attention_) {
      prefill_attention_->set_pcp_context(context);
    }
  }

  torch::Tensor forward(
      torch::Tensor& x,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_pre(
      const torch::Tensor& x,
      const torch::Tensor& hc_fn,
      const torch::Tensor& hc_scale,
      const torch::Tensor& hc_base,
      const torch::Tensor& norm_weight,
      bool* output_is_normalized);

  torch::Tensor hc_post(const torch::Tensor& x,
                        const torch::Tensor& residual,
                        const torch::Tensor& post,
                        const torch::Tensor& comb);

  DeepseekV4Attention attention_{nullptr};
  DeepseekV4Attention prefill_attention_{nullptr};
  DeepseekV4MoE moe_mlp_{nullptr};
  DeepseekV4Gate gate_{nullptr};
  RMSNorm attn_norm_{nullptr};
  RMSNorm ffn_norm_{nullptr};

  int64_t hc_mult_ = 1;
  int64_t hc_sinkhorn_iters_ = 0;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;
  const dsv4_musa_sp::DeepseekV4SPContext* sequence_parallel_context_ = nullptr;
  dsv4_musa_pcp::Context* pcp_context_ = nullptr;
  torch::Tensor hc_pre_output_workspace_;
  torch::Tensor hc_pre_post_workspace_;
  torch::Tensor hc_pre_comb_workspace_;
  torch::Tensor hc_post_output_workspace_;
  torch::Tensor rms_norm_output_workspace_;

  DEFINE_WEIGHT(hc_attn_fn);
  DEFINE_WEIGHT(hc_ffn_fn);
  DEFINE_WEIGHT(hc_attn_base);
  DEFINE_WEIGHT(hc_ffn_base);
  DEFINE_WEIGHT(hc_attn_scale);
  DEFINE_WEIGHT(hc_ffn_scale);
};
TORCH_MODULE(DeepseekV4DecoderLayer);

}  // namespace layer
}  // namespace xllm
