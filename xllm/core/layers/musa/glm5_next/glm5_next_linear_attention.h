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

#include <array>
#include <string>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/musa/kda.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/musa/rms_norm_gated.h"

namespace xllm {
namespace layer {

// Packs the four head-sharded projections and two replicated low-rank
// projections into one local GEMM. Checkpoint shards may arrive in any order.
class Glm5NextLinearAttentionInputProjectionImpl final
    : public torch::nn::Module {
 public:
  Glm5NextLinearAttentionInputProjectionImpl(
      int64_t hidden_size,
      int64_t num_heads,
      int64_t head_dim,
      int32_t rank,
      int32_t tp_size,
      const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix = "") const;
  torch::Tensor forward(const torch::Tensor& hidden_states);

 private:
  int32_t rank_;
  int32_t tp_size_;
  std::array<int64_t, 6> shard_sizes_;
  std::array<bool, 6> loaded_{};
  torch::Tensor weight_;
};
TORCH_MODULE(Glm5NextLinearAttentionInputProjection);

// Packed eager KDA self-attention: [T, hidden] BF16 -> [T, hidden] BF16.
// Cache uses framework MUSA layouts: conv [slots, 3*Hlocal*128, 3] BF16,
// recurrent [slots, Hlocal, 128, 128] FP32 (V first). Caller restores any
// prefix state into unique destination slots before forward. Fresh prefill
// resets both histories via has_initial_states=false. MTP, mixed batches,
// graph capture, distinct source slots and CP are explicitly unsupported.
// Weight prefixes are stripped by the caller (typically self_attn.).
class Glm5NextLinearAttentionImpl final : public torch::nn::Module {
 public:
  Glm5NextLinearAttentionImpl() = default;
  Glm5NextLinearAttentionImpl(const ModelArgs& args,
                              const QuantArgs& quant_args,
                              const ParallelArgs& parallel_args,
                              const torch::TensorOptions& options,
                              float gate_lower_bound = -5.0f);
  explicit Glm5NextLinearAttentionImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix = "") const;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  torch::Tensor get_linear_state_indices(const ModelInputParams& input_params,
                                         const torch::Device& device) const;
  int64_t get_checkpoint_stride(const KVCache& kv_cache) const;

  int64_t num_heads_ = 0;
  int64_t local_num_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t projection_size_ = 0;
  int64_t local_projection_size_ = 0;
  int64_t tp_size_ = 1;
  int64_t rank_ = 0;
  int32_t conv_kernel_size_ = 0;
  float gate_lower_bound_ = -5.0f;

  Glm5NextLinearAttentionInputProjection in_proj_qkvbfg_a_{nullptr};
  ColumnParallelLinear f_b_proj_{nullptr};
  ColumnParallelLinear g_b_proj_{nullptr};
  ColumnParallelLinear q_conv1d_{nullptr};
  ColumnParallelLinear k_conv1d_{nullptr};
  ColumnParallelLinear v_conv1d_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  musa::RmsNormGated o_norm_{nullptr};

  bool o_norm_is_loaded_ = false;

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);
};

TORCH_MODULE(Glm5NextLinearAttention);

using Glm5NextKDAInputProjectionImpl =
    Glm5NextLinearAttentionInputProjectionImpl;
using Glm5NextKDAInputProjection = Glm5NextLinearAttentionInputProjection;
using Glm5NextKDAImpl = Glm5NextLinearAttentionImpl;
using Glm5NextKDA = Glm5NextLinearAttention;

}  // namespace layer
}  // namespace xllm
