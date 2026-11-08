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
#include <string>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/musa/glm5_next/glm5_next_linear_attention.h"
#include "core/layers/musa/glm5_next/glm5_next_mla_attention.h"
#include "core/layers/musa/hyper_connection.h"
#include "core/layers/musa/routed_moe.h"
#include "core/layers/musa/routed_moe_gate.h"

namespace xllm::layer {

enum class Glm5NextAttentionRole : int8_t {
  KDA = 0,
  DSA = 1,
};

enum class Glm5NextMlpRole : int8_t {
  DENSE = 0,
  SPARSE = 1,
};

struct Glm5NextLayerRole final {
  Glm5NextAttentionRole attention = Glm5NextAttentionRole::DSA;
  Glm5NextMlpRole mlp = Glm5NextMlpRole::SPARSE;
};

Glm5NextLayerRole resolve_glm5_next_layer_role(const ModelArgs& args,
                                               int32_t layer_id);

// Model-independent mHC around the MUSA KDA/DSA attention and Dense/Sparse-MoE
// FFN layers.
class Glm5NextDecoderLayerImpl final : public torch::nn::Module {
 public:
  Glm5NextDecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  torch::Tensor forward(
      torch::Tensor& hidden_states,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params,
      std::optional<PendingMHC>* pending_mhc = nullptr,
      bool is_last_layer = true,
      kernel::musa::Glm5NextIndexerSelection* shared_selection = nullptr);

 private:
  int32_t layer_id_ = 0;
  Glm5NextLayerRole role_;
  bool enable_indexer_ = true;

  MHCPre attn_hc_pre_{nullptr};
  MHCPre ffn_hc_pre_{nullptr};
  MHCPost hc_post_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  Glm5NextLinearAttention linear_attention_{nullptr};
  Glm5NextMLAAttention mla_attention_{nullptr};
  DenseMLP dense_mlp_{nullptr};
  RoutedMoeGate sparse_gate_{nullptr};
  RoutedMoe sparse_moe_{nullptr};
};

TORCH_MODULE(Glm5NextDecoderLayer);

}  // namespace xllm::layer
