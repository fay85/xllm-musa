/* Copyright 2026 The xLLM Authors.

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

#include <string>

#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_attention.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"
#include "layers/mlu/hyper_connection.h"
#include "layers/mlu/sparse_moe_block.h"

namespace xllm::layer {
class DeepseekV41DecoderLayerImpl final : public torch::nn::Module {
 public:
  DeepseekV41DecoderLayerImpl(const ModelContext& context, int32_t layer_id);
  void load_state_dict(const StateDict& state);
  std::unique_ptr<DeepseekV41EngramImpl::HashState> create_hash_state(
      int64_t rows) const;
  void load_hash_state(const StateDict& state);
  void verify_loaded_weights(const std::string& prefix) const;
  torch::Tensor forward(torch::Tensor hidden,
                        torch::Tensor& pre_mix,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const torch::Tensor& rope,
                        std::vector<KVCache>& caches,
                        const ModelInputParams& input,
                        Csa2Context& shared,
                        DeepseekV41EngramImpl::HashState* hash_state = nullptr);

 private:
  DeepseekV41Attention attention_{nullptr};
  DeepseekV41Engram engram_{nullptr};
  RMSNorm attn_norm_{nullptr}, ffn_norm_{nullptr};
  MHCPre attn_hc_{nullptr}, ffn_hc_{nullptr};
  MHCPost post_{nullptr};
  SparseMoEBlock moe_{nullptr};
};
TORCH_MODULE(DeepseekV41DecoderLayer);
}  // namespace xllm::layer
