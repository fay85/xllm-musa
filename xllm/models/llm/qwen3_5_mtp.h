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

#include <memory>

#include "models/llm/qwen3_5.h"
#include "models/llm/qwen3_5_mtp_base.h"
#include "models/model_registry.h"

namespace xllm {

class Qwen3_5MtpModelImpl final : public Qwen3_5MtpModelImplBase {
 public:
  explicit Qwen3_5MtpModelImpl(const ModelContext& context)
      : Qwen3_5MtpModelImplBase(context) {}

#if defined(USE_MUSA)
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    if (input_params.meta.batch_forward_type.is_decode() ||
        input_params.is_spec_verify) {
      if (!input_params.embedding.mtp_token_embedding.defined() &&
          tokens.defined() && tokens.numel() > 0 &&
          this->embed_tokens_ && this->embed_tokens_->weight().defined()) {
        const int64_t vocab_size = this->embed_tokens_->weight().size(0);
        if (vocab_size > 0) {
          tokens = tokens.clamp(/*min=*/0, /*max=*/vocab_size - 1);
        }
      }
      return Qwen3_5MtpModelImplBase::forward(
          tokens, positions, kv_caches, input_params);
    }

    if (!input_params.enable_graph && input_params.attn_metadata != nullptr) {
      ModelInputParams params = input_params;
      params.attn_metadata.reset();
      return Qwen3_5MtpModelImplBase::forward(
          tokens, positions, kv_caches, params);
    }
    return Qwen3_5MtpModelImplBase::forward(
        tokens, positions, kv_caches, input_params);
  }
#endif
};

class Qwen3_5MtpForCausalLMImpl final : public Qwen3_5MtpForCausalLMImplBase {
 public:
  explicit Qwen3_5MtpForCausalLMImpl(const ModelContext& context)
      : Qwen3_5MtpForCausalLMImplBase(
            context,
            std::make_shared<Qwen3_5MtpModelImpl>(context)) {}
};
TORCH_MODULE(Qwen3_5MtpForCausalLM);

REGISTER_CAUSAL_MODEL(qwen3_5_mtp, Qwen3_5MtpForCausalLM);
REGISTER_CAUSAL_MODEL(qwen3_5_moe_mtp, Qwen3_5MtpForCausalLM);

REGISTER_MODEL_ARGS_LOADER(qwen3_5_mtp,
                           [](const JsonReader& json, ModelArgs* args) {
                             return qwen3_5_mtp::load_model_args(
                                 json, args, "qwen3_5_text", "qwen3_5_mtp");
                           });

REGISTER_MODEL_ARGS_LOADER(qwen3_5_moe_mtp,
                           [](const JsonReader& json, ModelArgs* args) {
                             return qwen3_5_mtp::load_model_args(
                                 json,
                                 args,
                                 "qwen3_5_moe_text",
                                 "qwen3_5_moe_mtp");
                           });

}  // namespace xllm
