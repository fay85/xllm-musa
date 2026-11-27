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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/message.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/request/structured_output_compiler.h"
#include "core/util/binary_payload.h"
#include "framework/chat_template/chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/request/request_state.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {

class Call;
class Tokenizer;

// Factory that assembles a domain Request aggregate for the VLM path from raw
// user input (a text prompt with multimodal data, or chat messages plus a
// multimodal payload) and RequestParams. It encapsulates multimodal
// preprocessing, prompt tokenization via the processor, chat-template
// rendering, length validation, sampling/stopping parameter assembly, and
// optional structured-output (JSON object / schema / regex / EBNF) constraints.
//
// The factory holds non-owning references to model configuration owned by
// VLMMaster (processor, chat template, tokenizer, model args, options, rate
// limiter); the owner must outlive the factory. Grammar compilers and caches
// are model-scoped and owned here; sequences retain their immutable
// compilation results.
class VLMRequestFactory final {
 public:
  VLMRequestFactory(MultimodalProcessorBase* processor,
                    ChatTemplate* chat_template,
                    const Tokenizer* tokenizer,
                    const ModelArgs* model_args,
                    const Options* options,
                    RateLimiter* rate_limiter,
                    OutputsFunc rpc_response_handler);

  ~VLMRequestFactory() = default;

  Status validate_multimodal_request(const RequestParams& sp) const;
  Status preprocess(PreparedMultimodalInput& input) const;

  // completion: a text prompt plus already-decoded multimodal data.
  std::shared_ptr<Request> create_request(
      std::string prompt,
      MMData mm_data,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback,
      std::optional<ChatTemplateGenerationMode> generation_mode = std::nullopt);

  // chat: decodes the multimodal payload and renders messages into a prompt,
  // then delegates to the completion overload above.
  std::shared_ptr<Request> create_request(std::vector<Message> messages,
                                          const RequestParams& sp,
                                          BinaryPayload payload,
                                          std::optional<Call*> call,
                                          OutputCallback callback,
                                          bool use_prompt_as_is = false);

 private:
  OutputsFunc rpc_response_handler_;

  Status process_media(const std::vector<Message>& messages,
                       BinaryPayload payload,
                       MMData& mm_data) const;

  std::shared_ptr<Request> build_request(
      std::string prompt,
      std::vector<int32_t> prompt_tokens,
      MMData mm_data,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback,
      std::optional<ChatTemplateGenerationMode> generation_mode);

  // Builds the stopping checker, encoding any stop sequences. Returns
  // std::nullopt after firing an error callback when a stop sequence fails to
  // encode.
  std::optional<StoppingChecker> build_stopping_checker(
      const RequestParams& sp,
      uint32_t max_tokens,
      int32_t max_context_len,
      const OutputCallback& callback);

  // Attaches the JSON-object grammar to req_state when requested. Returns
  // false after firing an error callback on failure.
  bool apply_json_object_grammar(
      RequestState& req_state,
      const RequestParams& sp,
      std::optional<ChatTemplateGenerationMode> generation_mode,
      const OutputCallback& callback);

  MultimodalProcessorBase* processor_ = nullptr;
  ChatTemplate* chat_template_ = nullptr;
  const Tokenizer* tokenizer_ = nullptr;
  const ModelArgs* model_args_ = nullptr;
  const Options* options_ = nullptr;
  RateLimiter* rate_limiter_ = nullptr;
  mutable MMInputTransfer mm_input_transfer_;
  std::unique_ptr<StructuredOutputCompiler> structured_output_;
};

}  // namespace xllm
