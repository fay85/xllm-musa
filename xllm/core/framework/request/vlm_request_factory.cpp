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

#include "core/framework/request/vlm_request_factory.h"

#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/macros.h"
#include "common/metrics.h"
#include "core/common/message.h"
#include "core/framework/chat_template/thinking_mode_resolver.h"
#include "core/framework/chat_template/tool_call_arguments.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/multimodal_config.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "framework/multimodal/mm_batch_data.h"
#include "framework/tokenizer/tokenizer.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

bool has_valid_pretokenized_text(const std::vector<int32_t>& token_ids,
                                 const Tokenizer& tokenizer) {
  // Reject a negative first token before FastTokenizer materializes its vocab.
  if (token_ids.empty() || token_ids.front() < 0) {
    return false;
  }
  const int64_t vocab_size = static_cast<int64_t>(tokenizer.vocab_size());
  return vocab_size > 0 &&
         !util::find_out_of_vocab_token(token_ids, vocab_size).has_value();
}

// Extracts the final prompt from a single message, concatenating its text
// contents in order and skipping non-text (image/video/audio) contents.
//
// The offline "prompt as-is" contract expects the caller to hand in exactly
// one message whose text is the final prompt already assembled with the
// chat template (images may ride along as non-text contents). Multi-turn
// conversations must go through the chat template path instead; joining
// their texts would silently mash the roles together, so any other message
// shape returns std::nullopt and fails the request.
std::optional<std::string> join_message_texts(
    const std::vector<Message>& messages) {
  if (messages.size() != 1) {
    return std::nullopt;
  }
  const auto& message = messages.front();
  if (auto text = std::get_if<std::string>(&message.content)) {
    return *text;
  }
  if (auto contents = std::get_if<MMContentVec>(&message.content)) {
    std::string prompt;
    for (const auto& content : *contents) {
      if (content.type == "text") {
        prompt += content.text;
      }
    }
    return prompt;
  }
  return std::nullopt;
}

// Throws std::invalid_argument describing the violation when messages do not
// satisfy the offline "prompt as-is" contract (see join_message_texts).
// Offline-only: raised on the caller thread so pybind surfaces it to the
// Python caller at the call site, instead of returning an empty prompt that
// is hard to trace.
[[noreturn]] void throw_prompt_as_is_error(
    const std::vector<Message>& messages) {
  std::string roles;
  for (const auto& message : messages) {
    if (!roles.empty()) {
      roles += ", ";
    }
    roles += message.role;
  }
  throw std::invalid_argument(
      "use_prompt_as_is expects exactly 1 message holding the final prompt "
      "(images allowed as non-text contents), got " +
      std::to_string(messages.size()) + " messages [" + roles +
      "]; apply the chat template for multi-turn conversations");
}

}  // namespace

VLMRequestFactory::VLMRequestFactory(MultimodalProcessorBase* processor,
                                     ChatTemplate* chat_template,
                                     const Tokenizer* tokenizer,
                                     const ModelArgs* model_args,
                                     const Options* options,
                                     RateLimiter* rate_limiter,
                                     OutputsFunc rpc_response_handler)
    : rpc_response_handler_(std::move(rpc_response_handler)),
      processor_(processor),
      chat_template_(chat_template),
      tokenizer_(tokenizer),
      model_args_(model_args),
      options_(options),
      rate_limiter_(rate_limiter),
      mm_input_transfer_(processor ? processor->video_frame_selector()
                                   : VideoFrameSelector{}) {
  CHECK(processor_ != nullptr);
  CHECK(chat_template_ != nullptr);
  CHECK(tokenizer_ != nullptr);
  CHECK(model_args_ != nullptr);
  CHECK(options_ != nullptr);
  CHECK(rate_limiter_ != nullptr);
  structured_output_ = std::make_unique<StructuredOutputCompiler>(
      *tokenizer_, *model_args_, *options_);
}

Status VLMRequestFactory::preprocess(PreparedMultimodalInput& input) const {
  std::vector<Message> messages;
  messages.reserve(1);
  messages.emplace_back("user", std::move(input.media));
  Status status =
      process_media(messages, std::move(input.payload), input.metadata);
  if (!status.ok()) {
    return status;
  }
  if (input.prompt && !processor_->process_prompt(
                          *input.prompt, input.metadata, input.token_ids)) {
    return {StatusCode::INVALID_ARGUMENT, "Failed to process prompt."};
  }
  return {};
}

Status VLMRequestFactory::process_media(const std::vector<Message>& messages,
                                        BinaryPayload payload,
                                        MMData& mm_data) const {
  const bool has_mm_content =
      std::any_of(messages.begin(), messages.end(), [](const Message& message) {
        return message.has_mm_content();
      });
  MMInput mm_inputs(std::move(payload));
  Timer transfer_timer;
  const MMErrCode code = mm_input_transfer_.trans(messages, mm_inputs);
  if (has_mm_content) {
    HISTOGRAM_OBSERVE(
        mm_transfer_latency_milliseconds,
        static_cast<int64_t>(transfer_timer.elapsed_milliseconds()));
    MULTI_COUNTER_ADD(mm_input_stage_failures_total,
                      "transfer",
                      code == MMErrCode::SUCCESS ? 0 : 1);
  }
  if (code != MMErrCode::SUCCESS) {
    const std::string error_message = MMErrToString(code);
    LOG(ERROR) << error_message;
    if (code == MMErrCode::VIDEO_DECODE_BUSY) {
      return {StatusCode::RESOURCE_EXHAUSTED,
              error_message,
              FailureReason::TEMPORARY_OVERLOAD};
    }
    return {StatusCode::INVALID_ARGUMENT, error_message};
  }
  if (!mm_inputs.empty()) {
    Timer processor_timer;
    const bool ok = processor_->process_multimodal(mm_inputs, mm_data);
    HISTOGRAM_OBSERVE(
        mm_processor_latency_milliseconds,
        static_cast<int64_t>(processor_timer.elapsed_milliseconds()));
    MULTI_COUNTER_ADD(mm_input_stage_failures_total, "processor", ok ? 0 : 1);
    if (!ok) {
      return {StatusCode::INVALID_ARGUMENT,
              "Failed to process multimodal input."};
    }
  }
  return {};
}

Status VLMRequestFactory::validate_multimodal_request(
    const RequestParams& sp) const {
  if (!sp.typed_input_status.ok()) {
    return sp.typed_input_status;
  }

  if (!sp.multimodal_prompt) {
    return sp.validate_multimodal_input();
  }
  const auto invalid = [] {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Typed media requires VLM Prefill with disaggregated PD");
  };
  if (!options_->enable_disagg_pd() || options_->enable_pd_ooc() ||
      InstanceRole(options_->instance_role()) != InstanceRole::PREFILL ||
      !options_->enable_service_routing() || sp.decode_address.empty()) {
    return invalid();
  }
  return {};
}

bool VLMRequestFactory::apply_json_object_grammar(
    RequestState& req_state,
    const RequestParams& sp,
    std::optional<ChatTemplateGenerationMode> generation_mode,
    const OutputCallback& callback) {
  bool reasoning_enabled = false;
  if (generation_mode == ChatTemplateGenerationMode::REASONING) {
    reasoning_enabled = true;
  } else if (generation_mode == ChatTemplateGenerationMode::CHAT) {
    reasoning_enabled = false;
  } else {
    // Multimodal chat templates such as Qwen3-VL do not end the rendered
    // prompt with a think marker, so the generation mode is unknown. Fall
    // back to the request's thinking switch, mirroring the no-mode path.
    reasoning_enabled = thinking_mode::is_enabled(
        sp.chat_template_kwargs,
        ModelConfig::get_instance().reasoning_parser());
  }
  return structured_output_->apply_json_object_grammar(
      req_state, sp, reasoning_enabled, callback);
}

std::shared_ptr<Request> VLMRequestFactory::create_request(
    std::string prompt,
    MMData mm_data,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback,
    std::optional<ChatTemplateGenerationMode> generation_mode) {
  // Guard the rate-limit slot acquired at the service entry. build_request
  // installs its own guard, so we dismiss ours before forwarding.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  Status validation_status = validate_multimodal_request(sp);
  if (!validation_status.ok()) {
    callback({std::move(validation_status),
              sp.service_request_id,
              sp.source_xservice_addr});
    return nullptr;
  }

  std::vector<int32_t> prompt_tokens;
  if (sp.multimodal_prompt) {
    if (!mm_data.empty()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Typed image input cannot include legacy media",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
    const auto& typed_prompt = *sp.multimodal_prompt;
    const auto interrupted = [&]() -> Status {
      if (call.has_value() && *call != nullptr && (*call)->is_disconnected()) {
        return {StatusCode::CANCELLED, "Image request cancelled"};
      }
      return {};
    };
    Status status = interrupted();
    if (status.ok()) {
      status = [&]() -> Status {
        std::vector<Message> messages;
        messages.reserve(1);
        messages.emplace_back("user", typed_prompt.media);
        Status media_status =
            process_media(messages, typed_prompt.payload, mm_data);
        if (!media_status.ok()) {
          return media_status;
        }
        if (typed_prompt.prompt) {
          prompt = *typed_prompt.prompt;
          std::vector<int32_t> tokens;
          if (!processor_->process_prompt(prompt, mm_data, tokens) ||
              tokens != typed_prompt.token_ids) {
            return {StatusCode::INVALID_ARGUMENT,
                    "Prepared media tokens changed"};
          }
        }
        if (!mm_data.hold<MMItemVec>() ||
            !typed_prompt.metadata.hold<MMItemVec>() ||
            mm_data.size() != typed_prompt.metadata.size()) {
          return {StatusCode::INVALID_ARGUMENT, "Prepared media count changed"};
        }
        const auto& expected_items = typed_prompt.metadata.items<MMItemVec>();
        for (size_t index = 0; index < mm_data.size(); ++index) {
          const auto& item = mm_data.items<MMItemVec>()[index];
          const auto& expected = expected_items[index];
          if (item.type() != expected.type() ||
              item.state().schedule_data().key !=
                  expected.state().schedule_data().key) {
            return {StatusCode::INVALID_ARGUMENT,
                    "Local media key differs from prepared metadata"};
          }
          if (!typed_prompt.prompt) {
            const auto grid = item.get<torch::Tensor>("image_grid_thw");
            const auto expected_grid =
                expected.get<torch::Tensor>("image_grid_thw");
            if (!grid || !expected_grid ||
                !torch::equal(*grid, *expected_grid)) {
              return {
                  StatusCode::INVALID_ARGUMENT,
                  "Local processed image grid differs from prepared metadata"};
            }
          }
        }
        if (!typed_prompt.prompt &&
            !processor_->bind_prompt_tokens(typed_prompt.token_ids, mm_data)) {
          return {StatusCode::INVALID_ARGUMENT,
                  "Failed to bind authoritative image tokens"};
        }
        for (size_t index = 0; index < mm_data.size(); ++index) {
          const auto& state = mm_data.items<MMItemVec>()[index].state();
          const auto& expected = expected_items[index].state();
          if (state.token_pos().offset != expected.token_pos().offset ||
              state.token_pos().length != expected.token_pos().length ||
              state.mm_token_num() != expected.mm_token_num()) {
            return {StatusCode::INVALID_ARGUMENT,
                    "Prepared media spans changed"};
          }
        }
        return {};
      }();
      const Status interrupted_status = interrupted();
      if (!interrupted_status.ok()) {
        status = interrupted_status;
      }
    }
    if (!status.ok()) {
      callback(
          {std::move(status), sp.service_request_id, sp.source_xservice_addr});
      return nullptr;
    }
    prompt_tokens = typed_prompt.token_ids;
  } else if (sp.prompt_tokens) {
    // Tokens are authoritative only for text. Media must carry typed spans or
    // pass through the processor so its mRoPE and embedding metadata agree.
    if (mm_data.valid() ||
        !has_valid_pretokenized_text(*sp.prompt_tokens, *tokenizer_)) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Invalid pretokenized VLM text input",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
    if (!processor_->bind_prompt_tokens(*sp.prompt_tokens, mm_data)) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Unbound media marker in pretokenized VLM text input",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
    prompt_tokens = *sp.prompt_tokens;
  } else {
    if (prompt.empty() && mm_data.empty()) {
      LOG(ERROR) << "Prompt and multimodal data cannot be both empty.";
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Prompt and multimodal data are both empty.",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
    if (!processor_->process_prompt(prompt, mm_data, prompt_tokens)) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Failed to process prompt.",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
  }
  rate_limit_guard.dismiss();
  return build_request(std::move(prompt),
                       std::move(prompt_tokens),
                       std::move(mm_data),
                       sp,
                       call,
                       std::move(callback),
                       generation_mode);
}

std::shared_ptr<Request> VLMRequestFactory::build_request(
    std::string prompt,
    std::vector<int32_t> prompt_tokens,
    MMData mm_data,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback,
    std::optional<ChatTemplateGenerationMode> generation_mode) {
  // Guard the rate-limit slot acquired at the service entry. Any early
  // return below releases it; success path dismisses right before Request
  // takes ownership.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  std::shared_ptr<const CompiledXGrammar> xgrammar_constraint;
  GrammarSpec response_format_spec;
  if (!structured_output_->build_response_format_constraint(
          sp, callback, xgrammar_constraint, response_format_spec)) {
    return nullptr;
  }

  const int32_t max_context_len = model_args_->max_position_embeddings();
  int32_t prompt_token_limit = max_context_len;
  if (!options_->enable_chunked_prefill()) {
    prompt_token_limit =
        std::min(prompt_token_limit, options_->max_tokens_per_batch());
  }
  if (prompt_tokens.size() >= static_cast<size_t>(prompt_token_limit)) {
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    callback({Status{StatusCode::INVALID_ARGUMENT,
                     "Prompt is too long",
                     FailureReason::PROMPT_TOO_LONG},
              sp.service_request_id,
              sp.source_xservice_addr});
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens, TODO: add image token size as well.
  size_t capacity = prompt_tokens.size() + max_tokens +
                    options_->num_speculative_tokens() + /*bonus_token*/ 1;
  if (options_->enable_schedule_overlap()) {
    capacity += options_->num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param = sp.to_sampling_param(best_of);
  sampling_param.json_object = structured_output_->uses_builtin_json_object(sp);
  const bool json_object = sampling_param.json_object;
  // Model-aware complement to verify_params' 2000 cap: an oversized top-k
  // would throw inside the sampler and take the whole batch down with it.
  if (const auto error =
          sampling_param.top_logprobs_vocab_error(model_args_->vocab_size())) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        *error,
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  std::optional<StoppingChecker> stopping_checker =
      build_stopping_checker(sp, max_tokens, max_context_len, callback);
  if (!stopping_checker.has_value()) {
    return nullptr;
  }

  // results cannot be streamed when best_of != n
  bool stream = sp.streaming;
  if (best_of != sp.n) {
    stream = false;
  }

  OutputsFunc batch_callback;
  if (options_->enable_service_routing()) {
    // Requests can outlive the factory while the scheduler drains responses.
    batch_callback = [handler = rpc_response_handler_](
                         const std::vector<RequestOutput>& outputs) {
      for (const auto& output : outputs) {
        output.log_request_status();
      }
      return handler(outputs);
    };
  }

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(mm_data),
                         std::move(sampling_param),
                         std::move(stopping_checker.value()),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         options_->enable_schedule_overlap(),
                         callback,
                         batch_callback,
                         /*decode_address=*/sp.decode_address,
                         call);
  req_state.include_stop_str_in_output = sp.include_stop_str_in_output;
  req_state.scheduler_param = sp.to_scheduler_param();
  req_state.timing = sp.timing;
  if (json_object &&
      !apply_json_object_grammar(req_state, sp, generation_mode, callback)) {
    return nullptr;
  }
  structured_output_->attach_xgrammar_constraint(
      req_state,
      std::move(xgrammar_constraint),
      std::move(response_format_spec));

  rate_limit_guard.dismiss();
  // add one sequence, rest will be added by scheduler
  return std::make_shared<Request>(sp.request_id,
                                   sp.x_request_id,
                                   sp.x_request_time,
                                   std::move(req_state),
                                   sp.service_request_id,
                                   sp.source_xservice_addr,
                                   rate_limiter_);
}

std::optional<StoppingChecker> VLMRequestFactory::build_stopping_checker(
    const RequestParams& sp,
    uint32_t max_tokens,
    int32_t max_context_len,
    const OutputCallback& callback) {
  std::unordered_set<int32_t> stop_tokens;
  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
  } else if (!sp.ignore_eos) {
    stop_tokens = model_args_->stop_token_ids();
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int32_t> tmp_tokens;
      if (!tokenizer_->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence",
                            sp.service_request_id,
                            sp.source_xservice_addr);
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return std::nullopt;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  return StoppingChecker(max_tokens,
                         max_context_len,
                         model_args_->eos_token_id(),
                         sp.ignore_eos,
                         std::move(stop_tokens),
                         std::move(stop_sequences));
}

std::shared_ptr<Request> VLMRequestFactory::create_request(
    std::vector<Message> messages,
    const RequestParams& sp,
    BinaryPayload payload,
    std::optional<Call*> call,
    OutputCallback callback,
    bool use_prompt_as_is) {
  // Guard the rate-limit slot acquired at the service entry. The next hop
  // (create_request(prompt, ...)) installs its own guard, so we dismiss ours
  // before forwarding.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  if (sp.multimodal_prompt) {
    if (!payload.empty() ||
        std::any_of(
            messages.begin(), messages.end(), [](const Message& message) {
              return message.has_mm_content();
            })) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Typed image input cannot include legacy media",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
    rate_limit_guard.dismiss();
    return create_request("", MMData{}, sp, call, std::move(callback));
  }

  Status arguments_status = validate_tool_call_arguments(messages);
  if (!arguments_status.ok()) {
    callback({std::move(arguments_status),
              sp.service_request_id,
              sp.source_xservice_addr});
    return nullptr;
  }

  const bool has_mm_content =
      std::any_of(messages.begin(), messages.end(), [](const Message& message) {
        return message.has_mm_content();
      });
  if (sp.prompt_tokens) {
    if (has_mm_content || !payload.empty()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Pretokenized VLM text cannot include legacy media",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
    rate_limit_guard.dismiss();
    return create_request("", MMData{}, sp, call, std::move(callback));
  }
  MMData mm_data;
  Status media_status = process_media(messages, std::move(payload), mm_data);
  if (!media_status.ok()) {
    callback({std::move(media_status),
              sp.service_request_id,
              sp.source_xservice_addr});
    return nullptr;
  }

  std::optional<std::string> prompt;
  std::optional<ChatTemplateGenerationMode> generation_mode;
  if (use_prompt_as_is) {
    // The request hands in a final prompt that is already assembled with
    // the chat template; reuse it as-is to avoid double assembly.
    VLOG(1) << "Offline request: use the prompt as-is, skip chat template "
               "application.";
    prompt = join_message_texts(messages);
    if (!prompt.has_value()) {
      // Loud failure for direct callers; the request entry validates the
      // shape earlier, so this only fires on a missed contract violation.
      throw_prompt_as_is_error(messages);
    }
  } else {
    AUTO_COUNTER(chat_template_latency_seconds);
    // Server-side template context for audio models (audio_token_id set):
    // the CTC region follows --use_ctc, so the rendered prompt and the
    // embedding injection cannot disagree. A request kwarg of the same name
    // is overridden on purpose.
    nlohmann::json chat_template_kwargs = sp.chat_template_kwargs;
    if (model_args_->audio_token_id() != 0) {
      chat_template_kwargs["use_ctc"] =
          MultimodalConfig::get_instance().use_ctc();
    }
    // Render with the generation mode so structured-output constraints can
    // pick the reasoning or plain grammar variant, mirroring the LLM path.
    const std::optional<ChatTemplateRenderResult> render_result =
        chat_template_->apply_with_generation_mode(
            messages, sp.tools, chat_template_kwargs);
    if (render_result.has_value()) {
      prompt = render_result->prompt;
      generation_mode = render_result->generation_mode;
    }
  }
  if (!prompt.has_value()) {
    std::string error_message = "Failed to construct prompt from messages";
    LOG(ERROR) << error_message;
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        error_message,
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  rate_limit_guard.dismiss();
  return create_request(std::move(prompt.value()),
                        std::move(mm_data),
                        sp,
                        call,
                        std::move(callback),
                        generation_mode);
}

}  // namespace xllm
