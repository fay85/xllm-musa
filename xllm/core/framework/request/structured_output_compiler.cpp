/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/request/structured_output_compiler.h"

#include <glog/logging.h>

#include <unordered_set>
#include <utility>

#include "common/macros.h"
#include "common/options.h"
#include "core/common/types.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/constraint/builtin_json_machine.h"
#include "core/framework/constraint/xgrammar_machine.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/request_params.h"
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

const char* response_format_name(ResponseFormatType format) {
  switch (format) {
    case ResponseFormatType::JSON_OBJECT:
      return "json_object";
    case ResponseFormatType::JSON_SCHEMA:
      return "json_schema";
    case ResponseFormatType::REGEX:
      return "regex";
    case ResponseFormatType::EBNF:
      return "ebnf";
    case ResponseFormatType::NONE:
      break;
  }
  return "none";
}

}  // namespace

StructuredOutputCompiler::StructuredOutputCompiler(const Tokenizer& tokenizer,
                                                   const ModelArgs& model_args,
                                                   const Options& options)
    : tokenizer_(tokenizer),
      model_args_(model_args),
      options_(options),
      enable_xgrammar_output_(ServiceConfig::get_instance().use_xgrammar()) {
  if (enable_xgrammar_output_) {
    std::string error;
    xgrammar_compiler_ = XGrammarCompiler::create(
        tokenizer_,
        model_args_,
        ServiceConfig::get_instance().xgrammar_cache_max_memory_bytes(),
        error);
    if (xgrammar_compiler_ == nullptr) {
      LOG(ERROR) << "XGrammar initialization failed (" << error
                 << "); structured output requests will be rejected";
    }
  }
}

std::shared_ptr<const JsonObjectGrammar>
StructuredOutputCompiler::get_json_object_grammar(bool reasoning_enabled,
                                                  std::string& error) {
  std::lock_guard<std::mutex> lock(json_object_grammar_mutex_);
  std::shared_ptr<const JsonObjectGrammar>& grammar =
      reasoning_enabled ? json_reasoning_grammar_ : json_object_grammar_;
  if (grammar == nullptr) {
    grammar =
        JsonObjectGrammar::create_from_tokenizer(tokenizer_,
                                                 model_args_.eos_token_id(),
                                                 model_args_.stop_token_ids(),
                                                 model_args_.vocab_size(),
                                                 reasoning_enabled,
                                                 &error);
  }
  return grammar;
}

bool StructuredOutputCompiler::validate_speculative_support(
    const RequestParams& sp,
    std::string& error) const {
  if (sp.response_format == ResponseFormatType::NONE ||
      options_.num_speculative_tokens() == 0 ||
      SpeculativeConfig::drives_constraint_channel(
          options_.speculative_algorithm())) {
    return true;
  }
  error = "structured output (response_format=" +
          std::string(response_format_name(sp.response_format)) +
          ") is not supported with speculative algorithm " +
          options_.speculative_algorithm() +
          ": the constraint mask is applied per speculative position and this "
          "drafter cannot drive it; use MTP, DFlash, DFlash2, DSpark or "
          "Suffix, or disable "
          "speculative decoding";
  return false;
}

bool StructuredOutputCompiler::uses_builtin_json_object(
    const RequestParams& sp) const {
  return ServiceConfig::get_instance().enable_json_object_output() &&
         sp.response_format == ResponseFormatType::JSON_OBJECT &&
         !enable_xgrammar_output_;
}

bool StructuredOutputCompiler::apply_json_object_grammar(
    RequestState& req_state,
    const RequestParams& sp,
    bool reasoning_enabled,
    const OutputCallback& callback) {
  std::string grammar_error;
  req_state.json_object_grammar =
      get_json_object_grammar(reasoning_enabled, grammar_error);
  if (req_state.json_object_grammar == nullptr) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Failed to initialize json_object constraint: " + grammar_error,
        sp.service_request_id,
        sp.source_xservice_addr);
    return false;
  }
  req_state.constraint_machine =
      wrap_builtin_machine(req_state.json_object_grammar);
  req_state.json_reasoning_enabled = reasoning_enabled;
  return true;
}

bool StructuredOutputCompiler::build_response_format_constraint(
    const RequestParams& sp,
    const OutputCallback& callback,
    std::shared_ptr<const CompiledXGrammar>& constraint,
    GrammarSpec& spec) {
  std::string error;
  if (!validate_speculative_support(sp, error)) {
    callback({Status::invalid_parameter(std::move(error), "response_format"),
              sp.service_request_id,
              sp.source_xservice_addr});
    return false;
  }
  constraint = compile_response_format(sp, spec, error);
  if (!error.empty()) {
    callback({Status::invalid_parameter(std::move(error), "response_format"),
              sp.service_request_id,
              sp.source_xservice_addr});
    return false;
  }
  return true;
}

void StructuredOutputCompiler::attach_xgrammar_constraint(
    RequestState& req_state,
    std::shared_ptr<const CompiledXGrammar> constraint,
    GrammarSpec spec) {
  req_state.xgrammar_constraint = std::move(constraint);
  if (req_state.constraint_machine != nullptr) {
    return;
  }
  // The spec rides the machine so its snapshots identify the grammar
  // cross-process: a decode worker rebuilds the exact source and replays the
  // committed history (G9). It is the one compile_response_format built the
  // grammar from, so the rebuilt grammar is equivalent by construction. The
  // member keeps a copy for the PD scheduler's proto fill; passing that lvalue
  // to the by-value sink is exactly one more copy into the machine's retained
  // identity.
  req_state.xgrammar_spec = std::move(spec);
  req_state.constraint_machine = wrap_xgrammar_machine(
      req_state.xgrammar_constraint, req_state.xgrammar_spec);
  // A compiled xgrammar constraint without its machine means the build lacks
  // USE_XGRAMMAR support; Sequence would dereference the null machine on its
  // first grammar_id() read. Unreachable through compile_response_format today
  // (it rejects non-JSON formats when the backend is off), but the invariant is
  // cheaper to enforce here than to debug at the dereference site.
  CHECK(req_state.xgrammar_constraint == nullptr ||
        req_state.constraint_machine != nullptr)
      << "an xgrammar constraint requires a build with USE_XGRAMMAR";
}

std::shared_ptr<const CompiledXGrammar>
StructuredOutputCompiler::compile_response_format(const RequestParams& sp,
                                                  GrammarSpec& spec,
                                                  std::string& error) {
  if (sp.response_format == ResponseFormatType::NONE) {
    return nullptr;
  }
  if (!enable_xgrammar_output_) {
    if (sp.response_format != ResponseFormatType::JSON_OBJECT) {
      error = "response_format=" +
              std::string(response_format_name(sp.response_format)) +
              " requires structured_output_backend=xgrammar";
    }
    return nullptr;
  }
  if (xgrammar_compiler_ == nullptr) {
    error =
        "XGrammar requires a valid ByteLevel tokenizer and model vocabulary; "
        "check initialization logs";
    return nullptr;
  }
  if (sp.beam_width > 1 || sp.is_sample_request || sp.ignore_eos) {
    error =
        "XGrammar constraints require ordinary generation without beam search, "
        "sample requests or ignore_eos";
    return nullptr;
  }
  if (sp.stop_token_ids.has_value()) {
    std::unordered_set<int32_t> model_stops = model_args_.stop_token_ids();
    std::unordered_set<int32_t> request_stops(sp.stop_token_ids->begin(),
                                              sp.stop_token_ids->end());
    const int32_t eos_token_id = model_args_.eos_token_id();
    if (eos_token_id >= 0) {
      model_stops.insert(eos_token_id);
      request_stops.insert(eos_token_id);
    }
    if (request_stops != model_stops) {
      error =
          "XGrammar constraints require the model's default stop token "
          "IDs; overriding them is not supported";
      return nullptr;
    }
  }
  if (sp.response_format == ResponseFormatType::REGEX) {
    spec.type = GrammarSpec::Type::REGEX;
    spec.source = sp.response_format_regex;
    return xgrammar_compiler_->compile_regex(spec.source, error);
  }
  if (sp.response_format == ResponseFormatType::EBNF) {
    spec.type = GrammarSpec::Type::EBNF;
    spec.source = sp.response_format_ebnf;
    // compile_ebnf also probes the compiled grammar (empty-language
    // rejection), so an unhealthy grammar fails as a request error here.
    return xgrammar_compiler_->compile_ebnf(spec.source, error);
  }
  const bool json_object =
      sp.response_format == ResponseFormatType::JSON_OBJECT;
  spec.type = GrammarSpec::Type::JSON_SCHEMA;
  spec.strict = !json_object && sp.response_format_strict;
  spec.source =
      json_object ? R"({"type":"object"})" : sp.response_format_schema;
  return xgrammar_compiler_->compile_json_schema(
      spec.source, spec.strict, error);
}

}  // namespace xllm
