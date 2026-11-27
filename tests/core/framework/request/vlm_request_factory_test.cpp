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

#include "framework/request/vlm_request_factory.h"

#include <absl/strings/escaping.h>
#include <brpc/controller.h>
#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "api_service/chat_json_parser.h"
#include "api_service/completion_json_parser.h"
#include "common/metrics.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/constants.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_codec.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/scheduler/async_response_processor.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/request/sequence.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "processors/deepseek_v41_image_processor.h"
#include "processors/deepseek_v41_prompt_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen2_vl_prompt_processor.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {
namespace {

constexpr int32_t kPretokenizedTextVocabSize = 1000;

constexpr char kTinyPngDataUrl[] =
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQIHWP4z8AAAAMB"
    "AQBb2/lEAAAAAElFTkSuQmCC";

struct InputMetricsSnapshot {
  int64_t transfer_count = HISTOGRAM_mm_transfer_latency_milliseconds.count();
  int64_t processor_count = HISTOGRAM_mm_processor_latency_milliseconds.count();
  double template_seconds = COUNTER_VALUE(chat_template_latency_seconds);
  double transfer_failures = MULTI_COUNTER_mm_input_stage_failures_total
                                 .get_stats(/*labels=*/{"transfer"})
                                 ->get_value();
  double processor_failures = MULTI_COUNTER_mm_input_stage_failures_total
                                  .get_stats(/*labels=*/{"processor"})
                                  ->get_value();
};

void expect_input_stage_failures(const InputMetricsSnapshot& before,
                                 int64_t transfer = 0,
                                 int64_t processor = 0) {
  EXPECT_DOUBLE_EQ(MULTI_COUNTER_mm_input_stage_failures_total
                       .get_stats(/*labels=*/{"transfer"})
                       ->get_value(),
                   before.transfer_failures + transfer);
  EXPECT_DOUBLE_EQ(MULTI_COUNTER_mm_input_stage_failures_total
                       .get_stats(/*labels=*/{"processor"})
                       ->get_value(),
                   before.processor_failures + processor);
}

// Deterministic tokenizer used for factory tests. It counts vocabulary lookups
// and fails to encode text containing "FAIL" to exercise stop-sequence errors.
class FakeTokenizer final : public Tokenizer {
 public:
  explicit FakeTokenizer(int32_t vocab_size) : vocab_size_(vocab_size) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
    if (text.find("FAIL") != std::string_view::npos) {
      return false;
    }
    ids->clear();
    for (const char c : text) {
      ids->push_back(static_cast<int32_t>(static_cast<unsigned char>(c)) %
                     vocab_size_);
    }
    if (ids->empty()) {
      ids->push_back(1);
    }
    return true;
  }

  size_t vocab_size() const override {
    ++vocab_size_calls_;
    return static_cast<size_t>(vocab_size_);
  }

  size_t vocab_size_calls() const { return vocab_size_calls_; }

  // Non-empty, deterministic pieces so the builtin JSON-object grammar can be
  // built for structured-output tests. Only spelling is synthetic.
  std::string decode_token(int32_t id) const override {
    if (id < 0) {
      return {};
    }
    return std::string(1, static_cast<char>('a' + (id % 26)));
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>(*this);
  }

  bool get_constraint_vocab(std::vector<std::string>* pieces) const override {
    pieces->assign(static_cast<size_t>(vocab_size_), std::string());
    for (int32_t id = 0; id < vocab_size_; ++id) {
      (*pieces)[id] = std::string(1, static_cast<char>(id));
    }
    return true;
  }

 private:
  int32_t vocab_size_ = 1000;
  mutable size_t vocab_size_calls_ = 0;
};

// Fake multimodal processor that lets tests drive process_prompt success and
// the tokens it emits, without pulling in any real vision/audio pipeline.
class FakeProcessor final : public MultimodalProcessorBase {
 public:
  FakeProcessor() : MultimodalProcessorBase(/*tokenizer=*/nullptr) {}

  bool process_prompt(std::string& /*prompt*/,
                      MMData& /*mm_data*/,
                      std::vector<int32_t>& token_ids) override {
    if (!process_prompt_succeed_) {
      return false;
    }
    token_ids = prompt_tokens_;
    return true;
  }

  bool process_multimodal(const MMInput& /*inputs*/,
                          MMData& /*data*/) const override {
    return process_multimodal_succeed_;
  }

  void set_process_prompt_succeed(bool succeed) {
    process_prompt_succeed_ = succeed;
  }

  void set_process_multimodal_succeed(bool succeed) {
    process_multimodal_succeed_ = succeed;
  }

  void set_prompt_tokens(std::vector<int32_t> tokens) {
    prompt_tokens_ = std::move(tokens);
  }

 private:
  bool process_prompt_succeed_ = true;
  bool process_multimodal_succeed_ = true;
  std::vector<int32_t> prompt_tokens_{1, 2, 3};
};

// Jinja chat template stub. The base constructor parses a trivial literal
// template (no Jinja tags, so it never fails to parse), while apply is
// overridden to return a fixed prompt or report failure on demand.
class FakeJinjaChatTemplate final : public JinjaChatTemplate {
 public:
  FakeJinjaChatTemplate() : JinjaChatTemplate(make_template_args()) {}

  std::optional<std::string> apply(
      const ChatMessages& /*messages*/,
      const std::vector<xllm::JsonTool>& /*json_tools*/,
      const nlohmann::ordered_json& /*chat_template_kwargs*/) const override {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!succeed_) {
      return std::nullopt;
    }
    return rendered_prompt_;
  }

  void set_succeed(bool succeed) { succeed_ = succeed; }

 private:
  static TokenizerArgs make_template_args() {
    TokenizerArgs args;
    args.chat_template("hello");
    return args;
  }

  bool succeed_ = true;
  std::string rendered_prompt_ = "rendered prompt";
};

class FakeCall final : public Call {
 public:
  explicit FakeCall(brpc::Controller* controller) : Call(controller) {}

  bool is_disconnected() const override { return disconnected_; }

  void disconnect() { disconnected_ = true; }

 private:
  bool disconnected_ = false;
};

// Records the last error surfaced through the OutputCallback.
struct CallbackCapture {
  bool called = false;
  std::optional<Status> status;
};

OutputCallback make_capture_callback(CallbackCapture* capture) {
  return [capture](RequestOutput output) {
    capture->called = true;
    capture->status = output.status;
    return false;
  };
}

class VLMRequestFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_backend_ =
        ServiceConfig::get_instance().structured_output_backend();
    // Simulate the caller (service entry) having acquired a rate-limit slot.
    rate_limiter_.is_limited();
    ASSERT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  }

  void TearDown() override {
    ServiceConfig::get_instance().structured_output_backend(previous_backend_);
  }

  std::unique_ptr<VLMRequestFactory> make_factory(int32_t vocab_size = 1000,
                                                  int32_t max_position = 2048) {
    processor_ = std::make_unique<FakeProcessor>();
    chat_template_ = std::make_unique<FakeJinjaChatTemplate>();
    tokenizer_ = std::make_unique<FakeTokenizer>(vocab_size);
    model_args_.vocab_size(vocab_size)
        .max_position_embeddings(max_position)
        .eos_token_id(-1);
    // enable_chunked_prefill defaults true; keep it so the prompt length limit
    // is exactly max_position_embeddings.
    options_.enable_service_routing(false).num_speculative_tokens(0);
    return std::make_unique<VLMRequestFactory>(
        processor_.get(),
        chat_template_.get(),
        tokenizer_.get(),
        &model_args_,
        &options_,
        &rate_limiter_,
        /*rpc_response_handler=*/nullptr);
  }

  std::unique_ptr<FakeProcessor> processor_;
  std::unique_ptr<FakeJinjaChatTemplate> chat_template_;
  std::unique_ptr<FakeTokenizer> tokenizer_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
  std::string previous_backend_;
};

// Pretokenized Service text must keep token identity and Decode routing even
// when local prompt processing would fail. Media still uses typed validation.
TEST_F(VLMRequestFactoryTest, PreservesServiceTextTokensWithoutProcessing) {
  auto factory =
      make_factory(kPretokenizedTextVocabSize, /*max_position=*/32768);
  processor_->set_process_prompt_succeed(false);
  proto::ChatRequest wire;
  // Long Service text must not rebuild the vocabulary for every input token.
  constexpr int32_t kPromptTokenCount = 10032;
  wire.mutable_token_ids()->Resize(kPromptTokenCount, 7);
  wire.set_token_ids(kPromptTokenCount - 1, kPretokenizedTextVocabSize - 1);
  wire.mutable_routing()->set_decode_name("decode");
  RequestParams params(wire, "", "");
  params.prompt_tokens.emplace(wire.token_ids().begin(),
                               wire.token_ids().end());
  params.max_tokens = 4;
  const size_t vocab_lookups_before = tokenizer_->vocab_size_calls();
  auto request = factory->create_request(
      "", MMData{}, params, std::nullopt, [](const RequestOutput&) {
        return true;
      });
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->state().prompt_tokens, *params.prompt_tokens);
  EXPECT_EQ(request->state().decode_address, "decode");
  EXPECT_LE(tokenizer_->vocab_size_calls() - vocab_lookups_before, 1u);
}

TEST_F(VLMRequestFactoryTest, RejectsNegativeFirstTokenWithoutVocabLookup) {
  auto factory = make_factory(kPretokenizedTextVocabSize);
  RequestParams params;
  params.prompt_tokens = std::vector<int32_t>{-1, 7};
  CallbackCapture capture;
  const size_t vocab_lookups_before = tokenizer_->vocab_size_calls();
  auto request = factory->create_request(
      "", MMData{}, params, std::nullopt, make_capture_callback(&capture));
  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(tokenizer_->vocab_size_calls(), vocab_lookups_before);
}

TEST_F(VLMRequestFactoryTest, RejectsTokenEqualToVocabSizeAfterValidPrefix) {
  auto factory = make_factory(kPretokenizedTextVocabSize);
  RequestParams params;
  params.prompt_tokens = std::vector<int32_t>{kPretokenizedTextVocabSize - 1,
                                              kPretokenizedTextVocabSize};
  CallbackCapture capture;
  auto request = factory->create_request(
      "", MMData{}, params, std::nullopt, make_capture_callback(&capture));
  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(VLMRequestFactoryTest,
       RejectsEmptyPromptAndMmDataReleasesRateLimitSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create_request(/*prompt=*/"",
                                         /*mm_data=*/MMData{},
                                         sp,
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.called);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "Prompt and multimodal data are both empty.");
  // The factory must release the rate-limit slot on every early return.
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, FailsWhenProcessPromptFails) {
  auto factory = make_factory();
  processor_->set_process_prompt_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  const InputMetricsSnapshot before;

  auto request = factory->create_request(/*prompt=*/"describe this",
                                         /*mm_data=*/MMData{},
                                         sp,
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(), "Failed to process prompt.");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count);
  expect_input_stage_failures(before);
}

TEST_F(VLMRequestFactoryTest, RejectsPromptLongerThanContext) {
  auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/8);
  processor_->set_prompt_tokens(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8});
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create_request(/*prompt=*/"describe this",
                                         /*mm_data=*/MMData{},
                                         sp,
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(), "Prompt is too long");
  EXPECT_EQ(capture.status->reason(), FailureReason::PROMPT_TOO_LONG);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, FailsWhenStopSequenceEncodingFails) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.stop = std::vector<std::string>{"FAIL-STOP"};

  auto request = factory->create_request(/*prompt=*/"describe this",
                                         /*mm_data=*/MMData{},
                                         sp,
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Failed to encode stop sequence");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest,
       CreatesRequestForValidPromptAndKeepsRateLimitSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-1";
  sp.max_tokens = 16;

  auto request = factory->create_request(/*prompt=*/"describe this",
                                         /*mm_data=*/MMData{},
                                         sp,
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  // On success the slot stays held; it is later released when the request is
  // completed/destroyed by the scheduler, not by the factory.
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(VLMRequestFactoryTest, AttachesBuiltinJsonObjectConstraint) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 16;
  sp.response_format = ResponseFormatType::JSON_OBJECT;

  auto request = factory->create_request(/*prompt=*/"describe this",
                                         /*mm_data=*/MMData{},
                                         sp,
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_TRUE(request->state().sampling_param.json_object);
  EXPECT_NE(request->state().json_object_grammar, nullptr);
  EXPECT_NE(request->state().constraint_machine, nullptr);
  EXPECT_EQ(request->state().xgrammar_constraint, nullptr);
}

TEST_F(VLMRequestFactoryTest, RejectsJsonSchemaOnBuiltinBackend) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 16;
  sp.response_format = ResponseFormatType::JSON_SCHEMA;
  sp.response_format_schema = R"({"type":"object"})";

  auto request = factory->create_request(/*prompt=*/"describe this",
                                         /*mm_data=*/MMData{},
                                         sp,
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->param(), "response_format");
  EXPECT_NE(capture.status->message().find("structured_output_backend"),
            std::string::npos);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

#if defined(USE_XGRAMMAR)
// XGrammar cannot apply a grammar mask to beam search, sample requests,
// ignore_eos or a custom stop-token set, so each must be rejected up front.
TEST_F(VLMRequestFactoryTest, XGrammarRejectsUnsupportedRequestModes) {
  ServiceConfig::get_instance().structured_output_backend("xgrammar");
  auto factory = make_factory(/*vocab_size=*/256);
  for (int32_t mode = 0; mode < 5; ++mode) {
    RequestParams sp;
    sp.max_tokens = 16;
    sp.response_format = ResponseFormatType::REGEX;
    sp.response_format_regex = mode == 0 ? "[" : "yes";
    sp.beam_width = mode == 1 ? 2 : 0;
    sp.is_sample_request = mode == 2;
    sp.ignore_eos = mode == 3;
    if (mode == 4) {
      sp.stop_token_ids = std::vector<int32_t>{257};
    }
    CallbackCapture capture;
    EXPECT_EQ(factory->create_request(/*prompt=*/"FAIL",
                                      /*mm_data=*/MMData{},
                                      sp,
                                      /*call=*/std::nullopt,
                                      make_capture_callback(&capture)),
              nullptr);
    ASSERT_TRUE(capture.status.has_value());
    EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(capture.status->param(), "response_format");
    // Pin the exact guard so a disabled xgrammar backend cannot make this
    // test pass for the wrong reason.
    const char* expected = mode == 0
                               ? "Failed to initialize XGrammar constraint"
                           : mode == 1 ? "beam search"
                           : mode == 2 ? "sample requests"
                           : mode == 3 ? "ignore_eos"
                                       : "default stop token";
    EXPECT_NE(capture.status->message().find(expected), std::string::npos)
        << capture.status->message();
    EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
    rate_limiter_.is_limited();
  }
  // Without a response_format the same modes stay supported.
  RequestParams plain;
  plain.max_tokens = 16;
  plain.ignore_eos = true;
  CallbackCapture capture;
  EXPECT_NE(factory->create_request(/*prompt=*/"hello",
                                    /*mm_data=*/MMData{},
                                    plain,
                                    /*call=*/std::nullopt,
                                    make_capture_callback(&capture)),
            nullptr);
  EXPECT_FALSE(capture.called);
}
#endif

// Eagle3 drafts with a reduced vocabulary; the target-vocabulary constraint
// mask would abort the sampler on the draft logits, so the request must be
// rejected before execution. The fixture keeps the default builtin backend,
// pinning that the admission rule covers builtin json_object and not only
// xgrammar formats.
TEST_F(VLMRequestFactoryTest, RejectsConstraintWithUnsupportedSpeculative) {
  auto factory = make_factory();
  options_.num_speculative_tokens(2);
  options_.speculative_algorithm("Eagle3");
  RequestParams sp;
  sp.max_tokens = 16;
  sp.response_format = ResponseFormatType::JSON_OBJECT;

  CallbackCapture capture;
  EXPECT_EQ(factory->create_request(/*prompt=*/"x",
                                    /*mm_data=*/MMData{},
                                    sp,
                                    /*call=*/std::nullopt,
                                    make_capture_callback(&capture)),
            nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->param(), "response_format");
  EXPECT_NE(capture.status->message().find("speculative algorithm"),
            std::string::npos);
  EXPECT_NE(capture.status->message().find("response_format=json_object"),
            std::string::npos);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);

  // MTP drives the constraint channel and stays supported.
  rate_limiter_.is_limited();
  options_.speculative_algorithm("MTP");
  CallbackCapture mtp_capture;
  EXPECT_NE(factory->create_request(/*prompt=*/"x",
                                    /*mm_data=*/MMData{},
                                    sp,
                                    /*call=*/std::nullopt,
                                    make_capture_callback(&mtp_capture)),
            nullptr);
  EXPECT_FALSE(mtp_capture.called);
}

TEST_F(VLMRequestFactoryTest, MessageOverloadFailsWhenTemplateRejects) {
  auto factory = make_factory();
  chat_template_->set_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  sp.service_request_id = "template-failure";
  sp.source_xservice_addr = "owner-service:19000";

  // MMInputTransfer requires MMContentVec content; a text part is enough.
  MMContentVec content;
  content.emplace_back("text", std::string("hi"));
  std::vector<Message> messages;
  messages.emplace_back("user", content);

  const InputMetricsSnapshot before;
  auto request = factory->create_request(
      messages,
      sp,
      /*payload=*/BinaryPayload(),
      /*call=*/std::nullopt,
      [&](const RequestOutput& output) {
        capture.called = true;
        capture.status = output.status;
        EXPECT_EQ(output.service_request_id, sp.service_request_id);
        EXPECT_EQ(output.target_xservice_addr, sp.source_xservice_addr);

        return false;
      });

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "Failed to construct prompt from messages");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count);
  EXPECT_GT(COUNTER_VALUE(chat_template_latency_seconds),
            before.template_seconds);
  expect_input_stage_failures(before);
}

TEST_F(VLMRequestFactoryTest, MessageOverloadCreatesRequestOnSuccess) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.timing = std::make_shared<RequestTiming>();
  sp.request_id = "req-chat";
  sp.max_tokens = 16;
  // MMInputTransfer requires MMContentVec content; a text part is enough.
  MMContentVec content;
  content.emplace_back("text", std::string("hi"));
  std::vector<Message> messages;
  messages.emplace_back("user", content);

  const InputMetricsSnapshot before;
  auto request = factory->create_request(messages,
                                         sp,
                                         /*payload=*/BinaryPayload(),
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->state().timing, sp.timing);
  EXPECT_EQ(request->sequences().front()->request_timing(), sp.timing);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count);
  EXPECT_GT(COUNTER_VALUE(chat_template_latency_seconds),
            before.template_seconds);
  expect_input_stage_failures(before);
}

TEST_F(VLMRequestFactoryTest, MultiImagePromptAsIsRecordsOneTransferSample) {
  auto factory = make_factory();
  chat_template_->set_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 16;
  sp.n = 2;
  MMContentVec content;
  content.reserve(3);
  content.emplace_back("text", std::string("describe these"));
  content.emplace_back("image_url", ImageURL{.url = kTinyPngDataUrl});
  content.emplace_back("image_url", ImageURL{.url = kTinyPngDataUrl});
  std::vector<Message> messages;
  messages.reserve(1);
  messages.emplace_back("user", content);

  const InputMetricsSnapshot before;
  auto request = factory->create_request(std::move(messages),
                                         sp,
                                         /*payload=*/BinaryPayload(),
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture),
                                         /*use_prompt_as_is=*/true);

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(request->state().prompt, "describe these");
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count + 1);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count + 1);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(chat_template_latency_seconds),
                   before.template_seconds);
  expect_input_stage_failures(before);
}

TEST_F(VLMRequestFactoryTest, FailedMediaTransferStillRecordsOneSample) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  MMContentVec content;
  content.reserve(2);
  content.emplace_back(
      "image_url", ImageURL{.url = "data:image/png;base64,bm90LWFuLWltYWdl"});
  content.emplace_back(
      "image_url", ImageURL{.url = "data:image/png;base64,bm90LWFuLWltYWdl"});
  std::vector<Message> messages;
  messages.reserve(1);
  messages.emplace_back("user", content);

  const InputMetricsSnapshot before;
  auto request = factory->create_request(std::move(messages),
                                         sp,
                                         /*payload=*/BinaryPayload(),
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(), "Failed to decode multimodal input.");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count + 1);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(chat_template_latency_seconds),
                   before.template_seconds);
  expect_input_stage_failures(before, /*transfer=*/1, /*processor=*/0);
}

TEST_F(VLMRequestFactoryTest, ProcessorFailureRetainsInputStageSamples) {
  auto factory = make_factory();
  processor_->set_process_multimodal_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  MMContentVec content;
  content.reserve(1);
  content.emplace_back("image_url", ImageURL{.url = kTinyPngDataUrl});
  std::vector<Message> messages;
  messages.reserve(1);
  messages.emplace_back("user", content);

  const InputMetricsSnapshot before;
  auto request = factory->create_request(std::move(messages),
                                         sp,
                                         /*payload=*/BinaryPayload(),
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Failed to process multimodal input.");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count + 1);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count + 1);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(chat_template_latency_seconds),
                   before.template_seconds);
  expect_input_stage_failures(before, /*transfer=*/0, /*processor=*/1);
}

enum class RequestInput { PROMPT, MESSAGES };

class VLMRequestFactoryConnectionTest
    : public VLMRequestFactoryTest,
      public ::testing::WithParamInterface<RequestInput> {
 protected:
  std::shared_ptr<Request> create_request(VLMRequestFactory& factory,
                                          const RequestParams& sp,
                                          std::optional<Call*> call,
                                          OutputCallback callback) {
    if (GetParam() == RequestInput::PROMPT) {
      return factory.create_request(/*prompt=*/"describe this",
                                    /*mm_data=*/MMData{},
                                    sp,
                                    call,
                                    std::move(callback));
    }

    MMContentVec content;
    content.reserve(1);
    content.emplace_back("text", std::string("hi"));
    std::vector<Message> messages;
    messages.reserve(1);
    messages.emplace_back("user", std::move(content));
    return factory.create_request(std::move(messages),
                                  sp,
                                  /*payload=*/BinaryPayload(),
                                  call,
                                  std::move(callback));
  }
};

TEST_P(VLMRequestFactoryConnectionTest, OnlineCallLifetimeAndCancellation) {
  auto factory = make_factory();
  brpc::Controller controller;
  auto call = std::make_shared<FakeCall>(&controller);
  std::weak_ptr<FakeCall> weak_call = call;
  RequestParams sp;
  sp.max_tokens = 16;
  sp.n = 2;

  // Match the API callback's ownership of the non-owning RequestState call.
  auto request = create_request(
      *factory, sp, call.get(), [call](const RequestOutput& /*output*/) {
        return !call->is_disconnected();
      });

  ASSERT_NE(request, nullptr);
  ASSERT_TRUE(request->state().call_.has_value());
  EXPECT_EQ(request->state().call_.value(), call.get());
  ASSERT_TRUE(request->expand_sequences(/*share_prefix=*/false));
  ASSERT_EQ(request->sequences().size(), 2);
  request->update_connection_status();
  EXPECT_FALSE(request->cancelled());
  for (const auto& sequence : request->sequences()) {
    EXPECT_FALSE(sequence->cancelled());
  }

  call->disconnect();
  call.reset();
  ASSERT_FALSE(weak_call.expired());
  request->update_connection_status();
  request->update_connection_status();
  EXPECT_TRUE(request->cancelled());
  for (const auto& sequence : request->sequences()) {
    EXPECT_TRUE(sequence->cancelled());
  }

  request.reset();
  EXPECT_TRUE(weak_call.expired());
}

TEST_P(VLMRequestFactoryConnectionTest, OfflineCallDoesNotCancel) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 16;
  sp.n = 2;

  auto request = create_request(
      *factory, sp, /*call=*/std::nullopt, make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(request->state().call_.has_value());
  ASSERT_TRUE(request->expand_sequences(/*share_prefix=*/false));
  ASSERT_EQ(request->sequences().size(), 2);
  request->update_connection_status();
  EXPECT_FALSE(request->cancelled());
  for (const auto& sequence : request->sequences()) {
    EXPECT_FALSE(sequence->cancelled());
  }
  EXPECT_FALSE(capture.called);
}

INSTANTIATE_TEST_SUITE_P(PromptAndMessages,
                         VLMRequestFactoryConnectionTest,
                         ::testing::Values(RequestInput::PROMPT,
                                           RequestInput::MESSAGES));

TEST_F(VLMRequestFactoryTest, OfflinePromptAsIsSkipsTemplateAndHasNoCall) {
  auto factory = make_factory();
  chat_template_->set_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 16;
  MMContentVec content;
  content.reserve(2);
  content.emplace_back("text", std::string("final "));
  content.emplace_back("text", std::string("prompt"));
  std::vector<Message> messages;
  messages.reserve(1);
  messages.emplace_back("user", std::move(content));

  const InputMetricsSnapshot before;
  auto request = factory->create_request(std::move(messages),
                                         sp,
                                         /*payload=*/BinaryPayload(),
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture),
                                         /*use_prompt_as_is=*/true);

  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->state().prompt, "final prompt");
  EXPECT_FALSE(request->state().call_.has_value());
  request->update_connection_status();
  EXPECT_FALSE(request->cancelled());
  ASSERT_EQ(request->sequences().size(), 1);
  EXPECT_FALSE(request->sequences().front()->cancelled());
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(chat_template_latency_seconds),
                   before.template_seconds);
  expect_input_stage_failures(before);
}

TEST_F(VLMRequestFactoryTest, InvalidHistoryReleasesSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  Message assistant("assistant", "");
  Message::ToolCall call;
  call.function.arguments = "null";
  assistant.tool_calls = Message::ToolCallVec{call};
  const InputMetricsSnapshot before;
  auto request = factory->create_request(ChatMessages{assistant},
                                         sp,
                                         /*payload=*/BinaryPayload(),
                                         /*call=*/std::nullopt,
                                         make_capture_callback(&capture),
                                         /*use_prompt_as_is=*/false);
  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->reason(), FailureReason::INVALID_PARAMETER);
  EXPECT_EQ(capture.status->param(),
            "messages[0].tool_calls[0].function.arguments");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
  EXPECT_EQ(HISTOGRAM_mm_transfer_latency_milliseconds.count(),
            before.transfer_count);
  EXPECT_EQ(HISTOGRAM_mm_processor_latency_milliseconds.count(),
            before.processor_count);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(chat_template_latency_seconds),
                   before.template_seconds);
  expect_input_stage_failures(before);
}

TEST_F(VLMRequestFactoryTest, PreservesDecodeRouteForOrdinaryVlmPd) {
  auto factory = make_factory();
  options_.enable_disagg_pd(true);
  RequestParams params;
  params.max_tokens = 16;
  params.decode_address = "decode.example:1234";
  auto request = factory->create_request(
      "hello", MMData{}, params, {}, [](const RequestOutput&) { return true; });
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->state().decode_address, params.decode_address);
  EXPECT_EQ(request->state().clone().decode_address, params.decode_address);
  request.reset();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

// These tests exercise the native processor and request factory used by
// typed Chat/Completion dispatch, using real CPU Qwen image preprocessing.
class MediaTestProcessor final : public MultimodalProcessorBase {
 public:
  explicit MediaTestProcessor(const ModelArgs& args)
      : MultimodalProcessorBase(nullptr), real_(args, nullptr, {}) {}
  bool process_prompt(std::string&, MMData&, std::vector<int32_t>&) override {
    ++render_calls;
    return false;
  }
  bool process_multimodal(const MMInput& inputs, MMData& data) const override {
    if (before_process) {
      before_process();
    }
    ++image_calls;
    return real_.process_multimodal(inputs, data);
  }
  bool bind_prompt_tokens(const std::vector<int32_t>& tokens,
                          MMData& data) const override {
    return real_.bind_prompt_tokens(tokens, data);
  }
  mutable std::atomic<int32_t> image_calls{0};
  std::atomic<int32_t> render_calls{0};
  std::function<void()> before_process;

 private:
  MultimodalProcessor<Qwen2VLPromptProcessor, Qwen2VLImageProcessor> real_;
};

class TypedImageRequestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    args_.model_type("qwen2_vl")
        .vocab_size(200000)
        .max_position_embeddings(4096)
        .eos_token_id(-1)
        .vision_start_token_id(151652)
        .vision_end_token_id(151653)
        .image_token_id(151655)
        .video_token_id(151656)
        .mm_image_patch_size(14)
        .mm_image_merge_size(2)
        .mm_image_temporal_patch_size(2)
        .mm_image_min_pixels(3136)
        .mm_image_max_pixels(12845056)
        .mm_image_normalize_mean({0.5, 0.5, 0.5})
        .mm_image_normalize_std({0.5, 0.5, 0.5});
    processor_ = std::make_unique<MediaTestProcessor>(args_);
    key_.set_provider_id(proto::PROVIDER_ID_XLLM_NATIVE);
    key_.set_engine_uid("engine");
    key_.set_profile_digest("profile");
    key_.set_incarnation_id("incarnation");
    options_.enable_service_routing(true).enable_disagg_pd(true).instance_role(
        InstanceRole::PREFILL);
    factory_ = std::make_unique<VLMRequestFactory>(
        processor_.get(),
        &chat_template_,
        &tokenizer_,
        &args_,
        &options_,
        &limiter_,
        [outputs = batch_outputs_](const std::vector<RequestOutput>& batch) {
          outputs->insert(outputs->end(), batch.begin(), batch.end());
          return std::vector<bool>(batch.size(), true);
        });
    // Derive authoritative metadata from the full native processor output.
    MMData processed;
    process_media({MMContent("image_url", ImageURL{kTinyPngDataUrl})},
                  processed);
    ASSERT_EQ(processed.size(), 1);
    proto::MMDataItem media_item;
    ASSERT_TRUE(
        media_metadata_to_proto(processed.items<MMItemVec>()[0], &media_item));
    prompt_.set_model_revision("revision");
    prompt_.set_metadata_format("qwen-image-v1");
    auto* source = prompt_.add_sources();
    source->mutable_media()->set_type("image_url");
    source->mutable_media()->mutable_image_url()->set_url(kTinyPngDataUrl);
    *source->mutable_preprocessing_engine() = key_;
    auto* entry = prompt_.mutable_mm_data();
    entry->set_is_item_vec(true);
    entry->set_type(MMType::IMAGE);
    *entry->add_items() = std::move(media_item);
    const auto& grid = entry->items(0)
                           .dict()
                           .at("image_grid_thw")
                           .single_tensor()
                           .contents()
                           .int64_contents();
    const int32_t length = static_cast<int32_t>(grid.Get(1) * grid.Get(2) / 4);
    prompt_.add_token_ids(42);
    prompt_.add_token_ids(args_.vision_start_token_id());
    for (int32_t index = 0; index < length; ++index) {
      prompt_.add_token_ids(args_.image_token_id());
    }
    prompt_.add_token_ids(args_.vision_end_token_id());
    auto* state = entry->mutable_items(0)->mutable_state();
    state->set_token_pos_offset(2);
    state->set_token_pos_length(length);
    state->set_mm_token_num(length);
  }
  // Obtain real metadata for each occurrence, then construct the exact typed
  // prompt consumed by the factory. Only transport/provider RPCs are absent.
  void prepare_images(const std::vector<std::string>& bytes) {
    std::vector<proto::MMInputData> images(bytes.size());
    MMContentVec sources;
    sources.reserve(images.size());
    for (size_t index = 0; index < images.size(); ++index) {
      images[index].set_type("image_url");
      images[index].mutable_image_url()->set_url(
          "data:image/png;base64," + absl::Base64Escape(bytes[index]));
      MMContent content("image_url", "");
      ASSERT_TRUE(decode_media_input(images[index], content).ok());
      sources.emplace_back(std::move(content));
    }
    MMData processed;
    process_media(std::move(sources), processed);
    ASSERT_EQ(processed.size(), images.size());
    prompt_.clear_sources();
    prompt_.clear_token_ids();
    auto* metadata = prompt_.mutable_mm_data();
    metadata->clear_items();
    prompt_.add_token_ids(42);
    for (size_t index = 0; index < images.size(); ++index) {
      auto* source = prompt_.add_sources();
      *source->mutable_media() = images[index];
      *source->mutable_preprocessing_engine() = key_;
      auto* item = metadata->add_items();
      ASSERT_TRUE(
          media_metadata_to_proto(processed.items<MMItemVec>()[index], item));
      const auto& grid = item->dict()
                             .at("image_grid_thw")
                             .single_tensor()
                             .contents()
                             .int64_contents();
      const int32_t length =
          static_cast<int32_t>(grid.Get(1) * grid.Get(2) / 4);
      prompt_.add_token_ids(args_.vision_start_token_id());
      auto* state = item->mutable_state();
      state->set_token_pos_offset(prompt_.token_ids_size());
      state->set_token_pos_length(length);
      state->set_mm_token_num(length);
      for (int32_t token = 0; token < length; ++token) {
        prompt_.add_token_ids(args_.image_token_id());
      }
      prompt_.add_token_ids(args_.vision_end_token_id());
    }
  }
  std::string image_bytes(uint8_t color) const {
    cv::Mat image(28, 56, CV_8UC3, cv::Scalar(color, 60, 90));
    std::vector<unsigned char> bytes;
    EXPECT_TRUE(cv::imencode(".png", image, bytes));
    return std::string(bytes.begin(), bytes.end());
  }
  void process_media(MMContentVec media,
                     MMData& data,
                     BinaryPayload payload = {}) const {
    std::vector<Message> messages;
    messages.reserve(1);
    messages.emplace_back("user", std::move(media));
    MMInput input(std::move(payload));
    MMInputTransfer transfer;
    ASSERT_EQ(transfer.trans(messages, input), MMErrCode::SUCCESS);
    ASSERT_TRUE(processor_->process_multimodal(input, data));
  }
  std::shared_ptr<PreparedMultimodalInput> prepared_input(
      const proto::MultimodalPrompt& prompt) const {
    auto input = std::make_shared<PreparedMultimodalInput>();
    EXPECT_TRUE(decode_prepared_multimodal_input(prompt, *input).ok());
    return input;
  }

  void expect_key_rejection(const proto::MultimodalPrompt& prompt) {
    auto sp = params();
    sp.multimodal_prompt = prepared_input(prompt);
    ASSERT_NE(sp.multimodal_prompt, nullptr);
    ASSERT_TRUE(factory_->validate_multimodal_request(sp).ok());
    ASSERT_FALSE(limiter_.is_limited());
    CallbackCapture capture;
    EXPECT_EQ(factory_->create_request(
                  "", {}, sp, std::nullopt, make_capture_callback(&capture)),
              nullptr);
    ASSERT_TRUE(capture.called);
    ASSERT_TRUE(capture.status.has_value());
    EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(capture.status->message(),
              "Local media key differs from prepared metadata");
    EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
    EXPECT_TRUE(batch_outputs_->empty());
  }

  RequestParams params() const {
    RequestParams sp;
    sp.request_id = "typed-request";
    sp.timing = RequestTiming::start_ingress({});
    sp.multimodal_prompt = prepared_input(prompt_);
    sp.service_request_id = "019a6123-4567-789a-8abc-0123456789ab";
    sp.decode_address = "decode";
    sp.max_tokens = 4;
    return sp;
  }

  std::pair<Status, MMData> process_prepared(
      const proto::MultimodalPrompt& prompt) {
    auto sp = params();
    sp.multimodal_prompt = prepared_input(prompt);
    EXPECT_FALSE(limiter_.is_limited());
    CallbackCapture capture;
    auto request = factory_->create_request(
        "", {}, sp, std::nullopt, make_capture_callback(&capture));
    if (!request) {
      return {capture.status.value(), {}};
    }
    return {Status{}, request->state().mm_data};
  }
  ModelArgs args_;
  Options options_;
  RateLimiter limiter_;
  FakeJinjaChatTemplate chat_template_;
  FakeTokenizer tokenizer_{200000};
  proto::ProviderEngineKey key_;
  proto::MultimodalPrompt prompt_;
  std::unique_ptr<MediaTestProcessor> processor_;
  std::shared_ptr<std::vector<RequestOutput>> batch_outputs_ =
      std::make_shared<std::vector<RequestOutput>>();
  std::unique_ptr<VLMRequestFactory> factory_;
};

// Only token spelling is synthetic; image processing, prompt expansion, layout
// adjustment, content hashing and prepared-input validation use real
// processors.
class MediaTokenTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
    const std::vector<std::pair<std::string_view, int32_t>> spellings = {
        {"<|vision_start|>", 151652},
        {"<|vision_end|>", 151653},
        {"<|image_pad|>", 151655},
        {"<|video_pad|>", 151656},
        {kDeepseekV41ImagePlaceholder, 129264},
        {kDeepseekV41ImagePadPlaceholder, 129265}};
    ids->clear();
    for (size_t index = 0; index < text.size();) {
      bool matched = false;
      for (const auto& [spelling, token] : spellings) {
        if (text.substr(index, spelling.size()) == spelling) {
          ids->emplace_back(token);
          index += spelling.size();
          matched = true;
          break;
        }
      }
      if (!matched) {
        ids->emplace_back(static_cast<unsigned char>(text[index++]));
      }
    }
    return true;
  }
  size_t vocab_size() const override { return 200000; }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<MediaTokenTokenizer>(*this);
  }
};

TEST_F(TypedImageRequestTest,
       GenericPromptUsesLoadedProcessorAndRejectsTampering) {
  for (const bool deepseek : {false, true}) {
    ModelArgs args = args_;
    args.model_type(deepseek ? "deepseek_v41" : "qwen2_vl");
    args.image_token_id(deepseek ? 129264 : 151655);
    std::unique_ptr<MultimodalProcessorBase> processor;
    auto tokenizer = std::make_shared<MediaTokenTokenizer>();
    if (deepseek) {
      processor =
          std::make_unique<MultimodalProcessor<DeepseekV41PromptProcessor,
                                               DeepseekV41ImageProcessor>>(
              args, tokenizer, TokenizerArgs{});
    } else {
      processor = std::make_unique<
          MultimodalProcessor<Qwen2VLPromptProcessor, Qwen2VLImageProcessor>>(
          args, tokenizer, TokenizerArgs{});
    }
    const std::string text =
        deepseek ? "a<image>b"
                 : "a<|vision_start|><|image_pad|><|vision_end|>b";
    proto::MMInputData source;
    source.set_type("image_url");
    source.mutable_image_url()->set_url(kTinyPngDataUrl);
    MMData authoritative;
    std::vector<int32_t> tokens;
    std::vector<Message> messages;
    messages.reserve(1);
    messages.emplace_back(
        "user",
        MMContentVec{MMContent("image_url", ImageURL{kTinyPngDataUrl})});
    MMInput inputs;
    MMInputTransfer transfer;
    ASSERT_EQ(transfer.trans(messages, inputs), MMErrCode::SUCCESS);
    ASSERT_TRUE(processor->process_multimodal(inputs, authoritative));
    std::string rendered = text;
    ASSERT_TRUE(processor->process_prompt(rendered, authoritative, tokens));
    proto::MultimodalPrompt prepared;
    prepared.set_model_revision("revision");
    prepared.set_prompt_text(text);
    prepared.mutable_token_ids()->Add(tokens.begin(), tokens.end());
    ASSERT_TRUE(mmdata_entry_to_proto(
        authoritative, prepared.mutable_mm_data(), false));
    auto* media = prepared.add_sources();
    *media->mutable_media() = source;
    *media->mutable_preprocessing_engine() = key_;
    MMData actual;
    VLMRequestFactory factory(processor.get(),
                              &chat_template_,
                              tokenizer.get(),
                              &args,
                              &options_,
                              &limiter_,
                              [](const std::vector<RequestOutput>& outputs) {
                                return std::vector<bool>(outputs.size(), true);
                              });
    const auto execute = [&](const proto::MultimodalPrompt& input) {
      auto sp = params();
      sp.multimodal_prompt = std::make_shared<PreparedMultimodalInput>();
      sp.typed_input_status =
          decode_prepared_multimodal_input(input, *sp.multimodal_prompt);
      EXPECT_FALSE(limiter_.is_limited());
      CallbackCapture capture;
      auto request = factory.create_request(
          "", {}, sp, std::nullopt, make_capture_callback(&capture));
      if (!request) {
        return capture.status.value();
      }
      actual = request->state().mm_data;
      return Status{};
    };
    ASSERT_TRUE(execute(prepared).ok()) << args.model_type();
    EXPECT_TRUE(torch::equal(
        authoritative.items<MMItemVec>()[0].state().mm_token_mask(),
        actual.items<MMItemVec>()[0].state().mm_token_mask()));
    auto tampered = prepared;
    tampered.set_token_ids(0, 99);
    EXPECT_EQ(execute(tampered).code(), StatusCode::INVALID_ARGUMENT);
    tampered = prepared;
    tampered.mutable_mm_data()
        ->mutable_items(0)
        ->mutable_state()
        ->set_schedule_data_key(std::string(16, 'x'));
    EXPECT_EQ(execute(tampered).code(), StatusCode::INVALID_ARGUMENT);
    tampered = prepared;
    tampered.mutable_mm_data()
        ->mutable_items(0)
        ->mutable_state()
        ->set_token_pos_length(0);
    EXPECT_EQ(execute(tampered).code(), StatusCode::INVALID_ARGUMENT);
    tampered = prepared;
    tampered.mutable_sources(0)->mutable_media()->mutable_image_url()->set_url(
        "file:///tmp/image.png");
    EXPECT_EQ(execute(tampered).code(), StatusCode::INVALID_ARGUMENT);
  }
}

TEST_F(TypedImageRequestTest,
       FactoryConsumesAuthoritativeTokensWithoutRendering) {
  auto sp = params();
  ASSERT_TRUE(factory_->validate_multimodal_request(sp).ok());
  ASSERT_FALSE(limiter_.is_limited());
  CallbackCapture capture;
  auto request = factory_->create_request(
      "", {}, sp, std::nullopt, make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(processor_->render_calls.load(), 0);
  EXPECT_EQ(processor_->image_calls.load(), 2);
  EXPECT_EQ(request->state().prompt_tokens,
            std::vector<int32_t>(prompt_.token_ids().begin(),
                                 prompt_.token_ids().end()));
  EXPECT_EQ(request->state().decode_address, "decode");
  ASSERT_TRUE(request->state().mm_data.hold<MMItemVec>());
  EXPECT_TRUE(request->state()
                  .mm_data.items<MMItemVec>()[0]
                  .get<torch::Tensor>("pixel_values")
                  .has_value());
  request.reset();
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
}

// Exercise the same factory and response pools as service-routed VLM Prefill.
TEST_F(TypedImageRequestTest,
       RoutesTypedImageStreamingBatchAfterFactoryDestruction) {
  auto sp = params();
  sp.streaming = true;
  sp.source_xservice_addr = "owner-service";
  ASSERT_FALSE(limiter_.is_limited());
  auto request = factory_->create_request(
      "", {}, sp, std::nullopt, [](const RequestOutput&) {
        ADD_FAILURE() << "Service output must use the batch RPC handler";
        return false;
      });
  ASSERT_NE(request, nullptr);
  request->sequences().front()->kv_state().set_kv_cache_tokens_num(
      request->state().prompt_tokens.size());
  request->sequences().front()->append_token(Token(65));
  factory_.reset();
  std::atomic<int32_t> cancellations{0};
  AsyncResponseProcessor response(
      &tokenizer_,
      InstanceRole::PREFILL,
      /*enable_service_routing=*/true,
      /*disable_log_stats=*/true,
      [&cancellations](std::shared_ptr<Request>) { ++cancellations; });
  std::vector<std::shared_ptr<Request>> requests{request};
  response.process_stream_requests(requests);
  response.wait_completion();
  EXPECT_EQ(cancellations.load(), 0);
  ASSERT_EQ(batch_outputs_->size(), 1);
  const auto& output = batch_outputs_->front();
  EXPECT_EQ(output.request_id, sp.request_id);
  EXPECT_EQ(output.service_request_id, sp.service_request_id);
  EXPECT_EQ(output.target_xservice_addr, sp.source_xservice_addr);
  EXPECT_TRUE(output.finished_on_prefill_instance);
}

TEST_F(TypedImageRequestTest,
       RoutesTypedImageCompletedBatchAfterFactoryDestruction) {
  auto sp = params();
  sp.streaming = false;
  sp.max_tokens = 1;
  sp.source_xservice_addr = "owner-service";
  ASSERT_FALSE(limiter_.is_limited());
  auto request = factory_->create_request(
      "", {}, sp, std::nullopt, [](const RequestOutput&) {
        ADD_FAILURE() << "Service output must use the batch RPC handler";
        return false;
      });
  ASSERT_NE(request, nullptr);
  request->sequences().front()->kv_state().set_kv_cache_tokens_num(
      request->state().prompt_tokens.size());
  request->sequences().front()->append_token(Token(65));
  factory_.reset();
  AsyncResponseProcessor response(&tokenizer_,
                                  InstanceRole::PREFILL,
                                  /*enable_service_routing=*/true,
                                  /*disable_log_stats=*/true,
                                  [](std::shared_ptr<Request>) {});
  std::vector<std::shared_ptr<Request>> requests{request};
  response.process_completed_requests(requests, /*terminal=*/false);
  response.wait_completion();
  ASSERT_EQ(batch_outputs_->size(), 1);
  const auto& output = batch_outputs_->front();
  EXPECT_EQ(output.request_id, sp.request_id);
  EXPECT_EQ(output.service_request_id, sp.service_request_id);
  EXPECT_EQ(output.target_xservice_addr, sp.source_xservice_addr);
  ASSERT_TRUE(output.status);
  EXPECT_TRUE(output.status->ok());
  EXPECT_TRUE(output.finished_on_prefill_instance);
}

TEST_F(TypedImageRequestTest, AcceptsDifferentCompatibleMetadataProvider) {
  auto other = prompt_;
  auto* provider = other.mutable_sources(0)->mutable_preprocessing_engine();
  provider->set_engine_uid("metadata-A");
  provider->set_incarnation_id("metadata-A-incarnation");
  provider->set_profile_digest("metadata-A-profile");
  auto sp = params();
  sp.multimodal_prompt = prepared_input(other);
  ASSERT_TRUE(factory_->validate_multimodal_request(sp).ok());
  ASSERT_FALSE(limiter_.is_limited());
  CallbackCapture capture;
  auto request = factory_->create_request(
      "", {}, sp, std::nullopt, make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(processor_->image_calls.load(), 2);
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 1);
  request.reset();
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(TypedImageRequestTest, RejectsLocalGridMismatchAndReleasesFactorySlot) {
  auto invalid = prompt_;
  auto* grid = (*invalid.mutable_mm_data()
                     ->mutable_items(0)
                     ->mutable_dict())["image_grid_thw"]
                   .mutable_single_tensor()
                   ->mutable_contents();
  grid->set_int64_contents(1, grid->int64_contents(1) * 2);
  const int32_t extra = invalid.mm_data().items(0).state().token_pos_length();
  auto* state = invalid.mutable_mm_data()->mutable_items(0)->mutable_state();
  state->set_token_pos_length(extra * 2);
  state->set_mm_token_num(extra * 2);
  invalid.mutable_token_ids()->RemoveLast();
  for (int32_t index = 0; index < extra; ++index) {
    invalid.add_token_ids(args_.image_token_id());
  }
  invalid.add_token_ids(args_.vision_end_token_id());
  auto sp = params();
  sp.multimodal_prompt = prepared_input(invalid);
  ASSERT_FALSE(limiter_.is_limited());
  CallbackCapture capture;
  EXPECT_EQ(factory_->create_request(
                "", {}, sp, std::nullopt, make_capture_callback(&capture)),
            nullptr);
  EXPECT_TRUE(capture.called);
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(TypedImageRequestTest, RejectsSameGridDifferentImageBeforeServing) {
  const std::string original_bytes = image_bytes(/*color=*/20);
  const std::string changed_bytes = image_bytes(/*color=*/21);
  prepare_images({original_bytes});
  const auto original = prompt_;
  prepare_images({changed_bytes});
  ASSERT_EQ(prompt_.mm_data()
                .items(0)
                .dict()
                .at("image_grid_thw")
                .SerializeAsString(),
            original.mm_data()
                .items(0)
                .dict()
                .at("image_grid_thw")
                .SerializeAsString());
  ASSERT_EQ(std::vector<int32_t>(prompt_.token_ids().begin(),
                                 prompt_.token_ids().end()),
            std::vector<int32_t>(original.token_ids().begin(),
                                 original.token_ids().end()));
  ASSERT_NE(prompt_.mm_data().items(0).state().schedule_data_key(),
            original.mm_data().items(0).state().schedule_data_key());
  // Equivalent to a URL returning new bytes between preparation and serving.
  auto stale = original;
  *stale.mutable_sources(0)->mutable_media() = prompt_.sources(0).media();
  expect_key_rejection(stale);
  auto valid = process_prepared(original);
  ASSERT_TRUE(valid.first.ok());
  EXPECT_EQ(std::string(
                reinterpret_cast<const char*>(valid.second.items<MMItemVec>()[0]
                                                  .state()
                                                  .schedule_data()
                                                  .key.data),
                XXH3_128BITS_HASH_VALUE_LEN),
            original.mm_data().items(0).state().schedule_data_key());
}

TEST_F(TypedImageRequestTest,
       RejectsEveryPreparedKeyByteIncludingEmbeddedNull) {
  const std::string original =
      prompt_.mm_data().items(0).state().schedule_data_key();
  ASSERT_EQ(original.size(), XXH3_128BITS_HASH_VALUE_LEN);
  for (size_t index = 0; index < original.size(); ++index) {
    SCOPED_TRACE(index);
    auto changed = prompt_;
    std::string key = original;
    key[index] ^= 1;
    changed.mutable_mm_data()
        ->mutable_items(0)
        ->mutable_state()
        ->set_schedule_data_key(key);
    expect_key_rejection(changed);
  }
  auto changed = prompt_;
  std::string nul_key = original;
  nul_key[0] = '\0';
  nul_key[15] ^= 1;
  changed.mutable_mm_data()
      ->mutable_items(0)
      ->mutable_state()
      ->set_schedule_data_key(nul_key);
  expect_key_rejection(changed);
  EXPECT_TRUE(process_prepared(prompt_).first.ok());
}

TEST_F(TypedImageRequestTest, RejectsExchangedKeysForSameGridImageOccurrences) {
  prepare_images({image_bytes(/*color=*/20), image_bytes(/*color=*/21)});
  ASSERT_EQ(prompt_.mm_data()
                .items(0)
                .dict()
                .at("image_grid_thw")
                .SerializeAsString(),
            prompt_.mm_data()
                .items(1)
                .dict()
                .at("image_grid_thw")
                .SerializeAsString());
  const std::string first =
      prompt_.mm_data().items(0).state().schedule_data_key();
  const std::string second =
      prompt_.mm_data().items(1).state().schedule_data_key();
  ASSERT_NE(first, second);
  auto exchanged = prompt_;
  exchanged.mutable_mm_data()
      ->mutable_items(0)
      ->mutable_state()
      ->set_schedule_data_key(second);
  exchanged.mutable_mm_data()
      ->mutable_items(1)
      ->mutable_state()
      ->set_schedule_data_key(first);
  expect_key_rejection(exchanged);
  auto valid = process_prepared(prompt_);
  ASSERT_TRUE(valid.first.ok());
  ASSERT_EQ(valid.second.size(), 2u);
  const auto& items = valid.second.items<MMItemVec>();
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(
                            items[0].state().schedule_data().key.data),
                        16),
            first);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(
                            items[1].state().schedule_data().key.data),
                        16),
            second);
}

TEST_F(TypedImageRequestTest,
       RejectsDifferentPreparationSeedAndReleasesCapacity) {
  // The input hash is recomputed by the actual image handler, not patched in
  // the fixture. Only the execution-side seed changes after preparation.
  const uint32_t seed = KVCacheConfig::get_instance().xxh3_128bits_seed();
  KVCacheConfig::get_instance().xxh3_128bits_seed(seed ^ 1);
  expect_key_rejection(prompt_);
  // Drain callbacks before restoring process-wide configuration.
  KVCacheConfig::get_instance().xxh3_128bits_seed(seed);
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(TypedImageRequestTest, FactoryRechecksCancellationAfterProcessing) {
  brpc::Controller controller;
  FakeCall call(&controller);
  processor_->before_process = [&] { call.disconnect(); };
  auto sp = params();
  ASSERT_FALSE(limiter_.is_limited());
  CallbackCapture capture;
  EXPECT_EQ(factory_->create_request(
                "", {}, sp, &call, make_capture_callback(&capture)),
            nullptr);
  ASSERT_TRUE(capture.status);
  EXPECT_EQ(capture.status->code(), StatusCode::CANCELLED);
  EXPECT_EQ(processor_->image_calls.load(), 2);
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(TypedImageRequestTest, DisconnectedFactoryCallDoesNotProcessImages) {
  brpc::Controller controller;
  FakeCall call(&controller);
  call.disconnect();
  auto sp = params();
  ASSERT_FALSE(limiter_.is_limited());
  CallbackCapture capture;
  EXPECT_EQ(factory_->create_request(
                "", {}, sp, &call, make_capture_callback(&capture)),
            nullptr);
  ASSERT_TRUE(capture.status);
  EXPECT_EQ(capture.status->code(), StatusCode::CANCELLED);
  EXPECT_EQ(processor_->image_calls.load(), 1);
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(TypedImageRequestTest, PublicJsonAdaptersRetainTypedTokensAndEmptyText) {
  proto::CompletionRequest completion;
  *completion.mutable_multimodal_prompt() = prompt_;
  std::string json;
  ASSERT_TRUE(
      google::protobuf::util::MessageToJsonString(completion, &json).ok());
  auto [status, prepared] = preprocess_completion_prompt(json);
  ASSERT_TRUE(status.ok());
  proto::CompletionRequest decoded;
  ASSERT_TRUE(
      google::protobuf::util::JsonStringToMessage(prepared, &decoded).ok());
  RequestParams params(decoded, "", "");
  ASSERT_TRUE(bool(params.multimodal_prompt));
  EXPECT_EQ(params.multimodal_prompt->token_ids.size(),
            prompt_.token_ids_size());
  proto::ChatRequest chat;
  *chat.mutable_multimodal_prompt() = prompt_;
  auto* message = chat.add_messages();
  message->set_role("user");
  message->set_content("describe");
  // MessageToJsonString appends; the chat adapter needs its own JSON document.
  std::string chat_json_input;
  ASSERT_TRUE(
      google::protobuf::util::MessageToJsonString(chat, &chat_json_input).ok());
  auto [chat_status, chat_json] =
      ChatJsonParser::get(ServingMode::LLM).preprocess(chat_json_input);
  ASSERT_TRUE(chat_status.ok()) << chat_status.message();
  proto::ChatRequest decoded_chat;
  ASSERT_TRUE(
      google::protobuf::util::JsonStringToMessage(chat_json, &decoded_chat)
          .ok());
  EXPECT_EQ(decoded_chat.multimodal_prompt().SerializeAsString(),
            prompt_.SerializeAsString());
}

}  // namespace
}  // namespace xllm
