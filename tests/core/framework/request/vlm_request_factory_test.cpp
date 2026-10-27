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

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/request/sequence.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {
namespace {

constexpr char kTinyPngDataUrl[] =
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQIHWP4z8AAAAMB"
    "AQBb2/lEAAAAAElFTkSuQmCC";

struct InputMetricsSnapshot {
  int64_t transfer_count = HISTOGRAM_mm_transfer_latency_milliseconds.count();
  int64_t processor_count = HISTOGRAM_mm_processor_latency_milliseconds.count();
  double template_seconds = COUNTER_VALUE(chat_template_latency_seconds);
  double transfer_failures =
      MULTI_COUNTER_VALUE(mm_input_stage_failures_total, "transfer");
  double processor_failures =
      MULTI_COUNTER_VALUE(mm_input_stage_failures_total, "processor");
};

void expect_input_stage_failures(const InputMetricsSnapshot& before,
                                 int64_t transfer = 0,
                                 int64_t processor = 0) {
  EXPECT_DOUBLE_EQ(
      MULTI_COUNTER_VALUE(mm_input_stage_failures_total, "transfer"),
      before.transfer_failures + transfer);
  EXPECT_DOUBLE_EQ(
      MULTI_COUNTER_VALUE(mm_input_stage_failures_total, "processor"),
      before.processor_failures + processor);
}

// Deterministic tokenizer used for factory tests. It only participates in stop
// sequence encoding here; encoding fails for any text containing "FAIL" so the
// stop-sequence error path can be exercised independently.
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
    return static_cast<size_t>(vocab_size_);
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>(*this);
  }

 private:
  int32_t vocab_size_ = 1000;
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
    // Simulate the caller (service entry) having acquired a rate-limit slot.
    rate_limiter_.is_limited();
    ASSERT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
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
    return std::make_unique<VLMRequestFactory>(processor_.get(),
                                               chat_template_.get(),
                                               tokenizer_.get(),
                                               &model_args_,
                                               &options_,
                                               &rate_limiter_);
  }

  std::unique_ptr<FakeProcessor> processor_;
  std::unique_ptr<FakeJinjaChatTemplate> chat_template_;
  std::unique_ptr<FakeTokenizer> tokenizer_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
};

TEST_F(VLMRequestFactoryTest,
       RejectsEmptyPromptAndMmDataReleasesRateLimitSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"",
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

  auto request = factory->create(/*prompt=*/"describe this",
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

  auto request = factory->create(/*prompt=*/"describe this",
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

  auto request = factory->create(/*prompt=*/"describe this",
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

  auto request = factory->create(/*prompt=*/"describe this",
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

TEST_F(VLMRequestFactoryTest, MessageOverloadFailsWhenTemplateRejects) {
  auto factory = make_factory();
  chat_template_->set_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  // MMInputTransfer requires MMContentVec content; a text part is enough.
  MMContentVec content;
  content.emplace_back("text", std::string("hi"));
  std::vector<Message> messages;
  messages.emplace_back("user", content);

  const InputMetricsSnapshot before;
  auto request = factory->create(messages,
                                 sp,
                                 /*payload=*/BinaryPayload(),
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

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
  auto request = factory->create(messages,
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
  auto request = factory->create(std::move(messages),
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
  auto request = factory->create(std::move(messages),
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
  auto request = factory->create(std::move(messages),
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
      return factory.create(/*prompt=*/"describe this",
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
    return factory.create(std::move(messages),
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
  auto request = factory->create(std::move(messages),
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
  auto request = factory->create(ChatMessages{assistant},
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

}  // namespace
}  // namespace xllm
