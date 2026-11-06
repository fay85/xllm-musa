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

#include <gtest/gtest.h>
#include <json2pb/pb_to_json.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "api_service/stream_call.h"
#include "api_service/utils.h"
#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm {
namespace {

std::vector<JsonTool> make_test_tools() {
  return {JsonTool("function",
                   JsonFunction("get_weather",
                                "Get the weather",
                                nlohmann::json{{"type", "object"}}))};
}

template <typename Request, typename Response>
class InspectableStreamCall final : public StreamCall<Request, Response> {
 public:
  using Base = StreamCall<Request, Response>;
  using Base::Base;

  // Exercise the production SSE serializer without publishing a frame.
  using Base::serialize_stream_response;

  std::string serialized_frame() const { return this->io_buf_.to_string(); }
};

TEST(UsageJsonTest, ChatUsageSerializesOpenAICachedTokensField) {
  Usage usage;
  usage.num_prompt_tokens = 1024;
  usage.num_generated_tokens = 50;
  usage.num_total_tokens = 1074;
  usage.num_cached_tokens = 896;

  proto::ChatResponse response;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  options.always_print_primitive_fields = true;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  nlohmann::json json = nlohmann::json::parse(json_text);
  ASSERT_TRUE(json.contains("usage"));
  EXPECT_EQ(json["usage"]["prompt_tokens"], 1024);
  EXPECT_EQ(json["usage"]["completion_tokens"], 50);
  EXPECT_EQ(json["usage"]["total_tokens"], 1074);
  ASSERT_TRUE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 896);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["audio_tokens"], 0);
  ASSERT_TRUE(json["usage"].contains("completion_tokens_details"));
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["audio_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"].size(), 2);
}

TEST(UsageJsonTest, ChatUsagePrintsZeroCachedTokens) {
  Usage usage;
  usage.num_prompt_tokens = 12;
  usage.num_generated_tokens = 3;
  usage.num_total_tokens = 15;
  usage.num_cached_tokens = 0;

  proto::ChatResponse response;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  options.always_print_primitive_fields = true;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  nlohmann::json json = nlohmann::json::parse(json_text);
  ASSERT_TRUE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["audio_tokens"], 0);
  ASSERT_TRUE(json["usage"].contains("completion_tokens_details"));
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["audio_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"].size(), 2);
}

TEST(UsageJsonTest, CompletionUsageChunkKeepsEmptyChoices) {
  proto::CompletionRequest request;
  request.set_stream(true);
  proto::CompletionResponse response;
  response.set_id("cmpl_usage");
  response.set_object("text_completion");
  response.set_created(123);
  response.set_model("test_model");
  Usage usage;
  usage.num_prompt_tokens = 1024;
  usage.num_generated_tokens = 32;
  usage.num_total_tokens = 1056;
  usage.num_cached_tokens = 512;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  brpc::Controller controller;
  InspectableStreamCall<proto::CompletionRequest, proto::CompletionResponse>
      call(&controller,
           brpc::DoNothing(),
           &request,
           &response,
           /*use_arena=*/true);

  // The default frame omits every empty repeated field, including choices.
  ASSERT_TRUE(call.serialize_stream_response(response));
  const nlohmann::json default_json =
      nlohmann::json::parse(call.serialized_frame().substr(6));
  EXPECT_FALSE(default_json.contains("choices"));
  EXPECT_FALSE(default_json.contains("output_tensors"));

  ASSERT_TRUE(call.serialize_stream_response(response,
                                             /*empty_array_field=*/"choices"));
  const std::string frame = call.serialized_frame();
  ASSERT_EQ(frame.substr(0, 6), "data: ");
  ASSERT_EQ(frame.substr(frame.size() - 2), "\n\n");
  nlohmann::json json = nlohmann::json::parse(frame.substr(6));
  ASSERT_TRUE(json.contains("choices"));
  ASSERT_TRUE(json["choices"].is_array());
  EXPECT_TRUE(json["choices"].empty());
  EXPECT_EQ(json["id"], "cmpl_usage");
  EXPECT_EQ(json["object"], "text_completion");
  EXPECT_EQ(json["created"], 123);
  EXPECT_EQ(json["model"], "test_model");
  EXPECT_EQ(json["usage"]["prompt_tokens"], 1024);
  EXPECT_EQ(json["usage"]["completion_tokens"], 32);
  EXPECT_EQ(json["usage"]["total_tokens"], 1056);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 512);
  EXPECT_FALSE(json.contains("output_tensors"));
  json.erase("choices");
  EXPECT_EQ(json, default_json);
}

TEST(UsageJsonTest, ChatUsageChunkKeepsEmptyChoices) {
  proto::ChatRequest request;
  request.set_stream(true);
  proto::ChatResponse response;
  response.set_id("chatcmpl_usage");
  response.set_object("chat.completion.chunk");
  response.set_created(456);
  response.set_model("test_model");
  Usage usage;
  usage.num_prompt_tokens = 64;
  usage.num_generated_tokens = 8;
  usage.num_total_tokens = 72;
  usage.num_cached_tokens = 0;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  brpc::Controller controller;
  InspectableStreamCall<proto::ChatRequest, proto::ChatResponse> call(
      &controller, brpc::DoNothing(), &request, &response, /*use_arena=*/true);

  ASSERT_TRUE(call.serialize_stream_response(response,
                                             /*empty_array_field=*/"choices"));
  const std::string frame = call.serialized_frame();
  ASSERT_EQ(frame.substr(0, 6), "data: ");
  ASSERT_EQ(frame.substr(frame.size() - 2), "\n\n");
  const nlohmann::json json = nlohmann::json::parse(frame.substr(6));
  ASSERT_TRUE(json.contains("choices"));
  ASSERT_TRUE(json["choices"].is_array());
  EXPECT_TRUE(json["choices"].empty());
  EXPECT_EQ(json["id"], "chatcmpl_usage");
  EXPECT_EQ(json["object"], "chat.completion.chunk");
  EXPECT_EQ(json["created"], 456);
  EXPECT_EQ(json["model"], "test_model");
  EXPECT_EQ(json["usage"]["prompt_tokens"], 64);
  EXPECT_EQ(json["usage"]["completion_tokens"], 8);
  EXPECT_EQ(json["usage"]["total_tokens"], 72);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
}

TEST(ToolCallResultTest, KeepsNormalTextWhenNoToolCallMarkerExists) {
  const std::string text = "The weather is sunny.";

  const auto result =
      api_service::process_tool_calls(text, make_test_tools(), "glm45", "stop");

  EXPECT_FALSE(result.tool_calls.has_value());
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.finish_reason, "stop");
}

TEST(ToolCallResultTest, KeepsNormalTextWhenMarkerDoesNotParse) {
  const std::string text =
      "The weather is sunny. <tool_call>unknown_tool\n"
      "<arg_key>location</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n"
      "</tool_call>";

  const auto result =
      api_service::process_tool_calls(text, make_test_tools(), "glm45", "stop");

  EXPECT_FALSE(result.tool_calls.has_value());
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.finish_reason, "stop");
}

TEST(ToolCallResultTest, PreservesParsedToolCall) {
  const auto result = api_service::process_tool_calls(
      "I will check. <tool_call>get_weather\n"
      "<arg_key>location</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n"
      "</tool_call>",
      make_test_tools(),
      "glm45",
      "stop");

  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->at(0).function().name(), "get_weather");
  EXPECT_EQ(result.finish_reason, "tool_calls");
}

TEST(ChatResponseJsonTest, OmitsEmptyToolCallsForNormalText) {
  auto* request = new proto::ChatRequest();
  request->set_stream(false);
  auto* response = new proto::ChatResponse();
  auto* choice = response->add_choices();
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  message->set_content("The weather is sunny.");

  brpc::Controller controller;
  {
    StreamCall<proto::ChatRequest, proto::ChatResponse> call(
        &controller, brpc::DoNothing(), request, response);
    ASSERT_TRUE(call.write_and_finish(*response));
  }

  const nlohmann::json json =
      nlohmann::json::parse(controller.response_attachment().to_string());
  ASSERT_TRUE(json["choices"][0].contains("message"));
  EXPECT_FALSE(json["choices"][0]["message"].contains("tool_calls"));
}

TEST(ChatResponseJsonTest, KeepsNonEmptyToolCalls) {
  proto::ChatResponse response;
  auto* choice = response.add_choices();
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  auto* tool_call = message->add_tool_calls();
  tool_call->set_id("call_test");
  tool_call->set_type("function");
  tool_call->mutable_function()->set_name("get_weather");
  tool_call->mutable_function()->set_arguments("{\"location\":\"Beijing\"}");

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = false;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  const nlohmann::json json = nlohmann::json::parse(json_text);
  const auto& tool_calls = json["choices"][0]["message"]["tool_calls"];
  ASSERT_TRUE(tool_calls.is_array());
  ASSERT_EQ(tool_calls.size(), 1);
  EXPECT_EQ(tool_calls[0]["function"]["name"], "get_weather");
}

TEST(AnthropicCallJsonTest, KeepsEmptyContentArray) {
  auto* request = new proto::AnthropicMessagesRequest();
  request->set_stream(false);
  auto* response = new proto::AnthropicMessagesResponse();

  brpc::Controller controller;
  {
    AnthropicCall call(&controller, brpc::DoNothing(), request, response);
    ASSERT_TRUE(call.write_and_finish(*response));
  }

  const nlohmann::json json =
      nlohmann::json::parse(controller.response_attachment().to_string());
  ASSERT_TRUE(json.contains("content"));
  EXPECT_TRUE(json["content"].is_array());
  EXPECT_TRUE(json["content"].empty());
}

}  // namespace
}  // namespace xllm
