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

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/multimodal/mm_handler.h"
#include "core/framework/multimodal/mm_input.h"

namespace xllm {
namespace {

constexpr char kTinyPngDataUrl[] =
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQIHWP4z8AAAAMB"
    "AQBb2/lEAAAAAElFTkSuQmCC";

struct ItemCounts {
  double success;
  double failed;
};

ItemCounts item_counts(const char* modality, const char* representation) {
  return {MULTI_COUNTER_mm_input_items_total
              .get_stats({modality, representation, "success"})
              ->get_value(),
          MULTI_COUNTER_mm_input_items_total
              .get_stats({modality, representation, "failed"})
              ->get_value()};
}

void expect_item_counts(const char* modality,
                        const char* representation,
                        const ItemCounts& before,
                        double success,
                        double failed) {
  const ItemCounts after = item_counts(modality, representation);
  EXPECT_EQ(after.success, before.success + success);
  EXPECT_EQ(after.failed, before.failed + failed);
}

MMErrCode transfer_content(MMContent content, MMInput& inputs) {
  MMContentVec contents;
  contents.reserve(1);
  contents.emplace_back(std::move(content));
  std::vector<Message> messages;
  messages.reserve(1);
  messages.emplace_back("user", std::move(contents));
  MMInputTransfer transfer;
  return transfer.trans(messages, inputs);
}

proto::Embedding make_embedding() {
  proto::Embedding embedding;
  embedding.set_hash_key("00000000000000000000000000000001");
  proto::Tensor* tensor = embedding.mutable_embedding();
  tensor->set_datatype("FP32");
  tensor->add_shape(1);
  tensor->add_shape(2);
  tensor->mutable_contents()->add_fp32_contents(1.0f);
  tensor->mutable_contents()->add_fp32_contents(2.0f);
  return embedding;
}

struct MediaCase {
  const char* modality;
  const char* raw_type;
  const char* embedding_type;
};

class MMInputModalityMetricsTest : public ::testing::TestWithParam<MediaCase> {
};

TEST_P(MMInputModalityMetricsTest, CountsRawFailureByModality) {
  const MediaCase& media = GetParam();
  MMContent content(media.raw_type, /*text=*/"");
  content.image_url.url = "data:image/png;base64,%%%";
  content.video_url.url = "data:video/mp4;base64,%%%";
  content.audio_url.url = "data:audio/wav;base64,%%%";
  MMInput inputs;
  const ItemCounts raw_before = item_counts(media.modality, "raw");
  const ItemCounts unknown_representation_before =
      item_counts(media.modality, "unknown");
  const ItemCounts unknown_before = item_counts("unknown", "unknown");

  EXPECT_EQ(transfer_content(std::move(content), inputs),
            MMErrCode::LOAD_DATA_ERR);

  EXPECT_TRUE(inputs.empty());
  expect_item_counts(
      media.modality, "raw", raw_before, /*success=*/0, /*failed=*/1);
  expect_item_counts(media.modality,
                     "unknown",
                     unknown_representation_before,
                     /*success=*/0,
                     /*failed=*/0);
  expect_item_counts(
      "unknown", "unknown", unknown_before, /*success=*/0, /*failed=*/0);
}

TEST_P(MMInputModalityMetricsTest, CountsPreembeddedInputWithoutRawMedia) {
  const MediaCase& media = GetParam();
  MMContent content(media.embedding_type, make_embedding());
  MMInput inputs;
  const ItemCounts embedded_before = item_counts(media.modality, "embedding");
  const ItemCounts raw_before = item_counts(media.modality, "raw");

  ASSERT_EQ(transfer_content(std::move(content), inputs), MMErrCode::SUCCESS);

  ASSERT_EQ(inputs.size(), 1u);
  const MMInputItem& input = inputs.items().front();
  ASSERT_TRUE(input.is_embedding());
  EXPECT_TRUE(
      torch::equal(input.embedding.embedding, torch::tensor({{1.0f, 2.0f}})));
  EXPECT_TRUE(input.raw_data.empty());
  expect_item_counts(media.modality,
                     "embedding",
                     embedded_before,
                     /*success=*/1,
                     /*failed=*/0);
  expect_item_counts(
      media.modality, "raw", raw_before, /*success=*/0, /*failed=*/0);
}

TEST_P(MMInputModalityMetricsTest, CountsEmbeddingFailureByModality) {
  const MediaCase& media = GetParam();
  MMContent content(media.embedding_type, proto::Embedding());
  MMInput inputs;
  const ItemCounts embedded_before = item_counts(media.modality, "embedding");
  const ItemCounts raw_before = item_counts(media.modality, "raw");
  const ItemCounts unknown_representation_before =
      item_counts(media.modality, "unknown");
  const ItemCounts unknown_before = item_counts("unknown", "unknown");

  EXPECT_EQ(transfer_content(std::move(content), inputs),
            MMErrCode::PARSE_EMB_ERR);

  EXPECT_TRUE(inputs.empty());
  expect_item_counts(media.modality,
                     "embedding",
                     embedded_before,
                     /*success=*/0,
                     /*failed=*/1);
  expect_item_counts(
      media.modality, "raw", raw_before, /*success=*/0, /*failed=*/0);
  expect_item_counts(media.modality,
                     "unknown",
                     unknown_representation_before,
                     /*success=*/0,
                     /*failed=*/0);
  expect_item_counts(
      "unknown", "unknown", unknown_before, /*success=*/0, /*failed=*/0);
}

INSTANTIATE_TEST_SUITE_P(
    ImageVideoAudio,
    MMInputModalityMetricsTest,
    ::testing::Values(MediaCase{"image", "image_url", "image_embedding"},
                      MediaCase{"video", "video_url", "video_embedding"},
                      MediaCase{"audio", "audio_url", "audio_embedding"}));

TEST(MMInputMetricsTest, CountsRawImageSuccess) {
  MMContent content("image_url", ImageURL{.url = kTinyPngDataUrl});
  MMInput inputs;
  const ItemCounts before = item_counts("image", "raw");

  ASSERT_EQ(transfer_content(std::move(content), inputs), MMErrCode::SUCCESS);

  ASSERT_EQ(inputs.size(), 1u);
  EXPECT_TRUE(inputs.items().front().decode_image.defined());
  expect_item_counts("image", "raw", before, /*success=*/1, /*failed=*/0);
}

TEST(MMInputMetricsTest, CountsDecodeFailureOnlyOnce) {
  MMContent content("image_url",
                    ImageURL{.url = "data:image/png;base64,bm90LWFuLWltYWdl"});
  MMInput inputs;
  const ItemCounts raw_before = item_counts("image", "raw");
  const ItemCounts unknown_representation_before =
      item_counts("image", "unknown");
  const ItemCounts unknown_before = item_counts("unknown", "unknown");

  EXPECT_EQ(transfer_content(std::move(content), inputs),
            MMErrCode::DECODE_ERR);

  EXPECT_TRUE(inputs.empty());
  expect_item_counts("image", "raw", raw_before, /*success=*/0, /*failed=*/1);
  expect_item_counts("image",
                     "unknown",
                     unknown_representation_before,
                     /*success=*/0,
                     /*failed=*/0);
  expect_item_counts(
      "unknown", "unknown", unknown_before, /*success=*/0, /*failed=*/0);
}

TEST(MMInputMetricsTest, DirectHandlerCallsDoNotCountInputItems) {
  const MMContent content("image_url", ImageURL{.url = kTinyPngDataUrl});
  MMHandlerSet handlers;
  MMInputItem input;
  const ItemCounts before = item_counts("image", "raw");

  ASSERT_EQ(handlers.process(content.type, content, input, BinaryPayload()),
            MMErrCode::SUCCESS);

  EXPECT_TRUE(input.decode_image.defined());
  expect_item_counts("image", "raw", before, /*success=*/0, /*failed=*/0);
}

TEST(MMInputMetricsTest, ParallelImagesCountItemsRatherThanRequests) {
  MMContentVec content;
  content.reserve(3);
  content.emplace_back("text", std::string("describe"));
  content.emplace_back("image_url", ImageURL{.url = kTinyPngDataUrl});
  content.emplace_back("image_url", ImageURL{.url = kTinyPngDataUrl});
  const std::vector<Message> messages = {Message("user", std::move(content))};
  MMInputTransfer transfer;
  MMInput inputs;
  const ItemCounts before = item_counts("image", "raw");

  ASSERT_EQ(transfer.trans(messages, inputs), MMErrCode::SUCCESS);

  EXPECT_EQ(inputs.size(), 2u);
  expect_item_counts("image", "raw", before, /*success=*/2, /*failed=*/0);
}

TEST(MMInputMetricsTest, TextOnlyDoesNotCountMediaItems) {
  const std::vector<Message> messages = {
      Message("user", MMContentVec{MMContent("text", std::string("hi"))})};
  MMInputTransfer transfer;
  MMInput inputs;
  const ItemCounts image_before = item_counts("image", "raw");
  const ItemCounts unknown_before = item_counts("unknown", "unknown");

  ASSERT_EQ(transfer.trans(messages, inputs), MMErrCode::SUCCESS);

  EXPECT_TRUE(inputs.empty());
  expect_item_counts("image", "raw", image_before, /*success=*/0, /*failed=*/0);
  expect_item_counts(
      "unknown", "unknown", unknown_before, /*success=*/0, /*failed=*/0);
}

TEST(MMInputMetricsTest, UnsupportedTypesShareFixedUnknownLabels) {
  MMInput inputs;
  const ItemCounts before = item_counts("unknown", "unknown");

  for (const char* type : {"unsupported_one",
                           "unsupported_two",
                           "image_url_extra",
                           "video_url_extra",
                           "audio_embedding_extra",
                           "other_url",
                           "other_embedding"}) {
    SCOPED_TRACE(type);
    EXPECT_EQ(transfer_content(MMContent(type, /*text=*/""), inputs),
              MMErrCode::HANDLER_ERR);
    EXPECT_TRUE(inputs.empty());
  }

  expect_item_counts("unknown", "unknown", before, /*success=*/0, /*failed=*/7);
}

TEST(MMInputMetricsTest, UnstartedMediaTasksDoNotCountAsFailures) {
  MMContent unsupported(/*type=*/"unsupported", /*text=*/"");
  MMContentVec content;
  content.reserve(2);
  content.emplace_back("image_url", ImageURL{.url = kTinyPngDataUrl});
  content.emplace_back(std::move(unsupported));
  const std::vector<Message> messages = {Message("user", std::move(content))};
  MMInputTransfer transfer;
  MMInput inputs;
  const ItemCounts image_before = item_counts("image", "raw");
  const ItemCounts unknown_representation_before =
      item_counts("image", "unknown");
  const ItemCounts unknown_before = item_counts("unknown", "unknown");

  EXPECT_EQ(transfer.trans(messages, inputs), MMErrCode::HANDLER_ERR);

  EXPECT_TRUE(inputs.empty());
  expect_item_counts("image", "raw", image_before, /*success=*/0, /*failed=*/0);
  expect_item_counts("image",
                     "unknown",
                     unknown_representation_before,
                     /*success=*/0,
                     /*failed=*/0);
  expect_item_counts(
      "unknown", "unknown", unknown_before, /*success=*/0, /*failed=*/1);
}

}  // namespace
}  // namespace xllm
