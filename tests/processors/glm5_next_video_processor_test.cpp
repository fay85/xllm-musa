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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "processors/glm4v_prompt_processor.h"
#include "processors/glm4v_video_processor.h"

namespace xllm {
namespace {

ModelArgs video_args(const std::string& model_type = "glm5_next") {
  ModelArgs args;
  args.model_type(model_type)
      .mm_video_patch_size(14)
      .mm_video_temporal_patch_size(2)
      .mm_video_merge_size(2)
      .mm_image_merge_size(2)
      .mm_video_normalize_mean({0.5, 0.5, 0.5})
      .mm_video_normalize_std({0.5, 0.5, 0.5})
      .mm_video_min_tokens(1)
      .mm_video_max_tokens(240000);
  return args;
}

class Glm5NextVideoAliasTest : public ::testing::TestWithParam<std::string> {};

TEST_P(Glm5NextVideoAliasTest, SamplesVideoAndExpandsWholeSecondTimestamps) {
  const ModelArgs args = video_args(GetParam());
  const Glm4VVideoProcessor processor(
      args, Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 3.0;
  metadata.total_num_frames = 36;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({36, 3, 28, 28}, torch::kUInt8), metadata, item));
  const auto sampled = item.get_metadata<VideoMetadata>();
  ASSERT_TRUE(sampled.has_value());
  EXPECT_TRUE(
      torch::equal(sampled->frame_indices,
                   torch::tensor({0, 7, 14, 21, 28, 35}, torch::kLong)));
  EXPECT_TRUE(torch::equal(*item.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{3, 2, 2}}, torch::kLong)));
  EXPECT_EQ(item.get<torch::Tensor>("pixel_values_videos")->size(0), 12);
  EXPECT_EQ(item.get<torch::Tensor>("pixel_values_videos")->size(1), 1176);

  MMData data;
  data.add(MMType::VIDEO, item);
  GLM4VPromptProcessor prompt_processor(args);
  std::string prompt = "<|begin_of_video|><|video|><|end_of_video|>";
  ASSERT_TRUE(prompt_processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "<|begin_of_video|>"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|end_of_video|>");
}

INSTANTIATE_TEST_SUITE_P(ModelTypes,
                         Glm5NextVideoAliasTest,
                         ::testing::Values("glm5_next", "glm5_3_flash"));

TEST(Glm5NextVideoProcessorTest, HandlesSubsecondVideoWithOneSourceFrame) {
  const Glm4VVideoProcessor processor(
      video_args(), Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 1.0 / 12.0;
  metadata.total_num_frames = 1;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({1, 3, 28, 28}, torch::kUInt8), metadata, item));
  EXPECT_TRUE(torch::equal(item.get_metadata<VideoMetadata>()->frame_indices,
                           torch::tensor({0, 0}, torch::kLong)));
  EXPECT_TRUE(torch::equal(*item.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{1, 2, 2}}, torch::kLong)));
}

TEST(Glm5NextVideoProcessorTest, HonorsConfiguredSamplingRate) {
  ModelArgs args = video_args();
  args.mm_fps(1.0);
  const Glm4VVideoProcessor processor(
      args, Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 3.0;
  metadata.total_num_frames = 36;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({36, 3, 28, 28}, torch::kUInt8), metadata, item));
  EXPECT_TRUE(torch::equal(item.get_metadata<VideoMetadata>()->frame_indices,
                           torch::tensor({0, 17, 35, 35}, torch::kLong)));
}

TEST(Glm5NextVideoProcessorTest, KeepsOriginalIndicesForPreselectedFrames) {
  ModelArgs args = video_args();
  args.mm_fps(1.0).dtype("bfloat16");
  const Glm4VVideoProcessor processor(
      args, Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 3.0;
  metadata.total_num_frames = 36;
  const auto selector = processor.video_frame_selector();
  ASSERT_TRUE(selector);
  EXPECT_EQ(selector(metadata), (std::vector<int64_t>{0, 17, 35}));
  const torch::Tensor video = torch::arange(36 * 3 * 28 * 28, torch::kInt64)
                                  .remainder(256)
                                  .to(torch::kUInt8)
                                  .reshape({36, 3, 28, 28});
  MMDataItem baseline(MMType::NONE);
  ASSERT_TRUE(processor.process(video, metadata, baseline));
  metadata.frames_preselected = true;
  metadata.frame_indices = torch::tensor(selector(metadata), torch::kInt64);
  const auto selected = video.index_select(0, metadata.frame_indices);
  for (const auto format :
       {at::MemoryFormat::Contiguous, at::MemoryFormat::ChannelsLast}) {
    MMDataItem actual(MMType::NONE);
    ASSERT_TRUE(
        processor.process(selected.contiguous(format), metadata, actual));
    const auto processed = actual.get_metadata<VideoMetadata>();
    ASSERT_TRUE(processed);
    EXPECT_TRUE(torch::equal(processed->frame_indices,
                             torch::tensor({0, 17, 35, 35}, torch::kInt64)));
    EXPECT_TRUE(torch::equal(*actual.get<torch::Tensor>("video_grid_thw"),
                             *baseline.get<torch::Tensor>("video_grid_thw")));
    ASSERT_EQ(actual.get<torch::Tensor>("pixel_values_videos")->scalar_type(),
              torch::kBFloat16);
    EXPECT_TRUE(
        torch::equal(*actual.get<torch::Tensor>("pixel_values_videos"),
                     *baseline.get<torch::Tensor>("pixel_values_videos")));
  }
}

TEST(Glm5NextVideoProcessorTest, MatchesVaTruncationNearIntegerBoundary) {
  const Glm4VVideoProcessor processor(
      video_args(), Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 30.0;
  metadata.total_num_frames = 231;
  metadata.duration = metadata.total_num_frames / metadata.fps;
  const auto selector = processor.video_frame_selector();
  ASSERT_TRUE(selector);
  EXPECT_TRUE(selector.match_opencv_rgb);
  const auto selected = selector(metadata);
  ASSERT_GT(selected.size(), 7);
  EXPECT_EQ(selected[7], 114);
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({231, 3, 28, 28}, torch::kUInt8), metadata, item));
  EXPECT_EQ(
      item.get_metadata<VideoMetadata>()->frame_indices[7].item<int64_t>(),
      114);
}

TEST(Glm5NextVideoProcessorTest, PadsRightAndBottomWithoutStretching) {
  const Glm4VVideoProcessor processor(
      video_args(), Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 2.0;
  metadata.duration = 1.0;
  metadata.total_num_frames = 2;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::full({2, 3, 29, 30}, 255, torch::kUInt8), metadata, item));
  EXPECT_TRUE(torch::equal(*item.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{1, 4, 4}}, torch::kLong)));
  const auto pixels = *item.get<torch::Tensor>("pixel_values_videos");
  EXPECT_EQ(pixels.eq(-1).sum().item<int64_t>(), 2 * 3 * (56 * 56 - 29 * 30));
  EXPECT_EQ(pixels.eq(1).sum().item<int64_t>(), 2 * 3 * 29 * 30);
}

TEST(Glm5NextVideoProcessorTest, AppliesVaTokenCapOnlyInFixedFpsMode) {
  using SamplingMode = Glm4VVideoProcessor::SamplingMode;
  ModelArgs args = video_args();
  args.mm_video_patch_size(1).mm_video_merge_size(1);
  VideoMetadata metadata;
  metadata.fps = 2.0;
  metadata.duration = 1.0;
  metadata.total_num_frames = 2;
  metadata.frame_indices = torch::tensor({0, 1}, torch::kInt64);
  metadata.frames_preselected = true;
  const auto video = torch::zeros({2, 3, 180, 180}, torch::kUInt8);
  MMDataItem fixed(MMType::NONE);
  MMDataItem dynamic(MMType::NONE);
  MMDataItem stricter(MMType::NONE);

  ASSERT_TRUE(Glm4VVideoProcessor(args, SamplingMode::FIXED_FPS)
                  .process(video, metadata, fixed));
  ASSERT_TRUE(Glm4VVideoProcessor(args).process(video, metadata, dynamic));
  args.mm_video_max_tokens(10000);
  ASSERT_TRUE(Glm4VVideoProcessor(args, SamplingMode::FIXED_FPS)
                  .process(video, metadata, stricter));

  EXPECT_TRUE(torch::equal(*fixed.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{1, 173, 173}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(*dynamic.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{1, 180, 180}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(*stricter.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{1, 100, 100}}, torch::kInt64)));
}

TEST(Glm5NextVideoProcessorTest,
     KeepsDefaultDynamicSamplingAndFloorsTimestamps) {
  const ModelArgs args = video_args("glm4v");
  const Glm4VVideoProcessor processor(args);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 3.0;
  metadata.total_num_frames = 36;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({36, 3, 28, 28}, torch::kUInt8), metadata, item));
  EXPECT_EQ(item.get_metadata<VideoMetadata>()->frame_indices.numel(), 18);
  EXPECT_EQ((*item.get<torch::Tensor>("video_grid_thw"))[0][0].item<int64_t>(),
            9);

  MMData data;
  data.add(MMType::VIDEO, item);
  GLM4VPromptProcessor prompt_processor(args);
  std::string prompt = "<|begin_of_video|><|video|><|end_of_video|>";
  ASSERT_TRUE(prompt_processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "<|begin_of_video|>"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|end_of_video|>");
}

TEST(Glm5NextVideoProcessorTest, KeepsDefaultDynamicFrameCap) {
  const Glm4VVideoProcessor processor(video_args("glm4v"));
  VideoMetadata metadata;
  metadata.fps = 1.0;
  metadata.duration = 3000.0;
  metadata.total_num_frames = 3000;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({3000, 3, 28, 28}, torch::kUInt8), metadata, item));
  const auto sampled = item.get_metadata<VideoMetadata>();
  ASSERT_TRUE(sampled.has_value());
  EXPECT_EQ(sampled->frame_indices.numel(), 640);
  EXPECT_EQ(sampled->frame_indices[0].item<int64_t>(), 0);
  EXPECT_EQ(sampled->frame_indices[-1].item<int64_t>(), 2999);
}

}  // namespace
}  // namespace xllm
