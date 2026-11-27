/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/multimodal/mm_type.h"
#include "provider.pb.h"

namespace xllm {

struct PreparedMultimodalInput;
Status decode_prepared_multimodal_input(const proto::MultimodalPrompt& prompt,
                                        PreparedMultimodalInput& output);

Status decode_media_input(const proto::MMInputData& source, MMContent& content);
bool valid_image_media_input(const proto::ImageMediaInput& image);
Status decode_image_media_input(const proto::ImageMediaInput& image,
                                MMContent& content,
                                std::string& payload);

class OpenCVImageDecoder {
 public:
  OpenCVImageDecoder() = default;
  ~OpenCVImageDecoder() = default;

  bool decode(std::string_view raw_data,
              torch::Tensor& t,
              bool preserve_alpha = false);
};

class OpenCVImageEncoder {
 public:
  OpenCVImageEncoder() = default;
  ~OpenCVImageEncoder() = default;

  bool encode(const torch::Tensor& t, std::string& raw_data);

 private:
  bool valid(const torch::Tensor& t);
};

enum class SampledVideoDecodeStatus {
  SUCCESS,
  DECODE_ERROR,
  VIDEO_LIMIT,
  BUSY
};

class FFmpegVideoDecoder {
 public:
  FFmpegVideoDecoder() = default;
  ~FFmpegVideoDecoder() = default;

  bool decode(std::string_view raw_data, torch::Tensor& t, VideoMetadata& meta);
  SampledVideoDecodeStatus decode_sampled(
      std::string_view raw_data,
      const VideoFrameSelector& select_frames,
      torch::Tensor& tensor,
      VideoMetadata& metadata,
      uint64_t max_video_bytes,
      uint64_t max_total_bytes);
};

class FFmpegAudioDecoder {
 public:
  FFmpegAudioDecoder() = default;
  ~FFmpegAudioDecoder() = default;

  bool decode(std::string_view raw_data,
              torch::Tensor& t,
              AudioMetadata& meta,
              int64_t target_sr = 16000,
              int32_t target_channels = 1);
};

class FFmpegVideoEncoder final {
 public:
  FFmpegVideoEncoder() = default;
  ~FFmpegVideoEncoder() = default;

  // Encode CPU RGB video [T, 3, H, W]: float32 in [0, 1] or contiguous
  // uint8 in [0, 255], with explicit container format ("mp4", "avi", etc.).
  bool encode(const torch::Tensor& video,
              double fps,
              const std::string& format,
              std::string& raw_data);

  bool encode(const torch::Tensor& video,
              const torch::Tensor& audio,
              double fps,
              int32_t sample_rate,
              const std::string& format,
              std::string& raw_data);
};
}  // namespace xllm
