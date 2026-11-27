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

#include "processors/glm4v_video_processor.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <unordered_set>

#include "processors/transforms.h"

namespace xllm {

namespace {

using Size = std::pair<int32_t, int32_t>;

std::optional<Size> smart_resize(int32_t num_frames,
                                 int32_t height,
                                 int32_t width,
                                 int32_t temporal_factor,
                                 int32_t factor = 28,
                                 int32_t min_pixels = 56 * 56,
                                 int32_t max_pixels = 14 * 14 * 4 * 1280) {
  if (height < factor || width < factor) {
    LOG(ERROR) << "Height or width must be larger than factor";
    return std::nullopt;
  }
  if (num_frames < temporal_factor) {
    LOG(ERROR) << "num_frames must be larger than temporal_factor, num_frames: "
               << num_frames << ", temporal_factor: " << temporal_factor;
    return std::nullopt;
  }

  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200, height: "
               << height << ", width: " << width;
    return std::nullopt;
  }
  int32_t t_bar = static_cast<int32_t>(std::rint(
                      num_frames / static_cast<double>(temporal_factor))) *
                  temporal_factor;
  int32_t h_bar =
      static_cast<int32_t>(std::rint(height / static_cast<double>(factor))) *
      factor;
  int32_t w_bar =
      static_cast<int32_t>(std::rint(width / static_cast<double>(factor))) *
      factor;

  if (t_bar * h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((num_frames * height * width) /
                            static_cast<double>(max_pixels));
    h_bar = static_cast<int32_t>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int32_t>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (t_bar * h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(min_pixels /
                            static_cast<double>(height * width * num_frames));
    h_bar = static_cast<int32_t>(
                std::ceil(height * beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int32_t>(
                std::ceil(width * beta / static_cast<double>(factor))) *
            factor;
  }

  return std::make_pair(h_bar, w_bar);
}

// token-budget smart_resize for video, mirroring HF
// Glm5NextVideoProcessor. See glm4v_image_processor.cpp for the algorithm.
std::optional<Size> smart_resize_tokens(int32_t num_frames,
                                        int32_t height,
                                        int32_t width,
                                        int32_t temporal_factor,
                                        int32_t factor,
                                        int32_t min_tokens,
                                        int32_t max_tokens,
                                        bool match_va) {
  if (height <= 0 || width <= 0 ||
      (!match_va && (height < factor || width < factor))) {
    LOG(ERROR) << "Height or width must be larger than factor";
    return std::nullopt;
  }
  if (num_frames <= 0 || (!match_va && num_frames < temporal_factor)) {
    LOG(ERROR) << "num_frames must be larger than temporal_factor, num_frames: "
               << num_frames << ", temporal_factor: " << temporal_factor;
    return std::nullopt;
  }

  const int64_t pixels_per_token =
      static_cast<int64_t>(temporal_factor) * factor * factor;
  int64_t min_pixels = static_cast<int64_t>(min_tokens) * pixels_per_token;
  int64_t max_pixels = static_cast<int64_t>(max_tokens) * pixels_per_token;

  auto align = [](int32_t value, int32_t f) {
    return ((value + f - 1) / f) * f;
  };

  auto fit_within_budget = [&](int32_t aligned_frames) -> Size {
    int32_t low = 1;
    int32_t high = height;
    int32_t best_h = factor;
    int32_t best_w = factor;
    while (low <= high) {
      int32_t content_h = (low + high) / 2;
      int32_t content_w =
          std::max(1,
                   static_cast<int32_t>(std::floor(static_cast<double>(width) *
                                                   content_h / height)));
      int32_t cand_h = align(content_h, factor);
      int32_t cand_w = align(content_w, factor);
      int64_t budget = static_cast<int64_t>(aligned_frames) * cand_h * cand_w;
      if (budget <= max_pixels) {
        best_h = cand_h;
        best_w = cand_w;
        low = content_h + 1;
      } else {
        high = content_h - 1;
      }
    }
    return std::make_pair(best_h, best_w);
  };

  int32_t aligned_frames = std::max(
      temporal_factor,
      static_cast<int32_t>(
          std::rint(num_frames / static_cast<double>(temporal_factor))) *
          temporal_factor);
  int32_t aligned_h = align(height, factor);
  int32_t aligned_w = align(width, factor);
  int64_t aligned_budget =
      static_cast<int64_t>(aligned_frames) * aligned_h * aligned_w;

  if (aligned_budget < min_pixels) {
    double scale =
        std::sqrt(static_cast<double>(min_pixels) /
                  (static_cast<double>(num_frames) * height * width));
    aligned_h = align(
        std::max(1, static_cast<int32_t>(std::ceil(height * scale))), factor);
    aligned_w = align(
        std::max(1, static_cast<int32_t>(std::ceil(width * scale))), factor);
    aligned_budget =
        static_cast<int64_t>(aligned_frames) * aligned_h * aligned_w;
  }

  if (aligned_budget > max_pixels) {
    if (match_va &&
        static_cast<int64_t>(aligned_frames) * factor * factor > max_pixels) {
      return std::nullopt;
    }
    auto fitted = fit_within_budget(aligned_frames);
    aligned_h = fitted.first;
    aligned_w = fitted.second;
  }

  return std::make_pair(aligned_h, aligned_w);
}

}  // namespace

Glm4VVideoProcessor::Glm4VVideoProcessor(const ModelArgs& args,
                                         SamplingMode sampling_mode)
    : VideoProcessor(args), sampling_mode_(sampling_mode) {
  video_mean_ = torch::tensor(args.mm_video_normalize_mean(),
                              torch::dtype(torch::kFloat32));
  video_std_ = torch::tensor(args.mm_video_normalize_std(),
                             torch::dtype(torch::kFloat32));

  video_min_pixels_ = args.mm_video_shortest_edge();
  video_max_pixels_ = args.mm_video_longest_edge();

  video_patch_size_ = args.mm_video_patch_size();
  video_temporal_patch_size_ = args.mm_video_temporal_patch_size();
  video_merge_size_ = args.mm_video_merge_size();
  video_min_tokens_ = args.mm_video_min_tokens();
  video_max_tokens_ = args.mm_video_max_tokens();
  if (args.mm_fps() > 0.0) {
    target_fps_ = args.mm_fps();
  }

  if (do_rescale_ && do_normalize_) {
    video_mean_.mul_(1.0 / rescale_factor_);
    video_std_.mul_(1.0 / rescale_factor_);
    do_rescale_ = false;
  }
}

namespace {

std::vector<int64_t> sample_fixed_video_indices(const VideoMetadata& metadata,
                                                double target_fps,
                                                int32_t temporal_patch_size) {
  std::vector<int64_t> indices;
  const int32_t total_frames = metadata.total_num_frames;
  if (total_frames <= 0 || !(metadata.fps > 0.0) ||
      !std::isfinite(metadata.fps) || !(target_fps > 0.0) ||
      !std::isfinite(target_fps) || temporal_patch_size <= 0) {
    return indices;
  }
  constexpr int32_t kMaxFrameCount = 2048;
  const double duration =
      metadata.duration > 0.0
          ? metadata.duration
          : std::rint((total_frames - 1) / metadata.fps) + 1;
  const int32_t count = std::max(
      1,
      static_cast<int32_t>(std::min(duration * target_fps,
                                    static_cast<double>(kMaxFrameCount))));
  indices.reserve(count);
  if (total_frames < count) {
    for (int64_t index = 0; index < count; ++index) {
      indices.push_back(index * total_frames / count);
    }
  } else {
    const double step = 1.0 / (temporal_patch_size * target_fps);
    const double seconds_per_frame = 1.0 / metadata.fps;
    double next_second = 0.0;
    for (int32_t index = 0; index < total_frames; ++index) {
      if (index * seconds_per_frame < next_second) {
        continue;
      }
      indices.push_back(index);
      next_second += step;
      if (next_second >= std::floor(duration) || indices.size() > count) {
        break;
      }
    }
    if (indices.size() != count) {
      const bool extend = !indices.empty() && indices.size() < count;
      const int64_t start = extend ? indices.front() : 0;
      const int64_t end = extend ? indices.back() : total_frames - 1;
      const double stride =
          count > 1 ? static_cast<double>(end - start) / (count - 1) : 0.0;
      indices.resize(count);
      for (int32_t index = 0; index < count; ++index) {
        const double offset = index * stride;
        indices[index] = static_cast<int64_t>(start + offset);
      }
      if (count > 1) {
        indices.back() = end;
      }
    }
  }
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}

}  // namespace

VideoFrameSelector Glm4VVideoProcessor::video_frame_selector() const {
  if (sampling_mode_ != SamplingMode::FIXED_FPS || !do_sample_frame_) {
    return {};
  }
  return {.select =
              [target_fps = target_fps_,
               temporal_patch_size =
                   video_temporal_patch_size_](const VideoMetadata& metadata) {
                return sample_fixed_video_indices(
                    metadata, target_fps, temporal_patch_size);
              },
          .apply_display_rotation = true,
          .match_opencv_rgb = true};
}

torch::Tensor Glm4VVideoProcessor::sample_frames(
    const VideoMetadata& metadata) const {
  if (sampling_mode_ == SamplingMode::FIXED_FPS) {
    return torch::tensor(sample_fixed_video_indices(
                             metadata, target_fps_, video_temporal_patch_size_),
                         torch::kLong);
  }
  const int32_t total_frames = metadata.total_num_frames;
  if (total_frames <= 0) {
    return torch::empty({0}, torch::dtype(torch::kLong));
  }

  if (metadata.fps <= 0.0) {
    LOG(FATAL) << "invalid metadata.fps <= 0";
  }

  const int32_t max_frame_idx = total_frames - 1;

  double duration = metadata.duration;
  if (duration <= 0.0) {
    duration =
        std::round(static_cast<double>(max_frame_idx) / metadata.fps) + 1.0;
  }

  constexpr double kMaxDuration = 2400.0;
  constexpr int32_t kMaxFrameCount = 640;
  const double effective_duration = std::min(duration, kMaxDuration);
  const double target_fps = effective_duration <= 30.0    ? 3.0
                            : effective_duration <= 300.0 ? 1.0
                                                          : 0.5;
  const int32_t extract_t = std::min(
      static_cast<int32_t>(effective_duration * target_fps *
                           static_cast<double>(video_temporal_patch_size_)),
      kMaxFrameCount);

  const double duration_per_frame = 1.0 / metadata.fps;
  std::vector<double> timestamps(total_frames);
  for (int32_t i = 0; i < total_frames; ++i) {
    timestamps[i] = static_cast<double>(i) * duration_per_frame;
  }
  const int32_t max_second = static_cast<int32_t>(duration);

  torch::Tensor frame_indices;

  if (total_frames < extract_t) {
    frame_indices = torch::linspace(
        0, total_frames - 1, extract_t, torch::dtype(torch::kLong));
  } else {
    std::vector<int64_t> tmp;
    tmp.reserve(static_cast<size_t>(total_frames));
    double current_second = 0.0;
    // Keep the temporal-patch-scaled walk and redistribute excess candidates
    // uniformly below.
    const double inv_fps =
        1.0 / (static_cast<double>(video_temporal_patch_size_) * target_fps);

    for (int32_t frame_index = 0; frame_index < total_frames; frame_index++) {
      if (timestamps[frame_index] >= current_second) {
        current_second += inv_fps;
        tmp.push_back(frame_index);
        if (current_second >= static_cast<double>(max_second)) {
          break;
        }
      }
    }
    frame_indices =
        torch::tensor(tmp, torch::TensorOptions().dtype(torch::kLong));
  }
  int64_t len = frame_indices.size(0);
  if (len < extract_t) {
    int64_t start, end;
    if (len == 0) {
      start = 0;
      end = std::max<int64_t>(total_frames - 1, 0);
    } else {
      start = frame_indices[0].item<int64_t>();
      end = frame_indices[len - 1].item<int64_t>();
    }
    frame_indices =
        torch::linspace(start, end, extract_t, torch::dtype(torch::kLong));
  } else if (len > extract_t) {
    frame_indices = torch::linspace(
        0, total_frames - 1, extract_t, torch::dtype(torch::kLong));
  }

  len = frame_indices.size(0);
  std::unordered_set<int64_t> seen;
  seen.reserve(static_cast<size_t>(len) * 2);
  std::vector<int64_t> uniq;
  uniq.reserve(static_cast<size_t>(len));

  for (int64_t i = 0; i < len; ++i) {
    auto idx = frame_indices[i].item<int64_t>();
    if (seen.insert(idx).second) {
      uniq.push_back(idx);
    }
  }

  if (!uniq.empty() && (uniq.size() & 1)) {
    uniq.push_back(uniq.back());
  }

  return torch::tensor(uniq, torch::TensorOptions().dtype(torch::kLong));
}

bool Glm4VVideoProcessor::process(const torch::Tensor& origin_video,
                                  const VideoMetadata& metadata,
                                  MMDataItem& output_item) const {
  torch::Tensor pixel_values;
  torch::Tensor thw;
  VideoMetadata output_metadata = metadata;
  if (!process_video(origin_video, output_metadata, pixel_values, thw)) {
    return false;
  }
  output_item = MMDataItem(
      MMType::VIDEO,
      MMDict{{"pixel_values_videos", pixel_values}, {"video_grid_thw", thw}},
      output_metadata);
  return true;
}

bool Glm4VVideoProcessor::process_video(const torch::Tensor& origin_video,
                                        VideoMetadata& metadata,
                                        torch::Tensor& pixel_values,
                                        torch::Tensor& thw) const {
  if (origin_video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }

  torch::Tensor indices;
  torch::Tensor video;
  const bool match_va = sampling_mode_ == SamplingMode::FIXED_FPS;
  if (metadata.frames_preselected) {
    if (!metadata.frame_indices.defined() ||
        metadata.frame_indices.numel() != origin_video.size(0) ||
        origin_video.size(0) == 0) {
      return false;
    }
    indices = metadata.frame_indices;
    video = origin_video;
  } else if (do_sample_frame_) {
    indices = sample_frames(metadata);
  } else {
    indices = torch::arange(0,
                            static_cast<int64_t>(origin_video.size(0)),
                            torch::TensorOptions().dtype(torch::kLong));
  }
  if (!video.defined()) {
    video = origin_video.index_select(/*dim=*/0, indices);
  }
  if (video.size(0) == 0) {
    return false;
  }
  if (match_va && indices.size(0) % 2 != 0) {
    indices = torch::cat({indices, indices.slice(0, -1)}, 0);
  }
  const int64_t sampled_total_frames = indices.size(0);

  metadata.frame_indices = indices;
  metadata.timestamps.clear();
  metadata.timestamps.reserve(static_cast<size_t>(sampled_total_frames));
  double fps_for_ts = (metadata.fps > 0.0) ? metadata.fps : 24.0;
  for (int64_t i = 0; i < sampled_total_frames; ++i) {
    int64_t frame_idx = metadata.frame_indices[i].item<int64_t>();
    metadata.timestamps.push_back(static_cast<double>(frame_idx) / fps_for_ts);
  }

  if (metadata.total_num_frames > 0 && metadata.fps > 0.0) {
    metadata.sampled_fps = double(sampled_total_frames) /
                           double(metadata.total_num_frames) * metadata.fps;
  } else {
    metadata.sampled_fps = fps_for_ts;
  }

  auto shape = video.sizes();
  auto channel = shape[1];
  auto time_len = shape[0];
  auto resized_height = shape[2];
  auto resized_width = shape[3];

  if (do_resize_) {
    std::optional<Size> size;
    if (video_min_tokens_ > 0 && video_max_tokens_ > 0) {
      constexpr int32_t kVaMaxVideoTokens = 30000;
      const int32_t max_tokens =
          match_va ? std::min(video_max_tokens_, kVaMaxVideoTokens)
                   : video_max_tokens_;
      size = smart_resize_tokens(time_len,
                                 resized_height,
                                 resized_width,
                                 video_temporal_patch_size_,
                                 video_patch_size_ * video_merge_size_,
                                 video_min_tokens_,
                                 max_tokens,
                                 match_va);
    } else {
      size = smart_resize(time_len,
                          resized_height,
                          resized_width,
                          video_temporal_patch_size_,
                          video_patch_size_ * video_merge_size_,
                          video_min_pixels_,
                          video_max_pixels_);
    }
    if (!size) {
      return false;
    }
    std::tie(resized_height, resized_width) = *size;
  }

  auto out_video = video;
  if (do_resize_) {
    int64_t content_height = resized_height;
    int64_t content_width = resized_width;
    if (match_va) {
      const int64_t factor = video_patch_size_ * video_merge_size_;
      const int64_t min_pixels =
          video_min_tokens_ > 0 && video_max_tokens_ > 0
              ? video_min_tokens_ * video_temporal_patch_size_ * factor * factor
              : video_min_pixels_;
      double scale = std::min(static_cast<double>(resized_height) / shape[2],
                              static_cast<double>(resized_width) / shape[3]);
      if (time_len * shape[2] * shape[3] >= min_pixels) {
        scale = std::min(1.0, scale);
      }
      content_height = std::clamp<int64_t>(shape[2] * scale, 1, resized_height);
      content_width = std::clamp<int64_t>(shape[3] * scale, 1, resized_width);
    }
    if (!match_va || content_height != shape[2] || content_width != shape[3]) {
      out_video = transforms::resize(
          out_video, {content_height, content_width}, resample_, true);
    }
    if (match_va &&
        (content_height != resized_height || content_width != resized_width)) {
      out_video = torch::constant_pad_nd(out_video,
                                         {0,
                                          resized_width - content_width,
                                          0,
                                          resized_height - content_height});
    }
  }
  if (do_normalize_) {
    out_video = transforms::normalize(
        out_video, video_mean_, video_std_, output_dtype_);
  }
  if (do_rescale_) {
    out_video = transforms::rescale(out_video, rescale_factor_);
  }

  auto pad_t = (video_temporal_patch_size_ -
                (out_video.size(0) % video_temporal_patch_size_)) %
               video_temporal_patch_size_;
  if (pad_t != 0) {
    auto last = out_video.index({out_video.size(0) - 1})
                    .unsqueeze(0)
                    .repeat({pad_t, 1, 1, 1});
    out_video = torch::cat({out_video, last}, 0);
  }

  shape = out_video.sizes();
  auto grid_h = resized_height / video_patch_size_;
  auto grid_w = resized_width / video_patch_size_;
  auto grid_t = shape[0] / video_temporal_patch_size_;

  out_video = out_video.contiguous();

  auto patches = out_video.view({grid_t,
                                 video_temporal_patch_size_,
                                 channel,
                                 grid_h / video_merge_size_,
                                 video_merge_size_,
                                 video_patch_size_,
                                 grid_w / video_merge_size_,
                                 video_merge_size_,
                                 video_patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape({grid_t * grid_h * grid_w,
                             channel * video_temporal_patch_size_ *
                                 video_patch_size_ * video_patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).reshape({-1, 3});

  return true;
}

}  // namespace xllm
