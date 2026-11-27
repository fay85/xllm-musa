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
#include "core/framework/multimodal/mm_codec.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/cpu.h>
#include <libavutil/display.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include "core/framework/multimodal/mm_batch_data.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {
constexpr size_t kMaxImageBytes = 16 * 1024 * 1024;
constexpr size_t kMaxImageHeaderBytes = 16 * 1024;
constexpr size_t kDataUrlOverheadBytes = 256;
constexpr size_t kMaxHeaderCount = 32;

bool valid_remote_image(
    std::string_view url,
    size_t byte_count,
    const std::vector<std::pair<std::string_view, std::string_view>>& headers) {
  if (byte_count > kMaxImageBytes ||
      url.size() > ((kMaxImageBytes + 2) / 3) * 4 + kDataUrlOverheadBytes) {
    return false;
  }
  if (headers.size() > kMaxHeaderCount) {
    return false;
  }
  size_t header_bytes = 0;
  for (const auto& [key, value] : headers) {
    header_bytes += key.size() + value.size();
    if (key.empty() || key.find_first_of("\r\n:") != std::string_view::npos ||
        value.find_first_of("\r\n") != std::string_view::npos) {
      return false;
    }
  }
  if (header_bytes > kMaxImageHeaderBytes) {
    return false;
  }
  if (byte_count == 0 && url.substr(0, 7) != "http://" &&
      url.substr(0, 8) != "https://" && url.substr(0, 11) != "data:image/") {
    return false;
  }
  return true;
}

}  // namespace

Status decode_media_input(const proto::MMInputData& source,
                          MMContent& content) {
  const std::string& type = source.type();
  ImageURL url;
  if (type == "image_url" && source.has_image_url()) {
    url.url = source.image_url().url();
    for (const auto& [key, value] : source.image_url().headers()) {
      url.headers.emplace(key, value);
    }
  } else if (type == "video_url" && source.has_video_url()) {
    url.url = source.video_url().url();
    for (const auto& [key, value] : source.video_url().headers()) {
      url.headers.emplace(key, value);
    }
  } else if (type == "audio_url" && source.has_audio_url()) {
    url.url = source.audio_url().url();
    for (const auto& [key, value] : source.audio_url().headers()) {
      url.headers.emplace(key, value);
    }
  } else {
    return {StatusCode::INVALID_ARGUMENT, "Missing Native media source"};
  }
  ImageURL validated = url;
  const std::string prefix = type == "video_url"   ? "data:video/"
                             : type == "audio_url" ? "data:audio/"
                                                   : "data:image/";
  if (url.url.rfind("data:", 0) == 0) {
    if (url.url.rfind(prefix, 0) != 0) {
      return {StatusCode::INVALID_ARGUMENT, "Mismatched media data URL"};
    }
    validated.url.replace(0, prefix.size(), "data:image/");
  }
  std::vector<std::pair<std::string_view, std::string_view>> headers;
  headers.reserve(validated.headers.size());
  for (const auto& [key, value] : validated.headers) {
    headers.emplace_back(key, value);
  }
  if (!valid_remote_image(validated.url, 0, headers)) {
    return {StatusCode::INVALID_ARGUMENT, "Invalid bounded media source"};
  }
  content.type = type;
  content.image_url = url;
  content.video_url = {url.url, url.headers};
  content.audio_url = {url.url, url.headers};
  return {};
}

bool valid_image_media_input(const proto::ImageMediaInput& image) {
  if (image.headers_size() > kMaxHeaderCount ||
      image.content_case() == proto::ImageMediaInput::CONTENT_NOT_SET ||
      (image.has_data() && image.data().empty())) {
    return false;
  }
  std::vector<std::pair<std::string_view, std::string_view>> headers;
  headers.reserve(image.headers_size());
  for (const auto& [key, value] : image.headers()) {
    headers.emplace_back(key, value);
  }
  return valid_remote_image(image.url(), image.data().size(), headers);
}

Status decode_image_media_input(const proto::ImageMediaInput& image,
                                MMContent& content,
                                std::string& payload) {
  if (!valid_image_media_input(image)) {
    return {StatusCode::INVALID_ARGUMENT, "Invalid image input"};
  }
  content.type = "image_url";
  auto& url = content.image_url;
  url.url = image.url();
  if (image.has_data()) {
    url.url = "data:image/jpeg;binary," + std::to_string(payload.size()) + "," +
              std::to_string(image.data().size());
    payload.append(image.data());
  }
  url.headers.clear();
  url.headers.reserve(image.headers_size());
  for (const auto& [key, value] : image.headers()) {
    url.headers.emplace(key, value);
  }
  return {};
}

Status decode_prepared_multimodal_input(const proto::MultimodalPrompt& prompt,
                                        PreparedMultimodalInput& output) {
  PreparedMultimodalInput input;
  input.token_ids.assign(prompt.token_ids().begin(), prompt.token_ids().end());
  proto::MMDataEntry metadata = prompt.mm_data();
  for (auto& item : *metadata.mutable_items()) {
    item.mutable_state()->clear_mm_token_mask();
  }
  auto decoded = proto_to_mmdata_entry(metadata);
  if (!decoded) {
    return {StatusCode::INVALID_ARGUMENT, "Invalid prepared media metadata"};
  }
  input.metadata = std::move(*decoded);
  if (!prompt.prompt_text().empty()) {
    input.prompt = prompt.prompt_text();
  }
  input.media.reserve(prompt.sources_size());
  for (const auto& source : prompt.sources()) {
    MMContent content("text", "");
    Status status = decode_media_input(source.media(), content);
    if (!status.ok()) {
      return status;
    }
    input.media.emplace_back(std::move(content));
  }
  output = std::move(input);
  return {};
}

namespace {

struct MemCtx {
  const uint8_t* mem_ptr;
  int64_t size;
  int64_t offset;
};

struct Reader {
  // AVIO read callback
  static int32_t read(void* opaque, uint8_t* buf, int32_t buf_size) {
    auto* mc = static_cast<MemCtx*>(opaque);
    if (mc->offset < 0) {
      return AVERROR(EINVAL);
    }
    int64_t remain = mc->size - mc->offset;
    int64_t n = std::min(remain, static_cast<int64_t>(buf_size));
    if (n <= 0) {
      return AVERROR_EOF;
    }
    std::memcpy(buf, mc->mem_ptr + mc->offset, static_cast<size_t>(n));
    mc->offset += n;
    return static_cast<int32_t>(n);
  }

  // AVIO seek callback
  static int64_t seek(void* opaque, int64_t offset, int32_t whence) {
    auto* mc = static_cast<MemCtx*>(opaque);

    if (whence == AVSEEK_SIZE) {
      return mc->size;
    }

    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos = mc->offset + offset;
        break;
      case SEEK_END:
        pos = mc->size + offset;
        break;
      default:
        return AVERROR(EINVAL);
    }

    if (pos < 0 || pos > mc->size) {
      return AVERROR(EINVAL);
    }

    mc->offset = pos;
    return pos;
  }
};

// Downstream multimodal preprocess expects a 3-channel RGB image. For BGRA
// input, blend alpha onto a white background instead of dropping alpha
// directly, so transparent regions keep visually stable colors.
void blend_bgra_to_rgb_with_white_background(const cv::Mat& image,
                                             cv::Mat& rgb_image) {
  rgb_image = cv::Mat(image.rows, image.cols, CV_8UC3);

  for (int32_t row = 0; row < image.rows; ++row) {
    const cv::Vec4b* src_row = image.ptr<cv::Vec4b>(row);
    cv::Vec3b* dst_row = rgb_image.ptr<cv::Vec3b>(row);

    for (int32_t col = 0; col < image.cols; ++col) {
      const float alpha = src_row[col][3] / 255.0f;
      const float blue = src_row[col][0];
      const float green = src_row[col][1];
      const float red = src_row[col][2];

      dst_row[col][0] = static_cast<uint8_t>(
          std::round(red * alpha + 255.0f * (1.0f - alpha)));
      dst_row[col][1] = static_cast<uint8_t>(
          std::round(green * alpha + 255.0f * (1.0f - alpha)));
      dst_row[col][2] = static_cast<uint8_t>(
          std::round(blue * alpha + 255.0f * (1.0f - alpha)));
    }
  }
}

// OpenCV decodes different image formats into different channel layouts:
// 1 channel -> GRAY, 3 channels -> BGR, 4 channels -> BGRA. Normalize them
// here so the decoder path always returns a 3-channel RGB image.
void convert_decoded_image_to_rgb(const cv::Mat& image, cv::Mat& rgb_image) {
  const int32_t channels = image.channels();

  if (channels == 4) {
    blend_bgra_to_rgb_with_white_background(image, rgb_image);
  } else if (channels == 3) {
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
  } else if (channels == 1) {
    cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
  } else {
    LOG(FATAL) << "unsupported channel count: " << channels;
  }
}

bool use_neon_rgba_for_video(const AVFrame& frame) {
#if defined(__aarch64__) && LIBSWSCALE_VERSION_MAJOR == 7
  return (av_get_cpu_flags() & AV_CPU_FLAG_NEON) &&
         frame.format == AV_PIX_FMT_YUV420P &&
         frame.color_range != AVCOL_RANGE_JPEG && frame.width % 16 == 0 &&
         frame.height % 2 == 0;
#else
  return false;
#endif
}

struct VideoColorConversion {
  int32_t width;
  int32_t height;
  AVPixelFormat input;
  AVPixelFormat output;
  AVColorSpace colorspace;
  AVColorRange range;
  AVChromaLocation chroma;

  bool operator==(const VideoColorConversion&) const = default;
};

SwsContext* make_opencv_color_context(const VideoColorConversion& color) {
  const auto* desc = av_pix_fmt_desc_get(color.input);
  if (!desc) {
    return nullptr;
  }
  auto* context = sws_alloc_context();
  if (!context) {
    return nullptr;
  }
  const auto chroma = color.chroma == AVCHROMA_LOC_UNSPECIFIED
                          ? AVCHROMA_LOC_CENTER
                          : color.chroma;
  int32_t chroma_x = 0;
  int32_t chroma_y = 0;
  if (av_chroma_location_enum_to_pos(&chroma_x, &chroma_y, chroma) == 0) {
    av_opt_set_int(context,
                   "src_h_chr_pos",
                   chroma_x * ((1 << desc->log2_chroma_w) - 1),
                   0);
    av_opt_set_int(context,
                   "src_v_chr_pos",
                   chroma_y * ((1 << desc->log2_chroma_h) - 1),
                   0);
  }
  av_opt_set_int(context, "srcw", color.width, 0);
  av_opt_set_int(context, "srch", color.height, 0);
  av_opt_set_int(context, "dstw", color.width, 0);
  av_opt_set_int(context, "dsth", color.height, 0);
  av_opt_set_pixel_fmt(context, "src_format", color.input, 0);
  av_opt_set_pixel_fmt(context, "dst_format", color.output, 0);
  const bool full_range = color.range == AVCOL_RANGE_JPEG;
  av_opt_set_int(context, "src_range", full_range, 0);
  av_opt_set_int(context, "sws_flags", SWS_BICUBIC, 0);
  if (sws_init_context(context, nullptr, nullptr) < 0 ||
      sws_setColorspaceDetails(context,
                               sws_getCoefficients(color.colorspace),
                               full_range,
                               sws_getCoefficients(SWS_CS_DEFAULT),
                               1,
                               0,
                               1 << 16,
                               1 << 16) < 0) {
    sws_freeContext(context);
    return nullptr;
  }
  return context;
}

std::optional<cv::RotateFlags> get_video_rotation(
    const AVCodecParameters& params) {
  const AVPacketSideData* side_data =
      av_packet_side_data_get(params.coded_side_data,
                              params.nb_coded_side_data,
                              AV_PKT_DATA_DISPLAYMATRIX);
  std::array<int32_t, 9> matrix{};
  if (!side_data || side_data->size < sizeof(matrix)) {
    return std::nullopt;
  }
  std::memcpy(matrix.data(), side_data->data, sizeof(matrix));
  const double angle = -av_display_rotation_get(matrix.data());
  if (!std::isfinite(angle)) {
    return std::nullopt;
  }
  switch (std::lround(angle)) {
    case 90:
      return cv::ROTATE_90_CLOCKWISE;
    case -90:
      return cv::ROTATE_90_COUNTERCLOCKWISE;
    case -180:
    case 180:
      return cv::ROTATE_180;
    default:
      return std::nullopt;
  }
}

std::atomic_flag parallel_video_decoder_in_use{};

}  // namespace

class MemoryMediaReader {
 public:
  MemoryMediaReader(const uint8_t* data, size_t size) {
    mc_.mem_ptr = data;
    if (size > static_cast<size_t>(INT64_MAX)) {
      LOG(FATAL) << "MemCtx size too large";
    }
    mc_.size = static_cast<int64_t>(size);
    mc_.offset = 0;
  }

  ~MemoryMediaReader() {
    if (frm_) {
      av_frame_free(&frm_);
    }
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (parallel_video_decode_) {
      parallel_video_decoder_in_use.clear(std::memory_order_relaxed);
    }
    if (fmt_ctx_) {
      avformat_close_input(&fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    } else if (avio_buf_) {
      av_freep(reinterpret_cast<void**>(&avio_buf_));
    }
  }

  bool init(AVMediaType type,
            int32_t decoder_threads = 0,
            bool prefer_parallel_video = false) {
    fmt_ctx_ = avformat_alloc_context();
    if (!fmt_ctx_) {
      return false;
    }
    constexpr int32_t avio_buf_sz = 1 << 16;
    avio_buf_ =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(avio_buf_sz)));
    if (!avio_buf_) {
      return false;
    }

    avio_ctx_ = avio_alloc_context(
        avio_buf_, avio_buf_sz, 0, &mc_, &Reader::read, nullptr, &Reader::seek);
    if (!avio_ctx_) {
      return false;
    }
    avio_buf_ = nullptr;

    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    if (avformat_open_input(&fmt_ctx_, nullptr, nullptr, nullptr) < 0) {
      return false;
    }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
      return false;
    }

    for (uint32_t i = 0; i < fmt_ctx_->nb_streams; ++i) {
      AVStream* stream = fmt_ctx_->streams[i];
      if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&
          !(stream->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
        has_video_stream_ = true;
        break;
      }
    }

    stream_index_ = av_find_best_stream(fmt_ctx_, type, -1, -1, nullptr, 0);
    if (stream_index_ < 0) {
      return false;
    }

    AVStream* st = fmt_ctx_->streams[stream_index_];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) {
      return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }

    constexpr int32_t kMaxVideoDecoderThreads = 16;
    constexpr int64_t kMinParallelVideoPixels = 1280 * 720;
    constexpr int64_t kMinParallelVideoFrames = 120;
    if (prefer_parallel_video && st->nb_frames >= kMinParallelVideoFrames &&
        static_cast<int64_t>(st->codecpar->width) * st->codecpar->height >=
            kMinParallelVideoPixels &&
        (codec->capabilities & AV_CODEC_CAP_FRAME_THREADS)) {
      const int32_t thread_limit =
          std::min<int32_t>(av_cpu_count(), kMaxVideoDecoderThreads);
      if (thread_limit > decoder_threads) {
        parallel_video_decode_ = !parallel_video_decoder_in_use.test_and_set(
            std::memory_order_relaxed);
        if (parallel_video_decode_) {
          decoder_threads = thread_limit;
        }
      }
    }
    if (decoder_threads > 0) {
      codec_ctx_->thread_count = decoder_threads;
    }
    if (avcodec_parameters_to_context(codec_ctx_, st->codecpar) < 0 ||
        avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
      return false;
    }

    pkt_ = av_packet_alloc();
    frm_ = av_frame_alloc();
    if (!pkt_ || !frm_) {
      return false;
    }

    return true;
  }

  bool decode() {
    CHECK(fmt_ctx_ && codec_ctx_ && pkt_ && frm_) << "ffmpeg init failed";
    CHECK_GE(stream_index_, 0) << "stream index not found";

    // read packets, send to decoder, pull frames
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
      if (pkt_->stream_index == stream_index_) {
        if (avcodec_send_packet(codec_ctx_, pkt_) == 0) {
          while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
            // handle_frame: video->push RGB frame, audio->append PCM
            if (!handle_frame(frm_)) {
              av_packet_unref(pkt_);
              return false;
            }
          }
        }
      }
      av_packet_unref(pkt_);
    }

    // flush decoder at end of stream
    avcodec_send_packet(codec_ctx_, nullptr);
    while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
      if (!handle_frame(frm_)) {
        return false;
      }
    }

    return true;
  }

  // video->RGB tensor, audio->PCM samples
  virtual bool handle_frame(AVFrame* f) = 0;

 protected:
  AVFormatContext* fmt_ctx_ = nullptr;
  uint8_t* avio_buf_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frm_ = nullptr;
  MemCtx mc_{nullptr, 0, 0};
  int32_t stream_index_ = -1;
  bool has_video_stream_ = false;
  bool parallel_video_decode_ = false;
};

namespace {

std::atomic<uint64_t> sampled_video_bytes{0};

class SampledVideoReservation final {
 public:
  ~SampledVideoReservation() {
    sampled_video_bytes.fetch_sub(bytes_, std::memory_order_relaxed);
  }

  bool acquire(uint64_t bytes, uint64_t budget) {
    if (bytes > budget) {
      return false;
    }
    uint64_t used = sampled_video_bytes.load(std::memory_order_relaxed);
    while (used <= budget - bytes) {
      if (sampled_video_bytes.compare_exchange_weak(
              used, used + bytes, std::memory_order_relaxed)) {
        bytes_ = bytes;
        return true;
      }
    }
    return false;
  }

 private:
  uint64_t bytes_ = 0;
};

}  // namespace

class MemoryVideoReader : public MemoryMediaReader {
 public:
  MemoryVideoReader(const uint8_t* data, size_t size)
      : MemoryMediaReader(data, size) {}

  ~MemoryVideoReader() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (rgb_frame_) {
      av_frame_free(&rgb_frame_);
    }
  }

  bool init(VideoMetadata& metadata,
            int32_t decoder_threads = 0,
            bool prefer_parallel_video = false) {
    if (!MemoryMediaReader::init(
            AVMEDIA_TYPE_VIDEO, decoder_threads, prefer_parallel_video)) {
      return false;
    }

    // init VideoMetadata
    AVStream* st = fmt_ctx_->streams[stream_index_];
    AVRational r =
        st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    metadata.fps = (r.num && r.den) ? av_q2d(r) : 0.0;
    metadata.total_num_frames = 0;
    metadata.duration = 0.0;
    return true;
  }

  int32_t frame_count_hint() const {
    const AVStream* stream = fmt_ctx_->streams[stream_index_];
    if (stream->nb_frames > 0 && stream->nb_frames <= INT32_MAX) {
      return static_cast<int32_t>(stream->nb_frames);
    }
    return 0;
  }

  int32_t count_video_packets() {
    int32_t count = 0;
    int32_t result = 0;
    while ((result = av_read_frame(fmt_ctx_, pkt_)) >= 0) {
      const bool is_video = pkt_->stream_index == stream_index_;
      av_packet_unref(pkt_);
      if (!is_video) {
        continue;
      }
      if (count == INT32_MAX) {
        return 0;
      }
      ++count;
    }
    return result == AVERROR_EOF ? count : 0;
  }

  int32_t decoded_count() const { return scan_complete_ ? decoded_count_ : 0; }

  bool scan() {
    selective_ = true;
    scan_complete_ = decode();
    return scan_complete_;
  }

  bool read_sampled(const VideoFrameSelector& select_frames,
                    torch::Tensor& tensor,
                    VideoMetadata& metadata,
                    uint64_t max_video_bytes,
                    uint64_t max_total_bytes) {
    selective_ = true;
    if (select_frames.apply_display_rotation) {
      rotation_ =
          get_video_rotation(*fmt_ctx_->streams[stream_index_]->codecpar);
    }
    planar_selected_frames_ = parallel_video_decode_ && !rotation_;
    match_opencv_rgb_ = select_frames.match_opencv_rgb;
    max_video_bytes_ = std::min(max_video_bytes, max_total_bytes);
    max_total_bytes_ = max_total_bytes;
    selected_ = select_frames(metadata);
    if (!std::is_sorted(selected_.begin(), selected_.end()) ||
        (!selected_.empty() &&
         (selected_.front() < 0 ||
          selected_.back() >= metadata.total_num_frames))) {
      return false;
    }
    selected_.erase(std::unique(selected_.begin(), selected_.end()),
                    selected_.end());
    if (!scan()) {
      return false;
    }
    if (decoded_count_ == metadata.total_num_frames && !selected_.empty() &&
        next_selected_ == selected_.size()) {
      tensor = planar_selected_frames_ ? selected_frames_
                                       : selected_frames_.permute({0, 3, 1, 2});
      metadata.frame_indices = torch::tensor(selected_, torch::kInt64);
      metadata.frames_preselected = true;
      return true;
    }
    return false;
  }

  SampledVideoDecodeStatus sampled_failure() const { return sampled_failure_; }

  bool read(torch::Tensor& tensor, VideoMetadata& metadata) {
    CHECK(frames_.empty()) << "frames is not cleared before read";

    if (!decode()) {
      return false;
    }
    if (frames_.empty()) {
      return false;
    }

    tensor = torch::stack(frames_);  // [T,C,H,W]
    metadata.total_num_frames = static_cast<int32_t>(frames_.size());
    metadata.duration =
        (metadata.fps > 0.0)
            ? static_cast<double>(metadata.total_num_frames) / metadata.fps
            : 0.0;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    if (!frame_size_) {
      frame_size_.emplace(f->width, f->height);
    } else if (frame_size_->width != f->width ||
               frame_size_->height != f->height) {
      const size_t frame_index =
          selective_ ? static_cast<size_t>(decoded_count_) : frames_.size();
      LOG(ERROR) << "Video frame " << frame_index + 1 << " has dimensions "
                 << f->width << "x" << f->height << ", expected "
                 << frame_size_->width << "x" << frame_size_->height
                 << " (width x height). All video frames must have the same "
                    "dimensions.";
      return false;
    }

    if (selective_) {
      if (decoded_count_ == INT32_MAX) {
        return false;
      }
      const int32_t frame_index = decoded_count_++;
      if (next_selected_ == selected_.size() ||
          selected_[next_selected_] != frame_index) {
        return true;
      }
      if (!prepare_selected_frames(f->height, f->width)) {
        return false;
      }
    }
    const bool use_rgba = match_opencv_rgb_ && use_neon_rgba_for_video(*f);
    const AVPixelFormat output_format = use_rgba            ? AV_PIX_FMT_RGBA
                                        : match_opencv_rgb_ ? AV_PIX_FMT_BGR24
                                                            : AV_PIX_FMT_RGB24;
    const auto input_format = static_cast<AVPixelFormat>(f->format);
    if (match_opencv_rgb_) {
      const VideoColorConversion color{f->width,
                                       f->height,
                                       input_format,
                                       output_format,
                                       f->colorspace,
                                       f->color_range,
                                       f->chroma_location};
      if (!color_conversion_ || *color_conversion_ != color) {
        SwsContext* context = make_opencv_color_context(color);
        if (!context) {
          return false;
        }
        sws_freeContext(sws_ctx_);
        sws_ctx_ = context;
        color_conversion_ = color;
      }
    } else {
      sws_ctx_ = sws_getCachedContext(sws_ctx_,
                                      f->width,
                                      f->height,
                                      input_format,
                                      f->width,
                                      f->height,
                                      output_format,
                                      SWS_BILINEAR,
                                      nullptr,
                                      nullptr,
                                      nullptr);
    }
    if (!sws_ctx_) {
      return false;
    }

    // use an FFmpeg-allocated frame so sws_scale writes into a buffer with the
    // correct padded linesize
    if (!rgb_frame_) {
      rgb_frame_ = av_frame_alloc();
      if (!rgb_frame_) {
        return false;
      }
    }

    // (re)allocate the RGB buffer when input changes
    if (rgb_frame_->width != f->width || rgb_frame_->height != f->height ||
        rgb_frame_->format != output_format || !rgb_frame_->data[0]) {
      av_frame_unref(rgb_frame_);
      rgb_frame_->format = output_format;
      rgb_frame_->width = f->width;
      rgb_frame_->height = f->height;
      if (av_frame_get_buffer(rgb_frame_, 0) < 0) {
        return false;
      }
    }
    if (av_frame_make_writable(rgb_frame_) < 0) {
      return false;
    }

    if (sws_scale(sws_ctx_,
                  f->data,
                  f->linesize,
                  0,
                  f->height,
                  rgb_frame_->data,
                  rgb_frame_->linesize) != f->height) {
      return false;
    }

    if (selective_) {
      return store_selected_frame(output_format);
    }

    // build CHW uint8 tensor
    const int64_t H = f->height;
    const int64_t W = f->width;
    const int64_t src_ls = rgb_frame_->linesize[0];
    constexpr int64_t pixel_stride = 3;

    auto rgb = torch::from_blob(rgb_frame_->data[0],
                                {3, H, W},  // [C,H,W]
                                {1, src_ls, pixel_stride},
                                torch::TensorOptions().dtype(torch::kUInt8))
                   .contiguous();

    frames_.emplace_back(rgb.clone());
    return true;
  }

 private:
  bool prepare_selected_frames(int32_t height, int32_t width) {
    const bool swap_dimensions = rotation_ && *rotation_ != cv::ROTATE_180;
    const int32_t output_width = swap_dimensions ? height : width;
    const int32_t output_height = swap_dimensions ? width : height;
    if (selected_frames_.defined()) {
      return selected_frames_.size(planar_selected_frames_ ? 2 : 1) ==
                 output_height &&
             selected_frames_.size(planar_selected_frames_ ? 3 : 2) ==
                 output_width;
    }
    if (height <= 0 || width <= 0) {
      return false;
    }
    const uint64_t frame_bytes = static_cast<uint64_t>(height) * width * 3;
    if (selected_.size() > max_video_bytes_ / frame_bytes) {
      sampled_failure_ = SampledVideoDecodeStatus::VIDEO_LIMIT;
      return false;
    }
    auto reservation = std::make_shared<SampledVideoReservation>();
    if (!reservation->acquire(selected_.size() * frame_bytes,
                              max_total_bytes_)) {
      sampled_failure_ = SampledVideoDecodeStatus::BUSY;
      return false;
    }
    const int64_t count = static_cast<int64_t>(selected_.size());
    const std::array<int64_t, 4> shape =
        planar_selected_frames_
            ? std::array<int64_t, 4>{count, 3, output_height, output_width}
            : std::array<int64_t, 4>{count, output_height, output_width, 3};
    auto owned = torch::empty(shape, torch::kUInt8);
    auto* data = owned.data_ptr<uint8_t>();
    selected_frames_ = torch::from_blob(
        data,
        shape,
        [owned = std::move(owned),
         reservation = std::move(reservation)](void*) mutable {
          owned.reset();
          reservation.reset();
        },
        torch::kUInt8);
    return true;
  }

  bool store_selected_frame(AVPixelFormat format) {
    const int32_t width = rgb_frame_->width;
    const int32_t height = rgb_frame_->height;
    const int64_t plane_bytes = static_cast<int64_t>(width) * height;
    uint8_t* destination =
        selected_frames_.data_ptr<uint8_t>() + next_selected_ * plane_bytes * 3;
    const bool use_rgba = format == AV_PIX_FMT_RGBA;
    const bool convert = use_rgba || format == AV_PIX_FMT_BGR24;
    const auto conversion = use_rgba ? cv::COLOR_RGBA2RGB : cv::COLOR_BGR2RGB;
    cv::Mat source(height,
                   width,
                   use_rgba ? CV_8UC4 : CV_8UC3,
                   rgb_frame_->data[0],
                   rgb_frame_->linesize[0]);
    if (planar_selected_frames_) {
      const bool bgr = format == AV_PIX_FMT_BGR24;
      std::array<cv::Mat, 4> planes = {
          cv::Mat(height,
                  width,
                  CV_8UC1,
                  destination + (bgr ? 2 * plane_bytes : 0)),
          cv::Mat(height, width, CV_8UC1, destination + plane_bytes),
          cv::Mat(height,
                  width,
                  CV_8UC1,
                  destination + (bgr ? 0 : 2 * plane_bytes)),
          planar_alpha_};
      cv::split(source, planes.data());
      planar_alpha_ = std::move(planes[3]);
    } else if (rotation_) {
      if (convert) {
        cv::cvtColor(source, rotation_buffer_, conversion);
        source = rotation_buffer_;
      }
      cv::Mat rgb(selected_frames_.size(1),
                  selected_frames_.size(2),
                  CV_8UC3,
                  destination);
      cv::rotate(source, rgb, *rotation_);
    } else if (convert) {
      cv::Mat rgb(height, width, CV_8UC3, destination);
      cv::cvtColor(source, rgb, conversion);
    } else {
      const int64_t row_bytes = static_cast<int64_t>(width) * 3;
      for (int32_t row = 0; row < height; ++row) {
        std::memcpy(destination + row * row_bytes,
                    rgb_frame_->data[0] +
                        static_cast<int64_t>(row) * rgb_frame_->linesize[0],
                    row_bytes);
      }
    }
    ++next_selected_;
    return true;
  }

  std::optional<cv::Size> frame_size_;
  SwsContext* sws_ctx_ = nullptr;
  std::optional<VideoColorConversion> color_conversion_;
  AVFrame* rgb_frame_ = nullptr;
  std::vector<torch::Tensor> frames_;
  bool selective_ = false;
  bool match_opencv_rgb_ = false;
  bool planar_selected_frames_ = false;
  uint64_t max_video_bytes_ = 0;
  uint64_t max_total_bytes_ = 0;
  SampledVideoDecodeStatus sampled_failure_ =
      SampledVideoDecodeStatus::DECODE_ERROR;
  bool scan_complete_ = false;
  int32_t decoded_count_ = 0;
  size_t next_selected_ = 0;
  std::vector<int64_t> selected_;
  torch::Tensor selected_frames_;
  std::optional<cv::RotateFlags> rotation_;
  cv::Mat rotation_buffer_;
  cv::Mat planar_alpha_;
};

class MemoryAudioReader : public MemoryMediaReader {
 public:
  MemoryAudioReader(const uint8_t* data,
                    size_t size,
                    int64_t target_sr = 16000,
                    int32_t target_channels = 1)
      : MemoryMediaReader(data, size) {
    target_sr_ = target_sr;
    target_ch_ = target_channels;
  }

  ~MemoryAudioReader() {
    if (swr_ctx_) {
      swr_free(&swr_ctx_);
    }
  }

  bool init(AudioMetadata& metadata) {
    if (!MemoryMediaReader::init(AVMEDIA_TYPE_AUDIO)) {
      return false;
    }

    AVStream* st = fmt_ctx_->streams[stream_index_];
    codec_ctx_->pkt_timebase = st->time_base;

    // setup resampler
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
      return false;
    }

    AVChannelLayout in_layout;
    if (av_channel_layout_copy(&in_layout, &codec_ctx_->ch_layout) < 0) {
      return false;
    }

    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, target_ch_);

    if (swr_alloc_set_opts2(&swr_ctx_,
                            &out_layout,
                            AV_SAMPLE_FMT_FLT,
                            target_sr_,
                            &in_layout,
                            codec_ctx_->sample_fmt,
                            codec_ctx_->sample_rate,
                            0,
                            nullptr) < 0) {
      av_channel_layout_uninit(&out_layout);
      av_channel_layout_uninit(&in_layout);
      return false;
    }

    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);

    // Preserve explicit averaging for standalone audio; use FFmpeg for video
    // audio.
    int32_t in_ch = codec_ctx_->ch_layout.nb_channels;
    if (target_ch_ == 1 && in_ch == 2 && !has_video_stream_) {
      constexpr double matrix[2] = {0.5, 0.5};
      if (swr_set_matrix(swr_ctx_, matrix, in_ch) < 0) {
        return false;
      }
    }

    if (swr_init(swr_ctx_) < 0) {
      return false;
    }

    // init AudioMetadata
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    metadata.duration = 0.0;
    return true;
  }

  bool read(torch::Tensor& tensor, AudioMetadata& metadata) {
    CHECK(swr_ctx_) << "SwrContext is null";
    CHECK(pcm_.empty()) << "PCM buffer is not cleared before read";

    if (!decode()) {
      return false;
    }

    // flush resampler buffered samples after decode
    while (true) {
      if (resample_to_pcm(nullptr, 0) <= 0) {
        break;
      }
    }

    if (pcm_.empty()) {
      return false;
    }

    // Keep decoded audio uniformly shaped as [channels, samples].
    CHECK_EQ(pcm_.size() % static_cast<size_t>(target_ch_), 0);
    const int64_t sample_count =
        static_cast<int64_t>(pcm_.size() / static_cast<size_t>(target_ch_));
    tensor = torch::from_blob(pcm_.data(),
                              {sample_count, target_ch_},
                              torch::TensorOptions().dtype(torch::kFloat32))
                 .permute({1, 0})
                 .clone()
                 .contiguous();
    metadata.duration = static_cast<double>(sample_count) / target_sr_;
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    return resample_to_pcm((const uint8_t**)f->extended_data, f->nb_samples) >=
           0;
  }

  int32_t resample_to_pcm(const uint8_t** in_data, int32_t nb_samples) {
    int32_t out_nb = swr_get_out_samples(swr_ctx_, nb_samples);
    if (out_nb < 0) {
      return out_nb;
    }
    if (out_nb == 0) {
      return 0;
    }

    std::vector<float> out_buf(static_cast<size_t>(out_nb) *
                               static_cast<size_t>(target_ch_));
    uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(out_buf.data())};

    // convert input frame samples to target format
    int32_t converted =
        swr_convert(swr_ctx_, out_data, out_nb, in_data, nb_samples);
    if (converted < 0) {
      return converted;
    }
    if (converted == 0) {
      return 0;
    }

    // append converted samples to pcm buffer
    const int64_t n = static_cast<int64_t>(converted * target_ch_);
    pcm_.reserve(pcm_.size() + static_cast<size_t>(n));
    pcm_.insert(pcm_.end(), out_buf.data(), out_buf.data() + n);
    return converted;
  }

 private:
  SwrContext* swr_ctx_ = nullptr;
  int32_t target_sr_ = 16000;
  int32_t target_ch_ = 1;
  std::vector<float> pcm_;
};

bool OpenCVImageDecoder::decode(std::string_view raw_data,
                                torch::Tensor& t,
                                bool preserve_alpha) {
  if (raw_data.empty() ||
      raw_data.size() >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    LOG(ERROR) << "opencv image decode got empty or oversized data";
    return false;
  }
  cv::Mat buffer(1,
                 static_cast<int32_t>(raw_data.size()),
                 CV_8UC1,
                 const_cast<char*>(raw_data.data()));
  cv::Mat image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    LOG(INFO) << "opencv image decode failed";
    return false;
  }

  cv::Mat rgb_image;
  if (preserve_alpha) {
    if (image.channels() == 4) {
      cv::cvtColor(image, rgb_image, cv::COLOR_BGRA2RGBA);
    } else if (image.channels() == 3) {
      cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGBA);
    } else if (image.channels() == 1) {
      cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGBA);
    } else {
      LOG(ERROR) << "Unsupported image channel count: " << image.channels();
      return false;
    }
    if (rgb_image.depth() != CV_8U) {
      LOG(ERROR) << "RGBA DiT inputs currently require 8-bit pixels";
      return false;
    }
  } else {
    convert_decoded_image_to_rgb(image, rgb_image);
  }

  if (rgb_image.depth() == CV_8U && rgb_image.isContinuous()) {
    t = torch::from_blob(
            rgb_image.data,
            {rgb_image.rows, rgb_image.cols, rgb_image.channels()},
            [owner = rgb_image](void*) mutable { owner.release(); },
            torch::kUInt8)
            .permute({2, 0, 1});
    return true;
  }

  torch::Tensor tensor =
      torch::from_blob(rgb_image.data,
                       {rgb_image.rows, rgb_image.cols, rgb_image.channels()},
                       torch::TensorOptions().dtype(torch::kUInt8));

  t = tensor.permute({2, 0, 1}).clone();  // [C, H, W]
  return true;
}

bool OpenCVImageEncoder::encode(const torch::Tensor& t, std::string& raw_data) {
  if (!valid(t)) {
    return false;
  }

  auto img = t.permute({1, 2, 0}).contiguous();
  const int32_t channels = static_cast<int32_t>(t.size(0));
  cv::Mat mat(static_cast<int32_t>(img.size(0)),
              static_cast<int32_t>(img.size(1)),
              CV_MAKETYPE(CV_32F, channels),
              img.data_ptr<float>());

  cv::Mat mat_8u;
  mat.convertTo(mat_8u, CV_MAKETYPE(CV_8U, channels), 255.0);
  cv::cvtColor(
      mat_8u, mat_8u, channels == 4 ? cv::COLOR_RGBA2BGRA : cv::COLOR_RGB2BGR);

  std::vector<uchar> data;
  if (!cv::imencode(".png", mat_8u, data)) {
    LOG(ERROR) << "image encode failed";
    return false;
  }

  raw_data.assign(data.begin(), data.end());
  return true;
}

bool OpenCVImageEncoder::valid(const torch::Tensor& t) {
  if (t.dim() != 3 || (t.size(0) != 3 && t.size(0) != 4)) {
    LOG(ERROR) << "input tensor must be RGB/RGBA CHW tensor";
    return false;
  }

  if (t.scalar_type() != torch::kFloat32 || !t.device().is_cpu()) {
    LOG(ERROR) << "tensor must be cpu float32";
    return false;
  }

  return true;
}

bool FFmpegVideoDecoder::decode(std::string_view raw_data,
                                torch::Tensor& t,
                                VideoMetadata& metadata) {
  MemoryVideoReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size());

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "video decode failed";
    return false;
  }
  return true;
}

SampledVideoDecodeStatus FFmpegVideoDecoder::decode_sampled(
    std::string_view raw_data,
    const VideoFrameSelector& select_frames,
    torch::Tensor& tensor,
    VideoMetadata& metadata,
    uint64_t max_video_bytes,
    uint64_t max_total_bytes) {
  using Status = SampledVideoDecodeStatus;
  if (!select_frames) {
    return Status::DECODE_ERROR;
  }
  const auto* bytes = reinterpret_cast<const uint8_t*>(raw_data.data());
  constexpr int32_t kSampledDecoderThreads = 4;
  int32_t observed_frames = 0;
  std::optional<int32_t> packet_count;
  const auto read = [&](int32_t frame_count) {
    MemoryVideoReader reader(bytes, raw_data.size());
    VideoMetadata candidate;
    if (!reader.init(candidate,
                     kSampledDecoderThreads,
                     select_frames.match_opencv_rgb)) {
      return Status::DECODE_ERROR;
    }
    if (!(candidate.fps > 0.0)) {
      if (!select_frames.match_opencv_rgb) {
        return Status::DECODE_ERROR;
      }
      candidate.fps = 24.0;
    }
    candidate.total_num_frames =
        frame_count > 0 ? frame_count : reader.frame_count_hint();
    if (candidate.total_num_frames == 0) {
      packet_count = reader.count_video_packets();
      return Status::DECODE_ERROR;
    }
    candidate.duration = candidate.total_num_frames / candidate.fps;
    if (!reader.read_sampled(select_frames,
                             tensor,
                             candidate,
                             max_video_bytes,
                             max_total_bytes)) {
      if (reader.decoded_count() != candidate.total_num_frames) {
        observed_frames = reader.decoded_count();
      }
      return reader.sampled_failure();
    }
    metadata = std::move(candidate);
    return Status::SUCCESS;
  };
  const auto status = read(0);
  if (!packet_count) {
    return status == Status::DECODE_ERROR && observed_frames > 0
               ? read(observed_frames)
               : status;
  }
  if (*packet_count > 0) {
    const auto packet_status = read(*packet_count);
    if (packet_status == Status::SUCCESS) {
      return packet_status;
    }
    if (observed_frames > 0) {
      return read(observed_frames);
    }
  }
  {
    MemoryVideoReader counter(bytes, raw_data.size());
    VideoMetadata ignored;
    if (!counter.init(
            ignored, kSampledDecoderThreads, select_frames.match_opencv_rgb) ||
        !counter.scan() || counter.decoded_count() <= 0) {
      return Status::DECODE_ERROR;
    }
    observed_frames = counter.decoded_count();
  }
  return read(observed_frames);
}

bool FFmpegAudioDecoder::decode(std::string_view raw_data,
                                torch::Tensor& t,
                                AudioMetadata& metadata,
                                int64_t target_sr,
                                int32_t target_channels) {
  MemoryAudioReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size(),
                           target_sr,
                           target_channels);

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "audio decode failed";
    return false;
  }
  return true;
}

// ---- MemoryMediaWriter (in-memory encoding base class) ----

namespace {

struct MemWriteCtx {
  std::vector<uint8_t>* buf;
  int64_t pos;
};

struct Writer {
  static int32_t write(void* opaque, uint8_t* buf, int32_t buf_size) {
    auto* mc = static_cast<MemWriteCtx*>(opaque);
    int64_t end_pos = mc->pos + buf_size;
    if (end_pos > static_cast<int64_t>(mc->buf->size())) {
      mc->buf->resize(static_cast<size_t>(end_pos), 0);
    }
    std::memcpy(mc->buf->data() + mc->pos, buf, static_cast<size_t>(buf_size));
    mc->pos = end_pos;
    return buf_size;
  }

  static int64_t seek(void* opaque, int64_t offset, int32_t whence) {
    auto* mc = static_cast<MemWriteCtx*>(opaque);
    if (whence == AVSEEK_SIZE) {
      return static_cast<int64_t>(mc->buf->size());
    }
    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos = mc->pos + offset;
        break;
      case SEEK_END:
        pos = static_cast<int64_t>(mc->buf->size()) + offset;
        break;
      default:
        return AVERROR(EINVAL);
    }
    if (pos < 0) {
      return AVERROR(EINVAL);
    }
    mc->pos = pos;
    return pos;
  }
};

}  // namespace

class MemoryMediaWriter {
 public:
  MemoryMediaWriter() = default;

  virtual ~MemoryMediaWriter() {
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
      if (!finished_) {
        av_write_trailer(fmt_ctx_);
      }
      avformat_free_context(fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    }
  }

 protected:
  // Initializes the output format context and in-memory AVIO writer.
  bool init_memory_output_context(const char* format) {
    constexpr int32_t kAvioBufferSize = 1 << 16;
    uint8_t* avio_buffer =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(kAvioBufferSize)));
    if (avio_buffer == nullptr) {
      return false;
    }

    avio_ctx_ = avio_alloc_context(avio_buffer,
                                   kAvioBufferSize,
                                   1,
                                   &write_ctx_,
                                   nullptr,
                                   &Writer::write,
                                   &Writer::seek);
    if (avio_ctx_ == nullptr) {
      av_freep(reinterpret_cast<void**>(&avio_buffer));
      return false;
    }
    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;

    const AVOutputFormat* output_format =
        av_guess_format(format, nullptr, nullptr);
    if (output_format == nullptr) {
      LOG(ERROR) << "MemoryMediaWriter: no muxer for " << format;
      return false;
    }

    if (avformat_alloc_output_context2(
            &fmt_ctx_, output_format, nullptr, nullptr) < 0 ||
        fmt_ctx_ == nullptr) {
      return false;
    }
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;
    return true;
  }

  bool init(const char* format,
            AVCodecID codec_id,
            int32_t width,
            int32_t height,
            double fps,
            AVPixelFormat pix_fmt,
            AVDictionary** opts = nullptr) {
    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!codec) {
      LOG(ERROR) << "MemoryMediaWriter: encoder not found, codec_id="
                 << avcodec_get_name(codec_id);
      return false;
    }

    if (!init_memory_output_context(format)) {
      return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }

    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = {1, static_cast<int32_t>(std::llround(fps))};
    codec_ctx_->framerate = {static_cast<int32_t>(std::llround(fps)), 1};
    codec_ctx_->pix_fmt = pix_fmt;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
      codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(codec_ctx_, codec, opts) < 0) {
      LOG(ERROR) << "MemoryMediaWriter: avcodec_open2 failed for "
                 << codec->name;
      return false;
    }

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!stream_) {
      return false;
    }
    stream_->time_base = codec_ctx_->time_base;
    if (avcodec_parameters_from_context(stream_->codecpar, codec_ctx_) < 0) {
      return false;
    }

    if (avformat_write_header(fmt_ctx_, nullptr) < 0) {
      LOG(ERROR) << "MemoryMediaWriter: avformat_write_header failed";
      return false;
    }

    pkt_ = av_packet_alloc();
    if (!pkt_) {
      return false;
    }

    LOG(INFO) << "MemoryMediaWriter: initialized " << codec->name << " ["
              << format << "] " << width << "x" << height << " @ " << fps
              << " fps";
    return true;
  }

  bool send_frame(AVFrame* frame) {
    if (avcodec_send_frame(codec_ctx_, frame) < 0) {
      return false;
    }
    return drain_packets();
  }

  bool finish() {
    avcodec_send_frame(codec_ctx_, nullptr);
    if (!drain_packets()) {
      return false;
    }
    av_write_trailer(fmt_ctx_);
    finished_ = true;
    return true;
  }

  std::vector<uint8_t> take_output() { return std::move(out_buf_); }

  AVCodecContext* codec_ctx() { return codec_ctx_; }
  AVStream* stream() { return stream_; }

  bool drain_packets() {
    while (avcodec_receive_packet(codec_ctx_, pkt_) == 0) {
      if (pkt_->duration == 0) {
        pkt_->duration = av_rescale_q(
            1, av_inv_q(codec_ctx_->framerate), codec_ctx_->time_base);
      }
      av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
      pkt_->stream_index = stream_->index;
      if (av_interleaved_write_frame(fmt_ctx_, pkt_) < 0) {
        av_packet_unref(pkt_);
        return false;
      }
      av_packet_unref(pkt_);
    }
    return true;
  }

  AVFormatContext* fmt_ctx_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVStream* stream_ = nullptr;
  MemWriteCtx write_ctx_{&out_buf_, 0};
  std::vector<uint8_t> out_buf_;
  bool finished_ = false;
};

class MemoryVideoWriter final : public MemoryMediaWriter {
 public:
  MemoryVideoWriter() = default;

  ~MemoryVideoWriter() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (yuv_frame_) {
      av_frame_free(&yuv_frame_);
    }
  }

  bool write(const torch::Tensor& video,
             double fps,
             const std::string& format,
             std::string& raw_data) {
    if (video.dim() != 4 || video.size(1) != 3) {
      LOG(ERROR) << "MemoryVideoWriter: expects [T,C,H,W] with C=3, got "
                 << video.sizes();
      return false;
    }
    if (!video.device().is_cpu() || (video.scalar_type() != torch::kUInt8 &&
                                     video.scalar_type() != torch::kFloat32)) {
      LOG(ERROR) << "MemoryVideoWriter: expects cpu float32 or uint8 tensor";
      return false;
    }
    if (video.scalar_type() == torch::kUInt8 && !video.is_contiguous()) {
      LOG(ERROR) << "MemoryVideoWriter: expects contiguous uint8 tensor";
      return false;
    }
    const AVPixelFormat src_pix_fmt = video.scalar_type() == torch::kUInt8
                                          ? AV_PIX_FMT_GBRP
                                          : AV_PIX_FMT_RGB24;

    const int64_t T = video.size(0);
    const int64_t H = video.size(2);
    const int64_t W = video.size(3);
    if (T == 0 || H == 0 || W == 0) {
      LOG(ERROR) << "MemoryVideoWriter: empty dimensions T=" << T << " H=" << H
                 << " W=" << W;
      return false;
    }

    AVCodecID codec_id;
    AVPixelFormat pix_fmt;
    AVDictionary* opts = nullptr;

    if (format == "avi") {
      codec_id = AV_CODEC_ID_MJPEG;
      pix_fmt = AV_PIX_FMT_YUVJ420P;
    } else {
      const AVCodec* x264_codec = avcodec_find_encoder_by_name("libx264");

      if (x264_codec) {
        codec_id = x264_codec->id;
        pix_fmt = AV_PIX_FMT_YUV420P;
        av_dict_set(&opts, "crf", "18", 0);
        av_dict_set(&opts, "preset", "veryfast", 0);
        av_dict_set(&opts, "profile", "high", 0);
        av_dict_set(&opts, "level", "4.1", 0);
        LOG(INFO) << "Using libx264 H.264 encoder with CRF=18 preset=veryfast"
                  << " input_dtype="
                  << (video.scalar_type() == torch::kUInt8 ? "uint8"
                                                           : "float32")
                  << " source_pix_fmt="
                  << (video.scalar_type() == torch::kUInt8 ? "gbrp" : "rgb24");
      } else {
        codec_id = AV_CODEC_ID_MPEG4;
        pix_fmt = AV_PIX_FMT_YUV420P;
        av_dict_set(&opts, "mbd", "2", 0);
        LOG(WARNING) << "libx264 not available, using MPEG4 fallback";
      }
    }

    if (!init(format.c_str(),
              codec_id,
              static_cast<int32_t>(W),
              static_cast<int32_t>(H),
              fps,
              pix_fmt,
              &opts)) {
      if (opts) av_dict_free(&opts);
      return false;
    }
    if (opts) av_dict_free(&opts);

    sws_ctx_ = sws_getContext(static_cast<int32_t>(W),
                              static_cast<int32_t>(H),
                              src_pix_fmt,
                              static_cast<int32_t>(W),
                              static_cast<int32_t>(H),
                              pix_fmt,
                              SWS_BILINEAR,
                              nullptr,
                              nullptr,
                              nullptr);
    if (!sws_ctx_) {
      LOG(ERROR) << "MemoryVideoWriter: sws_getContext failed";
      return false;
    }

    yuv_frame_ = av_frame_alloc();
    if (!yuv_frame_) {
      return false;
    }
    yuv_frame_->format = pix_fmt;
    yuv_frame_->width = static_cast<int32_t>(W);
    yuv_frame_->height = static_cast<int32_t>(H);
    if (av_frame_get_buffer(yuv_frame_, 0) < 0) {
      return false;
    }

    const auto encode_frame = [this, H](const uint8_t* const* src_data,
                                        const int32_t* src_linesize,
                                        int64_t pts) -> bool {
      if (av_frame_make_writable(yuv_frame_) < 0) {
        return false;
      }
      sws_scale(sws_ctx_,
                src_data,
                src_linesize,
                /*src_slice_y=*/0,
                static_cast<int32_t>(H),
                yuv_frame_->data,
                yuv_frame_->linesize);
      yuv_frame_->pts = pts;
      return send_frame(yuv_frame_);
    };

    if (video.scalar_type() == torch::kUInt8) {
      const int64_t plane_size = H * W;
      const uint8_t* video_data = video.const_data_ptr<uint8_t>();
      const int32_t src_linesize[AV_NUM_DATA_POINTERS] = {
          static_cast<int32_t>(W),
          static_cast<int32_t>(W),
          static_cast<int32_t>(W),
          0};
      const uint8_t* src_data[AV_NUM_DATA_POINTERS] = {};
      for (int64_t t = 0; t < T; ++t) {
        const uint8_t* frame_data = video_data + t * 3 * plane_size;
        // GBRP planes are G/B/R; the tensor channels are R/G/B.
        src_data[0] = frame_data + plane_size;
        src_data[1] = frame_data + 2 * plane_size;
        src_data[2] = frame_data;
        if (!encode_frame(src_data, src_linesize, t)) {
          return false;
        }
      }
    } else {
      const auto video_acc = video.accessor<float, 4>();
      const int64_t stride = W * 3;
      std::vector<uint8_t> rgb_buf(static_cast<size_t>(H * stride));
      const uint8_t* src_data[AV_NUM_DATA_POINTERS] = {
          rgb_buf.data(), nullptr, nullptr, nullptr};
      const int32_t src_linesize[AV_NUM_DATA_POINTERS] = {
          static_cast<int32_t>(stride), 0, 0, 0};
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t y = 0; y < H; ++y) {
          for (int64_t x = 0; x < W; ++x) {
            rgb_buf[static_cast<size_t>(y * stride + x * 3 + 0)] =
                static_cast<uint8_t>(
                    std::clamp(video_acc[t][0][y][x] * 255.0f, 0.0f, 255.0f));
            rgb_buf[static_cast<size_t>(y * stride + x * 3 + 1)] =
                static_cast<uint8_t>(
                    std::clamp(video_acc[t][1][y][x] * 255.0f, 0.0f, 255.0f));
            rgb_buf[static_cast<size_t>(y * stride + x * 3 + 2)] =
                static_cast<uint8_t>(
                    std::clamp(video_acc[t][2][y][x] * 255.0f, 0.0f, 255.0f));
          }
        }
        if (!encode_frame(src_data, src_linesize, t)) {
          return false;
        }
      }
    }

    if (!finish()) {
      return false;
    }

    auto out = take_output();
    raw_data.assign(out.begin(), out.end());

    LOG(INFO) << "MemoryVideoWriter: encoded " << T << " frames (" << W << "x"
              << H << ") at " << fps << " fps [" << format << "], output "
              << out.size() << " bytes";
    return true;
  }

 private:
  SwsContext* sws_ctx_ = nullptr;
  AVFrame* yuv_frame_ = nullptr;
};

class MemoryAudioWriter {
 public:
  MemoryAudioWriter() = default;

  ~MemoryAudioWriter() {
    av_frame_free(&frame_);
    swr_free(&swr_ctx_);
    avcodec_free_context(&codec_ctx_);
  }

  // Set up the AAC encoder and a FLTP -> codec-format resampler for
  // `channels` planar-float input at `sample_rate`.
  bool open(int32_t sample_rate, int32_t channels, bool global_header) {
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!codec) {
      LOG(ERROR) << "MemoryAudioWriter: AAC encoder not available";
      return false;
    }
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }
    codec_ctx_->bit_rate = 192000;
    codec_ctx_->sample_rate = sample_rate;
    codec_ctx_->time_base = {1, sample_rate};
    codec_ctx_->sample_fmt = AV_SAMPLE_FMT_FLTP;
    if (codec->sample_fmts) {
      codec_ctx_->sample_fmt = codec->sample_fmts[0];
      for (const AVSampleFormat* sample_fmt = codec->sample_fmts;
           *sample_fmt != AV_SAMPLE_FMT_NONE;
           ++sample_fmt) {
        if (*sample_fmt == AV_SAMPLE_FMT_FLTP) {
          codec_ctx_->sample_fmt = *sample_fmt;
          break;
        }
      }
    }
    av_channel_layout_default(&codec_ctx_->ch_layout, channels);
    if (global_header) {
      codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
      LOG(ERROR) << "MemoryAudioWriter: failed to open AAC encoder";
      return false;
    }

    AVChannelLayout input_layout;
    av_channel_layout_default(&input_layout, channels);
    const bool swr_ok = swr_alloc_set_opts2(&swr_ctx_,
                                            &codec_ctx_->ch_layout,
                                            codec_ctx_->sample_fmt,
                                            sample_rate,
                                            &input_layout,
                                            AV_SAMPLE_FMT_FLTP,
                                            sample_rate,
                                            0,
                                            nullptr) >= 0 &&
                        swr_ctx_ && swr_init(swr_ctx_) >= 0;
    av_channel_layout_uninit(&input_layout);
    if (!swr_ok) {
      LOG(ERROR) << "MemoryAudioWriter: failed to initialize audio resampler";
      return false;
    }

    frame_size_ = codec_ctx_->frame_size > 0 ? codec_ctx_->frame_size : 1024;
    frame_ = av_frame_alloc();
    if (!frame_) {
      return false;
    }
    frame_->format = codec_ctx_->sample_fmt;
    frame_->sample_rate = sample_rate;
    frame_->nb_samples = frame_size_;
    if (av_channel_layout_copy(&frame_->ch_layout, &codec_ctx_->ch_layout) <
            0 ||
        av_frame_get_buffer(frame_, 0) < 0) {
      return false;
    }
    channels_ = channels;
    return true;
  }

  // Create the output audio stream in `out` from the opened encoder.
  AVStream* add_stream(AVFormatContext* out) {
    AVStream* stream = avformat_new_stream(out, nullptr);
    if (!stream) {
      return nullptr;
    }
    stream->time_base = codec_ctx_->time_base;
    if (avcodec_parameters_from_context(stream->codecpar, codec_ctx_) < 0) {
      return nullptr;
    }
    return stream;
  }

  // Encode a [channels, samples] float32 CPU tensor into AAC packets rescaled
  // to `stream_tb` and tagged with `stream_index`.
  bool encode(const torch::Tensor& audio,
              AVRational stream_tb,
              int32_t stream_index,
              std::vector<AVPacket*>& packets) {
    const int64_t samples = audio.size(1);
    const float* audio_ptr = audio.data_ptr<float>();
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
      return false;
    }
    bool encoding_succeeded = true;
    auto drain = [&]() {
      while (avcodec_receive_packet(codec_ctx_, packet) == 0) {
        AVPacket* out_packet = av_packet_clone(packet);
        av_packet_unref(packet);
        if (!out_packet) {
          encoding_succeeded = false;
          return;
        }
        av_packet_rescale_ts(out_packet, codec_ctx_->time_base, stream_tb);
        out_packet->stream_index = stream_index;
        packets.push_back(out_packet);
      }
    };

    int64_t sample_offset = 0;
    int64_t audio_pts = 0;
    while (sample_offset < samples) {
      if (av_frame_make_writable(frame_) < 0 ||
          av_samples_set_silence(
              frame_->data, 0, frame_size_, channels_, codec_ctx_->sample_fmt) <
              0) {
        encoding_succeeded = false;
        break;
      }
      const int32_t valid_samples = static_cast<int32_t>(
          std::min<int64_t>(frame_size_, samples - sample_offset));
      const uint8_t* input_data[AV_NUM_DATA_POINTERS] = {};
      for (int32_t channel = 0; channel < channels_; ++channel) {
        input_data[channel] = reinterpret_cast<const uint8_t*>(
            audio_ptr + channel * samples + sample_offset);
      }
      if (swr_convert(
              swr_ctx_, frame_->data, frame_size_, input_data, valid_samples) <
          0) {
        LOG(ERROR) << "MemoryAudioWriter: audio conversion failed";
        encoding_succeeded = false;
        break;
      }
      frame_->nb_samples = frame_size_;
      frame_->pts = audio_pts;
      if (avcodec_send_frame(codec_ctx_, frame_) < 0) {
        encoding_succeeded = false;
        break;
      }
      drain();
      if (!encoding_succeeded) {
        break;
      }
      sample_offset += valid_samples;
      audio_pts += frame_size_;
    }

    if (encoding_succeeded && sample_offset >= samples &&
        avcodec_send_frame(codec_ctx_, nullptr) >= 0) {
      drain();
    } else {
      encoding_succeeded = false;
    }
    av_packet_free(&packet);
    return encoding_succeeded && !packets.empty();
  }

 private:
  AVCodecContext* codec_ctx_ = nullptr;
  SwrContext* swr_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  int32_t frame_size_ = 1024;
  int32_t channels_ = 0;
};

class MemoryVideoAudioWriter final : public MemoryMediaWriter {
 public:
  MemoryVideoAudioWriter() = default;

  bool write(const torch::Tensor& video,
             const torch::Tensor& audio_in,
             double fps,
             int32_t sample_rate,
             const std::string& format,
             std::string& raw_data) {
    // Normalize audio to planar [channels, samples].
    torch::Tensor audio = audio_in.cpu().to(torch::kFloat32).contiguous();
    if (audio.dim() == 1) {
      audio = audio.unsqueeze(0);
    }
    if (audio.dim() != 2 || audio.size(0) <= 0 ||
        audio.size(0) > AV_NUM_DATA_POINTERS || audio.size(1) <= 0 ||
        sample_rate <= 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: invalid audio tensor "
                 << audio.sizes();
      return false;
    }
    const int32_t channels = static_cast<int32_t>(audio.size(0));

    std::string video_data;
    {
      MemoryVideoWriter video_writer;
      if (!video_writer.write(video, fps, format, video_data)) {
        return false;
      }
    }

    if (!init_memory_output_context(format.c_str())) {
      return false;
    }

    AVStream* out_video_stream = nullptr;
    std::vector<AVPacket*> video_packets;
    if (!copy_video_packets(video_data, out_video_stream, video_packets)) {
      free_packets(video_packets);
      return false;
    }

    MemoryAudioWriter audio_writer;
    const bool global_header =
        (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) != 0;
    if (!audio_writer.open(sample_rate, channels, global_header)) {
      free_packets(video_packets);
      return false;
    }
    AVStream* out_audio_stream = audio_writer.add_stream(fmt_ctx_);
    if (!out_audio_stream || avformat_write_header(fmt_ctx_, nullptr) < 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: failed to initialize mux";
      free_packets(video_packets);
      return false;
    }

    std::vector<AVPacket*> audio_packets;
    if (!audio_writer.encode(audio,
                             out_audio_stream->time_base,
                             out_audio_stream->index,
                             audio_packets)) {
      LOG(ERROR) << "MemoryVideoAudioWriter: AAC encoder produced no packets";
      free_packets(video_packets);
      free_packets(audio_packets);
      return false;
    }

    const bool mux_succeeded = interleave_and_write(
        out_video_stream, out_audio_stream, video_packets, audio_packets);
    free_packets(video_packets);
    free_packets(audio_packets);
    if (!mux_succeeded) {
      return false;
    }

    auto out = take_output();
    raw_data.assign(out.begin(), out.end());
    finished_ = true;
    return !raw_data.empty();
  }

 private:
  static void free_packets(std::vector<AVPacket*>& packets) {
    for (AVPacket*& packet : packets) {
      if (packet) {
        av_packet_free(&packet);
      }
    }
    packets.clear();
  }

  // Demux the already-encoded video bytestream, create a matching output video
  // stream (parameter copy) in fmt_ctx_, and collect its packets rescaled to
  // that stream. Reuses the in-memory read helpers (MemCtx/Reader).
  bool copy_video_packets(const std::string& video_data,
                          AVStream*& out_video_stream,
                          std::vector<AVPacket*>& packets) {
    MemCtx input_mem{reinterpret_cast<const uint8_t*>(video_data.data()),
                     static_cast<int64_t>(video_data.size()),
                     0};
    constexpr int32_t kAvioBufferSize = 1 << 16;
    uint8_t* input_buffer =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(kAvioBufferSize)));
    if (input_buffer == nullptr) {
      return false;
    }

    AVIOContext* input_io = avio_alloc_context(input_buffer,
                                               kAvioBufferSize,
                                               0,
                                               &input_mem,
                                               &Reader::read,
                                               nullptr,
                                               &Reader::seek);
    if (input_io == nullptr) {
      av_free(input_buffer);
      return false;
    }
    ScopeGuard input_io_guard([&input_io] {
      av_freep(&input_io->buffer);
      avio_context_free(&input_io);
    });

    AVFormatContext* input_format = avformat_alloc_context();
    if (input_format == nullptr) {
      return false;
    }
    ScopeGuard input_format_guard(
        [&input_format] { avformat_close_input(&input_format); });

    input_format->pb = input_io;
    input_format->flags |= AVFMT_FLAG_CUSTOM_IO;
    if (avformat_open_input(&input_format, nullptr, nullptr, nullptr) < 0 ||
        avformat_find_stream_info(input_format, nullptr) < 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: failed to open encoded video";
      return false;
    }

    const int32_t input_video_index = av_find_best_stream(
        input_format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (input_video_index < 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: encoded video stream not found";
      return false;
    }
    AVStream* input_video_stream = input_format->streams[input_video_index];

    out_video_stream = avformat_new_stream(fmt_ctx_, nullptr);
    if (out_video_stream == nullptr ||
        avcodec_parameters_copy(out_video_stream->codecpar,
                                input_video_stream->codecpar) < 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: failed to copy video parameters";
      return false;
    }
    out_video_stream->codecpar->codec_tag = 0;
    out_video_stream->time_base = input_video_stream->time_base;

    AVPacket* packet = av_packet_alloc();
    if (packet == nullptr) {
      return false;
    }
    ScopeGuard packet_guard([&packet] { av_packet_free(&packet); });

    while (av_read_frame(input_format, packet) >= 0) {
      if (packet->stream_index != input_video_index) {
        av_packet_unref(packet);
        continue;
      }

      AVPacket* video_packet = av_packet_clone(packet);
      av_packet_unref(packet);
      if (video_packet == nullptr) {
        LOG(ERROR) << "MemoryVideoAudioWriter: failed to clone video packet";
        return false;
      }
      av_packet_rescale_ts(video_packet,
                           input_video_stream->time_base,
                           out_video_stream->time_base);
      video_packet->stream_index = out_video_stream->index;
      packets.push_back(video_packet);
    }

    if (packets.empty()) {
      LOG(ERROR) << "MemoryVideoAudioWriter: encoded video has no packets";
      return false;
    }
    return true;
  }

  static int64_t packet_timestamp(const AVPacket* packet) {
    return packet->dts != AV_NOPTS_VALUE ? packet->dts : packet->pts;
  }

  // Merge the pre-collected video/audio packets in timestamp order and write
  // them to fmt_ctx_, then finalize the container.
  bool interleave_and_write(AVStream* video_stream,
                            AVStream* audio_stream,
                            std::vector<AVPacket*>& video_packets,
                            std::vector<AVPacket*>& audio_packets) {
    size_t video_index = 0;
    size_t audio_index = 0;
    while (video_index < video_packets.size() ||
           audio_index < audio_packets.size()) {
      bool use_video = audio_index >= audio_packets.size();
      if (!use_video && video_index < video_packets.size()) {
        const AVPacket* video_packet = video_packets[video_index];
        const AVPacket* audio_packet = audio_packets[audio_index];
        use_video = av_compare_ts(packet_timestamp(video_packet),
                                  video_stream->time_base,
                                  packet_timestamp(audio_packet),
                                  audio_stream->time_base) <= 0;
      }
      AVPacket*& next_packet = use_video ? video_packets[video_index++]
                                         : audio_packets[audio_index++];
      const int32_t write_result =
          av_interleaved_write_frame(fmt_ctx_, next_packet);
      av_packet_free(&next_packet);
      if (write_result < 0) {
        LOG(ERROR) << "MemoryVideoAudioWriter: failed to write muxed MP4";
        return false;
      }
    }
    if (av_write_trailer(fmt_ctx_) < 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: failed to write muxed MP4";
      return false;
    }
    avio_flush(avio_ctx_);
    return true;
  }
};

bool FFmpegVideoEncoder::encode(const torch::Tensor& video,
                                double fps,
                                const std::string& format,
                                std::string& raw_data) {
  MemoryVideoWriter writer;
  return writer.write(video, fps, format, raw_data);
}

bool FFmpegVideoEncoder::encode(const torch::Tensor& video,
                                const torch::Tensor& audio,
                                double fps,
                                int32_t sample_rate,
                                const std::string& format,
                                std::string& raw_data) {
  MemoryVideoAudioWriter writer;
  return writer.write(video, audio, fps, sample_rate, format, raw_data);
}

}  // namespace xllm
