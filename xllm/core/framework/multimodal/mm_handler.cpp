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

#include "mm_handler.h"

#include <butil/base64.h>
#include <butil/strings/string_number_conversions.h>
#include <glog/logging.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <deque>
#include <list>
#include <thread>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "common/global_flags.h"
#include "core/framework/config/multimodal_config.h"
#include "core/util/hash_util.h"
#include "core/util/http_downloader.h"
#include "mm_codec.h"
#include "mm_embedding_handler.h"

namespace xllm {

namespace {

// Parse --mm_download_headers once; it is fixed after startup.
const std::unordered_map<std::string, std::string>& global_download_headers() {
  static const std::unordered_map<std::string, std::string> headers =
      parse_headers_json(
          MultimodalConfig::get_instance().mm_download_headers());
  return headers;
}

class DecodedImageUrlCache final {
 public:
  struct CachedImage {
    XXH3Key hash_key;
    torch::Tensor decode_image;
  };

  struct InFlightSlot {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::optional<CachedImage> result;
  };

  struct AcquireOutcome {
    std::optional<CachedImage> hit;
    std::shared_ptr<InFlightSlot> waiter;
    bool is_leader = false;
  };

  static DecodedImageUrlCache& instance() {
    static DecodedImageUrlCache cache(/*max_entries=*/4096);
    return cache;
  }

  std::optional<CachedImage> lookup(const XXH3Key& url_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(url_key);
    if (it == entries_.end()) {
      return std::nullopt;
    }
    lru_.splice(lru_.end(), lru_, it->second.lru_it);
    return it->second.value;
  }

  AcquireOutcome acquire_or_join(const XXH3Key& url_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(url_key);
    if (it != entries_.end()) {
      lru_.splice(lru_.end(), lru_, it->second.lru_it);
      return AcquireOutcome{it->second.value, nullptr, false};
    }
    auto flight_it = inflight_.find(url_key);
    if (flight_it != inflight_.end()) {
      return AcquireOutcome{std::nullopt, flight_it->second, false};
    }
    auto slot = std::make_shared<InFlightSlot>();
    inflight_.emplace(url_key, slot);
    return AcquireOutcome{std::nullopt, std::move(slot), true};
  }

  void complete_inflight(const XXH3Key& url_key,
                         const std::shared_ptr<InFlightSlot>& slot,
                         std::optional<CachedImage> value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (value.has_value() && value->decode_image.defined()) {
        insert_locked(url_key, value->hash_key, value->decode_image);
      }
      auto flight_it = inflight_.find(url_key);
      if (flight_it != inflight_.end() && flight_it->second == slot) {
        inflight_.erase(flight_it);
      }
    }
    if (slot != nullptr) {
      {
        std::lock_guard<std::mutex> slot_lock(slot->mutex);
        slot->result = std::move(value);
        slot->done = true;
      }
      slot->cv.notify_all();
    }
  }

  void insert(const XXH3Key& url_key,
              const XXH3Key& hash_key,
              const torch::Tensor& decode_image) {
    if (!decode_image.defined()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    insert_locked(url_key, hash_key, decode_image);
  }

 private:
  explicit DecodedImageUrlCache(size_t max_entries)
      : max_entries_(max_entries) {}

  void insert_locked(const XXH3Key& url_key,
                     const XXH3Key& hash_key,
                     const torch::Tensor& decode_image) {
    auto it = entries_.find(url_key);
    if (it != entries_.end()) {
      lru_.splice(lru_.end(), lru_, it->second.lru_it);
      return;
    }
    while (entries_.size() >= max_entries_ && !lru_.empty()) {
      const XXH3Key& oldest = lru_.front();
      entries_.erase(oldest);
      lru_.pop_front();
    }
    lru_.push_back(url_key);
    entries_.emplace(
        url_key,
        Entry{CachedImage{hash_key, decode_image}, std::prev(lru_.end())});
  }

  struct Entry {
    CachedImage value;
    std::list<XXH3Key>::iterator lru_it;
  };

  size_t max_entries_;
  std::mutex mutex_;
  std::list<XXH3Key> lru_;
  std::unordered_map<XXH3Key, Entry, FixedStringKeyHash, FixedStringKeyEqual>
      entries_;
  std::unordered_map<XXH3Key,
                     std::shared_ptr<InFlightSlot>,
                     FixedStringKeyHash,
                     FixedStringKeyEqual>
      inflight_;
};

bool shared_mm_cache_enabled() {
  const char* env = std::getenv("XLLM_SHARED_MM_CACHE");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return std::string(env) != "0" && std::string(env) != "false" &&
         std::string(env) != "FALSE";
}

const std::string& shared_mm_cache_dir() {
  static const std::string dir = [] {
    const char* env = std::getenv("XLLM_SHARED_MM_CACHE_DIR");
    std::string d = (env != nullptr && env[0] != '\0')
                        ? std::string(env)
                        : std::string("/tmp/xllm_shared_mm_cache");
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    return d;
  }();
  return dir;
}

size_t shared_mm_cache_max_items() {
  const char* env = std::getenv("XLLM_SHARED_MM_CACHE_MAX_ITEMS");
  if (env == nullptr || env[0] == '\0') {
    return 1024;
  }
  char* endp = nullptr;
  const unsigned long long v = std::strtoull(env, &endp, 10);
  return (endp != env && v > 0) ? static_cast<size_t>(v) : 1024;
}

std::string xxh3_to_hex(const XXH3Key& key) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(sizeof(key.data) * 2, '0');
  for (size_t i = 0; i < sizeof(key.data); ++i) {
    const uint8_t b = static_cast<uint8_t>(key.data[i]);
    out[2 * i] = kHex[(b >> 4) & 0xf];
    out[2 * i + 1] = kHex[b & 0xf];
  }
  return out;
}

struct SharedDecodedImageHeader {
  uint32_t magic = 0x494D4D58u;  // "XMMI"
  uint32_t version = 1u;
  uint8_t content_key[16];
  int64_t dims[3];
};

constexpr int64_t kMaxSharedDecodedImageBytes = 32LL * 1024 * 1024;

void maybe_prune_shared_mm_dir() {
  static std::atomic<uint64_t> write_counter{0};
  if ((write_counter.fetch_add(1, std::memory_order_relaxed) & 63u) != 0u) {
    return;
  }
  const size_t max_files = shared_mm_cache_max_items() * 2;
  std::error_code ec;
  std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>>
      files;
  for (const auto& entry :
       std::filesystem::directory_iterator(shared_mm_cache_dir(), ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    const auto path = entry.path();
    if (path.extension() != ".bin") {
      continue;
    }
    auto ftime = entry.last_write_time(ec);
    if (!ec) {
      files.emplace_back(ftime, path);
    }
  }
  if (files.size() <= max_files) {
    return;
  }
  std::sort(files.begin(),
            files.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  const size_t to_remove = files.size() - max_files;
  for (size_t i = 0; i < to_remove; ++i) {
    std::filesystem::remove(files[i].second, ec);
  }
}

std::optional<DecodedImageUrlCache::CachedImage> load_shared_decoded_image(
    const XXH3Key& url_key) {
  if (!shared_mm_cache_enabled()) {
    return std::nullopt;
  }
  const std::string final_path =
      shared_mm_cache_dir() + "/img_" + xxh3_to_hex(url_key) + ".bin";
  FILE* fp = std::fopen(final_path.c_str(), "rb");
  if (fp == nullptr) {
    return std::nullopt;
  }
  SharedDecodedImageHeader hdr{};
  if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
      hdr.magic != 0x494D4D58u || hdr.version != 1u ||
      hdr.dims[0] <= 0 || hdr.dims[0] > 4 || hdr.dims[1] <= 0 ||
      hdr.dims[2] <= 0 || hdr.dims[1] > 16384 || hdr.dims[2] > 16384) {
    std::fclose(fp);
    return std::nullopt;
  }
  const int64_t nbytes = hdr.dims[0] * hdr.dims[1] * hdr.dims[2];
  if (nbytes <= 0 || nbytes > kMaxSharedDecodedImageBytes) {
    std::fclose(fp);
    return std::nullopt;
  }
  torch::Tensor img = torch::empty(
      {hdr.dims[0], hdr.dims[1], hdr.dims[2]},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  const size_t got =
      std::fread(img.data_ptr(), 1, static_cast<size_t>(nbytes), fp);
  std::fclose(fp);
  if (got != static_cast<size_t>(nbytes)) {
    return std::nullopt;
  }
  XXH3Key content_key;
  std::memcpy(content_key.data, hdr.content_key, sizeof(content_key.data));
  return DecodedImageUrlCache::CachedImage{content_key, std::move(img)};
}

void save_shared_decoded_image_sync(const XXH3Key& url_key,
                                    const XXH3Key& hash_key,
                                    const torch::Tensor& decode_image) {
  if (!shared_mm_cache_enabled() || !decode_image.defined() ||
      !decode_image.device().is_cpu() ||
      decode_image.scalar_type() != torch::kUInt8 || decode_image.dim() != 3) {
    return;
  }
  const int64_t nbytes = decode_image.numel();
  if (nbytes <= 0 || nbytes > kMaxSharedDecodedImageBytes) {
    return;
  }
  const std::string final_path =
      shared_mm_cache_dir() + "/img_" + xxh3_to_hex(url_key) + ".bin";
  std::error_code ec;
  if (std::filesystem::exists(final_path, ec)) {
    return;
  }
  torch::Tensor contig =
      decode_image.is_contiguous() ? decode_image : decode_image.contiguous();
  static std::atomic<uint64_t> tmp_seq{0};
  const std::string tmp_path =
      final_path + ".tmp." + std::to_string(::getpid()) + "." +
      std::to_string(tmp_seq.fetch_add(1, std::memory_order_relaxed));
  FILE* fp = std::fopen(tmp_path.c_str(), "wb");
  if (fp == nullptr) {
    return;
  }
  SharedDecodedImageHeader hdr{};
  std::memcpy(hdr.content_key, hash_key.data, sizeof(hdr.content_key));
  hdr.dims[0] = contig.size(0);
  hdr.dims[1] = contig.size(1);
  hdr.dims[2] = contig.size(2);
  const bool ok =
      (std::fwrite(&hdr, sizeof(hdr), 1, fp) == 1) &&
      (std::fwrite(contig.data_ptr(), 1, static_cast<size_t>(nbytes), fp) ==
       static_cast<size_t>(nbytes));
  std::fclose(fp);
  if (!ok) {
    std::filesystem::remove(tmp_path, ec);
    return;
  }
  std::filesystem::rename(tmp_path, final_path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
    return;
  }
  maybe_prune_shared_mm_dir();
}

void save_shared_decoded_image(const XXH3Key& url_key,
                               const XXH3Key& hash_key,
                               const torch::Tensor& decode_image) {
  if (!shared_mm_cache_enabled() || !decode_image.defined()) {
    return;
  }
  struct Task {
    XXH3Key url_key;
    XXH3Key hash_key;
    torch::Tensor img;
  };
  struct AsyncWriter {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Task> queue;
    AsyncWriter() {
      std::thread([this]() {
        for (;;) {
          Task t;
          {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [this] { return !queue.empty(); });
            t = std::move(queue.front());
            queue.pop_front();
          }
          save_shared_decoded_image_sync(t.url_key, t.hash_key, t.img);
        }
      }).detach();
    }
  };
  static AsyncWriter* writer = new AsyncWriter();
  {
    std::lock_guard<std::mutex> lk(writer->mu);
    if (writer->queue.size() >= 128) {
      return;
    }
    writer->queue.push_back(Task{url_key, hash_key, decode_image});
  }
  writer->cv.notify_one();
}

struct PendingImageLoadState {
  XXH3Key url_key;
  std::shared_ptr<DecodedImageUrlCache::InFlightSlot> inflight_slot;
};

thread_local std::optional<PendingImageLoadState> tls_pending_image_state;

}  // namespace

MMErrCode MMHandlerBase::process(const MMContent& content,
                                 MMInputItem& input,
                                 MMPayload& payload) {
  MMErrCode code = this->load(content, input, payload);
  if (code != MMErrCode::SUCCESS) return code;

  code = this->decode(input);
  if (code != MMErrCode::SUCCESS) return code;

  if (!input.raw_data.empty() && !input.hash_key.has_value()) {
    input.hash_key = hash_string(input.raw_data);
  }

  return MMErrCode::SUCCESS;
}

MMErrCode MMHandlerBase::load_from_dataurl(const std::string& url,
                                           std::string& raw_data,
                                           MMPayload& payload) {
  size_t pos = url.find_first_of(';');
  if (pos == std::string::npos) return MMErrCode::LOAD_DATA_ERR;

  butil::StringPiece sub(url, pos + 1);
  pos = sub.find_first_of(',');
  if (pos == std::string::npos) return MMErrCode::LOAD_DATA_ERR;

  butil::StringPiece type(sub, 0, pos);
  butil::StringPiece data(sub, pos + 1);

  if (type == "base64") {
    if (!butil::Base64Decode(data, &raw_data)) {
      return MMErrCode::LOAD_DATA_ERR;
    }
    return MMErrCode::SUCCESS;
  } else if (type == "binary") {
    size_t len = 0;
    bool res = butil::StringToSizeT(data, &len);
    if (res) {
      if (!payload.get(raw_data, len)) {
        LOG(ERROR) << "load data from binary url failed, url is: " << url;
        return MMErrCode::LOAD_DATA_ERR;
      }
      return MMErrCode::SUCCESS;
    } else {
      LOG(ERROR) << " data url is invalid, url is " << url;
      return MMErrCode::LOAD_DATA_ERR;
    }
  } else {
    LOG(ERROR) << " data url is invalid, url is " << url;
    return MMErrCode::LOAD_DATA_ERR;
  }
}

MMErrCode MMHandlerBase::load_from_local(const std::string& url,
                                         std::string& data) {
  std::string path = url;
  const std::string prefix = "file://";
  if (path.compare(0, prefix.size(), prefix) == 0) {
    path = path.substr(prefix.size());
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    LOG(ERROR) << "failed to open local file: " << path;
    return MMErrCode::LOAD_LOCAL_ERR;
  }

  data.assign(std::istreambuf_iterator<char>(in),
              std::istreambuf_iterator<char>());
  return MMErrCode::SUCCESS;
}

MMErrCode MMHandlerBase::load_from_http(
    const std::string& url,
    std::string& data,
    const std::unordered_map<std::string, std::string>& headers) {
  BRpcDownloader downloader(global_download_headers());
  if (!downloader.fetch_data(url, data, headers)) {
    return MMErrCode::LOAD_HTTP_ERR;
  }
  return MMErrCode::SUCCESS;
}

MMErrCode ImageHandler::load(const MMContent& content,
                             MMInputItem& input,
                             MMPayload& payload) {
  input.clear();
  tls_pending_image_state.reset();

  const auto& image_url = content.image_url;
  const auto& url = image_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url
    input.type = MMType::IMAGE;
    const bool is_base64 = url.find(";base64,") != std::string::npos;
    if (is_base64) {
      const XXH3Key url_key = hash_string(url);
      auto acq = DecodedImageUrlCache::instance().acquire_or_join(url_key);
      if (acq.hit.has_value()) {
        input.hash_key = acq.hit->hash_key;
        input.decode_image = acq.hit->decode_image;
        return MMErrCode::SUCCESS;
      }
      if (!acq.is_leader && acq.waiter != nullptr) {
        std::unique_lock<std::mutex> wait_lock(acq.waiter->mutex);
        acq.waiter->cv.wait(wait_lock, [&]() { return acq.waiter->done; });
        if (acq.waiter->result.has_value()) {
          input.hash_key = acq.waiter->result->hash_key;
          input.decode_image = acq.waiter->result->decode_image;
          return MMErrCode::SUCCESS;
        }
      }
      if (auto shared_hit = load_shared_decoded_image(url_key);
          shared_hit.has_value()) {
        input.hash_key = shared_hit->hash_key;
        input.decode_image = shared_hit->decode_image;
        if (acq.is_leader) {
          DecodedImageUrlCache::instance().complete_inflight(
              url_key, acq.waiter, std::move(shared_hit));
        } else {
          DecodedImageUrlCache::instance().insert(
              url_key, *input.hash_key, input.decode_image);
        }
        return MMErrCode::SUCCESS;
      }
      MMErrCode code = this->load_from_dataurl(url, input.raw_data, payload);
      if (code == MMErrCode::SUCCESS) {
        tls_pending_image_state = PendingImageLoadState{
            url_key, acq.is_leader ? std::move(acq.waiter) : nullptr};
      } else if (acq.is_leader) {
        DecodedImageUrlCache::instance().complete_inflight(
            url_key, acq.waiter, std::nullopt);
      }
      return code;
    }
    return this->load_from_dataurl(url, input.raw_data, payload);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url
    input.type = MMType::IMAGE;
    if (content.image_url.headers.empty()) {
      const XXH3Key url_key = hash_string(url);
      auto acq = DecodedImageUrlCache::instance().acquire_or_join(url_key);
      if (acq.hit.has_value()) {
        input.hash_key = acq.hit->hash_key;
        input.decode_image = acq.hit->decode_image;
        return MMErrCode::SUCCESS;
      }
      if (!acq.is_leader && acq.waiter != nullptr) {
        std::unique_lock<std::mutex> wait_lock(acq.waiter->mutex);
        acq.waiter->cv.wait(wait_lock, [&]() { return acq.waiter->done; });
        if (acq.waiter->result.has_value()) {
          input.hash_key = acq.waiter->result->hash_key;
          input.decode_image = acq.waiter->result->decode_image;
          return MMErrCode::SUCCESS;
        }
        // Leader failed; retry directly so transient single-request error can recover.
        MMErrCode retry_code = this->load_from_http(
            url, input.raw_data, content.image_url.headers);
        if (retry_code == MMErrCode::SUCCESS) {
          tls_pending_image_state = PendingImageLoadState{url_key, nullptr};
        }
        return retry_code;
      }
      if (auto shared_hit = load_shared_decoded_image(url_key);
          shared_hit.has_value()) {
        input.hash_key = shared_hit->hash_key;
        input.decode_image = shared_hit->decode_image;
        DecodedImageUrlCache::instance().complete_inflight(
            url_key, acq.waiter, std::move(shared_hit));
        return MMErrCode::SUCCESS;
      }
      MMErrCode code =
          this->load_from_http(url, input.raw_data, content.image_url.headers);
      if (code == MMErrCode::SUCCESS) {
        tls_pending_image_state =
            PendingImageLoadState{url_key, std::move(acq.waiter)};
      } else {
        DecodedImageUrlCache::instance().complete_inflight(
            url_key, acq.waiter, std::nullopt);
      }
      return code;
    }
    return this->load_from_http(url, input.raw_data, content.image_url.headers);
  } else {
    // treat as local path or file:// url
    input.type = MMType::IMAGE;
    if (this->load_from_local(url, input.raw_data) == MMErrCode::SUCCESS) {
      return MMErrCode::SUCCESS;
    }
    LOG(ERROR) << " image url is invalid, url is " << url;
    return MMErrCode::INVALID_URL_ERR;
  }
}

MMErrCode ImageHandler::decode(MMInputItem& input) {
  std::optional<PendingImageLoadState> pending =
      std::move(tls_pending_image_state);
  tls_pending_image_state.reset();
  if (input.decode_image.defined() && input.hash_key.has_value()) {
    if (pending.has_value() && pending->inflight_slot != nullptr) {
      DecodedImageUrlCache::instance().complete_inflight(
          pending->url_key,
          pending->inflight_slot,
          DecodedImageUrlCache::CachedImage{*input.hash_key,
                                            input.decode_image});
    }
    return MMErrCode::SUCCESS;
  }
  OpenCVImageDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_image)) {
    if (pending.has_value() && pending->inflight_slot != nullptr) {
      DecodedImageUrlCache::instance().complete_inflight(
          pending->url_key, pending->inflight_slot, std::nullopt);
    }
    return MMErrCode::DECODE_ERR;
  }
  if (!input.raw_data.empty()) {
    input.hash_key = hash_string(input.raw_data);
  }
  if (pending.has_value() && input.hash_key.has_value()) {
    save_shared_decoded_image(
        pending->url_key, *input.hash_key, input.decode_image);
    if (pending->inflight_slot != nullptr) {
      DecodedImageUrlCache::instance().complete_inflight(
          pending->url_key,
          pending->inflight_slot,
          DecodedImageUrlCache::CachedImage{*input.hash_key,
                                            input.decode_image});
    } else {
      DecodedImageUrlCache::instance().insert(
          pending->url_key, *input.hash_key, input.decode_image);
    }
    std::string().swap(input.raw_data);
  } else if (pending.has_value() && pending->inflight_slot != nullptr) {
    DecodedImageUrlCache::instance().complete_inflight(
        pending->url_key, pending->inflight_slot, std::nullopt);
  }
  return MMErrCode::SUCCESS;
}

MMErrCode VideoHandler::load(const MMContent& content,
                             MMInputItem& input,
                             MMPayload& payload) {
  input.clear();

  const auto& video_url = content.video_url;
  const auto& url = video_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::VIDEO;
    return this->load_from_dataurl(url, input.raw_data, payload);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::VIDEO;
    return this->load_from_http(url, input.raw_data, content.video_url.headers);
  } else {
    // treat as local path or file:// url (mirrors ImageHandler so local
    // video files are accepted, not just data/http URLs).
    input.type = MMType::VIDEO;
    if (this->load_from_local(url, input.raw_data) == MMErrCode::SUCCESS) {
      return MMErrCode::SUCCESS;
    }
    LOG(ERROR) << " video url is invalid, url is " << url;
    return MMErrCode::INVALID_URL_ERR;
  }
}

MMErrCode VideoHandler::decode(MMInputItem& input) {
  if (::xllm::MultimodalConfig::get_instance().use_audio_in_video()) {
    FFmpegAudioDecoder audio_decoder;
    if (audio_decoder.decode(
            input.raw_data, input.decode_audio, input.audio_meta)) {
      input.type |= MMType::AUDIO;
    } else {
      LOG(ERROR) << "decode audio in video failed";
      return MMErrCode::DECODE_ERR;
    }
  }

  FFmpegVideoDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_video, input.video_meta)) {
    return MMErrCode::DECODE_ERR;
  }
  return MMErrCode::SUCCESS;
}

MMErrCode AudioHandler::load(const MMContent& content,
                             MMInputItem& input,
                             MMPayload& payload) {
  input.clear();

  const auto& audio_url = content.audio_url;
  const auto& url = audio_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::AUDIO;
    return this->load_from_dataurl(url, input.raw_data, payload);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::AUDIO;
    return this->load_from_http(url, input.raw_data, content.audio_url.headers);
  } else {
    LOG(ERROR) << " audio url is invalid, url is " << url;
    return MMErrCode::INVALID_URL_ERR;
  }
}

MMErrCode AudioHandler::decode(MMInputItem& input) {
  FFmpegAudioDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_audio, input.audio_meta)) {
    return MMErrCode::DECODE_ERR;
  }
  return MMErrCode::SUCCESS;
}

MMHandlerSet::MMHandlerSet() {
  handlers_["image_url"] = std::make_unique<ImageHandler>();
  handlers_["video_url"] = std::make_unique<VideoHandler>();
  handlers_["audio_url"] = std::make_unique<AudioHandler>();
  handlers_["image_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::IMAGE);
  handlers_["video_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::VIDEO);
  handlers_["audio_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::AUDIO);
}

MMHandlerSet::~MMHandlerSet() {}

MMErrCode MMHandlerSet::process(const std::string& type,
                                const MMContent& content,
                                MMInputItem& input,
                                MMPayload& payload) {
  auto itor = handlers_.find(type);
  if (itor == handlers_.end()) {
    return MMErrCode::HANDLER_ERR;
  }

  auto& handler = itor->second;
  return handler->process(content, input, payload);
}

}  // namespace xllm
