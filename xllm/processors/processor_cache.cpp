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

#include "processors/processor_cache.h"

#include <glog/logging.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <utility>
#include <condition_variable>
#include <deque>
#include <thread>
#include <vector>

#include "core/framework/config/model_config.h"

namespace xllm {

namespace {

constexpr uint32_t kSharedProcMagic = 0x504D4D58u;  // "XMMP"
constexpr uint32_t kSharedProcVersion = 1u;
constexpr uint64_t kMaxSharedProcBytes = 64ULL * 1024 * 1024;

bool shared_proc_cache_enabled() {
  const char* env = std::getenv("XLLM_SHARED_MM_CACHE");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return std::string(env) != "0" && std::string(env) != "false" &&
         std::string(env) != "FALSE";
}

const std::string& shared_proc_cache_dir() {
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

std::string key_to_hex_str(const XXH3Key& key, size_t max_bytes = 16) {
  static constexpr char kHex[] = "0123456789abcdef";
  const size_t n = std::min<size_t>(sizeof(key.data), max_bytes);
  std::string out(n * 2, '0');
  for (size_t i = 0; i < n; ++i) {
    const uint8_t b = static_cast<uint8_t>(key.data[i]);
    out[2 * i] = kHex[(b >> 4) & 0xf];
    out[2 * i + 1] = kHex[b & 0xf];
  }
  return out;
}

const std::string& model_proc_tag() {
  static const std::string tag = [] {
    const std::string& model_id = ModelConfig::get_instance().model();
    const XXH3Key k = hash_string(model_id);
    return key_to_hex_str(k, 4);
  }();
  return tag;
}

std::string shared_proc_path(const XXH3Key& key) {
  return shared_proc_cache_dir() + "/proc_" + model_proc_tag() + "_" +
         key_to_hex_str(key) + ".bin";
}

struct SharedProcFileHeader {
  uint32_t magic = kSharedProcMagic;
  uint32_t version = kSharedProcVersion;
  uint32_t mm_type = 0;
  uint32_t num_tensors = 0;
  uint8_t schedule_key[16];
};

std::optional<MMDataItem> load_shared_proc_item(const XXH3Key& key) {
  if (!shared_proc_cache_enabled()) {
    return std::nullopt;
  }
  const std::string path = shared_proc_path(key);
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) {
    return std::nullopt;
  }
  SharedProcFileHeader hdr{};
  if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
      hdr.magic != kSharedProcMagic || hdr.version != kSharedProcVersion ||
      hdr.mm_type != static_cast<uint32_t>(MMType::IMAGE) ||
      hdr.num_tensors == 0 || hdr.num_tensors > 16) {
    std::fclose(fp);
    return std::nullopt;
  }
  MMDict dict;
  uint64_t total_bytes = 0;
  for (uint32_t idx = 0; idx < hdr.num_tensors; ++idx) {
    uint32_t name_len = 0;
    int32_t dtype_i32 = 0;
    uint32_t ndim = 0;
    if (std::fread(&name_len, sizeof(name_len), 1, fp) != 1 ||
        name_len == 0 || name_len > 256 ||
        std::fread(&dtype_i32, sizeof(dtype_i32), 1, fp) != 1 ||
        std::fread(&ndim, sizeof(ndim), 1, fp) != 1 || ndim == 0 || ndim > 8) {
      std::fclose(fp);
      return std::nullopt;
    }
    std::string name(name_len, '\0');
    if (std::fread(name.data(), 1, name_len, fp) != name_len) {
      std::fclose(fp);
      return std::nullopt;
    }
    std::vector<int64_t> sizes(ndim, 0);
    if (std::fread(sizes.data(), sizeof(int64_t), ndim, fp) != ndim) {
      std::fclose(fp);
      return std::nullopt;
    }
    for (int64_t s : sizes) {
      if (s < 0 || s > (1LL << 30)) {
        std::fclose(fp);
        return std::nullopt;
      }
    }
    uint64_t nbytes = 0;
    if (std::fread(&nbytes, sizeof(nbytes), 1, fp) != 1 ||
        nbytes > kMaxSharedProcBytes ||
        total_bytes + nbytes > kMaxSharedProcBytes) {
      std::fclose(fp);
      return std::nullopt;
    }
    total_bytes += nbytes;
    const auto stype = static_cast<c10::ScalarType>(dtype_i32);
    torch::Tensor t = torch::empty(
        sizes, torch::TensorOptions().dtype(stype).device(torch::kCPU));
    if (static_cast<uint64_t>(t.nbytes()) != nbytes) {
      std::fclose(fp);
      return std::nullopt;
    }
    if (nbytes > 0 &&
        std::fread(t.data_ptr(), 1, static_cast<size_t>(nbytes), fp) !=
            static_cast<size_t>(nbytes)) {
      std::fclose(fp);
      return std::nullopt;
    }
    dict.emplace(std::move(name), std::move(t));
  }
  std::fclose(fp);
  MMDataItem item(MMType::IMAGE, dict);
  XXH3Key skey;
  std::memcpy(skey.data, hdr.schedule_key, sizeof(skey.data));
  item.mutable_state().mutable_schedule_data().key = skey;
  return item;
}

void save_shared_proc_item_sync(const XXH3Key& key, const MMDataItem& item) {
  if (!shared_proc_cache_enabled() || !item.valid() ||
      item.type() != MMType::IMAGE || item.is_embedded() ||
      item.data().empty() || item.data().size() > 16) {
    return;
  }
  const std::string final_path = shared_proc_path(key);
  std::error_code ec;
  if (std::filesystem::exists(final_path, ec)) {
    return;
  }
  uint64_t total_bytes = 0;
  std::vector<std::pair<std::string, torch::Tensor>> tensors;
  tensors.reserve(item.data().size());
  for (const auto& [k, v] : item.data()) {
    if (!std::holds_alternative<torch::Tensor>(v) || k.empty() ||
        k.size() > 256) {
      return;
    }
    const torch::Tensor& src = std::get<torch::Tensor>(v);
    if (!src.defined() || !src.device().is_cpu()) {
      return;
    }
    torch::Tensor contig = src.is_contiguous() ? src : src.contiguous();
    const uint64_t nb = static_cast<uint64_t>(contig.nbytes());
    if (nb > kMaxSharedProcBytes || total_bytes + nb > kMaxSharedProcBytes) {
      return;
    }
    total_bytes += nb;
    tensors.emplace_back(k, std::move(contig));
  }

  static std::atomic<uint64_t> tmp_seq{0};
  const std::string tmp_path =
      final_path + ".tmp." + std::to_string(::getpid()) + "." +
      std::to_string(tmp_seq.fetch_add(1, std::memory_order_relaxed));
  FILE* fp = std::fopen(tmp_path.c_str(), "wb");
  if (fp == nullptr) {
    return;
  }
  SharedProcFileHeader hdr{};
  hdr.mm_type = static_cast<uint32_t>(item.type());
  hdr.num_tensors = static_cast<uint32_t>(tensors.size());
  const XXH3Key& skey = item.state().schedule_data().key;
  std::memcpy(hdr.schedule_key, skey.data, sizeof(hdr.schedule_key));

  bool ok = (std::fwrite(&hdr, sizeof(hdr), 1, fp) == 1);
  for (const auto& [name, t] : tensors) {
    if (!ok) {
      break;
    }
    const uint32_t name_len = static_cast<uint32_t>(name.size());
    const int32_t dtype_i32 = static_cast<int32_t>(t.scalar_type());
    const uint32_t ndim = static_cast<uint32_t>(t.dim());
    const auto sizes = t.sizes();
    const uint64_t nbytes = static_cast<uint64_t>(t.nbytes());
    ok = (std::fwrite(&name_len, sizeof(name_len), 1, fp) == 1) &&
         (std::fwrite(&dtype_i32, sizeof(dtype_i32), 1, fp) == 1) &&
         (std::fwrite(&ndim, sizeof(ndim), 1, fp) == 1) &&
         (std::fwrite(name.data(), 1, name_len, fp) == name_len) &&
         (std::fwrite(sizes.data(), sizeof(int64_t), ndim, fp) == ndim) &&
         (std::fwrite(&nbytes, sizeof(nbytes), 1, fp) == 1) &&
         (nbytes == 0 ||
          std::fwrite(t.data_ptr(), 1, static_cast<size_t>(nbytes), fp) ==
              static_cast<size_t>(nbytes));
  }
  std::fclose(fp);
  if (!ok) {
    std::filesystem::remove(tmp_path, ec);
    return;
  }
  std::filesystem::rename(tmp_path, final_path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
  }
}

void save_shared_proc_item(const XXH3Key& key, const MMDataItem& item) {
  if (!shared_proc_cache_enabled() || !item.valid() ||
      item.type() != MMType::IMAGE || item.is_embedded()) {
    return;
  }
  struct Task {
    XXH3Key key;
    MMDataItem item;
  };
  struct AsyncWriter {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Task> queue;
    AsyncWriter() {
      std::thread([this]() {
        for (;;) {
          std::optional<Task> t;
          {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [this] { return !queue.empty(); });
            t.emplace(std::move(queue.front()));
            queue.pop_front();
          }
          save_shared_proc_item_sync(t->key, t->item);
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
    writer->queue.push_back(Task{key, item});
  }
  writer->cv.notify_one();
}

}  // namespace

ProcessorCache::ProcessorCache(int64_t max_items) : max_items_(max_items) {
  CHECK_GE(max_items, 0) << "ProcessorCache max_items must be non-negative";
}

std::optional<MMDataItem> ProcessorCache::lookup(const XXH3Key& key) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    EntryMap::iterator it = entries_.find(key);
    if (it != entries_.end()) {
      touch(it);
      return it->second.item;
    }
  }
  std::optional<MMDataItem> shared_hit = load_shared_proc_item(key);
  if (shared_hit.has_value() && max_items_ > 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    EntryMap::iterator it = entries_.find(key);
    if (it == entries_.end()) {
      evict_if_full();
      lru_keys_.push_back(key);
      entries_.emplace(
          key, Entry{*shared_hit, std::prev(lru_keys_.end())});
    } else {
      touch(it);
    }
  }
  return shared_hit;
}

void ProcessorCache::insert(const XXH3Key& key, const MMDataItem& item) {
  if (max_items_ == 0 || !item.valid()) {
    return;
  }

  {
    MMDataItem cached_item = item;
    std::lock_guard<std::mutex> lock(mutex_);
    EntryMap::iterator it = entries_.find(key);
    if (it != entries_.end()) {
      touch(it);
    } else {
      evict_if_full();
      lru_keys_.push_back(key);
      entries_.emplace(
          key, Entry{std::move(cached_item), std::prev(lru_keys_.end())});
    }
  }
  save_shared_proc_item(key, item);
}

void ProcessorCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  lru_keys_.clear();
}

void ProcessorCache::touch(EntryMap::iterator it) {
  lru_keys_.splice(lru_keys_.end(), lru_keys_, it->second.lru_it);
}

void ProcessorCache::erase(EntryMap::iterator it) {
  lru_keys_.erase(it->second.lru_it);
  entries_.erase(it);
}

void ProcessorCache::evict_if_full() {
  while (!lru_keys_.empty() &&
         static_cast<int64_t>(entries_.size()) >= max_items_) {
    const XXH3Key& evict_key = lru_keys_.front();
    EntryMap::iterator it = entries_.find(evict_key);
    CHECK(it != entries_.end());
    erase(it);
  }
}

}  // namespace xllm
