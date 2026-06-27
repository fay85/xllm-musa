/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE
==============================================================================*/

#include "layer_hidden_dumper.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace xllm::debug {

LayerHiddenDumper& LayerHiddenDumper::instance() {
  static LayerHiddenDumper inst;
  return inst;
}

LayerHiddenDumper::LayerHiddenDumper() {
  const char* dir_env = std::getenv("XLLM_DUMP_HIDDEN_DIR");
  if (dir_env == nullptr || std::strlen(dir_env) == 0) {
    return;
  }
  dir_ = dir_env;
  const char* tag_env = std::getenv("XLLM_DUMP_TAG");
  tag_ = (tag_env != nullptr && std::strlen(tag_env) > 0) ? tag_env : "run";
  const char* steps_env = std::getenv("XLLM_DUMP_STEPS");
  if (steps_env != nullptr) {
    try {
      max_steps_to_dump_ = std::stoi(steps_env);
    } catch (...) {
      // keep default
    }
  }
  const char* reserved_env = std::getenv("XLLM_DUMP_RESERVED_TOKENS");
  if (reserved_env != nullptr) {
    try {
      reserved_max_tokens_ = std::stoll(reserved_env);
    } catch (...) {
      // keep default
    }
  }
  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);
  enabled_ = true;
  LOG(INFO) << "[LayerHiddenDumper] enabled: dir=" << dir_
            << ", tag=" << tag_ << ", max_steps=" << max_steps_to_dump_;
}

void LayerHiddenDumper::ensure_buffers(int num_slots,
                                       int64_t max_tokens,
                                       int64_t hidden_size,
                                       const torch::Device& device,
                                       torch::ScalarType dtype) {
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mtx_);

  // Grow-only: never re-allocate buffers once captured into a graph. Resizing
  // would invalidate captured copy_ op pointers (graph replay writes to freed
  // storage). Pre-allocate to the largest expected shape on the first call;
  // subsequent calls with smaller shapes reuse the same storage.
  const int64_t alloc_tokens = std::max<int64_t>(max_tokens, reserved_max_tokens_);
  if (!bufs_.empty()) {
    if (num_slots_ == num_slots && hidden_size_ == hidden_size &&
        max_tokens_ >= max_tokens) {
      return;  // existing buffers are sufficient
    }
    LOG(WARNING) << "[LayerHiddenDumper] shape change requires re-alloc "
                 << "(num_slots " << num_slots_ << "->" << num_slots
                 << ", max_tokens " << max_tokens_ << "->" << alloc_tokens
                 << ", hidden " << hidden_size_ << "->" << hidden_size
                 << "). This INVALIDATES any captured graph. Avoid by "
                 << "setting XLLM_DUMP_RESERVED_TOKENS large enough.";
    bufs_.clear();
  }
  bufs_.reserve(num_slots);
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  for (int i = 0; i < num_slots; ++i) {
    bufs_.push_back(torch::zeros({alloc_tokens, hidden_size}, opts));
  }
  num_slots_ = num_slots;
  max_tokens_ = alloc_tokens;
  hidden_size_ = hidden_size;
  LOG(INFO) << "[LayerHiddenDumper] allocated " << num_slots
            << " GPU buffers shape=[" << alloc_tokens << ", " << hidden_size
            << "] dtype=" << dtype
            << " (reserved >= requested " << max_tokens << ")";
}

void LayerHiddenDumper::record(int slot, const torch::Tensor& h) {
  if (!enabled_) return;
  if (slot < 0 || slot >= num_slots_) {
    LOG_FIRST_N(WARNING, 4) << "[LayerHiddenDumper] slot " << slot
                            << " out of range [0, " << num_slots_ << ")";
    return;
  }
  if (!h.defined()) return;
  const int64_t n = h.size(0);
  if (n > max_tokens_) {
    LOG_FIRST_N(WARNING, 4) << "[LayerHiddenDumper] tensor too large for slot "
                            << slot << " (got " << n << " > " << max_tokens_
                            << ")";
    return;
  }
  // 2D buffer flatten-safe: copy first n rows.
  bufs_[slot].narrow(0, 0, n).copy_(h.reshape({n, -1}));
}

bool LayerHiddenDumper::flush(int64_t actual_num_tokens) {
  if (!enabled_) return false;
  const int step = step_counter_.fetch_add(1);
  if (step >= max_steps_to_dump_) return false;
  if (actual_num_tokens <= 0) actual_num_tokens = max_tokens_;
  if (actual_num_tokens > max_tokens_) actual_num_tokens = max_tokens_;
  std::lock_guard<std::mutex> lock(mtx_);
  for (int i = 0; i < num_slots_; ++i) {
    auto cpu = bufs_[i]
                   .narrow(0, 0, actual_num_tokens)
                   .detach()
                   .to(torch::kCPU)
                   .to(torch::kFloat32)
                   .contiguous();
    const std::string path = dir_ + "/" + tag_ + "_step" +
                             std::to_string(step) + "_slot" +
                             std::to_string(i) + ".pt";
    auto bytes = torch::pickle_save(cpu);
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
      LOG(WARNING) << "[LayerHiddenDumper] cannot open " << path
                   << " for writing";
      continue;
    }
    ofs.write(bytes.data(), bytes.size());
  }
  LOG(INFO) << "[LayerHiddenDumper] flushed step " << step << " ("
            << num_slots_ << " slots, " << actual_num_tokens << " tokens)";
  return true;
}

}  // namespace xllm::debug
