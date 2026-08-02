/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#if defined(USE_MUSA)
#include <musa_runtime_api.h>

#include "torch_musa/csrc/core/MUSAStream.h"
#elif defined(USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace xllm {

// Opt-in Phase E.3 / G attribution. Enable with XLLM_PREFILL_BREAKDOWN=1.
// Uses raw CUDA/MUSA events so common_layers (no MusaMapping plugin) can
// include this header. Mate is nested inside gdn_attn.
class PrefillBreakdown final {
 public:
  enum class Bucket : int32_t {
    kEmbed = 0,
    kFullAttn = 1,
    kGdnAttn = 2,
    kMate = 3,
    kMlpGateUp = 4,
    kMlpAct = 5,
    kMlpDown = 6,
    kNorm = 7,
    // Nested under kGdnAttn (excluded from accounted sum, like kMate).
    kGdnProj = 8,
    kGdnSplit = 9,
    kGdnConv = 10,
    kGdnGate = 11,
    kGdnLayout = 12,
    kGdnOProj = 13,
    // Nested under kFullAttn (Phase G.5b).
    kFullQkv = 14,
    kFullPrep = 15,   // slice + QK-norm + RoPE (+ optional gate slice)
    kFullFa = 16,     // FlashInfer FA2/FA3 / AttentionImpl::forward
    kFullOProj = 17,  // output gate + o_proj
    // Nested inside the projection buckets above. These isolate the native
    // block-FP8 preparation, activation quantizer, and Mate GEMM.
    kFp8Prepare = 18,
    kFp8Quant = 19,
    kFp8Gemm = 20,
    // Nested under the Qwen3.5 routed-MoE layer scope. These distinguish the
    // routed expert path from the shared expert and from the outer MoE total.
    kMoeRoute = 21,
    kMoePreprocess = 22,
    kMoeGateUp = 23,
    kMoeAct = 24,
    kMoeDown = 25,
    kMoeCombine = 26,
    kMoeShared = 27,
    // Additional GDN-core attribution buckets.
    kGdnStatePrep = 28,
    kGdnStateWrite = 29,
    kGdnNorm = 30,
    kCount = 31,
  };

  static bool enabled() {
    static const bool on = [] {
      const char* env = std::getenv("XLLM_PREFILL_BREAKDOWN");
      return env != nullptr && std::string(env) == "1";
    }();
    return on;
  }

  static void begin() {
    if (!enabled()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex());
    records().clear();
  }

  static void end_and_log(int64_t n_tokens,
                          int32_t batch_bs,
                          double wall_fwd_ms) {
    if (!enabled()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex());
    auto& recs = records();
    if (recs.empty()) {
      LOG(INFO) << "[PREFILL_BREAKDOWN] n_tokens=" << n_tokens
                << " batch_bs=" << batch_bs << " wall_ms=" << wall_fwd_ms
                << " (no samples)";
      return;
    }

    sync_device();

    std::array<double, static_cast<size_t>(Bucket::kCount)> sums{};
    std::array<int32_t, static_cast<size_t>(Bucket::kCount)> counts{};
    for (auto& rec : recs) {
      if (!rec.finished) {
        continue;
      }
      const float ms = elapsed_ms(rec.start, rec.end);
      const size_t idx = static_cast<size_t>(rec.bucket);
      sums[idx] += static_cast<double>(ms);
      counts[idx] += 1;
    }

    // Mate / GDN / full-attn sub-stages are nested; exclude them from the
    // accounted sum so wall_ms ≈ accounted + other still holds.
    auto is_nested = [](Bucket b) {
      return b == Bucket::kMate || b == Bucket::kGdnProj ||
             b == Bucket::kGdnSplit || b == Bucket::kGdnConv ||
             b == Bucket::kGdnGate || b == Bucket::kGdnLayout ||
             b == Bucket::kGdnOProj || b == Bucket::kFullQkv ||
             b == Bucket::kFullPrep || b == Bucket::kFullFa ||
             b == Bucket::kFullOProj || b == Bucket::kFp8Prepare ||
             b == Bucket::kFp8Quant || b == Bucket::kFp8Gemm ||
             b == Bucket::kMoeRoute || b == Bucket::kMoePreprocess ||
             b == Bucket::kMoeGateUp || b == Bucket::kMoeAct ||
             b == Bucket::kMoeDown || b == Bucket::kMoeCombine ||
             b == Bucket::kMoeShared || b == Bucket::kGdnStatePrep ||
             b == Bucket::kGdnStateWrite || b == Bucket::kGdnNorm;
    };
    double accounted = 0.0;
    for (size_t i = 0; i < sums.size(); ++i) {
      if (is_nested(static_cast<Bucket>(i))) {
        continue;
      }
      accounted += sums[i];
    }
    const double other = wall_fwd_ms - accounted;
    const double mlp_ms = sums[static_cast<size_t>(Bucket::kMlpGateUp)] +
                          sums[static_cast<size_t>(Bucket::kMlpAct)] +
                          sums[static_cast<size_t>(Bucket::kMlpDown)];
    const double gdn_non_mate_ms = sums[static_cast<size_t>(Bucket::kGdnAttn)] -
                                   sums[static_cast<size_t>(Bucket::kMate)];

    LOG(INFO)
        << "[PREFILL_BREAKDOWN] n_tokens=" << n_tokens
        << " batch_bs=" << batch_bs << " wall_ms=" << wall_fwd_ms
        << " accounted_ms=" << accounted << " other_ms=" << other
        << " embed_ms=" << sums[static_cast<size_t>(Bucket::kEmbed)]
        << " full_attn_ms=" << sums[static_cast<size_t>(Bucket::kFullAttn)]
        << " gdn_attn_ms=" << sums[static_cast<size_t>(Bucket::kGdnAttn)]
        << " mate_ms=" << sums[static_cast<size_t>(Bucket::kMate)]
        << " gdn_non_mate_ms=" << gdn_non_mate_ms
        << " gdn_proj_ms=" << sums[static_cast<size_t>(Bucket::kGdnProj)]
        << " gdn_split_ms=" << sums[static_cast<size_t>(Bucket::kGdnSplit)]
        << " gdn_conv_ms=" << sums[static_cast<size_t>(Bucket::kGdnConv)]
        << " gdn_gate_ms=" << sums[static_cast<size_t>(Bucket::kGdnGate)]
        << " gdn_layout_ms=" << sums[static_cast<size_t>(Bucket::kGdnLayout)]
        << " gdn_o_proj_ms=" << sums[static_cast<size_t>(Bucket::kGdnOProj)]
        << " gdn_state_prep_ms="
        << sums[static_cast<size_t>(Bucket::kGdnStatePrep)]
        << " gdn_state_write_ms="
        << sums[static_cast<size_t>(Bucket::kGdnStateWrite)]
        << " gdn_norm_ms=" << sums[static_cast<size_t>(Bucket::kGdnNorm)]
        << " full_qkv_ms=" << sums[static_cast<size_t>(Bucket::kFullQkv)]
        << " full_prep_ms=" << sums[static_cast<size_t>(Bucket::kFullPrep)]
        << " full_fa_ms=" << sums[static_cast<size_t>(Bucket::kFullFa)]
        << " full_o_proj_ms=" << sums[static_cast<size_t>(Bucket::kFullOProj)]
        << " fp8_prepare_ms=" << sums[static_cast<size_t>(Bucket::kFp8Prepare)]
        << " fp8_quant_ms=" << sums[static_cast<size_t>(Bucket::kFp8Quant)]
        << " fp8_gemm_ms=" << sums[static_cast<size_t>(Bucket::kFp8Gemm)]
        << " n_fp8_quant=" << counts[static_cast<size_t>(Bucket::kFp8Quant)]
        << " n_fp8_gemm=" << counts[static_cast<size_t>(Bucket::kFp8Gemm)]
        << " moe_route_ms=" << sums[static_cast<size_t>(Bucket::kMoeRoute)]
        << " moe_preprocess_ms="
        << sums[static_cast<size_t>(Bucket::kMoePreprocess)]
        << " moe_gate_up_ms=" << sums[static_cast<size_t>(Bucket::kMoeGateUp)]
        << " moe_act_ms=" << sums[static_cast<size_t>(Bucket::kMoeAct)]
        << " moe_down_ms=" << sums[static_cast<size_t>(Bucket::kMoeDown)]
        << " moe_combine_ms=" << sums[static_cast<size_t>(Bucket::kMoeCombine)]
        << " moe_shared_ms=" << sums[static_cast<size_t>(Bucket::kMoeShared)]
        << " mlp_ms=" << mlp_ms
        << " mlp_gate_up_ms=" << sums[static_cast<size_t>(Bucket::kMlpGateUp)]
        << " mlp_act_ms=" << sums[static_cast<size_t>(Bucket::kMlpAct)]
        << " mlp_down_ms=" << sums[static_cast<size_t>(Bucket::kMlpDown)]
        << " norm_ms=" << sums[static_cast<size_t>(Bucket::kNorm)]
        << " n_full=" << counts[static_cast<size_t>(Bucket::kFullAttn)]
        << " n_gdn=" << counts[static_cast<size_t>(Bucket::kGdnAttn)]
        << " n_mate=" << counts[static_cast<size_t>(Bucket::kMate)];

    static const char* kNames[] = {
        "embed",          "full_attn",       "gdn_attn",       "mate",
        "mlp_gate_up",    "mlp_act",         "mlp_down",       "norm",
        "gdn_proj",       "gdn_split",       "gdn_conv",       "gdn_gate",
        "gdn_layout",     "gdn_o_proj",      "full_qkv",       "full_prep",
        "full_fa",        "full_o_proj",     "fp8_prepare",    "fp8_quant",
        "fp8_gemm",       "moe_route",       "moe_preprocess", "moe_gate_up",
        "moe_act",        "moe_down",        "moe_combine",    "moe_shared",
        "gdn_state_prep", "gdn_state_write", "gdn_norm"};
    for (size_t i = 0; i < sums.size(); ++i) {
      if (counts[i] == 0) {
        continue;
      }
      LOG(INFO) << "[PREFILL_BREAKDOWN_BUCKET] name=" << kNames[i]
                << " ms=" << sums[i] << " calls=" << counts[i]
                << " pct=" << (100.0 * sums[i] / wall_fwd_ms);
    }
    recs.clear();
  }

  class Scope final {
   public:
    explicit Scope(Bucket bucket) : bucket_(bucket), active_(enabled()) {
      if (!active_) {
        return;
      }
      std::lock_guard<std::mutex> lock(mutex());
      auto& recs = records();
      record_index_ = static_cast<int32_t>(recs.size());
      recs.emplace_back();
      auto& rec = recs.back();
      rec.bucket = bucket_;
      create_event(&rec.start);
      create_event(&rec.end);
      record_event(rec.start);
    }

    ~Scope() {
      if (!active_ || record_index_ < 0) {
        return;
      }
      std::lock_guard<std::mutex> lock(mutex());
      auto& recs = records();
      if (static_cast<size_t>(record_index_) >= recs.size()) {
        return;
      }
      auto& rec = recs[static_cast<size_t>(record_index_)];
      record_event(rec.end);
      rec.finished = true;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    Bucket bucket_;
    bool active_ = false;
    int32_t record_index_ = -1;
  };

 private:
#if defined(USE_MUSA)
  using Event = musaEvent_t;
#elif defined(USE_CUDA)
  using Event = cudaEvent_t;
#else
  using Event = void*;
#endif

  struct EventPair {
    Bucket bucket = Bucket::kEmbed;
    Event start{};
    Event end{};
    bool finished = false;

    EventPair() = default;
    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;
    EventPair(EventPair&& other) noexcept {
      bucket = other.bucket;
      start = other.start;
      end = other.end;
      finished = other.finished;
      other.start = {};
      other.end = {};
      other.finished = false;
    }
    EventPair& operator=(EventPair&& other) noexcept {
      if (this != &other) {
        destroy_event(start);
        destroy_event(end);
        bucket = other.bucket;
        start = other.start;
        end = other.end;
        finished = other.finished;
        other.start = {};
        other.end = {};
        other.finished = false;
      }
      return *this;
    }
    ~EventPair() {
      destroy_event(start);
      destroy_event(end);
    }
  };

  static void sync_device() {
#if defined(USE_MUSA)
    musaDeviceSynchronize();
#elif defined(USE_CUDA)
    cudaDeviceSynchronize();
#endif
  }

  static void create_event(Event* event) {
#if defined(USE_MUSA)
    musaEventCreateWithFlags(event, musaEventDefault);
#elif defined(USE_CUDA)
    cudaEventCreateWithFlags(event, cudaEventDefault);
#else
    *event = nullptr;
#endif
  }

  static void destroy_event(Event event) {
    if (event == Event{}) {
      return;
    }
#if defined(USE_MUSA)
    musaEventDestroy(event);
#elif defined(USE_CUDA)
    cudaEventDestroy(event);
#endif
  }

  static void record_event(Event event) {
#if defined(USE_MUSA)
    musaEventRecord(event, c10::musa::getCurrentMUSAStream().stream());
#elif defined(USE_CUDA)
    cudaEventRecord(event, /*stream=*/nullptr);
#else
    (void)event;
#endif
  }

  static float elapsed_ms(Event start, Event end) {
#if defined(USE_MUSA)
    float ms = 0.0f;
    musaEventElapsedTime(&ms, start, end);
    return ms;
#elif defined(USE_CUDA)
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    return ms;
#else
    (void)start;
    (void)end;
    return 0.0f;
#endif
  }

  static std::mutex& mutex() {
    static std::mutex m;
    return m;
  }

  static std::vector<EventPair>& records() {
    static std::vector<EventPair> r;
    return r;
  }
};

}  // namespace xllm
