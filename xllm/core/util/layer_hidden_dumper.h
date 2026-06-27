/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace xllm::debug {

// Debug helper for snapshotting per-layer hidden_states across eager and
// graph-mode forward passes, so the two can be diff'd to localize the layer
// where graph-mode diverges from eager-mode.
//
// Lifecycle:
//   1. Set env XLLM_DUMP_HIDDEN_DIR=/tmp/dumps (enables the dumper). Optional
//      env XLLM_DUMP_TAG=eager|graph (defaults to "run") and
//      XLLM_DUMP_STEPS=N (number of forward calls to dump; default 2).
//   2. Model forward() calls ensure_buffers() once (idempotent), then
//      record(slot, h) after the embedding output, after each decoder layer,
//      and after the final norm.
//   3. Executor (eager or graph) calls flush(actual_num_tokens) after each
//      forward / graph replay completes. This copies the GPU buffers to CPU
//      and writes them as .pt files.
//
// Graph-safety:
//   - ensure_buffers() allocates GPU tensors (must not run during capture).
//     The "first forward" warmup call hits this; subsequent capture replays
//     reuse the existing buffers.
//   - record() does only narrow()+copy_() which are captured as kernel
//     launches and replayed at inference (no allocations).
//   - flush() runs OUTSIDE the captured region. The graph executor must call
//     it after graph_.replay() (and the eager executor after forward()).
class LayerHiddenDumper {
 public:
  static LayerHiddenDumper& instance();

  bool enabled() const { return enabled_; }

  // Pre-allocate GPU buffers, one per slot, sized [max_tokens, hidden_size].
  // Idempotent: only re-allocates when shape or slot count changes.
  void ensure_buffers(int num_slots,
                      int64_t max_tokens,
                      int64_t hidden_size,
                      const torch::Device& device,
                      torch::ScalarType dtype);

  // Snapshot h into slot. h shape must be [n, hidden_size] with n<=max_tokens.
  void record(int slot, const torch::Tensor& h);

  // Save first `actual_num_tokens` rows of each buffer to
  // ${dir}/${tag}_step${N}_slot${i}.pt. Returns true if a flush occurred.
  // Honours XLLM_DUMP_STEPS (default 2) so we don't fill disk on long runs.
  bool flush(int64_t actual_num_tokens);

 private:
  LayerHiddenDumper();

  std::string dir_;
  std::string tag_;
  bool enabled_{false};
  int max_steps_to_dump_{2};
  // Grow-only reservation. The first ensure_buffers() call allocates
  // max(max_tokens, reserved_max_tokens_) rows so the captured copy_ op
  // pointers stay valid across prefill (larger tokens) <-> decode (1 token)
  // transitions. Override via XLLM_DUMP_RESERVED_TOKENS env.
  int64_t reserved_max_tokens_{256};

  std::mutex mtx_;
  int num_slots_{0};
  int64_t max_tokens_{0};
  int64_t hidden_size_{0};
  std::vector<torch::Tensor> bufs_;
  std::atomic<int> step_counter_{0};
};

}  // namespace xllm::debug
