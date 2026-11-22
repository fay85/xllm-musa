/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace xllm::layer {

inline constexpr int64_t kDeepseekV41PrefillTileRows = 512;
inline constexpr int64_t kDeepseekV41PrefillMaxRows = 8192;

// An intermediate chunk can omit CED replay only when the next chunk has
// more than one window of rows. Move a full window past a short final cut,
// retaining the chunk's row alignment and the exact input token order.
inline size_t deepseek_v41_rebalance_final_prefill_tail(size_t computed,
                                                        size_t target,
                                                        size_t sequence,
                                                        size_t window) {
  CHECK_LE(computed, target);
  CHECK_LE(target, sequence);
  const size_t remaining = sequence - target;
  if (remaining != 0 && remaining <= window) {
    CHECK_GT(target - computed, 2 * window);
    return target - window;
  }
  return target;
}

// A worker prepares and executes its synchronous target forward on the
// same thread. No sampling selector can consume a cache-only result.
class DeepseekV41CacheOnlyPrefillGuard {
 public:
  explicit DeepseekV41CacheOnlyPrefillGuard(bool enabled) : previous_(active_) {
    active_ = enabled;
  }
  DeepseekV41CacheOnlyPrefillGuard(const DeepseekV41CacheOnlyPrefillGuard&) =
      delete;
  DeepseekV41CacheOnlyPrefillGuard& operator=(
      const DeepseekV41CacheOnlyPrefillGuard&) = delete;
  ~DeepseekV41CacheOnlyPrefillGuard() { active_ = previous_; }

  static bool active() { return active_; }
  static void retain_if(bool consensus) { active_ = active_ && consensus; }

 private:
  inline static thread_local bool active_ = false;
  bool previous_;
};

// Legacy 512-row kernels retain their exact input shapes within wide P chunks.
template <class Visit>
void deepseek_v41_for_each_prefill_tile(int64_t rows, Visit&& visit) {
  for (int64_t start = 0; start < rows; start += kDeepseekV41PrefillTileRows) {
    visit(start, std::min(rows - start, kDeepseekV41PrefillTileRows));
  }
}

// MCCL transports the candidate mask as bytes. This is an all-gather only:
// preserve its 0/1 bits and restore Bool before any logical consumer. Do not
// broaden the process group's reduction datatype contract for this case.
inline torch::Tensor deepseek_v41_tail_collective_input(
    const torch::Tensor& rows) {
  CHECK(rows.scalar_type() == torch::kBool ||
        rows.scalar_type() == torch::kInt32 ||
        rows.scalar_type() == torch::kInt64);
  const torch::Tensor contiguous = rows.contiguous();
  return rows.scalar_type() == torch::kBool ? contiguous.view(torch::kUInt8)
                                            : contiguous;
}

// Keep external prompt row indices while exposing only computed CED tail
// rows. Consumers must slice the tail before context-KV projection, and
// prompt logits outside that tail remain unsupported.
inline torch::Tensor deepseek_v41_pad_prefill_tail(const torch::Tensor& tail,
                                                   int64_t full_rows) {
  CHECK_EQ(tail.dim(), 2);
  CHECK_GT(tail.size(0), 0);
  CHECK_GE(full_rows, tail.size(0));
  torch::Tensor output =
      torch::zeros({full_rows, tail.size(1)}, tail.options());
  output.narrow(0, full_rows - tail.size(0), tail.size(0)).copy_(tail);
  return output;
}

// Metadata was gathered in the original contiguous query-rank order. CED
// changes row ownership; take this rank's slice of the *global* tail, not
// a slice of its old local query rows.
inline torch::Tensor deepseek_v41_partition_prefill_tail(
    const torch::Tensor& full,
    int64_t tail_rows,
    int64_t rank,
    int64_t world) {
  CHECK_EQ(full.dim(), 2);
  CHECK_GT(world, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world);
  CHECK_GT(tail_rows, 0);
  CHECK_GE(full.size(0), tail_rows);
  CHECK_EQ(tail_rows % world, 0);
  const int64_t count = tail_rows / world;
  return full.narrow(0, full.size(0) - tail_rows + rank * count, count)
      .contiguous();
}

}  // namespace xllm::layer
