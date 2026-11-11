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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace xllm {

// DeepSeek V4.1 index-plane policy (production).
//
// INT8 plane + FP16 per-token scale: the only encoding on NPU, so the INT8-only
// fused ``quant_lightning_indexer_v2_va`` op is always usable. All dtype
// resolution is centralized in ``get_dsv4_cache_policy`` so the allocator, the
// estimator and the Python write/read paths agree. MLU is the one exception:
// it keeps the index plane in the model dtype with no per-token scale.

// DeepSeek V4.1 compressor state ring length (V4.1-only).
//
// The ratio-2 compressor parks each span's trailing partial group in a
// per-sequence ring; only the final ``ratio - 1`` tokens are ever re-read, so
// the ring is fixed to 32 rows to match vllm-ascend's ``STATE_RING_ROWS`` and
// the Triton ``CACHE_SIZE``. Addressing: ``ring_block *
// kDsv41CompressStateRows + (position % kDsv41CompressStateRows)``.
constexpr int64_t kDsv41CompressStateRows = 32;

struct DeepSeekV4CachePolicy {
  torch::ScalarType index_dtype = torch::kInt8;
  int64_t index_dtype_size = 1;
  bool has_indexer_cache_scale = true;
  torch::ScalarType scale_dtype = torch::kFloat16;
  int64_t scale_dtype_size = 2;
};

// Both allocator and estimator use this model-specific choice. BF16 carries
// the original FP4 QDQ values exactly; it is not the final packed cache layout.
inline bool dsv41_original_attention_qdq_enabled([[maybe_unused]] bool is_v41) {
#if defined(USE_MUSA)
  if (!is_v41) {
    return false;
  }
  const char* value = std::getenv("XLLM_DSV41_MUSA_ATTENTION_QUANT");
  const std::string_view mode = value == nullptr ? "reference" : value;
  CHECK(mode == "reference" || mode == "original_qdq")
      << "Invalid XLLM_DSV41_MUSA_ATTENTION_QUANT: " << mode;
  return mode == "original_qdq";
#else
  return false;
#endif
}

inline DeepSeekV4CachePolicy get_dsv4_cache_policy(
    [[maybe_unused]] torch::ScalarType model_dtype,
    bool is_v41 = false) {
  DeepSeekV4CachePolicy policy;
#if defined(USE_MLU)
  // MLU keeps the index plane in the model dtype with no per-token scale.
  policy.index_dtype = model_dtype;
  policy.has_indexer_cache_scale = false;
#endif
  if (dsv41_original_attention_qdq_enabled(is_v41)) {
    CHECK_EQ(model_dtype, torch::kBFloat16)
        << "Original V4.1 attention QDQ cache requires BF16 model storage";
    policy.index_dtype = torch::kBFloat16;
    policy.has_indexer_cache_scale = false;
  }
  policy.index_dtype_size =
      static_cast<int64_t>(torch::elementSize(policy.index_dtype));
  policy.scale_dtype_size =
      policy.has_indexer_cache_scale
          ? static_cast<int64_t>(torch::elementSize(policy.scale_dtype))
          : 0;
  return policy;
}

}  // namespace xllm
