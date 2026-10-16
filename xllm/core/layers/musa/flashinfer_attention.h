/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/torch.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"
#include "layers/cuda/base_attention_impl.h"

namespace xllm {
namespace layer {

namespace musa {

inline bool is_fa3_shape_supported(int64_t head_size,
                                   int64_t num_heads,
                                   int64_t num_kv_heads) {
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return false;
  }
  const int64_t gqa_ratio = num_heads / num_kv_heads;
  return (head_size == 128 &&
          (gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 7 ||
           gqa_ratio == 8)) ||
         (head_size == 256 && (gqa_ratio == 6 || gqa_ratio == 8));
}

inline bool is_fa3_prefill_requested() {
  static const bool requested = [] {
    const char* env = std::getenv("XLLM_USE_FA3");
    return env != nullptr && std::string(env) == "1";
  }();
  return requested;
}

inline bool is_fa3_decode_requested() {
  static const bool requested = [] {
    const char* env = std::getenv("XLLM_USE_FA3_DECODE");
    if (env == nullptr) {
      env = std::getenv("XLLM_USE_FA3");
    }
    return env != nullptr && std::string(env) == "1";
  }();
  return requested;
}

inline bool should_use_fa3_prefill(torch::ScalarType query_dtype,
                                   int64_t head_size,
                                   int64_t num_heads,
                                   int64_t num_kv_heads) {
  const bool default_to_fa3 =
      query_dtype == torch::kBFloat16 &&
      is_fa3_shape_supported(head_size, num_heads, num_kv_heads);
  static const int32_t setting = [] {
    const char* env = std::getenv("XLLM_USE_FA3");
    if (env == nullptr) {
      return int32_t{-1};
    }
    return std::string(env) == "1" ? int32_t{1} : int32_t{0};
  }();
  return setting < 0 ? default_to_fa3 : setting == 1;
}

inline bool is_fa3_decode_page_size_supported(int64_t page_size) {
  return page_size == 64;
}

inline bool should_use_fa3_decode(torch::ScalarType query_dtype,
                                  int64_t head_size,
                                  int64_t num_heads,
                                  int64_t num_kv_heads,
                                  int64_t page_size) {
  if (query_dtype != torch::kBFloat16 ||
      !is_fa3_shape_supported(head_size, num_heads, num_kv_heads) ||
      !is_fa3_decode_page_size_supported(page_size)) {
    return false;
  }
  static const int32_t setting = [] {
    const char* env = std::getenv("XLLM_USE_FA3_DECODE");
    if (env == nullptr) {
      env = std::getenv("XLLM_USE_FA3");
    }
    if (env == nullptr) {
      return int32_t{1};  // default on when shape+dtype+page ok
    }
    return std::string(env) == "1" ? int32_t{1} : int32_t{0};
  }();
  return setting == 1;
}

}  // namespace musa


class FlashInferAttentionImpl final : public BaseAttentionImpl {
 public:
  FlashInferAttentionImpl(int64_t num_heads,
                          int64_t head_size,
                          float scale,
                          int64_t num_kv_heads,
                          int64_t sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      torch::Tensor& output,
      KVCache& kv_cache) override;

 private:
  void prefill_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& output,
                       std::optional<torch::Tensor>& output_lse,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache);

  void chunked_prefill_forward(const AttentionMetadata& attn_metadata,
                               torch::Tensor& query,
                               const torch::Tensor& key,
                               torch::Tensor& output,
                               std::optional<torch::Tensor>& output_lse,
                               const torch::Tensor& k_cache,
                               const torch::Tensor& v_cache);

  void decoder_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       const torch::Tensor& key,
                       torch::Tensor& output,
                       std::optional<torch::Tensor>& output_lse,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache);

 private:
  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;
  torch::Tensor prefill_lse_buf_;
  torch::Tensor decode_lse_buf_;
};

}  // namespace layer
}  // namespace xllm
