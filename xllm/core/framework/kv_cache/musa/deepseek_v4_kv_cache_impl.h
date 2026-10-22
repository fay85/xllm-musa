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

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "framework/kv_cache/kv_cache_impl.h"

namespace xllm {

inline constexpr int64_t kMusaDsv4Model1BytesPerToken = 584;
inline constexpr int64_t kMusaDsv4Model1PageAlignment = 576;
inline constexpr int64_t kMusaDsv4FlashMlaRopeHeadDim = 64;

// FlashMLA consumes the 512-value latent row plus a second 64-value RoPE
// view.  The MUSA scatter kernel materializes that view directly from the
// model row, so this is a cache-layout contract rather than a model dimension.
int64_t musa_dsv4_flashmla_cache_head_dim(int64_t model_head_dim);

int64_t musa_dsv4_model1_page_stride(int64_t block_size);

torch::Tensor create_musa_dsv4_model1_cache(int64_t block_count,
                                            int64_t block_size,
                                            const torch::Device& device);

std::unique_ptr<KVCacheImpl> create_musa_dsv4_kv_cache_impl(
    const DeepSeekV4KVCacheTensors& tensors);

}  // namespace xllm
