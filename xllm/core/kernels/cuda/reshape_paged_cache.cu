/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/cuda/CUDAStream.h>

#include "core/kernels/cuda/cuda_ops_api.h"
#include "core/kernels/cuda/device_utils.cuh"

namespace xllm::kernel::cuda {

template <typename SourceT, typename CacheT = SourceT>
__global__ void XLLM_KERNEL_ATTR(1024) reshape_paged_cache_kernel(
    const int* __restrict__ slot_ids,    // [n_tokens]
    const SourceT* __restrict__ keys,    // [n_tokens, n_heads, head_dim]
    const SourceT* __restrict__ values,  // [n_tokens, n_heads, head_dim]
    CacheT* __restrict__ key_cache,
    CacheT* __restrict__ value_cache,
    int64_t k_stride,
    int64_t v_stride,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  // block/token index
  const int64_t bid = blockIdx.x;
  // which slot to write to
  const int64_t slot_id = slot_ids[bid];
  if (slot_id < 0) {
    return;
  }
  // block index
  const int64_t block_idx = slot_id / block_size;
  // offset within block
  const int64_t block_offset = slot_id % block_size;
  // base index for the block in cache
  const int64_t block_base_idx = block_idx * block_size * n_kv_heads * head_dim;
  // copy value one by one for the token
  for (int64_t i = threadIdx.x; i < n_kv_heads * head_dim; i += blockDim.x) {
    const int64_t k_src_idx = bid * k_stride + i;
    const int64_t v_src_idx = bid * v_stride + i;
    // cache: [n_blocks, block_size, n_heads, head_dim]
    const int64_t head_base_idx =
        block_base_idx + block_offset * n_kv_heads * head_dim;
    // which head to write to
    const int head_idx = i / head_dim;
    // which dim within head to write to
    const int head_offset = i % head_dim;
    const int64_t dst_idx = head_base_idx + head_idx * head_dim + head_offset;
    key_cache[dst_idx] = static_cast<CacheT>(keys[k_src_idx]);
    value_cache[dst_idx] = static_cast<CacheT>(values[v_src_idx]);
  }
}

void reshape_paged_cache(
    torch::Tensor slot_ids,   // [n_tokens]
    torch::Tensor keys,       // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor values,     // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor key_cache,  // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor value_cache) {
  // keys and values should be continuous at n_kv_heads and head_dim dims
  CHECK(keys.stride(-1) == 1 && keys.stride(-2) == keys.size(-1));
  CHECK(values.stride(-1) == 1 && values.stride(-2) == values.size(-1));
  const int64_t n_tokens = keys.size(-3);
  const int64_t n_kv_heads = keys.size(-2);
  const int64_t head_dim = keys.size(-1);
  const int64_t block_size = key_cache.size(-3);
  bool materialize_cache_dtype = true;
#if defined(USE_MUSA)
  // Fuse equal-dtype MUSA projections. Preserve the legacy conversion path
  // for mixed source dtypes instead of narrowing the shared operator API.
  materialize_cache_dtype = keys.scalar_type() != values.scalar_type();
#endif
  if (materialize_cache_dtype &&
      (key_cache.scalar_type() != keys.scalar_type() ||
       value_cache.scalar_type() != values.scalar_type())) {
    CHECK(key_cache.scalar_type() == torch::kFloat8_e4m3fn ||
          key_cache.scalar_type() == torch::kFloat8_e5m2)
        << "paged cache dtype conversion only supports FP8 storage";
    CHECK_EQ(value_cache.scalar_type(), key_cache.scalar_type());
    keys = keys.to(key_cache.scalar_type());
    values = values.to(value_cache.scalar_type());
  }
  CHECK_EQ(keys.scalar_type(), values.scalar_type());
  CHECK_EQ(key_cache.scalar_type(), value_cache.scalar_type());
#if defined(USE_MUSA)
  const bool convert_to_fp8 = key_cache.scalar_type() != keys.scalar_type();
  if (convert_to_fp8) {
    CHECK(key_cache.scalar_type() == torch::kFloat8_e4m3fn ||
          key_cache.scalar_type() == torch::kFloat8_e5m2)
        << "paged cache dtype conversion only supports FP8 storage";
  }
#endif
  // Keep source strides: projection views can interleave K/V with Q and gates.
  // The scatter converts directly into cache storage without temporary tensors.
  const int64_t k_stride = keys.stride(-3);
  const int64_t v_stride = values.stride(-3);
  const int64_t n = n_kv_heads * head_dim;
  dim3 grid(n_tokens);
  dim3 block(std::min<int>(n, 1024));
#define DISPATCH_PAGED_CACHE_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(                                                        \
      TYPE,                                                                  \
      NAME,                                                                  \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) AT_DISPATCH_CASE( \
          at::ScalarType::Half, __VA_ARGS__)                                 \
          AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)            \
              AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)   \
                  AT_DISPATCH_CASE(at::ScalarType::Float8_e5m2, __VA_ARGS__))
#if defined(USE_MUSA)
  if (convert_to_fp8) {
    AT_DISPATCH_SWITCH(
        key_cache.scalar_type(),
        "reshape_paged_cache_fp8",
        AT_DISPATCH_CASE(torch::kFloat8_e4m3fn, [&] {
          using CacheT = scalar_t;
          DISPATCH_PAGED_CACHE_TYPES(
              keys.scalar_type(), "reshape_paged_cache_fp8_input", [&] {
                reshape_paged_cache_kernel<scalar_t, CacheT>
                    <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                        slot_ids.data_ptr<int>(),
                        keys.data_ptr<scalar_t>(),
                        values.data_ptr<scalar_t>(),
                        key_cache.data_ptr<CacheT>(),
                        value_cache.data_ptr<CacheT>(),
                        k_stride,
                        v_stride,
                        n_kv_heads,
                        head_dim,
                        block_size);
              });
        }) AT_DISPATCH_CASE(torch::kFloat8_e5m2, [&] {
          using CacheT = scalar_t;
          DISPATCH_PAGED_CACHE_TYPES(
              keys.scalar_type(), "reshape_paged_cache_fp8_input", [&] {
                reshape_paged_cache_kernel<scalar_t, CacheT>
                    <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                        slot_ids.data_ptr<int>(),
                        keys.data_ptr<scalar_t>(),
                        values.data_ptr<scalar_t>(),
                        key_cache.data_ptr<CacheT>(),
                        value_cache.data_ptr<CacheT>(),
                        k_stride,
                        v_stride,
                        n_kv_heads,
                        head_dim,
                        block_size);
              });
        }));
  } else
#endif
  {
    DISPATCH_PAGED_CACHE_TYPES(
        keys.scalar_type(), "reshape_paged_cache_kernel", [&] {
          reshape_paged_cache_kernel<scalar_t>
              <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                  slot_ids.data_ptr<int>(),
                  keys.data_ptr<scalar_t>(),
                  values.data_ptr<scalar_t>(),
                  key_cache.data_ptr<scalar_t>(),
                  value_cache.data_ptr<scalar_t>(),
                  k_stride,
                  v_stride,
                  n_kv_heads,
                  head_dim,
                  block_size);
        });
  }
#undef DISPATCH_PAGED_CACHE_TYPES
}

}  // namespace xllm::kernel::cuda
