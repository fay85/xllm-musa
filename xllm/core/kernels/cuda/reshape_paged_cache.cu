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
#if defined(USE_MUSA)
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <musa_fp8.h>
#endif

#include <cstdint>
#include <type_traits>

#include "core/kernels/cuda/cuda_ops_api.h"
#include "core/kernels/cuda/device_utils.cuh"

namespace xllm::kernel::cuda {

namespace {

template <typename SourceT, typename CacheT>
__device__ __forceinline__ CacheT convert_paged_cache_elem(SourceT v) {
  if constexpr (std::is_same_v<SourceT, CacheT>) {
    return v;
  } else {
    return static_cast<CacheT>(static_cast<float>(v));
  }
}

#if defined(USE_MUSA)
__device__ __forceinline__ uint32_t
fix_fp8_e4m3_exception_bytes(uint32_t result, uint64_t input_bits) {
  const uint32_t value0 = static_cast<uint32_t>(input_bits) & 0xffffu;
  const uint32_t value1 = static_cast<uint32_t>(input_bits >> 16) & 0xffffu;
  const uint32_t value2 = static_cast<uint32_t>(input_bits >> 32) & 0xffffu;
  const uint32_t value3 = static_cast<uint32_t>(input_bits >> 48);
  const uint32_t mask0 = (value0 & 0x7fffu) > 0x43e8u ? 0xffu : 0u;
  const uint32_t mask1 = (value1 & 0x7fffu) > 0x43e8u ? 0xff00u : 0u;
  const uint32_t mask2 = (value2 & 0x7fffu) > 0x43e8u ? 0xff0000u : 0u;
  const uint32_t mask3 = (value3 & 0x7fffu) > 0x43e8u ? 0xff000000u : 0u;
  const uint32_t mask = mask0 | mask1 | mask2 | mask3;
  const uint32_t sign0 =
      (value0 & 0x7fffu) > 0x7f80u ? 0u : ((value0 >> 8) & 0x80u);
  const uint32_t sign1 =
      (value1 & 0x7fffu) > 0x7f80u ? 0u : ((value1 >> 8) & 0x80u);
  const uint32_t sign2 =
      (value2 & 0x7fffu) > 0x7f80u ? 0u : ((value2 >> 8) & 0x80u);
  const uint32_t sign3 =
      (value3 & 0x7fffu) > 0x7f80u ? 0u : ((value3 >> 8) & 0x80u);
  const uint32_t nan_bytes = (sign0 | 0x7fu) | ((sign1 | 0x7fu) << 8) |
                             ((sign2 | 0x7fu) << 16) | ((sign3 | 0x7fu) << 24);
  return (result & ~mask) | (nan_bytes & mask);
}

__global__ void XLLM_KERNEL_ATTR(256) reshape_paged_cache_bf16_to_fp8_vec4_kernel(
    const int* __restrict__ slot_ids,
    const c10::BFloat16* __restrict__ keys,
    const c10::BFloat16* __restrict__ values,
    c10::Float8_e4m3fn* __restrict__ key_cache,
    c10::Float8_e4m3fn* __restrict__ value_cache,
    int64_t k_stride,
    int64_t v_stride,
    int64_t row_elems) {
  const int64_t bid = blockIdx.x;
  const int64_t slot_id = slot_ids[bid];
  if (slot_id < 0) {
    return;
  }
  const int64_t dst_base = slot_id * row_elems;
  const int64_t k_base = bid * k_stride;
  const int64_t v_base = bid * v_stride;
  for (int64_t vec_i = threadIdx.x; vec_i * 4 < row_elems; vec_i += blockDim.x) {
    const int64_t off = vec_i * 4;
    const uint64_t k_bits = *reinterpret_cast<const uint64_t*>(keys + k_base + off);
    const uint64_t v_bits = *reinterpret_cast<const uint64_t*>(values + v_base + off);
    const uint32_t kl = static_cast<uint32_t>(k_bits);
    const uint32_t ku = static_cast<uint32_t>(k_bits >> 32);
    const uint32_t vl = static_cast<uint32_t>(v_bits);
    const uint32_t vu = static_cast<uint32_t>(v_bits >> 32);
    const float4 kf = make_float4(__uint_as_float(kl << 16),
                                  __uint_as_float(kl & 0xffff0000u),
                                  __uint_as_float(ku << 16),
                                  __uint_as_float(ku & 0xffff0000u));
    const float4 vf = make_float4(__uint_as_float(vl << 16),
                                  __uint_as_float(vl & 0xffff0000u),
                                  __uint_as_float(vu << 16),
                                  __uint_as_float(vu & 0xffff0000u));
    const uint32_t kp = static_cast<uint32_t>(
        __musa_cvt_float4_to_fp8x4(kf, __MT_SATFINITE, __MT_E4M3));
    const uint32_t vp = static_cast<uint32_t>(
        __musa_cvt_float4_to_fp8x4(vf, __MT_SATFINITE, __MT_E4M3));
    *reinterpret_cast<uint32_t*>(key_cache + dst_base + off) =
        fix_fp8_e4m3_exception_bytes(kp, k_bits);
    *reinterpret_cast<uint32_t*>(value_cache + dst_base + off) =
        fix_fp8_e4m3_exception_bytes(vp, v_bits);
  }
}
#endif

}  // namespace

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
    key_cache[dst_idx] = convert_paged_cache_elem<SourceT, CacheT>(keys[k_src_idx]);
    value_cache[dst_idx] = convert_paged_cache_elem<SourceT, CacheT>(values[v_src_idx]);
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
  if (n_tokens <= 0) {
    return;
  }
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
#if defined(USE_MUSA)
  if (convert_to_fp8 &&
      keys.scalar_type() == torch::kBFloat16 &&
      key_cache.scalar_type() == torch::kFloat8_e4m3fn &&
      key_cache.is_contiguous() && value_cache.is_contiguous() &&
      n % 4 == 0 && k_stride % 4 == 0 && v_stride % 4 == 0 &&
      reinterpret_cast<uintptr_t>(keys.data_ptr<c10::BFloat16>()) %
              (sizeof(c10::BFloat16) * 4) ==
          0 &&
      reinterpret_cast<uintptr_t>(values.data_ptr<c10::BFloat16>()) %
              (sizeof(c10::BFloat16) * 4) ==
          0 &&
      reinterpret_cast<uintptr_t>(key_cache.data_ptr<c10::Float8_e4m3fn>()) %
              4 ==
          0 &&
      reinterpret_cast<uintptr_t>(value_cache.data_ptr<c10::Float8_e4m3fn>()) %
              4 ==
          0) {
    dim3 vec_block(std::min<int>(static_cast<int>(n / 4), 256));
    reshape_paged_cache_bf16_to_fp8_vec4_kernel
        <<<grid, vec_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            slot_ids.data_ptr<int>(),
            keys.data_ptr<c10::BFloat16>(),
            values.data_ptr<c10::BFloat16>(),
            key_cache.data_ptr<c10::Float8_e4m3fn>(),
            value_cache.data_ptr<c10::Float8_e4m3fn>(),
            k_stride,
            v_stride,
            n);
    return;
  }
#endif
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
