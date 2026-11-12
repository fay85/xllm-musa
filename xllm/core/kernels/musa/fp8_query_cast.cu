/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <glog/logging.h>
#include <musa_fp8.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tuple>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kQueryCastThreads = 256;
constexpr int64_t kQueryCastMaxBlocks = 4096;
constexpr int32_t kPackedQueryHeadDim = 256;
constexpr int32_t kPackedQueryElements = 4;
constexpr int32_t kPackedQueryHeadThreads =
    kPackedQueryHeadDim / kPackedQueryElements;
constexpr int32_t kPackedQueryLargeHeadsPerBlock = 4;
constexpr int64_t kPackedQuerySmallTokenLimit = 16;
constexpr int32_t kDequantKvThreads = 256;
constexpr int64_t kDefaultMaxPrefillScratchBlocks = 2048;  // 128K tokens @ page=64

__host__ __device__ constexpr uint16_t fp8_e4m3fn_to_bf16_bits(uint8_t bits) {
  const uint16_t sign = static_cast<uint16_t>(bits & 0x80u) << 8;
  const uint32_t exp = (bits >> 3) & 0x0fu;
  const uint32_t mant = bits & 0x07u;
  if (exp == 0u) {
    if (mant == 0u) {
      return sign;
    }
    // Subnormal E4M3FN: (-1)^s * 2^-6 * (mant / 8) = (-1)^s * mant * 2^-9.
    if (mant == 1u) {
      return sign | static_cast<uint16_t>(118u << 7);
    }
    if (mant < 4u) {
      return sign | static_cast<uint16_t>((119u << 7) | ((mant & 1u) << 6));
    }
    return sign | static_cast<uint16_t>((120u << 7) | ((mant & 3u) << 5));
  }
  if (exp == 15u && mant == 7u) {
    // E4M3FN NaN -> BF16 quiet NaN preserving sign.
    return sign | 0x7fc0u;
  }
  // Normal E4M3FN: bias 7 -> BF16 bias 127 (+120), 3-bit mantissa -> 7-bit (<<4).
  return sign | static_cast<uint16_t>(((exp + 120u) << 7) | (mant << 4));
}

static_assert(fp8_e4m3fn_to_bf16_bits(0x00u) == 0x0000u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x80u) == 0x8000u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x01u) == 0x3b00u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x02u) == 0x3b80u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x03u) == 0x3bc0u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x04u) == 0x3c00u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x07u) == 0x3c60u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x08u) == 0x3c80u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x38u) == 0x3f80u);
static_assert(fp8_e4m3fn_to_bf16_bits(0xb8u) == 0xbf80u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x7eu) == 0x43e0u);
static_assert(fp8_e4m3fn_to_bf16_bits(0x7fu) == 0x7fc0u);
static_assert(fp8_e4m3fn_to_bf16_bits(0xffu) == 0xffc0u);

__device__ __forceinline__ uint64_t lut_fp8x4_to_bf16x4(
    uint32_t packed,
    const uint16_t* __restrict__ s_lut) {
  const uint64_t b0 = s_lut[packed & 0xffu];
  const uint64_t b1 = s_lut[(packed >> 8) & 0xffu];
  const uint64_t b2 = s_lut[(packed >> 16) & 0xffu];
  const uint64_t b3 = s_lut[packed >> 24];
  return b0 | (b1 << 16) | (b2 << 32) | (b3 << 48);
}

__global__ void dequant_fp8_paged_kv_to_bf16_kernel(
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seqused_k,
    uint16_t* __restrict__ k_out,
    uint16_t* __restrict__ v_out,
    int32_t* __restrict__ ident_page_table,
    int64_t total_pool_blocks,
    int64_t scratch_blocks,
    int64_t page_table_cols,
    int64_t page_size,
    int64_t page_elems,
    bool vec8_aligned) {
  __shared__ uint16_t s_lut[256];
  s_lut[threadIdx.x] =
      fp8_e4m3fn_to_bf16_bits(static_cast<uint8_t>(threadIdx.x));
  __syncthreads();

  const int64_t logical_block = static_cast<int64_t>(blockIdx.x);
  if (logical_block == 0) {
    const int32_t clamp_max = static_cast<int32_t>(scratch_blocks - 1);
    for (int64_t col = threadIdx.x; col < page_table_cols; col += blockDim.x) {
      const int32_t idx = static_cast<int32_t>(col);
      ident_page_table[col] = idx < clamp_max ? idx : clamp_max;
    }
  }

  const int32_t total_kv = seqused_k[0];
  const int64_t active_blocks =
      total_kv > 0 ? (static_cast<int64_t>(total_kv) + page_size - 1) / page_size
                   : 0;
  if (logical_block >= active_blocks) {
    return;
  }
  const int32_t phys_block = page_table[logical_block];
  if (phys_block < 0 || static_cast<int64_t>(phys_block) >= total_pool_blocks) {
    return;
  }

  const int64_t src_base = static_cast<int64_t>(phys_block) * page_elems;
  const int64_t dst_base = logical_block * page_elems;
  if (vec8_aligned) {
    const int64_t vec8_count = page_elems >> 3;
    const uint2* __restrict__ k_src8 =
        reinterpret_cast<const uint2*>(k_cache + src_base);
    const uint2* __restrict__ v_src8 =
        reinterpret_cast<const uint2*>(v_cache + src_base);
    uint4* __restrict__ k_dst8 = reinterpret_cast<uint4*>(k_out + dst_base);
    uint4* __restrict__ v_dst8 = reinterpret_cast<uint4*>(v_out + dst_base);
    for (int64_t i = threadIdx.x; i < vec8_count; i += blockDim.x) {
      const uint2 kv_k = k_src8[i];
      const uint2 kv_v = v_src8[i];
      const uint64_t k_lo = lut_fp8x4_to_bf16x4(kv_k.x, s_lut);
      const uint64_t k_hi = lut_fp8x4_to_bf16x4(kv_k.y, s_lut);
      const uint64_t v_lo = lut_fp8x4_to_bf16x4(kv_v.x, s_lut);
      const uint64_t v_hi = lut_fp8x4_to_bf16x4(kv_v.y, s_lut);
      k_dst8[i] = make_uint4(static_cast<uint32_t>(k_lo),
                             static_cast<uint32_t>(k_lo >> 32),
                             static_cast<uint32_t>(k_hi),
                             static_cast<uint32_t>(k_hi >> 32));
      v_dst8[i] = make_uint4(static_cast<uint32_t>(v_lo),
                             static_cast<uint32_t>(v_lo >> 32),
                             static_cast<uint32_t>(v_hi),
                             static_cast<uint32_t>(v_hi >> 32));
    }
    return;
  }

  for (int64_t i = threadIdx.x; i < page_elems; i += blockDim.x) {
    k_out[dst_base + i] = s_lut[k_cache[src_base + i]];
    v_out[dst_base + i] = s_lut[v_cache[src_base + i]];
  }
}

int64_t max_prefill_scratch_blocks() {
  static const int64_t blocks = [] {
    const char* env = std::getenv("XLLM_FP8_PREFILL_SCRATCH_BLOCKS");
    if (env != nullptr) {
      const int64_t parsed = std::atoll(env);
      if (parsed > 0) {
        return parsed;
      }
    }
    return kDefaultMaxPrefillScratchBlocks;
  }();
  return blocks;
}

__device__ __forceinline__ uint32_t
fix_fp8_exception_bytes(uint32_t result, uint64_t input_bits) {
  // Hardware SATFINITE conversion is fast but clamps overflow. Restore c10's
  // unscaled E4M3FN behavior: values above the 464 tie, Inf and NaN become a
  // signed overflow NaN. Input NaNs canonicalize to positive NaN like the
  // legacy torch_musa cast; the BF16 value 464 rounds to finite 448.
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

template <int32_t HeadsPerBlock>
__global__ void fp8_query_cast_packed_kernel(
    const c10::BFloat16* __restrict__ query,
    c10::Float8_e4m3fn* __restrict__ output,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t token_stride,
    int64_t head_stride) {
  const int64_t head =
      static_cast<int64_t>(blockIdx.x) * HeadsPerBlock +
      static_cast<int32_t>(threadIdx.x) / kPackedQueryHeadThreads;
  if (head >= num_heads) {
    return;
  }
  const int64_t dim =
      (static_cast<int32_t>(threadIdx.x) % kPackedQueryHeadThreads) *
      kPackedQueryElements;
  for (int64_t token = static_cast<int64_t>(blockIdx.y); token < num_tokens;
       token += static_cast<int64_t>(gridDim.y)) {
    const int64_t input_offset =
        token * token_stride + head * head_stride + dim;
    const uint64_t input_bits =
        *reinterpret_cast<const uint64_t*>(query + input_offset);
    const uint32_t lower = static_cast<uint32_t>(input_bits);
    const uint32_t upper = static_cast<uint32_t>(input_bits >> 32);
    const float4 values = make_float4(__uint_as_float(lower << 16),
                                      __uint_as_float(lower & 0xffff0000u),
                                      __uint_as_float(upper << 16),
                                      __uint_as_float(upper & 0xffff0000u));
    const uint32_t packed = static_cast<uint32_t>(
        __musa_cvt_float4_to_fp8x4(values, __MT_SATFINITE, __MT_E4M3));
    const int64_t output_offset =
        (token * num_heads + head) * kPackedQueryHeadDim + dim;
    *reinterpret_cast<uint32_t*>(output + output_offset) =
        fix_fp8_exception_bytes(packed, input_bits);
  }
}

template <int32_t HeadsPerBlock>
void launch_packed_query_cast(const torch::Tensor& query,
                              torch::Tensor& output) {
  const int64_t num_heads = query.size(1);
  const uint32_t head_blocks =
      static_cast<uint32_t>((num_heads + HeadsPerBlock - 1) / HeadsPerBlock);
  const uint32_t token_blocks =
      static_cast<uint32_t>(std::min(query.size(0), kQueryCastMaxBlocks));
  const dim3 grid(head_blocks, token_blocks);
  fp8_query_cast_packed_kernel<HeadsPerBlock>
      <<<grid,
         kPackedQueryHeadThreads * HeadsPerBlock,
         0,
         current_musa_stream(query.device())>>>(
          query.data_ptr<c10::BFloat16>(),
          output.data_ptr<c10::Float8_e4m3fn>(),
          query.size(0),
          num_heads,
          query.stride(0),
          query.stride(1));
}

__global__ void fp8_query_cast_kernel(const c10::BFloat16* __restrict__ query,
                                      c10::Float8_e4m3fn* __restrict__ output,
                                      int64_t num_rows,
                                      int64_t num_heads,
                                      int64_t head_dim,
                                      int64_t token_stride,
                                      int64_t head_stride,
                                      int64_t dim_stride) {
  // A row is one query head. Read the projection view directly and write a
  // contiguous FP8 row without an intermediate contiguous BF16 allocation.
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < num_rows;
       row += static_cast<int64_t>(gridDim.x)) {
    const int64_t token = row / num_heads;
    const int64_t head = row % num_heads;
    const int64_t input_base = token * token_stride + head * head_stride;
    const int64_t output_base = row * head_dim;
    for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < head_dim;
         dim += static_cast<int64_t>(blockDim.x)) {
      // Use the same c10 conversion as the existing unscaled FP8 cache cast.
      // In particular, do not silently introduce finite-value saturation.
      output[output_base + dim] = static_cast<c10::Float8_e4m3fn>(
          static_cast<float>(query[input_base + dim * dim_stride]));
    }
  }
}

}  // namespace

torch::Tensor query_cast_fp8_e4m3(const torch::Tensor& query) {
  CHECK(query.defined());
  CHECK(query.device().is_privateuseone())
      << "query_cast_fp8_e4m3 requires a MUSA tensor";
  CHECK_EQ(query.dim(), 3)
      << "query_cast_fp8_e4m3 expects [tokens, heads, dim]";
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "query_cast_fp8_e4m3 requires BF16 input";
  const XLLMDeviceGuard device_guard(query.device());
  torch::Tensor output =
      torch::empty(query.sizes(), query.options().dtype(torch::kFloat8_e4m3fn));
  if (query.numel() == 0) {
    return output;
  }

  // Vector loads require aligned four-BF16 groups. Projected Q with head256
  // satisfies these conditions even when tokens or heads are interleaved.
  const bool packed_supported =
      query.size(2) == kPackedQueryHeadDim && query.stride(2) == 1 &&
      query.stride(0) % kPackedQueryElements == 0 &&
      query.stride(1) % kPackedQueryElements == 0 &&
      reinterpret_cast<uintptr_t>(query.data_ptr<c10::BFloat16>()) %
              (sizeof(c10::BFloat16) * kPackedQueryElements) ==
          0;
  if (packed_supported) {
    if (query.size(0) <= kPackedQuerySmallTokenLimit) {
      launch_packed_query_cast<1>(query, output);
    } else {
      launch_packed_query_cast<kPackedQueryLargeHeadsPerBlock>(query, output);
    }
    check_musa_kernel_launch("query_cast_fp8_e4m3_packed");
    return output;
  }

  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t num_rows = query.size(0) * num_heads;
  const uint32_t num_blocks =
      static_cast<uint32_t>(std::min(num_rows, kQueryCastMaxBlocks));
  fp8_query_cast_kernel<<<num_blocks,
                          kQueryCastThreads,
                          0,
                          current_musa_stream(query.device())>>>(
      query.data_ptr<c10::BFloat16>(),
      output.data_ptr<c10::Float8_e4m3fn>(),
      num_rows,
      num_heads,
      head_dim,
      query.stride(0),
      query.stride(1),
      query.stride(2));
  check_musa_kernel_launch("query_cast_fp8_e4m3");
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dequant_fp8_paged_kv_to_bf16(const torch::Tensor& k_cache,
                             const torch::Tensor& v_cache,
                             const torch::Tensor& page_table,
                             const torch::Tensor& seqused_k) {
  CHECK(k_cache.defined() && v_cache.defined() && page_table.defined() &&
        seqused_k.defined());
  CHECK(k_cache.device().is_privateuseone());
  CHECK_EQ(k_cache.dim(), 4);
  CHECK_EQ(k_cache.sizes(), v_cache.sizes());
  CHECK_EQ(k_cache.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(v_cache.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(page_table.dim(), 2);
  CHECK_EQ(page_table.size(0), 1)
      << "dequant_fp8_paged_kv_to_bf16 supports single-sequence prefill";
  CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() &&
        page_table.is_contiguous() && seqused_k.is_contiguous());

  const XLLMDeviceGuard device_guard(k_cache.device());
  const int64_t total_pool_blocks = k_cache.size(0);
  const int64_t page_size = k_cache.size(1);
  const int64_t n_kv_heads = k_cache.size(2);
  const int64_t head_dim = k_cache.size(3);
  const int64_t page_table_cols = page_table.size(1);
  const int64_t scratch_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(
          page_table_cols,
          std::min<int64_t>(total_pool_blocks, max_prefill_scratch_blocks())));

  const auto bf16_opts = k_cache.options().dtype(torch::kBFloat16);
  torch::Tensor k_bf16 =
      torch::empty({scratch_blocks, page_size, n_kv_heads, head_dim}, bf16_opts);
  torch::Tensor v_bf16 =
      torch::empty({scratch_blocks, page_size, n_kv_heads, head_dim}, bf16_opts);
  torch::Tensor ident_page_table = torch::empty_like(page_table);

  const int64_t page_elems = page_size * n_kv_heads * head_dim;
  const bool vec8_aligned =
      (page_elems % 8 == 0) &&
      (reinterpret_cast<uintptr_t>(k_cache.data_ptr()) % 8 == 0) &&
      (reinterpret_cast<uintptr_t>(v_cache.data_ptr()) % 8 == 0) &&
      (reinterpret_cast<uintptr_t>(k_bf16.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(v_bf16.data_ptr()) % 16 == 0);

  const dim3 grid(static_cast<uint32_t>(scratch_blocks));
  dequant_fp8_paged_kv_to_bf16_kernel<<<grid,
                                        kDequantKvThreads,
                                        0,
                                        current_musa_stream(k_cache.device())>>>(
      reinterpret_cast<const uint8_t*>(k_cache.data_ptr<c10::Float8_e4m3fn>()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr<c10::Float8_e4m3fn>()),
      page_table.data_ptr<int32_t>(),
      seqused_k.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(k_bf16.data_ptr<c10::BFloat16>()),
      reinterpret_cast<uint16_t*>(v_bf16.data_ptr<c10::BFloat16>()),
      ident_page_table.data_ptr<int32_t>(),
      total_pool_blocks,
      scratch_blocks,
      page_table_cols,
      page_size,
      page_elems,
      vec8_aligned);
  check_musa_kernel_launch("dequant_fp8_paged_kv_to_bf16");
  return {std::move(k_bf16), std::move(v_bf16), std::move(ident_page_table)};
}

}  // namespace xllm::kernel::musa
