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

#include <ATen/record_function.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "core/kernels/musa/flash_mla_sparse.h"
#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {
__global__ void write_fp8_cache(const c10::BFloat16* latent,
                                const int64_t* slots,
                                uint8_t* cache,
                                int64_t capacity) {
  const int64_t slot = slots[blockIdx.x];
  if (slot < 0 || slot >= capacity) {
    return;
  }
  __shared__ float values[128];
  const int32_t lane = threadIdx.x;
  for (int32_t group = 0; group < 4; ++group) {
    const float value =
        static_cast<float>(latent[blockIdx.x * 512 + group * 128 + lane]);
    values[lane] = fabsf(value);
    __syncthreads();
    for (int32_t offset = 64; offset > 0; offset /= 2) {
      if (lane < offset) {
        values[lane] = fmaxf(values[lane], values[lane + offset]);
      }
      __syncthreads();
    }
    const float scale = fmaxf(__fdiv_rn(values[0], 448.0f), 1.0e-4f);
    if (lane == 0) {
      reinterpret_cast<float*>(cache + slot * 528 + 512)[group] = scale;
    }
    reinterpret_cast<c10::Float8_e4m3fn*>(cache +
                                          slot * 528)[group * 128 + lane] =
        QuantConverter<c10::Float8_e4m3fn>::convert(__fdiv_rn(value, scale));
    __syncthreads();
  }
}

__global__ void pack_fp8_cache(const c10::BFloat16* latent, uint8_t* packed) {
  const int64_t row = blockIdx.x;
  const int32_t group = threadIdx.x / 32;
  const int32_t lane = threadIdx.x % 32;
  const int32_t first = group * 128 + lane * 4;
  float values[4];
  float maximum = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    values[i] = static_cast<float>(latent[row * 512 + first + i]);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset, 32));
  }
  const float scale = fmaxf(__fdiv_rn(maximum, 448.0f), 1.0e-4f);
  uint8_t* dest = packed + row * 528;
  if (lane == 0) {
    reinterpret_cast<float*>(dest + 512)[group] = scale;
  }
  const float4 scaled = {__fdiv_rn(values[0], scale),
                         __fdiv_rn(values[1], scale),
                         __fdiv_rn(values[2], scale),
                         __fdiv_rn(values[3], scale)};
  *reinterpret_cast<uint32_t*>(dest + first) =
      __musa_cvt_float4_to_fp8x4(scaled, __MT_SATFINITE, __MT_E4M3);
}

__global__ void write_packed_fp8_cache(const uint8_t* packed,
                                       const int64_t* slots,
                                       uint8_t* cache,
                                       int64_t capacity) {
  const int64_t row = blockIdx.x;
  const int64_t slot = slots[row];
  if (slot < 0 || slot >= capacity) {
    return;
  }
  for (int64_t byte = threadIdx.x; byte < 528; byte += blockDim.x) {
    cache[slot * 528 + byte] = packed[row * 528 + byte];
  }
}
__global__ void write_cache(const c10::BFloat16* latent,
                            const int64_t* slots,
                            c10::BFloat16* cache,
                            int64_t capacity) {
  const int64_t slot = slots[blockIdx.x];
  if (slot < 0 || slot >= capacity) {
    return;
  }
  for (int64_t dim = threadIdx.x; dim < 512; dim += blockDim.x) {
    cache[slot * 512 + dim] = latent[blockIdx.x * 512 + dim];
  }
}

__global__ void map_indices(const int32_t* logical,
                            const int32_t* rows,
                            const int64_t* positions,
                            const int32_t* table,
                            int32_t* output,
                            int64_t queries,
                            int64_t width,
                            int64_t batch,
                            int64_t table_width,
                            int32_t page_size,
                            int32_t page_shift,
                            int64_t pages) {
  for (int64_t query = blockIdx.x; query < queries; query += gridDim.x) {
    const int64_t row = rows[query];
    const int64_t position = positions[query];
    for (int64_t col = blockIdx.y * blockDim.x + threadIdx.x; col < width;
         col += gridDim.y * blockDim.x) {
      const int64_t offset = query * width + col;
      const int32_t token = logical[offset];
      int32_t slot = -1;
      if (row >= 0 && row < batch && token >= 0 && token <= position) {
        const int32_t block =
            page_shift >= 0 ? token >> page_shift : token / page_size;
        if (block < table_width) {
          const int32_t page = table[row * table_width + block];
          if (page >= 0 && page < pages) {
            slot = page * page_size + (token - block * page_size);
          }
        }
      }
      output[offset] = slot;
    }
  }
}

void check_tensor(const torch::Tensor& tensor,
                  const torch::Tensor& anchor,
                  torch::ScalarType dtype) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), anchor.device());
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.is_contiguous());
}
}  // namespace

void sparse_mla_cache_write(const torch::Tensor& latent,
                            const torch::Tensor& slots,
                            torch::Tensor& cache) {
  RECORD_USER_SCOPE("kernel: sparse_mla_cache_write");
  CHECK(latent.device().is_privateuseone());
  check_tensor(latent, latent, torch::kBFloat16);
  check_tensor(cache, latent, torch::kBFloat16);
  check_tensor(slots, latent, torch::kInt64);
  CHECK_EQ(latent.dim(), 2);
  CHECK_EQ(latent.size(1), 512);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(slots.numel(), latent.size(0));
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.size(2), 1);
  CHECK_EQ(cache.size(3), 512);
  CHECK_GT(cache.size(1), 0);
  if (latent.size(0) == 0) {
    return;
  }
  XLLMDeviceGuard device_guard(latent.device());
  write_cache<<<latent.size(0), 128, 0, current_musa_stream(latent.device())>>>(
      latent.data_ptr<c10::BFloat16>(),
      slots.data_ptr<int64_t>(),
      cache.data_ptr<c10::BFloat16>(),
      cache.size(0) * cache.size(1));
  check_musa_kernel_launch("sparse MLA cache write");
}

void sparse_mla_fp8_cache_write(const torch::Tensor& latent,
                                const torch::Tensor& slots,
                                torch::Tensor& cache) {
  RECORD_USER_SCOPE("kernel: sparse_mla_fp8_cache_write");
  CHECK(latent.device().is_privateuseone());
  check_tensor(latent, latent, torch::kBFloat16);
  check_tensor(cache, latent, torch::kUInt8);
  check_tensor(slots, latent, torch::kInt64);
  CHECK_EQ(latent.dim(), 2);
  CHECK_EQ(latent.size(1), 512);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(slots.numel(), latent.size(0));
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.size(2), 1);
  CHECK_EQ(cache.size(3), 528);
  CHECK_GT(cache.size(1), 0);
  if (latent.size(0) == 0) {
    return;
  }
  XLLMDeviceGuard device_guard(latent.device());
  write_fp8_cache<<<latent.size(0),
                    128,
                    0,
                    current_musa_stream(latent.device())>>>(
      latent.data_ptr<c10::BFloat16>(),
      slots.data_ptr<int64_t>(),
      cache.data_ptr<uint8_t>(),
      cache.size(0) * cache.size(1));
  check_musa_kernel_launch("sparse MLA FP8 cache write");
}

torch::Tensor sparse_mla_fp8_cache_pack(const torch::Tensor& latent) {
  RECORD_USER_SCOPE("kernel: sparse_mla_fp8_cache_pack");
  CHECK(latent.device().is_privateuseone());
  check_tensor(latent, latent, torch::kBFloat16);
  CHECK_EQ(latent.dim(), 2);
  CHECK_EQ(latent.size(1), 512);
  auto packed = torch::empty({latent.size(0), 528},
                             latent.options().dtype(torch::kUInt8));
  if (latent.size(0) == 0) {
    return packed;
  }
  XLLMDeviceGuard device_guard(latent.device());
  pack_fp8_cache<<<latent.size(0),
                   128,
                   0,
                   current_musa_stream(latent.device())>>>(
      latent.data_ptr<c10::BFloat16>(), packed.data_ptr<uint8_t>());
  check_musa_kernel_launch("sparse MLA FP8 cache pack");
  return packed;
}

void sparse_mla_fp8_cache_write_packed(const torch::Tensor& packed,
                                       const torch::Tensor& slots,
                                       torch::Tensor& cache) {
  RECORD_USER_SCOPE("kernel: sparse_mla_fp8_cache_write_packed");
  CHECK(packed.device().is_privateuseone());
  check_tensor(packed, packed, torch::kUInt8);
  check_tensor(slots, packed, torch::kInt64);
  check_tensor(cache, packed, torch::kUInt8);
  CHECK_EQ(packed.dim(), 2);
  CHECK_EQ(packed.size(1), 528);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(slots.numel(), packed.size(0));
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.size(2), 1);
  CHECK_EQ(cache.size(3), 528);
  CHECK_GT(cache.size(1), 0);
  if (packed.size(0) == 0) {
    return;
  }
  XLLMDeviceGuard device_guard(packed.device());
  write_packed_fp8_cache<<<packed.size(0),
                           128,
                           0,
                           current_musa_stream(packed.device())>>>(
      packed.data_ptr<uint8_t>(),
      slots.data_ptr<int64_t>(),
      cache.data_ptr<uint8_t>(),
      cache.size(0) * cache.size(1));
  check_musa_kernel_launch("sparse MLA packed FP8 cache write");
}

torch::Tensor sparse_mla_map_indices(const torch::Tensor& logical,
                                     const torch::Tensor& rows,
                                     const torch::Tensor& positions,
                                     const torch::Tensor& block_table,
                                     int64_t page_size,
                                     int64_t num_pages) {
  RECORD_USER_SCOPE("kernel: sparse_mla_map_indices");
  CHECK(logical.device().is_privateuseone());
  check_tensor(logical, logical, torch::kInt32);
  check_tensor(rows, logical, torch::kInt32);
  check_tensor(positions, logical, torch::kInt64);
  check_tensor(block_table, logical, torch::kInt32);
  CHECK_EQ(logical.dim(), 2);
  CHECK_EQ(rows.dim(), 1);
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(rows.numel(), logical.size(0));
  CHECK_EQ(positions.numel(), logical.size(0));
  CHECK_GT(page_size, 0);
  CHECK_GT(num_pages, 0);
  CHECK_LE(num_pages, std::numeric_limits<int32_t>::max() / page_size);
  auto output = torch::empty_like(logical);
  if (logical.numel() == 0) {
    return output;
  }
  XLLMDeviceGuard device_guard(logical.device());
  const dim3 blocks(std::min<int64_t>(logical.size(0), 65535),
                    std::min<int64_t>((logical.size(1) + 255) / 256, 65535));
  const int32_t page_shift =
      (page_size & (page_size - 1)) == 0
          ? __builtin_ctz(static_cast<unsigned>(page_size))
          : -1;
  map_indices<<<blocks, 256, 0, current_musa_stream(logical.device())>>>(
      logical.data_ptr<int32_t>(),
      rows.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(),
      block_table.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      logical.size(0),
      logical.size(1),
      block_table.size(0),
      block_table.size(1),
      static_cast<int32_t>(page_size),
      page_shift,
      num_pages);
  check_musa_kernel_launch("sparse MLA logical to physical mapping");
  return output;
}
}  // namespace xllm::kernel::musa
