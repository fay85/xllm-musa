/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <glog/logging.h>

#include "core/kernels/musa/mrope.h"

namespace xllm::kernel::musa {
namespace {

template <typename ScalarType, typename IndexType>
__global__ void partial_mrope_kernel(const IndexType* positions,
                                     ScalarType* query,
                                     ScalarType* key,
                                     const ScalarType* cos_sin_cache,
                                     int64_t axis_stride,
                                     int64_t position_stride,
                                     int64_t query_stride,
                                     int64_t key_stride,
                                     int64_t query_heads,
                                     int64_t key_heads,
                                     int64_t head_size,
                                     int64_t rotary_dim,
                                     int64_t time_section,
                                     int64_t height_section,
                                     int64_t width_section,
                                     bool interleaved) {
  const int64_t token_index = blockIdx.x;
  const int64_t half_rotary_dim = rotary_dim / 2;
  const int64_t rotation_count = (query_heads + key_heads) * half_rotary_dim;
  for (int64_t rotation_index = threadIdx.x; rotation_index < rotation_count;
       rotation_index += blockDim.x) {
    const int64_t frequency_index = rotation_index % half_rotary_dim;
    int64_t axis_index = 0;
    if (interleaved) {
      const int64_t phase = frequency_index % 3;
      if (phase == 1 && frequency_index < height_section * 3) {
        axis_index = 1;
      } else if (phase == 2 && frequency_index < width_section * 3) {
        axis_index = 2;
      }
    } else if (frequency_index >= time_section + height_section) {
      axis_index = 2;
    } else if (frequency_index >= time_section) {
      axis_index = 1;
    }
    const int64_t position =
        positions[axis_index * axis_stride + token_index * position_stride];
    const float cosine = static_cast<float>(
        cos_sin_cache[position * rotary_dim + frequency_index]);
    const float sine =
        static_cast<float>(cos_sin_cache[position * rotary_dim +
                                         half_rotary_dim + frequency_index]);
    const int64_t combined_head = rotation_index / half_rotary_dim;
    ScalarType* head =
        combined_head < query_heads
            ? query + token_index * query_stride + combined_head * head_size
            : key + token_index * key_stride +
                  (combined_head - query_heads) * head_size;
    const float first = static_cast<float>(head[frequency_index]);
    const float second =
        static_cast<float>(head[frequency_index + half_rotary_dim]);
    head[frequency_index] =
        static_cast<ScalarType>(first * cosine - second * sine);
    head[frequency_index + half_rotary_dim] =
        static_cast<ScalarType>(second * cosine + first * sine);
  }
}

}  // namespace

void partial_mrope_embedding(const torch::Tensor& positions,
                             torch::Tensor& query,
                             torch::Tensor& key,
                             const torch::Tensor& cos_sin_cache,
                             int64_t head_size,
                             int64_t rotary_dim,
                             const std::vector<int64_t>& mrope_section,
                             bool interleaved) {
  CHECK_EQ(positions.dim(), 2);
  CHECK_EQ(positions.size(0), 3);
  CHECK_EQ(query.dim(), 2);
  CHECK_EQ(key.dim(), 2);
  CHECK_EQ(query.size(0), positions.size(1));
  CHECK_EQ(key.size(0), positions.size(1));
  CHECK_GT(head_size, 0);
  CHECK_GT(rotary_dim, 0);
  CHECK_LE(rotary_dim, head_size);
  CHECK_EQ(rotary_dim % 2, 0);
  CHECK_EQ(query.size(1) % head_size, 0);
  CHECK_EQ(key.size(1) % head_size, 0);
  CHECK_EQ(query.stride(1), 1);
  CHECK_EQ(key.stride(1), 1);
  CHECK_EQ(mrope_section.size(), 3);
  CHECK_EQ(mrope_section[0] + mrope_section[1] + mrope_section[2],
           rotary_dim / 2);
  CHECK_EQ(cos_sin_cache.dim(), 2);
  CHECK_EQ(cos_sin_cache.size(1), rotary_dim);
  CHECK(cos_sin_cache.is_contiguous());
  CHECK_EQ(query.scalar_type(), key.scalar_type());
  CHECK_EQ(query.scalar_type(), cos_sin_cache.scalar_type());
  CHECK_EQ(query.device(), key.device());
  CHECK_EQ(query.device(), positions.device());
  CHECK_EQ(query.device(), cos_sin_cache.device());
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  const int64_t num_tokens = positions.size(1);
  if (num_tokens == 0) {
    return;
  }
  const c10::DeviceGuard device_guard(query.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query.get_device());
  const dim3 grid(static_cast<uint32_t>(num_tokens));
  const dim3 block(256);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      query.scalar_type(),
      "partial_mrope",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            positions.scalar_type(), "mrope_positions", [&] {
              partial_mrope_kernel<scalar_t, index_t>
                  <<<grid, block, 0, stream>>>(
                      positions.data_ptr<index_t>(),
                      query.data_ptr<scalar_t>(),
                      key.data_ptr<scalar_t>(),
                      cos_sin_cache.data_ptr<scalar_t>(),
                      positions.stride(0),
                      positions.stride(1),
                      query.stride(0),
                      key.stride(0),
                      query.size(1) / head_size,
                      key.size(1) / head_size,
                      head_size,
                      rotary_dim,
                      mrope_section[0],
                      mrope_section[1],
                      mrope_section[2],
                      interleaved);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::musa
