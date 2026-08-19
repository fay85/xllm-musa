/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// DeepSeek-V4 prefill sequence parallelism pads the tail rank before an
// all-gather.  torch::zeros routes through MUDNN Fill and is not reliable on
// the schedule-overlap compute stream for an empty local shard.  Materialize
// the complete result with stream-ordered device copy/memset instead.

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

template <typename index_t>
__global__ void deepseek_v4_state_slots_kernel(const index_t* block_table,
                                               int64_t blocks,
                                               int64_t state_rows,
                                               int64_t block_size,
                                               int64_t* output) {
  if (threadIdx.x != 0) {
    return;
  }
  int64_t last_block = -1;
  for (int64_t column = 0; column < blocks; ++column) {
    const int64_t block = static_cast<int64_t>(block_table[column]);
    if (block >= 0) {
      last_block = block;
    }
  }
  for (int64_t row = 0; row < state_rows; ++row) {
    output[row] = last_block * block_size + row;
  }
}

template <typename index_t>
__global__ void deepseek_v4_state_gather_kernel(const uint4* cache,
                                                const index_t* block_table,
                                                int64_t blocks,
                                                int64_t state_rows,
                                                int64_t block_size,
                                                int64_t vectors_per_row,
                                                int64_t* slots,
                                                uint4* output) {
  __shared__ int64_t last_block;
  if (threadIdx.x == 0) {
    last_block = -1;
    for (int64_t column = 0; column < blocks; ++column) {
      const int64_t block = static_cast<int64_t>(block_table[column]);
      if (block >= 0) {
        last_block = block;
      }
    }
  }
  __syncthreads();

  for (int64_t row = threadIdx.x; row < state_rows; row += blockDim.x) {
    slots[row] = last_block * block_size + row;
  }
  if (last_block < 0) {
    return;
  }

  const int64_t source_offset = last_block * block_size * vectors_per_row;
  const int64_t vector_count = state_rows * vectors_per_row;
  for (int64_t index = threadIdx.x; index < vector_count; index += blockDim.x) {
    output[index] = cache[source_offset + index];
  }
}

}  // namespace

void deepseek_v4_sp_pad_local_out(const torch::Tensor& input,
                                  torch::Tensor& output) {
  CHECK(input.defined() && input.dim() > 0 && input.is_contiguous());
  CHECK(output.defined() && output.dim() == input.dim() &&
        output.is_contiguous());
  CHECK(!input.device().is_cpu());
  CHECK(input.device() == output.device());
  CHECK(input.scalar_type() == output.scalar_type());
  CHECK(output.size(0) >= input.size(0));
  for (int64_t dim = 1; dim < input.dim(); ++dim) {
    CHECK(input.size(dim) == output.size(dim));
  }

  const XLLMOptionalDeviceGuard device_guard(input.device());
  const musaStream_t stream = current_musa_stream(input.device());
  const size_t input_bytes =
      static_cast<size_t>(input.numel()) * input.element_size();
  const size_t output_bytes =
      static_cast<size_t>(output.numel()) * output.element_size();

  if (input_bytes > 0) {
    const musaError_t copy_status = musaMemcpyAsync(output.data_ptr(),
                                                    input.data_ptr(),
                                                    input_bytes,
                                                    musaMemcpyDeviceToDevice,
                                                    stream);
    CHECK(copy_status == musaSuccess) << "DeepSeek-V4 SP local copy failed: "
                                      << musaGetErrorString(copy_status);
  }
  const musaError_t zero_status =
      musaMemsetAsync(static_cast<char*>(output.data_ptr()) + input_bytes,
                      0,
                      output_bytes - input_bytes,
                      stream);
  CHECK(zero_status == musaSuccess) << "DeepSeek-V4 SP local padding failed: "
                                    << musaGetErrorString(zero_status);
}

torch::Tensor deepseek_v4_sp_pad_local(const torch::Tensor& input,
                                       int64_t padded_rows) {
  CHECK(input.defined() && input.dim() > 0 && input.is_contiguous());
  CHECK(!input.device().is_cpu());
  CHECK(padded_rows >= input.size(0));

  if (padded_rows == input.size(0)) {
    return input;
  }

  auto output_shape = input.sizes().vec();
  output_shape[0] = padded_rows;
  auto output = torch::empty(output_shape, input.options());
  deepseek_v4_sp_pad_local_out(input, output);
  return output;
}

torch::Tensor deepseek_v4_state_slots(const torch::Tensor& block_table,
                                      int64_t sequence_index,
                                      int64_t state_rows,
                                      int64_t block_size,
                                      const torch::Tensor& output) {
  CHECK(block_table.defined() && block_table.dim() == 2);
  CHECK(sequence_index >= 0 && sequence_index < block_table.size(0));
  CHECK(state_rows > 0 && block_size >= state_rows);
  CHECK(output.defined() && output.dim() == 1 && output.size(0) >= state_rows &&
        output.scalar_type() == torch::kInt64 &&
        output.device() == block_table.device() && output.is_contiguous());
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64);

  const auto row = block_table.select(/*dim=*/0, sequence_index);
  const auto result = output.narrow(/*dim=*/0, /*start=*/0, state_rows);
  const XLLMOptionalDeviceGuard device_guard(at::device_of(block_table));
  const auto stream = current_musa_stream(block_table.device());
  if (block_table.scalar_type() == torch::kInt32) {
    deepseek_v4_state_slots_kernel<int32_t>
        <<<1, 1, 0, stream>>>(row.data_ptr<int32_t>(),
                              row.size(0),
                              state_rows,
                              block_size,
                              result.data_ptr<int64_t>());
  } else {
    deepseek_v4_state_slots_kernel<int64_t>
        <<<1, 1, 0, stream>>>(row.data_ptr<int64_t>(),
                              row.size(0),
                              state_rows,
                              block_size,
                              result.data_ptr<int64_t>());
  }
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 state-slot kernel failed: " << musaGetErrorString(status);
  return result;
}

torch::Tensor deepseek_v4_state_gather(const torch::Tensor& cache,
                                       const torch::Tensor& block_table,
                                       int64_t sequence_index,
                                       int64_t state_rows,
                                       int64_t block_size,
                                       const torch::Tensor& slots,
                                       const torch::Tensor& output) {
  CHECK(cache.defined() && cache.dim() == 3 && cache.is_contiguous());
  CHECK(block_table.defined() && block_table.dim() == 2 &&
        block_table.stride(1) == 1);
  CHECK(sequence_index >= 0 && sequence_index < block_table.size(0));
  CHECK(state_rows > 0 && block_size == cache.size(1) &&
        block_size >= state_rows);
  CHECK(slots.defined() && slots.dim() == 1 && slots.size(0) >= state_rows &&
        slots.scalar_type() == torch::kInt64 &&
        slots.device() == cache.device() && slots.is_contiguous());
  CHECK(output.defined() && output.dim() == 2 && output.size(0) >= state_rows &&
        output.size(1) == cache.size(2) &&
        output.scalar_type() == cache.scalar_type() &&
        output.device() == cache.device() && output.is_contiguous());
  CHECK(block_table.device() == cache.device());
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64);

  constexpr int64_t kVectorBytes = sizeof(uint4);
  const int64_t row_bytes = cache.size(2) * cache.element_size();
  CHECK(row_bytes % kVectorBytes == 0)
      << "DeepSeek-V4 state row must be 16-byte aligned, got " << row_bytes;
  const int64_t vectors_per_row = row_bytes / kVectorBytes;
  const auto result = output.narrow(/*dim=*/0, /*start=*/0, state_rows);
  const auto slot_result = slots.narrow(/*dim=*/0, /*start=*/0, state_rows);
  const auto* block_row =
      static_cast<const char*>(block_table.data_ptr()) +
      sequence_index * block_table.stride(0) * block_table.element_size();
  const XLLMOptionalDeviceGuard device_guard(at::device_of(cache));
  const auto stream = current_musa_stream(cache.device());
  constexpr int64_t kThreads = 256;
  if (block_table.scalar_type() == torch::kInt32) {
    deepseek_v4_state_gather_kernel<int32_t><<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(cache.data_ptr()),
        reinterpret_cast<const int32_t*>(block_row),
        block_table.size(1),
        state_rows,
        block_size,
        vectors_per_row,
        slot_result.data_ptr<int64_t>(),
        reinterpret_cast<uint4*>(result.data_ptr()));
  } else {
    deepseek_v4_state_gather_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(cache.data_ptr()),
        reinterpret_cast<const int64_t*>(block_row),
        block_table.size(1),
        state_rows,
        block_size,
        vectors_per_row,
        slot_result.data_ptr<int64_t>(),
        reinterpret_cast<uint4*>(result.data_ptr()));
  }
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess) << "DeepSeek-V4 state-gather kernel failed: "
                               << musaGetErrorString(status);
  return result;
}

}  // namespace xllm::kernel::musa
