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
#include <cstdlib>
#include <string>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kStateLookupThreads = 256;
constexpr int32_t kStateWarpSize = 32;

// Graph block-table width is capacity (often 8193 columns), even for a short
// request. Scanning it in thread 0 serializes thousands of dependent loads
// for EACH compressor/indexer state restore. Reduce logical column indices,
// not physical block IDs: rolling tables can have holes and arbitrary IDs.
template <typename index_t>
__device__ int64_t last_valid_state_block(const index_t* block_table,
                                         int64_t blocks,
                                         int64_t* columns) {
  int64_t last_column = -1;
  for (int64_t column = threadIdx.x; column < blocks; column += blockDim.x) {
    if (block_table[column] >= 0) {
      last_column = column;
    }
  }
  // Max is exact for integer column indices. Reduce within each warp before
  // using shared memory, replacing the eight block-wide reduction barriers.
  const int32_t lane = threadIdx.x % kStateWarpSize;
  const int32_t warp = threadIdx.x / kStateWarpSize;
  for (int32_t offset = kStateWarpSize / 2; offset > 0; offset /= 2) {
    const int64_t other = __shfl_down_sync(0xffffffff, last_column, offset);
    last_column = max(last_column, other);
  }
  if (lane == 0) {
    columns[warp] = last_column;
  }
  __syncthreads();
  if (warp == 0) {
    last_column = lane < kStateLookupThreads / kStateWarpSize ? columns[lane] : -1;
    for (int32_t offset = kStateWarpSize / 2; offset > 0; offset /= 2) {
      const int64_t other = __shfl_down_sync(0xffffffff, last_column, offset);
      last_column = max(last_column, other);
    }
    if (lane == 0) {
      columns[0] = last_column;
    }
  }
  __syncthreads();
  return columns[0] >= 0 ? static_cast<int64_t>(block_table[columns[0]]) : -1;
}

template <typename index_t>
__global__ void deepseek_v4_state_slots_kernel(const index_t* block_table,
                                               int64_t blocks,
                                               int64_t state_rows,
                                               int64_t block_size,
                                               int64_t* output) {
  __shared__ int64_t columns[kStateLookupThreads];
  const auto last_block = last_valid_state_block(block_table, blocks, columns);
  for (int64_t row = threadIdx.x; row < state_rows; row += blockDim.x) {
    output[row] = last_block * block_size + row;
  }
}

template <typename index_t>
__device__ void state_gather_copy(const uint4* cache,
                                                const index_t* block_table,
                                                const void* input_positions,
                                                bool positions_are_int64,
                                                int64_t blocks,
                                                int64_t state_rows,
                                                int64_t block_size,
                                                int64_t vectors_per_row,
                                                int64_t* slots,
                                                uint4* output,
                                                int64_t& last_block,
                                                int64_t& source_block,
                                                int64_t* columns) {
  const auto tail_block = last_valid_state_block(block_table, blocks, columns);
  if (threadIdx.x == 0) {
    last_block = tail_block;
    source_block = last_block;
    if (input_positions != nullptr) {
      const int64_t position =
          positions_are_int64 ? static_cast<const int64_t*>(input_positions)[0]
                              : static_cast<const int32_t*>(input_positions)[0];
      if (position > 0) {
        const int64_t previous_column = (position - 1) / block_size;
        if (previous_column < blocks && block_table[previous_column] >= 0) {
          source_block = block_table[previous_column];
        }
      }
    }
  }
  __syncthreads();

  if (blockIdx.x == 0) {
    for (int64_t row = threadIdx.x; row < state_rows; row += blockDim.x) {
      slots[row] = last_block * block_size + row;
    }
  }
  if (last_block < 0) {
    return;
  }

  // A newly allocated SWA page has no recurrent state yet. Read the previous
  // token's page, but publish slots for the current page so the updated state
  // follows the sequence across the page boundary.
  const int64_t source_offset = source_block * block_size * vectors_per_row;
  const int64_t vector_count = state_rows * vectors_per_row;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < vector_count; index += gridDim.x * blockDim.x) {
    output[index] = cache[source_offset + index];
  }
}

template <typename index_t>
__global__ void deepseek_v4_state_gather_kernel(
    const uint4* cache, const index_t* block_table, const void* input_positions,
    bool positions_are_int64, int64_t blocks, int64_t state_rows,
    int64_t block_size, int64_t vectors_per_row, int64_t* slots, uint4* output) {
  __shared__ int64_t last_block;
  __shared__ int64_t source_block;
  __shared__ int64_t columns[kStateLookupThreads];
  state_gather_copy(cache, block_table, input_positions, positions_are_int64,
                    blocks, state_rows, block_size, vectors_per_row, slots,
                    output, last_block, source_block, columns);
}

struct StateGatherRole {
  const uint4* cache;
  const void* block_table;
  int64_t* slots;
  uint4* output;
  int64_t blocks;
  int64_t rows;
  int64_t block_size;
  int64_t vectors_per_row;
  bool table_is_int64;
};

struct StateGatherRoles {
  StateGatherRole roles[4];
};

struct BatchedStateGatherRole {
  StateGatherRole role;
  int64_t table_stride_bytes;
  int64_t slots_stride;
  int64_t output_stride_vectors;
};

struct BatchedStateGatherRoles {
  BatchedStateGatherRole roles[4];
};

// Preserve state_gather_copy verbatim: grid.z changes request ownership only.
// No state arithmetic, state publication, page truncation or atomics are added.
__global__ void deepseek_v4_state_gather_batched_kernel(
    BatchedStateGatherRoles roles,
    const void* positions,
    bool positions_are_int64,
    int64_t position_stride_bytes) {
  const BatchedStateGatherRole descriptor = roles.roles[blockIdx.y];
  StateGatherRole role = descriptor.role;
  role.block_table = static_cast<const char*>(role.block_table) +
                     blockIdx.z * descriptor.table_stride_bytes;
  role.slots += blockIdx.z * descriptor.slots_stride;
  role.output += blockIdx.z * descriptor.output_stride_vectors;
  const void* position = static_cast<const char*>(positions) +
                         blockIdx.z * position_stride_bytes;
  __shared__ int64_t last_block;
  __shared__ int64_t source_block;
  __shared__ int64_t columns[kStateLookupThreads];
  if (role.table_is_int64) {
    state_gather_copy(role.cache, static_cast<const int64_t*>(role.block_table),
                      position, positions_are_int64, role.blocks, role.rows,
                      role.block_size, role.vectors_per_row, role.slots,
                      role.output, last_block, source_block, columns);
  } else {
    state_gather_copy(role.cache, static_cast<const int32_t*>(role.block_table),
                      position, positions_are_int64, role.blocks, role.rows,
                      role.block_size, role.vectors_per_row, role.slots,
                      role.output, last_block, source_block, columns);
  }
}

// Each y block is an independent cache role. Retain the exact scalar lookup
// and byte-copy semantics, including holes and the previous-page restore.
__global__ void deepseek_v4_state_gather_many_kernel(
    StateGatherRoles roles, const void* positions, bool positions_are_int64) {
  const StateGatherRole role = roles.roles[blockIdx.y];
  __shared__ int64_t last_block;
  __shared__ int64_t source_block;
  __shared__ int64_t columns[kStateLookupThreads];
  if (role.table_is_int64) {
    state_gather_copy(role.cache, static_cast<const int64_t*>(role.block_table),
                      positions, positions_are_int64, role.blocks, role.rows,
                      role.block_size, role.vectors_per_row, role.slots,
                      role.output, last_block, source_block, columns);
  } else {
    state_gather_copy(role.cache, static_cast<const int32_t*>(role.block_table),
                      positions, positions_are_int64, role.blocks, role.rows,
                      role.block_size, role.vectors_per_row, role.slots,
                      role.output, last_block, source_block, columns);
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
        <<<1, kStateLookupThreads, 0, stream>>>(row.data_ptr<int32_t>(),
                              row.size(0),
                              state_rows,
                              block_size,
                              result.data_ptr<int64_t>());
  } else {
    deepseek_v4_state_slots_kernel<int64_t>
        <<<1, kStateLookupThreads, 0, stream>>>(row.data_ptr<int64_t>(),
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

void deepseek_v4_state_gather_many(
    const std::vector<torch::Tensor>& caches,
    const std::vector<torch::Tensor>& block_tables,
    const std::vector<int64_t>& state_rows,
    const std::vector<torch::Tensor>& slots,
    const std::vector<torch::Tensor>& outputs,
    int64_t sequence_index,
    const torch::Tensor& input_positions) {
  const int64_t count = static_cast<int64_t>(caches.size());
  CHECK(count > 0 && count <= 4);
  CHECK(block_tables.size() == count && state_rows.size() == count &&
        slots.size() == count && outputs.size() == count);
  CHECK(!caches[0].device().is_cpu());
  if (input_positions.defined()) {
    CHECK(input_positions.device() == caches[0].device() &&
          input_positions.dim() == 1 && input_positions.numel() == 1 &&
          input_positions.is_contiguous());
    CHECK(input_positions.scalar_type() == torch::kInt32 ||
          input_positions.scalar_type() == torch::kInt64);
  }
  StateGatherRoles packed{};
  for (int64_t i = 0; i < count; ++i) {
    const auto& cache = caches[i];
    const auto& table = block_tables[i];
    const auto& output = outputs[i];
    CHECK(cache.defined() && cache.dim() == 3 && cache.is_contiguous());
    CHECK(cache.device() == caches[0].device());
    CHECK(table.defined() && table.dim() == 2 && table.stride(1) == 1 &&
          table.device() == cache.device());
    CHECK(table.scalar_type() == torch::kInt32 ||
          table.scalar_type() == torch::kInt64);
    CHECK(sequence_index >= 0 && sequence_index < table.size(0));
    CHECK(state_rows[i] > 0 && state_rows[i] <= cache.size(1));
    CHECK(slots[i].defined() && slots[i].dim() == 1 &&
          slots[i].size(0) >= state_rows[i] && slots[i].is_contiguous() &&
          slots[i].scalar_type() == torch::kInt64 &&
          slots[i].device() == cache.device());
    CHECK(output.defined() && output.dim() == 2 && output.is_contiguous() &&
          output.size(0) >= state_rows[i] && output.size(1) == cache.size(2) &&
          output.device() == cache.device() &&
          output.scalar_type() == cache.scalar_type());
    const int64_t row_bytes = cache.size(2) * cache.element_size();
    CHECK(row_bytes % sizeof(uint4) == 0);
    packed.roles[i] = {
        reinterpret_cast<const uint4*>(cache.data_ptr()),
        static_cast<const char*>(table.data_ptr()) +
            sequence_index * table.stride(0) * table.element_size(),
        slots[i].data_ptr<int64_t>(),
        reinterpret_cast<uint4*>(output.data_ptr()),
        table.size(1), state_rows[i], cache.size(1),
        row_bytes / static_cast<int64_t>(sizeof(uint4)),
        table.scalar_type() == torch::kInt64};
  }
  int32_t copy_blocks = 1;
  const char* blocks_env = std::getenv("XLLM_DSV4_STATE_COPY_BLOCKS");
  if (blocks_env != nullptr) {
    const std::string requested(blocks_env);
    CHECK(requested == "1" || requested == "2" || requested == "4" || requested == "8");
    copy_blocks = static_cast<int32_t>(requested[0] - '0');
  }
  const XLLMOptionalDeviceGuard device_guard(at::device_of(caches[0]));
  const auto stream = current_musa_stream(caches[0].device());
  const dim3 grid(copy_blocks, count);
  deepseek_v4_state_gather_many_kernel<<<grid, kStateLookupThreads, 0, stream>>>(
      packed, input_positions.defined() ? input_positions.data_ptr() : nullptr,
      input_positions.defined() && input_positions.scalar_type() == torch::kInt64);
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess) << "DeepSeek-V4 batched state gather failed: "
                               << musaGetErrorString(status);
}

void deepseek_v4_state_gather_batched(
    const std::vector<torch::Tensor>& caches,
    const std::vector<torch::Tensor>& block_tables,
    const std::vector<int64_t>& state_rows,
    const std::vector<torch::Tensor>& slots,
    const std::vector<torch::Tensor>& outputs,
    int64_t sequence_count,
    int64_t query_width,
    const torch::Tensor& input_positions) {
  const int64_t count = static_cast<int64_t>(caches.size());
  CHECK(count > 0 && count <= 4);
  CHECK(sequence_count > 1 && sequence_count <= 16 && query_width > 0);
  CHECK(block_tables.size() == count && state_rows.size() == count &&
        slots.size() == count && outputs.size() == count);
  CHECK(caches[0].defined() && !caches[0].device().is_cpu());
  CHECK(input_positions.defined() && input_positions.dim() == 1 &&
        input_positions.is_contiguous() &&
        input_positions.numel() >= sequence_count * query_width &&
        input_positions.device() == caches[0].device());
  CHECK(input_positions.scalar_type() == torch::kInt32 ||
        input_positions.scalar_type() == torch::kInt64);
  BatchedStateGatherRoles packed{};
  for (int64_t role = 0; role < count; ++role) {
    const auto& cache = caches[role];
    const auto& table = block_tables[role];
    const auto& slot = slots[role];
    const auto& output = outputs[role];
    CHECK(cache.defined() && cache.dim() == 3 && cache.is_contiguous() &&
          cache.device() == caches[0].device());
    CHECK(table.defined() && table.dim() == 2 && table.stride(1) == 1 &&
          table.size(0) >= sequence_count && table.size(1) > 0 &&
          table.device() == cache.device());
    CHECK(table.scalar_type() == torch::kInt32 ||
          table.scalar_type() == torch::kInt64);
    CHECK(state_rows[role] > 0 && state_rows[role] <= cache.size(1));
    CHECK(slot.defined() && slot.dim() == 2 && slot.is_contiguous() &&
          slot.size(0) == sequence_count && slot.size(1) >= state_rows[role] &&
          slot.scalar_type() == torch::kInt64 && slot.device() == cache.device());
    CHECK(output.defined() && output.dim() == 3 && output.is_contiguous() &&
          output.size(0) == sequence_count &&
          output.size(1) >= state_rows[role] && output.size(2) == cache.size(2) &&
          output.scalar_type() == cache.scalar_type() &&
          output.device() == cache.device());
    const int64_t row_bytes = cache.size(2) * cache.element_size();
    CHECK(row_bytes % sizeof(uint4) == 0);
    packed.roles[role] = {
        {reinterpret_cast<const uint4*>(cache.data_ptr()), table.data_ptr(),
         slot.data_ptr<int64_t>(), reinterpret_cast<uint4*>(output.data_ptr()),
         table.size(1), state_rows[role], cache.size(1),
         row_bytes / static_cast<int64_t>(sizeof(uint4)),
         table.scalar_type() == torch::kInt64},
        table.stride(0) * static_cast<int64_t>(table.element_size()),
        slot.stride(0),
        output.stride(0) * static_cast<int64_t>(output.element_size()) /
            static_cast<int64_t>(sizeof(uint4))};
  }
  int32_t copy_blocks = 1;
  const char* blocks_env = std::getenv("XLLM_DSV4_STATE_COPY_BLOCKS");
  if (blocks_env != nullptr) {
    const std::string requested(blocks_env);
    CHECK(requested == "1" || requested == "2" || requested == "4" || requested == "8");
    copy_blocks = static_cast<int32_t>(requested[0] - '0');
  }
  const XLLMOptionalDeviceGuard device_guard(at::device_of(caches[0]));
  const musaStream_t stream = current_musa_stream(caches[0].device());
  const dim3 grid(copy_blocks, count, sequence_count);
  deepseek_v4_state_gather_batched_kernel<<<grid, kStateLookupThreads, 0, stream>>>(
      packed, input_positions.data_ptr(),
      input_positions.scalar_type() == torch::kInt64,
      query_width * static_cast<int64_t>(input_positions.element_size()));
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 multi-request state restore failed: " << musaGetErrorString(status);
}

torch::Tensor deepseek_v4_state_gather(const torch::Tensor& cache,
                                       const torch::Tensor& block_table,
                                       int64_t sequence_index,
                                       int64_t state_rows,
                                       int64_t block_size,
                                       const torch::Tensor& slots,
                                       const torch::Tensor& output,
                                       const torch::Tensor& input_positions) {
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
  if (input_positions.defined()) {
    CHECK(input_positions.device() == cache.device() &&
          input_positions.dim() == 1 && input_positions.numel() == 1 &&
          input_positions.is_contiguous());
    CHECK(input_positions.scalar_type() == torch::kInt32 ||
          input_positions.scalar_type() == torch::kInt64);
  }

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
  constexpr int64_t kThreads = kStateLookupThreads;
  int32_t copy_blocks = 1;
  const char* blocks_env = std::getenv("XLLM_DSV4_STATE_COPY_BLOCKS");
  if (blocks_env != nullptr) {
    const std::string requested(blocks_env);
    CHECK(requested == "1" || requested == "2" || requested == "4" || requested == "8")
        << "State copy blocks must be 1, 2, 4 or 8";
    copy_blocks = static_cast<int32_t>(requested[0] - '0');
  }
  if (block_table.scalar_type() == torch::kInt32) {
    deepseek_v4_state_gather_kernel<int32_t><<<copy_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(cache.data_ptr()),
        reinterpret_cast<const int32_t*>(block_row),
        input_positions.defined() ? input_positions.data_ptr() : nullptr,
        input_positions.defined() &&
            input_positions.scalar_type() == torch::kInt64,
        block_table.size(1),
        state_rows,
        block_size,
        vectors_per_row,
        slot_result.data_ptr<int64_t>(),
        reinterpret_cast<uint4*>(result.data_ptr()));
  } else {
    deepseek_v4_state_gather_kernel<int64_t><<<copy_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(cache.data_ptr()),
        reinterpret_cast<const int64_t*>(block_row),
        input_positions.defined() ? input_positions.data_ptr() : nullptr,
        input_positions.defined() &&
            input_positions.scalar_type() == torch::kInt64,
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
