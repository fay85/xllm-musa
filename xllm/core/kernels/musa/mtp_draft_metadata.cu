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

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kNextDraftThreads = 128;
constexpr int32_t kAcceptThreads = 128;
constexpr int64_t kRowsPerSequence = 2;
constexpr int64_t kVectorBytes = 16;
constexpr int32_t kPagedKvThreads = 256;
constexpr int64_t kMaxPagedKvRows = 256;
constexpr int64_t kMaxPagedKvBlocks = 64;

// torch::floor_divide and torch::remainder round toward negative infinity.
__device__ __forceinline__ int64_t floor_div(int64_t value, int64_t divisor) {
  const int64_t quotient = value / divisor;
  const bool round_down = value % divisor != 0 && (value < 0) != (divisor < 0);
  return round_down ? quotient - 1 : quotient;
}

__device__ __forceinline__ int64_t floor_mod(int64_t value, int64_t divisor) {
  return value - floor_div(value, divisor) * divisor;
}

// An int32 or int64 integer tensor.
struct IndexBuffer {
  void* data;
  bool is_int64;
};

__device__ __forceinline__ int64_t load_index(const IndexBuffer& buffer,
                                              int64_t index) {
  return buffer.is_int64 ? static_cast<const int64_t*>(buffer.data)[index]
                         : static_cast<const int32_t*>(buffer.data)[index];
}

__device__ __forceinline__ void store_index(const IndexBuffer& buffer,
                                            int64_t index,
                                            int64_t value) {
  if (buffer.is_int64) {
    static_cast<int64_t*>(buffer.data)[index] = value;
  } else {
    static_cast<int32_t*>(buffer.data)[index] = static_cast<int32_t>(value);
  }
}

// One block per output row; rows alternate [repair, current] per sequence.
__global__ void __launch_bounds__(kNextDraftThreads)
    mtp_next_draft_inputs_kernel(
        IndexBuffer accepted_tokens,
        const uint4* __restrict__ accepted_embeddings,
        const uint4* __restrict__ embedding_placeholder,
        IndexBuffer base_positions,
        IndexBuffer base_kv_seq_lens,
        const int32_t* __restrict__ block_tables,
        int64_t block_table_stride,
        int64_t blocks_per_sequence,
        int32_t width,
        int64_t embedding_vectors,
        int64_t block_size,
        IndexBuffer token_ids,
        uint4* __restrict__ embeddings,
        IndexBuffer positions,
        IndexBuffer kv_seq_lens,
        int32_t* __restrict__ cache_slots) {
  // Accepted embedding copied into this row; -1 selects the placeholder.
  __shared__ int64_t source_index;
  const int64_t row = blockIdx.x;
  const int64_t sequence = row / kRowsPerSequence;
  const bool is_current = row % kRowsPerSequence == 1;
  if (threadIdx.x == 0) {
    const int64_t first_token = sequence * width;
    int64_t accepted = 0;
    for (int32_t i = 0; i < width; ++i) {
      accepted += load_index(accepted_tokens, first_token + i) >= 0 ? 1 : 0;
    }
    const int64_t last_index = accepted > 0 ? accepted - 1 : 0;
    const int64_t previous_index = accepted > 1 ? accepted - 2 : 0;
    const bool has_previous = accepted > 1;
    const int64_t position = load_index(base_positions, sequence) + accepted;
    const int64_t kv_seq_len =
        load_index(base_kv_seq_lens, sequence) + accepted;
    int64_t token = load_index(accepted_tokens, first_token + last_index);
    int64_t cache_position = position;
    int64_t row_offset = 0;
    int64_t embedding_index = last_index;
    if (!is_current) {
      if (has_previous) {
        token = load_index(accepted_tokens, first_token + previous_index);
      }
      // After a rejection the repair row writes a future scratch slot so it
      // cannot overwrite valid draft KV.
      cache_position = accepted == width ? position - 1 : position + 1;
      row_offset = -1;
      embedding_index = has_previous ? previous_index : -1;
    }
    const int64_t block = floor_div(cache_position, block_size);
    int64_t slot = -1;
    if (block >= 0 && block < blocks_per_sequence) {
      slot = static_cast<int64_t>(
                 block_tables[sequence * block_table_stride + block]) *
                 block_size +
             floor_mod(cache_position, block_size);
    }
    store_index(token_ids, row, token);
    store_index(positions, row, position + row_offset);
    store_index(kv_seq_lens, row, kv_seq_len + row_offset);
    cache_slots[row] = static_cast<int32_t>(slot);
    source_index = embedding_index;
  }
  __syncthreads();
  const uint4* source =
      source_index < 0
          ? embedding_placeholder
          : accepted_embeddings +
                (sequence * width + source_index) * embedding_vectors;
  uint4* destination = embeddings + row * embedding_vectors;
  for (int64_t i = threadIdx.x; i < embedding_vectors; i += blockDim.x) {
    destination[i] = source[i];
  }
}

// One thread per sequence.
__global__ void __launch_bounds__(kAcceptThreads)
    greedy_accepted_tokens_kernel(IndexBuffer draft_token_ids,
                                  IndexBuffer target_token_ids,
                                  int64_t batch,
                                  int32_t width,
                                  IndexBuffer accepted_token_ids) {
  const int64_t sequence =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sequence >= batch) {
    return;
  }
  const int32_t num_drafts = width - 1;
  int32_t first_rejected = num_drafts;
  for (int32_t i = 0; i < num_drafts; ++i) {
    if (load_index(draft_token_ids, sequence * num_drafts + i) !=
        load_index(target_token_ids, sequence * width + i)) {
      first_rejected = i;
      break;
    }
  }
  for (int32_t i = 0; i < width; ++i) {
    const int64_t index = sequence * width + i;
    store_index(accepted_token_ids,
                index,
                i <= first_rejected ? load_index(target_token_ids, index) : -1);
  }
}

// Every block rebuilds the page prefix sums of all rows, so no block waits for
// another; block 0 also writes the per-row outputs.
__global__ void __launch_bounds__(kPagedKvThreads)
    draft_paged_kv_metadata_kernel(const int32_t* __restrict__ kv_seq_lens,
                                   const int32_t* __restrict__ block_tables,
                                   int64_t block_table_stride,
                                   int64_t blocks_per_row,
                                   int32_t rows,
                                   int64_t num_indices,
                                   int32_t block_size,
                                   int32_t* __restrict__ kv_cu_seq_lens,
                                   int32_t* __restrict__ paged_kv_indptr,
                                   int32_t* __restrict__ paged_kv_last_page_len,
                                   int32_t* __restrict__ paged_kv_indices) {
  __shared__ int32_t indptr[kMaxPagedKvRows + 1];
  if (threadIdx.x == 0) {
    const bool writes_rows = blockIdx.x == 0;
    int64_t page_sum = 0;
    int64_t len_sum = 0;
    indptr[0] = 0;
    for (int32_t row = 0; row < rows; ++row) {
      const int32_t len = kv_seq_lens[row];
      const int32_t pages =
          static_cast<int32_t>(floor_div(len + (block_size - 1), block_size));
      page_sum += pages;
      len_sum += len;
      indptr[row + 1] = static_cast<int32_t>(page_sum);
      if (writes_rows) {
        kv_cu_seq_lens[row + 1] = static_cast<int32_t>(len_sum);
        paged_kv_indptr[row + 1] = static_cast<int32_t>(page_sum);
        paged_kv_last_page_len[row] = len - (pages - 1) * block_size;
      }
    }
    if (writes_rows) {
      kv_cu_seq_lens[0] = 0;
      paged_kv_indptr[0] = 0;
    }
  }
  __syncthreads();
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_indices;
       index += stride) {
    // Upper-bound search over indptr[1..rows], as torch::searchsorted(right).
    int32_t start = 0;
    int32_t end = rows;
    while (start < end) {
      const int32_t mid = start + ((end - start) >> 1);
      if (!(index < indptr[mid + 1])) {
        start = mid + 1;
      } else {
        end = mid;
      }
    }
    const int32_t row = start < rows - 1 ? start : rows - 1;
    int64_t local = index - indptr[row];
    local = local < 0 ? 0 : local;
    local = local > blocks_per_row - 1 ? blocks_per_row - 1 : local;
    paged_kv_indices[index] = block_tables[row * block_table_stride + local];
  }
}

bool is_16b_aligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kVectorBytes == 0;
}

bool is_index_type(torch::ScalarType type) {
  return type == torch::kInt || type == torch::kLong;
}

IndexBuffer index_buffer(const torch::Tensor& tensor) {
  return IndexBuffer{tensor.data_ptr(), tensor.scalar_type() == torch::kLong};
}

}  // namespace

bool mtp_next_draft_inputs_supported(const torch::Tensor& accepted_tokens,
                                     const torch::Tensor& accepted_embeddings,
                                     const torch::Tensor& embedding_placeholder,
                                     const torch::Tensor& base_positions,
                                     const torch::Tensor& base_kv_seq_lens,
                                     const torch::Tensor& block_tables,
                                     const MtpNextDraftDtypes& dtypes) {
  if (!is_index_type(dtypes.token_ids) || !is_index_type(dtypes.positions) ||
      !is_index_type(dtypes.kv_seq_lens) || !accepted_tokens.defined() ||
      !accepted_embeddings.defined() || !embedding_placeholder.defined() ||
      !base_positions.defined() || !base_kv_seq_lens.defined() ||
      !block_tables.defined() || accepted_tokens.is_cpu() ||
      accepted_tokens.dim() != 2 || accepted_embeddings.dim() != 3 ||
      block_tables.dim() != 2) {
    return false;
  }
  const torch::Device device = accepted_tokens.device();
  const int64_t batch = accepted_tokens.size(0);
  const int64_t width = accepted_tokens.size(1);
  const int64_t hidden = accepted_embeddings.size(2);
  const bool placeholder_is_row =
      embedding_placeholder.dim() == 1 ||
      (embedding_placeholder.dim() == 2 && embedding_placeholder.size(0) == 1);
  return batch > 0 && width > 0 && hidden > 0 &&
         is_index_type(accepted_tokens.scalar_type()) &&
         accepted_tokens.is_contiguous() &&
         accepted_embeddings.size(0) == batch &&
         accepted_embeddings.size(1) == width &&
         accepted_embeddings.device() == device &&
         accepted_embeddings.is_contiguous() &&
         hidden * accepted_embeddings.element_size() % kVectorBytes == 0 &&
         is_16b_aligned(accepted_embeddings) && placeholder_is_row &&
         embedding_placeholder.numel() == hidden &&
         embedding_placeholder.scalar_type() ==
             accepted_embeddings.scalar_type() &&
         embedding_placeholder.device() == device &&
         embedding_placeholder.is_contiguous() &&
         is_16b_aligned(embedding_placeholder) &&
         is_index_type(base_positions.scalar_type()) &&
         base_positions.device() == device && base_positions.is_contiguous() &&
         base_positions.numel() >= batch &&
         is_index_type(base_kv_seq_lens.scalar_type()) &&
         base_kv_seq_lens.device() == device &&
         base_kv_seq_lens.is_contiguous() &&
         base_kv_seq_lens.numel() >= batch &&
         block_tables.scalar_type() == torch::kInt &&
         block_tables.device() == device && block_tables.size(0) >= batch &&
         block_tables.size(1) > 0 && block_tables.stride(1) == 1;
}

MtpNextDraftInputs mtp_next_draft_inputs(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    const torch::Tensor& block_tables,
    int64_t block_size,
    const MtpNextDraftDtypes& dtypes) {
  CHECK(mtp_next_draft_inputs_supported(accepted_tokens,
                                        accepted_embeddings,
                                        embedding_placeholder,
                                        base_positions,
                                        base_kv_seq_lens,
                                        block_tables,
                                        dtypes))
      << "unsupported MTP next-draft inputs: accepted_tokens="
      << accepted_tokens.sizes()
      << ", accepted_embeddings=" << accepted_embeddings.sizes()
      << ", block_tables=" << block_tables.sizes();
  CHECK_GT(block_size, 0);
  const int64_t batch = accepted_tokens.size(0);
  const int64_t rows = batch * kRowsPerSequence;
  const int64_t hidden = accepted_embeddings.size(2);
  const torch::TensorOptions options = accepted_tokens.options();
  MtpNextDraftInputs inputs;
  inputs.token_ids = torch::empty({rows}, options.dtype(dtypes.token_ids));
  inputs.embeddings =
      torch::empty({rows, hidden}, accepted_embeddings.options());
  inputs.positions = torch::empty({rows}, options.dtype(dtypes.positions));
  inputs.kv_seq_lens = torch::empty({rows}, options.dtype(dtypes.kv_seq_lens));
  inputs.cache_slots = torch::empty({rows}, options.dtype(torch::kInt));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(accepted_tokens));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  mtp_next_draft_inputs_kernel<<<static_cast<uint32_t>(rows),
                                 kNextDraftThreads,
                                 0,
                                 stream>>>(
      index_buffer(accepted_tokens),
      reinterpret_cast<const uint4*>(accepted_embeddings.data_ptr()),
      reinterpret_cast<const uint4*>(embedding_placeholder.data_ptr()),
      index_buffer(base_positions),
      index_buffer(base_kv_seq_lens),
      block_tables.data_ptr<int32_t>(),
      block_tables.stride(0),
      block_tables.size(1),
      static_cast<int32_t>(accepted_tokens.size(1)),
      hidden * accepted_embeddings.element_size() / kVectorBytes,
      block_size,
      index_buffer(inputs.token_ids),
      reinterpret_cast<uint4*>(inputs.embeddings.data_ptr()),
      index_buffer(inputs.positions),
      index_buffer(inputs.kv_seq_lens),
      inputs.cache_slots.data_ptr<int32_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  return inputs;
}

bool greedy_accepted_tokens_supported(const torch::Tensor& draft_token_ids,
                                      const torch::Tensor& target_token_ids) {
  return draft_token_ids.defined() && target_token_ids.defined() &&
         !target_token_ids.is_cpu() &&
         draft_token_ids.device() == target_token_ids.device() &&
         draft_token_ids.dim() == 2 && target_token_ids.dim() == 2 &&
         target_token_ids.size(0) > 0 && target_token_ids.size(1) > 1 &&
         draft_token_ids.size(0) == target_token_ids.size(0) &&
         draft_token_ids.size(1) == target_token_ids.size(1) - 1 &&
         is_index_type(draft_token_ids.scalar_type()) &&
         is_index_type(target_token_ids.scalar_type()) &&
         draft_token_ids.is_contiguous() && target_token_ids.is_contiguous();
}

torch::Tensor greedy_accepted_tokens(const torch::Tensor& draft_token_ids,
                                     const torch::Tensor& target_token_ids) {
  CHECK(greedy_accepted_tokens_supported(draft_token_ids, target_token_ids))
      << "unsupported greedy acceptance: draft=" << draft_token_ids.sizes()
      << ", target=" << target_token_ids.sizes();
  torch::Tensor accepted_token_ids = torch::empty_like(target_token_ids);
  const int64_t batch = target_token_ids.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(target_token_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (batch + kAcceptThreads - 1) / kAcceptThreads;
  greedy_accepted_tokens_kernel<<<static_cast<uint32_t>(blocks),
                                  kAcceptThreads,
                                  0,
                                  stream>>>(
      index_buffer(draft_token_ids),
      index_buffer(target_token_ids),
      batch,
      static_cast<int32_t>(target_token_ids.size(1)),
      index_buffer(accepted_token_ids));
  C10_CUDA_CHECK(cudaGetLastError());
  return accepted_token_ids;
}

bool draft_paged_kv_metadata_supported(const torch::Tensor& kv_seq_lens,
                                       const torch::Tensor& block_tables,
                                       int64_t rows) {
  return rows > 0 && rows <= kMaxPagedKvRows && kv_seq_lens.defined() &&
         !kv_seq_lens.is_cpu() && kv_seq_lens.dim() == 1 &&
         kv_seq_lens.numel() == rows &&
         kv_seq_lens.scalar_type() == torch::kInt &&
         kv_seq_lens.is_contiguous() && block_tables.defined() &&
         block_tables.dim() == 2 && block_tables.scalar_type() == torch::kInt &&
         block_tables.device() == kv_seq_lens.device() &&
         block_tables.size(0) >= rows && block_tables.size(1) > 0 &&
         block_tables.stride(1) == 1;
}

DraftPagedKvMetadata draft_paged_kv_metadata(const torch::Tensor& kv_seq_lens,
                                             const torch::Tensor& block_tables,
                                             int64_t rows,
                                             int64_t num_indices,
                                             int64_t block_size) {
  CHECK(draft_paged_kv_metadata_supported(kv_seq_lens, block_tables, rows))
      << "unsupported draft paged-KV metadata: kv_seq_lens="
      << kv_seq_lens.sizes() << ", block_tables=" << block_tables.sizes()
      << ", rows=" << rows;
  CHECK_GT(num_indices, 0);
  CHECK_GT(block_size, 0);
  const torch::TensorOptions options = kv_seq_lens.options();
  DraftPagedKvMetadata metadata;
  metadata.kv_cu_seq_lens = torch::empty({rows + 1}, options);
  metadata.paged_kv_indptr = torch::empty({rows + 1}, options);
  metadata.paged_kv_last_page_len = torch::empty({rows}, options);
  metadata.paged_kv_indices = torch::empty({num_indices}, options);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_seq_lens));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks =
      std::clamp<int64_t>((num_indices + kPagedKvThreads - 1) / kPagedKvThreads,
                          1,
                          kMaxPagedKvBlocks);
  draft_paged_kv_metadata_kernel<<<static_cast<uint32_t>(blocks),
                                   kPagedKvThreads,
                                   0,
                                   stream>>>(
      kv_seq_lens.data_ptr<int32_t>(),
      block_tables.data_ptr<int32_t>(),
      block_tables.stride(0),
      block_tables.size(1),
      static_cast<int32_t>(rows),
      num_indices,
      static_cast<int32_t>(block_size),
      metadata.kv_cu_seq_lens.data_ptr<int32_t>(),
      metadata.paged_kv_indptr.data_ptr<int32_t>(),
      metadata.paged_kv_last_page_len.data_ptr<int32_t>(),
      metadata.paged_kv_indices.data_ptr<int32_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  return metadata;
}

}  // namespace xllm::kernel::musa
