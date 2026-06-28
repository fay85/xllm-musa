/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {
namespace {

// Fill each row of page_table from CSR-format paged KV metadata.
// Supports non-contiguous page_table views (e.g. slices of persistent graph
// buffers). row_stride is page_table.stride(0) in elements.
__global__ void build_page_table_kernel(
    const int32_t* __restrict__ indptr,
    const int32_t* __restrict__ indices,
    int32_t batch_size,
    int32_t max_pages_per_row,
    int64_t row_stride,
    int32_t* __restrict__ page_table) {
  const int32_t seq_id = static_cast<int32_t>(blockIdx.x);
  if (seq_id >= batch_size) {
    return;
  }

  const int32_t start = indptr[seq_id];
  const int32_t end = indptr[seq_id + 1];
  const int32_t num_pages = end - start;

  int32_t* row = page_table + seq_id * row_stride;
  for (int32_t j = static_cast<int32_t>(threadIdx.x); j < max_pages_per_row;
       j += static_cast<int32_t>(blockDim.x)) {
    if (j < num_pages) {
      row[j] = indices[start + j];
    } else {
      row[j] = -1;
    }
  }
}

}  // namespace

void build_page_table_from_paged_kv(
    torch::Tensor& page_table,
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices) {
  CHECK(page_table.defined()) << "page_table output must be pre-allocated";
  CHECK(paged_kv_indptr.defined());
  CHECK(paged_kv_indices.defined());
  CHECK_EQ(page_table.dim(), 2);
  CHECK_EQ(paged_kv_indptr.dim(), 1);
  CHECK_EQ(paged_kv_indices.dim(), 1);
  CHECK(paged_kv_indptr.is_contiguous());
  CHECK(paged_kv_indices.is_contiguous());
  CHECK_EQ(page_table.scalar_type(), torch::kInt32);

  torch::Tensor indptr = paged_kv_indptr.scalar_type() == torch::kInt32
                             ? paged_kv_indptr
                             : paged_kv_indptr.to(torch::kInt32);
  torch::Tensor indices = paged_kv_indices.scalar_type() == torch::kInt32
                              ? paged_kv_indices
                              : paged_kv_indices.to(torch::kInt32);

  CHECK_GE(indptr.size(0), 2);
  const int32_t batch_size = static_cast<int32_t>(indptr.size(0) - 1);
  CHECK_GT(batch_size, 0);
  CHECK_EQ(page_table.size(0), batch_size)
      << "page_table batch dim must match paged_kv_indptr";
  CHECK_GT(page_table.size(1), 0);

  const int32_t max_pages_per_row =
      static_cast<int32_t>(page_table.size(1));
  const int64_t row_stride = page_table.stride(0);
  CHECK_EQ(indptr.device(), page_table.device());
  CHECK_EQ(indices.device(), page_table.device());

  const c10::cuda::OptionalCUDAGuard device_guard(page_table.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  int32_t threads = std::min<int32_t>(max_pages_per_row, 256);
  threads = std::max<int32_t>(threads, 64);

  build_page_table_kernel<<<batch_size, threads, 0, stream>>>(
      indptr.data_ptr<int32_t>(),
      indices.data_ptr<int32_t>(),
      batch_size,
      max_pages_per_row,
      row_stride,
      page_table.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
