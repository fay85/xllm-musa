/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>

#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kPartitionThreads = 256;
constexpr int32_t kMaxPartitionSequences = 16;

__global__ void rebase_mate_partitions_kernel(
    int32_t* __restrict__ metadata,
    const int32_t* __restrict__ local_splits,
    int32_t* __restrict__ cumulative_splits,
    int32_t sequences,
    int32_t query_width,
    int32_t parts_per_sequence) {
  __shared__ int32_t split_prefix[kMaxPartitionSequences + 1];
  if (threadIdx.x == 0) {
    split_prefix[0] = 0;
    for (int32_t seq = 0; seq < sequences; ++seq) {
      split_prefix[seq + 1] =
          split_prefix[seq] +
          local_splits[seq * (query_width + 1) + query_width];
    }
  }
  __syncthreads();
  for (int32_t part = threadIdx.x; part < sequences * parts_per_sequence;
       part += blockDim.x) {
    const int32_t offset = (part / parts_per_sequence) * query_width;
    // Metadata fields1/3 are KV block bounds;field4 is the within-query
    // split index. Preserve them,including empty trailing scheduler parts.
    metadata[part * 8] += offset;
    metadata[part * 8 + 2] += offset;
  }
  for (int32_t row = threadIdx.x; row <= sequences * query_width;
       row += blockDim.x) {
    const int32_t seq = row / query_width;
    cumulative_splits[row] =
        seq == sequences
            ? split_prefix[seq]
            : split_prefix[seq] +
                  local_splits[seq * (query_width + 1) + row % query_width];
  }
}

}  // namespace

void deepseek_v4_rebase_mate_partitions(torch::Tensor metadata,
                                        const torch::Tensor& local_splits,
                                        torch::Tensor cumulative_splits,
                                        int64_t query_width) {
  CHECK(query_width == 7);
  CHECK(local_splits.dim() == 2 && local_splits.size(0) > 1 &&
        local_splits.size(0) <= kMaxPartitionSequences);
  const int64_t sequences = local_splits.size(0);
  CHECK(local_splits.size(1) == query_width + 1);
  CHECK(metadata.dim() == 2 && metadata.size(1) == 8 && metadata.size(0) > 0 &&
        metadata.size(0) % sequences == 0);
  CHECK(metadata.size(0) <= std::numeric_limits<int32_t>::max() / 8);
  CHECK(cumulative_splits.dim() == 1 &&
        cumulative_splits.numel() == sequences * query_width + 1);
  CHECK(!metadata.device().is_cpu());
  for (const auto& tensor : {metadata, local_splits, cumulative_splits}) {
    CHECK(tensor.scalar_type() == torch::kInt32 && tensor.is_contiguous());
    CHECK(tensor.device() == metadata.device());
  }
  const XLLMOptionalDeviceGuard guard(metadata.device());
  const musaStream_t stream = is_stream_capturing()
                                  ? c10::musa::getCurrentMUSAStream().stream()
                                  : current_musa_stream(metadata.device());
  rebase_mate_partitions_kernel<<<1, kPartitionThreads, 0, stream>>>(
      metadata.data_ptr<int32_t>(),
      local_splits.data_ptr<int32_t>(),
      cumulative_splits.data_ptr<int32_t>(),
      static_cast<int32_t>(sequences),
      static_cast<int32_t>(query_width),
      static_cast<int32_t>(metadata.size(0) / sequences));
  check_musa_kernel_launch("dsv4 per-request MATE partition rebasing");
}

}  // namespace xllm::kernel::musa
