/* Copyright 2026 The xLLM Authors.

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

#include "kernels/mlu/bangc/include/deepseek_v41.h"

#include <glog/logging.h>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/deepseek_v41/launch.h"
namespace xllm::kernel::mlu::bangc {
void deepseek_v41_decode_gather(const torch::Tensor& swa,
                                const torch::Tensor& compressed,
                                const torch::Tensor& current,
                                const torch::Tensor& swa_table,
                                const torch::Tensor& compressed_table,
                                const torch::Tensor& topk,
                                int64_t position,
                                int32_t window,
                                torch::Tensor output,
                                torch::Tensor length) {
  CHECK_EQ(swa.scalar_type(), torch::kBFloat16);
  CHECK_EQ(swa.size(-1), 512);
  CHECK(swa.is_contiguous() && compressed.is_contiguous() &&
        current.is_contiguous());
  CHECK_EQ(compressed.scalar_type(), swa.scalar_type());
  CHECK_EQ(current.scalar_type(), swa.scalar_type());
  CHECK_EQ(current.numel(), 512);
  CHECK(swa.device().is_privateuseone());
  CHECK(swa_table.device() == swa.device() &&
        compressed_table.device() == swa.device());
  CHECK(topk.device() == swa.device() && length.device() == swa.device());
  CHECK_EQ(swa_table.scalar_type(), torch::kInt32);
  CHECK_EQ(compressed_table.scalar_type(), torch::kInt32);
  CHECK_EQ(topk.scalar_type(), torch::kInt64);
  CHECK(swa_table.is_contiguous() && compressed_table.is_contiguous() &&
        topk.is_contiguous());
  CHECK_EQ(length.scalar_type(), torch::kInt32);
  CHECK_EQ(length.numel(), 1);
  CHECK(output.is_contiguous());
  CHECK_EQ(output.scalar_type(), swa.scalar_type());
  CHECK_GE(output.numel(),
           (std::min<int64_t>(position + 1, window) + topk.numel()) * 512);
  deepseek_v41_gather_launch(torch_mlu::getCurMLUStream(),
                             swa.data_ptr(),
                             compressed.data_ptr(),
                             current.data_ptr(),
                             swa_table.data_ptr<int32_t>(),
                             compressed_table.data_ptr<int32_t>(),
                             topk.data_ptr<int64_t>(),
                             topk.numel(),
                             position,
                             window,
                             swa_table.numel(),
                             output.data_ptr(),
                             length.data_ptr<int32_t>());
  // Check submission errors only: launches remain asynchronous. Device
  // execution failures surface at the caller's stream/graph synchronization
  // boundary.
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
}
}  // namespace xllm::kernel::mlu::bangc
