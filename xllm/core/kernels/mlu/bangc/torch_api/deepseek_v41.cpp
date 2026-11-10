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
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"
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
  CHECK(detail::check_tensor(swa, torch::kBFloat16))
      << "swa must be contiguous BF16 on MLU";
  const torch::Device device = swa.device();
  CHECK(detail::check_tensor(compressed, torch::kBFloat16, device))
      << "compressed must be contiguous BF16 on the same device as swa";
  CHECK(detail::check_tensor(current, torch::kBFloat16, device))
      << "current must be contiguous BF16 on the same device as swa";
  CHECK(detail::check_tensor(swa_table, torch::kInt32, device))
      << "swa_table must be contiguous int32 on the same device as swa";
  CHECK(detail::check_tensor(compressed_table, torch::kInt32, device))
      << "compressed_table must be contiguous int32 on the same device as swa";
  CHECK(detail::check_tensor(topk, torch::kInt64, device))
      << "topk must be contiguous int64 on the same device as swa";
  CHECK(detail::check_tensor(output, torch::kBFloat16, device))
      << "output must be contiguous BF16 on the same device as swa";
  CHECK(detail::check_tensor(length, torch::kInt32, device))
      << "length must be contiguous int32 on the same device as swa";
  CHECK_EQ(swa.size(-1), 512);
  CHECK_EQ(current.numel(), 512);
  CHECK_EQ(length.numel(), 1);
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
