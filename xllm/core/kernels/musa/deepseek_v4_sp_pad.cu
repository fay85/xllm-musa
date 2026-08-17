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

}  // namespace xllm::kernel::musa
