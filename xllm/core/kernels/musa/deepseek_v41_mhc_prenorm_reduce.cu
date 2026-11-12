/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <glog/logging.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

#include "core/kernels/musa/deepseek_v41_mhc_prenorm.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

// MATE publishes [split,row,24] partial projections and [split,row]
// sum-of-squares. Keep both sums and normalized coefficients in FP32.
__global__ void mhc_prenorm_reduce_kernel(const float* partial,
                                          const float* square_sum,
                                          float* output,
                                          int64_t rows,
                                          int64_t splits,
                                          float hidden_width,
                                          float epsilon) {
  const int64_t row = blockIdx.x;
  const int64_t column = threadIdx.x;
  if (column >= 24) {
    return;
  }
  float projection = 0.0f;
  float squares = 0.0f;
  for (int64_t split = 0; split < splits; ++split) {
    projection =
        __fadd_rn(projection, partial[(split * rows + row) * 24 + column]);
    squares = __fadd_rn(squares, square_sum[split * rows + row]);
  }
  const float mean_square = __fdiv_rn(squares, hidden_width);
  const float inverse = rsqrtf(__fadd_rn(mean_square, epsilon));
  output[row * 24 + column] = __fmul_rn(projection, inverse);
}

}  // namespace

torch::Tensor deepseek_v41_mhc_prenorm_reduce(const torch::Tensor& partial,
                                              const torch::Tensor& square_sum,
                                              int64_t hidden_width,
                                              double epsilon) {
  CHECK(partial.defined() && square_sum.defined());
  CHECK(partial.device().type() == torch::kPrivateUse1);
  CHECK_EQ(square_sum.device(), partial.device());
  CHECK_EQ(partial.scalar_type(), torch::kFloat32);
  CHECK_EQ(square_sum.scalar_type(), torch::kFloat32);
  CHECK(partial.is_contiguous() && square_sum.is_contiguous());
  CHECK_EQ(partial.dim(), 3);
  CHECK_EQ(square_sum.dim(), 2);
  CHECK(partial.size(0) == 16 || partial.size(0) == 32);
  CHECK_EQ(partial.size(2), 24);
  CHECK_EQ(square_sum.size(0), partial.size(0));
  CHECK_EQ(square_sum.size(1), partial.size(1));
  CHECK_GE(partial.size(1), 0);
  CHECK_LE(partial.size(1), 512);
  CHECK_GT(hidden_width, 0);
  CHECK(std::isfinite(epsilon) && epsilon > 0.0);
  torch::DeviceGuard guard(partial.device());
  torch::Tensor output = torch::empty({partial.size(1), 24}, partial.options());
  if (partial.size(1) > 0) {
    mhc_prenorm_reduce_kernel<<<static_cast<uint32_t>(partial.size(1)),
                                32,
                                0,
                                current_musa_stream(partial.device())>>>(
        partial.data_ptr<float>(),
        square_sum.data_ptr<float>(),
        output.data_ptr<float>(),
        partial.size(1),
        partial.size(0),
        static_cast<float>(hidden_width),
        static_cast<float>(epsilon));
    check_musa_kernel_launch("V4.1 fused mHC prenorm reduction");
  }
  return output;
}

}  // namespace xllm::kernel::musa
