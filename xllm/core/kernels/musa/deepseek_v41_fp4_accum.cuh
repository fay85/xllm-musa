/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <cstdint>

namespace xllm::kernel::musa {

// Small-M dot products can land exactly on a BF16 halfway point. Preserve
// low FP32 summation bits before that cast rather than relaxing output tests.
class V41CompensatedAccumulator final {
 public:
  __device__ V41CompensatedAccumulator() : sum_(0.0f), correction_(0.0f) {}

  __device__ __forceinline__ void add(float value) {
    const float adjusted = value - correction_;
    const float next = sum_ + adjusted;
    correction_ = (next - sum_) - adjusted;
    sum_ = next;
  }

  __device__ __forceinline__ float warp_sum() const {
    float sum = sum_;
    float error = -correction_;
#pragma unroll
    for (int32_t delta = 16; delta > 0; delta /= 2) {
      const float other = __shfl_xor_sync(0xffffffffu, sum, delta);
      const float other_error = __shfl_xor_sync(0xffffffffu, error, delta);
      const float next = sum + other;
      const float virtual_other = next - sum;
      const float roundoff =
          (sum - (next - virtual_other)) + (other - virtual_other);
      error += other_error + roundoff;
      sum = next;
    }
    return sum + error;
  }

 private:
  float sum_;
  float correction_;
};

}  // namespace xllm::kernel::musa
