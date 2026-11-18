/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <ATen/record_function.h>
#include <glog/logging.h>
#include <musa_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/sigmoid_gated_rms_norm.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kHeadDim = 128;
constexpr int32_t kRowsPerBlock = 8;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kValuesPerLane = 4;

union NormBf16Pack {
  uint2 vector;
  __mt_bfloat16 values[kValuesPerLane];
};

// The existing GDN H128 kernel uses one warp per row. Keep that reduction
// layout, vectorize its loads/stores, and use sigmoid rather than SiLU.
__global__ void sigmoid_gated_rms_norm_h128_kernel(
    const __mt_bfloat16* __restrict__ input,
    const __mt_bfloat16* __restrict__ gate,
    const __mt_bfloat16* __restrict__ weight,
    __mt_bfloat16* __restrict__ output,
    int64_t input_stride,
    int64_t gate_stride,
    int32_t rows,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock +
                      threadIdx.x / kWarpSize;
  if (row >= rows) {
    return;
  }
  const int32_t column = (threadIdx.x % kWarpSize) * kValuesPerLane;
  NormBf16Pack x;
  NormBf16Pack g;
  NormBf16Pack w;
  x.vector =
      *reinterpret_cast<const uint2*>(input + row * input_stride + column);
  g.vector = *reinterpret_cast<const uint2*>(gate + row * gate_stride + column);
  w.vector = *reinterpret_cast<const uint2*>(weight + column);
  float values[kValuesPerLane];
  float sum = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < kValuesPerLane; ++i) {
    values[i] = __bfloat162float(x.values[i]);
    sum += values[i] * values[i];
  }
#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xffffffffu, sum, offset, kWarpSize);
  }
  const float inverse_rms = rsqrtf(sum / kHeadDim + eps);
  NormBf16Pack result;
#pragma unroll
  for (int32_t i = 0; i < kValuesPerLane; ++i) {
    const float normalized =
        values[i] * inverse_rms * __bfloat162float(w.values[i]);
    const float sigmoid = 1.0f / (1.0f + expf(-__bfloat162float(g.values[i])));
    result.values[i] = __float2bfloat16_rn(normalized * sigmoid);
  }
  *reinterpret_cast<uint2*>(output + row * kHeadDim + column) = result.vector;
}

}  // namespace

torch::Tensor sigmoid_gated_rms_norm(const torch::Tensor& input,
                                     const torch::Tensor& gate,
                                     const torch::Tensor& weight,
                                     double eps) {
  RECORD_USER_SCOPE("kernel: sigmoid_gated_rms_norm");
  CHECK(input.device().is_privateuseone());
  CHECK_EQ(input.device(), gate.device());
  CHECK_EQ(input.device(), weight.device());
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(gate.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(gate.dim(), 2);
  CHECK_EQ(weight.dim(), 1);
  CHECK(input.sizes() == gate.sizes());
  CHECK_EQ(input.size(1), kHeadDim);
  CHECK_EQ(weight.numel(), kHeadDim);
  CHECK_EQ(input.stride(1), 1);
  CHECK_EQ(gate.stride(1), 1);
  CHECK(weight.is_contiguous());
  CHECK_GE(input.stride(0), kHeadDim);
  CHECK_GE(gate.stride(0), kHeadDim);
  CHECK_EQ(input.stride(0) % kValuesPerLane, 0);
  CHECK_EQ(gate.stride(0) % kValuesPerLane, 0);
  CHECK_EQ(reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(uint2), 0);
  CHECK_EQ(reinterpret_cast<uintptr_t>(gate.data_ptr()) % alignof(uint2), 0);
  CHECK_EQ(reinterpret_cast<uintptr_t>(weight.data_ptr()) % alignof(uint2), 0);
  CHECK(std::isfinite(eps) && eps > 0.0);
  CHECK_LE(input.size(0), std::numeric_limits<int32_t>::max());
  XLLMDeviceGuard device_guard(input.device());
  torch::Tensor output = torch::empty(input.sizes(), input.options());
  const int32_t rows = static_cast<int32_t>(input.size(0));
  if (rows == 0) {
    return output;
  }
  const int64_t blocks =
      (static_cast<int64_t>(rows) + kRowsPerBlock - 1) / kRowsPerBlock;
  sigmoid_gated_rms_norm_h128_kernel<<<static_cast<uint32_t>(blocks),
                                       kRowsPerBlock * kWarpSize,
                                       0,
                                       current_musa_stream(input.device())>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __mt_bfloat16*>(gate.data_ptr()),
      reinterpret_cast<const __mt_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr()),
      input.stride(0),
      gate.stride(0),
      rows,
      static_cast<float>(eps));
  check_musa_kernel_launch("sigmoid_gated_rms_norm");
  return output;
}

}  // namespace xllm::kernel::musa
