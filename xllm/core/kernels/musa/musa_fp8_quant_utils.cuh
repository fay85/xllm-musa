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

#include <c10/util/Float8_e4m3fn.h>
#include <musa_fp8.h>

#include <limits>
#include <type_traits>

namespace xllm::kernel::musa {

template <typename QuantType,
          typename =
              std::enable_if_t<std::is_same_v<QuantType, c10::Float8_e4m3fn> ||
                               std::is_same_v<QuantType, int8_t>>>
struct QuantTypeMax {
  static constexpr QuantType value() {
    return std::numeric_limits<QuantType>::max();
  }
};

template <typename QuantType>
struct QuantConverter {
  __device__ __forceinline__ static QuantType convert(float value) {
    return static_cast<QuantType>(value);
  }
};

template <>
struct QuantConverter<c10::Float8_e4m3fn> {
  __device__ __forceinline__ static c10::Float8_e4m3fn convert(float value) {
    const auto bits = static_cast<uint8_t>(
        __musa_cvt_float_to_fp8(value, __MT_SATFINITE, __MT_E4M3));
    return c10::Float8_e4m3fn(bits, c10::Float8_e4m3fn::from_bits());
  }
};

template <bool IsScaleInverted, typename QuantType>
__device__ __forceinline__ QuantType scaled_fp8_conversion(float value,
                                                           float scale) {
  const float scaled = IsScaleInverted ? value * scale : value / scale;
  const float maximum = static_cast<float>(QuantTypeMax<QuantType>::value());
  const float saturated = fmaxf(-maximum, fminf(scaled, maximum));
  return QuantConverter<QuantType>::convert(saturated);
}

}  // namespace xllm::kernel::musa
