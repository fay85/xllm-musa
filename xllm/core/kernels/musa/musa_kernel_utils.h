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

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>

#include <type_traits>

#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm::kernel::musa {

using XLLMDeviceGuard = c10::DeviceGuard;
using XLLMOptionalDeviceGuard = c10::OptionalDeviceGuard;

inline musaStream_t current_musa_stream(const c10::Device& device) {
  // This target is compiled through MusaMapping. ATen producers and the
  // worker2 reference select their current execution stream through the CUDA
  // compatibility API; using c10::musa here can select a different native
  // stream and race those producers/consumers.
  return c10::cuda::getCurrentCUDAStream(device.index());
}

inline void check_musa_kernel_launch(const char* operation) {
  const musaError_t status = musaGetLastError();
  CHECK(status == musaSuccess)
      << operation << " failed: " << musaGetErrorString(status);
}

template <typename Scalar>
__device__ float musa_convert_to_float(Scalar value) {
  return static_cast<float>(value);
}

}  // namespace xllm::kernel::musa
