/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>
#include <torch/types.h>

namespace xllm::kernel::mlu::bangc::detail {

inline void check_tensor_properties(const torch::Tensor& tensor,
                                    const char* name,
                                    const torch::Device& device,
                                    torch::ScalarType dtype) {
  CHECK(tensor.defined()) << name << " must be defined";
  CHECK(tensor.device() == device) << name << " must be on " << device;
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " has an unexpected dtype";
}

inline void check_tensor(const torch::Tensor& tensor,
                         const char* name,
                         const torch::Device& device,
                         torch::ScalarType dtype,
                         torch::IntArrayRef shape) {
  check_tensor_properties(tensor, name, device, dtype);
  CHECK(tensor.sizes() == shape)
      << name << " must have shape " << shape << ", got " << tensor.sizes();
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

}  // namespace xllm::kernel::mlu::bangc::detail
