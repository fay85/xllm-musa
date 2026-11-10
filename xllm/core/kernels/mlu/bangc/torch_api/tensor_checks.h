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

#include <torch/types.h>

#include <optional>

namespace xllm::kernel::mlu::bangc::detail {

// Shape, supported dtype sets and kernel-specific strides belong to the caller.
[[nodiscard]] inline bool check_tensor(
    const torch::Tensor& tensor,
    std::optional<torch::ScalarType> dtype = std::nullopt,
    std::optional<torch::Device> device = std::nullopt,
    bool require_contiguous = true) {
  return tensor.defined() && tensor.device().is_privateuseone() &&
         (!dtype.has_value() || tensor.scalar_type() == *dtype) &&
         (!device.has_value() || tensor.device() == *device) &&
         (!require_contiguous || tensor.is_contiguous());
}

}  // namespace xllm::kernel::mlu::bangc::detail
