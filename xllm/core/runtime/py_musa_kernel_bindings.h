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

#pragma once

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <cmath>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm {

// Shared by the embedded runtime and the standalone numerical-test module.
// This binds the existing V4 kernel; it does not implement a second kernel.
inline void register_musa_kernel_bindings(pybind11::module_& module) {
  module.def(
      "musa_shared_swiglu",
      [](const torch::Tensor& input, double limit) {
        CHECK(input.defined());
        CHECK(input.device().type() == torch::kPrivateUse1);
        CHECK(input.scalar_type() == torch::kBFloat16);
        CHECK_EQ(input.dim(), 2);
        CHECK_GT(input.size(1), 0);
        CHECK_EQ(input.size(1) % 2, 0);
        CHECK(input.is_contiguous());
        CHECK(std::isfinite(limit) && limit > 0.0 && limit < 1000000.0);
        torch::Tensor output =
            torch::empty({input.size(0), input.size(1) / 2}, input.options());
        if (input.size(0) == 0) {
          return output;
        }
        torch::Tensor result =
            kernel::musa::deepseek_v4_shared_swiglu(input, limit, output);
        CHECK(result.defined()) << "V4 shared SwiGLU rejected validated input";
        return result;
      },
      pybind11::arg("input"),
      pybind11::arg("limit"),
      pybind11::call_guard<pybind11::gil_scoped_release>());
}

}  // namespace xllm
