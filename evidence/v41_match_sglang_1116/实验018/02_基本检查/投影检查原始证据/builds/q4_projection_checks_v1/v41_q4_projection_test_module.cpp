/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/layers/musa/deepseek_v41/feed_forward.h"
PYBIND11_MODULE(v41_q4_projection_test_module, module) {
  module.def("supported",
             &xllm::kernel::musa::deepseek_v41_q4_projection_supported);
  module.def("gemm", &xllm::kernel::musa::deepseek_v41_q4_projection);
  using Dense = xllm::layer::DeepseekV41Dense;
  pybind11::class_<Dense>(module, "Dense")
      .def(pybind11::init<torch::Tensor, bool, bool>())
      .def("forward", &Dense::forward);
  using Shared = xllm::layer::DeepseekV41SharedMlp;
  pybind11::class_<Shared>(module, "Shared")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          double,
                          bool,
                          bool,
                          bool>())
      .def("forward", &Shared::forward);
}
