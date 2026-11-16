/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_w4a8.h"
#include "core/layers/musa/deepseek_v41/w4a8_experts.h"

PYBIND11_MODULE(v41_static_w4a8_test_module, module) {
  module.def("dispatch", &xllm::kernel::musa::deepseek_v41_w4a8_dispatch);
  module.def("gemm", &xllm::kernel::musa::deepseek_v41_w4a8_masked_gemm);
  module.def("swiglu", &xllm::kernel::musa::deepseek_v41_w4a8_swiglu_quant);
  module.def("combine", &xllm::kernel::musa::deepseek_v41_w4a8_combine);
  using Experts = xllm::layer::DeepseekV41W4A8Experts;
  pybind11::class_<Experts>(module, "Experts")
      .def(pybind11::init<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          int64_t,
                          double,
                          std::string,
                          double,
                          double>())
      .def("forward",
           &Experts::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
}
