/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_swa_decode_read.h"
#include "core/layers/musa/deepseek_v41/paged_attention.h"

PYBIND11_MODULE(v41_verify_read_test_module, module) {
  module.def("read", &xllm::kernel::musa::deepseek_v41_swa_verify_read);
  module.def("decode_read", &xllm::kernel::musa::deepseek_v41_swa_decode_read);
  using Paged = xllm::layer::DeepseekV41PagedAttention;
  pybind11::class_<Paged>(module, "Paged")
      .def(pybind11::init<torch::Tensor,
                          int64_t,
                          double,
                          std::string,
                          bool,
                          bool,
                          bool>())
      .def("forward", &Paged::forward_with_window_floor);
}
