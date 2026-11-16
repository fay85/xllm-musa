/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_dense_block32.h"

PYBIND11_MODULE(v41_prequant_projection_test_module, module) {
  module.def("forward",
             &xllm::kernel::musa::deepseek_v41_dense_prequant_block32,
             pybind11::call_guard<pybind11::gil_scoped_release>());
}
