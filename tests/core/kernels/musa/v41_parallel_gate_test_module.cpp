/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/musa_dsv4_ops_api.h"

PYBIND11_MODULE(v41_parallel_gate_test_module, module) {
  module.def("gate", &xllm::kernel::musa::deepseek_v41_gate_topk);
}
