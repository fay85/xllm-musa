/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/musa_dsv4_ops_api.h"

PYBIND11_MODULE(v41_hc_metadata_test_module, module) {
  module.def("metadata", &xllm::kernel::musa::deepseek_v4_hc_pre_metadata);
}
