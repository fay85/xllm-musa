/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_sparse_attention.h"
#include "core/kernels/musa/deepseek_v41_swa_decode_read.h"

PYBIND11_MODULE(v41_verify_graph_inputs_test_module, module) {
  module.def("read", &xllm::kernel::musa::deepseek_v41_swa_verify_read);
  module.def("attention",
             &xllm::kernel::musa::deepseek_v41_mate_sparse_attention);
  module.def("attention_prepared",
             &xllm::kernel::musa::deepseek_v41_mate_sparse_attention_prepared);
}
