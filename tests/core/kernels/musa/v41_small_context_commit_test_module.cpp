/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/musa_dsv4_ops_api.h"

namespace {

torch::Tensor scatter(torch::Tensor cache,
                      const torch::Tensor& slots,
                      const torch::Tensor& values) {
  xllm::kernel::musa::deepseek_v41_scatter_small_cache_native(
      cache, slots, values);
  return cache;
}

}  // namespace

PYBIND11_MODULE(v41_small_context_commit_test_module, module) {
  module.def("scatter", &scatter);
}
