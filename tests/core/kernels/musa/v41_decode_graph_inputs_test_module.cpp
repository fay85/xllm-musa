/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_pair_pool.h"
#include "core/layers/musa/deepseek_v41/token_cache.h"

PYBIND11_MODULE(v41_decode_graph_inputs_test_module, module) {
  module.def("pair_pool", &xllm::kernel::musa::deepseek_v41_pair_pool);
  module.def("pair_pool_decode",
             &xllm::kernel::musa::deepseek_v41_pair_pool_decode);
  using Cache = xllm::layer::DeepseekV41TokenCache;
  pybind11::class_<Cache>(module, "TokenCache")
      .def(pybind11::init<torch::Tensor>())
      .def("commit", &Cache::commit)
      .def("gather", &Cache::gather)
      .def("gather_visible", &Cache::gather_visible);
}
