/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/layers/musa/deepseek_v41/attention_projection.h"
#include "core/layers/musa/deepseek_v41/csa2_attention.h"

PYBIND11_MODULE(v41_attention_layout_test_module, module) {
  using Sparse = xllm::layer::DeepseekV41SparseAttention;
  pybind11::class_<Sparse>(module, "NativeV41SparseAttention")
      .def(pybind11::init<double, int64_t, std::string>())
      .def("forward", &Sparse::forward);
  using Projection = xllm::layer::DeepseekV41AttentionProjection;
  pybind11::class_<Projection>(module, "NativeV41AttentionProjection")
      .def(pybind11::init<std::vector<torch::Tensor>,
                          int64_t,
                          int64_t,
                          int64_t,
                          double,
                          bool,
                          bool>())
      .def("prepare", &Projection::prepare)
      .def("project_output", &Projection::project_output);
}
