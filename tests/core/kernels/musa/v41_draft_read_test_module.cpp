/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_swa_decode_read.h"
#include "core/layers/musa/deepseek_v41/csa2_attention.h"
#include "core/layers/musa/deepseek_v41/paged_attention.h"

PYBIND11_MODULE(v41_draft_read_test_module, module) {
  module.def("draft_read",
             &xllm::kernel::musa::deepseek_v41_swa_draft_read,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  using Sparse = xllm::layer::DeepseekV41SparseAttention;
  pybind11::class_<Sparse>(module, "Sparse")
      .def(pybind11::init<double, int64_t, std::string>())
      .def("forward",
           &Sparse::forward,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using Paged = xllm::layer::DeepseekV41PagedAttention;
  pybind11::class_<Paged>(module, "Paged")
      .def(pybind11::
               init<torch::Tensor, int64_t, double, std::string, bool, bool>())
      .def("forward_draft_block",
           &Paged::forward_draft_block,
           pybind11::call_guard<pybind11::gil_scoped_release>());
}
