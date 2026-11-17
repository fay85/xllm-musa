/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_mhc_prenorm.h"
#include "core/layers/musa/deepseek_v41/hyper_connection.h"

PYBIND11_MODULE(v41_mhc_dot_test_module, module) {
  module.def(
      "integrated",
      [](const torch::Tensor& residual,
         const torch::Tensor& carried_pre,
         const std::vector<torch::Tensor>& weights,
         const std::string& mate_directory,
         const std::string& dot_directory) {
        CHECK_EQ(weights.size(), 3);
        xllm::layer::DeepseekV41HyperConnection hc(
            weights[0],
            weights[1],
            weights[2],
            /*iterations=*/20,
            /*norm_eps=*/1e-20f,
            /*eps=*/1e-6f,
            mate_directory,
            /*decode_fused_prenorm_reduce=*/true,
            dot_directory);
        return hc.prepare(residual, carried_pre);
      },
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("forward",
             &xllm::kernel::musa::deepseek_v41_mhc_dot_sinkhorn,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "control",
      [](const torch::Tensor& residual,
         const torch::Tensor& carried_pre,
         const std::vector<torch::Tensor>& weights,
         const std::string& directory) {
        CHECK_EQ(weights.size(), 3);
        xllm::layer::DeepseekV41HyperConnection hc(
            weights[0],
            weights[1],
            weights[2],
            /*iterations=*/20,
            /*norm_eps=*/1e-20f,
            /*eps=*/1e-6f,
            directory,
            /*decode_fused_prenorm_reduce=*/true);
        return hc.prepare(residual, carried_pre);
      },
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("collapse",
             &xllm::kernel::musa::deepseek_v4_hc_pre_combine,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("combine",
             &xllm::kernel::musa::deepseek_v4_hc_post,
             pybind11::call_guard<pybind11::gil_scoped_release>());
}
