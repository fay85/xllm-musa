/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v41_dense_block32.h"
#include "core/layers/musa/deepseek_v41/feed_forward.h"
#include "models/llm/musa/deepseek_v41_checkpoint.h"

PYBIND11_MODULE(v41_prequant_projection_test_module, module) {
  module.def("forward",
             &xllm::kernel::musa::deepseek_v41_dense_prequant_block32,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("prepare_fp8",
             &xllm::DeepseekV41Checkpoint::prepare_dense_fp8,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "dense_forward",
      [](const torch::Tensor& input,
         const torch::Tensor& weight,
         const torch::Tensor& scale,
         const std::string& directory) {
        xllm::layer::DeepseekV41Dense dense(weight,
                                            /*original_fp8_activation=*/true,
                                            /*verify_skinny_projection=*/false,
                                            scale,
                                            directory);
        return dense.forward(input);
      },
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "shared_forward",
      [](const torch::Tensor& input,
         const std::vector<torch::Tensor>& weights,
         const std::vector<torch::Tensor>& scales,
         const std::string& directory) {
        CHECK_EQ(weights.size(), 3);
        xllm::layer::DeepseekV41SharedMlp shared(
            weights[0],
            weights[1],
            weights[2],
            /*limit=*/10.0,
            /*original_fp8_activation=*/true,
            /*decode_packed_gate_up=*/false,
            /*verify_skinny_projection=*/false,
            scales,
            directory);
        return shared.forward(input);
      },
      pybind11::call_guard<pybind11::gil_scoped_release>());
}
