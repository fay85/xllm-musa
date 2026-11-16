/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/musa_dsv4_ops_api.h"

namespace xllm::kernel::musa {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_metadata_warp_probe(
    const torch::Tensor& mixes,
    const torch::Tensor& scale,
    const torch::Tensor& base,
    int64_t iterations,
    double eps);
}  // namespace xllm::kernel::musa

PYBIND11_MODULE(v41_hc_metadata_warp_test_module, module) {
  module.def("reference", &xllm::kernel::musa::deepseek_v4_hc_pre_metadata);
  module.def("candidate", &xllm::kernel::musa::hc_metadata_warp_probe);
}
