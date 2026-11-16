/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>
namespace xllm::kernel::musa {
bool deepseek_v41_small_bf16_gemm_supported(const torch::Tensor&,
                                            const torch::Tensor&,
                                            const torch::Tensor&);
void deepseek_v41_small_bf16_gemm(const torch::Tensor&,
                                  const torch::Tensor&,
                                  torch::Tensor&);
}  // namespace xllm::kernel::musa
PYBIND11_MODULE(v41_small_gemm_test_module, module) {
  module.def("supported",
             &xllm::kernel::musa::deepseek_v41_small_bf16_gemm_supported);
  module.def("gemm", &xllm::kernel::musa::deepseek_v41_small_bf16_gemm);
}
