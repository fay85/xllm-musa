/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <torch/extension.h>

#include "core/kernels/musa/deepseek_v4_dsa_rope.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"

namespace xllm::kernel::musa {
torch::Tensor test_rope_qdq(const torch::Tensor& input,
                            const torch::Tensor& positions,
                            const torch::Tensor& table,
                            bool inverse);
}  // namespace xllm::kernel::musa

namespace {
torch::Tensor reference(const torch::Tensor& input,
                        const torch::Tensor& positions,
                        const torch::Tensor& table,
                        bool inverse) {
  const torch::Tensor rotated = xllm::kernel::musa::deepseek_v41_rope_table(
      input, positions, table, inverse);
  return std::get<0>(xllm::kernel::musa::deepseek_v41_block32_qdq(
                         rotated.reshape({-1, 512})))
      .reshape(input.sizes());
}
}  // namespace

PYBIND11_MODULE(v41_rope_qdq_test_module, module) {
  module.def("reference", &reference);
  module.def("candidate", &xllm::kernel::musa::test_rope_qdq);
  module.def("rope", &xllm::kernel::musa::deepseek_v41_rope_table);
  module.def("qdq", &xllm::kernel::musa::deepseek_v41_block32_qdq);
}
