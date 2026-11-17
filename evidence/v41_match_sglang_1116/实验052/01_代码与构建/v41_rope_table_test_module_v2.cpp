/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <glog/logging.h>
#include <torch/extension.h>

#include <cstdint>

#include "core/kernels/musa/deepseek_v4_dsa_rope.h"

namespace xllm::kernel::musa {
torch::Tensor deepseek_v41_rope_table_candidate(const torch::Tensor& input,
                                                const torch::Tensor& positions,
                                                const torch::Tensor& table,
                                                bool inverse,
                                                int32_t block_size,
                                                int32_t arithmetic);
}  // namespace xllm::kernel::musa

namespace {
torch::Tensor reference(const torch::Tensor& input,
                        const torch::Tensor& positions,
                        const torch::Tensor& table,
                        bool inverse) {
  torch::DeviceGuard guard(input.device());
  if (input.size(0) == 0) {
    return input.clone();
  }
  const int64_t width = table.size(1);
  const torch::Tensor rows = table.index_select(0, positions.to(torch::kInt64));
  const torch::Tensor cosine =
      rows.narrow(1, 0, width / 2).repeat_interleave(2, -1).contiguous();
  const torch::Tensor sine = rows.narrow(1, width / 2, width / 2)
                                 .repeat_interleave(2, -1)
                                 .contiguous();
  const torch::Tensor value = input.to(torch::kFloat32)
                                  .reshape({input.size(0), -1, input.size(-1)})
                                  .contiguous();
  torch::Tensor output;
  CHECK(xllm::kernel::musa::deepseek_v4_partial_rope(
      value, cosine, sine, input.size(-1) - width, width, inverse, output));
  return output.reshape(input.sizes()).to(input.scalar_type());
}
}  // namespace

PYBIND11_MODULE(v41_rope_table_test_module, module) {
  module.def("reference", &reference);
  module.def("candidate",
             &xllm::kernel::musa::deepseek_v41_rope_table_candidate);
}
