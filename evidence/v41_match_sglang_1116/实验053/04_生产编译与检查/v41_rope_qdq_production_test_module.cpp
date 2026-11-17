/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <torch/extension.h>

#include <memory>
#include <vector>

#include "core/kernels/musa/deepseek_v4_dsa_rope.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/layers/musa/deepseek_v41/attention_projection.h"

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

std::unique_ptr<xllm::layer::DeepseekV41AttentionProjection> projection(
    std::vector<torch::Tensor> weights,
    int64_t heads,
    int64_t groups,
    bool fused) {
  return std::make_unique<xllm::layer::DeepseekV41AttentionProjection>(
      std::move(weights),
      heads,
      groups,
      /*output_rank=*/1024,
      /*epsilon=*/1e-5,
      /*original_fp8_activation=*/true,
      /*decode_packed_qkv=*/false,
      /*verify_skinny_projection=*/true,
      std::vector<torch::Tensor>{},
      std::string{},
      /*direct_rope_table=*/true,
      fused);
}
}  // namespace

PYBIND11_MODULE(v41_rope_qdq_test_module, module) {
  module.def("reference", &reference);
  module.def("candidate", &xllm::kernel::musa::deepseek_v41_rope_table_qdq);
  module.def("rope", &xllm::kernel::musa::deepseek_v41_rope_table);
  module.def("qdq", &xllm::kernel::musa::deepseek_v41_block32_qdq);
  pybind11::class_<xllm::layer::DeepseekV41AttentionProjection>(module,
                                                                "Projection")
      .def("prepare", &xllm::layer::DeepseekV41AttentionProjection::prepare)
      .def("context",
           &xllm::layer::DeepseekV41AttentionProjection::prepare_context_kv)
      .def("output",
           &xllm::layer::DeepseekV41AttentionProjection::project_output);
  module.def("projection", &projection);
}
