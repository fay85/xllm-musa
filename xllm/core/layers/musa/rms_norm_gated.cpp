/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/musa/rms_norm_gated.h"

#include <algorithm>

#include "framework/state_dict/utils.h"
#include "kernels/ops_api.h"
#include "layers/musa/graph_output_buffers.h"

namespace xllm::layer::musa {

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : eps_(eps) {
  weight_ = register_parameter(
      "weight", torch::empty({dim}, options), /*requires_grad=*/false);
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input,
                                        std::optional<torch::Tensor> gate,
                                        bool use_transient_output) {
  xllm::kernel::GatedLayerNormParams params;
  params.x = input;
  params.weight = weight_;
  params.bias = torch::Tensor();
  params.eps = eps_;
  if (gate.has_value()) {
    params.z = gate;
  }
  params.group_size = input.size(-1);
  params.is_rms_norm = true;

  constexpr int64_t kPersistentMaxRows = 128;
  PiecewiseGraphMatmulBufferPool* pool =
      PiecewiseGraphMatmulBufferScope::current_buffer_pool();
  if (pool != nullptr) {
    params.output_buf = pool->get_gated_rms_norm_output(input);
  }
  if (!params.output_buf.has_value() && input.dim() >= 1 && input.numel() > 0 &&
      input.stride(-1) == 1) {
    const int64_t last_dim = input.size(-1);
    const int64_t rows = input.numel() / last_dim;
    if (rows <= kPersistentMaxRows) {
      const bool needs_realloc =
          !output_buf_.defined() || output_buf_.device() != input.device() ||
          output_buf_.scalar_type() != input.scalar_type() ||
          output_buf_.dim() != 2 || output_buf_.size(0) < rows ||
          output_buf_.size(1) != last_dim;
      if (needs_realloc) {
        output_buf_ = torch::empty({std::max<int64_t>(rows, 32), last_dim},
                                   input.options());
      }
      params.output_buf = output_buf_.narrow(0, 0, rows).view(input.sizes());
    } else if (use_transient_output && input.dim() == 2 &&
               input.is_contiguous() && gate.has_value() && gate->defined() &&
               gate->is_contiguous()) {
      params.output_buf = torch::empty_like(input);
    }
  }

  return xllm::kernel::gated_layer_norm(params);
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace xllm::layer::musa
