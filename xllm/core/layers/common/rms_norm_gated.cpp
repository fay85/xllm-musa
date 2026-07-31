/* Copyright 2025-2026 The xLLM Authors.

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

#include "rms_norm_gated.h"

#include <glog/logging.h>

#include "layers/common/linear.h"
#include "framework/state_dict/utils.h"
#include "xllm/core/kernels/ops_api.h"

namespace xllm {
namespace layer {

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter(
      "weight", torch::empty({dim}, options), /*requires_grad=*/false);
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input,
                                        std::optional<torch::Tensor> gate,
                                        bool use_transient_output) {
  xllm::kernel::GatedLayerNormParams params;
  params.x = input;
  params.weight = weight_;
  torch::Tensor bias;
  params.bias = bias;
  params.eps = eps_;
  if (gate.has_value()) {
    params.z = gate;
  }
  params.group_size = input.size(-1);
  params.is_rms_norm = true;

#if defined(USE_CUDA) || defined(USE_MUSA)
  // Provide a caller-owned output buffer so the CUDA/MUSA fused gated
  // RMSNorm kernel can avoid all `at::empty`-style allocations under
  // stream capture. The buffer lazily grows on the row dimension only;
  // shape must otherwise match `input`. Stays inert on other backends
  // (the kernel dispatch ignores `output_buf` when it falls back to the
  // libtorch ref impl).
  //
  // Cap at kRmsNormGatedBufMaxRows so prefill (large M) does not grow the
  // persistent buffer to prefill-batch size and hold that memory across all
  // 48 GDN layers. Eager prefill callers can opt into the transient buffer
  // below; other large shapes fall back to the reference implementation.
  constexpr int64_t kRmsNormGatedBufMaxRows = 128;
  if (auto* pool = PiecewiseGraphMatmulBufferScope::current_buffer_pool();
      pool != nullptr) {
    params.output_buf = pool->get_gated_rms_norm_output(input);
  }
  torch::Tensor transient_output;
  if (!params.output_buf.has_value() && input.dim() >= 1 &&
      input.numel() > 0 && input.stride(-1) == 1) {
    const auto target_sizes = input.sizes();
    const int64_t last_dim = target_sizes.back();
    const int64_t target_rows = input.numel() / last_dim;
    if (target_rows <= kRmsNormGatedBufMaxRows) {
      auto desired_options = input.options();

      const bool need_realloc = !out_buf_.defined() ||
                                out_buf_.dtype() != desired_options.dtype() ||
                                out_buf_.device() != desired_options.device() ||
                                out_buf_.dim() != input.dim() ||
                                out_buf_.size(-1) != last_dim ||
                                (out_buf_.numel() / last_dim) < target_rows;
      if (need_realloc) {
        std::vector<int64_t> alloc_shape(target_sizes.begin(),
                                         target_sizes.end());
        // Pre-allocate up to the maximum decode-graph row count we currently
        // capture (32). Smaller-than-32 rows reuse the same allocation via
        // narrow(); >32 rows fall through to a one-time re-grow.
        const int64_t kMaxRowsHint = 32;
        alloc_shape[0] = std::max<int64_t>(target_rows, kMaxRowsHint);
        out_buf_ = torch::empty(alloc_shape, desired_options);
      }
      // Narrow on the leading row dim so the view's shape exactly matches
      // `input.sizes()`. Underlying storage is preserved across replays.
      torch::Tensor view = out_buf_;
      if (out_buf_.size(0) != target_rows) {
        view = out_buf_.narrow(0, 0, target_rows);
      }
      if (view.sizes() == target_sizes && view.stride(-1) == 1) {
        params.output_buf = view;
      }
    }

    // A caller that owns an eager-only path can request a short-lived output
    // and still use the fused kernel without retaining one large buffer per
    // GDN layer. The returned tensor keeps the storage alive until its
    // downstream projection has consumed it. Graph paths must leave this off
    // because the allocation is not capture-safe.
    if (use_transient_output && target_rows > kRmsNormGatedBufMaxRows &&
        input.dim() == 2 && gate.has_value() && gate->defined() &&
        gate->dim() == 2 && input.is_contiguous() && gate->is_contiguous()) {
      transient_output = torch::empty_like(input);
      params.output_buf = transient_output;
    }
  }
#endif

  auto ret = xllm::kernel::gated_layer_norm(params);
  return ret;
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace xllm
