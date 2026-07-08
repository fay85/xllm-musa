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

#include "qwen3_next_rms_norm.h"

#include <glog/logging.h>

#include "xllm/core/kernels/ops_api.h"

namespace xllm {
namespace layer {

Qwen3NextRMSNormImpl::Qwen3NextRMSNormImpl(int64_t dim,
                                           double eps,
                                           const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter("weight", torch::empty({dim}, options), false);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
Qwen3NextRMSNormImpl::forward(torch::Tensor& input,
                              std::optional<torch::Tensor> residual) {
  if (!residual.has_value()) {
    // No residual: use the persistent output buffer. We pre-allocate
    // (`torch::empty_like`) lazily here on first call (during eager prefill or
    // graph warmup, when stream capture is NOT active). Subsequent calls with
    // the same shape reuse the buffer; that's what makes the gemma_rms_norm
    // call inside the captured graph safe -- no `EmptyStridedMUSA` happens
    // during capture.
    //
    // If a later forward sees a larger shape we re-allocate; this can only
    // happen outside capture (capture always replays the warmup's shape).
    //
    // Cap at kRmsNormBufMaxRows so prefill (large M) does not grow the buffer
    // to prefill-batch size and hold that memory permanently across all layers.
    // When input_rows exceeds the cap, norm_out is left undefined and the
    // dispatch allocates torch::empty_like(input) internally.
    constexpr int64_t kRmsNormBufMaxRows = 128;
    const int64_t input_rows =
        input.size(-1) > 0 ? input.numel() / input.size(-1) : 0;
    xllm::kernel::GemmaRMSNormParams norm_params;
    norm_params.x = input;
    norm_params.gamma = weight_;
    norm_params.epsilon = eps_;
    if (input_rows <= kRmsNormBufMaxRows) {
      if (!norm_out_buf_.defined() ||
          norm_out_buf_.sizes() != input.sizes() ||
          norm_out_buf_.scalar_type() != input.scalar_type() ||
          norm_out_buf_.device() != input.device()) {
        norm_out_buf_ = torch::empty_like(input);
      }
      norm_params.norm_out = norm_out_buf_;
    }
    xllm::kernel::gemma_rms_norm(norm_params);
    return std::make_tuple(norm_params.norm_out, std::nullopt);
  }

#if defined(USE_CUDA) && !defined(USE_DCU)
  // CUDA / MUSA-as-CUDA: dispatch to the fused gemma add+rmsnorm kernel
  // through the gemma_rms_norm API (which folds `weight + 1.0` inside the
  // device kernel). This avoids the host-side `(1.0 + weight_)` tensor
  // allocation that breaks MUSA graph capture on torch_musa 2.7.1 -- the
  // caching allocator's `musaMemMap` is forbidden during stream capture.
  // Routed via ops_api instead of cuda::fused_add_gemma_rms_norm directly so
  // we don't pull `c10/cuda/CUDAGuard.h` into a .cpp compiled without the
  // libMusaMapping plugin.
  xllm::kernel::GemmaRMSNormParams gemma_params;
  gemma_params.x = input;
  gemma_params.gamma = weight_;
  gemma_params.epsilon = eps_;
  gemma_params.residual = residual.value();
  xllm::kernel::gemma_rms_norm(gemma_params);
  return std::make_tuple(gemma_params.norm_out, gemma_params.residual_out);
#else
  // With residual: use fused_layernorm (which calls npu::add_rms_norm on NPU)
  xllm::kernel::FusedLayerNormParams fused_params;
  fused_params.input = input;
  fused_params.residual = residual;
#if !defined(USE_NPU)
  // NPU backend allocates outputs internally in npu::add_rms_norm,
  // so skip pre-allocation to avoid wasted memory.
  fused_params.output = torch::empty_like(input);
  fused_params.residual_out = torch::empty_like(residual.value());
#endif
  fused_params.weight = 1.0 + weight_;
  fused_params.eps = eps_;
  fused_params.mode = "rmsnorm";
  xllm::kernel::fused_layernorm(fused_params);
  return std::make_tuple(fused_params.output, fused_params.residual_out);
#endif
}

void Qwen3NextRMSNormImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace xllm
