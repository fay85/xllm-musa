/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/types.h>

#include <cstdint>

namespace xllm::kernel::mlu::bangc {

// Delayed-pre fused MHC boundary for hc_mult = 4 and hidden size 5120, tuned
// for decode token counts (T = batch or batch * draft tokens):
//
//   x [T,5120] bf16, residual_in [T,4,5120] bf16, post_in [T,4] f32,
//   comb_in [T,4,4] f32, pre_in [T,4] f32, hc_fn [24,20480] bf16,
//   hc_scale [3] f32, hc_base [24] f32, gamma [5120] bf16
//   -> norm_out [T,5120] bf16, residual_out [T,4,5120] bf16,
//      next_pre [T,4] f32, post_out [T,4] f32, comb_out [T,4,4] f32
//
// Every core of the device owns a hidden slice of a group of tokens, so small
// T keeps all cores busy; the cross-core partial sums go through a workspace
// allocated per call. Any T is accepted and only natural element alignment is
// required. Large calls whose tensors are all 64-byte aligned are handed to
// the prefill kernel when a cost model says it is faster.
// Enqueues on the guarded device's current stream and supports graph capture.
void fused_mhc_decode(const torch::Tensor& x,
                      const torch::Tensor& residual_in,
                      const torch::Tensor& post_in,
                      const torch::Tensor& comb_in,
                      const torch::Tensor& pre_in,
                      const torch::Tensor& hc_fn,
                      const torch::Tensor& hc_scale,
                      const torch::Tensor& hc_base,
                      const torch::Tensor& gamma,
                      int64_t sinkhorn_iter,
                      double hc_eps,
                      double norm_eps,
                      torch::Tensor& norm_out,
                      torch::Tensor& residual_out,
                      torch::Tensor& next_pre,
                      torch::Tensor& post_out,
                      torch::Tensor& comb_out);

}  // namespace xllm::kernel::mlu::bangc
