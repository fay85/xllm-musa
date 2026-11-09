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

#include <cstdint>

// Decode delayed-pre fused MHC boundary (hc_mult = 4, dim = 5120).
//
// in[9]  : x [T,5120] bf16, residual_in [T,4,5120] bf16, post_in [T,4] f32,
//          comb_in [T,4,4] f32, pre_in [T,4] f32, hc_fn [24,20480] bf16,
//          hc_scale [3] f32, hc_base [24] f32, gamma [5120] bf16
// out[5] : norm_out [T,5120] bf16, residual_out [T,4,5120] bf16,
//          next_pre [T,4] f32, post_out [T,4] f32, comb_out [T,4,4] f32
// workspace: device buffer of fused_mhc_decode_workspace_bytes(T) bytes for
//            the cross-core partial sums (0 bytes when the plan keeps them in
//            cluster SRAM); no state is carried across calls.
// All tensors are contiguous; only natural element alignment is assumed.
// Large 64-byte aligned calls may be handed to fused_mhc_prefill_launch.

// Workspace bytes for T tokens on the current device, -1 if the device query
// fails or the topology is unsupported.
extern "C" int64_t fused_mhc_decode_workspace_bytes(int64_t tokens);

// Returns 0 on success, non-zero on a rejected configuration (nothing is
// launched): 1 shape, 2 too many tokens, 3 unsupported device topology,
// 5 no kernel for the chosen group size. T == 0 launches nothing.
extern "C" int32_t fused_mhc_decode_launch(void* queue,
                                           const void* const* in,
                                           void* const* out,
                                           void* workspace,
                                           int64_t tokens,
                                           int32_t dim,
                                           int32_t sinkhorn_iter,
                                           float hc_eps,
                                           float norm_eps);
