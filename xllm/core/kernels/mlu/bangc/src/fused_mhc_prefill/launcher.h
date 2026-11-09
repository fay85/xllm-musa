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

constexpr int32_t kFusedMhcPrefillHcMult = 4;
constexpr int32_t kFusedMhcPrefillDim = 5120;
constexpr int32_t kFusedMhcPrefillMixes = 24;
// Every tensor base address must be aligned to this many bytes.
constexpr int64_t kFusedMhcPrefillAlignBytes = 64;

// Prefill delayed-pre fused MHC boundary (hc_mult = 4, dim = 5120).
//
// in[9]  : x [T,5120] bf16, residual_in [T,4,5120] bf16, post_in [T,4] f32,
//          comb_in [T,4,4] f32, pre_in [T,4] f32, hc_fn [24,20480] bf16,
//          hc_scale [3] f32, hc_base [24] f32, gamma [5120] bf16
// out[5] : norm_out [T,5120] bf16, residual_out [T,4,5120] bf16,
//          next_pre [T,4] f32, post_out [T,4] f32, comb_out [T,4,4] f32
// All tensors are contiguous. Returns 0 on success, non-zero on a rejected
// configuration (nothing is launched): 1 shape, 2 device query, 3 null or
// misaligned pointer.
extern "C" int32_t fused_mhc_prefill_launch(void* queue,
                                            const void* const* in,
                                            void* const* out,
                                            int64_t tokens,
                                            int32_t dim,
                                            int32_t sinkhorn_iter,
                                            float hc_eps,
                                            float norm_eps);
