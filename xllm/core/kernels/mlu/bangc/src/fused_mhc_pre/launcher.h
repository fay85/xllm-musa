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

// Standalone Pre + RMSNorm, HC=4, D=5120. Contiguous tensors, natural
// alignment; T=0 launches nothing. No inputs are mutated.
// in[6]: residual [T,4,5120] BF16, pre_in [T,4] FP32,
//        hc_fn [24,20480] BF16, hc_scale [3] FP32, hc_base [24] FP32,
//        gamma [5120] BF16.
// out[4]: norm [T,5120] BF16, new_pre/post [T,4] FP32, comb [T,4,4] FP32.
// Workspace bytes for T tokens on the current device (0 when the plan keeps
// the partial sums in cluster SRAM), -1 if the device query fails.
extern "C" int64_t fused_mhc_pre_workspace_bytes(int64_t tokens);

// Returns 0 on success, non-zero on a rejected configuration (nothing is
// launched): 1 shape, 2 too many tokens (or, for aligned T >= 257 handed to
// the prefill kernel, a failed device query), 3/4 unsupported device
// topology, 5 no kernel for the chosen group size. T == 0 launches nothing.
extern "C" int32_t fused_mhc_pre_launch(void* queue,
                                        const void* const* in,
                                        void* const* out,
                                        void* workspace,
                                        int64_t tokens,
                                        int32_t dim,
                                        int32_t sinkhorn_iter,
                                        float hc_eps,
                                        float norm_eps);
