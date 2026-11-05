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

#include <torch/torch.h>

#include <cstdint>
#include <optional>

namespace xllm::kernel::musa {

// Native counterpart of MATE's flash_kda.get_workspace_size. Internal scratch
// is allocated on the calling stream; no caller workspace is consumed.
int64_t flash_kda_workspace_size(int64_t total_tokens,
                                 int64_t num_heads,
                                 int64_t num_sequences = 1);

// Native counterpart of flash_kda.fwd (Q/K normalization is always enabled).
// Dense Q/K: [B,T,Hqk,128], V/G/out: [B,T,Hv,128], beta: [B,T,Hv].
// Varlen also accepts rank-3 Q/K/V/G/out and rank-2 beta, with int32/int64
// cu_seqlens. Hv is a multiple of Hqk. Activations are FP16 or BF16.
// Optional states are contiguous [N,Hv,128,128], V-first, activation dtype
// or FP32. A_log [Hv] and dt_bias [Hv,128] are paired optional FP32 tensors.
// Output and final_state are caller-owned and written in place. All device
// work uses the existing TVM-FFI stream guard, with ordering against the
// current Torch MUSA stream. Precompiled MATE modules are loaded from
// FLASHINFER_OPS_PATH, like the GDN operators.
void flash_kda(const torch::Tensor& q,
               const torch::Tensor& k,
               const torch::Tensor& v,
               const torch::Tensor& g,
               const torch::Tensor& beta,
               std::optional<double> scale,
               torch::Tensor& out,
               const std::optional<torch::Tensor>& a_log,
               const std::optional<torch::Tensor>& dt_bias,
               double lower_bound,
               const std::optional<torch::Tensor>& initial_state = std::nullopt,
               const std::optional<torch::Tensor>& final_state = std::nullopt,
               const std::optional<torch::Tensor>& cu_seqlens = std::nullopt);

}  // namespace xllm::kernel::musa
