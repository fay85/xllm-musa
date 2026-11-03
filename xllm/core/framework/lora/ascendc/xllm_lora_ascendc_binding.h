/*
 * Copyright (c) 2026 xllm authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AscendC LoRA kernel bindings for the bgmv decode path. sgmv variants
 * (prefill) are not implemented.
 * Backend: vllm-ascend libvllm_ascend_kernels.so, dlopen'd on first call.
 * The torch dispatcher registration path (torch.ops._C_xllm_lora) is unusable
 * here -- an orphan singleton under -Wl,-Bsymbolic -- so the slow path calls
 * the C++ wrappers directly.
 *
 * Header exposing the four AscendC LoRA kernel op wrappers defined in
 * xllm_lora_ascendc_binding.cpp for direct C++ callers. This avoids the
 * torch dispatcher registration path, which is affected by the
 * xllm/pybind/CMakeLists.txt:28 `-Wl,-Bsymbolic` link flag (needed for
 * ffmpeg neon PIC linking) that stops TORCH_LIBRARY static registration
 * from reaching the process-wide c10::Dispatcher singleton.
 *
 * The wrappers themselves are unchanged; they still dlopen
 * libvllm_ascend_kernels.so on first call, forward to the mangled
 * `vllm_ascend::*_impl` symbols with argtypes matching the exported symbols.
 */
#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace xllm {

// x:       [batch_tokens, hidden_in]  fp16 / bf16
// w:       [num_adapters, R, hidden_in]  fp16 / bf16    (or 4D with layers)
// indices: [batch_tokens]  int64  (per-token adapter slot in the stacked slab)
// y:       [batch_tokens, R]  fp32  (accumulator; kernel writes here)
// scale:   LoRA alpha / rank scaling
void bgmv_shrink(torch::Tensor x,
                 torch::Tensor w,
                 torch::Tensor indices,
                 torch::Tensor y,
                 double scale);

// x:            [batch_tokens, R]  fp32  (buffer from bgmv_shrink)
// w:            [num_adapters, hidden_out, R]  fp16 / bf16
// indices:      [batch_tokens]  int64
// y:            [batch_tokens, y_size]  fp16 / bf16  (in-place accumulate)
// slice_offset, slice_size: write to y[:, slice_offset :
// slice_offset+slice_size]
void bgmv_expand(torch::Tensor x,
                 torch::Tensor w,
                 torch::Tensor indices,
                 torch::Tensor y,
                 int64_t slice_offset,
                 int64_t slice_size);

// NOTE: sgmv_shrink / sgmv_expand (prefill path) are defined in
// xllm_lora_ascendc_binding.cpp for completeness but intentionally NOT
// declared here — Commit B wires only the decode (bgmv) path.
// Prefill (sgmv) wire lands in a follow-up commit once decode e2e
// numbers are in.

}  // namespace xllm
