# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MUSA normalization kernels."""

from __future__ import annotations

import torch

fused_add_rms_norm = torch.ops.xllm_ops.fused_add_rms_norm
gemma_rms_norm = torch.ops.xllm_ops.gemma_rms_norm
fused_add_gemma_rms_norm = torch.ops.xllm_ops.fused_add_gemma_rms_norm


def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(input)
    return torch.ops.xllm_ops.rms_norm(input, weight, eps, output)


def rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    value_2d = value.reshape(-1, value.size(-1))
    gate_2d = gate.reshape(-1, gate.size(-1))
    if output is None:
        output_2d = torch.empty_like(value_2d)
    else:
        output_2d = output.reshape(-1, value.size(-1))
        if output_2d.size(0) < value_2d.size(0):
            raise RuntimeError("rms_norm_gated output is shorter than value")
        output_2d = output_2d[: value_2d.size(0)]
    output_2d = torch.ops.xllm_ops.rms_norm_gated(
        value_2d, gate_2d, weight, eps, output_2d
    )
    return output_2d.reshape_as(value)


__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "gemma_rms_norm",
    "fused_add_gemma_rms_norm",
    "rms_norm_gated",
]
