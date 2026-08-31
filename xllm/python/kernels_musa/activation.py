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

"""MUSA activation kernels."""

from __future__ import annotations

import torch


mul_sigmoid_gate_inplace = torch.ops.xllm_ops.mul_sigmoid_gate_inplace


def silu_and_mul(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        sizes = list(input.shape)
        sizes[-1] //= 2
        output = input.new_empty(sizes)
    return torch.ops.xllm_ops.silu_and_mul(input, output)


def fused_swiglu_quant_fp8(
    input: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.xllm_ops.fused_swiglu_quant_fp8(input, group_size)


__all__ = ["silu_and_mul", "mul_sigmoid_gate_inplace", "fused_swiglu_quant_fp8"]
