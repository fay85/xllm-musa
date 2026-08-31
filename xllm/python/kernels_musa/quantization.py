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

"""MUSA block-FP8 linear kernels."""

from __future__ import annotations

import torch


def block_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_n: int,
    block_k: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("block_fp8_linear expects a 2D activation")
    if output is None:
        output = torch.empty(
            (x.size(0), weight.size(0)),
            dtype=x.dtype,
            device=x.device,
        )
    return torch.ops.xllm_ops.block_fp8_linear(
        x, weight, weight_scale_inv, block_n, block_k, output
    )


def block_fp8_linear_quantized(
    x: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_n: int,
    block_k: int,
    output: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("block_fp8_linear_quantized expects a 2D activation")
    if output is None:
        resolved_dtype = (
            output_dtype if output_dtype is not None else torch.bfloat16
        )
        output = torch.empty(
            (x.size(0), weight.size(0)),
            dtype=resolved_dtype,
            device=x.device,
        )
    return torch.ops.xllm_ops.block_fp8_linear_quantized(
        x, input_scale, weight, weight_scale_inv, block_n, block_k, output
    )


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("per_token_group_quant_fp8 expects a 2D activation")
    return torch.ops.xllm_ops.per_token_group_quant_fp8(x, group_size)


__all__ = [
    "block_fp8_linear",
    "block_fp8_linear_quantized",
    "per_token_group_quant_fp8",
]
