# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Explicit OFFLINE checkpoint conversion, not a CPU inference fallback.

GLM's public checkpoint stores FP8 E4M3 matrices and FLOAT32 block scales
named weight_scale_inv. Those are multiplicative dequantization scales, not
ue8m0 bytes (DeepSeek V4.1), int8 scales, or an instruction to divide.
This initial path expands weights to BF16 before copying them to MUSA.
It does NOT claim native FP8 activation/weight inference or FP8 performance.
"""

from __future__ import annotations

import torch


def _validate_fp8_block_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> None:
    """Validate the full offline checkpoint contract, including other shards."""
    if weight.device.type != "cpu" or scale_inv.device.type != "cpu":
        raise ValueError("Offline weight conversion expects CPU checkpoint tensors")
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        raise ValueError("GLM FP8 conversion requires a 2-D E4M3FN matrix")
    if len(block_size) != 2 or any(size <= 0 for size in block_size):
        raise ValueError("Weight block geometry must contain two positive dimensions")
    rows, columns = weight.shape
    block_rows, block_columns = block_size
    expected = (
        (rows + block_rows - 1) // block_rows,
        (columns + block_columns - 1) // block_columns,
    )
    if scale_inv.dtype != torch.float32 or tuple(scale_inv.shape) != expected:
        raise ValueError(f"FP8 checkpoint requires FP32 scale_inv shape {expected}")
    if not torch.isfinite(scale_inv).all().item() or (scale_inv < 0).any().item():
        raise ValueError("FP8 checkpoint scales must be finite and nonnegative")


def dequantize_fp8_block_to_bf16(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Convert a 2-D CPU checkpoint matrix with explicit FP32 block scales."""
    _validate_fp8_block_weight(weight, scale_inv, block_size)
    rows, columns = weight.shape
    block_rows, block_columns = block_size
    expanded_scale = scale_inv.repeat_interleave(block_rows, 0).repeat_interleave(
        block_columns, 1
    )
    return (weight.float() * expanded_scale[:rows, :columns]).to(torch.bfloat16)


def dequantize_fp8_block_shard_to_bf16(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dim: int,
    world: int,
    rank: int,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Expand only an aligned TP shard; retain the original unaligned path.

    This is offline CPU checkpoint loading, not inference. All original scale
    values are validated before slicing. Unaligned shard boundaries select the
    original full expansion explicitly, before any conversion is performed.
    """
    _validate_fp8_block_weight(weight, scale_inv, block_size)
    if dim not in (-2, -1, 0, 1) or world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid FP8 TP shard dimension, world, or rank")
    dim %= 2
    extent = weight.size(dim)
    if extent % world != 0:
        raise ValueError("FP8 TP shard dimension must be divisible by world")
    width = extent // world
    start = rank * width
    end = start + width
    block_width = block_size[dim]
    aligned = start % block_width == 0 and (
        end % block_width == 0 or end == extent
    )
    if not aligned:
        full = dequantize_fp8_block_to_bf16(weight, scale_inv, block_size)
        return full.narrow(dim, start, width).contiguous()
    shard = weight.narrow(dim, start, width).contiguous()
    scale_start = start // block_width
    scale_width = (width + block_width - 1) // block_width
    scales = scale_inv.narrow(dim, scale_start, scale_width).contiguous()
    return dequantize_fp8_block_to_bf16(shard, scales, block_size)
