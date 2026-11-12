# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""V4 partial-rotary kernel reuse with V4.1 FP32 arithmetic and tables."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Callable

import torch

from scripts.logger import logger

Rotary = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


@lru_cache(maxsize=1)
def select_rotary() -> Rotary | None:
    """Freeze the explicit MUSA mode once, before the first rotary execution."""
    mode = os.environ.get("XLLM_DSV41_MUSA_ROPE", "reference")
    if mode not in ("reference", "v4_fused"):
        raise ValueError(f"Invalid V4.1 MUSA RoPE mode: {mode!r}")
    logger.info("V4.1 MUSA RoPE mode=%s", mode)
    if mode == "reference":
        return None
    import xllm_runtime

    if not hasattr(xllm_runtime, "musa_partial_rope_fp32"):
        raise RuntimeError("Rebuild MUSA xllm: missing musa_partial_rope_fp32")
    native = xllm_runtime.musa_partial_rope_fp32

    def rotary(x: torch.Tensor, positions: torch.Tensor, table: torch.Tensor, inverse: bool) -> torch.Tensor:
        if x.device.type != "musa" or positions.device != x.device or table.device != x.device:
            raise ValueError("RoPE operands must reside on the same MUSA device")
        if x.ndim < 2 or positions.numel() != x.shape[0] or table.ndim != 2:
            raise ValueError("RoPE requires [T,...,D], T positions, and a 2D table")
        width = table.shape[1]
        if width <= 0 or width % 2 or width > x.shape[-1]:
            raise ValueError("RoPE table width must be positive, even, and fit the hidden axis")
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32) or not table.is_floating_point():
            raise ValueError("RoPE hidden must be BF16/FP16/FP32 and table must be floating point")
        if x.numel() == 0:
            return x.clone()
        half = width // 2
        rows = table.index_select(0, positions.reshape(-1).to(torch.int64)).float()
        # V4 takes expanded [cos0,cos0,cos1,cos1,...] rows; V4.1 stores
        # [cos0,cos1,...,sin0,sin1,...]. Keep tables and arithmetic FP32.
        cos = rows[:, :half].repeat_interleave(2, dim=-1).contiguous()
        sin = rows[:, half:].repeat_interleave(2, dim=-1).contiguous()
        hidden = x.float().reshape(x.shape[0], -1, x.shape[-1]).contiguous()
        return native(hidden, cos, sin, inverse).reshape(x.shape).to(x.dtype)

    return rotary
