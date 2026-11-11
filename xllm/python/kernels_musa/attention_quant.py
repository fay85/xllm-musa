# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Original V4.1 attention QDQ on MUSA, with a BF16 cache carrier.

These calls preserve original FP4/FP8 rounding. They do not define the
physical cache: allocator, estimator and transfer identity choose the same
explicit original_qdq mode before any request executes.
"""
from __future__ import annotations

import os
from typing import Callable

import torch

from scripts.logger import logger

Fp4Quant = Callable[[torch.Tensor, int, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
Fp8Quant = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class V41AttentionQuant:
    """Stateless GPU primitives; the backend owns all persistent state."""

    def __init__(self, fp4: Fp4Quant, fp8: Fp8Quant) -> None:
        self._fp4 = fp4
        self._fp8 = fp8

    @staticmethod
    def _flatten(x: torch.Tensor, group: int) -> torch.Tensor:
        if x.device.type != "musa" or x.dtype != torch.bfloat16:
            raise ValueError("Original attention QDQ requires BF16 MUSA inputs")
        if x.ndim < 2 or x.shape[-1] <= 0 or x.shape[-1] % group:
            raise ValueError(f"Attention QDQ last dimension must be a positive multiple of {group}")
        return x.reshape(-1, x.shape[-1]).contiguous()

    def index(self, x: torch.Tensor) -> torch.Tensor:
        return self._fp4(self._flatten(x, 32), 32, False)[0].reshape_as(x)

    def compressed(self, x: torch.Tensor) -> torch.Tensor:
        return self._fp4(self._flatten(x, 16), 16, True)[0].reshape_as(x)

    def fp8(self, x: torch.Tensor) -> torch.Tensor:
        return self._fp8(self._flatten(x, 32))[0].reshape_as(x)


def select_attention_quant(
    device: torch.device, dtype: torch.dtype
) -> V41AttentionQuant | None:
    if device.type != "musa":
        return None
    mode = os.environ.get("XLLM_DSV41_MUSA_ATTENTION_QUANT", "reference")
    if mode not in ("reference", "original_qdq"):
        raise ValueError(f"Invalid V4.1 attention quantization mode: {mode!r}")
    if mode == "reference":
        return None
    if dtype != torch.bfloat16:
        raise ValueError("Original V4.1 attention QDQ requires BF16 storage")
    import xllm_runtime

    required = ("musa_v41_attention_fp4_quant", "musa_v41_block32_qdq")
    if any(not hasattr(xllm_runtime, name) for name in required):
        raise RuntimeError("Rebuild xllm with original V4.1 attention quantization")
    logger.info("Original V4.1 attention QDQ selected; index cache carrier is BF16")
    return V41AttentionQuant(xllm_runtime.musa_v41_attention_fp4_quant,
                            xllm_runtime.musa_v41_block32_qdq)
