# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Explicit V4.1 GPU sqrt-softplus routing with unbiased route weights."""
from __future__ import annotations

import math
import os
from typing import Callable

import torch
from scripts.logger import logger

Gate = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def select_gate(device: torch.device, experts: int, topk: int, normalize: bool, scale: float) -> Gate | None:
    if device.type != "musa":
        return None
    mode = os.environ.get("XLLM_DSV41_MUSA_GATE", "reference")
    if mode not in ("reference", "v41_fused"):
        raise ValueError(f"Invalid V4.1 MUSA gate mode: {mode!r}")
    logger.info("V4.1 MUSA gate mode=%s experts=%s topk=%s", mode, experts, topk)
    if mode == "reference":
        return None
    if not 0 < topk <= experts <= 1024 or not math.isfinite(scale):
        raise ValueError("Fused V4.1 gate requires 0<topk<=experts<=1024 and finite scale")
    import xllm_runtime
    if not hasattr(xllm_runtime, "musa_v41_gate_topk"):
        raise RuntimeError("Rebuild MUSA xllm: missing musa_v41_gate_topk")
    native = xllm_runtime.musa_v41_gate_topk

    def gate(logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.device.type != "musa" or bias.device != logits.device:
            raise ValueError("Gate inputs must share one MUSA device")
        if logits.dtype != torch.float32 or bias.dtype != torch.float32:
            raise ValueError("Gate logits and correction bias must remain FP32")
        if logits.ndim != 2 or logits.shape[1] != experts:
            raise ValueError("Gate logits must have configured [T,E] shape")
        if tuple(bias.shape) not in ((experts,), tuple(logits.shape)):
            raise ValueError("Gate bias must be [E] or [T,E]")
        return native(logits.contiguous(), bias.contiguous(), topk, normalize, scale)

    return gate
