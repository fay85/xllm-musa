# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Explicit original Engram gate and delivered-SGLang-inspired MUSA fusion."""
from __future__ import annotations

import os
import math
import struct
from typing import Callable

import torch

from scripts.logger import logger

EngramGate = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def original_engram_gate(x: torch.Tensor, kv: torch.Tensor, qw: torch.Tensor,
                         kw: torch.Tensor, mask: torch.Tensor | None,
                         eps: float) -> torch.Tensor:
    """Official reference product/reduction boundaries, all on input device."""
    copies, dim = x.shape[-2:]
    keys, value = kv.split([copies * dim, dim], dim=-1)
    keys = keys.float().unflatten(-1, (copies, dim))
    hidden = x.float()
    weight = qw.float() * kw.float()
    rstd = torch.rsqrt(hidden.square().mean(-1) + eps) * torch.rsqrt(keys.square().mean(-1) + eps)
    dot = (hidden * weight * keys).sum(-1) * rstd * dim**-.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if mask is not None:
        gate = gate.masked_fill(~mask.reshape(-1, 1), 0)
    return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(1)).to(x.dtype)


def select_engram_gate(device: torch.device, dtype: torch.dtype, dim: int,
                       copies: int, eps: float, rotated: bool) -> EngramGate | None:
    if device.type != 'musa':
        return None
    mode = os.environ.get('XLLM_DSV41_MUSA_ENGRAM_GATE', 'reference')
    if mode not in ('reference', 'original', 'sglang_fused'):
        raise ValueError(f'Invalid MUSA Engram gate mode: {mode!r}')
    if mode == 'reference':
        return None
    if rotated or dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('Original MUSA Engram gate requires unrotated BF16/FP32 activations')
    if not math.isfinite(eps) or eps<=0:
        raise ValueError('Engram normalization epsilon must be positive and finite')
    logger.info('V4.1 Engram gate mode=%s, dim=%s, copies=%s, eps=%.17g', mode, dim, copies, eps)
    if mode == 'original':
        return lambda x, kv, qw, kw, mask: original_engram_gate(x, kv, qw, kw, mask, eps)
    # ModelArgs stores FP32 before PyDictVisitor widens it to a Python float.
    # Compare the actual scalar type used by the GPU computation, not FP64
    # spelling; 9.999999682655225e-21 and1e-20 produce identical FP32 bits.
    eps_matches = eps<=1 and struct.pack('=f',eps)==struct.pack('=f',1e-20)
    if dim != 5120 or copies != 4 or not eps_matches:
        raise ValueError('Fused Engram gate requires original V4.1 Flash D5120/HC4/eps1e-20')
    import xllm_runtime
    if not hasattr(xllm_runtime, 'musa_v41_engram_gate'):
        raise RuntimeError('Rebuild xllm with V4.1 Engram gate fusion')
    return xllm_runtime.musa_v41_engram_gate
