# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Independent V4 kernel reuse, preserving V4.1's carried pre_mix ordering.

Select each primitive at construction, before state is touched. No MUSA
failure is caught or silently replaced by a reference computation.
"""

from __future__ import annotations

import math
import os
from typing import Callable, NamedTuple

import torch

from scripts.logger import logger

Pre = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Post = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
Metadata = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class HyperConnectionOps(NamedTuple):
    pre: Pre | None
    post: Post | None
    metadata: Metadata | None


def _check_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(f"mHC requires {device}/{dtype}/{shape}, got {tensor.device}/{tensor.dtype}/{tuple(tensor.shape)}")


def select_hyper_connection_ops(
    device: torch.device, dtype: torch.dtype, hc_mult: int, iterations: int, eps: float
) -> HyperConnectionOps:
    """Return independently enabled GPU operations, or reference markers."""
    if device.type != "musa":
        return HyperConnectionOps(None, None, None)
    modes = {part: os.environ.get(f"XLLM_DSV41_MUSA_HC_{part.upper()}", "reference") for part in ("pre", "post", "metadata")}
    for part, mode in modes.items():
        if mode not in ("reference", "v4_fused"):
            raise ValueError(f"MUSA mHC {part} must be reference or v4_fused, got {mode!r}")
    logger.info("V4.1 MUSA mHC: %s", modes)
    if all(mode == "reference" for mode in modes.values()):
        return HyperConnectionOps(None, None, None)
    if dtype != torch.bfloat16 or hc_mult != 4:
        raise ValueError("V4.1 fused mHC requires BF16 and hc_mult=4")
    if not 1 <= iterations <= 1000 or not math.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("Invalid fused mHC Sinkhorn iterations or epsilon")
    import xllm_runtime

    names = {"pre": "musa_hc_pre_combine", "post": "musa_hc_post", "metadata": "musa_hc_metadata"}
    for part, mode in modes.items():
        if mode == "v4_fused" and not hasattr(xllm_runtime, names[part]):
            raise RuntimeError(f"Rebuild MUSA xllm: xllm_runtime.{names[part]} is missing")

    def pre(residual: torch.Tensor, carried_pre: torch.Tensor) -> torch.Tensor:
        if residual.device.type != "musa" or residual.dtype != torch.bfloat16 or residual.ndim != 3 or residual.shape[1] != 4 or residual.shape[2] == 0:
            raise ValueError("mHC pre requires a MUSA BF16 [T,4,H] residual, H>0")
        _check_tensor(carried_pre, residual.device, torch.float32, (residual.shape[0], 4))
        return xllm_runtime.musa_hc_pre_combine(residual.contiguous(), carried_pre.contiguous())

    def post(value: torch.Tensor, residual: torch.Tensor, coefficients: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        if residual.device.type != "musa" or residual.dtype != torch.bfloat16 or residual.ndim != 3 or residual.shape[1] != 4 or residual.shape[2] == 0:
            raise ValueError("mHC post requires a MUSA BF16 [T,4,H] residual, H>0")
        rows, _, hidden = residual.shape
        _check_tensor(value, residual.device, torch.bfloat16, (rows, hidden))
        _check_tensor(coefficients, residual.device, torch.float32, (rows, 4))
        _check_tensor(comb, residual.device, torch.float32, (rows, 4, 4))
        return xllm_runtime.musa_hc_post(value.contiguous(), residual.contiguous(), coefficients.contiguous(), comb.contiguous())

    def metadata(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mixes.device.type != "musa" or mixes.dtype != torch.float32 or mixes.ndim != 2 or mixes.shape[1] != 24:
            raise ValueError("mHC metadata requires MUSA FP32 [T,24] mixes")
        _check_tensor(scale, mixes.device, torch.float32, (3,))
        _check_tensor(base, mixes.device, torch.float32, (24,))
        return xllm_runtime.musa_hc_metadata(mixes.contiguous(), scale.contiguous(), base.contiguous(), iterations, eps)

    return HyperConnectionOps(
        pre if modes["pre"] == "v4_fused" else None,
        post if modes["post"] == "v4_fused" else None,
        metadata if modes["metadata"] == "v4_fused" else None,
    )
