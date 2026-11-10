# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Opt-in V4 kernel reuse for V4.1's unweighted shared-expert activation.

Select once during model construction, not on each token. Routed experts
must retain their FP32 routing-weight multiplication before the BF16 cast.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import torch

from scripts.logger import logger

SharedActivation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_MODE_ENV = "XLLM_DSV41_MUSA_SHARED_SWIGLU"


def select_shared_swiglu(
    device: torch.device, dtype: torch.dtype, limit: float
) -> SharedActivation | None:
    """Return an explicitly selected fused activation, or the reference marker.

    Other platforms retain their current implementation. An invalid mode or
    an unsupported MUSA configuration is an initialization error, never an
    implicit fallback after a failed device operation.
    """
    if device.type != "musa":
        return None
    mode = os.environ.get(_MODE_ENV, "reference")
    if mode not in ("reference", "v4_fused"):
        raise ValueError(f"{_MODE_ENV} must be reference or v4_fused, got {mode!r}")
    logger.info("V4.1 MUSA shared SwiGLU: mode=%s, dtype=%s, clamp=%s", mode, dtype, limit)
    if mode == "reference":
        return None
    if dtype != torch.bfloat16 or not math.isfinite(limit) or not 0.0 < limit < 1_000_000.0:
        raise ValueError("V4 fused shared SwiGLU requires BF16 and a finite clamp in (0, 1000000)")
    import xllm_runtime

    if not hasattr(xllm_runtime, "musa_shared_swiglu"):
        raise RuntimeError("Rebuild the MUSA xllm binary: xllm_runtime.musa_shared_swiglu is missing")
    native_op = xllm_runtime.musa_shared_swiglu

    def activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "musa" or gate.device != up.device:
            raise ValueError("Shared SwiGLU inputs must be on the same MUSA device")
        if gate.dtype != torch.bfloat16 or up.dtype != gate.dtype:
            raise ValueError("Shared SwiGLU inputs must both be BF16")
        if gate.ndim != 2 or gate.shape != up.shape or gate.shape[1] == 0:
            raise ValueError("Shared SwiGLU inputs must have matching [tokens, intermediate] shapes")
        # Include this packing and the native output allocation in benchmarks.
        return native_op(torch.cat((gate, up), dim=-1), limit)

    return activation
