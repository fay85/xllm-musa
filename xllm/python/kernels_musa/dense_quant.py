# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Original FP8 input rounding for already dequantized dense weights.

This is an explicit numerical integration path, not an FP8 GEMM. Only the
checkpoint loader knows which projections originally had FP8 weights: BF16
compressors, gates, Indexer weights_proj/wk and mHC must not be wrapped.
"""
from __future__ import annotations

import os
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from scripts.logger import logger

Fp8Quant = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class V41OriginalFp8Dense(nn.Module):
    """Share the loaded BF16 parameter, rounding activations before GEMM."""

    def __init__(self, original: nn.Linear, quantize: Fp8Quant) -> None:
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self._quantize = quantize

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError("Original FP8 dense input must end in the loaded K dimension")
        value, _ = self._quantize(x.reshape(-1, self.in_features).contiguous())
        return value.reshape_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self._prepare(x), self.weight, self.bias)


def share_original_fp8_input(
    x: torch.Tensor, gate: nn.Module, up: nn.Module
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Round the shared gate/up input once when both have the same contract."""
    if not isinstance(gate, V41OriginalFp8Dense) or not isinstance(up, V41OriginalFp8Dense):
        return None
    if gate.in_features != up.in_features or gate._quantize is not up._quantize:
        return None
    value = gate._prepare(x)
    return F.linear(value, gate.weight, gate.bias), F.linear(value, up.weight, up.bias)


def select_original_fp8_dense(module: nn.Module) -> nn.Module:
    """Called only after loading/dequantizing an original FP8 projection."""
    if isinstance(module, V41OriginalFp8Dense):
        raise ValueError("Original FP8 dense activation quantization was already installed")
    if not isinstance(module, nn.Linear):
        raise TypeError("Original FP8 dense adaptation requires a plain loaded nn.Linear")
    if module.weight.device.type != "musa":
        return module
    mode = os.environ.get("XLLM_DSV41_MUSA_DENSE_QUANT", "reference")
    if mode not in ("reference", "original_qdq"):
        raise ValueError(f"Invalid original FP8 dense mode: {mode!r}")
    if mode == "reference":
        return module
    if module.weight.dtype != torch.bfloat16 or module.in_features % 32 or module.bias is not None:
        raise ValueError("Original FP8 dense QDQ requires BF16 weights, K divisible by32 and no bias")
    import xllm_runtime

    if not hasattr(xllm_runtime, "musa_v41_block32_qdq"):
        raise RuntimeError("Rebuild xllm with original V4.1 block32 quantization")
    logger.info("Original FP8 dense activation QDQ selected for loaded shape (%d,%d); GEMM remains BF16",
                module.out_features, module.in_features)
    return V41OriginalFp8Dense(module, xllm_runtime.musa_v41_block32_qdq)
