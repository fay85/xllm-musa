# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Explicit original-E2M1 expert linear modes with an optional block32 QDQ."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from scripts.logger import logger

FP4Linear = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def select_fp4_linear(
    device: torch.device,
    dtype: torch.dtype,
    packed_fp4: bool,
    dequantize: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> FP4Linear | None:
    if device.type != "musa" or not packed_fp4:
        return None
    mode = os.environ.get("XLLM_DSV41_MUSA_FP4_LINEAR", "reference")
    if mode not in ("reference", "packed_bf16", "block32", "partition32"):
        raise ValueError(f"Invalid V4.1 packed FP4 mode: {mode!r}")
    logger.info("V4.1 packed FP4 linear mode=%s (partition32 uses BF16 mantissas)", mode)
    if mode == "reference":
        return None
    if dtype != torch.bfloat16:
        raise ValueError("Packed FP4 expert linear requires BF16 model activations")
    import xllm_runtime
    required = ("musa_v41_block32_qdq", "musa_v41_packed_fp4_gemv")
    if any(not hasattr(xllm_runtime, name) for name in required):
        raise RuntimeError("Rebuild MUSA xllm with V4.1 FP4/block32 primitives")

    bmm_so_path = ""
    max_partial_bytes = 512 * 1024 * 1024
    if mode == "partition32":
        bmm_so_path = os.environ.get("XLLM_DSV41_MATE_BMM_SO", "")
        if not bmm_so_path or not Path(bmm_so_path).is_absolute() or not Path(bmm_so_path).is_file():
            raise ValueError("partition32 requires an existing absolute XLLM_DSV41_MATE_BMM_SO path")
        if not hasattr(xllm_runtime, "musa_v41_partition_linear"):
            raise RuntimeError("Rebuild xllm with the native partition32 MATE adapter")

    def _linear(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or weight.ndim != 2 or scales.ndim != 2:
            raise ValueError("Expected input[M,K], packed weight[N,K/2], scales[N,K/32]")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.uint8 or scales.dtype != torch.uint8:
            raise ValueError("Expected BF16 input and original uint8 E2M1/UE8M0 storage")
        if x.device.type != "musa" or weight.device != x.device or scales.device != x.device:
            raise ValueError("Expert tensors must share the configured MUSA device")
        if x.shape[1] <= 0 or x.shape[1] % 32 or weight.shape[0] <= 0:
            raise ValueError("Original FP4 experts need positive dimensions and K divisible by32")
        if weight.shape[1] * 2 != x.shape[1] or tuple(scales.shape) != (weight.shape[0], x.shape[1] // 32):
            raise ValueError("FP4 packed weight/scale shape mismatch")
        value = x.contiguous()
        if mode == "partition32":
            groups = x.shape[1] // 32
            if groups * weight.shape[0] * 4 > max_partial_bytes:
                raise ValueError("partition32 byte budget cannot fit even one output row")
            # The native adapter tiles M before allocation, bounding partials
            # to512MiB without changing the chosen quantization semantics.
            return xllm_runtime.musa_v41_partition_linear(
                value, weight.contiguous(), scales.contiguous(), bmm_so_path, max_partial_bytes,
            )
        if mode == "block32":
            value, _ = xllm_runtime.musa_v41_block32_qdq(value)
        # Capability selection happens before execution; an operator failure
        # always propagates. Both shapes use the SAME selected A-quantization.
        if x.shape[0] <= 8:
            return xllm_runtime.musa_v41_packed_fp4_gemv(value, weight.contiguous(), scales.contiguous())
        return F.linear(value, dequantize(weight, scales))

    return _linear


def select_fp4_decode(device: torch.device, dtype: torch.dtype, packed_fp4: bool) -> Callable | None:
    """Select fixed-launch local experts; caller owns EP collective/shared add."""
    if device.type != "musa" or not packed_fp4:
        return None
    mode = os.environ.get("XLLM_DSV41_MUSA_FP4_MOE", "reference")
    if mode not in ("reference", "v41_decode"):
        raise ValueError(f"Invalid V4.1 packed MoE mode: {mode!r}")
    if mode == "reference":
        return None
    linear_mode = os.environ.get("XLLM_DSV41_MUSA_FP4_LINEAR", "reference")
    if dtype != torch.bfloat16 or linear_mode not in ("packed_bf16", "block32", "partition32"):
        raise ValueError("Decode MoE requires BF16 and an explicit packed_bf16/block32 linear mode")
    import xllm_runtime
    if not hasattr(xllm_runtime, "musa_v41_decode_moe"):
        raise RuntimeError("Rebuild xllm with V4.1 decode MoE kernels")
    logger.info("V4.1 GPU local expert decode selected; activation mode=%s", linear_mode)
    native = xllm_runtime.musa_v41_decode_moe
    block32 = linear_mode in ("block32", "partition32")

    def _decode(
        x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
        gate: torch.Tensor, gate_scale: torch.Tensor,
        up: torch.Tensor, up_scale: torch.Tensor,
        down: torch.Tensor, down_scale: torch.Tensor,
        first_expert: int, limit: float,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[0] > 8 or x.dtype != torch.bfloat16 or x.device.type != "musa":
            raise ValueError("GPU decode MoE requires BF16 MUSA input[M<=8,H]")
        if ids.dtype != torch.int64 or weights.dtype != torch.float32 or ids.shape != weights.shape:
            raise ValueError("Decode routes require int64 IDs and FP32 weights with equal shape")
        return native(x.contiguous(), ids.contiguous(), weights.contiguous(),
                      gate, gate_scale, up, up_scale, down, down_scale,
                      first_expert, limit, block32)

    return _decode
