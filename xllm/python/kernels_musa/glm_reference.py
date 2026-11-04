# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness-first GLM primitives implemented with real MUSA Torch ops.

These are explicitly UNFUSED GPU reference implementations. The hc_pre_fused
name preserves the model's interface, not a claim of a single fused kernel.
mHC semantics follow the INTERNAL A tree's hyper_connection_test.cpp FP32
reference, including pre epsilon, doubled post gate and Sinkhorn order.
There is no CPU inference fallback or exception-driven execution fallback.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def _require_musa(*tensors: torch.Tensor) -> None:
    if not tensors or tensors[0].device.type != "musa":
        raise ValueError("GLM MUSA reference kernels require MUSA tensors")
    if any(tensor.device != tensors[0].device for tensor in tensors):
        raise ValueError("GLM MUSA inputs must be on the same device")


def rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Normalize in FP32, apply weight, and cast once to the input dtype."""
    _require_musa(value, weight)
    if value.dim() < 1 or weight.dim() != 1 or value.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight must match the last input dimension")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    work = value.float()
    output = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + epsilon)
    return (output * weight.float()).to(value.dtype)


def rms_norm_sigmoid_gated(
    value: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """GLM uses sigmoid gating, not the SiLU gate used by Qwen GDN."""
    _require_musa(value, gate, weight)
    if (
        gate.shape != value.shape
        or weight.dim() != 1
        or weight.numel() != value.shape[-1]
    ):
        raise ValueError("Sigmoid RMSNorm input/gate/weight shapes do not match")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    work = value.float()
    normalized = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + epsilon)
    return (normalized * weight.float() * gate.float().sigmoid()).to(value.dtype)


def batch_matmul_transpose(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project [tokens,heads,in] by [heads,in,out], without CUDA aliases."""
    _require_musa(value, weight)
    if value.dim() != 3 or weight.dim() != 3 or value.shape[1:] != weight.shape[:2]:
        raise ValueError("MLA projection expects [T,H,D] and [H,D,O]")
    if value.dtype != weight.dtype:
        raise ValueError("MLA projection input and weight dtypes must match")
    return torch.bmm(value.transpose(0, 1), weight).transpose(0, 1).contiguous()


def hc_pre_fused(
    hidden: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unfused MUSA mHC reference implementing the model's HcPre contract."""
    _require_musa(hidden, hc_fn, hc_scale, hc_base)
    if hidden.dim() not in (3, 4) or hc_mult != 4 or hidden.shape[-2] != hc_mult:
        raise ValueError(
            "Initial MUSA mHC expects three/four-dimensional four-stream input"
        )
    width = hidden.shape[-1]
    mix_width = (2 + hc_mult) * hc_mult
    if (
        hc_fn.shape != (mix_width, hc_mult * width)
        or hc_scale.shape != (3,)
        or hc_base.shape != (mix_width,)
    ):
        raise ValueError(
            "mHC projection/scale/base shapes do not match the stream geometry"
        )
    if any(value.dtype != torch.float32 for value in (hc_fn, hc_scale, hc_base)):
        raise ValueError("mHC requires prepared FP32 projection/scale/base parameters")
    if hc_sinkhorn_iters < 1 or any(
        not math.isfinite(eps) or eps <= 0 for eps in (norm_eps, hc_eps)
    ):
        raise ValueError(
            "mHC requires positive Sinkhorn iterations and finite positive epsilons"
        )
    work = hidden.float()
    flat = work.flatten(-2)
    inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = functional.linear(flat, hc_fn) * inverse_rms
    pre = (mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]).sigmoid() + hc_eps
    post = (
        2
        * (
            mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        ).sigmoid()
    )
    combination = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    combination = combination.reshape(*hidden.shape[:-2], hc_mult, hc_mult)
    combination = combination.softmax(-1) + hc_eps
    combination = combination / (combination.sum(-2, keepdim=True) + hc_eps)
    for _ in range(1, hc_sinkhorn_iters):
        combination = combination / (combination.sum(-1, keepdim=True) + hc_eps)
        combination = combination / (combination.sum(-2, keepdim=True) + hc_eps)
    output = (work * pre.unsqueeze(-1)).sum(-2).to(hidden.dtype)
    return output, post, combination


def hc_post(
    value: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combination: torch.Tensor,
) -> torch.Tensor:
    """Place the new value and apply comb-transpose to the old streams."""
    _require_musa(value, residual, post, combination)
    if residual.shape[:-2] != value.shape[:-1] or residual.shape[-1] != value.shape[-1]:
        raise ValueError("mHC output and residual geometries do not match")
    if post.shape != residual.shape[:-1] or combination.shape != (
        *residual.shape[:-1],
        residual.shape[-2],
    ):
        raise ValueError("mHC post/combination geometries do not match the streams")
    mixed = torch.einsum("...ij,...id->...jd", combination.float(), residual.float())
    return (post.float().unsqueeze(-1) * value.float().unsqueeze(-2) + mixed).to(
        value.dtype
    )
