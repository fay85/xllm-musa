# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Model-level adapters for MATE normalization and hyper-connections."""

import torch
from mate.hyperconnection import mhc_pre
from mate.norm import rmsnorm


def rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Flatten token axes; preserve the model's normalization and output shape."""
    if value.device.type != "musa" or weight.device != value.device:
        raise ValueError("MATE RMSNorm requires MUSA input and weight")
    return rmsnorm(value.reshape(-1, value.shape[-1]), weight, eps=epsilon).reshape_as(value)


def hc_pre_fused(
    hidden: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, hc_mult: int, hc_sinkhorn_iters: int,
    norm_eps: float, hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map MATE's (post,comb,input) result to xLLM's (input,post,comb)."""
    if hidden.device.type != "musa" or hidden.shape[-2] != hc_mult:
        raise ValueError("MATE mHC input must have the configured MUSA residual streams")
    post, combination, collapsed = mhc_pre(
        hidden, hc_fn, hc_scale, hc_base,
        rms_eps=norm_eps, mhc_pre_eps=hc_eps, mhc_sinkhorn_eps=hc_eps,
        mhc_post_mult_value=2.0, sinkhorn_repeat=hc_sinkhorn_iters,
    )
    return collapsed, post.squeeze(-1), combination
