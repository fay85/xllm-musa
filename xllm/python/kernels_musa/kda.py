# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Adapt GLM's raw KDA gates and value-first states to MATE 0.2.8.post1."""

from __future__ import annotations

import math

import torch
from mate.kda import chunk_kda, gated_delta_rule_decode


def recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.Tensor | None = None,
    prefill_mode: str = "chunk",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run chunk Prefill or recurrent Decode without mutating the caller state.

    Gate and beta inputs are logits: MATE applies the GLM safe gate and
    sigmoid internally, exactly once. Both paths consume and return FP32
    state in [sequence, head, value, key] order. The optional recurrent prefill
    uses the decode kernel sequentially; packed inputs remain chunk-only.
    The scheduler owns slots;
    this adapter operates on dense snapshots, including prefix copy-on-write.
    """
    if prefill_mode not in ("chunk", "recurrent"):
        raise ValueError("MUSA KDA prefill mode must be chunk or recurrent")
    if prefill_mode == "recurrent" and cu_seqlens is not None:
        raise NotImplementedError("Recurrent KDA prefill requires dense per-sequence inputs")
    tensors = (query, key, value, raw_gate, beta_raw, a_log, dt_bias, initial_state)
    if query.device.type != "musa" or any(t.device != query.device for t in tensors):
        raise ValueError("KDA tensors must share one MUSA device")
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("GLM KDA requires equal [B,T,H,128] Q/K/V")
    batch, tokens, heads, width = query.shape
    if min(batch, tokens, heads) <= 0 or width != 128:
        raise ValueError("GLM KDA requires nonempty inputs and head dimension 128")
    if query.dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != query.dtype for t in (key, value, raw_gate, beta_raw)
    ):
        raise ValueError("MATE KDA Q/K/V/gate/beta must have one BF16/FP16 dtype")
    if raw_gate.shape != query.shape or beta_raw.shape != query.shape[:3]:
        raise ValueError("GLM KDA raw gate/beta geometry mismatch")
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("This GLM MATE adapter requires a finite negative safe gate bound")
    if a_log.numel() != heads or dt_bias.numel() != heads * width:
        raise ValueError("GLM KDA A_log/dt_bias geometry mismatch")
    sequences = batch
    if cu_seqlens is not None:
        if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.device != query.device:
            raise ValueError("Packed KDA requires batch one and MUSA cumulative lengths")
        sequences = cu_seqlens.numel() - 1
    if initial_state.dtype != torch.float32 or initial_state.shape != (sequences, heads, width, width):
        raise ValueError("GLM KDA state must be FP32 [sequence,head,value,key]")
    # MATE accepts strided projections only along supported axes. Canonicalize
    # the last dimension at this boundary, without a host compute path.
    q, k, v, gate, beta = (t.contiguous() for t in (query, key, value, raw_gate, beta_raw))
    state = initial_state.contiguous()
    if prefill_mode == "recurrent" and tokens > 1:
        # The same fused MATE recurrence executes every token in both modes.
        # This avoids chunk/recurrent arithmetic differences but sacrifices
        # parallel prefill throughput; select it explicitly for qualification.
        parts = []
        for index in range(tokens):
            part, state = recurrent_kda(
                q[:, index : index + 1], k[:, index : index + 1], v[:, index : index + 1],
                gate[:, index : index + 1], beta[:, index : index + 1], a_log, dt_bias, state,
                lower_bound=lower_bound, prefill_mode="recurrent",
            )
            parts.append(part)
        return torch.cat(parts, 1), state
    if tokens > 1 or cu_seqlens is not None:
        return chunk_kda(
            q, k, v, gate, beta, scale=width**-0.5,
            initial_state=state, output_final_state=True,
            cu_seqlens=cu_seqlens, A_log=a_log.float().contiguous(),
            dt_bias=dt_bias.float().reshape(heads, width).contiguous(),
            lower_bound=lower_bound, use_qk_l2norm_in_kernel=True,
        )
    output, final_state = gated_delta_rule_decode(
        q=q, k=k, v=v, state=state,
        A_log=a_log.float().contiguous(), a=gate,
        dt_bias=dt_bias.float().reshape(heads, width).contiguous(),
        num_accepted_tokens=None, cu_seqlens=None, b=beta,
        scale=width**-0.5, use_qk_l2norm=True,
        lower_bound=lower_bound, use_gate_in_kernel=True,
        use_lower_bound=True, apply_beta_sigmoid=True,
        allow_neg_eigval=False, use_initial_state=True, state_v_first=True,
        is_varlen=False, store_final_state=True, inplace_final_state=False,
        state_dtype=torch.float32,
    )
    if final_state is None:
        raise RuntimeError("MATE did not return the requested KDA final state")
    return output, final_state
