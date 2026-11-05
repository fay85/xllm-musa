# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""MATE block-FP8 experts with explicit workspace and reduced-output contracts.

Attention TP ranks supply replicated rows. Each EP rank owns complete experts;
MATE returns the complete routed result for that rank's input rows. The caller
must reduce TP-sharded shared experts separately, never reduce this result again.
"""

from __future__ import annotations

import math
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


class MegaMoEWorkspace:
    """One sequentially reused IPC workspace for a model's homogeneous layers."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        experts: int,
        hidden: int,
        intermediate: int,
        topk: int,
        max_tokens: int,
    ) -> None:
        from mate.mega_moe import get_symm_buffer_for_mega_moe

        if max_tokens < 129 or group.size() > 8 or experts % group.size():
            raise ValueError("MegaMoE requires capacity >=129 and an intranode expert partition")
        if hidden <= 0 or hidden % 256 or intermediate <= 0 or intermediate % 128:
            raise ValueError("MATE split-prefill requires hidden divisible by 256 and intermediate by 128")
        if not 0 < topk <= experts:
            raise ValueError("MegaMoE top-k must lie within the expert count")
        self.max_tokens = max_tokens
        self.group = group
        self.geometry = (experts, hidden, intermediate, topk)
        # MATE exchanges CPU IPC handles with all_gather_object. Some MUSA
        # bridges advertise CUDA for MCCL object collectives; do not allocate
        # CUDA tensors for this host metadata. Expert computation still uses
        # MUSA IPC, and the model's TP collectives remain MCCL.
        self.ipc_group = dist.new_group(
            ranks=dist.get_process_group_ranks(group),
            backend="gloo",
            timeout=timedelta(seconds=120),
            use_local_synchronization=True,
        )
        # Keep the safe worst-case pool; do not shrink it based on expected routing.
        self.buffer = get_symm_buffer_for_mega_moe(
            self.ipc_group,
            num_experts=experts,
            num_max_tokens_per_rank=max_tokens,
            num_topk=topk,
            hidden=hidden,
            intermediate_hidden=intermediate,
        )

    def close(self) -> None:
        """Call collectively after all model work, before process-group teardown."""
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
            dist.destroy_process_group(self.ipc_group)


class MusaMegaMoEExperts(nn.Module):
    """Preserve original E4M3/block scales until MATE's layout transformation."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down: torch.Tensor,
        down_scale: torch.Tensor,
        workspace: MegaMoEWorkspace,
        activation_clamp: float,
    ) -> None:
        super().__init__()
        from mate.mega_moe import transform_weights_for_mega_moe

        experts, hidden, intermediate, _ = workspace.geometry
        local_experts = experts // workspace.group.size()
        for name, weight, scale, shape in (
            ("gate_up", gate_up, gate_up_scale, (local_experts, 2 * intermediate, hidden)),
            ("down", down, down_scale, (local_experts, hidden, intermediate)),
        ):
            if weight.dtype != torch.float8_e4m3fn or weight.device.type not in ("cpu", "musa"):
                raise ValueError(f"{name} requires original FP8 checkpoint weights")
            if tuple(weight.shape) != shape or scale.dtype != torch.float32:
                raise ValueError(f"{name} weight/scale geometry does not match the expert partition")
            if tuple(scale.shape) != (shape[0], shape[1] // 128, shape[2] // 128):
                raise ValueError(f"{name} requires canonical 128x128 block scales")
            if scale.device != weight.device:
                raise ValueError(f"{name} scales must reside with the weights")
        if not math.isfinite(activation_clamp) or activation_clamp <= 0:
            raise ValueError("GLM SwiGLU clamp must be positive")
        # Offline byte/layout transformation only. On this TorchMUSA build,
        # MATE's GPU arange factory dispatches to CUDA; CPU packing preserves
        # the original FP8 bytes/scales, without dequantization or CPU inference.
        l1, l2 = transform_weights_for_mega_moe(
            (gate_up.cpu().contiguous(), gate_up_scale.cpu().contiguous()),
            (down.cpu().contiguous(), down_scale.cpu().contiguous()),
        )
        device = workspace.buffer.buffer.device
        if device.type != "musa":
            raise ValueError("MegaMoE workspace must be allocated on MUSA")
        self.register_buffer("gate_up", l1[0].to(device))
        self.register_buffer("gate_up_scale", l1[1].to(device))
        self.register_buffer("down", l2[0].to(device))
        self.register_buffer("down_scale", l2[1].to(device))
        self.workspace = workspace
        self.activation_clamp = activation_clamp

    def forward(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        from mate.mega_moe import fp8_fp8_mega_moe

        count = hidden.shape[0]
        _, width, _, topk = self.workspace.geometry
        if hidden.ndim != 2 or hidden.shape[1] != width or hidden.dtype != torch.bfloat16:
            raise ValueError("MegaMoE inputs must be BF16 [tokens, hidden]")
        if not 0 < count <= self.workspace.max_tokens:
            raise ValueError("MegaMoE token count exceeds configured scheduler capacity")
        if ids.shape != (count, topk) or weights.shape != ids.shape:
            raise ValueError("MegaMoE routing must contain one top-k row per token")
        buf = self.workspace.buffer
        if buf is None:
            raise RuntimeError("MegaMoE workspace is closed")
        # Clear sentinel/padding on every invocation, including prefill->decode.
        # MATE 0.2.8.post1's decode stage accepts but does not forward clamp.
        # Select its clamped split-prefill implementation explicitly; invalid
        # padded routes contribute nothing. This preserves GLM math for decode
        # too, until a validated clamped decode kernel is available.
        execution_rows = max(count, 129)
        rows = min(buf.x.shape[0], execution_rows + 1)
        buf.x[:rows].zero_()
        buf.x_sf[:rows].zero_()
        buf.topk_idx[:rows].fill_(-1)
        buf.topk_weights[:rows].zero_()
        groups = hidden.float().reshape(count, width // 128, 128)
        scales = groups.abs().amax(-1).clamp_min(1e-4) / 448.0
        quantized = (groups / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        buf.x[:count].copy_(quantized.reshape(count, width))
        buf.x_sf[:count].copy_(scales)
        buf.topk_idx[:count].copy_(ids.to(torch.int64))
        buf.topk_weights[:count].copy_(weights.float())
        # Peer IPC reads must not race another rank's asynchronous dispatch.
        torch.musa.synchronize(hidden.device)
        dist.barrier(group=self.workspace.ipc_group)
        output = torch.empty((execution_rows, width), dtype=hidden.dtype, device=hidden.device)
        fp8_fp8_mega_moe(
            output,
            (self.gate_up, self.gate_up_scale),
            (self.down, self.down_scale),
            buf,
            activation="swiglu",
            activation_clamp=self.activation_clamp,
        )
        torch.musa.synchronize(hidden.device)
        dist.barrier(group=self.workspace.ipc_group)
        return output[:count]
