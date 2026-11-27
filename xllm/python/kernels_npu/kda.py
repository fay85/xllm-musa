# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU kernel bindings for KDA speculative verification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch


class KdaSafeGateParameters(NamedTuple):
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float


def recurrent_kda_verify(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    *,
    safe_gate: KdaSafeGateParameters | None = None,
) -> torch.Tensor:
    """Read the accepted checkpoint and write the new checkpoints in place.

    The kernel consumes packed TND tensors and a request-by-checkpoint slot
    table. Each request reads checkpoint column num_accepted_tokens - 1
    and writes one checkpoint per verify row. With safe_gate, gate contains
    raw projections.
    """
    from fla_npu.ops.ascendc import recurrent_kda

    if safe_gate is None:
        gate_kwargs = dict(use_gate_in_kernel=False)
    else:
        gate_kwargs = dict(
            A_log=safe_gate.a_log,
            dt_bias=safe_gate.dt_bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=safe_gate.lower_bound,
        )
    result = recurrent_kda(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=ssm_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        layout="TND",
        scale=1.0 / (query.shape[-1] ** 0.5),
        output_final_state=True,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        **gate_kwargs,
    )
    return result[0] if isinstance(result, tuple) else result


def build_kda_replayssm_state_indices(slots: Sequence[int | None]) -> torch.Tensor:
    """Build CPU int32 indices from distinct live slots and None for padding."""
    indices: list[int] = []
    live_slots: set[int] = set()
    for slot in slots:
        if slot is None:
            indices.append(-1)
            continue
        if slot < 0:
            raise ValueError(f"invalid live ReplaySSM slot {slot}")
        if slot in live_slots:
            raise ValueError(f"duplicate ReplaySSM slot {slot}")
        live_slots.add(slot)
        indices.append(slot)
    return torch.tensor(indices, dtype=torch.int32, device="cpu")


def kda_replayssm_verify_record(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    normalized_key: torch.Tensor,
    correction: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """Verify eight tokens per request and write the FP32 ReplaySSM records.

    Q/K/V are BF16 TND tensors with eight 128-wide heads. Slots select the
    read-only committed states and the corresponding writable record rows.
    Construct state_indices from unique live slots and use -1 for padding,
    e.g. with build_kda_replayssm_state_indices before transferring to NPU.
    """
    return torch.ops.xllm_ops.kda_replayssm_verify_record(
        query,
        key,
        value,
        raw_gate,
        beta,
        ssm_state,
        a_log,
        dt_bias,
        state_indices,
        normalized_key,
        correction,
        decay,
        1.0 / (query.shape[-1] ** 0.5),
    )


def kda_replayssm_correction_fold(
    states: Sequence[torch.Tensor],
    normalized_keys: Sequence[torch.Tensor],
    corrections: Sequence[torch.Tensor],
    decays: Sequence[torch.Tensor],
    state_indices: torch.Tensor,
    accepted_lengths: torch.Tensor,
) -> None:
    """Apply accepted prefixes in place to disjoint layer states using the verify indices."""
    torch.ops.xllm_ops.kda_replayssm_correction_fold(
        states, normalized_keys, corrections, decays, state_indices, accepted_lengths
    )
