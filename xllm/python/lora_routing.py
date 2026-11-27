# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Python-side LoRA routing state for python-impl models.
#
# The API process assigns per-sequence adapter ids; the executor forwards them
# here, the model hooks read them back, and the ACL graph runner writes the
# per-token slot ids into the captured buffer. Keeping the state in one module
# avoids threading it through every forward signature.

from __future__ import annotations

import os
from contextvars import ContextVar

import torch

_ADAPTER_IDS: ContextVar[tuple[int, ...] | None] = ContextVar(
    "xllm_lora_adapter_ids", default=None
)
_Q_SEQ_LENS: ContextVar[tuple[int, ...] | None] = ContextVar(
    "xllm_lora_q_seq_lens", default=None
)

# Per-token adapter slot ids as a graph input. The ACL graph runner owns the
# buffer and copies this step's routing into it before replay; capture bakes
# the buffer's address, so the delta ops read fresh ids.
_SLOT_IDS: ContextVar[torch.Tensor | None] = ContextVar(
    "xllm_lora_slot_ids", default=None
)
_SLOT_ZERO_CACHE: dict[tuple[int, str], torch.Tensor] = {}
# Which captured graph variant is running: base (no delta ops) or adapter.
# The runner sets this around capture/eager so one bucket can own both graphs.
_SLOT_ACTIVE: ContextVar[bool] = ContextVar(
    "xllm_lora_slot_active", default=False
)


def set_lora_routing(adapter_ids, q_seq_lens) -> None:
    """Record this batch's per-sequence adapter routing."""
    _ADAPTER_IDS.set(tuple(adapter_ids) if adapter_ids else None)
    _Q_SEQ_LENS.set(tuple(q_seq_lens) if q_seq_lens else None)


def active_adapter_ids() -> tuple[int, ...] | None:
    return _ADAPTER_IDS.get()


def slot_graph_enabled() -> bool:
    """Whether the graph-compatible fixed-slot delta path is enabled."""
    if os.environ.get("XLLM_LORA_SLOT_GRAPH", "") == "1":
        return True
    # Backward compatibility with the flag used by the first pilot deployment.
    return (
        os.environ.get("XLLM_LORA_SPIKE", "") == "1"
        and os.environ.get("XLLM_LORA_SPIKE_SLOTS", "") == "1"
    )


def slot_mode() -> bool:
    return slot_graph_enabled()


def set_slot_active(active: bool) -> None:
    _SLOT_ACTIVE.set(active)


def slot_active() -> bool:
    # Deliberately independent of slot_graph_enabled(): eager/prefill steps do
    # not replay a graph, so the executor sets this per batch from the routing.
    # The graph runner still overrides it per captured variant (base vs
    # adapter) before replay.
    return _SLOT_ACTIVE.get()


def has_adapter_routing() -> bool:
    ids = _ADAPTER_IDS.get()
    return bool(ids) and any(ids)


def set_slot_ids(ids: torch.Tensor | None) -> None:
    _SLOT_IDS.set(ids)


def _row_slot_ids(batch_size: int) -> list[int]:
    """Map this step's host routing to one slot per token row (0 = base).

    Padding rows past the routed tokens are base. A routing vector that cannot
    describe this batch (length mismatch, or more routed tokens than rows) must
    fail loudly: silently zeroing the whole batch would run adapter tokens on
    base weights and, for speculative verify, accept tokens on the wrong
    logits.
    """
    ids = _ADAPTER_IDS.get()
    lens = _Q_SEQ_LENS.get()
    if not ids:
        return [0] * batch_size
    if lens is None or len(ids) != len(lens):
        raise RuntimeError(
            "LoRA routing cannot be mapped to token rows: adapter_ids and "
            f"q_seq_lens differ in length ({len(ids)} vs "
            f"{'None' if lens is None else len(lens)})"
        )
    total = sum(lens)
    if total > batch_size:
        raise RuntimeError(
            f"LoRA routing covers {total} tokens but the batch has "
            f"{batch_size} rows; refusing to run this batch on base weights"
        )
    rows: list[int] = []
    for adapter_id, length in zip(ids, lens):
        rows.extend([1 if adapter_id else 0] * length)
    if total < batch_size:
        rows.extend([0] * (batch_size - total))
    return rows


def fill_static_slot_ids(buffer: torch.Tensor, batch_size: int) -> bool:
    """Write this step's per-token slots into the captured device buffer."""
    if not slot_mode():
        return False
    rows = _row_slot_ids(batch_size)
    host = torch.tensor(rows, dtype=buffer.dtype)
    buffer[:batch_size].copy_(host)
    if buffer.numel() > batch_size:
        buffer[batch_size:].zero_()
    return True


def _slot_ids_for(x: torch.Tensor) -> torch.Tensor:
    ids = _SLOT_IDS.get()
    if ids is not None:
        return ids
    key = (x.size(0), str(x.device))
    cached = _SLOT_ZERO_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(
            _row_slot_ids(x.size(0)), dtype=torch.long, device=x.device
        )
        _SLOT_ZERO_CACHE[key] = cached
    return cached


def slot_mask(x: torch.Tensor) -> torch.Tensor:
    """0/1 row mask (dtype of x) telling the caller which rows are adapters."""
    ids = _slot_ids_for(x)
    return (ids != 0).to(x.dtype).unsqueeze(1)


def all_reduce(delta: torch.Tensor) -> torch.Tensor:
    from xllm.python import distributed

    try:
        group = distributed.tp_group(delta.device)
    except Exception:
        # No TP group on this device: TP1, nothing to reduce.
        return delta
    if group.world_size() <= 1:
        return delta
    # Only the "no group" case is skipped above. A real collective failure
    # must surface: swallowing it would leave each rank with a local partial
    # sum and silently wrong deltas.
    distributed.tp_all_reduce(delta)
    return delta
