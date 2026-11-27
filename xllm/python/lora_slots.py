# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xLLM/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-slot LoRA weight store for graph-compatible Python models.

P1-B keeps adapter weights at stable device addresses so a captured decode
graph can read the current values without re-running Python. A batch is
served by one adapter slot at a time: the delta uses that slot's weights and
zeroes the base rows with a per-token mask. A batch that mixes several
adapters is outside this path (the caller keeps it eager).

Slots are allocated per install and never move while the adapter stays
loaded. Unload frees the slot index; a later install reuses it with new
tensors, which invalidates every graph that captured the old ones. The
caller keys graph variants on :meth:`SlotStore.signature`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from safetensors import safe_open

BASE_SLOT = 0


@dataclass(frozen=True)
class ProjSpec:
    """One projection's local shape for the delta."""

    name: str
    in_local: int
    out_local: int
    row_parallel: bool = False


@dataclass
class AdapterWeights:
    """A loaded adapter's per-layer weights in the local TP shard."""

    adapter_id: int
    slot: int
    r: int
    scaling: float
    weights: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    packed: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )


class SlotStore:
    """Registry of per-adapter weights with stable slot indices."""

    def __init__(self, max_loras: int = 16) -> None:
        if max_loras < 1:
            raise ValueError("max_loras must be positive")
        self._max_loras = max_loras
        self._lock = threading.Lock()
        self._by_id: dict[int, AdapterWeights] = {}
        self._free_slots: list[int] = list(range(max_loras, BASE_SLOT, -1))
        # Bumped on every install/uninstall. Folded into the graph variant
        # key so a reinstall with the same id/slot can never replay a stale
        # graph that still points at the previous tensors.
        self._generation = 0
        self._listeners: list[Callable[[int, int], None]] = []

    # -- lifecycle -------------------------------------------------------

    def install(
        self,
        adapter_id: int,
        path: str,
        *,
        tp_world: int = 1,
        tp_rank: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Load one adapter into a free slot; returns the slot index."""
        if adapter_id <= BASE_SLOT:
            raise ValueError(f"adapter id must be > {BASE_SLOT}, got {adapter_id}")
        with self._lock:
            if adapter_id in self._by_id:
                return self._by_id[adapter_id].slot
            if not self._free_slots:
                raise RuntimeError(
                    f"LoRA slot pool is full (max_loras={self._max_loras})"
                )
            slot = self._free_slots.pop()
            try:
                entry = self._load(
                    adapter_id, slot, path, tp_world, tp_rank, device, dtype
                )
            except Exception:
                # A failed load must return the slot, or repeated failures
                # shrink the pool until every later install reports "full".
                self._free_slots.append(slot)
                raise
            self._by_id[adapter_id] = entry
            self._generation += 1
            listeners = list(self._listeners)
        # Outside the lock: a listener may drop graph variants and recapture.
        # Install changes the graph signature just like uninstall does, so the
        # runner must be told to discard its captured variants/pool.
        for listener in listeners:
            listener(adapter_id, slot)
        return slot

    def uninstall(self, adapter_id: int) -> bool:
        """Drop an adapter and return its slot to the pool."""
        with self._lock:
            entry = self._by_id.pop(adapter_id, None)
            if entry is None:
                return False
            self._free_slots.append(entry.slot)
            self._generation += 1
            listeners = list(self._listeners)
        # Outside the lock: a listener may drop graph variants and recapture.
        for listener in listeners:
            listener(adapter_id, entry.slot)
        return True

    def slot_of(self, adapter_id: int) -> int | None:
        entry = self._by_id.get(adapter_id)
        return None if entry is None else entry.slot

    def signature(self) -> tuple[int, tuple[tuple[int, int], ...]]:
        """``(generation, ((id, slot), ...))`` for graph-variant keying.

        The generation changes on every install/uninstall, so reusing an id
        or a slot always lands on a fresh variant key instead of replaying a
        graph that still references the previous tensors.
        """
        with self._lock:
            pairs = tuple(sorted((i, e.slot) for i, e in self._by_id.items()))
            return self._generation, pairs

    def add_listener(self, listener: Callable[[int, int], None]) -> None:
        """Register ``fn(adapter_id, slot)`` called after an unload."""
        with self._lock:
            self._listeners.append(listener)

    def loaded_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._by_id))

    # -- compute ---------------------------------------------------------

    def packed_qkv(
        self,
        adapter_id: int,
        layer: int,
        q_local: int,
        kv_local: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Packed q/k/v: A [3r, in], block-diagonal B [q + 2*kv, 3r]."""
        entry = self._by_id.get(adapter_id)
        if entry is None:
            return None
        key = (layer, q_local, kv_local)
        cached = entry.packed.get(key)
        if cached is not None:
            return cached
        pairs = [entry.weights.get((layer, name)) for name in ("q_proj", "k_proj", "v_proj")]
        if any(pair is None for pair in pairs):
            return None
        a = torch.cat([pair[0] for pair in pairs], dim=0)
        r = a.size(0) // len(pairs)
        total_out = sum(pair[1].size(0) for pair in pairs)
        b = torch.zeros(
            (total_out, len(pairs) * r), dtype=a.dtype, device=a.device
        )
        offset = 0
        for index, pair in enumerate(pairs):
            out = pair[1]
            b[offset : offset + out.size(0), index * r : (index + 1) * r] = out
            offset += out.size(0)
        entry.packed[key] = (a, b)
        return a, b

    def qkv_delta(
        self,
        adapter_id: int,
        layer: int,
        x: torch.Tensor,
        q_local: int,
        kv_local: int,
    ) -> torch.Tensor | None:
        packed = self.packed_qkv(adapter_id, layer, q_local, kv_local)
        if packed is None:
            return None
        a, b = packed
        entry = self._by_id[adapter_id]
        return ((x @ a.t()) @ b.t()) * entry.scaling

    def delta(
        self,
        adapter_id: int,
        layer: int,
        proj: str,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """``x @ A^T @ B^T * scaling`` for one projection, masked to adapters.

        ``mask`` is ``[T, 1]`` in ``x.dtype``; base rows are zero. The caller
        adds the result with ``addcmul_`` so the whole tail stays graph-safe.
        """
        entry = self._by_id.get(adapter_id)
        if entry is None:
            return None
        pair = entry.weights.get((layer, proj))
        if pair is None:
            return None
        a, b = pair
        if a.device != x.device or a.dtype != x.dtype:
            raise RuntimeError(
                f"adapter {adapter_id} weights are on {a.device}/{a.dtype}, "
                f"activation is on {x.device}/{x.dtype}"
            )
        return ((x @ a.t()) @ b.t()) * entry.scaling

    # -- loading ---------------------------------------------------------

    def _load(
        self,
        adapter_id: int,
        slot: int,
        path: str,
        tp_world: int,
        tp_rank: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> AdapterWeights:
        config_path = os.path.join(path, "adapter_config.json")
        weights_path = os.path.join(path, "adapter_model.safetensors")
        with open(config_path) as handle:
            config = json.load(handle)
        r = int(config["r"])
        scaling = float(config["lora_alpha"]) / r
        entry = AdapterWeights(
            adapter_id=adapter_id, slot=slot, r=r, scaling=scaling
        )
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                parsed = _parse_key(key)
                if parsed is None:
                    continue
                layer, proj, which = parsed
                base = key[: -len(f".lora_{which.upper()}.weight")]
                a_key = base + ".lora_A.weight"
                b_key = base + ".lora_B.weight"
                if a_key not in keys or b_key not in keys:
                    continue
                a = handle.get_tensor(a_key).to(device=device, dtype=dtype)
                b = handle.get_tensor(b_key).to(device=device, dtype=dtype)
                a, b = _shard(a, b, tp_world, tp_rank, proj)
                entry.weights[(layer, proj)] = (a, b)
        if not entry.weights:
            raise ValueError(f"adapter {path} has no supported projection weights")
        return entry


def _parse_key(key: str) -> tuple[int, str, str] | None:
    """Return (layer, proj, "a"|"b") for a PEFT LoRA weight key."""
    if key.endswith(".lora_A.weight"):
        which = "a"
    elif key.endswith(".lora_B.weight"):
        which = "b"
    else:
        return None
    parts = key.split(".")
    layer_index = None
    for index, part in enumerate(parts[:-2]):
        if part == "layers" and index + 1 < len(parts):
            layer_index = int(parts[index + 1])
            break
    if layer_index is None:
        return None
    proj = parts[-3]
    return layer_index, proj, which


# LoRA sharding follows the projection's parallelism, not tensor shape:
# column-parallel outputs are split across ranks (slice B's output rows),
# row-parallel inputs are split (slice A's input columns). Deciding by
# divisibility alone slices the wrong tensor for q/k/v at TP>1.
_ROW_PARALLEL_PROJS = frozenset({"o_proj", "down_proj", "o_b_proj"})
_COLUMN_PARALLEL_PROJS = frozenset(
    {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "q_a_proj", "kv_proj"}
)


def _shard(
    a: torch.Tensor,
    b: torch.Tensor,
    tp_world: int,
    tp_rank: int,
    proj: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice replicated A/B for this rank according to the projection kind."""
    if tp_world <= 1:
        return a, b
    if proj in _ROW_PARALLEL_PROJS:
        if a.size(1) % tp_world != 0:
            raise ValueError(
                f"row-parallel LoRA {proj}: in_dim {a.size(1)} not divisible "
                f"by tp_world {tp_world}"
            )
        width = a.size(1) // tp_world
        a = a[:, tp_rank * width : (tp_rank + 1) * width].contiguous()
        return a, b
    if proj in _COLUMN_PARALLEL_PROJS:
        if b.size(0) % tp_world != 0:
            raise ValueError(
                f"column-parallel LoRA {proj}: out_dim {b.size(0)} not "
                f"divisible by tp_world {tp_world}"
            )
        width = b.size(0) // tp_world
        b = b[tp_rank * width : (tp_rank + 1) * width].contiguous()
        return a, b
    # Unknown projection: no consumer records it, so keep both replicated
    # instead of guessing a direction.
    return a, b


def projections_in(path: str) -> Iterable[str]:
    """Projection names present in an adapter (diagnostic helper)."""
    with safe_open(os.path.join(path, "adapter_model.safetensors"), framework="pt") as handle:
        names = {_parse_key(key)[1] for key in handle.keys() if _parse_key(key)}
    return sorted(names)


# -- process-wide store used by the C++ worker for python-impl models -------

_STORE: SlotStore | None = None
_STORE_LOCK = threading.Lock()
_NAME_TO_ID: dict[str, int] = {}
_CHANGE_LISTENERS: list[Callable[[int, int], None]] = []


def add_change_listener(listener: Callable[[int, int], None]) -> None:
    """Register ``fn(adapter_id, slot)``; forwarded to later stores too."""
    with _STORE_LOCK:
        _CHANGE_LISTENERS.append(listener)
        store = _STORE
    if store is not None:
        store.add_listener(listener)


def batch_adapter(adapter_ids: Iterable[int] | None) -> int | None:
    """The batch's single non-zero adapter id, or None (base/multi-adapter).

    The slot path serves one adapter per batch; a batch that mixes several
    adapters returns None so the caller keeps it on the eager path.
    """
    if not adapter_ids:
        return None
    distinct = {int(adapter_id) for adapter_id in adapter_ids if adapter_id}
    if len(distinct) != 1:
        return None
    return distinct.pop()


def store_mode() -> bool:
    """Productized fixed-slot path (attention first, C++ forwards load)."""
    return os.environ.get("XLLM_LORA_SLOTS_STORE", "") == "1"


def get_store() -> SlotStore | None:
    return _STORE


def install_adapter(
    adapter_id: int,
    path: str,
    name: str,
    max_loras: int = 16,
    device_index: int = 0,
    dtype: torch.dtype | None = None,
) -> bool:
    """Install one adapter on this worker; weights are owned by Python.

    Called from ``WorkerImpl::load_lora_adapter`` for python-impl models, so
    the C++ LoRA runtime never materialises per-projection weights for them.
    ``dtype`` is the running model's dtype: defaulting to bf16 here makes a
    fp16 model fail on its first adapter batch, after the load already
    reported success.
    """
    if dtype is None:
        raise ValueError("install_adapter requires the model dtype")
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = SlotStore(max(1, int(max_loras)))
            for listener in _CHANGE_LISTENERS:
                _STORE.add_listener(listener)
        store = _STORE
        bound = _NAME_TO_ID.get(str(name))
        if bound is not None and bound != int(adapter_id):
            raise ValueError(
                f"adapter name '{name}' is already bound to id {bound}; "
                f"unload it before loading id {adapter_id}"
            )
    device = _resolve_device(device_index)
    world, rank = _tp_info(device)
    store.install(
        int(adapter_id),
        path,
        tp_world=world,
        tp_rank=rank,
        device=device,
        dtype=dtype,
    )
    with _STORE_LOCK:
        _NAME_TO_ID[str(name)] = int(adapter_id)
    return True


def uninstall_adapter(adapter_id: int | None = None, name: str | None = None) -> bool:
    """Unload by id or by the name recorded at install."""
    store = _STORE
    if store is None:
        return False
    if adapter_id is None:
        if name is None:
            return False
        with _STORE_LOCK:
            adapter_id = _NAME_TO_ID.get(str(name))
        if adapter_id is None:
            return False
    with _STORE_LOCK:
        if name is not None:
            _NAME_TO_ID.pop(str(name), None)
        else:
            for key, value in list(_NAME_TO_ID.items()):
                if value == int(adapter_id):
                    _NAME_TO_ID.pop(key, None)
    return store.uninstall(int(adapter_id))


def _resolve_device(device_index: int) -> torch.device:
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device(f"npu:{int(device_index)}")
    except Exception:  # pragma: no cover - CPU-only builds
        pass
    return torch.device("cpu")


def _tp_info(device: torch.device) -> tuple[int, int]:
    try:
        from xllm.python import distributed

        group = distributed.tp_group(device)
        return group.world_size(), group.rank()
    except Exception:  # pragma: no cover - no process group
        return 1, 0
