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

"""CPU tests for the fixed-slot LoRA weight store."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from xllm.python.lora_slots import BASE_SLOT, SlotStore, _parse_key


def _write_adapter(
    root,
    name: str,
    r: int,
    alpha: int,
    layers: tuple[int, ...] = (0, 1),
    projs: tuple[str, ...] = ("q_proj", "o_proj"),
) -> str:
    path = root / name
    path.mkdir()
    tensors = {}
    for layer in layers:
        for proj in projs:
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
            tensors[prefix + ".lora_A.weight"] = torch.randn(r, 8)
            tensors[prefix + ".lora_B.weight"] = torch.randn(6, r)
    save_file(tensors, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha})
    )
    return str(path)


def test_parse_key() -> None:
    key = "base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight"
    assert _parse_key(key) == (7, "q_proj", "a")
    assert _parse_key("lm_head.weight") is None


def test_install_assigns_distinct_slots(tmp_path) -> None:
    store = SlotStore(max_loras=2)
    first = store.install(1, _write_adapter(tmp_path, "a", 4, 8))
    second = store.install(2, _write_adapter(tmp_path, "b", 4, 8))
    assert first != BASE_SLOT and second != BASE_SLOT and first != second
    assert store.slot_of(1) == first
    assert store.loaded_ids() == (1, 2)


def test_delta_matches_reference_per_layer(tmp_path) -> None:
    path = _write_adapter(tmp_path, "a", 4, 8)
    store = SlotStore(max_loras=2)
    store.install(3, path, dtype=torch.float32)
    with torch.no_grad():
        x = torch.randn(5, 8)
        delta = store.delta(3, 1, "q_proj", x, None)
    entry = store._by_id[3]  # noqa: SLF001 - white-box numeric check
    weight_a, weight_b = entry.weights[(1, "q_proj")]
    expected = ((x @ weight_a.t()) @ weight_b.t()) * entry.scaling
    torch.testing.assert_close(delta, expected)


def test_missing_projection_returns_none(tmp_path) -> None:
    path = _write_adapter(tmp_path, "a", 4, 8, layers=(0,), projs=("q_proj",))
    store = SlotStore(max_loras=2)
    store.install(1, path)
    assert store.delta(1, 0, "v_proj", torch.randn(2, 8), None) is None
    assert store.delta(1, 5, "q_proj", torch.randn(2, 8), None) is None


def test_uninstall_reuses_slot(tmp_path) -> None:
    store = SlotStore(max_loras=2)
    slot = store.install(1, _write_adapter(tmp_path, "a", 4, 8))
    assert store.uninstall(1) is True
    assert store.uninstall(1) is False
    again = store.install(2, _write_adapter(tmp_path, "b", 4, 8))
    assert again == slot


def test_pool_full(tmp_path) -> None:
    store = SlotStore(max_loras=1)
    store.install(1, _write_adapter(tmp_path, "a", 4, 8))
    with pytest.raises(RuntimeError, match="slot pool is full"):
        store.install(2, _write_adapter(tmp_path, "b", 4, 8))


def test_signature_tracks_slots(tmp_path) -> None:
    store = SlotStore(max_loras=3)
    slot = store.install(5, _write_adapter(tmp_path, "a", 4, 8))
    assert store.signature() == (1, ((5, slot),))
    store.uninstall(5)
    assert store.signature() == (2, ())


def test_unload_listener_and_generation(tmp_path) -> None:
    store = SlotStore(max_loras=2)
    seen: list[tuple[int, int]] = []
    store.add_listener(lambda adapter_id, slot: seen.append((adapter_id, slot)))
    slot = store.install(7, _write_adapter(tmp_path, "a", 4, 8))
    assert store.signature()[0] == 1
    store.uninstall(7)
    # Install and uninstall both notify: each changes the graph signature.
    assert seen == [(7, slot), (7, slot)]
    # Same id reinstalled must land on a new generation even if the slot
    # index comes back: the old graph variant key can never match again.
    again = store.install(7, _write_adapter(tmp_path, "b", 4, 8))
    assert again == slot
    assert store.signature() == (3, ((7, slot),))


def test_packed_qkv_matches_per_projection(tmp_path) -> None:
    path = _write_adapter(
        tmp_path, "qkv", 4, 8, layers=(0,), projs=("q_proj", "k_proj", "v_proj")
    )
    store = SlotStore(max_loras=2)
    store.install(9, path, dtype=torch.float32)
    x = torch.randn(5, 8)
    delta = store.qkv_delta(9, 0, x, q_local=6, kv_local=6)
    entry = store._by_id[9]  # noqa: SLF001 - white-box numeric check
    parts = []
    for name in ("q_proj", "k_proj", "v_proj"):
        a, b = entry.weights[(0, name)]
        parts.append(((x @ a.t()) @ b.t()) * entry.scaling)
    torch.testing.assert_close(delta, torch.cat(parts, dim=1))


def test_batch_adapter() -> None:
    from xllm.python.lora_slots import batch_adapter

    assert batch_adapter(None) is None
    assert batch_adapter([0, 0]) is None
    assert batch_adapter([0, 3, 0]) == 3
    assert batch_adapter([1, 2]) is None


def test_change_listener_global(tmp_path) -> None:
    from xllm.python import lora_slots

    seen: list[tuple[int, int]] = []
    lora_slots.add_change_listener(lambda adapter_id, slot: seen.append((adapter_id, slot)))
    store = lora_slots.get_store()
    if store is None:
        assert lora_slots.install_adapter(
            41, _write_adapter(tmp_path, "g", 4, 8), "g", dtype=torch.float32
        )
    else:
        store.install(41, _write_adapter(tmp_path, "g", 4, 8))
    lora_slots.uninstall_adapter(name="g")
    # Both install and uninstall change the graph signature and must notify:
    # stale captured variants otherwise keep referencing the old weights.
    assert seen == [(41, 1), (41, 1)]


def test_shard_follows_projection_parallelism() -> None:
    from xllm.python.lora_slots import _shard

    a = torch.randn(4, 8)
    b = torch.randn(6, 4)
    # Column-parallel (q/k/v): A keeps the full input dim, B loses its output
    # rows to the rank.
    a_col, b_col = _shard(a, b, 2, 0, "q_proj")
    assert torch.equal(a_col, a)
    assert b_col.shape == (3, 4)
    # Row-parallel (o/down): A is split on its input dim, B stays full.
    a_row, b_row = _shard(a, b, 2, 1, "o_proj")
    assert a_row.shape == (4, 4)
    assert torch.equal(a_row, a[:, 4:])
    assert torch.equal(b_row, b)


def test_install_failure_returns_slot(tmp_path) -> None:
    from xllm.python.lora_slots import SlotStore

    store = SlotStore(1)
    with pytest.raises(Exception):
        store.install(1, str(tmp_path / "missing"))
    # The failed load must return its slot; otherwise one bad adapter pin
    # exhausts a small max_loras pool permanently.
    slot = store.install(2, _write_adapter(tmp_path, "ok", 4, 8))
    assert slot == 1


def test_install_adapter_requires_model_dtype(tmp_path) -> None:
    from xllm.python import lora_slots

    with pytest.raises(ValueError):
        lora_slots.install_adapter(
            43, _write_adapter(tmp_path, "h", 4, 8), "h"
        )
