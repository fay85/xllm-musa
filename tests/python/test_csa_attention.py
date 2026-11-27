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

"""Unit tests for the DeepSeek-V4 CSA/HCA attention backend.

Pure-Python: does not load compiled operators. Covers DSA metadata build,
decode-graph device refresh, and fused inverse RoPE wiring (see the
per-section notes below).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from xllm.python import kernels
from xllm.python.attention import csa_attention as csa_attention_module
from xllm.python.attention import dsa_metadata as dsa_metadata_module
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa_attention import (
    CsaAttentionBackend,
    DsaAttentionBackend,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.dsa_metadata import (
    DsaMetadataBuilder,
    build_cache_specs,
)
from xllm.python.model_executor.v4_cp_context import (
    build_deepseek_v4_cp_context,
)


def _make_backend(device: torch.device | None = None, *, index_topk: int = 512) -> CsaAttentionBackend:
    compress_ratios = [0, 4, 128]
    return CsaAttentionBackend(
        compress_ratios=compress_ratios,
        window_size=128,
        n_layers=3,
        num_heads=8,
        attn_head_dim=512,
        index_topk=index_topk,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=device or torch.device("cpu"),
        dtype=torch.bfloat16,
    )


def _bound_dummy_caches() -> list[LayerCache]:
    swa = lambda: torch.zeros((2, 128, 1, 512), dtype=torch.bfloat16)
    return [
        LayerCache(key=None, value=None, swa=swa()),
        LayerCache(key=torch.zeros((4, 512, 1, 512), dtype=torch.bfloat16), value=None, swa=swa()),
        LayerCache(key=torch.zeros((1, 16, 1, 512), dtype=torch.bfloat16), value=None, swa=swa()),
    ]


@pytest.mark.parametrize(
    ("layer_id", "ratio", "expected"),
    [(1, 4, (0, 1, 7, 2, 3, 4, 5, 6)), (2, 128, (0, -1, -1, 1, 2, 3, -1, -1)), (0, 1, (-1, -1, -1, 0, -1, -1, -1, -1))],
)
def test_cache_mapping_respects_layer_slot_abi(layer_id: int, ratio: int, expected: tuple[int, ...]) -> None:
    mapping = _make_backend()._resolve_cache_mapping(layer_id, ratio)
    assert (
        mapping.cmp_cache_idx,
        mapping.index_cache_idx,
        mapping.indexer_scale_cache_idx,
        mapping.ori_cache_idx,
        mapping.kv_state_cache_idx,
        mapping.score_state_cache_idx,
        mapping.index_kv_state_cache_idx,
        mapping.index_score_state_cache_idx,
    ) == expected


def test_get_layer_cache_tensor_bounds() -> None:
    tensors = [[torch.empty(0)], [torch.zeros(2), torch.zeros(3)]]
    assert _get_layer_cache_tensor(tensors, 0, 0) is tensors[0][0]
    assert _get_layer_cache_tensor(tensors, 1, 1) is tensors[1][1]
    assert _get_layer_cache_tensor(tensors, -1, 0) is None
    assert _get_layer_cache_tensor(tensors, 1, -1) is None
    assert _get_layer_cache_tensor(tensors, 5, 0) is None  # bad layer
    assert _get_layer_cache_tensor(tensors, 1, 9) is None  # bad cache idx


def test_scatter_by_slot_writes_rows() -> None:
    cache = torch.zeros(4, 3, dtype=torch.float32)
    # slot 0 -> row 0, slot 2 -> row 2, slot -1 skipped.
    slots = torch.tensor([0, -1, 2], dtype=torch.int32)
    value = torch.tensor([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [2.0, 2.0, 2.0]])
    _scatter_by_slot(cache, slots, value)
    torch.testing.assert_close(
        cache,
        torch.tensor([[1, 1, 1], [0, 0, 0], [2, 2, 2], [0, 0, 0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_scatter_by_slot_ignores_all_padded_rows() -> None:
    cache = torch.arange(12, dtype=torch.float32).view(4, 3)
    original = cache.clone()
    slots = torch.full((2,), -1, dtype=torch.int32)
    values = torch.full((2, 3), 99.0)

    _scatter_by_slot(cache, slots, values)

    assert torch.equal(cache, original)


def test_prepare_binds_compressed_metadata_to_current_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(
        backend,
        "_build_precomputed_metadata",
        lambda compressed_metadata, metadata: None,
    )

    def make_metadata(kv_len: int, is_prefill: bool) -> SimpleNamespace:
        q_len = kv_len if is_prefill else 1
        return SimpleNamespace(
            multi_block_tables=[],
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            is_prefill=is_prefill,
            is_chunked_prefill=False,
            dsa_metadata=None,
            dsa_positions=torch.arange(kv_len - q_len, kv_len, dtype=torch.int64),
            dsa_cos_sin=None,
            dsa_c4_cos_sin=None,
            dsa_c128_cos_sin=None,
            dsa_graph_block_table_cols=0,
            dsa_graph_mode=False,
        )

    prefill = make_metadata(84, True)
    backend.prepare(prefill)
    backend.prepare_dsa_metadata_for_forward()
    prefill_compressed_metadata = prefill.dsa_metadata
    assert prefill_compressed_metadata.max_query_len == 84

    decode = make_metadata(85, False)
    backend.prepare(decode)
    backend.prepare_dsa_metadata_for_forward()
    assert decode.dsa_metadata is not prefill_compressed_metadata
    assert decode.dsa_metadata.max_query_len == 1
    assert prefill.dsa_metadata is prefill_compressed_metadata


def test_prepare_clamps_target_block_tables_to_draft_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MTP draft backend consumes only its registered SWA manager table."""
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[
            torch.tensor([[10]], dtype=torch.int32),
            torch.tensor([[20]], dtype=torch.int32),
            torch.tensor([[30]], dtype=torch.int32),
        ],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        new_cache_slots_host_values=[1280],
        is_prefill=True,
        is_chunked_prefill=False,
        is_dummy=False,
        dsa_positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_graph_block_table_cols=0,
        dsa_graph_mode=False,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.block_tables[0][0].tolist() == [[10]]
    assert dsa.slot_mappings[0][0].tolist() == [1280]


@pytest.mark.parametrize(("index_topk", "c4_slot", "c128_slot"), [(512, 127, 3), (1024, 255, 7), (2048, 511, 15)])
def test_empty_dp_metadata_uses_safe_lengths_and_cache_tables(
    monkeypatch: pytest.MonkeyPatch, index_topk: int, c4_slot: int, c128_slot: int
) -> None:
    backend = _make_backend(index_topk=index_topk)
    backend.bind_kv_caches(_bound_dummy_caches())
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata(
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=True,
        is_dummy=True,
    )
    dsa = backend._build_dsa_metadata_for_forward(metadata)
    assert dsa.seq_lens.tolist() == [index_topk]
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.max_seq_len == index_topk and dsa.max_query_len == 1
    assert dsa.block_tables[1][0].tolist() == [[0]]
    assert dsa.slot_mappings[0][0].tolist() == [127]
    assert dsa.slot_mappings[1][0].tolist() == [c4_slot]
    assert dsa.slot_mappings[2][0].tolist() == [c128_slot]


@pytest.mark.parametrize(
    ("index_topk", "graph_cols", "swa_column", "c4_slot", "c128_slot"),
    [(512, 8, 3, 127, 3), (1024, 8, 7, 255, 7), (2048, 16, 15, 511, 15)],
)
def test_empty_dp_graph_metadata_preserves_bucket_rows(
    monkeypatch: pytest.MonkeyPatch, index_topk: int, graph_cols: int, swa_column: int, c4_slot: int, c128_slot: int
) -> None:
    backend = _make_backend(index_topk=index_topk)
    backend.bind_kv_caches(_bound_dummy_caches())
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata(
        multi_block_tables=[torch.full((4, graph_cols), -1, dtype=torch.int32) for _ in range(3)],
        kv_seq_lens_host_values=[1, 0, 0, 0],
        q_seq_lens=torch.ones(4, dtype=torch.int32),
        is_dummy=True,
        dsa_positions=torch.zeros(4, dtype=torch.int64),
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=graph_cols,
    )
    dsa = backend._build_dsa_metadata_for_forward(metadata)
    assert dsa.seq_lens.tolist() == [index_topk] * 4
    assert dsa.seq_lens_q.tolist() == [1] * 4
    expected_swa = torch.full((4, graph_cols), -1, dtype=torch.int32)
    expected_swa[:, swa_column] = 0
    torch.testing.assert_close(dsa.block_tables[0][0], expected_swa, rtol=0, atol=0)
    torch.testing.assert_close(dsa.block_tables[1][0], torch.zeros((4, graph_cols), dtype=torch.int32), rtol=0, atol=0)
    assert dsa.slot_mappings[0][0].tolist() == [127] * 4
    assert dsa.slot_mappings[1][0].tolist() == [c4_slot] * 4
    assert dsa.slot_mappings[2][0].tolist() == [c128_slot] * 4


def test_empty_dp_graph_rejects_insufficient_swa_capacity() -> None:
    backend = _make_backend()
    block_tables = [torch.full((4, 8), -1, dtype=torch.int32) for _ in range(3)]

    with pytest.raises(ValueError, match="SWA block table capacity is too small"):
        backend._build_empty_dp_block_tables(
            block_tables,
            4,
            2048,
            torch.device("cpu"),
            graph_mode=True,
            graph_block_table_capacity_cols=8,
        )


def test_prepare_clears_previous_forward_state() -> None:
    backend = _make_backend()
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_metadata=backend._builder.build([], [1], [1], torch.tensor([0], dtype=torch.int64), None, False, False),
        dsa_positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=torch.ones(1, 128),
        dsa_c4_cos_sin=torch.ones(1, 128),
        dsa_c128_cos_sin=torch.ones(1, 128),
        dsa_graph_mode=True,
    )
    backend.attach_compressor(lambda *_args: None)
    backend.attach_indexer(lambda *_args: None)
    backend.reset_forward(metadata)
    assert metadata.dsa_metadata is None
    assert metadata.dsa_positions is None
    assert metadata.dsa_cos_sin is None
    assert metadata.dsa_c4_cos_sin is None
    assert metadata.dsa_c128_cos_sin is None
    assert metadata.dsa_graph_mode is False
    assert not hasattr(backend, "_compressor_fn")
    assert not hasattr(backend, "_indexer_fn")


# -- sequence-length supply: expanded spec-verify + idle-shard dummy ---------


def _sequence_metadata(**overrides: Any) -> SimpleNamespace:
    """The minimal metadata surface the DSA builder reads.

    The three DSA compression levels are named c1/c4/c128 by their ratio
    (1/4/128): ``dsa_cos_sin`` is the full-resolution (c1) table and
    ``dsa_c4_cos_sin``/``dsa_c128_cos_sin`` the compressed ones. Upstream
    numbered naming (c0/c1/c2) maps to c1/c4/c128 here. The AICPU tiling
    fields (``c1_metadata``/``c4_metadata``/``c128_metadata``) are filled by
    the backend on the DSAMetadata object, not read by this builder surface.
    """
    values = {
        "multi_block_tables": [],
        "kv_seq_lens_host": None,
        "q_seq_lens_host": None,
        "q_seq_lens": None,
        "is_prefill": False,
        "is_chunked_prefill": False,
        "dsa_metadata": None,
        "dsa_positions": torch.tensor([0]),
        "dsa_cos_sin": None,
        "dsa_c4_cos_sin": None,
        "dsa_c128_cos_sin": None,
        "dsa_graph_block_table_cols": 0,
        "dsa_graph_mode": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_idle_shard_without_host_lengths_keeps_one_dummy_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy missing host lengths with physicalrow1 yields nonempty metadata."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata()  # one fake query row, no sequence lengths

    backend.prepare_dsa_metadata_for_forward(metadata)

    dsa = metadata.dsa_metadata
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.seq_lens.tolist() == [1]
    assert dsa.block_tables == []
    # The literal B4 raise point: csa2_attention maxes the per-sequence q lens.


def test_idle_shard_dummy_rows_follow_the_query_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dummy sequence count stays aligned with the padded query rows."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata(is_dummy=True, dsa_positions=torch.zeros(3, dtype=torch.int64))

    backend.prepare_dsa_metadata_for_forward(metadata)

    assert metadata.dsa_metadata.seq_lens_q.tolist() == [1, 1, 1]
    assert metadata.dsa_metadata.max_query_len == 1
    assert metadata.dsa_metadata.seq_lens.tolist() == [512, 512, 512]


def test_expanded_spec_verify_lengths_supply_the_missing_host_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packed verify batch keeps its lengths in the expanded view."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    expanded = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.tensor([7, 9], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([7, 9], dtype=torch.int32),
        kv_seq_lens_host_values=[7, 9],
        q_seq_lens_host=None,
        q_seq_lens_host_values=None,
    )
    metadata = _sequence_metadata(
        dsa_positions=torch.tensor([6, 8], dtype=torch.int64), expanded_decode_metadata=expanded
    )

    backend.prepare_dsa_metadata_for_forward(metadata)

    dsa = metadata.dsa_metadata
    assert dsa.seq_lens.tolist() == [7, 9]
    assert dsa.seq_lens_q.tolist() == [1, 1]


def test_real_host_lengths_are_never_replaced_by_the_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal batch keeps the C++ host view verbatim (no dummy, no expanded)."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    expanded = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.tensor([99], dtype=torch.int32),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[99],
        q_seq_lens_host=None,
        q_seq_lens_host_values=None,
    )
    metadata = _sequence_metadata(
        kv_seq_lens_host=torch.tensor([5, 6], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1, 1], dtype=torch.int32),
        expanded_decode_metadata=expanded,
        dsa_positions=torch.tensor([4, 5], dtype=torch.int64),
    )

    backend.prepare_dsa_metadata_for_forward(metadata)

    assert metadata.dsa_metadata.seq_lens.tolist() == [5, 6]
    assert metadata.dsa_metadata.seq_lens_q.tolist() == [1, 1]


def test_empty_forward_without_positions_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No query rows at all -> no synthetic sequence (nothing to align to)."""
    backend = _make_backend()
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = _sequence_metadata(dsa_positions=torch.empty(0, dtype=torch.int64))

    backend.prepare_dsa_metadata_for_forward(metadata)

    assert metadata.dsa_metadata.seq_lens_q.tolist() == []
    assert metadata.dsa_metadata.seq_lens.tolist() == []
    assert metadata.dsa_metadata.actual_seq_lengths_query.tolist() == [0]


def test_decode_precomputed_metadata_matches_cpp_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend()
    sparse_calls: list[dict] = []
    qli_calls: list[dict] = []

    def fake_sparse_metadata(**kwargs: Any) -> torch.Tensor:
        sparse_calls.append(kwargs)
        return torch.tensor([kwargs["cmp_ratio"]], dtype=torch.int32)

    def fake_qli_metadata(**kwargs: Any) -> torch.Tensor:
        qli_calls.append(kwargs)
        return torch.tensor([4], dtype=torch.int32)

    monkeypatch.setattr(
        kernels,
        "sparse_attn_sharedkv_metadata",
        fake_sparse_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "quant_lightning_indexer_metadata",
        fake_qli_metadata,
        raising=False,
    )
    compressed_metadata = backend._builder.build(
        [], [85], [1], torch.tensor([84], dtype=torch.int64), None, False, False
    )
    metadata = SimpleNamespace(
        max_query_len=1,
        max_seq_len=85,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([85], dtype=torch.int32),
    )

    backend._build_precomputed_metadata(compressed_metadata, metadata)

    assert [call["cmp_ratio"] for call in sparse_calls] == [1, 4, 128]
    assert all(call["head_dim"] == 512 for call in sparse_calls)
    assert all(call["cu_seqlens_q"].tolist() == [0, 1] for call in sparse_calls)
    assert all(call["cu_seqlens_ori_kv"].numel() == 0 for call in sparse_calls)
    assert sparse_calls[1]["cmp_topk"] == 512
    assert qli_calls[0]["actual_seq_lengths_query"].tolist() == [1]
    assert qli_calls[0]["actual_seq_lengths_key"].tolist() == [85]
    assert qli_calls[0]["head_dim"] == 128
    assert compressed_metadata.precomputed_metadata_inputs[0] is compressed_metadata.actual_seq_lengths_query
    owned_storage = {tensor.untyped_storage().data_ptr() for tensor in compressed_metadata.precomputed_metadata_inputs}
    for call in sparse_calls + qli_calls:
        for value in call.values():
            if isinstance(value, torch.Tensor) and value.numel():
                assert value.untyped_storage().data_ptr() in owned_storage


def test_dspark_routes_to_supported_native_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the sparse kernel and consume the matching SWA metadata."""
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_speculative_tokens=5,
    )
    swa = torch.zeros(2, 128, 1, 512)
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=swa)])
    dsa = backend._builder.build(
        multi_block_tables=[torch.tensor([[1]], dtype=torch.int32)],
        kv_seq_lens=[2],
        q_seq_lens=[1],
        positions=torch.tensor([1]),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
    lengths = torch.tensor([[2]], dtype=torch.int32)
    tiling = torch.zeros(1024, dtype=torch.int32)

    def reject_sharedkv(**_kwargs: object) -> None:
        pytest.fail("DSpark selected SharedKV with native sparse attention available")

    def run_native(
        q: torch.Tensor,
        _kv: torch.Tensor,
        _table: torch.Tensor,
        _cu: torch.Tensor,
        _lens: torch.Tensor,
        actual_indices: torch.Tensor,
        actual_lengths: torch.Tensor,
        _sinks: torch.Tensor | None,
        actual_tiling: torch.Tensor,
    ) -> torch.Tensor:
        assert actual_indices is indices
        assert actual_lengths is lengths
        assert actual_tiling is tiling
        return q + 1

    monkeypatch.setattr(kernels, "supports_sparse_flash_mla_swa", lambda: True, raising=False)
    monkeypatch.setattr(
        kernels, "sparse_flash_mla_swa_metadata", lambda *_args, **_kwargs: (indices, lengths, tiling), raising=False
    )
    monkeypatch.setattr(kernels, "sparse_flash_mla_swa", run_native, raising=False)
    monkeypatch.setattr(kernels, "sparse_attn_sharedkv_metadata", reject_sharedkv, raising=False)
    monkeypatch.setattr(kernels, "sparse_attn_sharedkv", reject_sharedkv, raising=False)
    metadata = SimpleNamespace(dsa_metadata=dsa, is_prefill=False, is_chunked_prefill=False)
    with monkeypatch.context() as build_context:
        build_context.setattr(backend, "device", SimpleNamespace(type="npu"))
        backend._build_precomputed_metadata(dsa, metadata)

    dsa.cos_table = dsa.sin_table = dsa.c4_cos = dsa.c128_cos = torch.empty(0)
    backend._metadata = metadata
    q, kv = torch.zeros(1, 8, 512), torch.ones(1, 1, 512)
    layer: Any = SimpleNamespace(layer_id=0, attn_sink_loaded=False)
    torch.testing.assert_close(backend.execute(q, kv, kv, layer), torch.ones_like(q))


def test_csa_execute_requires_model_compressor(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend()
    empty_cache = LayerCache(key=None, value=None)
    cmp_cache = torch.zeros(2, 512, 1, 512, dtype=torch.bfloat16)
    swa_cache = torch.zeros(2, 128, 1, 512, dtype=torch.bfloat16)
    backend.bind_kv_caches(
        [
            empty_cache,
            LayerCache(key=cmp_cache, value=None, swa=swa_cache),
            empty_cache,
        ]
    )
    # compress_ratios [0, 4, 128] bind two logical tables: SWA and TOKEN.
    compressed_metadata = backend._builder.build(
        [torch.tensor([[1]], dtype=torch.int32) for _ in range(2)],
        [1],
        [1],
        torch.tensor([0], dtype=torch.int64),
        None,
        False,
        False,
    )
    backend._metadata = SimpleNamespace(
        dsa_metadata=compressed_metadata,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with pytest.raises(RuntimeError, match="compressor is required"):
        backend.execute(
            torch.zeros(1, 8, 512, dtype=torch.bfloat16),
            torch.zeros(1, 1, 512, dtype=torch.bfloat16),
            torch.zeros(1, 1, 512, dtype=torch.bfloat16),
            SimpleNamespace(layer_id=1, attn_sink=None),
        )


def test_forward_rope_state_is_owned_by_each_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(
        backend,
        "_build_precomputed_metadata",
        lambda compressed_metadata, metadata: None,
    )

    def make_metadata(kv_len: int, q_len: int) -> SimpleNamespace:
        return SimpleNamespace(
            multi_block_tables=[],
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            is_prefill=q_len > 1,
            is_chunked_prefill=False,
            dsa_metadata=None,
            dsa_positions=None,
            dsa_cos_sin=None,
            dsa_c4_cos_sin=None,
            dsa_c128_cos_sin=None,
            dsa_graph_block_table_cols=0,
            dsa_graph_mode=False,
        )

    rope_cache = torch.arange(256 * 128, dtype=torch.float32).view(256, 128)
    prefill = make_metadata(84, 84)
    backend.prepare(prefill)
    backend.attach_rope_tables(
        torch.arange(84),
        rope_cache,
        csa_cos_sin=rope_cache,
        hca_cos_sin=rope_cache,
        metadata=prefill,
    )

    decode = make_metadata(85, 1)
    backend.prepare(decode)
    backend.attach_rope_tables(
        torch.tensor([84]),
        rope_cache,
        csa_cos_sin=rope_cache,
        hca_cos_sin=rope_cache,
        metadata=decode,
    )

    backend.prepare_dsa_metadata_for_forward(prefill)

    assert prefill.dsa_metadata.input_positions.numel() == 84
    assert decode.dsa_positions.numel() == 1
    assert prefill.dsa_positions.data_ptr() != decode.dsa_positions.data_ptr()
    torch.testing.assert_close(prefill.dsa_metadata.input_rope_by_ratio[1][0], rope_cache[:84, :64], rtol=0, atol=0)


def test_cp_localization_keeps_runtime_metadata_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend()
    dsa = backend._builder.build(
        multi_block_tables=[],
        kv_seq_lens=[4],
        q_seq_lens=[4],
        positions=torch.arange(4, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    class _ReadOnlyMetadata:
        def __init__(self) -> None:
            self.dsa_metadata = dsa
            self.dp_execution_token_counts = (4,)
            self.dsa_cos_sin = torch.arange(4 * 128, dtype=torch.float32).view(4, 128)
            self.dsa_c4_cos_sin = self.dsa_cos_sin
            self.dsa_c128_cos_sin = self.dsa_cos_sin

        @property
        def q_seq_lens_host(self) -> torch.Tensor:
            return torch.tensor([4], dtype=torch.int32)

        @property
        def kv_seq_lens_host(self) -> torch.Tensor:
            return torch.tensor([4], dtype=torch.int32)

        @property
        def max_query_len(self) -> int:
            return 4

        @property
        def max_seq_len(self) -> int:
            return 4

    captured: dict[str, int] = {}

    def capture_precomputed(
        _dsa,
        _metadata,
        *,
        cu_seqlens_ori_kv_override=None,
        max_query_len_override=None,
        max_seq_len_override=None,
    ) -> None:
        assert cu_seqlens_ori_kv_override.tolist() == [0, 2]
        captured["max_query_len"] = max_query_len_override
        captured["max_seq_len"] = max_seq_len_override

    monkeypatch.setattr(backend, "_build_precomputed_metadata", capture_precomputed)
    metadata = _ReadOnlyMetadata()
    cp_context = build_deepseek_v4_cp_context(
        2,
        0,
        [4],
        [4],
        torch.arange(4, dtype=torch.int64),
    )

    backend.localize_dsa_metadata_for_cp(cp_context, metadata)

    assert dsa.seq_lens_q.tolist() == [2]
    assert dsa.seq_lens.tolist() == [2]
    assert captured == {"max_query_len": 2, "max_seq_len": 2}
    assert metadata.q_seq_lens_host.tolist() == [4]
    assert metadata.kv_seq_lens_host.tolist() == [4]
    torch.testing.assert_close(dsa.input_rope_by_ratio[1][0], metadata.dsa_cos_sin[:2, :64], rtol=0, atol=0)


def test_graph_dsa_refresh_preserves_tensor_addresses() -> None:
    def make_metadata(value: int, seq_rows: int) -> SimpleNamespace:
        return SimpleNamespace(
            seq_lens=torch.full((seq_rows,), value, dtype=torch.int32),
            block_tables=[[torch.full((seq_rows, 2), value, dtype=torch.int32)]],
            slot_mappings=[[torch.full((seq_rows,), value, dtype=torch.int32)]],
            input_rope_by_ratio={
                1: (
                    torch.full((2, 2), value, dtype=torch.float32),
                    torch.full((2, 2), value, dtype=torch.float32),
                )
            },
            ori_sparse_indices=torch.full((seq_rows, 1, 128), value, dtype=torch.int32),
            sparse_metadata_ori_win_left=value,
            max_query_len=value,
            max_seq_len=value,
            is_acl_graph=True,
            precomputed_metadata_inputs=(),
        )

    persistent = make_metadata(1, 4)
    refreshed = make_metadata(7, 2)
    seq_lens_ptr = persistent.seq_lens.data_ptr()
    block_table_ptr = persistent.block_tables[0][0].data_ptr()
    slot_ptr = persistent.slot_mappings[0][0].data_ptr()
    swa_indices_ptr = persistent.ori_sparse_indices.data_ptr()
    rope_ptr = persistent.input_rope_by_ratio[1][0].data_ptr()

    CsaAttentionBackend._copy_graph_dsa_metadata(persistent, refreshed)

    assert persistent.seq_lens.data_ptr() == seq_lens_ptr
    assert persistent.seq_lens.tolist() == [7, 7, 0, 0]
    assert persistent.block_tables[0][0].data_ptr() == block_table_ptr
    assert persistent.block_tables[0][0].tolist() == [[7, 7], [7, 7], [-1, -1], [-1, -1]]
    assert persistent.slot_mappings[0][0].tolist() == [7, 7, -1, -1]
    assert persistent.slot_mappings[0][0].data_ptr() == slot_ptr
    assert persistent.ori_sparse_indices.data_ptr() == swa_indices_ptr
    expected_swa = torch.full((4, 1, 128), -1, dtype=torch.int32)
    expected_swa[:2] = 7
    torch.testing.assert_close(persistent.ori_sparse_indices, expected_swa, rtol=0, atol=0)
    assert persistent.input_rope_by_ratio[1][0].data_ptr() == rope_ptr
    assert persistent.input_rope_by_ratio[1][0].tolist() == [[7.0, 7.0], [7.0, 7.0]]
    assert persistent.max_query_len == 7
    assert persistent.max_seq_len == 7


def test_graph_dsa_build_uses_stable_host_length_values(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[9, 1],
        q_seq_lens_host=None,
        q_seq_lens=None,
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_positions=torch.tensor([8, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=4,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.seq_lens.tolist() == [9, 1]
    assert dsa.seq_lens_q.tolist() == [1, 1]
    assert dsa.start_pos.tolist() == [8, 0]


def test_prefill_and_dummy_decode_metadata_stay_in_sync_with_execute(
    monkeypatch,
) -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        num_speculative_tokens=5,
    )
    swa = torch.zeros(2, 128, 1, 512, dtype=torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=swa)])
    block_table = torch.tensor([[1]], dtype=torch.int32)
    layer = SimpleNamespace(
        layer_id=0,
        attn_sink=torch.tensor([1.0], dtype=torch.float32),
        attn_sink_loaded=False,
    )
    calls: list[dict] = []

    def fake_sparse_attn(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        return kwargs["q"].clone(), torch.empty(0)

    monkeypatch.setattr(csa_attention_module, "_sparse_attn_sharedkv", fake_sparse_attn)
    monkeypatch.setattr(
        kernels,
        "sparse_attn_sharedkv_metadata",
        lambda **kwargs: torch.tensor([kwargs["cmp_ratio"]]),
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "quant_lightning_indexer_metadata",
        lambda **_kwargs: torch.tensor([4]),
        raising=False,
    )

    def prepare_step(
        kv_len: int, q_len: int, is_prefill: bool, *, is_chunked_prefill: bool = False, is_dummy: bool = False
    ) -> Any:
        metadata = _sequence_metadata(
            multi_block_tables=[block_table],
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            is_prefill=is_prefill,
            is_chunked_prefill=is_chunked_prefill,
            is_dummy=is_dummy,
            dsa_positions=torch.arange(kv_len - q_len, kv_len, dtype=torch.int64),
        )
        backend.prepare(metadata)
        backend.prepare_dsa_metadata_for_forward(metadata)
        return metadata.dsa_metadata

    window_left = 127
    prefill_metadata = prepare_step(kv_len=2, q_len=2, is_prefill=True)
    assert prefill_metadata.ori_sparse_indices is None
    assert prefill_metadata.sparse_metadata_ori_win_left == window_left
    prefill_kv = torch.arange(2 * 512, dtype=torch.bfloat16).view(2, 1, 512)
    backend.execute(torch.zeros(2, 8, 512, dtype=torch.bfloat16), prefill_kv, prefill_kv, layer)
    assert torch.equal(swa[1, :2], prefill_kv)
    assert calls[-1]["ori_sparse_indices"] is None
    assert calls[-1]["ori_win_left"] == window_left

    decode_metadata = prepare_step(kv_len=3, q_len=1, is_prefill=False)
    assert decode_metadata.ori_sparse_indices is None
    assert decode_metadata.sparse_metadata_ori_win_left == window_left
    decode_kv = torch.full((1, 1, 512), 7.0, dtype=torch.bfloat16)
    backend.execute(torch.zeros(1, 8, 512, dtype=torch.bfloat16), decode_kv, decode_kv, layer)
    assert torch.equal(swa[1, 2], decode_kv[0])
    assert calls[-1]["cu_seqlens_ori_kv"] is None
    assert calls[-1]["ori_sparse_indices"] is None
    assert calls[-1]["ori_win_left"] == window_left
    assert calls[-1]["sinks"] is None

    dummy_metadata = prepare_step(kv_len=3, q_len=1, is_prefill=True, is_dummy=True)
    assert dummy_metadata.ori_sparse_indices is None
    assert dummy_metadata.sparse_metadata_ori_win_left == window_left
    backend.execute(torch.zeros(1, 8, 512, dtype=torch.bfloat16), decode_kv, decode_kv, layer)
    assert calls[-1]["cu_seqlens_ori_kv"] is None
    assert calls[-1]["ori_sparse_indices"] is None
    assert calls[-1]["ori_win_left"] == window_left

    chunked_dummy_metadata = prepare_step(
        kv_len=3,
        q_len=1,
        is_prefill=False,
        is_chunked_prefill=True,
        is_dummy=True,
    )
    assert chunked_dummy_metadata.ori_sparse_indices is None
    assert chunked_dummy_metadata.sparse_metadata_ori_win_left == window_left
    backend.execute(torch.zeros(1, 8, 512, dtype=torch.bfloat16), decode_kv, decode_kv, layer)
    assert calls[-1]["cu_seqlens_ori_kv"] is not None
    assert calls[-1]["ori_sparse_indices"] is None
    assert calls[-1]["ori_win_left"] == window_left

    layer.attn_sink_loaded = True
    prepare_step(kv_len=3, q_len=1, is_prefill=False)
    backend.execute(torch.zeros(1, 8, 512, dtype=torch.bfloat16), decode_kv, decode_kv, layer)
    assert torch.equal(calls[-1]["sinks"], layer.attn_sink)


# ---- graph device refresh ----
# Decode-graph DSA refresh chain (pure Python, CPU): replay alias dedup and
# the in-place refresh mechanism (driven by the builder's ``inplace_target`` /
# ``dst`` parameters directly).

# --- [S4a] Alias dedup ------------------------------------------------------
# Mirrors ``dsa_metadata.py`` block/slot assembly: one manager tensor per group
# is appended into every layer's list, so layers alias the same object.
_N_LAYERS = 4
_GROUPS = 3
_PERSISTENT_ROWS = 4
_REFRESHED_ROWS = 2
_COLS = 3

_TENSOR_FIELDS = (
    "seq_lens",
    "seq_lens_q",
    "actual_seq_lengths_kv",
    "actual_seq_lengths_query",
    "kv_cu_seq_lens",
    "max_seqlen_kv",
    "max_seqlen_q",
    "input_positions",
    "c4_pad_positions",
    "c128_pad_positions",
    "start_pos",
    "c4_cos",
    "c4_sin",
    "c128_cos",
    "c128_sin",
    "c4_input_cos",
    "c4_input_sin",
    "c128_input_cos",
    "c128_input_sin",
)

# --- [Candidate A] In-place refresh ----------------------------------------
_GRAPH_COLS = 8

# Value-derived fields the candidate writes in place on this graph path.
_INPLACE_FIELDS = (
    "seq_lens",
    "seq_lens_q",
    "actual_seq_lengths_kv",
    "actual_seq_lengths_query",
    "kv_cu_seq_lens",
    "max_seqlen_kv",
    "max_seqlen_q",
    "input_positions",
    "c4_pad_positions",
    "c128_pad_positions",
    "start_pos",
)

# --- [SWA device slot-mapping cases] ---------------------------------------
# (raw_bt, block_size, ctx_lens, q_lens, new_cache_slots)
_SWA_CASES = [
    # Plain two-row table; seq1's q_start crosses a block boundary.
    (torch.tensor([[0, 1], [2, 3]], dtype=torch.int32), 4, [5, 9], [1, 1], []),
    # Negative (placeholder) ring blocks must map to -1.
    (torch.tensor([[-1, 1], [2, -1]], dtype=torch.int32), 4, [3, 8], [1, 1], []),
    # q_start at the last position of a block and at position 0.
    (torch.tensor([[0, 1], [2, 3]], dtype=torch.int32), 4, [8, 1], [1, 1], []),
    # Production shape: one ring block per sequence (semantic_cols == 1).
    (torch.tensor([[5], [6]], dtype=torch.int32), 128, [130, 7], [1, 1], []),
    # Scheduler-provided slots short-circuit the ring formula (H2D, no D2H).
    (torch.tensor([[0, 1]], dtype=torch.int32), 4, [5], [1], [42]),
]

# --- [S4a] Alias dedup helpers ---------------------------------------------


def _make_metadata(persistent_seed: int, refreshed_seed: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Persistent/refreshed DSA metadata with group-level per-layer aliasing."""

    def persistent_tensor(value: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        return torch.full((_PERSISTENT_ROWS,), value, dtype=dtype)

    def refreshed_tensor(value: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        return torch.full((_REFRESHED_ROWS,), value, dtype=dtype)

    p_block = [
        torch.full((_PERSISTENT_ROWS, _COLS), persistent_seed * 100 + group, dtype=torch.int32)
        for group in range(_GROUPS)
    ]
    p_slots = [
        torch.full((_PERSISTENT_ROWS,), persistent_seed * 100 + group, dtype=torch.int32) for group in range(_GROUPS)
    ]
    r_block = [
        torch.full((_REFRESHED_ROWS, _COLS), refreshed_seed * 10 + group, dtype=torch.int32) for group in range(_GROUPS)
    ]
    r_slots = [
        torch.full((_REFRESHED_ROWS,), refreshed_seed * 10 + group, dtype=torch.int32) for group in range(_GROUPS)
    ]

    meta = SimpleNamespace(
        ori_sparse_indices=torch.full((_PERSISTENT_ROWS, 1, 128), persistent_seed, dtype=torch.int32),
        # New list objects per layer, but the *same* tensor object per group --
        # exactly the alias relation produced by ``dsa_metadata.py``:553-564.
        block_tables=[[p_block[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        slot_mappings=[[p_slots[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        input_rope_by_ratio={
            1: (persistent_tensor(persistent_seed, torch.float32), persistent_tensor(persistent_seed, torch.float32))
        },
        sparse_metadata_ori_win_left=persistent_seed,
        max_query_len=persistent_seed,
        max_seq_len=persistent_seed,
        is_acl_graph=True,
        graph_token_capacity=persistent_seed,
        precomputed_metadata_inputs=(),
    )
    for name in _TENSOR_FIELDS:
        dtype = torch.float32 if "cos" in name or "sin" in name else torch.int32
        setattr(meta, name, persistent_tensor(persistent_seed, dtype))
    # Attach the refreshed side under distinct names for readability; the copy
    # helpers only read the fields listed above, so we stage a second namespace.
    refreshed = SimpleNamespace(
        ori_sparse_indices=torch.full((_REFRESHED_ROWS, 1, 128), refreshed_seed, dtype=torch.int32),
        block_tables=[[r_block[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        slot_mappings=[[r_slots[group] for group in range(_GROUPS)] for _ in range(_N_LAYERS)],
        input_rope_by_ratio={
            1: (refreshed_tensor(refreshed_seed, torch.float32), refreshed_tensor(refreshed_seed, torch.float32))
        },
        sparse_metadata_ori_win_left=refreshed_seed,
        max_query_len=refreshed_seed,
        max_seq_len=refreshed_seed,
        is_acl_graph=True,
        graph_token_capacity=refreshed_seed,
        precomputed_metadata_inputs=(),
    )
    for name in _TENSOR_FIELDS:
        dtype = torch.float32 if "cos" in name or "sin" in name else torch.int32
        setattr(refreshed, name, refreshed_tensor(refreshed_seed, dtype))
    return meta, refreshed


# --- [Candidate A] In-place refresh helpers ---------------------------------


# Separate from plain ``_make_backend``: this variant additionally stubs
# ``_build_precomputed_metadata`` -- the AICPU metadata kernels are not
# available on CPU; the base builder only consumes them later, so a no-op
# keeps the field set under test intact.
def _inplace_make_backend() -> CsaAttentionBackend:
    backend = _make_backend()
    backend._build_precomputed_metadata = lambda *args, **kwargs: None  # noqa: SLF001
    return backend


def _multi_block_tables(batch: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Logical SWA/KV inputs shared by the physical cache groups."""
    tables = (
        torch.full((batch, 2), seed + 1, dtype=torch.int32),
        torch.full((batch, 4), seed + 2, dtype=torch.int32),
    )
    return tables


def _inplace_forward_metadata(
    kv_seq_lens: list[int],
    seed: int,
    *,
    positions: torch.Tensor | None = None,
    is_dummy: bool = False,
) -> SimpleNamespace:
    batch = len(kv_seq_lens)
    return SimpleNamespace(
        multi_block_tables=_multi_block_tables(batch, seed),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=list(kv_seq_lens),
        new_cache_slots_host_values=None,
        dsa_positions=positions
        if positions is not None
        else torch.tensor([kv_len - 1 for kv_len in kv_seq_lens], dtype=torch.int64),
        q_seq_lens_host=None,
        q_seq_lens=None,
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=_GRAPH_COLS,
        dsa_graph_token_capacity=_GRAPH_COLS * 2,
        is_dummy=is_dummy,
        is_prefill=False,
        is_chunked_prefill=False,
        expanded_decode_metadata=None,
    )


def _paired_metadata(seed: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Capture/replay metadata with the same graph bucket shapes.

    ``dsa_positions`` is one shared, stable tensor whose contents the runner
    rewrites in place every step (``_fill_graph_dsa_positions``), so the capture
    and the replay observe the same object -- exactly the graph contract.
    """
    batch = 4
    positions = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    capture = _inplace_forward_metadata([3, 5, 7, 9], seed, positions=positions)
    positions.copy_(torch.tensor([3, 5, 7, 9], dtype=torch.int64))
    replay = _inplace_forward_metadata([4, 6, 8, 10], seed + 1, positions=positions)
    return capture, replay


def _tensor_signature(value: torch.Tensor) -> tuple[int, tuple[int, ...], torch.dtype]:
    return (value.data_ptr(), tuple(value.shape), value.dtype)


def _persistent_field_tensors(dsa: Any) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name in _INPLACE_FIELDS:
        tensors[name] = getattr(dsa, name)
    for lid, layer in enumerate(dsa.block_tables):
        for ci, tensor in enumerate(layer):
            if tensor is not None and tensor.numel() > 0:
                tensors[f"block_tables[{lid}][{ci}]"] = tensor
    for lid, layer in enumerate(dsa.slot_mappings):
        for ci, tensor in enumerate(layer):
            if tensor is not None and tensor.numel() > 0:
                tensors[f"slot_mappings[{lid}][{ci}]"] = tensor
    return tensors


def _run_copy(persistent: Any, refreshed: Any, inplace_fields: frozenset[str] | None = None) -> None:
    CsaAttentionBackend._copy_graph_dsa_metadata(  # noqa: SLF001 - tested unit
        persistent, refreshed, inplace_fields=inplace_fields
    )


def _assert_fields_equal(left: Any, right: Any) -> None:
    # Scalar attributes.
    for name in ("max_query_len", "max_seq_len", "graph_token_capacity", "is_acl_graph"):
        assert getattr(left, name) == getattr(right, name), name
    # Value-derived tensors.
    for name in _INPLACE_FIELDS:
        a = getattr(left, name)
        b = getattr(right, name)
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert torch.equal(a, b), name
    # Per-layer manager tables (alias-aware, compare by storage contents).
    assert len(left.block_tables) == len(right.block_tables)
    seen: set[tuple[int, int]] = set()
    for lid, (ll, rr) in enumerate(zip(left.block_tables, right.block_tables, strict=True)):
        for ci, (a, b) in enumerate(zip(ll, rr, strict=True)):
            if a.numel() == 0 and b.numel() == 0:
                continue
            key = (a.data_ptr(), b.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            assert torch.equal(a, b), f"block_tables[{lid}][{ci}]"
    seen = set()
    for lid, (ll, rr) in enumerate(zip(left.slot_mappings, right.slot_mappings, strict=True)):
        for ci, (a, b) in enumerate(zip(ll, rr, strict=True)):
            if a.numel() == 0 and b.numel() == 0:
                continue
            key = (a.data_ptr(), b.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            assert torch.equal(a, b), f"slot_mappings[{lid}][{ci}]"


# --- [SWA device refresh helpers] ------------------------------------------


def _builder() -> DsaMetadataBuilder:
    caches_info, group_infos = build_cache_specs([0, 4, 128], 128, 3)
    return DsaMetadataBuilder(caches_info, group_infos)


def _run_swa(
    raw_bt: torch.Tensor,
    block_size: int,
    ctx_lens: list[int],
    q_lens: list[int],
    new_cache_slots: list[int],
    *,
    device_ctx_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    builder = _builder()
    return builder._process_swa_group(  # noqa: SLF001 - exercise the unit directly
        raw_bt,
        block_size,
        ctx_lens,
        q_lens,
        batch_size=len(ctx_lens),
        graph_slot_capacity=len(ctx_lens),
        graph_block_table_capacity_cols=int(raw_bt.size(1)),
        new_cache_slots=new_cache_slots,
        swa_device_ctx_lens=device_ctx_lens,
    )


# === [S4a] Alias dedup ===


def test_alias_dedup_preserves_values_and_owned_storage() -> None:
    persistent, refreshed = _make_metadata(persistent_seed=1, refreshed_seed=7)
    fields_before = {name: getattr(persistent, name).data_ptr() for name in _TENSOR_FIELDS + ("ori_sparse_indices",)}
    block_before = [[tensor.data_ptr() for tensor in layer] for layer in persistent.block_tables]
    slot_before = [[tensor.data_ptr() for tensor in layer] for layer in persistent.slot_mappings]
    rope_before = persistent.input_rope_by_ratio[1][0].data_ptr()
    asynchronous_input = torch.tensor([11, 12], dtype=torch.int32)
    input_storage = asynchronous_input.untyped_storage().data_ptr()
    refreshed.precomputed_metadata_inputs = (asynchronous_input,)
    CsaAttentionBackend._copy_graph_dsa_metadata(persistent, refreshed)
    for name in _TENSOR_FIELDS:
        tensor = getattr(persistent, name)
        assert tensor.data_ptr() == fields_before[name]
        expected = torch.zeros_like(tensor)
        expected[:_REFRESHED_ROWS] = 7
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    expected_indices = torch.full((_PERSISTENT_ROWS, 1, 128), -1, dtype=torch.int32)
    expected_indices[:_REFRESHED_ROWS] = 7
    assert persistent.ori_sparse_indices.data_ptr() == fields_before["ori_sparse_indices"]
    torch.testing.assert_close(persistent.ori_sparse_indices, expected_indices, rtol=0, atol=0)
    for layer_id in range(_N_LAYERS):
        for group in range(_GROUPS):
            table = persistent.block_tables[layer_id][group]
            slots = persistent.slot_mappings[layer_id][group]
            assert table.data_ptr() == block_before[layer_id][group]
            assert slots.data_ptr() == slot_before[layer_id][group]
            assert table.data_ptr() == persistent.block_tables[0][group].data_ptr()
            assert slots.data_ptr() == persistent.slot_mappings[0][group].data_ptr()
            expected_table = torch.full_like(table, -1)
            expected_table[:_REFRESHED_ROWS] = 70 + group
            expected_slots = torch.full_like(slots, -1)
            expected_slots[:_REFRESHED_ROWS] = 70 + group
            torch.testing.assert_close(table, expected_table, rtol=0, atol=0)
            torch.testing.assert_close(slots, expected_slots, rtol=0, atol=0)
    for tensor in persistent.input_rope_by_ratio[1]:
        expected = torch.zeros_like(tensor)
        expected[:_REFRESHED_ROWS] = 7
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    assert persistent.input_rope_by_ratio[1][0].data_ptr() == rope_before
    assert persistent.max_query_len == persistent.max_seq_len == persistent.graph_token_capacity == 7
    refreshed.precomputed_metadata_inputs = ()
    del asynchronous_input
    assert any(
        tensor.untyped_storage().data_ptr() == input_storage for tensor in persistent.precomputed_metadata_inputs
    )
    torch.testing.assert_close(
        persistent.precomputed_metadata_inputs[0], torch.tensor([11, 12], dtype=torch.int32), rtol=0, atol=0
    )


def test_alias_dedup_skips_only_duplicate_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    persistent, refreshed = _make_metadata(persistent_seed=1, refreshed_seed=7)
    pairs = {
        (a.data_ptr(), b.data_ptr()) for a, b in zip(persistent.block_tables[0], refreshed.block_tables[0], strict=True)
    }
    pairs |= {
        (a.data_ptr(), b.data_ptr())
        for a, b in zip(persistent.slot_mappings[0], refreshed.slot_mappings[0], strict=True)
    }
    observed: dict[tuple[int, int], int] = {}
    real_copy = torch.Tensor.copy_

    def count_copy(destination: torch.Tensor, source: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        pair = (destination.data_ptr(), source.data_ptr())
        if pair in pairs:
            observed[pair] = observed.get(pair, 0) + 1
        return real_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", count_copy)
    CsaAttentionBackend._copy_graph_dsa_metadata(persistent, refreshed)
    assert observed == {pair: 1 for pair in pairs}


def test_alias_dedup_keeps_last_write_for_shared_target_distinct_sources() -> None:
    # One target shared by two *distinct* refreshed sources must still be
    # written by every source in order (last write wins), as before dedup.
    target = torch.zeros((4, 2), dtype=torch.int32)
    source_a = torch.full((2, 2), 11, dtype=torch.int32)
    source_b = torch.full((2, 2), 22, dtype=torch.int32)
    seen: set[tuple[int, int]] = set()
    CsaAttentionBackend._copy_graph_tensor_list(
        [target, target], [source_a, source_b], "block_tables[0]", pad_value=-1, seen=seen
    )
    expected = torch.full((4, 2), -1, dtype=torch.int32)
    expected[:2] = 22
    assert torch.equal(target, expected)


def test_move_metadata_to_device_still_moves_cpu_fields() -> None:
    backend = _make_backend(torch.device("meta"))
    original = [
        torch.zeros((2,), dtype=torch.int32),
        torch.zeros((2, 2), dtype=torch.int32),
        torch.zeros((2,), dtype=torch.int32),
    ]
    metadata = SimpleNamespace(seq_lens=original[0], block_tables=[[original[1]]], slot_mappings=[[original[2]]])
    backend._move_metadata_to_device(metadata)
    actual = [metadata.seq_lens, metadata.block_tables[0][0], metadata.slot_mappings[0][0]]
    for source, target in zip(original, actual, strict=True):
        assert source.device.type == "cpu"
        assert target.device.type == "meta"
        assert target.shape == source.shape and target.dtype == source.dtype


def test_graph_metadata_h2d_staging_lives_until_event_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend()
    backend.device = Mock(type="npu", index=0)
    backend.supports_draft_to_target_overlap = True
    metadata = SimpleNamespace(
        is_acl_graph=True,
        seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        block_tables=[],
        slot_mappings=[],
    )

    class CompletionEvent:
        def __init__(self) -> None:
            self.complete = False
            self.recorded_stream = None

        def query(self) -> bool:
            return self.complete

        def record(self, stream: object) -> None:
            self.recorded_stream = stream

    class DeviceTensor:
        def __init__(self, device: torch.device) -> None:
            self.device = device

    current_stream = object()
    events: list[CompletionEvent] = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Event=lambda: events.append(CompletionEvent()) or events[-1], current_stream=lambda _: current_stream
        ),
        raising=False,
    )
    original_empty = torch.empty
    staging: list[torch.Tensor] = []

    def allocate(*args: object, **kwargs: object) -> torch.Tensor:
        assert kwargs.pop("pin_memory") is True
        tensor = original_empty(*args, **kwargs)
        staging.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    transfers: list[tuple[torch.Tensor, torch.device, bool]] = []

    def transfer(tensor: torch.Tensor, device: torch.device, *, non_blocking: bool = False) -> DeviceTensor:
        transfers.append((tensor, device, non_blocking))
        return DeviceTensor(device)

    monkeypatch.setattr(torch.Tensor, "to", transfer)

    source = metadata.seq_lens
    backend._move_metadata_to_device(metadata, tensor_fields=("seq_lens",), move_tables=False)
    metadata.seq_lens = source + 10
    backend._move_metadata_to_device(metadata, tensor_fields=("seq_lens",), move_tables=False)

    assert len(staging) == 2
    assert len(events) == 2
    assert transfers[0][0] is staging[0]
    assert transfers[1][0] is staging[1]
    assert transfers[0][2] is True and transfers[1][2] is True
    assert events[0].recorded_stream is current_stream
    torch.testing.assert_close(staging[0], source)
    torch.testing.assert_close(staging[1], source + 10)
    assert all(
        any(staged is saved for saved in staging)
        for uploads in backend._metadata_staging.values()
        for staged, _ in uploads
    )

    events[0].complete = True
    metadata.seq_lens = source + 20
    backend._move_metadata_to_device(metadata, tensor_fields=("seq_lens",), move_tables=False)

    assert len(staging) == 2
    assert transfers[2][0] is staging[0]
    torch.testing.assert_close(staging[0], source + 20)


def test_eager_metadata_move_keeps_synchronous_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    device = Mock(type="npu", index=0)
    backend = _make_backend()
    backend.device = device
    backend.supports_draft_to_target_overlap = True
    metadata = SimpleNamespace(
        is_acl_graph=False,
        seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        block_tables=[],
        slot_mappings=[],
    )

    class DeviceTensor:
        def __init__(self, target: torch.device) -> None:
            self.device = target

    transfers: list[tuple[torch.device, bool]] = []

    def transfer(tensor: torch.Tensor, target: torch.device, *, non_blocking: bool = False) -> DeviceTensor:
        transfers.append((target, non_blocking))
        return DeviceTensor(target)

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    backend._move_metadata_to_device(metadata, tensor_fields=("seq_lens",), move_tables=False)

    assert transfers == [(device, False)]
    assert not backend._metadata_staging


# === [Candidate A] In-place refresh ===


# -- (a) off path equivalence ------------------------------------------------


def test_graph_host_block_tables_match_device_build_without_device_reads() -> None:
    backend = _inplace_make_backend()
    backend.supports_draft_to_target_overlap = True
    _, metadata = _paired_metadata(seed=3)
    expected = backend._build_dsa_metadata_for_forward(metadata)
    metadata.multi_block_tables_host = metadata.multi_block_tables
    # Shape-only device tables cannot be read back. Every slot and ring table
    # must instead come from the scheduler's equivalent host copy.
    metadata.multi_block_tables = tuple(torch.empty_like(table, device="meta") for table in metadata.multi_block_tables)
    actual = backend._build_dsa_metadata_for_forward(metadata)
    _assert_fields_equal(expected, actual)


def test_graph_host_query_lengths_avoid_device_readback() -> None:
    backend = _inplace_make_backend()
    _, metadata = _paired_metadata(seed=3)
    expected = backend._build_dsa_metadata_for_forward(metadata)
    metadata.q_seq_lens_host_values = expected.seq_lens_q.tolist()
    metadata.q_seq_lens_host = None
    # Reading the device query lengths would fail for a shape-only tensor.
    metadata.q_seq_lens = torch.empty_like(expected.seq_lens_q, device="meta")
    actual = backend._build_dsa_metadata_for_forward(metadata)
    _assert_fields_equal(expected, actual)


# -- (b) on/off numerical equivalence ---------------------------------------


def test_inplace_refresh_matches_off_path_bitwise() -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)

    persistent_on = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    persistent_off = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    refreshed_on = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        replay_meta, inplace_target=persistent_on
    )
    assert refreshed_on.refresh_inplace_fields == frozenset(_INPLACE_FIELDS)

    refreshed_off = backend._build_dsa_metadata_for_forward(replay_meta)  # noqa: SLF001

    _run_copy(persistent_on, refreshed_on, refreshed_on.refresh_inplace_fields)
    _run_copy(persistent_off, refreshed_off)

    _assert_fields_equal(persistent_on, persistent_off)
    # Legacy values (built independently) must also match byte-for-byte.
    for name in _INPLACE_FIELDS:
        assert torch.equal(getattr(persistent_on, name), getattr(refreshed_off, name)), name


def test_inplace_refresh_matches_capture_values_for_same_metadata() -> None:
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    snapshot = {name: getattr(persistent, name).clone() for name in _INPLACE_FIELDS}

    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        capture_meta, inplace_target=persistent
    )
    _run_copy(persistent, refreshed, refreshed.refresh_inplace_fields)

    for name in _INPLACE_FIELDS:
        assert torch.equal(getattr(persistent, name), snapshot[name]), name


def test_query_descriptor_reuse_tracks_layout_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _inplace_make_backend()
    backend._builder.async_graph_refresh = True
    metadata = _inplace_forward_metadata(
        [127, 128, 2047, 2048], seed=3, positions=torch.tensor([126, 127, 2046, 2047, 0, 0, 0, 0], dtype=torch.int64)
    )
    metadata.dsa_graph_token_capacity = 8
    metadata.q_seq_lens_host_values = [1, 1, 1, 1]
    persistent = backend._build_dsa_metadata_for_forward(metadata)
    names = ("seq_lens_q", "actual_seq_lengths_query", "max_seqlen_q")
    by_name = {name: getattr(persistent, name).data_ptr() for name in names}
    addresses = set(by_name.values())
    copies = []
    original = backend._builder._copy_to_destination

    def copied(destination: torch.Tensor, source: torch.Tensor, *, non_blocking: bool = True) -> None:
        if destination.data_ptr() in addresses:
            copies.append(destination.data_ptr())
        original(destination, source, non_blocking=non_blocking)

    monkeypatch.setattr(backend._builder, "_copy_to_destination", copied)
    for values, expected_copies in (([1, 1, 1, 1], 0), ([2, 1, 2, 1], 3), ([2, 1, 2, 1], 0), ([1, 1, 1, 1], 3)):
        copies.clear()
        metadata.q_seq_lens_host_values = values
        positions = [
            position
            for kv_len, q_len in zip([127, 128, 2047, 2048], values, strict=True)
            for position in range(kv_len - q_len, kv_len)
        ]
        metadata.dsa_positions.zero_()
        metadata.dsa_positions[: len(positions)].copy_(torch.tensor(positions, dtype=torch.int64))
        backend._build_dsa_metadata_for_forward(metadata, inplace_target=persistent)
        expected = backend._build_dsa_metadata_for_forward(metadata)
        for name in names:
            assert torch.equal(getattr(persistent, name), getattr(expected, name)), name
        assert len(copies) == expected_copies
        assert persistent.query_lengths_host == tuple(values)
        assert persistent.seq_lens_q.tolist() == values
        assert persistent.actual_seq_lengths_query.tolist() == (
            [0, 1, 2, 3, 4] if values == [1, 1, 1, 1] else [0, 2, 3, 5, 6]
        )
        assert persistent.max_seqlen_q.item() == max(values)
        assert all(getattr(persistent, name).data_ptr() == by_name[name] for name in names)


# -- (c) captured-address stability -----------------------------------------


def test_inplace_refresh_preserves_captured_addresses() -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    before = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}

    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        replay_meta, inplace_target=persistent
    )
    _run_copy(persistent, refreshed, refreshed.refresh_inplace_fields)

    after = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}
    assert before == after

    # The value-derived fields alias the captured objects, not copies.
    for name in _INPLACE_FIELDS:
        assert getattr(persistent, name) is getattr(refreshed, name), name


# -- (d) no host sync / no new per-field ops --------------------------------


def test_inplace_refresh_does_not_host_sync_or_move_fields(monkeypatch) -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001

    def _forbidden_cpu(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected host sync .cpu() on {self.shape}")

    original_to = torch.Tensor.to
    to_calls: list[int] = []

    def counting_to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        to_calls.append(id(self))
        return original_to(self, *args, **kwargs)

    copy_calls: list[int] = []
    original_copy = torch.Tensor.copy_

    def counting_copy(self, src, *args, **kwargs):  # type: ignore[no-untyped-def]
        copy_calls.append(id(self))
        return original_copy(self, src, *args, **kwargs)

    alias_checks: list[str] = []
    original_assert = CsaAttentionBackend._assert_graph_tensor_alias  # noqa: SLF001

    def counting_assert(persistent_t, refreshed_t, name):  # type: ignore[no-untyped-def]
        alias_checks.append(name)
        return original_assert(persistent_t, refreshed_t, name)

    monkeypatch.setattr(torch.Tensor, "cpu", _forbidden_cpu)
    monkeypatch.setattr(torch.Tensor, "to", counting_to)
    monkeypatch.setattr(torch.Tensor, "copy_", counting_copy)
    monkeypatch.setattr(CsaAttentionBackend, "_assert_graph_tensor_alias", staticmethod(counting_assert))

    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        replay_meta, inplace_target=persistent
    )
    inplace_ids = {id(getattr(persistent, name)) for name in _INPLACE_FIELDS}
    moved = [ptr for ptr in to_calls if ptr in inplace_ids]
    assert moved == [], "in-place fields must not be moved with .to()"

    # Only the replay pass matters here: the build itself legitimately writes
    # the captured buffers (one ``copy_`` per in-place field).
    build_copy_count = len(copy_calls)
    _run_copy(persistent, refreshed, refreshed.refresh_inplace_fields)

    copied = [ptr for ptr in copy_calls[build_copy_count:] if ptr in inplace_ids]
    assert copied == [], "in-place fields must not be copied on the replay path"
    assert set(alias_checks) == set(_INPLACE_FIELDS)


def test_inplace_refresh_uses_no_device_d2h_in_device_slot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SWA device slot build, when handed a captured buffer, must not call
    # ``.tolist()`` (a D2H sync) and must write the target buffer in place.
    raw_bt = torch.tensor([[5], [7]], dtype=torch.int32)
    ctx = torch.tensor([3, 4], dtype=torch.int32)
    target = torch.full((4,), -1, dtype=torch.int32)

    def _forbidden_tolist(self):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected .tolist() in the device slot path")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", _forbidden_tolist)
        out = DsaMetadataBuilder._swa_slots_on_device(  # noqa: SLF001
            raw_bt,
            block_size=128,
            semantic_cols=1,
            out_slot_rows=4,
            batch_size=2,
            q_lens=[1, 1],
            new_cache_slots=None,
            query_total_tokens=2,
            device_ctx_lens=ctx,
            dst_slots=target,
        )

    assert out is target
    assert torch.equal(target, torch.tensor([5 * 128 + 2, 7 * 128 + 3, -1, -1], dtype=torch.int32))


# -- (e) ownership conflicts -------------------------------------------------


def test_reset_forward_does_not_touch_captured_tensors() -> None:
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    before = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}
    contents = {name: t.clone() for name, t in _persistent_field_tensors(persistent).items()}

    backend.reset_forward(capture_meta)

    assert capture_meta.dsa_metadata is None
    assert capture_meta.dsa_positions is None
    after = {name: _tensor_signature(t) for name, t in _persistent_field_tensors(persistent).items()}
    assert before == after
    for name, value in contents.items():
        assert torch.equal(_persistent_field_tensors(persistent)[name], value), name


def test_shared_metadata_draft_group_limit_is_eager_and_isolated() -> None:
    """Explicit shared-request slicing leaves independent captured storage untouched."""
    # The DSpark draft shares the request's multi_block_tables but owns one SWA
    # group and is always eager (no graph target): its build must never write
    # into the target's captured persistent metadata.
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    before = {name: t.clone() for name, t in _persistent_field_tensors(persistent).items()}

    draft = backend._build_dsa_metadata_for_forward(replay_meta, group_limit=1)  # noqa: SLF001
    assert draft.refresh_inplace_fields == frozenset()
    assert draft is not persistent

    after = _persistent_field_tensors(persistent)
    for name, value in before.items():
        assert torch.equal(after[name], value), name


def test_inplace_falls_back_when_shapes_do_not_match() -> None:
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    before = _tensor_signature(persistent.seq_lens)
    original_values = persistent.seq_lens.clone()

    # A different batch size cannot alias the captured buffer; the builder must
    # allocate fresh and let the copy path handle it, leaving the address alone.
    other_meta = _inplace_forward_metadata([1, 2, 3], seed=9)
    refreshed = backend._build_dsa_metadata_for_forward(  # noqa: SLF001
        other_meta, inplace_target=persistent
    )
    assert refreshed.seq_lens is not persistent.seq_lens
    assert "seq_lens" not in refreshed.refresh_inplace_fields
    assert _tensor_signature(persistent.seq_lens) == before
    torch.testing.assert_close(persistent.seq_lens, original_values, rtol=0, atol=0)


# -- (f) capture-time gating -------------------------------------------------


def test_capture_path_never_uses_inplace_target() -> None:
    # The capture entry point must build a fresh metadata that never aliases or
    # writes into an existing persistent object; the in-place mechanism is a
    # replay-time concern only.
    backend = _inplace_make_backend()
    capture_meta, _ = _paired_metadata(seed=0)

    backend.prepare_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001 - capture entry
    captured = capture_meta.dsa_metadata
    assert captured is not None
    assert captured.refresh_inplace_fields == frozenset()

    other_meta = _inplace_forward_metadata([3, 5, 7, 9], seed=7)
    fresh = backend._build_dsa_metadata_for_forward(other_meta)  # noqa: SLF001
    assert fresh.refresh_inplace_fields == frozenset()
    for name in _INPLACE_FIELDS:
        assert getattr(fresh, name) is not getattr(captured, name), name


# -- (g) host planning avoids per-slot tensor reads -------------------------


def test_host_slot_planning_avoids_per_row_item(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _inplace_make_backend()
    capture_meta, replay_meta = _paired_metadata(seed=0)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)  # noqa: SLF001
    expected = backend._build_dsa_metadata_for_forward(replay_meta)  # noqa: SLF001

    def reject_item(self: torch.Tensor) -> object:
        raise AssertionError("host planning must read block tables in bulk")

    monkeypatch.setattr(torch.Tensor, "item", reject_item)
    actual = backend._build_dsa_metadata_for_forward(replay_meta, inplace_target=persistent)  # noqa: SLF001
    expected_fields = _persistent_field_tensors(expected)
    for name, value in _persistent_field_tensors(actual).items():
        assert torch.equal(value, expected_fields[name]), name


def test_replay_copy_guard_skips_alias_and_copies_disjoint(monkeypatch) -> None:
    calls: list[int] = []
    original = torch.Tensor.copy_

    def counting_copy(self, src, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(self.data_ptr())
        return original(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counting_copy)
    aliased = torch.tensor([3, 3, 3], dtype=torch.int32)
    # Same storage -> data_ptr equality -> an aliased field needs no write.
    DsaAttentionBackend._copy_graph_tensor(aliased, aliased, "seq_lens")
    assert calls == []

    persistent = torch.tensor([0, 0, 0], dtype=torch.int32)
    refreshed = torch.tensor([1, 2, 3], dtype=torch.int32)
    DsaAttentionBackend._copy_graph_tensor(persistent, refreshed, "seq_lens")
    assert calls == [persistent.data_ptr()]
    assert torch.equal(persistent, refreshed)


# === SWA device refresh mechanism ===


def test_swa_device_slots_bit_identical_to_host_path() -> None:
    for raw_bt, block_size, ctx_lens, q_lens, new_cache_slots in _SWA_CASES:
        device_ctx = torch.tensor(ctx_lens, dtype=torch.int32)
        off_bt, off_slots = _run_swa(
            raw_bt.clone(),
            block_size,
            ctx_lens,
            q_lens,
            new_cache_slots,
            device_ctx_lens=None,
        )
        on_bt, on_slots = _run_swa(
            raw_bt.clone(),
            block_size,
            ctx_lens,
            q_lens,
            new_cache_slots,
            device_ctx_lens=device_ctx,
        )
        assert torch.equal(off_bt, on_bt), (raw_bt.tolist(), ctx_lens)
        assert torch.equal(off_slots, on_slots), (raw_bt.tolist(), ctx_lens)


def test_swa_device_refresh_writes_into_captured_storage_in_place() -> None:
    persistent_bt = torch.full((2, 3), -7, dtype=torch.int32)
    persistent_slots = torch.full((2,), -7, dtype=torch.int32)
    bt_ptr = persistent_bt.data_ptr()
    slots_ptr = persistent_slots.data_ptr()

    raw_bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    on_bt, on_slots = _run_swa(
        raw_bt,
        4,
        [5, 9],
        [1, 1],
        [],
        device_ctx_lens=torch.tensor([5, 9], dtype=torch.int32),
    )
    CsaAttentionBackend._copy_graph_tensor_list([persistent_bt], [on_bt], "block_tables[0]", pad_value=-1)
    CsaAttentionBackend._copy_graph_tensor_list([persistent_slots], [on_slots], "slot_mappings[0]", pad_value=-1)

    assert persistent_bt.data_ptr() == bt_ptr
    assert persistent_slots.data_ptr() == slots_ptr
    assert torch.equal(persistent_bt, on_bt)
    assert torch.equal(persistent_slots, on_slots)
    torch.testing.assert_close(persistent_bt, torch.tensor([[0, 1, -1], [-1, 3, 2]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(persistent_slots, torch.tensor([4, 8], dtype=torch.int32), rtol=0, atol=0)


# Native rotary dispatch and independent eager fallback contract.


def _build_cache(max_pos: int, rotary_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _stub_partial_rotary() -> Any:
    """Native-call recorder; it deliberately computes no fake rotary result."""
    calls: list[tuple[bool, tuple[int, ...], int, int]] = []

    def op(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_start_dim: int,
        rope_head_dim: int,
        inverse: bool = False,
    ) -> None:
        assert cos.dtype == sin.dtype == x.dtype
        calls.append((inverse, tuple(x.shape), rope_start_dim, rope_head_dim))

    op.calls = calls  # type: ignore[attr-defined]
    return op


@pytest.mark.parametrize(
    ("shape", "inverse"),
    [((8, 4, 512), True), ((8, 1, 512), True), ((5, 128), True), ((6, 3, 256), False)],
)
def test_fused_rotary_dispatch_preserves_inplace_inputs(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], inverse: bool
) -> None:
    op = _stub_partial_rotary()
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: op)
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda _x: True)
    cache = _build_cache(256, 64)
    x = torch.zeros(shape, dtype=torch.bfloat16)
    positions = torch.arange(shape[0], dtype=torch.int32)
    output = csa2.apply_interleaved_rope(x, positions, cache, inverse=inverse)
    native_shape = shape if len(shape) == 3 else (shape[0], 1, shape[1])
    assert op.calls == [(inverse, native_shape, shape[-1] - 64, 64)]
    assert output.shape == shape
    assert output.data_ptr() == x.data_ptr()


def test_inverse_guard_matrix(monkeypatch: Any) -> None:
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: lambda *_a, **_k: None)
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda x: True)

    cache = _build_cache(64, 64)
    x = torch.zeros(4, 2, 256, dtype=torch.bfloat16)
    positions = torch.arange(4, dtype=torch.int32)

    assert csa2._rope_fused_supported(x, positions, cache, True) is True
    assert csa2._rope_fused_supported(x, positions, cache, False) is True

    # dtype / rope width / cache dtype / contiguity / rank / device.
    assert csa2._rope_fused_supported(x.to(torch.float16), positions, cache, True) is False
    assert csa2._rope_fused_supported(x, positions, _build_cache(64, 16), True) is False
    assert csa2._rope_fused_supported(x, positions, cache.to(torch.bfloat16), True) is False
    assert (
        csa2._rope_fused_supported(torch.zeros(4, 2, 512, dtype=torch.bfloat16)[..., ::2], positions, cache, True)
        is False
    )
    assert csa2._rope_fused_supported(torch.zeros(4, 2, 1, 256, dtype=torch.bfloat16), positions, cache, True) is False
    # P2-a: an ODD table width (65 // 2 == 32, so the old 2*half == 64 passed).
    odd_cache = torch.randn(64, 65)
    assert csa2._rope_fused_supported(x, positions, odd_cache, True) is False
    assert csa2._rope_fused_supported(x, positions, odd_cache, False) is False
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda x: False)
    assert csa2._rope_fused_supported(x, positions, cache, True) is False


def test_inverse_falls_back_when_op_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(csa2, "_rope_fused_device_supported", lambda _x: True)
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: None)
    cache = torch.cat((torch.zeros(64, 32), torch.ones(64, 32)), dim=-1)
    x = torch.full((2, 1, 128), 9, dtype=torch.bfloat16)
    x[..., 64:] = torch.tensor([1, 2] * 32, dtype=torch.bfloat16)
    fallback = csa2.apply_interleaved_rope(x, torch.tensor([0, 1], dtype=torch.int32), cache, inverse=True)
    expected = torch.full_like(x, 9)
    expected[..., 64:] = torch.tensor([2, -1] * 32, dtype=torch.bfloat16)
    torch.testing.assert_close(fallback, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# P1-b: amplitude domain of the mixed tolerance
# ---------------------------------------------------------------------------

# Deterministic hash-like pattern (no RNG) so the scale boundary is stable
# across torch builds; found by the rev-3 stress sweep (scale=8 -> ratio ~0.82,
# scale=16 -> ~1.39).


@pytest.mark.parametrize("async_graph_inputs", [False, True])
def test_dsa_replay_inplace_refresh_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, async_graph_inputs: bool
) -> None:
    backend = _inplace_make_backend()
    backend.supports_draft_to_target_overlap = async_graph_inputs
    assert not backend._builder.async_graph_refresh
    capture_meta, replay_meta = _paired_metadata(seed=3)
    persistent = backend._build_dsa_metadata_for_forward(capture_meta)
    replay_meta.dsa_metadata = persistent
    original = backend._build_dsa_metadata_for_forward
    destinations = []

    def build(metadata: Any, **kwargs: Any) -> Any:
        destinations.append(kwargs.get("inplace_target"))
        return original(metadata, **kwargs)

    monkeypatch.setattr(backend, "_build_dsa_metadata_for_forward", build)
    aliases: list[str] = []
    original_assert = backend._assert_graph_tensor_alias

    def assert_alias(persistent_tensor: Any, refreshed_tensor: Any, name: str) -> None:
        aliases.append(name)
        original_assert(persistent_tensor, refreshed_tensor, name)

    monkeypatch.setattr(type(backend), "_assert_graph_tensor_alias", staticmethod(assert_alias))
    backend.refresh_dsa_metadata_for_graph_replay(replay_meta)
    assert len(destinations) == 1
    assert (destinations[0] is persistent) is async_graph_inputs
    assert bool(aliases) is async_graph_inputs
