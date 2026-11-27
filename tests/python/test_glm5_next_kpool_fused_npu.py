# Copyright 2026 The xLLM Authors.
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

"""Check paged pooling and prefix restoration against persistent CANN state."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from xllm.python.models.glm5_next_kpool import (  # noqa: E402
    _store_compressed_kpool_torch,
    prepare_fused_kpool_states,
    update_compressed_kpool,
)


@torch.inference_mode()
def test_fused_kpool_store_reuses_kernels_within_prefill_length_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.kernels_npu.triton import kpool_compress

    torch.npu.set_device(0)
    torch.manual_seed(42)
    launches: dict[str, list[object]] = {}

    def track_launch(name: str) -> None:
        kernel = getattr(kpool_compress, name)
        launch = kernel.run
        compiled_kernels: list[object] = []
        launches[name] = compiled_kernels

        def record_launch(*args: torch.Tensor, **kwargs: object) -> object:
            compiled = launch(*args, **kwargs)
            assert compiled is not None
            compiled_kernels.append(compiled)
            return compiled

        monkeypatch.setattr(kernel, "run", record_launch)

    track_launch("_store_fused_pools_kernel")
    track_launch("_store_fused_kpool_kernel")
    head_dim, rate, page_size = 128, 4, 16
    ids = torch.tensor([1], dtype=torch.int32, device="npu")
    ape = torch.zeros(rate, head_dim, device="npu")
    query_lengths = (16384, 576, 13, 15, 17, 25, 37, 63, 65, 97, 1, 2, 3, 4, 5, 8)
    for query_len in query_lengths:
        raw = torch.randn(query_len, head_dim, device="npu")
        gates = torch.randn_like(raw)
        valid = torch.ones(query_len, dtype=torch.bool, device="npu")
        positions = torch.arange(3, query_len + 3, device="npu")
        pool_count = (query_len + rate - 1) // rate
        pages = (pool_count + page_size - 1) // page_size
        table = torch.arange(1, pages + 1, device="npu", dtype=torch.int32).unsqueeze(0)
        cache = torch.zeros(257, page_size, 1, head_dim, dtype=torch.bfloat16, device="npu")
        tail_capacity = 5 if query_len in (2, 5, 8) else 9
        tail = torch.randn(2, 2, tail_capacity, head_dim, device="npu")
        states = [
            (
                torch.randn(pool_count, head_dim, dtype=torch.bfloat16, device="npu"),
                torch.randn(2, rate, head_dim, device="npu"),
            )
        ]
        expected_cache, expected_tail = cache.clone(), tail.clone()
        _store_compressed_kpool_torch(
            raw,
            gates,
            valid,
            valid,
            positions,
            expected_cache,
            expected_tail,
            ids,
            ids,
            table,
            [query_len],
            ape,
            rate,
            fused_states=states,
        )
        kpool_compress.store_fused_kpool(
            raw, gates, valid, valid, positions, cache, tail, ids, ids, table, query_len, rate, states
        )
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(tail, expected_tail, rtol=0, atol=0)

    pool_launches = launches["_store_fused_pools_kernel"]
    assert len(pool_launches) == 10
    assert all(kernel is pool_launches[0] for kernel in pool_launches)

    tail_launches = launches["_store_fused_kpool_kernel"]
    assert len(tail_launches) == len(query_lengths)
    for first in (0, 2, 4, 6, 8, 12, 14):
        assert tail_launches[first] is tail_launches[first + 1]
    assert len({id(kernel) for kernel in tail_launches}) == 9


@pytest.mark.parametrize(
    "query_lens,prefix_lens",
    [((16384,), (0,)), ((16384,), (3,)), ((1, 16383), (16383, 0))],
)
@torch.inference_mode()
def test_cp_tnd_selection_preserves_per_query_causal_tails(
    query_lens: tuple[int, ...], prefix_lens: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    torch.ops.load_library(artifact)
    from xllm.python.kernels_npu.sparse_attention import pool_key_indexer
    from xllm.python.model_executor.cp_utils import build_cp_context, cp_real_query_rows
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context
    from xllm.python.models import glm5_next
    from xllm.python.models.glm5_next import Glm5NextConfig, Glm5NextIndexer

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", pool_key_indexer, raising=False)

    torch.npu.set_device(0)
    torch.manual_seed(43)
    device = torch.device("npu:0")
    config = Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=32,
        index_head_dim=128,
        index_kpool=4,
        index_topk=2048,
        index_kpool_always_select_tail=True,
    )
    indexer = Glm5NextIndexer(config, 1, torch.bfloat16, device).to(device=device, dtype=torch.bfloat16)
    tokens = sum(query_lens)
    qr = torch.randn(tokens, 16, device=device, dtype=torch.bfloat16)
    weights = torch.randn(tokens, 32, device=device, dtype=torch.bfloat16)
    positions = torch.cat(
        [torch.arange(prefix, prefix + length, device=device) for prefix, length in zip(prefix_lens, query_lens)]
    )
    kv_lens = [prefix + length for prefix, length in zip(prefix_lens, query_lens)]
    pages = max((length + 63) // 64 for length in kv_lens) + 1
    table = torch.arange(1, pages * len(query_lens) + 1, device=device, dtype=torch.int32).reshape(
        len(query_lens), pages
    )
    cache = torch.randn(pages * len(query_lens) + 1, 16, 1, 128, device=device, dtype=torch.bfloat16)
    cp_context = build_cp_context(list(query_lens), kv_lens, 8, 0, device)
    rows = cp_real_query_rows(cp_context)
    hidden = torch.randn(1, tokens, 16, device=device, dtype=torch.bfloat16)
    full_keys, full_weights = indexer._project_key_weights(hidden)
    local_keys, local_weights = indexer._project_key_weights(hidden, rows)
    torch.testing.assert_close(local_keys, full_keys, rtol=0, atol=0)
    torch.testing.assert_close(
        local_weights[0].index_select(0, rows), full_weights[0].index_select(0, rows), rtol=0, atol=0
    )
    expected = torch.full((tokens, 1, indexer.topk_output_width), -1, device=device, dtype=torch.int32)
    start = 0
    for request, length in enumerate(query_lens):
        selected = rows[(rows >= start) & (rows < start + length)]
        if selected.numel():
            selected_indices = indexer._select_topk_fused_pa(
                indexer.wq_b(qr.index_select(0, selected)).reshape(1, selected.numel(), 32, 128),
                weights.index_select(0, selected).unsqueeze(0),
                torch.tensor([kv_lens[request]], device=device, dtype=torch.int32),
                torch.ones(1, selected.numel(), device=device, dtype=torch.bool),
                cache,
                table[request : request + 1],
                query_positions=positions.index_select(0, selected).unsqueeze(0),
                attention_mask_all_valid=True,
            )
            expected.index_copy_(0, selected, selected_indices.reshape(-1, 1, indexer.topk_output_width).int())
        start += length
    with forward_context(ForwardContext(None, device, None, [], cp_context=cp_context)):
        actual = indexer._select_topk_fused_cp_paged(qr, weights, positions, rows, cache, table, cp_context, tokens)
        local = indexer._select_topk_fused_cp_paged(
            qr, weights, positions, rows, cache, table, cp_context, tokens, local_output=True
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        local.index_select(0, cp_context.query_index), expected.index_select(0, rows), rtol=0, atol=0
    )
    padding = torch.ones(cp_context.total_local, dtype=torch.bool, device=device)
    padding[cp_context.query_index] = False
    assert torch.all(local[padding] == -1)


@pytest.mark.parametrize("chunks", [(64,), (1, 2, 3, 4, 17, 37), (4096,), (3, 4093), (16384,), (3, 16381)])
@pytest.mark.parametrize("restore_prefix", [False, True])
@torch.inference_mode()
def test_fused_kpool_preserves_fp32_state(chunks: tuple[int, ...], restore_prefix: bool) -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "key_pool"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _check_pooling(chunks, restore_prefix)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def _check_pooling(chunks: tuple[int, ...], restore_prefix: bool) -> None:
    tokens = sum(chunks)
    hidden = torch.randn(tokens, 4096, device="npu", dtype=torch.bfloat16)
    key_weight = torch.randn(128, 4096, device="npu", dtype=torch.bfloat16) * 0.02
    gate_weight = torch.randn_like(key_weight) * 0.02
    ape = torch.randn(4, 128, device="npu")
    norm_weight = torch.randn(128, device="npu")
    norm_bias = torch.randn_like(norm_weight)
    tail = torch.zeros(4, 2, 9, 128, device="npu")
    tail[:, 1].fill_(float("-inf"))
    pool_capacity = (tokens + 3) // 4 + 1
    cache = torch.zeros(2, pool_capacity, 1, 128, device="npu", dtype=torch.bfloat16)
    table = torch.tensor([[1]], device="npu", dtype=torch.int32)
    oracle_state = torch.zeros(pool_capacity + 1, 4, 256, device="npu")
    oracle_table = torch.arange(1, pool_capacity + 1, device="npu", dtype=torch.int32)[None]
    offset = 0
    read_slot = 1
    for chunk_index, length in enumerate(chunks):
        write_slot = 2 if restore_prefix and chunk_index > 0 else 1
        read_ids = torch.tensor([read_slot], device="npu", dtype=torch.int32)
        write_ids = torch.tensor([write_slot], device="npu", dtype=torch.int32)
        old_tail = tail[read_slot].clone()
        positions = torch.arange(offset, offset + length, device="npu")[None]
        inputs = hidden[offset : offset + length][None]
        valid = torch.ones(1, length, device="npu", dtype=torch.bool)
        states = prepare_fused_kpool_states(
            inputs,
            key_weight,
            gate_weight,
            ape,
            norm_weight,
            norm_bias,
            tail,
            read_ids,
            positions,
            [length],
            4,
            1e-6,
        )
        raw = F.layer_norm(F.linear(inputs, key_weight).float(), (128,), norm_weight, norm_bias, 1e-6)
        gates = F.linear(inputs, gate_weight).float()
        update_compressed_kpool(
            raw,
            gates,
            valid,
            valid,
            positions,
            cache,
            tail,
            read_ids,
            write_ids,
            table,
            [length],
            ape,
            4,
            fused_states=states,
        )
        expected = torch.ops.xllm_ops.key_pool(
            inputs[0],
            key_weight,
            gate_weight,
            ape,
            oracle_state,
            oracle_table,
            torch.tensor([offset], device="npu", dtype=torch.int32),
            norm_weight,
            norm_bias,
            torch.tensor([0, length], device="npu", dtype=torch.int32),
            4,
            1e-6,
        )
        count = (offset % 4 + length) // 4
        first_pool = offset // 4
        torch.testing.assert_close(
            cache[1, first_pool : first_pool + count, 0].cpu(),
            expected[:count].cpu(),
            rtol=0,
            atol=0,
        )
        expected_tail = old_tail.cpu()
        retained = min(length, tail.shape[2])
        retained_positions = positions[0, -retained:].cpu() % tail.shape[2]
        expected_tail[0].index_copy_(0, retained_positions, raw[0, -retained:].cpu())
        expected_tail[1].index_copy_(0, retained_positions, gates[0, -retained:].cpu())
        offset += length
        remainder = offset % 4
        if remainder:
            rows = torch.arange(offset - remainder, offset, device="npu") % tail.shape[2]
            partial = oracle_state[1 + offset // 4, :remainder].reshape(remainder, 2, 128).permute(1, 0, 2)
            expected_tail.index_copy_(1, rows.cpu(), partial.cpu())
        torch.testing.assert_close(tail[write_slot].cpu(), expected_tail, rtol=0, atol=0)
        if read_slot != write_slot:
            torch.testing.assert_close(tail[read_slot].cpu(), old_tail.cpu(), rtol=0, atol=0)
        read_slot = write_slot


@pytest.mark.parametrize("valid_tokens", [0, 2])
@torch.inference_mode()
def test_long_fused_kpool_finds_end_before_padding(valid_tokens: int) -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.kernels_npu.triton.kpool_compress import store_fused_kpool

    torch.npu.set_device(0)
    torch.manual_seed(42)
    query_len, head_dim, rate = 1024, 128, 4
    raw = torch.randn(query_len, head_dim, device="npu")
    gates = torch.randn_like(raw)
    positions = torch.arange(3, query_len + 3, device="npu")
    valid = torch.arange(query_len, device="npu") < valid_tokens
    positions.masked_fill_(~valid, -1)
    ids = torch.tensor([1], dtype=torch.int32, device="npu")
    table = ids[:, None].contiguous()
    tail = torch.randn(2, 2, 9, head_dim, device="npu")
    cache = torch.zeros(2, 257, 1, head_dim, dtype=torch.bfloat16, device="npu")
    states = [
        (
            torch.randn(257, head_dim, dtype=torch.bfloat16, device="npu"),
            torch.randn(2, rate, head_dim, device="npu"),
        )
    ]
    expected_tail, expected_cache = tail.clone(), cache.clone()
    _store_compressed_kpool_torch(
        raw,
        gates,
        valid,
        valid,
        positions,
        expected_cache,
        expected_tail,
        ids,
        ids,
        table,
        [query_len],
        torch.zeros(rate, head_dim, device="npu"),
        rate,
        fused_states=states,
    )
    store_fused_kpool(
        raw,
        gates,
        valid,
        valid,
        positions,
        cache,
        tail,
        ids,
        ids,
        table,
        query_len,
        rate,
        states,
    )
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(tail, expected_tail, rtol=0, atol=0)


@torch.inference_mode()
def test_batched_fused_kpool_keeps_requests_and_cached_geometry_independent() -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "key_pool"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    query_lens = [1, 0, 3, 5]
    hidden = torch.randn(sum(query_lens), 4096, dtype=torch.bfloat16, device="npu")
    key_weight = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu") * 0.02
    gate_weight = torch.randn_like(key_weight) * 0.02
    ape = torch.randn(4, 128, device="npu")
    norm_weight = torch.randn(128, device="npu")
    norm_bias = torch.randn_like(norm_weight)
    tail = torch.randn(4, 2, 9, 128, device="npu")
    ids = torch.tensor([3, 0, 2, 1], device="npu", dtype=torch.int32)
    positions = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13], device="npu")
    old_tail = tail.clone()
    batched = prepare_fused_kpool_states(
        hidden,
        key_weight,
        gate_weight,
        ape,
        norm_weight,
        norm_bias,
        tail,
        ids,
        positions,
        query_lens,
        4,
        1e-6,
    )
    offset = 0
    for request_idx, length in enumerate(query_lens):
        expected = prepare_fused_kpool_states(
            hidden[offset : offset + length],
            key_weight,
            gate_weight,
            ape,
            norm_weight,
            norm_bias,
            tail,
            ids[request_idx : request_idx + 1],
            positions[offset : offset + length],
            [length],
            4,
            1e-6,
        )[0]
        if length:
            pool_count = (int(positions[offset].item()) % 4 + length) // 4
            torch.testing.assert_close(batched[request_idx][0][:pool_count], expected[0][:pool_count], rtol=0, atol=0)
            torch.testing.assert_close(batched[request_idx][1], expected[1], rtol=0, atol=0)
        else:
            assert batched[request_idx][0].shape == expected[0].shape == (0, 128)
        offset += length
    torch.testing.assert_close(tail, old_tail, rtol=0, atol=0)


@pytest.mark.parametrize("query_lens", [(1, 4095), (3, 61), (3, 3)])
@torch.inference_mode()
def test_varlen_fused_storage_matches_portable_cache_and_tail(query_lens: tuple[int, ...]) -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "key_pool"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    hidden = torch.randn(sum(query_lens), 4096, dtype=torch.bfloat16, device="npu")
    key_weight = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu") * 0.02
    gate_weight = torch.randn_like(key_weight) * 0.02
    ape = torch.randn(4, 128, device="npu")
    norm_weight = torch.randn(128, device="npu")
    norm_bias = torch.randn_like(norm_weight)
    tail = torch.randn(4, 2, 9, 128, device="npu")
    ids = torch.tensor([2, 1], dtype=torch.int32, device="npu")
    positions = torch.cat([torch.arange(3, 3 + length, device="npu") for length in query_lens])
    table = torch.tensor([[1], [2]], dtype=torch.int32, device="npu")
    cache = torch.zeros(3, 4097, 1, 128, dtype=torch.bfloat16, device="npu")
    valid = torch.ones(sum(query_lens), dtype=torch.bool, device="npu")
    valid[-2] = False
    tail_valid = torch.ones_like(valid)
    states = prepare_fused_kpool_states(
        hidden,
        key_weight,
        gate_weight,
        ape,
        norm_weight,
        norm_bias,
        tail,
        ids,
        positions,
        query_lens,
        4,
        1e-6,
    )
    raw = F.layer_norm(F.linear(hidden, key_weight).float(), (128,), norm_weight, norm_bias, 1e-6)
    gates = F.linear(hidden, gate_weight)
    write_ids = ids if query_lens[0] == 1 else torch.tensor([1, 3], dtype=torch.int32, device="npu")
    expected_cache, expected_tail = cache.clone(), tail.clone()
    _store_compressed_kpool_torch(
        raw,
        gates,
        valid,
        tail_valid,
        positions,
        expected_cache,
        expected_tail,
        ids,
        write_ids,
        table,
        query_lens,
        ape,
        4,
        fused_states=states,
    )
    update_compressed_kpool(
        raw,
        gates,
        valid,
        tail_valid,
        positions,
        cache,
        tail,
        ids,
        write_ids,
        table,
        query_lens,
        ape,
        4,
        fused_states=states,
    )
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(tail, expected_tail, rtol=0, atol=0)


@pytest.mark.parametrize("query_width", [1, 3, 8])
@pytest.mark.parametrize("restore_prefix", [False, True])
@pytest.mark.parametrize("mask_pool_member", [False, True])
@torch.inference_mode()
def test_fused_kpool_graph_replay_matches_eager_with_padding(
    query_width: int, restore_prefix: bool, mask_pool_member: bool
) -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "key_pool"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    batch = 3
    key_weight = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu") * 0.02
    gate_weight = torch.randn_like(key_weight) * 0.02
    ape = torch.randn(4, 128, device="npu")
    norm_weight = torch.randn(128, device="npu")
    norm_bias = torch.randn_like(norm_weight)
    hidden = torch.zeros(batch, query_width, 4096, dtype=torch.bfloat16, device="npu")
    positions = torch.full((batch, query_width), -1, device="npu", dtype=torch.int64)
    valid = torch.zeros(batch, query_width, device="npu", dtype=torch.bool)
    pool_valid = valid.clone()
    read_ids = torch.zeros(batch, device="npu", dtype=torch.int32)
    write_ids = torch.zeros_like(read_ids) if restore_prefix else read_ids
    cache = torch.zeros(4, 32, 1, 128, dtype=torch.bfloat16, device="npu")
    tail = torch.randn(8, 2, 11, 128, device="npu")
    table = torch.arange(1, 4, dtype=torch.int32, device="npu")[:, None]
    oracle_cache = cache.clone()
    oracle_tail = tail.clone()

    def step() -> None:
        states = prepare_fused_kpool_states(
            hidden,
            key_weight,
            gate_weight,
            ape,
            norm_weight,
            norm_bias,
            tail,
            read_ids,
            positions,
            [query_width] * batch,
            4,
            1e-6,
            valid_rows=valid,
        )
        raw = F.layer_norm(F.linear(hidden, key_weight).float(), (128,), norm_weight, norm_bias, 1e-6)
        gates = F.linear(hidden, gate_weight).float()
        update_compressed_kpool(
            raw,
            gates,
            pool_valid,
            valid,
            positions,
            cache,
            tail,
            read_ids,
            write_ids,
            table,
            [query_width] * batch,
            ape,
            4,
            graph_mode=True,
            fused_states=states,
        )

    step()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        step()
    lengths = [
        [query_width, query_width, 0],
        [1, max(1, query_width - 1), 0],
        [0, 0, 0],
        [query_width, 1, query_width],
    ]
    starts = [5, 7, 12]
    for iteration, counts in enumerate(lengths):
        hidden.normal_()
        positions.fill_(-1)
        valid.zero_()
        read_ids.zero_()
        write_ids.zero_()
        for request_idx, count in enumerate(counts):
            if not count:
                continue
            positions[request_idx, :count] = torch.arange(
                starts[request_idx], starts[request_idx] + count, device="npu"
            )
            valid[request_idx, :count] = True
            read_slot = request_idx + 1
            write_slot = request_idx + 4 if restore_prefix else read_slot
            read_ids[request_idx] = read_slot
            write_ids[request_idx] = write_slot
            inputs = hidden[request_idx, :count]
            request_positions = positions[request_idx, :count]
            request_read = read_ids[request_idx : request_idx + 1]
            request_write = write_ids[request_idx : request_idx + 1] if restore_prefix else request_read
            expected_states = prepare_fused_kpool_states(
                inputs,
                key_weight,
                gate_weight,
                ape,
                norm_weight,
                norm_bias,
                oracle_tail,
                request_read,
                request_positions,
                [count],
                4,
                1e-6,
            )
            raw = F.layer_norm(F.linear(inputs, key_weight).float(), (128,), norm_weight, norm_bias, 1e-6)
            gates = F.linear(inputs, gate_weight).float()
            mask = torch.ones(count, dtype=torch.bool, device="npu")
            pool_mask = mask.clone()
            if mask_pool_member and request_idx == 1:
                pool_mask[0] = False
            _store_compressed_kpool_torch(
                raw.reshape(-1, 128),
                gates.reshape(-1, 128),
                pool_mask,
                mask,
                request_positions,
                oracle_cache,
                oracle_tail,
                request_read,
                request_write,
                table[request_idx : request_idx + 1],
                [count],
                ape,
                4,
                fused_states=expected_states,
            )
            starts[request_idx] += count
        pool_valid.copy_(valid)
        if mask_pool_member:
            pool_valid[1, 0] = False
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(cache.cpu(), oracle_cache.cpu(), rtol=0, atol=0, msg=f"cache replay {iteration}")
        torch.testing.assert_close(tail.cpu(), oracle_tail.cpu(), rtol=0, atol=0, msg=f"tail replay {iteration}")
