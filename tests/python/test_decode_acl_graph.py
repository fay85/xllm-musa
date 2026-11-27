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

"""Tests for the NPU ACL decode-graph runner."""

import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python import distributed
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import _v41_compressor_swa_gate
from xllm.python.attention.csa2_block_metadata import (
    Csa2BlockAttentionMetadata,
    Csa2BlockAttentionMetadataBuilder,
)
from xllm.python.attention.dsa_metadata import (
    READ_TABLE_LAYOUT_RING,
    DsaMetadataBuilder,
    build_cache_specs,
    build_cache_specs_v41,
)
from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import ModelOutput
from xllm.python.model_executor.mtp_topk import MtpTopkReuse
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
    _CaptureAttentionMetadata,
    _DecodeGraphEntry,
)
from xllm.python.model_executor.sampling_metadata import (
    ModelSamplingMetadata,
    ModelSamplingMetadataBuilder,
    SamplingMode,
)
from xllm.python.models.engram_config import EngramConfig
from xllm.python.models.multimodal import MultimodalInputs


class _SamplingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sampling_builder = ModelSamplingMetadataBuilder()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return input_ids + positions

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        return inputs.input_batch is not None and self.sampling_builder.can_execute_graph(inputs.input_batch.sampling)


class _PassThroughDecoder(nn.Module):
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return input_ids


def _runner(model: nn.Module | None = None, max_batch: int = 8, **kwargs: Any) -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(
        page_size=4,
        is_mla=False,
        graph_uses_dsa_metadata=lambda _: False,
    )
    return DecodeAclGraphRunner(
        _PassThroughDecoder() if model is None else model,
        attention_backend,
        torch.device("cpu"),
        max_batch=max_batch,
        max_model_len=8,
        **kwargs,
    )


@pytest.mark.parametrize("lengths", [[], [32768, 32769], [2147483647, 1]])
def test_generated_cumulative_lengths_use_int32(lengths: list[int]) -> None:
    values = torch.tensor(lengths, dtype=torch.int32)
    actual = DecodeAclGraphRunner._cumulative_lengths(values, None)
    expected = torch.cat((torch.zeros(1, dtype=torch.int64), values.cumsum(0))).to(torch.int32)
    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_draft_consumes_aligned_effective_shapes() -> None:
    width = 3
    runner = _runner(
        is_spec_draft=True,
        num_speculative_tokens=width,
        model_type="DSparkDraftModel",
        capture_sizes=(width, 2 * width, 3 * width),
    )
    assert runner._padded_token_rows(2 * width - 1) == 2 * width
    assert runner._padded_token_rows(3 * width + 1) == 0
    assert runner._engram_graph_token_capacity() == 3 * width
    with pytest.raises(ValueError, match="aligned"):
        _runner(
            is_spec_draft=True, num_speculative_tokens=width, model_type="DSparkDraftModel", capture_sizes=(width + 1,)
        )


def test_custom_capture_selects_buckets_and_preserves_actual_sequence_capacity() -> None:
    runner = _runner(max_batch=10, capture_sizes=(3, 7, 10))
    for rows, bucket in ((1, 3), (3, 3), (4, 7), (7, 7), (8, 10), (10, 10), (11, 0)):
        assert runner._padded_token_rows(rows) == bucket
    assert runner._engram_graph_token_capacity() == 10
    metadata = _aligned_dp_decode_metadata(9)
    metadata.is_spec_verify = False
    metadata.linear_state_indices = None
    assert runner.can_execute(ModelInput(torch.arange(9), torch.arange(9), metadata))
    runner.warmup(torch.arange(9), torch.arange(9), metadata)
    assert not runner.can_execute(ModelInput(torch.arange(11), torch.arange(11), metadata))


@pytest.mark.parametrize("counts,admitted,bucket", [((9, 0), True, 10), ((11, 0), False, 0), ((7, 3), True, 7)])
def test_custom_capture_dp_ranks_choose_same_shape_and_admission(
    counts: tuple[int, int], admitted: bool, bucket: int
) -> None:
    for rank in range(2):
        runner = _runner(max_batch=16, dp_size=2, dp_rank=rank, capture_sizes=(3, 7, 10))
        rows = max(1, counts[rank])
        metadata = _aligned_dp_decode_metadata(rows)
        metadata.is_spec_verify = False
        metadata.linear_state_indices = None
        metadata.dp_execution_token_counts = tuple(max(1, count) for count in counts)
        metadata.raw_dp_execution_token_counts = counts
        metadata.is_dummy = counts[rank] == 0
        metadata.block_table[:, 0] = torch.arange(1, rows + 1) if counts[rank] else 0
        metadata.paged_kv_indices = metadata.block_table[:, 0].clone()
        metadata.slot_mapping = metadata.block_table[:, 0] * 4
        if not counts[rank]:
            metadata.slot_mapping.fill_(-1)
        input_ids = torch.arange(rows)
        assert runner.can_execute(ModelInput(input_ids, input_ids, metadata)) is admitted
        assert runner._padded_token_rows(max(metadata.dp_execution_token_counts)) == bucket


def _query_batch(num_reqs: int, width: int) -> InputBatch:
    num_tokens = max(1, num_reqs) * width
    return InputBatch.from_runtime(
        torch.arange(num_tokens),
        torch.arange(num_tokens, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            num_scheduled_tokens=[width] * num_reqs,
            num_computed_tokens=[],
            query_start_loc=list(range(0, num_reqs * width + 1, width)),
            is_prefilling=[False] * num_reqs,
        ),
        is_dummy=num_reqs == 0,
    )


@pytest.mark.parametrize(("query_width", "num_sequences"), [(3, 1), (3, 5), (5, 2), (15, 3)])
def test_draft_graph_uses_same_whole_block_plan_for_active_and_empty_dp_shards(
    query_width: int,
    num_sequences: int,
) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False),
        torch.device("cpu"),
        max_batch=16,
        max_model_len=256,
        dp_size=2,
        is_spec_draft=True,
        num_speculative_tokens=query_width,
        model_type="DSparkDraftModel",
    )
    num_tokens = num_sequences * query_width
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=(num_tokens, query_width),
        raw_dp_execution_token_counts=(num_tokens, 0),
        dp_is_decode=(True, True),
    )
    assert runner._shared_graph_plan(metadata, num_tokens) == (False, 1, num_sequences)
    runner.dp_rank = 1
    assert runner._shared_graph_plan(metadata, query_width) == (False, 1, num_sequences)
    padded_rows = runner._padded_token_rows(max(metadata.dp_execution_token_counts))
    assert padded_rows >= num_tokens and padded_rows % query_width == 0


def test_draft_graph_communication_limit_uses_complete_block_padding() -> None:
    rejected = _runner(
        is_spec_draft=True, num_speculative_tokens=5, model_type="DSparkDraftModel", eplv2_graph_token_limit=16
    )
    assert rejected._padded_token_rows(15) == 20
    assert not rejected._eplv2_graph_admits(15)
    allowed = _runner(
        is_spec_draft=True, num_speculative_tokens=5, model_type="DSparkDraftModel", eplv2_graph_token_limit=20
    )
    assert allowed._eplv2_graph_admits(15)


@pytest.mark.parametrize("query_width", [3, 5, 15])
def test_expanded_draft_proposal_has_shared_plan_with_empty_dp(query_width: int) -> None:
    plans = []
    for dp_rank, num_reqs in ((0, 2), (1, 0)):
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            SimpleNamespace(page_size=128, is_mla=False),
            torch.device("cpu"),
            max_batch=8,
            max_model_len=256,
            dp_size=2,
            dp_rank=dp_rank,
            is_spec_draft=True,
            num_speculative_tokens=query_width,
            model_type="DSparkDraftModel",
        )
        batch = _query_batch(num_reqs, query_width)
        row_count = batch.num_tokens if num_reqs else 1
        metadata = _metadata(None if num_reqs else torch.zeros(1, dtype=torch.int32))
        metadata.is_chunked_prefill = True
        metadata.use_dflash_proposal_attention = True
        metadata.is_dummy = batch.is_dummy
        metadata.num_accepted_tokens = None
        metadata.block_table = torch.zeros((row_count, query_width), dtype=torch.int32)
        metadata.kv_seq_lens = torch.full((row_count,), query_width, dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [query_width] * row_count
        metadata.q_cu_seq_lens = torch.arange(row_count + 1, dtype=torch.int32) * (1 if num_reqs else query_width)
        metadata.kv_cu_seq_lens = torch.arange(row_count + 1, dtype=torch.int32) * query_width
        metadata.slot_mapping = torch.full((batch.num_tokens,), -1, dtype=torch.int32)
        metadata.dp_execution_token_counts = (2 * query_width, query_width)
        metadata.raw_dp_execution_token_counts = (2 * query_width, 0)
        metadata.dp_is_decode = (True, True)
        assert runner.can_execute(ModelInput(batch.input_ids, batch.input_ids, metadata, input_batch=batch))
        plans.append(runner._shared_graph_plan(metadata, batch.num_tokens))
    assert plans == [(True, query_width, 2)] * 2


@pytest.mark.parametrize(
    ("dp_size", "context_length", "extra_target_tables"),
    [(1, 256, True), (2, 1024, False), (4, 1025, True), (1, 8192, False), (4, 8192, True)],
)
def test_csa2_ring_paging_admission_and_dummy_replay(
    dp_size: int,
    context_length: int,
    extra_target_tables: bool,
) -> None:
    query_width, page_size = 4, 128
    _, group_infos = build_cache_specs_v41([0], [], [], page_size, 1)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=page_size, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8192,
        dp_size=dp_size,
        is_spec_draft=True,
        num_speculative_tokens=query_width,
        model_type="DSparkDraftModel",
    )
    builder = Csa2BlockAttentionMetadataBuilder(query_width, page_size, 8192)
    runner.bind_execution_metadata_builders((builder,))
    batch = replace(_query_batch(1, query_width), positions=torch.arange(context_length - query_width, context_length))
    metadata = _metadata(None)
    metadata.use_dflash_proposal_attention = True
    metadata.is_dummy = False
    metadata.num_accepted_tokens = None
    metadata.block_table = None
    metadata.multi_block_tables = (torch.tensor([[2, 5]] * query_width, dtype=torch.int32),)
    if extra_target_tables:
        metadata.multi_block_tables += (
            torch.full((query_width, 1), 31, dtype=torch.int32),
            torch.full((query_width, 1), 41, dtype=torch.int32),
        )
    metadata.kv_seq_lens = torch.full((query_width,), context_length, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [context_length] * query_width
    metadata.kv_cu_seq_lens = torch.arange(query_width + 1, dtype=torch.int32) * context_length
    page_count = (context_length + page_size - 1) // page_size
    metadata.paged_kv_indptr = torch.arange(query_width + 1, dtype=torch.int32) * page_count
    row_pages = [2 if page % 2 == 0 else 5 for page in range(page_count)]
    metadata.paged_kv_indices = torch.tensor(row_pages * query_width, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.full(
        (query_width,), (context_length - 1) % page_size + 1, dtype=torch.int32
    )
    block_ids = torch.tensor([2, 5], dtype=torch.int32)[batch.positions // page_size % 2]
    metadata.slot_mapping = block_ids * page_size + batch.positions % page_size
    metadata.new_cache_slots_host_values = metadata.slot_mapping.tolist()
    metadata.dp_execution_token_counts = (query_width,) * dp_size
    metadata.raw_dp_execution_token_counts = (query_width,) + (0,) * (dp_size - 1)
    metadata.dp_is_decode = (True,) * dp_size
    assert runner.can_execute(ModelInput(batch.input_ids, batch.input_ids, metadata, input_batch=batch))
    normalized = runner._normalize_dsa_metadata(metadata)
    assert len(normalized.multi_block_tables) == 1
    assert normalized.multi_block_tables[0] is metadata.multi_block_tables[0]
    assert len(metadata.multi_block_tables) == (3 if extra_target_tables else 1)
    if extra_target_tables:
        target_runner = DecodeAclGraphRunner(
            nn.Identity(),
            SimpleNamespace(page_size=page_size, is_mla=False, group_infos=group_infos),
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8192,
            dp_size=dp_size,
        )
        assert target_runner._normalize_dsa_metadata(metadata) is metadata
    entry = runner._allocate_entry(8, batch.input_ids, batch.positions, normalized, batch)
    target = entry.execution_contexts[Csa2BlockAttentionMetadata]
    addresses = tuple(tensor.data_ptr() for tensor in (target.context_lengths, target.block_table, target.slot_mapping))

    empty_batch = _query_batch(0, query_width)
    dummy = SimpleNamespace(**vars(metadata))
    dummy.is_dummy = True
    dummy.linear_state_indices = torch.zeros(query_width, dtype=torch.int32)
    dummy.block_table = torch.zeros((query_width, 1), dtype=torch.int32)
    dummy.multi_block_tables = ()
    dummy.kv_seq_lens = torch.ones(query_width, dtype=torch.int32)
    dummy.kv_seq_lens_host_values = [1] * query_width
    dummy.kv_cu_seq_lens = torch.arange(query_width + 1, dtype=torch.int32)
    dummy.slot_mapping = torch.full((query_width,), -1, dtype=torch.int32)
    dummy.new_cache_slots_host_values = [-1] * query_width
    dummy.paged_kv_indptr = torch.arange(query_width + 1, dtype=torch.int32)
    dummy.paged_kv_indices = torch.zeros(query_width, dtype=torch.int32)
    dummy.paged_kv_last_page_len = torch.ones(query_width, dtype=torch.int32)
    dummy.raw_dp_execution_token_counts = ((0, query_width) + (0,) * (dp_size - 2)) if dp_size > 1 else (0,)
    assert runner.can_execute(ModelInput(empty_batch.input_ids, empty_batch.input_ids, dummy, input_batch=empty_batch))
    assert runner._shared_graph_plan(metadata, query_width) == runner._shared_graph_plan(dummy, query_width)
    normalized_dummy = runner._normalize_dsa_metadata(dummy)
    assert normalized_dummy.multi_block_tables[0].shape == (query_width, 1)
    assert builder.build(empty_batch, dummy).slot_mapping.tolist() == [-1] * query_width

    def copy_legacy_inputs(*args: Any) -> None:
        for source, destination in ((args[0], args[7]), (args[1], args[8]), (args[2], args[9])):
            destination.zero_()
            destination[: source.numel()].copy_(source)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        side_effect=copy_legacy_inputs,
        create=True,
    ):
        for current_metadata, current_batch, expected_lengths in (
            (normalized, batch, [context_length, 0]),
            (normalized_dummy, empty_batch, [0, 0]),
            (normalized, batch, [context_length, 0]),
        ):
            runner._fill_entry(
                entry,
                current_batch.input_ids,
                current_batch.positions,
                current_metadata,
                query_width,
                None,
                current_batch,
            )
            assert target.context_lengths.tolist() == expected_lengths
            if current_batch.is_dummy:
                assert target.slot_mapping.tolist() == [-1] * 8
                assert not torch.count_nonzero(target.block_table)
    assert (
        tuple(tensor.data_ptr() for tensor in (target.context_lengths, target.block_table, target.slot_mapping))
        == addresses
    )


@pytest.mark.parametrize("dp_size", [1, 2])
def test_ordinary_paging_still_rejects_insufficient_history_capacity(dp_size: int) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=2048,
        dp_size=dp_size,
    )
    metadata = _metadata(None)
    metadata.kv_seq_lens = torch.full((4,), 1024, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [1024] * 4
    metadata.paged_kv_indptr = torch.tensor([0], dtype=torch.int32)
    metadata.paged_kv_indices = torch.empty(0, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.empty(0, dtype=torch.int32)
    metadata.dp_execution_token_counts = (4,) * dp_size
    metadata.dp_is_decode = (True,) * dp_size
    if dp_size == 1:
        assert not runner.can_execute(ModelInput(torch.arange(4), torch.arange(4), metadata))
    else:
        with pytest.raises(RuntimeError, match="cannot hold the KV length"):
            runner.can_execute(ModelInput(torch.arange(4), torch.arange(4), metadata))


def test_python_mtp_selection_survives_graph_output_reuse_and_resets_per_iteration() -> None:
    reuse = MtpTopkReuse()
    source = torch.arange(24, dtype=torch.int32).reshape(4, 1, 6)
    assert reuse.input_for_step(0, 4) is None
    reuse.finish_step(0, source, torch.tensor([1, 3], dtype=torch.int32))
    selected = reuse.input_for_step(1, 2)
    expected = source[[1, 3]].clone()
    source.fill_(-1)
    torch.testing.assert_close(selected, expected)
    assert DecodeAclGraphRunner._graph_key(2, False, None) != DecodeAclGraphRunner._graph_key(
        2, False, None, mtp_topk_indices=selected
    )
    reuse.finish_step(1, selected, None)
    assert reuse.input_for_step(2, 2) is selected
    assert reuse.input_for_step(0, 1) is None
    with pytest.raises(ValueError, match="consecutive"):
        reuse.input_for_step(1, 1)


def test_draft_graph_key_keeps_whole_query_blocks() -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=256,
        is_spec_draft=True,
        num_speculative_tokens=5,
        model_type="DSparkDraftModel",
    )
    captured_keys: list[tuple[Any, ...]] = []

    class ProbeCache(dict):
        def get(self, key: tuple[Any, ...], default: Any = None) -> Any:
            captured_keys.append(key)
            raise LookupError("graph key recorded")

    runner._graphs = ProbeCache()
    for num_sequences in (2, 3):
        batch = _query_batch(num_sequences, 5)
        metadata = _aligned_dp_decode_metadata(batch.num_tokens)
        metadata.is_spec_verify = False
        metadata.linear_state_indices = None
        metadata.block_table[:, 0] = torch.arange(1, batch.num_tokens + 1)
        metadata.kv_seq_lens = torch.ones(batch.num_tokens, dtype=torch.int32)
        metadata.slot_mapping = metadata.block_table[:, 0] * 128
        metadata.paged_kv_indices = metadata.block_table[:, 0].clone()
        with pytest.raises(LookupError, match="graph key recorded"):
            runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)
    assert [key[0] for key in captured_keys] == [10, 20]


@pytest.mark.parametrize(
    ("is_spec_draft", "num_speculative_tokens", "expected_capacity"),
    [(False, 0, 16), (False, 2, 48), (True, 2, 32)],
)
def test_legacy_paged_buffer_covers_target_and_draft_token_rows(
    is_spec_draft: bool,
    num_speculative_tokens: int,
    expected_capacity: int,
) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        num_speculative_tokens=num_speculative_tokens,
        is_spec_draft=is_spec_draft,
        model_type="DSparkDraftModel" if is_spec_draft else "",
    )
    input_ids = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(input_ids)
    entry = runner._allocate_entry(4, input_ids, input_ids, metadata)
    assert runner._paged_kv_indices_buffer.numel() == expected_capacity
    assert entry.static_metadata.paged_kv_indices is runner._paged_kv_indices_buffer


def test_csa2_builder_updates_alongside_legacy_buffers_and_protects_capture() -> None:
    _, group_infos = build_cache_specs_v41([0], [], [], 4, 1)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        is_spec_draft=True,
        num_speculative_tokens=2,
        model_type="DSparkDraftModel",
    )
    runner.bind_execution_metadata_builders((Csa2BlockAttentionMetadataBuilder(2, 4, 8),))
    batch = _query_batch(2, 2)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.linear_state_indices = None
    metadata.num_accepted_tokens = None
    metadata.is_dummy = False
    metadata.multi_block_tables = (torch.tensor([[5], [5], [6], [6]], dtype=torch.int32),)
    metadata.block_table = torch.tensor([[5, 0], [5, 0], [6, 0], [6, 0]], dtype=torch.int32)
    metadata.paged_kv_indices = torch.tensor([5, 5, 6, 6], dtype=torch.int32)
    metadata.slot_mapping = torch.tensor([20, 21, 26, 27], dtype=torch.int32)
    metadata.new_cache_slots_host_values = [20, 21, 26, 27]
    entry = runner._allocate_entry(8, batch.input_ids, batch.positions, metadata, batch)
    target = entry.execution_contexts[Csa2BlockAttentionMetadata]
    buffers = (target.context_lengths, target.block_table, target.slot_mapping)
    addresses = tuple(tensor.data_ptr() for tensor in buffers)
    assert entry.static_metadata.paged_kv_indices is runner._paged_kv_indices_buffer
    assert entry.static_metadata.slot_mapping.data_ptr() != target.slot_mapping.data_ptr()
    assert entry.static_metadata.multi_block_tables[0].data_ptr() != target.block_table.data_ptr()

    def copy_legacy_inputs(*args: Any) -> None:
        for source, destination in ((args[0], args[7]), (args[1], args[8]), (args[2], args[9])):
            destination.zero_()
            destination[: source.numel()].copy_(source)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        side_effect=copy_legacy_inputs,
        create=True,
    ) as legacy_update:
        runner._fill_entry(entry, batch.input_ids, batch.positions, _CaptureAttentionMetadata(metadata), 4, None, batch)
        assert target.context_lengths.tolist() == [0, 0, 0, 0]
        assert target.block_table.tolist() == [[0, 0]] * 4
        assert target.slot_mapping.tolist() == [-1] * 8
        assert not batch.is_dummy
        assert not entry.static_metadata.is_dummy

        runner._fill_entry(entry, batch.input_ids, batch.positions, metadata, 4, None, batch)
        assert legacy_update.call_count == 2

    assert target.query_lens == (2, 2, 2, 2)
    assert target.context_lengths.tolist() == [2, 4, 0, 0]
    assert target.block_table.tolist() == [[5, 5], [6, 6], [0, 0], [0, 0]]
    assert target.slot_mapping.tolist() == [20, 21, 26, 27, -1, -1, -1, -1]
    current_buffers = (target.context_lengths, target.block_table, target.slot_mapping)
    assert tuple(tensor.data_ptr() for tensor in current_buffers) == addresses
    assert entry.execution_contexts[Csa2BlockAttentionMetadata] is target
    assert entry.static_input_ids.tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
    assert entry.static_positions.tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
    assert entry.static_metadata.slot_mapping.tolist() == [20, 21, 26, 27, -1, -1, -1, -1]
    assert entry.static_metadata.kv_seq_lens_host_values == [1, 2, 3, 4, 1, 1, 1, 1]
    assert entry.static_metadata.multi_block_tables[0].tolist() == [
        [5, 5],
        [5, 5],
        [6, 6],
        [6, 6],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]


def _metadata(linear_state_indices: torch.Tensor | None) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        paged_kv_indptr=torch.arange(5, dtype=torch.int32),
        paged_kv_indices=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        paged_kv_last_page_len=torch.arange(1, 5, dtype=torch.int32),
        block_table=torch.tensor(
            [[10, 0], [20, 0], [30, 0], [40, 0]],
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.arange(1, 5, dtype=torch.int32),
        kv_seq_lens_host_values=[1, 2, 3, 4],
        kv_cu_seq_lens=torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32),
        q_cu_seq_lens=torch.arange(5, dtype=torch.int32),
        linear_state_indices=linear_state_indices,
        num_accepted_tokens=torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        expanded_decode_metadata=None,
        multi_block_tables=(),
        new_cache_slots_host_values=[0, 1, 2, 3],
        is_prefill=False,
        is_chunked_prefill=False,
    )


@pytest.mark.parametrize("sampling_mode", ["greedy", "non_greedy", "probabilities", "confidence", "missing"])
def test_sampling_graph_admission_checks_request_capability(sampling_mode: str) -> None:
    runner = _runner(_SamplingModel())
    runner.bind_execution_metadata_builders((ModelSamplingMetadataBuilder(),))
    sampling = SimpleNamespace(
        is_unconstrained_greedy=sampling_mode != "non_greedy",
        return_draft_logits=sampling_mode == "probabilities",
        return_confidence=sampling_mode == "confidence",
    )
    batch = replace(_query_batch(4, 1), sampling=None if sampling_mode == "missing" else sampling)
    metadata = _metadata(None)
    metadata.block_table[:, 0] = torch.arange(1, 5)
    metadata.paged_kv_indices = metadata.block_table[:, 0].clone()
    metadata.slot_mapping = metadata.block_table[:, 0] * 4 + batch.positions
    metadata.new_cache_slots_host_values = metadata.slot_mapping.tolist()

    assert runner.can_execute(ModelInput(batch.input_ids, batch.positions, metadata, input_batch=batch)) == (
        sampling_mode == "greedy"
    )


@pytest.mark.parametrize(
    ("empty_peer", "sampling_allowed", "layout_allowed"),
    [
        (True, (True, True), (True, True)),
        (False, (False, True), (True, True)),
        (True, (True, False), (True, True)),
        (False, (True, True), (False, True)),
    ],
)
def test_sampling_graph_admission_is_unanimous_across_dp(
    monkeypatch: pytest.MonkeyPatch,
    empty_peer: bool,
    sampling_allowed: tuple[bool, bool],
    layout_allowed: tuple[bool, bool],
) -> None:
    votes = [sampling and layout for sampling, layout in zip(sampling_allowed, layout_allowed, strict=True)]
    calls: list[int] = []

    def agree(allowed: bool, device: torch.device, world_size: int) -> bool:
        assert device == torch.device("cpu") and world_size == 2
        assert allowed is votes[rank]
        calls.append(rank)
        return all(votes)

    monkeypatch.setattr(distributed, "dp_all_true", agree, raising=False)
    counts = (4, 1) if empty_peer else (4, 4)
    for rank in range(2):
        runner = _runner(_SamplingModel(), dp_size=2, dp_rank=rank)
        runner.bind_execution_metadata_builders((ModelSamplingMetadataBuilder(),))
        is_dummy = empty_peer and rank == 1
        sampling = SimpleNamespace(
            is_unconstrained_greedy=sampling_allowed[rank], return_draft_logits=False, return_confidence=False
        )
        batch = _query_batch(0 if is_dummy else 4, 1)
        batch = replace(batch, sampling=sampling, positions=torch.zeros(batch.num_tokens, dtype=torch.int32))
        metadata = _aligned_dp_decode_metadata(batch.num_tokens)
        metadata.is_spec_verify = False
        metadata.linear_state_indices = None
        metadata.is_dummy = is_dummy
        metadata.dp_execution_token_counts = counts
        metadata.raw_dp_execution_token_counts = (4, 0) if empty_peer else counts
        metadata.block_table[:, 0] = 0 if is_dummy else torch.arange(1, batch.num_tokens + 1)
        metadata.paged_kv_indices = metadata.block_table[:, 0].clone()
        metadata.slot_mapping = metadata.block_table[:, 0] * 4
        if is_dummy:
            metadata.slot_mapping.fill_(-1)
        multimodal = None if layout_allowed[rank] else MultimodalInputs(torch.zeros(batch.num_tokens, 1))
        inputs = ModelInput(batch.input_ids, batch.positions, metadata, input_batch=batch, multimodal_inputs=multimodal)
        assert runner.can_execute(inputs) is all(votes)
    assert calls == [0, 1]


def test_sampling_graph_admission_propagates_dp_agreement_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(_SamplingModel(), dp_size=2)
    runner.bind_execution_metadata_builders((ModelSamplingMetadataBuilder(),))
    monkeypatch.setattr(
        distributed, "dp_all_true", MagicMock(side_effect=RuntimeError("DP host agreement failed")), raising=False
    )
    sampling = SimpleNamespace(is_unconstrained_greedy=True, return_draft_logits=False, return_confidence=False)
    batch = replace(_query_batch(4, 1), sampling=sampling, positions=torch.zeros(4, dtype=torch.int32))
    metadata = _aligned_dp_decode_metadata(4)
    metadata.is_spec_verify = False
    metadata.linear_state_indices = None
    with pytest.raises(RuntimeError, match="DP host agreement failed"):
        runner.can_execute(ModelInput(batch.input_ids, batch.positions, metadata, input_batch=batch))


def test_sampling_graph_admission_does_not_gate_models_without_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(dp_size=2)
    agreement = MagicMock(side_effect=AssertionError("unexpected DP admission collective"))
    monkeypatch.setattr(distributed, "dp_all_true", agreement, raising=False)
    metadata = _aligned_dp_decode_metadata(4)
    metadata.is_spec_verify = False
    metadata.linear_state_indices = None
    assert runner.can_execute(ModelInput(torch.arange(4), torch.zeros(4, dtype=torch.int32), metadata))
    agreement.assert_not_called()


def test_sampling_graph_metadata_does_not_require_host_cached_counts() -> None:
    runner = _runner()
    builder = ModelSamplingMetadataBuilder()
    runner.bind_execution_metadata_builders((builder,))

    first_sampling = SimpleNamespace(
        parameters=SimpleNamespace(sampling_mode=SamplingMode.GREEDY),
        is_unconstrained_greedy=True,
        return_draft_logits=False,
        return_confidence=False,
    )
    next_sampling = SimpleNamespace(
        parameters=SimpleNamespace(sampling_mode=SamplingMode.GREEDY),
        is_unconstrained_greedy=True,
        return_draft_logits=False,
        return_confidence=False,
    )
    input_ids = torch.arange(4)
    positions = torch.arange(4)
    metadata = _metadata(None)
    metadata.block_table = torch.tensor([[5, 0], [5, 0], [6, 0], [6, 0]], dtype=torch.int32)
    metadata.paged_kv_indices = torch.tensor([5, 5, 6, 6], dtype=torch.int32)
    metadata.slot_mapping = torch.tensor([20, 21, 26, 27], dtype=torch.int32)
    metadata.new_cache_slots_host_values = metadata.slot_mapping.tolist()
    metadata.num_accepted_tokens = None
    batch_metadata = SimpleNamespace(
        num_reqs=2,
        num_tokens=4,
        num_scheduled_tokens=[2, 2],
        num_computed_tokens=[],
        query_start_loc=[0, 2, 4],
        is_prefilling=[False, False],
    )
    input_batch = InputBatch.from_runtime(
        input_ids,
        positions,
        batch_metadata,
        is_dummy=False,
        sampling=first_sampling,
    )

    def copy_inputs(*args: Any) -> None:
        for source, destination in ((args[0], args[7]), (args[1], args[8]), (args[2], args[9])):
            destination.zero_()
            destination[: source.numel()].copy_(source)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        side_effect=copy_inputs,
        create=True,
    ):
        entry = runner._allocate_entry(8, input_ids, positions, metadata, input_batch)
        persistent = entry.execution_contexts[ModelSamplingMetadata]
        assert persistent.sampling is first_sampling
        input_address = entry.static_input_ids.data_ptr()

        for sampling in (first_sampling, next_sampling):
            input_batch = InputBatch.from_runtime(
                input_ids,
                positions,
                batch_metadata,
                is_dummy=False,
                sampling=sampling,
            )
            runner._fill_entry(
                entry,
                input_ids,
                positions,
                metadata,
                batch_size=4,
                input_embedding=None,
                input_batch=input_batch,
            )
            assert entry.execution_contexts[ModelSamplingMetadata] is persistent
            assert persistent.sampling is sampling
            assert persistent.num_tokens == 4
            assert entry.static_input_ids.data_ptr() == input_address
            assert entry.static_input_ids.tolist() == [0, 1, 2, 3, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "limit,sequences,expected",
    [(16, 1, True), (16, 2, True), (16, 3, False), (24, 3, False), (32, 3, True), (32, 4, True), (32, 5, False)],
)
def test_eplv2_graph_admission_uses_padded_token_rows_for_dflash2(limit: int, sequences: int, expected: bool) -> None:
    runner = _runner(max_batch=128, num_speculative_tokens=7, eplv2_graph_token_limit=limit)
    rows = sequences * 8
    metadata = _aligned_dp_decode_metadata(rows)
    metadata.linear_state_indices = torch.arange(sequences, dtype=torch.int32)
    metadata.is_spec_verify = True
    metadata.is_chunked_prefill = True
    metadata.q_cu_seq_lens = torch.arange(sequences + 1, dtype=torch.int32) * 8
    metadata.block_table = torch.arange(1, sequences * 2 + 1, dtype=torch.int32).view(sequences, 2)
    metadata.kv_seq_lens = torch.full((sequences,), 8, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [8] * sequences
    positions = torch.arange(8, dtype=torch.int32).repeat(sequences)
    tables = metadata.block_table.repeat_interleave(8, dim=0)
    lengths = positions + 1
    metadata.slot_mapping = tables.gather(1, (positions // 4).long().unsqueeze(1)).flatten() * 4 + positions % 4
    metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
        kv_seq_lens=lengths,
        block_table=tables,
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=lengths.clone(),
        kv_seq_lens_host_values=lengths.tolist(),
    )
    assert runner.can_execute(ModelInput(torch.arange(rows), positions, metadata)) is expected


def test_full_attention_graph_ignores_accepted_token_metadata() -> None:
    runner = _runner(max_batch=10, is_spec_draft=True, capture_sizes=(3, 7, 10))
    runner.bind_layer_caches([LayerCache(torch.zeros(64, 4, 1, 1), torch.zeros(64, 4, 1, 1))])
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.zeros(4, dtype=torch.int32)
    metadata = _aligned_dp_decode_metadata(4)
    metadata.is_spec_verify = False
    metadata.linear_state_indices = None
    accepted = torch.ones(4, dtype=torch.int32)
    metadata.num_accepted_tokens = accepted
    metadata.is_dummy = True
    metadata.slot_mapping.fill_(-1)
    entry = runner._allocate_entry(7, input_ids, positions, metadata)
    assert entry.static_metadata.num_accepted_tokens is None
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for is_dummy, counts in ((True, accepted), (False, None), (False, accepted)):
            metadata.is_dummy = is_dummy
            metadata.num_accepted_tokens = counts
            metadata.block_table[:, 0] = 0 if is_dummy else torch.arange(1, 5)
            metadata.paged_kv_indices = metadata.block_table[:, 0].clone()
            metadata.slot_mapping = metadata.block_table[:, 0] * 4
            if is_dummy:
                metadata.slot_mapping.fill_(-1)
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
            assert entry.static_metadata.num_accepted_tokens is None


def _verify_batch_metadata(num_reqs: int, width: int) -> tuple[InputBatch, SimpleNamespace]:
    batch = _query_batch(num_reqs, width)
    positions = torch.arange(width, dtype=torch.int32).repeat(num_reqs)
    batch = replace(batch, positions=positions)
    metadata = _metadata(torch.arange(1, num_reqs + 1, dtype=torch.int32))
    metadata.is_spec_verify = True
    metadata.is_chunked_prefill = True
    metadata.q_cu_seq_lens = torch.arange(num_reqs + 1, dtype=torch.int32) * width
    metadata.kv_cu_seq_lens = torch.arange(num_reqs + 1, dtype=torch.int32) * width
    metadata.kv_seq_lens = torch.full((num_reqs,), width, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [width] * num_reqs
    metadata.block_table = torch.arange(1, num_reqs * 2 + 1, dtype=torch.int32).view(num_reqs, 2)
    tables = metadata.block_table.repeat_interleave(width, dim=0)
    lengths = positions + 1
    metadata.slot_mapping = tables.gather(1, (positions // 4).long().unsqueeze(1)).flatten() * 4 + positions % 4
    metadata.new_cache_slots_host_values = metadata.slot_mapping.tolist()
    metadata.num_accepted_tokens = torch.full((num_reqs,), width, dtype=torch.int32)
    metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
        kv_seq_lens=lengths,
        block_table=tables,
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=lengths.clone(),
        kv_seq_lens_host_values=lengths.tolist(),
    )
    return batch, metadata


def test_accepted_tokens_use_live_per_sequence_graph_buffer() -> None:
    runner = _runner(num_speculative_tokens=1)
    runner.bind_layer_caches([LayerCache(None, None, conv=torch.zeros(8, 1, 2), ssm=torch.zeros(16, 1, 1, 1))])
    batch, metadata = _verify_batch_metadata(2, 2)
    entry = runner._allocate_entry(8, batch.input_ids, batch.positions, metadata)
    static_counts = entry.static_metadata.num_accepted_tokens
    address = static_counts.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for sequences, counts in ((2, [2, 1]), (2, [1, 2]), (1, [2])):
            batch, metadata = _verify_batch_metadata(sequences, 2)
            metadata.num_accepted_tokens = torch.tensor(counts, dtype=torch.int32)
            runner._fill_entry(entry, batch.input_ids, batch.positions, metadata, batch.num_tokens, None)
            assert static_counts.tolist() == counts + [1] * (8 - sequences)
            assert static_counts.data_ptr() == address
        dummy = _aligned_dp_decode_metadata(1)
        dummy.is_dummy = True
        dummy.slot_mapping.fill_(-1)
        dummy.num_accepted_tokens = None
        runner._fill_entry(entry, torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), dummy, 1, None)
    assert static_counts.data_ptr() == address
    assert static_counts.tolist() == [1] * 8


def test_host_dsa_fill_preserves_live_inputs_without_generic_paging(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = DecodeAclGraphRunner(
        _PassThroughDecoder(),
        SimpleNamespace(
            page_size=128, is_mla=False, supports_draft_to_target_overlap=True, graph_uses_dsa_metadata=lambda _: True
        ),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=16384,
    )

    def source(rows: int, dummy: bool = False) -> SimpleNamespace:
        metadata = _aligned_dp_decode_metadata(rows)
        metadata.is_dummy = dummy
        metadata.is_spec_verify = False
        metadata.linear_state_indices = None
        metadata.num_accepted_tokens = None
        metadata.kv_seq_lens_host_values = [1] * rows if dummy else [127, 128, 8193, 8194][:rows]
        metadata.kv_seq_lens = torch.tensor(metadata.kv_seq_lens_host_values, dtype=torch.int32)
        metadata.block_table = (
            torch.zeros((rows, 65), dtype=torch.int32)
            if dummy
            else torch.arange(1, rows * 65 + 1, dtype=torch.int32).view(rows, 65)
        )
        positions = metadata.kv_seq_lens - 1
        metadata.slot_mapping = (
            torch.full((rows,), -1, dtype=torch.int32)
            if dummy
            else metadata.block_table.gather(1, (positions // 128).long().unsqueeze(1)).flatten() * 128
            + positions % 128
        )
        metadata.new_cache_slots_host_values = metadata.slot_mapping.tolist()
        metadata.paged_kv_indptr, metadata.paged_kv_indices, metadata.paged_kv_last_page_len = (
            runner._build_row_aligned_paged_kv_metadata(metadata.block_table, metadata.kv_seq_lens_host_values)
        )
        return metadata

    initial = source(4)
    entry = runner._allocate_entry(8, torch.arange(4), initial.kv_seq_lens - 1, initial)
    monkeypatch.setattr(runner, "_decode_metadata", lambda *_: pytest.fail("unused paged metadata was rebuilt"))
    addresses = (entry.static_input_ids.data_ptr(), entry.static_positions.data_ptr())
    paging = entry.static_metadata.paged_kv_indices.clone()
    for rows, dummy in ((4, False), (2, False), (1, True), (4, False)):
        metadata = source(rows, dummy)
        ids = torch.zeros(rows, dtype=torch.int64) if dummy else torch.arange(rows) + 100
        positions = metadata.kv_seq_lens - 1
        runner._fill_entry(entry, ids, positions, metadata, rows, None)
        assert (entry.static_input_ids.data_ptr(), entry.static_positions.data_ptr()) == addresses
        assert torch.equal(entry.static_input_ids[:rows], ids)
        assert torch.equal(entry.static_positions[:rows], positions)
        assert torch.equal(entry.static_metadata.dsa_positions[:rows], positions.to(torch.int64))
        assert not entry.static_input_ids[rows:].any()
        assert not entry.static_positions[rows:].any()
        assert torch.equal(entry.static_metadata.slot_mapping[:rows], metadata.slot_mapping)
        assert torch.all(entry.static_metadata.slot_mapping[rows:] == -1)
        assert entry.static_metadata.kv_seq_lens_host_values == metadata.kv_seq_lens_host_values + [1] * (8 - rows)
        assert torch.equal(entry.static_metadata.paged_kv_indices, paging)


def test_verify_graph_key_tracks_width_not_active_sequence_count() -> None:
    runner = _runner(num_speculative_tokens=3)
    captured_keys: list[tuple[Any, ...]] = []

    class ProbeCache(dict):
        def get(self, key: tuple[Any, ...], default: Any = None) -> Any:
            captured_keys.append(key)
            raise LookupError("graph key recorded")

    runner._graphs = ProbeCache()
    for sequence_count, width in ((3, 4), (4, 4), (8, 2)):
        batch, metadata = _verify_batch_metadata(sequence_count, width)
        with pytest.raises(LookupError, match="graph key recorded"):
            runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)
    assert captured_keys[0] == captured_keys[1]
    assert captured_keys[1] != captured_keys[2]


@pytest.mark.parametrize(
    "accepted_counts,error_type",
    [
        (None, RuntimeError),
        (torch.ones(2, 2, dtype=torch.int32), ValueError),
        (torch.ones(0, dtype=torch.int32), ValueError),
        (torch.ones(5, dtype=torch.int32), ValueError),
    ],
)
def test_accepted_tokens_reject_missing_or_invalid_replay_metadata(
    accepted_counts: torch.Tensor | None, error_type: type[Exception]
) -> None:
    runner = _runner(num_speculative_tokens=1)
    runner.bind_layer_caches([LayerCache(None, None, conv=torch.zeros(8, 1, 2), ssm=torch.zeros(16, 1, 1, 1))])
    batch, metadata = _verify_batch_metadata(2, 2)
    metadata.is_dummy = True
    metadata.num_accepted_tokens = None
    metadata.slot_mapping.fill_(-1)
    entry = runner._allocate_entry(8, batch.input_ids, batch.positions, metadata)
    batch, metadata = _verify_batch_metadata(2, 2)
    metadata.num_accepted_tokens = accepted_counts
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(error_type, match="accepted-token metadata"),
    ):
        runner._fill_entry(entry, batch.input_ids, batch.positions, metadata, batch.num_tokens, None)


def test_slice_output_preserves_aux_hidden_tuple() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    aux_hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    output = DecodeAclGraphRunner._slice_output((hidden, aux_hidden), 3)

    assert isinstance(output, tuple)
    torch.testing.assert_close(output[0], hidden[:3])
    torch.testing.assert_close(output[1], aux_hidden[:3])


def test_linear_state_graph_buffers_are_stable() -> None:
    runner = _runner(num_speculative_tokens=1)
    runner.bind_layer_caches(
        [
            LayerCache(
                None,
                None,
                conv=torch.zeros(8, 2, 1),
                ssm=torch.zeros(16, 1, 1, 1),
                kpool_tail=torch.zeros((8, 1, 1, 1), dtype=torch.bfloat16),
            )
        ]
    )
    batch, metadata = _verify_batch_metadata(2, 2)
    metadata.kpool_query_lens = [2, 2]
    entry = runner._allocate_entry(8, batch.input_ids, batch.positions, metadata)
    static_indices = entry.static_metadata.linear_state_indices
    static_counts = entry.static_metadata.num_accepted_tokens
    pointers = (
        static_indices.data_ptr(),
        static_counts.data_ptr(),
        entry.static_metadata.kpool_query_lens_device.data_ptr(),
    )
    assert entry.static_metadata.kpool_query_lens == (2, 2, 2, 2)
    assert entry.static_metadata.kpool_query_lens_device.tolist() == [2, 2, 2, 2]
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for slots, counts in (([1, 2], [2, 1]), ([3, 4], [1, 2])):
            metadata.linear_state_indices = torch.tensor(slots, dtype=torch.int32)
            metadata.num_accepted_tokens = torch.tensor(counts, dtype=torch.int32)
            runner._fill_entry(entry, batch.input_ids, batch.positions, metadata, batch.num_tokens, None)
            assert static_indices.tolist() == [slot for slot in slots for _ in range(2)] + [0] * 4
            assert static_counts.tolist() == counts + [1] * 6
            assert (
                static_indices.data_ptr(),
                static_counts.data_ptr(),
                entry.static_metadata.kpool_query_lens_device.data_ptr(),
            ) == pointers
            assert entry.static_metadata.kpool_query_lens_device.tolist() == [2, 2, 2, 2]


def test_dflash_proposal_mode_uses_a_distinct_graph_key() -> None:
    runner = _runner(is_spec_draft=True, num_speculative_tokens=4, model_type="DSparkDraftModel")
    captured: list[tuple[Any, ...]] = []

    class ProbeCache(dict):
        def get(self, key: tuple[Any, ...], default: Any = None) -> Any:
            captured.append(key)
            raise LookupError("graph key recorded")

    runner._graphs = ProbeCache()
    batch, metadata = _verify_batch_metadata(2, 4)
    for proposal in (False, True):
        metadata.use_dflash_proposal_attention = proposal
        with pytest.raises(LookupError, match="graph key recorded"):
            runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)
    assert captured[0] != captured[1]


@pytest.mark.parametrize("configured_width", [1, 4])
def test_width_one_verify_uses_a_separate_graph_from_plain_decode(
    monkeypatch: pytest.MonkeyPatch, configured_width: int
) -> None:
    from xllm.python.attention.kda_metadata import KdaMetadata, KdaMetadataBuilder

    monkeypatch.setattr(torch, "npu", MagicMock(), raising=False)
    monkeypatch.setattr(
        import_module(DecodeAclGraphRunner.__module__), "device_capture_lock", lambda device: nullcontext()
    )
    runner = _runner(num_speculative_tokens=configured_width - 1)
    runner.attention_backend.prepare = MagicMock()
    builder = KdaMetadataBuilder()
    caches = [LayerCache(None, None, conv=torch.zeros(8, 5, 6), ssm=torch.zeros(32, 1, 2, 2))]
    builder.bind_layer_caches(caches)
    runner.bind_layer_caches(caches)
    runner.bind_execution_metadata_builders((builder,))

    batch, verify_metadata = _verify_batch_metadata(2, 1)
    plain_metadata = SimpleNamespace(**vars(verify_metadata))
    plain_metadata.is_spec_verify = False
    plain_metadata.is_chunked_prefill = False
    plain_metadata.expanded_decode_metadata = None

    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for metadata in (verify_metadata, plain_metadata):
            assert runner.can_execute(ModelInput(batch.input_ids, batch.positions, metadata, input_batch=batch))
            runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)

    verify_key = runner._graph_key(batch.num_tokens, True, None, verify_width=1)
    plain_key = runner._graph_key(batch.num_tokens, False, None, verify_width=1)
    verify = runner._graphs[verify_key].execution_contexts[KdaMetadata].spec_verify
    assert verify is not None
    assert verify.query_start_loc.tolist() == [0, 1, 2]
    assert verify.slot_indices.tolist() == [1, 2]
    assert runner._graphs[plain_key].execution_contexts[KdaMetadata].spec_verify is None


def test_kda_graph_refreshes_request_metadata_in_place() -> None:
    from xllm.python.attention.kda_metadata import KdaMetadata, KdaMetadataBuilder

    width = 3
    builder = KdaMetadataBuilder()
    caches = [LayerCache(None, None, conv=torch.zeros(8, 4, 6), ssm=torch.zeros(32, 1, 2, 2))]
    builder.bind_layer_caches(caches)
    runner = _runner(num_speculative_tokens=width - 1)
    runner.bind_execution_metadata_builders((builder,))
    runner.bind_layer_caches(caches)
    batch, metadata = _verify_batch_metadata(2, width)
    entry = runner._allocate_entry(8, batch.input_ids, batch.positions, metadata, batch, verify_width=width)
    verify = entry.execution_contexts[KdaMetadata].spec_verify
    assert verify is not None
    assert verify.tail_padding == 2
    assert verify.query_start_loc.tolist() == [0, 3, 6]
    buffers = (verify.slot_indices, verify.num_accepted_tokens, verify.ssm_state_indices)
    addresses = tuple(buffer.data_ptr() for buffer in buffers)

    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for slots, accepted in (([1, 2], [3, 2]), ([3], [2])):
            batch, metadata = _verify_batch_metadata(len(slots), width)
            metadata.linear_state_indices = torch.tensor(slots, dtype=torch.int32)
            metadata.num_accepted_tokens = torch.tensor(accepted, dtype=torch.int32)
            runner._fill_entry(
                entry, batch.input_ids, batch.positions, metadata, batch.num_tokens, None, batch, verify_width=width
            )
            expected_slots = slots + [0] * (2 - len(slots))
            assert verify.slot_indices.tolist() == expected_slots
            assert verify.num_accepted_tokens.tolist() == accepted + [1] * (2 - len(slots))
            assert verify.ssm_state_indices.tolist() == [list(range(slot * 4, slot * 4 + 4)) for slot in expected_slots]
            assert (
                tuple(
                    buffer.data_ptr()
                    for buffer in (verify.slot_indices, verify.num_accepted_tokens, verify.ssm_state_indices)
                )
                == addresses
            )
        batch = replace(
            _query_batch(1, 1),
            num_reqs=0,
            num_scheduled_tokens=(),
            num_computed_tokens=(),
            query_start_loc=(0,),
            is_prefilling=(),
            is_dummy=True,
        )
        metadata = _aligned_dp_decode_metadata(1)
        metadata.is_dummy = True
        runner._fill_entry(
            entry, batch.input_ids, batch.positions, metadata, batch.num_tokens, None, batch, verify_width=width
        )
        assert verify.slot_indices.tolist() == [0, 0]
        assert verify.num_accepted_tokens.tolist() == [1, 1]
        assert verify.ssm_state_indices.tolist() == [list(range(4))] * 2


@pytest.mark.parametrize("verify_width", [1, 2])
def test_capture_preserves_live_state_until_first_replay(monkeypatch: pytest.MonkeyPatch, verify_width: int) -> None:
    from xllm.python.attention.kda_metadata import KdaMetadata, KdaMetadataBuilder

    npu = MagicMock()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(
        import_module(DecodeAclGraphRunner.__module__), "device_capture_lock", lambda device: nullcontext()
    )
    builder = KdaMetadataBuilder()
    runner = _runner(num_speculative_tokens=verify_width - 1)
    runner.bind_execution_metadata_builders((builder,))
    batch, metadata = _verify_batch_metadata(4 // verify_width, verify_width)
    slots = torch.arange(1, 5, verify_width)
    metadata.linear_state_indices = slots
    metadata.has_initial_state = torch.ones_like(slots)
    conv = torch.arange(8, dtype=torch.float32)
    ssm = torch.arange(8 * verify_width, dtype=torch.float32)
    initial_conv, initial_ssm = conv.clone(), ssm.clone()
    caches = [LayerCache(None, None, conv=conv, ssm=ssm)]
    builder.bind_layer_caches(caches)
    runner.bind_layer_caches(caches)
    output = torch.empty(4, 1)
    verify = None

    def advance(capture_entry: _DecodeGraphEntry | None = None) -> torch.Tensor:
        nonlocal verify
        if capture_entry is not None:
            verify = capture_entry.execution_contexts[KdaMetadata].spec_verify
        assert verify is not None
        indices = verify.slot_indices.long()
        checkpoints = verify.ssm_state_indices.long()
        accepted = verify.num_accepted_tokens.long()
        values = ssm[checkpoints[torch.arange(indices.numel()), accepted - 1]] + 1
        conv.index_copy_(0, indices, conv[indices] + 1)
        ssm.index_copy_(0, checkpoints.flatten(), values.repeat_interleave(verify_width))
        output.copy_(values.repeat_interleave(verify_width).view(-1, 1))
        if npu.NPUGraph.return_value.replay.call_count == 0:
            torch.testing.assert_close(conv[1:], initial_conv[1:])
            torch.testing.assert_close(ssm[verify_width:], initial_ssm[verify_width:])
        return output

    runner.attention_backend.prepare = lambda *_args, **_kwargs: None
    monkeypatch.setattr(runner, "_forward_static", advance)
    npu.NPUGraph.return_value.replay.side_effect = advance
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        result = runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)
    expected = initial_ssm[slots * verify_width + verify_width - 1] + 1
    torch.testing.assert_close(result.flatten(), expected.repeat_interleave(verify_width))
    torch.testing.assert_close(conv[slots], initial_conv[slots] + 1)
    checkpoints = slots[:, None] * verify_width + torch.arange(verify_width)
    torch.testing.assert_close(ssm[checkpoints], expected[:, None].expand_as(checkpoints))
    torch.testing.assert_close(metadata.linear_state_indices, slots)
    assert metadata.has_initial_state.tolist() == [1] * slots.numel()
    assert metadata.num_accepted_tokens.tolist() == [verify_width] * slots.numel()
    npu.graph_pool_handle.assert_called_once_with()
    assert npu.graph.call_args.kwargs["pool"] is npu.graph_pool_handle.return_value


@pytest.mark.parametrize("verify_width,dp_counts", [(1, None), (1, (4, 7)), (2, (4, 6))])
def test_static_collection_uses_capture_padding_without_caching_graph(
    monkeypatch: pytest.MonkeyPatch, verify_width: int, dp_counts: tuple[int, int] | None
) -> None:
    npu = MagicMock()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(
        import_module(DecodeAclGraphRunner.__module__), "device_capture_lock", lambda device: nullcontext()
    )
    collection = MagicMock()
    collection.collect_forward.return_value = nullcontext()
    runner = _runner(dp_size=2 if dp_counts else 1, num_speculative_tokens=verify_width - 1)
    if verify_width > 1:
        batch, metadata = _verify_batch_metadata(4 // verify_width, verify_width)
    else:
        batch = _query_batch(4, 1)
        metadata = _aligned_dp_decode_metadata(4)
        metadata.is_spec_verify = False
        metadata.linear_state_indices = None
        metadata.block_table[:, 0] = torch.arange(1, 5)
        metadata.paged_kv_indices = metadata.block_table[:, 0].clone()
        metadata.slot_mapping = metadata.block_table[:, 0] * 4
        batch = replace(batch, positions=torch.zeros(4, dtype=torch.int32))
    if dp_counts:
        metadata.dp_execution_token_counts = dp_counts
        metadata.raw_dp_execution_token_counts = dp_counts
        metadata.dp_is_decode = (True, True)
    runner.attention_backend.prepare = MagicMock()
    layouts = []

    def forward(entry: Any) -> torch.Tensor:
        layouts.append(
            (
                entry.static_input_ids.shape,
                entry.static_positions.shape,
                entry.static_metadata.block_table.shape,
                None if entry.static_metadata.q_cu_seq_lens is None else entry.static_metadata.q_cu_seq_lens.shape,
            )
        )
        return torch.zeros(entry.batch_size, 1)

    monkeypatch.setattr(runner, "_forward_static", forward)
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        result = runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch, static_kernel=collection)
        collected_layout = layouts[0]
        assert collected_layout[0] == (8 if dp_counts else 4,)
        assert result.shape == (4, 1)
        assert not runner._graphs
        npu.graph.assert_not_called()
        npu.graph_pool_handle.assert_not_called()
        collection.collect_forward.assert_called_once_with()
        layouts.clear()
        runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)
    assert layouts and all(layout == collected_layout for layout in layouts)
    assert len(runner._graphs) == 1


def test_explicit_verify_reuses_row_aligned_paging_metadata() -> None:
    runner = _runner()
    runner.bind_layer_caches([LayerCache(None, None, conv=torch.zeros(1), ssm=torch.zeros(1))])
    runner.supports_draft_to_target_overlap = True
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
        kv_seq_lens=metadata.kv_seq_lens,
        block_table=metadata.block_table,
        paged_kv_indptr=metadata.paged_kv_indptr,
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
    )
    metadata.is_spec_verify = True
    metadata.is_chunked_prefill = True
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    metadata.q_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [2, 4]
    metadata.kv_cu_seq_lens = torch.tensor([0, 2, 6], dtype=torch.int32)
    metadata.block_table = metadata.block_table[1::2]

    with patch.object(
        runner,
        "_build_row_aligned_paged_kv_metadata",
        side_effect=AssertionError("row-aligned paging must come from C++"),
    ):
        resolved = runner._decode_metadata(metadata)

    assert resolved[3] is metadata.paged_kv_indptr
    assert resolved[4] is metadata.paged_kv_indices
    assert resolved[5] is metadata.paged_kv_last_page_len
    input_ids = torch.arange(4, dtype=torch.int32)
    assert runner.can_execute(ModelInput(input_ids, torch.zeros_like(input_ids), metadata))
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    assert entry.static_metadata.is_spec_verify
    assert entry.static_metadata.q_seq_lens.tolist() == [2, 2, 2, 2]
    assert entry.static_metadata.q_seq_lens_host_values == [2, 2, 2, 2]
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
    assert entry.static_metadata.num_accepted_tokens.tolist() == [4, 2, 1, 1, 1, 1, 1, 1]
    assert entry.static_metadata.linear_state_indices.tolist() == [1, 1, 2, 2, 0, 0, 0, 0]


def test_verify_graph_requires_explicit_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    assert not runner.can_execute(
        ModelInput(
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
    )


@pytest.mark.parametrize("dp_size", [1, 2])
def test_fill_entry_refreshes_mega_moe_mask_in_place(dp_size: int) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=dp_size,
        enable_mega_moe_token_mask=True,
    )
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    mask = entry.static_metadata.mega_moe_token_mask
    data_ptr = mask.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for remote_count in (3, 1):
            metadata.dp_execution_token_counts = (4, remote_count)[:dp_size]
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
            expected = [True] * 4 + [False] * 4
            if dp_size == 2:
                expected += [True] * remote_count + [False] * (8 - remote_count)
            assert mask.tolist() == expected
            assert entry.static_metadata.mega_moe_token_mask.data_ptr() == data_ptr


@pytest.mark.parametrize("ratios", [[4, 128], [1, 2], [2, 1]])
def test_dsa_graph_normalizes_logical_compressed_table(ratios: list[int]) -> None:
    if max(ratios) > 2:
        _, groups = build_cache_specs(ratios, 128, 2)
    else:
        _, groups = build_cache_specs_v41(ratios, [0, 1], [], 128, 2)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=groups),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=32768,
    )
    metadata = _metadata(torch.arange(1, dtype=torch.int32))
    swa = torch.tensor([[3, 4]], dtype=torch.int32)
    compressed = torch.tensor([[7, 2, 9]], dtype=torch.int32)
    metadata.multi_block_tables = (swa, compressed)
    normalized = runner._normalize_dsa_metadata(metadata)
    assert len(normalized.multi_block_tables) == 2
    assert normalized.multi_block_tables[1] is compressed
    static = SimpleNamespace(
        multi_block_tables=tuple(torch.zeros_like(table) for table in normalized.multi_block_tables)
    )
    runner._fill_dsa_block_tables(static, normalized, swa, 1)
    assert torch.equal(static.multi_block_tables[1], compressed)


def test_dsa_graph_tables_use_compressed_block_counts() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=32768,
    )
    runner._max_blocks_per_sequence = 256

    tables = runner._build_static_multi_block_tables(4, torch.device("cpu"))

    assert [tuple(table.shape) for table in tables] == [(4, 256), (4, 16)]
    assert all(torch.all(table == 0) for table in tables)

    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    indptr, indices, last_page_len = runner._build_row_aligned_paged_kv_metadata(
        block_table,
        [129, 132],
    )
    assert indptr.tolist() == [0, 2, 4]
    assert indices.tolist() == [10, 11, 20, 21]
    assert last_page_len.tolist() == [1, 4]

    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    assert not runner.can_execute(
        ModelInput(
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
    )


@pytest.mark.parametrize("async_graph_inputs", [False, True])
def test_dsa_graph_refreshes_every_manager_and_clears_tails(async_graph_inputs: bool) -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
        supports_draft_to_target_overlap=async_graph_inputs,
        graph_uses_dsa_metadata=lambda _: False,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10, 11], [12, 13]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 11, 10, 11],
        [12, 13, 12, 13],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [20, 0],
        [21, 0],
        [0, 0],
        [0, 0],
    ]
    if async_graph_inputs:
        for host, device in zip(
            static_metadata.multi_block_tables_host, static_metadata.multi_block_tables, strict=True
        ):
            assert torch.equal(host, device)
    else:
        assert static_metadata.multi_block_tables_host is None

    metadata.multi_block_tables = (
        torch.tensor([[40]], dtype=torch.int32),
        torch.tensor([[50]], dtype=torch.int32),
    )
    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=1,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [40, 40, 40, 40],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [50, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    if async_graph_inputs:
        for host, device in zip(
            static_metadata.multi_block_tables_host, static_metadata.multi_block_tables, strict=True
        ):
            assert torch.equal(host, device)
    else:
        assert static_metadata.multi_block_tables_host is None


def test_dsa_graph_clamps_target_tables_to_draft_groups() -> None:
    _, group_infos = build_cache_specs([1], 128, 1)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(multi_block_tables=(torch.full((4, 2), 99, dtype=torch.int32),))
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10], [11]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 10],
        [11, 11],
        [0, 0],
        [0, 0],
    ]


def test_dsa_graph_padding_uses_reserved_single_kv_length() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(
            page_size=128,
            is_mla=False,
            group_infos=group_infos,
        ),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1, 1, 1, 1]),
    )

    runner._fill_host_metadata(entry, [9, 17], batch_size=2)

    assert entry.static_metadata.kv_seq_lens_host_values == [9, 17, 1, 1]


def test_dsa_graph_pads_new_cache_slots_to_graph_bucket() -> None:
    entry = SimpleNamespace(
        batch_size=16,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[301, 302, 401, 402])

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=4,
    )

    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402] + [0] * 12


def test_dsa_graph_padding_preserves_scheduler_resolved_swa_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(
        new_cache_slots_host_values=[1280, 2560, 3840],
    )
    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=3,
    )

    caches_info, group_infos = build_cache_specs([0, 4, 128, 4], 128, 4)
    builder = DsaMetadataBuilder(caches_info, group_infos)
    dsa = builder.build(
        multi_block_tables=[
            torch.tensor(
                [
                    [10, 11, 0, 0],
                    [20, 21, 0, 0],
                    [30, 31, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
            torch.zeros((4, 4), dtype=torch.int32),
        ],
        kv_seq_lens=[257, 257, 257, 1],
        q_seq_lens=[1, 1, 1, 1],
        positions=torch.tensor([256, 256, 256, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=entry.static_metadata.new_cache_slots_host_values,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[0][0].tolist() == [1280, 2560, 3840, 0]


def test_dsa_graph_allows_dummy_without_scheduler_cache_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[99]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[], is_dummy=True)

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=1,
    )

    assert entry.static_metadata.new_cache_slots_host_values == []


def test_dsv4_decode_admits_scheduler_slots_when_top_level_mapping_is_empty() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.block_table = None
    metadata.multi_block_tables = (torch.tensor([[10, 0], [20, 0], [30, 0], [40, 0]], dtype=torch.int32),)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]

    assert runner.can_execute(
        ModelInput(
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
    )
    assert runner._effective_slot_mapping(metadata, 4, torch.device("cpu")).tolist() == [
        301,
        302,
        401,
        402,
    ]


def test_decode_rejects_mismatched_linear_state_validity_mask() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.has_initial_state = torch.ones(3, dtype=torch.int32)

    assert not runner.can_execute(
        ModelInput(
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
    )


def test_decode_rejects_mismatched_linear_state_indices_without_cache_probe() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(3, dtype=torch.int32))

    assert not runner.can_execute(
        ModelInput(
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
    )


def test_fill_entry_uses_scheduler_slots_for_dsv4_decode() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]
    entry = runner._allocate_entry(8, input_ids, positions, metadata)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ) as update_metadata:
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert update_metadata.call_args.args[2].tolist() == [301, 302, 401, 402]
    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402, 0, 0, 0, 0]


def test_dp_real_and_dummy_decode_share_graph_admission() -> None:
    def make_metadata(*, is_dummy: bool) -> SimpleNamespace:
        metadata = _metadata(torch.tensor([0], dtype=torch.int32))
        metadata.slot_mapping = torch.tensor([0], dtype=torch.int32) if is_dummy else torch.empty(0, dtype=torch.int32)
        metadata.block_table = None
        metadata.multi_block_tables = (torch.tensor([[0]], dtype=torch.int32),)
        metadata.kv_seq_lens = torch.tensor([1], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [1]
        metadata.kv_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([0], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
        metadata.new_cache_slots_host_values = [] if is_dummy else [301]
        metadata.dp_execution_token_counts = (1, 1)
        metadata.dp_is_decode = (1, 1)
        metadata.is_dummy = is_dummy
        return metadata

    real_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    dummy_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )
    input_ids = torch.tensor([1], dtype=torch.int32)

    assert real_runner.can_execute(
        ModelInput(
            input_ids,
            torch.zeros_like(input_ids),
            make_metadata(is_dummy=False),
        )
    )
    assert dummy_runner.can_execute(
        ModelInput(
            input_ids,
            torch.zeros_like(input_ids),
            make_metadata(is_dummy=True),
        )
    )


def test_dsa_graph_positions_are_refreshed_from_current_input() -> None:
    runner = _runner()
    entry = SimpleNamespace(
        batch_size=4,
        static_positions=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        static_metadata=SimpleNamespace(dsa_positions=None),
    )

    runner._fill_graph_dsa_positions(
        entry,
        torch.tensor([20, 21], dtype=torch.int32),
    )

    assert entry.static_metadata.dsa_positions.tolist() == [20, 21, 0, 0]


def _dp_metadata(
    token_counts: tuple[int, int],
    dp_is_decode: tuple[int, int] = (1, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        dp_is_decode=dp_is_decode,
    )


def test_dp_empty_rank_uses_group_wide_acl_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runner.can_execute(
            ModelInput(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                _dp_metadata((5, 0)),
            )
        )


def test_dp_mtp_target_dummy_rank_uses_shared_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    active_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_speculative_tokens=3,
    )
    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_speculative_tokens=3,
    )
    draft_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_speculative_tokens=0,
    )
    metadata = _dp_metadata((4, 0))

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert active_runner.can_execute(
            ModelInput(
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                metadata,
            )
        )
        assert empty_runner.can_execute(
            ModelInput(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                metadata,
            )
        )
        assert draft_runner.can_execute(
            ModelInput(
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                metadata,
            )
        )


def test_dp_expanded_verify_and_dummy_rank_share_graph_decision() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=3,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True
    active.is_chunked_prefill = True
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].can_execute(
            ModelInput(
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                active,
            )
        )
        assert runners[1].can_execute(
            ModelInput(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                dummy,
            )
        )
        busy_plan = runners[0]._shared_graph_plan(active, 4)
        empty_plan = runners[1]._shared_graph_plan(dummy, 1)
        assert busy_plan == empty_plan == (True, 4, 1)

    wide_runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=3,
        )
        for rank in range(2)
    ]
    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert wide_runners[0].can_execute(
            ModelInput(
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                active,
            )
        )
        assert wide_runners[1].can_execute(
            ModelInput(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                dummy,
            )
        )


def test_dp_plain_decode_with_empty_rank_keeps_width_one() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=3,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((4, 0))
    active.raw_dp_execution_token_counts = (4, 0)
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 4) == (False, 1, 4)
        assert runners[1]._shared_graph_plan(dummy, 1) == (False, 1, 4)
        assert (
            runners[0].can_execute(
                ModelInput(
                    torch.zeros(4, dtype=torch.int32),
                    torch.zeros(4, dtype=torch.int32),
                    active,
                )
            )
            is False
        )
        assert (
            runners[1].can_execute(
                ModelInput(
                    torch.zeros(1, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    dummy,
                )
            )
            is False
        )


def test_dp_empty_spec_rank_matches_busy_verify_graph_key() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=3,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        busy_plan = runners[0]._shared_graph_plan(active, 4)
        empty_plan = runners[1]._shared_graph_plan(dummy, 1)
        assert busy_plan == (True, 4, 1)
        assert empty_plan == busy_plan
        busy_key = DecodeAclGraphRunner._graph_key(4, busy_plan[0], None, (), verify_width=busy_plan[1])
        empty_key = DecodeAclGraphRunner._graph_key(4, empty_plan[0], None, (), verify_width=empty_plan[1])
        assert busy_key == empty_key
        assert runners[0].can_execute(
            ModelInput(
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                active,
            )
        )
        assert runners[1].can_execute(
            ModelInput(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                dummy,
            )
        )

        unmarked = _expanded_verify_metadata(4)
        unmarked.dp_execution_token_counts = (4, 1)
        unmarked.raw_dp_execution_token_counts = (4, 0)
        unmarked.is_spec_verify = False
        with pytest.raises(RuntimeError, match="group-wide spec verify"):
            runners[0]._shared_graph_plan(unmarked, 4)


def test_dflash_draft_graph_capacity_uses_sequence_count() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            is_spec_draft=True,
            num_speculative_tokens=8,
            model_type="DSparkDraftModel",
        )
        for rank in range(2)
    ]
    active = _dp_metadata((16, 0))
    active.raw_dp_execution_token_counts = (16, 0)
    dummy = _dp_metadata((16, 0))
    dummy.raw_dp_execution_token_counts = (16, 0)
    misaligned = _dp_metadata((10, 0))
    misaligned.raw_dp_execution_token_counts = (10, 0)

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 16) == (False, 1, 2)
        assert runners[1]._shared_graph_plan(dummy, 1) == (False, 1, 2)
        assert runners[0].can_execute(
            ModelInput(
                torch.zeros(16, dtype=torch.int32),
                torch.zeros(16, dtype=torch.int32),
                active,
            )
        )
        assert runners[1].can_execute(
            ModelInput(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                dummy,
            )
        )
        assert (
            runners[0].can_execute(
                ModelInput(
                    torch.zeros(10, dtype=torch.int32),
                    torch.zeros(10, dtype=torch.int32),
                    misaligned,
                )
            )
            is False
        )
        assert (
            runners[1].can_execute(
                ModelInput(
                    torch.zeros(1, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    misaligned,
                )
            )
            is False
        )


def test_dp_uneven_sequences_stay_in_graph_with_full_batch_capacity() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=5,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((48, 0))
    active.raw_dp_execution_token_counts = (48, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((48, 0))
    dummy.raw_dp_execution_token_counts = (48, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].max_batch == 8
        assert runners[0].can_execute(
            ModelInput(
                torch.zeros(48, dtype=torch.int32),
                torch.zeros(48, dtype=torch.int32),
                active,
            )
        )
        assert runners[1].can_execute(
            ModelInput(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                dummy,
            )
        )


def test_dp_spec_verify_width_six_pads_to_bucket_eight() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=5,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(6)
    active.dp_execution_token_counts = (6, 6)
    active.raw_dp_execution_token_counts = (6, 6)
    active.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 6) == (True, 6, 1)
        assert runners[0].can_execute(
            ModelInput(
                torch.zeros(6, dtype=torch.int32),
                torch.zeros(6, dtype=torch.int32),
                active,
            )
        )
        assert runners[1].can_execute(
            ModelInput(
                torch.zeros(6, dtype=torch.int32),
                torch.zeros(6, dtype=torch.int32),
                active,
            )
        )


def test_dp_spec_verify_refuses_bucket_not_divisible_by_width() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=2,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((2, 0))
    active.raw_dp_execution_token_counts = (2, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((2, 0))
    dummy.raw_dp_execution_token_counts = (2, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 2) == (True, 3, None)
        assert runners[1]._shared_graph_plan(dummy, 1) == (True, 3, None)
        assert (
            runners[0].can_execute(
                ModelInput(
                    torch.zeros(2, dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                    active,
                )
            )
            is False
        )
        assert (
            runners[1].can_execute(
                ModelInput(
                    torch.zeros(1, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    dummy,
                )
            )
            is False
        )


def test_dp_verification_shares_token_count_across_ranks() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=7,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(8)
    active.dp_execution_token_counts = (8, 1)
    active.raw_dp_execution_token_counts = (8, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((8, 0))
    dummy.raw_dp_execution_token_counts = (8, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 8) == (True, 8, 1)
        assert runners[1]._shared_graph_plan(dummy, 1) == (True, 8, 1)


def test_dp_spec_verify_rejects_measured_width_mismatch() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_speculative_tokens=7,
    )
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True

    with pytest.raises(RuntimeError, match="does not match verify_width"):
        runner._shared_graph_plan(active, 4)


def test_dp_spec_verify_stays_eager_when_active_count_is_not_a_multiple() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=3,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((6, 0))
    active.raw_dp_execution_token_counts = (6, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((6, 0))
    dummy.raw_dp_execution_token_counts = (6, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert (
            runners[0].can_execute(
                ModelInput(
                    torch.zeros(6, dtype=torch.int32),
                    torch.zeros(6, dtype=torch.int32),
                    active,
                )
            )
            is False
        )
        assert (
            runners[1].can_execute(
                ModelInput(
                    torch.zeros(1, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    dummy,
                )
            )
            is False
        )


def test_dp_kpool_spans_must_match_shared_bucket_layout() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_speculative_tokens=7,
    )
    runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16))]
    aligned = _dp_metadata((8, 8))
    aligned.is_spec_verify = True
    mismatched = _dp_metadata((8, 8))
    mismatched.is_spec_verify = True
    mismatched.kpool_query_lens = (1,) * 8

    with patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                aligned,
            )
        )
        with pytest.raises(RuntimeError, match="shared bucket layout"):
            runner.can_execute(
                ModelInput(
                    torch.zeros(8, dtype=torch.int32),
                    torch.zeros(8, dtype=torch.int32),
                    mismatched,
                )
            )
    assert runner._graph_kpool_query_lens(aligned, 8, 8, 8) == (8,)
    assert runner._graph_kpool_query_lens(aligned, 1, 8, 8) == (8,)

    uneven = _dp_metadata((16, 16))
    uneven.is_spec_verify = True
    uneven.kpool_query_lens = (10, 6)
    short = _dp_metadata((16, 16))
    short.is_spec_verify = True
    short.kpool_query_lens = (8,)
    nonpositive = _dp_metadata((8, 8))
    nonpositive.is_spec_verify = True
    nonpositive.kpool_query_lens = (0, 8)
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="must be uniform"),
    ):
        runner.can_execute(
            ModelInput(
                torch.zeros(16, dtype=torch.int32),
                torch.zeros(16, dtype=torch.int32),
                uneven,
            )
        )
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="positive lengths"),
    ):
        runner.can_execute(
            ModelInput(
                torch.zeros(16, dtype=torch.int32),
                torch.zeros(16, dtype=torch.int32),
                short,
            )
        )
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="positive lengths"),
    ):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                nonpositive,
            )
        )


def _aligned_dp_decode_metadata(rows: int) -> SimpleNamespace:
    metadata = _dp_metadata((rows, rows))
    metadata.is_spec_verify = True
    metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
    metadata.block_table = torch.zeros((rows, 2), dtype=torch.int32)
    metadata.kv_seq_lens = torch.ones(rows, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [1] * rows
    metadata.kv_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32)
    metadata.paged_kv_indptr = torch.arange(rows + 1, dtype=torch.int32)
    metadata.paged_kv_indices = torch.zeros(rows, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.ones(rows, dtype=torch.int32)
    metadata.linear_state_indices = torch.zeros(rows, dtype=torch.int32)
    metadata.expanded_decode_metadata = None
    return metadata


def test_dp_all_decode_row_mismatch_raises() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_speculative_tokens=7,
    )
    aligned = _aligned_dp_decode_metadata(8)
    mismatched = _aligned_dp_decode_metadata(8)
    mismatched.block_table = torch.zeros((1, 2), dtype=torch.int32)
    mismatched.kv_seq_lens = torch.ones(1, dtype=torch.int32)
    mismatched.kv_seq_lens_host_values = [1]
    mismatched.slot_mapping = torch.zeros(1, dtype=torch.int32)

    assert runner.can_execute(
        ModelInput(
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int32),
            aligned,
        )
    )
    assert runner.can_execute(
        ModelInput(
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int32),
            aligned,
            torch.zeros(8, 4),
        )
    )
    narrow = _aligned_dp_decode_metadata(8)
    narrow.paged_kv_indptr = None
    narrow.paged_kv_indices = None
    narrow.paged_kv_last_page_len = None
    narrow.block_table = torch.zeros((8, 1), dtype=torch.int32)
    narrow.kv_seq_lens = torch.full((8,), 5, dtype=torch.int32)
    narrow.kv_seq_lens_host_values = [5] * 8
    with pytest.raises(RuntimeError, match="cannot hold the KV length"):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                narrow,
            )
        )
    fits = _aligned_dp_decode_metadata(8)
    fits.paged_kv_indptr = None
    fits.paged_kv_indices = None
    fits.paged_kv_last_page_len = None
    fits.block_table = torch.zeros((8, 1), dtype=torch.int32)
    fits.kv_seq_lens = torch.full((8,), 4, dtype=torch.int32)
    fits.kv_seq_lens_host_values = [4] * 8
    assert runner.can_execute(
        ModelInput(
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int32),
            fits,
        )
    )
    with pytest.raises(RuntimeError, match="input_embedding does not match"):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                aligned,
                torch.zeros(3, 4),
            )
        )
    with pytest.raises(RuntimeError, match="does not match token rows"):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                mismatched,
            )
        )

    bad_q_cu = _aligned_dp_decode_metadata(8)
    bad_q_cu.q_cu_seq_lens = torch.zeros(3, dtype=torch.int32)
    bad_indices = _aligned_dp_decode_metadata(8)
    bad_indices.linear_state_indices = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="q_cu_seq_lens does not match"):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                bad_q_cu,
            )
        )
    with pytest.raises(RuntimeError, match="linear_state_indices does not match"):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                bad_indices,
            )
        )

    bad_initial_state = _aligned_dp_decode_metadata(8)
    bad_initial_state.has_initial_state = torch.ones(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="has_initial_state does not match"):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                bad_initial_state,
            )
        )

    runner.layer_caches = [SimpleNamespace(conv=object(), kpool_tail=None)]
    missing_indices = _aligned_dp_decode_metadata(8)
    missing_indices.linear_state_indices = None
    missing_indices.raw_dp_execution_token_counts = (8, 0)
    with pytest.raises(RuntimeError, match="missing linear_state_indices"):
        runner.can_execute(
            ModelInput(
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                missing_indices,
            )
        )

    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_speculative_tokens=7,
    )
    empty_runner.layer_caches = [SimpleNamespace(conv=object(), kpool_tail=None)]
    placeholder = _aligned_dp_decode_metadata(1)
    placeholder.dp_execution_token_counts = (8, 1)
    placeholder.raw_dp_execution_token_counts = (8, 0)
    placeholder.linear_state_indices = None
    assert empty_runner.can_execute(
        ModelInput(
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            placeholder,
        )
    )


def test_dp_kpool_spans_keep_complete_groups_only() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=5,
        )
        for rank in range(2)
    ]
    for runner in runners:
        runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16))]
    active = _dp_metadata((6, 6))
    active.is_spec_verify = True
    active.kpool_query_lens = (6,)
    empty = _dp_metadata((6, 6))
    empty.is_spec_verify = True

    with patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True):
        assert runners[0].can_execute(
            ModelInput(
                torch.zeros(6, dtype=torch.int32),
                torch.zeros(6, dtype=torch.int32),
                active,
            )
        )
        assert runners[1].can_execute(
            ModelInput(
                torch.zeros(6, dtype=torch.int32),
                torch.zeros(6, dtype=torch.int32),
                empty,
            )
        )
    assert runners[0]._graph_kpool_query_lens(active, 6, 8, 6) == (6,)
    assert runners[1]._graph_kpool_query_lens(empty, 6, 8, 6) == (6,)
    assert runners[0]._graph_kpool_query_lens(empty, 8, 16, 8) == (8, 8)


def test_empty_rank_kpool_spans_follow_shared_verify_width() -> None:
    runner = _runner()
    runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16))]
    metadata = SimpleNamespace()

    busy = runner._padded_kpool_query_lens(metadata, 8, 16, verify_width=8)
    empty = runner._padded_kpool_query_lens(metadata, 1, 16, verify_width=8)

    assert busy == (8, 8)
    assert empty == busy


def test_dp_expanded_verify_without_placeholder_folds_to_sequences() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=3,
        )
        for rank in range(2)
    ]
    metadata = _expanded_verify_metadata(8)
    metadata.dp_execution_token_counts = (8, 8)
    metadata.raw_dp_execution_token_counts = (8, 8)
    metadata.linear_state_indices = torch.tensor([0, 1], dtype=torch.int32)
    metadata.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        for runner in runners:
            assert runner.can_execute(
                ModelInput(
                    torch.zeros(8, dtype=torch.int32),
                    torch.zeros(8, dtype=torch.int32),
                    metadata,
                )
            )


def _expanded_verify_metadata(rows: int) -> SimpleNamespace:
    metadata = _dp_metadata((rows, rows))
    metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
    metadata.linear_state_indices = torch.tensor([0], dtype=torch.int32)
    metadata.expanded_decode_metadata = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
    )
    return metadata


@pytest.mark.parametrize(
    ("width", "disable_verify_graph"),
    [
        pytest.param(4, False, id="kda-verify-default"),
        pytest.param(4, True, id="verify-graph-disabled"),
        pytest.param(3, False, id="width-three-pads-inside-bucket"),
    ],
)
def test_dp_mtp_target_fallback_is_group_wide(
    width: int,
    disable_verify_graph: bool,
) -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_speculative_tokens=width - 1,
        )
        for rank in range(2)
    ]
    metadata = _expanded_verify_metadata(width)
    metadata.is_spec_verify = True
    admitted = not disable_verify_graph

    with (
        patch.dict(
            os.environ,
            {"XLLM_NO_VERIFY_GRAPH": "1" if disable_verify_graph else "0"},
        ),
        patch.object(
            DecodeAclGraphRunner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
    ):
        for runner in runners:
            assert (
                runner.can_execute(
                    ModelInput(
                        torch.zeros(width, dtype=torch.int32),
                        torch.zeros(width, dtype=torch.int32),
                        metadata,
                    )
                )
                is admitted
            )


def test_kda_verify_graph_is_enabled_by_default() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 3], dtype=torch.int32))
    metadata.q_cu_seq_lens = torch.arange(5, dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([5, 6, 9, 10], dtype=torch.int32)
    with (
        patch.dict(os.environ, {"XLLM_NO_VERIFY_GRAPH": "0"}),
        patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
    ):
        assert runner.can_execute(
            ModelInput(
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                metadata,
            )
        )


def test_dp_mixed_step_does_not_enter_acl_decode_graph() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert not runner.can_execute(
            ModelInput(
                torch.zeros(3, dtype=torch.int32),
                torch.zeros(3, dtype=torch.int32),
                _dp_metadata((3, 2), dp_is_decode=(0, 1)),
            )
        )
        missing_step_type = _dp_metadata((3, 2))
        del missing_step_type.dp_is_decode
        with pytest.raises(RuntimeError, match="requires dp_is_decode"):
            runner.can_execute(
                ModelInput(
                    torch.zeros(3, dtype=torch.int32),
                    torch.zeros(3, dtype=torch.int32),
                    missing_step_type,
                )
            )
        assert not runner.can_execute(
            ModelInput(
                torch.zeros(1, 3, dtype=torch.int32),
                torch.zeros(1, 3, dtype=torch.int32),
                _dp_metadata((3, 2), dp_is_decode=(0, 1)),
            )
        )
        with pytest.raises(RuntimeError, match="must be one-dimensional"):
            runner.can_execute(
                ModelInput(
                    torch.zeros(1, 8, dtype=torch.int32),
                    torch.zeros(1, 8, dtype=torch.int32),
                    _dp_metadata((8, 8)),
                )
            )


def test_dp_acl_graph_requires_group_wide_token_counts() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    metadata = _dp_metadata((3, 2))
    metadata.dp_execution_token_counts = (3,)

    with (
        patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="valid dp_execution_token_counts"),
    ):
        runner.can_execute(
            ModelInput(
                torch.zeros(3, dtype=torch.int32),
                torch.zeros(3, dtype=torch.int32),
                metadata,
            )
        )


def test_xfia_graph_accepts_device_only_lengths() -> None:
    runner = _runner()
    runner.attention_backend.requires_host_kv_lengths = False
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    assert runner._has_compatible_decode_metadata(torch.arange(4), metadata)
    table, lengths, host_lengths, *_ = runner._decode_metadata(metadata)
    assert table is metadata.block_table
    assert lengths is metadata.kv_seq_lens
    assert host_lengths is None
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1] * 4),
    )
    runner._fill_host_metadata(entry, None, batch_size=4)
    assert entry.static_metadata.kv_seq_lens_host_values == [1] * 4


@pytest.mark.parametrize("num_sequences", [1, 2, 3, 8])
def test_xfia_draft_proposal_graph_accepts_request_scoped_metadata(num_sequences: int) -> None:
    query_width = 15
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, requires_host_kv_lengths=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=256,
        is_spec_draft=True,
        num_speculative_tokens=query_width,
        model_type="DSparkDraftModel",
    )
    num_tokens = num_sequences * query_width
    metadata = _metadata(torch.arange(num_sequences, dtype=torch.int64))
    metadata.linear_state_indices = None
    metadata.is_chunked_prefill = True
    metadata.use_dflash_proposal_attention = True
    metadata.slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
    metadata.block_table = torch.arange(num_sequences * 2, dtype=torch.int32).reshape(num_sequences, 2)
    metadata.kv_seq_lens = torch.full((num_sequences,), 129, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [143] * num_sequences
    metadata.q_cu_seq_lens = torch.arange(1, num_sequences + 1, dtype=torch.int32) * query_width
    metadata.kv_cu_seq_lens = torch.arange(1, num_sequences + 1, dtype=torch.int32) * 129
    input_ids = torch.zeros(num_tokens, dtype=torch.int32)
    assert runner.can_execute(ModelInput(input_ids, torch.zeros_like(input_ids), metadata))
    table, lengths, host_lengths, *_ = runner._decode_metadata(metadata)
    assert table.shape == (num_tokens, 2)
    assert lengths.tolist() == [129] * num_tokens
    assert host_lengths is None
    metadata.use_dflash_proposal_attention = False
    assert not runner.can_execute(ModelInput(input_ids, torch.zeros_like(input_ids), metadata))


# ---------------------------------------------------------------------------
# Engram graph-visible staging capacity hint
# ---------------------------------------------------------------------------


def test_acl_runner_pre_sizes_engram_graph_staging_capacity() -> None:
    """The runner tells the model its largest decode bucket before capture.

    Capture is lazy and buckets warm up ascending, so without this hint the
    engram graph-visible block would be fixed at the first (smallest) bucket.
    """
    from xllm.python.model_executor.runners.decode_cuda_graph import _decode_bucket

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hints: list[int] = []

        def configure_engram_graph_capacity(self, max_tokens: int) -> bool:
            self.hints.append(int(max_tokens))
            return True

    model = _Model()
    DecodeAclGraphRunner(
        model,
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=64,
        max_model_len=8,
        num_speculative_tokens=1,
    )
    # 64 sequences x 2 token rows, folded through the decode bucket.
    assert model.hints == [_decode_bucket(64 * 2)]
    assert model.hints == [128]

    # ``decode_batch_size_limit`` clamps the captured buckets, so the hint too.
    limited = _Model()
    DecodeAclGraphRunner(
        limited,
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=64,
        max_model_len=8,
        decode_batch_size_limit=32,
        num_speculative_tokens=0,
    )
    assert limited.hints == [_decode_bucket(32)]

    # A spec-draft runner emits up to ``draft_query_width`` query rows per
    # sequence, which can exceed ``num_decoding_tokens``; the graph-visible
    # engram block must cover that wider bucket, not the narrower MTP one.
    draft = _Model()
    DecodeAclGraphRunner(
        draft,
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=64,
        max_model_len=8,
        is_spec_draft=True,
        num_speculative_tokens=8,
        model_type="DSparkDraftModel",
    )
    assert draft.hints == [_decode_bucket(64 * 8)]

    # A model without the hook is untouched.
    DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def test_acl_runner_warns_once_when_engram_staging_returns_false() -> None:
    """A False staging result on an offload-active model: capture warns, replay raises.

    The pre-graph staging hook's return value gates whether a replay re-reads
    the engram staging block. On a capture (``capturing=True``) this helper
    returns False and latches a single per-runner warning as a companion
    diagnostic; the runner's ``_engram_first_capture_gate`` then refuses the
    capture before it is cached (see the gate tests below), so no Engram-less
    graph is stored. On a replay (``capturing=False``) the graph was already
    captured *with* engram, so continuing would serve the previous step's stale
    rows as a silent wrong output (the eager path skips engram in the same
    condition); that is a fail-fast ``RuntimeError``.

    Discriminating: removing the capture warning makes ``len(warnings) == 1``
    fail; removing the replay raise makes ``pytest.raises`` fail; dropping the
    ``engram_offload_active`` gate makes the non-offload model warn/raise. The
    warning is asserted through the shared ``xllm`` logger (``scripts.logger``),
    not a per-module logger. That logger sets ``propagate = False``, so a
    root-attached ``caplog`` handler cannot see its records. Rather than flip the
    global ``propagate`` (which would leak into other tests, including ones
    running concurrently), the test attaches its own scoped handler straight to
    the ``xllm`` logger and removes it in ``finally``.
    """

    class _Model(nn.Module):
        def __init__(self, *, staged: bool, offload_active: bool) -> None:
            super().__init__()
            self._staged = staged
            self.engram_offload_active = offload_active
            self.layers: list = []
            self.calls = 0

        def stage_engram_rows_outside_graph(self, *args: object) -> bool:
            self.calls += 1
            return self._staged

    def make_runner(model: nn.Module) -> DecodeAclGraphRunner:
        return DecodeAclGraphRunner(
            model,
            SimpleNamespace(page_size=4, is_mla=False),
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
        )

    # Scoped handler on the shared ``xllm`` logger: collect WARNINGs without
    # touching the logger's global ``propagate`` (see the docstring).
    xllm_logger = logging.getLogger("xllm")
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture(level=logging.WARNING)
    xllm_logger.addHandler(handler)
    try:
        # Capture: a not-staged offload model warns exactly once and never
        # raises, even though a capture walks several buckets (the latch holds).
        failing = make_runner(_Model(staged=False, offload_active=True))
        assert failing._stage_engram_rows_outside_graph(None, None, None, capturing=True) is False
        assert failing._stage_engram_rows_outside_graph(None, None, None, capturing=True) is False
        assert failing._engram_stage_skip_warned is True
        warnings = [record for record in captured if "staged nothing" in record.getMessage()]
        assert len(warnings) == 1, "a capture-stage failure must warn exactly once"
        assert "capturing=True" in warnings[0].getMessage()

        # Replay: the same condition is a hard error, not a stale-row replay, and
        # it must not also latch the capture warning.
        captured.clear()
        replay = make_runner(_Model(staged=False, offload_active=True))
        with pytest.raises(RuntimeError, match="previous step's rows"):
            replay._stage_engram_rows_outside_graph(None, None, None, capturing=False)
        assert not [record for record in captured if "staged nothing" in record.getMessage()]

        # Legitimate no-ops: a successful stage and a non-offload model stay
        # silent (neither warning nor raise) on both capture and replay.
        captured.clear()
        for model in (_Model(staged=True, offload_active=True), _Model(staged=False, offload_active=False)):
            runner = make_runner(model)
            assert runner._stage_engram_rows_outside_graph(None, None, None, capturing=True) == model._staged
            assert runner._stage_engram_rows_outside_graph(None, None, None, capturing=False) == model._staged
        assert not [record for record in captured if "staged nothing" in record.getMessage()]
    finally:
        xllm_logger.removeHandler(handler)


def _engram_gate_runner(*, offload_active: bool) -> DecodeAclGraphRunner:
    """A CPU ACL runner whose model toggles ``engram_offload_active``."""

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.engram_offload_active = offload_active

    return DecodeAclGraphRunner(
        _Model(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def test_acl_runner_engram_first_capture_gate_refuses_when_not_staged() -> None:
    """A first-capture staging failure must fail fast before anything is cached.

    The gate runs before ``_capture`` and before the entry is registered, so a
    ``False`` stage on an offload-active model must raise and leave ``_graphs``
    untouched (there is no entry to drop). Discriminating: deleting the raise
    lets the Engram-less graph be captured and cached.
    """

    runner = _engram_gate_runner(offload_active=True)

    with pytest.raises(RuntimeError) as excinfo:
        runner._engram_first_capture_gate(staged=False)

    assert runner._graphs == {}, "the gate refuses before anything is cached"
    message = str(excinfo.value)
    assert "first ACL graph capture" in message
    assert "Engram-less" in message
    assert "engram_config.cpu_offload=false" in message
    # The hint must name the copy-pasteable CLI form, and that exact JSON must
    # parse with both switches off (the HBM path the message points at).
    example = '{"cpu_offload":false,"dp_shared_memory":false}'
    assert f"--engram-config '{example}'" in message
    assert EngramConfig.from_value(example) == EngramConfig(cpu_offload=False, dp_shared_memory=False)


def test_acl_runner_engram_first_capture_gate_passes_when_staged() -> None:
    """A first capture that staged engram is allowed through unchanged.

    ``staged=True`` short-circuits the gate, so a healthy capture proceeds and
    the gate is a pure pass-through (no raise, no cache mutation).
    """

    runner = _engram_gate_runner(offload_active=True)
    runner._engram_first_capture_gate(staged=True)
    assert runner._graphs == {}


def test_acl_runner_engram_first_capture_gate_passes_when_offload_inactive() -> None:
    """A non-offload model's ``False`` hook must not trip the gate.

    Non-engram / non-offload models legitimately return ``False`` from the
    staging hook (it is a no-op), so gating only on ``staged`` would raise for
    them; the ``engram_offload_active`` guard keeps them silent.
    """

    runner = _engram_gate_runner(offload_active=False)
    runner._engram_first_capture_gate(staged=False)
    assert runner._graphs == {}


class _StoppedAfterCaptureError(Exception):
    """Sentinel raised at the replay boundary, after a successful capture."""


class _StopAtReplayStream:
    """``self._stream`` stand-in that stops ``execute`` at the replay.

    ``execute`` only reaches ``self._stream.wait_stream(...)`` after the first
    capture has completed and been registered, so raising from ``wait_stream``
    keeps the test device-free while still exercising the capture-then-register
    ordering.
    """

    def wait_stream(self, *args: object, **kwargs: object) -> None:
        raise _StoppedAfterCaptureError


def _engram_execute_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    staged: bool,
    capture_error: Exception | None = None,
    capture_sizes: tuple[int, ...] = (),
) -> tuple[DecodeAclGraphRunner, Any, list[Any], Any]:
    """Wire a runner for the real ``DecodeAclGraphRunner.execute`` path.

    Only ``_allocate_entry`` / ``_fill_entry`` / ``_capture`` and the backend
    ``prepare`` are stubbed; ``execute`` itself -- the gate call, the capture
    call and the registration that follows it -- runs for real. ``_stream``
    raises :class:`_StoppedAfterCaptureError` from ``wait_stream`` so a
    successful capture stops right after registration, before the replay block
    (no device needed); when ``capture_error`` is set the stubbed ``_capture``
    raises it instead. ``torch.npu`` is stubbed to the single entry point used
    on that path so the test does not require an NPU.
    """

    class _EngramModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.engram_offload_active = True
            self.layers: list = []
            self.staged = staged
            self.stage_calls = 0

        def stage_engram_rows_outside_graph(self, *args: object) -> bool:
            self.stage_calls += 1
            return self.staged

    class _KeyRecordingDict(dict):
        def __init__(self) -> None:
            super().__init__()
            self.keys_written: list[Any] = []

        def __setitem__(self, key: Any, value: Any) -> None:
            self.keys_written.append(key)
            super().__setitem__(key, value)

    model = _EngramModel()
    backend = SimpleNamespace(page_size=4, is_mla=False, prepare=lambda *args, **kwargs: None)
    runner = DecodeAclGraphRunner(
        model, backend, torch.device("cpu"), max_batch=8, max_model_len=8, capture_sizes=capture_sizes
    )
    runner._stream = _StopAtReplayStream()
    graphs = _KeyRecordingDict()
    runner._graphs = graphs

    def fake_allocate_entry(
        padded_batch_size,
        input_ids,
        positions,
        metadata,
        input_batch=None,
        verify_width=1,
        slot_variant=0,
        adapter_signature=None,
    ):
        return SimpleNamespace(
            static_metadata=SimpleNamespace(),
            static_input_ids=input_ids,
            static_positions=positions,
            execution_state=object(),
            execution_contexts={},
            eplb=None,
            graph=None,
        )

    capture_calls: list[Any] = []

    def fake_capture(entry):
        capture_calls.append(entry)
        if capture_error is not None:
            raise capture_error

    monkeypatch.setattr(runner, "_allocate_entry", fake_allocate_entry)
    monkeypatch.setattr(runner, "_fill_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_capture", fake_capture)
    # The only ``torch.npu`` call reached on this path is the replay-boundary
    # ``current_stream`` lookup, right before ``_stream.wait_stream`` raises.
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: None), raising=False)
    return runner, model, capture_calls, graphs


def test_custom_capture_overflow_is_rejected_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, model, captures, graphs = _engram_execute_harness(monkeypatch, staged=True, capture_sizes=(3, 6))
    allocate = MagicMock(side_effect=AssertionError("overflow must not allocate"))
    monkeypatch.setattr(runner, "_allocate_entry", allocate)
    tokens, positions, metadata = _engram_execute_inputs(7)
    with pytest.raises(ValueError, match="exceed effective graph_capture_sizes"):
        runner.execute(tokens, positions, metadata)
    allocate.assert_not_called()
    assert model.stage_calls == 0
    assert not captures
    assert not graphs.keys_written


def _engram_execute_inputs(rows: int = 4) -> tuple[torch.Tensor, torch.Tensor, Any]:
    input_ids = torch.arange(rows, dtype=torch.int64)
    positions = torch.arange(rows, dtype=torch.int64)
    metadata = SimpleNamespace(linear_state_indices=torch.arange(rows, dtype=torch.int32))
    return input_ids, positions, metadata


@pytest.mark.parametrize("num_reqs", [0, 3, 4])
def test_acl_runner_slices_pipeline_replay_output_with_input_batch_counts(
    num_reqs: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _query_batch(num_reqs, 3)
    runner, _, _, _ = _engram_execute_harness(monkeypatch, staged=True)
    runner.max_batch = 16
    runner._stream = MagicMock()
    runner._update_stream = MagicMock()
    runner._replay_done_event = MagicMock()
    token_capacity = runner._padded_token_rows(batch.num_tokens)
    request_capacity = token_capacity // 3
    source = ModelOutput(
        draft_token_ids=torch.arange(request_capacity * 3).reshape(request_capacity, 3),
        draft_confidence_probs=torch.ones(request_capacity, 3),
    )
    replay = MagicMock()

    def _capture(entry: Any) -> None:
        entry.graph = SimpleNamespace(replay=replay)
        entry.graph_tasks = []
        entry.static_output = source

    monkeypatch.setattr(runner, "_capture", _capture)
    monkeypatch.setattr(runner, "_update_graph_tasks", MagicMock())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: runner._stream, stream=lambda stream: nullcontext()),
        raising=False,
    )
    metadata = SimpleNamespace(linear_state_indices=torch.arange(batch.num_tokens, dtype=torch.int32))

    output = runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)

    replay.assert_called_once()
    assert isinstance(output, ModelOutput)
    assert output.draft_token_ids is not None
    assert source.draft_token_ids is not None
    assert torch.equal(output.draft_token_ids, source.draft_token_ids[:num_reqs])
    assert torch.equal(output.draft_confidence_probs, source.draft_confidence_probs[:num_reqs])
    assert output.hidden_states is None
    if num_reqs:
        assert output.draft_token_ids.data_ptr() == source.draft_token_ids.data_ptr()
        assert output.draft_confidence_probs.data_ptr() == source.draft_confidence_probs.data_ptr()


def test_acl_runner_execute_refuses_first_capture_when_engram_staging_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute() wiring: the gate refuses the first capture before it is cached.

    Drives the real ``execute`` path with a ``False`` stage. Deleting the
    ``_engram_first_capture_gate`` call turns this red: ``_capture`` would run
    and the replay-boundary sentinel would surface instead of the expected
    ``RuntimeError``. The entry must never be written to ``_graphs``.
    """

    runner, model, capture_calls, graphs = _engram_execute_harness(monkeypatch, staged=False)
    input_ids, positions, metadata = _engram_execute_inputs()

    with pytest.raises(RuntimeError) as excinfo:
        runner.execute(input_ids, positions, metadata)

    assert "first ACL graph capture" in str(excinfo.value)
    assert model.stage_calls == 1
    assert capture_calls == [], "_capture must not run when the gate refuses"
    assert graphs.keys_written == [], "a refused capture must not register anything"
    assert runner._graphs == {}


def test_acl_runner_execute_registers_only_after_successful_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute() registers the entry only once ``_capture`` has succeeded.

    With staging recovered the first step captures, then registers the entry, so
    the graph is available for replay. Registering *before* ``_capture`` would
    cache a half-built entry whenever the capture fails.
    """

    runner, model, capture_calls, graphs = _engram_execute_harness(monkeypatch, staged=True, capture_sizes=(3, 6))
    for count in (4, 5, 6):
        input_ids, positions, metadata = _engram_execute_inputs(count)
        with pytest.raises(_StoppedAfterCaptureError):
            runner.execute(input_ids, positions, metadata)
        assert runner.last_padded_batch_size == 6

    assert len(capture_calls) == 1
    assert len(graphs.keys_written) == 1
    graph_key = graphs.keys_written[0]
    assert runner._graphs.get(graph_key) is capture_calls[0]
    assert len(runner._graphs) == 1


def test_csa2_execute_reuses_capture_across_context_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, capture_calls, graphs = _engram_execute_harness(monkeypatch, staged=True)
    runner.attention_backend.requires_paged_decode_graph = True
    runner._graph_token_capacity = 1048576

    for seq_len in (16, 2047, 2049, 131072, 1048576):
        input_ids, positions, _ = _engram_execute_inputs()
        metadata = _aligned_dp_decode_metadata(input_ids.numel())
        metadata.is_spec_verify = False
        metadata.kv_seq_lens.fill_(seq_len)
        metadata.kv_seq_lens_host_values = [seq_len] * input_ids.numel()
        assert runner.can_execute(ModelInput(input_ids, positions, metadata))
        with pytest.raises(_StoppedAfterCaptureError):
            runner.execute(input_ids, positions, metadata)

    assert len(graphs) == len(capture_calls) == 1
    assert capture_calls[0].static_metadata.dsa_graph_token_capacity == 1048576
    for seq_len in (0, 1048577):
        metadata.kv_seq_lens.fill_(seq_len)
        metadata.kv_seq_lens_host_values = [seq_len] * input_ids.numel()
        assert not runner.can_execute(ModelInput(input_ids, positions, metadata))


def test_acl_runner_execute_leaves_no_entry_when_capture_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``_capture`` failure must not leave a ``graph is None`` entry cached.

    Staging succeeds but ``_capture`` raises; because the entry is registered
    only after ``_capture`` returns, nothing is cached and the next step can
    re-capture. Registering before the capture would cache the partial entry
    (``entry.graph is None``) and the next step's replay branch would dereference
    ``entry.graph`` -> ``AttributeError``.
    """

    class _CaptureFailedError(Exception):
        """Sentinel raised by the stubbed ``_capture``."""

    runner, model, capture_calls, graphs = _engram_execute_harness(
        monkeypatch,
        staged=True,
        capture_error=_CaptureFailedError("capture sentinel"),
    )
    input_ids, positions, metadata = _engram_execute_inputs()

    with pytest.raises(_CaptureFailedError, match="capture sentinel"):
        runner.execute(input_ids, positions, metadata)

    assert len(capture_calls) == 1
    assert graphs.keys_written == [], "a failed capture must not register anything"
    assert runner._graphs == {}


def test_acl_runner_execute_recaptures_after_engram_staging_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute() recovery: once staging succeeds the bucket is re-captured.

    The failing first step must register nothing, so the next step takes the
    first-capture path again (a fresh staging attempt) and captures the
    Engram-baked graph. Registering before the capture would leave the refused
    entry cached, so the second step would take the replay branch and never
    re-capture.
    """

    runner, model, capture_calls, graphs = _engram_execute_harness(monkeypatch, staged=False)
    input_ids, positions, metadata = _engram_execute_inputs()

    with pytest.raises(RuntimeError):
        runner.execute(input_ids, positions, metadata)
    assert capture_calls == []
    assert graphs.keys_written == []

    model.staged = True
    with pytest.raises(_StoppedAfterCaptureError):
        runner.execute(input_ids, positions, metadata)

    assert len(capture_calls) == 1
    graph_key = graphs.keys_written[0]
    assert graph_key in runner._graphs
    assert len(runner._graphs) == 1
    assert graphs.keys_written == [graph_key]


@pytest.mark.parametrize("ring_blocks", [1, 2])
def test_dsa_graph_swa_capacity_padding_preserves_ring_addresses(ring_blocks: int) -> None:
    """Graph capacity must not turn positions after the ring into block zero."""
    caches_info, group_infos = build_cache_specs_v41(
        compress_ratios=[0], kv_source_layer_ids=[], index_source_layer_ids=[], swa_block_size=128, n_layers=1
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=1024,
    )
    runner._max_blocks_per_sequence = 8
    tables = runner._build_static_multi_block_tables(4, torch.device("cpu"))
    source = torch.arange(7, 7 + ring_blocks, dtype=torch.int32).view(1, -1)
    runner._fill_dsa_block_tables(
        SimpleNamespace(multi_block_tables=tables),
        SimpleNamespace(multi_block_tables=(source,)),
        torch.empty(0, dtype=torch.int32),
        batch_size=1,
    )
    for position in [127, 128, 255, 256, 511, 512, 1023]:
        builder = DsaMetadataBuilder(caches_info, group_infos, read_table_layout=READ_TABLE_LAYOUT_RING)
        dsa = builder.build(
            multi_block_tables=list(tables),
            kv_seq_lens=[position + 1, 1, 1, 1],
            q_seq_lens=[1] * 4,
            positions=torch.tensor([position, 0, 0, 0]),
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=False,
            enable_graph=True,
            graph_block_table_capacity_cols=8,
        )
        expected_block = source[0, (position // 128) % ring_blocks]
        assert dsa.slot_mappings[0][0][0] == expected_block * 128 + position % 128
        assert dsa.block_tables[0][0][0, position // 128] == expected_block
    assert torch.all(tables[0][1:] == 0)


def test_v41_compressor_admits_capacity_wide_graph_swa_table() -> None:
    """The fused compressor gate admits the ACL-graph capacity-wide SWA table.

    The graph runner feeds the builder a SWA manager padded to the bucket
    capacity (``_build_static_multi_block_tables``), then ``_fill_dsa_block_tables``
    repeats the width-1 scheduler ring table across the capacity. The gate must key
    on the config ring count (1), never on that padded width, so column 0 stays
    the sequence's ring block throughout.
    """
    cap = 16
    caches_info, group_infos = build_cache_specs_v41(
        compress_ratios=[0, 2],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        swa_block_size=128,
        n_layers=2,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    runner._max_blocks_per_sequence = cap

    static_tables = runner._build_static_multi_block_tables(4, torch.device("cpu"))
    swa_static = static_tables[0]
    assert tuple(swa_static.shape) == (4, cap)

    ring = torch.tensor([[7], [9], [11], [13]], dtype=torch.int32)  # width-1 source
    token_src = torch.zeros((4, 1), dtype=torch.int32)
    runner._fill_dsa_block_tables(
        SimpleNamespace(multi_block_tables=static_tables),
        SimpleNamespace(multi_block_tables=(ring, token_src)),
        torch.empty(0, dtype=torch.int32),
        batch_size=4,
    )

    # Every capacity column retains the physical ring's addressing semantics.
    assert swa_static[:, 0].tolist() == [7, 9, 11, 13]
    torch.testing.assert_close(swa_static, ring.expand_as(swa_static))

    builder = DsaMetadataBuilder(caches_info, group_infos)
    dsa = builder.build(
        multi_block_tables=list(static_tables),
        kv_seq_lens=[5, 5, 5, 5],
        q_seq_lens=[1, 1, 1, 1],
        positions=torch.arange(4, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        enable_graph=True,
        graph_block_table_capacity_cols=cap,
    )

    swa_idx = 2  # Layer 1's SWA entry (cache order: key, index, swa, kv_state...).
    read_table = dsa.block_tables[1][swa_idx]
    assert read_table[:, 0].tolist() == ring[:, 0].tolist()

    block_table, reason = _v41_compressor_swa_gate(dsa, 1, swa_idx, 1)
    assert block_table is read_table and reason is None
    _, reason = _v41_compressor_swa_gate(dsa, 1, swa_idx, 2)
    assert reason == "swa ring blocks per sequence 2 != 1"
    _, reason = _v41_compressor_swa_gate(dsa, -1, 0, 1)
    assert reason == "no swa block table"


def test_v41_engram_lookback_refreshes_graph_buffer_without_changing_address() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.engram_lookback_token_ids = torch.tensor([[4, 3], [8, 7], [12, 11], [16, 15]], dtype=torch.int32)
    metadata.engram_lookback_positions = torch.tensor([5, 9, 13, 17], dtype=torch.int32)
    input_ids = torch.arange(4, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    history = entry.static_metadata.engram_lookback_token_ids
    assert history is not None
    address = history.data_ptr()
    history_positions = entry.static_metadata.engram_lookback_positions
    positions_address = history_positions.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
        torch.testing.assert_close(history[:4], metadata.engram_lookback_token_ids)
        torch.testing.assert_close(history_positions[:4], metadata.engram_lookback_positions)
        assert torch.all(history_positions[4:] == -1)
        assert torch.all(history[4:] == -1)
        metadata.engram_lookback_token_ids = torch.full((4, 2), 42, dtype=torch.int32)
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
        assert torch.all(history[:4] == 42)
        assert torch.all(history[4:] == -1)
        metadata.engram_lookback_positions = torch.tensor([7, 11, 15, 19], dtype=torch.int32)
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
        torch.testing.assert_close(history_positions[:4], metadata.engram_lookback_positions)
        del metadata.engram_lookback_positions
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
        assert torch.all(history_positions == -1)
    assert history.data_ptr() == address
    assert history_positions.data_ptr() == positions_address


@pytest.mark.parametrize("shape", [(3,), (4, 1), (9,)])
def test_v41_engram_graph_replay_rejects_misaligned_history_positions(shape: tuple[int, ...]) -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.engram_lookback_token_ids = torch.ones(4, 2, dtype=torch.int32)
    metadata.engram_lookback_positions = torch.ones(shape, dtype=torch.int32)
    input_ids = torch.arange(4, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(ValueError, match="one position per history row"),
    ):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)


@pytest.mark.parametrize("rows", [8, 9])
def test_v41_engram_graph_replay_checks_history_batch_capacity(rows: int) -> None:
    runner = _runner()
    runner.model.cfg = SimpleNamespace(engram_layer_ids=[1], engram_max_ngram_size=3)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    input_ids = torch.arange(4, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    metadata.engram_lookback_token_ids = torch.full((rows, 2), 42, dtype=torch.int32)
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        if rows > 8:
            with pytest.raises(ValueError, match="Engram lookback batch exceeds the captured graph batch"):
                runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
        else:
            runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
            torch.testing.assert_close(
                entry.static_metadata.engram_lookback_token_ids, metadata.engram_lookback_token_ids
            )


@pytest.mark.parametrize("engram_enabled", [False, True])
def test_v41_engram_graph_warmup_without_history_allocates_only_when_enabled(engram_enabled: bool) -> None:
    runner = _runner()
    runner.model.cfg = SimpleNamespace(engram_layer_ids=[1] if engram_enabled else [], engram_max_ngram_size=3)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    input_ids = torch.arange(4, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    history = entry.static_metadata.engram_lookback_token_ids
    if not engram_enabled:
        assert history is None
        return
    assert history is not None and history.shape == (8, 2)
    address = history.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
        assert torch.all(history == -1)
        metadata.engram_lookback_token_ids = torch.full((4, 2), 42, dtype=torch.int32)
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
        assert torch.all(history[:4] == 42) and torch.all(history[4:] == -1)
    assert history.data_ptr() == address


@pytest.mark.parametrize("shape", [(4,), (4, 1), (4, 3)])
def test_v41_engram_graph_allocation_rejects_invalid_history_shape(shape: tuple[int, ...]) -> None:
    runner = _runner()
    runner.model.cfg = SimpleNamespace(engram_layer_ids=[1], engram_max_ngram_size=3)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.engram_lookback_token_ids = torch.zeros(shape, dtype=torch.int32)
    input_ids = torch.arange(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="Engram lookback"):
        runner._allocate_entry(8, input_ids, input_ids, metadata)


@pytest.mark.parametrize("shape", [(4,), (4, 1), (4, 3)])
def test_v41_engram_graph_replay_rejects_invalid_history_shape(shape: tuple[int, ...]) -> None:
    runner = _runner()
    runner.model.cfg = SimpleNamespace(engram_layer_ids=[1], engram_max_ngram_size=3)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    input_ids = torch.arange(4, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    metadata.engram_lookback_token_ids = torch.zeros(shape, dtype=torch.int32)
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(ValueError, match="Engram lookback depth"),
    ):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)


def test_metadata_uses_async_upload() -> None:
    runner = _runner()
    source = MagicMock()
    source.device = torch.device("cpu")
    source.to.return_value = source
    source.contiguous.return_value = source
    source.pin_memory.return_value = source
    device = MagicMock(type="npu")

    runner._stage_metadata_tensor(source, device, torch.int32)
    source.pin_memory.assert_called_once_with(device="npu")
    source.to.assert_any_call(device=device, dtype=torch.int32, non_blocking=True)


def test_metadata_staging_preserves_values_dtype_and_layout_on_cpu() -> None:
    runner = _runner()
    source = torch.arange(12, dtype=torch.int64).reshape(3, 4).t()
    expected = source.to(dtype=torch.int32).contiguous()
    with patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("CPU copies do not need pinning")):
        actual = runner._stage_metadata_tensor(source, torch.device("cpu"), torch.int32)
    assert actual.is_contiguous()
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("async_graph_inputs", [False, True])
def test_dsa_host_tables_are_staged_before_device_copy(async_graph_inputs: bool) -> None:
    runner = _runner()
    runner.supports_draft_to_target_overlap = async_graph_inputs
    source = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    target = MagicMock()
    target.shape = (4, 4)
    target.device = MagicMock(type="npu")
    target.dtype = torch.int32
    staged = MagicMock()
    staged.shape = source.shape
    runner._stage_block_table = MagicMock(return_value=staged)

    runner._fill_dsa_block_tables(
        SimpleNamespace(multi_block_tables=(target,)),
        SimpleNamespace(multi_block_tables=(source,)),
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    runner._stage_block_table.assert_called_once_with(source, target.device)
    target.__getitem__.return_value.copy_.assert_called_once_with(staged.__getitem__.return_value)


def test_row_aligned_paging_stages_all_host_metadata() -> None:
    runner = _runner()
    runner.attention_backend.page_size = 128
    block_table = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)

    with patch.object(runner, "_stage_metadata_tensor", wraps=runner._stage_metadata_tensor) as stage:
        indptr, indices, last_page_len = runner._build_row_aligned_paged_kv_metadata(
            block_table,
            [0, 128, 129],
        )

    assert stage.call_count == 3
    for call, dtype in zip(stage.call_args_list, (torch.int64, torch.int32, torch.int32)):
        source, device, target_dtype = call.args
        assert source.device.type == "cpu"
        assert source.dtype == dtype
        assert device == block_table.device
        assert target_dtype == dtype
    assert indptr.dtype == indices.dtype == last_page_len.dtype == torch.int32
    assert indptr.tolist() == [0, 1, 2, 4]
    assert indices.tolist() == [10, 20, 30, 31]
    assert last_page_len.tolist() == [1, 128, 1]


@pytest.mark.parametrize("dp_size", [1, 2])
def test_graph_admission_preserves_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
    dp_size: int,
) -> None:
    runner = _runner()
    runner.dp_size = dp_size
    monkeypatch.setattr(runner, "_can_execute_inner", MagicMock(return_value=False))
    admission = MagicMock(side_effect=AssertionError("Rejected geometry must skip model admission"))
    monkeypatch.setattr(runner.model, "can_execute_graph", admission, raising=False)
    agreement = MagicMock(return_value=False)
    monkeypatch.setattr(distributed, "dp_all_true", agreement, raising=False)
    batch = _query_batch(4, 1)

    assert not runner.can_execute(ModelInput(batch.input_ids, batch.positions, _metadata(None), input_batch=batch))
    admission.assert_not_called()
    assert agreement.call_count == (1 if dp_size > 1 else 0)


@pytest.mark.parametrize("dummy", [False, True])
def test_dsa_derived_tables_skip_intermediate_device_uploads(dummy: bool) -> None:
    _, groups = build_cache_specs([1, 4, 128], 128, 3)
    backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=groups,
        graph_uses_dsa_metadata=lambda metadata: True,
        supports_draft_to_target_overlap=True,
    )
    runner = DecodeAclGraphRunner(nn.Identity(), backend, torch.device("cpu"), max_batch=4, max_model_len=512)
    static = SimpleNamespace(multi_block_tables=tuple(torch.full((4, 4), 99, dtype=torch.int32) for _ in range(2)))
    tables = tuple(torch.tensor([[10 + idx, 20 + idx]], dtype=torch.int32) for idx in range(2))
    metadata = SimpleNamespace(multi_block_tables=() if dummy else tables, is_dummy=dummy)
    runner._stage_block_table = lambda *args: pytest.fail("intermediate table was uploaded")
    runner._fill_dsa_block_tables(static, metadata, torch.empty(0, dtype=torch.int32), batch_size=2)
    for idx, table in enumerate(static.multi_block_tables_host):
        assert torch.all(static.multi_block_tables[idx] == 99)
        if dummy:
            assert torch.count_nonzero(table) == 0
        elif idx == 0:
            assert table.tolist() == [[10, 20, 10, 20]] * 2 + [[0] * 4] * 2
        else:
            assert table.tolist() == [[10 + idx, 20 + idx, 0, 0]] * 2 + [[0] * 4] * 2


def test_host_dsa_fill_rejects_expanded_kv_length_beyond_table_capacity() -> None:
    runner = _runner()
    metadata = _metadata(torch.ones(4, dtype=torch.int32))
    input_ids = torch.arange(4)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    runner.supports_draft_to_target_overlap = True
    runner.attention_backend.graph_uses_dsa_metadata = lambda _: True
    metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
        kv_seq_lens=torch.full((4,), 5, dtype=torch.int32),
        block_table=torch.zeros((4, 1), dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[5] * 4,
    )
    with pytest.raises(RuntimeError, match="page count exceeds block-table capacity"):
        runner._fill_entry(entry, input_ids, input_ids, metadata, 4, None)


def test_multimodal_rejection_skips_graph_layout_but_participates_in_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(_SamplingModel())
    runner.dp_size = 2
    layout = MagicMock(side_effect=AssertionError("Multimodal eager inputs do not require graph metadata"))
    monkeypatch.setattr(runner, "_can_execute_inner", layout)
    agreement = MagicMock(return_value=False)
    monkeypatch.setattr(distributed, "dp_all_true", agreement, raising=False)
    assert not runner.can_execute(
        ModelInput(
            torch.ones(1),
            torch.ones(1),
            SimpleNamespace(),
            multimodal_inputs=SimpleNamespace(),
        )
    )
    layout.assert_not_called()
    agreement.assert_called_once_with(False, torch.device("cpu"), 2)
