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

"""Non-expanded (V4.1 DSpark) packed spec-verify graph admission.

A DSpark verify step packs ``num_spec+1`` token rows per logical sequence and
arrives decode-typed: V4.1 has no linear-attention layers, so it exports no
``linear_state_indices`` and ``is_spec_verify`` stays False. The C++ validate
builder exports the true packed width on ``AttentionMetadata.num_speculative_tokens``
so the runner can fold token rows back to sequences and admit the batch under
DP>1 without confusing it with a plain decode of the same row count.

The packed-verify admission is unconditional: whenever the exported
``num_speculative_tokens`` identifies a packed verification batch, the runner folds token rows
back to sequences.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
from xllm.python.model_executor.runners.decode_cuda_graph import _decode_bucket
from xllm.python.models.deepseek_v41 import DeepseekV41Model
from xllm.python.models.deepseek_v41_dspark import DeepseekV41DSparkModel


def _runner(
    *,
    dp_size: int = 1,
    dp_rank: int = 0,
    max_seqs_per_batch: int = 16,
    num_speculative_tokens: int = 5,
    limit: int | None = None,
    model_type: str = "",
) -> DecodeAclGraphRunner:
    return DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False),
        torch.device("cpu"),
        max_batch=max_seqs_per_batch,
        max_model_len=65536,
        dp_size=dp_size,
        dp_rank=dp_rank,
        decode_batch_size_limit=limit,
        num_speculative_tokens=num_speculative_tokens,
        is_spec_draft=bool(model_type),
        model_type=model_type,
    )


def _metadata(
    rows: int,
    *,
    num_speculative_tokens: int = 5,
    counts: tuple[int, ...] | None = None,
    dp_size: int = 1,
    per_sequence_tensors: bool = False,
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=False,
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        new_cache_slots_host_values=[],
        paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32),
        paged_kv_indices=torch.arange(rows, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(rows, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        kv_seq_lens_host_values=[1] * rows,
        max_seq_len=1,
        q_cu_seq_lens=torch.arange(rows + 1, dtype=torch.int32),
        kv_cu_seq_lens=None,
        linear_state_indices=None,
        has_initial_state=None,
        expanded_decode_metadata=None,
        multi_block_tables=(),
        kpool_query_lens=(),
        dp_execution_token_counts=counts if counts is not None else (rows,) * dp_size,
        dp_is_decode=(True,) * dp_size,
    )
    if per_sequence_tensors:
        # A packed verify carries per-LOGICAL-SEQUENCE state tensors even though
        # its paging metadata is per token row (the on-hardware V4.1 shape).
        seqs = rows // (num_speculative_tokens + 1)
        metadata.linear_state_indices = torch.arange(seqs, dtype=torch.int64)
        metadata.has_initial_state = torch.ones(seqs, dtype=torch.int32)
        metadata.num_accepted_tokens = torch.ones(seqs, dtype=torch.int32)
    metadata.num_speculative_tokens = num_speculative_tokens
    return metadata


# --------------------------------------------------------------------------
# Exported width folds rows to sequences symmetrically across DP
# --------------------------------------------------------------------------


def test_active_and_idle_ranks_fold_to_the_same_graph_bucket() -> None:
    counts = (36, 1, 1, 1)
    decisions = []
    for rank in range(4):
        rows = counts[rank]
        runner = _runner(dp_size=4, dp_rank=rank, max_seqs_per_batch=64)
        assert runner.max_batch == 64  # merged-tree scheduler publishes the full seq budget (no /dp_size)
        metadata = _metadata(rows, num_speculative_tokens=5, counts=counts, dp_size=4)
        row_tokens = torch.zeros(rows, dtype=torch.int32)
        decisions.append(runner.can_execute(ModelInput(row_tokens, torch.zeros_like(row_tokens), metadata)))
    assert decisions == [True, True, True, True]


def test_per_sequence_state_tensors_admit_a_packed_verify() -> None:
    # The on-hardware V4.1 verify batch carries one linear_state_indices /
    # has_initial_state / num_accepted_tokens entry per LOGICAL sequence while
    # its paging metadata is per token row; admission must fold those too.
    counts = (96, 96, 96, 96)
    runner = _runner(dp_size=4, dp_rank=0, max_seqs_per_batch=64)
    metadata = _metadata(96, num_speculative_tokens=5, counts=counts, dp_size=4, per_sequence_tensors=True)
    assert metadata.linear_state_indices.numel() == 16
    tokens = torch.zeros(96, dtype=torch.int32)
    assert runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata)) is True


def test_per_sequence_state_tensors_do_not_admit_a_plain_decode() -> None:
    # Without an exported packed width the per-sequence state tensors are not a
    # verify signal: the historical width-1 decision (eager under DP>1) holds.
    runner = _runner(dp_size=4, dp_rank=0, max_seqs_per_batch=64)
    metadata = _metadata(96, num_speculative_tokens=0, counts=(96, 96, 96, 96), dp_size=4, per_sequence_tensors=False)
    tokens = torch.zeros(96, dtype=torch.int32)
    assert runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata)) is False


def test_query_pack_row_mismatch_falls_back_without_raising() -> None:
    # A recurrent spec-verify step arrives chunked-prefill-typed with
    # is_spec_verify set (see DFlashWorkerImpl::run_validate) and keeps one
    # paging row per request, not per query token. The incompatible graph
    # shape must select eager, never abort ranks. Either flag alone stays
    # under the strict row check (expanded verify / plain chunked prefill).
    runner = _runner(dp_size=2, dp_rank=1, max_seqs_per_batch=16)
    metadata = _metadata(5, num_speculative_tokens=0, counts=(5, 5), dp_size=2)
    metadata.is_spec_verify = True
    metadata.is_chunked_prefill = True
    metadata.block_table = torch.zeros((1, 1), dtype=torch.int32)
    metadata.kv_seq_lens = torch.ones(1, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [1]
    tokens = torch.zeros(5, dtype=torch.int32)

    assert runner._has_compatible_decode_metadata(tokens, metadata, dp_all_decode=True) is False


# --------------------------------------------------------------------------
# Plain decode (exported width 1) must never change its decision
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16])
def test_plain_decode_inside_capacity_is_unchanged(rows: int) -> None:
    runner = _runner(max_seqs_per_batch=16, num_speculative_tokens=5)
    expected = _decode_bucket(rows) <= runner.max_batch
    metadata = _metadata(rows, num_speculative_tokens=0)
    tokens = torch.zeros(rows, dtype=torch.int32)
    assert runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata)) is expected


def test_width_one_keeps_the_gate_blocked() -> None:
    # A decode-typed batch the C++ marks as width 1 is not a verifiable pack:
    # the historical DP+MTP gate still rejects it.
    runner = _runner(dp_size=4, dp_rank=0, max_seqs_per_batch=16, num_speculative_tokens=5)
    metadata = _metadata(4, num_speculative_tokens=0, counts=(4, 4, 4, 4), dp_size=4)
    tokens = torch.zeros(4, dtype=torch.int32)
    assert runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata)) is False


# --------------------------------------------------------------------------
# The graph key partitions the width
# --------------------------------------------------------------------------


def test_graph_key_separates_plain_decode_from_verify() -> None:
    key_plain = DecodeAclGraphRunner._graph_key(48, False, None, (), verify_width=1)
    key_verify = DecodeAclGraphRunner._graph_key(48, False, None, (), verify_width=6)
    assert key_plain != key_verify


@pytest.mark.parametrize("rows,expected", [(6, 6), (18, 18), (30, 30), (48, 48), (54, 64)])
def test_v41_small_verify_buckets_preserve_complete_query_groups(rows: int, expected: int) -> None:
    runner = _runner()
    runner._model_decode_graph_bucket = DeepseekV41Model.decode_graph_token_bucket
    assert runner._padded_token_rows(rows, verify_width=6) == expected
    assert runner._padded_token_rows(rows, verify_width=1) == _decode_bucket(rows)
    assert runner._engram_graph_token_capacity() >= expected


def test_v41_verify_bucket_is_shared_with_empty_dp_ranks() -> None:
    counts = (18, 1, 12, 6)
    buckets = []
    for rank, rows in enumerate(counts):
        runner = _runner(dp_size=4, dp_rank=rank)
        runner._model_decode_graph_bucket = DeepseekV41Model.decode_graph_token_bucket
        metadata = _metadata(rows, counts=counts, dp_size=4)
        _, width, admission = runner._shared_graph_plan(metadata, rows)
        assert admission == 3
        buckets.append(runner._padded_token_rows(max(counts), width))
    assert buckets == [18] * 4


@pytest.mark.parametrize("limit,expected", [(19, False), (20, True), (31, True)])
@pytest.mark.parametrize("dp_size", [1, 4])
def test_v41_verify_window_admission_uses_the_capture_bucket(limit: int, expected: bool, dp_size: int) -> None:
    counts = (20,) if dp_size == 1 else (20, 1, 10, 5)
    decisions = []
    for rank, rows in enumerate(counts):
        runner = _runner(dp_size=dp_size, dp_rank=rank, num_speculative_tokens=4)
        runner._model_decode_graph_bucket = DeepseekV41Model.decode_graph_token_bucket
        runner._eplv2_graph_token_limit = limit
        metadata = _metadata(rows, num_speculative_tokens=4, counts=counts, dp_size=dp_size)
        _, width, _ = runner._shared_graph_plan(metadata, rows)
        assert runner._padded_token_rows(max(counts), width) == 20
        assert _decode_bucket(max(counts)) == 32
        tokens = torch.zeros(rows, dtype=torch.int32)
        decisions.append(runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata)))
    assert decisions == [expected] * dp_size


def test_generic_verify_window_keeps_the_generic_capture_bucket() -> None:
    runner = _runner(num_speculative_tokens=4)
    runner._eplv2_graph_token_limit = 20
    metadata = _metadata(20, num_speculative_tokens=4)
    tokens = torch.zeros(20, dtype=torch.int32)
    assert not runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata))


@pytest.mark.parametrize("limit,expected", [(20, False), (30, True)])
def test_explicit_capture_sizes_override_v41_small_buckets(limit: int, expected: bool) -> None:
    runner = _runner(num_speculative_tokens=4)
    runner._model_decode_graph_bucket = DeepseekV41Model.decode_graph_token_bucket
    runner._capture_sizes = (10, 30)
    runner._eplv2_graph_token_limit = limit
    metadata = _metadata(20, num_speculative_tokens=4)
    tokens = torch.zeros(20, dtype=torch.int32)
    assert runner._padded_token_rows(20, verify_width=5) == 30
    assert runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata)) is expected


def test_model_bucket_does_not_change_other_models() -> None:
    runner = _runner()
    assert runner._padded_token_rows(18, verify_width=6) == 32
    runner = _runner(model_type="DSparkDraftModel")
    assert runner._padded_token_rows(15, verify_width=6) == 20


@pytest.mark.parametrize("sequences", [1, 2, 3, 5, 7, 8, 9])
def test_v41_draft_bucket_uses_draft_query_width(sequences: int) -> None:
    runner = _runner(num_speculative_tokens=6, model_type="deepseek_v41_dspark")
    runner._model_decode_graph_bucket = DeepseekV41DSparkModel.decode_graph_token_bucket
    expected = sequences * 6 if sequences <= 8 else _decode_bucket(sequences) * 6
    # Draft carries per-token metadata (verify width 1), while the model must
    # retain complete six-query blocks and use the same bucket on every DP rank.
    assert runner._padded_token_rows(sequences * 6, verify_width=1) == expected
    assert runner._engram_graph_token_capacity() >= expected


def test_v41_draft_active_and_empty_ranks_share_the_same_bucket() -> None:
    counts = (30, 1, 18, 12)
    for rank, rows in enumerate(counts):
        runner = _runner(dp_size=4, dp_rank=rank, num_speculative_tokens=6, model_type="deepseek_v41_dspark")
        runner._model_decode_graph_bucket = DeepseekV41DSparkModel.decode_graph_token_bucket
        metadata = _metadata(rows, num_speculative_tokens=0, counts=counts, dp_size=4)
        metadata.raw_dp_execution_token_counts = (30, 0, 18, 12)
        expanded, width, admission = runner._shared_graph_plan(metadata, rows)
        assert not expanded
        assert width == 1
        assert admission == 5
        assert runner._padded_token_rows(max(counts), width) == 30


@pytest.mark.parametrize("limit,expected", [(29, False), (30, True)])
@pytest.mark.parametrize("dp_size", [1, 4])
def test_v41_draft_window_admission_uses_query_blocks(limit: int, expected: bool, dp_size: int) -> None:
    counts = (30,) if dp_size == 1 else (30, 1, 18, 12)
    for rank, rows in enumerate(counts):
        runner = _runner(dp_size=dp_size, dp_rank=rank, num_speculative_tokens=6, model_type="deepseek_v41_dspark")
        runner._model_decode_graph_bucket = DeepseekV41DSparkModel.decode_graph_token_bucket
        runner._eplv2_graph_token_limit = limit
        metadata = _metadata(rows, num_speculative_tokens=0, counts=counts, dp_size=dp_size)
        metadata.raw_dp_execution_token_counts = tuple(0 if count == 1 else count for count in counts)
        tokens = torch.zeros(rows, dtype=torch.int32)
        assert runner.can_execute(ModelInput(tokens, torch.zeros_like(tokens), metadata)) is expected
