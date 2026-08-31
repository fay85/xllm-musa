# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for C++-aligned MTP verify attention expansion."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

import xllm.python as python_pkg

_mock_kernels = MagicMock()
python_pkg.kernels = _mock_kernels
python_pkg._runtime_kernels = _mock_kernels
sys.modules["xllm.python.kernels"] = _mock_kernels
sys.modules.setdefault("xllm.python.layers.gated_delta_net", MagicMock())

from xllm.python.model_executor.runners.decode_musa_graph import (  # noqa: E402
    _can_expand_spec_verify,
    _conv_cache_state_len,
    _decode_bucket,
    _decode_graph_batch_size,
    _expand_spec_verify_attention,
    _has_prepared_expanded_attention,
    _input_embedding_matches_tokens,
    _is_fused_decode_conv_cache,
    _is_supported_prefill_conv_cache,
    _spec_verify_sequence_count,
)
from xllm.python.model_executor.executor import (  # noqa: E402
    _decode_graph_max_tokens,
)


@dataclass
class _VerifyMetadata:
    paged_kv_last_page_len: torch.Tensor
    q_cu_seq_lens: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    use_expanded_decode_for_spec_verify_attention: bool = False
    expanded_kv_seq_lens: torch.Tensor | None = None
    expanded_block_table: torch.Tensor | None = None
    expanded_kv_seq_lens_host: torch.Tensor | None = None
    expanded_paged_kv_indptr: torch.Tensor | None = None
    expanded_paged_kv_indices: torch.Tensor | None = None
    expanded_paged_kv_last_page_len: torch.Tensor | None = None
    max_seq_len: int = 0
    linear_state_indices: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None


def test_decode_bucket_rounds_regular_decode_width() -> None:
    assert _decode_bucket(2) == 2
    assert _decode_bucket(4) == 4
    assert _decode_bucket(6) == 8


def test_decode_graph_batch_size_keeps_verify_width_exact() -> None:
    assert _decode_graph_batch_size(3, is_spec_verify=True) == 3
    assert _decode_graph_batch_size(12, is_spec_verify=True) == 12
    assert _decode_graph_batch_size(6, is_spec_verify=False) == 8


def test_decode_graph_capacity_covers_all_verify_tokens() -> None:
    assert _decode_graph_max_tokens(1, {"num_speculative_tokens": 2}) == 8
    assert _decode_graph_max_tokens(4, {"num_speculative_tokens": 2}) == 12


def test_expand_spec_verify_matches_cpp_row_layout() -> None:
    metadata = _VerifyMetadata(
        paged_kv_last_page_len=torch.tensor([36], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        kv_seq_lens=torch.tensor([100], dtype=torch.int32),
        block_table=torch.tensor([[7, 8, 9], [1, 1, 1]], dtype=torch.int32)[:1],
        max_seq_len=100,
    )
    assert _can_expand_spec_verify(metadata, 2)
    assert not _has_prepared_expanded_attention(metadata, 2)

    expanded = _expand_spec_verify_attention(
        metadata, 2, page_size=64, device=torch.device("cpu")
    )
    assert expanded.kv_seq_lens.tolist() == [99, 100]
    assert expanded.kv_cu_seq_lens.tolist() == [0, 99, 199]
    assert expanded.block_table.tolist() == [[7, 8, 9], [7, 8, 9]]
    assert expanded.paged_kv_indptr.tolist() == [0, 2, 4]
    assert expanded.paged_kv_indices.tolist() == [7, 8, 7, 8]
    assert expanded.paged_kv_last_page_len.tolist() == [35, 36]
    assert expanded.max_seq_len == 100


def test_prepared_expanded_attention_is_preferred() -> None:
    metadata = _VerifyMetadata(
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        kv_seq_lens=torch.tensor([4], dtype=torch.int32),
        block_table=torch.tensor([[1, 2]], dtype=torch.int32),
        use_expanded_decode_for_spec_verify_attention=True,
        expanded_kv_seq_lens=torch.tensor([3, 4], dtype=torch.int32),
        expanded_block_table=torch.tensor([[5, 6], [5, 6]], dtype=torch.int32),
        expanded_paged_kv_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        expanded_paged_kv_indices=torch.tensor([5, 5], dtype=torch.int32),
        expanded_paged_kv_last_page_len=torch.tensor([3, 4], dtype=torch.int32),
        expanded_kv_seq_lens_host=torch.tensor([3, 4], dtype=torch.int32),
        max_seq_len=4,
    )
    assert _has_prepared_expanded_attention(metadata, 2)
    expanded = _expand_spec_verify_attention(
        metadata, 2, page_size=64, device=torch.device("cpu")
    )
    assert expanded.kv_seq_lens.tolist() == [3, 4]
    assert expanded.paged_kv_indices.tolist() == [5, 5]
    assert expanded.max_seq_len == 4


def test_fused_decode_conv_cache_rejects_mtp_width() -> None:
    conv_dim = 8
    kernel_size = 4
    decode_conv = torch.zeros(
        (3, conv_dim, kernel_size - 1), dtype=torch.float32
    )
    mtp_conv = torch.zeros((3, conv_dim, kernel_size), dtype=torch.float32)
    assert _conv_cache_state_len(decode_conv, conv_dim) == kernel_size - 1
    assert _is_fused_decode_conv_cache(decode_conv, conv_dim, kernel_size)
    assert not _is_fused_decode_conv_cache(mtp_conv, conv_dim, kernel_size)
    assert _is_supported_prefill_conv_cache(decode_conv, conv_dim, kernel_size)
    assert _is_supported_prefill_conv_cache(mtp_conv, conv_dim, kernel_size)


def test_spec_verify_sequence_count_prefers_gdn_slots() -> None:
    metadata = _VerifyMetadata(
        paged_kv_last_page_len=torch.tensor([35, 36], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        kv_seq_lens=torch.tensor([100], dtype=torch.int32),
        block_table=torch.tensor([[7, 8, 9]], dtype=torch.int32),
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )
    assert _spec_verify_sequence_count(metadata, 2) == 1
    assert _can_expand_spec_verify(metadata, 2)
    expanded = _expand_spec_verify_attention(
        metadata, 2, page_size=64, device=torch.device("cpu")
    )
    assert expanded.kv_seq_lens.tolist() == [99, 100]
    assert expanded.block_table.tolist() == [[7, 8, 9], [7, 8, 9]]


def test_draft_decode_graph_keeps_matching_embedding() -> None:
    class _Meta:
        input_embedding = None

    assert _input_embedding_matches_tokens(_Meta(), 1) is True
    _Meta.input_embedding = torch.ones(1, 8)
    assert _input_embedding_matches_tokens(_Meta(), 1) is True
    assert _input_embedding_matches_tokens(_Meta(), 2) is False
