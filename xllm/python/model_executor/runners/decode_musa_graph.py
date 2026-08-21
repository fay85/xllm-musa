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

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.runners.base import BaseRunner
from scripts.logger import logger

_CAPTURE_WARMUP_STEPS = 2


def _live_input_embedding(
    metadata: AttentionMetadata,
) -> torch.Tensor | None:
    live = getattr(metadata, "input_embedding", None)
    if live is None or int(live.numel()) <= 0:
        return None
    return live


def _input_embedding_matches_tokens(
    metadata: AttentionMetadata, num_tokens: int
) -> bool:
    live = _live_input_embedding(metadata)
    if live is None:
        return True
    return live.dim() == 2 and int(live.size(0)) == int(num_tokens)


def _ensure_static_input_embedding(
    current: torch.Tensor | None,
    live: torch.Tensor | None,
    num_rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Copy MTP target hidden into a persistent graph buffer.

    Capture and replay must keep the same storage so the in-graph
    ``metadata.input_embedding`` pointer stays valid.
    """
    if live is None:
        return current
    if live.dim() != 2:
        raise ValueError("input_embedding must be [tokens, hidden]")
    if int(live.size(0)) != num_rows:
        raise ValueError(
            "input_embedding rows must match live tokens "
            f"({int(live.size(0))} vs {num_rows})"
        )
    hidden = int(live.size(1))
    if current is None:
        current = torch.empty(
            (num_rows, hidden),
            dtype=live.dtype,
            device=device,
        )
    if (
        int(current.size(0)) != num_rows
        or int(current.size(1)) != hidden
        or current.dtype != live.dtype
    ):
        raise ValueError("input_embedding graph buffer shape/dtype changed")
    current.copy_(live)
    return current


def _decode_bucket(batch_size: int) -> int:
    if batch_size <= 1:
        return 1
    if batch_size <= 2:
        return 2
    if batch_size <= 4:
        return 4
    if batch_size <= 8:
        return 8
    return ((batch_size + 15) // 16) * 16


def _decode_graph_batch_size(token_count: int, is_spec_verify: bool) -> int:
    if is_spec_verify:
        return token_count
    return _decode_bucket(token_count)


def _conv_cache_state_len(conv: torch.Tensor, conv_dim: int) -> int:
    """Return the cached convolution history width for a GDN conv tensor."""
    if conv.dim() != 3:
        raise ValueError("linear-attention conv cache must be 3D")
    if int(conv.size(1)) == conv_dim:
        return int(conv.size(2))
    if int(conv.size(2)) == conv_dim:
        return int(conv.size(1))
    raise ValueError("linear-attention conv cache has an unexpected shape")


def _is_fused_decode_conv_cache(
    conv: torch.Tensor, conv_dim: int, kernel_size: int
) -> bool:
    """True when conv history is ``kernel-1``, the fused decode layout.

    MTP expands the same cache to ``history + K``, which
    ``causal_conv1d_decode_fused`` rejects. Plain decode without MTP keeps
    the fused width and can be captured like the C++ MusaGraph path.
    """
    return _conv_cache_state_len(conv, conv_dim) == kernel_size - 1


def _is_supported_prefill_conv_cache(
    conv: torch.Tensor, conv_dim: int, kernel_size: int
) -> bool:
    """Prefill writes the first ``kernel-1`` slots.

    MTP allocates ``history + K``. Extra slots stay unused until verify.
    """
    return _conv_cache_state_len(conv, conv_dim) >= kernel_size - 1


def _as_int32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device != device or tensor.dtype != torch.int32:
        return tensor.to(device=device, dtype=torch.int32).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _int_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _cumulative_int32(lengths: torch.Tensor) -> torch.Tensor:
    cumulative = torch.empty(
        lengths.numel() + 1, dtype=torch.int32, device=lengths.device
    )
    cumulative[0] = 0
    cumulative[1:].copy_(lengths.to(dtype=torch.int32).cumsum(0))
    return cumulative


def _sequence_query_widths(
    metadata: AttentionMetadata, token_count: int, sequence_count: int
) -> list[int] | None:
    query_cu = getattr(metadata, "q_cu_seq_lens", None)
    if query_cu is not None and int(query_cu.numel()) == sequence_count + 1:
        widths = _int_list(query_cu[1:] - query_cu[:-1])
        if sum(widths) == token_count and all(width > 0 for width in widths):
            return widths
    if sequence_count > 0 and token_count % sequence_count == 0:
        width = token_count // sequence_count
        if width > 0:
            return [width] * sequence_count
    return None


def _sequence_kv_lens(
    metadata: AttentionMetadata, sequence_count: int
) -> list[int] | None:
    kv_lens = getattr(metadata, "kv_seq_lens", None)
    if kv_lens is not None and int(kv_lens.numel()) == sequence_count:
        return _int_list(kv_lens)
    kv_cu = getattr(metadata, "kv_cu_seq_lens", None)
    if kv_cu is not None and int(kv_cu.numel()) == sequence_count + 1:
        return _int_list(kv_cu[1:] - kv_cu[:-1])
    host = getattr(metadata, "kv_seq_lens_host", None)
    if host is None:
        return None
    if int(host.numel()) == sequence_count:
        return _int_list(host)
    if int(host.numel()) == sequence_count + 1:
        return _int_list(host[1:] - host[:-1])
    return None


def _spec_verify_sequence_count(
    metadata: AttentionMetadata, token_count: int
) -> int | None:
    """Prefer GDN sequence slots over a token-expanded last_page_len."""
    candidates: list[int] = []
    indices = getattr(metadata, "linear_state_indices", None)
    if indices is not None and int(indices.numel()) > 0:
        candidates.append(int(indices.numel()))
    accepted = getattr(metadata, "num_accepted_tokens", None)
    if accepted is not None and int(accepted.numel()) > 0:
        accepted_count = int(accepted.numel())
        if accepted_count not in candidates:
            candidates.append(accepted_count)
    last_page_len = getattr(metadata, "paged_kv_last_page_len", None)
    if last_page_len is not None and int(last_page_len.numel()) > 0:
        last_count = int(last_page_len.numel())
        if last_count not in candidates:
            candidates.append(last_count)
    for sequence_count in candidates:
        if token_count % sequence_count != 0:
            continue
        query_widths = _sequence_query_widths(
            metadata, token_count, sequence_count
        )
        if query_widths is not None:
            return sequence_count
    return None


@dataclass(slots=True)
class _ExpandedDecodeAttention:
    kv_seq_lens: torch.Tensor
    kv_cu_seq_lens: torch.Tensor
    block_table: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    max_seq_len: int


def _has_prepared_expanded_attention(
    metadata: AttentionMetadata, token_count: int
) -> bool:
    if not bool(
        getattr(metadata, "use_expanded_decode_for_spec_verify_attention", False)
    ):
        return False
    kv_seq_lens = getattr(metadata, "expanded_kv_seq_lens", None)
    block_table = getattr(metadata, "expanded_block_table", None)
    indptr = getattr(metadata, "expanded_paged_kv_indptr", None)
    indices = getattr(metadata, "expanded_paged_kv_indices", None)
    last_page_len = getattr(metadata, "expanded_paged_kv_last_page_len", None)
    return (
        kv_seq_lens is not None
        and block_table is not None
        and indptr is not None
        and indices is not None
        and last_page_len is not None
        and int(kv_seq_lens.numel()) == token_count
        and int(last_page_len.numel()) == token_count
        and block_table.dim() == 2
        and int(block_table.size(0)) == token_count
        and int(indptr.numel()) == token_count + 1
    )


def _can_expand_spec_verify(
    metadata: AttentionMetadata, token_count: int
) -> bool:
    if _has_prepared_expanded_attention(metadata, token_count):
        return True
    sequence_count = _spec_verify_sequence_count(metadata, token_count)
    if sequence_count is None:
        return False
    query_widths = _sequence_query_widths(metadata, token_count, sequence_count)
    kv_lens = _sequence_kv_lens(metadata, sequence_count)
    block_table = getattr(metadata, "block_table", None)
    return (
        query_widths is not None
        and kv_lens is not None
        and block_table is not None
        and block_table.dim() == 2
        and int(block_table.size(0)) >= sequence_count
        and all(
            kv_len >= query_width
            for kv_len, query_width in zip(kv_lens, query_widths)
        )
    )


def _build_expanded_paged_kv(
    block_table: torch.Tensor,
    kv_seq_lens: list[int],
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if page_size <= 0:
        raise RuntimeError("expanded decode paged KV requires page_size > 0")
    table = block_table.detach().to(device="cpu", dtype=torch.int32)
    indptr = [0]
    indices: list[int] = []
    last_page_len: list[int] = []
    for row, kv_len in enumerate(kv_seq_lens):
        if kv_len <= 0:
            raise RuntimeError("expanded kv_len must be positive")
        block_count = (kv_len + page_size - 1) // page_size
        if block_count > int(table.size(1)):
            raise RuntimeError("expanded block_table is narrower than kv_len")
        indices.extend(int(block_id) for block_id in table[row, :block_count].tolist())
        indptr.append(len(indices))
        remainder = kv_len % page_size
        last_page_len.append(page_size if remainder == 0 else remainder)
    return (
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(last_page_len, dtype=torch.int32, device=device),
    )


def _expand_spec_verify_attention(
    metadata: AttentionMetadata,
    token_count: int,
    page_size: int,
    device: torch.device,
) -> _ExpandedDecodeAttention:
    """Mirror C++ `mtp_graph_buffers` expanded spec-verify decode attention."""
    if _has_prepared_expanded_attention(metadata, token_count):
        kv_seq_lens = _as_int32(metadata.expanded_kv_seq_lens, device)
        host_kv = getattr(metadata, "expanded_kv_seq_lens_host", None)
        max_seq_len = int(getattr(metadata, "max_seq_len", 0) or 0)
        if max_seq_len <= 0 and host_kv is not None and host_kv.numel() == token_count:
            max_seq_len = int(host_kv.max().item())
        if max_seq_len <= 0:
            max_seq_len = int(kv_seq_lens.max().item())
        return _ExpandedDecodeAttention(
            kv_seq_lens=kv_seq_lens,
            kv_cu_seq_lens=_cumulative_int32(kv_seq_lens),
            block_table=_as_int32(metadata.expanded_block_table, device),
            paged_kv_indptr=_as_int32(metadata.expanded_paged_kv_indptr, device),
            paged_kv_indices=_as_int32(metadata.expanded_paged_kv_indices, device),
            paged_kv_last_page_len=_as_int32(
                metadata.expanded_paged_kv_last_page_len, device
            ),
            max_seq_len=max_seq_len,
        )

    sequence_count = _spec_verify_sequence_count(metadata, token_count)
    query_widths = (
        None
        if sequence_count is None
        else _sequence_query_widths(metadata, token_count, sequence_count)
    )
    kv_lens = (
        None
        if sequence_count is None
        else _sequence_kv_lens(metadata, sequence_count)
    )
    block_table = getattr(metadata, "block_table", None)
    if (
        sequence_count is None
        or query_widths is None
        or kv_lens is None
        or block_table is None
    ):
        raise RuntimeError("MTP verify graph is missing sequence attention metadata")

    expanded_kv: list[int] = []
    expanded_rows: list[int] = []
    for sequence_index, query_width in enumerate(query_widths):
        kv_len = kv_lens[sequence_index]
        if kv_len < query_width:
            raise RuntimeError("MTP verify kv_len is shorter than query width")
        for token_index in range(query_width):
            expanded_kv.append(kv_len - query_width + token_index + 1)
            expanded_rows.append(sequence_index)
    if len(expanded_kv) != token_count:
        raise RuntimeError("expanded spec-verify rows must match token count")

    row_index = torch.tensor(
        expanded_rows, dtype=torch.long, device=block_table.device
    )
    expanded_table = _as_int32(block_table.index_select(0, row_index), device)
    paged_indptr, paged_indices, last_page_len = _build_expanded_paged_kv(
        expanded_table,
        expanded_kv,
        page_size,
        device,
    )
    kv_seq_lens = torch.tensor(expanded_kv, dtype=torch.int32, device=device)
    return _ExpandedDecodeAttention(
        kv_seq_lens=kv_seq_lens,
        kv_cu_seq_lens=_cumulative_int32(kv_seq_lens),
        block_table=expanded_table,
        paged_kv_indptr=paged_indptr,
        paged_kv_indices=paged_indices,
        paged_kv_last_page_len=last_page_len,
        max_seq_len=max(expanded_kv),
    )


@dataclass(slots=True)
class _StaticAttentionMetadata:
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None = None
    q_cu_seq_lens: torch.Tensor | None = None
    kv_cu_seq_lens: torch.Tensor | None = None
    kv_seq_lens_host: torch.Tensor | None = None
    paged_kv_indptr_host: torch.Tensor | None = None
    paged_kv_last_page_len_host: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    kv_seq_lens: torch.Tensor | None = None
    max_seq_len: int = 1
    max_query_len: int = 1
    is_prefill: bool = False
    is_chunked_prefill: bool = False
    linear_state_indices: torch.Tensor | None = None
    has_initial_state: torch.Tensor | None = None
    input_embedding: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
    is_spec_verify: bool = False
    use_expanded_decode_for_spec_verify_attention: bool = False
    expanded_kv_seq_lens: torch.Tensor | None = None
    expanded_block_table: torch.Tensor | None = None
    expanded_kv_seq_lens_host: torch.Tensor | None = None
    expanded_paged_kv_indptr: torch.Tensor | None = None
    expanded_paged_kv_indices: torch.Tensor | None = None
    expanded_paged_kv_last_page_len: torch.Tensor | None = None


class _DecodeGraphEntry:
    __slots__ = (
        "batch_size",
        "kind",
        "graph",
        "static_output",
        "static_logits",
        "static_input_ids",
        "static_positions",
        "static_input_embedding",
        "static_metadata",
        "kv_seq_lens_delta",
        "host_seq_lens",
        "host_block_counts",
    )


class DecodeMusaGraphRunner(BaseRunner):
    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        lm_head: nn.Module | None = None,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.max_batch = max_batch
        self.max_model_len = max_model_len
        self.lm_head = lm_head
        self.last_logits: torch.Tensor | None = None
        self._graphs: dict[tuple[int, str], _DecodeGraphEntry] = {}
        self._paged_kv_indices_buffer: torch.Tensor | None = None
        self._stream: torch.musa.Stream | None = None
        self._warmed_up = False
        from xllm.python.layers.gated_delta_net import Qwen3_5GatedDeltaNet

        self._has_linear_attention = any(
            isinstance(module, Qwen3_5GatedDeltaNet)
            for module in self.model.modules()
        )
        self._linear_conv_kernel = 0
        self._linear_conv_dim = 0
        for module in self.model.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                self._linear_conv_kernel = int(module.conv_kernel_size)
                self._linear_conv_dim = int(module.conv_dim)
                break
        self.attention_backend.reserve_decode_buffers(self.max_batch)

    def _linear_decode_cache_supported(self) -> bool:
        """Allow GDN decode graph only when conv cache is fused-decode width."""
        if not self._has_linear_attention:
            return True
        if self._linear_conv_kernel <= 1 or not self.layer_caches:
            return False
        found_conv = False
        for cache in self.layer_caches:
            if cache.conv is None or cache.conv.numel() == 0:
                continue
            found_conv = True
            if not _is_fused_decode_conv_cache(
                cache.conv,
                self._linear_conv_dim,
                self._linear_conv_kernel,
            ):
                return False
        return found_conv

    def can_execute(
        self, input_ids: torch.Tensor, metadata: AttentionMetadata
    ) -> bool:
        token_count = input_ids.shape[0]
        is_spec_verify = bool(getattr(metadata, "is_spec_verify", False))
        graph_batch_size = _decode_graph_batch_size(
            token_count, is_spec_verify
        )
        if token_count <= 0 or graph_batch_size > self.max_batch:
            return False
        if not _input_embedding_matches_tokens(metadata, token_count):
            return False
        if (
            _live_input_embedding(metadata) is not None
            and graph_batch_size != token_count
        ):
            return False
        sequence_count = int(metadata.paged_kv_last_page_len.numel())
        has_linear_state = (
            not self._has_linear_attention
            or (
                getattr(metadata, "linear_state_indices", None) is not None
                and getattr(metadata, "has_initial_state", None) is not None
            )
        )
        if is_spec_verify:
            if not has_linear_state:
                return False
            verify_sequences = _spec_verify_sequence_count(metadata, token_count)
            if verify_sequences is None:
                return False
            if self._has_linear_attention:
                accepted = getattr(metadata, "num_accepted_tokens", None)
                if accepted is None or int(accepted.numel()) != verify_sequences:
                    return False
            return _can_expand_spec_verify(metadata, token_count)
        if self._has_linear_attention and not self._linear_decode_cache_supported():
            # MTP expands conv cache to history+K. Fused decode requires
            # state_len == kernel-1, so only the verify graph may run then.
            return False
        return (
            not metadata.is_prefill
            and not metadata.is_chunked_prefill
            and token_count == sequence_count
            and has_linear_state
        )

    def warmup(self, _device: torch.device, _dtype: torch.dtype) -> None:
        # Dummy batches write live KV/GDN and omit MTP input_embedding.
        # Capture the first real decode instead, matching C++ lazy capture.
        self._warmed_up = True

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        is_spec_verify = bool(getattr(metadata, "is_spec_verify", False))
        padded_batch_size = _decode_graph_batch_size(
            batch_size, is_spec_verify
        )
        if padded_batch_size > self.max_batch:
            raise ValueError("decode batch exceeds MUSA graph capacity")
        graph_kind = "verify" if is_spec_verify else "decode"
        graph_key = (padded_batch_size, graph_kind)

        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(
                padded_batch_size, input_ids, positions, metadata, graph_kind
            )
            self._graphs[graph_key] = entry

        if self._stream is None:
            self._stream = torch.musa.Stream(device=input_ids.device)

        # Match NPU DecodeAclGraphRunner: fill + plan on the current stream,
        # then replay the captured graph on a dedicated stream.
        self._fill_entry(entry, input_ids, positions, metadata, batch_size)
        self.attention_backend.prepare(entry.static_metadata, graph_mode=True)

        self._stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(self._stream):
            with forward_context(
                ForwardContext(
                    self.attention_backend,
                    self.device,
                    entry.static_metadata,
                    self.layer_caches,
                    graph_mode=True,
                    activation_pool=None,
                )
            ):
                if first_capture:
                    self._capture(entry)
                entry.graph.replay()
                output = entry.static_output[:batch_size].clone()
                if entry.static_logits is not None:
                    self.last_logits = entry.static_logits[:batch_size].clone()
                else:
                    self.last_logits = None

        torch.musa.current_stream().wait_stream(self._stream)
        return output

    def _allocate_entry(
        self,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        graph_kind: str,
    ) -> _DecodeGraphEntry:
        device = input_ids.device
        if self._paged_kv_indices_buffer is None:
            page_size = self.attention_backend.page_size
            max_blocks_per_sequence = (
                self.max_model_len + page_size - 1
            ) // page_size
            self._paged_kv_indices_buffer = torch.zeros(
                self.max_batch * max_blocks_per_sequence,
                dtype=metadata.paged_kv_indices.dtype,
                device=device,
            )

        is_verify = graph_kind == "verify"
        live_indices = getattr(metadata, "linear_state_indices", None)
        if is_verify and live_indices is not None:
            linear_slots = int(live_indices.numel())
        else:
            linear_slots = padded_batch_size

        entry = _DecodeGraphEntry()
        entry.batch_size = padded_batch_size
        entry.kind = graph_kind
        entry.graph = None
        entry.static_output = None
        entry.static_logits = None
        entry.static_input_ids = torch.zeros(
            padded_batch_size, dtype=input_ids.dtype, device=device
        )
        entry.static_positions = torch.zeros(
            padded_batch_size, dtype=torch.int32, device=device
        )
        entry.static_input_embedding = None
        entry.static_metadata = _StaticAttentionMetadata(
            slot_mapping=torch.zeros(
                padded_batch_size,
                dtype=metadata.slot_mapping.dtype,
                device=device,
            ),
            paged_kv_indptr=torch.zeros(
                padded_batch_size + 1,
                dtype=metadata.paged_kv_indptr.dtype,
                device=device,
            ),
            paged_kv_indices=self._paged_kv_indices_buffer,
            paged_kv_last_page_len=torch.zeros(
                padded_batch_size,
                dtype=metadata.paged_kv_last_page_len.dtype,
                device=device,
            ),
            kv_cu_seq_lens=torch.zeros(
                padded_batch_size + 1,
                dtype=torch.int32,
                device=device,
            ),
            q_cu_seq_lens=torch.arange(
                padded_batch_size + 1, dtype=torch.int32, device=device
            ),
            kv_seq_lens=torch.zeros(
                padded_batch_size, dtype=torch.int32, device=device
            ),
            block_table=torch.full(
                (
                    padded_batch_size,
                    self._max_pages_per_sequence(metadata),
                ),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            linear_state_indices=torch.ones(
                linear_slots, dtype=torch.int32, device=device
            ),
            has_initial_state=torch.zeros(
                linear_slots, dtype=torch.bool, device=device
            ),
            num_accepted_tokens=(
                torch.ones(linear_slots, dtype=torch.int32, device=device)
                if is_verify
                else None
            ),
            is_spec_verify=is_verify,
            use_expanded_decode_for_spec_verify_attention=is_verify,
            max_seq_len=1,
            max_query_len=1,
        )
        entry.kv_seq_lens_delta = torch.empty(
            padded_batch_size, dtype=torch.int32, device=device
        )
        entry.host_seq_lens = None
        entry.host_block_counts = None
        return entry

    def _fill_entry(
        self,
        entry: _DecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        if positions.dtype == torch.int32 and positions.is_contiguous():
            graph_positions = positions
        else:
            graph_positions = positions.to(torch.int32).contiguous()

        if static_metadata.is_spec_verify:
            expanded = _expand_spec_verify_attention(
                metadata,
                batch_size,
                self.attention_backend.page_size,
                input_ids.device,
            )
            kv_cu_seq_lens = expanded.kv_cu_seq_lens
            paged_kv_indptr = expanded.paged_kv_indptr
            paged_kv_indices = expanded.paged_kv_indices
            paged_kv_last_page_len = expanded.paged_kv_last_page_len
            fa3_kv_seq_lens = expanded.kv_seq_lens
            fa3_block_table = expanded.block_table
            fa3_max_seq_len = expanded.max_seq_len
        else:
            if metadata.kv_cu_seq_lens is None:
                raise RuntimeError(
                    "decode MUSA graph requires device cumulative KV lengths"
                )
            kv_cu_seq_lens = metadata.kv_cu_seq_lens
            paged_kv_indptr = metadata.paged_kv_indptr
            paged_kv_indices = metadata.paged_kv_indices
            paged_kv_last_page_len = metadata.paged_kv_last_page_len
            fa3_kv_seq_lens = None
            fa3_block_table = None
            fa3_max_seq_len = int(getattr(metadata, "max_seq_len", 0) or 0)

        kernels.update_decode_graph_metadata(
            input_ids,
            graph_positions,
            metadata.slot_mapping,
            kv_cu_seq_lens,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            entry.static_input_ids,
            entry.static_positions,
            static_metadata.slot_mapping,
            static_metadata.kv_cu_seq_lens,
            entry.kv_seq_lens_delta,
            static_metadata.paged_kv_indptr,
            static_metadata.paged_kv_indices,
            static_metadata.paged_kv_last_page_len,
            padded_batch_size,
        )
        self._fill_fa3_metadata(
            entry,
            metadata,
            batch_size,
            kv_seq_lens=fa3_kv_seq_lens,
            block_table=fa3_block_table,
            max_seq_len=fa3_max_seq_len,
            max_query_len=1 if static_metadata.is_spec_verify else None,
        )
        self._fill_linear_state_metadata(entry, metadata)
        entry.static_input_embedding = _ensure_static_input_embedding(
            entry.static_input_embedding,
            _live_input_embedding(metadata),
            batch_size,
            input_ids.device,
        )
        static_metadata.input_embedding = entry.static_input_embedding

    def _fill_linear_state_metadata(
        self,
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
    ) -> None:
        static_metadata = entry.static_metadata
        if static_metadata.linear_state_indices is None:
            return
        live_indices = getattr(metadata, "linear_state_indices", None)
        live_has_state = getattr(metadata, "has_initial_state", None)
        if live_indices is None or live_has_state is None:
            if self._has_linear_attention:
                raise RuntimeError(
                    "Qwen3.5 MUSA graph requires linear state metadata"
                )
            static_metadata.linear_state_indices.zero_()
            static_metadata.has_initial_state.zero_()
            return
        index_slots = static_metadata.linear_state_indices.numel()
        if live_indices.numel() != index_slots:
            raise ValueError("linear_state_indices must match captured graph slots")
        if live_has_state.numel() != index_slots:
            raise ValueError("has_initial_state must match captured graph slots")
        static_metadata.linear_state_indices.copy_(
            live_indices.to(
                device=static_metadata.linear_state_indices.device,
                dtype=static_metadata.linear_state_indices.dtype,
            )
        )
        static_metadata.has_initial_state.copy_(
            live_has_state.to(
                device=static_metadata.has_initial_state.device,
                dtype=static_metadata.has_initial_state.dtype,
            )
        )
        if static_metadata.is_spec_verify:
            live_accepted = getattr(metadata, "num_accepted_tokens", None)
            if live_accepted is None or static_metadata.num_accepted_tokens is None:
                raise RuntimeError("MTP verify graph requires num_accepted_tokens")
            if live_accepted.numel() != static_metadata.num_accepted_tokens.numel():
                raise ValueError("num_accepted_tokens must match captured graph slots")
            static_metadata.num_accepted_tokens.copy_(
                live_accepted.to(
                    device=static_metadata.num_accepted_tokens.device,
                    dtype=static_metadata.num_accepted_tokens.dtype,
                )
            )

    def _max_pages_per_sequence(self, metadata: AttentionMetadata) -> int:
        page_size = self.attention_backend.page_size
        model_pages = (self.max_model_len + page_size - 1) // page_size
        live_tables = [
            getattr(metadata, "expanded_block_table", None),
            getattr(metadata, "block_table", None),
        ]
        live_width = 0
        for live_table in live_tables:
            if live_table is not None and live_table.dim() == 2:
                live_width = max(live_width, int(live_table.size(1)))
        return max(model_pages, live_width)

    def _fill_fa3_metadata(
        self,
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
        kv_seq_lens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        max_seq_len: int = 0,
        max_query_len: int | None = None,
    ) -> None:
        static_metadata = entry.static_metadata
        if static_metadata.kv_seq_lens is None or static_metadata.block_table is None:
            raise RuntimeError("decode MUSA graph is missing FA3 static buffers")

        live_seq = kv_seq_lens if kv_seq_lens is not None else metadata.kv_seq_lens
        if live_seq is None:
            if metadata.kv_cu_seq_lens is None:
                raise RuntimeError(
                    "decode MUSA graph requires per-sequence KV lengths"
                )
            live_seq = (
                metadata.kv_cu_seq_lens[1:] - metadata.kv_cu_seq_lens[:-1]
            )
        live_table = (
            block_table if block_table is not None else metadata.block_table
        )
        if live_table is None:
            raise RuntimeError("decode MUSA graph requires block_table for FA3")
        if live_table.size(1) > static_metadata.block_table.size(1):
            raise RuntimeError(
                "FA3 block_table is wider than the captured graph page table: "
                f"{int(live_table.size(1))} > {int(static_metadata.block_table.size(1))}"
            )
        kernels.update_fa3_graph_metadata(
            live_seq,
            live_table,
            static_metadata.kv_seq_lens,
            static_metadata.block_table,
            batch_size,
        )

        resolved_max_seq_len = max_seq_len
        if resolved_max_seq_len <= 0:
            resolved_max_seq_len = int(getattr(metadata, "max_seq_len", 0) or 0)
        if resolved_max_seq_len <= 0:
            host = metadata.kv_seq_lens_host
            if host is not None and host.numel() == batch_size:
                resolved_max_seq_len = int(host.max().item())
            elif host is not None and host.numel() == batch_size + 1:
                resolved_max_seq_len = int((host[1:] - host[:-1]).max().item())
            else:
                raise RuntimeError(
                    "decode MUSA graph requires max_seq_len or host KV lengths"
                )
        static_metadata.max_seq_len = resolved_max_seq_len
        if max_query_len is not None:
            static_metadata.max_query_len = max_query_len
        else:
            static_metadata.max_query_len = int(
                getattr(metadata, "max_query_len", 0) or 1
            )

    def _capture(self, entry: _DecodeGraphEntry) -> None:
        state_snapshot = self._snapshot_linear_state(entry)
        for _ in range(_CAPTURE_WARMUP_STEPS):
            hidden = self.model(entry.static_input_ids, entry.static_positions)
            if self.lm_head is not None:
                self.lm_head(hidden)
        self._restore_linear_state(state_snapshot)
        entry.graph = torch.musa.MUSAGraph()
        with torch.musa.graph(entry.graph, stream=self._stream):
            hidden = self.model(entry.static_input_ids, entry.static_positions)
            entry.static_output = hidden
            if self.lm_head is not None:
                entry.static_logits = self.lm_head(hidden)
        self._stream.synchronize()
        self._restore_linear_state(state_snapshot)
        logger.info(
            f"captured MUSA {entry.kind} graph batch={entry.batch_size} "
            f"logits={'on' if entry.static_logits is not None else 'off'}"
        )

    def _snapshot_linear_state(
        self, entry: _DecodeGraphEntry
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        indices = entry.static_metadata.linear_state_indices
        if not self._has_linear_attention or indices is None:
            return []
        snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        indices = indices.to(dtype=torch.long).contiguous()
        for cache in self.layer_caches:
            if cache.conv is not None and cache.conv.numel() > 0:
                snapshots.append(
                    (cache.conv, cache.conv.index_select(0, indices), indices)
                )
            if cache.ssm is not None and cache.ssm.numel() > 0:
                if cache.conv is None or cache.conv.numel() == 0:
                    raise RuntimeError("SSM snapshot requires a convolution cache")
                if cache.ssm.size(0) % cache.conv.size(0) != 0:
                    raise RuntimeError("SSM cache layout does not match conv cache")
                stride = cache.ssm.size(0) // cache.conv.size(0)
                offsets = torch.arange(stride, device=indices.device)
                expanded = (
                    indices[:, None] * stride + offsets[None, :]
                ).reshape(-1)
                snapshots.append(
                    (cache.ssm, cache.ssm.index_select(0, expanded), expanded)
                )
        return snapshots

    @staticmethod
    def _restore_linear_state(
        snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        for target, rows, indices in snapshots:
            target.index_copy_(0, indices, rows)
