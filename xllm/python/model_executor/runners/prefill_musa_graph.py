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

import bisect
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.graph_workspace import GraphActivationPool
from xllm.python.model_executor.runners.base import BaseRunner
from xllm.python.model_executor.runners.decode_musa_graph import (
    _ensure_static_input_embedding,
    _input_embedding_matches_tokens,
    _is_supported_prefill_conv_cache,
    _live_input_embedding,
)
from scripts.logger import logger

_CAPTURE_WARMUP_STEPS = 2
_PREFILL_CHUNK = 64


def _prefill_piecewise_enabled() -> bool:
    # C++ C1/384 currently captures one graph and zero GDN runners.
    # Splitting Mate out of Python MusaGraph is opt-in only.
    return os.environ.get("XLLM_PYTHON_PREFILL_PIECEWISE", "") == "1"


_PREFILL_SHOULDERS = (2080, 2112, 4352, 6400, 8448, 10496, 12544, 14592)
# Match C++ XLLM_QWEN35_C1_PIECEWISE_MAX_PADDING_TOKENS. Negative disables.
_C1_MAX_PADDING = int(
    os.environ.get("XLLM_QWEN35_C1_PIECEWISE_MAX_PADDING_TOKENS", "32")
)


def _generate_prefill_graph_tokens(max_tokens: int) -> tuple[int, ...]:
    """Match C++ generate_piecewise_prefill_graph_tokens."""
    sizes: list[int] = []

    def append_range(start: int, end_inclusive: int, step: int) -> None:
        size = start
        while size <= end_inclusive and size <= max_tokens:
            sizes.append(size)
            size += step

    append_range(4, 32, 4)
    append_range(48, 256, 16)
    append_range(288, 512, 32)
    append_range(576, 1024, 64)
    append_range(1280, 4096, 256)
    append_range(4608, max_tokens, 512)
    for shoulder in _PREFILL_SHOULDERS:
        if shoulder <= max_tokens:
            sizes.append(shoulder)
    unique = sorted(set(sizes))
    if max_tokens > 0 and (not unique or unique[-1] < max_tokens):
        unique.append(max_tokens)
    return tuple(unique)


def _align_qwen35_bucket(bucket: int, max_tokens: int) -> int | None:
    if bucket == 2080:
        return bucket if bucket <= max_tokens else None
    if bucket % _PREFILL_CHUNK == 0:
        return bucket if bucket <= max_tokens else None
    aligned = ((bucket + _PREFILL_CHUNK - 1) // _PREFILL_CHUNK) * _PREFILL_CHUNK
    if aligned <= max_tokens:
        return aligned
    return None


def _prefill_bucket(
    num_tokens: int,
    max_tokens: int,
    *,
    align_qwen35: bool,
) -> int | None:
    if num_tokens <= 0 or num_tokens > max_tokens:
        return None
    buckets = _generate_prefill_graph_tokens(max_tokens)
    index = bisect.bisect_left(buckets, num_tokens)
    raw = num_tokens if index >= len(buckets) else buckets[index]
    if not align_qwen35:
        return raw if raw <= max_tokens else None
    aligned = _align_qwen35_bucket(raw, max_tokens)
    if aligned is None:
        return None
    padding = aligned - num_tokens
    if _C1_MAX_PADDING >= 0 and padding > _C1_MAX_PADDING:
        return None
    return aligned


def _gdn_graph_tokens(num_tokens: int, max_tokens: int) -> int | None:
    """Exact-length GDN prefill graphs. T<64 stays eager."""
    if num_tokens < _PREFILL_CHUNK or num_tokens > max_tokens:
        return None
    return num_tokens


def _exact_prefill_tokens(num_tokens: int, max_tokens: int) -> int | None:
    """Exact-length non-GDN prefill graphs. Dummy pad tokens collapse MTP."""
    if num_tokens <= 0 or num_tokens > max_tokens:
        return None
    return num_tokens


def _gdn_kkt_tokens(num_tokens: int) -> int:
    """64-align live tokens for Mate KKT, matching C++ piecewise."""
    if num_tokens < _PREFILL_CHUNK:
        return num_tokens
    return ((num_tokens + _PREFILL_CHUNK - 1) // _PREFILL_CHUNK) * _PREFILL_CHUNK


def _host_prefill_has_gdn_prefix(
    metadata: AttentionMetadata,
    num_tokens: int,
    has_initial_state: torch.Tensor,
) -> bool:
    """Reject prefix GDN restore without a device ``.any()`` sync when possible."""
    host = getattr(metadata, "kv_seq_lens_host", None)
    if host is not None and host.numel() > 0 and host.device.type == "cpu":
        if host.numel() == 1:
            return int(host.item()) > num_tokens
        return int((host[-1] - host[0]).item()) > num_tokens
    if has_initial_state.device.type == "cpu":
        return bool(has_initial_state.any().item())
    return bool(has_initial_state.any())


def _is_single_sequence(metadata: AttentionMetadata) -> bool:
    cu_q = metadata.q_cu_seq_lens
    if cu_q is not None and cu_q.numel() != 2:
        return False
    cu_k = metadata.kv_cu_seq_lens
    if cu_k is not None and cu_k.numel() != 2:
        return False
    return True


def _pad_with_last(dst: torch.Tensor, src: torch.Tensor) -> None:
    actual = src.size(0)
    dst[:actual].copy_(src)
    padding = dst.size(0) - actual
    if padding > 0:
        dst[actual:].copy_(src[-1:].expand(padding))


@dataclass(slots=True)
class _StaticPrefillMetadata:
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None = None
    q_cu_seq_lens: torch.Tensor | None = None
    gdn_cu_seq_lens: torch.Tensor | None = None
    kv_cu_seq_lens: torch.Tensor | None = None
    kv_seq_lens_host: torch.Tensor | None = None
    paged_kv_indptr_host: torch.Tensor | None = None
    paged_kv_last_page_len_host: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    kv_seq_lens: torch.Tensor | None = None
    max_seq_len: int = 1
    max_query_len: int = 1
    is_prefill: bool = True
    linear_state_indices: torch.Tensor | None = None
    has_initial_state: torch.Tensor | None = None
    input_embedding: torch.Tensor | None = None
    is_chunked_prefill: bool = False


class _PrefillGraphEntry:
    __slots__ = (
        "num_tokens",
        "graph",
        "piecewise_handle",
        "static_output",
        "static_logits",
        "static_input_ids",
        "static_positions",
        "static_input_embedding",
        "static_metadata",
        "static_kkt_cu_seq_lens",
    )


class PrefillMusaGraphRunner(BaseRunner):
    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_tokens: int,
        lm_head: nn.Module | None = None,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.max_tokens = max_tokens
        self.lm_head = lm_head
        self.last_logits: torch.Tensor | None = None
        self._activation_pool = GraphActivationPool()
        self._graphs: dict[int, _PrefillGraphEntry] = {}
        from xllm.python.layers.gated_delta_net import Qwen3_5GatedDeltaNet

        self._has_linear_attention = any(
            isinstance(module, Qwen3_5GatedDeltaNet) for module in self.model.modules()
        )
        self._linear_conv_kernel = 0
        self._linear_conv_dim = 0
        for module in self.model.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                self._linear_conv_kernel = int(module.conv_kernel_size)
                self._linear_conv_dim = int(module.conv_dim)
                break
        self._stream: torch.musa.Stream | None = None

    def _linear_prefill_cache_supported(self) -> bool:
        """Allow GDN prefill graph for fused or MTP ``history+K`` conv."""
        if not self._has_linear_attention:
            return True
        if self._linear_conv_kernel <= 1 or not self.layer_caches:
            return False
        found_conv = False
        for cache in self.layer_caches:
            if cache.conv is None or cache.conv.numel() == 0:
                continue
            found_conv = True
            if not _is_supported_prefill_conv_cache(
                cache.conv,
                self._linear_conv_dim,
                self._linear_conv_kernel,
            ):
                return False
        return found_conv

    def can_execute(self, input_ids: torch.Tensor, metadata: AttentionMetadata) -> bool:
        if not metadata.is_prefill or metadata.is_chunked_prefill:
            return False
        if not _input_embedding_matches_tokens(metadata, input_ids.shape[0]):
            return False
        if not _is_single_sequence(metadata):
            return False
        if self._has_linear_attention:
            live_indices = getattr(metadata, "linear_state_indices", None)
            live_has_state = getattr(metadata, "has_initial_state", None)
            if live_indices is None or live_has_state is None:
                return False
            if _host_prefill_has_gdn_prefix(
                metadata, input_ids.shape[0], live_has_state
            ):
                # Prefix/chunk GDN state is not in the C++ piecewise zero-init
                # contract used by this graph.
                return False
            if not self._linear_prefill_cache_supported():
                return False
        if self._has_linear_attention:
            bucket = _gdn_graph_tokens(input_ids.shape[0], self.max_tokens)
        else:
            bucket = _exact_prefill_tokens(input_ids.shape[0], self.max_tokens)
        return bucket is not None and bucket <= self.max_tokens

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        actual_tokens = input_ids.shape[0]
        if self._has_linear_attention:
            bucket = _gdn_graph_tokens(actual_tokens, self.max_tokens)
        else:
            bucket = _exact_prefill_tokens(actual_tokens, self.max_tokens)
        if bucket is None or bucket > self.max_tokens:
            raise ValueError("prefill token count exceeds MUSA graph capacity")

        entry = self._graphs.get(bucket)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(bucket, input_ids, positions, metadata)
            self._graphs[bucket] = entry

        if self._stream is None:
            self._stream = torch.musa.Stream(device=input_ids.device)

        self._fill_entry(entry, input_ids, positions, metadata)
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
                    activation_pool=self._activation_pool,
                )
            ):
                if first_capture:
                    self._capture(entry)
                self._replay_entry(entry, actual_tokens)
                if entry.static_logits is not None:
                    self.last_logits = entry.static_logits.clone()
                else:
                    self.last_logits = None
                # MTP draft prefill cats token embeds with target hidden.
                # Last-row-only hidden makes that cat 382 vs 1 and crashes.
                output = entry.static_output[:actual_tokens].clone()
        torch.musa.current_stream().wait_stream(self._stream)
        return output

    def _allocate_entry(
        self,
        bucket: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> _PrefillGraphEntry:
        device = input_ids.device
        entry = _PrefillGraphEntry()
        entry.num_tokens = bucket
        entry.graph = None
        entry.piecewise_handle = None
        entry.static_output = None
        entry.static_logits = None
        entry.static_kkt_cu_seq_lens = torch.zeros(2, dtype=torch.int32, device=device)
        entry.static_input_ids = torch.zeros(
            bucket, dtype=input_ids.dtype, device=device
        )
        entry.static_positions = torch.zeros(bucket, dtype=torch.int32, device=device)
        entry.static_input_embedding = None
        cu_seqlens = torch.tensor([0, bucket], dtype=torch.int32, device=device)
        entry.static_metadata = _StaticPrefillMetadata(
            slot_mapping=torch.full(
                (bucket,),
                -1,
                dtype=metadata.slot_mapping.dtype,
                device=device,
            ),
            paged_kv_indptr=torch.zeros(
                2, dtype=metadata.paged_kv_indptr.dtype, device=device
            ),
            paged_kv_indices=torch.zeros(
                1, dtype=metadata.paged_kv_indices.dtype, device=device
            ),
            paged_kv_last_page_len=torch.zeros(
                1, dtype=metadata.paged_kv_last_page_len.dtype, device=device
            ),
            q_cu_seq_lens=cu_seqlens,
            gdn_cu_seq_lens=torch.zeros(2, dtype=torch.int32, device=device),
            kv_cu_seq_lens=cu_seqlens,
            linear_state_indices=torch.ones(1, dtype=torch.int32, device=device),
            has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
            max_seq_len=bucket,
            max_query_len=bucket,
            is_prefill=True,
        )
        return entry

    def _fill_entry(
        self,
        entry: _PrefillGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        if positions.dtype == torch.int32 and positions.is_contiguous():
            graph_positions = positions
        else:
            graph_positions = positions.to(torch.int32).contiguous()
        _pad_with_last(entry.static_input_ids, input_ids)
        _pad_with_last(entry.static_positions, graph_positions)
        entry.static_input_embedding = _ensure_static_input_embedding(
            entry.static_input_embedding,
            _live_input_embedding(metadata),
            int(input_ids.size(0)),
            input_ids.device,
        )
        entry.static_metadata.input_embedding = entry.static_input_embedding

        actual = input_ids.size(0)
        kkt_cu = entry.static_kkt_cu_seq_lens
        kkt_cu[0] = 0
        kkt_cu[1] = _gdn_kkt_tokens(actual)
        slot_mapping = metadata.slot_mapping
        if slot_mapping.dtype != entry.static_metadata.slot_mapping.dtype:
            slot_mapping = slot_mapping.to(entry.static_metadata.slot_mapping.dtype)
        entry.static_metadata.slot_mapping[:actual].copy_(slot_mapping)
        if actual < entry.num_tokens:
            # C++ piecewise pads cache slots with the last live slot so
            # FlashInfer overwrites that slot with the same value.
            entry.static_metadata.slot_mapping[actual:].copy_(
                entry.static_metadata.slot_mapping[actual - 1 : actual].expand(
                    entry.num_tokens - actual
                )
            )
        static_metadata = entry.static_metadata
        live_indices = getattr(metadata, "linear_state_indices", None)
        live_has_state = getattr(metadata, "has_initial_state", None)
        if live_indices is None or live_has_state is None:
            if self._has_linear_attention:
                raise RuntimeError("Qwen3.5 MUSA graph requires linear state metadata")
            static_metadata.linear_state_indices.zero_()
            static_metadata.has_initial_state.zero_()
            return
        if live_indices.numel() != 1 or live_has_state.numel() != 1:
            raise ValueError("prefill linear state metadata must be sequence-scoped")
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
        gdn_cu = static_metadata.gdn_cu_seq_lens
        if gdn_cu is None:
            raise RuntimeError("prefill graph requires gdn_cu_seq_lens")
        gdn_cu[0] = 0
        # Conv / recurrent Mate keep the live endpoint. FA uses the
        # bucket CU so dummy tokens overwrite the last cache slot.
        gdn_cu[1] = actual

    def _capture(self, entry: _PrefillGraphEntry) -> None:
        state_snapshot = self._snapshot_linear_state(entry)
        for _ in range(_CAPTURE_WARMUP_STEPS):
            self._run_captured_model(entry)
        self._restore_linear_state(state_snapshot)
        self._activation_pool.freeze()
        if _prefill_piecewise_enabled():
            self._capture_piecewise(entry)
        else:
            self._capture_musagraph(entry)
        self._stream.synchronize()
        self._restore_linear_state(state_snapshot)

    def _capture_musagraph(self, entry: _PrefillGraphEntry) -> None:
        entry.graph = torch.musa.MUSAGraph()
        with torch.musa.graph(entry.graph, stream=self._stream):
            hidden, logits = self._run_captured_model(entry)
            entry.static_output = hidden
            entry.static_logits = logits
        logger.info(
            f"captured MUSA prefill graph tokens={entry.num_tokens} "
            f"logits={'on' if entry.static_logits is not None else 'off'}"
        )

    def _capture_piecewise(self, entry: _PrefillGraphEntry) -> None:
        from xllm.python import kernels

        kernels.python_prefill_piecewise_begin(entry.static_input_ids)
        hidden, logits = self._run_captured_model(entry)
        entry.static_output = hidden
        entry.static_logits = logits
        handle, num_graphs, num_runners = kernels.python_prefill_piecewise_end(
            entry.static_input_ids
        )
        entry.piecewise_handle = int(handle)
        if self._has_linear_attention and int(num_runners) == 0:
            raise RuntimeError(
                "Python piecewise capture registered no GDN Mate runners"
            )
        logger.info(
            f"captured piecewise prefill graph tokens={entry.num_tokens} "
            f"graphs={int(num_graphs)} runners={int(num_runners)} "
            f"logits={'on' if entry.static_logits is not None else 'off'}"
        )

    def _replay_entry(self, entry: _PrefillGraphEntry, actual_tokens: int) -> None:
        if entry.piecewise_handle is not None:
            self._replay_piecewise(entry, actual_tokens)
            return
        if entry.graph is None:
            raise RuntimeError("prefill graph was not captured")
        entry.graph.replay()

    def _replay_piecewise(self, entry: _PrefillGraphEntry, actual_tokens: int) -> None:
        from xllm.python import kernels

        gdn_cu = entry.static_metadata.gdn_cu_seq_lens
        if gdn_cu is None:
            raise RuntimeError("prefill graph requires gdn_cu_seq_lens")
        kernels.python_prefill_piecewise_replay(
            entry.static_input_ids,
            entry.piecewise_handle,
            gdn_cu,
            entry.static_kkt_cu_seq_lens,
            actual_tokens,
            _gdn_kkt_tokens(actual_tokens),
        )

    def _run_captured_model(
        self, entry: _PrefillGraphEntry
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._activation_pool.reset()
        hidden = self.model(entry.static_input_ids, entry.static_positions)
        if self.lm_head is None:
            return hidden, None
        # Exact-length graphs keep the last row as the live token.
        logits = self.lm_head(hidden[-1:])
        return hidden, logits

    def _snapshot_linear_state(
        self, entry: _PrefillGraphEntry
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self._has_linear_attention:
            return []
        indices = entry.static_metadata.linear_state_indices
        if indices is None:
            return []
        indices = indices.to(dtype=torch.long).contiguous()
        snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for cache in self.layer_caches:
            if cache.conv is not None and cache.conv.numel() > 0:
                snapshots.append(
                    (cache.conv, cache.conv.index_select(0, indices), indices)
                )
            if cache.ssm is not None and cache.ssm.numel() > 0:
                if cache.conv is None or cache.conv.size(0) == 0:
                    raise RuntimeError("SSM snapshot requires a convolution cache")
                if cache.ssm.size(0) % cache.conv.size(0) != 0:
                    raise RuntimeError("SSM cache layout does not match conv cache")
                stride = cache.ssm.size(0) // cache.conv.size(0)
                offsets = torch.arange(stride, device=indices.device)
                expanded = (indices[:, None] * stride + offsets[None, :]).reshape(-1)
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
