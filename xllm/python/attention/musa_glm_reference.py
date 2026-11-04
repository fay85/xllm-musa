# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Experimental eager-only MUSA GLM attention components.

All tensor math and state storage remain on MUSA. Host reads are control
metadata only. This deliberately rejects graph, CP/KV sharding, speculative
verify, quantized caches and non-GLM attention before changing cache state.
Admission is experimental; complete checkpoint and PD validation are separate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import os

import torch
import torch.nn.functional as functional

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    MlaIndexContext,
    linear_state_checkpoint_stride,
    resolve_linear_state_io_indices,
)
from xllm.python.kernels_musa.kda import recurrent_kda

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


def _host_values(metadata: AttentionMetadata, name: str) -> list[int]:
    values = getattr(metadata, name + "_host_values", None)
    if values is not None and len(values):
        return [int(value) for value in values]
    tensor = getattr(metadata, name + "_host", None)
    if tensor is None:
        tensor = getattr(metadata, name, None)
    if tensor is None:
        raise ValueError(f"MUSA eager attention requires {name} control metadata")
    return [int(value) for value in tensor.reshape(-1).tolist()]


class MusaGlmReferenceBackend(AttentionBackend):
    """Paged NoPE sparse MLA plus FP32 value-first KDA, on real MUSA."""

    def __init__(self, device: torch.device) -> None:
        if device.type != "musa":
            raise ValueError("GLM MUSA backend requires a MUSA device")
        self._device = device
        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None
        self._q_cumulative: list[int] = []
        self._kv_lengths: list[int] = []
        self._cache_step_control = os.environ.get("XLLM_MUSA_STEP_METADATA_CACHE", "0") == "1"
        self._linear_control_cache: tuple[
            torch.Tensor, torch.Tensor, list[int], list[int], list[bool]
        ] | None = None

    @property
    def is_mla(self) -> bool:
        return True

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        if not kv_caches:
            raise ValueError("GLM MUSA requires explicit layer caches")
        for cache in kv_caches:
            for tensor in (
                cache.key,
                cache.index,
                cache.conv,
                cache.ssm,
                cache.kpool_tail,
            ):
                if tensor is not None and tensor.device != self._device:
                    raise ValueError(
                        "GLM MUSA caches must reside on the configured device"
                    )
        self._kv_caches = kv_caches

    def _key_cache(self) -> torch.Tensor:
        for cache in self._kv_caches:
            if cache.key is not None and cache.key.numel():
                return cache.key
        raise RuntimeError("No paged latent cache has been bound")

    @property
    def num_kv_blocks(self) -> int:
        return self._key_cache().shape[0]

    @property
    def page_size(self) -> int:
        return self._key_cache().shape[1]

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        if graph_mode or getattr(metadata, "is_spec_verify", False):
            raise NotImplementedError(
                "Initial MUSA GLM backend is eager-only without speculative verify"
            )
        if getattr(metadata, "kv_split_size", 1) != 1 or getattr(
            metadata, "has_kv_shard", False
        ):
            raise NotImplementedError(
                "Initial MUSA GLM backend requires replicated unsharded cache pages"
            )
        kv_lengths = _host_values(metadata, "kv_seq_lens")
        host_cumulative = getattr(metadata, "q_cu_seq_lens_host_values", None)
        device_cumulative = getattr(metadata, "q_cu_seq_lens", None)
        if (host_cumulative is not None and len(host_cumulative)) or device_cumulative is not None:
            q_cumulative = _host_values(metadata, "q_cu_seq_lens")
        else:
            # MUSA's C++ view can expose normalized per-sequence HOST lengths
            # without a cumulative DEVICE tensor. This is a supported metadata
            # form, not a compute fallback or exception-driven retry.
            query_lengths = _host_values(metadata, "q_seq_lens")
            q_cumulative = [0]
            for length in query_lengths:
                q_cumulative.append(q_cumulative[-1] + length)
        if not q_cumulative or q_cumulative[0] != 0:
            q_cumulative = [0] + q_cumulative
        if len(q_cumulative) != len(kv_lengths) + 1 or any(
            end <= start for start, end in zip(q_cumulative, q_cumulative[1:])
        ):
            raise ValueError(
                "Query cumulative lengths must describe nonempty logical sequences"
            )
        if any(
            kv < end - start
            for kv, start, end in zip(kv_lengths, q_cumulative, q_cumulative[1:])
        ):
            raise ValueError(
                "Logical KV lengths cannot be shorter than current queries"
            )
        if not self._kv_caches:
            raise RuntimeError("Bind caches before preparing a MUSA forward")
        self._q_cumulative, self._kv_lengths = q_cumulative, kv_lengths
        self._metadata = metadata
        self._linear_control_cache = None

    def _prepared(self) -> AttentionMetadata:
        if self._metadata is None:
            raise RuntimeError("MUSA backend prepare() has not been called")
        return self._metadata

    def _require_device(self, *tensors: torch.Tensor) -> None:
        if any(tensor.device != self._device for tensor in tensors):
            raise ValueError(
                "Inference tensors must remain on the configured MUSA device"
            )

    def execute(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Initial GLM MUSA backend only implements KDA and NoPE sparse MLA"
        )

    def _linear_control(
        self, metadata: AttentionMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[bool]]:
        """Resolve immutable per-forward control once, preserving layer COW checks."""
        if self._cache_step_control and self._linear_control_cache is not None:
            return self._linear_control_cache
        read_indices, write_indices = resolve_linear_state_io_indices(metadata)
        if read_indices is None or write_indices is None:
            raise ValueError(
                "KDA requires explicit read/write state slots"
            )
        read_ids, write_ids = read_indices.tolist(), write_indices.tolist()
        sequences = len(self._kv_lengths)
        if metadata.has_initial_state is not None:
            initial_validity = metadata.has_initial_state.tolist()
        elif not metadata.is_prefill and not metadata.is_chunked_prefill:
            # The C++ metadata builder deliberately materializes this mask
            # only for prefill. Ordinary decode consumes its existing live
            # state; a zero-history row is not a valid decode continuation.
            if any(
                kv <= end - start
                for kv, start, end in zip(
                    self._kv_lengths, self._q_cumulative, self._q_cumulative[1:]
                )
            ):
                raise ValueError("KDA decode requires existing sequence history")
            initial_validity = [True] * sequences
        else:
            raise ValueError("KDA prefill requires explicit initial-state validity")
        result = (read_indices, write_indices, read_ids, write_ids, initial_validity)
        if self._cache_step_control:
            self._linear_control_cache = result
        return result

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        beta_raw: torch.Tensor,
        layer: Attention,
        raw_gate_proj: torch.Tensor,
    ) -> torch.Tensor:
        metadata = self._prepared()
        cache = self._kv_caches[layer.layer_id]
        if cache.conv is None or cache.ssm is None:
            raise ValueError("KDA layer requires framework convolution and SSM caches")
        if linear_state_checkpoint_stride(cache.conv, cache.ssm) != 1:
            raise NotImplementedError(
                "Initial MUSA KDA requires one live SSM checkpoint per slot"
            )
        batch, channels, tokens = mixed_qkv.shape
        heads, dim, window = (
            layer.num_heads_local,
            layer.head_dim,
            layer.conv_kernel_size,
        )
        if channels != 3 * heads * dim or cache.conv.shape[1:] != (
            channels,
            window - 1,
        ):
            raise ValueError(
                "MUSA convolution cache must have [slots,3*heads*dim,window-1] layout"
            )
        self._require_device(mixed_qkv, beta_raw, raw_gate_proj, layer.conv_weight)
        read_indices, write_indices, read_ids, write_ids, initial_validity = (
            self._linear_control(metadata)
        )
        sequences = len(self._kv_lengths)
        if (
            len(read_ids) != sequences
            or len(write_ids) != sequences
            or len(initial_validity) != sequences
        ):
            raise ValueError("KDA state metadata must cover every logical sequence")
        if batch * tokens != self._q_cumulative[-1]:
            raise ValueError("Initial eager KDA does not support padded token buckets")
        if len(set(write_ids)) != sequences or any(
            not 0 < slot < cache.conv.shape[0] for slot in write_ids
        ):
            raise ValueError("KDA destinations must be unique nonzero in-bounds slots")
        if any(not 0 <= slot < cache.conv.shape[0] for slot in read_ids):
            raise ValueError("KDA source slots must be in bounds")
        inputs = mixed_qkv.transpose(1, 2).reshape(-1, channels)
        gates = raw_gate_proj.reshape(-1, heads, dim)
        betas = beta_raw.reshape(-1, heads)
        # Snapshot every read before writing any destination (prefix COW/permutation).
        conv_snapshots = cache.conv.index_select(
            0, read_indices.to(torch.int64)
        ).clone()
        ssm_snapshots = cache.ssm.index_select(0, read_indices.to(torch.int64)).clone()
        outputs, next_conv, next_ssm = [], [], []
        for index, (start, end) in enumerate(
            zip(self._q_cumulative, self._q_cumulative[1:])
        ):
            conv = (
                conv_snapshots[index]
                if initial_validity[index]
                else torch.zeros_like(conv_snapshots[index])
            )
            ssm = (
                ssm_snapshots[index : index + 1]
                if initial_validity[index]
                else torch.zeros_like(ssm_snapshots[index : index + 1])
            )
            concatenated = torch.cat((conv, inputs[start:end].T), -1)
            convolved = (
                concatenated.float().unfold(-1, window, 1)
                * layer.conv_weight.float().squeeze(1)[:, None, :]
            ).sum(-1)
            if layer.activation == "silu":
                convolved = functional.silu(convolved)
            elif layer.activation != "identity":
                raise NotImplementedError(
                    "Initial MUSA convolution supports identity/silu only"
                )
            query, key, value = convolved.T.to(mixed_qkv.dtype).chunk(3, -1)
            output, final = recurrent_kda(
                query.reshape(1, end - start, heads, dim),
                key.reshape(1, end - start, heads, dim),
                value.reshape(1, end - start, heads, dim),
                gates[start:end].unsqueeze(0),
                betas[start:end].unsqueeze(0),
                layer.forget_gate.A_log,
                layer.forget_gate.dt_bias,
                ssm,
                lower_bound=layer.forget_gate.safe_gate_lower_bound,
            )
            outputs.append(output.squeeze(0))
            next_conv.append(concatenated[:, -(window - 1) :])
            next_ssm.append(final.squeeze(0))
        destination = write_indices.to(torch.int64)
        cache.conv.index_copy_(0, destination, torch.stack(next_conv))
        cache.ssm.index_copy_(0, destination, torch.stack(next_ssm))
        return torch.cat(outputs, 0).reshape(batch, tokens, heads, dim)

    def _write_pages(self, cache: torch.Tensor, values: torch.Tensor) -> None:
        metadata = self._prepared()
        self._require_device(cache, values, metadata.slot_mapping)
        if cache.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise NotImplementedError(
                "Initial MUSA paging requires unquantized floating caches"
            )
        flat = cache.reshape(-1, cache.shape[-1])
        rows = values.reshape(-1, values.shape[-1])
        slots = metadata.slot_mapping.reshape(-1).to(torch.int64)
        if rows.shape != (slots.numel(), flat.shape[-1]):
            raise ValueError(
                "Paged write tensor geometry does not match cache and slot mapping"
            )
        valid = slots > 0
        selected = slots[valid]
        if selected.numel() and (
            int(selected.max().item()) >= flat.shape[0]
            or selected.unique().numel() != selected.numel()
        ):
            raise ValueError(
                "Paged writes require unique nonzero in-bounds physical slots"
            )
        flat.index_copy_(0, selected, rows[valid])

    def _block_table(self) -> torch.Tensor:
        table = self._prepared().block_table
        if table is None or table.dim() != 2 or table.shape[0] != len(self._kv_lengths):
            raise ValueError(
                "MUSA paging requires one block-table row per logical sequence"
            )
        self._require_device(table)
        return table.to(torch.int64)

    def gather_index_history(self, layer: Attention, batch_size: int) -> torch.Tensor:
        del batch_size
        cache = self._kv_caches[layer.layer_id].index
        if cache is None or cache.dim() != 4 or cache.shape[2] != 1:
            raise ValueError("Indexer requires a [blocks,page,1,width] cache")
        table = self._block_table()
        length = max(self._kv_lengths)
        positions = torch.arange(length, device=self._device)
        if (length - 1) // cache.shape[1] >= table.shape[1]:
            raise ValueError("Logical index history exceeds block-table capacity")
        physical = (
            table[:, positions // cache.shape[1]] * cache.shape[1]
            + positions % cache.shape[1]
        )
        valid = (
            positions[None]
            < torch.tensor(self._kv_lengths, device=self._device)[:, None]
        )
        safe = torch.where(valid, physical, torch.zeros_like(physical))
        history = cache.reshape(-1, cache.shape[-1]).index_select(0, safe.reshape(-1))
        return history.reshape(
            len(self._kv_lengths), length, cache.shape[-1]
        ) * valid.unsqueeze(-1)

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        metadata = self._prepared()
        cache = self._kv_caches[layer.layer_id]
        if cache.index is None or cache.indexer_scale is not None:
            raise NotImplementedError(
                "Initial GLM MUSA indexer requires an unquantized index cache"
            )
        table = self._block_table()
        read_indices, write_indices = resolve_linear_state_io_indices(metadata)

        def _update(values: torch.Tensor, scales: torch.Tensor | None) -> None:
            if scales is not None:
                raise NotImplementedError("Quantized index writes are unsupported")
            self._write_pages(cache.index, values)

        def _materialize() -> tuple[torch.Tensor, None, torch.Tensor]:
            return cache.index, None, table

        def _quant_metadata(
            heads: int, dim: int, count: int, ratio: int
        ) -> torch.Tensor:
            raise NotImplementedError("Quantized indexer metadata is unsupported")

        return MlaIndexContext(
            index_cache=cache.index,
            slot_mapping=metadata.slot_mapping,
            block_table=table,
            actual_seq_q=torch.tensor(
                self._q_cumulative[1:], dtype=torch.int32, device=self._device
            ),
            actual_seq_kv=torch.tensor(
                self._kv_lengths, dtype=torch.int32, device=self._device
            ),
            index_cache_scale=None,
            get_quant_indexer_metadata=_quant_metadata,
            update_index_cache=_update,
            materialize_index_cache=_materialize,
            kpool_tail=cache.kpool_tail,
            kpool_tail_read_indices=read_indices,
            kpool_tail_write_indices=write_indices,
            kpool_query_lens=tuple(
                end - start
                for start, end in zip(self._q_cumulative, self._q_cumulative[1:])
            ),
        )

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor | None,
        k_latent: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        layer: Attention,
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        self._prepared()
        if (
            q_pe is not None
            or k_pe is not None
            or topk is None
            or cache_is_preprocessed
        ):
            raise NotImplementedError(
                "Initial MUSA MLA requires explicit NoPE latent writes and sparse top-k"
            )
        cache = self._kv_caches[layer.layer_id].key
        if cache is None or k_latent is None or cache.dim() != 4 or cache.shape[2] != 1:
            raise ValueError("GLM latent cache must be [blocks,page,1,latent_dim]")
        self._require_device(q_latent, k_latent, topk)
        if (
            q_latent.shape[0] != self._q_cumulative[-1]
            or topk.shape[0] != q_latent.shape[0]
        ):
            raise ValueError(
                "Sparse MLA queries/top-k must match prepared logical query rows"
            )
        table = self._block_table()
        self._write_pages(cache, k_latent)
        flat = cache.reshape(-1, cache.shape[-1])
        outputs = []
        for sequence, (start, end) in enumerate(
            zip(self._q_cumulative, self._q_cumulative[1:])
        ):
            for offset in range(start, end, 8):
                limit = min(end, offset + 8)
                selected = (
                    topk[offset:limit].reshape(limit - offset, -1).to(torch.int64)
                )
                position = (
                    self._kv_lengths[sequence]
                    - (end - start)
                    + torch.arange(offset - start, limit - start, device=self._device)
                )
                valid = (selected >= 0) & (selected <= position[:, None])
                safe = selected.clamp(0, self._kv_lengths[sequence] - 1)
                page = table[sequence, safe // cache.shape[1]]
                physical = page * cache.shape[1] + safe % cache.shape[1]
                keys = flat.index_select(0, physical.reshape(-1)).reshape(
                    limit - offset, selected.shape[1], -1
                )
                scores = (
                    torch.einsum(
                        "qhd,qkd->qhk", q_latent[offset:limit].float(), keys.float()
                    )
                    * layer.scale
                )
                scores.masked_fill_(~valid[:, None, :], float("-inf"))
                probabilities = torch.nan_to_num(scores.softmax(-1), nan=0.0)
                output = torch.einsum("qhk,qkd->qhd", probabilities, keys.float())
                outputs.append(output.to(q_latent.dtype))
        return torch.cat(outputs, 0)
