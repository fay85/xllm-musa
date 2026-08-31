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

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadataLike,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


@dataclass(frozen=True, slots=True)
class LayerCache:
    """Named per-layer caches. Full attention uses key/value; GDN uses conv/ssm."""

    key: torch.Tensor | None
    value: torch.Tensor | None
    index: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    ssm: torch.Tensor | None = None
    index_scale: torch.Tensor | None = None


_LAYER_CACHE_SLOTS = ("key", "value", "index", "conv", "ssm", "index_scale")
LayerCacheInput = LayerCache | tuple[torch.Tensor | None, ...]
KVCache = tuple[torch.Tensor, torch.Tensor]


def normalize_layer_caches(caches: Sequence[LayerCacheInput]) -> list[LayerCache]:
    normalized: list[LayerCache] = []
    for cache in caches:
        if isinstance(cache, LayerCache):
            normalized.append(cache)
            continue
        if not 2 <= len(cache) <= len(_LAYER_CACHE_SLOTS):
            raise ValueError(
                "layer cache must hold between K/V and "
                f"{'/'.join(_LAYER_CACHE_SLOTS)} tensors"
            )
        slots = [
            None if tensor is None or not tensor.numel() else tensor
            for tensor in cache
        ]
        slots.extend([None] * (len(_LAYER_CACHE_SLOTS) - len(slots)))
        normalized.append(LayerCache(*slots))
    return normalized


class AttentionMetadata(Protocol):
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None
    q_cu_seq_lens: torch.Tensor | None
    gdn_cu_seq_lens: torch.Tensor | None
    kv_cu_seq_lens: torch.Tensor | None
    kv_seq_lens_host: torch.Tensor | None
    kv_seq_lens_host_values: list[int] | None
    q_seq_lens_host: torch.Tensor | None
    paged_kv_indptr_host: torch.Tensor | None
    paged_kv_last_page_len_host: torch.Tensor | None
    block_table: torch.Tensor | None
    kv_seq_lens: torch.Tensor | None
    q_seq_lens: torch.Tensor | None
    max_seq_len: int
    max_query_len: int
    is_prefill: bool
    is_chunked_prefill: bool
    linear_state_indices: torch.Tensor | None
    has_initial_state: torch.Tensor | None
    input_embedding: torch.Tensor | None
    num_accepted_tokens: torch.Tensor | None
    dp_token_counts: Sequence[int]
    dp_is_decode: Sequence[int]
    expanded_decode_metadata: ExpandedDecodeMetadataLike | None
    is_spec_verify: bool
    use_expanded_decode_for_spec_verify_attention: bool
    expanded_kv_seq_lens: torch.Tensor | None
    expanded_block_table: torch.Tensor | None
    expanded_kv_seq_lens_host: torch.Tensor | None
    expanded_paged_kv_indptr: torch.Tensor | None
    expanded_paged_kv_indices: torch.Tensor | None
    expanded_paged_kv_last_page_len: torch.Tensor | None


@dataclass(frozen=True)
class MlaIndexContext:
    """Public contract handed to an optional LightningIndexer."""

    index_cache: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    index_cache_scale: torch.Tensor | None
    get_quant_indexer_metadata: Callable[[int, int, int, int], torch.Tensor]
    update_index_cache: Callable[[torch.Tensor, torch.Tensor | None], None]


@dataclass(frozen=True)
class MlaPreprocessContext:
    """Decode cache tensors consumed by fused MLA preprocessing."""

    kv_cache: torch.Tensor
    rope_cache: torch.Tensor
    slot_mapping: torch.Tensor


class AttentionBackend(ABC):
    @abstractmethod
    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        pass

    @abstractmethod
    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_kv_blocks(self) -> int:
        pass

    @property
    @abstractmethod
    def page_size(self) -> int:
        pass

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        layer: "Attention",
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        del q_latent, q_pe, k_latent, k_pe, layer, topk, cache_is_preprocessed
        raise NotImplementedError(f"{type(self).__name__} does not support MLA")

    def mla_preprocess_context(
        self,
        layer: "Attention",
    ) -> MlaPreprocessContext | None:
        del layer
        return None

    def mla_index_context(self, layer: "Attention") -> MlaIndexContext:
        del layer
        raise NotImplementedError(
            f"{type(self).__name__} does not support MLA indexer"
        )
