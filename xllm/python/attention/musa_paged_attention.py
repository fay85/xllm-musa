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

"""MUSA FA3 paged-attention backend for the Python model executor.

Planner work (`fa3_decode_scheduler_metadata`) runs in `prepare()`, outside any
`MUSAGraph`. `execute()` only writes paged KV and runs `fa3_decode` /
`fa3_prefill` so capture/replay keep a static address set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from xllm.python import kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    normalize_layer_caches,
)
from scripts.logger import logger

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

_FA3_PAGE_SIZE = 64
_FA3_HEAD_DIMS = (128, 256)
_FA3_GQA_RATIOS = (2, 4, 6, 8)
# Matches C++ kDefaultFa3GraphNumSplits so capture/replay keep one kernel shape.
_FA3_GRAPH_NUM_SPLITS = 4


def _fa3_scheduler_numel(batch_size: int) -> int:
    rounded_batch = ((batch_size + 3) // 4) * 4
    return rounded_batch * 4


def _require_int32(tensor: torch.Tensor, _name: str) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        return tensor.to(dtype=torch.int32).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _as_head_tensor(
    tensor: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    # Packed QKV slices are strided on dim 0 but head-major on the last two
    # dims. C++ FA3 / reshape_paged_cache consume that layout; do not insert
    # a per-layer .contiguous() copy into the captured graph.
    viewed = tensor.view(-1, num_heads, head_dim)
    if viewed.stride(-1) == 1 and viewed.stride(-2) == head_dim:
        return viewed
    return viewed.contiguous()


class MusaPagedAttentionBackend(AttentionBackend):
    """Paged FA3 attention used by the MUSA C++ decoder path."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if dtype != torch.bfloat16:
            raise RuntimeError(
                "MUSA FA3 Python attention requires bfloat16, "
                f"got {dtype}"
            )
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise RuntimeError(
                "MUSA FA3 requires num_heads divisible by num_kv_heads, "
                f"got {num_heads} and {num_kv_heads}"
            )
        gqa_ratio = num_heads // num_kv_heads
        head_ok = (head_dim == 128 and gqa_ratio in (2, 4)) or (
            head_dim == 256 and gqa_ratio in (6, 8)
        )
        if not head_ok or gqa_ratio not in _FA3_GQA_RATIOS:
            raise RuntimeError(
                "MUSA FA3 does not support this attention shape: "
                f"head_dim={head_dim}, gqa_ratio={gqa_ratio}"
            )
        if head_dim not in _FA3_HEAD_DIMS:
            raise RuntimeError(
                f"MUSA FA3 does not support head_dim={head_dim}"
            )
        if sliding_window < -1:
            raise RuntimeError(
                f"MUSA FA3 requires sliding_window >= -1, got {sliding_window}"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.device = device
        self.dtype = dtype
        self._window_left = sliding_window

        self._kv_caches: list[KVCache] = []
        self._metadata: AttentionMetadata | None = None
        self._graph_mode = False
        self._cu_seqlens_q: torch.Tensor | None = None
        self._seqused_k: torch.Tensor | None = None
        self._page_table: torch.Tensor | None = None
        self._scheduler_metadata: torch.Tensor | None = None
        self._scheduler_buffers: dict[int, torch.Tensor] = {}
        self._seqused_k_buffers: dict[int, torch.Tensor] = {}
        self._cu_seqlens_q_buffers: dict[int, torch.Tensor] = {}
        self._page_table_buffers: dict[int, torch.Tensor] = {}
        self._max_seqlen_q = 1
        self._max_seqlen_k = 1
        self._num_splits = 0
        self._decode_output: torch.Tensor | None = None
        self._decode_lse_flat: torch.Tensor | None = None
        self._prefill_q: torch.Tensor | None = None
        self._prefill_k: torch.Tensor | None = None
        self._prefill_v: torch.Tensor | None = None
        self._prefill_output: torch.Tensor | None = None
        self._prefill_lse_flat: torch.Tensor | None = None
        logger.info(
            "MUSA Python attention uses FA3 "
            f"(gqa={gqa_ratio}, head_dim={head_dim})"
        )

    def reserve_decode_buffers(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise RuntimeError("FA3 decode reserve requires max_tokens > 0")
        need_output = (
            self._decode_output is None or self._decode_output.size(0) < max_tokens
        )
        if need_output:
            self._decode_output = torch.empty(
                (max_tokens, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        required_lse = self.num_heads * max_tokens
        need_lse = (
            self._decode_lse_flat is None
            or self._decode_lse_flat.numel() < required_lse
        )
        if need_lse:
            # Flat grow-only storage, then view to [heads, tokens] so the
            # captured FA3 LSE pointer stays contiguous across bucket sizes.
            self._decode_lse_flat = torch.empty(
                (required_lse,),
                dtype=torch.float32,
                device=self.device,
            )

    def reserve_prefill_buffers(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise RuntimeError("FA3 prefill reserve requires max_tokens > 0")
        need_q = self._prefill_q is None or self._prefill_q.size(0) < max_tokens
        if need_q:
            self._prefill_q = torch.empty(
                (max_tokens, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._prefill_output = torch.empty(
                (max_tokens, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        need_kv = self._prefill_k is None or self._prefill_k.size(0) < max_tokens
        if need_kv:
            self._prefill_k = torch.empty(
                (max_tokens, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._prefill_v = torch.empty(
                (max_tokens, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        required_lse = self.num_heads * max_tokens
        need_lse = (
            self._prefill_lse_flat is None
            or self._prefill_lse_flat.numel() < required_lse
        )
        if need_lse:
            self._prefill_lse_flat = torch.empty(
                (required_lse,),
                dtype=torch.float32,
                device=self.device,
            )

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        layer_caches = normalize_layer_caches(kv_caches)
        paged = [
            cache
            for cache in layer_caches
            if cache.key is not None and cache.value is not None
        ]
        if not paged:
            raise ValueError(
                "MusaPagedAttentionBackend requires at least one full-attention KV cache"
            )
        k_cache = paged[0].key
        if k_cache.dim() < 2 or k_cache.size(1) != _FA3_PAGE_SIZE:
            raise RuntimeError(
                "MUSA FA3 requires page size 64, "
                f"got cache shape {tuple(k_cache.shape)}"
            )
        self._layer_caches = layer_caches
        self._kv_caches = [
            (cache.key, cache.value)
            if cache.key is not None and cache.value is not None
            else None
            for cache in layer_caches
        ]

    def _first_paged_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        for cache in self._kv_caches:
            if cache is not None:
                return cache
        raise RuntimeError("KV caches are not bound")

    @property
    def num_kv_blocks(self) -> int:
        return self._first_paged_cache()[0].size(0)

    @property
    def page_size(self) -> int:
        return self._first_paged_cache()[0].size(1)

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        if not self._kv_caches:
            raise RuntimeError("KV caches are not bound")
        self._metadata = metadata
        self._graph_mode = graph_mode
        if metadata.is_chunked_prefill:
            self._prepare_decode(metadata)
            return
        if metadata.is_prefill and not metadata.is_chunked_prefill:
            return
        self._prepare_decode(metadata)

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("MusaPagedAttentionBackend.prepare() was not called")

        layer_kv = self._kv_caches[layer.layer_id]
        if layer_kv is None:
            raise RuntimeError(
                f"full-attention layer {layer.layer_id} is missing a paged KV cache"
            )
        k_cache, v_cache = layer_kv
        q_3d = _as_head_tensor(q, layer.num_heads, layer.head_dim)
        k_3d = _as_head_tensor(k, layer.num_kv_heads, layer.head_dim)
        v_3d = _as_head_tensor(v, layer.num_kv_heads, layer.head_dim)
        # FA3 prefill requires fully contiguous Q/K/V. Decode matches the C++
        # path and keeps packed-QKV slices (head-major, strided on dim 0).
        if metadata.is_prefill and not metadata.is_chunked_prefill:
            q_3d, k_3d, v_3d = self._materialize_prefill_qkv(q_3d, k_3d, v_3d)
        kernels.reshape_paged_cache(
            metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        if metadata.is_prefill and not metadata.is_chunked_prefill:
            return self._prefill(q_3d, k_3d, v_3d, metadata)
        return self._decode(q_3d, k_cache, v_cache)

    def _prepare_decode(self, metadata: AttentionMetadata) -> None:
        page_table = metadata.block_table
        if page_table is None:
            raise RuntimeError("MUSA FA3 decode requires a rectangular block_table")
        page_table = _require_int32(page_table, "block_table")
        batch_size = page_table.size(0)
        if batch_size <= 0:
            raise RuntimeError("MUSA FA3 decode requires a non-empty block_table")

        seqused_k = metadata.kv_seq_lens
        if seqused_k is None:
            if metadata.kv_cu_seq_lens is None:
                raise RuntimeError(
                    "MUSA FA3 decode requires kv_seq_lens or kv_cu_seq_lens"
                )
            seqused_k = metadata.kv_cu_seq_lens[1:] - metadata.kv_cu_seq_lens[:-1]
        seqused_k = _require_int32(seqused_k, "seqused_k")
        if seqused_k.numel() != batch_size:
            raise RuntimeError(
                "MUSA FA3 seqused_k must match block_table batch, "
                f"got {seqused_k.numel()} vs {batch_size}"
            )

        cu_seqlens_q = metadata.qo_indptr
        if cu_seqlens_q is None:
            cu_seqlens_q = metadata.q_cu_seq_lens
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=page_table.device
            )
        cu_seqlens_q = _require_int32(cu_seqlens_q, "cu_seqlens_q")
        if cu_seqlens_q.numel() != batch_size + 1:
            raise RuntimeError(
                "MUSA FA3 cu_seqlens_q must have batch+1 entries, "
                f"got {cu_seqlens_q.numel()} for batch {batch_size}"
            )

        max_seqlen_q = int(getattr(metadata, "max_query_len", 0) or 1)
        max_seqlen_k = int(getattr(metadata, "max_seq_len", 0) or 0)
        if max_seqlen_k <= 0:
            host = metadata.kv_seq_lens_host
            if host is None:
                raise RuntimeError(
                    "MUSA FA3 decode requires max_seq_len or host KV lengths"
                )
            if host.numel() == batch_size:
                max_seqlen_k = int(host.max().item())
            elif host.numel() == batch_size + 1:
                max_seqlen_k = int((host[1:] - host[:-1]).max().item())
            else:
                raise RuntimeError(
                    "MUSA FA3 host KV lengths must be batch or batch+1, "
                    f"got {host.numel()}"
                )
        if max_seqlen_k <= 0:
            raise RuntimeError("MUSA FA3 decode requires max_seq_len > 0")

        device = page_table.device
        if self._graph_mode:
            # DecodeMusaGraphRunner already wrote the static FA3 buffers.
            seqused_buf = seqused_k
            cu_q_buf = cu_seqlens_q
            page_buf = page_table
        else:
            seqused_buf = self._seqused_k_buffers.get(batch_size)
            if seqused_buf is None or seqused_buf.numel() != batch_size:
                seqused_buf = torch.empty(
                    batch_size, dtype=torch.int32, device=device
                )
                self._seqused_k_buffers[batch_size] = seqused_buf
            seqused_buf.copy_(seqused_k)

            cu_q_buf = self._cu_seqlens_q_buffers.get(batch_size)
            if cu_q_buf is None or cu_q_buf.numel() != batch_size + 1:
                cu_q_buf = torch.empty(
                    batch_size + 1, dtype=torch.int32, device=device
                )
                self._cu_seqlens_q_buffers[batch_size] = cu_q_buf
            cu_q_buf.copy_(cu_seqlens_q)

            page_cols = page_table.size(1)
            page_buf = self._page_table_buffers.get(batch_size)
            if (
                page_buf is None
                or page_buf.size(0) != batch_size
                or page_buf.size(1) != page_cols
            ):
                page_buf = torch.empty(
                    (batch_size, page_cols), dtype=torch.int32, device=device
                )
                self._page_table_buffers[batch_size] = page_buf
            page_buf.copy_(page_table)

        num_splits = _FA3_GRAPH_NUM_SPLITS if self._graph_mode else 0
        required = _fa3_scheduler_numel(batch_size)
        scheduler = self._scheduler_buffers.get(batch_size)
        if scheduler is None or scheduler.numel() != required:
            scheduler = torch.empty(required, dtype=torch.int32, device=device)
            self._scheduler_buffers[batch_size] = scheduler
        kernels.fa3_decode_scheduler_metadata(
            cu_q_buf,
            seqused_buf,
            batch_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim,
            max_seqlen_q,
            max_seqlen_k,
            self._window_left,
            0,
            num_splits,
            scheduler,
        )

        self._cu_seqlens_q = cu_q_buf
        self._seqused_k = seqused_buf
        self._page_table = page_buf
        self._scheduler_metadata = scheduler
        self._max_seqlen_q = max_seqlen_q
        self._max_seqlen_k = max_seqlen_k
        self._num_splits = num_splits

    def _decode(
        self,
        q_3d: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self._cu_seqlens_q is None
            or self._seqused_k is None
            or self._page_table is None
            or self._scheduler_metadata is None
        ):
            raise RuntimeError("MUSA FA3 decode planner did not run")
        token_count = q_3d.size(0)
        self.reserve_decode_buffers(token_count)
        decode_output = self._decode_output
        decode_lse_flat = self._decode_lse_flat
        if decode_output is None or decode_lse_flat is None:
            raise RuntimeError("FA3 decode buffers were not reserved")
        output = decode_output[:token_count]
        output_lse = decode_lse_flat[: self.num_heads * token_count].view(
            self.num_heads, token_count
        )
        kernels.fa3_decode(
            q_3d,
            k_cache,
            v_cache,
            self._cu_seqlens_q,
            self._seqused_k,
            self._page_table,
            self._scheduler_metadata,
            self._max_seqlen_q,
            self._window_left,
            0,
            float(self.scale),
            self._num_splits,
            output,
            output_lse,
        )
        return output.reshape(token_count, self.num_heads * self.head_dim)

    def _materialize_prefill_qkv(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_count = q_3d.size(0)
        prefill_q = self._prefill_q
        prefill_k = self._prefill_k
        prefill_v = self._prefill_v
        buffers_ready = (
            prefill_q is not None
            and prefill_k is not None
            and prefill_v is not None
            and prefill_q.size(0) >= token_count
            and prefill_k.size(0) >= token_count
            and prefill_v.size(0) >= token_count
        )
        if self._graph_mode and not buffers_ready:
            raise RuntimeError(
                "FA3 prefill graph buffers were not reserved before capture"
            )
        if buffers_ready:
            if prefill_q is None or prefill_k is None or prefill_v is None:
                raise RuntimeError("FA3 prefill QKV buffers were not reserved")
            q_view = prefill_q[:token_count]
            k_view = prefill_k[:token_count]
            v_view = prefill_v[:token_count]
            # fused_qk_norm_rope writes packed QKV in place. Q and V are
            # already contiguous slices; only the gated-K middle slice
            # needs a pack. Skip the extra D2D when FA3 can take the view.
            q_out = q_3d if q_3d.is_contiguous() else q_view.copy_(q_3d)
            k_out = k_3d if k_3d.is_contiguous() else k_view.copy_(k_3d)
            v_out = v_3d if v_3d.is_contiguous() else v_view.copy_(v_3d)
            return q_out, k_out, v_out
        if not q_3d.is_contiguous():
            q_3d = q_3d.contiguous()
        if not k_3d.is_contiguous():
            k_3d = k_3d.contiguous()
        if not v_3d.is_contiguous():
            v_3d = v_3d.contiguous()
        return q_3d, k_3d, v_3d

    def _prefill(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = q_3d.size(0)
        cu_seqlens_q = metadata.q_cu_seq_lens
        cu_seqlens_k = metadata.kv_cu_seq_lens
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor(
                [0, num_tokens], dtype=torch.int32, device=q_3d.device
            )
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        cu_seqlens_q = _require_int32(cu_seqlens_q, "cu_seqlens_q")
        cu_seqlens_k = _require_int32(cu_seqlens_k, "cu_seqlens_k")

        max_seqlen_q = int(getattr(metadata, "max_query_len", 0) or 0)
        max_seqlen_k = int(getattr(metadata, "max_seq_len", 0) or 0)
        if max_seqlen_q <= 0:
            host = metadata.kv_seq_lens_host
            if host is not None and host.numel() >= 2:
                max_seqlen_q = int((host[1:] - host[:-1]).max().item())
            else:
                max_seqlen_q = num_tokens
        if max_seqlen_k <= 0:
            max_seqlen_k = max_seqlen_q

        if self._graph_mode:
            output_ready = (
                self._prefill_output is not None
                and self._prefill_lse_flat is not None
                and self._prefill_output.size(0) >= num_tokens
                and self._prefill_lse_flat.numel() >= self.num_heads * num_tokens
            )
            if not output_ready:
                raise RuntimeError(
                    "FA3 prefill graph output buffers were not reserved"
                )
        else:
            self.reserve_prefill_buffers(num_tokens)
        prefill_output = self._prefill_output
        prefill_lse_flat = self._prefill_lse_flat
        if prefill_output is None or prefill_lse_flat is None:
            raise RuntimeError("FA3 prefill buffers were not reserved")
        output = prefill_output[:num_tokens]
        output_lse = prefill_lse_flat[: self.num_heads * num_tokens].view(
            self.num_heads, num_tokens
        )
        kernels.fa3_prefill(
            q_3d,
            k_3d,
            v_3d,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            self._window_left,
            0,
            float(self.scale),
            output,
            output_lse,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)
