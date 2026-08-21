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

"""MUSA paged-attention support kernels."""

from __future__ import annotations

import torch

reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache
update_decode_graph_metadata = torch.ops.xllm_ops.update_decode_graph_metadata
update_fa3_graph_metadata = torch.ops.xllm_ops.update_fa3_graph_metadata
fa3_decode = torch.ops.xllm_ops.fa3_decode


def _fa3_scheduler_numel(batch_size: int) -> int:
    rounded_batch = ((batch_size + 3) // 4) * 4
    return rounded_batch * 4


def fa3_decode_scheduler_metadata(
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim_qk: int,
    head_dim_vo: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int,
    window_size_right: int,
    num_splits: int,
    scheduler_metadata: torch.Tensor | None = None,
) -> torch.Tensor:
    if scheduler_metadata is None:
        scheduler_metadata = cu_seqlens_q.new_empty(
            (_fa3_scheduler_numel(batch_size),)
        )
    return torch.ops.xllm_ops.fa3_decode_scheduler_metadata(
        cu_seqlens_q,
        seqused_k,
        batch_size,
        num_heads_q,
        num_heads_kv,
        head_dim_qk,
        head_dim_vo,
        max_seqlen_q,
        max_seqlen_k,
        window_size_left,
        window_size_right,
        num_splits,
        scheduler_metadata,
    )


def fa3_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_left: int,
    window_right: int,
    sm_scale: float,
    output: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(query)
    if output_lse is None:
        output_lse = torch.empty(
            (query.size(1), query.size(0)),
            dtype=torch.float32,
            device=query.device,
        )
    return torch.ops.xllm_ops.fa3_prefill(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_left,
        window_right,
        sm_scale,
        output,
        output_lse,
    )


__all__ = [
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "update_fa3_graph_metadata",
    "fa3_decode_scheduler_metadata",
    "fa3_decode",
    "fa3_prefill",
]
