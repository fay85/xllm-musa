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

"""MUSA rotary-embedding kernels."""

from __future__ import annotations

import torch


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    *,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    interleaved: bool = False,
    k_head_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-head QK-RMSNorm and RoPE to a packed QKV projection."""

    del cos, sin
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    fused = torch.ops.xllm_ops.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        interleaved,
        position_ids,
        k_head_offset,
    )
    key_start = (k_head_offset if k_head_offset > 0 else num_heads_q) * head_dim
    return (
        fused[:, :q_size],
        fused[:, key_start : key_start + kv_size],
        fused[:, key_start + kv_size : key_start + 2 * kv_size],
    )


__all__ = ["fused_qk_norm_rope"]
