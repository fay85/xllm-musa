# Copyright 2025-2026 The xLLM Authors.
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

"""GLM-5.3-Flash-VL multimodal model (Python model executor target).

Architecture: GlmOcr-based ViT + GLM-5.3-Flash LLM + lm_head.

This module contains both the vision tower and the VLM composition logic,
mirroring the qwen3_vl.py structure. The LLM backbone is imported from
``glm5_next.py``.

Vision tower (GlmOcr-based):
  - ``VisionPatchEmbed``: 3D-conv patch projection.
  - ``VisionRotaryEmbedding``: RoPE inv_freq table (half head_dim).
  - ``VisionAttention``: MHA with QK-RMSNorm + RoPE, varlen via cu_seqlens.
  - ``VisionMLP``: packed gate/up projection and row-parallel down projection.
  - ``VisionBlock``: pre-norm (RMSNorm) transformer block.
  - ``VisionPatchMerger``: proj + LayerNorm + gated MLP, with
    ``context_dim = projection_intermediate_size`` for GLM-5.3-Flash.
  - ``downsample``: Conv2d spatial merge.
  - ``post_layernorm``: RMSNorm after blocks.

Data flow:
  pixel_values + grid_thw → vision_model → image_embeds
  input_ids → embed_tokens → text_embeds
  text_embeds.masked_scatter(image_mask, image_embeds) → inputs_embeds
  inputs_embeds + positions → language_model → hidden → lm_head
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from xllm.python import distributed, kernels
from xllm.python.layers import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_loader.module_loaders import quant_description_weight_is_quantized, read_quant_description
from xllm.python.models.base import PyModelBase
from xllm.python.models.glm5_next import (
    Glm5NextConfig,
    Glm5NextForCausalLM,
    Glm5NextModel,
    _glm5_next_swiglu,
)
from xllm.python.models.multimodal import (
    EncoderOutputs,
    EncoderTokenCounts,
    MultimodalData,
    MultimodalInputs,
    encode_batched_vision,
    merge_multimodal_embeddings,
)

# ---------------------------------------------------------------------------
# Position helpers (mirror transformers.vision_utils, pure-tensor, no dep)
# ---------------------------------------------------------------------------


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Cumulative sequence lengths per image from ``grid_thw``.

    ``grid_thw`` is ``(num_images, 3)`` (T, H, W). Returns ``(total_patches+1,)``
    int32 with a leading 0, matching varlen attention layouts.
    """
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """(row, col) position ids for the vision RoPE.

    Returns ``(total_tokens, 2)`` long. Each image contributes
    ``T * H * W`` tokens (before spatial merge); the rotary embedding is
    applied per-patch, and the spatial merge happens later via the Conv2d
    downsample + merger.
    """
    device = grid_thw.device
    position_ids: list[torch.Tensor] = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = (
            hpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .transpose(1, 2)
            .flatten()
        )

        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = (
            wpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .transpose(1, 2)
            .flatten()
        )
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


# ---------------------------------------------------------------------------
# Vision config
# ---------------------------------------------------------------------------


@dataclass
class Glm5NextVisionConfig:
    """Configuration for the GLM-5.3-Flash-VL vision tower (GlmOcr-based).

    Field names match the HuggingFace ``GlmOcrVisionConfig`` plus the
    GLM-5.3-Flash-specific ``projection_intermediate_size`` used by the SGLang
    adaptation to override the merger context dimension.

    ``projection_intermediate_size`` defaults to ``None``: when unset, the
    merger context_dim falls back to ``out_hidden_size * in_channels`` (the
    base GlmOcr behavior used by GLM-OCR). When set (GLM-5.3-Flash SGLang
    adaptation), it overrides the merger context_dim.
    """

    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int = 336
    patch_size: int = 14
    rms_norm_eps: float = 1e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 1536
    intermediate_size: int = 4096
    initializer_range: float = 0.02
    # GLM-5.3-Flash override: merger context dim. None = fall back to
    # out_hidden_size * in_channels (base GlmOcr / GLM-OCR behavior).
    projection_intermediate_size: Optional[int] = None
    # GLM-5.3-Flash SwiGLU clamp limit. HF's Glm5NextConfig injects this into the
    # vision config from text_config.swiglu_limit (always 10.0 for this
    # checkpoint). Applied to gate/up projections in VisionMLP + VisionPatchMerger.
    swiglu_limit: float = 10.0
    tp_size: int = 1
    tp_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> Glm5NextVisionConfig:
        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None and d[k] != -1 and d[k] != "":
                    return d[k]
            return default

        # The C++ ModelArgs default (``PROPERTY(float, swiglu_limit) = 0.0f``,
        # model_args.h) leaks into the flat config dict because the C++ side
        # does not carry the nested ``vision_config.swiglu_limit``. A picked
        # 0/0.0 is therefore the unset default, not a real limit — and a limit
        # of 0 would zero the SwiGLU gate/up (clamp(max=0) kills all positive
        # values), collapsing the whole ViT output to zeros. Fall back to the
        # HF default 10.0 (config.json vision_config.swiglu_limit).
        swiglu_limit = pick("swiglu_limit", "mm_swiglu_limit", default=10.0)
        if not swiglu_limit:
            swiglu_limit = 10.0

        return cls(
            depth=int(pick("mm_num_hidden_layers", "depth", default=24)),
            hidden_size=int(pick("mm_hidden_size", "hidden_size", default=1024)),
            hidden_act=str(pick("mm_hidden_act", "hidden_act", default="silu")),
            # NOTE: only the ``mm_``-prefixed key is honored. The C++ flat
            # config dict carries the TEXT model's ``attention_bias = false``
            # (ModelArgs default), which would silently disable every vision
            # bias (qkv/proj/mlp) even though the checkpoint ships them —
            # shape checks can't catch a missing bias.
            attention_bias=bool(pick("mm_attention_bias", default=True)),
            attention_dropout=float(pick("mm_dropout", "attention_dropout", default=0.0)),
            num_heads=int(pick("mm_num_attention_heads", "num_heads", default=16)),
            in_channels=int(pick("mm_num_channels", "in_channels", "in_chans", default=3)),
            image_size=int(pick("mm_image_size", "image_size", default=336)),
            patch_size=int(pick("mm_patch_size", "patch_size", default=14)),
            rms_norm_eps=float(pick("mm_layer_norm_eps", "rms_norm_eps", default=1e-5)),
            spatial_merge_size=int(pick("mm_spatial_merge_size", "spatial_merge_size", default=2)),
            temporal_patch_size=int(pick("mm_temporal_patch_size", "temporal_patch_size", default=2)),
            out_hidden_size=int(pick("mm_projection_dim", "out_hidden_size", default=1536)),
            intermediate_size=int(pick("mm_intermediate_size", "intermediate_size", default=4096)),
            initializer_range=float(pick("mm_initializer_range", "initializer_range", default=0.02)),
            projection_intermediate_size=pick(
                "projection_intermediate_size",
                "context_size",
                "mm_projection_intermediate_size",
                # GLM-5.3-Flash vision projector (config.json
                # vision_config.projection_intermediate_size = 10240). The C++
                # ModelArgs loader should expose it as mm_projection_intermediate_size;
                # 10240 is the fallback so the merger matches the checkpoint when the
                # C++ side hasn't been extended yet.
                default=10240,
            ),
            swiglu_limit=swiglu_limit,
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
        )

    def merger_context_dim(self) -> int:
        """Resolve the merger context_dim.

        - GLM-5.3-Flash (SGLang override): ``projection_intermediate_size``.
        - GLM-OCR (base): ``out_hidden_size * in_channels``.
        """
        if self.projection_intermediate_size is not None:
            return int(self.projection_intermediate_size)
        return self.out_hidden_size * self.in_channels


# ---------------------------------------------------------------------------
# Vision building blocks
# ---------------------------------------------------------------------------


class VisionRMSNorm(nn.Module):
    """Weighted RMSNorm with one rounding after FP32 normalization and scaling."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return kernels.fused_add_rms_norm(hidden_states, residual, self.weight, self.variance_epsilon)
        return kernels.rms_norm(hidden_states, self.weight, self.variance_epsilon)


class VisionPatchEmbed(nn.Module):
    """3D-conv patch embedding (temporal, height, width), no bias."""

    def __init__(self, cfg: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.patch_size = cfg.patch_size
        self.temporal_patch_size = cfg.temporal_patch_size
        self.in_channels = cfg.in_channels
        self.embed_dim = cfg.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class VisionPatchDownsample(nn.Conv2d):
    """Project each spatial merge group with the checkpoint's flattened kernel."""

    def __init__(self, cfg: Glm5NextVisionConfig) -> None:
        super().__init__(
            cfg.hidden_size,
            cfg.out_hidden_size,
            kernel_size=cfg.spatial_merge_size,
            stride=cfg.spatial_merge_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = F.linear(hidden_states.flatten(1), self.weight.flatten(1), self.bias)
        return projected.view(-1, self.out_channels, 1, 1)


class VisionRotaryEmbedding(nn.Module):
    """inv_freq table for the vision RoPE (half head_dim)."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> VisionRotaryEmbedding:
        # Model.to(bfloat16) must not round the frequencies before multiplying
        # by positions. Moving an already rounded table back to FP32 loses data.
        inv_freq = self.inv_freq
        super()._apply(fn, recurse)
        self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.device.type == "npu" and q.dtype == torch.bfloat16 and q.shape[-1] == 64:
        # RotaryMul keeps the FP32 multiply/add rounding used by the
        # reference while avoiding the separate casts and elementwise ops.
        cos_full = cos.unsqueeze(0).unsqueeze(-2)
        sin_full = sin.unsqueeze(0).unsqueeze(-2)
        return (
            kernels.vision_rotary_mul(q, cos_full, sin_full),
            kernels.vision_rotary_mul(k, cos_full, sin_full),
        )
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class VisionRowParallelLinear(RowParallelLinear):
    """Add the replicated bias once, inside rank zero's local GEMM."""

    def __init__(
        self,
        in_features_per_partition: int,
        out_features: int,
        tp_size: int,
        *,
        tp_rank: int,
        bias: bool,
    ) -> None:
        super().__init__(in_features_per_partition, out_features, tp_size, bias=bias)
        self.tp_rank = tp_rank

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        output = F.linear(hidden_states, self.weight, bias)
        if self.tp_size > 1:
            distributed.tp_all_reduce(output)
        return output


class VisionAttention(nn.Module):
    """MHA with QK-RMSNorm + RoPE, varlen via ``cu_seqlens`` chunking.

    Q/K/V are projected, Q and K get per-head RMSNorm, RoPE is applied, then
    each chunk is attended independently (non-causal). ``cu_seqlens`` lists
    cumulative chunk ends without the leading zero. The outputs are
    concatenated, matching the HF GlmOcr non-flash path.

    TP sharding (mirrors va ``Qwen2VisionAttention``):
      - ``qkv``: :class:`ColumnParallelLinear` (output dim sharded, per-head).
        Each rank owns ``num_heads // tp_size`` heads of ALL THREE of q/k/v, so
        the per-partition output is ``3 * num_heads_local * head_dim``. The
        varlen SDPA runs over the LOCAL heads only.
      - ``proj``: :class:`RowParallelLinear` (input dim sharded, all-reduced).
        Consumes the local-head attention output (``num_heads_local *
        head_dim``) and produces a partial ``dim`` output summed across ranks
        via all-reduce; the post-reduce output is the full ``dim`` so the
        surrounding residual stays replicated (identical on every rank).

    At ``tp_size==1`` ``num_heads_local == num_heads`` and the parallel layers
    skip their collectives, so the path is byte-identical to plain ``nn.Linear``.
    """

    def __init__(self, cfg: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = self.dim // self.num_heads
        assert self.num_heads % cfg.tp_size == 0, (
            f"vision num_heads {self.num_heads} not divisible by tp_size {cfg.tp_size}"
        )
        # Local (per-rank) head count: each rank computes its head-subset's
        # q/k/v and attends only those heads. The qkv ColumnParallelLinear owns
        # 3*num_heads_local*head_dim output channels (q/k/v each contribute
        # num_heads_local heads). At tp==1 this equals num_heads (no-op).
        self.num_heads_local = self.num_heads // cfg.tp_size
        self.tp_size = cfg.tp_size
        qkv_out_local = 3 * self.num_heads_local * self.head_dim
        self.qkv = ColumnParallelLinear(
            self.dim,
            qkv_out_local,
            cfg.tp_size,
            gather_output=False,
            bias=cfg.attention_bias,
        )
        self.proj = VisionRowParallelLinear(
            self.num_heads_local * self.head_dim,
            self.dim,
            cfg.tp_size,
            tp_rank=cfg.tp_rank,
            bias=cfg.attention_bias,
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.q_norm = VisionRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = VisionRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # Reshape over the LOCAL heads (num_heads_local); the qkv
        # ColumnParallelLinear emits [seq, 3*num_heads_local*head_dim].
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads_local, self.head_dim).permute(1, 0, 2, 3)
        query_states, key_states, value_states = qkv.unbind(0)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        attn_output = kernels.packed_infer_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            actual_seq_lengths=cu_seqlens,
            num_heads=self.num_heads_local,
            num_key_value_heads=self.num_heads_local,
            scale=self.scaling,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class VisionMLP(nn.Module):
    """Gated MLP: down_proj(act_fn(gate_proj(x)) * up_proj(x)).

    TP sharding (mirrors va ``Qwen2VisionMLP``): packed gate/up outputs
    are :class:`ColumnParallelLinear` (intermediate dim sharded, each rank owns
    ``intermediate_size // tp_size`` channels); ``down_proj`` is
    :class:`RowParallelLinear` (input dim sharded, all-reduced to the full
    ``hidden_size``). At ``tp_size==1`` the parallel layers hold full weights
    and skip their collectives, so the path is byte-identical to ``nn.Linear``.
    """

    def __init__(self, cfg: Glm5NextVisionConfig, bias: bool = True) -> None:
        super().__init__()
        assert cfg.intermediate_size % cfg.tp_size == 0, (
            f"vision intermediate_size {cfg.intermediate_size} not divisible by tp_size {cfg.tp_size}"
        )
        inter_local = cfg.intermediate_size // cfg.tp_size
        self.gate_up_proj = ColumnParallelLinear(
            cfg.hidden_size,
            2 * inter_local,
            cfg.tp_size,
            bias=bias,
        )
        self.down_proj = VisionRowParallelLinear(
            inter_local,
            cfg.hidden_size,
            cfg.tp_size,
            tp_rank=cfg.tp_rank,
            bias=bias,
        )
        if cfg.hidden_act == "silu":
            self.act_fn = F.silu
        elif cfg.hidden_act == "gelu":
            self.act_fn = F.gelu
        elif cfg.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            raise ValueError(f"Unsupported hidden_act: {cfg.hidden_act}")
        # GLM-5.3-Flash SwiGLU clamp (HF Glm5NextVisionMLP): clamp gate/up before
        # the gated multiply. gate is upper-clamped only; up is +/- clamped.
        self.swiglu_limit = cfg.swiglu_limit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_state)
        if self.act_fn is F.silu:
            return self.down_proj(_glm5_next_swiglu(gate_up, self.swiglu_limit))
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class VisionBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm: LN -> attn -> residual, LN -> mlp -> residual."""

    def __init__(self, cfg: Glm5NextVisionConfig) -> None:
        super().__init__()
        # Block and final tower norms use the checkpoint's vision epsilon.
        self.norm1 = VisionRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm2 = VisionRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg, bias=cfg.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        normalized, residual = self.norm2(hidden_states, residual=attn_output)
        return residual + self.mlp(normalized)


class VisionPatchMerger(nn.Module):
    """Patch merger matching ``GlmOcrVisionPatchMerger``.

    proj -> GELU(LayerNorm(proj(x))) -> down_proj(act_fn(gate_proj(x)) * up_proj(x))

    For GLM-5.3-Flash: ``dim = out_hidden_size``, ``context_dim =
    projection_intermediate_size``, ``bias = False``.

    The first projection is column-sharded and gathered before LayerNorm.
    Gate/up are packed and column-sharded; down_proj is row-sharded and
    reduced. These GEMM shapes and the bias placement are part of the BF16
    arithmetic contract used by the GLM reference implementation.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_act: str,
        bias: bool = False,
        tp_size: int = 1,
        swiglu_limit: float = 10.0,
        tp_rank: int = 0,
    ) -> None:
        super().__init__()
        assert context_dim % tp_size == 0, f"merger context_dim {context_dim} not divisible by tp_size {tp_size}"
        self.proj = ColumnParallelLinear(dim, dim // tp_size, tp_size, gather_output=True, bias=bias)
        self.post_projection_norm = nn.LayerNorm(dim)
        context_local = context_dim // tp_size
        self.gate_up_proj = ColumnParallelLinear(dim, 2 * context_local, tp_size, bias=bias)
        self.down_proj = VisionRowParallelLinear(context_local, dim, tp_size, tp_rank=tp_rank, bias=bias)
        self.act1 = nn.GELU()
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")
        # GLM-5.3-Flash SwiGLU clamp (HF Glm5NextVisionPatchMerger).
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        gate_up = self.gate_up_proj(hidden_state)
        if self.act_fn is F.silu:
            return self.down_proj(_glm5_next_swiglu(gate_up, self.swiglu_limit))
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class VisionModelOutput:
    """Mirror of HF ``BaseModelOutputWithPooling`` (used fields only).

    - ``last_hidden_state``: post-downsample hidden states ``(total_tokens, out_hidden_size)``.
    - ``pooler_output``: post-merger tokens ``(total_tokens, out_hidden_size)``.
    """

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


# ---------------------------------------------------------------------------
# Full vision model
# ---------------------------------------------------------------------------


class Glm5NextVisionModel(nn.Module):
    """GLM-5.3-Flash-VL vision tower (GlmOcr-based), checkpoint-compatible with HF.

    Forward contract matches the GLM-5.3-Flash-VL ``self.visual`` usage:
    ``self.visual(pixel_values, grid_thw=image_grid_thw)`` returns an object with
    ``.pooler_output`` and ``.last_hidden_state``.

    The merger uses ``projection_intermediate_size`` as context_dim (SGLang
    adaptation), overriding the base GlmOcr default of
    ``out_hidden_size * in_channels``.
    """

    def __init__(
        self,
        cfg: Glm5NextVisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.device = device

        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.patch_embed = VisionPatchEmbed(cfg)

        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.depth)])
        # GLM-5.3-Flash merger: context_dim from projection_intermediate_size if
        # set, else fall back to GlmOcr base (out_hidden_size * in_channels).
        # bias=False matches the SGLang adaptation. Pass tp_size so the
        # gate/up/down projections shard per the va Qwen2VisionPatchMerger map.
        self.merger = VisionPatchMerger(
            dim=cfg.out_hidden_size,
            context_dim=cfg.merger_context_dim(),
            hidden_act=cfg.hidden_act,
            bias=False,
            tp_size=cfg.tp_size,
            swiglu_limit=cfg.swiglu_limit,
            tp_rank=cfg.tp_rank,
        )
        self.downsample = VisionPatchDownsample(cfg)
        # Keep the final norm consistent with the configured block norms.
        self.post_layernorm = VisionRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        self.to(dtype=dtype, device=device)

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> VisionModelOutput:
        """Run the vision tower.

        Args:
            hidden_states: ``pixel_values`` reshaped to
                ``(total_patches, in_channels * temporal_patch_size *
                patch_size * patch_size)``.
            grid_thw: ``(num_images, 3)`` T/H/W per image.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)[1:].tolist()

        hidden_states = self.patch_embed(hidden_states)
        rotary_emb = self.rotary_pos_emb(position_ids)
        emb = torch.cat((rotary_emb, rotary_emb), dim=-1)
        position_embeddings = (emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype))

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.post_layernorm(hidden_states)

        # Spatial merge via Conv2d downsample.
        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.cfg.out_hidden_size)

        merged_hidden_states = self.merger(hidden_states)

        return VisionModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    # -- weight loading ----------------------------------------------------

    def load_weights(self, state_dicts: list, prefix: str = "model.visual.") -> None:
        """Load HF checkpoint weights into this vision tower.

        ``state_dicts`` is the list of ``StateDict`` objects from the xLLM weight
        loader. ``prefix`` is the key prefix in the checkpoint (``model.visual.``
        for a full GLM-5.3-Flash-VL checkpoint).
        """

        def find(name: str):
            full = prefix + name
            for sd in state_dicts:
                if sd.has(full):
                    return sd
            return None

        def load_tensor(name: str) -> torch.Tensor:
            sd = find(name)
            assert sd is not None, f"checkpoint tensor not found: {prefix + name}"
            return sd.get_tensor(prefix + name)

        def copy_in(param_name: str, tensor: torch.Tensor) -> None:
            param = self.get_parameter(param_name)
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))

        tp = self.cfg.tp_size
        rank = self.cfg.tp_rank

        def shard(t: torch.Tensor, dim: int) -> torch.Tensor:
            """Narrow ``t`` to this rank's contiguous slice along ``dim``.

            Mirrors ``W8A8WeightLoader.shard`` / the LLM loader. At ``tp==1`` it
            is a no-op so the byte-identical replicated load is preserved.
            """
            if tp <= 1:
                return t
            cs = t.size(dim) // tp
            return t.narrow(dim, rank * cs, cs).contiguous()

        def shard_qkv(t: torch.Tensor) -> torch.Tensor:
            """Shard the merged qkv weight/bias per-head across TP ranks.

            The checkpoint layout is ``[q(num_heads*hd) | k(...) | v(...)]``
            along dim 0 (the output dim of the ``nn.Linear(dim, 3*dim)``). A
            contiguous ``shard(t, dim=0)`` would split across the q/k boundary
            (rank 0 of a tp=3 run would own all of q and nothing of k/v), so
            each third is narrowed to this rank's head block SEPARATELY and
            re-cat'd, giving the local layout ``[q_local | k_local | v_local]``
            that matches the ``reshape(seq, 3, num_heads_local, head_dim)`` in
            :meth:`VisionAttention.forward`. Mirrors the KDA conv1d / MoE
            gate_up shard-then-cat fix in ``glm5_next.py``. At ``tp==1`` the
            shard is a no-op, leaving the merged tensor byte-identical.
            """
            if tp <= 1:
                return t
            third = t.size(0) // 3
            parts = [shard(t.narrow(0, j * third, third), dim=0) for j in range(3)]
            return torch.cat(parts, dim=0).contiguous()

        # patch_embed (Conv3d, REPLICATED — Conv TP-sharding is non-standard).
        copy_in("patch_embed.proj.weight", load_tensor("patch_embed.proj.weight"))
        copy_in("patch_embed.proj.bias", load_tensor("patch_embed.proj.bias"))

        # blocks
        for i in range(self.cfg.depth):
            p = f"blocks.{i}."
            # norms (RMSNorm, replicated).
            copy_in(p + "norm1.weight", load_tensor(p + "norm1.weight"))
            copy_in(p + "norm2.weight", load_tensor(p + "norm2.weight"))
            copy_in(p + "attn.q_norm.weight", load_tensor(p + "attn.q_norm.weight"))
            copy_in(p + "attn.k_norm.weight", load_tensor(p + "attn.k_norm.weight"))
            # attn.qkv: ColumnParallelLinear — shard the merged qkv per-head.
            copy_in(p + "attn.qkv.weight", shard_qkv(load_tensor(p + "attn.qkv.weight")))
            # attn.proj: RowParallelLinear — shard the INPUT dim (dim 1); the
            # The replicated output bias is included in rank zero's GEMM.
            copy_in(p + "attn.proj.weight", shard(load_tensor(p + "attn.proj.weight"), dim=1))
            # mlp.gate/up: ColumnParallelLinear — shard the intermediate (dim 0).
            copy_in(
                p + "mlp.gate_up_proj.weight",
                torch.cat(
                    (
                        shard(load_tensor(p + "mlp.gate_proj.weight"), dim=0),
                        shard(load_tensor(p + "mlp.up_proj.weight"), dim=0),
                    ),
                    dim=0,
                ),
            )
            # mlp.down_proj: RowParallelLinear — shard the input dim (dim 1);
            # bias replicated.
            copy_in(p + "mlp.down_proj.weight", shard(load_tensor(p + "mlp.down_proj.weight"), dim=1))
            if self.cfg.attention_bias:
                # qkv bias follows the same per-head shard as the weight; the
                # gate/up ColumnParallel biases also shard on dim 0; the
                # RowParallel (proj, down_proj) biases are replicated.
                copy_in(p + "attn.qkv.bias", shard_qkv(load_tensor(p + "attn.qkv.bias")))
                copy_in(p + "attn.proj.bias", load_tensor(p + "attn.proj.bias"))
                copy_in(
                    p + "mlp.gate_up_proj.bias",
                    torch.cat(
                        (
                            shard(load_tensor(p + "mlp.gate_proj.bias"), dim=0),
                            shard(load_tensor(p + "mlp.up_proj.bias"), dim=0),
                        ),
                        dim=0,
                    ),
                )
                copy_in(p + "mlp.down_proj.bias", load_tensor(p + "mlp.down_proj.bias"))

        # post_layernorm (RMSNorm, replicated).
        copy_in("post_layernorm.weight", load_tensor("post_layernorm.weight"))

        # downsample (Conv2d, REPLICATED).
        copy_in("downsample.weight", load_tensor("downsample.weight"))
        copy_in("downsample.bias", load_tensor("downsample.bias"))

        # Merger proj is column-sharded and gathered before LayerNorm; gate/up
        # are packed from separately sharded checkpoint tensors. Down projection
        # shards dim 1. The GLM merger has no projection biases.
        copy_in("merger.proj.weight", shard(load_tensor("merger.proj.weight"), dim=0))
        copy_in("merger.post_projection_norm.weight", load_tensor("merger.post_projection_norm.weight"))
        copy_in("merger.post_projection_norm.bias", load_tensor("merger.post_projection_norm.bias"))
        copy_in(
            "merger.gate_up_proj.weight",
            torch.cat(
                (
                    shard(load_tensor("merger.gate_proj.weight"), dim=0),
                    shard(load_tensor("merger.up_proj.weight"), dim=0),
                ),
                dim=0,
            ),
        )
        copy_in("merger.down_proj.weight", shard(load_tensor("merger.down_proj.weight"), dim=1))

        # ViT RowParallelLinear: do NOT call ``process_weights_after_loading``
        # (which would transpose + Fractal-NZ-cast the weight and route forward
        # through ``torch.matmul``). The ViT is bf16 and unquantized; keeping
        # the checkpoint ``[N, K]`` layout makes ``RowParallelLinear.forward``
        # fall through to ``F.linear`` — bit-identical to the transformers
        # reference (verified per-block, 317/317 tensors, with fused-bias). The
        # NZ-fractal matmul is a pure NPU perf optimization that the small ViT
        # (one forward per prefill) does not need.


# ---------------------------------------------------------------------------
# Top-level conditional generation model
# ---------------------------------------------------------------------------


def _load_vision_output_rotation(
    model_path: str | None, vision_model: nn.Module, hidden_size: int
) -> torch.Tensor | None:
    """Adapt a separate, unexported vision tower to a ModelSlim-rotated LLM."""
    description = read_quant_description(model_path)
    if description is None:
        return None
    is_rot_used = description.get("is_rot_used", False)
    if not isinstance(is_rot_used, bool):
        raise ValueError("quant_model_description.json is_rot_used must be a boolean")
    if not is_rot_used:
        return None

    # The export manifest uses separate gate/up keys, while the runtime packs
    # them. Full FLOAT coverage means ModelSlim already folded Q into down_proj;
    # zero coverage identifies a separate vision tower that still needs Q.
    expected_visual = set()
    for name, _ in vision_model.named_parameters():
        names = (name.replace("gate_up_proj", "gate_proj"), name.replace("gate_up_proj", "up_proj"))
        expected_visual.update("model.visual." + key for key in names)
    exported_visual = {key for key in description if key.startswith("model.visual.")}
    if exported_visual:
        if exported_visual != expected_visual or any(
            quant_description_weight_is_quantized(description, key) is not False for key in exported_visual
        ):
            raise ValueError(
                "Rotated GLM visual export must have complete FLOAT manifest coverage or no visual entries"
            )
        return None

    optional = description.get("optional")
    quarot = optional.get("quarot") if isinstance(optional, dict) else None
    rotation_map = quarot.get("rotation_map") if isinstance(quarot, dict) else None
    mapped_path = rotation_map.get("global_rotation") if isinstance(rotation_map, dict) else None
    if not isinstance(mapped_path, str) or not mapped_path.strip():
        raise ValueError("Unexported GLM vision requires optional.quarot.rotation_map.global_rotation")
    rotation_path = Path(mapped_path)
    if rotation_path.is_absolute() or ".." in rotation_path.parts:
        raise ValueError("optional global_rotation must map to a checkpoint-relative file")
    with safe_open(Path(model_path) / rotation_path, framework="pt", device="cpu") as tensors:
        if "global_rotation" not in tensors.keys():
            raise ValueError("Mapped QuaRot file must contain global_rotation, not rot.weight")
        rotation = tensors.get_tensor("global_rotation")
    if rotation.shape != (hidden_size, hidden_size):
        raise ValueError(f"global_rotation must have shape ({hidden_size}, {hidden_size}), got {tuple(rotation.shape)}")
    if not rotation.is_floating_point() or not torch.isfinite(rotation).all().item():
        raise ValueError("global_rotation must be a finite floating-point matrix")
    return rotation.float()


class Glm5NextVLModel(Glm5NextForCausalLM):
    """GLM-5.3-Flash-VL composition: ViT + LLM + lm_head.

    Wiring (``forward``):
      pixel_values + grid_thw → vision_model → image_embeds
      input_ids → embed_tokens → text_embeds
      text_embeds.masked_scatter(image_mask, image_embeds) → inputs_embeds
      inputs_embeds + positions → language_model → hidden → lm_head

    The ``self.model`` attribute (required by ``PyModelBase``) is set to the
    language model, so the executor's runner calls
    ``language_model(input_ids, positions, inputs_embeds=...)``. Vision encoding happens in
    ``encode_multimodal`` / ``prepare_multimodal_inputs`` before the runner kicks in.
    """

    cache_multimodal_embeddings = True

    def __init__(self, config: dict) -> None:
        # NOTE: we inherit Glm5NextForCausalLM to reuse its weight-loader helpers
        # (_load_kda_attn / _load_dsa_attn / _load_mlp / ...), but we deliberately
        # do NOT call its __init__ (it requires a text-only config and would build
        # a redundant language model). Initialize nn.Module directly instead.
        nn.Module.__init__(self)

        # PyCausalLM hands us a FLAT ModelArgs dict (built by build_config_dict
        # via visit_properties): text fields are top-level (hidden_size,
        # n_layers, n_heads, tie_word_embeddings, tp_size, ...), vision fields
        # are "mm_"-prefixed. Both Glm5NextTextConfig.from_dict and
        # Glm5NextVisionConfig.from_dict read this flat layout directly.
        # Fall back to nested vision_config/text_config for standalone tests.
        vision_cfg_dict = config.get("vision_config", config)

        vcfg = Glm5NextVisionConfig.from_dict(vision_cfg_dict)
        vcfg.tp_size = int(config.get("tp_size", 1))
        vcfg.tp_rank = int(config.get("tp_rank", 0))

        # Merge runtime overrides before deriving MoE-TP defaults or validating
        # the text configuration. Mutating attention TP afterwards leaves the
        # derived MoE axes stale for nested text_config inputs.
        tcfg = Glm5NextConfig.from_dict(config)

        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device

        # --- Vision tower ---
        self.vision_cfg = vcfg
        self.vision_model = Glm5NextVisionModel(vcfg, dtype=dtype, device=device)

        # --- Language model ---
        self.text_cfg = tcfg
        # Register under ``self.model`` FIRST (the primary name). The weight
        # loader builds its param table from ``named_parameters()``, and PyTorch
        # de-duplicates a module bound to two attribute names — the prefix of
        # the FIRST binding is the one kept. Glm5NextForCausalLM.load_weights
        # (delegated below) looks up params as ``model.<...>``, so ``self.model``
        # must be registered before the ``self.language_model`` alias.
        self.model = Glm5NextModel(tcfg, dtype=dtype, device=device)
        self.language_model = self.model

        # --- LM head ---
        tp = tcfg.tp_size
        assert tcfg.vocab_size % tp == 0
        self.lm_head = ColumnParallelLinear(
            tcfg.hidden_size,
            tcfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        # Cast the whole graph (vision + LM + lm_head) to the target dtype/device
        # — mirrors Glm5NextForCausalLM.__init__ so KDA conv1d / indexer params
        # (which default to float32 inside Glm5NextModel) are bf16 like the rest.
        self.to(device=device, dtype=dtype)

        # Register after the whole-model cast so Q retains FP32 precision.
        rotation = _load_vision_output_rotation(config.get("model_path"), self.vision_model, tcfg.hidden_size)
        if rotation is not None and vcfg.out_hidden_size != tcfg.hidden_size:
            raise ValueError("Rotated GLM vision output width must match the language model hidden_size")
        self.register_buffer(
            "_vision_output_rotation", rotation.to(device=device) if rotation is not None else None, persistent=False
        )

    # ------------------------------------------------------------------
    # Connection logic: ViT → LLM
    # ------------------------------------------------------------------

    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 1: vision encode → image_embeds.

        Args:
            pixel_values: ``(total_patches, C*t*p*p)`` flattened patches.
            grid_thw: ``(num_images, 3)`` T/H/W per image.

        Returns:
            ``image_embeds`` of shape ``(total_image_tokens, out_hidden_size)``
            where ``total_image_tokens = sum(grid_thw.prod(-1) // merge**2)``.
        """
        pixel_values = pixel_values.type(self.vision_model.dtype)
        out = self.vision_model(pixel_values, grid_thw)
        features = out.pooler_output
        if self._vision_output_rotation is not None:
            if features.shape[-1] != self._vision_output_rotation.shape[0]:
                raise ValueError("GLM vision feature width does not match global_rotation")
            features = (features.float() @ self._vision_output_rotation).to(dtype=features.dtype)
        return features

    def encode_multimodal(self, data: MultimodalData, token_counts: EncoderTokenCounts) -> EncoderOutputs:
        return encode_batched_vision(self.encode, data, token_counts)

    def prepare_multimodal_inputs(self, input_ids: torch.Tensor, data: MultimodalData) -> MultimodalInputs:
        return MultimodalInputs(merge_multimodal_embeddings(self.language_model.embed_tokens(input_ids), data))

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Load ViT + LLM + lm_head weights from checkpoint.

        Checkpoint key layout (GLM-5.3-Flash-VL):
          - ``model.visual.*``      → vision_model
          - ``model.language_model.*`` → language_model (Side A aliases
            ``model.`` ↔ ``model.language_model.`` so both prefixes load)
          - ``lm_head.*``           → lm_head (or tied to embed_tokens)
        """
        # 1. Vision tower — TP-sharded per rank (reads self.cfg.tp_size /
        #    tp_rank set in __init__ from the flat ModelArgs). Conv + norms +
        #    merger.proj stay replicated; qkv/proj/gate/up/down shard per the
        #    va Qwen2-VL ViT map (see Glm5NextVisionModel.load_weights).
        self.vision_model.load_weights(state_dicts, prefix="model.visual.")

        # 2. LLM + lm_head: delegate to the full Glm5NextForCausalLM loader
        #    (handles MoE expert stacking, absorbed-MLA W_UK/W_UV split, mHC,
        #    TP sharding, and the model.↔model.language_model. checkpoint alias).
        self.cfg = self.text_cfg
        Glm5NextForCausalLM.load_weights(self, state_dicts, tp_rank, tp_size)
