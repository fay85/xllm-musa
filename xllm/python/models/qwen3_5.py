# Copyright 2025-2026 The xLLM Authors.
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

"""Qwen3.5 hybrid-attention causal LM for the MUSA Python executor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GemmaRMSNorm,
    HiddenParallelEmbedding,
    RowParallelLinear,
)
from xllm.python.layers.gated_delta_net import Qwen3_5GatedDeltaNet
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.model_executor.prefill_breakdown import scope as breakdown_scope
from xllm.python.models.base import PyModelBase
from xllm.python.platform import current_platform


@dataclass
class Qwen3_5Config:
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    vocab_size: int
    layer_types: list[str]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    attention_bias: bool
    attn_output_gate: bool
    use_sliding_window: bool
    sliding_window: int
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: list[int]
    norm_topk_prob: bool
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    tp_size: int
    tp_rank: int
    dp_size: int
    dp_rank: int
    world_size: int
    moe_tp_size: int
    moe_tp_rank: int
    ep_size: int
    ep_rank: int

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3_5Config:
        def _pick(*keys: str, default: object = None) -> object:
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return default

        n_layers = int(_pick("n_layers", "num_hidden_layers", default=0))
        interval = int(_pick("full_attention_interval", default=4))
        layer_types = list(_pick("layer_types", default=[]))
        if not layer_types:
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(n_layers)
            ]
        if len(layer_types) != n_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")

        hidden_size = int(_pick("hidden_size", default=0))
        n_heads = int(_pick("n_heads", "num_attention_heads", default=0))
        tp_size = int(_pick("tp_size", default=1))
        dp_size = int(_pick("dp_size", default=1))
        world_size = int(_pick("world_size", default=tp_size * dp_size))
        ep_size = int(_pick("ep_size", default=1))
        mlp_only = _pick("mlp_only_layers", default=[])
        return cls(
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=int(_pick("n_kv_heads", "num_key_value_heads", default=0)),
            head_dim=int(_pick("head_dim", default=hidden_size // max(n_heads, 1))),
            intermediate_size=int(_pick("intermediate_size", default=0)),
            rms_norm_eps=float(_pick("rms_norm_eps", default=1e-6)),
            rope_theta=float(_pick("rope_theta", default=1e7)),
            partial_rotary_factor=float(_pick("partial_rotary_factor", default=0.25)),
            max_position_embeddings=int(
                _pick("max_position_embeddings", default=262144)
            ),
            vocab_size=int(_pick("vocab_size", default=248320)),
            layer_types=layer_types,
            linear_conv_kernel_dim=int(_pick("linear_conv_kernel_dim", default=4)),
            linear_key_head_dim=int(_pick("linear_key_head_dim", default=128)),
            linear_value_head_dim=int(_pick("linear_value_head_dim", default=128)),
            linear_num_key_heads=int(_pick("linear_num_key_heads", default=16)),
            linear_num_value_heads=int(_pick("linear_num_value_heads", default=48)),
            attention_bias=bool(_pick("attention_bias", default=False)),
            attn_output_gate=bool(_pick("attn_output_gate", default=True)),
            use_sliding_window=bool(_pick("use_sliding_window", default=False)),
            sliding_window=int(_pick("sliding_window", default=4096)),
            tie_word_embeddings=bool(_pick("tie_word_embeddings", default=False)),
            num_experts=int(_pick("num_experts", "n_routed_experts", default=0)),
            num_experts_per_tok=int(_pick("num_experts_per_tok", default=0)),
            decoder_sparse_step=int(_pick("decoder_sparse_step", default=1)),
            mlp_only_layers=[int(layer_id) for layer_id in mlp_only],
            norm_topk_prob=bool(_pick("norm_topk_prob", default=True)),
            moe_intermediate_size=int(_pick("moe_intermediate_size", default=0)),
            shared_expert_intermediate_size=int(
                _pick("shared_expert_intermediate_size", default=0)
            ),
            tp_size=tp_size,
            tp_rank=int(_pick("tp_rank", default=0)),
            dp_size=dp_size,
            dp_rank=int(_pick("dp_rank", default=0)),
            world_size=world_size,
            moe_tp_size=int(
                _pick("moe_tp_size", default=world_size // max(ep_size, 1))
            ),
            moe_tp_rank=int(_pick("moe_tp_rank", default=0)),
            ep_size=ep_size,
            ep_rank=int(_pick("ep_rank", default=0)),
        )

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("invalid Qwen3.5 model dimensions")
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if self.num_experts > 0:
            if current_platform.is_musa():
                raise ValueError(
                    "MUSA Python Qwen3.5 currently supports dense 27B only"
                )
            if min(self.dp_size, self.moe_tp_size, self.ep_size) <= 0:
                raise ValueError("parallel sizes must be positive")
            if self.tp_size * self.dp_size != self.world_size:
                raise ValueError("world_size must equal tp_size * dp_size")
            if self.ep_size not in (1, self.world_size):
                raise ValueError(
                    "Qwen3.5 Python supports only ep_size=1 or world_size"
                )
            if self.moe_tp_size * self.ep_size != self.world_size:
                raise ValueError("world_size must equal moe_tp_size * ep_size")
            if not 0 <= self.dp_rank < self.dp_size:
                raise ValueError("dp_rank must be in [0, dp_size)")
            if not 0 <= self.moe_tp_rank < self.moe_tp_size:
                raise ValueError("moe_tp_rank must be in [0, moe_tp_size)")
            if not 0 <= self.ep_rank < self.ep_size:
                raise ValueError("ep_rank must be in [0, ep_size)")
            if self.decoder_sparse_step <= 0:
                raise ValueError("decoder_sparse_step must be positive")
            if self.num_experts_per_tok <= 0:
                raise ValueError("num_experts_per_tok must be positive for MoE")
            if self.num_experts % self.ep_size:
                raise ValueError("num_experts must be divisible by ep_size")
            if self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive for MoE")
            if self.moe_intermediate_size % self.moe_tp_size:
                raise ValueError(
                    "moe_intermediate_size must be divisible by moe_tp_size"
                )
            if self.shared_expert_intermediate_size <= 0:
                raise ValueError(
                    "shared_expert_intermediate_size must be positive for Qwen3.5 MoE"
                )
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if self.use_sliding_window and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when enabled")
        for name, count in (
            ("attention heads", self.n_heads),
            ("linear key heads", self.linear_num_key_heads),
            ("linear value heads", self.linear_num_value_heads),
        ):
            if count % self.tp_size:
                raise ValueError(f"{name} must be divisible by tp_size")

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts > 0
            and (layer_id + 1) % self.decoder_sparse_step == 0
            and layer_id not in self.mlp_only_layers
        )

    def full_head_split(self) -> tuple[int, int, int]:
        num_heads = self.n_heads // self.tp_size
        if self.n_kv_heads >= self.tp_size:
            if self.n_kv_heads % self.tp_size:
                raise ValueError("num_key_value_heads must be divisible by tp_size")
            return num_heads, self.n_kv_heads // self.tp_size, 1
        if self.tp_size % self.n_kv_heads:
            raise ValueError("tp_size must be divisible by num_key_value_heads")
        return num_heads, 1, self.tp_size // self.n_kv_heads


def _reorder_gated_q_rows(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Turn per-head [q|g] rows into grouped [Q|G] rows."""
    hidden = weight.size(-1)
    grouped = weight.view(num_heads, 2 * head_dim, hidden)
    query = grouped[:, :head_dim].reshape(num_heads * head_dim, hidden)
    gate = grouped[:, head_dim:].reshape(num_heads * head_dim, hidden)
    return torch.cat((query, gate), dim=0)


def _reorder_gated_q_scale(
    scale: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_n: int = 128,
) -> torch.Tensor:
    """Reorder block-FP8 N-tiles to match grouped [Q|G] rows."""
    if head_dim % block_n:
        raise ValueError("gated QKV FP8 reorder requires head_dim % block_n == 0")
    tiles_per_head = (2 * head_dim) // block_n
    query_tiles = head_dim // block_n
    tiled = scale.view(num_heads, tiles_per_head, scale.size(-1))
    query_scale = tiled[:, :query_tiles].reshape(num_heads * query_tiles, -1)
    gate_scale = tiled[:, query_tiles:].reshape(num_heads * query_tiles, -1)
    return torch.cat((query_scale, gate_scale), dim=0)


def _store_fused_qk_norm_weight(norm: GemmaRMSNorm, dtype: torch.dtype) -> None:
    """Keep Gemma Q/K weights in the fused-kernel activation dtype.

    ``fused_qk_norm_rope`` reads ``q_weight`` / ``k_weight`` as the same
    scalar type as ``qkv``. GemmaRMSNorm stores float32, so a raw Parameter
    would be alias-read as bf16 and silently corrupt QK scales.
    """
    weight = norm.weight.data
    if weight.dtype != dtype or not weight.is_contiguous():
        norm.weight.data = weight.to(dtype=dtype).contiguous()


def pack_gated_qkv_projection(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_scale: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pack checkpoint Q/K/V into the C++ [Q|G|K|V] fused-RoPE layout."""
    packed_query = _reorder_gated_q_rows(query, num_heads, head_dim)
    packed = torch.cat((packed_query, key, value), dim=0)
    if query_scale is None or key_scale is None or value_scale is None:
        return packed, None
    packed_query_scale = _reorder_gated_q_scale(query_scale, num_heads, head_dim)
    packed_scale = torch.cat((packed_query_scale, key_scale, value_scale), dim=0)
    return packed, packed_scale


class Qwen3_5SparseMoEBlock(nn.Module):
    def __init__(
        self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        from xllm.python import distributed
        from xllm.python.layers.fused_moe import FusedMoE

        # Routed and shared branches share the same ranks when every group
        # spans the whole world, so one fused all-reduce is exact. DP splits
        # those groups, so each branch keeps its own reduction.
        self._distributed = distributed
        self.fuse_reductions = cfg.dp_size == 1 and cfg.tp_size > 1
        self.experts = FusedMoE(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_tok,
            renormalize=cfg.norm_topk_prob,
            moe_tp_size=cfg.moe_tp_size,
            moe_tp_rank=cfg.moe_tp_rank,
            ep_size=cfg.ep_size,
            ep_rank=cfg.ep_rank,
            dp_size=cfg.dp_size,
            dp_rank=cfg.dp_rank,
            dtype=dtype,
            device=device,
            reduce_results=not self.fuse_reductions,
        )
        self.shared_expert = GatedMLP(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            cfg.tp_size,
            dtype,
            device,
            reduce_results=not self.fuse_reductions,
        )
        self.shared_expert_gate = nn.Linear(
            cfg.hidden_size, 1, bias=False, dtype=dtype, device=device
        )

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        self.shared_expert.reserve_graph_workspace(max_tokens)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        routed = self.experts(hidden)
        shared = self.shared_expert(hidden)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden))
        output = routed + shared * shared_gate
        if self.fuse_reductions:
            self._distributed.all_reduce_(output)
        return output


class PartialRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        rope_theta: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("partial rotary dimension must be positive and even")
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                / rotary_dim
            )
        )
        freqs = torch.outer(
            torch.arange(max_position, dtype=torch.float32, device=device), inv_freq
        )
        cos_half = freqs.cos().to(dtype)
        sin_half = freqs.sin().to(dtype)
        self.register_buffer("cos", cos_half, persistent=False)
        self.register_buffer("sin", sin_half, persistent=False)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((cos_half, sin_half), dim=-1).contiguous(),
            persistent=False,
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        rotary, passthrough = x.split(
            [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1
        )
        pos = positions.to(torch.long)
        cos = torch.cat((self.cos[pos], self.cos[pos]), dim=-1).unsqueeze(1)
        sin = torch.cat((self.sin[pos], self.sin[pos]), dim=-1).unsqueeze(1)
        rotary = rotary * cos + self._rotate_half(rotary) * sin
        return torch.cat((rotary, passthrough), dim=-1)


class Qwen3_5Attention(nn.Module):
    def __init__(
        self,
        cfg: Qwen3_5Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads, self.num_kv_heads, _ = cfg.full_head_split()
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        q_multiplier = 2 if cfg.attn_output_gate else 1
        self.attn_output_gate = cfg.attn_output_gate
        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            q_multiplier * self.q_size + 2 * self.kv_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = GemmaRMSNorm(
            self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.k_norm = GemmaRMSNorm(
            self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.rotary = rotary
        self.attn = Attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim**-0.5,
            cfg.sliding_window if cfg.use_sliding_window else -1,
            layer_id,
        )

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        self.qkv_proj.reserve_graph_workspace(max_tokens)
        self.o_proj.reserve_graph_workspace(max_tokens)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        with breakdown_scope("full_qkv"):
            qkv = self.qkv_proj(hidden)
        q_weight = self.q_norm.weight
        k_weight = self.k_norm.weight
        if q_weight.dtype != qkv.dtype or k_weight.dtype != qkv.dtype:
            raise RuntimeError(
                "fused QK-norm weights must match qkv dtype "
                f"(q={q_weight.dtype}, k={k_weight.dtype}, qkv={qkv.dtype})"
            )
        k_head_offset = 2 * self.num_heads if self.attn_output_gate else 0
        with breakdown_scope("full_prep"):
            q, k, v = kernels.fused_qk_norm_rope(
                qkv,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.q_norm.eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=self.rotary.cos_sin_cache,
                position_ids=positions,
                k_head_offset=k_head_offset,
            )
        with breakdown_scope("full_fa"):
            output = self.attn(q, k, v)
        with breakdown_scope("full_o_proj"):
            if self.attn_output_gate:
                gate = qkv[:, self.q_size : 2 * self.q_size]
                if current_platform.is_musa():
                    kernels.mul_sigmoid_gate_inplace(output, gate)
                else:
                    output = output * torch.sigmoid(gate)
            return self.o_proj(output)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3_5Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_type = cfg.layer_types[layer_id]
        self.input_layernorm = GemmaRMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(cfg, layer_id, dtype, device, rotary)
        elif self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(cfg, layer_id, dtype, device)
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.layer_type}")
        self.post_attention_layernorm = GemmaRMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        if cfg.is_moe_layer(layer_id):
            self.mlp = Qwen3_5SparseMoEBlock(cfg, dtype, device)
        else:
            self.mlp = GatedMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.tp_size,
                dtype,
                device,
            )

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        self.input_layernorm.reserve_graph_workspace(max_tokens)
        if self.layer_type == "full_attention":
            self.self_attn.reserve_graph_workspace(max_tokens)
        else:
            self.linear_attn.reserve_graph_workspace(max_tokens)
        self.post_attention_layernorm.reserve_graph_workspace(max_tokens)
        self.mlp.reserve_graph_workspace(max_tokens)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with breakdown_scope("norm"):
            if residual is None:
                residual = hidden
                hidden = self.input_layernorm(hidden)
            else:
                hidden, residual = self.input_layernorm(hidden, residual)
        if self.layer_type == "full_attention":
            with breakdown_scope("full_attn"):
                hidden = self.self_attn(positions, hidden)
        else:
            with breakdown_scope("gdn_attn"):
                hidden = self.linear_attn(hidden)
        with breakdown_scope("norm"):
            hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.tp_size:
            raise ValueError("hidden_size must be divisible by tp_size")
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // cfg.tp_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        self.rotary = PartialRotaryEmbedding(
            cfg.head_dim,
            rotary_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype,
            device,
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(cfg, i, dtype, device, self.rotary)
            for i in range(cfg.n_layers)
        )
        self.norm = GemmaRMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )

    def reserve_graph_workspaces(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise RuntimeError("Qwen3.5 graph reserve requires max_tokens > 0")
        for layer in self.layers:
            layer.reserve_graph_workspace(max_tokens)
        self.norm.reserve_graph_workspace(max_tokens)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        with breakdown_scope("embed"):
            hidden = self.embed_tokens(input_ids)
            if current_platform.is_musa():
                if positions.dtype != torch.int32:
                    positions = positions.to(torch.int32)
                if not positions.is_contiguous():
                    positions = positions.contiguous()
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions)
        with breakdown_scope("norm"):
            hidden, _ = self.norm(hidden, residual)
        return hidden


class Qwen3_5ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen3_5Config.from_dict(config)
        self.cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        if self.cfg.vocab_size % self.cfg.tp_size:
            raise ValueError("vocab_size must be divisible by tp_size")
        self.model = Qwen3_5Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def reserve_graph_workspaces(self, max_tokens: int) -> None:
        self.model.reserve_graph_workspaces(max_tokens)
        self.lm_head.reserve_graph_workspace(max_tokens)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        cfg = self.cfg

        def find(name: str) -> object | None:
            for state_dict in state_dicts:
                if state_dict.has(name):
                    return state_dict
            return None

        prefixes = ("model.language_model.", "model.", "")
        model_prefix = next(
            (prefix for prefix in prefixes if find(prefix + "embed_tokens.weight")),
            None,
        )
        if model_prefix is None:
            raise KeyError("Qwen3.5 embedding weight was not found")

        def tensor(name: str) -> torch.Tensor:
            state_dict = find(name)
            if state_dict is None:
                raise KeyError(f"checkpoint tensor not found: {name}")
            return state_dict.get_tensor(name)

        def shard_tensor(
            name: str, dim: int, rank: int = tp_rank, world: int = tp_size
        ) -> torch.Tensor:
            value = tensor(name)
            if world == 1:
                return value
            if value.size(dim) % world:
                raise ValueError(f"cannot shard {name} across {world} ranks")
            return value.chunk(world, dim=dim)[rank].contiguous()

        def copy_in(name: str, value: torch.Tensor) -> None:
            parameter = self.get_parameter(name)
            parameter.data.copy_(value)

        def load_linear(module: nn.Module, name: str, dim: int) -> None:
            weight = shard_tensor(name, dim)
            scale_name = name.replace(".weight", ".weight_scale_inv")
            if find(scale_name) is not None:
                scale = shard_tensor(scale_name, dim)
                module.load_fp8(weight, scale)
                return
            module.weight.data.copy_(weight)

        def shard_kv(name: str, dim: int = 0) -> torch.Tensor:
            if cfg.n_kv_heads >= tp_size:
                return shard_tensor(name, dim)
            replicas = tp_size // cfg.n_kv_heads
            return shard_tensor(name, dim, tp_rank // replicas, cfg.n_kv_heads)

        copy_in(
            "model.embed_tokens.weight",
            shard_tensor(model_prefix + "embed_tokens.weight", 1),
        )
        for layer_id, layer_type in enumerate(cfg.layer_types):
            source = f"{model_prefix}layers.{layer_id}."
            layer = self.model.layers[layer_id]
            for norm in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                copy_in(f"model.layers.{layer_id}.{norm}", tensor(source + norm))

            if layer_type == "full_attention":
                q = shard_tensor(source + "self_attn.q_proj.weight", 0)
                k = shard_kv(source + "self_attn.k_proj.weight")
                v = shard_kv(source + "self_attn.v_proj.weight")
                q_scale = find(source + "self_attn.q_proj.weight_scale_inv")
                local_heads = layer.self_attn.num_heads
                if not cfg.attn_output_gate:
                    packed = torch.cat((q, k, v))
                    if q_scale is not None:
                        qs = shard_tensor(
                            source + "self_attn.q_proj.weight_scale_inv", 0
                        )
                        ks = shard_kv(source + "self_attn.k_proj.weight_scale_inv")
                        vs = shard_kv(source + "self_attn.v_proj.weight_scale_inv")
                        layer.self_attn.qkv_proj.load_fp8(
                            packed, torch.cat((qs, ks, vs))
                        )
                    else:
                        layer.self_attn.qkv_proj.weight.data.copy_(packed)
                elif q_scale is not None:
                    qs = shard_tensor(source + "self_attn.q_proj.weight_scale_inv", 0)
                    ks = shard_kv(source + "self_attn.k_proj.weight_scale_inv")
                    vs = shard_kv(source + "self_attn.v_proj.weight_scale_inv")
                    packed, packed_scale = pack_gated_qkv_projection(
                        q,
                        k,
                        v,
                        qs,
                        ks,
                        vs,
                        local_heads,
                        cfg.head_dim,
                    )
                    layer.self_attn.qkv_proj.load_fp8(packed, packed_scale)
                else:
                    packed, _ = pack_gated_qkv_projection(
                        q,
                        k,
                        v,
                        None,
                        None,
                        None,
                        local_heads,
                        cfg.head_dim,
                    )
                    layer.self_attn.qkv_proj.weight.data.copy_(packed)
                load_linear(
                    layer.self_attn.o_proj, source + "self_attn.o_proj.weight", 1
                )
                copy_in(
                    f"model.layers.{layer_id}.self_attn.q_norm.weight",
                    tensor(source + "self_attn.q_norm.weight"),
                )
                copy_in(
                    f"model.layers.{layer_id}.self_attn.k_norm.weight",
                    tensor(source + "self_attn.k_norm.weight"),
                )
                _store_fused_qk_norm_weight(layer.self_attn.q_norm, self.dtype)
                _store_fused_qk_norm_weight(layer.self_attn.k_norm, self.dtype)
            else:
                linear = source + "linear_attn."
                qkv = tensor(linear + "in_proj_qkv.weight")
                global_key = cfg.linear_num_key_heads * cfg.linear_key_head_dim
                global_value = cfg.linear_num_value_heads * cfg.linear_value_head_dim
                q, k, v = qkv.split((global_key, global_key, global_value), dim=0)
                q = q.chunk(tp_size, dim=0)[tp_rank]
                k = k.chunk(tp_size, dim=0)[tp_rank]
                v = v.chunk(tp_size, dim=0)[tp_rank]
                fused_qkv = torch.cat((q, k, v))
                qkv_scale_name = linear + "in_proj_qkv.weight_scale_inv"
                if find(qkv_scale_name) is not None:
                    qkv_scale = tensor(qkv_scale_name)
                    sq, sk, sv = qkv_scale.split(
                        (
                            (global_key + 127) // 128,
                            (global_key + 127) // 128,
                            (global_value + 127) // 128,
                        ),
                        dim=0,
                    )
                    fused_scale = torch.cat(
                        (
                            sq.chunk(tp_size, dim=0)[tp_rank],
                            sk.chunk(tp_size, dim=0)[tp_rank],
                            sv.chunk(tp_size, dim=0)[tp_rank],
                        )
                    )
                    if layer.linear_attn.in_proj_qkvz is not None:
                        z_weight = tensor(linear + "in_proj_z.weight")
                        z_scale = tensor(linear + "in_proj_z.weight_scale_inv")
                        layer.linear_attn.in_proj_qkvz.load_fp8(
                            torch.cat((fused_qkv, z_weight)),
                            torch.cat((fused_scale, z_scale)),
                        )
                    else:
                        layer.linear_attn.in_proj_qkv.load_fp8(
                            fused_qkv, fused_scale
                        )
                        load_linear(
                            layer.linear_attn.in_proj_z,
                            linear + "in_proj_z.weight",
                            0,
                        )
                elif layer.linear_attn.in_proj_qkvz is not None:
                    z_weight = tensor(linear + "in_proj_z.weight")
                    layer.linear_attn.in_proj_qkvz.weight.data.copy_(
                        torch.cat((fused_qkv, z_weight))
                    )
                else:
                    layer.linear_attn.in_proj_qkv.weight.data.copy_(fused_qkv)
                    load_linear(
                        layer.linear_attn.in_proj_z,
                        linear + "in_proj_z.weight",
                        0,
                    )
                if layer.linear_attn.in_proj_ba is not None:
                    b_weight = shard_tensor(linear + "in_proj_b.weight", 0)
                    a_weight = shard_tensor(linear + "in_proj_a.weight", 0)
                    b_scale_name = linear + "in_proj_b.weight_scale_inv"
                    a_scale_name = linear + "in_proj_a.weight_scale_inv"
                    if (
                        find(b_scale_name) is not None
                        and find(a_scale_name) is not None
                    ):
                        layer.linear_attn.in_proj_ba.load_fp8(
                            torch.cat((b_weight, a_weight), dim=0),
                            torch.cat(
                                (
                                    shard_tensor(b_scale_name, 0),
                                    shard_tensor(a_scale_name, 0),
                                ),
                                dim=0,
                            ),
                        )
                    else:
                        layer.linear_attn.in_proj_ba.weight.data.copy_(
                            torch.cat((b_weight, a_weight), dim=0)
                        )
                else:
                    for projection in ("in_proj_b", "in_proj_a"):
                        load_linear(
                            getattr(layer.linear_attn, projection),
                            linear + f"{projection}.weight",
                            0,
                        )
                conv = tensor(linear + "conv1d.weight").squeeze(1)
                cq, ck, cv = conv.split((global_key, global_key, global_value), dim=0)
                copy_in(
                    f"model.layers.{layer_id}.linear_attn.conv1d_weight",
                    torch.cat(
                        (
                            cq.chunk(tp_size)[tp_rank],
                            ck.chunk(tp_size)[tp_rank],
                            cv.chunk(tp_size)[tp_rank],
                        )
                    ),
                )
                for name in ("A_log", "dt_bias"):
                    copy_in(
                        f"model.layers.{layer_id}.linear_attn.{name}",
                        shard_tensor(linear + name, 0),
                    )
                copy_in(
                    f"model.layers.{layer_id}.linear_attn.norm_weight",
                    tensor(linear + "norm.weight"),
                )
                load_linear(
                    layer.linear_attn.out_proj, linear + "out_proj.weight", 1
                )

            if cfg.is_moe_layer(layer_id):
                moe = source + "mlp."
                copy_in(
                    f"model.layers.{layer_id}.mlp.experts.gate.weight",
                    tensor(moe + "gate.weight"),
                )
                copy_in(
                    f"model.layers.{layer_id}.mlp.shared_expert_gate.weight",
                    tensor(moe + "shared_expert_gate.weight"),
                )
                gate_up = tensor(moe + "experts.gate_up_proj")
                start_expert = cfg.ep_rank * (cfg.num_experts // cfg.ep_size)
                gate_up = gate_up.narrow(
                    0, start_expert, cfg.num_experts // cfg.ep_size
                )
                gate, up = gate_up.chunk(2, dim=1)
                gate = gate.chunk(cfg.moe_tp_size, dim=1)[cfg.moe_tp_rank]
                up = up.chunk(cfg.moe_tp_size, dim=1)[cfg.moe_tp_rank]
                # Checkpoint is [gate, up]; CUTLASS SwiGLU consumes [up, gate].
                copy_in(
                    f"model.layers.{layer_id}.mlp.experts.w13",
                    torch.cat((up, gate), dim=1),
                )
                down = tensor(moe + "experts.down_proj").narrow(
                    0, start_expert, cfg.num_experts // cfg.ep_size
                )
                copy_in(
                    f"model.layers.{layer_id}.mlp.experts.w2",
                    down.chunk(cfg.moe_tp_size, dim=2)[cfg.moe_tp_rank],
                )
                shared = moe + "shared_expert."
                shared_gate = shard_tensor(shared + "gate_proj.weight", 0)
                shared_up = shard_tensor(shared + "up_proj.weight", 0)
                copy_in(
                    f"model.layers.{layer_id}.mlp.shared_expert.gate_up_proj.weight",
                    torch.cat((shared_gate, shared_up)),
                )
                copy_in(
                    f"model.layers.{layer_id}.mlp.shared_expert.down_proj.weight",
                    shard_tensor(shared + "down_proj.weight", 1),
                )
            else:
                gate_name = source + "mlp.gate_proj.weight"
                up_name = source + "mlp.up_proj.weight"
                gate = shard_tensor(gate_name, 0)
                up = shard_tensor(up_name, 0)
                if find(gate_name.replace(".weight", ".weight_scale_inv")) is not None:
                    gate_scale = shard_tensor(
                        gate_name.replace(".weight", ".weight_scale_inv"), 0
                    )
                    up_scale = shard_tensor(
                        up_name.replace(".weight", ".weight_scale_inv"), 0
                    )
                    layer.mlp.gate_up_proj.load_fp8(
                        torch.cat((gate, up)), torch.cat((gate_scale, up_scale))
                    )
                else:
                    layer.mlp.gate_up_proj.weight.data.copy_(torch.cat((gate, up)))
                load_linear(layer.mlp.down_proj, source + "mlp.down_proj.weight", 1)

        copy_in("model.norm.weight", tensor(model_prefix + "norm.weight"))
        lm_head_name = "lm_head.weight"
        if cfg.tie_word_embeddings or find(lm_head_name) is None:
            lm_head_name = model_prefix + "embed_tokens.weight"
        copy_in("lm_head.weight", shard_tensor(lm_head_name, 0))
        self.lm_head.prefer_nn_gemm_layout()
