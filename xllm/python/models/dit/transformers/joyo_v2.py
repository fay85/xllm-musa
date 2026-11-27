# Copyright 2026 The xLLM Authors.
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

"""JoyOV2 transformer (T2VA triple-stream Expert-Chosen MoE DiT).

Port of ``xllm/models/dit/transformers/transformer_joyo_v2.h``. The 200B
checkpoint cannot fit on a single device, so the port keeps the C++
tensor-parallel shard semantics: column/row-parallel projections are plain
``nn.Linear`` modules holding this rank's slice, with the matching
all-gather / all-reduce driven through the ``dit_tp`` named group. Ulysses
sequence parallelism inside the mix layers uses the ``dit_sp`` group and the
:class:`SequenceParallelMixin` scatter/gather wrapper exactly like the C++
``sequence_parallel_forward`` call.

The three refiner streams (context_text / noise / audio) run full-sequence
(no Ulysses); only the packed mix layers reshard. Rolling load
(``DitRollingLoadManager``) has no Python counterpart yet — the whole
transformer is resident after the executor's ``to(device)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.distributed import collectives
from xllm.python.models.dit.dit_component_base import DiTComponentBase, prefixed, require_loaded
from xllm.python.models.dit.transformers.common import RMSNorm
from xllm.python.parallel import SequenceParallelMixin

# Named process groups bridged by the C++ shell (see py_dit_executor.cpp).
TP_GROUP = "dit_tp"
SP_GROUP = "dit_sp"


class RopeMeta(NamedTuple):
    """Per-stream rope metadata (seqlens [n], position_id [3, S])."""

    seqlens: torch.Tensor
    position_id: torch.Tensor


class PackedRopeMeta(NamedTuple):
    """Rope metadata for the packed T2VA forward (C++ JoyOV2PackedRopeMeta)."""

    text: RopeMeta
    pixel: RopeMeta | None
    audio: RopeMeta | None
    mix: RopeMeta


def tp_group_size(device: torch.device | str | None) -> int:
    if device is None:
        return 1
    return collectives.named_group_size(TP_GROUP, device)


def tp_group_rank(device: torch.device | str | None) -> int:
    if device is None:
        return 0
    return collectives.named_group_rank(TP_GROUP, device)


def _shard_column(weight: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """[out, in] full weight -> this rank's contiguous output slice."""
    return weight.chunk(world, dim=0)[rank]


def _shard_row(weight: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """[out, in] full weight -> this rank's input-dim slice."""
    return weight.chunk(world, dim=1)[rank]


def _shard_sections(
    weight: torch.Tensor,
    sections: list[int],
    rank: int,
    world: int,
) -> torch.Tensor:
    """Shard each packed output section independently, then concatenate.

    Mirrors ``ColumnParallelLinear::load_state_dict(shard_tensor_count=N)``:
    a packed [gate|up] (or [Q|KV|gate]) weight is N logical tensors, each
    column-sharded on its own, NOT one flat split of the packed dim.
    """
    parts: list[torch.Tensor] = []
    start = 0
    for size in sections:
        parts.append(weight[start : start + size].chunk(world, dim=0)[rank])
        start += size
    return torch.cat(parts, dim=0)


def _load_norm_weight(norm: RMSNorm, state_dicts: list, name: str = "weight") -> None:
    """RMSNorm has no load_weights; fetch its single weight directly."""
    for state_dict in state_dicts:
        if state_dict.has(name):
            norm.weight.data.copy_(state_dict.get_tensor(name))
            return
    raise RuntimeError(f"weights are not loaded for {type(norm).__name__}: ['{name}']")


class ColumnParallelLinear(nn.Module):
    """Column-sharded Linear with optional output all-gather.

    Python counterpart of ``layer::ColumnParallelLinear`` for the shapes the
    JoyOV2 model uses (no bias except on the embedders / out projections).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        if out_features % tp_size:
            raise ValueError(f"ColumnParallelLinear out_features {out_features} not divisible by tp_size {tp_size}")
        self.tp_size = tp_size
        self.gather_output = gather_output and tp_size > 1
        self.linear = nn.Linear(in_features, out_features // tp_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        if self.gather_output:
            y = collectives.all_gather(y.contiguous(), -1, self.tp_size, TP_GROUP)
        return y

    def load_full(self, state_dicts: list, prefix: str, rank: int, world: int) -> None:
        names = ["weight"] + (["bias"] if self.linear.bias is not None else [])
        loaded = {name: False for name in names}
        for state_dict in state_dicts:
            for name in names:
                full = f"{prefix}.{name}"
                if not loaded[name] and state_dict.has(full):
                    tensor = state_dict.get_tensor(full)
                    if name == "weight":
                        self.linear.weight.data.copy_(_shard_column(tensor, rank, world))
                    else:
                        self.linear.bias.data.copy_(tensor.chunk(world, dim=0)[rank])
                    loaded[name] = True
        require_loaded(loaded, prefix)


class MergedColumnParallelLinear(nn.Module):
    """Packed column-parallel Linear whose output is several logical sections.

    The local weight is the per-rank concatenation of each section's slice
    (e.g. [Q|KV|gate] with sizes [q_dim, kv_dim, gate_dim]). No gather: the
    caller consumes the local shard directly (attention / shared experts).
    """

    def __init__(
        self,
        in_features: int,
        sections: list[int],
        bias: bool = False,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        for section in sections:
            if section % tp_size:
                raise ValueError(f"MergedColumnParallelLinear section {section} not divisible by tp_size {tp_size}")
        self.sections = list(sections)
        self.tp_size = tp_size
        self.linear = nn.Linear(in_features, sum(section // tp_size for section in sections), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def load_full(self, state_dicts: list, prefix: str, rank: int, world: int) -> None:
        names = ["weight"] + (["bias"] if self.linear.bias is not None else [])
        loaded = {name: False for name in names}
        for state_dict in state_dicts:
            for name in names:
                full = f"{prefix}.{name}"
                if not loaded[name] and state_dict.has(full):
                    tensor = state_dict.get_tensor(full)
                    if name == "weight":
                        self.linear.weight.data.copy_(_shard_sections(tensor, self.sections, rank, world))
                    else:
                        self.linear.bias.data.copy_(
                            _shard_sections(tensor.unsqueeze(0), self.sections, rank, world).squeeze(0)
                        )
                    loaded[name] = True
        require_loaded(loaded, prefix)


class RowParallelLinear(nn.Module):
    """Row-sharded Linear over a parallelized input with result reduction."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        if in_features % tp_size:
            raise ValueError(f"RowParallelLinear in_features {in_features} not divisible by tp_size {tp_size}")
        self.tp_size = tp_size
        self.linear = nn.Linear(in_features // tp_size, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        if self.tp_size > 1:
            collectives.all_reduce_(y, TP_GROUP)
        return y

    def load_full(self, state_dicts: list, prefix: str, rank: int, world: int) -> None:
        names = ["weight"] + (["bias"] if self.linear.bias is not None else [])
        loaded = {name: False for name in names}
        for state_dict in state_dicts:
            for name in names:
                full = f"{prefix}.{name}"
                if not loaded[name] and state_dict.has(full):
                    tensor = state_dict.get_tensor(full)
                    if name == "weight":
                        self.linear.weight.data.copy_(_shard_row(tensor, rank, world))
                    else:
                        # Row-parallel bias is replicated (added after the
                        # reduction on every rank).
                        self.linear.bias.data.copy_(tensor)
                    loaded[name] = True
        require_loaded(loaded, prefix)


def apply_rotary_neox(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """NeoX-style RoPE. x: [B,S,H,D], cos/sin: [S, D/2]."""
    orig_dtype = x.dtype
    c = cos.to(orig_dtype).unsqueeze(0).unsqueeze(2)
    s = sin.to(orig_dtype).unsqueeze(0).unsqueeze(2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Per-sample SDPA over packed [sample0|sample1|...] (SGLang _sdpa_varlen).

    Full-sequence SDPA would let CFG pos/neg attend across samples and
    collapse generation into noise. q/k/v: [B=1, H, S, D].
    """
    if q.size(0) != 1:
        raise ValueError(f"sdpa_varlen expects batch 1, got {q.size(0)}")
    cu = cu_seqlens.to("cpu", torch.int64)
    outs = []
    for i in range(cu.numel() - 1):
        start = int(cu[i])
        end = int(cu[i + 1])
        if end <= start:
            raise ValueError(f"sdpa_varlen empty sample {i}: [{start}, {end})")
        outs.append(F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]))
    return torch.cat(outs, dim=2)


def expand_per_sample(
    per_sample: torch.Tensor,  # [N, D]
    seqlens: torch.Tensor,  # [N]
    total_s: int,
) -> torch.Tensor:
    repeats = seqlens.to(torch.int64).to(per_sample.device)
    return torch.repeat_interleave(per_sample, repeats, dim=0, output_size=total_s).unsqueeze(0)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.to(torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class JoyOV2Modulate(nn.Module):
    """``(modulate_table + x)`` chunked factor-ways; table [1, factor*H] fp32."""

    def __init__(self, hidden_size: int, factor: int = 6) -> None:
        super().__init__()
        self.factor = factor
        self.hidden_size = hidden_size
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor * hidden_size, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, timestep_emb: torch.Tensor) -> list[torch.Tensor]:
        # timestep_emb: [N, factor*H] -> factor tensors of [N, H]
        fused = (self.modulate_table.to(timestep_emb.dtype) + timestep_emb).view(-1, self.factor, self.hidden_size)
        return list(fused.unbind(dim=1))

    def load_weights(self, state_dicts: list) -> None:
        loaded = False
        for state_dict in state_dicts:
            if state_dict.has("modulate_table"):
                self.modulate_table.data.copy_(state_dict.get_tensor("modulate_table"))
                loaded = True
        if not loaded:
            raise RuntimeError("weight is not loaded for modulate_table")


class JoyOV2DenseFFN(nn.Module):
    """SwiGLU FFN with TP-sharded w1 (packed [gate|up]) and w2."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        if ffn_hidden_size % tp_size:
            raise ValueError(f"ffn_hidden_size {ffn_hidden_size} not divisible by tp_size {tp_size}")
        local = ffn_hidden_size // tp_size
        self.w1 = nn.Linear(hidden_size, 2 * local, bias=False)
        self.w2 = nn.Linear(local, hidden_size, bias=False)
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        y = self.w2(F.silu(gate) * up)
        if self.tp_size > 1:
            collectives.all_reduce_(y, TP_GROUP)
        return y

    def load_weights(self, state_dicts: list, rank: int, world: int) -> None:
        loaded = {"w1.weight": False, "w2.weight": False}
        for state_dict in state_dicts:
            if not loaded["w1.weight"] and state_dict.has("w1.weight"):
                # Checkpoint packs [gate|up]; TP shards each half independently.
                ffn = self.w2.in_features * world
                self.w1.weight.data.copy_(_shard_sections(state_dict.get_tensor("w1.weight"), [ffn, ffn], rank, world))
                loaded["w1.weight"] = True
            if not loaded["w2.weight"] and state_dict.has("w2.weight"):
                self.w2.weight.data.copy_(_shard_row(state_dict.get_tensor("w2.weight"), rank, world))
                loaded["w2.weight"] = True
        require_loaded(loaded, type(self).__name__)


class JoyOV2MoEFeedForward(nn.Module):
    """Expert-Chosen MoE + sigmoid-gated shared expert.

    Routing is sample-level (ceil(seqlen_i * top_p) tokens per expert per
    sample) when per-sample ``seqlens`` are supplied, else a global
    ceil(S * top_p) topk. EP sharding reuses the TP group: with tp_size > 1
    each rank holds num_experts/tp experts, computes their contribution on
    the full token set and all-reduces (compute-over-replication, matching
    the C++ pre-dispatcher path).
    """

    def __init__(
        self,
        hidden_size: int,
        moe_hidden_size: int,
        num_experts: int,
        top_p: float,
        share_expert_dim: int,
        routed_scaling_factor: float,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        if num_experts % tp_size:
            raise ValueError(f"num_experts {num_experts} not divisible by tp_size {tp_size}")
        if share_expert_dim % tp_size:
            raise ValueError(f"share_expert_dim {share_expert_dim} not divisible by tp_size {tp_size}")
        self.hidden_size = hidden_size
        self.moe_hidden_size = moe_hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // tp_size
        self.ep_size = tp_size
        self.ep_rank = 0  # resolved from the group at load/forward time
        self.top_p = top_p
        self.share_expert_dim = share_expert_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.tp_size = tp_size

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        # Diffusers layout: experts.w1 [E, 2*H_moe, H] / experts.w2 [E, H, H_moe]
        self.experts_w1 = nn.Parameter(
            torch.empty(self.num_local_experts, moe_hidden_size * 2, hidden_size),
            requires_grad=False,
        )
        self.experts_w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, moe_hidden_size),
            requires_grad=False,
        )
        self.shared_w1 = MergedColumnParallelLinear(hidden_size, [share_expert_dim, share_expert_dim], tp_size=tp_size)
        self.shared_w2 = RowParallelLinear(share_expert_dim, hidden_size, tp_size=tp_size)
        self.shared_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, D]
        seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        b, s, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        num_tokens = x.size(0)

        gate_logits = self.gate(x)
        gate_prob = torch.sigmoid(gate_logits.to(torch.float32))
        expert_token_ids, expert_token_weights = expert_chosen_routing(gate_prob, num_tokens, seqlens, self.top_p)

        ep_rank = self.ep_rank
        if self.ep_size == 1:
            expert_input = x[expert_token_ids]  # [E, C, D]
            expert_out = self._local_expert_compute(expert_input, expert_token_weights)
            routed = torch.zeros_like(x)
            routed.index_add_(
                0,
                expert_token_ids.reshape(-1),
                expert_out.reshape(-1, d).to(x.dtype),
            )
        else:
            start = ep_rank * self.num_local_experts
            end = start + self.num_local_experts
            local_ids = expert_token_ids[start:end]
            local_weights = expert_token_weights[start:end]
            expert_input = x[local_ids]
            expert_out = self._local_expert_compute(expert_input, local_weights)
            routed = torch.zeros_like(x)
            routed.index_add_(0, local_ids.reshape(-1), expert_out.reshape(-1, d).to(x.dtype))
            collectives.all_reduce_(routed, TP_GROUP)

        routed = routed * self.routed_scaling_factor
        shared = self._shared_expert_forward(x)
        return (routed + shared).view(b, s, d)

    def _shared_expert_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.shared_w1(x).chunk(2, dim=-1)
        out = self.shared_w2(F.silu(gate) * up)
        g = torch.sigmoid(self.shared_gate(out))
        return g * out

    def _local_expert_compute(
        self,
        expert_input: torch.Tensor,  # [E_loc, C, D]
        expert_weights: torch.Tensor,  # [E_loc, C]
    ) -> torch.Tensor:
        in_dtype = expert_input.dtype
        h = torch.bmm(expert_input.to(self.experts_w1.dtype), self.experts_w1.transpose(1, 2))
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up
        out = torch.bmm(h, self.experts_w2.transpose(1, 2))
        out = out * expert_weights.unsqueeze(-1).to(out.dtype)
        return out.to(in_dtype)

    def load_weights(
        self,
        state_dicts: list,
        rank: int,
        world: int,
    ) -> None:
        self.ep_rank = rank if world == self.ep_size else 0
        loaded = {
            "gate.weight": False,
            "experts.w1": False,
            "experts.w2": False,
            "shared_expert.gate.weight": False,
        }
        for state_dict in state_dicts:
            if not loaded["gate.weight"] and state_dict.has("gate.weight"):
                self.gate.weight.data.copy_(state_dict.get_tensor("gate.weight"))
                loaded["gate.weight"] = True
            if not loaded["experts.w1"] and state_dict.has("experts.w1"):
                w1 = state_dict.get_tensor("experts.w1")
                if w1.size(0) != self.num_experts:
                    raise RuntimeError(f"experts.w1 expert count {w1.size(0)} != {self.num_experts}")
                start = self.ep_rank * self.num_local_experts
                self.experts_w1.data.copy_(w1[start : start + self.num_local_experts])
                loaded["experts.w1"] = True
            if not loaded["experts.w2"] and state_dict.has("experts.w2"):
                w2 = state_dict.get_tensor("experts.w2")
                if w2.size(0) != self.num_experts:
                    raise RuntimeError(f"experts.w2 expert count {w2.size(0)} != {self.num_experts}")
                start = self.ep_rank * self.num_local_experts
                self.experts_w2.data.copy_(w2[start : start + self.num_local_experts])
                loaded["experts.w2"] = True
            if not loaded["shared_expert.gate.weight"] and state_dict.has("shared_expert.gate.weight"):
                self.shared_gate.weight.data.copy_(state_dict.get_tensor("shared_expert.gate.weight"))
                loaded["shared_expert.gate.weight"] = True
        require_loaded(loaded, type(self).__name__)
        self.shared_w1.load_full(prefixed(state_dicts, "shared_expert."), "w1", rank, world)
        self.shared_w2.load_full(prefixed(state_dicts, "shared_expert."), "w2", rank, world)


def expert_chosen_routing(
    gate_prob: torch.Tensor,  # [S, E] fp32
    num_tokens: int,
    seqlens: torch.Tensor | None,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert-Chosen routing -> (expert_token_ids [E, C], weights [E, C])."""
    gate_t = gate_prob.transpose(0, 1)  # [E, S]

    if seqlens is not None and seqlens.numel() > 1:
        lens = seqlens.to("cpu", torch.int64).tolist()
        kept_cols: list[int] = []
        start = 0
        for length in lens:
            k = min(math.ceil(length * top_p), length)
            kept_cols.extend(range(start, start + k))
            start += length
        device = gate_t.device
        sample_ids = torch.repeat_interleave(
            torch.arange(len(lens), dtype=torch.int64, device=device),
            seqlens.to(torch.int64).to(device),
            dim=0,
            output_size=num_tokens,
        )
        # Offsets keep each sample's block contiguous in the sort order.
        sort_key = gate_t - sample_ids.unsqueeze(0).to(gate_t.dtype) * 2.0
        _, expert_token_ids = torch.sort(sort_key, dim=1, descending=True, stable=True)
        kept = torch.tensor(kept_cols, dtype=torch.int64, device=device)
        expert_token_ids = expert_token_ids.index_select(1, kept)
        expert_token_weights = gate_t.gather(1, expert_token_ids)
    else:
        tokens_per_expert = math.ceil(num_tokens * top_p)
        expert_token_weights, expert_token_ids = torch.topk(gate_t, tokens_per_expert, dim=1, largest=True, sorted=True)

    # Normalize weights per token over its selected experts...
    token_weight_sums = torch.zeros(num_tokens, dtype=expert_token_weights.dtype, device=expert_token_weights.device)
    token_weight_sums.scatter_add_(0, expert_token_ids.reshape(-1), expert_token_weights.reshape(-1))
    expert_token_weights = expert_token_weights / token_weight_sums[expert_token_ids].clamp_min(1e-9)
    # ...then scale by selected-expert count^0.5 (Joytron default).
    token_counts = torch.zeros_like(token_weight_sums)
    token_counts.scatter_add_(
        0,
        expert_token_ids.reshape(-1),
        torch.ones(
            expert_token_ids.numel(),
            dtype=expert_token_weights.dtype,
            device=expert_token_weights.device,
        ),
    )
    expert_token_weights = expert_token_weights * token_counts[expert_token_ids].pow(0.5)
    return expert_token_ids, expert_token_weights


class JoyOV2Attention(nn.Module):
    """Packed-QKV+gate attention with per-head sigmoid output gating.

    Checkpoint layout: ``to_qkv`` packs [Q | KV per-head interleaved K_iV_i |
    per-Q-head gate scalar]; q/k are RMSNorm'd, GQA-repeated, Ulysses
    resharded, RoPE'd (neox), attended per-sample, reverse resharded and
    gated. ``to_out`` is row-parallel over the head shard.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        sp_group: str | None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        if num_heads % tp_size or num_kv_heads % tp_size:
            raise ValueError(f"heads ({num_heads}, {num_kv_heads}) not divisible by tp_size {tp_size}")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.local_heads = num_heads // tp_size
        self.local_kv_heads = num_kv_heads // tp_size
        self.sp_group = sp_group
        self.tp_size = tp_size

        qkv_out = (num_heads + 2 * num_kv_heads) * head_dim + num_heads
        self.to_qkv = MergedColumnParallelLinear(
            hidden_size,
            [
                num_heads * head_dim,
                num_kv_heads * head_dim * 2,
                num_heads,
            ],
            tp_size=tp_size,
        )
        self.to_out = RowParallelLinear(num_heads * head_dim, hidden_size, tp_size=tp_size)
        self.norm_q = RMSNorm(head_dim, norm_eps)
        self.norm_k = RMSNorm(head_dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,  # [B, S, H]
        freqs_cis: torch.Tensor | None,  # [S, head_dim] packed [cos|sin]
        cu_seqlens: torch.Tensor | None,
        ulysses_sp: bool = False,
    ) -> torch.Tensor:
        b, s = x.size(0), x.size(1)
        qkvg = self.to_qkv(x)
        q_size = self.local_heads * self.head_dim
        kv_size = self.local_kv_heads * self.head_dim * 2
        gate_size = self.local_heads
        q, kv, gate = qkvg.split([q_size, kv_size, gate_size], dim=-1)

        q = q.view(b, s, self.local_heads, self.head_dim)
        kv_view = kv.view(b, s, self.local_kv_heads, self.head_dim * 2)
        k, v = kv_view.chunk(2, dim=-1)
        v = v.contiguous()

        q = self.norm_q(q.reshape(-1, self.head_dim)).view(b, s, self.local_heads, self.head_dim)
        k = self.norm_k(k.reshape(-1, self.head_dim)).view(b, s, self.local_kv_heads, self.head_dim)

        if self.local_kv_heads != self.local_heads:
            rep = self.local_heads // self.local_kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)

        sp_size = collectives.named_group_size(self.sp_group, x.device) if self.sp_group is not None else 1
        use_sp = ulysses_sp and sp_size > 1
        if use_sp:
            # [b, shard_seq, h, d] -> [b, seq, h/sp, d] (before RoPE)
            q = collectives.all_to_all_4D(q.contiguous(), 2, 1, self.sp_group)
            k = collectives.all_to_all_4D(k.contiguous(), 2, 1, self.sp_group)
            v = collectives.all_to_all_4D(v.contiguous(), 2, 1, self.sp_group)

        if freqs_cis is not None:
            cos, sin = freqs_cis.chunk(2, dim=-1)
            if cos.size(0) != q.size(1):
                raise RuntimeError(
                    f"JoyOV2Attention RoPE length mismatch: freqs={cos.size(0)} q_s={q.size(1)} ulysses={use_sp}"
                )
            q = apply_rotary_neox(q, cos, sin)
            k = apply_rotary_neox(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # After the Ulysses all2all QKV span the full packed sequence; isolate
        # CFG samples with cu_seqlens built from the unsplit mix seqlens.
        allow_varlen = cu_seqlens is not None and cu_seqlens.numel() > 2
        if allow_varlen:
            out = sdpa_varlen(q, k, v, cu_seqlens)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous()

        if use_sp:
            out = collectives.all_to_all_4D(out, 1, 2, self.sp_group)

        out = out.reshape(b, out.size(1), self.local_heads * self.head_dim)
        gate_s = torch.sigmoid(gate).view(b, s, self.local_heads, 1)
        out = (
            (out.view(b, s, self.local_heads, self.head_dim) * gate_s)
            .contiguous()
            .view(b, s, self.local_heads * self.head_dim)
        )
        return self.to_out(out)

    def load_weights(self, state_dicts: list, rank: int, world: int) -> None:
        self.to_qkv.load_full(state_dicts, "to_qkv", rank, world)
        self.to_out.load_full(state_dicts, "to_out", rank, world)
        loaded = {"norm_q.weight": False, "norm_k.weight": False}
        params = {
            "norm_q.weight": self.norm_q.weight,
            "norm_k.weight": self.norm_k.weight,
        }
        for state_dict in state_dicts:
            for name in loaded:
                if not loaded[name] and state_dict.has(name):
                    params[name].data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        require_loaded(loaded, type(self).__name__)


class JoyOV2TransformerBlock(nn.Module):
    """Single JoyOV2 block: modulated attention + sandwich norms + FFN/MoE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        sandwich_norm: bool,
        modulation: bool,
        use_moe: bool,
        moe_hidden_size: int,
        num_experts: int,
        top_p: float,
        share_expert_dim: int,
        routed_scaling_factor: float,
        sp_group: str | None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.sandwich_norm = sandwich_norm
        self.modulation = modulation
        self.use_moe = use_moe

        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.norm2 = RMSNorm(hidden_size, norm_eps)
        if sandwich_norm:
            self.norm1_post = RMSNorm(hidden_size, norm_eps)
            self.norm2_post = RMSNorm(hidden_size, norm_eps)
        if modulation:
            self.mod = JoyOV2Modulate(hidden_size, 6)
        self.attn = JoyOV2Attention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            norm_eps,
            sp_group,
            tp_size,
        )
        if use_moe:
            self.ff = JoyOV2MoEFeedForward(
                hidden_size,
                moe_hidden_size,
                num_experts,
                top_p,
                share_expert_dim,
                routed_scaling_factor,
                tp_size,
            )
        else:
            self.ff = JoyOV2DenseFFN(hidden_size, ffn_hidden_size, tp_size)

    def forward(
        self,
        x: torch.Tensor,  # [B, S, H]
        timestep_emb: torch.Tensor | None,  # [N, 6H]
        freqs: torch.Tensor | None,
        seqlens: torch.Tensor | None,
        ulysses_sp: bool = False,
        attn_seqlens: torch.Tensor | None = None,
        token_sample_ids: torch.Tensor | None = None,
        audio_timestep_emb: torch.Tensor | None = None,
        modulation_sample_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale_msa = gate_msa = shift_msa = None
        scale_mlp = gate_mlp = shift_mlp = None
        modulated = False
        if self.modulation and timestep_emb is not None:
            mods = self.mod(timestep_emb)
            if audio_timestep_emb is not None:
                if modulation_sample_ids is None:
                    raise ValueError("audio timestep modulation requires per-token modulation sample ids")
                audio_mods = self.mod(audio_timestep_emb)
                mods = [torch.cat([video, audio], dim=0) for video, audio in zip(mods, audio_mods)]
            scale_msa, gate_msa, shift_msa, scale_mlp, gate_mlp, shift_mlp = mods
            modulated = True

        s = x.size(1)

        def expand(t: torch.Tensor) -> torch.Tensor:
            if modulation_sample_ids is not None:
                if modulation_sample_ids.numel() != s:
                    raise ValueError(f"modulation_sample_ids {modulation_sample_ids.numel()} != seq {s}")
                return t.index_select(0, modulation_sample_ids).unsqueeze(0)
            if token_sample_ids is not None:
                if token_sample_ids.numel() != s:
                    raise ValueError(f"token_sample_ids {token_sample_ids.numel()} != seq {s}")
                return t.index_select(0, token_sample_ids).unsqueeze(0)
            if seqlens is None:
                raise ValueError("block modulation needs seqlens or token_sample_ids")
            return expand_per_sample(t, seqlens, s)

        cu_seqlens = None
        if attn_seqlens is not None and attn_seqlens.numel() > 1:
            lens = attn_seqlens
        elif seqlens is not None and seqlens.numel() > 1:
            lens = seqlens
        else:
            lens = None
        if lens is not None:
            lens_dev = lens.to(torch.int32).to(x.device)
            cu = torch.zeros(lens_dev.numel() + 1, dtype=torch.int32, device=x.device)
            cu[1:] = torch.cumsum(lens_dev, dim=0)
            cu_seqlens = cu

        attn_in = self.norm1(x)
        if modulated:
            attn_in = attn_in * (1 + expand(scale_msa)) + expand(shift_msa)
        attn_out = self.attn(attn_in, freqs, cu_seqlens, ulysses_sp)
        if self.sandwich_norm:
            attn_out = self.norm1_post(attn_out)
        if modulated:
            attn_out = attn_out * expand(gate_msa)
        h = x + attn_out

        ffn_in = self.norm2(h)
        if modulated:
            ffn_in = ffn_in * (1 + expand(scale_mlp)) + expand(shift_mlp)
        if self.use_moe:
            # Sample-level routing only applies to unsplit sequences; inside
            # Ulysses the shard has no per-sample boundaries.
            ffn_out = self.ff(ffn_in, None if ulysses_sp else seqlens)
        else:
            ffn_out = self.ff(ffn_in)
        if self.sandwich_norm:
            ffn_out = self.norm2_post(ffn_out)
        if modulated:
            ffn_out = ffn_out * expand(gate_mlp)
        return h + ffn_out

    def load_weights(self, state_dicts: list, rank: int, world: int) -> None:
        loaded = {"norm1.weight": False, "norm2.weight": False}
        if self.sandwich_norm:
            loaded["norm1_post.weight"] = False
            loaded["norm2_post.weight"] = False
        params = {
            "norm1.weight": self.norm1.weight,
            "norm2.weight": self.norm2.weight,
        }
        if self.sandwich_norm:
            params["norm1_post.weight"] = self.norm1_post.weight
            params["norm2_post.weight"] = self.norm2_post.weight
        for state_dict in state_dicts:
            for name in loaded:
                if not loaded[name] and state_dict.has(name):
                    params[name].data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        require_loaded(loaded, type(self).__name__)
        if self.modulation:
            self.mod.load_weights(prefixed(state_dicts, "mod."))
        self.attn.load_weights(prefixed(state_dicts, "attn."), rank, world)
        self.ff.load_weights(prefixed(state_dicts, "ff."), rank, world)


class JoyOV2Transformer3DModel(DiTComponentBase, SequenceParallelMixin):
    """T2VA triple-stream DiT: 3 refiner streams + packed mix MoE layers."""

    def __init__(
        self,
        config: dict,
        sp_group: str | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        DiTComponentBase.__init__(self)
        SequenceParallelMixin.__init__(self, sp_group=sp_group)
        self._dtype = dtype if dtype is not None else torch.bfloat16
        self._device = device

        self.hidden_size = int(config.get("hidden_size", 6144))
        self.num_heads = int(config.get("num_attention_heads", 48))
        self.num_kv_heads = int(config.get("num_kv_heads", 16))
        self.head_dim = int(config.get("head_dim", 128))
        self.num_layers = int(config.get("num_layers", 51))
        self.num_encoder_layers = int(config.get("num_encoder_layers", 2))
        self.num_experts = int(config.get("num_experts", 48))
        self.moe_hidden_size = int(config.get("moe_hidden_size", 4096))
        self.share_expert_dim = int(config.get("share_expert_dim", 8192))
        self.ffn_hidden_size = int(config.get("ffn_hidden_size", 16384))
        self.latent_channels = int(config.get("latent_channels", 128))
        self.text_dim = int(config.get("text_dim", 5120))
        self.rope_theta = float(config.get("rope_theta", 10000.0))
        # The checkpoint stores null for unset optionals (diffusers
        # convention); ModelArgs on the C++ side heals null the same way.
        self.mrope_section = [int(v) for v in (config.get("mrope_section") or [16, 24, 24])]
        self.norm_eps = float(config.get("norm_eps", 1e-6))
        self.sandwich_norm = bool(config.get("sandwich_norm", True))
        self.top_p = float(config.get("top_p", 0.125))
        self.routed_scaling_factor = float(config.get("routed_scaling_factor", 10.0))

        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                f"num_attention_heads*head_dim ({self.num_heads * self.head_dim}) != hidden_size ({self.hidden_size})"
            )

        tp_size = tp_group_size(device)
        self.tp_size = tp_size
        self.tp_rank = tp_group_rank(device)

        def column(in_f: int, out_f: int, bias: bool = True) -> ColumnParallelLinear:
            return ColumnParallelLinear(in_f, out_f, bias=bias, gather_output=True, tp_size=tp_size)

        self.x_embedder = column(self.latent_channels, self.hidden_size)
        self.audio_embedder = column(self.latent_channels, self.hidden_size)
        self.context_embedder_norm = RMSNorm(self.text_dim, self.norm_eps)
        self.context_embedder_proj = column(self.text_dim, self.hidden_size)
        # Checkpoint: time_embedder.linear_1 / linear_2 (no bias)
        self.time_embedder_linear_1 = column(256, self.hidden_size, bias=False)
        self.time_embedder_linear_2 = column(self.hidden_size, self.hidden_size * 6, bias=False)
        self.proj_out_norm = RMSNorm(self.hidden_size, self.norm_eps)
        self.proj_out_linear = column(self.hidden_size, self.latent_channels)
        self.audio_proj_out_norm = RMSNorm(self.hidden_size, self.norm_eps)
        self.audio_proj_out_linear = column(self.hidden_size, self.latent_channels)

        def make_block(modulation: bool, use_moe: bool) -> JoyOV2TransformerBlock:
            return JoyOV2TransformerBlock(
                self.hidden_size,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.ffn_hidden_size,
                self.norm_eps,
                self.sandwich_norm,
                modulation,
                use_moe,
                self.moe_hidden_size,
                self.num_experts,
                self.top_p,
                self.share_expert_dim,
                self.routed_scaling_factor,
                sp_group,
                tp_size,
            )

        self.noise_refiner = nn.ModuleList(
            make_block(modulation=True, use_moe=False) for _ in range(self.num_encoder_layers)
        )
        self.context_text_refiner = nn.ModuleList(
            make_block(modulation=False, use_moe=False) for _ in range(self.num_encoder_layers)
        )
        self.audio_refiner = nn.ModuleList(
            make_block(modulation=True, use_moe=False) for _ in range(self.num_encoder_layers)
        )
        # Layer 0 is dense; layers 1..N-1 are Expert-Chosen MoE.
        self.transformer_blocks = nn.ModuleList(
            make_block(modulation=True, use_moe=layer_index > 0) for layer_index in range(self.num_layers)
        )

        self.to(self._dtype)

    # ---- rope ---------------------------------------------------------------

    def compute_freqs_cis(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Interleaved 3D MRoPE -> [S, head_dim] packed [cos|sin]."""
        if position_ids.size(0) != 3:
            raise ValueError(f"MRoPE position_ids must be [3, S], got {position_ids.shape}")
        half = self.head_dim // 2
        device = position_ids.device
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
        cos = freqs.cos()
        sin = freqs.sin()

        sec_h = self.mrope_section[1] if len(self.mrope_section) > 1 else 0
        sec_w = self.mrope_section[2] if len(self.mrope_section) > 2 else 0

        def pack(t: torch.Tensor) -> torch.Tensor:
            result = t[0].clone()
            for i in range(sec_h):
                idx = 1 + i * 3
                if idx < half:
                    result[..., idx] = t[1][..., idx]
            for i in range(sec_w):
                idx = 2 + i * 3
                if idx < half:
                    result[..., idx] = t[2][..., idx]
            return result

        return torch.cat([pack(cos), pack(sin)], dim=-1)

    # ---- forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,  # [S, latent_channels] packed pixels
        text_embeddings: torch.Tensor,  # [T, text_dim] packed text
        timestep: torch.Tensor,  # [N]
        rope_meta: PackedRopeMeta,
        audio_hidden_states: torch.Tensor | None = None,  # [A, latent_channels]
        audio_timestep: torch.Tensor | None = None,  # [N], defaults to video timestep
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = hidden_states.device
        dtype = hidden_states.dtype

        t_emb = timestep_embedding(timestep * 1000.0, 256).to(dtype)
        timestep_emb = self.time_embedder_linear_2(F.silu(self.time_embedder_linear_1(t_emb)))
        audio_timestep_emb = timestep_emb
        if audio_hidden_states is not None and audio_timestep is not None:
            if audio_timestep.shape != timestep.shape:
                raise ValueError("audio and video timestep shapes must match")
            audio_t_emb = timestep_embedding(audio_timestep * 1000.0, 256).to(dtype)
            audio_timestep_emb = self.time_embedder_linear_2(F.silu(self.time_embedder_linear_1(audio_t_emb)))

        pixel_emb = self.x_embedder(hidden_states)
        text_emb = self.context_embedder_proj(self.context_embedder_norm(text_embeddings))
        audio_emb = None
        if audio_hidden_states is not None:
            audio_emb = self.audio_embedder(audio_hidden_states)

        # Refine text (no modulation). Refiners stay full-sequence (no Ulysses).
        text_freqs = self.compute_freqs_cis(rope_meta.text.position_id)
        x = text_emb.unsqueeze(0)
        for block in self.context_text_refiner:
            x = block(x, None, text_freqs, rope_meta.text.seqlens)
        text_emb = x.squeeze(0)

        # Refine pixel
        pixel_freqs = self.compute_freqs_cis(rope_meta.pixel.position_id)
        x = pixel_emb.unsqueeze(0)
        for block in self.noise_refiner:
            x = block(x, timestep_emb, pixel_freqs, rope_meta.pixel.seqlens)
        pixel_emb = x.squeeze(0)

        if audio_emb is not None and rope_meta.audio is not None:
            audio_freqs = self.compute_freqs_cis(rope_meta.audio.position_id)
            x = audio_emb.unsqueeze(0)
            for block in self.audio_refiner:
                x = block(x, audio_timestep_emb, audio_freqs, rope_meta.audio.seqlens)
            audio_emb = x.squeeze(0)

        # Pack [text | pixel | audio] per sample.
        text_lens = rope_meta.text.seqlens.to("cpu", torch.int64).tolist()
        pixel_lens = rope_meta.pixel.seqlens.to("cpu", torch.int64).tolist()
        audio_lens = (
            rope_meta.audio.seqlens.to("cpu", torch.int64).tolist()
            if audio_emb is not None and rope_meta.audio is not None
            else None
        )
        n = len(text_lens)
        if len(pixel_lens) != n:
            raise ValueError("text/pixel seqlen count mismatch")
        pieces = []
        text_off = pix_off = aud_off = 0
        pix_per = pixel_lens[0]
        for i in range(n):
            t_len = text_lens[i]
            t_slice = text_emb[text_off : text_off + t_len]
            text_off += t_len
            p_slice = pixel_emb[pix_off : pix_off + pix_per]
            pix_off += pix_per
            if audio_emb is not None:
                a_len = audio_lens[i]
                a_slice = audio_emb[aud_off : aud_off + a_len]
                aud_off += a_len
                pieces.append(torch.cat([t_slice, p_slice, a_slice], dim=0))
            else:
                pieces.append(torch.cat([t_slice, p_slice], dim=0))
        x = torch.cat(pieces, dim=0).unsqueeze(0)
        mix_freqs = self.compute_freqs_cis(rope_meta.mix.position_id)

        sp_size = collectives.named_group_size(self.sp_group, device) if self.sp_group is not None else 1
        mix_seqlens = rope_meta.mix.seqlens
        mix_attn_seqlens = mix_seqlens
        audio_mix_emb = None
        modulation_ids = None
        if audio_emb is not None and audio_timestep is not None:
            audio_mix_emb = audio_timestep_emb
            # Text/video tokens select [0, N); audio tokens select [N, 2N).
            modulation_ids = torch.cat(
                [
                    torch.cat(
                        [
                            torch.full((text_lens[i] + pix_per,), i, dtype=torch.int64, device=device),
                            torch.full((audio_lens[i],), n + i, dtype=torch.int64, device=device),
                        ]
                    )
                    for i in range(n)
                ]
            )
        if sp_size > 1:
            # The mix pad must also lengthen the LAST sample's attn seqlens so
            # varlen SDPA keeps CFG samples isolated; freqs are padded (not
            # scattered) because RoPE runs after the Ulysses all2all.
            s_orig = x.size(1)
            pad = (sp_size - s_orig % sp_size) % sp_size
            mix_lens_cpu = mix_seqlens.to("cpu", torch.int64)
            sample_ids = torch.repeat_interleave(torch.arange(n, dtype=torch.int64), mix_lens_cpu, dim=0).to(device)
            if pad > 0:
                attn_lens = mix_lens_cpu.clone()
                attn_lens[-1] += pad
                mix_attn_seqlens = attn_lens.to(torch.int32).to(device)
                mix_freqs = F.pad(mix_freqs, (0, 0, 0, pad), mode="constant", value=0)

            sp_inputs = {"x": (x, 1), "sample_ids": (sample_ids.unsqueeze(0), 1)}
            if modulation_ids is not None:
                sp_inputs["modulation_ids"] = (modulation_ids.unsqueeze(0), 1)
            sp_outputs = self.sequence_parallel_forward(
                sp_inputs,
                lambda local: {
                    "x": (
                        self._run_mix_layers(
                            local["x"],
                            timestep_emb,
                            mix_seqlens,
                            mix_attn_seqlens,
                            mix_freqs,
                            local["sample_ids"].squeeze(0),
                            audio_mix_emb,
                            local["modulation_ids"].squeeze(0) if "modulation_ids" in local else None,
                        ),
                        1,
                    )
                },
            )
            x = sp_outputs["x"][0]
        else:
            x = self._run_mix_layers(
                x, timestep_emb, mix_seqlens, mix_attn_seqlens, mix_freqs, None, audio_mix_emb, modulation_ids
            )

        # Slice each sample's pixel / audio tail back out of the mix output.
        pix_outs = []
        aud_outs = []
        off = 0
        for i in range(n):
            off += text_lens[i]
            pix_outs.append(x[0, off : off + pix_per])
            off += pix_per
            if audio_emb is not None:
                a_len = audio_lens[i]
                aud_outs.append(x[0, off : off + a_len])
                off += a_len
        pix_h = torch.cat(pix_outs, dim=0)
        pixel_vel = self.proj_out_linear(self.proj_out_norm(pix_h))

        audio_vel = None
        if aud_outs:
            aud_h = torch.cat(aud_outs, dim=0)
            audio_vel = self.audio_proj_out_linear(self.audio_proj_out_norm(aud_h))
        return pixel_vel, audio_vel

    def _run_mix_layers(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        mix_seqlens: torch.Tensor,
        mix_attn_seqlens: torch.Tensor,
        mix_freqs: torch.Tensor,
        token_sample_ids: torch.Tensor | None,
        audio_timestep_emb: torch.Tensor | None = None,
        modulation_sample_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ulysses = token_sample_ids is not None
        for block in self.transformer_blocks:
            x = block(
                x,
                timestep_emb,
                mix_freqs,
                mix_seqlens,
                ulysses_sp=ulysses,
                attn_seqlens=mix_attn_seqlens,
                token_sample_ids=token_sample_ids,
                audio_timestep_emb=audio_timestep_emb,
                modulation_sample_ids=modulation_sample_ids,
            )
        return x

    # ---- weights -------------------------------------------------------------

    def load_weights(self, state_dicts: list) -> None:
        rank = self.tp_rank
        world = self.tp_size
        self.x_embedder.load_full(state_dicts, "x_embedder", rank, world)
        self.audio_embedder.load_full(state_dicts, "audio_embedder", rank, world)
        self.context_embedder_proj.load_full(prefixed(state_dicts, "context_embedder."), "proj", rank, world)
        _load_norm_weight(
            self.context_embedder_norm,
            prefixed(state_dicts, "context_embedder."),
            "norm.weight",
        )
        self.time_embedder_linear_1.load_full(prefixed(state_dicts, "time_embedder."), "linear_1", rank, world)
        self.time_embedder_linear_2.load_full(prefixed(state_dicts, "time_embedder."), "linear_2", rank, world)
        self.proj_out_linear.load_full(prefixed(state_dicts, "proj_out."), "linear", rank, world)
        _load_norm_weight(self.proj_out_norm, prefixed(state_dicts, "proj_out."), "norm.weight")
        self.audio_proj_out_linear.load_full(prefixed(state_dicts, "audio_proj_out."), "linear", rank, world)
        _load_norm_weight(
            self.audio_proj_out_norm,
            prefixed(state_dicts, "audio_proj_out."),
            "norm.weight",
        )
        for i, block in enumerate(self.noise_refiner):
            block.load_weights(prefixed(state_dicts, f"noise_refiner.{i}."), rank, world)
        for i, block in enumerate(self.context_text_refiner):
            block.load_weights(prefixed(state_dicts, f"context_text_refiner.{i}."), rank, world)
        for i, block in enumerate(self.audio_refiner):
            block.load_weights(prefixed(state_dicts, f"audio_refiner.{i}."), rank, world)
        for i, block in enumerate(self.transformer_blocks):
            block.load_weights(prefixed(state_dicts, f"transformer_blocks.{i}."), rank, world)
