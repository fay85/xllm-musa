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

"""Qwen3.5 MTP draft model for the MUSA Python executor."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, GemmaRMSNorm, HiddenParallelEmbedding
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3_5 import (
    PartialRotaryEmbedding,
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
    _store_fused_qk_norm_weight,
    pack_gated_qkv_projection,
)
from xllm.python.platform import current_platform


class Qwen3_5MtpModel(nn.Module):
    def __init__(
        self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device
    ) -> None:
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
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.fc = nn.Linear(
            2 * cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        rotary = PartialRotaryEmbedding(
            cfg.head_dim,
            rotary_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype,
            device,
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(cfg, i, dtype, device, rotary)
            for i in range(cfg.n_layers)
        )
        self.norm = GemmaRMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.embed_tokens(input_ids)
        metadata = get_forward_context().metadata
        if input_embedding is None:
            input_embedding = getattr(metadata, "input_embedding", None)
        hidden = embedding if input_embedding is None else input_embedding
        embedding = self.pre_fc_norm_embedding(embedding)
        hidden = self.pre_fc_norm_hidden(hidden)
        hidden = self.fc(torch.cat((embedding, hidden), dim=-1))
        if current_platform.is_musa():
            if positions.dtype != torch.int32:
                positions = positions.to(torch.int32)
            if not positions.is_contiguous():
                positions = positions.contiguous()

        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class Qwen3_5MtpForCausalLM(PyModelBase):
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
        self.model = Qwen3_5MtpModel(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        cfg = self.cfg

        def find(name: str) -> object | None:
            for state_dict in state_dicts:
                if state_dict.has(name):
                    return state_dict
            return None

        def resolve(*names: str) -> str:
            for name in names:
                if find(name) is not None:
                    return name
            raise KeyError(f"checkpoint tensor not found; tried {names}")

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
            self.get_parameter(name).data.copy_(value)

        def load_linear(module: nn.Module, name: str, dim: int) -> None:
            weight = shard_tensor(name, dim)
            scale_name = name.replace(".weight", ".weight_scale_inv")
            if find(scale_name) is not None:
                module.load_fp8(weight, shard_tensor(scale_name, dim))
            else:
                module.weight.data.copy_(weight)

        def shard_kv(name: str) -> torch.Tensor:
            if cfg.n_kv_heads >= tp_size:
                return shard_tensor(name, 0)
            replicas = tp_size // cfg.n_kv_heads
            return shard_tensor(name, 0, tp_rank // replicas, cfg.n_kv_heads)

        embedding_name = resolve(
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        )
        copy_in("model.embed_tokens.weight", shard_tensor(embedding_name, 1))
        copy_in(
            "model.pre_fc_norm_embedding.weight",
            tensor(resolve("mtp.pre_fc_norm_embedding.weight")),
        )
        copy_in(
            "model.pre_fc_norm_hidden.weight",
            tensor(resolve("mtp.pre_fc_norm_hidden.weight")),
        )
        copy_in("model.fc.weight", tensor(resolve("mtp.fc.weight")))

        for layer_id, layer_type in enumerate(cfg.layer_types):
            if layer_type != "full_attention":
                raise ValueError("Qwen3.5 MTP body must use full attention")
            source = f"mtp.layers.{layer_id}."
            layer = self.model.layers[layer_id]
            for norm in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                copy_in(f"model.layers.{layer_id}.{norm}", tensor(resolve(source + norm)))

            q_name = resolve(source + "self_attn.q_proj.weight")
            k_name = resolve(source + "self_attn.k_proj.weight")
            v_name = resolve(source + "self_attn.v_proj.weight")
            q = shard_tensor(q_name, 0)
            k = shard_kv(k_name)
            v = shard_kv(v_name)
            q_scale_name = q_name.replace(".weight", ".weight_scale_inv")
            local_heads = layer.self_attn.num_heads
            if find(q_scale_name) is not None:
                qs = shard_tensor(q_scale_name, 0)
                ks = shard_kv(k_name.replace(".weight", ".weight_scale_inv"))
                vs = shard_kv(v_name.replace(".weight", ".weight_scale_inv"))
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
                layer.self_attn.o_proj,
                resolve(source + "self_attn.o_proj.weight"),
                1,
            )
            copy_in(
                f"model.layers.{layer_id}.self_attn.q_norm.weight",
                tensor(resolve(source + "self_attn.q_norm.weight")),
            )
            copy_in(
                f"model.layers.{layer_id}.self_attn.k_norm.weight",
                tensor(resolve(source + "self_attn.k_norm.weight")),
            )
            _store_fused_qk_norm_weight(layer.self_attn.q_norm, self.dtype)
            _store_fused_qk_norm_weight(layer.self_attn.k_norm, self.dtype)

            gate_name = resolve(source + "mlp.gate_proj.weight")
            up_name = resolve(source + "mlp.up_proj.weight")
            gate = shard_tensor(gate_name, 0)
            up = shard_tensor(up_name, 0)
            gate_scale_name = gate_name.replace(".weight", ".weight_scale_inv")
            if find(gate_scale_name) is not None:
                gate_scale = shard_tensor(gate_scale_name, 0)
                up_scale = shard_tensor(
                    up_name.replace(".weight", ".weight_scale_inv"), 0
                )
                layer.mlp.gate_up_proj.load_fp8(
                    torch.cat((gate, up)), torch.cat((gate_scale, up_scale))
                )
            else:
                layer.mlp.gate_up_proj.weight.data.copy_(torch.cat((gate, up)))
            load_linear(
                layer.mlp.down_proj,
                resolve(source + "mlp.down_proj.weight"),
                1,
            )

        copy_in("model.norm.weight", tensor(resolve("mtp.norm.weight")))
        copy_in("lm_head.weight", shard_tensor(resolve("lm_head.weight"), 0))
        self.lm_head.prefer_nn_gemm_layout()

