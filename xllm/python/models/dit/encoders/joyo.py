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

"""Joyo caption adapter over the existing Python Qwen3-VL text backbone."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, LayerCache
from xllm.python.distributed.collectives import named_group, tp_group_scope
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader, load_causal_lm_weights
from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_vl import Qwen3VLModel


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class _TextAttention(AttentionBackend):
    """Single-prompt causal prefill over current Q/K/V, with no paged KV pool."""

    supports_anchor_context_window = False

    def __init__(self) -> None:
        self._causal_mask: torch.Tensor | None = None

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        if kv_caches:
            raise ValueError("Joyo text encoding does not use a paged KV cache")

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        if graph_mode:
            raise ValueError("Joyo text encoder does not support paged graph execution")

    @property
    def num_kv_blocks(self) -> int:
        return 0

    @property
    def page_size(self) -> int:
        return 0

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        length = q.shape[0]
        q = q.reshape(length, layer.num_heads, layer.head_dim)
        k = k.reshape(length, layer.num_kv_heads, layer.head_dim)
        v = v.reshape(length, layer.num_kv_heads, layer.head_dim)
        if q.device.type == "npu":
            if self._causal_mask is None:
                self._causal_mask = torch.ones((2048, 2048), dtype=torch.int8, device=q.device).triu_(1)
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                atten_mask=self._causal_mask,
                actual_seq_lengths=[length],
                actual_seq_lengths_kv=[length],
                num_heads=layer.num_heads,
                num_key_value_heads=layer.num_kv_heads,
                scale=layer.scale,
                input_layout="TND",
                sparse_mode=3,
                softmax_lse_flag=False,
            )
        else:
            ratio = layer.num_heads // layer.num_kv_heads
            output = (
                F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0),
                    v.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0),
                    is_causal=True,
                    scale=layer.scale,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        return output.reshape(length, -1)


class JoyoTextEncoder(nn.Module):
    """Own one text-only network; return full unpadded pre-norm residuals."""

    def __init__(
        self,
        config: dict,
        tokenizer: _Tokenizer,
        device: torch.device,
        dtype: torch.dtype,
        group_name: str = "dit_text_encoder_tp",
    ) -> None:
        super().__init__()
        if config.get("model_type") != "qwen3_vl":
            raise ValueError("Joyo text encoding currently requires a Qwen3-VL checkpoint")
        group = named_group(group_name, device)
        if group is None:
            raise ValueError(f"Text encoder TP group {group_name} is not registered for {device}")
        self.group_name = group_name
        self.tp_rank, self.tp_size = group.rank(), group.size()
        text = dict(config.get("text_config", config))
        text.update(tp_rank=self.tp_rank, tp_size=self.tp_size)
        cfg = Qwen3Config.from_dict(text)
        cfg.mrope_section = list((text.get("rope_scaling") or {}).get("mrope_section", []))
        if not text.get("use_sliding_window", config.get("use_sliding_window", False)):
            cfg.sliding_window = 0
        if cfg.sliding_window:
            raise ValueError("Joyo text prefill requires full causal attention, not sliding-window attention")
        self.model = Qwen3VLModel(cfg, dtype=dtype, device=device)
        self.tokenizer = tokenizer
        self.target_device = device
        self.dtype = dtype
        self._attention = _TextAttention()

    def load_weights(self, state_dicts: list) -> None:
        weights = ScopedWeightLoader(state_dicts, src_prefixes=("model.language_model.", "model.", ""))
        load_causal_lm_weights(
            self.model,
            None,
            weights,
            ParallelLoadContext(self.tp_rank, self.tp_size),
            tie_word_embeddings=self.model.cfg.tie_word_embeddings,
        )

    def set_on_device(self, on_device: bool) -> None:
        self.model.to(self.target_device if on_device else torch.device("cpu"))

    @torch.inference_mode()
    def encode_joyo_prompts(self, prompts: list[str]) -> list[torch.Tensor]:
        if self.model.embed_tokens.weight.device != self.target_device:
            raise RuntimeError("Joyo text encoder is offloaded; call set_on_device(True) before encoding prompts")
        result = []
        context = ForwardContext(self._attention, self.target_device, None, [])
        with tp_group_scope(self.group_name), forward_context(context):
            for prompt in prompts:
                ids = self.tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
                if len(ids) > self.model.cfg.max_position_embeddings:
                    raise ValueError("Joyo prompt exceeds the text encoder position limit")
                if not ids:
                    raise ValueError("Joyo tokenizer returned an empty prompt")
                tokens = torch.tensor(ids, dtype=torch.int32, device=self.target_device)
                positions = torch.arange(len(ids), dtype=torch.int32, device=self.target_device)
                _, residual = self.model.forward_with_residual(tokens, positions)
                result.append(residual.to(dtype=self.dtype))
        return result
