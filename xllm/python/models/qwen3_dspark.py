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

"""Qwen3 DSpark draft model for the Python NPU executor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_loader import ParallelLoadContext, load_draft_lm_head_if_present
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.qwen3_dflash import (
    DFlashQwen3Config,
    DFlashQwen3Model,
)


@dataclass
class Qwen3DSparkConfig(DFlashQwen3Config):
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False
    sliding_window_non_causal: bool = False
    layer_types: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3DSparkConfig:
        nested = d.get("dflash_config")
        if nested is not None and not isinstance(nested, dict):
            raise ValueError("dflash_config must be a dict when present")
        nested = nested or {}
        base = DFlashQwen3Config.from_dict(cls._apply_swa_aliases(d, nested))
        non_causal = d.get("sliding_window_non_causal")
        if non_causal is None:
            non_causal = not bool(nested.get("causal", True)) if nested.get("causal") is not None else False
        layer_types = tuple(d.get("layer_types") or ())
        if not layer_types:
            layer_types = ("sliding_attention" if base.use_sliding_window else "full_attention",) * base.n_layers
        return cls(
            **(base.__dict__ | {"layer_types": layer_types}),
            markov_rank=int(d.get("markov_rank", 0)),
            enable_confidence_head=bool(d.get("enable_confidence_head", False)),
            confidence_head_with_markov=bool(d.get("confidence_head_with_markov", False)),
            sliding_window_non_causal=bool(non_causal),
        )

    def validate(self) -> None:
        super().validate()
        if self.markov_rank <= 0:
            raise ValueError("Qwen3 DSpark requires markov_rank > 0")
        if len(self.layer_types) != self.n_layers:
            raise ValueError("DSpark layer_types must contain one entry per hidden layer")
        if any(layer_type not in ("sliding_attention", "full_attention") for layer_type in self.layer_types):
            raise ValueError("DSpark layer_types must use sliding_attention or full_attention")
        if not self.use_sliding_window:
            if "sliding_attention" in self.layer_types:
                raise ValueError("sliding_attention layers require use_sliding_window=true")
            return
        if self.sliding_window <= 0:
            raise ValueError("sliding-window DSpark requires sliding_window > 0")
        if self.cp_size != 1:
            raise ValueError("sliding-window DSpark requires cp_size=1")
        if self.layer_types != ("sliding_attention",) * self.n_layers:
            raise ValueError("sliding-window DSpark requires one sliding_attention layer type per draft layer")

    def _allows_reduced_vocab(self) -> bool:
        return True


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash draft backbone used with DSpark-specific output heads."""

    def _configure_sliding_attention(self, cfg: Qwen3DSparkConfig) -> None:
        # DSpark owns anchor windows rather than DFlash's per-query windows.
        if cfg.use_sliding_window:
            for layer in self.layers:
                layer.self_attn.attn.causal = not cfg.sliding_window_non_causal
                layer.self_attn.attn.attention_window = cfg.sliding_window


class Qwen3DSparkForCausalLM(DSparkForCausalLMBase):
    uses_block_query_graph_metadata: bool = True

    def __init__(self, config: dict) -> None:
        cfg = Qwen3DSparkConfig.from_dict(config)
        cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu"))
        super().__init__(
            backbone=Qwen3DSparkModel(cfg, dtype, device),
            lm_head=None,
            query_width=cfg.num_speculative_tokens,
            vocab_size=cfg.vocab_size,
            draft_vocab_size=cfg.draft_vocab_size,
            markov_rank=cfg.markov_rank,
            hidden_size=cfg.hidden_size,
            enable_confidence_head=cfg.enable_confidence_head,
            confidence_head_with_markov=cfg.confidence_head_with_markov,
            dtype=dtype,
            device=device,
        )
        self.cfg = cfg
        self.dtype = dtype
        self.device = device

    def supports_device_only_decode_inputs(self, kv_page_size: int) -> bool:
        import xllm_runtime

        return (
            type(self) is Qwen3DSparkForCausalLM
            and not self.cfg.use_sliding_window
            and self.cfg.n_heads % self.cfg.tp_size == 0
            and self.cfg.n_kv_heads % self.cfg.tp_size == 0
            and self.cfg.head_dim == 128
            and kv_page_size == 128
            and xllm_runtime.is_ascend_a3()
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        pipeline = self.model
        all_weights = pipeline.backbone.load_weights(state_dicts, tp_rank, tp_size)
        if all_weights.has("d2t"):
            offsets = all_weights.get_tensor("d2t").to(device=self.device, dtype=torch.long)
            pipeline.hot_token_ids = offsets + torch.arange(offsets.numel(), device=self.device)
        load_draft_lm_head_if_present(
            pipeline,
            all_weights,
            context=ParallelLoadContext(tp_rank, tp_size),
        )
        all_weights.load_tensor(pipeline.markov_head.markov_w1.weight, "markov_head.markov_w1.weight")
        all_weights.load_tensor(pipeline.markov_head.markov_w2.weight, "markov_head.markov_w2.weight")
        if pipeline.confidence_head is not None:
            all_weights.load_tensor(pipeline.confidence_head.proj.weight, "confidence_head.proj.weight")
            all_weights.load_tensor(pipeline.confidence_head.proj.bias, "confidence_head.proj.bias")

    def adapt_weights_for_reference_model(
        self,
        reference_model_path: str,
    ) -> None:
        self.model.backbone.adapt_weights_for_reference_model(reference_model_path)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        return self.model.backbone.write_context_kv(
            target_hidden,
            positions,
            cache_slots,
            kv_caches,
            layer_synchronizer,
        )
