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

"""Text-only Joyo encoding must return the pre-norm residual, not final hidden."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xllm.python import kernels

from .conftest import load_source_module

_collectives = load_source_module(
    Path(__file__).parents[2] / "xllm/python/distributed/collectives.py", "_xllm_collectives_under_test"
)
sys.modules.setdefault("xllm.python.distributed.collectives", _collectives)
from xllm.python.models.dit.encoders import joyo


def test_text_encoder_reuses_backbone_and_preserves_pre_norm_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(joyo, "named_group", lambda *_: SimpleNamespace(rank=lambda: 0, size=lambda: 1))
    tokenizer_calls = []

    def tokenize(text: str) -> list[int]:
        tokenizer_calls.append(text)
        return [1, 2, 3]

    def norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight

    def fused_norm(value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple:
        combined = value + residual
        return norm(combined, weight, eps), combined

    def qkv_rope(qkv: torch.Tensor, **kwargs: object) -> tuple:
        from xllm.python.layers.rotary_embedding import apply_rotary_half, gather_half_rope_cos_sin

        dim, heads, kv_heads = kwargs["head_dim"], kwargs["num_heads_q"], kwargs["num_heads_k"]
        q, k, v = torch.split(qkv, [heads * dim, kv_heads * dim, kv_heads * dim], dim=-1)
        q, k, v = q.reshape(-1, heads, dim), k.reshape(-1, kv_heads, dim), v.reshape(-1, kv_heads, dim)
        q = norm(q, kwargs["q_weight"], kwargs["eps"])
        k = norm(k, kwargs["k_weight"], kwargs["eps"])
        cos, sin = gather_half_rope_cos_sin(kwargs["cos_sin_cache"], kwargs["position_ids"])
        return apply_rotary_half(q, cos, sin), apply_rotary_half(k, cos, sin), v

    monkeypatch.setattr(kernels, "fused_add_rms_norm", fused_norm, raising=False)
    monkeypatch.setattr(kernels, "fused_qk_norm_rope", qkv_rope, raising=False)
    monkeypatch.setattr(
        kernels,
        "silu_and_mul",
        lambda x: torch.nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
        raising=False,
    )
    encoder = joyo.JoyoTextEncoder(
        {
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 16,
                "max_position_embeddings": 32,
                "vocab_size": 16,
                "rms_norm_eps": 1e-6,
            },
        },
        SimpleNamespace(encode=tokenize),
        torch.device("cpu"),
        torch.float32,
    )
    torch.manual_seed(17)
    for parameter in encoder.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    before_norm = []

    def capture(_module: object, args: tuple) -> None:
        before_norm.append((args[0] + args[1]).detach().clone())

    handle = encoder.model.norm.register_forward_pre_hook(capture)
    output = encoder.encode_joyo_prompts(["positive", "negative"])
    handle.remove()
    assert len(output) == 2
    for actual, expected in zip(output, before_norm):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (3, 8)
    assert tokenizer_calls == [
        "<|im_start|>user\npositive<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nnegative<|im_end|>\n<|im_start|>assistant\n",
    ]
    assert not hasattr(encoder, "vision_model")
    assert not hasattr(encoder, "lm_head")

    with joyo.forward_context(joyo.ForwardContext(encoder._attention, torch.device("cpu"), None, [])):
        hidden = encoder.model(torch.tensor([1, 2, 3]), torch.arange(3))
    torch.testing.assert_close(hidden, norm(output[0], encoder.model.norm.weight, encoder.model.norm.eps))
    encoder.model.to("meta")
    with pytest.raises(RuntimeError, match="offloaded"):
        encoder.encode_joyo_prompts(["must fail before tokenizing"])
    assert len(tokenizer_calls) == 2
