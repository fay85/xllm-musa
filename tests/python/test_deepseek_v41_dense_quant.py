# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Checkpoint-driven dense activation quantization, without quantizing BF16 layers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xllm.python.kernels_musa.dense_quant import V41OriginalFp8Dense, share_original_fp8_input
from xllm.python.models import deepseek_v41 as v41
from tests.python.test_deepseek_v41_model import (
    _DEEPSEEK_V41_TINY_CONFIG, _DEEPSEEK_V41_FORWARD_CONFIG, _FakeStateDict, _build_tiny_checkpoint,
)


@pytest.mark.parametrize("shape", [(0, 32), (1, 32), (2, 3, 32)])
def test_wrapper_shares_parameter_and_quantizes_before_projection(shape: tuple[int, ...]) -> None:
    original = nn.Linear(32, 19, bias=False)
    calls = []

    def _round(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(x.clone())
        return x.round(), torch.ones(x.shape[0], 1)

    wrapped = V41OriginalFp8Dense(original, _round)
    x = torch.randn(shape)
    torch.testing.assert_close(wrapped(x), original(x.round()), rtol=0, atol=0)
    assert wrapped.weight is original.weight
    assert list(wrapped.state_dict()) == ["weight"]
    assert len(calls) == 1 and calls[0].shape == (x.numel() // 32, 32)
    with pytest.raises(ValueError, match="loaded K"):
        wrapped(torch.ones(1, 31))


def test_fp8_loader_marks_only_original_quantized_projections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = v41.DeepseekV41Config.from_dict(_DEEPSEEK_V41_TINY_CONFIG)
    config = {**_DEEPSEEK_V41_TINY_CONFIG, "device": "cpu", "dtype": "float32", "model_path": str(tmp_path)}
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    model = v41.DeepseekV41ForCausalLM(config)
    tensors = _build_tiny_checkpoint(cfg)
    # Independently enumerate the official quantized projections; compressor,
    # routing gate, mHC and Indexer weights_proj/wk intentionally stay plain.
    prefixes = []
    for index in range(cfg.n_layers):
        base = f"layers.{index}."
        prefixes.extend(base + "attn." + name for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"))
        if base + "attn.indexer.wq_b.weight" in tensors:
            prefixes.append(base + "attn.indexer.wq_b")
        prefixes.extend(base + "ffn.shared_experts." + name for name in ("w1", "w2", "w3"))
        if index in cfg.engram_layer_ids:
            prefixes.append(base + "engram.wkv")
        for expert in range(cfg.n_routed_experts):
            for name in ("w1", "w2", "w3"):
                prefix = base + f"ffn.experts.{expert}.{name}"
                n, k = tensors[prefix + ".weight"].shape
                tensors[prefix + ".weight"] = torch.full((n, k // 2), 0x22, dtype=torch.uint8)
                tensors[prefix + ".scale"] = torch.full((n, k // 32), 120, dtype=torch.uint8)
    for prefix in prefixes:
        weight = tensors[prefix + ".weight"]
        tensors[prefix + ".weight"] = weight.to(torch.float8_e4m3fn)
        tensors[prefix + ".scale"] = torch.full((-(weight.shape[0] // -32), -(weight.shape[1] // -32)), 127, dtype=torch.uint8)
    seen = []

    def _record(module: nn.Module) -> nn.Module:
        seen.append(module)
        return module

    monkeypatch.setattr(v41, "select_original_fp8_dense", _record)
    model.load_weights([_FakeStateDict(tensors)], 0, 1)
    names = {id(module): name for name, module in model.named_modules()}
    actual = {names[id(module)] for module in seen}
    expected = set()
    for index in range(cfg.n_layers):
        base = f"model.layers.{index}."
        expected.update(base + "self_attn." + name for name in ("q_a_proj", "q_b_proj", "kv_proj", "o_b_proj"))
        expected.update(base + "mlp.shared_" + name for name in ("w1", "w2", "w3"))
        if model.model.layers[index].self_attn.indexer is not None:
            expected.add(base + "self_attn.indexer.wq_b")
        if index in cfg.engram_layer_ids:
            expected.add(base + "engram.wkv")
    assert actual == expected
    assert len(seen) == len(expected)


def test_checkpoint_fixture_preserves_attention_geometry_after_engram() -> None:
    cfg = v41.DeepseekV41Config.from_dict(_DEEPSEEK_V41_FORWARD_CONFIG)
    assert cfg.head_dim != cfg.engram_head_dim
    tensors = _build_tiny_checkpoint(cfg)
    for layer in range(cfg.n_layers):
        assert tensors[f"layers.{layer}.attn.wq_b.weight"].shape == (cfg.n_heads * cfg.head_dim, cfg.q_lora_rank)
        assert tensors[f"layers.{layer}.attn.wkv.weight"].shape == (cfg.head_dim, cfg.hidden_size)
    for layer in cfg.index_source_layer_ids:
        assert f"layers.{layer}.attn.indexer.wq_b.weight" in tensors
    # Reindex owns only the query and scoring projection, not a second K.
    assert "layers.5.attn.indexer.wk.weight" not in tensors


def test_shared_gate_and_up_keep_independent_weights_and_one_activation_round() -> None:
    calls = []

    def _round(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(x.shape)
        return x.round(), torch.ones(x.size(0), 1)

    gate = V41OriginalFp8Dense(nn.Linear(32, 13, bias=False), _round)
    up = V41OriginalFp8Dense(nn.Linear(32, 17, bias=False), _round)
    x = torch.randn(2, 3, 32)
    expected = gate(x), up(x)
    assert len(calls) == 2
    calls.clear()
    actual = share_original_fp8_input(x, gate, up)
    assert actual is not None
    assert calls == [(6, 32)]
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert share_original_fp8_input(x, gate, nn.Linear(32, 17, bias=False)) is None
