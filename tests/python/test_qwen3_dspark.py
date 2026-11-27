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

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python import kernels
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.dspark_model import DSparkModel
from xllm.python.models.qwen3_dspark import (
    Qwen3DSparkConfig,
    Qwen3DSparkForCausalLM,
)


def _config(**overrides) -> Qwen3DSparkConfig:
    values = _config_dict(**overrides)
    config = Qwen3DSparkConfig.from_dict(values)
    config.validate()
    return config


def _config_dict(**overrides) -> dict:
    values = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 16,
        "vocab_size": 8,
        "draft_vocab_size": 8,
        "markov_rank": 2,
        "num_speculative_tokens": 3,
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_config_requires_positive_markov_rank() -> None:
    with pytest.raises(ValueError, match="markov_rank > 0"):
        _config(markov_rank=0)


@pytest.mark.parametrize("query_width", [1, 3, 15])
def test_pipeline_uses_runtime_query_width(query_width: int) -> None:
    model = Qwen3DSparkForCausalLM(_config_dict(num_speculative_tokens=query_width))

    assert model.model._query_width == query_width


@pytest.mark.parametrize("is_ascend_a3", [False, True])
def test_device_only_decode_capability_requires_a3(
    monkeypatch: pytest.MonkeyPatch,
    is_ascend_a3: bool,
) -> None:
    runtime = types.ModuleType("xllm_runtime")
    runtime.is_ascend_a3 = Mock(return_value=is_ascend_a3)
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    model = Qwen3DSparkForCausalLM(_config_dict(hidden_size=128, head_dim=128, dtype="bfloat16"))

    assert model.supports_device_only_decode_inputs(kv_page_size=128) is is_ascend_a3


@pytest.mark.parametrize(
    "head_dim,kv_page_size,expected",
    [
        (64, 128, False),
        (256, 128, False),
        (128, 64, False),
        (128, 256, False),
        (64, 64, False),
        (128, 128, True),
    ],
)
def test_device_only_decode_capability_requires_supported_geometry(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    kv_page_size: int,
    expected: bool,
) -> None:
    runtime = types.ModuleType("xllm_runtime")
    runtime.is_ascend_a3 = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    model = Qwen3DSparkForCausalLM(_config_dict(hidden_size=head_dim, head_dim=head_dim, dtype="bfloat16"))

    assert model.supports_device_only_decode_inputs(kv_page_size) is expected


def test_checkpoint_loads_dspark_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_row_parallel_weight = Mock(side_effect=lambda weight: (weight, False))
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        prepare_row_parallel_weight,
        raising=False,
    )
    model = Qwen3DSparkForCausalLM(
        _config_dict(
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
    )
    tensors = {
        "d2t": torch.tensor([0, 1, 2, 3, 0, 0, 0, 0]),
        "fc.weight": torch.eye(4),
        "hidden_norm.weight": torch.ones(4),
        "layers.0.input_layernorm.weight": torch.ones(4),
        "layers.0.post_attention_layernorm.weight": torch.ones(4),
        "layers.0.self_attn.q_norm.weight": torch.ones(4),
        "layers.0.self_attn.k_norm.weight": torch.ones(4),
        "layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
        "layers.0.self_attn.k_proj.weight": torch.full((4, 4), 2.0),
        "layers.0.self_attn.v_proj.weight": torch.full((4, 4), 3.0),
        "layers.0.self_attn.o_proj.weight": torch.full((4, 4), 4.0),
        "layers.0.mlp.gate_proj.weight": torch.full((8, 4), 5.0),
        "layers.0.mlp.up_proj.weight": torch.full((8, 4), 6.0),
        "layers.0.mlp.down_proj.weight": torch.full((4, 8), 7.0),
        "norm.weight": torch.ones(4),
        "markov_head.markov_w1.weight": torch.full((8, 2), 8.0),
        "markov_head.markov_w2.weight": torch.full((8, 2), 9.0),
        "confidence_head.proj.weight": torch.full((1, 6), 10.0),
        "confidence_head.proj.bias": torch.full((1,), 11.0),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    assert isinstance(model.model, DSparkModel)
    assert "model.backbone.fc.weight" in dict(model.named_parameters())
    torch.testing.assert_close(model.model.hot_token_ids, tensors["d2t"] + torch.arange(8))

    assert prepare_row_parallel_weight.call_count == 2
    torch.testing.assert_close(
        model.model.markov_head.markov_w1.weight,
        tensors["markov_head.markov_w1.weight"],
    )
    torch.testing.assert_close(
        model.model.markov_head.markov_w2.weight,
        tensors["markov_head.markov_w2.weight"],
    )
    assert model.model.confidence_head is not None
    torch.testing.assert_close(
        model.model.confidence_head.proj.weight,
        tensors["confidence_head.proj.weight"],
    )
    torch.testing.assert_close(
        model.model.confidence_head.proj.bias,
        tensors["confidence_head.proj.bias"],
    )


def test_community_swa_config_sets_anchor_attention() -> None:
    values = _config_dict(dflash_config={"use_swa": True, "swa_window_size": 4, "causal": False})
    model = Qwen3DSparkForCausalLM(values)
    attention = model.model.backbone.layers[0].self_attn.attn
    assert attention.attention_window == 4
    assert not attention.causal
    assert model.cfg.layer_types == ("sliding_attention",)
    values.update(use_sliding_window=False, sliding_window=8, sliding_window_non_causal=False)
    cfg = Qwen3DSparkConfig.from_dict(values)
    cfg.validate()
    assert not cfg.use_sliding_window
    assert not cfg.sliding_window_non_causal
    assert cfg.sliding_window == 8
    values.update(use_sliding_window=True, sliding_window=0)
    with pytest.raises(ValueError, match="sliding_window > 0"):
        Qwen3DSparkConfig.from_dict(values).validate()


@pytest.mark.parametrize("causal", [False, True])
def test_swa_eager_and_graph_keep_one_anchor_for_the_whole_proposal(
    monkeypatch: pytest.MonkeyPatch, causal: bool
) -> None:
    def scatter(slots, key, value, kc, vc):
        kc.flatten(0, 1).index_copy_(0, slots, key)
        vc.flatten(0, 1).index_copy_(0, slots, value)

    def fia(query, key, value, **kwargs):
        assert kwargs["input_layout"] == "BSND"
        scores = query.transpose(1, 2) @ key.permute(0, 2, 3, 1)
        scores.masked_fill_(kwargs["atten_mask"], -torch.inf)
        output = scores.softmax(-1) @ value.transpose(1, 2)
        return output.transpose(1, 2).contiguous(), None

    monkeypatch.setattr(kernels, "reshape_paged_cache", scatter, raising=False)
    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", fia, raising=False)
    table = torch.tensor([[2, 0, 1], [6, 7, 4]], dtype=torch.int32)
    slots = torch.tensor([11, 0, 1, 16, 17, 18])
    query = key = torch.zeros(6, 1, 1)
    value = torch.tensor([3, 4, 5, 108, 109, 110], dtype=torch.float32).view(6, 1, 1)
    expected = [1.5, 2, 2.5, 106, 106.5, 107] if causal else [2.5] * 3 + [107] * 3
    for graph_mode in (False, True):
        cache = LayerCache(torch.full((8, 4, 1, 1), float("nan")), torch.full((8, 4, 1, 1), float("nan")))
        for row, anchor in enumerate((3, 8)):
            positions = torch.arange(anchor)
            physical = table[row, positions // 4].long() * 4 + positions % 4
            cache.key.flatten(0, 1).index_fill_(0, physical, 0)
            cache.value.flatten(0, 1)[physical] = (positions + 100 * row).float().view(-1, 1, 1)
        metadata = SimpleNamespace(
            block_table=table.repeat_interleave(3, dim=0) if graph_mode else table,
            kv_seq_lens=torch.tensor([6] * 3 + [11] * 3 if graph_mode else [6, 11], dtype=torch.int32),
            kv_seq_lens_host_values=[6] * 3 + [11] * 3 if graph_mode else [6, 11],
            slot_mapping=slots,
            q_cu_seq_lens=None,
            q_cu_seq_lens_host_values=[3, 6],
            is_prefill=False,
            is_chunked_prefill=not graph_mode,
            is_mixed=False,
            is_spec_verify=False,
            use_dflash_proposal_attention=True,
            kv_split_size=1,
            dsa_graph_token_capacity=8,
        )
        backend = NpuPagedAttentionBackend(
            1,
            1,
            1,
            1.0,
            4,
            False,
            torch.device("cpu"),
            torch.float32,
            draft_query_width=3,
            draft_graph_window_size=4,
        )
        backend.bind_kv_caches([cache])
        layer = SimpleNamespace(layer_id=0, causal=causal, attention_window=4)
        context = ForwardContext(backend, torch.device("cpu"), metadata, [cache])
        with forward_context(context):
            backend.prepare(metadata, graph_mode=graph_mode)
            output = backend.execute(query, key, value, layer)
        torch.testing.assert_close(output.flatten(), torch.tensor(expected))
