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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python import distributed, kernels
from xllm.python.models.qwen3 import Qwen3Model
from xllm.python.models.qwen3_dflash import (
    DFlashContextProjection,
    DFlashQwen3Config,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


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
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def _config(**overrides) -> DFlashQwen3Config:
    config = DFlashQwen3Config.from_dict(_config_dict(**overrides))
    config.validate()
    return config


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_top_level_and_nested_rope_config_formats_are_supported() -> None:
    top_level_rope = _config(model_type="qwen3", rope_theta=10000.0)
    nested_rope = _config(
        model_type="qwen3",
        rope_theta=None,
        rope_parameters={"rope_theta": 1e7},
    )

    assert top_level_rope.rope_theta == 10000.0
    assert nested_rope.rope_theta == 1e7


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dp_size": 2, "world_size": 1}, "world_size must equal"),
        ({"dp_size": 2, "dp_rank": 2, "world_size": 2}, "dp_rank must be"),
        ({"tp_rank": 1}, "tp_rank must be"),
        ({"cp_size": 2, "world_size": 4}, "world_size must equal"),
        ({"cp_size": 0}, "parallel sizes must be positive"),
    ],
)
def test_config_rejects_invalid_parallel_settings(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_config_accepts_context_parallel_world_size() -> None:
    config = _config(dp_size=2, cp_size=2, world_size=4)

    assert config.cp_size == 2


def test_config_defaults_world_size_with_cp_dimension() -> None:
    config = _config(dp_size=2, cp_size=2)

    assert config.world_size == 4


def test_config_rejects_reduced_draft_vocabulary() -> None:
    with pytest.raises(ValueError, match="reduced-vocabulary"):
        _config(draft_vocab_size=4)


def test_config_rejects_aux_hidden_capture() -> None:
    with pytest.raises(ValueError, match="does not support layers_to_capture"):
        _config(layers_to_capture=[0])


def test_draft_attention_is_non_causal() -> None:
    model = DFlashQwen3Model(_config(), torch.float32, torch.device("cpu"))

    assert isinstance(model, Qwen3Model)
    assert model.embed_tokens is None
    assert not model.layers[0].self_attn.attn.causal


def test_context_projection_uses_tensor_parallel_output_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = DFlashContextProjection(
        out_features=4,
        tp_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    rank_zero_output = torch.nn.functional.linear(hidden, weight[:2])
    all_gather = Mock(
        side_effect=lambda local_output, **_: torch.cat(
            (rank_zero_output, local_output),
            dim=-1,
        )
    )
    monkeypatch.setattr(distributed, "all_gather", all_gather, raising=False)

    projection.load_weight(weight, tp_rank=1)
    output = projection(hidden)

    torch.testing.assert_close(projection.weight, weight[2:])
    torch.testing.assert_close(output, torch.nn.functional.linear(hidden, weight))
    all_gather.assert_called_once()
    torch.testing.assert_close(
        all_gather.call_args.args[0],
        torch.nn.functional.linear(hidden, weight[2:]),
    )
    assert all_gather.call_args.kwargs == {"dim": -1, "world_size": 2}


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_context_projection_writes_each_layer_cache(
    monkeypatch: pytest.MonkeyPatch,
    position_dtype: torch.dtype,
) -> None:
    config = _config()
    model = DFlashQwen3Model(config, torch.float32, torch.device("cpu"))
    attention = model.layers[0].self_attn
    with torch.no_grad():
        model.fc.load_weight(torch.eye(config.hidden_size), tp_rank=0)
        model.hidden_norm.weight.fill_(1.0)
        attention.qkv_proj.weight.zero_()
        attention.qkv_proj.weight[attention.q_size : attention.q_size + attention.kv_size].copy_(
            torch.eye(config.hidden_size)
        )
        attention.qkv_proj.weight[attention.q_size + attention.kv_size :].copy_(2.0 * torch.eye(config.hidden_size))
        attention.k_norm.weight.fill_(1.0)
        model._build_context_kv_buffers()
    monkeypatch.setattr(
        kernels,
        "rms_norm",
        lambda hidden, weight, eps: (
            hidden * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + eps) * weight
        ),
        raising=False,
    )
    reshape_paged_cache = Mock()
    monkeypatch.setattr(
        kernels,
        "reshape_paged_cache",
        reshape_paged_cache,
        raising=False,
    )
    synchronizer = Mock()
    key_cache = torch.empty(1, 1, 1, config.head_dim)
    value_cache = torch.empty_like(key_cache)
    target_hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    projected = model.write_context_kv(
        target_hidden,
        torch.tensor([0], dtype=position_dtype),
        torch.tensor([0], dtype=torch.int32),
        [(key_cache, value_cache, None, None, None)],
        synchronizer,
    )

    reshape_paged_cache.assert_called_once()
    call_args = reshape_paged_cache.call_args.args
    torch.testing.assert_close(
        call_args[2],
        2.0 * projected.view(1, 1, config.head_dim),
    )
    assert call_args[3] is key_cache
    assert call_args[4] is value_cache
    synchronizer.record_event.assert_called_once_with(0)


def test_context_projection_rejects_mrope_positions() -> None:
    model = DFlashQwen3Model(_config(), torch.float32, torch.device("cpu"))

    with pytest.raises(ValueError, match="token-aligned 1-D"):
        model.write_context_kv(
            torch.zeros(2, 4),
            torch.zeros(3, 2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            [],
            None,
        )


def test_context_projection_rejects_position_count_mismatch() -> None:
    model = DFlashQwen3Model(_config(), torch.float32, torch.device("cpu"))

    with pytest.raises(ValueError, match="position/hidden count mismatch"):
        model.write_context_kv(
            torch.zeros(2, 4),
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            [],
            None,
        )


def test_context_projection_rejects_float_positions() -> None:
    model = DFlashQwen3Model(_config(), torch.float32, torch.device("cpu"))

    with pytest.raises(ValueError, match="int32 or int64"):
        model.write_context_kv(
            torch.zeros(2, 4),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.int32),
            [],
            None,
        )


def test_checkpoint_weight_names_load_into_fused_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_row_parallel_weight = Mock(side_effect=lambda weight: (weight, False))
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        prepare_row_parallel_weight,
        raising=False,
    )
    model = DFlashQwen3ForCausalLM(_config_dict())
    tensors = {
        "fc.weight": torch.ones(4, 8),
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
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    qkv_weight = model.model.layers[0].self_attn.qkv_proj.weight
    torch.testing.assert_close(
        qkv_weight[:4],
        tensors["layers.0.self_attn.q_proj.weight"],
    )
    torch.testing.assert_close(
        qkv_weight[4:8],
        tensors["layers.0.self_attn.k_proj.weight"],
    )
    torch.testing.assert_close(
        qkv_weight[8:12],
        tensors["layers.0.self_attn.v_proj.weight"],
    )
    assert model.model._fused_kv_weight.shape == (8, 4)
    assert model.model.fc.weight.shape == (4, 8)
    assert model.model.embed_tokens is None
    assert model.lm_head is None
    assert prepare_row_parallel_weight.call_count == 2


# ---------------------------------------------------------------------------
# DFlash2 nested dflash_config parsing (incoai/GLM-5.3-DFlash2 layout).
# ---------------------------------------------------------------------------


def test_dflash2_nested_dflash_config_geometry_is_parsed() -> None:
    """Community DFlash2 checkpoints nest geometry under dflash_config.

    incoai/GLM-5.3-DFlash2 and the GLM-5.3-Flash pairing keep block_size,
    conv_* and selector_* inside a nested "dflash_config" object while our
    own exports use flat top-level dflash2_* keys. Both must parse to the
    same config values (the C++ qwen3.h loader reads only the nested path;
    the flat-key preference is a Python-side extension).
    """
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Config

    geometry = {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "selector_rank": 256,
        "selector_top_k": 16,
    }
    base = _config_dict(
        model_type="qwen3",
        sliding_window=2048,
        use_sliding_window=True,
        hidden_size=16,
    )
    base.pop("sliding_window")
    base.pop("use_sliding_window")
    nested = dict(base, dflash_config=dict(geometry, use_swa=True, swa_window_size=2048))
    flat = dict(base, sliding_window=2048, use_sliding_window=True, **{f"dflash2_{k}": v for k, v in geometry.items()})

    nested_cfg = DFlash2Qwen3Config.from_dict(nested)
    flat_cfg = DFlash2Qwen3Config.from_dict(flat)
    for key in (
        "num_speculative_tokens",
        "dflash2_conv_group_size",
        "dflash2_conv_kernel_size",
        "dflash2_selector_rank",
        "dflash2_selector_top_k",
    ):
        assert getattr(nested_cfg, key) == getattr(flat_cfg, key), key
    assert nested_cfg.num_speculative_tokens == 7
    assert nested_cfg.dflash2_conv_group_size == 16
    assert nested_cfg.dflash2_conv_kernel_size == 2
    assert nested_cfg.dflash2_selector_rank == 256
    assert nested_cfg.dflash2_selector_top_k == 16
    assert nested_cfg.sliding_window == flat_cfg.sliding_window == 2048
    assert nested_cfg.use_sliding_window == flat_cfg.use_sliding_window is True


def test_dflash2_flat_keys_take_precedence_over_nested_geometry() -> None:
    """A config carrying both forms must resolve to the flat top-level key."""
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Config

    base = _config_dict(model_type="qwen3", sliding_window=2048, use_sliding_window=True, hidden_size=16)
    cfg_dict = dict(
        base,
        dflash_config={
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 256,
            "selector_top_k": 16,
        },
        dflash2_block_size=4,
    )
    cfg_dict["use_sliding_window"] = False
    cfg_dict["dflash_config"].update(use_swa=True, swa_window_size=1024)
    cfg = DFlash2Qwen3Config.from_dict(cfg_dict)
    assert cfg.num_speculative_tokens == 3
    assert cfg.dflash2_selector_rank == 256
    assert cfg.sliding_window == 2048
    assert not cfg.use_sliding_window


def test_dflash2_non_dict_dflash_config_raises() -> None:
    """A malformed dflash_config (truthy non-dict) must fail loudly.

    ``or {}`` alone cannot recover from e.g. a string value; the explicit
    isinstance check turns a downstream AttributeError into a clear config
    error.
    """
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Config

    base = _config_dict(model_type="qwen3", sliding_window=2048, use_sliding_window=True, hidden_size=16)
    cfg_dict = dict(base, dflash_config="not-a-dict")
    with pytest.raises(ValueError, match="dflash_config must be a dict"):
        DFlash2Qwen3Config.from_dict(cfg_dict)


def test_dflash_mixed_windows_preserve_dflash2_block_attention() -> None:
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Config, DFlash2Qwen3Model

    values = _config_dict(
        num_hidden_layers=2,
        sliding_window=8,
        use_sliding_window=True,
        layer_types=["sliding_attention", "full_attention"],
    )
    cfg = DFlashQwen3Config.from_dict(values)
    cfg.validate()
    model = DFlashQwen3Model(cfg, torch.float32, torch.device("cpu"))
    sliding, full = (layer.self_attn.attn for layer in model.layers)
    assert sliding.causal and sliding.attention_window == (7, 0)
    assert not full.causal and full.attention_window is None

    values.update(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        dflash_config=dict(block_size=3, conv_group_size=2, conv_kernel_size=2, selector_rank=2, selector_top_k=2),
    )
    cfg2 = DFlash2Qwen3Config.from_dict(values)
    cfg2.validate()
    model2 = DFlash2Qwen3Model(cfg2, torch.float32, torch.device("cpu"))
    attention2 = model2.layers[0].self_attn.attn
    assert not attention2.causal and attention2.attention_window == (7, 2)

    values.update(num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"])
    with pytest.raises(ValueError, match="DFlash2 applies the block window to every layer"):
        DFlash2Qwen3Config.from_dict(values).validate()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({}, "use_sliding_window=true"),
        ({"use_sliding_window": False}, "use_sliding_window=true"),
    ],
)
def test_dflash_sliding_rejects_disabled_configuration(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(layer_types=["sliding_attention"], sliding_window=8, **overrides)


@pytest.mark.parametrize("dflash2", [False, True])
def test_sliding_draft_accepts_target_context_parallelism(dflash2: bool) -> None:
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Config

    values = _config_dict(
        hidden_size=16,
        cp_size=2,
        world_size=2,
        use_sliding_window=True,
        sliding_window=8,
        layer_types=["sliding_attention"],
        dflash2_block_size=4,
        dflash2_conv_group_size=16,
        dflash2_conv_kernel_size=2,
        dflash2_selector_rank=4,
        dflash2_selector_top_k=4,
    )
    config_cls = DFlash2Qwen3Config if dflash2 else DFlashQwen3Config
    config = config_cls.from_dict(values)
    config.validate()
    assert config.cp_size == 2
    assert config.layer_types == ("sliding_attention",)


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("full", True),
        ("full-cp2", True),
        ("scalar-window", False),
        ("causal", False),
        ("mixed", False),
        ("dflash2", True),
        ("dflash2-cp2", True),
    ],
)
def test_device_only_decode_admits_only_matching_attention(
    monkeypatch: pytest.MonkeyPatch, variant: str, expected: bool
) -> None:
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

    runtime = SimpleNamespace(is_ascend_a3=lambda: True)
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    config = _config_dict(
        hidden_size=256,
        head_dim=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_speculative_tokens=3,
    )
    model_cls = DFlashQwen3ForCausalLM
    if variant.startswith("dflash2"):
        model_cls = DFlash2Qwen3ForCausalLM
        config.update(
            sliding_window=8,
            use_sliding_window=True,
            dflash2_block_size=4,
            dflash2_conv_group_size=16,
            dflash2_conv_kernel_size=2,
            dflash2_selector_rank=4,
            dflash2_selector_top_k=4,
        )
    elif variant == "scalar-window":
        config["sliding_window"] = 8
    elif variant in ("causal", "mixed"):
        config.update(
            sliding_window=8,
            use_sliding_window=True,
            layer_types=["sliding_attention", "sliding_attention" if variant == "causal" else "full_attention"],
        )
    if variant.endswith("-cp2"):
        config.update(cp_size=2, world_size=2)
    model = model_cls(config)
    assert model.supports_device_only_decode_inputs(128) is expected
    assert not model.supports_device_only_decode_inputs(64)
    runtime.is_ascend_a3 = lambda: False
    assert not model.supports_device_only_decode_inputs(128)
