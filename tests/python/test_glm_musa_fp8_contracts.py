# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Checkpoint preservation and expert/shared reduction contract regressions."""

import pytest
import torch
from torch import nn

from xllm.python.models.glm5_next import Glm5NextConfig, Glm5NextMoE
from xllm.python.models.glm5_next_weight import W8A8WeightLoader


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def has(self, key: str) -> bool:
        return key in self.tensors

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]


def test_raw_checkpoint_path_preserves_fp8_bytes_and_multiplicative_scales() -> None:
    weight = torch.arange(256 * 128).reshape(256, 128).remainder(63).float().to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.25], [2.0]], dtype=torch.float32)
    state = _StateDict({"expert.weight": weight, "expert.weight_scale_inv": scale})
    loader = W8A8WeightLoader(nn.Module(), [state], 1, 0)
    actual, actual_scale = loader.load_raw_fp8("expert.weight")
    assert actual.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(actual.view(torch.uint8), weight.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, scale, rtol=0, atol=0)
    reference = weight.float() * scale.repeat_interleave(128, 0)
    torch.testing.assert_close(loader.load_tensor("expert.weight"), reference.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("bad_scale", [float("nan"), -1.0])
def test_raw_checkpoint_rejects_invalid_scale(bad_scale: float) -> None:
    state = _StateDict(
        {"x.weight": torch.zeros(128, 128).to(torch.float8_e4m3fn), "x.weight_scale_inv": torch.tensor([[bad_scale]])}
    )
    with pytest.raises(ValueError, match="scale values"):
        W8A8WeightLoader(nn.Module(), [state], 1, 0).load_raw_fp8("x.weight")


@pytest.mark.parametrize(
    "overrides",
    [{"ep_size": 4}, {"moe_tp_size": 2}, {"cp_size": 2}, {"hidden_size": 128}, {"musa_mega_moe_max_tokens": 128}],
)
def test_mega_moe_rejects_inconsistent_parallel_axes(overrides: dict[str, int]) -> None:
    cfg = dict(tp_size=8, ep_size=8, moe_tp_size=1, musa_moe_backend="mega_moe")
    with pytest.raises(ValueError, match="MUSA.*MegaMoE"):
        Glm5NextConfig.from_dict(cfg | overrides)


class _Routed(nn.Module):
    def forward(self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return hidden * 3


class _Shared(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 2


def test_complete_routed_result_is_not_all_reduced_again(monkeypatch: pytest.MonkeyPatch) -> None:
    # CPU stand-ins test orchestration only; distributed GPU numerical tests
    # independently validate the actual MATE path and shared collective.
    cfg = Glm5NextConfig(
        hidden_size=8, n_routed_experts=16, num_experts_per_tok=8, moe_intermediate_size=16, tp_size=2, moe_tp_size=2
    )
    layer = Glm5NextMoE(cfg, torch.bfloat16, torch.device("cpu"))
    cfg.musa_moe_backend = "mega_moe"
    layer.musa_mega_experts = _Routed()
    layer.shared_experts = _Shared()
    reductions = []

    def _all_reduce(value: torch.Tensor) -> None:
        reductions.append(value.clone())
        value.mul_(2)

    monkeypatch.setattr("xllm.python.models.glm5_next.distributed.all_reduce_", _all_reduce)
    hidden = torch.arange(16).reshape(2, 8).bfloat16()
    actual = layer(hidden)
    torch.testing.assert_close(actual, hidden * 7, rtol=0, atol=0)
    assert len(reductions) == 1
    torch.testing.assert_close(reductions[0], hidden * 2, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides", [{"musa_kda_prefill_mode": "unknown"}, {"musa_tp_reduce_precision": "unknown"}]
)
def test_invalid_numerical_modes_fail_before_model_execution(overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="MUSA"):
        Glm5NextConfig.from_dict(overrides)


def test_explicit_numerical_mode_config_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLLM_MUSA_GLM_KDA_PREFILL_MODE", "recurrent")
    monkeypatch.setenv("XLLM_MUSA_GLM_TP_REDUCE_PRECISION", "fp32")
    selected = Glm5NextConfig.from_dict({})
    assert (selected.musa_kda_prefill_mode, selected.musa_tp_reduce_precision) == ("recurrent", "fp32")
    overridden = Glm5NextConfig.from_dict({"musa_kda_prefill_mode": "chunk", "musa_tp_reduce_precision": "native"})
    assert (overridden.musa_kda_prefill_mode, overridden.musa_tp_reduce_precision) == ("chunk", "native")
