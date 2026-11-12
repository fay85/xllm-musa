# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Guard against narrowing routed activations before route-weight scaling."""

import sys
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from xllm.python.kernels_musa.shared_swiglu import select_shared_swiglu
from xllm.python.models.deepseek_v41 import _swiglu_clamped

# Selection only reads metadata. CPU CI need not import/register TorchMUSA.
_MUSA_DEVICE = cast(torch.device, SimpleNamespace(type="musa"))


def test_reference_does_not_require_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLLM_DSV41_MUSA_SHARED_SWIGLU", "reference")
    monkeypatch.delitem(sys.modules, "xllm_runtime", raising=False)
    assert select_shared_swiglu(_MUSA_DEVICE, torch.bfloat16, 10.0) is None


@pytest.mark.parametrize("mode", ["auto", "typo", ""])
def test_invalid_mode_is_not_silent_fallback(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLLM_DSV41_MUSA_SHARED_SWIGLU", mode)
    with pytest.raises(ValueError, match="reference or v4_fused"):
        select_shared_swiglu(_MUSA_DEVICE, torch.bfloat16, 10.0)


@pytest.mark.parametrize("dtype,limit", [(torch.float16, 10.0), (torch.bfloat16, 0.0), (torch.bfloat16, float("nan"))])
def test_unsupported_contract_fails_at_construction(
    dtype: torch.dtype, limit: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XLLM_DSV41_MUSA_SHARED_SWIGLU", "v4_fused")
    with pytest.raises(ValueError, match="requires BF16"):
        select_shared_swiglu(_MUSA_DEVICE, dtype, limit)


def test_stale_binary_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLLM_DSV41_MUSA_SHARED_SWIGLU", "v4_fused")
    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace())
    with pytest.raises(RuntimeError, match="Rebuild the MUSA xllm binary"):
        select_shared_swiglu(_MUSA_DEVICE, torch.bfloat16, 10.0)


def test_routing_weight_is_applied_before_bf16_rounding() -> None:
    torch.manual_seed(1109)
    gate = torch.randn(17, 288).to(torch.bfloat16)
    up = torch.randn_like(gate)
    weight = torch.rand(17)
    expected = (torch.nn.functional.silu(gate.float()) * up.float() * weight[:, None]).bfloat16()
    actual = _swiglu_clamped(gate, gate, up, torch.nn.Identity(), 10.0, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    prematurely_rounded = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
    assert torch.any((prematurely_rounded.float() * weight[:, None]).bfloat16() != actual)
    with pytest.raises(ValueError, match="routing-weight"):
        _swiglu_clamped(
            gate, gate, up, torch.nn.Identity(), 10.0, weight,
            shared_activation=lambda g, u: torch.empty_like(g),
        )
