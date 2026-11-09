# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Staged MC2 shared-expert dependencies and lifetime ownership."""

from types import SimpleNamespace

import pytest
import torch

from tests.python.test_deepseek_v4_shared_expert import _install_fake_npu, _owner
from xllm.python.models import deepseek_v41 as v41


def test_staged_shared_events_fence_vector_and_cube_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_npu(monkeypatch)
    monkeypatch.setattr(v41, "shared_expert_stream", lambda device: fake.side)
    phases: list[str] = []
    hidden = torch.randn(8, 4)

    def quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phases.append("quantize")
        return value, torch.ones(value.shape[0])

    def gate_up(value: torch.Tensor) -> torch.Tensor:
        phases.append("gate_up")
        return value * 2

    def activate(value: torch.Tensor, scale: torch.Tensor, *, clamp_limit: float) -> tuple[torch.Tensor, torch.Tensor]:
        phases.append("activate")
        assert clamp_limit == 10.0
        return value.clamp_max(clamp_limit), scale

    def down(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        phases.append("down")
        return value + 1

    monkeypatch.setattr(v41.kernels, "dynamic_quant", quantize, raising=False)
    owner = _owner(
        cfg=SimpleNamespace(swiglu_limit=10.0),
        shared_experts=SimpleNamespace(
            gate_up_proj=SimpleNamespace(forward_accumulated=gate_up),
            activate_quantized_gate_up=activate,
            down_proj=SimpleNamespace(forward_quantized=down),
        ),
        shared_expert_gate_is_loaded=False,
    )
    state = v41.DeepseekV41W8A8MoE._prepare_mc2_shared_expert(owner, hidden)
    assert phases == ["quantize"]
    v41.DeepseekV41W8A8MoE._project_mc2_shared_gate_up(owner, state)
    assert phases == ["quantize", "gate_up"]
    v41.DeepseekV41W8A8MoE._activate_mc2_shared_expert(owner, state)
    result = v41.DeepseekV41W8A8MoE._project_mc2_shared_down(owner, state)
    assert phases == ["quantize", "gate_up", "activate", "down"]
    assert fake.side.waited == [
        owner._mc2_shared_input_ready,
        owner._mc2_shared_router_ready,
        owner._mc2_shared_dispatch_done,
        owner._mc2_shared_combine_start,
    ]
    assert all(event.recorded_on is fake.current for event in fake.side.waited)
    assert owner._shared_expert_done_event.recorded_on is fake.side
    assert state.quantized_input is hidden
    assert state.gate_up is not None and state.activation is not None
    assert result[1] is hidden
    torch.testing.assert_close(result[0], (hidden * 2).clamp_max(10.0) + 1)
