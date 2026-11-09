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

"""Shared-expert overlap launch/join dependencies and input ownership.

Pure-Python: fakes the NPU stream surface so the side-stream protocol can be
checked without an accelerator.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from xllm.python.models import deepseek_v4 as v4
from xllm.python.models import deepseek_v41 as v41


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded_on: object | None = None

    def record(self, stream: object | None = None) -> None:
        self.recorded_on = stream


class _FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited: list[object] = []

    def wait_event(self, event: _FakeEvent) -> None:
        self.waited.append(event)


class _FakeNpu:
    def __init__(self) -> None:
        self.current = _FakeStream("current")
        self.side = _FakeStream("side")

    def Event(self) -> _FakeEvent:
        return _FakeEvent()

    def current_stream(self, device: object | None = None) -> _FakeStream:
        return self.current

    def stream(self, stream: object) -> object:
        return contextlib.nullcontext()


def _install_fake_npu(monkeypatch: pytest.MonkeyPatch) -> _FakeNpu:
    fake = _FakeNpu()
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    monkeypatch.setattr(v4, "shared_expert_stream", lambda device: fake.side)
    return fake


def _owner(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "_shared_expert_start_event": None,
        "_shared_expert_done_event": None,
        "_shared_expert_graph_logged": False,
        "_run_gated_shared_experts": lambda x: x * 2,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("model_type", ["v4", "v41"])
@pytest.mark.parametrize("device_type", ["cpu", "npu", "privateuseone"])
def test_overlap_is_selected_at_construction_without_config(
    monkeypatch: pytest.MonkeyPatch, model_type: str, device_type: str
) -> None:
    """Exercise the constructor's default without allocating NPU tensors."""
    empty = torch.empty
    linear = torch.nn.Linear

    def host_empty(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs["device"] = torch.device("cpu")
        return empty(*args, **kwargs)

    def host_linear(*args: object, **kwargs: object) -> torch.nn.Module:
        kwargs["device"] = torch.device("cpu")
        return linear(*args, **kwargs)

    def shared_mlp(*args: object, **kwargs: object) -> torch.nn.Module:
        return torch.nn.Identity()

    monkeypatch.setattr(torch, "empty", host_empty)
    monkeypatch.setattr(torch.nn, "Linear", host_linear)
    cls = v4.DeepseekV4MoE if model_type == "v4" else v41.DeepseekV41W8A8MoE
    cfg_cls = v4.DeepseekV4Config if model_type == "v4" else v41.DeepseekV41Config
    monkeypatch.setattr(cls, "_SHARED_MLP_CLS", staticmethod(shared_mlp))
    cfg = cfg_cls(hidden_size=4, n_routed_experts=4, moe_intermediate_size=4, n_hash_layers=0)
    assert "enable_shared_expert_overlap" not in cfg.__dataclass_fields__
    moe = cls(cfg, 0, torch.bfloat16, SimpleNamespace(type=device_type))
    assert moe._shared_expert_overlap_enabled is (device_type != "cpu")


def test_mx_shared_expert_keeps_its_serial_path(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = torch.empty
    linear = torch.nn.Linear

    def host_empty(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs["device"] = torch.device("cpu")
        return empty(*args, **kwargs)

    def host_linear(*args: object, **kwargs: object) -> torch.nn.Module:
        kwargs["device"] = torch.device("cpu")
        return linear(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)
    monkeypatch.setattr(torch.nn, "Linear", host_linear)
    monkeypatch.setattr(v4, "NpuDeepseekV4MLPMXFP8", lambda *args, **kwargs: torch.nn.Identity())
    cfg = v4.DeepseekV4Config(
        hidden_size=32, n_routed_experts=4, moe_intermediate_size=32, n_hash_layers=0, weight_quant_method="fp8"
    )
    moe = v4.DeepseekV4MoE(cfg, 0, torch.bfloat16, SimpleNamespace(type="npu"))
    assert not moe._shared_expert_overlap_enabled


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_overlap_launch_join_matches_the_synchronous_path(monkeypatch: pytest.MonkeyPatch, noncontiguous: bool) -> None:
    fake = _install_fake_npu(monkeypatch)
    hidden = torch.randn(18, 3)[::2] if noncontiguous else torch.randn(9, 3)
    owner = _owner()

    state = v4.DeepseekV4MoE._launch_shared_expert_overlap(owner, hidden)
    out = v4.DeepseekV4MoE._finish_shared_expert_overlap(owner, state)

    assert state[1] is hidden
    assert owner._shared_expert_start_event.recorded_on is fake.current
    assert fake.side.waited == [owner._shared_expert_start_event]
    assert owner._shared_expert_done_event.recorded_on is fake.side
    assert fake.current.waited == [owner._shared_expert_done_event]
    torch.testing.assert_close(out, hidden * 2, rtol=0, atol=0)


def test_overlap_events_are_reused_after_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_npu(monkeypatch)
    hidden = torch.randn(9, 3)
    owner = _owner()
    state = v4.DeepseekV4MoE._launch_shared_expert_overlap(owner, hidden)
    v4.DeepseekV4MoE._finish_shared_expert_overlap(owner, state)
    start = owner._shared_expert_start_event
    done = owner._shared_expert_done_event

    state = v4.DeepseekV4MoE._launch_shared_expert_overlap(owner, hidden)
    v4.DeepseekV4MoE._finish_shared_expert_overlap(owner, state)

    assert owner._shared_expert_start_event is start
    assert owner._shared_expert_done_event is done
