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

"""MC2 overlaps the complete shared MLP with one fork/join dependency."""

from typing import Any

import pytest
import torch

from xllm.python.model_executor.forward_context import forward_context
from xllm.python.models import deepseek_v41 as v41

from .test_deepseek_v41_mc2 import (
    _context,
    _install_mc2_streams,
    _make_moe,
    _StreamTestTensor,
    mc2,
)


@pytest.mark.usefixtures("mc2")
@pytest.mark.parametrize("overlap", [False, True])
def test_mc2_shared_mlp_is_submitted_once_before_dispatch_and_joined_after_combine(
    monkeypatch: pytest.MonkeyPatch, overlap: bool
) -> None:
    events: list[str] = []
    streams = _install_mc2_streams(monkeypatch, events)
    hidden = torch.randn(3, 4, dtype=torch.bfloat16).as_subclass(_StreamTestTensor)
    moe = _make_moe(dp_size=4)
    moe.cfg.eplv2_shared_overlap = overlap
    dispatch = v41._moe_mc2_dispatch
    combine = v41._moe_mc2_combine

    def record_stream(value: torch.Tensor, stream: object) -> None:
        assert stream is (streams.side if value is hidden else streams.current)
        events.append("retain_input" if value is hidden else "retain_output")

    def shared(value: torch.Tensor) -> torch.Tensor:
        assert value is hidden
        events.append("shared_mlp")
        return torch.full_like(value, 0.25)

    def routed_dispatch(**kwargs: Any) -> tuple:
        events.append("dispatch")
        return dispatch(**kwargs)

    def routed_combine(**kwargs: Any) -> torch.Tensor:
        events.append("combine")
        return combine(**kwargs)

    monkeypatch.setattr(_StreamTestTensor, "record_stream", record_stream)
    monkeypatch.setattr(moe, "_run_shared_experts", shared)
    monkeypatch.setattr(v41, "_moe_mc2_dispatch", routed_dispatch)
    monkeypatch.setattr(v41, "_moe_mc2_combine", routed_combine)
    with forward_context(_context(token_counts=[3] * 4, dp_is_decode=[1] * 4)):
        output = moe._forward_mc2_decode(hidden)

    assert events == (
        ["wait_input", "retain_input", "shared_mlp", "dispatch", "combine", "join_shared", "retain_output"]
        if overlap
        else ["dispatch", "combine", "shared_mlp"]
    )
    torch.testing.assert_close(output, torch.full_like(output, 0.75), rtol=0, atol=0)
