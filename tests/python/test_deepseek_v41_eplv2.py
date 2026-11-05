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

"""V4.1 source ownership, global transport admission and unchanged decode."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xllm.python import distributed, kernels
from xllm.python.layers.sequence_parallel import sp_shard
from xllm.python.layers.sequence_parallel_metadata import EpSpTokenLayout, SpMetadata
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41W8A8MoE,
    _eplv2_prefill_step,
    _validate_eplv2,
)


def _config(rank: int = 0) -> DeepseekV41Config:
    dp, tp = divmod(rank, 2)
    return DeepseekV41Config(
        expert_parallel_degree=2,
        tp_size=2,
        tp_rank=tp,
        dp_size=2,
        dp_rank=dp,
        ep_size=4,
        ep_rank=rank,
        n_routed_experts=4,
    )


def _context(raw: tuple[int, int], flags: tuple[int, int]) -> ForwardContext:
    metadata = SimpleNamespace(
        raw_dp_execution_token_counts=raw,
        dp_execution_token_counts=tuple(max(1, n) for n in raw),
        dp_is_decode=flags,
        is_spec_verify=False,
    )
    return ForwardContext(None, torch.device("cpu"), metadata, [])


@pytest.mark.parametrize(
    "changes",
    [
        dict(cp_size=2),
        dict(moe_tp_size=2),
        dict(ep_size=1),
        dict(ep_rank=1),
        dict(n_routed_experts=5),
        dict(expert_parallel_degree=3),
    ],
)
def test_unsupported_eplv2_fails_before_collectives(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="V4.1"):
        _validate_eplv2(replace(_config(), **changes), "ascend")


def test_eplv2_requires_w8a8_and_static_experts() -> None:
    _validate_eplv2(_config(), "ascend")
    _validate_eplv2(replace(_config(), enable_dsa_cp=True), "ascend")
    for mode, eplb in (("none", False), ("fp8", False), ("ascend", True)):
        with pytest.raises(ValueError, match="V4.1 EPLv2"):
            _validate_eplv2(_config(), mode, eplb)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize(
    "raw,flags,expected",
    [
        ((7, 0), (0, 1), True),
        ((0, 7), (1, 0), True),
        ((7, 1), (0, 1), True),
        ((1, 7), (1, 0), True),
        ((1, 0), (1, 0), False),
        ((0, 0), (0, 0), False),
        ((1, 1), (1, 1), False),
    ],
)
def test_transport_decision_matches_on_all_peers(
    rank: int, raw: tuple[int, int], flags: tuple[int, int], expected: bool
) -> None:
    context = _context(raw, flags)
    with forward_context(context):
        assert _eplv2_prefill_step(_config(rank)) == expected
    with forward_context(replace(context, acl_graph=object())):
        assert not _eplv2_prefill_step(_config(rank))


@pytest.mark.parametrize("raw", [(7, 3), (7, 0), (0, 7), (1, 1)])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("fc1", [False, True])
@pytest.mark.parametrize("mega_capacity", [0, 1, 8])
@pytest.mark.parametrize("execution_padding", [0, 5])
def test_source_rows_are_dispatched_once_and_shared_expert_is_not_reduced(
    raw: tuple[int, int],
    rank: int,
    fc1: bool,
    mega_capacity: int,
    execution_padding: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(rank)
    context = _context(raw, (0, 0))
    context.metadata.dp_execution_token_counts = tuple(max(1, n) + execution_padding for n in raw)
    dispatch_rows = (max(context.metadata.dp_execution_token_counts) + cfg.tp_size - 1) // cfg.tp_size
    rows = context.metadata.dp_execution_token_counts[cfg.dp_rank]
    hidden = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) / 32 + 1
    ep = EpSpTokenLayout.from_token_counts(
        context.metadata.dp_execution_token_counts,
        raw,
        tp_size=2,
        tp_rank=cfg.tp_rank,
        dp_size=2,
        dp_rank=cfg.dp_rank,
        compacted=True,
    )
    layout = SpMetadata.from_execution_rows(rows, 2, cfg.tp_rank, ep_layout=ep)
    source = sp_shard(hidden, layout)
    module = DeepseekV41W8A8MoE.__new__(DeepseekV41W8A8MoE)
    nn.Module.__init__(module)
    module.cfg = cfg
    module.layer_id = 0
    module.num_total_experts = 4
    for name in ("experts_w13", "experts_w2", "experts_w13_scale", "experts_w2_scale"):
        setattr(module, name, torch.empty(0))
    module._eplv2_mega_moe_enabled = mega_capacity > 0
    module._mega_moe_num_max_tokens_per_rank = mega_capacity
    module._mega_moe_ccl_buffer_size = 8192
    module.mega_moe_context = torch.zeros(8, dtype=torch.int32)
    module.mega_moe_w13_scale = torch.ones(1, dtype=torch.int64)
    module.mega_moe_w2_scale = torch.ones(1, dtype=torch.int64)
    module.shared_expert_gate_is_loaded = False
    module._run_shared_experts = lambda value: value * 0.125

    def route(value: torch.Tensor, norm: nn.Module | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert norm is not None
        normalized = norm(value)
        weights = torch.tensor([0.25, 0.75]).expand(value.shape[0], 2)
        indices = torch.tensor([0, 3], dtype=torch.int32).expand(value.shape[0], 2)
        return normalized, weights, indices

    module._w8a8_bias_topk = route
    calls = []

    def experts(
        value: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        calls.append(value)
        torch.testing.assert_close(value, source * 2)
        # Remote experts must retain their real router weights until dispatch.
        torch.testing.assert_close(weights[:, 1], torch.full((value.shape[0],), 0.75))
        assert indices[:, 1].eq(3).all()
        mask = kwargs["x_active_mask"]
        assert mask.tolist() == [i < ep.active_rows for i in range(layout.local_rows)]
        return value * (0.25 + 0.75 * 4)

    monkeypatch.setattr(kernels, "ep_moe_w8a8_alltoall", experts, raising=False)

    def mega_experts(
        context: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        assert context is module.mega_moe_context
        assert dispatch_rows <= mega_capacity
        assert value.shape[0] == dispatch_rows
        calls.append(value)
        torch.testing.assert_close(value[: source.shape[0]], source * 2)
        expected_mask = [i < ep.active_rows for i in range(dispatch_rows)]
        assert kwargs["x_active_mask"].tolist() == expected_mask
        return value * (0.25 + 0.75 * 4)

    monkeypatch.setattr(kernels, "mega_moe", mega_experts, raising=False)
    expected = hidden * 2 * (0.25 + 0.75 * 4 + 0.125)
    expected[raw[cfg.dp_rank] :] = 0
    # Restore the other TP owner's independently computed source rows.
    padded = torch.zeros(layout.padded_rows, 3)
    padded[:rows] = expected
    monkeypatch.setattr(distributed, "tp_all_gather", lambda *args: padded, raising=False)

    class _Norm(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * 2

    with forward_context(context):
        output = module._forward_eplv2(source if fc1 else hidden, _Norm(), layout if fc1 else None)
    assert len(calls) == 1
    torch.testing.assert_close(output, sp_shard(expected, layout) if fc1 else expected, rtol=0, atol=0)
