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

"""Admission and unique EP token ownership, including idle and SWA layouts."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from xllm.python import distributed, kernels
from xllm.python.layers.sequence_parallel_metadata import build_flashcomm1_layout
from xllm.python.model_executor.forward_context import ForwardContext
from xllm.python.models.deepseek_v4 import DeepseekV4MoE
from xllm.python.models.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41W8A8MoE,
    _validate_flashcomm1,
)


def _config(**changes: object) -> DeepseekV41Config:
    return replace(
        DeepseekV41Config(),
        **dict(
            enable_flashcomm1=True,
            flashcomm1_min_prefill_tokens=4,
            tp_size=2,
            dp_size=2,
            ep_size=4,
            moe_tp_size=1,
            **changes,
        ),
    )


def _context(counts: tuple[int, ...], raw: tuple[int, ...], flags: tuple[int, ...]) -> ForwardContext:
    metadata = SimpleNamespace(
        dp_execution_token_counts=counts,
        raw_dp_execution_token_counts=raw,
        dp_is_decode=flags,
        is_spec_verify=False,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    return ForwardContext(None, torch.device("cpu"), metadata, [])


def test_eager_admission_is_global() -> None:
    cfg = _config()
    _validate_flashcomm1(cfg, "ascend")
    # C++ reflects frequency=0 for this checkpoint; the decoder treats 0/1
    # identically as an all-MoE stack.
    _validate_flashcomm1(replace(cfg, moe_layer_freq=0), "ascend")
    # Both the active DP0 and idle DP1 must enter the EP collectives.
    context = _context((7, 1), (7, 0), (0, 1))
    assert build_flashcomm1_layout(cfg, context, 7) is not None
    idle = replace(cfg, dp_rank=1, ep_rank=2)
    layout = build_flashcomm1_layout(idle, context, 1)
    assert layout is not None and layout.ep_layout.active_rows == 0
    for state in (dict(acl_graph=object()), dict(execution_state=object())):
        assert build_flashcomm1_layout(cfg, replace(context, **state), 7) is None
    assert build_flashcomm1_layout(cfg, _context((7, 1), (7, 0), (1, 1)), 7) is None
    # SWA consumers use their new row counts, never the source's 2048-row layout.
    assert build_flashcomm1_layout(cfg, _context((2, 1), (2, 0), (0, 1)), 2) is None
    context.metadata.is_spec_verify = True
    assert build_flashcomm1_layout(cfg, context, 7) is None
    with pytest.raises(ValueError, match="execution rows"):
        build_flashcomm1_layout(cfg, _context((7, 1), (7, 0), (0, 1)), 6)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("capture", [False, True])
def test_decode_graph_admission_keeps_uniform_capacity(rank: int, capture: bool) -> None:
    dp, tp = divmod(rank, 2)
    cfg = _config(dp_rank=dp, tp_rank=tp, ep_rank=rank)
    context = _context((8, 8), (), (1, 1))
    context.metadata.is_spec_verify = True
    context = replace(context, **({"acl_graph": object()} if capture else {"execution_state": object()}))
    layout = build_flashcomm1_layout(cfg, context, 8)
    assert layout is not None and layout.local_rows == 4
    assert layout.ep_layout.dispatch_counts == (4, 4, 4, 4)
    assert not layout.ep_layout.compacted
    for counts in [(8, 1), (7, 7), (0, 0)]:
        context.metadata.dp_execution_token_counts = counts
        assert build_flashcomm1_layout(cfg, context, counts[dp]) is None


@pytest.mark.parametrize("rank", range(4))
def test_draft_block_admission_preserves_uneven_capacities(rank: int) -> None:
    dp, tp = divmod(rank, 2)
    cfg = _config(dp_rank=dp, tp_rank=tp, ep_rank=rank)
    context = _context((5, 1), (5, 0), (1, 1))
    context.metadata.is_dflash_proposal = True
    layout = build_flashcomm1_layout(cfg, context, (5, 1)[dp])
    assert layout is not None and layout.local_rows == (3, 1)[dp]
    assert layout.ep_layout.dispatch_counts == (3, 3, 1, 1)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize(
    "raw,flags,admitted",
    [
        ((8192, 1), (0, 1), False),
        ((1, 8192), (1, 0), False),
        ((8192, 3), (0, 0), True),
        ((8192, 0), (0, 0), True),
        ((8192, 0), (0, 1), True),
        ((0, 8192), (0, 0), True),
        ((0, 8192), (1, 0), True),
        ((128, 0), (1, 0), False),
        ((0, 128), (0, 1), False),
        ((128, 1), (1, 1), False),
        ((0, 0), (0, 0), False),
        ((0, 0), (1, 1), False),
    ],
)
def test_only_active_prefill_replicas_admit_fc1(
    rank: int, raw: tuple[int, ...], flags: tuple[int, ...], admitted: bool
) -> None:
    """All EP peers agree despite mixed traffic and arbitrary idle flags."""
    dp, tp = divmod(rank, 2)
    cfg = _config(dp_rank=dp, tp_rank=tp, ep_rank=rank)
    counts = tuple(max(1, rows) for rows in raw)
    context = _context(counts, raw, flags)
    # Local forward types differ in mixed steps and on dummy replicas.
    context.metadata.is_prefill = flags[dp] == 0
    layout = build_flashcomm1_layout(cfg, context, counts[dp])
    assert (layout is not None) == admitted
    if layout is not None and raw[dp] == 0:
        assert layout.ep_layout.active_rows == 0


@pytest.mark.parametrize("flags", [(), (0,), (0, 0, 0)])
def test_dp_admission_requires_complete_global_flags(flags: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="decode flag"):
        build_flashcomm1_layout(_config(), _context((7, 1), (7, 0), flags), 7)


@pytest.mark.parametrize("prefill", [False, True])
def test_single_replica_without_flags_uses_local_forward_type(prefill: bool) -> None:
    cfg = replace(_config(), dp_size=1, ep_size=2)
    context = _context((7,), (7,), ())
    context.metadata.is_chunked_prefill = prefill
    assert (build_flashcomm1_layout(cfg, context, 7) is not None) == prefill


@pytest.mark.parametrize(
    "changes",
    [
        dict(tp_size=1),
        dict(cp_size=2),
        dict(moe_tp_size=2),
        dict(ep_size=2),
        dict(ep_rank=1),
        dict(first_k_dense_replace=1),
        dict(moe_layer_freq=2),
        dict(flashcomm1_min_prefill_tokens=1),
    ],
)
def test_unsupported_topology_disables_flashcomm1(changes: dict[str, object]) -> None:
    cfg = replace(_config(), **changes)
    _validate_flashcomm1(cfg, "ascend")
    assert not cfg.enable_flashcomm1
    assert build_flashcomm1_layout(cfg, _context((7, 1), (7, 0), (0, 1)), 7) is None
    cfg = _config()
    _validate_flashcomm1(cfg, "none")
    assert not cfg.enable_flashcomm1


def test_dsa_cp_preserves_flashcomm1_admission() -> None:
    cfg = replace(_config(), enable_dsa_cp=True)
    _validate_flashcomm1(cfg, "ascend")
    assert cfg.enable_flashcomm1
    assert build_flashcomm1_layout(cfg, _context((7, 1), (7, 0), (0, 1)), 7) is not None


@pytest.mark.parametrize("raw", [(7, 3), (7, 0), (1, 7), (129, 4)])
@pytest.mark.parametrize("rank", range(4))
def test_ep_experts_sum_to_token_owners_without_padding_leaks(
    raw: tuple[int, ...], rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real EP packing/restoration and V4.1 shared-expert boundary.

    Simulate each rank's linear expert contribution, and compare the reduced
    result with the full-token reference. Uneven and empty sources exercise
    compact gather offsets and the padded reduce-scatter unpacking.
    """
    dp, tp = divmod(rank, 2)
    cfg = replace(_config(), dp_rank=dp, tp_rank=tp, ep_rank=rank)
    counts = tuple(max(1, n) for n in raw)
    layout = build_flashcomm1_layout(cfg, _context(counts, raw, (0, 0)), counts[dp])
    assert layout is not None
    inputs = [torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 32 + 1 for n in raw]
    sources = []
    for source_rank in range(4):
        source_dp, source_tp = divmod(source_rank, 2)
        capacity = (counts[source_dp] + 1) // 2
        source = inputs[source_dp][source_tp * capacity : (source_tp + 1) * capacity]
        sources.append(source if source.shape[0] else torch.zeros(1, 3))
    gathered = torch.cat(sources)
    local = torch.full((layout.local_rows, 3), 12345.0)
    active = layout.ep_layout.active_rows
    local[:active] = sources[rank][:active]

    def gather_variable(value: torch.Tensor, dispatch_counts: list[int], ep_rank: int, group: str) -> torch.Tensor:
        assert group == "moe_ep" and ep_rank == rank
        assert dispatch_counts == list(layout.ep_layout.dispatch_counts)
        torch.testing.assert_close(value, sources[rank])
        return gathered.clone()

    def reduce_scatter(value: torch.Tensor, size: int, group: str) -> torch.Tensor:
        assert size == 4 and group == "moe_ep"
        # Each expert rank contributes (rank+1)*x; sum over all four experts.
        return value.chunk(size)[rank] * (10 / (rank + 1))

    monkeypatch.setattr(distributed, "all_gather_variable", gather_variable, raising=False)
    monkeypatch.setattr(
        distributed,
        "all_gather",
        lambda value, **kwargs: gather_variable(
            value, list(layout.ep_layout.dispatch_counts), rank, kwargs["group_name"]
        ),
        raising=False,
    )
    monkeypatch.setattr(distributed, "reduce_scatter", reduce_scatter, raising=False)
    module = object.__new__(DeepseekV41W8A8MoE)
    nn.Module.__init__(module)
    module.shared_expert_gate_is_loaded = False

    def experts(self: nn.Module, value: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        return norm(value) * (rank + 1)

    module._forward_flashcomm1_routed = MethodType(experts, module)
    module._run_shared_experts = lambda value: value * 3
    result = module._forward_flashcomm1(local, nn.Identity(), layout)
    expected = torch.zeros_like(local)
    expected[:active] = sources[rank][:active] * 13
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("with_norm", [False, True])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("ep_rank", [0, 1])
@torch.inference_mode()
def test_v41_local_experts_match_unchanged_v4_path(
    with_norm: bool, scoring: str, ep_rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the original gate/GMM contract after removing the V4 refactor."""
    cfg = replace(
        DeepseekV41Config(),
        hidden_size=8,
        moe_intermediate_size=8,
        n_routed_experts=4,
        n_activated_experts=2,
        n_hash_layers=0,
        tp_size=2,
        tp_rank=ep_rank,
        ep_size=2,
        ep_rank=ep_rank,
        moe_tp_size=1,
        scoring_func=scoring,
    )
    module = DeepseekV41W8A8MoE(cfg, 0, torch.bfloat16, torch.device("cpu"))
    module.e_score_correction_bias.data.zero_()
    module._run_shared_experts = torch.zeros_like
    gate_calls: list[dict[str, object]] = []
    expert_calls: list[tuple[object, ...]] = []

    def gate(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, None]:
        gate_calls.append(kwargs)
        scores = kwargs["x"].float().softmax(dim=-1)
        weights, indices = scores.topk(2, dim=-1)
        return weights, indices, None

    def experts(*args: object) -> torch.Tensor:
        expert_calls.append(args)
        hidden, weights = args[:2]
        return hidden * weights.sum(-1, keepdim=True).to(hidden.dtype)

    monkeypatch.setattr(kernels, "moe_gating_top_k_hash", gate, raising=False)
    monkeypatch.setattr(kernels, "grouped_moe_with_selected_experts", experts, raising=False)
    monkeypatch.setattr(distributed, "moe_ep_all_reduce", lambda value: None, raising=False)
    hidden = torch.arange(40, dtype=torch.bfloat16).reshape(5, 8) / 32
    norm = nn.LayerNorm(8, dtype=torch.bfloat16) if with_norm else None
    reference = DeepseekV4MoE.forward(module, hidden, post_norm=norm)
    result = module._forward_flashcomm1_routed(hidden, norm)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert len(gate_calls) == len(expert_calls) == 2
    assert gate_calls[0].keys() == gate_calls[1].keys()
    for key, value in gate_calls[0].items():
        other = gate_calls[1][key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        else:
            assert value == other
    for value, other in zip(*expert_calls):
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        else:
            assert value == other
