# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Bitwise codes/scales, ACL graph replay, and JIT reuse for fused indexer INT8."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture(scope="module")
def indexer_quant() -> ModuleType:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for Triton indexer quantization")
    source = Path(__file__).resolve().parents[2] / "xllm/python/kernels_npu/triton/indexer_quant.py"
    spec = importlib.util.spec_from_file_location("indexer_quant_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    torch.npu.set_deterministic_level(1)
    return module


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = x.float().abs().amax(dim=-1)
    scale = (maximum / 127.0).half().float().clamp_min(2.0**-24)
    codes = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return codes, scale.half()


@pytest.mark.parametrize(
    "shape",
    [
        (0, 128),
        (1, 128),
        (7, 128),
        (1, 32, 128),
        (6, 32, 128),
        (17, 32, 128),
        (3, 7, 128),
        (2, 1, 128),
        (1024, 32, 128),
    ],
)
def test_indexer_quant_live_graph_is_bitwise_exact(indexer_quant: ModuleType, shape: tuple[int, ...]) -> None:
    quantize = indexer_quant.quantize_indexer_rows_int8
    torch.manual_seed(42)
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    for _ in range(3):
        quantize(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = quantize(x)
    for kind in ["random", "zero", "tiny", "halfway", "new_input"]:
        x.normal_()
        if kind == "zero":
            x.zero_()
        elif kind == "tiny":
            x.mul_(1e-9)
        elif kind == "halfway":
            pattern = torch.tensor(
                [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 127], device="npu", dtype=torch.bfloat16
            ).repeat(16)
            x.copy_(pattern)
        graph.replay()
        expected = _reference(x)
        for lhs, rhs in zip(actual, expected):
            torch.testing.assert_close(lhs.cpu(), rhs.cpu(), rtol=0, atol=0)


def test_indexer_quant_reuses_kernel_for_prefill_tail_rows(
    indexer_quant: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = indexer_quant._quantize_rows.run
    launches: list[tuple[tuple[int, ...], int, object]] = []

    def record_launch(
        *args: torch.Tensor | int, grid: tuple[int, ...], BLOCK_ROWS: int, **kwargs: int | bool
    ) -> object:
        kernel = launch(*args, grid=grid, BLOCK_ROWS=BLOCK_ROWS, **kwargs)
        assert kernel is not None
        launches.append((grid, BLOCK_ROWS, kernel))
        return kernel

    monkeypatch.setattr(indexer_quant._quantize_rows, "run", record_launch)
    for shape in ((16384, 32, 128), (9, 32, 128), (1, 128), (8, 32, 128)):
        values = torch.ones(shape, device="npu", dtype=torch.bfloat16)
        for actual, expected in zip(indexer_quant.quantize_indexer_rows_int8(values), _reference(values)):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)

    full, large_tail, single_row, small_tail = launches
    assert [launch[:2] for launch in launches] == [((16384,), 32), ((9,), 32), ((1,), 4), ((64,), 4)]
    assert full[2] is large_tail[2]
    assert single_row[2] is small_tail[2]
    assert full[2] is not small_tail[2]
