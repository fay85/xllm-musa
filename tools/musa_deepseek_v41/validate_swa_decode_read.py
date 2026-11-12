# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Basic q1 SWA read/index equivalence and deferred-commit checks on real MUSA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Callable
import torch
import torch_musa
import musa_python_kernel_test_module as native

sys.path.insert(0, "/work/repos/xllm-musa")
from scripts.logger import logger


def _reference(
    cache: torch.Tensor,
    current: torch.Tensor,
    table: torch.Tensor,
    positions: torch.Tensor,
    window: int,
    compressed: torch.Tensor | None,
    extra: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = positions.to(torch.int64)
    history = pos[0] - window + torch.arange(window, device=cache.device)
    safe = history.clamp_min(0)
    columns = torch.floor_divide(safe, cache.size(1)).remainder(table.size(1))
    blocks = table[0].index_select(0, columns).long().clamp_min(0)
    slots = blocks * cache.size(1) + safe.remainder(cache.size(1))
    gathered = cache.view(-1, cache.size(2)).index_select(0, slots)
    ring = torch.where(history.ge(0).unsqueeze(-1), gathered, 0)
    rows = torch.cat([ring, current], 0)
    desired = pos.view(1, 1) - window + 1 + torch.arange(window, device=cache.device)
    picks = torch.where(desired.ge(0), desired - pos.view(1, 1) + window, -1)
    if compressed is not None:
        mapped = torch.where(
            extra.ge(0).logical_and(extra.lt(compressed.size(0))),
            extra.long() + window + 1,
            -1,
        )
        picks = torch.cat([picks, mapped], -1)
        rows = torch.cat([rows, compressed], 0)
    block = (
        table[0]
        .index_select(
            0, torch.floor_divide(pos, cache.size(1)).remainder(table.size(1))
        )
        .long()
    )
    slot = block * cache.size(1) + pos.remainder(cache.size(1))
    return rows, picks, slot


def _exact(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.allclose(a, b, rtol=0, atol=0, equal_nan=True))


def _timed(function: Callable[[], torch.Tensor]) -> float:
    for _ in range(3):
        function()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(30):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 30


def _run(args: argparse.Namespace) -> dict:
    assert torch.musa.device_count() == 1
    torch.manual_seed(111211)
    device = torch.device("musa:0")
    cases = []
    positions = (0, 1, 31, 32, 127, 128, 129, 255, 256, 257, 1024)
    for width in (128, 512):
        for dtype in (torch.int32, torch.int64):
            for nondefault in (False, True):
                stream = (
                    torch.musa.Stream() if nondefault else torch.musa.current_stream()
                )
                with torch.musa.stream(stream):
                    for pos in positions:
                        for count in (None, 0, 3, 128):
                            cache = torch.randn(
                                5, 128, width, device=device, dtype=torch.bfloat16
                            )
                            if pos < 128:
                                cache[0].fill_(float("nan"))
                            before = cache.clone()
                            current = torch.randn(
                                1, width, device=device, dtype=torch.bfloat16
                            )
                            table = torch.tensor(
                                [[3, 1, 4, 2]], device=device, dtype=dtype
                            )
                            position = torch.tensor([pos], device=device, dtype=dtype)
                            compressed = (
                                None
                                if count is None
                                else torch.randn(
                                    count, width, device=device, dtype=torch.bfloat16
                                )
                            )
                            extra = (
                                None
                                if count is None
                                else torch.tensor(
                                    [[-1, 0, count - 1, count, count + 1]],
                                    device=device,
                                    dtype=dtype,
                                )
                            )
                            expected = _reference(
                                cache, current, table, position, 128, compressed, extra
                            )
                            actual = native.v41_swa_decode_read(
                                cache, current, table, position, 128, compressed, extra
                            )
                            assert all(
                                _exact(a.cpu(), b.cpu())
                                for a, b in zip(actual, expected)
                            ), (width, dtype, nondefault, pos, count)
                            assert _exact(
                                cache.cpu(), before.cpu()
                            ), "read kernel mutated cache"
                            cases.append(
                                {
                                    "kind": "read_metadata",
                                    "width": width,
                                    "dtype": str(dtype),
                                    "other_stream": nondefault,
                                    "position": pos,
                                    "compressed_rows": count,
                                    "rows_picks_slot_exact": True,
                                    "cache_read_only": True,
                                }
                            )
    full = []
    for table_values in ([3, 1, 4, 2], [-1, 1, 4, 2]):
        for pos in positions:
            for use_compressed in (False, True):
                cache = torch.randn(5, 128, 512, device=device, dtype=torch.bfloat16)
                other = cache.clone()
                table = torch.tensor([table_values], device=device, dtype=torch.int32)
                position = torch.tensor([pos], device=device, dtype=torch.int64)
                current = torch.randn(1, 512, device=device, dtype=torch.bfloat16)
                query = torch.randn(1, 8, 512, device=device, dtype=torch.bfloat16)
                sink = torch.randn(8, device=device)
                compressed = (
                    torch.randn(17, 512, device=device, dtype=torch.bfloat16)
                    if use_compressed
                    else None
                )
                extra = (
                    torch.tensor([[-1, 0, 5, 16, 17]], device=device, dtype=torch.int32)
                    if use_compressed
                    else None
                )
                old = native.NativeV41PagedAttention(
                    cache, 128, 512**-0.5, args.sparse_library, True, False
                )
                new = native.NativeV41PagedAttention(
                    other, 128, 512**-0.5, args.sparse_library, True, True
                )
                ref = old.forward(
                    query, current, table, pos + 1, sink, compressed, extra
                )
                actual = new.forward_with_positions(
                    query, current, table, pos + 1, sink, compressed, extra, position
                )
                assert _exact(actual.cpu(), ref.cpu()), (
                    "attention",
                    table_values,
                    pos,
                    use_compressed,
                )
                assert _exact(cache.cpu(), other.cpu()), (
                    "deferred_commit",
                    table_values,
                    pos,
                )
                full.append(
                    {
                        "position": pos,
                        "table": table_values,
                        "compressed": use_compressed,
                        "output_cache_exact": True,
                    }
                )
    samples = []
    for repeat in range(4):
        funcs = {
            "old": lambda: old.forward(
                query, current, table, pos + 1, sink, compressed, extra
            ),
            "fused": lambda: new.forward_with_positions(
                query, current, table, pos + 1, sink, compressed, extra, position
            ),
        }
        labels = ["old", "fused"] if repeat % 2 == 0 else ["fused", "old"]
        row = {label: _timed(funcs[label]) for label in labels}
        samples.append(row)
    return {
        "status": "PASS_SWA_FUSED_READ_AND_DEFERRED_COMMIT",
        "read_cases": cases,
        "attention_cases": full,
        "alternating_micro_ms": samples,
        "old_ms": statistics.median(r["old"] for r in samples),
        "fused_ms": statistics.median(r["fused"] for r in samples),
        "timing_scope": "MUSA event interval includes enqueue gaps, not full-model performance",
        "accuracy_scope": "仅通过基础功能检查，未完成模型精度验收",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-library", required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        result = _run(args)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info(
        "SWA basic checks passed: read=%d attention=%d",
        len(result["read_cases"]),
        len(result["attention_cases"]),
    )


if __name__ == "__main__":
    main()
