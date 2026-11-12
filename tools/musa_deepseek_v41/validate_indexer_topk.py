# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require exact native stable top-k IDs, including masks and tied scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def _time(function: Callable[[], Any]) -> float:
    for _ in range(2):
        function()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    device = torch.device("musa:0")
    torch.manual_seed(111208)
    weights = [
        torch.zeros(4096, 128, device=device, dtype=torch.bfloat16),
        torch.zeros(32, 128, device=device, dtype=torch.bfloat16),
    ]
    reference = native.NativeV41Indexer(weights, 32, 128, 1e-20, True)
    target = native.NativeV41Indexer(
        weights, 32, 128, 1e-20, True, str(args.library_directory)
    )
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for width in (31, 129, 512, 1025, 4096, 8192):
            columns = torch.arange(width, device=device)
            for rows in (1, 17, 256):
                # BF16-valued scores create many exact ties at the cutoff.
                scores = torch.randn(rows, width, device=device).bfloat16().float()
                visible = torch.randint(0, width + 1, (rows,), device=device)
                scores.masked_fill_(columns[None, :] >= visible[:, None], -torch.inf)
                scores[:, 0] = -0.0
                scores[:, 1] = 0.0
                scores[:, 2] = torch.nan
                scores[:, 3] = torch.inf
                scores[:, 4] = -torch.inf
                for topk in (1, 64, 256, 512):
                    for masked in (False, True):
                        candidates = (
                            torch.rand(rows, width, device=device) > 0.4
                            if masked
                            else None
                        )
                        if candidates is not None:
                            candidates[0].fill_(False)
                        expected = reference.select(scores, visible, topk, candidates)
                        actual = target.select(scores, visible, topk, candidates)
                        cpu_scores = scores.cpu()
                        cpu_masked = (
                            cpu_scores.masked_fill(~candidates.cpu(), -torch.inf)
                            if candidates is not None
                            else cpu_scores
                        )
                        cpu_order = (
                            torch.argsort(
                                cpu_masked, dim=-1, descending=True, stable=True
                            )[:, : min(topk, width)]
                            .sort(-1)
                            .values
                        )
                        cpu_valid = (
                            cpu_order < visible.cpu()[:, None]
                        ) & torch.isfinite(cpu_masked.gather(-1, cpu_order))
                        cpu_expected = torch.where(cpu_valid, cpu_order, -1).int()
                        if not torch.equal(expected.cpu(), cpu_expected):
                            raise AssertionError(
                                "C++ reference does not match independent stable CPU reference"
                            )
                        if not torch.equal(actual, expected):
                            mismatch = (actual != expected).nonzero()
                            cpu_scores = scores.cpu()
                            cpu_masked = (
                                cpu_scores.masked_fill(~candidates.cpu(), -torch.inf)
                                if candidates is not None
                                else cpu_scores
                            )
                            cpu_order = (
                                torch.argsort(
                                    cpu_masked, dim=-1, descending=True, stable=True
                                )[:, : min(topk, width)]
                                .sort(-1)
                                .values
                            )
                            cpu_valid = (
                                cpu_order < visible.cpu()[:, None]
                            ) & torch.isfinite(cpu_masked.gather(-1, cpu_order))
                            cpu_expected = torch.where(cpu_valid, cpu_order, -1).int()
                            failing_rows = mismatch[:, 0].unique().cpu()
                            details = {
                                "rows": rows,
                                "width": width,
                                "topk": topk,
                                "masked": masked,
                                "mismatches": mismatch[:20].cpu().tolist(),
                                "actual_first": actual[0, :20].cpu().tolist(),
                                "expected_first": expected[0, :20].cpu().tolist(),
                                "actual_matches_cpu_reference": torch.equal(
                                    actual.cpu(), cpu_expected
                                ),
                                "musa_reference_matches_cpu": torch.equal(
                                    expected.cpu(), cpu_expected
                                ),
                                "failing_rows": [
                                    {
                                        "row": int(row),
                                        "scores": cpu_masked[row].tolist(),
                                        "visible": int(visible[row].item()),
                                        "actual": actual[row].cpu().tolist(),
                                        "musa_reference": expected[row].cpu().tolist(),
                                        "cpu_reference": cpu_expected[row].tolist(),
                                    }
                                    for row in failing_rows[:5]
                                ],
                            }
                            args.output.write_text(
                                json.dumps({"status": "FAIL", **details}, indent=2)
                            )
                            raise AssertionError(f"Stable top-k differs: {details}")
                        timings = {}
                        if (
                            width in (1025, 8192)
                            and rows in (1, 256)
                            and topk in (256, 512)
                        ):
                            baseline = _time(
                                lambda: reference.select(
                                    scores, visible, topk, candidates
                                )
                            )
                            fused = _time(
                                lambda: target.select(scores, visible, topk, candidates)
                            )
                            timings = {
                                "reference_gpu_ms": baseline,
                                "native_gpu_ms": fused,
                                "speedup": baseline / fused,
                                "warmups": 2,
                                "repeats": 10,
                            }
                        results.append(
                            {
                                "rows": rows,
                                "width": width,
                                "topk": topk,
                                "masked": masked,
                                "ids_bitwise_equal": True,
                                **timings,
                            }
                        )
                args.output.write_text(
                    json.dumps({"status": "IN_PROGRESS", "results": results}, indent=2)
                )
                logger.info(
                    "Stable top-k M%d width%d K1/64/256/512 mask/no-mask exact PASS",
                    rows,
                    width,
                )
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_NATIVE_INDEXER_RADIX_TOPK",
                "results": results,
                "scope": "exact native selection IDs with signed zero,NaN,Inf,causal/candidate masks; not full-model accuracy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
