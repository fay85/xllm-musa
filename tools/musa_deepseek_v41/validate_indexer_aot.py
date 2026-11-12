# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the native fused indexer against official BF16 rounding stages."""

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


def _reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    committed: int,
    ratio: int,
) -> torch.Tensor:
    rows = torch.arange(keys.shape[0], device=keys.device)
    safe = torch.where((rows < committed)[:, None], keys, 0)
    dot = torch.einsum("shd,td->sht", query, safe)
    weighted = torch.relu(dot) * weights[:, :, None]
    result = weighted.sum(1).float()
    visible = ((positions + 1) // ratio).clamp(0, committed)
    return result.masked_fill(rows[None, :] >= visible[:, None], -torch.inf)


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

    torch.manual_seed(1112)
    device = torch.device("musa:0")
    model_weights = [
        torch.zeros(4096, 128, device=device, dtype=torch.bfloat16),
        torch.zeros(32, 128, device=device, dtype=torch.bfloat16),
    ]
    reference = native.NativeV41Indexer(model_weights, 32, 128, 1e-20, True)
    target = native.NativeV41Indexer(
        model_weights, 32, 128, 1e-20, True, str(args.library_directory)
    )
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for count in (129, 512, 1025):
            for rows in (1, 17, 128, 256):
                query = torch.randn(rows, 32, 128, device=device, dtype=torch.bfloat16)
                query = native.musa_v41_attention_fp4_quant(
                    query.reshape(-1, 128), 32, False
                )[0].reshape(query.shape)
                keys = native.musa_v41_attention_fp4_quant(
                    torch.randn(count, 128, device=device, dtype=torch.bfloat16),
                    32,
                    False,
                )[0]
                committed = count - 13
                keys[committed:].fill_(float("nan"))
                weights = torch.randn(rows, 32, device=device, dtype=torch.bfloat16)
                for ratio in (1, 2):
                    positions = (
                        torch.arange(rows, device=device, dtype=torch.int64)
                        + count // 2
                    )
                    expected = _reference(
                        query, keys, weights, positions, committed, ratio
                    )
                    cpp_reference = reference.score(
                        query, keys, weights, positions, committed, ratio
                    )
                    torch.testing.assert_close(cpp_reference, expected, atol=0, rtol=0)
                    actual = target.score(
                        query, keys, weights, positions, committed, ratio
                    )
                    finite = torch.isfinite(expected)
                    if not torch.equal(torch.isfinite(actual), finite):
                        raise AssertionError("Causal/committed masks differ")
                    difference = actual[finite] - expected[finite]
                    rms = (
                        difference.square().mean().sqrt()
                        / expected[finite].square().mean().sqrt().clamp_min(1e-10)
                    ).item()
                    if rms > 0.005 or not torch.isfinite(difference).all():
                        raise AssertionError(
                            f"M{rows} N{count} R{ratio}: RMS {rms} >0.5%"
                        )
                    native_ms = _time(
                        lambda: target.score(
                            query, keys, weights, positions, committed, ratio
                        )
                    )
                    reference_ms = _time(
                        lambda: reference.score(
                            query, keys, weights, positions, committed, ratio
                        )
                    )
                    k = min(64, count)
                    order_actual = torch.argsort(
                        actual, dim=-1, descending=True, stable=True
                    )[:, :k]
                    order_expected = torch.argsort(
                        expected, dim=-1, descending=True, stable=True
                    )[:, :k]
                    exact_fraction = (
                        (order_actual == order_expected).float().mean().item()
                    )
                    zero = target.score(query, keys, weights, positions, 0, ratio)
                    if not torch.isneginf(zero).all():
                        raise AssertionError(
                            "Zero committed must produce masked scores"
                        )
                    record = {
                        "rows": rows,
                        "keys": count,
                        "committed": committed,
                        "ratio": ratio,
                        "relative_rms": rms,
                        "max_abs": difference.abs().max().item(),
                        "finite_value_exact_fraction": (difference == 0)
                        .float()
                        .mean()
                        .item(),
                        "stable_top64_order_exact_fraction": exact_fraction,
                        "reference_gpu_ms": reference_ms,
                        "native_gpu_ms": native_ms,
                        "speedup": reference_ms / native_ms,
                        "warmups": 2,
                        "repeats": 10,
                    }
                    results.append(record)
                    args.output.write_text(
                        json.dumps(
                            {"status": "IN_PROGRESS", "results": results}, indent=2
                        )
                    )
                    logger.info(
                        "Indexer M%d N%d R%d RMS %.5g top64 %.4f %.4f -> %.4f ms",
                        rows,
                        count,
                        ratio,
                        rms,
                        exact_fraction,
                        reference_ms,
                        native_ms,
                    )
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_NATIVE_INDEXER_AOT",
                "results": results,
                "scope": "FP4 QDQ BF16 inputs, official BF16 score rounding, exact masks; report top-k differences without hiding them; not whole model accuracy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
