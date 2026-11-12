# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the native MATE sparse call, masks and local timing against FP32."""

from __future__ import annotations

import argparse
import json
import math
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


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - reference.float()
    scale = reference.float().square().mean().sqrt().clamp_min(1e-20)
    return {
        "relative_rms": (difference.square().mean().sqrt() / scale).item(),
        "max_abs": difference.abs().max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(1112)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    scale = 1 / math.sqrt(512)
    reference = native.NativeV41SparseAttention(scale, 32 * 1024**2)
    target = native.NativeV41SparseAttention(scale, 32 * 1024**2, str(args.library))
    device = torch.device("musa:0")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for rows in (1, 17, 128, 256):
            for width in (63, 128, 384):
                query = torch.randn(rows, 8, 512, device=device, dtype=torch.bfloat16)
                kv = torch.randn(1024, 512, device=device, dtype=torch.bfloat16)
                # Invalid picks must not leak NaNs from otherwise unused rows.
                kv[0].fill_(float("nan"))
                indices = torch.randint(1, 1024, (rows, width), device=device)
                indices[:, ::11] = -1
                indices[:, -1] = 2**40
                sink = torch.randn(8, device=device)
                expected = reference.forward(query, kv, indices, sink)
                actual = target.forward(query, kv, indices, sink)
                torch.musa.synchronize()
                error = _error(actual, expected)
                if not torch.isfinite(actual).all() or error["relative_rms"] > 0.01:
                    raise AssertionError(f"M{rows} K{width}: {error}; threshold1%")
                baseline_ms = _time(lambda: reference.forward(query, kv, indices, sink))
                mate_ms = _time(lambda: target.forward(query, kv, indices, sink))
                # An entirely masked row has zero values and finite sink mass.
                empty = torch.full_like(indices, -1)
                masked = target.forward(query, kv, empty, sink)
                torch.testing.assert_close(
                    masked, torch.zeros_like(masked), atol=0, rtol=0
                )
                record = {
                    "rows": rows,
                    "width": width,
                    "heads": 8,
                    "head_dim": 512,
                    "kv_rows": 1024,
                    "error": error,
                    "all_invalid_exact_zero": True,
                    "reference_gpu_ms": baseline_ms,
                    "mate_gpu_ms": mate_ms,
                    "speedup": baseline_ms / mate_ms,
                    "warmups": 2,
                    "repeats": 10,
                }
                results.append(record)
                args.output.write_text(
                    json.dumps({"status": "IN_PROGRESS", "results": results}, indent=2)
                )
                logger.info(
                    "MATE sparse M%d K%d RMS %.5g %.4f -> %.4f ms",
                    rows,
                    width,
                    error["relative_rms"],
                    baseline_ms,
                    mate_ms,
                )
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_NATIVE_MATE_SPARSE_H8",
                "results": results,
                "scope": "native sparse class, Q/KV BF16 carrier and invalid masks; not full-model or CED equivalence",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
