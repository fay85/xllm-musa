# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the reused xLLM MUSA weighted RMSNorm and measure local cost."""

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
    value: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    fp32 = value.float()
    return (
        fp32
        * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + epsilon)
        * weight.float()
    ).to(value.dtype)


def _time(function: Callable[[], Any]) -> float:
    for _ in range(2):
        function()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    stop = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(1112)
    device = torch.device("musa:0")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for hidden in (128, 512, 1280, 5120):
            weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
            for rows in (1, 17, 256):
                for amplitude in (0.0, 1e-12, 1.0, 1e4):
                    inputs = (torch.randn(rows, hidden, device=device) * amplitude).to(
                        torch.bfloat16
                    )
                    for epsilon in (1e-20, 1e-6):
                        expected = _reference(inputs, weight, epsilon)
                        actual = native.v41_rms_norm(inputs, weight, epsilon)
                        delta = actual.float() - expected.float()
                        relative = (
                            delta.square().mean().sqrt()
                            / expected.float().square().mean().sqrt().clamp_min(1e-30)
                        ).item()
                        if not torch.isfinite(actual).all() or relative > 0.0015:
                            raise AssertionError(
                                f"RMSNorm H{hidden}/M{rows}/amp{amplitude}/eps{epsilon}: RMS {relative} >0.15%"
                            )
                        row = {
                            "hidden": hidden,
                            "rows": rows,
                            "amplitude": amplitude,
                            "epsilon": epsilon,
                            "relative_rms": relative,
                            "max_abs": delta.abs().max().item(),
                        }
                        if amplitude == 1.0 and epsilon == 1e-20:
                            baseline_ms = _time(
                                lambda: _reference(inputs, weight, epsilon)
                            )
                            fused_ms = _time(
                                lambda: native.v41_rms_norm(inputs, weight, epsilon)
                            )
                            row.update(
                                reference_gpu_ms=baseline_ms,
                                native_gpu_ms=fused_ms,
                                speedup=baseline_ms / fused_ms,
                                warmups=2,
                                repeats=10,
                            )
                            logger.info(
                                "RMSNorm H%d M%d %.4f -> %.4f ms",
                                hidden,
                                rows,
                                baseline_ms,
                                fused_ms,
                            )
                        results.append(row)
                args.output.write_text(
                    json.dumps({"status": "IN_PROGRESS", "results": results}, indent=2)
                )
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_NATIVE_MUSA_RMSNORM",
                "cases": len(results),
                "results": results,
                "scope": "ordinary BF16 weighted RMSNorm; not Gemma, not F32 pooled compressor; no full-model accuracy claim",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
