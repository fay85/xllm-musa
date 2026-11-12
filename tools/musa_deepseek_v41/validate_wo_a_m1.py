# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the reused native wo_a GEMV on real target weights before rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def _time(function: Callable[[], Any]) -> float:
    for _ in range(10):
        function()
    torch.musa.synchronize()
    begin, end = torch.musa.Event(enable_timing=True), torch.musa.Event(
        enable_timing=True
    )
    begin.record()
    for _ in range(100):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(111219)
    weight = torch.load(args.fixture, map_location="cpu", weights_only=True)["weight"]
    if weight.shape != (1, 1024, 4096) or weight.dtype != torch.bfloat16:
        raise ValueError("Wrong real TP8 wo_a fixture")
    gpu = weight.to("musa:0")
    cases = []
    with torch.inference_mode():
        for amplitude in (0.001, 1.0, 100.0):
            for seed in range(20):
                value = (torch.randn(1, 1, 4096) * amplitude).to(torch.bfloat16)
                expected = (
                    torch.nn.functional.linear(
                        value.float().flatten(0, 1), weight.float().flatten(0, 1)
                    )
                    .to(torch.bfloat16)
                    .reshape(1, 1, 1024)
                )
                device_value = value.to("musa:0")
                actual = native.v41_wo_a_m1(device_value, gpu).cpu()
                reference = torch.einsum("tgd,grd->tgr", device_value, gpu).cpu()
                scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
                error = float(
                    (actual.float() - expected.float()).square().mean().sqrt() / scale
                )
                baseline_error = float(
                    (reference.float() - expected.float()).square().mean().sqrt()
                    / scale
                )
                if error > 0.004 or not torch.isfinite(actual).all():
                    raise AssertionError(
                        f"wo_a real weight amp{amplitude} seed{seed} RMS{error}"
                    )
                cases.append(
                    {
                        "amplitude": amplitude,
                        "sample": seed,
                        "native_relative_rms": error,
                        "einsum_relative_rms": baseline_error,
                    }
                )
        zero = torch.zeros(1, 1, 4096, device="musa:0", dtype=torch.bfloat16)
        if native.v41_wo_a_m1(zero, gpu).count_nonzero().item() != 0:
            raise AssertionError("Zero input must give exact zero")
        timings = []
        for trial in range(3):
            value = torch.randn(1, 1, 4096, device="musa:0", dtype=torch.bfloat16)
            baseline = _time(lambda: torch.einsum("tgd,grd->tgr", value, gpu))
            proposed = _time(lambda: native.v41_wo_a_m1(value, gpu))
            timings.append(
                {
                    "trial": trial,
                    "einsum_gpu_ms": baseline,
                    "native_gpu_ms": proposed,
                    "warmups": 10,
                    "repeats": 100,
                }
            )
            logger.info("wo_a trial%d %.6f -> %.6fms", trial, baseline, proposed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_REAL_WO_A_M1",
                "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                "cases": cases,
                "zero_exact": True,
                "timings": timings,
                "scope": "one projection TP8 rank0 actual weights; CPU FP32 oracle with BF16 boundary, not whole-model speedup",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
