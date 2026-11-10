# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Single-GPU production-binding validation; does not claim 1P1D acceptance.

Build musa_python_kernel_test_module with the same CMake tree as xllm. Put
its output directory on PYTHONPATH and run this script in the matched image.
The test module contains the exact header used by xllm_runtime, not a mock.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import torch_musa  # noqa: F401

from scripts.logger import logger
from xllm.python import initialize_runtime
from xllm.python.kernels_musa.shared_swiglu import select_shared_swiglu


def _reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.silu(gate.float().clamp(max=10.0)) * up.float().clamp(-10.0, 10.0)).bfloat16()


def _timings(operation: Callable[[], torch.Tensor], iterations: int) -> dict:
    for _ in range(10):
        operation()
    torch.musa.synchronize()
    wall_samples = []
    device_samples = []
    for _ in range(5):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - begin) * 1e6 / iterations)
        device_samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "iterations_per_sample": iterations, "samples": 5,
        "wall_us": wall_samples, "device_us": device_samples,
        "median_wall_us": statistics.median(wall_samples),
        "median_device_us": statistics.median(device_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError("Expose exactly one idle GPU to this validation")
    import musa_python_kernel_test_module as native

    sys.modules["xllm_runtime"] = native
    initialize_runtime()
    from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41MoE

    torch.manual_seed(1110)
    device = torch.device("musa:0")
    os.environ["XLLM_DSV41_MUSA_SHARED_SWIGLU"] = "v4_fused"
    fused = select_shared_swiglu(device, torch.bfloat16, 10.0)
    assert fused is not None
    result = {
        "status": "RUNNING", "torch": torch.__version__, "mate": version("mate"),
        "device": torch.musa.get_device_name(0), "seed": 1110,
        "scope": "single GPU; existing V4 kernel + production Python binding; no TP/PD/model acceptance",
        "cases": [], "timings": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2))

    def check(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        if actual.device.type != "musa":
            raise AssertionError("No host-compute fallback is allowed")
        actual_cpu = actual.cpu().float()
        expected_cpu = expected.cpu().float()
        finite = torch.isfinite(actual_cpu) & torch.isfinite(expected_cpu)
        delta = (actual_cpu[finite] - expected_cpu[finite]).abs()
        row = {
            "name": name, "elements": actual.numel(),
            "max_abs_error": delta.max().item() if delta.numel() else 0.0,
            "mismatched_elements": int(torch.count_nonzero(
                (actual_cpu != expected_cpu) & ~(torch.isnan(actual_cpu) & torch.isnan(expected_cpu))
            )),
            "rtol": 1 / 128, "atol": 1e-7, "status": "CHECKING",
        }
        result["cases"].append(row)
        save()
        # One BF16 rounding interval. NaNs/Infs must be preserved, not clamped
        # to finite values. This threshold is fixed before measuring.
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=1 / 128, atol=1e-7, equal_nan=True)
        row["status"] = "PASS"
        save()

    with torch.inference_mode():
        for tokens in (0, 1, 17, 256, 2048):
            for width in (288, 2304):
                # Noncontiguous views also exercise the real packing adapter.
                gate = (torch.randn(tokens, width * 2) * 8).bfloat16().to(device)[:, ::2]
                up = (torch.randn(tokens, width * 2) * 8).bfloat16().to(device)[:, ::2]
                check(f"activation_{tokens}x{width}", fused(gate, up), _reference(gate, up))
        values = torch.tensor(
            [-float("inf"), -100, -10.0625, -10, -9.9375, -0.0, 0, 9.9375, 10, 10.0625, 100, float("inf"), float("nan")],
            dtype=torch.bfloat16, device=device,
        )
        gate = values[:, None].expand(-1, values.numel())
        up = values[None, :].expand(values.numel(), -1)
        check("clamp_boundaries_and_nonfinite", fused(gate, up), _reference(gate, up))
        # Producer -> native launch -> consumer must use the same non-default
        # stream. Default-stream-only tests can hide stream-selection errors.
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            gate = torch.randn(17, 288, device=device, dtype=torch.bfloat16)
            up = torch.randn_like(gate)
            observed = fused(gate + 1, up - 1) * 2
            expected = _reference(gate + 1, up - 1) * 2
        stream.synchronize()
        check("nondefault_stream", observed, expected)

        # The actual model-level MoE caller and packed routed expert path.
        cfg = DeepseekV41Config(
            hidden_size=64, moe_intermediate_size=32, n_routed_experts=4,
            n_activated_experts=2, routed_scaling_factor=1.5, swiglu_limit=10.0,
        )
        moe = DeepseekV41MoE(cfg, 0, torch.bfloat16, device, packed_fp4=True)
        for name, parameter in moe.named_parameters():
            if parameter.dtype == torch.uint8:
                parameter.copy_(torch.randint(0, 256, parameter.shape, dtype=torch.uint8, device=device))
            else:
                parameter.copy_(torch.randn_like(parameter.float()).to(parameter.dtype) * 0.1)
        for name in ("w1", "w2", "w3"):
            getattr(moe, f"experts_{name}_scale").fill_(122)
        for tokens in (17, 1):
            hidden = torch.randn(tokens, 64, dtype=torch.bfloat16, device=device)
            moe._shared_activation = None
            reference = moe(hidden)
            moe._shared_activation = fused
            check(f"packed_moe_{tokens}_tokens", moe(hidden), reference)

        for tokens in (1, 256, 2048):
            for width in (288, 2304):
                gate = torch.randn(tokens, width, dtype=torch.bfloat16, device=device)
                up = torch.randn_like(gate)
                packed = torch.cat((gate, up), -1)
                timings = {
                    "tokens": tokens, "intermediate": width,
                    "reference": _timings(lambda: _reference(gate, up), 50),
                    "fused_including_cat_allocation_binding": _timings(lambda: fused(gate, up), 50),
                    "native_prepacked_including_allocation_binding": _timings(lambda: native.musa_shared_swiglu(packed, 10.0), 50),
                }
                # Include all three projections, not only activation time.
                # These are synthetic weights at the real hidden dimension;
                # they are not an official-checkpoint accuracy measurement.
                hidden = torch.randn(tokens, 5120, dtype=torch.bfloat16, device=device)
                w1 = torch.randn(width, 5120, dtype=torch.bfloat16, device=device) * (5120**-0.5)
                w3 = torch.randn_like(w1) * (5120**-0.5)
                w2 = torch.randn(5120, width, dtype=torch.bfloat16, device=device) * (width**-0.5)

                def mlp(activation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
                    return F.linear(activation(F.linear(hidden, w1), F.linear(hidden, w3)), w2)

                check(f"shared_mlp_{tokens}x{width}", mlp(fused), mlp(_reference))
                timings["full_shared_mlp_reference"] = _timings(lambda: mlp(_reference), 10)
                timings["full_shared_mlp_fused"] = _timings(lambda: mlp(fused), 10)
                result["timings"].append(timings)
                save()
    result["status"] = "PASS"
    save()
    logger.info("Shared SwiGLU numerical checks and timings saved to %s", args.output)


if __name__ == "__main__":
    main()
